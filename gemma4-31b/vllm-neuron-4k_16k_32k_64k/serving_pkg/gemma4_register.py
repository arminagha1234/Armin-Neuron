# SPDX-License-Identifier: Apache-2.0
"""Register the custom Gemma4 model into vLLM + vllm_neuron registries.

Gemma4 is NOT in the vLLM-Neuron Beta (SDK 2.30) supported list (only Llama3 +
GPT-OSS are). This shim injects our local `gemma4` package so `vllm serve` can
dispatch HF arch "Gemma4ForConditionalGeneration" to it.

Pattern mirrors the validated Qwen3.5 pathB register.py (force-replace into
vllm.ModelRegistry + post-plugin re-register hook).

Usage: import gemma4_register; gemma4_register.install_post_plugin_hook()
       (or `python -m gemma4_register` to verify registration).
"""
import dataclasses
import importlib
import logging

logger = logging.getLogger(__name__)
_PATCHED = False
ARCH = "Gemma4ForConditionalGeneration"


def register() -> None:
    global _PATCHED
    if _PATCHED:
        return
    from gemma4.factory import Gemma4ForConditionalGeneration

    # 1. vllm_neuron's own registry (used by some code paths + our smoke test)
    try:
        registry = importlib.import_module("vllm_neuron.model.registry")
        original_get_models = registry.get_models

        def patched_get_models():
            models = list(original_get_models())
            if ARCH not in {n for n, _ in models}:
                models.append((ARCH, Gemma4ForConditionalGeneration))
                logger.info("Registered %s in vllm_neuron", ARCH)
            return models

        registry.get_models = patched_get_models
    except Exception as exc:
        logger.warning("Could not patch vllm_neuron.model.registry: %r", exc)

    # 2. vLLM's _ModelRegistry (the one neuron_model_runner.load_model queries).
    #    Force-replace any built-in stub and mark as text-generation model.
    try:
        vllm_registry_mod = importlib.import_module("vllm.model_executor.models.registry")
        ModelRegistry = vllm_registry_mod.ModelRegistry
        _RegisteredModel = vllm_registry_mod._RegisteredModel
        slot = _RegisteredModel.from_model_cls(Gemma4ForConditionalGeneration)
        try:
            new_interfaces = dataclasses.replace(
                slot.interfaces, is_text_generation_model=True
            )
            slot = dataclasses.replace(slot, interfaces=new_interfaces)
        except Exception as exc:
            logger.warning("Could not set is_text_generation_model: %r", exc)
        ModelRegistry.models[ARCH] = slot
        logger.info("Force-registered %s in vllm.ModelRegistry", ARCH)
    except Exception as exc:
        logger.warning("Could not patch vllm.ModelRegistry: %r", exc)

    _PATCHED = True


def install_post_plugin_hook() -> None:
    """Re-apply registration AFTER vllm_neuron's plugin loads (which resets the registry)."""
    try:
        import vllm.plugins as _plugins_mod
    except Exception as exc:
        logger.warning("Could not import vllm.plugins: %r", exc)
        return
    if getattr(_plugins_mod.load_general_plugins, "_gemma4_wrapped", False):
        return
    _orig = _plugins_mod.load_general_plugins

    def _wrapped(*a, **k):
        r = _orig(*a, **k)
        global _PATCHED
        _PATCHED = False
        try:
            register()
        except Exception as exc:
            logger.warning("post-plugin re-register failed: %r", exc)
        return r

    _wrapped._gemma4_wrapped = True
    _plugins_mod.load_general_plugins = _wrapped
    logger.info("Installed post-plugin re-register hook for %s", ARCH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    register()
    from vllm_neuron.model.registry import get_models
    names = [n for n, _ in get_models()]
    print("Registered architectures:")
    for n in names:
        print(f"  - {n}{'  <-- new' if n == ARCH else ''}")
    assert ARCH in names, "Gemma4 registration FAILED"
    print("OK")
