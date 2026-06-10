# SPDX-License-Identifier: Apache-2.0
"""Register Qwen3.6-27B in vllm_neuron's model registry without forking it.

Run before `vllm serve` (e.g. via `python -m qwen3_6.register` in the
container, or by importing this module from a startup hook).

After this call, vllm_neuron will dispatch HF arch
"Qwen3_5ForConditionalGeneration" to our local 27B implementation.

NOTE: The HF arch class name is the same `Qwen3_5ForConditionalGeneration`
that the 3.5 family uses. The same registry slot in vLLM thus dispatches
to our 27B model. If you also have the 3.5 adapter on the PYTHONPATH,
whichever is imported LAST wins — keep them in separate containers.
"""

import importlib
import logging

logger = logging.getLogger(__name__)


_PATCHED = False
_HF_ARCH = "Qwen3_5ForConditionalGeneration"


def register() -> None:
    """Idempotently inject Qwen3.6-27B factory into the registry under the HF arch name."""
    global _PATCHED
    if _PATCHED:
        logger.debug("Qwen3.6 already registered, skipping")
        return

    from . import Qwen3_6ForConditionalGeneration

    # 1. Register into vllm_neuron's own registry (used by some code paths
    #    and for our smoke test).
    registry = importlib.import_module("vllm_neuron.model.registry")
    original_get_models = registry.get_models

    def patched_get_models() -> list[tuple[str, type]]:
        models = list(original_get_models())
        # Drop any pre-existing slot for this arch — including Qwen3.5's
        # if both adapters happen to be on the PYTHONPATH.
        models = [m for m in models if m[0] != _HF_ARCH]
        models.append((_HF_ARCH, Qwen3_6ForConditionalGeneration))
        logger.info("Registered %s -> Qwen3_6 in vllm_neuron", _HF_ARCH)
        return models

    registry.get_models = patched_get_models

    # 2. Register into vLLM's `_ModelRegistry` (the one
    #    `neuron_model_runner.load_model` actually queries via
    #    `ModelRegistry.resolve_model_cls`). This is the registry that
    #    matters for `vllm serve`. Without this, the resolver falls back
    #    to the transformers impl which has no `from_configs` method.
    #
    # IMPORTANT: vLLM ships a built-in `Qwen3_5ForConditionalGeneration`
    # at `vllm.model_executor.models.qwen3_5` registered as a
    # `_LazyRegisteredModel`. That's the WRONG class for us — it's
    # vLLM's transformers-style class without `from_configs`. We must
    # FORCE-REPLACE the slot, not skip if it exists.
    try:
        vllm_registry_mod = importlib.import_module(
            "vllm.model_executor.models.registry"
        )
        ModelRegistry = vllm_registry_mod.ModelRegistry  # the singleton
        _RegisteredModel = vllm_registry_mod._RegisteredModel  # eager wrapper

        # Build the eager-registered slot wrapping our class directly.
        slot = _RegisteredModel.from_model_cls(Qwen3_6ForConditionalGeneration)

        # vLLM's introspection on our factory class doesn't pick up that
        # this is a text generation model, so force the flag. _ModelInfo
        # AND _RegisteredModel are both frozen dataclasses, so we have to
        # rebuild both with dataclasses.replace.
        try:
            import dataclasses
            new_interfaces = dataclasses.replace(
                slot.interfaces, is_text_generation_model=True
            )
            slot = dataclasses.replace(slot, interfaces=new_interfaces)
        except Exception as exc:
            logger.warning(
                "Could not patch _ModelInfo.is_text_generation_model: %r", exc
            )

        # Force-write — bypasses register_model's de-dupe check that
        # would skip our class because vLLM's built-in lazy entry exists.
        ModelRegistry.models[_HF_ARCH] = slot
        logger.info(
            "Force-registered %s -> Qwen3_6 in vllm.ModelRegistry "
            "(replaced vLLM's built-in stub, marked as text-generation model)",
            _HF_ARCH,
        )
    except Exception as exc:
        logger.warning("Could not patch vllm.ModelRegistry: %r", exc)

    _PATCHED = True


def install_post_plugin_hook() -> None:
    """Hook into vLLM's plugin loader so our register runs AFTER vllm_neuron's plugin.

    Why: vllm_neuron's `register` plugin (entry point in vllm.platform_plugins)
    re-initializes `ModelRegistry.models[]` from defaults, which OVERWRITES
    the slot we just patched. We need to re-apply the patch after
    `load_general_plugins()` finishes. We monkey-patch the loader so any
    caller (including the worker subprocesses) gets our patch reapplied.

    Idempotent — safe to call multiple times. The wrapped loader resets
    `_PATCHED` to False so register() runs again on the post-plugin path.
    """
    try:
        import vllm.plugins as _plugins_mod
    except Exception as exc:
        logger.warning("Could not import vllm.plugins to install hook: %r", exc)
        return

    if getattr(_plugins_mod.load_general_plugins, "_qwen36_wrapped", False):
        return  # already wrapped

    _orig_loader = _plugins_mod.load_general_plugins

    def _wrapped_loader(*args, **kwargs):
        result = _orig_loader(*args, **kwargs)
        # Force re-register AFTER plugins have run.
        global _PATCHED
        _PATCHED = False
        try:
            register()
        except Exception as exc:
            logger.warning("post-plugin re-register failed: %r", exc)
        return result

    _wrapped_loader._qwen36_wrapped = True  # type: ignore[attr-defined]
    _plugins_mod.load_general_plugins = _wrapped_loader
    logger.info("Installed post-plugin re-register hook on load_general_plugins")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    register()

    # Quick verification: re-list the registry and check we're in it.
    from vllm_neuron.model.registry import get_models

    names = [name for name, _ in get_models()]
    print("Registered model architectures:")
    for n in names:
        marker = " <-- new" if n == _HF_ARCH else ""
        print(f"  - {n}{marker}")
