#!/usr/bin/env python3
"""Make qwen3_5's factory.from_configs accept vLLM's text_neuron_config kwarg.

vllm_neuron's neuron_model_runner calls:
    model_cls.from_configs(hf_config=..., text_neuron_config=..., vision_neuron_config=...)
but qwen3_5/factory.py declares from_configs(hf_config, neuron_config) ->
    TypeError: got an unexpected keyword argument 'text_neuron_config'

model_bf16.py already carries the fix; factory.py is the class the registry
actually resolves (see qwen3_5/__init__.py), so BOTH need it. Idempotent.
Run after any overlay of the qwen3_5 sources.
"""
import py_compile, shutil, sys, os

F = "/opt/conda/lib/python3.13/site-packages/vllm_neuron/model/qwen3_5/factory.py"

OLD = """    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None,
    ) -> nn.Module:
        return cls._select_implementation(hf_config, neuron_config)
"""

NEW = """    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None = None,
        *,
        text_neuron_config: NeuronConfig | None = None,
        **kwargs,
    ) -> nn.Module:
        # vLLM passes `text_neuron_config` (plus `vision_neuron_config`) for
        # multimodal-wrapper checkpoints; Qwen3.5 nests its decoder under
        # `text_config`. Accept both and treat as the decoder's neuron_config.
        if neuron_config is None:
            neuron_config = text_neuron_config
        return cls._select_implementation(hf_config, neuron_config)
"""

src = open(F).read()
if "text_neuron_config" in src:
    print("COMPAT_ALREADY_APPLIED"); sys.exit(0)
if OLD not in src:
    print("COMPAT_ANCHOR_NOT_FOUND — factory.py differs from expected"); sys.exit(2)
bak = F + ".pre_compat"
if not os.path.exists(bak):
    shutil.copy2(F, bak)
open(F, "w").write(src.replace(OLD, NEW, 1))
try:
    py_compile.compile(F, doraise=True)
    print(f"COMPAT_OK backup={bak}")
except py_compile.PyCompileError as e:
    shutil.copy2(bak, F)
    print("COMPAT_REVERTED syntax error:", e); sys.exit(3)
