"""
Patch vLLM-Neuron model registry to add GLM 5.1 support.

GLM 5.1 (`GlmMoeDsaForCausalLM`) shares its architecture (MLA + MoE + DSA)
with DeepSeek V3.2. Until vLLM-Neuron upstream adds a dedicated entry,
we register GLM as a route to the existing DeepSeek V3.2 model class
(installed via PR #2025).

Run inside the vLLM-Neuron container BEFORE importing vllm:
    python patch_registry.py
"""

import os
import sys

REGISTRY_PATH = "/opt/conda/lib/python3.12/site-packages/vllm_neuron/model/registry.py"


def main():
    if not os.path.exists(REGISTRY_PATH):
        print(f"ERROR: {REGISTRY_PATH} not found")
        print("Are you inside the vLLM-Neuron container?")
        sys.exit(1)

    with open(REGISTRY_PATH) as f:
        content = f.read()

    if "deepseek_v32" in content:
        print("Registry already patched")
        return

    # Add the import
    content = content.replace(
        "from .qwen3_vl import Qwen3VLForConditionalGeneration",
        "from .qwen3_vl import Qwen3VLForConditionalGeneration\n"
        "from .deepseek_v32 import DeepseekV32ForCausalLM",
    )

    # Add both DeepSeek V3.2 and GLM 5.1 entries
    # GLM 5.1 routes to the same factory because the architectures match
    content = content.replace(
        '("Qwen3VLForConditionalGeneration", Qwen3VLForConditionalGeneration),',
        '("Qwen3VLForConditionalGeneration", Qwen3VLForConditionalGeneration),\n'
        '        ("DeepseekV32ForCausalLM", DeepseekV32ForCausalLM),\n'
        '        ("GlmMoeDsaForCausalLM", DeepseekV32ForCausalLM),',
    )

    with open(REGISTRY_PATH, "w") as f:
        f.write(content)

    print("Registry patched: DeepseekV32ForCausalLM + GlmMoeDsaForCausalLM registered")


if __name__ == "__main__":
    main()
