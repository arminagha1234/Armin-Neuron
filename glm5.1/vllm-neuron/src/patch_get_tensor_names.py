"""
Patch the DeepSeek V3.2 model code for an API drift in
SafetensorsCheckpoint.

The DeepSeek V3.2 model (PR #2025) calls `checkpoint.get_tensor_names()`,
but the current vLLM-Neuron `SafetensorsCheckpoint` class does not expose
that method. The available data is in `_tensor_name_to_file` (a dict whose
keys are the tensor names).

This patch rewrites the call to use the existing private dict, and forces
indexing first so the dict is populated.

Run inside the vLLM-Neuron container after `patch_registry.py`:
    python patch_get_tensor_names.py
"""

import os
import sys

MODEL_PATH = "/opt/conda/lib/python3.12/site-packages/vllm_neuron/model/deepseek_v32/model.py"


def main():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found")
        print("Run patch_registry.py first to install the DeepSeek V3.2 model code.")
        sys.exit(1)

    with open(MODEL_PATH) as f:
        content = f.read()

    if "_tensor_name_to_file" in content:
        print("get_tensor_names already patched")
        return

    if "checkpoint.get_tensor_names()" not in content:
        print("WARN: did not find checkpoint.get_tensor_names() in model.py")
        print("Maybe the upstream code has been updated. No-op.")
        return

    # Replace the call with a version that ensures the index is built and
    # returns the keys of the existing private dict.
    content = content.replace(
        "checkpoint.get_tensor_names()",
        "(list(checkpoint._tensor_name_to_file.keys()) "
        "if checkpoint._tensor_name_to_file "
        "else (checkpoint._ensure_indexed() or list(checkpoint._tensor_name_to_file.keys())))",
    )

    with open(MODEL_PATH, "w") as f:
        f.write(content)

    print("Patched: checkpoint.get_tensor_names() → _tensor_name_to_file.keys()")


if __name__ == "__main__":
    main()
