#!/usr/bin/env python3
"""DeepSeek V3.2 — 2-layer smoke test on trn2.48xl.

Verifies:
  1. The injected vllm_neuron.model.deepseek_v32 package is wired.
  2. PR #2025's FP8-aware weight loader handles the native HF FP8 ckpt.
  3. 2 layers compile through the Neuron compiler without OOM/HLO failure.

Includes a runtime shim adding `get_tensor_names()` to the v5 beta's
SafetensorsCheckpoint (PR #2025 expects this method but the v5 image
predates it).
"""
import os
import time
import sys

os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")
os.environ.setdefault("NEURON_SCRATCHPAD_PAGE_SIZE", "512")
os.environ.setdefault("NEURON_CC_FLAGS", (
    "--enable-saturate-infinity "
    "--enable-mixed-precision-accumulation "
    "--auto-cast=none "
    "--model-type transformer "
    "-O1 "
    "--hbm-scratchpad-page-size=512 "
    "--tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2' "
    "--tensorizer-options='--vectorize-strided-dma' "
    "--internal-hlo2tensorizer-options='--verify-hlo=true'"
))
os.environ["VLLM_NEURON_MIN_KV_BUDGET_GIB"] = "0"

import torch._dynamo
torch._dynamo.config.cache_size_limit = 64

# === SHIM: PR #2025 expects SafetensorsCheckpoint.get_tensor_names() ===
# The v5 beta image's vllm_neuron.utils.checkpoints predates this method.
# Fortunately _ensure_indexed() + _tensor_name_to_file gives us everything.
import vllm_neuron.utils.checkpoints as _ckpt_mod
if not hasattr(_ckpt_mod.SafetensorsCheckpoint, "get_tensor_names"):
    def _get_tensor_names(self):
        """Return all tensor names available in the checkpoint."""
        self._ensure_indexed()
        return list(self._tensor_name_to_file.keys())
    _ckpt_mod.SafetensorsCheckpoint.get_tensor_names = _get_tensor_names
    print("[shim] Patched SafetensorsCheckpoint.get_tensor_names()")

from vllm import LLM, SamplingParams

NUM_LAYERS = int(os.environ.get("NUM_LAYERS", "2"))

print("=" * 70)
print(f"DeepSeek V3.2 — {NUM_LAYERS}-layer smoke (TP=64)")
print("=" * 70)
print(f"Time: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
sys.stdout.flush()

t0 = time.time()
llm = LLM(
    model="deepseek-ai/DeepSeek-V3.2",
    tensor_parallel_size=64,
    max_model_len=128,
    max_num_seqs=1,
    gpu_memory_utilization=0.92,
    hf_overrides={
        "num_hidden_layers": NUM_LAYERS,
        "quantization_config": {},
    },
)
t_load = time.time() - t0
print(f"\nMODEL LOAD: {t_load:.0f}s ({t_load/60:.1f} min)", flush=True)

params = SamplingParams(max_tokens=10, temperature=0)
for prompt in ["The capital of France is", "1 + 1 ="]:
    t1 = time.time()
    out = llm.generate([prompt], params)[0].outputs[0]
    dt = (time.time() - t1) * 1000
    print(f"  {prompt!r} -> {out.text!r}  ({len(out.token_ids)} tok, {dt:.0f} ms)")

print("\n" + "=" * 70)
print(f"SMOKE TEST PASSED ({NUM_LAYERS} layers)")
print(f"Load: {t_load/60:.1f} min")
print("=" * 70)
