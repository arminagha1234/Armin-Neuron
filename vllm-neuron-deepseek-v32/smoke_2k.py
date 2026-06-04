#!/usr/bin/env python3
"""DeepSeek V3.2 — 8L compile under max_model_len=2048.

The PR #2025 model has dsa_max_seq_len=3072 hard-coded in config; vLLM
errors at compile when max_model_len > 3072 because the DSA indexer's
slice_scatter buffers are sized at 3072.

Cap max_model_len at 2048 to test compile cleanly. Then we can scan
TTFT at [256, 512, 1024, 2048].
"""
import os
import time
import sys
import json

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

from vllm import LLM, SamplingParams

NUM_LAYERS = int(os.environ.get("NUM_LAYERS", "8"))
MAX_LEN = int(os.environ.get("MAX_LEN", "2048"))
print("=" * 70)
print(f"DeepSeek V3.2 - {NUM_LAYERS}L max_model_len={MAX_LEN} (TP=64)")
print("=" * 70)
print(f"Time: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
sys.stdout.flush()

t0 = time.time()
llm = LLM(
    model="deepseek-ai/DeepSeek-V3.2",
    tensor_parallel_size=64,
    max_model_len=MAX_LEN,
    max_num_seqs=1,
    gpu_memory_utilization=0.92,
    enable_chunked_prefill=False,
    hf_overrides={
        "num_hidden_layers": NUM_LAYERS,
        "quantization_config": {},
    },
)
t_load = time.time() - t0
print(f"\nMODEL LOAD: {t_load:.0f}s ({t_load/60:.1f} min)", flush=True)

# TTFT scan with unique random prompts
print("\n--- TTFT scan ---")
import random
random.seed(0)
results = []

for seq_target in [256, 512, 1024, 2000]:  # 2000 < 2048 to leave room for the BOS token
    seed_words = ("The patient is a 64-year-old male presenting with acute chest pain and "
                  "shortness of breath, with a history of hypertension and type 2 diabetes. ")
    # Build a long string then truncate by token count once vLLM tokenizes
    reps = (seq_target * 5 + len(seed_words) - 1) // len(seed_words)
    prompt = (seed_words * reps)[: seq_target * 5]
    params_warm = SamplingParams(max_tokens=1, temperature=0)
    # Warmup once
    _ = llm.generate([prompt], params_warm)
    # Measure 3 times
    timings_ms = []
    for _ in range(3):
        t1 = time.time()
        out = llm.generate([prompt], params_warm)[0].outputs[0]
        timings_ms.append((time.time() - t1) * 1000)
    median = sorted(timings_ms)[len(timings_ms) // 2]
    print(f"  seq~{seq_target:>5}  TTFT median={median:.0f} ms  raw={[round(t) for t in timings_ms]}")
    results.append({"target_seq": seq_target, "median_ms": median, "raw_ms": timings_ms})

# Generation sample
print("\n--- Generation sample (max_tokens=20) ---")
gen_params = SamplingParams(max_tokens=20, temperature=0)
gen_results = []
for prompt in ["The capital of France is", "1 + 1 ="]:
    t1 = time.time()
    out = llm.generate([prompt], gen_params)[0].outputs[0]
    dt_total = (time.time() - t1) * 1000
    print(f"  {prompt!r} -> {out.text!r}  ({len(out.token_ids)} tok, {dt_total:.0f} ms)")
    gen_results.append({"prompt": prompt, "output": out.text, "n_tokens": len(out.token_ids), "total_ms": dt_total})

print("\n" + "=" * 70)
print(f"DONE (NUM_LAYERS={NUM_LAYERS}, max_model_len={MAX_LEN})")
print(f"Load: {t_load/60:.1f} min")
print("=" * 70)

print("\nJSON_RESULTS:", json.dumps({
    "config": {
        "model": "deepseek-ai/DeepSeek-V3.2",
        "num_layers": NUM_LAYERS,
        "tp": 64,
        "max_model_len": MAX_LEN,
        "enable_chunked_prefill": False,
    },
    "load_time_s": round(t_load, 1),
    "ttft": results,
    "generation": gen_results,
}, indent=2))
