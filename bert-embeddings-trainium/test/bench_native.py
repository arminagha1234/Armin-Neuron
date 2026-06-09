#!/usr/bin/env python3
"""
Native BERT bench — apples-to-apples with Path A's vLLM bench shape.

Same throughput sweep (N=32/128/512), same 64-call latency loop.
Toggle USE_NKI_ATTN=1 to swap in the NKI fused attention kernel.

Models tested:
  - sentence-transformers/all-MiniLM-L6-v2 (22M, head_dim=32)
  - BAAI/bge-base-en-v1.5 (110M, head_dim=64) — bigger, more representative

Env:
  MODEL=...           HF model id (default MiniLM)
  MAX_LEN=128
  USE_NKI_ATTN=1      route through NKI fused attention (else stock matmul)
  N_LATENCY=64
  OUT=/tmp/bench_native.json
"""
import json
import os
import sys
import time

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

# import the model code from the same dir
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from native_bert_model import BertEncoder, load_from_hf

MODEL = os.environ.get("MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_LEN = int(os.environ.get("MAX_LEN", "128"))
N_LATENCY = int(os.environ.get("N_LATENCY", "64"))
OUT_PATH = os.environ.get("OUT", "/tmp/bench_native.json")
USE_NKI = os.environ.get("USE_NKI_ATTN", "0") == "1"

CORPUS = [
    "Encoder benchmark sentence one",
    "second sentence for embedding",
    "the quick brown fox jumps over the lazy dog",
    "Trainium2 NeuronCore embedding workload",
    "sentence transformers all MiniLM L6 v2",
    "vLLM offline embedding throughput test",
    "production serving with paged batching",
    "AWS Neuron compiler NEFF graph execution",
]


def percentile(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def tokenize_batch(tok, prompts, dev, dtype):
    enc = tok(prompts, return_tensors="pt", padding="max_length",
              max_length=MAX_LEN, truncation=True)
    ids = enc["input_ids"].to(dev)
    am = enc["attention_mask"].to(dev)
    pos = torch.arange(MAX_LEN, device=dev).unsqueeze(0).expand_as(ids)
    return ids, pos, am.to(dtype)


def main():
    import torch_neuronx  # noqa: F401  registers neuron device
    print(f"[bench-native] model={MODEL} max_len={MAX_LEN} USE_NKI={USE_NKI}", flush=True)

    dtype = torch.bfloat16
    dev = torch.device("neuron")

    # 1. load HF model on CPU, copy weights into our module
    print("[bench-native] loading HF weights...", flush=True)
    hf = AutoModel.from_pretrained(MODEL, return_dict=False).eval()
    hf_cfg = hf.config
    ours = BertEncoder(hf_cfg, dtype=dtype).eval()
    load_from_hf(hf, ours)

    # 2. cast to bf16 and move to neuron
    ours = ours.to(dtype).to(dev)

    # Optional: torch.compile (Beta 3 native eager-mode default; compile is
    # opt-in via USE_COMPILE=1). Per the Beta 3 guide:
    #   - Dynamic shapes are NOT supported. We have to compile per-batch-size.
    #   - reduce-overhead and max-autotune fall back to default mode.
    # Strategy: compile a separate module for each throughput batch size, and
    # an eager module for single-prompt latency.
    use_compile = os.environ.get("USE_COMPILE", "0") == "1"
    eager_model = ours  # always keep eager for latency
    compiled_modules = {}  # batch_size -> compiled module

    def get_model(batch_size):
        if not use_compile:
            return eager_model
        if batch_size not in compiled_modules:
            print(f"[bench-native] torch.compile for batch={batch_size}...", flush=True)
            compiled_modules[batch_size] = torch.compile(
                eager_model, backend="neuron", dynamic=False
            )
        return compiled_modules[batch_size]

    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"[bench-native] hidden={hf_cfg.hidden_size} heads={hf_cfg.num_attention_heads} "
          f"head_dim={hf_cfg.hidden_size // hf_cfg.num_attention_heads} "
          f"layers={hf_cfg.num_hidden_layers}", flush=True)

    # 3. cold forward (compile for batch=4 first to validate pipeline)
    cold_prompts = CORPUS[:4]
    ids, pos, am = tokenize_batch(tok, cold_prompts, dev, dtype)
    m4 = get_model(4)
    t0 = time.time()
    with torch.no_grad():
        _ = m4(ids, pos, am)
        torch_neuronx.synchronize()
    cold_s = time.time() - t0
    print(f"[bench-native] cold forward (incl. compile): {cold_s:.1f}s", flush=True)

    results = {"model": MODEL, "use_nki": USE_NKI, "use_compile": use_compile,
               "cold_s": round(cold_s, 2)}

    # 4. throughput
    for N in [32, 128, 512]:
        prompts = (CORPUS * (N // len(CORPUS) + 1))[:N]
        ids, pos, am = tokenize_batch(tok, prompts, dev, dtype)
        mN = get_model(N)
        # warmup
        with torch.no_grad():
            _ = mN(ids, pos, am)
            torch_neuronx.synchronize()
        # timed (3 iters, take min)
        best = None
        for _ in range(3):
            t = time.time()
            with torch.no_grad():
                emb = mN(ids, pos, am)
                torch_neuronx.synchronize()
            dt = time.time() - t
            best = dt if best is None or dt < best else best
        results[f"throughput_N{N}"] = {
            "prompts": N, "seconds": round(best, 4),
            "seq_per_s": round(N / best, 1),
            "ms_per_prompt": round(best / N * 1000, 3),
        }
        print(f"[bench-native] N={N:>4}  {best:6.4f}s  {N/best:7.1f} seq/s  "
              f"{best/N*1000:6.3f} ms/prompt", flush=True)

    # 5. single-prompt latency (warmup first to avoid recompile per call)
    # Pin to batch=1 shape, run multiple warmups to settle the graph cache.
    # NOTE: latency loop always uses eager_model so we get a real per-call
    # number — torch.compile of batch=1 forward can be added later if useful.
    ids, pos, am = tokenize_batch(tok, [CORPUS[0]], dev, dtype)
    m1 = get_model(1)
    for _ in range(8):
        with torch.no_grad():
            _ = m1(ids, pos, am)
            torch_neuronx.synchronize()

    lat = []
    for i in range(N_LATENCY):
        ids, pos, am = tokenize_batch(tok, [CORPUS[i % len(CORPUS)]], dev, dtype)
        t = time.time()
        with torch.no_grad():
            _ = m1(ids, pos, am)
            torch_neuronx.synchronize()
        lat.append((time.time() - t) * 1000)
    lat.sort()
    L = {
        "calls": len(lat),
        "p50_ms": round(percentile(lat, 0.50), 3),
        "p90_ms": round(percentile(lat, 0.90), 3),
        "p99_ms": round(percentile(lat, 0.99), 3),
        "min_ms": round(lat[0], 3),
        "max_ms": round(lat[-1], 3),
        "mean_ms": round(sum(lat) / len(lat), 3),
    }
    results["latency_single"] = L
    print(f"[bench-native] single-prompt (n={L['calls']}): "
          f"p50={L['p50_ms']:.2f} p90={L['p90_ms']:.2f} p99={L['p99_ms']:.2f} "
          f"mean={L['mean_ms']:.2f} ms", flush=True)

    json.dump(results, open(OUT_PATH, "w"), indent=2)
    print(f"[bench-native] wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[bench-native] FAIL: {type(e).__name__}: {e}", flush=True)
        sys.exit(1)
