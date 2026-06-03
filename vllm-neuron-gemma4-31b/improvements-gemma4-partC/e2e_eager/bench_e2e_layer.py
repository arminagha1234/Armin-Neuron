#!/usr/bin/env python3
"""End-to-end eager Gemma4 decoder-layer benchmark: torch vs NKI, on device.

For one SWA layer and one Global layer:
  1. Verify the NKI path output matches the torch path (correctness).
  2. Time per-token decode latency for each path.
  3. Project a full 60-layer (49 SWA + 11 Global) per-token estimate.

Honest: this is ONE layer of each type with random weights at a fixed cached
seq length. It measures the decode-time compute of a real Gemma4 layer with the
kernels swapped in vs the eager fallback. It does NOT include sampler, embedding,
lm_head, or scheduler overhead — those are layer-count-independent and small
relative to 60 layers of attention+MLP.
"""
import sys, os, time
sys.path.insert(0, "/work")
os.environ["NEURON_SKIP_EFA_AFFINITY"] = "1"
import torch
import torch_neuronx
from gemma4_eager_layer import Gemma4DecoderLayer

dev = torch.device("privateuseone:0")
S = 512            # cached tokens
ITERS = 30
WARMUP = 5
N_SWA, N_GLOBAL = 49, 11


def timeit(fn):
    for _ in range(WARMUP):
        fn()
    torch_neuronx.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        fn()
    torch_neuronx.synchronize()
    return (time.time() - t0) / ITERS * 1e3  # ms/iter


def run_layer(is_global, label):
    layer = Gemma4DecoderLayer(dev, is_global=is_global, dtype=torch.float32)
    tok = layer.new_token()
    K, V = layer.new_cache(S)

    # correctness: same inputs, both paths
    out_t = layer.forward_torch(tok, K, V)
    out_n = layer.forward_nki(tok, K, V)
    torch_neuronx.synchronize()
    diff = (out_t - out_n).cpu().abs().max().item()
    rel = diff / (out_t.cpu().abs().max().item() + 1e-9)
    print(f"\n=== {label} layer (head_dim={layer.hd}, kv_heads={layer.n_kv}) ===")
    print(f"  correctness: max abs diff torch-vs-NKI = {diff:.5f}  (rel {rel:.4f})")

    t_torch = timeit(lambda: layer.forward_torch(tok, K, V))
    t_nki = timeit(lambda: layer.forward_nki(tok, K, V))
    sp = t_torch / t_nki
    tag = f"{sp:.2f}x faster" if sp >= 1 else f"{1/sp:.2f}x SLOWER"
    print(f"  per-layer decode: torch {t_torch:.3f}ms   NKI {t_nki:.3f}ms   -> {tag}")
    return t_torch, t_nki


print(f"=== Gemma4 eager decoder-layer bench (S={S}, iters={ITERS}) ===")
swa_t, swa_n = run_layer(False, "SWA")
glo_t, glo_n = run_layer(True, "Global")

full_torch = N_SWA * swa_t + N_GLOBAL * glo_t
full_nki = N_SWA * swa_n + N_GLOBAL * glo_n
print("\n=== Projected full-model per-token decode (49 SWA + 11 Global) ===")
print(f"  torch path: {full_torch:.1f} ms/token   ({1000/full_torch:.1f} tok/s)")
print(f"  NKI path:   {full_nki:.1f} ms/token   ({1000/full_nki:.1f} tok/s)")
print(f"  speedup:    {full_torch/full_nki:.2f}x")
print("\n(Projection = sum of per-layer decode times; excludes embed/lm_head/"
      "sampler/scheduler, which are layer-count-independent.)")
print("\nDone.")
