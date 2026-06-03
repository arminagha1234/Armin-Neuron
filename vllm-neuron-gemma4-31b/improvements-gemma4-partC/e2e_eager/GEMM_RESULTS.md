# Part C — GEMM & Fusion Results (the real bottleneck)

The e2e layer benchmark showed the decoder layer is **GEMM-bound** — the
projection matmuls (q/k/v/o/gate/up/down) dominate, and the elementwise NKI
kernels target <5% of layer time. So the only place a real end-to-end win can
come from is the GEMMs. This doc tests that directly.

## Two questions

1. Can a hand-written NKI GEMM **match the framework matmul** on the eager build?
2. Does **fusing the two MLP projections** (gate+up+gelu+multiply into one
   kernel, keeping the intermediate off HBM) beat the torch 3-op path?

## Method

Beta 2 DLC, `privateuseone:0`, fp32, 30 iters after warmup. Gemma4 MLP shapes:
hidden=5376, intermediate=21504. Tested at decode (M=1) and prefill (M=512).
Drivers: `test_gemm.py`, `test_fused_geglu_gemm.py`
(in `customers/Hippocratic/gemma4_vllm/`, copies in this folder).

## Results

### 1. Plain GEMM: C = A @ B  (`nki_gemm.py`)

| Shape | NKI | torch | result | correctness |
|---|---|---|---|---|
| decode M=1, [1,5376]@[5376,21504] | 1.671 ms | 1.553 ms | torch 1.08× | rel 0.00000 ✅ |
| prefill M=512, [512,5376]@[5376,21504] | 7.011 ms | 3.168 ms | torch 2.21× | rel 0.00000 ✅ |

- **Decode (M=1) is a near-tie (within 8%).** A matrix-vector product is
  memory-bound — the cost is reading the full weight once, and both paths hit
  the same HBM-read roofline. NKI can't go faster than reading the weights.
- **Prefill (M=512) loses 2.2×.** torch's matmul is much better compute-tiled
  (pipelined K-accumulation, better SBUF reuse) than the naive 128×128×512
  tiling here. Beating a vendor matmul at compute-bound shapes in eager NKI is a
  deep rabbit hole — not worth it for this exercise.

### 2. Fully-fused GeGLU GEMM (`nki_fused_geglu_gemm.py`)

`act = gelu_tanh(x @ Wg) * (x @ Wu)` in one kernel vs torch's 3 ops (gate, up,
gelu*up) with the [M,21504] intermediate written to and read from HBM.

| Shape | NKI fused | torch 3-op | result | correctness |
|---|---|---|---|---|
| decode M=1 | 3.107 ms | 3.237 ms | **NKI 1.04× faster** | rel 0.00000 ✅ |
| prefill M=512 | 13.087 ms | 6.847 ms | torch 1.91× | rel 0.00000 ✅ |

- **Decode: fusion wins (marginally, 1.04×).** Even though the raw matmul is
  slightly behind torch, fusing both projections + gelu + multiply eliminates
  the intermediate HBM round-trips, and the net is a small but real win at the
  memory-bound decode shape. This is the **first end-to-end-relevant win** in
  Part C that survives realistic conditions.
- **Prefill: still loses 1.9×.** The compute-bound matmul deficit dominates the
  bandwidth savings from fusion.

## Bottom line

| Path | Best NKI result | Verdict |
|---|---|---|
| Elementwise fusions (norm/GeGLU-gate/softcap) | 4-16× per-op | irrelevant e2e (<5% of layer) |
| Single-head attention | 16× per-op | killed by 32× dispatch; batched = 1.24× slower than bmm |
| Plain GEMM | tie at decode | can't beat vendor matmul |
| **Fused GeGLU GEMM** | **1.04× at decode** | **the one real, if small, decode win** |

**The honest conclusion across all of Part C:** on this eager build, where the
framework's matmul and batched SDPA already work well, hand-written NKI doesn't
deliver a big end-to-end decode speedup. The single place it nets a (small) win
is **operator fusion that removes HBM round-trips at the memory-bound decode
shape** — the fused GeGLU GEMM at 1.04×. To get that to a meaningful number you
would fuse more aggressively (e.g. qkv into one matmul + the o-projection with
the residual add) and, critically, **match the vendor matmul's tiling** — which
is the hard part and the real ceiling.

Where NKI's big wins genuinely live (and why they're not here):
- **When SDPA truly falls back** (head_dim 256/512 unsupported by the fused
  flash kernel in the *serving* image) — then the alternative is an unfused,
  element-wise attention path, and the NKI kernel's 0.4 ms batched attention
  beats it. We can't reproduce that fallback on this eager build because torch's
  batched `bmm` doesn't degrade.
- **In a fully compiled graph** (torch.compile / a traced NEFF), where the NKI
  kernels are fused into the graph and the per-dispatch eager overhead
  disappears. Eager `wrap_nki` pays a launch cost per call that a compiled graph
  would not.

## Reproduce

```bash
docker exec beta2_nki bash -lc 'cd /work && python3 test_gemm.py'
docker exec beta2_nki bash -lc 'cd /work && python3 test_fused_geglu_gemm.py'
```
