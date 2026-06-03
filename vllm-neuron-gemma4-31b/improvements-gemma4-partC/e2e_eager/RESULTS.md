# Part C — End-to-End Eager Layer Results (the honest story)

This is the capstone of Part C: instead of timing kernels in isolation, we built
**one faithful Gemma4 decoder layer** (real projections, QK/V-norm, RoPE, GQA,
attention, GeGLU MLP, 4 norms + 2 residuals) and ran the full decode path two
ways with the **same weights** — pure PyTorch-on-Neuron vs the NKI kernels
swapped in — on the Neuron device in the Beta 2 DLC.

**The headline reverses the microbenchmark.** Per-op, the kernels look great
(Part C `RESULTS.md`: 4-16×). In a real layer, the end-to-end decode win is
**~1.01× — essentially nothing.** This doc explains exactly why, with the
measurements that prove it. This is the kind of result worth keeping: it stops
us shipping a "16× faster" claim that wouldn't survive integration.

## What we measured

Driver: `bench_e2e_layer.py` + `gemma4_eager_layer.py` (this folder),
`diag_dispatch.py` + `test_mh_attn.py` (in `customers/Hippocratic/gemma4_vllm/`).
Gemma4 31B dims: hidden=5376, inter=21504, 32 heads, head_dim 256 (SWA) / 512
(global), 49 SWA + 11 global layers. Decode = 1 new token, S=512 cached. fp32.

### 1. Full decoder layer, torch vs NKI (same weights)

| Layer | correctness (max abs diff) | torch | NKI | result |
|---|---|---|---|---|
| SWA (hd=256, 16 KV) | 0.00007 | 16.78 ms | 16.86 ms | 1.00× (tie) |
| Global (hd=512, 4 KV) | 0.00002 | 21.05 ms | 19.57 ms | 1.08× faster |
| **Projected full (49+11)** | — | **1053.8 ms/tok** | **1041.4 ms/tok** | **1.01×** |

Correctness is excellent (the NKI layer reproduces the torch layer to 5
decimals). But the speedup is ~nothing.

### 2. Why: the diagnostic (`diag_dispatch.py`)

| Measurement | NKI | torch | result |
|---|---|---|---|
| A) 32-head attention (32 single-head NKI calls vs 1 batched bmm) | 6.21 ms | 0.32 ms | **torch 19.6× faster** |
| B) norm at decode shape [1, 5376] | 0.10 ms | 0.70 ms | NKI 6.9× |
| C) norm at prefill shape [512, 5376] | 0.13 ms | 0.69 ms | NKI 5.3× |

Two effects cancel the wins:

1. **Per-head dispatch overhead dominates attention.** The single-head kernel
   beats one SDPA call, but a layer has 32 heads. Calling the kernel 32× costs
   ~0.19 ms/dispatch × 32 ≈ 6 ms, while torch batches all 32 heads into one
   `bmm` at 0.32 ms. The microbenchmark compared 1 NKI call vs 1 SDPA call —
   the wrong unit. The right unit is **the whole head group**.

2. **The norms genuinely win 5-7×, but they're rounding error in the layer.**
   Each norm is ~0.1 ms; the layer is ~17 ms. The layer time is dominated by
   the big projection GEMMs (q/k/v/o/gate/up/down) — which these kernels don't
   touch. Saving 0.6 ms across four norms out of 17 ms is <4%.

### 3. The fix that recovered most of the attention gap (`test_mh_attn.py`)

Batching all 32 heads into **one** kernel dispatch (heads looped inside the
kernel: `nki_decode_attention_hd256_mh.py`):

| Path | time | vs torch |
|---|---|---|
| torch batched bmm | 0.320 ms | — |
| NKI, 32 separate calls | 2.562 ms | 8.0× slower |
| **NKI, 1 batched call** | **0.396 ms** | **1.24× slower** |

Batching cut the attention kernel from 19.6× slower to 1.24× slower, confirming
dispatch overhead was the killer. But torch's batched `bmm` is still slightly
ahead for plain decode attention at S=512.

## Conclusions (honest)

1. **For decode on this eager build, PyTorch-on-Neuron is already well
   optimized.** Its batched matmul path for attention and its norms-as-fused-ops
   are hard to beat with hand-written NKI in eager mode. The microbenchmark's
   4-16× wins were **real but measured at the wrong granularity** (one op / one
   head), and they evaporate once you account for batching and the fact that
   norms are a tiny fraction of layer time.

2. **Where NKI would actually help Gemma4 decode:**
   - **Only if SDPA can't be used at all.** The original #1552 bottleneck was a
     genuine *fallback* (head_dim 256/512 unsupported by the fused flash
     kernel) forcing an unfused, memory-inefficient path — not the clean batched
     `bmm` we compare against here. If the serving path truly degrades to
     per-element ops, the NKI kernel's 0.4 ms batched attention beats that. On
     this eager build torch's `bmm` is *not* degrading, so there's no win to
     capture.
   - **By fusing the big GEMMs**, not the norms. The layer is GEMM-bound; a
     kernel that fuses, e.g., the GeGLU gate *with* the down-projection, or qkv
     into one matmul, is where real decode time lives. The elementwise fusions
     we wrote are correct and fast per-op but target <5% of layer time.
   - **At prefill, not decode.** The norm fusions win 5× at [512, H] too, and
     prefill processes many tokens — but prefill is also GEMM-bound, so the
     ceiling is still the projections.

3. **The kernels are still valuable as validated building blocks.** All 7
   compile, are numerically exact, and the batched attention + S-tiling fixes
   make them production-shaped. They're the right primitives *if and when* the
   serving path needs to bypass an SDPA fallback. They are not a free
   end-to-end speedup on a path where SDPA already works well.

## What changed in the kernels during this work

- **S>512 fix:** the decode-attention score matmul exceeded the hardware moving-
  free-dim limit of 512 at S=513 (512 cached + 1 new token). Both `hd256` and
  `hd512` now tile the score matmul over S in ≤512 chunks. Validated at
  S=64/512/513/1024, diff ≤ 0.000001.
- **New `nki_decode_attention_hd256_mh.py`:** all-heads-in-one-dispatch variant.
  Validated (diff 0.00001), 6.5× faster than the 32-call version.

## Reproduce

```bash
docker exec beta2_nki bash -lc 'cd /work && python3 bench_e2e_layer.py'   # layer torch vs NKI
docker exec beta2_nki bash -lc 'cd /work && python3 diag_dispatch.py'     # why
docker exec beta2_nki bash -lc 'cd /work && python3 test_mh_attn.py'      # batched-attn fix
```
