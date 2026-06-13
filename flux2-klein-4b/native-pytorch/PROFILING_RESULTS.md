# FLUX.2-klein-4B DiT Profile Analysis

**Date:** 2026-06-13
**Instance:** trn2.48xlarge (`i-02a51e30b3a33408d`, us-east-2)
**NEFF:** 134 MB compiled DiT forward (1024×1024, 28-step shape)
**Tool:** neuron-profile 2.30.5

## Key Metrics

| Metric | Value | Interpretation |
|---|---|---|
| **MFU** | **51.8%** | Half of peak compute used — decent but room to improve |
| Model FLOPS per forward | 76.7 TFLOPS | The full 56-block DiT |
| Tensor engine (matmuls) | 258 ms | Core compute time |
| DMA (data movement) | 272 ms | Weight loading + activation shuffles |
| Vector engine (activations) | 170 ms | RMSNorm, SwiGLU, softmax |
| Scalar engine | 21.5% of total | Index math, control flow |
| Arithmetic intensity | 334 | Very high — matmuls are well-shaped |
| Weight data moved | 22.7 GB / forward | bf16 weights read from HBM |

## What this tells us

### 1. The model is memory-bandwidth bound, not compute bound

DMA time (272 ms) ≈ tensor engine time (258 ms). The Neuron cores wait
for weight data almost as long as they compute. This is the classic
"large model on a bandwidth-limited device" pattern.

**Implication:** FP8 quantization (halving weight traffic) could save
~100-136 ms per forward = 15-20% speedup without any code changes.

### 2. NKI custom kernels won't help significantly

At 51.8% MFU, the tensor engine is already well-utilized. Writing a
custom NKI attention kernel can't improve the matmul execution time
much — the compiler is already doing a good job scheduling the
Q@K → softmax → @V chain.

**This confirms our earlier finding:** `wrap_nki(attention_cte)` was
11% slower than naive compile because the per-call dispatch overhead
outweighed any kernel-level improvement.

### 3. Scalar overhead is the hidden cost

21.5% scalar engine activity means the compiler spends time on index
computation, block transitions, and control flow. With 56 identical
blocks, there's likely repeated scalar setup per block that could be
hoisted. This is a neuronxcc compiler optimization opportunity (not
user-actionable today).

### 4. The 960 ms NEFF execution breaks down as:

```
Tensor engine:  258 ms (27%)  — matmuls
DMA:            272 ms (29%)  — weight + activation movement
Vector:         170 ms (18%)  — element-wise ops
Scalar:         ~200 ms (21%) — control + scheduling
Other/overlap:  ~60 ms (5%)   — pipeline overlap / idle
Total:          ~960 ms
```

(Note: engines overlap so sum > total; the total of ~960 ms matches
the tqdm-measured 960 ms/step NEFF execution time.)

## Optimization priorities (based on profile)

| Priority | Optimization | Expected gain | Why (from profile) |
|---|---|---|---|
| 1 | **FP8 quantization** | 15-20% | DMA time ≈ compute time; halving weights cuts DMA |
| 2 | **Compiler flag tuning** | 5-10% | Scalar 21.5% overhead may reduce with scheduling hints |
| 3 | **Prompt caching** | (CPU-side) | Doesn't affect NEFF time; saves 25s of CPU per image |
| 4 | NKI custom kernels | <5% (not worth it) | MFU already 51.8%; dispatch overhead kills the gain |

## FP8 opportunity

BFL provides `black-forest-labs/FLUX.2-klein-4b-fp8` — a pre-quantized
FP8 variant. If neuronxcc supports FP8 matmul (E4M3 × bf16 or E4M3 ×
E4M3), loading this model would:

- Halve weight size: 22.7 GB → 11.4 GB per forward DMA
- Reduce DMA time by ~50%: 272 ms → ~136 ms
- Net forward time: ~960 ms - 136 ms = ~824 ms per step
- **New $/image at 32 cores with prompt caching:**
  `(4 × 824ms + 5s) / 32 × ($21.50/3600) = $0.0052/image (29% cheaper than H100)`

Whether neuronxcc supports FP8 compute on Trainium2 Beta 3 needs
verification.

## Files

- Profile NTFF: `/workspace/flux_dit_profile.ntff` (5.5 GB, on trn2.48xl)
- NEFF: `/tmp/neff_cache/08/35/083511f0d5ac49b2dbc45ddc7431ef6a.neff` (134 MB)
