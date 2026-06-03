# Part C — Throughput GEMM Results (optimized tiling + bf16)

Follow-up to `GEMM_RESULTS.md`. The customer wants **throughput** (tokens/min),
which means large effective batch (large M) — the compute-bound regime where the
first naive GEMM lost ~2.2×. We applied the two standard throughput tricks from
the official NKI matmul tutorial and tested the real serving dtype.

## What we changed

1. **Hoist-load tiling** (`nki_gemm_opt.py`, `nki_fused_geglu_gemm_opt.py`):
   load each M-block's lhsT K-tiles ONCE and reuse across all N-tiles (and, for
   the fused GeGLU, across both gate and up matmuls). Pattern lifted from
   `nki_matmul_hoist_load_` in the NKI tutorial + Jim's `fused_silu_gate` reuse.
2. **bf16** — the serving dtype, and where Trainium's PE array runs ~4× the fp32
   rate. A throughput comparison in fp32 is misleading; bf16 is the real test.

## Results

### Optimized GEMM (hoist-load) vs torch matmul

| dtype | M | NKI | torch | result |
|---|---|---|---|---|
| fp32 | 128 | 1.588 ms | 0.849 ms | torch 1.87× |
| fp32 | 512 | 6.225 ms | 3.172 ms | torch 1.96× |
| fp32 | 1024 | 12.321 ms | 6.281 ms | torch 1.96× |
| **bf16** | 128 | 1.227 ms | 0.391 ms | **torch 3.14×** |
| **bf16** | 512 | 4.796 ms | 0.855 ms | **torch 5.61×** |
| **bf16** | 1024 | 9.539 ms | 1.624 ms | **torch 5.87×** |

### Optimized fused GeGLU vs torch 3-op

| dtype | M | NKI | torch | result |
|---|---|---|---|---|
| fp32 | 128 | 3.102 ms | 1.931 ms | torch 1.61× |
| fp32 | 512 | 12.215 ms | 6.860 ms | torch 1.78× |
| fp32 | 1024 | 24.487 ms | 13.407 ms | torch 1.83× |
| **bf16** | 128 | 2.363 ms | 1.062 ms | **torch 2.22×** |
| **bf16** | 512 | 9.327 ms | 2.238 ms | **torch 4.17×** |
| **bf16** | 1024 | 18.629 ms | 4.161 ms | **torch 4.48×** |

All correct (rel ≤ 0.0085, bf16 tolerance).

## The honest finding

- **Hoist-load helped fp32** (prefill GEMM 2.21×→1.96×, fused GeGLU 1.91×→1.78×)
  but didn't close the gap.
- **In bf16 the gap WIDENS to ~4-6×.** torch's matmul hits the bf16 PE-array
  fast path; the hand-tiled NKI kernel does not get the same systolic-array
  utilization. The eager `wrap_nki` path also can't pipeline DMA with compute
  across tiles the way the vendor's compiled matmul does.
- **Fusion's bandwidth savings can't beat a 4-6× compute deficit.** Removing the
  intermediate HBM round-trip is real, but when the matmul itself is 5× slower,
  the fused kernel is still ~4× behind.

## Conclusion: stop hand-writing GEMMs in eager NKI

The data is consistent across fp32/bf16 × M=128/512/1024: **a hand-tiled eager
NKI GEMM does not beat the framework matmul**, and in the serving dtype (bf16)
it's 4-6× slower. The vendor matmul is the product of deep PE-array scheduling
work that a tutorial-pattern kernel won't match in eager mode.

Throughput on Gemma4 is **GEMM-bound**, and the GEMMs are already running on the
optimal path in the framework. So:

- **NKI elementwise/attention kernels don't move throughput** — they target the
  <5% of time that isn't GEMM (shown in `RESULTS.md`).
- **NKI GEMMs don't beat the framework** — shown here.
- **Therefore eager-NKI is not the throughput lever for Gemma4 decode/prefill on
  this stack.** The real throughput levers are the ones Part B identified and
  PR #1552 is pursuing: better bucketing, the fused flash-attention kernel for
  head_dim>128 (so attention stops falling back), FP8/quantization, and larger
  batch scheduling — all in the compiled serving graph, not eager hand-kernels.

Where eager NKI *would* still win: a genuine op the framework has **no good
kernel for** (e.g. an unusual fused activation, a custom sparse pattern, or
attention when SDPA truly falls back to element-wise). Plain GEMMs and standard
GeGLU are not in that category — the framework already nails them.

## Reproduce

```bash
docker exec beta2_nki bash -lc 'cd /work && python3 test_gemm_opt.py'    # fp32
docker exec beta2_nki bash -lc 'cd /work && python3 test_gemm_bf16.py'   # bf16
```
