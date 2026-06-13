# Autoresearch on Trainium2 — Benchmark

**Date:** 2026-06-13
**Model:** GPT (autoresearch default: depth=8, dim=512, 4 heads, 50.3M params)
**Data:** FineWeb-Edu (autoresearch default dataset)
**Config:** seq_len=2048, batch_size=16, grad_accum=16, total_batch=524K tokens,
bf16, Muon+AdamW optimizer, 5-min time budget
**Instance:** trn2.48xlarge, single logical core (NEURON_RT_VISIBLE_CORES=0-1)
**Stack:** Beta 3 DLC, torch 2.11.0, neuronxcc (walrus driver)

## Numbers

| Phase | Time |
|---|---|
| Compile (first-run, one-time) | ~19 min |
| Training (5-min budget) | 300.4 s |
| Total wall clock (compile + train + eval) | 1449.3 s |
| Per optimizer step | ~13 s |
| Eval (val_bpb computation) | ~12 min (one-time compile for eval graph) |

| Metric | Value |
|---|---|
| **val_bpb** | **1.834** |
| Final train loss (smoothed) | 5.392 |
| Throughput | 40,000 tok/sec |
| MFU | 5.27% |
| Steps completed | 34 |
| Tokens processed | 17.8M |
| Epochs | 1 |

## Training Curve

```
step 00  loss 9.011  (first step, includes compile warmup)
step 05  loss 8.074
step 10  loss 6.985
step 15  loss 6.321
step 20  loss 5.916
step 25  loss 5.664
step 30  loss 5.479
step 33  loss 5.392  (final, budget exhausted)
```

Smooth monotonic decrease — model is learning correctly on Trainium.

## Cost Analysis

| Instance | $/hr | 5-min run cost | 100 experiments (overnight) |
|---|---|---|---|
| **trn2.3xlarge** | $2.23 | $0.19 | **$18.50** |
| **trn2.48xlarge** | $21.50 | $1.79 | $179 |
| p5.48xlarge (H100) | $32.77 | $2.73 | $272 |

Note: trn2.48xlarge is overkill for this (uses 1 of 64 cores). A
trn2.3xlarge ($2.23/hr) is the right instance — same single-core
performance, 10× cheaper. The $18.50 overnight figure assumes
subsequent runs skip the ~19-min compile (cached NEFFs).

## Per-Step Breakdown

At 13s per optimizer step (524K tokens):
- Forward pass (compiled NEFF): ~8s estimate
- Backward pass: ~4s estimate
- Optimizer (Muon + AdamW, eager): ~1s estimate

The forward/backward NEFF includes the full 8-layer GPT with SDPA
attention. Compile fuses the whole model into a single execution graph.

## Comparison Notes

This is NOT directly comparable to H100 autoresearch results because:
1. **DEVICE_BATCH_SIZE** is 16 here vs 128 on H100 (memory constraint)
2. **Window attention (SSSL)** isn't fully implemented (window_size not
   passed through SDPA)
3. **MFU** is 5.27% vs ~50% on H100 (batch underutilization)

A fair comparison would require matching batch sizes (needs multi-core
DP on Trainium) and implementing windowed attention via NKI.
