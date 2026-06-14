# Step 2 (FP8) Findings — FLUX.2-klein-4B

**Date:** 2026-06-14
**Box:** `3.15.152.199` (trn2.48xl, Beta 3)

## Why I went back and did this

I initially dismissed Step 2 (FP8) as "moot because it stacks on TP."
That reasoning was wrong — the DiT profile is **bandwidth-bound**
(DMA 272ms ≈ tensor-engine 258ms), so FP8 (halving weight DMA) has an
independent single-rank rationale. So I actually tested it.

## Two FP8 paths

1. **Compiler auto-cast to fp8** (`--auto-cast=all --auto-cast-type=fp8_e4m3`)
   — quick to test, no checkpoint change. **TESTED.**
2. **fp8-stored weights** (Photoroom FP8 checkpoint + OCP→FP8_EXP4
   rescale) — genuinely reduces HBM weight traffic. **NOT tested**
   (multi-day: download + rescale + validate).

## Result: auto-cast FP8 — slower AND broken

`bench_cached.py` with `NEURON_CC_FLAGS="--auto-cast=all --auto-cast-type=fp8_e4m3"`:

| Config | Variant 3 warm | quality |
|---|---:|---|
| bf16 (Phase A) | **6.86 s** | std=18.15 sharp |
| FP8 auto-cast=all | **7.15 s** | **std=0.0 — ALL BLACK (corrupted)** |

Two failures:
1. **Not faster** (7.15s vs 6.86s, ~0.3s slower). auto-cast converts
   bf16→fp8 at compute time but the weights are still STORED bf16 in
   HBM — so it does NOT reduce the weight DMA the profile flagged. The
   DMA-halving benefit requires fp8-*stored* weights, not compute-time
   casting.
2. **Broken output.** `--auto-cast=all` casts ALL ops (not just
   matmuls) to fp8, which destroys numerics — the decoded image is
   all zeros (std=0.0, mean=0.0). A NaN/underflow cascade.

A narrower `--auto-cast=matmult --auto-cast-type=fp8_e4m3` might avoid
the corruption, but it still wouldn't reduce weight DMA (same
compute-time-cast limitation), so the upside is absent regardless.

## The real FP8 path (untested, low expected value)

To actually get the DMA win, you'd load `Photoroom/FLUX.2-klein-4b-fp8-diffusers`,
rescale OCP e4m3fn (±448) → Neuron FP8_EXP4 (±240) per the NxDI recipe:

```python
FP8_SCALING_FACTOR = 448.0 / 240.0
final_weight = (w.bfloat16() / FP8_SCALING_FACTOR).to(torch.float8_e4m3fn)
final_scale  = scale * FP8_SCALING_FACTOR
```

Profile-estimated upside: DMA 272ms → ~136ms, per-step 960→824ms,
~540ms over 4 steps. BUT:
- That's a DiT-loop saving, and the DiT loop is only ~2.9s of the 6.86s
  — saving 0.5s there → ~6.4s end-to-end, still ~7× H100.
- FP8 diffusion quality needs validation (cosine ≥ 0.999); the
  auto-cast corruption is a warning sign.
- Multi-day effort (download, rescale, integrate, validate).

Given every other lever came up empty at this model size, the expected
value of the multi-day fp8-weights path is low: best case ~6.4s
(vs 6.86s today), still far from H100. Documented for completeness; not
pursued.

## Honest status of Step 2

- Quick FP8 (auto-cast): TESTED — slower + broken. ✗
- Real FP8 (fp8 weights): NOT tested — multi-day, low expected value
  given the DiT loop is only 2.9s of the 6.86s total.

## Updated complete optimization map (every lever now addressed)

| Lever | Result |
|---|---|
| Phase A caching | 34s → **6.86s** ✅ SHIPPED |
| Step 1 TP=4 | 57s ✗ |
| Step 1.1 attention_cte single-rank | 8.11s ✗ |
| **Step 2 FP8 auto-cast** | **7.15s + corrupted ✗** |
| Step 2 FP8 weights | not run (multi-day, low EV) |
| Step 3 context parallelism | not run (TP base loses) |
| Step 4 fused kernels | blocked (no vllm_neuron in beta3) |
| Step 5 NKI RoPE | not run (~30ms ceiling) |
| Steps 6/7/9 requires_grad/RoPE | neutral ✗ |
| Step 8 functional rotate_half | not run |
| Step 10 single-NEFF verify | not run (diagnostic) |
| Step 11 auto-cast=matmult | 7.69s ✗ |
| Phase B VAE→Neuron | 7.73s ✗ (slower at 1024²) |

**6.86s single-rank Phase A remains the empirical optimum.** Of the
untested steps (3,4,5,8,10 + fp8-weights), none has a plausible path to
beating it: they're either blocked, sub-noise, or (fp8-weights) target
the 2.9s DiT loop that's a minority of the 6.86s total.

Box clean, no instance stopped.
