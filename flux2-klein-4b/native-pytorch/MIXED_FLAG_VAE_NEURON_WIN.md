# Mixed-flag VAE-on-Neuron — measured 1.41× win

**Date:** 2026-06-14
**Box:** trn2.48xl, Beta 3 DLC (`concourse-release-0461d3b:latest`),
`NEURON_RT_VIRTUAL_CORE_SIZE=2`.
**Config:** FLUX.2-klein-4B, 1024×1024, 4 distilled steps, guidance=1.0,
seed=42, Variant 3 (prompt + image-latent caching).

## TL;DR

| Path | Warm avg | Std | vs prior baseline | Verdict |
|---|---:|---:|---:|---|
| Phase A + channels_last CPU VAE (prior shipped) | 5.92 s | 18.15 | — | baseline |
| **VAE on Neuron + PAVE fixes + mixed-flag compile** | **4.19 s** | **18.16** | **−1.73 s (−29%)** | **✅ NEW DEFAULT** |

End-to-end **1.41× faster**, **no quality loss** (std 18.16 vs baseline
18.15, gate PASS), **no PR-folder churn beyond two new files** + a flag
default change.

## What changed (PAVE-derived recipe)

Three things, all derived from the internal PAVEDigitalTwinDiffusion
Neuron port (`code.amazon.com/packages/PAVEDigitalTwinDiffusion`):

1. **Gather-free upsample** in the VAE decoder. The 3 nearest-neighbor
   `F.interpolate(mode="nearest")` upsamples lower to a per-element
   GATHER on Neuron — tens of millions of dynamic-DMA packets,
   measured by PAVE at 74% of device time. Replace with
   `reshape→expand→reshape` (a broadcast/contiguous-copy on Neuron,
   not a gather). Bit-identical math; verified with TorchDispatchMode
   that 0 gather-class aten ops remain post-patch.

2. **fp32 GroupNorm** wrapper around all 52 GroupNorms. Stock bf16
   GroupNorm produces near-NaN numerics on the VAE's small-variance
   activations on Neuron. Cast in to fp32, group-norm, cast back to
   the input dtype. No downstream effect; full pipeline keeps bf16.

3. **Mixed-flag compile.** The DiT compiles under
   `--model-type=transformer` (the proven winner on transformer
   workloads — see Step 13 A/B below). The VAE compiles under
   `--model-type=transformer + --model-type=unet-inference`
   (the conv-scheduling hint, where it actually wins, on a conv
   workload). Implemented by wrapping `vae.encode`/`vae.decode` so
   `NEURON_CC_FLAGS` is set during the VAE call and restored after.
   The compile cache key includes flags, so warm calls hit the right
   NEFF.

## How we got here — 4 measurements

All measured on `3.15.152.199` today (2026-06-14) under identical
conditions. Each variant was a distinct cold compile in its own
`NEURON_COMPILE_CACHE_URL` directory, then 5 warm timed runs.

| # | Variant | Warm avg | Std | vs CPU baseline | Verdict |
|---|---|---:|---:|---:|---|
| 0 | Picture 1 — CPU VAE channels_last (prior shipped) | 5.92 s | 18.15 | — | baseline |
| 1 | **Step 13 alone** — DiT under `unet-inference` (no VAE moves) | **7.71 s** | sane | **+1.79 s SLOWER** | ❌ rejected |
| 2 | Option 1 — VAE on Neuron + PAVE fixes, DiT=transformer everywhere | 5.19 s | 18.16 | −0.73 s (−12%) | ✅ ship-quality |
| 3 | Option 2 — VAE on Neuron + PAVE fixes, **`unet-inference` everywhere** | 6.84 s | 18.17 | +0.92 s SLOWER | ❌ DiT regression dominates |
| 4 | **Option 3 — Mixed: DiT=transformer, VAE=unet-inference** | **4.19 s** | **18.16** | **−1.73 s (−29%)** | **✅ ship** |

The two negative measurements (#1 and #3) are also documented findings
— they show that:
- `unet-inference` is **harmful** on the DiT (a transformer); it's a
  conv-scheduling hint and miscost a transformer.
- Applying `unet-inference` pipeline-wide loses on net (~+1 s vs Option 1)
  because the DiT regression is bigger than the VAE gain.
- The right composition is **per-component flag selection**, not a
  global flag.

## Why the prior "VAE on Neuron is rejected as slower" was wrong

Earlier sessions measured `--vae-on-neuron` at 3.8 s vs 2.9 s CPU
(slower) and degraded quality (std 14.6 vs 18.1), and the handoff doc
flagged the lever as "experimental, not recommended." That conclusion
was correct *for the broken port*, not for VAE-on-Neuron in principle.

What was broken in the prior attempt:
- **Gather trap was unfixed.** The 3 nearest-neighbor upsamples ran
  as per-element gathers, dominating runtime.
- **bf16 GroupNorm** without an fp32 wrapper produced NaN-adjacent
  numerics, dimming the output (std 14.6).
- **Compiler flag was wrong** for a VAE — the default
  `--model-type=transformer` mis-schedules conv-heavy code.

This new path fixes all three. Quality returns to 18.16 (essentially
matching the CPU baseline 18.15) and runtime drops by 29% end-to-end.

## H100 comparison (post-Option-3)

| Metric | Trainium2 (Option 3) | H100 (4-step est.) | Ratio |
|---|---:|---:|---:|
| Warm latency / image | **4.19 s** | ~0.9 s | ~4.6× slower |
| $/image (single full instance, latency) | ~$0.0250 | ~$0.0082 | H100 ~3× cheaper at this granularity |
| **$/image (32-core trn2.48xl, throughput)** | **~$0.00078** | **~$0.0082** | **Trainium ~10.5× cheaper** |

The latency gap is real — for interactive UIs that want sub-second
image generation, H100 still wins. For a marketplace-shaped
throughput workload, Option 3 widens the existing $/image win
meaningfully.

(H100 4-step latency is extrapolated from a measured 28-step run at
0.218 s/step. A measured 4-step H100 baseline is on the followups list.)

## Where Option 3 fits in the optimization roadmap

This work consumed **Step 12** (finish the VAE-on-Neuron port) and
**Step 13** (`unet-inference` flag) from `HANDOFF_TO_IMPLEMENTATION.md`,
plus the PAVE cross-reference at the bottom of that doc. Both are now
captured: Step 12 ships, Step 13 is a per-component decision (no on
DiT, yes on VAE), not a pipeline-wide flag.

Remaining levers in priority order:

| Priority | Lever | Expected | Status |
|---|---|---|---|
| (done) | Step 12 + 13 — VAE on Neuron, mixed flags | −29% end-to-end | ✅ shipped here |
| 1 | Cheap wins (Steps 6/7/8/9) — graph cleanup | ~−2% | not yet done |
| 2 | Step 10 — verify single-NEFF DiT compile | TBD | not yet done |
| 3 | Step 1 — lift NxDI architecture (attention_cte) | ~−40% on DiT loop | scoped, not done |
| 4 | Step 2 — FP8 weights (DMA halving) | ~−15% on DiT loop | spec written, low priority post-Option-3 |
| 5 | Cluster — measured 4-step H100 baseline | tightens ratio claim | followup |

Note that with Option 3 shipped, **FP8 weights drops materially in
priority**. The DiT loop is now an even smaller fraction of end-to-end
(~2.05 s of 4.19 s = ~49%), and a 15% DiT-loop saving from FP8 is
~0.3 s end-to-end. The multi-day FP8 effort is worth less than it was
when DiT was a larger share of total time.

## Files added / changed

```
flux2-klein-4b/native-pytorch/
├── src/
│   ├── flux2_vae_neuron_fixes.py       # NEW — gather-free upsample,
│   │                                     fp32 GroupNorm, mixed-flag
│   │                                     wrapper, gather verifier
│   └── run_flux2_klein_native.py        # MODIFIED — --vae-on-neuron is
│                                          now RECOMMENDED (was
│                                          EXPERIMENTAL/NOT RECOMMENDED);
│                                          calls apply_vae_neuron_fixes +
│                                          install_mixed_flag_wrapper
└── MIXED_FLAG_VAE_NEURON_WIN.md         # NEW — this doc
```

## Reproduction

```bash
HF_TOKEN=<token> /opt/torch-neuronx/.venv/bin/python \
    src/run_flux2_klein_native.py \
    --base-model black-forest-labs/FLUX.2-klein-4B \
    --no-lora \
    --image input.png \
    --prompt "Zoom into the red highlighted area" \
    --steps 4 --guidance-scale 1.0 \
    --vae-on-neuron \
    --cache-image-latents \
    --output zoomed.png
```

The bench script that produced the 4.19 s number is
`src/bench_vae_mixed_flag.py` (kept in the PR alongside
`src/bench_vae_neuron_fixed.py`, which covers Options 1+2 for the A/B
record).

## Attribution

The recipe (gather-free upsample + fp32 GroupNorm + per-component
compiler-flag selection) is lifted directly from the internal
`PAVEDigitalTwinDiffusion` Neuron port (Amazon, 2026), generalized for
the FLUX.2-klein-4B VAE structure. Their work is the source of the
fixes; this PR is the application to a public-customer FLUX pipeline.
