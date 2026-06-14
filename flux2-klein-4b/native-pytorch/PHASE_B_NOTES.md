# Phase B Notes — VAE on Neuron + torch.compile

**Date:** 2026-06-13
**Status:** ⚠️ blocked on compiler instruction-count limit; not shipped

## Goal

Move the VAE encoder + decoder onto Neuron and `torch.compile` them
to eliminate the residual ~2.9 s of CPU-eager `vae.decode` time in the
Phase A shipped state. Projected end-to-end win: 6.86 s → ~4.2 s.

## What was tried

1. Added `vae_on_neuron=True` opt-in flag to
   `NeuronFlux2KleinPipeline.apply_neuron_patches()`.
2. When set:
   - Skips the CPU-coerce wrapper on `vae.decode`
   - `pipe.to(device)` now moves the VAE alongside the transformer
   - `_encode_vae_image` routes the VAE input to Neuron instead of CPU
3. Wrapped `pipe.vae.decode` and `pipe.vae.encode` with
   `torch.compile(backend="neuron", dynamic=False)`
4. Ran the bench script `bench_phase_b.py` on the test box.

## Result: compiler hit hard ceiling

The Neuron compiler (`neuronx-cc 2.25`) refused the VAE graph:

```
COMPILATION FAILED: Command failed (neuronx-cc compilation) with exit code 70:
[INTERNAL_ERROR] [NCC_IXTP002] Number of instructions (11453027) is over the
threshold (10000000). Tiling could potentially do a better job.
```

The VAE decode/encode is producing a ~11.4M instruction graph, over the
compiler's 10M ceiling. The DiT was ~134 MB / ~9.5M instructions and
compiled fine — the VAE at 1024×1024 spatial resolution generates
larger conv-heavy graphs than the DiT.

## Why it fails (likely)

- VAE decodes from 128×128 latents to 1024×1024 pixels — multiple
  upsampling stages, each with 3×3 convs at increasing spatial sizes.
- conv-heavy graphs unroll large numbers of instructions per layer.
- At 1024×1024 spatial output the unrolled graph crosses the 10M
  threshold the compiler enforces.

## What would unblock it

In rough effort order:

1. **Compile decoder and encoder separately.** They're independent
   nn.Modules — wrapping each with its own `torch.compile` produces
   two smaller graphs. ~2 hours, low risk.
2. **Compile per stage of the decoder.** The VAE has well-defined
   downsample/upsample blocks; each block compiled separately stays
   well under the threshold. Requires writing a small wrapper that
   dispatches per-block. Half-day.
3. **Targeted NxDI compile.** NxDI has chunked-NEFF support for
   exactly this kind of large graph. Would re-platform the VAE to the
   NxDI inference framework. 1-2 days.
4. **Bigger compiler threshold.** Open a bug with the Neuron team to
   raise NCC_IXTP002 — it's a soft tiling threshold, not a hardware
   limit. Out of band, no timeline.

## Recommendation

**Ship Phase A** (already on GitHub at commit `a955dc5`) as the
customer-facing result. It delivers a 5× wall-clock speedup
(34.6 s → 6.86 s) and brings Trainium2 to 1.3× the cost of H100,
which is a customer-shippable win for a single-image-many-prompts
zoom-LoRA pattern.

Treat Phase B as a follow-up engineering project (option 1 first,
option 3 if it doesn't unblock). Realistic week-of-work, not
afternoon-of-work.

## Files

- `src/neuron_flux2_klein_native.py` — has the `vae_on_neuron` flag
  wired in (off by default). Ready for option 1 to plug into.
- `src/bench_phase_b.py` — the bench harness, reproduces the failure
  in ~30 min compile time. Useful when the next attempt is made.
- `results/bench_phase_b_attempt.log` — failure log with the exact
  compiler error message.

## Rollback

The Phase A code path (default `vae_on_neuron=False`) is unchanged.
Customers running `run_flux2_klein_native.py --cache-image-latents`
get the shipped 6.86 s/call. No action required.
