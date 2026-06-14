# Step 12 Resolution — "move VAE off CPU" verified, and why it's already handled

**Date:** 2026-06-14
**Directive:** Verify the 3 Step-12 questions against shipped code; if any
"no", Step 12 is the ~30s Priority-#0 win.

## The 3 questions, answered against GitHub-shipped neuron_flux2_klein_native.py

| Question | Answer |
|---|---|
| 1. Does `apply_neuron_patches()` move VAE to Neuron? | **No** (only if `vae_on_neuron=True`; default `False`) |
| 2. Is patch #9 (`_patched_decode` `z.to("cpu")`) deleted? | **No** — active in the default path |
| 3. Is `PatchedGroupNorm` applied? | **No** — never implemented |

By the handoff's decision tree, "any no → do Step 12 (~30s win)." But
that conclusion rests on a premise that's no longer true.

## Why the ~30s is already recovered (the stale premise)

Step 12's "~30s" is the cost of `prepare_image_latents` — the VAE
**encode + patchify + batch-norm** that the handoff's device-audit
table lists at **~24s** per image. That was the dominant Phase-2 tax.

**Image-latent caching (Phase A, shipped) already eliminated it.** In a
zoom/repeat-image session, the input image is encoded ONCE; every
subsequent inference is a cache hit (~0ms). That is *why* the shipped
number is 6.86s (now 5.97s), not 34s.

**Proof — cached steady-state per-stage breakdown (the re-profile Step 12
asks for):**
```
encode_prompt           0.6 ms   (cached)
prepare_latents        14.6 ms
scheduler.set_timesteps 0.3 ms
vae.decode           2048.9 ms   (channels_last)
prepare_image_latents     —      ← NOT PRESENT (cache hit, ~0ms)
(sum)                2064 ms
wall-clock           5.97 s
```

The "~28s pre-loop" the handoff warns about does NOT exist in the
shipped cached path. There is no 24s patchify to recover — caching
recovered it. Step 12's headline win is already banked, via a different
mechanism (cache, not device-move).

## What Step 12 would change *now*, and why it's a regression

With the 24s encode/patchify gone, the only thing left for Step 12 to
move is VAE **decode** (~2.0-2.9s). Both options were measured:

| VAE decode path | time | verdict |
|---|---:|---|
| CPU, default NCHW (old shipped) | 2.93 s | baseline |
| **CPU, channels_last (NEW, shipped)** | **2.05 s** | ✅ fastest |
| Neuron, per-block compiled (Step 12's ask) | 3.8 s | ✗ slower |

**Moving VAE decode to Neuron (Step 12) is a 1.8s regression vs the
CPU channels_last path.** At 1024² the decode is conv-bound; the host
CPU's mature NHWC conv kernels beat the 5-boundary-crossed per-block
Neuron NEFFs (a single fused VAE NEFF would be faster but hits the
NCC_IXTP002 10M-instruction compile limit).

The handoff author explicitly flagged this possibility ("Phase B may
have already moved the VAE — verify first; if so just confirm"). The
verification result is stronger than "already moved": moving it is
actively worse for this model, and caching already covers the
expensive encode half.

## The one valid Step 12 point: PatchedGroupNorm

Step 12 says `PatchedGroupNorm` is mandatory-for-correctness IF the VAE
runs on Neuron in bf16 (stock GroupNorm → NaN/washed-out). We are
keeping VAE on CPU, so it's not needed for the shipped path.

BUT: the optional `--vae-on-neuron` flag moves VAE to Neuron WITHOUT
PatchedGroupNorm. The earlier `--vae-on-neuron` full-pipeline output
measured std=14.56 (vs CPU 18.15) — that lower std is consistent with
exactly the bf16-GroupNorm degradation Step 12 warns about. So:

**Action taken:** mark `--vae-on-neuron` as experimental/not-recommended
in the runner help (it's both slower AND quality-degraded without
PatchedGroupNorm). The shipped, recommended path is CPU VAE +
channels_last + image-latent caching.

## Local-copy-vs-GitHub staleness check

The directive worried the local copy might be stale vs GitHub. Checked:
the push clone (= GitHub) and the local `.tmp/arminneuron-pr` copy are
the SAME on the three Step-12 questions — both have the conditional
`vae_on_neuron` flag (default off), both keep patch #9 for the default
path, neither has PatchedGroupNorm. Not stale; consistent.

## Bottom line

- Step 12's verification: VAE is on CPU by default (all 3 answers "no").
- Step 12's ~30s premise: already recovered by image-latent caching,
  NOT outstanding. Re-profile confirms no 24s pre-loop tax in the
  cached path.
- Step 12's prescribed fix (move VAE decode to Neuron): measured as a
  1.8s regression vs the channels_last CPU path just shipped.
- Net: **the port is functionally at Phase 3 for the hot path** — the
  only things on CPU are the encode (cached away), the decode (faster on
  CPU than Neuron here), text encode (handoff says leave it), and
  legitimate orchestration (scheduler/generator). Nothing scaling-with-
  work is needlessly on CPU in the steady state.

The shipped 5.97s is the correct, finished-port number for this model.

Box clean, no instance stopped.
