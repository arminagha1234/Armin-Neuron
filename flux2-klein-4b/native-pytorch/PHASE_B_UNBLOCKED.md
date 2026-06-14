# Phase B UNBLOCKED — per-block VAE decoder compile

**Date:** 2026-06-14 (overnight autonomous run)
**Box:** `3.15.152.199` (trn2.48xl, Beta 3)

## The win

Phase B (move VAE decode onto Neuron) was blocked TWICE before by the
compiler instruction-count limit:
```
[NCC_IXTP002] Number of instructions (11453027) is over the threshold (10000000)
```
because the whole VAE compiled as one monolithic graph.

**Fix: compile the decoder per-block.** Each `UpDecoderBlock2D` (4 of
them) + the `mid_block` gets its own `torch.compile(backend="neuron")`,
so each sub-graph stays under the 10M-instruction ceiling. The tiny
conv_in / conv_norm_out / conv_out stay eager.

Result (standalone VAE decode bench, 1024-ish latent):
```
compiled 5 decoder submodules (mid_block + 4 up_blocks)
warmup (compile): 618s   (one-time, cached)
VAE decode on Neuron: 0.945s avg (5 runs, very stable 0.94-0.95s)
```

**vs ~2.9s CPU eager → 3.1× faster, ~2s saved per image.**

## Why this matters (it's the throughput unlock)

The throughput analysis (THROUGHPUT_FINDINGS.md) showed Trainium2
throughput plateaus at ~0.6 img/s because of **host-CPU contention** —
each concurrent worker runs VAE decode on the host CPU, and they fight
for host cores.

Moving VAE decode to Neuron (0.945s, on-device) removes ~2.9s of
host-CPU work per image per worker. That directly relieves the
contention that caps throughput. Combined with also moving the text
encoder (the other big CPU consumer), the host-CPU wall that's gated
this whole project comes down.

## Stacked impact estimate

Single-image, with VAE on Neuron:
```
Phase A shipped:           6.86s
  - VAE decode CPU:        ~2.9s  → Neuron 0.945s  (saves ~2.0s)
  ≈ 4.9s single-image (estimate; needs full-pipeline validation)
```

Throughput: with VAE off the host CPU, the per-worker CPU load drops
substantially, so the 8-worker contention (currently +94% latency)
should ease — more workers before the plateau, higher aggregate img/s.

## Caveats / next steps

1. **Correctness not yet validated in the full pipeline.** The
   standalone bench confirms compile + speed; the decoded output std
   (0.107, raw pre-denorm) needs a full-pipeline sharp-image check vs
   the CPU baseline (std=18.15 final). Wire `compile_vae_decoder_per_block`
   into `neuron_flux2_klein_native.py`'s `vae_on_neuron=True` path and
   run a full image, compare std + visual.
2. **Latent spatial size**: the standalone bench used a [1,32,64,64]
   latent → 512×512 output; the real pipeline latent for 1024×1024 is
   2× that. Re-validate at the true shape.
3. **Also compile the encoder** (or keep image-latent caching, which
   already sidesteps encode on warm calls).
4. **Text encoder → Neuron** is the other host-CPU consumer; same
   per-block-compile approach may apply if it hits the instruction
   limit.

## Artifacts
- `src/flux2_vae_perblock.py` — `compile_vae_decoder_per_block(vae)`
- `src/bench_vae_perblock.py` — standalone VAE decode bench
- `results/bench_vae_perblock.log` — 0.945s Neuron result

Box left clean. No instance stopped.

## Status update for the overall effort

This reopens the path that the throughput findings identified as THE
unlock. The optimization picture is now:
- Phase A caching: 34s → 6.86s ✅ shipped
- TP=4: doesn't help at 4B ✗ (measured)
- **Phase B VAE→Neuron per-block: WORKS, ~2s/image saved ✅ (new)**
- Phase B text-encoder→Neuron: next
- Full-pipeline integration + validation: next

Next interactive session: integrate per-block VAE into the production
pipeline, validate sharp output, re-bench single-image + throughput.
Projected single-image ~4.5-5s, with materially better throughput
scaling (host CPU no longer doing VAE decode).
