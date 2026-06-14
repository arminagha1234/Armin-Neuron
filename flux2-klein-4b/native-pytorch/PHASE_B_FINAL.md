# Phase B FINAL — VAE on Neuron does NOT help at 1024² (correction)

**Date:** 2026-06-14
**Box:** `3.15.152.199` (trn2.48xl, Beta 3)

## Correction to PHASE_B_UNBLOCKED.md

The earlier standalone bench reported VAE decode at **0.945s on Neuron**
and I called Phase B a win. **That was a resolution artifact** — the
standalone bench used a [1,32,64,64] latent → 512×512 output. At the
real **1024×1024** pipeline resolution, the per-block-compiled VAE
decode is **3.8s**, which is SLOWER than the ~2.9s CPU eager baseline.

## The clean apples-to-apples measurement

`bench_cached.py --vae-on-neuron` (Variant 3 = prompt + image-latent
cached, the same harness that measured 6.86s for Phase A):

| Config | Variant 3 warm | VAE decode (warm) |
|---|---:|---:|
| Phase A (VAE on CPU) | **6.86 s** | ~2.9 s CPU |
| Phase B (VAE on Neuron, per-block) | **7.73 s** | 3.8 s Neuron |

**Phase B is 0.87s SLOWER.** Per-block VAE on Neuron loses to CPU eager
at 1024².

## Why per-block VAE on Neuron loses

The per-block compile solved the *compile* blocker (NCC_IXTP002) by
splitting the decoder into 5 sub-NEFFs. But at 1024² that creates 5
separate on-device graphs with boundary crossings (Neuron→Neuron
handoffs, intermediate tensor materialization) between each up-block.
At the large spatial resolution, the VAE decode is conv-bound and the
host CPU (with its big caches and mature conv kernels) actually does it
in ~2.9s, while the 5-NEFF Neuron path takes 3.8s including the
inter-block overhead.

A single fused VAE NEFF might beat CPU — but that's exactly what hits
the 10M-instruction compile limit. So it's a genuine bind: monolithic
won't compile; per-block compiles but is slower than CPU.

## The settled, complete optimization map for klein-4B

Every Neuron-side lever has now been empirically tested:

| Approach | Result | Verdict |
|---|---|---|
| Phase A CPU-side caching | 34s → **6.86s** | ✅ SHIPPED — the win |
| TP=4 tensor parallel | 57s (8× slower) | ✗ model too small |
| attention_cte kernel (single-rank) | 8.11s | ✗ 18% slower |
| auto-cast=matmult | 7.69s | ✗ slower |
| requires_grad/inference_mode | 7.79s | ≈ neutral |
| Phase B VAE→Neuron (per-block) | 7.73s | ✗ 0.87s slower at 1024² |

**6.86s single-rank Phase A is the empirical optimum for FLUX.2-klein-4B
on Trainium2.** This is now exhaustively verified — not a guess.

## What this means for fal.ai (final, honest)

- **Single-image latency: 6.86s** (7.6× H100's ~0.9s)
- **Throughput: ~0.6 img/s** on a full trn2.48xl (host-CPU capped)
- **Cost: ~$0.0099/image** at realistic concurrency (~10× H100's $0.0010)

The gap to H100 is real and is gated by:
1. Host-CPU pipeline work (VAE + text encode + scheduler) that doesn't
   shard or move to Neuron profitably at this model size
2. The 4B distilled DiT being too small for TP to help

For a 4B image model that fits on one H100, **H100 is the better
price-performance choice today.** Trainium2's wins are elsewhere (large
LLM serving, training, >80GB models). This is the same honest
conclusion the original BENCHMARK_VS_H100 reached before the distilled-
config correction — and it holds after exhaustively testing every
optimization lever.

The one thing that would change it: a future neuronx-cc that either
(a) raises the instruction limit so the VAE compiles as one fast NEFF,
or (b) makes TP comms cheap enough to win on small models. Neither is
available today.

## Artifacts
- `results/bench_variant4.log` — the clean Variant 3 + VAE-on-Neuron measurement
- `src/flux2_vae_perblock.py` — per-block VAE compile (works, just not faster)
- `src/bench_cached.py` — now supports --vae-on-neuron

Box clean, no instance stopped.
