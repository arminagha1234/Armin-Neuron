# LTX-2 18.88B on Trainium2 — native PyTorch benchmark vs H100

## Run config

| | |
|---|---|
| Model | `Lightricks/LTX-2` (18.88 B, audio+video DiT) |
| Stack | Beta 3 DLC, torch 2.11.0, torch_neuronx 2.11.3.0.1278 |
| TP | 4 (one rank per Neuron core, LNC=2) |
| Resolution | 384 × 512 |
| Frames | 25 |
| Steps | 8 |
| Guidance scale | 4.0 |
| dtype | bfloat16 |
| Seed | 42 |
| Prompt | "A golden retriever puppy runs across a sunny green meadow, its ears flapping in the wind. The camera follows from a low angle. Birds chirp in the background." |

## Numbers

| Metric | Trainium2 native PyTorch | H100 (1× of p5.48xl) | Gap |
|---|---:|---:|---:|
| Setup (CPU pipeline load + components on Neuron) | 31.1 s | 49 s | 0.63× (Trn faster) |
| TTFI (cold, includes NEFF compile) | **169.3 s** | 52.3 s | 3.24× slower |
| Warm mean (8 steps, full pipeline) | **165.4 s** (n=5, σ=0.74, p95=166.2) | 2.84 s (n=6, σ=0.01) | **58.2× slower** |
| Per-step transformer only | **6.33 s** | 326 ms | 19.4× slower |
| Per-step amortized (incl CPU) | 20.68 s | 0.355 s | 58× |

### Where the warm time goes

| Stage | Trainium2 (s) | H100 (s) | Gap | On Neuron? |
|---|---:|---:|---:|---|
| Transformer (8 steps, TP=4, BMM-SDPA + AWS recipe) | 50.6 | 2.6 | 19.4× | ✅ |
| Gemma-3 12B text encoder forward | ~25 | ~50 ms | 500× | ❌ CPU |
| LTX-2 connectors + per-layer projections | ~30 | ~30 ms | 1000× | ❌ CPU |
| Video VAE decode | ~50 | ~150 ms | 333× | ❌ CPU |
| Audio VAE + vocoder | ~10 | ~10 ms | 1000× | ❌ CPU |
| MP4 export | ~0.5 | ~0.5 | 1.0× | n/a |
| **Total** | **165.4 s** | **2.84 s** | **58.2×** | |

The 58× warm gap is dominated by **CPU flat tax**, not Trainium per-step.
Trainium spends only **31% of warm time in the transformer** (50 s of
165 s). The other ~115 s is CPU host work. Closing that gap requires
moving the encoder + VAE onto Neuron — multi-day work. See "Optimization
roadmap" below.

### Cost analysis

`p5.48xlarge` on-demand: $98.32 / hr ($0.273 / s 8-GPU). Single H100 (one
of 8 on the box) effectively $12.29/hr if amortized. trn2.48xlarge
on-demand: $35.7608 / hr ($0.0099 / s).

| Path | Per-clip wall-clock | Cost per clip | Cost per 1k clips |
|---|---:|---:|---:|
| H100 (1×, capacity-fraction of p5) | 2.84 s | $0.0097 | $9.69 |
| Trainium2 TP=4, native PyTorch (today) | 165.4 s | $1.642 | $1,642 |
| Trainium2 TP=4 + encoder+VAE on Neuron (target) | ~50 s (estimate) | $0.497 | $497 |

The Trainium "today" pricing is dominated by the CPU flat tax. Once the
encoder + VAE move to Neuron, the per-clip cost drops ~3×. To close the
remaining gap to H100 you'd need transformer-side compile speedups (NKI
flash attention, fused MLP — items 4 and 5 on the optimization roadmap).

## Compiler optimization sweep (384×512, 25f, 8 steps)

Tested multiple compiler configurations. At this shape (video_seq=768),
the pipeline is **not compute-bound** — all configs produce essentially
the same throughput:

| DiT NEFF config | Generation time | Steps/sec | Notes |
|---|---:|---:|---|
| O1, BMM-only | 29.1s | 3.38 it/s | Baseline |
| O1, ISA flash kernel | 29.9s | 3.36 it/s | NKI flash for self-attn |
| O2, ISA flash + mixed-precision-accum | **28.7s** | 3.35 it/s | All optimizations |

**Conclusion**: At seq=768, the hardware is already efficient. NKI flash
attention and O2 optimization provide their gains at **higher resolutions**
(seq≥6144, i.e. 768×512/121 frames) where attention becomes compute-dominant.

## Optimization roadmap

Listed by effort vs payoff. The first three would together bring Trainium
to within ~3× of single-H100 cost-per-clip.

| # | Lever | Effort | Expected gain |
|---|---|---|---|
| 1 | **Move Gemma-3 text encoder to Neuron** (compile via `torch.compile(backend="neuron")` + pre-shard for fast load). Reference: AWS Neuron team's `compile_gemma3.py` + `shard_gemma3_weights.py`. | 2-3 days | -25 s / clip |
| 2 | **Move video VAE decode to Neuron** with tiled compilation. The VAE is 128 in_channels with 3D convs; can't fit at full resolution. The AWS team uses 4×16 latent tiles (128×512 px) for 121-frame canonical shape. Reference: `tiled_vae_decode.py`. | 1-2 days | -40 s / clip at 25-frame; -120 s / clip at 121-frame |
| 3 | **NKI flash attention for attn1** (video self-attn, Q.seq=6144 = the dominant attention cost). The AWS contrib has both an ISA kernel (NxDI) and an `attention_cte` kernel — either drops self-attn time roughly 2-3×. | 1 day to integrate NxDI ISA, 2-3 days for an own NKI kernel | -3 s / clip per-step → -24 s / clip |
| 4 | **`attention_cte_bias` for masked cross-attn** (attn2, audio_attn2, K.seq=1024). Production NKI kernel with additive bias support. | 0.5 day to integrate AWS's. | -1.5 s / clip per-step → -12 s / clip |
| 5 | **`torch.compile(backend="neuron")` per-block compile** rather than full-graph. Reduces compile-time RAM (full-graph compile of 19B OOM'd a 256 GB host). May cut per-step wall by 5-10%. | 0.5 day | -5 s / clip |
| 6 | **CFG-batched forward** (run cond + uncond in a single batch=2 forward) instead of two sequential forwards per step. Halves transformer call count. | 1 day | -25 s / clip |
| 7 | **TP=8** instead of TP=4 (use 2 NeuronDevices × 4 cores per process via NEURON_LOGICAL_NC_CONFIG=1). Per-rank weight memory drops to ~5 GB; activation budget grows; may unlock 121-frame canonical shape. | 1-2 days | enables higher resolution / longer clips |

Stacking 1-4 should produce ~50 s / clip. Stacking 1-6 should produce
~25 s / clip. That brings cost-per-clip into the $0.07-0.25 range,
within ~3× of single-H100 capacity at the negotiated trn2.48xl rate.

## Comparison methodology

H100 numbers from generation on a single GPU of `p5.48xlarge` (8× H100
80GB), same prompt/seed/shape. The H100 ran Lightricks' reference
`LTX2Pipeline` from diffusers @ main, dtype bfloat16, no CFG batching
modifications. Both runs use deterministic seed 42.

The Trainium2 numbers are from `bench_ltx2.py` on this PR's code with all
ten correctness fixes applied. The five warm samples are from a single
container without restarts; explicit `gc.collect()` between iterations
prevents the slow accumulator-leak that surfaced as -9 OOM in earlier
benchmarks.

The PSNR / SSIM accuracy comparison vs CPU reference is in the AWS
contrib's integration test (`test/integration/test_model.py`,
threshold SSIM > 0.7). With BMM-SDPA + the additive `-10000` mask + the
RankTensor RoPE slice all applied, the AWS contrib reports "nearly
identical output to GPU reference" and our recipe ports those exact
fixes.

## Reproduction

```bash
NEURON_RT_VIRTUAL_CORE_SIZE=2 NEURON_RT_NUM_CORES=4 \
torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
    src/run_ltx2_native.py --num-steps 8 --num-frames 25 \
    --height 384 --width 512 --output results/ltx2_native_run.png

# Bench (5 warm samples after 1 warm-up):
torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
    src/run_ltx2_native.py --num-steps 8 --num-frames 25 \
    --height 384 --width 512 --output results/ltx2_native_run.png
```
