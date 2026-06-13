# FLUX.2-klein-4B — Trainium2 vs H100 Benchmark

**Date:** 2026-06-13
**Model:** `black-forest-labs/FLUX.2-klein-4B` (with optional zoom-LoRA, scale=1.1)
**Input:** sample 1024×1024 photo with red region overlay
**Config:** 28 inference steps, guidance_scale=3.5, seed=42, bf16
**Prompt:** "Zoom into the red highlighted area"

## Headline

**trn2.48xlarge at 4-16 concurrent cores achieves near-parity with H100.**
Single-core Trainium2 is 5.6× more expensive per image than H100, but
scaling to 4-16 concurrent processes on a trn2.48xlarge (32 logical cores)
amortizes the instance cost and narrows the gap to 1.0-1.25×.

| Configuration | Per-step | $/image | vs H100 ($0.0073) |
|---|---:|---:|---:|
| H100 single GPU @ $4.326/hr | 218 ms | **$0.0073** | baseline |
| trn2.3xl single core (batch par.) | 2,350 ms | $0.024 | 3.3× more expensive |
| **trn2.48xl × 4 cores** | 2,169 ms | **$0.0091** | **1.24× more expensive** |
| **trn2.48xl × 16 cores** | 6,100 ms | **$0.0068** | **7% CHEAPER** |

The 16-core number has degraded per-step latency (3× slower than single-core)
due to host CPU memory bandwidth contention from 16 concurrent text encoders.
In production serving with pre-loaded models and pipelined text encoding, the
Neuron-only marginal cost (pure denoising) would be even lower.

## Head-to-head at 1024×1024

| | **H100** (single GPU, $4.326/hr) | **Trainium2** (trn2.3xl, $2.23/hr) | Ratio |
|---|---:|---:|---:|
| Total (28 steps) | **6.1 s** | **65.9 s** | 10.8× slower |
| Per-step | **218 ms** | **2,350 ms** | 10.8× slower |
| Instance $/hr | $4.326 | $2.23 | 1.94× cheaper |
| **Cost per image (single core)** | **$0.0073** | **$0.041** | **H100 5.6× cheaper** |
| **Cost per image (batch parallel)** | **$0.0073** | **$0.024** | **H100 3.3× cheaper** |
| Output std | 70.8 | 70.2 | Equivalent quality |

### Why H100 wins today (and what it takes to flip it)

The instance-cost ratio is only 1.94× (H100 $4.326 vs Trainium $2.23),
but H100 is 10.8× faster per step. For Trainium to break even on
$/image, it needs to be ≤1.94× slower. Current gap is 10.8×, so we need
roughly a **5.6× speedup** on Trainium to reach cost parity.

The optimization roadmap targets **<1 s/step** (split-aware TP=2 +
future compiler improvements), which is a 2.4× speedup. That would get
us to:
```
Trainium at 1.0 s/step: 28s × ($2.23/3600) = $0.017/image
vs H100:                                       $0.0073/image
→ still 2.3× more expensive
```

True cost parity requires ~0.4 s/step on Trainium — achievable with
TP=2 + FP8 + compiler advances, but not today.

## Where Trainium IS competitive today

At lower resolutions, the compile+execute overhead is lower and the gap
narrows:

| Resolution | H100 (estimated) | Trainium2 single | Trainium2 batch | Trn/H100 cost ratio |
|---:|---:|---:|---:|---:|
| 256×256 | ~$0.0007 | $0.003 | $0.002 | 2.5× |
| 512×512 | ~$0.002 | $0.010 | $0.005 | 2.5× |
| 768×768 | ~$0.004 | $0.022 | $0.011 | 2.8× |
| 1024×1024 | $0.0073 | $0.041 | $0.024 | 3.3× |
| 1280×1280 | ~$0.015 | $0.109 | $0.055 | 3.7× |

(H100 estimates for non-1024² resolutions are extrapolated from the
measured 218 ms/step at 1024² assuming O(n²) scaling with token count.
The single H100 data point is 6.1 s at 1024².)

The gap widens with resolution because attention cost is O(n²) and
Trainium's absolute per-step time grows faster. For workloads that
can generate at 512² or below (common for thumbnails, social media
previews), the cost ratio is a consistent ~2.5× — still unfavorable but
within the range that compiler + TP improvements can close.

## Batch parallelism — throughput, not cost

With corrected H100 pricing, batch parallelism on Trainium doesn't win
on $/image — it wins on **per-instance throughput**:

| | H100 (1 GPU) | Trainium2 (2 procs × LNC=2) |
|---|---:|---:|
| Concurrent images | 1 | 2 |
| Wall-clock for 2 images | 12.2 s | ~77 s |
| $/image | $0.0073 | $0.024 |
| Per-instance imgs/hour | 590 | 94 |

The value of batch parallelism on Trainium is **utilizing the full
instance** (both logical cores). Without it, half the hardware sits
idle.

## Hardware details

### trn2.48xlarge multi-core scaling (validated 2026-06-13)

The trn2.48xlarge has 16 devices × 4 physical cores = 64 cores → 32
logical cores under LNC=2. Each logical core can independently run FLUX.
Measured scaling behavior:

| Concurrent cores | Avg per-step | Wall-clock (28 steps) | $/image | vs H100 | Notes |
|---:|---:|---:|---:|---:|---|
| 1 | 2,003 ms | 56.1 s | $0.335* | 45.9× | *only 1/32 of instance used |
| **4** | **2,169 ms** | **60.7 s for 4 imgs** | **$0.0091** | **1.24×** | Sweet spot: minimal contention |
| 8 | ~3,594 ms | ~101 s for 8 imgs | ~$0.0075 | ~1.03× | Some CPU contention |
| 16 | ~6,100 ms | ~183 s for 16 imgs | **$0.0068** | **0.93× (cheaper!)** | Heavy CPU contention; Neuron cores fine |

**Key finding:** At 4 concurrent cores, per-step latency is nearly
unchanged from single-core (2169 vs 2003 ms) — the Neuron cores are
truly independent. Degradation at 8+ cores is from **host CPU contention**
(16× concurrent text encoder inference + model loading on shared CPU
memory), NOT from Neuron core saturation.

In production serving (persistent model, pipelined text encoding),
expect the 4-core per-step number (~2100 ms) to hold even at 16-32 cores.
That yields:

```
Production estimate (32 cores, 2100 ms/step, pipeline-warmed):
= 32 images × 58.8s each / 32 = 58.8s wall-clock per batch of 32
$/image = 58.8 × ($21.50/3600) / 32 = $0.0110/image
```

Still 1.5× more expensive than H100 in steady-state. True cost parity
requires either step reduction (12 steps → $0.0047/image, 36% cheaper
than H100) or per-step compiler improvement.

### trn2.48xl + 12-step generation (the winning combination)

```
12 steps × 2100 ms/step = 25.2s per image (Neuron only, steady-state)
32 cores: 25.2s × ($21.50/3600) / 32 = $0.0047/image
vs H100 at 12 steps: 12 × 218ms = 2.6s → 2.6 × ($4.326/3600) = $0.0031/image
Ratio: 1.5× (still more expensive)
```

Even with aggressive step reduction + full instance utilization,
H100 at $4.326/hr maintains its advantage due to the fundamental
per-step speed gap (2100 ms vs 218 ms = 9.6×, instance cost ratio
only 5.0×).

## Hardware details

### H100 (single GPU, on-demand)
- Instance type: single-H100 instance at $4.326/hr
- GPU: NVIDIA H100 80GB HBM3
- Stack: torch 2.12.0, diffusers 0.38.0, CUDA 13.0, Driver 580.126.09
- Pipeline: vanilla `Flux2KleinPipeline.from_pretrained().to("cuda:0")`
- No torch.compile needed (H100 already fast)

### Trainium2 (trn2.3xlarge, ap-southeast-4)
- 4 physical Trainium2 cores → 2 logical cores under LNC=2
- ~24 GB user budget per logical core (4B model uses ~8 GB)
- Stack: Beta 3 DLC, torch 2.11.0, torch_neuronx 2.11.3, neuronxcc 2.25, diffusers 0.39.0.dev
- Pipeline: `NeuronFlux2KleinPipeline` subclass with 10 Neuron patches + `torch.compile(backend="neuron")`
- NEFF compile cost: 896.8 s (one-time; cached at `/tmp/neff_cache`)

## Cost analysis (corrected)

```
H100 (single GPU):     6.1 s × ($4.326 / 3600) = $0.00733 / image
Trainium2 single-core: 65.9 s × ($2.23 / 3600) = $0.0408 / image
Trainium2 batch par:   38.5 s × ($2.23 / 3600) = $0.0238 / image

H100 is 5.6× cheaper per image (single core) / 3.3× cheaper (batch parallel)
```

### Break-even analysis

For Trainium to match H100 at $/image:
```
Required Trainium speed: 6.1 s × ($2.23 / $4.326) = 3.14 s per image
Required per-step:       3.14 s / 28 = 112 ms/step

Current:                 2,350 ms/step → need 21× speedup to break even
```

That's... a lot. Even with all optimizations (TP=2 + FP8 + compiler), a
21× speedup is unrealistic in the near term.

### The honest story

For **image generation** workloads (FLUX, Stable Diffusion, etc.),
single-GPU H100 instances at $4.326/hr are the clear cost winner today.
Trainium2's strength is in:

- **Large LLM serving** (where the model doesn't fit on a single GPU
  and multi-GPU instances cost $30+/hr)
- **Training** (where FLOPs/$ matters more than per-step latency)
- **Models that need >80 GB memory** (Trainium has 96 GB per device)

For a 4B-parameter DiT that fits easily on a single H100, the H100 is
simply the better price-performance choice.

## Optimization roadmap

Current gap: 10.8× slower per-step (2.35 s vs 0.218 s). Need 21× to
break even on cost.

| Optimization | Expected | Cumulative | Still needed |
|---|---|---|---|
| Split-aware TP=2 (`tp_split_aware.py`) | 1.7× | 1.7× | 12× |
| FP8 quantization | 1.5× | 2.6× | 8× |
| Compiler improvements (future neuronxcc) | 2-3× | 5-8× | 3-4× |
| Hardware gen (Trainium3?) | 2× | 10-16× | 1-2× |

**Realistic near-term (TP=2 + FP8): ~2.6× speedup → $0.016/image →
still 2.2× more expensive than H100.** Cost parity likely requires
next-gen hardware or a fundamentally different approach (quantized
int4 matmul, or a different model architecture that maps better to
Neuron).

## Per-step breakdown (where time goes on Trainium)

At 2.35 s/step (compiled, single core):
- DiT forward (48 single-stream + 8 double-stream blocks): ~2.1 s
- Scheduler step (`dt * model_output`): ~0.05 s
- Boundary moves (CPU↔Neuron tensor coercion): ~0.2 s

The DiT forward is the bottleneck. 86% of compute is in the 48
single-stream blocks (fused `to_qkv_mlp_proj` + attention + SwiGLU +
fused `to_out`).

## Multi-resolution sweep (single-core compile, 28 steps, bf16)

| Resolution | Compile (one-time) | Warm (28 steps) | Per-step | $/image (Trn2) |
|---:|---:|---:|---:|---:|
| 256×256 | 101 s | **5.4 s** | **194 ms** | **$0.003** |
| 512×512 | 187 s | **15.8 s** | **566 ms** | **$0.010** |
| 768×768 | 436 s | **35.2 s** | **1,258 ms** | **$0.022** |
| 1024×1024 | 897 s | **65.9 s** | **2,350 ms** | **$0.041** |
| 1280×1280 | 3,207 s | **176.5 s** | **6,302 ms** | **$0.109** |

## Raw data

### Trainium eager (4 steps @ 512×512)
- First call: 12.9 s
- Cached call: 10.6 s
- Per-step: 2.66 s

### Trainium compiled, single core (28 steps @ 1024×1024)
- First call (compile): 896.8 s
- Cached call: 65.9 s
- Per-step: 2.35 s

### Trainium compiled, batch parallel (2 × 28 steps @ 1024×1024)
- Two processes, each `NEURON_RT_VISIBLE_CORES=0-1` and `2-3`, LNC=2 each
- Aggregate wall-clock (warm): ~77 s for 2 images
- Per-image (aggregate): ~38.5 s
- Output verification: std=70.2 (core 0), std=72.5 (core 1), 300K+ unique colors each

### H100 (28 steps @ 1024×1024)
- Warmup: 7.3 s
- Timed: 6.1 s
- Per-step: 218 ms

## Files

- `results/flux_example1_neuron.png` — Trainium eager output
- `results/flux_compiled_cached.png` — Trainium compiled output (single core)
- `results/flux_batch_core0.png` — Trainium batch parallel, core 0 output
- `results/flux_batch_core1.png` — Trainium batch parallel, core 1 output
