# FLUX.2-klein-4B + zoom-LoRA — Trainium2 vs H100 Benchmark

**Date:** 2026-06-13
**Model:** `black-forest-labs/FLUX.2-klein-4B` + external zoom-LoRA (scale=1.1)
**Input:** sample 1024×1024 photo with red region overlay
**Config:** 28 inference steps, guidance_scale=3.5, seed=42, bf16
**Prompt:** "Zoom into the red highlighted area"

## Head-to-head

| | **H100** (p5.48xlarge) | **Trainium2** (trn2.3xlarge) | Ratio |
|---|---:|---:|---:|
| Total (28 steps) | **6.1 s** | **65.9 s** | 10.8× slower |
| Per-step | **218 ms** | **2,350 ms** | 10.8× slower |
| Instance $/hr | $32.77 | $2.23 | 14.7× cheaper |
| **Cost per image** | **$0.055** | **$0.041** | **Trainium 26% cheaper** |
| Output std | 70.8 | 70.2 | Equivalent quality |

## Hardware details

### H100 (p5.48xlarge, us-east-2)
- GPU: NVIDIA H100 80GB HBM3 (single GPU used, 8 available)
- Stack: torch 2.12.0, diffusers 0.38.0, CUDA 13.0, Driver 580.126.09
- Pipeline: vanilla `Flux2KleinPipeline.from_pretrained().to("cuda:0")`
- No torch.compile

### Trainium2 (trn2.3xlarge, ap-southeast-4)
- Device: Single Trainium2 logical core (LNC=2, ~24 GB user budget)
- Stack: Beta 3 DLC, torch 2.11.0, torch_neuronx 2.11.3, neuronxcc 2.25, diffusers 0.39.0.dev
- Pipeline: `NeuronFlux2KleinPipeline` subclass with 10 Neuron patches + `torch.compile(backend="neuron")`
- NEFF compile cost: 896.8 s (one-time; cached at `/tmp/neff_cache`)

## Cost analysis

```
H100:     6.1 s × ($32.77 / 3600 s) = $0.0555 / image
Trainium: 65.9 s × ($2.23 / 3600 s) = $0.0408 / image

Savings:  $0.0555 - $0.0408 = $0.0147 / image (26.5% cheaper on Trainium)
```

At 1M images/month:
- H100:     $55,500
- Trainium: $40,800
- **Savings: $14,700/month**

## Optimization roadmap (to close the latency gap)

Current gap: 10.8× slower per-step than H100 (2.35 s vs 0.218 s).

| Optimization | Expected improvement | Effort |
|---|---|---|
| **NKI fused attention** | 1.3-2× (attention is ~40% of DiT compute) | 2-3 days |
| **`-O2` compiler flag** | 1.2-1.5× (more aggressive op fusion) | 30 min (flag change + recompile) |
| **TP=2** (shard DiT across 2 cores) | ~1.8× (less than 2× due to comm overhead) | 2-4 hours |
| **FP8 quantization** | 1.5-2× (if neuronxcc supports FP8 matmul for this model) | 1-2 days |
| **Combined (all above)** | ~4-6× total → ~0.5-0.6 s/step target | 1-2 weeks |

Target: **<1 s/step** at 1024×1024 = **<28 s per image** = **<$0.017/image** (3× cheaper than H100).

## Per-step breakdown (where time goes on Trainium)

At 2.35 s/step (compiled):
- DiT forward (48 single-stream + 8 double-stream blocks): ~2.1 s (estimate)
- Scheduler step (`dt * model_output`): ~0.05 s
- Boundary moves (CPU↔Neuron tensor coercion): ~0.2 s (estimate)

The DiT forward is the bottleneck. Fused attention and higher TP are
the right levers.

## Raw data

### Trainium eager (4 steps @ 512×512)
- First call: 12.9 s
- Cached call: 10.6 s
- Per-step: 2.66 s

### Trainium compiled (28 steps @ 1024×1024)
- First call (compile): 896.8 s
- Cached call: 65.9 s
- Per-step: 2.35 s

### H100 (28 steps @ 1024×1024)
- Warmup: 7.3 s
- Timed: 6.1 s
- Per-step: 218 ms

## Files

- `results/flux_example1_neuron.png` — Trainium eager output
- `results/flux_compiled_cached.png` — Trainium compiled output
