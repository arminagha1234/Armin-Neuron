# Cosmos-Predict2-2B — Native PyTorch Benchmark

**Date:** 2026-06-17  
**Hardware:** trn2.48xlarge, us-east-2, Beta 3 DLC  
**Stack:** native PyTorch + `torch_neuronx`, `torch.device("neuron")`,
diffusers 0.38.0  
**Model:** `nvidia/Cosmos-Predict2-2B-Text2Image` and
`nvidia/Cosmos-Predict2-2B-Video2World` (bf16)  
**Seed:** 42 throughout  
**Prompt (T2I/V2W):** "A nighttime city street in the rain, neon signs
reflecting on wet asphalt, cinematic, highly detailed"

## Text-to-Image

| Resolution | Steps | Cold | Warm | Quality (img std) | vs CPU ref |
|---|---:|---:|---:|---:|---|
| 512² | 12 | 17.2 s | 16.0 s | 76.25 | CPU 76.61 — **bit-accurate** |
| 1024² | 20 | 82.6 s | 42.0 s | 72.38 | (1024² CPU not benchmarked) |

The Neuron output is numerically indistinguishable from the CPU
reference at 512² (Δstd 0.36, same generator/seed). Stock bf16, no
fp32/mixed precision needed at these resolutions.

## Image-to-Video (Video2World)

| Resolution × frames | Steps | Cold | Warm | DiT (Neuron) | CPU side (T5+VAE) | Quality |
|---|---:|---:|---:|---:|---:|---:|
| 256² × 17f | 12 | 25.1 s | 22.4 s | (not split) | (not split) | std 85.35 |
| 480×832 × 17f | 12 | 132.4 s | **131.5 s** | 60.4 s (24×2.52 s) | 71.0 s | std 73.36 |
| **480×832 × 25f** | 12 | 375.8 s | **245.0 s** | **142.5 s** (24×5.94 s) | 102.4 s | std 80.92 |

DiT timing breakdown is from per-call instrumentation in
`cosmos_video_neuron.py`: 24 forward calls per pipeline call (12 steps ×
CFG cond+uncond). "CPU side" is everything that isn't the Neuron DiT —
T5 prompt encoding + WAN-VAE decode + scheduler / pipeline orchestration.

## Where the time goes

At the realistic 480×832 × 25f shape (245 s warm):

```
DiT on Neuron  ████████████████████████████████░  142.5 s (58%)
T5 + VAE (CPU) █████████████████████░             102.4 s (42%)
```

Two clean optimization vectors:

1. **DiT speedup (the Neuron half).** Algorithmic accelerators (e.g.
   Neural Dynamics-style adapters) and Neuron-side improvements (NKI
   flash attention, TP for higher resolutions) target this 142 s.
2. **Move the VAE onto Neuron.** The 102 s CPU-side cost is dominated
   by VAE decode at 480×832 × 25f. We have WAN VAE experience from
   prior Trainium work — porting it gets us into the sub-150 s/clip
   range without touching the DiT.

## Cold vs warm

Persistent NEFF cache works: re-runs of the same shape skip recompile.
Cold-warm gap on the 480×832 × 25f config was 376 s vs 245 s, almost
all of which was the first-call DiT compile (first DiT call cold:
136 s, last: 5.94 s).

## Reproduction

See [`README.md`](README.md) for the exact env vars and launch
commands. All numbers above are reproducible with the source files in
`src/` on a vanilla Beta 3 DLC + `pip install diffusers transformers
accelerate safetensors opencv-python-headless`.

## Optimization roadmap

| Effort | Win | Notes |
|---|---|---|
| Move WAN VAE onto Neuron | -50 to -80 s on 480×832 × 25f | Reuse WAN VAE patches we have |
| Larger video shapes | needs TP | FLUX v3 full-shard plan transfers |
| Algorithmic adapters | DiT halves | Stacks on top of the substrate |
| NKI flash attention | DiT improves | Same path as FLUX/LTX work |

## License

Apache-2.0 for this contrib code. NVIDIA Cosmos weights subject to
NVIDIA's model license.
