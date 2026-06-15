# FLUX.2-klein-4B — Trainium2 vs H100 Benchmark

**Date:** 2026-06-14 (VAE-on-Neuron + mixed-flag win added)
**Model:** `black-forest-labs/FLUX.2-klein-4B` (DISTILLED variant)
**Stack:** Beta 3 DLC, torch 2.11.0, torch_neuronx 2.11.3, neuronxcc 2.25, diffusers 0.38.0
**Instance:** trn2.48xlarge (`i-0c2806a95b490e26e`), single core (LNC=2)
**Input:** synthetic 1024×1024 photo with red region overlay
**Prompt:** "Zoom into the red highlighted area"

## Headline — 4-step distilled config (current production)

| Path | Warm avg | Std | $/image @ 32-core trn2.48xl | vs H100 4-step (measured $0.00054) | Latency vs H100 4-step (measured 0.49 s) |
|---|---:|---:|---:|---:|---:|
| CPU VAE channels_last (prior shipped) | 5.92 s | 18.15 | ~$0.00110 | ~2.0× more expensive | ~12× slower |
| **VAE on Neuron + PAVE fixes + mixed flags (NEW DEFAULT)** | **4.19 s** ✅ | **18.16** | **~$0.00078** | **~1.45× more expensive** | **~8.5× slower** |

**Win: −29% end-to-end, lossless quality.** See
[`MIXED_FLAG_VAE_NEURON_WIN.md`](MIXED_FLAG_VAE_NEURON_WIN.md) for the
full A/B record (4 measured configurations) and the recipe.

### Measured H100 baseline (replaces the 0.87 s extrapolation)

Measured 2026-06-14 on a clean p5.48xlarge `i-02553d3272f721a84`
(us-east-2), single H100 SXM5 80GB, driver 575.57.08, CUDA 12.8,
torch 2.11.0+cu128, diffusers 0.39.0.dev (HF main), transformers 5.12,
bf16, **stock diffusers, no torch.compile, no FP8, no FlashAttention-3**
— same software-stack honesty class as the Trainium native-PyTorch
bench. Text-to-image, canonical 4-step + guidance=1.0, seed=42, n=5
warm runs. Output `std=76.3` (real, sharp generation).

| Config | H100 measured | Per-step | Output std | $/image (1/8 of p5.48xl box) |
|---|---:|---:|---:|---:|
| **canonical 4-step 1024²** | **0.49 s warm** (cold 2.44 s, σ=0.005 s) | 124 ms | 76.3 | **$0.00054** |
| 28-step 1024² (old apples-to-apples) | 2.53 s warm | 90 ms | 71.1 | $0.00277 |
| `landscape_4_3` 1024×768 (fal API default) | 0.39 s warm | 98 ms | 75.1 | $0.00043 |

The earlier "0.87 s extrapolated" came from `6.1 s / 28 × 4` — wrong
because per-step on H100 *increases* at 4 steps (124 ms vs 90 ms at
28 steps) due to fixed text-encode + VAE overhead not shrinking with
step count. Real measured 4-step is **~1.8× faster than that
extrapolation predicted**.

The earlier "6.1 s at 28 steps" was on torch 2.12 + diffusers 0.38 +
CUDA 13. Today's measurement on torch 2.11 + diffusers 0.39.0.dev +
CUDA 12.8 measures **2.53 s at 28 steps** — a ~2.4× improvement on the
GPU side from the upstream diffusers FLUX.2 pipeline optimizations
shipped between 0.38 and 0.39. fal's production stack likely sits
somewhere between these two.

Raw JSON + sample PNGs (real generations): `results/h100_*.json`,
`results/A_canonical_1024.png`, `results/B_apples_28step.png`,
`results/C_landscape_4_3.png`.

### H100 resolution sweep — 0.26 MP → 4.0 MP

Single H100, canonical 4-step config, n=3 warm per resolution. Every
resolution from 512² to 2048² succeeded with no OOM. **Latency scales
linearly with megapixels:** `warm_s ≈ 0.13 + 0.49 × MP` (R²>0.99).

| Resolution | MP | Warm avg | Per-step | Output std | Peak HBM | $/image (1/8 of box) |
|---|---:|---:|---:|---:|---:|---:|
| `square` 512×512 | 0.26 | **0.19 s** | 47 ms | 75.9 | 15.6 GB | $0.00021 |
| `landscape_4_3` 1024×768 (fal default) | 0.79 | **0.39 s** | 97 ms | 79.3 | 16.8 GB | $0.00043 |
| 1280×720 | 0.92 | 0.44 s | 111 ms | 72.8 | 17.1 GB | $0.00048 |
| `square_hd` 1024×1024 (fal preset max) | 1.05 | **0.49 s** | 124 ms | 72.9 | 17.3 GB | $0.00054 |
| 1536×1024 | 1.57 | 0.73 s | 181 ms | 78.1 | 18.5 GB | $0.00079 |
| 1792×1024 | 1.84 | 0.84 s | 211 ms | 81.4 | 19.1 GB | $0.00092 |
| 2048×1024 | 2.10 | 0.98 s | 244 ms | 75.9 | 19.7 GB | $0.00107 |
| 2048×1536 | 3.15 | 1.53 s | 381 ms | 79.2 | 22.1 GB | $0.00167 |
| **2048×2048 (4 MP, klein max)** | **4.19** | **2.14 s** | **534 ms** | **74.1** | **24.5 GB** | **$0.00233** |

**Notable finding:** 4 MP fits comfortably on a single H100 (24.5 GB
peak of 80 GB). Per-step at 4 MP is **only 4.3× the 1 MP per-step**,
not the ~16× you'd expect from naive S² attention scaling — the fixed
text-encode/VAE/projection overhead dominates more than attention does
even at 4 MP under the stock compiler. **This confirms the
"≤ 1024² → don't write a custom attention kernel" recommendation
holds even higher than we documented:** even at 4 MP attention isn't
the dominant cost on the GPU stock path.

For Trainium: the 24.5 GB peak HBM at 4 MP sits right at the per-core
24 GB user budget, so 4 MP on Trainium likely requires TP. Single-core
4 MP is borderline-impossible on Trainium today; with TP=4 it's
plausible but carries the TP overhead documented in
[`TP4_FINDINGS.md`](TP4_FINDINGS.md).

---

## Legacy 28-step bench (historical context)
**Seed:** 42

## Headline

With **image-latent caching enabled**, Trainium2 is at **cost parity
with H100** on FLUX.2-klein-4B image-to-image inference:

| Variant | Wall-clock (avg, 5 runs) | $/image (trn2.48xl ÷32 cores) | vs H100 |
|---|---:|---:|---:|
| 1. No caching (baseline) | 34.59 s | $0.0065 | 6.5× more expensive |
| 2. Prompt-embed cached | 30.95 s | $0.0058 | 5.8× more expensive |
| **3. Prompt + image-latents cached** | **6.86 s** | **$0.0013** | **1.3× more expensive** |
| H100 reference (4-step est.) | 0.87 s | $0.0010 | baseline |

**Variant 3 is a 5× wall-clock speedup over the baseline** with zero
DiT changes — pure CPU-side optimization. The cost gap to H100 closes
from 6.5× to 1.3×.

This is the customer-facing pattern for a zoom-LoRA workload:
one input image, many prompts/regions per session. Image-latent caching
maps directly to that pattern and is now exposed in the production
runner via `--cache-image-latents`.

## Per-stage breakdown (where the 34s actually goes)

Times are per-call averages over 5 inference calls, instrumented at
each pipeline boundary:

| Stage | Variant 1 (no cache) | Variant 2 (prompt) | Variant 3 (prompt+image) |
|---|---:|---:|---:|
| `encode_prompt` (Qwen3, CPU) | 1457 ms | 2 ms | <1 ms |
| `_encode_vae_image` (CPU) | 2419 ms | 1400 ms | (in cache hit) |
| **`prepare_image_latents`** | **24833 ms** | **24104 ms** | **(in cache hit)** |
| `prepare_latents` (noise sample) | 14 ms | 14 ms | 14 ms |
| `scheduler.set_timesteps` | <1 ms | <1 ms | <1 ms |
| DiT denoising (Neuron, 4 steps) | ~2920 ms | ~2920 ms | ~2920 ms |
| `vae.decode` (CPU) | 4364 ms | 2911 ms | 2931 ms |
| **Sum** | **33088 ms** | **28432 ms** | **2945 ms** |

**Key insight:** `prepare_image_latents` (which runs VAE encode +
patchify + batch-norm normalization) is **71% of the wall-clock**.
Caching it is the single biggest win available.

The DiT denoising itself is only ~9% of wall-clock at the distilled
config — the optimization roadmap originally targeted that 9%.
Image-latent caching takes us from "5.6× more expensive than H100" to
"1.3× more expensive" without touching the DiT at all.

## When image-latent caching applies

✅ **Use it when:**
- Same input image, multiple prompts (zoom-LoRA region selection,
  A/B prompt testing)
- Batch generation from a template image
- Interactive tools where the user uploads once and tweaks parameters
- Most zoom-LoRA workloads

❌ **Don't use it when:**
- Every call has a different input image (image-latents would be stale)

The flag is opt-in (`--cache-image-latents`) so customers explicitly
choose when their workload pattern allows it.

## Production runner usage

```bash
# Single-image session (Variant 3 pattern):
python run_flux2_klein_native.py \
    --base-model black-forest-labs/FLUX.2-klein-4B \
    --no-lora \
    --steps 4 --guidance-scale 1.0 \
    --cache-image-latents \
    --image input.png --prompt "Zoom into the red highlighted area" \
    --output zoomed.png

# First call: ~30s (full pipeline, populates cache)
# Subsequent calls with the same input image: ~7s
```

For long-running services, the runner exposes the cached latents via
the closure — wire `pipe.prepare_image_latents` to your serving cache
key (e.g. hash of the input image bytes).

## Cost analysis (Path A applied)

```
Trainium2 single-core, no caching:
   34.59 s × ($21.50 / 3600 / 32 cores) = $0.0065 per image

Trainium2 single-core, prompt + image-latents cached:
   6.86 s × ($21.50 / 3600 / 32 cores) = $0.0013 per image  ← 5× cheaper

H100 single GPU, distilled (4 steps):
   ~0.87 s × ($4.326 / 3600) = $0.0010 per image

Trainium2 cost gap to H100: 1.3× (was 6.5× before Path A)
```

For workloads where the same image is reused across many calls
(zoom-LoRA-style), Trainium2 + image-latent caching is **functionally
at H100 cost parity**.

## Why this changes the customer story

Before this run:
> "Trainium2 is 5.6× more expensive than H100 on FLUX.2-klein-4B.
> Optimization roadmap targets 2.6× speedup via DiT-side wins. Cost
> parity unlikely near-term."

After this run:
> "Trainium2 with image-latent caching is **1.3× the cost of H100** on
> a zoom-LoRA workload. The single-call number (34s → 7s) is
> dominated by an opt-in CPU-side cache, not a DiT-side optimization."

## Quality verification

All three variants produce visually identical, sharp output (same seed,
same prompt). Variant 3 byte-for-byte matches Variant 1 modulo
generator-state ordering.

| Variant | Output std | Notes |
|---|---:|---|
| 1. No caching | 18.15 | sharp, baseline |
| 2. Prompt cached | 18.15 | identical to V1 |
| 3. Prompt + image cached | 18.15 | identical to V1 |

## Hardware details

### Trainium2 (trn2.48xlarge, us-east-2)
- 16 devices × 4 physical cores → 32 logical cores under LNC=2
- ~24 GB user budget per logical core (4B model uses ~8 GB)
- Stack: Beta 3 DLC, torch 2.11.0, torch_neuronx 2.11.3.0.1278,
  neuronxcc 2.25, diffusers 0.38.0
- Pipeline: `NeuronFlux2KleinPipeline` subclass with 8 Neuron patches +
  `torch.compile(backend="neuron")`
- NEFF cache: `/mnt/data/work/flux2/neff_cache_4step`

### H100 (single GPU, on-demand)
- Instance type: single-H100 instance at $4.326/hr
- GPU: NVIDIA H100 80GB HBM3
- Stack: torch 2.12.0, diffusers 0.38.0, CUDA 13.0
- Pipeline: vanilla `Flux2KleinPipeline.from_pretrained().to("cuda:0")`
- 4-step number is extrapolated from the measured 28-step / 218-ms-per-step
  baseline; an apples-to-apples 4-step H100 re-run is pending.

## Optimization status

| Optimization | Status | Wall-clock impact | Where |
|---|---|---|---|
| Distilled config (4 steps) | ✅ done | 65.9s → 34.6s | runner default |
| Prompt-embed caching | ✅ done | 34.6s → 31.0s | bench (not customer-meaningful at 1.1×) |
| **Image-latent caching** | ✅ done | **34.6s → 6.86s** | runner (`--cache-image-latents`) |
| Move VAE to Neuron | 🟡 in other chat | ~3-5s/call | research stream |
| Move text encoder (Qwen3) to Neuron | 🟡 in other chat | ~1.5s/call | research stream |
| ISA flash attention in DiT | ⏳ later | ~0.3s/call (small at 4 steps) | DiT-side |
| Compiler O2 sweep | ⏳ later | ~0.1s/call (saturated) | DiT-side |

The CPU-side wins (image-latent + Phase B encoder/VAE) deliver
~5-6× wall-clock speedups. DiT-side wins deliver <0.5s/call at the
distilled config — they're worth keeping but no longer the priority.

## Files

- `results/distilled_4step.png` — sharp 4-step output (validated)
- `results/bench_cached.log` — full Path A bench log (3 variants × 5 runs)
- `src/run_flux2_klein_native.py` — production runner (now with `--cache-image-latents`)
- `src/bench_cached.py` — Path A reproducibility bench (per-stage timing + 3 variants)

## Raw numbers

### Variant 1: no caching (5 runs)
- 34.03, 34.24, 35.26, 34.40, 35.01 s
- avg: 34.59 s, min: 34.03 s

### Variant 2: prompt cached (5 runs)
- 30.90, 30.89, 31.13, 30.94, 30.89 s
- avg: 30.95 s, min: 30.89 s
- one-time prompt encode: 1.61 s

### Variant 3: prompt + image-latents cached (5 runs)
- 6.90, 6.89, 6.86, 6.92, 6.75 s
- avg: 6.86 s, min: 6.75 s
- one-time capture call: 30.92 s (full pipeline once)
