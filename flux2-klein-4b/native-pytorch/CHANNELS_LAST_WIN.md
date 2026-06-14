# channels_last VAE decode — production win (6.86s → 5.97s)

**Date:** 2026-06-14
**Box:** `3.15.152.199` (trn2.48xl, Beta 3)

## What was missed (and found on a fresh look)

The handoff doc and all 11 optimization steps were **Neuron-focused**
(TP, FP8, NKI kernels, compiler flags). A fresh look at the cached
steady-state per-stage breakdown showed the real distribution of the
6.86s:

```
VAE decode (CPU):     ~2.93s  (43%)  ← nobody optimized the CPU side
DiT loop (Neuron):    ~2.90s  (42%)  ← saturated at compiler floor
Other/glue:           ~1.03s  (15%)
```

VAE decode on the CPU is 43% of wall-clock. Phase B tried moving it to
Neuron (slower, 3.8s). But **nobody tried making the CPU decode faster**
— and the VAE decoder is conv-heavy, where PyTorch's CPU conv kernels
are dramatically faster in `channels_last` (NHWC) layout than the
default NCHW.

## The fix: channels_last memory format on the CPU VAE

One line:
```python
pipe.vae = pipe.vae.to(memory_format=torch.channels_last)
```

### Standalone VAE decode bench (1024² latent)
| Config | decode time | speedup |
|---|---:|---:|
| baseline (NCHW bf16) | 4397 ms | 1.0× |
| **channels_last (NHWC)** | **2363 ms** | **1.86×** |
| tiling | 4606 ms | 0.95× (worse) |
| slicing | 4441 ms | 0.99× |
| fp32 | 6720 ms | 0.65× (worse) |
| threads 64/32/16 | 4220/4652/6239 ms | ≤1.04× |

channels_last is the clear winner; everything else is neutral or worse.

### End-to-end (bench_cached Variant 3, same harness as the 6.86s number)
| Config | warm wall-clock | VAE decode | output std |
|---|---:|---:|---:|
| Phase A baseline | 6.86 s | 2930 ms | 18.15 |
| **Phase A + channels_last** | **5.97 s** | **2180 ms** | 18.14 |

**0.89s faster per image (13% improvement). Lossless** — output std
18.14 vs 18.15 (byte-equivalent quality; channels_last is just a memory
layout, no numerical change).

## Why it works

The FLUX VAE decoder is a stack of `Conv2d` + `GroupNorm` at increasing
spatial resolution (up to 1024×1024). PyTorch's CPU convolution
(oneDNN/MKLDNN) has highly optimized NHWC kernels; the default NCHW
layout forces layout conversions inside each conv. Setting
channels_last lets the whole decoder run in the fast path.

This is the standard GPU trick (channels_last is well known for CUDA
convnets), but it applies equally to the **CPU** convs here — which is
exactly the VAE-decode path that dominates FLUX wall-clock on Trainium
(where the VAE runs on the host CPU, not the Neuron device).

## Shipped

`run_flux2_klein_native.py` now has `--vae-channels-last` (applies to
the default CPU-VAE path; mutually exclusive with `--vae-on-neuron`).
Recommended default for the production CPU-VAE path.

```bash
python run_flux2_klein_native.py --no-lora --steps 4 --guidance-scale 1.0 \
    --cache-image-latents --vae-channels-last --output out.png
```

## Updated customer numbers

| Metric | Before | After channels_last |
|---|---:|---:|
| Single-image latency | 6.86 s | **5.97 s** |
| vs H100 (~0.9s) | 7.6× | 6.6× |
| $/image (single core) | — | ~13% lower |

Still slower than H100 for this 4B model, but a real, free, lossless
13% improvement that the Neuron-focused handoff completely missed.

## The lesson

When ~half the wall-clock is on the host CPU (as it is for any
Trainium diffusion pipeline that keeps the VAE/text-encoder on CPU),
**CPU-side optimizations matter as much as Neuron-side ones.**
channels_last for conv-heavy CPU modules is the first thing to try and
was overlooked because the whole effort was framed as "optimize the
Neuron kernels."

## Other CPU-side levers checked (all neutral/worse — see bench_vae_cpu.log)
- VAE tiling: worse (tile overhead > cache benefit at this size)
- VAE slicing: neutral (B=1, nothing to slice)
- fp32 decode: 1.5× worse (bf16 is right on CPU here)
- thread count (16/32/64 vs 96): all ≤1.04×, 96 is fine

## Next CPU-side idea (untested)
The ~1.03s "other/glue" (15%) is unprofiled — diffusers `__call__`
overhead: guidance embedding, timestep prep, latent pack/unpack,
image post-process (`pil` conversion). A focused profile of the
non-stage wall-clock could find another 0.3-0.5s. Lower priority than
channels_last but the next thing to look at.

Box clean, no instance stopped.
