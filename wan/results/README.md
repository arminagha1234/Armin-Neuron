# Wan 2.2 TI2V-5B on one Trainium2 (trn2.3xl): results

End-to-end native-PyTorch pipeline (T5 → DiT → VAE) at 480×832, 49 frames, 50 steps, CFG×2 (100 DiT
forwards). Prompt: "a cat playing piano, cinematic." **All numbers are measured on-device.**

## Measured results

| Folder | Config | Full e2e (all on Neuron) | H100 target | Gap vs H100 | Fidelity |
|---|---|---|---|---|---|
| **A_exact_nocache** | TP=4, no cache | **69.3 s** (measured) | 33 s | **2.1×** | exact, correctness PASS |
| **B_teacache_54pct** | TP=4 + TeaCache 54% skip | ~53 s | 33 s | 1.6× | softer, different sample |
| **C_teacache_74pct** | TP=4 + TeaCache 74% skip | ~34 s | 33 s | ~1.0× (≈ parity) | hazy (un-calibrated) |

*All stages on Neuron (T5 0.704 s on-device). Gap = our time ÷ H100's 33 s. No-cache was re-run end-to-end
(69.3 s measured); TeaCache rows = measured DiT+VAE pipe (50.4 s / 31.4 s) + 0.704 s T5 + export. Exact
run split: T5 0.7 s + DiT 51.1 s + VAE 15.3 s.*

## Notes
- **Exact (A)** is the reference. **TeaCache** (step-skipping) trades fidelity for speed; these runs used
  an un-calibrated rescale, so quality drops with skip (softer at 54%, hazy at 74%). A calibrated
  polynomial recovers fidelity-per-skip.
- Correctness: decode parity (Neuron bf16 vs CPU fp32) PSNR 56.7 dB / SSIM 0.9994 → no visible difference.
- The clean path below H100 for this model is multi-chip (trn2.48xl).
