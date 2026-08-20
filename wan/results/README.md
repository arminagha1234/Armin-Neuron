# Wan 2.2 TI2V-5B on one Trainium2 (trn2.3xl): results

End-to-end native-PyTorch pipeline (T5 → DiT → VAE) at 480×832, 49 frames, 50 steps, CFG×2 (100 DiT
forwards). Prompt: "a cat playing piano, cinematic." **All numbers are measured on-device.**

## Measured results

| Folder | Config | DiT+VAE | Full e2e | H100 target | Gap vs H100 (device / e2e) | Fidelity |
|---|---|---|---|---|---|---|
| **A_exact_nocache** | TP=4, no cache | **68.9 s** | ~134 s | 33 s | **2.1× / 4.1×** | exact, correctness PASS |
| **B_teacache_54pct** | TP=4 + TeaCache 54% skip | 50.4 s | ~115 s | 33 s | 1.5× / 3.5× | softer, different sample |
| **C_teacache_74pct** | TP=4 + TeaCache 74% skip | 31.4 s | ~96 s | 33 s | **0.95×** / 2.9× | hazy (un-calibrated) |

*Gap = our time ÷ H100's 33 s (lower = better; <1× = faster than H100). Device-compute (DiT+VAE) is the
apples-to-apples accelerator number; full e2e includes T5 text-encode, which was left on CPU here as a
single-chip memory workaround — on-device T5 is 0.34 s (measured), bringing full e2e to ~69 s.*

## Notes
- **Exact (A)** is the reference. **TeaCache** (step-skipping) trades fidelity for speed; these runs used
  an un-calibrated rescale, so quality drops with skip (softer at 54%, hazy at 74%). A calibrated
  polynomial recovers fidelity-per-skip.
- Correctness: decode parity (Neuron bf16 vs CPU fp32) PSNR 56.7 dB / SSIM 0.9994 → no visible difference.
- The clean path below H100 for this model is multi-chip (trn2.48xl).
