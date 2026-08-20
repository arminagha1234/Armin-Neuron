# B — TeaCache 54% skip (thresh 0.05, un-calibrated rescale)

- **Video:** `wan_tp4_tc0p05.mp4` (49 frames). Frames: `frame_tc0p05_0/mid.png`.
- **Measured:** 46/100 DiT forwards executed (**54% skipped**), pipe **50.4 s**, full e2e **~115 s**.
- **Fidelity:** latent-cos **0.837** vs exact — a coherent, on-prompt video but a *different, softer* sample; frame-std ~52 (some contrast loss).
- **Note:** ran with identity rescale (no calibrated polynomial). A fitted polynomial would give higher fidelity at this skip. Needs sign-off + calibration before use.
