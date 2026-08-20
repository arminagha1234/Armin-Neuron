# C — TeaCache 74% skip (thresh 0.10, un-calibrated rescale)

- **Video:** `wan_tp4_tc0p1.mp4` (49 frames). Frames: `frame_tc0p1_0/mid.png`.
- **Measured:** 26/100 DiT forwards executed (**74% skipped**), pipe **31.4 s** (< H100 33 s on device compute), full e2e **~96 s**.
- **Fidelity:** latent-cos **0.744** vs exact — coherent but noticeably hazy, dynamic range fades over the clip (frame-std 52→36). NOT recommended without a calibrated rescale.
- **Note:** the only config whose DiT+VAE pipe dips under 33 s, but at meaningful quality cost with the un-calibrated rescale. Calibration is the missing piece for high-skip + high-fidelity.
