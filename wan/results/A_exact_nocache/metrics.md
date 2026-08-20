# A — Exact output (TP=4, no cache)

- **Video:** `wan_tp4.mp4` (49 frames, 832×480, 16 fps). Frames: `frame_nocache_0/mid.png`.
- **Measured e2e:** DiT denoise 51.1 s (100 fwd @ 511 ms) + VAE 15.5 s = **68.9 s (DiT+VAE)**;
  + T5-on-CPU 65 s = **~134 s full**. With T5 on-device (0.34 s), full e2e → **~69 s**.
- **Correctness:** decode parity vs CPU-fp32 PSNR 56.7 dB, SSIM 0.9994, 49/49 frames, no NaN → **PASS**.
- **vs H100 target (33 s):** DiT+VAE ≈ 2.1×; full e2e ~2.1× once T5 is on-device.
- Fidelity: exact — reference for the TeaCache comparisons.
