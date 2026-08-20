# A — Exact output (TP=4, no cache)

- **Video:** `wan_tp4.mp4` (49 frames, 832×480, 16 fps). Frames: `frame_nocache_0/mid.png`.
- **Measured full e2e: 69.3 s** (all on Neuron) — T5 0.704 s + DiT denoise 51.1 s (100 fwd @ 511 ms) +
  VAE 15.3 s + export.
- **Correctness:** decode parity vs CPU-fp32 PSNR 56.6 dB, SSIM 0.9994, 49/49 frames, no NaN → **PASS**.
- **vs H100 target (33 s):** ~2.1×.
- Fidelity: exact — reference for the TeaCache comparisons.
- Note: text-encode on CPU had inflated this to ~134 s; moving T5 on-device (0.7 s vs 65 s) fixed it.
