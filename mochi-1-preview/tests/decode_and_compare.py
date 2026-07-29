"""Decode Neuron + CPU latents through the SAME VAE and compare frames.

The latent comparison (compare_latents.py) proves DiT fidelity. This is the
end-to-end perceptual confirmation: run both latent tensors through one
deterministic VAE and measure per-frame agreement on the pixels a user sees.

    python decode_and_compare.py <snapshot_dir> <lat_neuron.pt> <lat_cpu.pt>

Run on the box (needs the VAE weights + diffusers). Writes side-by-side
frames and prints per-frame PSNR.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


def denorm(latents, vae):
    cfg = vae.config
    if getattr(cfg, "latents_mean", None) is not None and \
       getattr(cfg, "latents_std", None) is not None:
        mean = torch.tensor(cfg.latents_mean).view(1, 12, 1, 1, 1).to(latents)
        std = torch.tensor(cfg.latents_std).view(1, 12, 1, 1, 1).to(latents)
        return latents * std / cfg.scaling_factor + mean
    return latents / cfg.scaling_factor


def decode(latents, vae, processor):
    with torch.no_grad():
        video = vae.decode(denorm(latents, vae).to(torch.float32), return_dict=False)[0]
    return processor.postprocess_video(video, output_type="np")[0]  # (F,H,W,C) in [0,1]


def main():
    snapshot, p_neuron, p_cpu = sys.argv[1], sys.argv[2], sys.argv[3]

    from diffusers import AutoencoderKLMochi
    from diffusers.video_processor import VideoProcessor

    vae = AutoencoderKLMochi.from_pretrained(snapshot, subfolder="vae",
                                             torch_dtype=torch.float32).eval()
    processor = VideoProcessor(vae_scale_factor=8)

    lat_n = torch.load(p_neuron, map_location="cpu", weights_only=False)["latents"].float()
    lat_c = torch.load(p_cpu, map_location="cpu", weights_only=False)["latents"].float()

    fr_n = decode(lat_n, vae, processor)
    fr_c = decode(lat_c, vae, processor)

    n = min(len(fr_n), len(fr_c))
    a = (fr_n[:n] * 255).astype(np.float32)
    b = (fr_c[:n] * 255).astype(np.float32)

    print("=" * 60)
    print("Decoded-frame agreement: Neuron vs CPU (same VAE)")
    print("=" * 60)
    print(f"frames: {n}  resolution: {a.shape[1]}x{a.shape[2]}")

    mse_all = ((a - b) ** 2).mean()
    psnr_all = float("inf") if mse_all == 0 else 20 * np.log10(255.0 / np.sqrt(mse_all))
    print(f"whole-clip PSNR: {psnr_all:.2f} dB")
    print(f"whole-clip MAE : {np.abs(a - b).mean():.3f} (0-255 scale)")

    print("\nper-frame PSNR (dB):")
    for i in range(n):
        mse = ((a[i] - b[i]) ** 2).mean()
        psnr = float("inf") if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
        bar = "#" * int(min(psnr, 60) / 60 * 40)
        print(f"  f{i:2d}: {psnr:5.1f}  {bar}")

    # Side-by-side of the middle frame for eyeballing.
    mid = n // 2
    sbs = np.concatenate([a[mid], b[mid]], axis=1).clip(0, 255).astype(np.uint8)
    out = Path(p_neuron).parent / "cmp_sidebyside_mid.png"
    try:
        import imageio.v2 as iio
        iio.imwrite(out, sbs)
        print(f"\nwrote side-by-side middle frame (neuron | cpu): {out}")
    except Exception as e:
        print(f"(side-by-side write skipped: {e})")

    print("\nverdict:", "STRONG perceptual match" if psnr_all > 30 else
          ("acceptable" if psnr_all > 25 else "DIVERGENT — investigate"))
    print("=" * 60)


if __name__ == "__main__":
    main()
