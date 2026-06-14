#!/usr/bin/env python3
"""Standalone VAE-decode bench — per-block compile (Phase B retry).

Loads just the FLUX.2-klein-4B VAE, moves it to Neuron, compiles the
decoder per-block, and times a decode of a realistic latent. Fast
iteration (no full pipeline) so compile failures surface in minutes.

Compares:
  A. VAE decode on CPU eager (the shipped Phase A path)
  B. VAE decode on Neuron, per-block compiled

Usage:
    NEURON_RT_VIRTUAL_CORE_SIZE=2 python bench_vae_perblock.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import flux2_vae_perblock as vpb


def neuron_sync():
    if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--config", choices=["a", "b", "both"], default="b")
    args = ap.parse_args()

    from diffusers import AutoencoderKLFlux2

    print("=" * 60)
    print("VAE per-block compile bench")
    print("=" * 60)

    t0 = time.time()
    vae = AutoencoderKLFlux2.from_pretrained(
        args.base_model, subfolder="vae", torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
    )
    vae.eval()
    print(f"VAE loaded in {time.time()-t0:.1f}s")

    # Latent shape for 1024x1024: VAE downsamples by 8, patch 2 → /16,
    # latent_channels=32. So latent is [1, 32, 64, 64] approx. Use the
    # config to be exact.
    lc = vae.config.latent_channels
    # spatial: height / (2^(num_down) ) ; klein uses patchify (2,2) too.
    # Use a safe [1, lc, H//16, W//16].
    h = args.height // 16
    w = args.width // 16
    latent = torch.randn(1, lc, h, w, dtype=torch.bfloat16)
    print(f"latent shape: {tuple(latent.shape)}")

    if args.config in ("a", "both"):
        print("\n[A] CPU eager decode")
        times = []
        for i in range(args.runs):
            t0 = time.time()
            with torch.no_grad():
                _ = vae.decode(latent).sample
            times.append(time.time() - t0)
            print(f"  run {i}: {times[-1]:.2f}s")
        print(f"  avg: {sum(times)/len(times):.2f}s")

    if args.config in ("b", "both"):
        print("\n[B] Neuron per-block compiled decode")
        device = torch.device("neuron")
        vae.to(device)
        n = vpb.compile_vae_decoder_per_block(vae)
        print(f"  compiled {n} decoder submodules")
        latent_n = latent.to(device)

        # warmup (compiles)
        t0 = time.time()
        with torch.no_grad():
            out = vae.decode(latent_n).sample
        neuron_sync()
        print(f"  warmup (compile): {time.time()-t0:.1f}s")

        times = []
        for i in range(args.runs):
            t0 = time.time()
            with torch.no_grad():
                out = vae.decode(latent_n).sample
            neuron_sync()
            times.append(time.time() - t0)
            print(f"  run {i}: {times[-1]:.2f}s")
        print(f"  avg: {sum(times)/len(times):.3f}s")
        print(f"  output: shape={tuple(out.shape)} "
              f"std={out.float().std().item():.3f} on {out.device}")


if __name__ == "__main__":
    main()
