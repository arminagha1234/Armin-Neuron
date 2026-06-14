#!/usr/bin/env python3
"""CPU-side VAE decode optimization bench for FLUX.2-klein-4B.

The cached steady-state breakdown shows VAE decode (on CPU) is ~2.93s
= 43% of the 6.86s wall-clock. Phase B (move to Neuron) made it SLOWER
(3.8s). Nobody tried making the CPU decode itself faster. This bench
tests genuine untried CPU-side levers:

  A. baseline (current production: plain CPU bf16 decode)
  B. enable_tiling()        — process the decode in spatial tiles (cache locality)
  C. enable_slicing()       — process batch slices (less relevant at B=1)
  D. channels_last memory format
  E. fewer threads (96 → 32; oversubscription can hurt conv)
  F. fp32 decode (sometimes faster than bf16 on CPU due to native kernels)

Times the VAE decode in isolation on a realistic [1,32,128,128] latent
(1024x1024 output).
"""
from __future__ import annotations

import os
import time

import torch


def neuron_sync():
    if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()


def time_decode(vae, latent, runs=5, label=""):
    # warmup
    with torch.no_grad():
        _ = vae.decode(latent).sample
    ts = []
    for _ in range(runs):
        t0 = time.time()
        with torch.no_grad():
            out = vae.decode(latent).sample
        ts.append(time.time() - t0)
    avg = sum(ts) / len(ts)
    print(f"  {label:40s} {avg*1000:8.1f} ms  (min {min(ts)*1000:.0f})", flush=True)
    return avg, out


def main():
    from diffusers import AutoencoderKLFlux2

    print("=" * 64)
    print("VAE decode CPU optimization bench (FLUX.2-klein-4B)")
    print("=" * 64)
    tok = os.environ.get("HF_TOKEN")

    vae = AutoencoderKLFlux2.from_pretrained(
        "black-forest-labs/FLUX.2-klein-4B", subfolder="vae",
        torch_dtype=torch.bfloat16, token=tok,
    ).eval()

    lc = vae.config.latent_channels
    # 1024x1024 → latent [1, lc, 128, 128] (VAE downsample 8x).
    latent = torch.randn(1, lc, 128, 128, dtype=torch.bfloat16)
    print(f"latent: {tuple(latent.shape)}, host threads={torch.get_num_threads()}\n")

    results = {}

    # A. baseline
    print("[A] baseline (CPU bf16, current production)")
    results["A_baseline"], ref = time_decode(vae, latent, label="baseline")

    # B. tiling
    print("[B] enable_tiling()")
    vae.enable_tiling()
    results["B_tiling"], _ = time_decode(vae, latent, label="tiling")
    vae.disable_tiling()

    # C. slicing
    print("[C] enable_slicing()")
    vae.enable_slicing()
    results["C_slicing"], _ = time_decode(vae, latent, label="slicing")
    vae.disable_slicing()

    # D. channels_last
    print("[D] channels_last memory format")
    try:
        vae_cl = vae.to(memory_format=torch.channels_last)
        lat_cl = latent.to(memory_format=torch.channels_last)
        results["D_channels_last"], _ = time_decode(vae_cl, lat_cl, label="channels_last")
        vae = vae.to(memory_format=torch.contiguous_format)
    except Exception as e:
        print(f"    channels_last failed: {e}")

    # E. fewer threads
    print("[E] thread count sweep")
    orig = torch.get_num_threads()
    for nt in (32, 16, 64):
        torch.set_num_threads(nt)
        results[f"E_threads_{nt}"], _ = time_decode(vae, latent, label=f"threads={nt}")
    torch.set_num_threads(orig)

    # F. fp32 decode
    print("[F] fp32 decode")
    try:
        vae_fp32 = AutoencoderKLFlux2.from_pretrained(
            "black-forest-labs/FLUX.2-klein-4B", subfolder="vae",
            torch_dtype=torch.float32, token=tok,
        ).eval()
        latent_fp32 = latent.float()
        results["F_fp32"], _ = time_decode(vae_fp32, latent_fp32, label="fp32")
    except Exception as e:
        print(f"    fp32 failed: {e}")

    # F2. fp32 + tiling (often the best CPU combo)
    print("[F2] fp32 + tiling")
    try:
        vae_fp32.enable_tiling()
        results["F2_fp32_tiling"], _ = time_decode(vae_fp32, latent_fp32, label="fp32+tiling")
    except Exception as e:
        print(f"    fp32+tiling failed: {e}")

    print("\n" + "=" * 64)
    print("SUMMARY (VAE decode, lower is better)")
    print("=" * 64)
    base = results.get("A_baseline", 1.0)
    for k, v in sorted(results.items(), key=lambda x: x[1]):
        speedup = base / v
        tag = " ← BEST" if v == min(results.values()) else ""
        print(f"  {k:24s} {v*1000:8.1f} ms   {speedup:.2f}x vs baseline{tag}")
    print(f"\nIf best beats baseline, end-to-end 6.86s improves by "
          f"~{(base - min(results.values())):.2f}s")


if __name__ == "__main__":
    main()
