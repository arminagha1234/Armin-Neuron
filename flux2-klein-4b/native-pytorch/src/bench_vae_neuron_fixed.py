#!/usr/bin/env python3
"""Bench the VAE-on-Neuron path WITH the PAVE fixes (gather-free upsample +
fp32 GroupNorm + fp32 storage), Variant 3 (prompt + image-latent cached).

Compares:
  - CONFIG A: VAE on Neuron + fixes, default flags (--model-type=transformer)
  - CONFIG B: VAE on Neuron + fixes, --model-type=unet-inference

Run each in its own NEFF cache dir (set NEURON_COMPILE_CACHE_URL + optionally
NEURON_CC_FLAGS before launching). One config per invocation:

    NEURON_COMPILE_CACHE_URL=/mnt/data/work/flux2/neff_vaefix_xfmr \
      python bench_vae_neuron_fixed.py

    NEURON_CC_FLAGS="--model-type=unet-inference" \
    NEURON_COMPILE_CACHE_URL=/mnt/data/work/flux2/neff_vaefix_unet \
      python bench_vae_neuron_fixed.py

Baseline to beat: 5.92 s (CPU VAE + channels_last, Variant 3).
Quality gate: decoded std ~= 18.15, finite (no NaN/black image).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import statistics

import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from neuron_flux2_klein_native import NeuronFlux2KleinPipeline
import flux2_vae_neuron_fixes as vfix
import flux2_vae_perblock as vpb


def neuron_sync():
    if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()


def quality(p):
    im = np.array(Image.open(p))
    return {"shape": im.shape, "std": float(im.std()), "mean": float(im.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance-scale", type=float, default=1.0)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default="Zoom into the red highlighted area")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--label", default="vaefix")
    args = ap.parse_args()

    flags = os.environ.get("NEURON_CC_FLAGS", "(default/transformer)")
    cache = os.environ.get("NEURON_COMPILE_CACHE_URL", "(default)")
    print("=" * 70)
    print("VAE-on-Neuron + PAVE fixes bench")
    print(f"  NEURON_CC_FLAGS       = {flags}")
    print(f"  NEURON_COMPILE_CACHE  = {cache}")
    print("=" * 70)

    device = torch.device("neuron")

    t0 = time.time()
    pipe = NeuronFlux2KleinPipeline.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"  pipeline loaded            {time.time()-t0:.2f} s")

    # Apply transformer Neuron patches with vae_on_neuron=True
    pipe.apply_neuron_patches(device, dtype=torch.bfloat16, vae_on_neuron=True)

    # --- PAVE VAE fixes (BEFORE moving to device / compiling) ---
    # Start with bf16 storage (matches pipeline contract); just upsample +
    # GroupNorm-fp32 fixes. If quality fails we'll escalate to fp32 storage.
    summary = vfix.apply_vae_neuron_fixes(pipe.vae, fp32_storage=False)
    print(f"  vae fixes: {summary}")

    # --- VERIFY no gather ops remain (on CPU, before device move) ---
    # Build a sample latent matching the decode input shape and dtype.
    lat_ch = pipe.vae.config.latent_channels
    vae_dtype = next(pipe.vae.parameters()).dtype
    zc = torch.randn(1, lat_ch, args.height // 8, args.width // 8,
                     dtype=vae_dtype)
    try:
        ver = vfix.verify_no_gather(pipe.vae, zc)
        print(f"  gather check: clean={ver['clean']} ops={ver['gather_ops']}")
        if not ver["clean"]:
            print("  !! WARNING: gather ops still present — upsample patch "
                  "did not cover all sites")
    except Exception as e:
        print(f"  gather check skipped (decode probe failed: {e})")

    # Move transformer + VAE to Neuron
    pipe.transformer.to(device)
    pipe.vae.to(device)
    neuron_sync()

    # Compile transformer (as production) + VAE decoder per-block
    pipe.transformer.inner = torch.compile(
        pipe.transformer.inner, backend="neuron", dynamic=False,
    )
    n_vae = vpb.compile_vae_decoder_per_block(pipe.vae)
    print(f"  VAE decoder per-block compiled ({n_vae} submodules)")

    # Synthetic input image
    img = Image.new("RGB", (args.width, args.height), color=(180, 180, 180))
    draw = ImageDraw.Draw(img)
    draw.rectangle([args.width // 4, args.height // 4,
                    3 * args.width // 4, 3 * args.height // 4],
                   outline=(255, 0, 0), width=8)
    input_image = img

    prompt_embeds, _ = pipe.encode_prompt(
        prompt=args.prompt, device=pipe._execution_device,
        num_images_per_prompt=1,
    )
    neuron_sync()

    # capture image-latents, then cache (Variant 3)
    captured = {}
    orig_pil = pipe.prepare_image_latents
    def capturing_prepare(*a, **kw):
        out = orig_pil(*a, **kw)
        captured["il"] = out[0]; captured["ids"] = out[1]
        return out
    pipe.prepare_image_latents = capturing_prepare

    print("  [capture/compile call] ...")
    t0 = time.time()
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    out = pipe(prompt_embeds=prompt_embeds, image=input_image,
               height=args.height, width=args.width,
               num_inference_steps=args.steps,
               guidance_scale=args.guidance_scale, generator=gen)
    neuron_sync()
    print(f"  capture call: {time.time()-t0:.2f} s")

    def cached_prepare(*a, **kw):
        return captured["il"], captured["ids"]
    pipe.prepare_image_latents = cached_prepare

    times = []
    print(f"  [{args.runs} warm runs]")
    for i in range(args.runs):
        t0 = time.time()
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        out = pipe(prompt_embeds=prompt_embeds, image=input_image,
                   height=args.height, width=args.width,
                   num_inference_steps=args.steps,
                   guidance_scale=args.guidance_scale, generator=gen)
        neuron_sync()
        times.append(time.time() - t0)
        print(f"    run {i}: {times[-1]:.2f} s")

    out_path = f"/tmp/{args.label}_warm.png"
    out.images[0].save(out_path)
    q = quality(out_path)

    print()
    print("=" * 70)
    print("RESULT")
    print(f"  warm avg: {statistics.mean(times):.2f} s   min: {min(times):.2f} s")
    print(f"  quality:  std={q['std']:.2f} mean={q['mean']:.2f} shape={q['shape']}")
    print(f"  baseline (CPU VAE channels_last): 5.92 s")
    print(f"  delta vs baseline: {5.92 - statistics.mean(times):+.2f} s")
    print(f"  GATE std~=18.15: {'PASS' if 16.0 < q['std'] < 20.0 else 'CHECK'}")


if __name__ == "__main__":
    main()
