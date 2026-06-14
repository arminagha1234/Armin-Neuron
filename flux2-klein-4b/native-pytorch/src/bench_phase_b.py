#!/usr/bin/env python3
"""Phase B bench: VAE on Neuron + torch.compile.

Builds on Phase A (image-latent caching). Adds:
  1. VAE moved to Neuron via apply_neuron_patches(..., vae_on_neuron=True)
  2. torch.compile(backend="neuron") on vae.decode and vae.encode
  3. Image-latent caching still on (Phase A baseline)

Compares two configurations:
  A. Phase A only (VAE on CPU eager, image-latent cached) — the shipped 6.86s state
  B. Phase B (VAE on Neuron compiled, image-latent cached)

Quality check: also dumps the output PNG from each so we can verify
the Neuron VAE doesn't introduce blur (a known risk).
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


def neuron_sync():
    if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()


def load_pipe(args, vae_on_neuron: bool):
    """Load a NeuronFlux2KleinPipeline ready for inference."""
    print(f"\n=== loading pipe (vae_on_neuron={vae_on_neuron}) ===")
    t0 = time.time()
    pipe = NeuronFlux2KleinPipeline.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"  pipeline loaded                {time.time()-t0:.2f} s")

    t0 = time.time()
    device = torch.device("neuron")
    pipe.apply_neuron_patches(device, dtype=torch.bfloat16,
                              vae_on_neuron=vae_on_neuron)
    pipe.transformer.to(device)
    if vae_on_neuron:
        pipe.vae.to(device, dtype=torch.bfloat16)
    neuron_sync()
    print(f"  patches + .to(neuron)          {time.time()-t0:.2f} s")

    # torch.compile on DiT
    pipe.transformer.inner = torch.compile(
        pipe.transformer.inner, backend="neuron", dynamic=False,
    )
    if vae_on_neuron:
        # torch.compile on the VAE decode + encode submodules
        pipe.vae.decode = torch.compile(
            pipe.vae.decode, backend="neuron", dynamic=False,
        )
        pipe.vae.encode = torch.compile(
            pipe.vae.encode, backend="neuron", dynamic=False,
        )
        print("  torch.compile applied to DiT + vae.decode + vae.encode")
    else:
        print("  torch.compile applied to DiT only")
    return pipe


def install_image_latents_cache(pipe):
    """Install image-latent caching (Phase A)."""
    captured = {}
    orig_pil = pipe.prepare_image_latents

    def caching_prepare(images, batch_size, generator, device, dtype):
        if "image_latents" in captured:
            return captured["image_latents"], captured["image_latent_ids"]
        out = orig_pil(images, batch_size, generator, device, dtype)
        captured["image_latents"] = out[0]
        captured["image_latent_ids"] = out[1]
        return out

    pipe.prepare_image_latents = caching_prepare
    return captured


def quality(out_path):
    im = np.array(Image.open(out_path))
    return {
        "shape": im.shape,
        "std": float(im.std()),
        "mean": float(im.mean()),
    }


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
    ap.add_argument("--config", choices=["a", "b", "both"], default="both")
    args = ap.parse_args()

    print("=" * 70)
    print("Phase B bench: VAE on Neuron + torch.compile")
    print("=" * 70)
    print(f"Model:  {args.base_model}")
    print(f"Config: {args.steps} steps, guidance={args.guidance_scale}, "
          f"{args.height}x{args.width}, seed={args.seed}")

    # Synthetic image
    img = Image.new("RGB", (args.width, args.height), color=(180, 180, 180))
    draw = ImageDraw.Draw(img)
    x0, y0 = args.width // 4, args.height // 4
    x1, y1 = 3 * args.width // 4, 3 * args.height // 4
    draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=8)
    input_image = img

    summary = {}

    for label, vae_on_neuron in [("A", False), ("B", True)]:
        if args.config != "both" and args.config != label.lower():
            continue
        print(f"\n{'='*70}\nCONFIG {label}: vae_on_neuron={vae_on_neuron}\n{'='*70}")

        pipe = load_pipe(args, vae_on_neuron=vae_on_neuron)
        captured = install_image_latents_cache(pipe)

        # Prompt cache
        t0 = time.time()
        prompt_embeds, _ = pipe.encode_prompt(
            prompt=args.prompt, device=pipe._execution_device,
            num_images_per_prompt=1,
        )
        neuron_sync()
        prompt_cache_s = time.time() - t0
        print(f"  one-time prompt encode: {prompt_cache_s:.2f} s")

        # Capture call (populates image-latent cache, compiles VAE NEFFs if B)
        print(f"  [capture call — populates caches, may compile NEFFs]")
        t0 = time.time()
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        out = pipe(prompt_embeds=prompt_embeds, image=input_image,
                   height=args.height, width=args.width,
                   num_inference_steps=args.steps,
                   guidance_scale=args.guidance_scale, generator=gen)
        neuron_sync()
        capture_s = time.time() - t0
        print(f"  capture call: {capture_s:.2f} s")
        cap_path = f"/tmp/phase_b_capture_{label}.png"
        out.images[0].save(cap_path)

        # Warm runs
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
            elapsed = time.time() - t0
            times.append(elapsed)
            print(f"    run {i}: {elapsed:.2f} s")

        out_path = f"/tmp/phase_b_warm_{label}.png"
        out.images[0].save(out_path)
        q = quality(out_path)
        print(f"  output quality: std={q['std']:.2f}, mean={q['mean']:.2f}")

        summary[label] = {
            "vae_on_neuron": vae_on_neuron,
            "prompt_cache_s": prompt_cache_s,
            "capture_s": capture_s,
            "warm_avg": statistics.mean(times),
            "warm_min": min(times),
            "warm_times": times,
            "quality": q,
            "out_path": out_path,
        }

        # Drop pipe to free memory before loading the next one
        del pipe
        import gc; gc.collect()

    # ---- Summary ----
    print(f"\n{'='*70}\nPHASE B SUMMARY\n{'='*70}")
    if "A" in summary and "B" in summary:
        a = summary["A"]
        b = summary["B"]
        print(f"Config A (Phase A baseline, VAE on CPU eager):")
        print(f"  warm avg: {a['warm_avg']:.2f} s   min: {a['warm_min']:.2f} s")
        print(f"  quality:  std={a['quality']['std']:.2f}")
        print(f"Config B (Phase B, VAE on Neuron compiled):")
        print(f"  warm avg: {b['warm_avg']:.2f} s   min: {b['warm_min']:.2f} s")
        print(f"  quality:  std={b['quality']['std']:.2f}")
        print(f"\nDelta:    {a['warm_avg'] - b['warm_avg']:+.2f} s "
              f"({a['warm_avg']/b['warm_avg']:.2f}× faster)")
        rate = 21.50 / 3600 / 32
        print(f"\nCost on trn2.48xl ($21.50/hr ÷ 32 cores):")
        print(f"  Config A:  ${a['warm_avg']*rate:.4f} per image")
        print(f"  Config B:  ${b['warm_avg']*rate:.4f} per image")
        print(f"  H100 ref:  $0.0010 per image (4-step est.)")

        # Quality check
        std_diff = abs(a['quality']['std'] - b['quality']['std'])
        if std_diff > 5.0:
            print(f"\n⚠️  WARNING: output std differs by {std_diff:.2f} — "
                  f"possible blur or quality regression")
        else:
            print(f"\n✓ quality OK: std diff {std_diff:.2f} (within tolerance)")
    elif "A" in summary:
        a = summary["A"]
        print(f"Config A only:  warm avg {a['warm_avg']:.2f} s, "
              f"std={a['quality']['std']:.2f}")
    elif "B" in summary:
        b = summary["B"]
        print(f"Config B only:  warm avg {b['warm_avg']:.2f} s, "
              f"std={b['quality']['std']:.2f}")


if __name__ == "__main__":
    main()
