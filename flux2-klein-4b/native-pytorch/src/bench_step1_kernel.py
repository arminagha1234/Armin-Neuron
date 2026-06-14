#!/usr/bin/env python3
"""Step 1.1 bench: kernel-only attention_cte swap.

Compares two configurations:
  A. Phase A baseline (default SDPA inside diffusers Flux2AttnProcessor)
  B. attention_cte kernel installed (LNC=2 sharded, no TP=4)

Both use prompt + image-latent caching from Phase A. Same NEFF cache
location.

If B is faster than A: ship Step 1.1, push to GitHub.
If B is slower or equal: skip the kernel-only path, document the test
as confirmation of the v3 finding, move on to Step 1.2 (TP=4 lift).
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
import flux2_attention_cte as kernel_mod


def neuron_sync():
    if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()


def load_pipe(args, install_kernel: bool):
    print(f"\n=== loading pipe (kernel={install_kernel}) ===")
    if install_kernel:
        kernel_mod.install_attention_cte_processor(None)
        print("  attention_cte kernel installed via class patch")
    else:
        kernel_mod.restore_default_attention(None)
        print("  default SDPA processor (Phase A baseline)")

    t0 = time.time()
    pipe = NeuronFlux2KleinPipeline.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"  pipeline loaded                {time.time()-t0:.2f} s")

    t0 = time.time()
    device = torch.device("neuron")
    pipe.apply_neuron_patches(device, dtype=torch.bfloat16)
    pipe.transformer.to(device)
    neuron_sync()
    print(f"  patches + .to(neuron)          {time.time()-t0:.2f} s")

    pipe.transformer.inner = torch.compile(
        pipe.transformer.inner, backend="neuron", dynamic=False,
    )
    print("  torch.compile applied to DiT")
    return pipe


def install_image_latents_cache(pipe):
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


def quality(out_path):
    im = np.array(Image.open(out_path))
    return {
        "shape": im.shape,
        "std": float(im.std()),
        "mean": float(im.mean()),
    }


def run_config(args, label, install_kernel, input_image):
    pipe = load_pipe(args, install_kernel=install_kernel)
    install_image_latents_cache(pipe)

    # Prompt cache
    t0 = time.time()
    prompt_embeds, _ = pipe.encode_prompt(
        prompt=args.prompt, device=pipe._execution_device,
        num_images_per_prompt=1,
    )
    neuron_sync()
    print(f"  one-time prompt encode: {time.time()-t0:.2f} s")

    # Capture call (populates image-latent cache, may compile NEFFs)
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
    out.images[0].save(f"/tmp/step1_{label}_capture.png")

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

    out_path = f"/tmp/step1_{label}_warm.png"
    out.images[0].save(out_path)
    q = quality(out_path)
    print(f"  output quality: std={q['std']:.2f}, mean={q['mean']:.2f}")

    del pipe
    import gc; gc.collect()

    return {
        "label": label,
        "install_kernel": install_kernel,
        "capture_s": capture_s,
        "warm_avg": statistics.mean(times),
        "warm_min": min(times),
        "warm_times": times,
        "quality": q,
        "out_path": out_path,
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
    print("Step 1.1 bench — kernel-only attention_cte swap")
    print("=" * 70)
    print(f"Model:  {args.base_model}")
    print(f"Config: {args.steps} steps, guidance={args.guidance_scale}, "
          f"{args.height}x{args.width}, seed={args.seed}")

    # Synthetic image (same as Phase A bench)
    img = Image.new("RGB", (args.width, args.height), color=(180, 180, 180))
    draw = ImageDraw.Draw(img)
    x0, y0 = args.width // 4, args.height // 4
    x1, y1 = 3 * args.width // 4, 3 * args.height // 4
    draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=8)
    input_image = img

    summary = {}
    for label, install_kernel in [("A", False), ("B", True)]:
        if args.config != "both" and args.config != label.lower():
            continue
        print(f"\n{'='*70}\nCONFIG {label}: install_kernel={install_kernel}\n{'='*70}")
        summary[label] = run_config(args, label, install_kernel, input_image)

    # ---- Summary ----
    print(f"\n{'='*70}\nSTEP 1.1 SUMMARY\n{'='*70}")
    if "A" in summary and "B" in summary:
        a = summary["A"]
        b = summary["B"]
        print(f"Config A (Phase A baseline, default SDPA):")
        print(f"  warm avg: {a['warm_avg']:.2f} s   min: {a['warm_min']:.2f} s")
        print(f"  quality:  std={a['quality']['std']:.2f}")
        print(f"Config B (attention_cte LNC=2):")
        print(f"  warm avg: {b['warm_avg']:.2f} s   min: {b['warm_min']:.2f} s")
        print(f"  quality:  std={b['quality']['std']:.2f}")
        delta = a['warm_avg'] - b['warm_avg']
        ratio = a['warm_avg'] / b['warm_avg']
        print(f"\nDelta:    {delta:+.2f} s  ({ratio:.2f}× B vs A)")
        if delta > 0.5:
            print("✓ Kernel WIN — ship Step 1.1")
        elif delta > -0.5:
            print("≈ Kernel TIE — likely not worth shipping (compile risk)")
        else:
            print("✗ Kernel LOSS — confirms v3 finding, skip to Step 1.2 (TP=4 lift)")

        std_diff = abs(a['quality']['std'] - b['quality']['std'])
        if std_diff > 5.0:
            print(f"⚠️  WARNING: std differs by {std_diff:.2f} — quality regression")
        else:
            print(f"✓ quality OK: std diff {std_diff:.2f}")


if __name__ == "__main__":
    main()
