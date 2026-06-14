#!/usr/bin/env python3
"""Steps 6-11 bench: cheap wins (requires_grad False + inference_mode +
auto-cast=matmult flag) vs Phase A baseline.

Run config A (baseline, no cheap wins) and config B (cheap wins applied)
back to back. Both with prompt + image-latent caching.

Step 11 (--auto-cast=matmult) is applied via NEURON_CC_FLAGS env var —
set it BEFORE launching this script for config B:

    NEURON_CC_FLAGS="--auto-cast=matmult --auto-cast-type=bf16" \\
      python bench_cheap_wins.py --config b

Quality gate: cosine/std must match baseline (std=18.15). auto-cast
can introduce bf16 artifacts — verify std doesn't drop.
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
import flux2_cheap_wins as cheap


def neuron_sync():
    if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()


def load_pipe(args, apply_cheap: bool):
    print(f"\n=== loading pipe (cheap_wins={apply_cheap}) ===")
    t0 = time.time()
    pipe = NeuronFlux2KleinPipeline.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"  pipeline loaded                {time.time()-t0:.2f} s")

    device = torch.device("neuron")
    pipe.apply_neuron_patches(device, dtype=torch.bfloat16)

    if apply_cheap:
        cheap.apply_cheap_wins(pipe, enable_functional_rope=args.functional_rope)
        print(f"  cheap wins applied (functional_rope={args.functional_rope})")

    pipe.transformer.to(device)
    neuron_sync()

    pipe.transformer.inner = torch.compile(
        pipe.transformer.inner, backend="neuron", dynamic=False,
    )
    print(f"  NEURON_CC_FLAGS={os.environ.get('NEURON_CC_FLAGS', '(default)')}")
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


def quality(p):
    im = np.array(Image.open(p))
    return {"shape": im.shape, "std": float(im.std()), "mean": float(im.mean())}


def run_config(args, label, apply_cheap, input_image):
    pipe = load_pipe(args, apply_cheap=apply_cheap)
    install_image_latents_cache(pipe)

    prompt_embeds, _ = pipe.encode_prompt(
        prompt=args.prompt, device=pipe._execution_device, num_images_per_prompt=1,
    )
    neuron_sync()

    print("  [capture call]")
    t0 = time.time()
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    out = pipe(prompt_embeds=prompt_embeds, image=input_image,
               height=args.height, width=args.width,
               num_inference_steps=args.steps,
               guidance_scale=args.guidance_scale, generator=gen)
    neuron_sync()
    print(f"  capture call: {time.time()-t0:.2f} s")

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

    out_path = f"/tmp/cheap_{label}_warm.png"
    out.images[0].save(out_path)
    q = quality(out_path)
    print(f"  quality: std={q['std']:.2f}")

    del pipe
    import gc; gc.collect()
    return {"warm_avg": statistics.mean(times), "warm_min": min(times),
            "times": times, "quality": q}


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
    ap.add_argument("--config", choices=["a", "b", "both"], default="b")
    ap.add_argument("--functional-rope", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("Steps 6-11 bench — cheap wins")
    print("=" * 70)

    img = Image.new("RGB", (args.width, args.height), color=(180, 180, 180))
    draw = ImageDraw.Draw(img)
    x0, y0 = args.width // 4, args.height // 4
    x1, y1 = 3 * args.width // 4, 3 * args.height // 4
    draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=8)
    input_image = img

    summary = {}
    for label, apply_cheap in [("A", False), ("B", True)]:
        if args.config != "both" and args.config != label.lower():
            continue
        print(f"\n{'='*70}\nCONFIG {label}: cheap_wins={apply_cheap}\n{'='*70}")
        summary[label] = run_config(args, label, apply_cheap, input_image)

    print(f"\n{'='*70}\nCHEAP WINS SUMMARY\n{'='*70}")
    for k, v in summary.items():
        print(f"Config {k}: warm avg {v['warm_avg']:.2f}s  min {v['warm_min']:.2f}s  "
              f"std={v['quality']['std']:.2f}")
    if "A" in summary and "B" in summary:
        a, b = summary["A"], summary["B"]
        d = a["warm_avg"] - b["warm_avg"]
        print(f"\nDelta: {d:+.2f}s ({a['warm_avg']/b['warm_avg']:.3f}× B vs A)")
    elif "B" in summary:
        print(f"\n(Compare B against Phase A baseline 6.86s: "
              f"{6.86 - summary['B']['warm_avg']:+.2f}s)")


if __name__ == "__main__":
    main()
