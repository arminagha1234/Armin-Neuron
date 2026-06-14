#!/usr/bin/env python3
"""One throughput worker: a single-rank cached FLUX.2-klein-4B pipeline.

Pinned to its own core pair via NEURON_RT_VISIBLE_CORES (set by the
launcher). Runs `--images` warm inferences after a warmup, prints
per-image times. The launcher aggregates across workers.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from neuron_flux2_klein_native import NeuronFlux2KleinPipeline


def neuron_sync():
    if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--images", type=int, default=3)
    ap.add_argument("--base-model", default="black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    args = ap.parse_args()

    w = args.worker
    print(f"[worker {w}] cores={os.environ.get('NEURON_RT_VISIBLE_CORES')}", flush=True)

    pipe = NeuronFlux2KleinPipeline.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
    )
    device = torch.device("neuron")
    pipe.apply_neuron_patches(device, dtype=torch.bfloat16)
    pipe.transformer.to(device)
    pipe.transformer.inner = torch.compile(
        pipe.transformer.inner, backend="neuron", dynamic=False,
    )

    # image-latent cache
    cache = {}
    orig = pipe.prepare_image_latents

    def cached(images, batch_size, generator, device=None, dtype=None):
        if "il" in cache:
            return cache["il"], cache["ilids"]
        out = orig(images, batch_size, generator, device, dtype)
        cache["il"], cache["ilids"] = out[0], out[1]
        return out

    pipe.prepare_image_latents = cached

    img = Image.new("RGB", (args.width, args.height), color=(180, 180, 180))
    d = ImageDraw.Draw(img)
    d.rectangle([args.width//4, args.height//4, 3*args.width//4, 3*args.height//4],
                outline=(255, 0, 0), width=8)

    prompt_embeds, _ = pipe.encode_prompt(
        prompt="Zoom into the red highlighted area",
        device=pipe._execution_device, num_images_per_prompt=1,
    )
    neuron_sync()

    # warmup (compiles / loads NEFF)
    t0 = time.time()
    gen = torch.Generator(device="cpu").manual_seed(42)
    pipe(prompt_embeds=prompt_embeds, image=img, height=args.height,
         width=args.width, num_inference_steps=args.steps,
         guidance_scale=1.0, generator=gen)
    neuron_sync()
    print(f"[worker {w}] warmup {time.time()-t0:.1f}s", flush=True)

    # timed images
    times = []
    for i in range(args.images):
        t0 = time.time()
        gen = torch.Generator(device="cpu").manual_seed(42 + i)
        pipe(prompt_embeds=prompt_embeds, image=img, height=args.height,
             width=args.width, num_inference_steps=args.steps,
             guidance_scale=1.0, generator=gen)
        neuron_sync()
        times.append(time.time() - t0)
        print(f"[worker {w}] warm image {i}: {times[-1]:.2f}s", flush=True)

    print(f"[worker {w}] avg {sum(times)/len(times):.2f}s", flush=True)


if __name__ == "__main__":
    main()
