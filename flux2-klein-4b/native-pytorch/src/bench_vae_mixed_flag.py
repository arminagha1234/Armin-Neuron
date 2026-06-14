#!/usr/bin/env python3
"""Option 3 — mixed compiler flags. DiT compiled with --model-type=transformer
(the proven win on the DiT, A/B 5.92 vs 7.71). VAE compiled with
--model-type=unet-inference (PAVE's conv-scheduling win, conv workload).

Mechanism: the Neuron compile cache is keyed by (graph, flags, shapes), so
each component caches under whichever NEURON_CC_FLAGS were active at compile
trigger time. We wrap pipe.vae.encode and pipe.vae.decode so that the flag
is set whenever they're called — both at compile time AND at warm-call time
(so the cache lookup matches the compiled NEFF's key).
"""
from __future__ import annotations

import argparse
import contextlib
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

VAE_FLAG = "--model-type=unet-inference"


def neuron_sync():
    if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()


def quality(p):
    im = np.array(Image.open(p))
    return {"shape": im.shape, "std": float(im.std()), "mean": float(im.mean())}


@contextlib.contextmanager
def vae_flags():
    """Set NEURON_CC_FLAGS to include --model-type=unet-inference for the
    duration. Append to whatever's there, dedupe."""
    saved = os.environ.get("NEURON_CC_FLAGS")
    cur = saved or ""
    if VAE_FLAG not in cur:
        os.environ["NEURON_CC_FLAGS"] = (
            (cur + " " if cur else "") + VAE_FLAG
        )
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("NEURON_CC_FLAGS", None)
        else:
            os.environ["NEURON_CC_FLAGS"] = saved


def install_vae_flag_wrappers(vae):
    """Wrap vae.encode and vae.decode so the unet-inference flag is set
    whenever they execute. This applies at BOTH compile time and warm
    runtime, so the cache key is consistent."""
    orig_encode = vae.encode
    orig_decode = vae.decode

    def encode_w(*a, **kw):
        with vae_flags():
            return orig_encode(*a, **kw)

    def decode_w(*a, **kw):
        with vae_flags():
            return orig_decode(*a, **kw)

    vae.encode = encode_w
    vae.decode = decode_w


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
    ap.add_argument("--label", default="mixed")
    args = ap.parse_args()

    print("=" * 70)
    print("Mixed-flag VAE-on-Neuron bench")
    print(f"  DiT flag at compile: {os.environ.get('NEURON_CC_FLAGS', '(default/transformer)')}")
    print(f"  VAE flag at compile: ^^^ + {VAE_FLAG}")
    print(f"  cache: {os.environ.get('NEURON_COMPILE_CACHE_URL', '(default)')}")
    print("=" * 70)

    device = torch.device("neuron")

    t0 = time.time()
    pipe = NeuronFlux2KleinPipeline.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"  pipeline loaded            {time.time()-t0:.2f} s")

    pipe.apply_neuron_patches(device, dtype=torch.bfloat16, vae_on_neuron=True)

    summary = vfix.apply_vae_neuron_fixes(pipe.vae, fp32_storage=False)
    print(f"  vae fixes: {summary}")

    # Install the per-call flag wrapper BEFORE moving to device / compiling.
    install_vae_flag_wrappers(pipe.vae)
    print(f"  vae call wrappers installed (sets {VAE_FLAG} per call)")

    # CPU gather check
    lat_ch = pipe.vae.config.latent_channels
    vae_dtype = next(pipe.vae.parameters()).dtype
    zc = torch.randn(1, lat_ch, args.height // 8, args.width // 8, dtype=vae_dtype)
    ver = vfix.verify_no_gather(pipe.vae, zc)
    print(f"  gather check: clean={ver['clean']} ops={ver['gather_ops']}")

    pipe.transformer.to(device)
    pipe.vae.to(device)
    neuron_sync()

    pipe.transformer.inner = torch.compile(
        pipe.transformer.inner, backend="neuron", dynamic=False,
    )
    n_vae = vpb.compile_vae_decoder_per_block(pipe.vae)
    print(f"  VAE decoder per-block wrapped ({n_vae} submodules)")

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

    captured = {}
    orig_pil = pipe.prepare_image_latents
    def capturing_prepare(*a, **kw):
        out = orig_pil(*a, **kw)
        captured["il"] = out[0]; captured["ids"] = out[1]
        return out
    pipe.prepare_image_latents = capturing_prepare

    print("\n  [capture/compile call] ...")
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
    print(f"\n  [{args.runs} warm runs]")
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
    print("MIXED-FLAG RESULT (DiT=transformer, VAE=unet-inference)")
    print(f"  warm avg: {statistics.mean(times):.2f} s   min: {min(times):.2f} s")
    print(f"  quality:  std={q['std']:.2f} mean={q['mean']:.2f}")
    print(f"  CPU VAE baseline:               5.92 s")
    print(f"  Option 1 (VAE Neuron, xformer): 5.19 s")
    print(f"  Option 2 (VAE Neuron, unet/all): 6.84 s (regression)")
    print(f"  delta vs Option 1: {5.19 - statistics.mean(times):+.2f} s")
    print(f"  GATE std~=18.15: {'PASS' if 16.0 < q['std'] < 20.0 else 'CHECK'}")


if __name__ == "__main__":
    main()
