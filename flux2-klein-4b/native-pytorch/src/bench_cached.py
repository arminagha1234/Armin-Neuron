#!/usr/bin/env python3
"""Path A bench: prompt caching + image-latent caching + scheduler hoist.

Measures the wall-clock cost of each FLUX.2-klein inference stage
separately by monkey-patching the pipeline's internal methods to time
each one. Then re-runs with each cache layer applied to verify
end-to-end savings.

Usage (inside the Beta 3 container):

    /opt/torch-neuronx/.venv/bin/python bench_cached.py --runs 5

The DiT NEFF must already be cached at /mnt/data/work/flux2/neff_cache_4step
from a prior run.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import statistics

import torch
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from neuron_flux2_klein_native import NeuronFlux2KleinPipeline

# Toggle: set via --vae-on-neuron to test Variant 4 (Phase A + VAE on Neuron)
VAE_ON_NEURON = "--vae-on-neuron" in sys.argv
VAE_CHANNELS_LAST = "--vae-channels-last" in sys.argv


def neuron_sync():
    if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()


# ---------------------------------------------------------------------------
# Per-stage timing instrumentation
# ---------------------------------------------------------------------------

class StageTimer:
    """Captures elapsed time per pipeline stage. Reset between runs."""
    def __init__(self):
        self.timings = {}

    def reset(self):
        self.timings = {}

    def record(self, name, elapsed):
        self.timings.setdefault(name, []).append(elapsed)

    def report(self, label):
        print(f"  --- per-stage timings ({label}) ---")
        total = 0.0
        for k, vs in self.timings.items():
            avg = sum(vs) / len(vs)
            total += avg
            print(f"    {k:40s} {avg*1000:8.1f} ms")
        print(f"    {'(sum)':40s} {total*1000:8.1f} ms")


def instrument_pipeline(pipe, timer: StageTimer):
    """Wrap each major pipeline method to record its elapsed time."""
    methods_to_time = [
        ("encode_prompt",         pipe.encode_prompt),
        ("_encode_vae_image",     pipe._encode_vae_image),
        ("prepare_latents",       pipe.prepare_latents),
        ("prepare_image_latents", pipe.prepare_image_latents),
    ]
    for name, fn in methods_to_time:
        def make_wrap(_name, _fn):
            def wrapper(*a, **kw):
                t0 = time.time()
                out = _fn(*a, **kw)
                neuron_sync()
                timer.record(_name, time.time() - t0)
                return out
            return wrapper
        setattr(pipe, name, make_wrap(name, fn))

    # VAE decode: sits inside the pipeline `__call__` after the loop
    orig_decode = pipe.vae.decode
    def decode_timed(*a, **kw):
        t0 = time.time()
        out = orig_decode(*a, **kw)
        neuron_sync()
        timer.record("vae.decode", time.time() - t0)
        return out
    pipe.vae.decode = decode_timed

    # Scheduler: set_timesteps. Use functools.wraps so signature
    # introspection in retrieve_timesteps still finds the `sigmas`
    # parameter on the wrapped function.
    import functools as _ft
    orig_sched = pipe.scheduler.set_timesteps
    @_ft.wraps(orig_sched)
    def sched_timed(*a, **kw):
        t0 = time.time()
        out = orig_sched(*a, **kw)
        timer.record("scheduler.set_timesteps", time.time() - t0)
        return out
    pipe.scheduler.set_timesteps = sched_timed


def install_image_latents_cache(pipe, cached_latents, cached_ids):
    """Override prepare_image_latents to return pre-computed values."""
    def cached_prepare(images, batch_size, generator, device, dtype):
        return cached_latents, cached_ids
    pipe.prepare_image_latents = cached_prepare


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

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
    ap.add_argument("--vae-on-neuron", action="store_true",
                    help="Variant 4: Phase A caching + VAE on Neuron per-block")
    ap.add_argument("--vae-channels-last", action="store_true",
                    help="Variant 5: Phase A caching + CPU VAE channels_last")
    args = ap.parse_args()

    print("=" * 70)
    print("Path A bench — caching breakdown")
    print("=" * 70)
    print(f"Model:  {args.base_model}")
    print(f"Config: {args.steps} steps, guidance={args.guidance_scale}, "
          f"{args.height}x{args.width}, seed={args.seed}")
    print()

    # ---- Setup ----
    print("[setup]")
    t0 = time.time()
    pipe = NeuronFlux2KleinPipeline.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"  pipeline loaded                        {time.time()-t0:.2f} s")

    t0 = time.time()
    device = torch.device("neuron")
    pipe.apply_neuron_patches(device, dtype=torch.bfloat16,
                              vae_on_neuron=VAE_ON_NEURON)
    pipe.transformer.to(device)
    if VAE_ON_NEURON:
        import flux2_vae_perblock as vpb
        pipe.vae.to(device)
        nvae = vpb.compile_vae_decoder_per_block(pipe.vae)
        print(f"  VAE on Neuron, per-block compiled ({nvae} submodules)")
    elif VAE_CHANNELS_LAST:
        pipe.vae = pipe.vae.to(memory_format=torch.channels_last)
        print(f"  CPU VAE converted to channels_last")
    neuron_sync()
    print(f"  apply patches + transformer.to(neuron) {time.time()-t0:.2f} s")

    pipe.transformer.inner = torch.compile(
        pipe.transformer.inner, backend="neuron", dynamic=False,
    )

    # Synthetic input image
    img = Image.new("RGB", (args.width, args.height), color=(180, 180, 180))
    draw = ImageDraw.Draw(img)
    x0, y0 = args.width // 4, args.height // 4
    x1, y1 = 3 * args.width // 4, 3 * args.height // 4
    draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=8)
    input_image = img

    # ---- Variant 0: warm baseline (no caching), instrumented ----
    timer = StageTimer()
    instrument_pipeline(pipe, timer)

    # Warmup: one call to load NEFF + warm caches
    print()
    print("[warmup call (compiles or loads NEFF)]")
    t0 = time.time()
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    out = pipe(prompt=args.prompt, image=input_image,
               height=args.height, width=args.width,
               num_inference_steps=args.steps,
               guidance_scale=args.guidance_scale, generator=gen)
    neuron_sync()
    warmup_s = time.time() - t0
    print(f"  warmup wall-clock: {warmup_s:.2f} s")
    timer.report("warmup")
    out.images[0].save("/tmp/v0_warmup.png")

    # ---- Variant 1: warm baseline, no caching ----
    timer.reset()
    times_baseline = []
    print()
    print(f"[variant 1: no caching, {args.runs} runs]")
    for i in range(args.runs):
        t0 = time.time()
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        out = pipe(prompt=args.prompt, image=input_image,
                   height=args.height, width=args.width,
                   num_inference_steps=args.steps,
                   guidance_scale=args.guidance_scale, generator=gen)
        neuron_sync()
        times_baseline.append(time.time() - t0)
        print(f"  run {i}: {times_baseline[-1]:.2f} s")
    out.images[0].save("/tmp/v1_baseline.png")
    timer.report("variant 1 (no caching)")
    avg_baseline = statistics.mean(times_baseline)
    print(f"  avg: {avg_baseline:.2f} s   min: {min(times_baseline):.2f} s")

    # ---- Variant 2: prompt cached ----
    print()
    print(f"[variant 2: prompt_embeds cached, {args.runs} runs]")
    t0 = time.time()
    prompt_embeds, _text_ids = pipe.encode_prompt(
        prompt=args.prompt, device=pipe._execution_device,
        num_images_per_prompt=1,
    )
    neuron_sync()
    print(f"  one-time encode_prompt: {time.time()-t0:.2f} s")

    timer.reset()
    times_pcache = []
    for i in range(args.runs):
        t0 = time.time()
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        out = pipe(prompt_embeds=prompt_embeds, image=input_image,
                   height=args.height, width=args.width,
                   num_inference_steps=args.steps,
                   guidance_scale=args.guidance_scale, generator=gen)
        neuron_sync()
        times_pcache.append(time.time() - t0)
        print(f"  run {i}: {times_pcache[-1]:.2f} s")
    out.images[0].save("/tmp/v2_pcache.png")
    timer.report("variant 2 (prompt cached)")
    avg_pcache = statistics.mean(times_pcache)
    print(f"  avg: {avg_pcache:.2f} s   min: {min(times_pcache):.2f} s")

    # ---- Variant 3: prompt + image-latents cached ----
    # Capture the image-latents output on the next call by side-channel,
    # then install a cache that returns them. This avoids the
    # parent's 3D/4D image plumbing.
    print()
    print(f"[variant 3: prompt + image_latents cached, {args.runs} runs]")

    captured = {}
    orig_pil = pipe.prepare_image_latents
    def capturing_prepare(images, batch_size, generator, device, dtype):
        out = orig_pil(images, batch_size, generator, device, dtype)
        captured["image_latents"] = out[0]
        captured["image_latent_ids"] = out[1]
        return out
    pipe.prepare_image_latents = capturing_prepare

    # One call to capture
    print("  capturing image_latents from a one-time call...")
    t0 = time.time()
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    _ = pipe(prompt_embeds=prompt_embeds, image=input_image,
             height=args.height, width=args.width,
             num_inference_steps=args.steps,
             guidance_scale=args.guidance_scale, generator=gen)
    neuron_sync()
    print(f"  captured in {time.time()-t0:.2f} s")
    print(f"  image_latents shape: {tuple(captured['image_latents'].shape)} "
          f"on {captured['image_latents'].device}")

    # Now install the cache override (constant return)
    install_image_latents_cache(
        pipe, captured["image_latents"], captured["image_latent_ids"],
    )

    timer.reset()
    times_full = []
    for i in range(args.runs):
        t0 = time.time()
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        out = pipe(prompt_embeds=prompt_embeds, image=input_image,
                   height=args.height, width=args.width,
                   num_inference_steps=args.steps,
                   guidance_scale=args.guidance_scale, generator=gen)
        neuron_sync()
        times_full.append(time.time() - t0)
        print(f"  run {i}: {times_full[-1]:.2f} s")
    out.images[0].save("/tmp/v3_full.png")
    timer.report("variant 3 (prompt + image_latents cached)")
    avg_full = statistics.mean(times_full)
    print(f"  avg: {avg_full:.2f} s   min: {min(times_full):.2f} s")

    # ---- Summary ----
    print()
    print("=" * 70)
    print("PATH A SUMMARY")
    print("=" * 70)
    print(f"Variant 1 (no caching):                 {avg_baseline:6.2f} s   "
          f"(baseline)")
    print(f"Variant 2 (prompt cached):              {avg_pcache:6.2f} s   "
          f"({avg_baseline-avg_pcache:+.2f} s, "
          f"{avg_baseline/avg_pcache:.2f}× faster)")
    print(f"Variant 3 (prompt + image_latents):     {avg_full:6.2f} s   "
          f"({avg_baseline-avg_full:+.2f} s, "
          f"{avg_baseline/avg_full:.2f}× faster)")
    print()
    print("Per-image cost on trn2.48xl ($21.50/hr, 32 logical cores):")
    rate = 21.50 / 3600 / 32
    print(f"  Variant 1:  ${avg_baseline*rate:.4f} per image")
    print(f"  Variant 2:  ${avg_pcache*rate:.4f} per image")
    print(f"  Variant 3:  ${avg_full*rate:.4f} per image")
    print(f"  H100 ref:   $0.0010 per image (4-step est.)")


if __name__ == "__main__":
    main()
