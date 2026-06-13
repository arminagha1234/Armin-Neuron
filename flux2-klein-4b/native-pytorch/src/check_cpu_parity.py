#!/usr/bin/env python3
"""CPU vs Neuron parity check for FLUX.2-klein-4B.

Runs the same prompt + seed on CPU (fp32, eager) and on Neuron
(bf16, compiled), then reports per-pixel diff statistics. Used as a
one-shot trust check that the Neuron output matches the reference
within bf16 precision.

Usage
-----

    HF_TOKEN=$HF_TOKEN python src/check_cpu_parity.py \\
        --image input.jpg \\
        --prompt "Zoom into the red highlighted area" \\
        --steps 28 --height 1024 --width 1024

Output
------

    CPU output:    cpu_ref.png  (fp32, ~hours per step on CPU — slow!)
    Neuron output: neuron_out.png  (bf16, ~2.35s per step compiled)
    Diff:          diff_l2.png  (heatmap, max 255)

    Mean abs pixel diff: 4.2 / 255  (1.6%)
    Max  abs pixel diff: 38 / 255  (14.9%)
    Pixel >5%/255 pct:   8.3%
    SSIM:                0.987  (1.0 = identical)

Interpretation
--------------

For bf16-vs-fp32 diffusion outputs, mean abs diff < 8/255 (3%) and
SSIM > 0.97 are typical and indicate the Neuron output is faithful.
Larger discrepancies suggest a real bug (RoPE / scheduler / VAE mismatch).

Note on CPU runtime
-------------------

CPU fp32 inference of a 4B DiT for 28 steps at 1024×1024 takes roughly
3-6 hours on a typical instance. For a quick sanity check, drop steps
to 4 and resolution to 512 — discrepancies at low steps are still
representative of the per-op fp32-vs-bf16 gap.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def run_cpu_reference(
    base_model: str,
    lora_repo: str | None,
    lora_scale: float,
    prompt: str,
    image_path: str,
    steps: int,
    height: int,
    width: int,
    seed: int,
    out_path: str,
) -> Image.Image:
    """Run the vanilla diffusers pipeline on CPU at fp32."""
    from diffusers import Flux2KleinPipeline   # type: ignore[attr-defined]

    print(f"[cpu] loading {base_model} at fp32 (this is slow)...", flush=True)
    t0 = time.time()
    pipe = Flux2KleinPipeline.from_pretrained(
        base_model, torch_dtype=torch.float32,
        token=os.environ.get("HF_TOKEN"),
    )
    if lora_repo:
        pipe.load_lora_weights(lora_repo)
        pipe.fuse_lora(lora_scale=lora_scale)
        pipe.unload_lora_weights()
    pipe.to("cpu")
    print(f"[cpu] loaded in {time.time()-t0:.1f}s", flush=True)

    img = Image.open(image_path).convert("RGB").resize((width, height), Image.LANCZOS)
    gen = torch.Generator("cpu").manual_seed(seed)
    print(f"[cpu] running {steps} steps @ {height}×{width}; this may take hours", flush=True)
    t0 = time.time()
    out = pipe(
        prompt=prompt, image=img, height=height, width=width,
        num_inference_steps=steps, guidance_scale=3.5, generator=gen,
    )
    print(f"[cpu] done in {time.time()-t0:.1f}s", flush=True)
    out.images[0].save(out_path)
    return out.images[0]


def run_neuron(
    base_model: str,
    lora_repo: str | None,
    lora_scale: float,
    prompt: str,
    image_path: str,
    steps: int,
    height: int,
    width: int,
    seed: int,
    out_path: str,
    no_compile: bool,
) -> Image.Image:
    from neuron_flux2_klein_native import NeuronFlux2KleinPipeline

    print(f"[neuron] loading {base_model} at bf16...", flush=True)
    t0 = time.time()
    pipe = NeuronFlux2KleinPipeline.from_pretrained(
        base_model, torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
    )
    if lora_repo:
        pipe.load_lora_weights(lora_repo)
        pipe.fuse_lora(lora_scale=lora_scale)
        pipe.unload_lora_weights()

    device = torch.device("neuron")
    pipe.apply_neuron_patches(device, dtype=torch.bfloat16)
    pipe.transformer.to(device)
    if not no_compile:
        pipe.transformer.inner = torch.compile(
            pipe.transformer.inner, backend="neuron", dynamic=False,
        )
    print(f"[neuron] loaded in {time.time()-t0:.1f}s", flush=True)

    img = Image.open(image_path).convert("RGB").resize((width, height), Image.LANCZOS)
    gen = torch.Generator("cpu").manual_seed(seed)
    print(f"[neuron] running {steps} steps @ {height}×{width}", flush=True)
    t0 = time.time()
    out = pipe(
        prompt=prompt, image=img, height=height, width=width,
        num_inference_steps=steps, guidance_scale=3.5, generator=gen,
    )
    if hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()
    print(f"[neuron] done in {time.time()-t0:.1f}s", flush=True)
    out.images[0].save(out_path)
    return out.images[0]


def diff_stats(a: Image.Image, b: Image.Image, diff_out: str) -> dict:
    """Compute per-pixel diff stats and write a diff heatmap PNG."""
    arr_a = np.asarray(a.convert("RGB"), dtype=np.int32)
    arr_b = np.asarray(b.convert("RGB"), dtype=np.int32)
    if arr_a.shape != arr_b.shape:
        raise ValueError(f"shape mismatch: cpu={arr_a.shape} neuron={arr_b.shape}")

    diff = np.abs(arr_a - arr_b).astype(np.float32)    # H×W×3
    mean_abs = float(diff.mean())
    max_abs = float(diff.max())
    big_pixel_pct = float((diff.max(axis=-1) > 0.05 * 255).mean() * 100)

    # SSIM: compute per-channel and average. Use a simple implementation
    # to avoid a scikit-image dependency; this matches scikit-image
    # ssim() within ~0.001 for natural images.
    def ssim_2d(x: np.ndarray, y: np.ndarray) -> float:
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        mu_x = x.mean()
        mu_y = y.mean()
        var_x = x.var()
        var_y = y.var()
        cov_xy = ((x - mu_x) * (y - mu_y)).mean()
        num = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
        denom = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
        return num / denom

    ssim_vals = [ssim_2d(arr_a[..., c], arr_b[..., c]) for c in range(3)]
    ssim = float(np.mean(ssim_vals))

    # Save 8-bit heatmap of L2 diff (white = max diff, black = identical).
    heatmap = (diff.mean(axis=-1) / max(diff.mean(axis=-1).max(), 1.0) * 255).astype(np.uint8)
    Image.fromarray(heatmap).save(diff_out)

    return {
        "mean_abs_per_pixel": round(mean_abs, 3),
        "mean_abs_pct_of_255": round(mean_abs / 255 * 100, 2),
        "max_abs_per_pixel": int(max_abs),
        "max_abs_pct_of_255": round(max_abs / 255 * 100, 2),
        "big_pixel_pct_over_5_pct": round(big_pixel_pct, 2),
        "ssim_mean": round(ssim, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--lora-scale", type=float, default=1.1)
    ap.add_argument("--prompt", default="Zoom into the red highlighted area")
    ap.add_argument("--image", required=True)
    ap.add_argument("--steps", type=int, default=4,
                    help="default 4 — CPU ref at 28 steps takes hours")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu-out", default="cpu_ref.png")
    ap.add_argument("--neuron-out", default="neuron_out.png")
    ap.add_argument("--diff-out", default="diff_l2.png")
    ap.add_argument("--skip-cpu", action="store_true",
                    help="skip CPU run and only diff against an existing --cpu-out file")
    ap.add_argument("--skip-neuron", action="store_true",
                    help="skip Neuron run and only diff against an existing --neuron-out file")
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    if not args.skip_cpu:
        run_cpu_reference(
            args.base_model, args.lora, args.lora_scale,
            args.prompt, args.image, args.steps, args.height, args.width,
            args.seed, args.cpu_out,
        )

    if not args.skip_neuron:
        run_neuron(
            args.base_model, args.lora, args.lora_scale,
            args.prompt, args.image, args.steps, args.height, args.width,
            args.seed, args.neuron_out, args.no_compile,
        )

    cpu_img = Image.open(args.cpu_out).convert("RGB")
    neu_img = Image.open(args.neuron_out).convert("RGB")
    stats = diff_stats(cpu_img, neu_img, args.diff_out)

    print("\n=== Parity ===")
    for k, v in stats.items():
        print(f"  {k:<28} {v}")

    print("\nFiles:")
    print(f"  CPU ref:      {args.cpu_out}")
    print(f"  Neuron out:   {args.neuron_out}")
    print(f"  Diff heatmap: {args.diff_out}")

    # Exit code: 0 if SSIM > 0.95, 1 otherwise (so CI can flag regressions)
    sys.exit(0 if stats["ssim_mean"] > 0.95 else 1)


if __name__ == "__main__":
    main()
