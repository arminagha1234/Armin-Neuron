"""CPU reference run — diffusers QwenImageEditPlusPipeline with merged
LoRA. Same input image + prompt + seed as run_simple.py, but everything
on CPU. Used as the ground-truth target for cosine comparison vs the
Trainium output.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--merged-transformer", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--prompt", default="show_from_a_different_camera_angle")
    p.add_argument("--num-steps", type=int, default=28)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/output_cpu_ref.png")
    args = p.parse_args()

    from diffusers import QwenImageEditPlusPipeline, QwenImageTransformer2DModel

    print(f"[cpu_ref] loading base pipeline")
    t0 = time.time()
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.base_model_path, torch_dtype=torch.bfloat16,
    )
    print(f"[cpu_ref] base loaded in {time.time() - t0:.1f}s")

    print(f"[cpu_ref] loading merged transformer (LoRA fused)")
    t0 = time.time()
    pipe.transformer = QwenImageTransformer2DModel.from_pretrained(
        args.merged_transformer, torch_dtype=torch.bfloat16,
    )
    print(f"[cpu_ref] merged transformer loaded in {time.time() - t0:.1f}s")

    img = Image.open(args.image).convert("RGB")
    print(f"[cpu_ref] running pipeline ({args.num_steps} steps, {args.height}×{args.width})")
    torch.manual_seed(args.seed)
    t0 = time.time()
    out = pipe(
        image=img,
        prompt=args.prompt,
        num_inference_steps=args.num_steps,
        true_cfg_scale=1.0,
        height=args.height,
        width=args.width,
    )
    print(f"[cpu_ref] pipeline done in {time.time() - t0:.1f}s")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.images[0].save(out_path)
    print(f"[cpu_ref] WROTE {out_path}")


if __name__ == "__main__":
    main()
