# SPDX-License-Identifier: Apache-2.0
"""Merge fal/flux-2-klein-4B-zoom-lora into FLUX.2-klein-4B base offline.

Why offline merge instead of `pipe.load_lora_weights(...)` at runtime:
  - vllm-omni's pipeline construction happens inside spawned engine
    workers; there is no driver-side hook to inject LoRA weights before
    the framework's strict-load check runs.
  - The cleanest v1 path is to merge LoRA → base weights once, point
    --model-path at the merged dir, and let vllm-omni load it like any
    other base checkpoint.
  - The fal LoRA is a transformer-only adapter (~76 MB), so the merge
    only touches the transformer subfolder; the encoder/VAE/tokenizer
    pass through unchanged.

This script uses the diffusers FluxPipeline-style merging API. It runs
on CPU, takes ~1-2 minutes, and writes the merged checkpoint to
--out-dir. After it finishes, run:

    python run_flux2_klein_omni.py --model-path <out-dir> --image <input.png>

Usage (inside the vllm_omni container or any container with diffusers 0.38+):

    python merge_lora.py \\
        --base-model black-forest-labs/FLUX.2-klein-4B \\
        --lora fal/flux-2-klein-4B-zoom-lora \\
        --lora-file flux-red-zoom-lora.safetensors \\
        --lora-scale 1.1 \\
        --out-dir /work/flux2_klein_merged_zoom_lora

The output dir contains the FLUX.2-klein-4B layout (scheduler/, vae/,
text_encoder/, tokenizer/, transformer/) with the LoRA fused into the
transformer's safetensors shards.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="black-forest-labs/FLUX.2-klein-4B")
    parser.add_argument("--lora", default="fal/flux-2-klein-4B-zoom-lora")
    parser.add_argument("--lora-file",
                        default="flux-red-zoom-lora.safetensors",
                        help="Specific safetensors file in the LoRA repo to use.")
    parser.add_argument("--lora-scale", type=float, default=1.1)
    parser.add_argument("--out-dir", required=True,
                        help="Where to write the merged checkpoint.")
    parser.add_argument("--dtype", default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    args = parser.parse_args()

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[merge_lora] loading base FLUX.2-klein from {args.base_model}", flush=True)
    # We use the upstream diffusers Flux2 pipeline here for merging, NOT
    # vllm-omni's Flux2KleinPipeline (which expects vllm parallel layers).
    from diffusers import Flux2Pipeline
    pipe = Flux2Pipeline.from_pretrained(args.base_model, torch_dtype=dtype)

    print(f"[merge_lora] loading LoRA: {args.lora}/{args.lora_file} "
          f"(scale={args.lora_scale})", flush=True)
    pipe.load_lora_weights(args.lora, weight_name=args.lora_file)
    print("[merge_lora] fusing LoRA into base weights...", flush=True)
    pipe.fuse_lora(lora_scale=args.lora_scale)
    pipe.unload_lora_weights()

    print(f"[merge_lora] saving merged checkpoint to {out_dir}", flush=True)
    pipe.save_pretrained(str(out_dir))

    print(f"[merge_lora] done. Use --model-path {out_dir} in run_flux2_klein_omni.py",
          flush=True)


if __name__ == "__main__":
    main()
