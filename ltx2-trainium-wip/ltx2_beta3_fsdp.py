"""LTX-2 19B native PyTorch on Trainium2 — Beta 3 + FSDP across 4 cores.

Per Beta 3 user guide:
  NEURON_RT_VIRTUAL_CORE_SIZE=2 NEURON_RT_NUM_CORES=4 \
    torchrun --nproc_per_node=4 --rdzv_backend c10d \
             --rdzv_endpoint localhost:29500 ltx2_beta3_fsdp.py

This version shards the LTX-2 transformer across 4 logical cores via
FSDP. The text encoder (Gemma3-12B) and VAE stay on CPU since they're
load-once components and the text encoder doesn't fit alongside the
transformer's shard.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
import torch_neuronx  # noqa: F401


def setup_distributed():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not dist.is_initialized():
        # Beta 3 PG backend is still 'neuron' (same as Beta 2). The
        # change in Beta 3 is the device API (torch.device("neuron"))
        # and the rdzv backend (--rdzv_backend c10d). The PG backend
        # itself is unchanged.
        dist.init_process_group(backend="neuron")
    return rank, world_size


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num-frames", type=int, default=25)
    p.add_argument("--num-steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default=(
        "A golden retriever puppy runs across a sunny green meadow, "
        "its ears flapping in the wind. The camera follows from a low angle. "
        "Birds chirp in the background."
    ))
    p.add_argument("--output", default="/opt/dlami/nvme/ltx2/results/ltx2_beta3_fsdp.png")
    p.add_argument("--compile", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    rank, world_size = setup_distributed()

    if rank == 0:
        print(f"[ltx2-beta3-fsdp] torch={torch.__version__}, "
              f"torch_neuronx={torch_neuronx.__version__}", flush=True)
        print(f"[ltx2-beta3-fsdp] world_size={world_size}", flush=True)

    device = torch.device("neuron")
    cpu = torch.device("cpu")

    # Probe device on each rank
    x = torch.randn(4, 4, device=device)
    y = x @ x.T
    if rank == 0:
        print(f"[ltx2-beta3-fsdp] device probe ok: {y.shape}", flush=True)

    # Load LTX-2 pipeline on CPU first
    if rank == 0:
        print(f"\n[ltx2-beta3-fsdp] loading Lightricks/LTX-2 (CPU, bf16)...",
              flush=True)
    t0 = time.time()
    from diffusers import LTX2Pipeline
    pipe = LTX2Pipeline.from_pretrained(
        "Lightricks/LTX-2",
        torch_dtype=torch.bfloat16,
    )
    if rank == 0:
        print(f"[ltx2-beta3-fsdp] loaded in {time.time() - t0:.1f}s", flush=True)

    # Wrap transformer with FSDP for cross-rank sharding
    if world_size > 1:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy

        if rank == 0:
            print(f"[ltx2-beta3-fsdp] wrapping transformer with FSDP "
                  f"({world_size} ranks)...", flush=True)
        t0 = time.time()
        # FSDP needs the model on Neuron first, then we wrap
        pipe.transformer = pipe.transformer.to(device)
        if rank == 0:
            print(f"[ltx2-beta3-fsdp] transformer.to(neuron) done in "
                  f"{time.time() - t0:.1f}s", flush=True)

        pipe.transformer = FSDP(
            pipe.transformer,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            device_id=None,  # Neuron isn't accessed via cuda device id
        )
        if rank == 0:
            print(f"[ltx2-beta3-fsdp] FSDP wrap done", flush=True)
    else:
        if rank == 0:
            print(f"[ltx2-beta3-fsdp] world_size=1, skipping FSDP wrap", flush=True)
        pipe.transformer = pipe.transformer.to(device)

    if args.compile and rank == 0:
        print(f"[ltx2-beta3-fsdp] wrapping with torch.compile(backend='neuron')",
              flush=True)
        pipe.transformer = torch.compile(
            pipe.transformer, backend="neuron",
            dynamic=False, fullgraph=False,
        )

    # Generate (only rank 0 produces output, but all ranks run lockstep)
    if rank == 0:
        print(f"\n[ltx2-beta3-fsdp] generating: {args.width}×{args.height}, "
              f"{args.num_frames} frames, {args.num_steps} steps...", flush=True)

    t0 = time.time()
    try:
        with torch.no_grad():
            result = pipe(
                prompt=args.prompt,
                height=args.height, width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_steps,
                guidance_scale=4.0,
                max_sequence_length=1024,
                generator=torch.Generator(device="cpu").manual_seed(args.seed),
                output_type="pil",
            )
    except Exception as e:
        elapsed = time.time() - t0
        if rank == 0:
            import traceback
            print(f"[ltx2-beta3-fsdp] ✗ generation FAILED after {elapsed:.1f}s",
                  flush=True)
            traceback.print_exc()
        return 1

    elapsed = time.time() - t0
    if rank == 0:
        print(f"[ltx2-beta3-fsdp] ✓ generated in {elapsed:.1f}s "
              f"({elapsed/args.num_steps:.2f}s/step avg)", flush=True)
        frames = result.frames[0]
        print(f"[ltx2-beta3-fsdp] frames: {len(frames)}", flush=True)

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        if frames:
            frames[0].save(out)
            print(f"[ltx2-beta3-fsdp] WROTE {out}", flush=True)
            try:
                from diffusers.utils import export_to_video
                mp4 = out.with_suffix(".mp4")
                export_to_video(frames, str(mp4), fps=24)
                print(f"[ltx2-beta3-fsdp] WROTE {mp4}", flush=True)
            except Exception as e:
                print(f"[ltx2-beta3-fsdp] mp4 export skip: {e}", flush=True)

    if dist.is_initialized():
        dist.barrier()
    return 0


if __name__ == "__main__":
    sys.exit(main())
