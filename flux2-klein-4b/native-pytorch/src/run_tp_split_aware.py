#!/usr/bin/env python3
"""Run FLUX.2-klein-4B with split-aware TP=N inside a single Neuron logical core.

Launches via torchrun:

    NEURON_RT_VISIBLE_CORES=0-1 NEURON_RT_VIRTUAL_CORE_SIZE=2 \\
        torchrun --nproc_per_node=2 --rdzv_backend=c10d \\
            --rdzv_endpoint=localhost:29500 \\
            src/run_tp_split_aware.py \\
            --image input.jpg --steps 28

Each rank holds 1/N of every Flux2ParallelSelfAttention module's
weights; the all_reduce in to_out keeps outputs identical to the
unsharded run (within bf16 precision).

Status (2026-06-13)
-------------------

UNTESTED on Trainium. The split-aware sharding math is verified by
`tp_split_aware_smoke.py` on CPU at fp32 (max diff 2.4e-7). What's not
yet verified:

1. `init_process_group(backend="c10d")` on Beta 3 — the Beta 2 rule said
   `backend="neuron"`, but Beta 3 release notes claim c10d works.
2. `init_device_mesh("neuron", (world,))` on Beta 3.
3. `dist.all_reduce` performance under `torch.compile(backend="neuron")`
   when called inside the 48 single-stream blocks.

Expected speedup vs single-core compile baseline (65.9 s / 28 steps):
~1.6-1.8× per step. Combined with batch parallelism (2 procs × TP=2
spread across 4 cores) we'd target ~$0.012/image, but that requires
LNC=1 (currently blocked at the host driver level — see batch_parallel
docs).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from neuron_flux2_klein_native import NeuronFlux2KleinPipeline
from tp_split_aware import apply_split_aware_tp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--lora", default=None)
    ap.add_argument("--lora-scale", type=float, default=1.1)
    ap.add_argument("--image", required=True)
    ap.add_argument("--prompt", default="Zoom into the red highlighted area")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None)
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--backend", default="c10d",
                    help="Backend for init_process_group. Beta 3: c10d. Beta 2: neuron.")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    def log(msg):
        print(f"[rank{rank}/{world}] {msg}", flush=True)

    log(f"initializing distributed (backend={args.backend})")
    dist.init_process_group(backend=args.backend)

    from torch.distributed.device_mesh import init_device_mesh
    mesh = init_device_mesh("neuron", (world,))
    log(f"mesh ready: size={mesh.size()}, local_rank={mesh.get_local_rank()}")

    log(f"loading {args.base_model}")
    t0 = time.time()
    pipe = NeuronFlux2KleinPipeline.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
    )
    if args.lora:
        pipe.load_lora_weights(args.lora)
        pipe.fuse_lora(lora_scale=args.lora_scale)
        pipe.unload_lora_weights()
        log(f"LoRA fused (scale={args.lora_scale}) in {time.time()-t0:.1f}s")
    else:
        log(f"base loaded in {time.time()-t0:.1f}s")

    # Apply split-aware TP BEFORE moving to device — sharding works on
    # the CPU weights, then the per-rank slices move to Neuron.
    log("applying split-aware TP")
    n_replaced = apply_split_aware_tp(pipe.transformer.inner, mesh)
    log(f"sharded {n_replaced} attention modules across {world} ranks")

    device = torch.device("neuron")
    pipe.apply_neuron_patches(device, dtype=torch.bfloat16)
    pipe.transformer.to(device)
    log("transformer on neuron")

    if not args.no_compile:
        pipe.transformer.inner = torch.compile(
            pipe.transformer.inner, backend="neuron", dynamic=False,
        )
        log("torch.compile applied")

    img = Image.open(args.image).convert("RGB").resize(
        (args.width, args.height), Image.LANCZOS,
    )
    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    log(f"=== warmup ({args.steps} steps, {args.height}×{args.width}) ===")
    t0 = time.time()
    out = pipe(
        prompt=args.prompt, image=img,
        height=args.height, width=args.width,
        num_inference_steps=args.steps, guidance_scale=3.5, generator=gen,
    )
    if hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()
    log(f"warmup: {time.time()-t0:.1f}s")

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    log(f"=== timed ===")
    t0 = time.time()
    out = pipe(
        prompt=args.prompt, image=img,
        height=args.height, width=args.width,
        num_inference_steps=args.steps, guidance_scale=3.5, generator=gen,
    )
    if hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()
    elapsed = time.time() - t0
    log(f"timed: {elapsed:.1f}s ({elapsed*1000/args.steps:.0f} ms/step)")

    # Only rank 0 saves the output (all ranks should produce identical PNG).
    if rank == 0:
        out_path = args.output or f"./flux_tp{world}_{args.height}.png"
        out.images[0].save(out_path)
        log(f"saved {out_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
