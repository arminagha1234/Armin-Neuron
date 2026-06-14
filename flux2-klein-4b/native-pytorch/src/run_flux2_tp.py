#!/usr/bin/env python3
"""FLUX.2-klein-4B TP=4 runner — native PyTorch + Beta 3.

Shards the DiT across 4 Neuron cores via torch.distributed tensor
parallelism (parallelize_module), keeping text encoder + VAE replicated
on CPU per rank. Goal: split the ~730ms/step single-rank DiT floor
across 4 cores.

Launch:
    NEURON_RT_VIRTUAL_CORE_SIZE=2 NEURON_RT_NUM_CORES=8 \\
    NEURON_SKIP_EFA_AFFINITY=1 FI_PROVIDER=efa \\
    NEURON_RT_ROOT_COMM_ID=localhost:48620 \\
    HF_TOKEN=... HF_HOME=/mnt/data/hf_cache \\
    torchrun --nproc_per_node=4 --rdzv_backend c10d \\
        --rdzv_endpoint localhost:29500 run_flux2_tp.py --runs 3
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
# CRITICAL imports — register Neuron device + neuron PG backend
import torch_neuronx              # noqa: F401
import torch_neuronx.distributed  # noqa: F401
import torch.distributed as dist
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from neuron_flux2_klein_native import NeuronFlux2KleinPipeline
import flux2_tp_plan as tp
import flux2_attention_cte as kernel_mod


def neuron_sync():
    if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance-scale", type=float, default=1.0)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default="Zoom into the red highlighted area")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--output", default="/tmp/flux2_tp_out.png")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    def log(msg):
        if rank == 0:
            print(f"[tp run] {msg}", flush=True)

    # ── Init distributed ──────────────────────────────────────────────
    from datetime import timedelta
    dist.init_process_group(backend="neuron", timeout=timedelta(minutes=30))
    log(f"init_process_group OK, world_size={world_size}")

    device = torch.device("neuron")

    # ── Build device mesh for TP ──────────────────────────────────────
    from torch.distributed.device_mesh import init_device_mesh
    mesh = init_device_mesh("neuron", (world_size,), mesh_dim_names=("tp",))
    log(f"device mesh built: {mesh}")

    # ── Load pipeline on CPU ──────────────────────────────────────────
    t0 = time.time()
    pipe = NeuronFlux2KleinPipeline.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
    )
    log(f"pipeline loaded on CPU in {time.time()-t0:.1f}s")

    # Apply the CPU-side Neuron patches (scheduler, RoPE, pos_embed, VAE)
    # but DON'T move transformer to neuron yet — we shard first.
    pipe.apply_neuron_patches(device, dtype=torch.bfloat16)

    # Install the attention_cte flash kernel. At TP=4 the per-rank
    # attention is [1, 6, 8704, 128] — the default SDPA can't fit the
    # 8704x8704 score matrix in SBUF (NCC_INLA001 memory-out-of-bound).
    # attention_cte flash-tiles the sequence so it fits. THIS is where
    # the kernel pays off (vs single-rank where it was 18% slower).
    kernel_mod.install_attention_cte_processor(None)
    log("attention_cte flash kernel installed for sharded attention")

    # The wrapper's .inner is the real DiT. Shard it.
    inner = pipe.transformer.inner

    # ── Parallelize the DiT ───────────────────────────────────────────
    from torch.distributed.tensor.parallel import parallelize_module
    plan = tp.flux2_tp_plan(world_size)
    t0 = time.time()
    parallelize_module(inner, mesh, plan)
    log(f"parallelize_module applied ({len(plan)} entries) in {time.time()-t0:.1f}s")

    # Patch attn.heads / inner_dim / mlp_hidden_dim for sharded widths
    tp.apply_tp_fixes(inner, world_size, rank)

    # ── Phase A image-latent caching ──────────────────────────────────
    # Without this, prepare_image_latents (VAE encode + patchify +
    # batch-norm) runs every call on CPU (~24s) and dominates wall-clock.
    # Cache it after the first call.
    _cache = {}
    _orig_pil = pipe.prepare_image_latents

    def _caching_prepare_image_latents(images, batch_size, generator, device=None, dtype=None):
        if "il" in _cache:
            return _cache["il"], _cache["ilids"]
        out = _orig_pil(images, batch_size, generator, device, dtype)
        _cache["il"], _cache["ilids"] = out[0], out[1]
        return out

    pipe.prepare_image_latents = _caching_prepare_image_latents

    # Prompt caching: encode once, reuse.
    _prompt_embeds_cache = {}

    # ── Move sharded DiT to Neuron ────────────────────────────────────
    t0 = time.time()
    pipe.transformer.to(device)
    neuron_sync()
    log(f"sharded DiT on neuron in {time.time()-t0:.1f}s")

    # ── Build input ───────────────────────────────────────────────────
    img = Image.new("RGB", (args.width, args.height), color=(180, 180, 180))
    draw = ImageDraw.Draw(img)
    x0, y0 = args.width // 4, args.height // 4
    x1, y1 = 3 * args.width // 4, 3 * args.height // 4
    draw.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=8)
    input_image = img

    # ── First call (compiles) ─────────────────────────────────────────
    log("=== first call (compiles sharded graph) ===")
    # Pre-encode prompt once (Phase A prompt caching)
    prompt_embeds, _ = pipe.encode_prompt(
        prompt=args.prompt, device=pipe._execution_device, num_images_per_prompt=1,
    )
    neuron_sync()
    t0 = time.time()
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    out = pipe(
        prompt_embeds=prompt_embeds, image=input_image,
        height=args.height, width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale, generator=gen,
    )
    neuron_sync()
    log(f"first call: {time.time()-t0:.1f}s")
    if rank == 0:
        out.images[0].save(args.output)
        log(f"wrote {args.output}")

    # ── Warm runs (prompt + image-latents cached) ─────────────────────
    times = []
    for i in range(args.runs):
        t0 = time.time()
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        out = pipe(
            prompt_embeds=prompt_embeds, image=input_image,
            height=args.height, width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale, generator=gen,
        )
        neuron_sync()
        dt = time.time() - t0
        times.append(dt)
        log(f"warm run {i}: {dt:.2f}s")

    if rank == 0 and times:
        avg = sum(times) / len(times)
        import numpy as np
        im = np.array(out.images[0])
        print(f"\n=== FLUX.2-klein-4B TP={world_size} SUMMARY ===", flush=True)
        print(f"  warm avg: {avg:.2f}s   min: {min(times):.2f}s", flush=True)
        print(f"  quality: std={im.std():.2f}", flush=True)
        print(f"  vs Phase A single-rank baseline 6.86s: {6.86-avg:+.2f}s", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
