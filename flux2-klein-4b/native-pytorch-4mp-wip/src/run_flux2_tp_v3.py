#!/usr/bin/env python3
"""FLUX.2-klein-4B TP runner — v3 FULL sharding (attention + SwiGLU FFN +
single-stream fused blocks) in fp32.

v3 shards the bulk of the model (the 20 single-stream blocks + all FFNs),
not just the 5 double-stream attentions. This shards the replicated
activation that OOMs fp32 at >=3MP, so fp32 (the only correct recipe)
should now fit at higher resolution.

Launch:
    NEURON_RT_VIRTUAL_CORE_SIZE=2 NEURON_LOGICAL_NC_CONFIG=2 \
    NEURON_SKIP_EFA_AFFINITY=1 HF_TOKEN=... \
    torchrun --nproc_per_node=4 --rdzv_backend c10d \
        --rdzv_endpoint localhost:29500 run_flux2_tp_v3.py \
        --height 1792 --width 1792
"""
from __future__ import annotations

import argparse, os, sys, time
from datetime import timedelta

import torch
import torch_neuronx           # noqa: F401
import torch_neuronx.distributed   # noqa: F401
import torch.distributed as dist
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from neuron_flux2_klein_native import NeuronFlux2KleinPipeline
import flux2_tp_plan_v3 as v3
import flux2_attention_manual_flash as manual_attn
import flux2_attention_sdpa as sdpa_attn


def neuron_sync():
    if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance-scale", type=float, default=1.0)
    ap.add_argument("--height", type=int, default=1792)
    ap.add_argument("--width", type=int, default=1792)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--prompt", default="Zoom into the red highlighted area")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--flash-tile", type=int, default=0,
                    help="manual flash tile size (0=default 1024); larger "
                         "reduces online-softmax accumulation steps at high res")
    ap.add_argument("--probe-blocks", action="store_true",
                    help="print per-block hidden_states std on the first "
                         "denoising step (localizes high-res detail collapse)")
    ap.add_argument("--attn", choices=["manual", "sdpa"], default="manual",
                    help="attention impl: pure-Python tile flash (correct, slow) "
                         "or SDPA (fast, Neuron-fusible)")
    ap.add_argument("--output", default="/tmp/flux2_tp_v3.png")
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    dt = torch.float32 if args.dtype == "fp32" else torch.bfloat16

    def log(m):
        if rank == 0:
            print(f"[tp-v3] {m}", flush=True)

    dist.init_process_group(backend="neuron", timeout=timedelta(minutes=50))
    log(f"init OK ws={world_size} dtype={args.dtype}")
    device = torch.device("neuron")

    from torch.distributed.device_mesh import init_device_mesh
    mesh = init_device_mesh("neuron", (world_size,), mesh_dim_names=("tp",))

    t0 = time.time()
    pipe = NeuronFlux2KleinPipeline.from_pretrained(
        args.base_model, torch_dtype=dt, token=os.environ.get("HF_TOKEN"))
    log(f"loaded CPU {time.time()-t0:.1f}s")

    pipe.apply_neuron_patches(device, dtype=dt)
    log("neuron patches applied")

    # double-stream blocks use the manual flash processor; single-stream
    # uses the v3 processor (installed by restructure_for_tp).
    if args.flash_tile > 0:
        manual_attn.set_tile_size(args.flash_tile, args.flash_tile)
        log(f"manual flash tile size = {args.flash_tile}")
    if args.attn == "sdpa":
        # Install SDPA AFTER restructure_for_tp() so it overrides the v3
        # single-stream processor too. We still call install_manual to set
        # up _v3_split detection, but SDPA replaces both processors.
        manual_attn.install_manual_flash_processor(None)
        sdpa_attn.install_sdpa_processor()
        log("attention: SDPA (fast, Neuron-fusible)")
    else:
        manual_attn.install_manual_flash_processor(None)
        log("attention: manual tile flash (slow, correctness scaffold)")

    inner = pipe.transformer.inner

    # v3: split fused linears for sharding (CPU, weight-preserving)
    v3.restructure_for_tp(inner, rank=rank)

    from torch.distributed.tensor.parallel import parallelize_module
    plan = v3.flux2_tp_plan_v3(world_size)
    t0 = time.time()
    parallelize_module(inner, mesh, plan)
    log(f"parallelize_module ({len(plan)} entries) {time.time()-t0:.1f}s")
    v3.apply_tp_fixes_v3(inner, world_size, rank)

    # Per-block latent-magnitude probe: register forward hooks that print
    # the std of each block's hidden_states output on the FIRST forward
    # only (one denoising step), to localize where high-res detail dies.
    if args.probe_blocks and rank == 0:
        _probe_state = {"calls": 0, "rows": []}

        def _mk_hook(tag):
            def hook(mod, inp, out):
                if _probe_state["calls"] > 25 * 2:  # ~ first 2 steps worth
                    return
                hs = out[-1] if isinstance(out, (tuple, list)) else out
                try:
                    s = hs.float().std().item()
                    _probe_state["rows"].append((tag, s))
                    print(f"[probe] {tag}: hs_std={s:.4f} shape={tuple(hs.shape)}",
                          flush=True)
                except Exception as e:
                    print(f"[probe] {tag}: err {e}", flush=True)
                _probe_state["calls"] += 1
            return hook

        for i, blk in enumerate(inner.transformer_blocks):
            blk.register_forward_hook(_mk_hook(f"double[{i}]"))
        for i, blk in enumerate(inner.single_transformer_blocks):
            blk.register_forward_hook(_mk_hook(f"single[{i}]"))
        log("per-block probe hooks registered")

    _cache = {}
    _orig = pipe.prepare_image_latents

    def _cached(images, batch_size, generator, device=None, dtype=None):
        if "il" in _cache:
            return _cache["il"], _cache["ilids"]
        out = _orig(images, batch_size, generator, device, dtype)
        _cache["il"], _cache["ilids"] = out[0], out[1]
        return out

    pipe.prepare_image_latents = _cached

    # Diagnostic: print the DiT latent std right before VAE decode to
    # localize any high-res collapse (DiT latent vs VAE).
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "decode"):
        _orig_decode = pipe.vae.decode

        def _decode_dbg(z, *a, **k):
            if rank == 0:
                try:
                    zz = z.float()
                    log(f"[latent] pre-VAE std={zz.std().item():.4f} "
                        f"mean={zz.mean().item():.4f} shape={tuple(z.shape)}")
                except Exception:
                    pass
            return _orig_decode(z, *a, **k)

        pipe.vae.decode = _decode_dbg

    t0 = time.time()
    pipe.transformer.to(device)
    neuron_sync()
    log(f"sharded DiT on neuron {time.time()-t0:.1f}s")

    img = Image.new("RGB", (args.width, args.height), color=(180, 180, 180))
    d = ImageDraw.Draw(img)
    d.rectangle([args.width//4, args.height//4, 3*args.width//4,
                 3*args.height//4], outline=(255, 0, 0), width=8)

    prompt_embeds, _ = pipe.encode_prompt(
        prompt=args.prompt, device=pipe._execution_device,
        num_images_per_prompt=1)
    neuron_sync()
    log("prompt encoded")

    log("=== first call (compile) ===")
    t0 = time.time()
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    out = pipe(prompt_embeds=prompt_embeds, image=img, height=args.height,
               width=args.width, num_inference_steps=args.steps,
               guidance_scale=args.guidance_scale, generator=gen)
    neuron_sync()
    log(f"first call {time.time()-t0:.1f}s")
    if rank == 0:
        out.images[0].save(args.output)
        log(f"wrote {args.output}")

    times = []
    for i in range(args.runs):
        t0 = time.time()
        gen = torch.Generator(device="cpu").manual_seed(args.seed)
        out = pipe(prompt_embeds=prompt_embeds, image=img, height=args.height,
                   width=args.width, num_inference_steps=args.steps,
                   guidance_scale=args.guidance_scale, generator=gen)
        neuron_sync()
        times.append(time.time() - t0)
        log(f"warm {i}: {times[-1]:.2f}s")

    if rank == 0 and times:
        import numpy as np
        im = np.array(out.images[0])
        print(f"\n=== FLUX.2 v3 TP={world_size} {args.dtype} "
              f"{args.height}x{args.width} ===", flush=True)
        print(f"  warm avg {sum(times)/len(times):.2f}s  min {min(times):.2f}s",
              flush=True)
        print(f"  quality std={im.std():.2f} (correct ~18, blank ~2-5)",
              flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
