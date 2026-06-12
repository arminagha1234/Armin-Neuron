"""Path C — Simplified end-to-end runner. ALL RANKS run the same code.

Key insight: when using TP with parallelize_module + DTensor, every
rank already runs the same forward in lockstep — that's how the
TP collectives stay synchronized. So we don't need a broadcast wrapper
at all. Instead, ALL RANKS run:
    1. encoder + VAE on CPU (same inputs → same outputs, deterministic)
    2. transformer forward (TP collectives just work)
    3. VAE decode on CPU on rank 0 only (rank 0 saves the image)

This eliminates the broadcast wrapper entirely. The duplicate CPU work
is a few seconds — worth it to avoid the cross-rank desync problems.

We use the diffusers pipeline directly (no monkey-patching) but
override `_execution_device` so it places latents on Neuron for the
transformer call.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image

import torch_neuronx  # noqa: F401
try:
    import torch_neuronx.distributed  # noqa: F401
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from qwen_edit_meta_loader import load_weights_sharded  # noqa: E402
from qwen_edit_tp_plan import (  # noqa: E402
    HEAD_DIM, INNER_DIM, N_HEADS_FULL, N_LAYERS,
    apply_tp_fixes, qwen_edit_tp_plan,
)


def setup_distributed() -> tuple[int, int, int]:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world_size > 1 and not dist.is_initialized():
        from torch.distributed.distributed_c10d import Backend
        from datetime import timedelta
        backend = "neuron" if "neuron" in Backend.backend_type_map else "xla"
        if rank == 0:
            print(f"[dist] backend={backend} world={world_size}")
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
    return rank, world_size, local_rank


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--merged-transformer", default="")
    p.add_argument("--images", nargs="+", required=False, default=[])
    p.add_argument("--prompt", default="show the subject from a different angle")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num-steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default="results/output.png")
    p.add_argument("--tp", type=int, default=4)
    return p.parse_args()


def build_transformer(base_path, merged_path, rank, world_size, device):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import parallelize_module
    from diffusers import QwenImageTransformer2DModel

    transformer_dir = merged_path if merged_path else os.path.join(base_path, "transformer")
    cfg = json.loads((Path(transformer_dir) / "config.json").read_text())
    if rank == 0:
        print(f"[r{rank}] transformer source: {transformer_dir}")
        print(f"[r{rank}] heads={cfg['num_attention_heads']} head_dim={cfg['attention_head_dim']} layers={cfg['num_layers']}")

    with torch.device("meta"):
        model = QwenImageTransformer2DModel.from_config(cfg)

    mesh = init_device_mesh("neuron", (world_size,))
    plan = qwen_edit_tp_plan(world_size)
    parallelize_module(model, mesh, plan)
    apply_tp_fixes(model, world_size=world_size, rank=rank)

    t0 = time.time()
    load_weights_sharded(model, transformer_dir,
                         tp_local_rank=rank, world_size=world_size,
                         dtype=torch.bfloat16, device=device)
    if rank == 0:
        print(f"[r{rank}] weights streamed in {time.time() - t0:.1f}s")

    rope_mod = model.pos_embed
    if rope_mod.pos_freqs.is_meta or rope_mod.neg_freqs.is_meta:
        if rank == 0:
            print(f"[r{rank}] rebuilding QwenEmbedRope freqs")
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        rope_mod.pos_freqs = torch.cat([
            rope_mod.rope_params(pos_index, rope_mod.axes_dim[0], rope_mod.theta),
            rope_mod.rope_params(pos_index, rope_mod.axes_dim[1], rope_mod.theta),
            rope_mod.rope_params(pos_index, rope_mod.axes_dim[2], rope_mod.theta),
        ], dim=1)
        rope_mod.neg_freqs = torch.cat([
            rope_mod.rope_params(neg_index, rope_mod.axes_dim[0], rope_mod.theta),
            rope_mod.rope_params(neg_index, rope_mod.axes_dim[1], rope_mod.theta),
            rope_mod.rope_params(neg_index, rope_mod.axes_dim[2], rope_mod.theta),
        ], dim=1)

    model.eval()
    return model, cfg


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"privateuseone:{local_rank}")
    cpu = torch.device("cpu")

    if rank == 0:
        print(f"[r0] args: {vars(args)}")

    # ─── 1. Build TP transformer (all ranks, lockstep) ────────────────
    transformer, cfg = build_transformer(
        args.base_model_path, args.merged_transformer, rank, args.tp, device
    )

    # ─── 2. Load pipeline on ALL ranks (encoder + VAE + scheduler) ────
    # All ranks run the same setup. encoder/VAE on CPU (small model
    # work is duplicated 4× — cheap, ~30s total).
    if rank == 0:
        print(f"[r{rank}] loading pipeline (all ranks)")
    from diffusers import QwenImageEditPlusPipeline
    t0 = time.time()
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.base_model_path, torch_dtype=torch.bfloat16,
    )
    del pipe.transformer
    import gc; gc.collect()
    pipe.transformer = transformer  # all ranks plug in their TP-sharded transformer
    if rank == 0:
        print(f"[r{rank}] pipeline loaded in {time.time() - t0:.1f}s")

    # Override _execution_device so the pipeline puts inputs on Neuron
    QwenImageEditPlusPipeline._execution_device = property(  # type: ignore[assignment]
        lambda self: device
    )

    # encoder + VAE stay on CPU
    # (we monkey-patch _encode_vae_image to handle CPU→Neuron crossing)
    _original_encode_vae_image = pipe._encode_vae_image.__func__

    def _patched_encode_vae_image(self, image, generator):
        if hasattr(image, "to"):
            image = image.to(cpu)
        latents = _original_encode_vae_image(self, image=image, generator=generator)
        return latents.to(device)

    import types
    pipe._encode_vae_image = types.MethodType(_patched_encode_vae_image, pipe)

    # encoder→neuron crossing for prompt embeds
    _original_get_qwen_prompt_embeds = pipe._get_qwen_prompt_embeds.__func__

    def _patched_get_qwen_prompt_embeds(self, prompt, image=None, device=None, dtype=None):
        embeds, mask = _original_get_qwen_prompt_embeds(
            self, prompt, image=image, device=cpu, dtype=dtype
        )
        embeds = embeds.to(torch.device(f"privateuseone:{local_rank}"))
        if mask is not None:
            mask = mask.to(torch.device(f"privateuseone:{local_rank}"))
        return embeds, mask

    pipe._get_qwen_prompt_embeds = types.MethodType(_patched_get_qwen_prompt_embeds, pipe)

    # Patch vae.decode to move latents to CPU first (VAE is on CPU)
    _original_vae_decode = pipe.vae.decode

    def _patched_vae_decode(z, return_dict=True):
        if hasattr(z, "to"):
            z = z.to(cpu)
        result = _original_vae_decode(z, return_dict=return_dict)
        # Move result back to device for compatibility (then it gets
        # moved to CPU at postprocess time anyway)
        if hasattr(result, "sample") and hasattr(result.sample, "to"):
            return result
        return result

    pipe.vae.decode = _patched_vae_decode

    # Also patch the latents normalisation tensors that the pipeline
    # creates from `vae.config.latents_mean/std` — they go to whatever
    # device the latent is on. The unpacked latents come off Neuron;
    # they go through `.to(latents.device)` which is Neuron, but the
    # VAE conv expects CPU. Easiest: monkey-patch _unpack_latents to
    # produce a CPU-side latent. Actually simpler: the pipeline does
    # `latents = latents / latents_std + latents_mean` and then
    # `image = self.vae.decode(latents, return_dict=False)`.
    # We just need vae.decode to handle Neuron→CPU.

    # cache_context no-op for our wrapper (transformer doesn't use a cache)
    from contextlib import nullcontext
    transformer.cache_context = lambda *a, **kw: nullcontext()

    # ─── 3. Load images (all ranks) ───────────────────────────────────
    if args.images:
        input_images = [Image.open(p).convert("RGB") for p in args.images]
    else:
        input_images = [Image.new("RGB", (args.width, args.height), (180, 200, 150))]
    if rank == 0:
        print(f"[r{rank}] loaded {len(input_images)} input image(s)")

    # ─── 4. Call pipeline (all ranks run the same pipeline; transformer
    # internally does TP collectives in lockstep) ─────────────────────
    if rank == 0:
        print(f"[r{rank}] running pipeline ({args.num_steps} steps, {args.height}×{args.width})")
    torch.manual_seed(args.seed)
    t0 = time.time()
    result = pipe(
        image=input_images if len(input_images) > 1 else input_images[0],
        prompt=args.prompt,
        num_inference_steps=args.num_steps,
        true_cfg_scale=1.0,  # no CFG for simplicity
        height=args.height,
        width=args.width,
    )
    if rank == 0:
        print(f"[r{rank}] pipeline done in {time.time() - t0:.1f}s")

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(out_path)
        print(f"[r{rank}] WROTE {out_path}")

    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
