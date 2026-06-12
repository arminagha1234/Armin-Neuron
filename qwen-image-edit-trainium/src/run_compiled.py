"""Path C — torch.compile(backend='neuron') variant.

Identical to run_simple.py except: after building the TP-sharded
transformer, we wrap it with torch.compile so neuronx-cc captures the
entire 60-block forward as a single big NEFF.

Expected wins vs eager mode:
  - Cross-block op fusion (eager fires many small NEFFs per block)
  - Single large NEFF gets better memory layout + register allocation
  - No per-op dispatch overhead between blocks
  - Should drop ~3.8s/step → maybe 1.5-2s/step

Trade-off: first-compile is much longer (could be 10-30 min for a 20B
transformer), but the NEFF cache makes subsequent runs fast.

Launch:
    NEURON_RT_NUM_CORES=4 \\
    torchrun --nproc_per_node=4 --standalone run_compiled.py \\
        --base-model-path .../Qwen-Image-Edit-2511/snapshots/<sha> \\
        --merged-transformer .../merged_lora/transformer \\
        --images img.png \\
        --prompt "..." \\
        --num-steps 4
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
from rope_real import install_real_rope  # noqa: E402


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
    p.add_argument("--output", default="results/output_compiled.png")
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--no-compile", action="store_true",
                   help="Skip torch.compile (eager mode for A/B comparison)")
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

    # Install real-valued RoPE (REQUIRED for torch.compile — stock complex
    # RoPE crashes MLIR lowering with `DecomposeComplexOps pass crashed`).
    if rank == 0:
        print(f"[r{rank}] installing real-valued RoPE (compile-compatible)")
    install_real_rope(model)

    model.eval()
    return model, cfg


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"privateuseone:{local_rank}")
    cpu = torch.device("cpu")

    if rank == 0:
        print(f"[r0] args: {vars(args)}")

    transformer, cfg = build_transformer(
        args.base_model_path, args.merged_transformer, rank, args.tp, device
    )

    # ─── KEY DIFFERENCE FROM run_simple.py ─────────────────────────────
    # Wrap the transformer with torch.compile so neuronx-cc captures the
    # entire 60-block forward as one large NEFF.
    if not args.no_compile:
        if rank == 0:
            print(f"[r{rank}] wrapping transformer with torch.compile(backend='neuron')")
        # dynamic=False forces static shape capture (we always run with the
        # same height/width for a given session)
        # fullgraph=False allows graph breaks at non-tensor control flow
        # (img_shapes is a list of tuples — torch.compile needs to break there)
        transformer = torch.compile(
            transformer,
            backend="neuron",
            dynamic=False,
            fullgraph=False,
        )
    else:
        if rank == 0:
            print(f"[r{rank}] EAGER mode (--no-compile)")

    # ─── Pipeline (same as run_simple) ────────────────────────────────
    from diffusers import QwenImageEditPlusPipeline
    if rank == 0:
        print(f"[r{rank}] loading pipeline")
    t0 = time.time()
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.base_model_path, torch_dtype=torch.bfloat16,
    )
    del pipe.transformer
    import gc; gc.collect()
    pipe.transformer = transformer
    if rank == 0:
        print(f"[r{rank}] pipeline loaded in {time.time() - t0:.1f}s")

    QwenImageEditPlusPipeline._execution_device = property(  # type: ignore[assignment]
        lambda self: device
    )

    _original_get_qwen_prompt_embeds = pipe._get_qwen_prompt_embeds.__func__

    def _patched_get_qwen_prompt_embeds(self, prompt, image=None, device=None, dtype=None):
        embeds, mask = _original_get_qwen_prompt_embeds(
            self, prompt, image=image, device=cpu, dtype=dtype
        )
        embeds = embeds.to(torch.device(f"privateuseone:{local_rank}"))
        if mask is not None:
            mask = mask.to(torch.device(f"privateuseone:{local_rank}"))
        return embeds, mask

    import types
    pipe._get_qwen_prompt_embeds = types.MethodType(_patched_get_qwen_prompt_embeds, pipe)

    _original_encode_vae_image = pipe._encode_vae_image.__func__

    def _patched_encode_vae_image(self, image, generator):
        if hasattr(image, "to"):
            image = image.to(cpu)
        latents = _original_encode_vae_image(self, image=image, generator=generator)
        return latents.to(device)

    pipe._encode_vae_image = types.MethodType(_patched_encode_vae_image, pipe)

    _original_vae_decode = pipe.vae.decode

    def _patched_vae_decode(z, return_dict=True):
        if hasattr(z, "to"):
            z = z.to(cpu)
        return _original_vae_decode(z, return_dict=return_dict)

    pipe.vae.decode = _patched_vae_decode

    from contextlib import nullcontext
    # cache_context lives on the un-compiled transformer, so we set it on
    # the inner module if we wrapped:
    inner = transformer._orig_mod if hasattr(transformer, "_orig_mod") else transformer
    inner.cache_context = lambda *a, **kw: nullcontext()

    # ─── Load images ──────────────────────────────────────────────────
    if args.images:
        input_images = [Image.open(p).convert("RGB") for p in args.images]
    else:
        input_images = [Image.new("RGB", (args.width, args.height), (180, 200, 150))]
    if rank == 0:
        print(f"[r{rank}] loaded {len(input_images)} input image(s)")

    # ─── Run pipeline ─────────────────────────────────────────────────
    if rank == 0:
        print(f"[r{rank}] running pipeline ({args.num_steps} steps, {args.height}×{args.width})")
        print(f"[r{rank}] WARNING: first compile of the entire transformer may take 5-15 minutes")
    torch.manual_seed(args.seed)
    t0 = time.time()
    result = pipe(
        image=input_images if len(input_images) > 1 else input_images[0],
        prompt=args.prompt,
        num_inference_steps=args.num_steps,
        true_cfg_scale=1.0,
        height=args.height,
        width=args.width,
    )
    elapsed = time.time() - t0
    if rank == 0:
        print(f"[r{rank}] pipeline done in {elapsed:.1f}s ({elapsed/args.num_steps:.2f}s/step avg)")

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result.images[0].save(out_path)
        print(f"[r{rank}] WROTE {out_path}")

    if dist.is_initialized():
        dist.barrier()


if __name__ == "__main__":
    main()
