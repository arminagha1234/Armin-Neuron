"""LTX-2 19B native PyTorch on Trainium2 — Beta 3 + TP=4.

End-to-end pattern adapted from Qwen-Image-Edit Path C `run_simple.py`:

  1. setup_distributed() — torchrun launches 4 ranks; backend='neuron'
  2. Build the LTX-2 transformer under torch.device("meta")
  3. parallelize_module(...) with the LTX-2 TP plan
  4. apply_tp_fixes() — patch attn.heads to heads/N
  5. load_weights_sharded() — stream weights from disk per-rank
  6. Build the diffusers LTX2Pipeline on CPU and swap in our sharded transformer
  7. Patch CPU↔Neuron boundaries (text encoder + VAE stay on CPU)
  8. Generate

Launch:
    NEURON_RT_VIRTUAL_CORE_SIZE=2 NEURON_RT_NUM_CORES=4 \\
    torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint localhost:29500 \\
        ltx2_run.py --num-steps 4 --num-frames 25 --width 512 --height 384
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
import torch_neuronx  # noqa: F401

# Add this dir to path so we can import ltx2_tp_plan / ltx2_meta_loader
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ltx2_tp_plan import (
    ltx2_tp_plan, apply_tp_fixes, patch_rope_rank_slice,
    install_adaptive_qk_norm,
)
from ltx2_meta_loader import load_weights_sharded


def setup_distributed():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world_size > 1 and not dist.is_initialized():
        from datetime import timedelta
        dist.init_process_group(backend="neuron", timeout=timedelta(minutes=30))
    return rank, world_size, local_rank


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num-frames", type=int, default=25)
    p.add_argument("--num-steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default=(
        "A golden retriever puppy runs across a sunny green meadow, "
        "its ears flapping in the wind. The camera follows from a low angle. "
        "Birds chirp in the background."
    ))
    p.add_argument("--output", default="/opt/dlami/nvme/ltx2/results/ltx2_run.png")
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--no-compile", action="store_true",
                   help="Skip torch.compile (eager mode)")
    return p.parse_args()


def build_transformer(rank, world_size, local_rank, device, dtype):
    """Construct the LTX-2 transformer under meta-init, apply TP, load weights."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import parallelize_module
    from diffusers import LTX2VideoTransformer3DModel
    from huggingface_hub import hf_hub_download, snapshot_download

    if rank == 0:
        print(f"[ltx2_run] downloading transformer config + weights...", flush=True)

    # 1. Get transformer config (small JSON)
    config_path = hf_hub_download(
        "Lightricks/LTX-2", "transformer/config.json",
    )
    with open(config_path) as f:
        cfg = json.load(f)

    # 2. Get the local path of the transformer subdir (will pull weights into HF cache)
    snap = snapshot_download(
        "Lightricks/LTX-2", allow_patterns=["transformer/*"],
    )
    weights_dir = os.path.join(snap, "transformer")
    if rank == 0:
        print(f"[ltx2_run] weights dir: {weights_dir}", flush=True)

    # 3. Build under meta-init
    if rank == 0:
        print(f"[ltx2_run] building model under torch.device('meta')...", flush=True)
    with torch.device("meta"):
        model = LTX2VideoTransformer3DModel.from_config(cfg)
    if rank == 0:
        n_total = sum(p.numel() for p in model.parameters())
        print(f"[ltx2_run] meta-model: {n_total/1e9:.2f}B total params", flush=True)

    # 4. Apply TP plan
    mesh = init_device_mesh("neuron", (world_size,))
    plan = ltx2_tp_plan(world_size)
    if rank == 0:
        print(f"[ltx2_run] applying TP plan ({len(plan)} entries) on world={world_size}...",
              flush=True)
    parallelize_module(model, mesh, plan)

    # 5. Apply non-DTensor fixes (attn.heads patching, RoPE rank slice)
    apply_tp_fixes(model, world_size=world_size, rank=rank)

    # 6. Load weights sharded
    if rank == 0:
        print(f"[ltx2_run] loading weights sharded...", flush=True)
    t0 = time.time()
    load_weights_sharded(
        model, weights_dir,
        tp_local_rank=rank, world_size=world_size,
        dtype=dtype, device=device,
    )
    if rank == 0:
        print(f"[ltx2_run] weights loaded in {time.time() - t0:.1f}s", flush=True)

    # 6a. Safety net: materialize ANY param still on meta by looking it
    # up in the checkpoint by full dotted name. Catches norm_q/norm_k and
    # any other param the module-walk resolver missed post-parallelize.
    _materialize_remaining_meta(model, weights_dir, rank, world_size, device, dtype)

    # 6b. NOW install adaptive QK norm (AFTER weights load, so the loader
    # could materialize norm_q/norm_k.weight before we wrap them).
    install_adaptive_qk_norm(model, world_size=world_size, rank=rank)

    # 7. Apply RoPE rank slice AFTER weights load (since rope module
    # state is meta-initialized too; we need the patch on the resolved
    # forward).
    patch_rope_rank_slice(model, world_size=world_size, rank=rank)

    model.eval()
    return model, cfg


def _materialize_remaining_meta(model, weights_dir, rank, world_size, device, dtype):
    """Fill any param still on `meta` with its checkpoint tensor.

    norm_q/norm_k (rms_norm_across_heads) are full inner_dim and NOT in
    the TP plan, so they're replicated — load the full tensor on every
    rank. Other meta params get the same treatment.
    """
    import json as _json
    from pathlib import Path as _Path
    from safetensors import safe_open as _safe_open

    idx_path = _Path(weights_dir) / "diffusion_pytorch_model.safetensors.index.json"
    weight_map = _json.loads(idx_path.read_text())["weight_map"]

    # Group remaining meta params by shard file
    meta_params = {}
    for name, p in model.named_parameters():
        if getattr(p, "is_meta", False):
            meta_params[name] = p
    if rank == 0:
        print(f"[ltx2_run] materialize pass: {len(meta_params)} params still on meta",
              flush=True)
    if not meta_params:
        return

    by_shard = {}
    for name in meta_params:
        sf = weight_map.get(name)
        if sf is None:
            continue
        by_shard.setdefault(sf, []).append(name)

    filled = 0
    for shard_file, names in by_shard.items():
        with _safe_open(_Path(weights_dir) / shard_file, framework="pt", device="cpu") as f:
            for name in names:
                full = f.get_tensor(name).to(dtype)
                # Walk to parent module and set the leaf
                parts = name.split(".")
                *parent_path, leaf = parts
                mod = model
                ok = True
                for p in parent_path:
                    if not hasattr(mod, p):
                        ok = False
                        break
                    mod = getattr(mod, p)
                if not ok:
                    continue
                setattr(mod, leaf, torch.nn.Parameter(full.to(device), requires_grad=False))
                filled += 1
    if rank == 0:
        print(f"[ltx2_run] materialize pass: filled {filled} meta params", flush=True)


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()

    if rank == 0:
        print(f"[ltx2_run] torch={torch.__version__}", flush=True)
        print(f"[ltx2_run] world_size={world_size}, args={vars(args)}", flush=True)

    # Beta 3 device API
    device = torch.device("neuron")
    cpu = torch.device("cpu")
    dtype = torch.bfloat16

    # Sanity: probe the device
    x = torch.randn(4, 4, device=device)
    _ = x @ x.T
    if rank == 0:
        print(f"[ltx2_run] neuron device probe ok", flush=True)

    # Build TP-sharded transformer
    transformer, cfg = build_transformer(rank, world_size, local_rank, device, dtype)

    # Optionally compile
    if not args.no_compile:
        if rank == 0:
            print(f"[ltx2_run] wrapping transformer with torch.compile(backend='neuron')",
                  flush=True)
        transformer = torch.compile(
            transformer, backend="neuron",
            dynamic=False, fullgraph=False,
        )

    # Build the pipeline (CPU) and swap in our transformer
    if rank == 0:
        print(f"[ltx2_run] loading LTX2Pipeline (CPU)...", flush=True)
    t0 = time.time()
    from diffusers import LTX2Pipeline
    pipe = LTX2Pipeline.from_pretrained(
        "Lightricks/LTX-2", torch_dtype=dtype,
    )
    if rank == 0:
        print(f"[ltx2_run] pipeline loaded in {time.time() - t0:.1f}s", flush=True)

    # Free the CPU transformer; we have our own
    del pipe.transformer
    import gc; gc.collect()

    # Wrap our sharded transformer so ALL tensor inputs (hidden states,
    # encoder_hidden_states from CPU connectors, masks, rope tables, etc.)
    # are moved to Neuron before the forward runs. This is the single
    # CPU→Neuron chokepoint into the transformer — avoids chasing each
    # individual arg.
    import types as _types

    _transformer_inner = transformer

    def _move_to_neuron(obj):
        if torch.is_tensor(obj):
            return obj.to(device) if obj.device != device else obj
        if isinstance(obj, (list, tuple)):
            return type(obj)(_move_to_neuron(o) for o in obj)
        if isinstance(obj, dict):
            return {k: _move_to_neuron(v) for k, v in obj.items()}
        return obj

    class _NeuronTransformerWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            # expose config for the pipeline
            self.config = inner.config if hasattr(inner, "config") else (
                inner._orig_mod.config if hasattr(inner, "_orig_mod") else None
            )

        def forward(self, *args, **kwargs):
            args = tuple(_move_to_neuron(a) for a in args)
            kwargs = {k: _move_to_neuron(v) for k, v in kwargs.items()}
            return self.inner(*args, **kwargs)

        def __getattr__(self, name):
            # delegate unknown attrs (e.g. dtype, gradient_checkpointing) to inner
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.inner, name)

    pipe.transformer = _NeuronTransformerWrapper(transformer)

    # Force the pipeline's _execution_device to Neuron at class level
    # (the default property returns CPU because text_encoder is on CPU)
    LTX2Pipeline._execution_device = property(lambda self: device)

    # Patch text encoder calls to keep them on CPU but move outputs to Neuron
    cpu_dev = torch.device("cpu")

    # Wrap _get_gemma_prompt_embeds: force CPU forward, then move embeds to Neuron
    _orig_get = pipe._get_gemma_prompt_embeds

    def _patched_get_gemma(prompt, device=None, dtype=None, **kw):
        # Force CPU for the encoder forward (it's not on Neuron)
        out = _orig_get(prompt, device=cpu_dev, dtype=dtype, **kw)
        # _get_gemma_prompt_embeds returns (embeds, mask)
        # Keep on CPU for now — connectors expect CPU input. Pipeline will
        # transfer to neuron after connectors run.
        return out
    pipe._get_gemma_prompt_embeds = _patched_get_gemma

    # Connectors module also runs on CPU — its `per_layer_masked_mean_norm`
    # produces a 3GB intermediate that won't fit on Neuron alongside the
    # transformer shard.
    if hasattr(pipe, "connectors"):
        try:
            pipe.connectors = pipe.connectors.to(cpu_dev)
            if rank == 0:
                print(f"[ltx2_run] connectors moved to CPU", flush=True)
        except Exception as e:
            if rank == 0:
                print(f"[ltx2_run] connectors.to(cpu) skipped: {e}", flush=True)

    # Wrap VAE encode/decode similarly — VAE stays on CPU
    if hasattr(pipe, "_encode_vae_image") and hasattr(pipe, "_encode_vae_image"):
        _orig_vae_enc = pipe._encode_vae_image

        def _patched_vae_enc(image, generator):
            if hasattr(image, "to"):
                image = image.to(cpu_dev)
            latents = _orig_vae_enc(image, generator)
            return latents.to(torch.device('neuron'))
        pipe._encode_vae_image = _patched_vae_enc

    _orig_vae_dec = pipe.vae.decode

    def _patched_vae_dec(z, return_dict=True):
        if hasattr(z, "to"):
            z = z.to(cpu_dev)
        return _orig_vae_dec(z, return_dict=return_dict)
    pipe.vae.decode = _patched_vae_dec

    if rank == 0:
        print(f"\n[ltx2_run] generating: {args.width}×{args.height}, "
              f"{args.num_frames} frames, {args.num_steps} steps...", flush=True)
        print(f"[ltx2_run] (cold first call includes NEFF compilation; expect 5-15 min)",
              flush=True)

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
            print(f"[ltx2_run] ✗ generation FAILED after {elapsed:.1f}s",
                  flush=True)
            traceback.print_exc()
        if dist.is_initialized():
            dist.barrier()
        return 1

    elapsed = time.time() - t0
    if rank == 0:
        print(f"[ltx2_run] ✓ generated in {elapsed:.1f}s "
              f"({elapsed/args.num_steps:.2f}s/step avg)", flush=True)
        frames = result.frames[0]
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        if frames:
            frames[0].save(out)
            print(f"[ltx2_run] WROTE {out} ({len(frames)} frames)", flush=True)
            try:
                from diffusers.utils import export_to_video
                mp4 = out.with_suffix(".mp4")
                export_to_video(frames, str(mp4), fps=24)
                print(f"[ltx2_run] WROTE {mp4}", flush=True)
            except Exception as e:
                print(f"[ltx2_run] mp4 export skip: {e}", flush=True)

    if dist.is_initialized():
        dist.barrier()
    return 0


if __name__ == "__main__":
    sys.exit(main())
