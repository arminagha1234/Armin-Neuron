"""LTX-2 19B native PyTorch on Trainium2 — Beta 3 + TP=4 — benchmark.

Loads the model ONCE, then runs N iterations across configurations.
Captures TTFI (cold first call), warm steady-state, and a sweep over
(num_steps, num_frames, resolution).

Launch:
    NEURON_RT_VIRTUAL_CORE_SIZE=2 NEURON_RT_NUM_CORES=4 \\
    torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint localhost:29500 \\
        bench_ltx2.py
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
import types
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
import torch_neuronx  # noqa: F401

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
    p.add_argument("--canonical-steps", type=int, default=8,
                   help="step count for canonical warm runs")
    p.add_argument("--canonical-frames", type=int, default=25)
    p.add_argument("--canonical-w", type=int, default=512)
    p.add_argument("--canonical-h", type=int, default=384)
    p.add_argument("--n-canonical-warm", type=int, default=3)
    p.add_argument("--sweep", action="store_true",
                   help="run additional (steps, frames, resolution) sweep")
    p.add_argument("--prompt", default=(
        "A golden retriever puppy runs across a sunny green meadow, "
        "its ears flapping in the wind. The camera follows from a low angle. "
        "Birds chirp in the background."
    ))
    p.add_argument("--out-json", default="/opt/dlami/nvme/ltx2/results/bench_ltx2.json")
    return p.parse_args()


def build_pipe(rank, world_size, dtype, device):
    """Build TP-sharded transformer + LTX2Pipeline with CPU↔Neuron patches."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import parallelize_module
    from diffusers import LTX2VideoTransformer3DModel, LTX2Pipeline
    from huggingface_hub import hf_hub_download, snapshot_download

    if rank == 0:
        print(f"[bench] building meta-init transformer...", flush=True)
    cfg_path = hf_hub_download("Lightricks/LTX-2", "transformer/config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    snap = snapshot_download("Lightricks/LTX-2", allow_patterns=["transformer/*"])
    weights_dir = os.path.join(snap, "transformer")

    with torch.device("meta"):
        model = LTX2VideoTransformer3DModel.from_config(cfg)

    mesh = init_device_mesh("neuron", (world_size,))
    plan = ltx2_tp_plan(world_size)
    parallelize_module(model, mesh, plan)
    apply_tp_fixes(model, world_size=world_size, rank=rank)

    if rank == 0:
        print(f"[bench] loading sharded weights...", flush=True)
    t0 = time.time()
    load_weights_sharded(
        model, weights_dir,
        tp_local_rank=rank, world_size=world_size,
        dtype=dtype, device=device,
    )
    if rank == 0:
        print(f"[bench] weights loaded in {time.time() - t0:.1f}s", flush=True)

    install_adaptive_qk_norm(model, world_size=world_size, rank=rank)
    patch_rope_rank_slice(model, world_size=world_size, rank=rank)
    model.eval()

    if rank == 0:
        print(f"[bench] loading LTX2Pipeline (CPU)...", flush=True)
    pipe = LTX2Pipeline.from_pretrained("Lightricks/LTX-2", torch_dtype=dtype)
    del pipe.transformer
    gc.collect()

    cpu_dev = torch.device("cpu")

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
            self.config = inner.config if hasattr(inner, "config") else None

        def forward(self, *args, **kwargs):
            args = tuple(_move_to_neuron(a) for a in args)
            kwargs = {k: _move_to_neuron(v) for k, v in kwargs.items()}
            return self.inner(*args, **kwargs)

        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.inner, name)

    pipe.transformer = _NeuronTransformerWrapper(model)
    LTX2Pipeline._execution_device = property(lambda self: device)

    if hasattr(pipe, "connectors"):
        try:
            pipe.connectors = pipe.connectors.to(cpu_dev)
        except Exception:
            pass

    try:
        pipe.vae = pipe.vae.to(cpu_dev)
    except Exception:
        pass
    if hasattr(pipe, "audio_vae"):
        try:
            pipe.audio_vae = pipe.audio_vae.to(cpu_dev)
        except Exception:
            pass

    _orig_get = pipe._get_gemma_prompt_embeds

    def _patched_get_gemma(prompt, device=None, dtype=None, **kw):
        return _orig_get(prompt, device=cpu_dev, dtype=dtype, **kw)
    pipe._get_gemma_prompt_embeds = _patched_get_gemma

    if hasattr(pipe, "_encode_vae_image"):
        _orig_vae_enc = pipe._encode_vae_image

        def _patched_vae_enc(image, generator):
            if hasattr(image, "to"):
                image = image.to(cpu_dev)
            latents = _orig_vae_enc(image, generator)
            return latents.to(device)
        pipe._encode_vae_image = _patched_vae_enc

    _orig_vae_dec = pipe.vae.decode

    def _patched_vae_dec(*args, **kwargs):
        new_args = [a.to(cpu_dev) if torch.is_tensor(a) else a for a in args]
        new_kwargs = {k: (v.to(cpu_dev) if torch.is_tensor(v) else v)
                      for k, v in kwargs.items()}
        return _orig_vae_dec(*new_args, **new_kwargs)
    pipe.vae.decode = _patched_vae_dec

    if hasattr(pipe, "audio_vae"):
        _orig_avd = pipe.audio_vae.decode

        def _patched_avd(*args, **kwargs):
            new_args = [a.to(cpu_dev) if torch.is_tensor(a) else a for a in args]
            new_kwargs = {k: (v.to(cpu_dev) if torch.is_tensor(v) else v)
                          for k, v in kwargs.items()}
            return _orig_avd(*new_args, **new_kwargs)
        pipe.audio_vae.decode = _patched_avd

    return pipe


def time_call(pipe, *, prompt, h, w, frames, steps, seed=42):
    t0 = time.time()
    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            height=h, width=w,
            num_frames=frames,
            num_inference_steps=steps,
            guidance_scale=4.0,
            max_sequence_length=1024,
            generator=torch.Generator(device="cpu").manual_seed(seed),
            output_type="pil",
        )
    elapsed = time.time() - t0
    # Drop result frames, force gc + cuda-equivalent buffer release
    del result
    gc.collect()
    return elapsed, None


def percentile(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    idx = q / 100.0 * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = torch.device("neuron")
    dtype = torch.bfloat16

    if rank == 0:
        print(f"[bench] torch={torch.__version__} world_size={world_size}", flush=True)
        print(f"[bench] canonical: {args.canonical_w}x{args.canonical_h} "
              f"frames={args.canonical_frames} steps={args.canonical_steps} "
              f"warm_iters={args.n_canonical_warm}", flush=True)

    # Probe device
    _ = torch.randn(4, 4, device=device) @ torch.randn(4, 4, device=device).T

    # Build pipeline (heavy)
    t_setup0 = time.time()
    pipe = build_pipe(rank, world_size, dtype, device)
    setup_s = time.time() - t_setup0
    if rank == 0:
        print(f"[bench] full setup: {setup_s:.1f}s", flush=True)

    results = {
        "stack": "Beta 3, native PyTorch, torch_neuronx, TP=4",
        "torch_version": torch.__version__,
        "world_size": world_size,
        "setup_seconds": setup_s,
        "canonical": {
            "width": args.canonical_w, "height": args.canonical_h,
            "frames": args.canonical_frames, "steps": args.canonical_steps,
        },
    }

    # ===== Cold first call (TTFI) =====
    if rank == 0:
        print(f"\n[bench] === TTFI (cold first call, includes NEFF compile) ===", flush=True)
    ttfi_s, _ = time_call(
        pipe, prompt=args.prompt,
        h=args.canonical_h, w=args.canonical_w,
        frames=args.canonical_frames, steps=args.canonical_steps,
        seed=42,
    )
    results["ttfi_seconds"] = ttfi_s
    if rank == 0:
        print(f"[bench] TTFI: {ttfi_s:.1f}s", flush=True)

    # ===== Warm steady-state =====
    if rank == 0:
        print(f"\n[bench] === Warm steady-state ({args.n_canonical_warm} iters) ===", flush=True)
    warm_runs = []
    for i in range(args.n_canonical_warm):
        t, _ = time_call(
            pipe, prompt=args.prompt,
            h=args.canonical_h, w=args.canonical_w,
            frames=args.canonical_frames, steps=args.canonical_steps,
            seed=42 + i + 1,
        )
        warm_runs.append(t)
        if rank == 0:
            print(f"[bench]   warm[{i}]: {t:.2f}s", flush=True)

    if rank == 0 and warm_runs:
        results["warm"] = {
            "samples_seconds": warm_runs,
            "n": len(warm_runs),
            "mean": statistics.mean(warm_runs),
            "median": statistics.median(warm_runs),
            "stdev": statistics.stdev(warm_runs) if len(warm_runs) > 1 else 0.0,
            "p50": percentile(warm_runs, 50),
            "p95": percentile(warm_runs, 95) if len(warm_runs) >= 3 else None,
            "p99": percentile(warm_runs, 99) if len(warm_runs) >= 3 else None,
            "per_step_mean_s": statistics.mean(warm_runs) / args.canonical_steps,
        }
        print(f"\n[bench] === WARM SUMMARY ===", flush=True)
        print(f"  mean   {results['warm']['mean']:.2f}s", flush=True)
        print(f"  median {results['warm']['median']:.2f}s", flush=True)
        print(f"  stdev  {results['warm']['stdev']:.2f}s", flush=True)
        print(f"  per-step {results['warm']['per_step_mean_s']:.2f}s/step", flush=True)

    # ===== Sweep (optional) =====
    if args.sweep:
        if rank == 0:
            print(f"\n[bench] === Sweep ===", flush=True)
        sweep = {}

        # Steps sweep at canonical resolution
        steps_sweep = []
        for steps in [4, 8, 16, 25]:
            t, _ = time_call(
                pipe, prompt=args.prompt,
                h=args.canonical_h, w=args.canonical_w,
                frames=args.canonical_frames, steps=steps, seed=99 + steps,
            )
            steps_sweep.append({"steps": steps, "seconds": t,
                                "per_step": t / steps})
            if rank == 0:
                print(f"[bench]   steps={steps}: {t:.2f}s ({t/steps:.2f}s/step)", flush=True)
        sweep["steps"] = steps_sweep

        # Frames sweep
        frames_sweep = []
        for frames in [9, 25, 41]:
            try:
                t, _ = time_call(
                    pipe, prompt=args.prompt,
                    h=args.canonical_h, w=args.canonical_w,
                    frames=frames, steps=args.canonical_steps, seed=200 + frames,
                )
                frames_sweep.append({"frames": frames, "seconds": t,
                                     "ok": True})
                if rank == 0:
                    print(f"[bench]   frames={frames}: {t:.2f}s", flush=True)
            except Exception as e:
                frames_sweep.append({"frames": frames, "ok": False, "error": str(e)[:200]})
                if rank == 0:
                    print(f"[bench]   frames={frames}: FAIL: {str(e)[:100]}", flush=True)
        sweep["frames"] = frames_sweep

        # Resolution sweep
        res_sweep = []
        for w, h in [(384, 256), (512, 384), (576, 432)]:
            try:
                t, _ = time_call(
                    pipe, prompt=args.prompt,
                    h=h, w=w,
                    frames=args.canonical_frames, steps=args.canonical_steps,
                    seed=300 + w,
                )
                res_sweep.append({"w": w, "h": h, "seconds": t, "ok": True})
                if rank == 0:
                    print(f"[bench]   {w}x{h}: {t:.2f}s", flush=True)
            except Exception as e:
                res_sweep.append({"w": w, "h": h, "ok": False, "error": str(e)[:200]})
                if rank == 0:
                    print(f"[bench]   {w}x{h}: FAIL: {str(e)[:100]}", flush=True)
        sweep["resolution"] = res_sweep

        results["sweep"] = sweep

    if rank == 0:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\n[bench] WROTE {out}", flush=True)

    if dist.is_initialized():
        dist.barrier()
    return 0


if __name__ == "__main__":
    sys.exit(main())
