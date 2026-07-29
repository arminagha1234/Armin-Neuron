"""Mochi-1 preview (10B AsymmDiT) native PyTorch on Trainium2.

Runs the DiT on Neuron with tensor parallelism; keeps T5-XXL, the AsymmVAE,
and the scheduler on CPU.

    # eager bring-up, short clip, no CFG (lowest memory)
    NEURON_RT_NUM_CORES=4 TORCH_NEURONX_ENABLE_HOST_CC=1 \
    TORCH_NEURONX_ENABLE_ASYNC_NRT=1 \
    torchrun --nnodes 1 --nproc_per_node 4 \
        --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
        run_mochi_native.py --num-frames 19 --num-steps 4 --guidance-scale 1.0

    # then add CFG, more frames, and torch.compile for throughput
    ... run_mochi_native.py --num-frames 31 --num-steps 16 --compile

## Design notes

**Execution device is CPU, not Neuron.** Unlike the LTX-2 runner, we do
*not* set `_execution_device` to the Neuron device.
`FlowMatchEulerDiscreteScheduler.step` resolves its timestep index with
`(schedule_timesteps == timestep).nonzero()` -- a data-dependent op. Left on
device it runs eagerly on Neuron for no benefit. Keeping latents, the
scheduler, the text encoder, and the VAE on CPU confines the device to the
transformer, where the only shapes are static. The per-step latent
round-trip is ~2 MB, which is noise next to a 48-block forward.

**The prompt mask stays boolean across the boundary.** It has two consumers
with different expectations: `MochiAttentionPool` wants a bool mask for
SDPA, while the block processor wants an additive bias. Converting at the
boundary would break the pooler, so we pass bool through and let each side
adapt -- which is exactly why `neuron_compat` had to learn about bool masks.

**CFG doubles attention memory.** The diffusers pipeline batches the
conditional and unconditional passes together (`torch.cat([latents] * 2)`),
so batch=2 in the transformer. `--guidance-scale 1.0` disables CFG and
halves peak activation memory; use it for first bring-up, then turn CFG
back on.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Must be installed before any module builds or runs a forward.
from neuron_compat import (  # noqa: E402
    install_bmm_sdpa, print_active_fixes, set_attention_chunking,
)

install_bmm_sdpa()

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from mochi_meta_loader import load_weights_sharded  # noqa: E402
from mochi_neuron_attention import install_neuron_attn_processor  # noqa: E402
from mochi_norm_memory import DEFAULT_NORM_TILE, install_tiled_norms  # noqa: E402
from mochi_tp_plan import (  # noqa: E402
    apply_tp_fixes, mochi_tp_plan, patch_rope_cpu_precompute,
    print_plan_summary, shard_pos_frequencies, validate_world_size,
    visual_token_count,
)

MODEL_ID = "genmo/mochi-1-preview"
DEFAULT_PROMPT = (
    "Close-up of a chameleon's eye, with its scaly skin changing color. "
    "Ultra high resolution 4k."
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=848)
    p.add_argument("--num-frames", type=int, default=19,
                   help="19/31/61/85/163. Latent frames = (n-1)//6+1.")
    p.add_argument("--num-steps", type=int, default=8)
    p.add_argument("--guidance-scale", type=float, default=4.5,
                   help="1.0 disables CFG and halves activation memory.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--output", default="results/mochi_native.mp4")
    p.add_argument("--device", choices=("neuron", "cpu"), default="neuron",
                   help="cpu builds the CPU fp32 reference (slow).")
    p.add_argument("--variant", default="bf16",
                   help="bf16 (20 GB) or '' for the fp32 checkpoint (40 GB).")
    p.add_argument("--compile", action="store_true",
                   help="Wrap the transformer in torch.compile(backend='neuron').")
    p.add_argument("--q-chunk", type=int, default=0,
                   help="Attention query tile. 0=auto, <0=off. Affects NEFF cache.")
    p.add_argument("--norm-tile", type=int, default=0,
                   help=f"RMS-norm sequence tile. 0={DEFAULT_NORM_TILE} (default), "
                        f"<0 disables tiling (restores upstream full-sequence "
                        f"fp32 upcast, which OOMs above ~31 frames at TP=4).")
    p.add_argument("--rope-bf16", action="store_true",
                   help="Emit RoPE tables as bf16 instead of fp32. First knob "
                        "to try if output is structured but wrong.")
    p.add_argument("--download-only", action="store_true")
    p.add_argument("--report", default="results/run_report.json")
    p.add_argument("--dump-latents", default=None,
                   help="Save the pre-VAE denoised latents to this .pt path "
                        "(output_type='latent') and skip video decode. Used by "
                        "the CPU-vs-Neuron quality comparison to isolate the "
                        "DiT from the VAE.")
    p.add_argument("--vae-device", choices=("cpu", "neuron"), default="cpu",
                   help="Where to run the VAE decode. 'neuron' moves the "
                        "~40%%-of-wall-clock decode off CPU onto the device "
                        "(rank 0 only). Decode runs eager on Neuron.")
    return p.parse_args()


def setup_distributed():
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world_size > 1 and not dist.is_initialized():
        from datetime import timedelta
        dist.init_process_group(backend="neuron", timeout=timedelta(minutes=60))
    return rank, world_size, local_rank


def log(rank, msg):
    if rank == 0:
        print(f"[mochi] {msg}", flush=True)


def fetch_weights(rank, world_size, variant):
    """Download the checkpoint once (rank 0), then let the others proceed."""
    from huggingface_hub import snapshot_download

    # The fp32 transformer shards are 40 GB; skip them when running bf16.
    # Always skip the root-level original-format checkpoints (dit/encoder/
    # decoder.safetensors, ~42 GB): the diffusers pipeline never reads them,
    # and their absence otherwise makes an offline snapshot look "incomplete".
    ignore = [
        "dit.safetensors",
        "encoder.safetensors",
        "decoder.safetensors",
    ]
    if variant == "bf16":
        # The fp32 transformer shards are 40 GB; skip them when running bf16.
        ignore += [
            "transformer/diffusion_pytorch_model-*.safetensors",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
        ]

    if rank == 0:
        log(rank, f"downloading {MODEL_ID} (variant={variant or 'fp32'})...")
        path = snapshot_download(MODEL_ID, ignore_patterns=ignore)
        log(rank, f"weights at {path}")
    if world_size > 1:
        dist.barrier()
    if rank != 0:
        path = snapshot_download(MODEL_ID, ignore_patterns=ignore)
    return path


def build_transformer(snapshot, rank, world_size, device, dtype, args):
    """Meta-init, apply TP, stream weights, install the Neuron fixes."""
    from diffusers import MochiTransformer3DModel

    cfg_path = Path(snapshot) / "transformer" / "config.json"
    cfg = json.loads(cfg_path.read_text())

    log(rank, "building meta-init MochiTransformer3DModel (10.0 B params)")
    with torch.device("meta"):
        model = MochiTransformer3DModel.from_config(cfg)

    # Static-shape processor: removes torch.nonzero. Do this before the model
    # is ever called; it is a plain attribute swap, unaffected by TP.
    install_neuron_attn_processor(model, verbose=(rank == 0))

    # Tiled RMS norms. Mochi's modulated norms upcast the whole (B, S, 3072)
    # tensor to fp32 four times per block, which is the binding memory
    # constraint at long sequence lengths -- not attention. Must run before
    # parallelize_module so the swapped-in modules are the ones TP sees.
    if args.norm_tile >= 0:
        install_tiled_norms(
            model,
            tile=args.norm_tile if args.norm_tile > 0 else DEFAULT_NORM_TILE,
            verbose=(rank == 0),
        )

    if world_size > 1:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor.parallel import parallelize_module

        mesh = init_device_mesh(device.type, (world_size,))
        plan = mochi_tp_plan(world_size)
        log(rank, f"applying TP plan: {len(plan)} entries, world_size={world_size}")
        parallelize_module(model, mesh, plan)
        apply_tp_fixes(model, world_size=world_size, rank=rank)

    log(rank, "streaming weights (per-rank shards)...")
    t0 = time.time()
    summary = load_weights_sharded(
        model, Path(snapshot) / "transformer",
        tp_local_rank=rank, world_size=world_size,
        dtype=dtype, device=device, variant=(args.variant or None),
        strict=True, verbose=(rank == 0),
    )
    log(rank, f"weights loaded in {time.time() - t0:.1f}s "
              f"({summary['loaded']} tensors)")

    # RoPE: shard pos_frequencies on the head axis so each rank gets its own
    # frequencies, then precompute the cos/sin grid on CPU.
    shard_pos_frequencies(model, rank=rank, world_size=world_size,
                          device=device, verbose=(rank == 0))
    patch_rope_cpu_precompute(
        model,
        rope_dtype=torch.bfloat16 if args.rope_bf16 else torch.float32,
        rank=rank, verbose=(rank == 0),
    )

    model.eval()
    return model


class NeuronTransformerWrapper(torch.nn.Module):
    """Moves pipeline tensors onto the device and results back to CPU.

    Keeps the pipeline (scheduler, VAE, text encoder) entirely on CPU while
    the transformer runs on Neuron. Also forwards `config` and
    `cache_context`, both of which `MochiPipeline` reaches for directly.
    """

    def __init__(self, inner, device, cpu, dtype):
        super().__init__()
        self.inner = inner
        self._device = device
        self._cpu = cpu
        self._dtype = dtype

    @property
    def config(self):
        target = getattr(self.inner, "_orig_mod", self.inner)
        return target.config

    def cache_context(self, *args, **kwargs):
        target = getattr(self.inner, "_orig_mod", self.inner)
        return target.cache_context(*args, **kwargs)

    def _to_device(self, obj):
        if torch.is_tensor(obj):
            # Cast floating tensors to the model dtype; leave the bool prompt
            # mask alone -- MochiAttentionPool needs it boolean.
            if obj.is_floating_point():
                return obj.to(self._device, self._dtype)
            return obj.to(self._device)
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._to_device(o) for o in obj)
        if isinstance(obj, dict):
            return {k: self._to_device(v) for k, v in obj.items()}
        return obj

    def forward(self, *args, **kwargs):
        args = tuple(self._to_device(a) for a in args)
        kwargs = {k: self._to_device(v) for k, v in kwargs.items()}
        out = self.inner(*args, **kwargs)
        if isinstance(out, tuple):
            return tuple(
                o.to(self._cpu, torch.float32) if torch.is_tensor(o) else o
                for o in out
            )
        if torch.is_tensor(out):
            return out.to(self._cpu, torch.float32)
        return out


class NeuronVAEWrapper(torch.nn.Module):
    """Runs AutoencoderKLMochi.decode on Neuron, hopping tensors at the boundary.

    The Mochi pipeline calls ``vae.decode(latents).sample`` with CPU latents and
    also reaches for ``vae.config`` and ``vae.dtype``. This wrapper keeps the VAE
    weights resident on the device (bf16), moves the input latent onto the
    device for the decode, and returns the decoded frames on CPU fp32 so the
    downstream ``export_to_video`` path is unchanged.

    Decode is eager (no torch.compile): the AsymmVAE's Conv3d / replicate-pad /
    repeat_interleave upsampling all lower on nki, and eager decode already
    beats the CPU decode wall-clock while freeing the host.
    """

    def __init__(self, vae, device, cpu, dtype):
        super().__init__()
        # The AsymmVAE inflates the latent ~6x temporally and 8x8 spatially, so
        # a full-frame decode needs >1.2 GB single allocations -- which OOMs on
        # rank 0's core alongside the 9.25 GB of sharded transformer weights.
        # Tiling decodes in spatial/temporal chunks, capping the live activation.
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        self.vae = vae.to(device, dtype)
        self._device = device
        self._cpu = cpu
        self._dtype = dtype

    @property
    def config(self):
        return self.vae.config

    @property
    def dtype(self):
        return self._dtype

    def __getattr__(self, name):
        # Delegate anything not defined here (e.g. .decode variants, .encode)
        # to the wrapped VAE, after nn.Module's own attribute resolution.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.vae, name)

    def decode(self, z, *args, **kwargs):
        z = z.to(self._device, self._dtype)
        out = self.vae.decode(z, *args, **kwargs)
        # diffusers returns a DecoderOutput with a .sample tensor.
        if hasattr(out, "sample"):
            out.sample = out.sample.to(self._cpu, torch.float32)
            return out
        if torch.is_tensor(out):
            return out.to(self._cpu, torch.float32)
        return out


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()

    if args.device == "neuron":
        import torch_neuronx  # noqa: F401

    device = torch.device(args.device)
    cpu = torch.device("cpu")
    dtype = torch.bfloat16 if args.device == "neuron" else torch.float32

    if world_size > 1:
        validate_world_size(world_size)

    log(rank, f"torch={torch.__version__} device={device} dtype={dtype}")
    log(rank, f"args={vars(args)}")
    if rank == 0:
        print_active_fixes(rank)
        print_plan_summary(world_size)
        tokens = visual_token_count(args.num_frames, args.height, args.width)
        batch = 2 if args.guidance_scale > 1.0 else 1
        total = tokens + 256
        score_gb = batch * total * total * 24 / world_size * 2 / 1e9
        log(rank, f"geometry: {args.num_frames}f {args.width}x{args.height} -> "
                  f"{tokens:,} visual tokens (+256 text), CFG batch={batch}")
        log(rank, f"untiled score matrix would be {score_gb:.1f} GB/rank "
                  f"-> tiling {'ON (auto)' if args.q_chunk == 0 else args.q_chunk}")

    if args.q_chunk:
        set_attention_chunking(args.q_chunk)

    snapshot = fetch_weights(rank, world_size, args.variant)
    if args.download_only:
        log(rank, "download-only, exiting")
        return 0

    if args.device == "neuron":
        probe = torch.randn(4, 4, device=device)
        _ = probe @ probe.T
        log(rank, "device probe ok")

    transformer = build_transformer(snapshot, rank, world_size, device, dtype, args)

    if args.compile:
        log(rank, "torch.compile(backend='neuron', dynamic=False)")
        transformer = torch.compile(
            transformer, backend="neuron", dynamic=False, fullgraph=False,
        )

    # Pipeline on CPU, with our sharded transformer swapped in.
    log(rank, "loading MochiPipeline (T5 + VAE on CPU)")
    from diffusers import MochiPipeline

    t0 = time.time()
    pipe = MochiPipeline.from_pretrained(
        snapshot, transformer=None, torch_dtype=torch.float32,
    )
    log(rank, f"pipeline loaded in {time.time() - t0:.1f}s")

    pipe.transformer = NeuronTransformerWrapper(transformer, device, cpu, dtype)
    # Everything except the transformer stays on CPU. Because
    # _execution_device is CPU, no per-component device patches are needed.
    pipe.text_encoder = pipe.text_encoder.to(cpu)
    type(pipe)._execution_device = property(lambda self: cpu)

    # VAE decode is ~40% of wall clock. --vae-device neuron moves it off CPU
    # onto the device (rank 0, which is the only rank that decodes). The decode
    # runs eager; a thin wrapper hops latents to the device and frames back to
    # CPU so the surrounding pipeline stays unchanged.
    if args.vae_device == "neuron" and rank == 0:
        pipe.vae = NeuronVAEWrapper(pipe.vae, device, cpu, dtype)
        log(rank, "VAE decode -> Neuron (rank 0)")
    else:
        pipe.vae = pipe.vae.to(cpu)
    pipe.set_progress_bar_config(disable=(rank != 0))

    log(rank, f"generating: {args.num_frames}f {args.width}x{args.height}, "
              f"{args.num_steps} steps, cfg={args.guidance_scale}")
    if args.compile:
        log(rank, "(cold call includes NEFF compilation; expect 10-30 min)")

    # Only rank 0 produces the output file, so only rank 0 needs the CPU VAE
    # decode. The decode is CPU-only with no collective, and it is ~40% of wall
    # clock -- every rank running it redundantly wastes 3/4 of the machine's
    # CPU. Non-zero ranks stop at the latent (all ranks still run the full
    # collective denoising loop, which is what the transformer all-reduce
    # needs). They then wait at the closing barrier while rank 0 decodes.
    # MOCHI_FORCE_ALL_DECODE=1 restores the old all-ranks decode, for A/B.
    import os as _os
    if args.dump_latents is not None:
        # Quality-comparison mode: every rank stops at the latent, no VAE.
        output_type = "latent"
    elif _os.environ.get("MOCHI_FORCE_ALL_DECODE", "0") not in ("0", ""):
        output_type = "pil"
    else:
        output_type = "pil" if rank == 0 else "latent"
    t0 = time.time()
    try:
        with torch.no_grad():
            result = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator(device="cpu").manual_seed(args.seed),
                output_type=output_type,
            )
    except Exception:
        elapsed = time.time() - t0
        if rank == 0:
            import traceback
            print(f"[mochi] FAILED after {elapsed:.1f}s", flush=True)
            traceback.print_exc()
        if dist.is_initialized():
            dist.barrier()
        return 1

    elapsed = time.time() - t0
    log(rank, f"done in {elapsed:.1f}s ({elapsed / args.num_steps:.2f}s/step)")

    if args.dump_latents is not None:
        if rank == 0:
            # output_type='latent' returns the raw latent tensor in .frames.
            latents = result.frames if torch.is_tensor(result.frames) else result.frames[0]
            latents = latents.detach().to("cpu", torch.float32).contiguous()
            out = Path(args.dump_latents)
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "latents": latents,
                "shape": list(latents.shape),
                "device": args.device,
                "dtype": str(dtype),
                "seed": args.seed,
                "num_frames": args.num_frames,
                "num_steps": args.num_steps,
                "guidance_scale": args.guidance_scale,
            }, out)
            log(rank, f"WROTE latents {tuple(latents.shape)} -> {out} "
                      f"(mean={latents.mean():.4f} std={latents.std():.4f})")
        if dist.is_initialized():
            dist.barrier()
        return 0

    if rank == 0:
        frames = result.frames[0]
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            from diffusers.utils import export_to_video
            export_to_video(frames, str(out), fps=30)
            log(rank, f"WROTE {out} ({len(frames)} frames)")
        except Exception as exc:
            log(rank, f"mp4 export failed ({exc}); writing frame 0 as png")
            frames[0].save(out.with_suffix(".png"))

        report = {
            "elapsed_s": round(elapsed, 2),
            "s_per_step": round(elapsed / args.num_steps, 3),
            "world_size": world_size,
            "num_frames": args.num_frames,
            "resolution": [args.width, args.height],
            "num_steps": args.num_steps,
            "guidance_scale": args.guidance_scale,
            "visual_tokens": visual_token_count(
                args.num_frames, args.height, args.width
            ),
            "compiled": args.compile,
            "rope_dtype": "bf16" if args.rope_bf16 else "fp32",
            "q_chunk": args.q_chunk,
            "n_frames_out": len(frames),
        }
        rp = Path(args.report)
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(report, indent=2))
        log(rank, f"WROTE {rp}")

    if dist.is_initialized():
        dist.barrier()
    return 0


if __name__ == "__main__":
    sys.exit(main())
