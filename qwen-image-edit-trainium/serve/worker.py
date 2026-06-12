"""Persistent worker for FastAPI serving.

Runs inside torchrun (4 ranks). Loads the transformer + pipeline ONCE,
then loops on a request queue. Each request triggers a pipeline call
and writes the result image. Rank 0 communicates with the FastAPI
parent process via a Unix socket.

Protocol (JSON over newline-delimited Unix socket):
  Request:  {"req_id": str, "prompt": str, "image_b64": str|None,
             "height": int, "width": int, "num_steps": int, "seed": int}
  Response: {"req_id": str, "ok": bool, "image_b64": str|None, "error": str|None,
             "latency_ms": int}
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import socket
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image

import torch_neuronx  # noqa: F401
try:
    import torch_neuronx.distributed  # noqa: F401
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen_edit_meta_loader import load_weights_sharded  # noqa: E402
from qwen_edit_tp_plan import apply_tp_fixes, qwen_edit_tp_plan  # noqa: E402
from rope_real import install_real_rope  # noqa: E402


SOCKET_PATH = "/tmp/fal_pipeline.sock"


# Module-level per-request timer dict. Reset at start of each request,
# populated by the pipeline-level monkey-patches installed in
# build_pipeline(). Worker copies into the response after the call.
_TIMINGS_MS = {
    "encoder_ms": 0,
    "vae_encode_ms": 0,
    "vae_decode_ms": 0,
}


def setup_distributed():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if world_size > 1 and not dist.is_initialized():
        from torch.distributed.distributed_c10d import Backend
        from datetime import timedelta
        backend = "neuron" if "neuron" in Backend.backend_type_map else "xla"
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
    return rank, world_size, local_rank


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--merged-transformer", default="")
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--socket-path", default=SOCKET_PATH)
    return p.parse_args()


def build_pipeline(args, rank, world_size, local_rank, device):
    """Build the transformer + pipeline once. Same logic as run_compiled.py."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.parallel import parallelize_module
    from diffusers import QwenImageTransformer2DModel, QwenImageEditPlusPipeline

    transformer_dir = (
        args.merged_transformer
        if args.merged_transformer
        else os.path.join(args.base_model_path, "transformer")
    )
    cfg = json.loads((Path(transformer_dir) / "config.json").read_text())

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
        print(f"[worker r{rank}] transformer streamed in {time.time() - t0:.1f}s",
              flush=True)

    rope_mod = model.pos_embed
    if rope_mod.pos_freqs.is_meta or rope_mod.neg_freqs.is_meta:
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

    install_real_rope(model)
    model = torch.compile(model, backend="neuron", dynamic=False, fullgraph=False)
    model.eval()

    pipe = QwenImageEditPlusPipeline.from_pretrained(
        args.base_model_path, torch_dtype=torch.bfloat16,
    )
    del pipe.transformer
    import gc; gc.collect()
    pipe.transformer = model

    QwenImageEditPlusPipeline._execution_device = property(  # type: ignore[assignment]
        lambda self: device
    )

    cpu = torch.device("cpu")

    _orig_get = pipe._get_qwen_prompt_embeds.__func__

    def _patched_get(self, prompt, image=None, device=None, dtype=None):
        t = time.time()
        embeds, mask = _orig_get(self, prompt, image=image, device=cpu, dtype=dtype)
        embeds = embeds.to(torch.device(f"privateuseone:{local_rank}"))
        if mask is not None:
            mask = mask.to(torch.device(f"privateuseone:{local_rank}"))
        _TIMINGS_MS["encoder_ms"] += int((time.time() - t) * 1000)
        return embeds, mask

    import types
    pipe._get_qwen_prompt_embeds = types.MethodType(_patched_get, pipe)

    _orig_enc = pipe._encode_vae_image.__func__

    def _patched_enc(self, image, generator):
        t = time.time()
        if hasattr(image, "to"):
            image = image.to(cpu)
        latents = _orig_enc(self, image=image, generator=generator)
        out = latents.to(device)
        _TIMINGS_MS["vae_encode_ms"] += int((time.time() - t) * 1000)
        return out

    pipe._encode_vae_image = types.MethodType(_patched_enc, pipe)

    _orig_dec = pipe.vae.decode

    def _patched_dec(z, return_dict=True):
        t = time.time()
        if hasattr(z, "to"):
            z = z.to(cpu)
        out = _orig_dec(z, return_dict=return_dict)
        _TIMINGS_MS["vae_decode_ms"] += int((time.time() - t) * 1000)
        return out

    pipe.vae.decode = _patched_dec

    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    inner.cache_context = lambda *a, **kw: nullcontext()

    return pipe


def run_pipeline(pipe, prompt, image_b64, height, width, num_steps, seed,
                 num_images=1):
    """Run a single pipeline call. Returns (image_bytes, latency_ms, stages).

    Per-stage timings are populated by the install-time monkey-patches on
    `_get_qwen_prompt_embeds` (encoder), `_encode_vae_image` (vae_encode),
    and `vae.decode` (vae_decode). We reset the module-level dict at
    start, run the pipeline, then compute denoise = total - sum(stages).
    """
    if image_b64:
        image_bytes_in = base64.b64decode(image_b64)
        image_one = Image.open(io.BytesIO(image_bytes_in)).convert("RGB")
        image_in = [image_one] * num_images if num_images > 1 else image_one
    else:
        image_one = Image.new("RGB", (width, height), (180, 200, 150))
        image_in = [image_one] * num_images if num_images > 1 else image_one

    # Reset per-request timers
    _TIMINGS_MS["encoder_ms"] = 0
    _TIMINGS_MS["vae_encode_ms"] = 0
    _TIMINGS_MS["vae_decode_ms"] = 0

    # Wrap postprocess (cheap; install per-call so we don't pollute pipe)
    postprocess_ms = [0]
    _orig_post = pipe.image_processor.postprocess

    def _timed_post(*args, **kw):
        t = time.time()
        out = _orig_post(*args, **kw)
        postprocess_ms[0] += int((time.time() - t) * 1000)
        return out

    pipe.image_processor.postprocess = _timed_post

    torch.manual_seed(seed)
    t0 = time.time()
    try:
        result = pipe(
            image=image_in,
            prompt=prompt,
            num_inference_steps=num_steps,
            true_cfg_scale=1.0,
            height=height, width=width,
        )
    finally:
        pipe.image_processor.postprocess = _orig_post

    total_ms = int((time.time() - t0) * 1000)
    enc_ms = _TIMINGS_MS["encoder_ms"]
    venc_ms = _TIMINGS_MS["vae_encode_ms"]
    vdec_ms = _TIMINGS_MS["vae_decode_ms"]
    post_ms = postprocess_ms[0]
    denoise_ms = max(0, total_ms - enc_ms - venc_ms - vdec_ms - post_ms)

    stages = {
        "encoder_ms": enc_ms,
        "vae_encode_ms": venc_ms,
        "vae_decode_ms": vdec_ms,
        "postprocess_ms": post_ms,
        "denoise_ms": denoise_ms,
        "total_ms": total_ms,
    }

    out_buf = io.BytesIO()
    result.images[0].save(out_buf, format="PNG")
    return out_buf.getvalue(), total_ms, stages


def _broadcast_request(request, rank):
    """Fan rank 0's request out to all ranks. Uses a shared file under a
    barrier — `dist.broadcast_object_list` requires a backend that
    supports object collectives, which the Beta 2 'neuron' PG does not.
    The shared-file + barrier pattern matches the all-ranks-same-pipeline
    invariant documented in .kiro/steering/neuron-tp-on-beta2.md.

    The filename includes the req_id so concurrent workers (if any) and
    leftover files from a crashed run don't collide.
    """
    if not dist.is_initialized():
        return request

    if rank == 0:
        req_id = request["req_id"]
    else:
        req_id = None

    # Phase 1: rank 0 picks the path, broadcasts via a tiny known file.
    # We use a fixed handoff file and a barrier — single in-flight
    # request is enforced by the FastAPI lock, so no overlap.
    handoff = "/tmp/fal_request.json"
    if rank == 0:
        with open(handoff, "w") as f:
            json.dump(request, f)
    dist.barrier()
    if rank != 0:
        with open(handoff, "r") as f:
            request = json.load(f)
    dist.barrier()
    return request


def serve_loop(pipe, args, rank, world_size):
    """Rank 0 reads from socket; all ranks run pipeline in lockstep."""
    sock = None
    if rank == 0:
        try:
            os.unlink(args.socket_path)
        except FileNotFoundError:
            pass
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(args.socket_path)
        os.chmod(args.socket_path, 0o666)
        sock.listen(1)
        print(f"[worker r0] listening on {args.socket_path}", flush=True)

    while True:
        request = None
        conn = None
        if rank == 0:
            conn, _ = sock.accept()
            try:
                line = b""
                while not line.endswith(b"\n"):
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    line += chunk
                if line:
                    request = json.loads(line.decode("utf-8"))
                else:
                    conn.close()
                    continue
            except Exception as e:
                print(f"[worker r0] socket read error: {e}", flush=True)
                try:
                    conn.close()
                except Exception:
                    pass
                continue

        # Sync with other ranks to start a request
        if dist.is_initialized():
            # Signal "request inbound" with a barrier (cheap, just a sync)
            dist.barrier()

        request = _broadcast_request(request, rank)

        # All ranks now have the request — run pipeline in lockstep
        try:
            image_bytes, latency_ms, stages = run_pipeline(
                pipe,
                prompt=request["prompt"],
                image_b64=request.get("image_b64"),
                height=request.get("height", 512),
                width=request.get("width", 512),
                num_steps=request.get("num_steps", 28),
                seed=request.get("seed", 42),
                num_images=request.get("num_images", 1),
            )
            response = {
                "req_id": request["req_id"],
                "ok": True,
                "image_b64": base64.b64encode(image_bytes).decode("ascii"),
                "error": None,
                "latency_ms": latency_ms,
                "stages": stages,
            }
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            if rank == 0:
                print(f"[worker r0] pipeline error: {tb}", flush=True)
            response = {
                "req_id": request["req_id"],
                "ok": False,
                "image_b64": None,
                "error": f"{type(e).__name__}: {e}",
                "latency_ms": 0,
            }

        if rank == 0:
            try:
                conn.sendall((json.dumps(response) + "\n").encode("utf-8"))
            except Exception as e:
                print(f"[worker r0] send error: {e}", flush=True)
            try:
                conn.close()
            except Exception:
                pass

        if dist.is_initialized():
            dist.barrier()


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"privateuseone:{local_rank}")

    if rank == 0:
        print(f"[worker r0] starting; building pipeline...", flush=True)
    pipe = build_pipeline(args, rank, world_size, local_rank, device)
    if rank == 0:
        print(f"[worker r0] pipeline ready, entering serve loop", flush=True)

    serve_loop(pipe, args, rank, world_size)


if __name__ == "__main__":
    main()
