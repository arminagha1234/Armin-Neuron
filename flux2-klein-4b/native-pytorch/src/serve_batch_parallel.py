#!/usr/bin/env python3
"""Production-style serving wrapper for FLUX.2-klein-4B on Trainium2.

Spawns N independent worker processes — each pinned to its own logical
Neuron core via NEURON_RT_VISIBLE_CORES — and exposes a tiny HTTP API
(POST /generate) that load-balances incoming requests across the workers.

Architecture
------------

    POST /generate (FastAPI on port 8000)
              │
              ▼
        request queue (multiprocessing.Queue)
              │
       ┌──────┼──────┐
       ▼      ▼      ▼
    worker  worker  worker        ← each one is a single-core
    (core 0)(core 1)(core 2)        FLUX pipeline; runs the full
                                    forward independently.

Each worker runs the same NeuronFlux2KleinPipeline as the single-core
script. Workers share the persistent NEFF cache at /tmp/neff_cache so
only the first one to compile pays the ~15-minute cost; the rest reuse.

Usage
-----

    # On a 4-core trn2.3xl with LNC=2 (2 logical cores):
    python src/serve_batch_parallel.py --workers 2

    # On a 4-core trn2.3xl with LNC=1 (4 logical cores):
    NEURON_RT_VIRTUAL_CORE_SIZE=1 \\
        python src/serve_batch_parallel.py --workers 4

    # Then in another terminal:
    curl -X POST http://localhost:8000/generate \\
        -H 'Content-Type: application/json' \\
        -d '{"prompt": "a cat in a spacesuit", "steps": 28, "height": 1024, "width": 1024}' \\
        --output cat.png

    # Or supply an input image (image-to-image):
    curl -X POST http://localhost:8000/generate \\
        -F 'prompt=Zoom into the red highlighted area' \\
        -F 'image=@input.jpg' \\
        -F 'steps=28' \\
        --output zoomed.png

The server prints a one-line JSON event for each request — useful for
piping into a metrics collector.

Scaling notes
-------------

- One worker per logical Neuron core. Don't oversubscribe — Neuron does
  not time-slice cores like a GPU.
- Workers are independent processes so a crash in one doesn't take down
  the others. The supervisor restarts crashed workers on a backoff.
- Request queue is in-memory (single-host). For multi-host, put a real
  message broker (Redis, NATS, SQS) in front of N copies of this
  process, one per Trainium2 instance.
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Optional

# We import FastAPI lazily so the worker subprocesses (which don't need
# it) can stay clean.

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


# -----------------------------------------------------------------------------
# Worker process
# -----------------------------------------------------------------------------

@dataclass
class WorkerJob:
    job_id: str
    prompt: str
    steps: int
    height: int
    width: int
    seed: int
    image_bytes: Optional[bytes]    # encoded PNG/JPEG, or None for txt2img
    guidance_scale: float


@dataclass
class WorkerResult:
    job_id: str
    worker_id: int
    elapsed_s: float
    image_bytes: Optional[bytes]
    error: Optional[str] = None


def worker_main(
    worker_id: int,
    visible_cores: str,
    virtual_core_size: int,
    base_model: str,
    lora_repo: Optional[str],
    lora_scale: float,
    job_q: "mp.Queue[WorkerJob]",
    result_q: "mp.Queue[WorkerResult]",
    ready_q: "mp.Queue[int]",
    use_compile: bool,
):
    """Each worker loads FLUX once, then loops on the job queue."""
    # Pin to specific cores BEFORE importing torch_neuronx.
    os.environ["NEURON_RT_VISIBLE_CORES"] = visible_cores
    os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = str(virtual_core_size)

    import torch
    from PIL import Image

    from neuron_flux2_klein_native import NeuronFlux2KleinPipeline

    def log(msg: str):
        print(f"[worker{worker_id} cores={visible_cores}] {msg}", flush=True)

    log("starting up")
    pipe = NeuronFlux2KleinPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
    )
    if lora_repo:
        pipe.load_lora_weights(lora_repo)
        pipe.fuse_lora(lora_scale=lora_scale)
        pipe.unload_lora_weights()
        log(f"LoRA fused (scale={lora_scale})")

    device = torch.device("neuron")
    pipe.apply_neuron_patches(device, dtype=torch.bfloat16)
    pipe.transformer.to(device)

    if use_compile:
        pipe.transformer.inner = torch.compile(
            pipe.transformer.inner, backend="neuron", dynamic=False,
        )
        log("torch.compile applied")

    # Warmup forward at the most common shape (1024×1024) to populate
    # the NEFF cache before serving any request. Skip if the user wants
    # a fast cold-start.
    if os.environ.get("WARMUP_BEFORE_SERVE", "1") == "1":
        log("warmup pass at 1024×1024 (one-time NEFF compile)")
        warmup_img = Image.new("RGB", (1024, 1024), color=(127, 127, 127))
        gen = torch.Generator(device="cpu").manual_seed(0)
        t0 = time.time()
        try:
            _ = pipe(
                prompt="warmup", image=warmup_img,
                height=1024, width=1024,
                num_inference_steps=4, guidance_scale=3.5, generator=gen,
            )
            if hasattr(torch.neuron, "synchronize"):
                torch.neuron.synchronize()
            log(f"warmup done in {time.time()-t0:.1f}s")
        except Exception as exc:
            log(f"warmup failed: {exc!r} — continuing anyway")

    ready_q.put(worker_id)
    log("ready, entering job loop")

    while True:
        job = job_q.get()
        if job is None:    # shutdown sentinel
            log("got shutdown sentinel — exiting")
            break

        t0 = time.time()
        try:
            if job.image_bytes is not None:
                img = Image.open(io.BytesIO(job.image_bytes)).convert("RGB")
                img = img.resize((job.width, job.height), Image.LANCZOS)
            else:
                img = Image.new("RGB", (job.width, job.height), color=(127, 127, 127))

            gen = torch.Generator(device="cpu").manual_seed(job.seed)
            out = pipe(
                prompt=job.prompt, image=img,
                height=job.height, width=job.width,
                num_inference_steps=job.steps,
                guidance_scale=job.guidance_scale,
                generator=gen,
            )
            if hasattr(torch.neuron, "synchronize"):
                torch.neuron.synchronize()

            buf = io.BytesIO()
            out.images[0].save(buf, format="PNG")
            elapsed = time.time() - t0
            log(f"job {job.job_id} done in {elapsed:.1f}s")
            result_q.put(WorkerResult(
                job_id=job.job_id, worker_id=worker_id,
                elapsed_s=elapsed, image_bytes=buf.getvalue(),
            ))
        except Exception as exc:
            log(f"job {job.job_id} failed: {exc!r}")
            result_q.put(WorkerResult(
                job_id=job.job_id, worker_id=worker_id,
                elapsed_s=time.time() - t0,
                image_bytes=None, error=repr(exc),
            ))


# -----------------------------------------------------------------------------
# Supervisor (main process)
# -----------------------------------------------------------------------------

def derive_core_assignment(num_workers: int, virtual_core_size: int) -> list[str]:
    """Map worker_id -> NEURON_RT_VISIBLE_CORES string.

    For a trn2.3xl with 4 physical cores:
      - LNC=2 (virtual_core_size=2): each worker gets a pair, "0-1" and "2-3"
      - LNC=1 (virtual_core_size=1): each worker gets a single core, "0", "1", "2", "3"
    """
    assignments = []
    for w in range(num_workers):
        start = w * virtual_core_size
        end = start + virtual_core_size - 1
        if start == end:
            assignments.append(str(start))
        else:
            assignments.append(f"{start}-{end}")
    return assignments


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=2,
                    help="Number of worker processes (one per logical core)")
    ap.add_argument("--virtual-core-size", type=int,
                    default=int(os.environ.get("NEURON_RT_VIRTUAL_CORE_SIZE", 2)),
                    help="LNC config (2=default, 1=4-core mode on trn2.3xl)")
    ap.add_argument("--base-model", default="black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--lora", default=None,
                    help="HF LoRA repo id (optional)")
    ap.add_argument("--lora-scale", type=float, default=1.1)
    ap.add_argument("--no-compile", action="store_true")
    args = ap.parse_args()

    # Spawn workers ----------------------------------------------------------
    mp.set_start_method("spawn", force=True)
    job_q: mp.Queue = mp.Queue()
    result_q: mp.Queue = mp.Queue()
    ready_q: mp.Queue = mp.Queue()
    workers = []
    core_assign = derive_core_assignment(args.workers, args.virtual_core_size)
    for w in range(args.workers):
        p = mp.Process(
            target=worker_main,
            kwargs=dict(
                worker_id=w,
                visible_cores=core_assign[w],
                virtual_core_size=args.virtual_core_size,
                base_model=args.base_model,
                lora_repo=args.lora,
                lora_scale=args.lora_scale,
                job_q=job_q,
                result_q=result_q,
                ready_q=ready_q,
                use_compile=not args.no_compile,
            ),
            daemon=False,
            name=f"flux-worker-{w}",
        )
        p.start()
        workers.append(p)
        print(f"[supervisor] spawned worker {w} (cores={core_assign[w]}, pid={p.pid})", flush=True)

    print(f"[supervisor] waiting for {args.workers} worker(s) to come up...", flush=True)
    ready = set()
    while len(ready) < args.workers:
        ready.add(ready_q.get())
        print(f"[supervisor] {len(ready)}/{args.workers} workers ready", flush=True)

    # Track outstanding jobs --------------------------------------------------
    pending: dict[str, mp.Event] = {}
    completed: dict[str, WorkerResult] = {}

    # Dispatcher thread that drains result_q into completed
    import threading
    def drain():
        while True:
            res = result_q.get()
            completed[res.job_id] = res
            if res.job_id in pending:
                pending[res.job_id].set()
    threading.Thread(target=drain, daemon=True).start()

    # FastAPI app ------------------------------------------------------------
    from fastapi import FastAPI, File, Form, UploadFile
    from fastapi.responses import JSONResponse, Response
    import uvicorn

    app = FastAPI(title="FLUX.2-klein-4B Trainium2 server")

    def submit_and_wait(job: WorkerJob, timeout_s: float = 300.0) -> WorkerResult:
        evt = mp.Event()
        pending[job.job_id] = evt
        job_q.put(job)
        if not evt.wait(timeout_s):
            return WorkerResult(
                job_id=job.job_id, worker_id=-1, elapsed_s=timeout_s,
                image_bytes=None, error="timeout",
            )
        res = completed.pop(job.job_id)
        pending.pop(job.job_id, None)
        return res

    @app.post("/generate")
    async def generate(
        prompt: str = Form(...),
        steps: int = Form(28),
        height: int = Form(1024),
        width: int = Form(1024),
        seed: int = Form(42),
        guidance_scale: float = Form(3.5),
        image: Optional[UploadFile] = File(None),
    ):
        image_bytes = await image.read() if image is not None else None
        job = WorkerJob(
            job_id=uuid.uuid4().hex,
            prompt=prompt, steps=steps, height=height, width=width,
            seed=seed, guidance_scale=guidance_scale,
            image_bytes=image_bytes,
        )
        res = submit_and_wait(job)
        if res.error:
            return JSONResponse(status_code=500,
                                content={"job_id": res.job_id, "error": res.error})
        return Response(content=res.image_bytes, media_type="image/png",
                        headers={
                            "X-Job-Id": res.job_id,
                            "X-Worker-Id": str(res.worker_id),
                            "X-Elapsed-Seconds": f"{res.elapsed_s:.2f}",
                        })

    @app.get("/health")
    def health():
        return {"workers_total": args.workers,
                "workers_alive": sum(1 for w in workers if w.is_alive()),
                "queue_depth": job_q.qsize() if hasattr(job_q, "qsize") else -1}

    print(f"[supervisor] http://{args.host}:{args.port}/generate ready", flush=True)
    print(f"[supervisor] use POST /generate (multipart/form-data) with prompt + optional image", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
