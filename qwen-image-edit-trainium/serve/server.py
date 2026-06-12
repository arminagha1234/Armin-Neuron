"""FastAPI front-end for the persistent worker.

The worker process (worker.py) runs under torchrun (4 ranks) and listens
on a Unix socket. This server is a thin HTTP wrap that:

  POST /edit   -> forwards JSON to the worker, returns base64 PNG
  GET  /health -> returns 200 if the worker socket is reachable

Single-process FastAPI with an asyncio.Lock around the socket — the
worker handles one pipeline call at a time, so we serialize requests
here rather than crashing the worker with concurrent reads.

Per Requirement 8 AC#2 in customers/fal/.kiro-spec/requirements.md,
queueing inside FastAPI is acceptable for Phase 3.0.

Launch:
    uvicorn server:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


SOCKET_PATH = os.environ.get("FAL_SOCKET_PATH", "/tmp/fal_pipeline.sock")
REQUEST_TIMEOUT_S = int(os.environ.get("FAL_REQUEST_TIMEOUT_S", "180"))


app = FastAPI(title="fal-qwen-image-edit-trainium", version="0.1.0")
_socket_lock = asyncio.Lock()


class EditRequest(BaseModel):
    prompt: str = Field(..., description="Edit instruction")
    image_b64: str | None = Field(
        None, description="Base64 PNG/JPG; if omitted, a synthetic blob is used"
    )
    height: int = Field(512, ge=64, le=2048)
    width: int = Field(512, ge=64, le=2048)
    num_steps: int = Field(28, ge=1, le=100)
    seed: int = Field(42)
    num_images: int = Field(1, ge=1, le=3,
                            description="How many input images (Plus pipeline supports up to 3)")


class EditResponse(BaseModel):
    ok: bool
    image_b64: str | None = None
    error: str | None = None
    latency_ms: int
    req_id: str
    stages: dict | None = None


def _send_request(payload: dict) -> dict:
    """Sync Unix-socket round-trip (called from a thread executor)."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(REQUEST_TIMEOUT_S)
    sock.connect(SOCKET_PATH)
    try:
        sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))
        # Read newline-terminated response. Image payloads can be ~1MB
        # base64-encoded, so loop until we see a newline.
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = sock.recv(65536)
            if not chunk:
                break
            buf += chunk
        if not buf:
            raise RuntimeError("worker closed socket without reply")
        return json.loads(buf.decode("utf-8"))
    finally:
        sock.close()


@app.get("/health")
async def health():
    """200 if the worker socket exists and accepts connections."""
    if not os.path.exists(SOCKET_PATH):
        return JSONResponse(
            {"status": "down", "reason": f"socket {SOCKET_PATH} missing"},
            status_code=503,
        )
    try:
        # Just probe the connect; don't send a real request.
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(2)
        s.connect(SOCKET_PATH)
        s.close()
    except Exception as e:
        return JSONResponse(
            {"status": "down", "reason": f"connect failed: {e}"},
            status_code=503,
        )
    return {"status": "ok", "socket": SOCKET_PATH}


@app.post("/edit", response_model=EditResponse)
async def edit(req: EditRequest):
    req_id = uuid.uuid4().hex[:12]
    payload = {
        "req_id": req_id,
        "prompt": req.prompt,
        "image_b64": req.image_b64,
        "height": req.height,
        "width": req.width,
        "num_steps": req.num_steps,
        "seed": req.seed,
        "num_images": req.num_images,
    }

    t0 = time.time()
    async with _socket_lock:
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, _send_request, payload
            )
        except FileNotFoundError:
            raise HTTPException(503, f"worker socket {SOCKET_PATH} not found")
        except socket.timeout:
            raise HTTPException(504, f"worker timed out after {REQUEST_TIMEOUT_S}s")
        except Exception as e:
            raise HTTPException(500, f"socket error: {type(e).__name__}: {e}")
    server_latency_ms = int((time.time() - t0) * 1000)

    if not response.get("ok"):
        return EditResponse(
            ok=False,
            error=response.get("error", "unknown worker error"),
            latency_ms=server_latency_ms,
            req_id=response.get("req_id", req_id),
        )

    return EditResponse(
        ok=True,
        image_b64=response["image_b64"],
        latency_ms=response.get("latency_ms", server_latency_ms),
        req_id=response.get("req_id", req_id),
        stages=response.get("stages"),
    )
