"""Smoke-test client for the FastAPI server.

Sends one /edit request with the same input/seed/prompt as
run_compiled_28step.sh so the response can be cosine-compared against
results/run_compiled_28step/output.png.

Usage:
    python serve/test_client.py [--host http://localhost:8000] \\
        [--input results/test_input.png] \\
        [--output results/serve_test_output.png]
"""
from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import urllib.request


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://localhost:8000")
    p.add_argument("--input", default="results/test_input.png",
                   help="Input PNG (relative to /work/path_c)")
    p.add_argument("--output", default="results/serve_test_output.png")
    p.add_argument("--prompt", default="show_from_a_different_camera_angle")
    p.add_argument("--num-steps", type=int, default=28)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--no-image", action="store_true",
                   help="Skip the image; let the worker synthesize one")
    return p.parse_args()


def health_check(host):
    url = f"{host}/health"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            print(f"[health] {resp.status} {resp.read().decode()}")
            return resp.status == 200
    except Exception as e:
        print(f"[health] FAILED: {e}")
        return False


def main():
    args = parse_args()

    if not health_check(args.host):
        print("Health check failed — is the worker up and the server running?")
        return 2

    payload = {
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "num_steps": args.num_steps,
        "seed": args.seed,
    }
    if not args.no_image:
        img_path = Path(args.input)
        if not img_path.exists():
            print(f"[error] input image not found: {img_path}")
            return 2
        payload["image_b64"] = base64.b64encode(img_path.read_bytes()).decode("ascii")

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{args.host}/edit",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    print(f"[POST /edit] prompt={args.prompt!r} steps={args.num_steps} "
          f"size={args.width}×{args.height} seed={args.seed}")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        elapsed = time.time() - t0
        status = resp.status
        body = resp.read().decode("utf-8")
    response = json.loads(body)

    print(f"[response] status={status} ok={response.get('ok')} "
          f"latency_ms={response.get('latency_ms')} "
          f"client_elapsed={elapsed:.1f}s req_id={response.get('req_id')}")

    if not response.get("ok"):
        print(f"[error] {response.get('error')}")
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(base64.b64decode(response["image_b64"]))
    print(f"[wrote] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
