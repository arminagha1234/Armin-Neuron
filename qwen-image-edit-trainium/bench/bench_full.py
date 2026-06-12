"""Customer-grade benchmark suite for the Phase 3 HTTP server.

Captures the metrics that matter for an image-edit serving customer:

  • TTFI — time-to-first-image (process spawn → weights → first response)
  • Cold (first inference) vs warm (steady-state) latency
  • Per-stage breakdown (encoder / vae_encode / denoise / vae_decode /
    postprocess) — emitted by the worker
  • Steady-state p50 / p95 / p99
  • Resolution sweep (512 / 768 / 1024)
  • Step-count sweep (4 / 8 / 16 / 28)
  • Multi-image input sweep (1 / 2 / 3 — Plus pipeline ceiling)
  • Per-rank / per-core peak memory (read from `neuron-ls`)
  • $ / image at trn2.48xlarge on-demand pricing

Concurrency note: a single worker is single-in-flight (FastAPI lock).
Throughput results below are wall-clock under serial dispatch. The
"box throughput" extrapolation assumes 4 worker replicas can run
concurrently on the 16 cores of a trn2.48xl (TP=4 each); the
data-parallel multi-worker bench is `bench_dp_box.sh` (separate
script, requires 4 socket paths and 4 server ports).

Output: writes JSON to --out (default
`results/bench/<timestamp>/results.json`).
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path


# ── Pricing — change here, re-render report. ──────────────────────────
# trn2.48xlarge — negotiated rate (us-east-2, June 2026, per Armin)
TRN2_48XL_PRICE_USD_HR = 35.7608
# p5.48xlarge — negotiated rate (us-east-2, June 2026, per Armin)
# (exposed for parallel comparison reports — not consumed in the math
# unless --include-gpu-comparison is set)
P5_48XL_PRICE_USD_HR = 31.464

# Single trn2.48xl == 16 Neuron cores. With TP=4 per worker we can
# theoretically run 4 worker replicas in parallel for data-parallel
# throughput. Bench reports both single-worker and 4-worker math.
WORKERS_PER_BOX_DEFAULT = 4


def http_post_json(host, path, body, timeout):
    req = urllib.request.Request(
        f"{host}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def http_get(host, path, timeout=5):
    with urllib.request.urlopen(f"{host}{path}", timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8")


def wait_for_health(host, max_wait_s=600, interval_s=2):
    """Poll /health until 200 or timeout. Returns elapsed seconds."""
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        try:
            status, body = http_get(host, "/health")
            if status == 200:
                return time.time() - t0
        except Exception:
            pass
        time.sleep(interval_s)
    raise RuntimeError(f"server not healthy after {max_wait_s}s")


def percentile(values, p):
    """p in [0, 100]. Linear interpolation."""
    if not values:
        return 0.0
    s = sorted(values)
    if p <= 0:
        return s[0]
    if p >= 100:
        return s[-1]
    k = (len(s) - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def stats(values):
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "min": min(values),
        "max": max(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def stage_means(samples):
    """samples = list of stages dicts. Returns mean per stage in ms."""
    if not samples:
        return {}
    keys = samples[0].keys()
    return {k: statistics.fmean([s.get(k, 0) for s in samples]) for k in keys}


def read_neuron_memory():
    """Return per-core memory from neuron-ls. Tries multiple invocations
    since the JSON flag varies across SDK versions."""
    candidates = [
        ["neuron-ls", "-j"],
        ["neuron-ls", "--json"],
        ["neuron-ls", "--json-output"],
    ]
    last_err = None
    for cmd in candidates:
        try:
            out = subprocess.check_output(
                cmd, stderr=subprocess.STDOUT, timeout=5,
            )
            data = json.loads(out.decode())
            cores = []
            # Different SDK versions use different field shapes; flatten
            # whichever one we get.
            entries = data if isinstance(data, list) else data.get("neuron_devices", [])
            for dev in entries:
                dev_id = dev.get("nd_id") or dev.get("neuron_device") or dev.get("id")
                nc_list = (dev.get("nc_status") or dev.get("nc")
                           or dev.get("neuron_cores") or [])
                for nc in nc_list:
                    cores.append({
                        "device": dev_id,
                        "core": nc.get("nc_id") or nc.get("id"),
                        "mem_used_mb": nc.get("mem_used_mb") or nc.get("memory_used_mb"),
                        "mem_total_mb": nc.get("mem_total_mb") or nc.get("memory_total_mb"),
                    })
            if cores:
                return cores
            # Fall through if the structure was unexpected
            last_err = f"empty cores from {' '.join(cmd)}: {out[:200]!r}"
        except subprocess.CalledProcessError as e:
            last_err = f"{' '.join(cmd)}: rc={e.returncode} {e.output[:120]!r}"
        except FileNotFoundError as e:
            last_err = str(e); break
        except Exception as e:
            last_err = f"{' '.join(cmd)}: {type(e).__name__}: {e}"
    # Fallback: parse text output of `neuron-ls`
    try:
        out = subprocess.check_output(
            ["neuron-ls"], stderr=subprocess.STDOUT, timeout=5,
        ).decode()
        # Just return the raw stdout — at least the bench captures something.
        return {"raw_text": out, "note": last_err}
    except Exception as e2:
        return {"error": f"{last_err}; fallback {type(e2).__name__}: {e2}"}


def load_test_image(path, height, width):
    p = Path(path)
    if p.exists():
        return base64.b64encode(p.read_bytes()).decode("ascii")
    # Synthesize one if no file given
    from io import BytesIO
    from PIL import Image
    img = Image.new("RGB", (width, height), (180, 200, 150))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def call_edit(host, image_b64, prompt, height, width, num_steps, seed,
              num_images, timeout):
    body = {
        "prompt": prompt,
        "image_b64": image_b64,
        "height": height, "width": width,
        "num_steps": num_steps, "seed": seed,
        "num_images": num_images,
    }
    t0 = time.time()
    resp = http_post_json(host, "/edit", body, timeout=timeout)
    client_latency_ms = int((time.time() - t0) * 1000)
    return resp, client_latency_ms


def steady_state_run(host, n, base_args, timeout, *, label=""):
    """Run n requests, drop the first (cold), return stats on remaining.
    Per-request errors are recorded but do not abort the sweep."""
    samples = []
    stages_samples = []
    raw = []
    for i in range(n):
        try:
            resp, client_ms = call_edit(host, **base_args, timeout=timeout)
            ok = resp.get("ok", False)
            worker_ms = resp.get("latency_ms", -1)
            st = resp.get("stages") or {}
            print(f"  [{label}] run {i+1}/{n}: ok={ok} worker={worker_ms}ms "
                  f"client={client_ms}ms denoise={st.get('denoise_ms', '?')}ms",
                  flush=True)
            if not ok:
                raw.append({"i": i, "ok": False, "error": resp.get("error")})
                continue
            raw.append({
                "i": i, "ok": True,
                "worker_ms": worker_ms, "client_ms": client_ms,
                "stages": st,
            })
            samples.append(worker_ms)
            if st:
                stages_samples.append(st)
        except Exception as e:
            print(f"  [{label}] run {i+1}/{n}: FAILED {type(e).__name__}: {e}",
                  flush=True)
            raw.append({"i": i, "ok": False,
                        "error": f"{type(e).__name__}: {e}"})
            # keep going through the rest of the sweep
    if len(samples) <= 1:
        cold = samples[0] if samples else None
        warm_stats = stats([])
        warm_stages = {}
    else:
        cold = samples[0]
        warm_stats = stats(samples[1:])
        warm_stages = stage_means(stages_samples[1:]) if len(stages_samples) > 1 else {}
    return {
        "label": label,
        "config": {k: v for k, v in base_args.items() if k != "image_b64"},
        "cold_ms": cold,
        "warm": warm_stats,
        "warm_stages_ms": warm_stages,
        "raw": raw,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="http://localhost:8000")
    p.add_argument("--input", default="results/test_input.png")
    p.add_argument("--prompt", default="show_from_a_different_camera_angle")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None,
                   help="Output JSON path (default results/bench/<ts>/results.json)")
    p.add_argument("--ttfi-bench", action="store_true",
                   help="Measure TTFI from worker spawn. Requires --launch-worker-cmd.")
    p.add_argument("--launch-worker-cmd", default=None,
                   help="Shell command to launch worker (only used with --ttfi-bench)")
    p.add_argument("--max-wait-s", type=int, default=900,
                   help="Health-poll timeout for TTFI bench")
    p.add_argument("--n-warm", type=int, default=10,
                   help="N requests for steady-state percentile bench at the canonical config (28 step / 512 / single-image)")
    p.add_argument("--skip-resolution", action="store_true")
    p.add_argument("--skip-steps", action="store_true")
    p.add_argument("--skip-multi-image", action="store_true")
    p.add_argument("--workers-per-box", type=int, default=WORKERS_PER_BOX_DEFAULT)
    p.add_argument("--request-timeout", type=int, default=900,
                   help="Per-request HTTP timeout (s)")
    p.add_argument("--skip-canonical", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out_path = (Path(args.out) if args.out
                else Path(f"results/bench/{datetime.now().strftime('%Y%m%d_%H%M%S')}/results.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "meta": {
            "started": datetime.now().isoformat(),
            "host": args.host,
            "machine": "trn2.48xlarge",
            "price_usd_hr": TRN2_48XL_PRICE_USD_HR,
            "workers_per_box": args.workers_per_box,
            "tool": "customers/fal/path_c/serve/bench_full.py",
        },
        "ttfi": None,
        "canonical": None,
        "resolution_sweep": None,
        "step_sweep": None,
        "multi_image_sweep": None,
        "memory_pre_run": read_neuron_memory(),
        "memory_post_run": None,
    }

    # ── Optional: TTFI from full cold start ──────────────────────────
    if args.ttfi_bench:
        if not args.launch_worker_cmd:
            print("--ttfi-bench requires --launch-worker-cmd", file=sys.stderr)
            return 2
        print(f"[ttfi] launching worker: {args.launch_worker_cmd}", flush=True)
        t_spawn = time.time()
        proc = subprocess.Popen(
            args.launch_worker_cmd, shell=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        # Wait for /health to return 200 (server already up; worker socket opens last)
        try:
            health_elapsed = wait_for_health(args.host, max_wait_s=args.max_wait_s)
        except Exception as e:
            print(f"[ttfi] health timeout: {e}", flush=True)
            results["ttfi"] = {"error": str(e)}
            return 2
        t_health = time.time() - t_spawn
        # First image
        b64 = load_test_image(args.input, 512, 512)
        first_resp, first_client = call_edit(
            args.host, image_b64=b64, prompt=args.prompt,
            height=512, width=512, num_steps=28, seed=args.seed, num_images=1,
            timeout=args.request_timeout,
        )
        t_total = time.time() - t_spawn
        results["ttfi"] = {
            "spawn_to_health_s": round(health_elapsed, 1),
            "spawn_to_total_health_s": round(t_health, 1),
            "first_image_worker_ms": first_resp.get("latency_ms"),
            "first_image_client_ms": first_client,
            "spawn_to_first_image_s": round(t_total, 1),
            "ok": first_resp.get("ok"),
        }
        print(f"[ttfi] spawn→health {health_elapsed:.1f}s → "
              f"spawn→first_image {t_total:.1f}s "
              f"(worker {first_resp.get('latency_ms')}ms)", flush=True)

    # Health gate before any sweep
    print(f"[gate] checking health at {args.host}/health", flush=True)
    wait_for_health(args.host, max_wait_s=10)

    def checkpoint():
        out_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"[bench] checkpoint -> {out_path}", flush=True)

    # ── Canonical: 28 steps, 512×512, 1 image, n_warm samples ────────
    if not args.skip_canonical:
        print(f"\n[canonical] 28 steps / 512×512 / 1 image — {args.n_warm} samples", flush=True)
        b64_512 = load_test_image(args.input, 512, 512)
        canonical = steady_state_run(
            args.host, args.n_warm,
            {"image_b64": b64_512, "prompt": args.prompt,
             "height": 512, "width": 512, "num_steps": 28,
             "seed": args.seed, "num_images": 1},
            timeout=args.request_timeout,
            label="canonical",
        )
        results["canonical"] = canonical
        checkpoint()

    # ── Resolution sweep (28 steps, 1 image, n=3 each) ───────────────
    if not args.skip_resolution:
        print(f"\n[resolution sweep] 512 / 768 / 1024 — 3 samples each "
              f"(1 cold + 2 warm to take a min latency)", flush=True)
        sweep = []
        for h, w in [(512, 512), (768, 768), (1024, 1024)]:
            b64 = load_test_image(args.input, h, w)
            r = steady_state_run(
                args.host, 3,
                {"image_b64": b64, "prompt": args.prompt,
                 "height": h, "width": w, "num_steps": 28,
                 "seed": args.seed, "num_images": 1},
                timeout=args.request_timeout,
                label=f"{h}x{w}",
            )
            sweep.append({"height": h, "width": w, **r})
        results["resolution_sweep"] = sweep
        checkpoint()

    # ── Step-count sweep (512×512, 1 image, n=3 each) ────────────────
    if not args.skip_steps:
        print(f"\n[step sweep] 4 / 8 / 16 / 28 steps — 3 samples each", flush=True)
        sweep = []
        b64 = load_test_image(args.input, 512, 512)
        for steps in [4, 8, 16, 28]:
            r = steady_state_run(
                args.host, 3,
                {"image_b64": b64, "prompt": args.prompt,
                 "height": 512, "width": 512, "num_steps": steps,
                 "seed": args.seed, "num_images": 1},
                timeout=args.request_timeout,
                label=f"{steps}step",
            )
            sweep.append({"num_steps": steps, **r})
        results["step_sweep"] = sweep
        checkpoint()

    # ── Multi-image input sweep (1 / 2 / 3 images, 28 steps) ─────────
    if not args.skip_multi_image:
        print(f"\n[multi-image sweep] 1 / 2 / 3 input images — 3 samples each",
              flush=True)
        sweep = []
        b64 = load_test_image(args.input, 512, 512)
        for n_img in [1, 2, 3]:
            r = steady_state_run(
                args.host, 3,
                {"image_b64": b64, "prompt": args.prompt,
                 "height": 512, "width": 512, "num_steps": 28,
                 "seed": args.seed, "num_images": n_img},
                timeout=args.request_timeout,
                label=f"{n_img}img",
            )
            sweep.append({"num_images": n_img, **r})
        results["multi_image_sweep"] = sweep
        checkpoint()

    results["memory_post_run"] = read_neuron_memory()
    results["meta"]["ended"] = datetime.now().isoformat()

    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[bench] WROTE {out_path}", flush=True)
    print(f"[bench] tip: render with serve/bench_report.py {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
