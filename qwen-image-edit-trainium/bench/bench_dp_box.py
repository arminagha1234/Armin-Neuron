"""Concurrent multi-worker bench — measures full-box throughput.

The single-worker bench measures latency per image but a single worker
serializes (FastAPI lock + worker single-in-flight). To answer "what
does a customer get from a whole trn2.48xl box?", we need to drive
multiple workers concurrently.

Approach: spawn N concurrent client threads, each issuing requests in
a tight loop against the SAME `/edit` endpoint. The single worker
serializes them, so this isn't true throughput — it's the queue
latency profile under load. For real DP throughput we'd need 4 worker
replicas behind a load balancer; that's launched by `bench_dp_box.sh`
which starts 4 workers on disjoint core sets and 4 servers on disjoint
ports, then this script can target each in round-robin.

Usage:
    # Single worker, N concurrent clients (queue latency curve):
    python serve/bench_dp_box.py \\
        --hosts http://localhost:8000 \\
        --concurrency 4 --duration 600

    # 4-worker DP setup (after running bench_dp_box.sh to launch them):
    python serve/bench_dp_box.py \\
        --hosts http://localhost:8000,http://localhost:8001,http://localhost:8002,http://localhost:8003 \\
        --concurrency 4 --duration 600
"""
from __future__ import annotations

import argparse
import base64
import itertools
import json
import statistics
import sys
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hosts", required=True,
                   help="Comma-separated server URLs (1 to 4)")
    p.add_argument("--input", default="results/test_input.png")
    p.add_argument("--prompt", default="show_from_a_different_camera_angle")
    p.add_argument("--num-steps", type=int, default=28)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--concurrency", type=int, default=4,
                   help="Number of concurrent client threads")
    p.add_argument("--duration", type=int, default=300,
                   help="Bench wall-clock duration (seconds)")
    p.add_argument("--out", default=None,
                   help="Output JSON path")
    p.add_argument("--request-timeout", type=int, default=900)
    return p.parse_args()


def load_b64(path):
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    if p <= 0:
        return s[0]
    if p >= 100:
        return s[-1]
    import math
    k = (len(s) - 1) * p / 100
    f = math.floor(k); c = math.ceil(k)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def main():
    args = parse_args()
    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    image_b64 = load_b64(args.input)
    body_template = {
        "prompt": args.prompt,
        "image_b64": image_b64,
        "height": args.height, "width": args.width,
        "num_steps": args.num_steps, "seed": 42, "num_images": 1,
    }

    print(f"[bench-dp] hosts={hosts} concurrency={args.concurrency} "
          f"duration={args.duration}s", flush=True)

    results_lock = threading.Lock()
    results = []
    stop = threading.Event()
    host_cycle = itertools.cycle(hosts)
    host_lock = threading.Lock()

    def next_host():
        with host_lock:
            return next(host_cycle)

    def worker(thread_id):
        while not stop.is_set():
            host = next_host()
            seed = (thread_id * 1000 + len(results)) & 0xFFFF
            body = dict(body_template, seed=seed)
            req = urllib.request.Request(
                f"{host}/edit",
                data=json.dumps(body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=args.request_timeout) as resp:
                    body_resp = json.loads(resp.read().decode("utf-8"))
                client_ms = int((time.time() - t0) * 1000)
                with results_lock:
                    results.append({
                        "thread": thread_id, "host": host,
                        "client_ms": client_ms,
                        "worker_ms": body_resp.get("latency_ms"),
                        "ok": body_resp.get("ok", False),
                    })
            except Exception as e:
                with results_lock:
                    results.append({
                        "thread": thread_id, "host": host,
                        "client_ms": int((time.time() - t0) * 1000),
                        "ok": False, "error": str(e),
                    })

    t_start = time.time()
    threads = [threading.Thread(target=worker, args=(i,), daemon=True)
               for i in range(args.concurrency)]
    for t in threads:
        t.start()

    # Status loop
    while time.time() - t_start < args.duration:
        time.sleep(10)
        with results_lock:
            n = len(results)
            ok = sum(1 for r in results if r.get("ok"))
        print(f"[bench-dp] +{int(time.time() - t_start)}s: "
              f"completed={n} ok={ok}", flush=True)

    stop.set()
    for t in threads:
        t.join(timeout=args.request_timeout + 30)

    elapsed_s = time.time() - t_start
    ok_results = [r for r in results if r.get("ok")]
    client_ms = [r["client_ms"] for r in ok_results]
    worker_ms = [r["worker_ms"] for r in ok_results if r.get("worker_ms")]

    summary = {
        "meta": {
            "started": datetime.fromtimestamp(t_start).isoformat(),
            "elapsed_s": round(elapsed_s, 1),
            "hosts": hosts,
            "concurrency": args.concurrency,
            "config": body_template | {"image_b64": "<elided>"},
        },
        "totals": {
            "n_completed": len(results),
            "n_ok": len(ok_results),
            "n_failed": len(results) - len(ok_results),
            "throughput_imgs_per_min": (
                len(ok_results) / elapsed_s * 60) if elapsed_s else 0,
            "throughput_imgs_per_hr": (
                len(ok_results) / elapsed_s * 3600) if elapsed_s else 0,
        },
        "client_latency_ms": {
            "n": len(client_ms),
            "mean": statistics.fmean(client_ms) if client_ms else 0,
            "p50": percentile(client_ms, 50),
            "p95": percentile(client_ms, 95),
            "p99": percentile(client_ms, 99),
            "min": min(client_ms) if client_ms else 0,
            "max": max(client_ms) if client_ms else 0,
        },
        "worker_latency_ms": {
            "n": len(worker_ms),
            "mean": statistics.fmean(worker_ms) if worker_ms else 0,
            "p50": percentile(worker_ms, 50),
            "p95": percentile(worker_ms, 95),
            "p99": percentile(worker_ms, 99),
        },
        "results": results,
    }

    out = (Path(args.out) if args.out
           else Path(f"results/bench/dp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[bench-dp] WROTE {out}")
    print(f"[bench-dp] {summary['totals']['n_ok']} ok in {elapsed_s:.0f}s "
          f"= {summary['totals']['throughput_imgs_per_min']:.2f} img/min "
          f"= {summary['totals']['throughput_imgs_per_hr']:.1f} img/hr")
    print(f"[bench-dp] client p50={summary['client_latency_ms']['p50']:.0f}ms "
          f"p99={summary['client_latency_ms']['p99']:.0f}ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
