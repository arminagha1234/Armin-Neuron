#!/usr/bin/env python3
"""Gemma4 31B throughput sweep — max tokens/min via concurrent clients.

Reports aggregate output tokens/s and tokens/min at increasing concurrency,
so we can find the max sustained throughput (the customer's metric).

Run inside the container with the server already up on :8000.
    python3 bench_throughput.py --model /root/models/gemma-4-31b-it \
        --concurrency 1,4,8,16,32 --input-tokens 1024 --output-tokens 256
"""
import argparse, concurrent.futures, json, random, string, time
import requests
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/root/models/gemma-4-31b-it")
ap.add_argument("--base-url", default="http://localhost:8000")
ap.add_argument("--concurrency", default="1,4,8,16,32")
ap.add_argument("--input-tokens", type=int, default=1024)
ap.add_argument("--output-tokens", type=int, default=256)
ap.add_argument("--reqs-per-level", type=int, default=4, help="requests per concurrency = N x concurrency")
ap.add_argument("--out", default="throughput_results.json")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model)
URL = f"{args.base_url}/v1/completions"
CONC = [int(x) for x in args.concurrency.split(",")]


def make_prompt(n):
    words = ["".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
             for _ in range(n * 2)]
    ids = tok(" ".join(words)).input_ids[:n]
    return tok.decode(ids, skip_special_tokens=True)


PROMPT = make_prompt(args.input_tokens)
ACTUAL_IN = len(tok(PROMPT).input_ids)


def one_request():
    t0 = time.time()
    r = requests.post(URL, json={
        "model": args.model, "prompt": PROMPT,
        "max_tokens": args.output_tokens, "temperature": 0,
    }, timeout=600)
    dt = time.time() - t0
    u = r.json().get("usage", {})
    return {"latency_s": dt, "out_tokens": u.get("completion_tokens", 0)}


print(f"Input tokens: {ACTUAL_IN}, output tokens: {args.output_tokens}")
print("Warmup...")
one_request()

results = []
for c in CONC:
    n = c * args.reqs_per_level
    print(f"\n--- concurrency={c}, {n} requests ---")
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=c) as ex:
        resps = [f.result() for f in [ex.submit(one_request) for _ in range(n)]]
    wall = time.time() - t0
    total_out = sum(r["out_tokens"] for r in resps)
    tps = total_out / wall
    tpm = tps * 60.0
    avg_lat = sum(r["latency_s"] for r in resps) / len(resps)
    print(f"  wall={wall:.1f}s  out_tokens={total_out}")
    print(f"  THROUGHPUT: {tps:.1f} tok/s = {tpm:,.0f} tok/min  (avg latency {avg_lat:.1f}s)")
    results.append({"concurrency": c, "requests": n, "wall_s": round(wall, 2),
                    "out_tokens": total_out, "tokens_per_s": round(tps, 1),
                    "tokens_per_min": round(tpm), "avg_latency_s": round(avg_lat, 2)})

best = max(results, key=lambda r: r["tokens_per_min"])
print("\n" + "=" * 56)
print(f"MAX THROUGHPUT: {best['tokens_per_min']:,} tok/min at concurrency={best['concurrency']}")
print(f"{'conc':>5} | {'tok/s':>8} | {'tok/min':>10} | {'avg lat (s)':>11}")
for r in results:
    print(f"{r['concurrency']:>5} | {r['tokens_per_s']:>8.1f} | {r['tokens_per_min']:>10,} | {r['avg_latency_s']:>11.1f}")

with open(args.out, "w") as f:
    json.dump({"model": args.model, "input_tokens": ACTUAL_IN,
               "output_tokens": args.output_tokens, "results": results,
               "max_tokens_per_min": best["tokens_per_min"],
               "max_at_concurrency": best["concurrency"]}, f, indent=2)
print(f"\nSaved {args.out}")
