#!/usr/bin/env python3
"""Gemma4-31B serving benchmark — concurrency sweep measuring TTFT, TPOT, and E2E.

Sends N concurrent requests (a shared context prefix + a short unique query) and records, per
concurrency level:
  - TTFT  (Time To First Token, s)      : request sent -> first output token   [prefill latency]
  - TPOT  (Time Per Output Token, ms)   : steady-state decode latency per token [(E2E-TTFT)/(out-1)]
  - E2E   (End-to-End, s)               : request sent -> last output token
  - throughput (output tokens/sec, aggregate and per-request)

Requires only the Python standard library. Talks to any OpenAI-compatible /v1/completions endpoint.
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request

CHUNK = "The history of distributed computing spans many decades. "  # ~10 tokens


def build_ctx(approx_tokens):
    return CHUNK * max(1, approx_tokens // 10)


def one_request(base_url, model, ctx, qid, gen, timeout, results, idx):
    url = base_url.rstrip("/") + "/v1/completions"
    prompt = ctx + f"\nQuestion {qid}: In one sentence, summarize the passage above. Answer:"
    body = json.dumps({
        "model": model, "prompt": prompt, "max_tokens": gen,
        "temperature": 0, "stream": True, "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }).encode()
    t0 = time.perf_counter()
    ttft = None
    n = 0
    ptok = None
    try:
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        resp = urllib.request.urlopen(req, timeout=timeout)
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            choice = (obj.get("choices") or [{}])[0]
            if choice.get("text"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                n += 1
            if obj.get("usage"):
                ptok = obj["usage"].get("prompt_tokens")
        results[idx] = {"ttft": ttft, "n": n, "end": time.perf_counter() - t0, "ptok": ptok}
    except Exception as e:
        results[idx] = {"error": repr(e)[:160]}


def run_level(base_url, model, ctx, concurrency, gen, timeout):
    results = [None] * concurrency
    threads = [threading.Thread(target=one_request,
                                args=(base_url, model, ctx, i, gen, timeout, results, i))
               for i in range(concurrency)]
    t0 = time.perf_counter()
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    wall = time.perf_counter() - t0

    ok = [r for r in results if r and "error" not in r and r.get("ttft") is not None]
    errs = [r for r in results if r and "error" in r]
    ttfts = sorted(r["ttft"] for r in ok)
    ends = sorted(r["end"] for r in ok)
    tpots = sorted((r["end"] - r["ttft"]) / (r["n"] - 1)
                   for r in ok if r["n"] and r["n"] > 1 and r["ttft"] is not None)
    tot_out = sum(r["n"] for r in ok)

    def p99(vals):
        if not vals:
            return None
        return round(vals[min(len(vals) - 1, int(len(vals) * 0.99))], 3)

    return {
        "concurrency": concurrency, "ok": len(ok), "errors": len(errs),
        "ttft_mean_s": round(statistics.mean(ttfts), 3) if ttfts else None,
        "ttft_p50_s": round(statistics.median(ttfts), 3) if ttfts else None,
        "ttft_p99_s": p99(ttfts),
        "tpot_mean_ms": round(1000 * statistics.mean(tpots), 2) if tpots else None,
        "tpot_p50_ms": round(1000 * statistics.median(tpots), 2) if tpots else None,
        "e2e_mean_s": round(statistics.mean(ends), 3) if ends else None,
        "e2e_p50_s": round(statistics.median(ends), 3) if ends else None,
        "output_tok_s_per_req": round(tot_out / wall / len(ok), 2) if ok and wall > 0 else None,
        "agg_output_tok_s": round(tot_out / wall, 1) if wall > 0 else None,
        "prompt_tokens": (ok[0].get("ptok") if ok else None),
        "gen_tokens": gen,
        "wall_s": round(wall, 3),
        "first_err": (errs[0]["error"] if errs else None),
    }


def main():
    ap = argparse.ArgumentParser(description="Gemma4-31B concurrency benchmark (TTFT / TPOT / E2E)")
    ap.add_argument("--base-url", default="http://localhost:8000", help="OpenAI-compatible server base URL")
    ap.add_argument("--model", default="gemma4", help="served model name")
    ap.add_argument("--ctx-tokens", type=int, default=4096, help="approx input/context tokens")
    ap.add_argument("--gen", type=int, default=40, help="output tokens per request")
    ap.add_argument("--levels", default="1,2,4,8,16,32", help="comma-separated concurrency levels")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    ctx = build_ctx(args.ctx_tokens)
    levels = [int(x) for x in args.levels.split(",")]

    print(f"warming shared ~{args.ctx_tokens}-token prefix (2 requests)...", flush=True)
    warm = [None]
    one_request(args.base_url, args.model, ctx, 999, 4, args.timeout, warm, 0)
    one_request(args.base_url, args.model, ctx, 998, 4, args.timeout, warm, 0)
    if warm[0] and "error" in warm[0]:
        print(f"ERROR contacting server: {warm[0]['error']}", flush=True)

    hdr = (f"{'conc':>4} {'ok':>3} {'err':>3} {'TTFT_s':>7} {'TTFT_p99':>8} "
           f"{'TPOT_ms':>8} {'E2E_s':>7} {'tok/s':>7} {'tok/s/req':>9}")
    print("\n" + hdr, flush=True)
    print("-" * len(hdr), flush=True)
    rows = []
    for c in levels:
        r = run_level(args.base_url, args.model, ctx, c, args.gen, args.timeout)
        rows.append(r)
        print(f"{r['concurrency']:>4} {r['ok']:>3} {r['errors']:>3} "
              f"{str(r['ttft_mean_s']):>7} {str(r['ttft_p99_s']):>8} "
              f"{str(r['tpot_mean_ms']):>8} {str(r['e2e_mean_s']):>7} "
              f"{str(r['agg_output_tok_s']):>7} {str(r['output_tok_s_per_req']):>9}"
              + (f"  ERR:{r['first_err']}" if r['errors'] else ""), flush=True)

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
