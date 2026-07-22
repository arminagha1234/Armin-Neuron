#!/usr/bin/env python3
"""Random-prompt serving benchmark — concurrency sweep (TTFT / TPOT / E2E).

Unlike bench.py (which sends a SHARED context prefix + short unique query — the RAG /
prefix-cache-hit pattern), this script gives EVERY request a UNIQUE random prompt so that
automatic prefix caching (APC) CANNOT serve it from cache. This measures honest cold-prefill
TTFT under load — the "random dataset" pattern (same idea as `vllm bench serve --dataset-name
random`).

Why it defeats prefix caching: APC keys on the token prefix. Each request here starts with a
fresh random token sequence, so no two requests share a cacheable prefix — every request pays
full prefill.

Same metrics + JSON schema as bench.py so results are directly comparable:
  - TTFT (s), TPOT (ms), E2E (s), aggregate + per-request output tok/s.

Stdlib only. Talks to any OpenAI-compatible /v1/completions endpoint.
"""
import argparse
import json
import os
import random
import statistics
import threading
import time
import urllib.request

# A fixed pool of pseudo-words (stable across runs, seeded), sampled randomly PER REQUEST so
# each prompt is unique. Pseudo-words (not real English) keep tokenization roughly uniform and
# avoid the model latching onto meaningful shared n-grams.
def _build_word_pool(n=4000, seed=1234):
    rng = random.Random(seed)
    cons = "bcdfghjklmnpqrstvwxyz"
    vow = "aeiou"
    pool = []
    for _ in range(n):
        ln = rng.choice((3, 4, 5, 6, 7))
        w = []
        for i in range(ln):
            w.append(rng.choice(cons) if i % 2 == 0 else rng.choice(vow))
        pool.append("".join(w))
    return pool


_POOL = _build_word_pool()
_uid_lock = threading.Lock()
_uid_counter = [0]


def _next_uid():
    with _uid_lock:
        _uid_counter[0] += 1
        return _uid_counter[0]


def make_random_prompt(n_words, range_ratio=0.0):
    """Unique random-word prompt. A fresh RNG per call (seeded by a global counter + urandom)
    guarantees a distinct prefix, so prefix caching cannot hit."""
    uid = _next_uid()
    seed = (uid << 20) ^ int.from_bytes(os.urandom(4), "little")
    rng = random.Random(seed)
    if range_ratio > 0:
        lo = int(n_words * (1 - range_ratio))
        hi = int(n_words * (1 + range_ratio))
        n_words = rng.randint(max(1, lo), max(1, hi))
    # Lead with a unique id token so even the very first token differs across requests.
    words = [f"{uid}x{seed & 0xffff:04x}"]
    words += [rng.choice(_POOL) for _ in range(max(1, n_words - 1))]
    return " ".join(words)


def one_request(base_url, model, prompt, gen, timeout, results, idx):
    url = base_url.rstrip("/") + "/v1/completions"
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


def calibrate_words(base_url, model, target_tokens, timeout):
    """Send one probe to learn tokens-per-word for THIS server's tokenizer, then size prompts."""
    guess = max(1, int(target_tokens / 2.6))
    probe = [None]
    one_request(base_url, model, make_random_prompt(guess), 4, timeout, probe, 0)
    if probe[0] and probe[0].get("ptok"):
        tpw = probe[0]["ptok"] / guess
        if tpw > 0:
            n = max(1, int(target_tokens / tpw))
            return n, probe[0]["ptok"], round(tpw, 3)
    return guess, (probe[0].get("ptok") if probe[0] else None), None


def run_level(base_url, model, n_words, range_ratio, concurrency, gen, timeout):
    # Build a UNIQUE prompt per request up front.
    prompts = [make_random_prompt(n_words, range_ratio) for _ in range(concurrency)]
    results = [None] * concurrency
    threads = [threading.Thread(target=one_request,
                                args=(base_url, model, prompts[i], gen, timeout, results, i))
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
    ptoks = [r["ptok"] for r in ok if r.get("ptok")]

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
        "prompt_tokens_mean": round(statistics.mean(ptoks)) if ptoks else None,
        "gen_tokens": gen,
        "wall_s": round(wall, 3),
        "first_err": (errs[0]["error"] if errs else None),
    }


def main():
    ap = argparse.ArgumentParser(description="Random-prompt concurrency benchmark (defeats prefix caching)")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="gemma4", help="served model name")
    ap.add_argument("--ctx-tokens", type=int, default=4096, help="approx input tokens per request")
    ap.add_argument("--gen", type=int, default=40, help="output tokens per request")
    ap.add_argument("--levels", default="1,2,4,8,16,32", help="comma-separated concurrency levels")
    ap.add_argument("--range-ratio", type=float, default=0.1,
                    help="vary each prompt length by +/- this fraction (0 = fixed). Mirrors vllm bench --random-range-ratio")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default="results_random.json")
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",")]

    print(f"calibrating random-prompt length for ~{args.ctx_tokens} tokens ...", flush=True)
    n_words, probe_ptok, tpw = calibrate_words(args.base_url, args.model, args.ctx_tokens, args.timeout)
    print(f"  -> {n_words} words/prompt (probe measured {probe_ptok} prompt_tokens, "
          f"{tpw} tokens/word). Each request gets a UNIQUE random prompt (no prefix-cache reuse).",
          flush=True)

    hdr = (f"{'conc':>4} {'ok':>3} {'err':>3} {'in_tok':>6} {'TTFT_s':>7} {'TTFT_p99':>8} "
           f"{'TPOT_ms':>8} {'E2E_s':>7} {'tok/s':>7} {'tok/s/req':>9}")
    print("\n" + hdr, flush=True)
    print("-" * len(hdr), flush=True)
    rows = []
    for c in levels:
        r = run_level(args.base_url, args.model, n_words, args.range_ratio, c, args.gen, args.timeout)
        rows.append(r)
        print(f"{r['concurrency']:>4} {r['ok']:>3} {r['errors']:>3} "
              f"{str(r['prompt_tokens_mean']):>6} "
              f"{str(r['ttft_mean_s']):>7} {str(r['ttft_p99_s']):>8} "
              f"{str(r['tpot_mean_ms']):>8} {str(r['e2e_mean_s']):>7} "
              f"{str(r['agg_output_tok_s']):>7} {str(r['output_tok_s_per_req']):>9}"
              + (f"  ERR:{r['first_err']}" if r['errors'] else ""), flush=True)

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nsaved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
