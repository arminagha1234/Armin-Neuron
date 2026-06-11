#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""TTFT + throughput benchmark for Path B Qwen3.5-4B over vLLM HTTP.

Phase 8 deliverable. Mirrors `customers/Scaledown/bench_ttft.py` but
defaults to Path B's serve.sh defaults (max_model_len=4096, port=8000)
and computes:

  - TTFT (ms) at fixed seq_lens
  - Decode throughput (tok/s) for a 200-token completion
  - End-to-end latency for the customer's 20K-in / 200-out shape
  - $/M tokens given a fixed $/hr

Run after `serve.sh` is up:

    python bench_ttft.py --model /root/models/Qwen3.5-4B \\
        --seq-lengths 1024,4096,10000,16000,20000

Or just:

    ./bench_ttft.py
"""

import argparse
import json
import os
import random
import string
import time
from typing import Optional

import requests


def make_unique_prompt(target_tokens: int) -> str:
    """Build a random ASCII-words prompt aimed at ~target_tokens tokens.

    Random ASCII words tokenize to ~1.6-1.9 tokens/word with the Qwen3.5
    BPE tokenizer, so this is approximate. For exact-token benchmarks
    use the tokenizer-driven approach in customers/Scaledown/bench_ttft.py.
    """
    num_words = max(1, int(target_tokens * 0.55))
    words = []
    for _ in range(num_words):
        n = random.randint(3, 8)
        words.append("".join(random.choices(string.ascii_lowercase, k=n)))
    return " ".join(words)


def measure_ttft(
    url: str,
    model: str,
    target_tokens: int,
    num_runs: int = 3,
) -> tuple[Optional[float], list[float], list[int]]:
    """Median TTFT (ms), all run times, all actual prompt-token counts."""
    times: list[float] = []
    actuals: list[int] = []
    for run in range(num_runs):
        prompt = make_unique_prompt(target_tokens)
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "stream": False,
        }
        t0 = time.time()
        resp = requests.post(url, json=payload, timeout=180)
        elapsed = (time.time() - t0) * 1000
        if resp.status_code != 200:
            print(f"  Run {run+1}: ERROR {resp.status_code}: {resp.text[:200]}")
            continue
        data = resp.json()
        actual = int(data.get("usage", {}).get("prompt_tokens", -1))
        times.append(elapsed)
        actuals.append(actual)
        print(f"  Run {run+1}: {elapsed:.1f} ms (actual_input_tokens={actual})")
    if not times:
        return None, times, actuals
    times_sorted = sorted(times)
    return times_sorted[len(times_sorted) // 2], times, actuals


def measure_decode_throughput(
    url: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    num_runs: int = 2,
) -> tuple[Optional[float], list[dict]]:
    """Median decode throughput (tok/s) over `num_runs` requests."""
    runs: list[dict] = []
    for run in range(num_runs):
        prompt = make_unique_prompt(input_tokens)
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": output_tokens,
            "temperature": 0,
            "stream": False,
        }
        t0 = time.time()
        resp = requests.post(url, json=payload, timeout=300)
        total_ms = (time.time() - t0) * 1000
        if resp.status_code != 200:
            print(f"  Throughput run {run+1}: ERROR {resp.status_code}")
            continue
        data = resp.json()
        usage = data.get("usage", {})
        completion_tokens = int(usage.get("completion_tokens", output_tokens))
        prompt_tokens = int(usage.get("prompt_tokens", -1))
        # Throughput here is overall (prefill + decode): more useful for
        # the customer than isolated decode rate. For decode-only rate
        # subtract a measured prefill TTFT.
        tok_per_s = (completion_tokens / total_ms) * 1000 if total_ms > 0 else 0
        runs.append({
            "run": run + 1,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_ms": round(total_ms, 1),
            "tok_per_s": round(tok_per_s, 2),
        })
        print(
            f"  Throughput run {run+1}: {tok_per_s:.1f} tok/s "
            f"(in={prompt_tokens}, out={completion_tokens}, total={total_ms:.0f}ms)"
        )
    if not runs:
        return None, runs
    sorted_runs = sorted(r["tok_per_s"] for r in runs)
    return sorted_runs[len(sorted_runs) // 2], runs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/models/Qwen3.5-4B",
                        help="Model path (the same one passed to vllm serve)")
    parser.add_argument("--url", default="http://localhost:8000/v1/completions")
    parser.add_argument("--seq-lengths",
                        default="1024,4096,8192,10000,16000,20000",
                        help="Comma-separated input lengths to TTFT-bench")
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--customer-shape",
                        default="20000,200",
                        help="Customer shape <input>,<output> for end-to-end")
    parser.add_argument("--cost-per-hr", type=float, default=2.23,
                        help="Instance cost per hour (trn2.3xl=2.23, trn2.48xl=~28)")
    parser.add_argument("--output", default="/tmp/qwen35_4b_pathB_results.json")
    args = parser.parse_args()

    seq_lengths = [int(x) for x in args.seq_lengths.split(",")]
    customer_in, customer_out = (int(x) for x in args.customer_shape.split(","))

    print("=" * 70)
    print(f"Path B Qwen3.5-4B benchmark — {args.url}")
    print(f"Cost: ${args.cost_per_hr:.2f}/hr")
    print("=" * 70)

    # Phase 8.a: TTFT sweep
    ttft_results = []
    for sl in seq_lengths:
        print(f"\n--- TTFT @ target={sl} ---")
        med_ms, all_ms, actuals = measure_ttft(
            args.url, args.model, sl, args.num_runs
        )
        if med_ms is not None:
            ttft_results.append({
                "target_tokens": sl,
                "median_ttft_ms": round(med_ms, 1),
                "all_runs_ms": [round(x, 1) for x in all_ms],
                "actual_input_tokens": actuals,
            })
            print(f"  Median TTFT: {med_ms:.1f} ms")

    # Phase 8.b: customer-shape end-to-end
    print(f"\n--- Customer end-to-end: {customer_in} in / {customer_out} out ---")
    e2e_throughput, e2e_runs = measure_decode_throughput(
        args.url, args.model, customer_in, customer_out, num_runs=2
    )

    # Phase 8.c: $/M tokens computation
    cost_per_request = None
    cost_per_m_tokens = None
    if e2e_runs:
        avg_total_ms = sum(r["total_ms"] for r in e2e_runs) / len(e2e_runs)
        cost_per_request = (args.cost_per_hr / 3600) * (avg_total_ms / 1000)
        avg_in = sum(r["prompt_tokens"] for r in e2e_runs) / len(e2e_runs)
        avg_out = sum(r["completion_tokens"] for r in e2e_runs) / len(e2e_runs)
        # $/M tokens against (input + output)
        total_tokens = avg_in + avg_out
        if total_tokens > 0:
            cost_per_m_tokens = cost_per_request * 1_000_000 / total_tokens

    # Summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"{'Input':>10} | {'TTFT (ms)':>12}")
    print(f"{'-'*10}-+-{'-'*12}")
    for r in ttft_results:
        print(f"{r['target_tokens']:>10} | {r['median_ttft_ms']:>12.1f}")
    print()
    if e2e_throughput is not None:
        print(f"Customer {customer_in} in / {customer_out} out:")
        print(f"  Throughput:        {e2e_throughput:.1f} tok/s (median)")
        print(f"  Cost per request:  ${cost_per_request:.5f}")
        print(f"  Cost per M tokens: ${cost_per_m_tokens:.4f}")

    # Save
    summary = {
        "model": args.model,
        "url": args.url,
        "cost_per_hr": args.cost_per_hr,
        "customer_shape": [customer_in, customer_out],
        "ttft_results": ttft_results,
        "end_to_end": {
            "median_throughput_tok_per_s": e2e_throughput,
            "runs": e2e_runs,
            "cost_per_request_usd": cost_per_request,
            "cost_per_m_tokens_usd": cost_per_m_tokens,
        },
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
