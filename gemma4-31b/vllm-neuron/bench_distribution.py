#!/usr/bin/env python3
"""Distribution-aware TTFT bench for Gemma4 31B on vLLM-Neuron.

Measures TTFT at the input sizes that match a customer's payload mix and
computes a weighted-average TTFT.

Default distribution (representative customer mix):
  <=0.5K  24.8%   <=1K  53.1%   <=2K  9.5%   <=4K  12.7%

Run inside the container after the server is up:
    python3 bench_distribution.py --model /root/models/gemma-4-31b-it
"""
import argparse, json, time, random, string
from collections import OrderedDict
import requests
from transformers import AutoTokenizer

# size_target -> share of traffic
DEFAULT_DIST = OrderedDict([
    (256, 0.248),   # <=0.5K
    (950, 0.531),   # <=1K
    (1900, 0.095),  # <=2K
    (4090, 0.127),  # <=4K (4090 leaves room for the 1-token output within 4096)
])


def build_unique_prompt(tok, target_tokens):
    """Build a prompt that tokenizes to ~target_tokens with random words.

    Each call returns a different prompt -> no prefix-cache hits between runs.
    """
    words = []
    while True:
        words.append("".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8))))
        if len(words) % 200 == 0:
            ids = tok(" ".join(words), add_special_tokens=False).input_ids
            if len(ids) >= target_tokens + 50:
                break
    ids = tok(" ".join(words), add_special_tokens=False).input_ids[:target_tokens]
    return tok.decode(ids)


def measure_ttft(base_url, model, tok, target_tokens, runs):
    url = f"{base_url}/v1/completions"
    times = []
    for r in range(runs):
        prompt = build_unique_prompt(tok, target_tokens)
        n = len(tok(prompt, add_special_tokens=False).input_ids)
        t0 = time.time()
        resp = requests.post(url, json={
            "model": model, "prompt": prompt, "max_tokens": 1,
            "temperature": 0, "stream": True,
        }, stream=True, timeout=180)
        first = None
        for chunk in resp.iter_lines():
            if chunk and b"data: " in chunk and b"[DONE]" not in chunk:
                first = (time.time() - t0) * 1000
                break
        resp.close()
        if first is not None:
            times.append(first)
            print(f"    run {r+1} ({n} tok): {first:.1f} ms")
    if not times:
        return None
    times.sort()
    return times[len(times) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/gemma-4-31b-it")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--out", default="distribution_results.json")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"Distribution-aware TTFT bench  model={args.model}")

    results = OrderedDict()
    for size, share in DEFAULT_DIST.items():
        print(f"--- size={size} ({share*100:.1f}% of traffic) ---")
        ms = measure_ttft(args.base_url, args.model, tok, size, args.runs)
        if ms is None:
            print(f"  FAILED at size={size}")
        else:
            print(f"  -> median {ms:.1f} ms")
        results[size] = {"size_tokens": size, "share": share, "ttft_ms": ms}

    weighted = 0.0
    covered = 0.0
    for r in results.values():
        if r["ttft_ms"] is not None:
            weighted += r["share"] * r["ttft_ms"]
            covered += r["share"]
    print("\n" + "=" * 50)
    print(f"  Coverage: {covered*100:.1f}% of distribution")
    if covered > 0:
        print(f"  Weighted-average TTFT: {weighted/covered if covered<1 else weighted:.1f} ms")
    print("=" * 50)

    payload = {
        "model": args.model,
        "distribution": list(results.values()),
        "weighted_avg_ms": round(weighted, 1) if covered > 0.99 else None,
        "covered_fraction": covered,
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
