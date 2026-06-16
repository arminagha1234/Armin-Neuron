#!/usr/bin/env python3
"""TTFT benchmark for Gemma4 31B on vLLM-Neuron vs the 500 ms latency target.

Streaming completions, unique random-token prompts (no prefix-cache inflation),
median of N runs per sequence length. JSON out. Mirrors the Llama70B repro
harness used for the Llama-70B latency study.

Run INSIDE the container after the server is up:
    python3 /work/bench_ttft.py --model google/gemma-4-31b-it \
        --tag pathA --out /work/results/gemma4_ttft_pathA.json
"""
import argparse, json, random, string, time
import requests

TARGET_MS = 500.0


def make_unique_prompt(target_tokens: int) -> str:
    # ~0.45 words/token keeps us under the target for gemma's efficient tokenizer.
    # Random words ensure no prefix-cache hit between requests.
    n_words = max(1, int(target_tokens * 0.45))
    return " ".join(
        "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
        for _ in range(n_words)
    )


def measure_ttft(base_url, model, target_tokens, runs):
    url = f"{base_url}/v1/completions"
    times = []
    actual = None
    for r in range(runs):
        prompt = make_unique_prompt(target_tokens)
        t0 = time.time()
        resp = requests.post(
            url,
            json={"model": model, "prompt": prompt, "max_tokens": 1,
                  "temperature": 0, "stream": True},
            stream=True, timeout=180,
        )
        first = None
        for chunk in resp.iter_lines():
            if chunk and b"data: " in chunk and b"[DONE]" not in chunk:
                first = (time.time() - t0) * 1000.0
                break
        resp.close()  # don't re-iterate a consumed stream
        if first is not None:
            times.append(first)
            print(f"    run {r+1}: {first:.1f} ms")
    if not times:
        return None, []
    times.sort()
    return times[len(times) // 2], times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-31b-it")
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--seq-lens", default="256,512,1024,2048,4096,8192")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--tag", default="pathA")
    ap.add_argument("--out", default="/work/results/gemma4_ttft.json")
    args = ap.parse_args()

    seq_lens = [int(x) for x in args.seq_lens.split(",")]
    print(f"Gemma4 TTFT bench [{args.tag}]  model={args.model}  seq_lens={seq_lens}")
    results = []
    for sl in seq_lens:
        print(f"--- seq_len={sl} ---")
        med, allt = measure_ttft(args.base_url, args.model, sl, args.runs)
        if med is None:
            print("    FAILED")
            results.append({"seq_len": sl, "ttft_ms": None, "pass": False})
            continue
        passed = med < TARGET_MS
        print(f"  -> median {med:.1f} ms  {'PASS' if passed else 'FAIL'} (target {TARGET_MS:.0f})")
        results.append({"seq_len": sl, "ttft_ms": round(med, 1),
                        "all_ms": [round(t, 1) for t in allt],
                        "pass": passed})

    print("\n" + "=" * 48)
    print(f"Gemma4 TTFT [{args.tag}] vs {TARGET_MS:.0f} ms")
    print(f"{'seq_len':>8} | {'TTFT ms':>9} | result")
    print("-" * 32)
    for r in results:
        ms = "n/a" if r["ttft_ms"] is None else f"{r['ttft_ms']:.1f}"
        res = "PASS" if r.get("pass") else "FAIL"
        print(f"{r['seq_len']:>8} | {ms:>9} | {res}")

    payload = {"tag": args.tag, "model": args.model, "target_ms": TARGET_MS,
               "results": results}
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
