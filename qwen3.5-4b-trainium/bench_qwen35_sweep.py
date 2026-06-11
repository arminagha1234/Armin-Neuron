#!/usr/bin/env python3
"""TTFT + throughput sweep for Qwen3.5-4B on vllm-neuron.

Hits the local vllm endpoint with prompts of increasing length
(200 → 4k tokens) and measures:
  - TTFT (time to first token) — captured via stream=True
  - Decode throughput (tok/s) over the generated tail
  - Total latency

Run on the trn2.48xl after `vllm serve` is up at MAX_LEN=4096.
"""
import argparse
import json
import time
import urllib.request

# Word repetition produces predictable token counts. "Hello world. " is
# ~3 tokens, so to hit 200 tokens of input we repeat ~67 times. We just
# overshoot a bit and let the tokenizer handle it; we report the actual
# usage.prompt_tokens from the server.
BASE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Pack my box with five dozen liquor jugs. "
    "How vexingly quick daft zebras jump. "
    "Sphinx of black quartz, judge my vow. "
    "The five boxing wizards jump quickly. "
)

# Chars per token ~4 → multiply target_tokens by ~4 to get char count.
CHARS_PER_TOKEN = 4


def build_prompt(target_input_tokens: int) -> str:
    """Build a prompt approximately `target_input_tokens` tokens long."""
    target_chars = target_input_tokens * CHARS_PER_TOKEN
    prompt = ""
    while len(prompt) < target_chars:
        prompt += BASE_TEXT
    # Add a clear continuation cue so the model produces tokens after.
    prompt = prompt[:target_chars] + " Continue the story:"
    return prompt


def bench_one(
    endpoint: str,
    model: str,
    target_input_tokens: int,
    max_new_tokens: int = 64,
) -> dict:
    """Run one bench. Stream the response to capture TTFT cleanly."""
    prompt = build_prompt(target_input_tokens)
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": 0,
        "stream": True,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        endpoint + "/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    t_request = time.perf_counter()
    ttft = None
    n_tokens = 0
    last_chunk_text_total = ""
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            for line in resp:
                line = line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices", [])
                if not choices:
                    continue
                text = choices[0].get("text", "")
                if text:
                    if ttft is None:
                        ttft = time.perf_counter() - t_request
                    n_tokens += 1
                    last_chunk_text_total += text
    except Exception as e:
        return {"error": str(e), "input_target": target_input_tokens}
    t_total = time.perf_counter() - t_request

    decode_time = (t_total - ttft) if ttft else 0.0
    decode_tok_s = (n_tokens - 1) / decode_time if decode_time > 0 and n_tokens > 1 else 0.0

    return {
        "input_target": target_input_tokens,
        "input_chars": len(prompt),
        "max_new_tokens": max_new_tokens,
        "ttft_s": round(ttft, 3) if ttft is not None else None,
        "total_s": round(t_total, 3),
        "decode_s": round(decode_time, 3),
        "decoded_tokens": n_tokens,
        "decode_tok_s": round(decode_tok_s, 2),
        "first_64_chars": last_chunk_text_total[:64].replace("\n", "\\n"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8000")
    ap.add_argument("--model", default="/root/models/Qwen3.5-4B")
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument(
        "--input-lengths",
        default="200,500,1000,2000,4000",
        help="Comma-sep list of target input token counts",
    )
    ap.add_argument("--warmup", action="store_true", default=True)
    args = ap.parse_args()

    lengths = [int(x) for x in args.input_lengths.split(",")]

    if args.warmup:
        print("=== warmup (200 input, 8 new) ===", flush=True)
        r = bench_one(args.endpoint, args.model, 200, max_new_tokens=8)
        print(r, flush=True)

    print(f"\n=== Sweep ({len(lengths)} configs, max_new_tokens={args.max_new}) ===")
    print(f"{'input':>6}  {'TTFT(s)':>8}  {'total(s)':>9}  {'decode(s)':>10}  "
          f"{'decoded':>7}  {'tok/s':>7}  preview", flush=True)

    rows = []
    for n in lengths:
        r = bench_one(args.endpoint, args.model, n, max_new_tokens=args.max_new)
        if "error" in r:
            print(f"{n:>6}  ERROR: {r['error']}", flush=True)
            rows.append(r)
            continue
        print(
            f"{r['input_target']:>6}  {r['ttft_s']:>8}  {r['total_s']:>9}  "
            f"{r['decode_s']:>10}  {r['decoded_tokens']:>7}  "
            f"{r['decode_tok_s']:>7}  {r['first_64_chars']!r}",
            flush=True,
        )
        rows.append(r)

    print("\n=== JSON ===")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
