#!/usr/bin/env python3
"""20k input × 200 output bench for Qwen3.5-4B (customer shape).

Designed for Makora's pattern: long-context prefill (~20k tokens) and
short answer (~200 tokens). Run AFTER serving with MAX_LEN=20480
(or higher) and BUCKET=4096 for chunked prefill.

Reports:
  - TTFT (= 5×4k chunked-prefill cost since 20480/4096 = 5 chunks)
  - Decode tok/s over 200 tokens
  - Total wall time
  - $/M tokens at trn2.48xl pricing ($21.50/hr)
"""
import argparse
import json
import time
import urllib.request

# Repeat a long passage to hit ~20k tokens. "lorem ipsum" style works
# but we use a more realistic English text so the model has something
# to summarize (more representative of customer payloads than random).
LONG_TEXT = """
The history of artificial intelligence as a research field spans
several distinct eras. In the 1950s, pioneers like Alan Turing and
John McCarthy laid the theoretical foundations, with Turing's seminal
paper "Computing Machinery and Intelligence" introducing the famous
Turing test. The 1960s saw the development of early AI programs such
as ELIZA and SHRDLU, demonstrating limited but impressive natural
language understanding within constrained domains.
The 1970s and early 1980s brought the rise of expert systems, which
encoded human expertise as rule-based knowledge bases. Systems like
MYCIN for medical diagnosis and DENDRAL for chemistry showcased the
practical potential of symbolic AI. However, the limitations of pure
symbolic approaches became apparent, leading to what historians call
the "AI winter" of the late 1980s and 1990s, when funding and interest
declined sharply due to unmet expectations.
The resurgence of AI began in the 2000s with the convergence of three
factors: vastly increased computational power, the availability of
massive datasets, and improvements in machine learning algorithms,
particularly neural networks. The breakthrough moment came in 2012
when AlexNet, a deep convolutional neural network, dramatically won
the ImageNet competition. This event sparked the deep learning
revolution that continues to this day.
""".strip()

CHARS_PER_TOKEN = 4


def build_prompt(target_input_tokens: int) -> str:
    target_chars = target_input_tokens * CHARS_PER_TOKEN
    prompt = ""
    while len(prompt) < target_chars:
        prompt += LONG_TEXT + "\n\n"
    prompt = prompt[:target_chars]
    prompt += "\n\nPlease summarize the key milestones in the history of AI in a numbered list:\n"
    return prompt


def bench_one(endpoint, model, target_input, max_new):
    prompt = build_prompt(target_input)
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_new,
        "temperature": 0,
        "stream": True,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        endpoint + "/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )

    t_req = time.perf_counter()
    ttft = None
    n_tokens = 0
    text = ""
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
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
                t = choices[0].get("text", "")
                if t:
                    if ttft is None:
                        ttft = time.perf_counter() - t_req
                    n_tokens += 1
                    text += t
    except Exception as e:
        return {"error": str(e)}

    total = time.perf_counter() - t_req
    decode_t = total - ttft if ttft else 0
    decode_tps = (n_tokens - 1) / decode_t if decode_t > 0 else 0

    # $/M math at trn2.48xl on-demand $21.50/hr
    hourly = 21.50
    seconds_per_hour = 3600
    cost_per_sec = hourly / seconds_per_hour
    # Input pricing: TTFT-based — TTFT seconds spent processing N input tokens
    input_dollar_per_M = (
        cost_per_sec * ttft / target_input * 1_000_000 if ttft else 0
    )
    # Output pricing: 1 / (decode_tps tok/s) $/tok × 1M
    output_dollar_per_M = (cost_per_sec / decode_tps) * 1_000_000 if decode_tps else 0

    return {
        "input_target": target_input,
        "input_chars": len(prompt),
        "max_new": max_new,
        "ttft_s": round(ttft, 3) if ttft else None,
        "total_s": round(total, 3),
        "decode_s": round(decode_t, 3),
        "decoded_tokens": n_tokens,
        "decode_tok_s": round(decode_tps, 2),
        "input_$_per_M": round(input_dollar_per_M, 3),
        "output_$_per_M": round(output_dollar_per_M, 3),
        "first_120_chars": text[:120].replace("\n", "\\n"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:8000")
    ap.add_argument("--model", default="/root/models/Qwen3.5-4B")
    ap.add_argument("--input-tokens", type=int, default=20000)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--warmup", action="store_true", default=True)
    args = ap.parse_args()

    if args.warmup:
        print("=== warmup (200 in, 8 out) ===", flush=True)
        r = bench_one(args.endpoint, args.model, 200, 8)
        print(json.dumps(r, indent=2), flush=True)

    print(f"\n=== Config B: {args.input_tokens} input × {args.max_new} output ===")
    r = bench_one(args.endpoint, args.model, args.input_tokens, args.max_new)
    print(json.dumps(r, indent=2))


if __name__ == "__main__":
    main()
