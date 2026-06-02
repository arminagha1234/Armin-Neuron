#!/usr/bin/env python3
"""Throughput benchmark: concurrent requests at varying concurrency levels.
Measures generation throughput (tok/s), TTFT, and E2E latency.
"""
import requests, time, random, string, json, concurrent.futures
from transformers import AutoTokenizer

MODEL = "/root/models/gemma-4-31b-it"
URL = "http://localhost:8000/v1/completions"
tok = AutoTokenizer.from_pretrained(MODEL)

# Build a ~3000 token prompt (fits in 4096 bucket with room for output)
words = ["".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8))) for _ in range(8000)]
base = " ".join(words)
ids = tok(base).input_ids[:3000]
PROMPT = tok.decode(ids, skip_special_tokens=True)
PROMPT_TOKENS = len(tok(PROMPT).input_ids)
OUTPUT_TOKENS = 128
print(f"Prompt tokens: {PROMPT_TOKENS}, output tokens: {OUTPUT_TOKENS}")


def send_request():
    t0 = time.time()
    resp = requests.post(URL, json={
        "model": MODEL, "prompt": PROMPT,
        "max_tokens": OUTPUT_TOKENS, "temperature": 0,
    }, timeout=120)
    elapsed = time.time() - t0
    d = resp.json()
    usage = d.get("usage", {})
    return {
        "elapsed_s": elapsed,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


# Warmup
print("Warmup (1 request)...")
w = send_request()
print(f"  warmup: {w['elapsed_s']:.1f}s, {w['completion_tokens']} tokens")

results = []
for conc in [1, 2, 4]:
    num_reqs = conc * 3
    print(f"\n--- Concurrency={conc}, {num_reqs} requests ---")
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as ex:
        futures = [ex.submit(send_request) for _ in range(num_reqs)]
        resps = [f.result() for f in futures]
    wall_time = time.time() - t0
    total_gen = sum(r["completion_tokens"] for r in resps)
    total_prompt = sum(r["prompt_tokens"] for r in resps)
    throughput = total_gen / wall_time
    avg_e2e = sum(r["elapsed_s"] for r in resps) / len(resps)
    print(f"  wall_time={wall_time:.1f}s  total_gen_tokens={total_gen}")
    print(f"  throughput={throughput:.1f} gen tok/s  avg_e2e={avg_e2e:.1f}s")
    results.append({
        "concurrency": conc,
        "num_requests": num_reqs,
        "wall_time_s": round(wall_time, 2),
        "gen_tokens": total_gen,
        "throughput_tok_s": round(throughput, 1),
        "avg_e2e_s": round(avg_e2e, 2),
    })

print("\n=== Summary ===")
for r in results:
    print(f"  conc={r['concurrency']}  throughput={r['throughput_tok_s']} tok/s  avg_e2e={r['avg_e2e_s']}s")

with open("/work/results/throughput_bench.json", "w") as f:
    json.dump({"model": MODEL, "prompt_tokens": PROMPT_TOKENS,
               "output_tokens": OUTPUT_TOKENS, "results": results}, f, indent=2)
print("\nSaved /work/results/throughput_bench.json")
