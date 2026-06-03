#!/usr/bin/env python3
"""8K TTFT bench — hand-built prompt sized via tokenizer to fit budget.

The shipped bench_ttft.py uses target * 0.45 word heuristic that overshoots
near max_model_len, causing requests to be rejected. This sizes the prompt
exactly via the tokenizer.
"""
import time, json, requests, random, string, sys
from transformers import AutoTokenizer

MODEL = "/root/models/gemma-4-31b-it"
URL = "http://localhost:8000/v1/completions"
TARGET_TOKENS = 7800   # leave headroom for the 1-token output within 8192
RUNS = 5

tok = AutoTokenizer.from_pretrained(MODEL)

# Build a prompt that tokenizes to exactly TARGET_TOKENS by trimming
words = []
while True:
    w = "".join(random.choices(string.ascii_lowercase, k=random.randint(3, 8)))
    words.append(w)
    if len(words) % 200 == 0:
        ids = tok(" ".join(words), add_special_tokens=False).input_ids
        if len(ids) >= TARGET_TOKENS + 50:
            break

ids = tok(" ".join(words), add_special_tokens=False).input_ids[:TARGET_TOKENS]
prompt = tok.decode(ids)
n_actual = len(tok(prompt, add_special_tokens=False).input_ids)
print(f"built prompt: target {TARGET_TOKENS} tokens, actual {n_actual}")

times = []
for r in range(RUNS):
    t0 = time.time()
    resp = requests.post(URL, json={
        "model": MODEL, "prompt": prompt, "max_tokens": 1,
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
        print(f"  run {r+1}: {first:.1f} ms")

if times:
    times.sort()
    median = times[len(times)//2]
    out = {
        "tag": "pathB_tp32_8k_clean",
        "actual_input_tokens": n_actual,
        "ttft_median_ms": round(median, 1),
        "ttft_min_ms": round(min(times), 1),
        "ttft_max_ms": round(max(times), 1),
        "all_ms": [round(t, 1) for t in times],
        "pass": median < 500.0,
    }
    print(json.dumps(out, indent=2))
    json.dump(out, open(sys.argv[1] if len(sys.argv)>1 else "/work/results/repo_update/run2_ttft_8k_clean.json", "w"), indent=2)
else:
    print("no successful runs")
