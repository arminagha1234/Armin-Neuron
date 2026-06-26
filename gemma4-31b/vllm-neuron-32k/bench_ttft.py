"""TTFT-only sweep for the gemma4-31b vLLM-Neuron 32K serve.
Run INSIDE the container:  python3 -u /work/bench_ttft.py
Measures time-to-first-token (streaming) across input sizes. Decode is bound
at ~2.9 tok/s by the head_dim>128 decode path, so TTFT is the headline metric
for a long-context serve. 3 runs per size; reports median TTFT + prompt_tokens.
"""
import urllib.request, json, time, statistics

URL = "http://localhost:8000/v1/completions"
CHUNK = "The history of distributed computing spans many decades. "  # ~10 tok


def build_prompt(approx_tokens):
    reps = max(1, approx_tokens // 10)
    return (CHUNK * reps) + "\nIn one sentence, summarize the text. Answer:"


def ttft_once(approx_tokens):
    prompt = build_prompt(approx_tokens)
    body = json.dumps({"model": "gemma4", "prompt": prompt, "max_tokens": 2,
                       "temperature": 0, "stream": True, "ignore_eos": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    r = urllib.request.urlopen(req, timeout=900)
    ttft = None; ptok = None
    for raw in r:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"):
            continue
        d = line[5:].strip()
        if d == "[DONE]":
            break
        try:
            obj = json.loads(d)
        except Exception:
            continue
        ch = (obj.get("choices") or [{}])[0]
        if ch.get("text") and ttft is None:
            ttft = time.perf_counter() - t0
        if obj.get("usage"):
            ptok = obj["usage"].get("prompt_tokens")
    return ttft, ptok


if __name__ == "__main__":
    print("warming up..."); ttft_once(2048)
    rows = []
    for ctx in [1024, 2048, 4096, 8192, 16384, 32000]:
        samples = []
        ptok = None
        for _ in range(3):
            try:
                t, ptok = ttft_once(ctx)
                if t: samples.append(t)
            except Exception as e:
                print(ctx, "err", repr(e)[:120])
        med = round(statistics.median(samples), 3) if samples else None
        mn = round(min(samples), 3) if samples else None
        row = {"in_target": ctx, "prompt_tokens": ptok, "ttft_median_s": med, "ttft_min_s": mn}
        print(json.dumps(row)); rows.append(row)
    print("\n=== TTFT SUMMARY ===")
    print(f"{'in_target':>9} {'prompt_tok':>10} {'TTFT_med(s)':>12} {'TTFT_min(s)':>12}")
    for r in rows:
        print(f"{r['in_target']:>9} {str(r['prompt_tokens']):>10} {str(r['ttft_median_s']):>12} {str(r['ttft_min_s']):>12}")
