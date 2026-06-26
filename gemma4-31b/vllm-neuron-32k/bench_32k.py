"""32K-in / 500-out TTFT + throughput benchmark for the gemma4-31b vLLM-Neuron serve.
Run INSIDE the container:  python3 /work/bench_32k.py
Measures, via streaming: TTFT (time to first token), decode throughput (tok/s),
and end-to-end total, at several input sizes with a fixed 500-token output.
"""
import urllib.request, json, time

URL = "http://localhost:8000/v1/completions"
CHUNK = "The history of distributed computing spans many decades. "  # ~10 tok


def build_prompt(approx_tokens):
    reps = max(1, approx_tokens // 10)
    return (CHUNK * reps) + "\nIn one sentence, summarize the text. Answer:"


def post(prompt, max_tokens):
    body = json.dumps({
        "model": "gemma4", "prompt": prompt,
        "max_tokens": max_tokens, "temperature": 0, "stream": True,
        "ignore_eos": True, "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=1800)


def measure(approx_tokens, gen_tokens=500):
    prompt = build_prompt(approx_tokens)
    t0 = time.perf_counter()
    r = post(prompt, gen_tokens)
    ttft = None; n_tok = 0; prompt_tokens = None; last = t0
    for raw in r:
        line = raw.decode("utf-8", "ignore").strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            obj = json.loads(data)
        except Exception:
            continue
        now = time.perf_counter()
        ch = (obj.get("choices") or [{}])[0]
        if ch.get("text"):
            if ttft is None:
                ttft = now - t0
            n_tok += 1; last = now
        if obj.get("usage"):
            prompt_tokens = obj["usage"].get("prompt_tokens")
    total = last - t0
    decode_time = (total - ttft) if (ttft and n_tok > 1) else None
    tps = ((n_tok - 1) / decode_time) if decode_time and decode_time > 0 else None
    return {"target": approx_tokens, "prompt_tokens": prompt_tokens, "gen_tokens": n_tok,
            "ttft_s": round(ttft, 3) if ttft else None,
            "decode_tps": round(tps, 2) if tps else None,
            "total_s": round(total, 2)}


if __name__ == "__main__":
    rows = []
    # warm the path once (first request includes lazy warmup), then measure
    print("warming up..."); 
    try: measure(2048, gen_tokens=8)
    except Exception as e: print("warmup err", e)
    for ctx in [1024, 4096, 16384, 32000]:
        try:
            res = measure(ctx, gen_tokens=500)
        except Exception as e:
            res = {"target": ctx, "error": repr(e)[:200]}
        print(json.dumps(res)); rows.append(res)
    print("\n=== SUMMARY (500-token output) ===")
    print(f"{'in_target':>9} {'prompt_tok':>10} {'out':>4} {'TTFT(s)':>8} {'decode_tok/s':>12} {'total(s)':>9}")
    for r in rows:
        if "error" in r:
            print(f"{r['target']:>9}  ERROR {r['error']}")
        else:
            print(f"{r['target']:>9} {str(r['prompt_tokens']):>10} {r['gen_tokens']:>4} "
                  f"{str(r['ttft_s']):>8} {str(r['decode_tps']):>12} {str(r['total_s']):>9}")
