"""
Correctness + serving smoke test for the Gemma4-31B vLLM-Neuron 32K serve.
Run INSIDE the container:  python3 /tmp/test_serving.py
Exercises: health, model list, short coherent generation, and needle-in-haystack
retrieval at multiple depths across a ~32K context (stresses the windowed-SWA
attention path — a broken window/mask would fail the deep needles).
"""
import urllib.request, json, time, sys

BASE = "http://localhost:8000"
MODEL = "gemma4"
PASS, FAIL = "PASS", "FAIL"
results = []


def _get(path):
    return urllib.request.urlopen(BASE + path, timeout=30)


def _complete(prompt, max_tokens=24, temperature=0.0):
    body = json.dumps({"model": MODEL, "prompt": prompt,
                       "max_tokens": max_tokens, "temperature": temperature}).encode()
    req = urllib.request.Request(BASE + "/v1/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    o = json.loads(urllib.request.urlopen(req, timeout=600).read().decode())
    dt = time.perf_counter() - t0
    ch = o["choices"][0]
    return ch["text"], o["usage"]["prompt_tokens"], dt


def check(name, ok, detail=""):
    results.append((name, ok))
    print(f"[{PASS if ok else FAIL}] {name}  {detail}")


# 1. Health endpoint
try:
    code = _get("/health").getcode()
    check("health endpoint 200", code == 200, f"(got {code})")
except Exception as e:
    check("health endpoint 200", False, repr(e)[:80])

# 2. Model is registered
try:
    models = json.loads(_get("/v1/models").read().decode())
    ids = [m["id"] for m in models.get("data", [])]
    check("model listed", MODEL in ids, f"(models={ids})")
except Exception as e:
    check("model listed", False, repr(e)[:80])

# 3. Short coherent generation (greedy, deterministic)
try:
    txt, _, dt = _complete("The capital of France is", max_tokens=8)
    check("short factual generation", "Paris" in txt, f"-> {txt!r} ({dt:.2f}s)")
except Exception as e:
    check("short factual generation", False, repr(e)[:80])

# 4. Arithmetic / coherence sanity
try:
    txt, _, _ = _complete("What is 2 plus 2? The answer is", max_tokens=6)
    check("simple arithmetic", "4" in txt, f"-> {txt!r}")
except Exception as e:
    check("simple arithmetic", False, repr(e)[:80])

# 5. Needle-in-haystack at multiple depths across ~32K context.
#    A broken windowed-SWA gather/mask would miss needles at some depths.
chunk = "The history of distributed computing spans many decades. "  # ~10 tok
NEEDLE = "The secret passcode is BANANA-7731."
total_chunks = 3000  # ~30K tokens
for label, frac in [("needle@5%", 0.05), ("needle@50%", 0.50), ("needle@95%", 0.95)]:
    before = int(total_chunks * frac)
    after = total_chunks - before
    prompt = (chunk * before) + f" IMPORTANT: {NEEDLE} " + (chunk * after) + \
             "\nQuestion: What is the secret passcode? Answer:"
    try:
        txt, ptok, dt = _complete(prompt, max_tokens=16)
        check(f"{label} (ctx~{ptok} tok)", "BANANA-7731" in txt, f"-> {txt.strip()!r} ({dt:.1f}s)")
    except Exception as e:
        check(f"{label}", False, repr(e)[:80])

# Summary
npass = sum(1 for _, ok in results if ok)
print(f"\n==== {npass}/{len(results)} checks passed ====")
sys.exit(0 if npass == len(results) else 1)
