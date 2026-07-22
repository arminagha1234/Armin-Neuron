#!/usr/bin/env python3
"""Deterministic long-prompt parity check for the SWA windowed-prior fix.
Builds a FIXED ~18k-token prompt (seed=42) so full-span vs windowed runs use
byte-identical input; prints prompt_tokens + the 40 greedy completion tokens.
Run before (full-span) and after (windowed) the patch — completions MUST match.
"""
import json, random, urllib.request

def build_prompt(n_words, seed=42):
    rng = random.Random(seed)
    cons = "bcdfghjklmnpqrstvwxyz"; vow = "aeiou"
    pool = []
    for _ in range(4000):
        ln = rng.choice((3, 4, 5, 6, 7)); w = []
        for i in range(ln):
            w.append(rng.choice(cons) if i % 2 == 0 else rng.choice(vow))
        pool.append("".join(w))
    return " ".join(rng.choice(pool) for _ in range(n_words))

prompt = build_prompt(8000, seed=42)   # ~18k tokens -> 3 segmented chunks at seg=8192
body = json.dumps({"model": "gemma4", "prompt": prompt, "max_tokens": 40,
                   "temperature": 0, "ignore_eos": True}).encode()
req = urllib.request.Request("http://localhost:8000/v1/completions", data=body,
                            headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=600))
print("prompt_tokens=", r["usage"]["prompt_tokens"])
print("COMPLETION_REPR=", repr(r["choices"][0]["text"]))
