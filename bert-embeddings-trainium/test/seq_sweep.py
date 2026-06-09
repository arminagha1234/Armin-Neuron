#!/usr/bin/env python3
"""
Sequence length sweep for native + torch.compile BERT.

Sweeps {128, 256, 512} (BERT-base max) on MiniLM-L6 and bge-base, then
attempts longer if env var ALLOW_LONG=1 (will need a long-context model).

For each (model, max_len) measures:
  - Compile time
  - Throughput at N=128 (one canonical batch size)
  - Latency p50/p99 at batch=1

Writes /tmp/seq_sweep.json with full results.
"""
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/tmp")  # find native_bert_model.py
from native_bert_model import BertEncoder, load_from_hf

CORPUS_BASE = (
    "Encoder benchmark sentence runs embeddings at very high throughput. "
    "The quick brown fox jumps over the lazy dog repeatedly to fill the context. "
    "Sentence transformers MiniLM L6 and BAAI bge base are both BERT-style encoders. "
    "Production embedding services serve thousands of queries per second per device. "
)


def make_corpus(n, target_chars):
    """Build a list of n prompts each ~target_chars long for tokenizer to chunk to seq_len."""
    long_text = (CORPUS_BASE * 50)[:target_chars]
    return [long_text] * n


def percentile(xs, p):
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def bench_one(model_name, max_len, dtype=torch.bfloat16):
    print(f"\n=== {model_name} @ seq_len={max_len} ===", flush=True)
    import torch_neuronx  # noqa
    dev = torch.device("neuron")

    cfg = AutoConfig.from_pretrained(model_name)
    if cfg.max_position_embeddings < max_len:
        return {
            "model": model_name, "seq_len": max_len,
            "skipped": f"max_pos={cfg.max_position_embeddings} < {max_len}",
        }

    print("  loading HF + copying weights...", flush=True)
    hf = AutoModel.from_pretrained(model_name, return_dict=False).eval()
    tok = AutoTokenizer.from_pretrained(model_name)
    ours = BertEncoder(hf.config, dtype=dtype).eval()
    load_from_hf(hf, ours)
    ours = ours.to(dtype).to(dev)

    BATCH_TPUT = 128
    BATCH_LAT = 1

    def tokenize(prompts, B):
        # Make text long enough that tokenizer fills max_len after truncation.
        long = make_corpus(len(prompts), max_len * 8)  # 8 chars-ish per tok
        enc = tok(long, return_tensors="pt", padding="max_length",
                  max_length=max_len, truncation=True)
        ids = enc["input_ids"].to(dev)
        am = enc["attention_mask"].to(dev).to(dtype)
        pos = torch.arange(max_len, device=dev).unsqueeze(0).expand_as(ids)
        return ids, pos, am

    # Compile two modules: throughput batch and latency batch.
    print(f"  torch.compile batch={BATCH_TPUT}...", flush=True)
    t0 = time.time()
    m_tput = torch.compile(ours, backend="neuron", dynamic=False)
    ids, pos, am = tokenize(["x"] * BATCH_TPUT, BATCH_TPUT)
    with torch.no_grad():
        _ = m_tput(ids, pos, am)
        torch_neuronx.synchronize()
    compile_tput_s = time.time() - t0
    print(f"    done {compile_tput_s:.1f}s", flush=True)

    print(f"  torch.compile batch={BATCH_LAT}...", flush=True)
    t0 = time.time()
    m_lat = torch.compile(ours, backend="neuron", dynamic=False)
    ids1, pos1, am1 = tokenize(["x"], BATCH_LAT)
    with torch.no_grad():
        _ = m_lat(ids1, pos1, am1)
        torch_neuronx.synchronize()
    compile_lat_s = time.time() - t0
    print(f"    done {compile_lat_s:.1f}s", flush=True)

    # Throughput @ N=128 (3 iters min)
    ids_t, pos_t, am_t = tokenize(["x"] * BATCH_TPUT, BATCH_TPUT)
    best = None
    for _ in range(3):
        t = time.time()
        with torch.no_grad():
            _ = m_tput(ids_t, pos_t, am_t)
            torch_neuronx.synchronize()
        dt = time.time() - t
        best = dt if best is None or dt < best else best
    tput_seqps = BATCH_TPUT / best
    tput_ms_per = best / BATCH_TPUT * 1000
    print(f"  throughput N={BATCH_TPUT}: {tput_seqps:.1f} seq/s ({tput_ms_per:.3f} ms/prompt)", flush=True)

    # Latency batch=1, 32 calls
    ids_l, pos_l, am_l = tokenize(["x"], BATCH_LAT)
    for _ in range(8):
        with torch.no_grad():
            _ = m_lat(ids_l, pos_l, am_l)
            torch_neuronx.synchronize()
    lat = []
    for _ in range(32):
        t = time.time()
        with torch.no_grad():
            _ = m_lat(ids_l, pos_l, am_l)
            torch_neuronx.synchronize()
        lat.append((time.time() - t) * 1000)
    p50 = percentile(lat, 0.50); p99 = percentile(lat, 0.99)
    print(f"  latency batch=1: p50={p50:.2f} ms, p99={p99:.2f} ms", flush=True)

    return {
        "model": model_name, "seq_len": max_len,
        "compile_tput_s": round(compile_tput_s, 2),
        "compile_lat_s": round(compile_lat_s, 2),
        "throughput_N128_seqps": round(tput_seqps, 1),
        "throughput_N128_ms_per": round(tput_ms_per, 3),
        "latency_b1_p50_ms": round(p50, 3),
        "latency_b1_p99_ms": round(p99, 3),
    }


def main():
    configs = []
    # MiniLM-L6 and bge-base both cap at 512
    for model in [
        "sentence-transformers/all-MiniLM-L6-v2",
        "BAAI/bge-base-en-v1.5",
    ]:
        for sl in [128, 256, 512]:
            configs.append((model, sl))

    results = []
    for model, sl in configs:
        try:
            r = bench_one(model, sl)
        except Exception as e:
            import traceback; traceback.print_exc()
            r = {"model": model, "seq_len": sl, "error": str(e)}
        results.append(r)
        json.dump(results, open("/tmp/seq_sweep.json", "w"), indent=2)

    print("\n=== SUMMARY ===")
    print(f"{'model':<45}{'seq':>6}{'tput@N128':>14}{'p50':>10}{'p99':>10}")
    for r in results:
        if r.get("error") or r.get("skipped"):
            print(f"{r['model'][-44:]:<45}{r['seq_len']:>6}  {r.get('error') or r.get('skipped')}")
            continue
        print(f"{r['model'][-44:]:<45}{r['seq_len']:>6}"
              f"{r['throughput_N128_seqps']:>14}"
              f"{r['latency_b1_p50_ms']:>10}"
              f"{r['latency_b1_p99_ms']:>10}")


if __name__ == "__main__":
    main()
