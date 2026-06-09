#!/usr/bin/env python3
"""DP=2 native+compile worker. One per logical core, embeds chunk of corpus."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

WID = os.environ["WORKER_ID"]
N = int(os.environ.get("N_PROMPTS", "2048"))
OUT = os.environ["OUT_FILE"]
MODEL = os.environ.get("MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_LEN = int(os.environ.get("MAX_LEN", "128"))

CORPUS = [
    "Encoder benchmark sentence one",
    "second sentence for embedding",
    "the quick brown fox jumps over the lazy dog",
    "Trainium2 NeuronCore embedding workload",
]


def log(m): print(f"[dp-{WID}] {m}", flush=True)


def main():
    import torch
    import torch_neuronx
    from transformers import AutoModel, AutoTokenizer
    from native_bert_model import BertEncoder, load_from_hf

    log(f"loaded; CORES={os.environ.get('NEURON_VISIBLE_DEVICES')}")

    dtype = torch.bfloat16
    dev = torch.device("neuron")
    hf = AutoModel.from_pretrained(MODEL, return_dict=False).eval()
    ours = BertEncoder(hf.config, dtype=dtype).eval()
    load_from_hf(hf, ours)
    ours = ours.to(dtype).to(dev)
    ours = torch.compile(ours, backend="neuron", dynamic=False)

    tok = AutoTokenizer.from_pretrained(MODEL)
    BATCH = 512
    prompts = (CORPUS * (BATCH // len(CORPUS) + 1))[:BATCH]
    enc = tok(prompts, return_tensors="pt", padding="max_length",
              max_length=MAX_LEN, truncation=True)
    ids = enc["input_ids"].to(dev)
    am = enc["attention_mask"].to(dev).to(dtype)
    pos = torch.arange(MAX_LEN, device=dev).unsqueeze(0).expand_as(ids)

    t_boot = time.time()
    with torch.no_grad():
        _ = ours(ids, pos, am)
        torch_neuronx.synchronize()
    boot_s = time.time() - t_boot
    log(f"boot+compile {boot_s:.1f}s; running {N} prompts in batches of {BATCH}")

    t_run = time.time()
    n_done = 0
    while n_done < N:
        with torch.no_grad():
            _ = ours(ids, pos, am)
        n_done += BATCH
    torch_neuronx.synchronize()
    run_s = time.time() - t_run

    res = {
        "worker_id": WID, "model": MODEL, "n_prompts": n_done,
        "boot_s": round(boot_s, 2), "run_s": round(run_s, 4),
        "seq_per_s": round(n_done / run_s, 1),
    }
    json.dump(res, open(OUT, "w"), indent=2)
    log(f"DONE — {n_done} in {run_s:.2f}s = {n_done/run_s:.1f} seq/s")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback; traceback.print_exc()
        log(f"FAIL: {type(e).__name__}: {e}")
        sys.exit(1)
