#!/usr/bin/env python3
"""
Correctness check: native + torch.compile embeddings vs CPU HF reference.
Must hit cos >= 0.99 to claim numerical correctness on the new path.
"""
import os
import sys

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from native_bert_model import BertEncoder, load_from_hf

MODEL = os.environ.get("MODEL", "sentence-transformers/all-MiniLM-L6-v2")
MAX_LEN = int(os.environ.get("MAX_LEN", "128"))
USE_COMPILE = os.environ.get("USE_COMPILE", "1") == "1"
PROMPTS = [
    "Encoder benchmark sentence one",
    "second sentence for embedding",
    "the quick brown fox",
]


def cos(a, b):
    a = np.asarray(a, "float64").reshape(-1)
    b = np.asarray(b, "float64").reshape(-1)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    import torch_neuronx  # noqa: F401

    tok = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModel.from_pretrained(MODEL, return_dict=False).eval()

    # CPU reference (masked-mean, what our model returns)
    refs = []
    with torch.no_grad():
        for p in PROMPTS:
            enc = tok(p, return_tensors="pt", padding="max_length",
                      max_length=MAX_LEN, truncation=True)
            out = hf(enc["input_ids"], enc["attention_mask"], enc["token_type_ids"])
            h = out[0][0]                                       # [S, H]
            m = enc["attention_mask"][0].unsqueeze(-1).float()  # [S, 1]
            mm = (h * m).sum(0) / m.sum().clamp(min=1e-9)
            refs.append(mm.float().numpy().tolist())

    # Native + compile path
    dev = torch.device("neuron")
    dtype = torch.bfloat16
    ours = BertEncoder(hf.config, dtype=dtype).eval()
    load_from_hf(hf, ours)
    ours = ours.to(dtype).to(dev)
    if USE_COMPILE:
        ours = torch.compile(ours, backend="neuron", dynamic=False)

    out_neuron = []
    with torch.no_grad():
        for p in PROMPTS:
            enc = tok(p, return_tensors="pt", padding="max_length",
                      max_length=MAX_LEN, truncation=True)
            ids = enc["input_ids"].to(dev)
            am = enc["attention_mask"].to(dev).to(dtype)
            pos = torch.arange(MAX_LEN, device=dev).unsqueeze(0).expand_as(ids)
            emb = ours(ids, pos, am)
            torch_neuronx.synchronize()
            out_neuron.append(emb.detach().to(torch.float32).cpu().reshape(-1).tolist())

    print(f"=== Native + torch.compile vs CPU HF reference (MODEL={MODEL}) ===")
    print(f"{'prompt':<40}{'cosine':>10}")
    cs = []
    for i, p in enumerate(PROMPTS):
        c = cos(out_neuron[i], refs[i])
        cs.append(c)
        print(f"{p[:38]:<40}{c:>10.5f}")
    mn = min(cs)
    print(f"\nmin cosine: {mn:.5f}")
    print("VERDICT:", "PASS (>=0.99)" if mn >= 0.99 else
          ("CLOSE bf16 (0.95-0.99)" if mn >= 0.95 else "FAIL (<0.95)"))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"FAIL: {type(e).__name__}: {e}")
        sys.exit(1)
