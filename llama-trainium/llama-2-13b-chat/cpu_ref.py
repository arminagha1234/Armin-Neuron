"""Precompute a CPU fp32 reference (per-position argmax) and save it, so the
distributed TP run can compare without doing slow solo work mid-collective
(which would desync the ranks). Run once, single process:

    python3 cpu_ref.py <model_path> ref.pt
"""
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

mp, out = sys.argv[1], sys.argv[2]
prompt = ("Artificial intelligence is transforming the world. In this paper we "
          "describe how large language models can be trained efficiently on custom "
          "accelerators such as AWS Trainium. The key idea is to")
tok = AutoTokenizer.from_pretrained(mp)
ids = tok(prompt, return_tensors="pt").input_ids
m = AutoModelForCausalLM.from_pretrained(mp, dtype=torch.float32, attn_implementation="eager")
with torch.no_grad():
    pred = m(ids, use_cache=False).logits.float().argmax(-1)[0]
torch.save({"ids": ids, "cpu_pred": pred}, out)
print(f"saved CPU ref to {out}  tokens={ids.shape[1]}")
