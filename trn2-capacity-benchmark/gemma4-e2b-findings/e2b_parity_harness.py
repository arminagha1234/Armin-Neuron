#!/usr/bin/env python3
"""Per-layer parity harness: PORT (vllm-neuron, tensor_capture) vs HF (CPU).

Localizes the first decoder layer whose hidden-state output diverges from the
HuggingFace reference. Runs inside the vllm-neuron 0.21 container.

- PORT: offline vllm.LLM with neuron_config.tensor_capture on model.layers.0..N-1,
  single short prompt, max_tokens=1 (one prefill). Captures per-layer outputs.
- HF:   AutoModelForCausalLM on CPU, output_hidden_states=True, same prompt/tokens.
- COMPARE: cosine + max-abs per layer; print the FIRST layer with cosine < 0.99.
"""
import os, sys, glob, json
import torch

O = os.environ.get("O", "/tmp/parity"); os.makedirs(O, exist_ok=True)
HF_MODEL = os.environ.get("HF_MODEL", "/tmp/models/e2b")
PORT_MODEL = os.environ.get("PORT_MODEL", "/tmp/models/e2b-text")
PROMPT = "The capital of France is"
CAPS = f"{O}/caps"; os.makedirs(CAPS, exist_ok=True)
NLAYERS = int(os.environ.get("NLAYERS", "35"))

def log(*a):
    print("[parity]", *a, flush=True)

# ---------------- HF reference (per-layer hidden states) ----------------
def run_hf_dtype(dtype, tok, ids):
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL, dtype=dtype, attn_implementation="eager")
    model.eval()
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    hs = [h[0].float() for h in out.hidden_states]  # [embed, L0..L(N-1)] pre-per-layer; last is POST-final-norm
    nxt = out.logits[0, -1].float().argmax().item()
    del model
    return hs, nxt

def run_hf():
    log("=== HF reference (bf16 + fp32 noise floor) ===")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(HF_MODEL)
    ids = tok(PROMPT, return_tensors="pt")["input_ids"]
    log("HF token ids:", ids.tolist())
    hs_bf16, nxt_bf16 = run_hf_dtype(torch.bfloat16, tok, ids)
    log("HF bf16 next token:", nxt_bf16, "->", repr(tok.decode([nxt_bf16])))
    hs_fp32, nxt_fp32 = run_hf_dtype(torch.float32, tok, ids)
    log("HF fp32 next token:", nxt_fp32, "->", repr(tok.decode([nxt_fp32])))
    log("HF hidden count:", len(hs_bf16), "shapes:", hs_bf16[0].shape)
    torch.save({"bf16": hs_bf16, "fp32": hs_fp32, "ids": ids}, f"{O}/hf.pt")
    return hs_bf16, hs_fp32, ids

# ---------------- PORT capture via tensor_capture ----------------
def run_port(ids):
    log("=== PORT (vllm-neuron tensor_capture) ===")
    from vllm import LLM, SamplingParams
    modules = [f"model.layers.{i}" for i in range(NLAYERS)] + ["model.norm"]
    llm = LLM(
        model=PORT_MODEL, tensor_parallel_size=1, max_model_len=128,
        max_num_seqs=1, max_num_batched_tokens=128,
        enable_prefix_caching=False, num_gpu_blocks_override=128,
        additional_config={"neuron_config": {
            "tensor_capture": {"modules": modules, "capture_dir": CAPS},
            "num_batched_tokens_buckets": [128], "num_seqs_buckets": [1],
            "on_device_sampling_config": {"all_greedy": True},
        }})
    sp = SamplingParams(max_tokens=1, temperature=0.0)
    tokids = ids[0].tolist()
    log("PORT prompt_token_ids:", tokids)
    out = llm.generate(prompts=[{"prompt_token_ids": tokids}], sampling_params=sp)
    log("PORT next token:", repr(out[0].outputs[0].text))
    files = sorted(glob.glob(f"{CAPS}/**/*", recursive=True))
    log("capture files (%d):" % len(files))
    for f in files[:60]:
        log("  ", f.replace(CAPS, ""))
    return files

# ---------------- compare ----------------
def load_capture(path):
    try:
        if path.endswith(".pt"): return torch.load(path, map_location="cpu")
        if path.endswith(".npy"):
            import numpy as np; return torch.from_numpy(np.load(path))
    except Exception as e:
        log("load fail", path, repr(e)[:100])
    return None

def _cos(a, b):
    n = min(a.shape[0], b.shape[0])
    a = a[:n].reshape(-1).float(); b = b[:n].reshape(-1).float()
    if a.numel() != b.numel(): return None
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def compare(hs, files, hs_fp32=None):
    log("=== COMPARE (cosine per layer) ===")
    import re
    # index port captures by layer idx from filename
    port = {}
    for f in files:
        if os.path.isdir(f): continue
        m = re.search(r"layers[._](\d+)", f)
        if not m: continue
        t = load_capture(f)
        if t is None: continue
        if isinstance(t, (list, tuple)): t = t[0]
        if hasattr(t, "float"):
            port[int(m.group(1))] = t.detach().float().reshape(-1, t.shape[-1])
    log("port layers captured:", sorted([k for k in port if isinstance(k,int)])[:40], "norm:", "norm" in port)
    log("  layer | port-vs-HFbf16 | HFbf16-vs-fp32(noise) | excess")
    first_div = None
    for i in range(NLAYERS):
        if i not in port: continue
        cos = _cos(hs[i+1], port[i])          # port vs HF bf16 (layer i output, pre-norm — valid for i<last)
        noise = _cos(hs[i+1], hs_fp32[i+1]) if hs_fp32 is not None else None
        if cos is None: log(f"  L{i}: SHAPE MISMATCH"); continue
        shared = i >= (NLAYERS - 20)
        excess = (noise - cos) if noise is not None else None
        # "real" divergence = port disagrees with HF far MORE than bf16-vs-fp32 noise does
        real = excess is not None and excess > 0.02 and cos < 0.995
        if real and first_div is None: first_div = i
        log(f"  L{i:2d} {'SH' if shared else '  '} cos={cos:.4f}  noise={noise if noise is None else round(noise,4)}  excess={excess if excess is None else round(excess,4)}{'  <<< REAL DIVERGENCE' if real else ''}")
    # final-norm-aligned check: port's model.norm output vs HF's last hidden (post-norm)
    if "norm" in port:
        cn = _cos(hs[-1], port["norm"])
        nn = _cos(hs[-1], hs_fp32[-1]) if hs_fp32 is not None else None
        log(f"  FINAL-NORM  port-vs-HFbf16 cos={cn:.4f}  noise={nn if nn is None else round(nn,4)}")
    log("FIRST REAL DIVERGENT LAYER (excess>0.02):", first_div)

def main():
    hs_bf16, hs_fp32, ids = run_hf()
    files = run_port(ids)
    compare(hs_bf16, files, hs_fp32)
    log("DONE")

if __name__ == "__main__":
    main()
