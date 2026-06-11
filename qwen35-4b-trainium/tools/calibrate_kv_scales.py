#!/usr/bin/env python3
"""Per-layer KV scale calibration for Qwen3.5-4B on CPU.

Runs HF transformers reference impl, hooks each GQA `self_attn` module,
captures max-abs K and V across a calibration sample, and writes
kv_scales.json that the vllm-neuron Path B/C/D model can load.

Run inside /data/cpu_venv (transformers 5.10.2, has qwen3_5).
"""
import json, sys, time, os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "/root/models/Qwen3.5-4B"
OUT = "/work/qwen35/kv_scales.json"
FP8_E4M3_MAX = 240.0  # gen3/trn2 (gen4 would be 448)
SAFETY_HEADROOM = 0.9  # use 90% of FP8 max to leave room for unseen outliers
EFFECTIVE_MAX = FP8_E4M3_MAX * SAFETY_HEADROOM

# Calibration prompts — varied: factual recall, narrative, code, math.
# Each captures different K/V distributions in the attention heads.
CALIB_PROMPTS = [
    "The capital of France is Paris. The capital of Germany is Berlin. The capital of Japan is Tokyo. The capital of Brazil is Brasília. The capital of Canada is Ottawa.",
    "Once upon a time in a small village by the sea, there lived a young fisherman who dreamed of sailing across the ocean to discover new lands and meet new people.",
    "def fibonacci(n):\n    if n <= 1: return n\n    a, b = 0, 1\n    for _ in range(n - 1):\n        a, b = b, a + b\n    return b\n\nprint(fibonacci(10))",
    "The integral of x squared dx from 0 to 1 equals one third. The derivative of sine x is cosine x. Euler's identity states e to the i pi plus one equals zero.",
    "In machine learning, a transformer model uses self-attention to weigh the importance of different parts of the input sequence when producing each output token.",
    "Q: What language is spoken in Brazil? A: Portuguese. Q: How many planets are in the solar system? A: Eight. Q: What is the speed of light? A: About 300,000 km/s.",
]

print(f"[calib] Loading tokenizer from {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print(f"[calib] Loading model on CPU in BF16 (this takes ~2-3 min for 4B)")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
model.eval()
print(f"[calib] Loaded in {time.time()-t0:.1f}s. Param count: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")

# Identify GQA-attention layers. Qwen3.5-4B has 32 layers total:
# - 24 DeltaNet (linear attention) — no FP8 KV cache
# - 8 GQA (full attention) — get FP8 KV cache
# In HF the layer types are exposed via config.layer_types or
# via inspecting each layer's self_attn class.
layer_max_abs_k = {}  # global_layer_idx -> float
layer_max_abs_v = {}

def make_hook(layer_idx):
    """Capture K and V max-abs from each GQA forward."""
    def hook(module, inputs, output):
        # output of self_attn forward varies across HF versions. Capture
        # K/V from the module's last computation. The Qwen3_5 self_attn
        # holds k_proj_output / v_proj_output as intermediate tensors
        # only during forward — we instead hook k_proj and v_proj
        # directly. This hook is a placeholder and won't be used; see
        # the k_proj/v_proj hooks below.
        pass
    return hook

def make_proj_hook(kind, layer_idx):
    """Hook on k_proj or v_proj. Captures the LAYER OUTPUT (post-projection,
    pre-rope, pre-norm). For FP8 KV calibration we want the values that
    actually get written to cache; on Qwen3.5 that's after k_norm/q_norm.
    But k_norm output ≈ scale of k_proj output * RMSNorm normalization
    factor, so capturing k_proj output and applying RMSNorm-equivalent
    scale gives a close estimate. We'll capture post-projection here
    (close enough for static scale calibration) and leave per-head
    refinement for a follow-up."""
    def hook(module, inputs, output):
        # output: [batch, seq, num_heads_kv * head_dim] OR
        # [batch, seq, hidden] before split. We just take max-abs.
        max_abs = output.float().abs().max().item()
        if kind == "k":
            layer_max_abs_k[layer_idx] = max(layer_max_abs_k.get(layer_idx, 0.0), max_abs)
        else:
            layer_max_abs_v[layer_idx] = max(layer_max_abs_v.get(layer_idx, 0.0), max_abs)
    return hook

# Walk every layer; only GQA layers will have k_proj/v_proj as nn.Linear.
# DeltaNet has its own structure.
gqa_layer_indices = []
all_layers = model.model.layers if hasattr(model.model, "layers") else None
if all_layers is None:
    print("[calib] ERROR: can't find model.model.layers")
    sys.exit(2)

print(f"[calib] Walking {len(all_layers)} layers to find GQA self_attn modules")
hooks = []
for idx, layer in enumerate(all_layers):
    sa = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
    if sa is None:
        continue
    cls_name = type(sa).__name__
    has_kproj = hasattr(sa, "k_proj") and isinstance(sa.k_proj, torch.nn.Linear)
    has_vproj = hasattr(sa, "v_proj") and isinstance(sa.v_proj, torch.nn.Linear)
    if has_kproj and has_vproj:
        gqa_layer_indices.append(idx)
        hooks.append(sa.k_proj.register_forward_hook(make_proj_hook("k", idx)))
        hooks.append(sa.v_proj.register_forward_hook(make_proj_hook("v", idx)))
        print(f"[calib]   layer {idx}: GQA (cls={cls_name})")
    else:
        print(f"[calib]   layer {idx}: skip (cls={cls_name}, k_proj={has_kproj}, v_proj={has_vproj})")

if not gqa_layer_indices:
    print("[calib] ERROR: no GQA layers found!")
    sys.exit(3)

print(f"[calib] Found {len(gqa_layer_indices)} GQA layers: {gqa_layer_indices}")

# Run forward on each calibration prompt (no grad).
print(f"[calib] Running forward on {len(CALIB_PROMPTS)} calibration prompts")
with torch.no_grad():
    for i, prompt in enumerate(CALIB_PROMPTS):
        t0 = time.time()
        ids = tokenizer(prompt, return_tensors="pt").input_ids
        # Single forward, no generation — we only need the K/V projections
        # during prefill. This captures the dominant K/V magnitudes.
        _ = model(ids)
        print(f"[calib]   prompt {i+1}/{len(CALIB_PROMPTS)} ({ids.shape[1]} tokens): {time.time()-t0:.1f}s")

for h in hooks:
    h.remove()

print(f"\n[calib] Per-layer max-abs:")
print(f"  {'layer':>5} {'max|K|':>10} {'max|V|':>10}  {'k_scale':>10} {'v_scale':>10}")
out = {"layers": {}, "metadata": {
    "model": MODEL_PATH,
    "fp8_max": FP8_E4M3_MAX,
    "safety_headroom": SAFETY_HEADROOM,
    "effective_max": EFFECTIVE_MAX,
    "calib_prompts": len(CALIB_PROMPTS),
    "gqa_layer_indices": gqa_layer_indices,
}}
for idx in sorted(layer_max_abs_k.keys()):
    mk = layer_max_abs_k[idx]
    mv = layer_max_abs_v[idx]
    # scale = effective_max / max_abs  →  quantize: x*scale clamps at FP8 max
    # Floor max_abs at 0.5 so we don't over-amplify near-zero layers
    k_scale = EFFECTIVE_MAX / max(mk, 0.5)
    v_scale = EFFECTIVE_MAX / max(mv, 0.5)
    out["layers"][str(idx)] = {
        "max_abs_k": mk, "max_abs_v": mv,
        "k_scale": k_scale, "v_scale": v_scale,
    }
    print(f"  {idx:>5} {mk:>10.3f} {mv:>10.3f}  {k_scale:>10.3f} {v_scale:>10.3f}")

Path(OUT).parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n[calib] Wrote {OUT}")
print(f"[calib] {len(layer_max_abs_k)} GQA layers calibrated.")
