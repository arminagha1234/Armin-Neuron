#!/usr/bin/env python3
"""Pure-PyTorch parity test for the Qwen3.6 GQA layer (layer 3).

Loads the real weights from safetensors, reimplements the GQA attention
the way OUR adapter does (per-head Q/gate splice, q/k norm, partial RoPE,
sigmoid gate), runs it on CPU for the same input HF saw, and compares the
layer-3 output to the HF reference captured in /mnt/data/hf_ref.pt.

This isolates math/weight-loading correctness from the NF-kernel + TP
plumbing. If this matches HF, the bug is in the device-side plumbing.
If it mismatches, the bug is in our math/weight interpretation — and
this script lets us fix it in seconds instead of 40-min device compiles.

Run in the hfref venv:
    source /mnt/data/hfref_venv/bin/activate
    python /mnt/data/parity_layer3.py
"""
import json
import os
import torch
import torch.nn.functional as F
from safetensors import safe_open

MODEL = "/mnt/data/models/Qwen3.6-27B"
REF = "/mnt/data/hf_ref.pt"
HF = "model.language_model"
LAYER = 3  # first full-attention layer


def load(key):
    with open(os.path.join(MODEL, "model.safetensors.index.json")) as f:
        idx = json.load(f)
    fname = idx["weight_map"][key]
    with safe_open(os.path.join(MODEL, fname), framework="pt") as f:
        return f.get_tensor(key)


def rms_norm(x, w, eps=1e-6):
    dt = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    # Qwen3.6 (1 + weight) convention
    return (x * (1.0 + w.float())).to(dt)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def main():
    cfg = json.load(open(os.path.join(MODEL, "config.json")))["text_config"]
    hidden = cfg["hidden_size"]      # 5120
    n_q = cfg["num_attention_heads"] # 24
    n_kv = cfg["num_key_value_heads"]# 4
    hd = cfg["head_dim"]             # 256
    eps = cfg["rms_norm_eps"]
    theta = cfg["rope_parameters"]["rope_theta"]
    prf = cfg["partial_rotary_factor"]  # 0.25
    rotary_dim = int(round(hd * prf))   # 64

    ref = torch.load(REF)
    ids = ref["ids"]
    # Input to layer 3 = HF's hidden state going INTO layer 3.
    # We captured layer0_out and layer3_out (outputs). We need the INPUT to
    # layer 3. Easiest: re-run our math on the layer-3 INPUT which is the
    # output of layer 2. We didn't capture that — so instead compare the
    # TRANSFORM layer 3 applies: feed HF's actual layer-3 input.
    # We captured 'layer0_out' (output of layer 0). Not layer 2. So this
    # harness instead validates the attention sub-computation in isolation
    # using a known input: HF's embed_out broadcast (a proxy). For a real
    # parity we need the exact layer-3 input — capture it below.
    if "layer3_in" not in ref:
        print("[parity] NOTE: need layer3_in capture. Re-run hf_ref with the")
        print("         layer-3 INPUT hook (pre-forward). Using layer2 proxy off.")
    x = ref.get("layer3_in")
    if x is None:
        print("[parity] layer3_in missing — cannot run exact parity. Exiting.")
        print("[parity] (will add input hook to hf_ref and re-capture.)")
        return

    x = x.to(torch.bfloat16)  # (T, hidden)
    T = x.shape[0]

    # Load weights
    p = f"{HF}.layers.{LAYER}.self_attn"
    q_proj = load(f"{p}.q_proj.weight")   # (n_q*hd*2, hidden) per-head interleaved
    k_proj = load(f"{p}.k_proj.weight")   # (n_kv*hd, hidden)
    v_proj = load(f"{p}.v_proj.weight")
    o_proj = load(f"{p}.o_proj.weight")   # (hidden, n_q*hd)
    q_norm = load(f"{p}.q_norm.weight")   # (hd,)
    k_norm = load(f"{p}.k_norm.weight")
    in_ln = load(f"{HF}.layers.{LAYER}.input_layernorm.weight")
    post_ln = load(f"{HF}.layers.{LAYER}.post_attention_layernorm.weight")
    gate_proj = load(f"{HF}.layers.{LAYER}.mlp.gate_proj.weight")
    up_proj = load(f"{HF}.layers.{LAYER}.mlp.up_proj.weight")
    down_proj = load(f"{HF}.layers.{LAYER}.mlp.down_proj.weight")

    # === Attention (HF-faithful) ===
    residual = x
    h = rms_norm(x, in_ln, eps)

    qg = (h @ q_proj.T.to(h.dtype))            # (T, n_q*hd*2)
    qg = qg.view(T, n_q, hd * 2)
    q, gate = qg.chunk(2, dim=-1)              # each (T, n_q, hd) — per-head split
    gate = gate.reshape(T, -1)                 # (T, n_q*hd)

    k = (h @ k_proj.T.to(h.dtype)).view(T, n_kv, hd)
    v = (h @ v_proj.T.to(h.dtype)).view(T, n_kv, hd)

    q = rms_norm(q, q_norm, eps)               # (T, n_q, hd)
    k = rms_norm(k, k_norm, eps)               # (T, n_kv, hd)

    # RoPE (partial, NeoX)
    pos = torch.arange(T).float()
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
    freqs = pos[:, None] * inv_freq[None, :]   # (T, rotary_dim/2)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # (T, rotary_dim)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)

    def apply_rope(t):  # t: (T, nheads, hd)
        t_rot = t[..., :rotary_dim]
        t_pass = t[..., rotary_dim:]
        c = cos[:, None, :].to(t.dtype)
        s = sin[:, None, :].to(t.dtype)
        t_rot = t_rot * c + rotate_half(t_rot) * s
        return torch.cat([t_rot, t_pass], dim=-1)

    q = apply_rope(q)
    k = apply_rope(k)

    # GQA: repeat kv
    rep = n_q // n_kv
    k = k.repeat_interleave(rep, dim=1)   # (T, n_q, hd)
    v = v.repeat_interleave(rep, dim=1)

    # attention (causal)
    q = q.transpose(0, 1)   # (n_q, T, hd)
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    scale = hd ** -0.5
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale  # (n_q, T, T)
    mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
    scores = scores + mask
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v.float())   # (n_q, T, hd)
    out = out.transpose(0, 1).reshape(T, n_q * hd).to(h.dtype)  # (T, n_q*hd)

    # gate (sigmoid)
    gate_act = torch.sigmoid(gate.float())
    print(f"[parity] gate sigmoid: mean={gate_act.mean():.4f} std={gate_act.std():.4f} "
          f"min={gate_act.min():.4f} max={gate_act.max():.4f}")
    out_nogate = out.clone()
    out = out * gate_act.to(out.dtype)

    attn_out = out @ o_proj.T.to(out.dtype)   # (T, hidden)
    h = residual + attn_out

    # === MLP ===
    residual = h
    hn = rms_norm(h, post_ln, eps)
    gate_x = hn @ gate_proj.T.to(hn.dtype)
    up_x = hn @ up_proj.T.to(hn.dtype)
    mlp = (F.silu(gate_x.float()).to(hn.dtype) * up_x) @ down_proj.T.to(hn.dtype)
    h = residual + mlp

    # Compare to HF layer3_out
    ref_out = ref["layer3_out"].float()
    while ref_out.dim() > 2:
        ref_out = ref_out.squeeze(0)
    ours = h.float()
    cos_sim = F.cosine_similarity(ours.flatten(), ref_out.flatten(), dim=0)
    max_abs = (ours - ref_out).abs().max()
    print(f"[parity] layer3 ours vs HF:  cos={cos_sim.item():.6f}  max_abs={max_abs.item():.4f}")
    print(f"[parity] ours  mean={ours.mean():.5f} std={ours.std():.5f}")
    print(f"[parity] HF    mean={ref_out.mean():.5f} std={ref_out.std():.5f}")

    print(f"[parity] shapes: ours={tuple(ours.shape)} ref={tuple(ref_out.shape)}")
    # Per-token cosine to see if a specific position is off
    if ours.shape == ref_out.shape and ours.dim() == 2:
        for t in range(ours.shape[0]):
            a = ours[t]; b = ref_out[t]
            c = (a @ b) / (a.norm() * b.norm() + 1e-9)
            print(f"[parity]   tok{t}: cos={c.item():.5f} ours_std={a.std().item():.4f} hf_std={b.std().item():.4f}")


if __name__ == "__main__":
    main()
