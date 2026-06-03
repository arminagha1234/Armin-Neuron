#!/usr/bin/env python3
"""HONEST test: NKI split-K decode attention vs the REAL NxDI decode fallback.

When attn_kernel_enabled=False (which NxDI forces for Gemma4 head_dim>128),
decode attention runs the *decomposed* path, NOT clean F.scaled_dot_product_attention.
The decomposed path (from NxDI attention_base.compute_for_flash_decoding /
compute_for_token_gen) is:

    K = repeat_kv(K, n_rep)                       # GQA expand to n_q_heads
    V = repeat_kv(V, n_rep)
    scores = matmul(Q, K.transpose(-1,-2)) * scale
    probs  = softmax(scores, dim=-1, dtype=fp32)  # fp32 upcast
    out    = matmul(probs, V)

That's what actually executes for Gemma4 decode today. THIS is the right
baseline to compare the NKI kernel against — not the fused SDPA megakernel that
isn't available for head_dim>128.

We test per-LAYER (all 32 q-heads, GQA), the realistic decode unit:
  SWA layer:    32 q-heads, 16 kv-heads, head_dim=256, S=512
  Global layer: 32 q-heads,  4 kv-heads, head_dim=512, S=512

NKI path uses the batched multi-head kernel (one dispatch). For the global
layer (hd=512) we need an mh variant too; here we loop the hd512 single-head
kernel for heads but note that in the table.
"""
import sys, os, time
sys.path.insert(0, "/work")
os.environ["NEURON_SKIP_EFA_AFFINITY"] = "1"
import torch, torch.nn.functional as F, torch_neuronx
from torch_neuronx import wrap_nki

dev = torch.device("privateuseone:0")
ITERS, WARMUP = 20, 5

def timeit(fn):
    for _ in range(WARMUP): fn()
    torch_neuronx.synchronize()
    t0 = time.time()
    for _ in range(ITERS): fn()
    torch_neuronx.synchronize()
    return (time.time()-t0)/ITERS*1e3

def repeat_kv(x, n_rep):
    # x: [n_kv, S, D] -> [n_kv*n_rep, S, D]
    if n_rep == 1:
        return x
    n_kv, S, D = x.shape
    return x[:, None, :, :].expand(n_kv, n_rep, S, D).reshape(n_kv * n_rep, S, D)

def decomposed_fallback(q, K, V, n_rep, scale):
    """The real NxDI decode fallback (decomposed GQA attention)."""
    Kf = repeat_kv(K, n_rep)               # [n_q, S, D]
    Vf = repeat_kv(V, n_rep)
    qh = q.unsqueeze(1)                     # [n_q, 1, D]
    scores = torch.bmm(qh, Kf.transpose(1, 2)) * scale   # [n_q,1,S]
    probs = torch.softmax(scores.float(), dim=-1).to(q.dtype)
    return torch.bmm(probs, Vf).reshape(1, q.shape[0] * q.shape[1])  # [1, n_q*D]

# ---- SWA layer: hd=256, 32 q / 16 kv ----
from nki_decode_attention_hd256_mh import nki_decode_attention_hd256_mh
w256 = wrap_nki(nki_decode_attention_hd256_mh)
# ---- Global layer: hd=512, 32 q / 4 kv ----
from nki_decode_attention_hd512_mh import nki_decode_attention_hd512_mh
w512 = wrap_nki(nki_decode_attention_hd512_mh)

def run_layer(kernel, NH, NKV, HD, S=512, dtype=torch.float32):
    n_rep = NH // NKV
    scale = 1.0
    q = torch.randn(NH, HD, dtype=dtype)
    K = torch.randn(NKV, S, HD, dtype=dtype)
    V = torch.randn(NKV, S, HD, dtype=dtype)
    qd, Kd, Vd = q.to(dev), K.to(dev), V.to(dev)

    Kf = repeat_kv(K, n_rep); Vf = repeat_kv(V, n_rep)   # [NH,S,HD]
    q_t = q.reshape(NH*HD, 1).contiguous().to(dev)
    k_t = Kf.transpose(1, 2).reshape(NH*HD, S).contiguous().to(dev)
    v_d = Vf.reshape(NH*S, HD).contiguous().to(dev)

    r = kernel(q_t, k_t, v_d, scale, NH); torch_neuronx.synchronize()
    exp = decomposed_fallback(qd, Kd, Vd, n_rep, scale)
    diff = (r.cpu().float().reshape(NH, HD) - exp.cpu().float().reshape(NH, HD)).abs().max().item()

    t_fb = timeit(lambda: decomposed_fallback(qd, Kd, Vd, n_rep, scale))
    t_nki = timeit(lambda: kernel(q_t, k_t, v_d, scale, NH))
    return diff, t_fb, t_nki

def run_swa(S=512, dtype=torch.float32):
    return run_layer(w256, 32, 16, 256, S, dtype)

print("=== SWA layer (32 q-heads, 16 kv, hd=256, S=512) — NKI vs REAL fallback ===")
for dt, name in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
    try:
        diff, t_fb, t_nki = run_layer(w256, 32, 16, 256, 512, dt)
        sp = t_fb / t_nki
        tag = f"NKI {sp:.2f}x faster" if t_nki < t_fb else f"fallback {t_nki/t_fb:.2f}x faster"
        print(f"  [{name}] diff {diff:.4f} | fallback {t_fb:.3f}ms  NKI {t_nki:.3f}ms  -> {tag}")
    except Exception as e:
        print(f"  [{name}] FAIL: {e}")

print("\n=== Global layer (32 q-heads, 4 kv, hd=512, S=512) — NKI vs REAL fallback ===")
for dt, name in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
    try:
        diff, t_fb, t_nki = run_layer(w512, 32, 4, 512, 512, dt)
        sp = t_fb / t_nki
        tag = f"NKI {sp:.2f}x faster" if t_nki < t_fb else f"fallback {t_nki/t_fb:.2f}x faster"
        print(f"  [{name}] diff {diff:.4f} | fallback {t_fb:.3f}ms  NKI {t_nki:.3f}ms  -> {tag}")
    except Exception as e:
        print(f"  [{name}] FAIL: {e}")

# bf16 sanity: relative error vs the magnitude of the output (0.117 abs may just
# be bf16 noise on large values). Report relative error too.
print("\n=== bf16 correctness sanity (relative error) ===")
NH, NKV, HD, S = 32, 16, 256, 512
n_rep = NH // NKV
q = torch.randn(NH, HD, dtype=torch.bfloat16)
K = torch.randn(NKV, S, HD, dtype=torch.bfloat16)
V = torch.randn(NKV, S, HD, dtype=torch.bfloat16)
Kf = repeat_kv(K, n_rep); Vf = repeat_kv(V, n_rep)
q_t = q.reshape(NH*HD,1).contiguous().to(dev)
k_t = Kf.transpose(1,2).reshape(NH*HD, S).contiguous().to(dev)
v_d = Vf.reshape(NH*S, HD).contiguous().to(dev)
r = w256(q_t, k_t, v_d, 1.0, NH); torch_neuronx.synchronize()
# fp32 golden (the TRUE answer both approximate)
exp32 = decomposed_fallback(q.float().to(dev), K.float().to(dev), V.float().to(dev), n_rep, 1.0)
exp_bf = decomposed_fallback(q.to(dev), K.to(dev), V.to(dev), n_rep, 1.0)
g = exp32.cpu().float().reshape(NH, HD)
nki_err = (r.cpu().float().reshape(NH,HD) - g).abs().max().item() / (g.abs().max().item()+1e-9)
fb_err  = (exp_bf.cpu().float().reshape(NH,HD) - g).abs().max().item() / (g.abs().max().item()+1e-9)
print(f"  vs fp32 golden: NKI rel-err {nki_err:.4f} | bf16 fallback rel-err {fb_err:.4f}")
print(f"  (if comparable, the NKI 'diff' is just bf16 rounding, not a kernel bug)")

print("\nDone.")
