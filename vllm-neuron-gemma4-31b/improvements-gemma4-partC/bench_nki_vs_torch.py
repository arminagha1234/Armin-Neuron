#!/usr/bin/env python3
"""On-device microbenchmark: NKI kernels vs PyTorch-on-Neuron equivalents.

Both paths run on the SAME Neuron device (privateuseone:0) in the Beta 2 DLC.
We warm up (first call compiles the NEFF), then time N iterations with a single
sync at the end of each measured batch. This answers: for the exact op a kernel
fuses, is the NKI kernel faster than letting the framework run the unfused ops?

Honest notes baked in:
 - These are isolated-op microbenchmarks, NOT end-to-end serving throughput.
 - PyTorch-on-Neuron here is eager (the same path SDPA-fallback uses in serving).
 - Shapes are picked to mirror Gemma4 31B (hidden=5376, inter=30720, hd=256/512).
"""
import sys, os, time
sys.path.insert(0, "/work")
os.environ["NEURON_SKIP_EFA_AFFINITY"] = "1"
import torch
import torch.nn.functional as F
import torch_neuronx
from torch_neuronx import wrap_nki

dev = torch.device("privateuseone:0")
ITERS = 50
WARMUP = 5


def timeit(fn, iters=ITERS, warmup=WARMUP):
    """Run fn() warmup times, then time `iters` runs with one sync at the end."""
    for _ in range(warmup):
        fn()
    torch_neuronx.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch_neuronx.synchronize()
    return (time.time() - t0) / iters * 1e3  # ms per iter


def bench(name, nki_fn, torch_fn):
    try:
        nki_ms = timeit(nki_fn)
    except Exception as e:
        nki_ms = None
        print(f"  [{name}] NKI failed: {e}")
    try:
        torch_ms = timeit(torch_fn)
    except Exception as e:
        torch_ms = None
        print(f"  [{name}] torch failed: {e}")
    if nki_ms is not None and torch_ms is not None:
        speedup = torch_ms / nki_ms
        tag = f"{speedup:.2f}x faster" if speedup >= 1 else f"{1/speedup:.2f}x SLOWER"
        print(f"  {name:28s} NKI {nki_ms:8.3f}ms   torch {torch_ms:8.3f}ms   -> {tag}")
    else:
        print(f"  {name:28s} NKI {nki_ms}   torch {torch_ms}")


print(f"=== NKI vs PyTorch-on-Neuron microbench (iters={ITERS}, warmup={WARMUP}) ===")
print(f"device={dev}\n")

# ---- 1. RMSNorm + residual ----
from nki_fused_rmsnorm_residual import nki_fused_rmsnorm_residual
w_rms = wrap_nki(nki_fused_rmsnorm_residual)
T, H = 512, 5376
res = torch.randn(T, H, dtype=torch.float32).to(dev)
mo = torch.randn(T, H, dtype=torch.float32).to(dev)
wt = torch.ones(1, H, dtype=torch.float32).to(dev)
wt_flat = wt.squeeze(0)

def nki_rms():
    return w_rms(res, mo, wt, 1e-6)

def torch_rms():
    x = mo
    var = x.pow(2).mean(-1, keepdim=True)
    return res + (x * torch.rsqrt(var + 1e-6)) * wt_flat

bench("rmsnorm_residual [512,5376]", nki_rms, torch_rms)

# ---- 2. GeGLU ----
from nki_geglu_mlp import nki_geglu_mlp
w_geglu = wrap_nki(nki_geglu_mlp)
Ti, I = 512, 30720
gate = torch.randn(Ti, I, dtype=torch.float32).to(dev)
up = torch.randn(Ti, I, dtype=torch.float32).to(dev)

def nki_geglu():
    return w_geglu(gate, up)

def torch_geglu():
    return F.gelu(gate, approximate="tanh") * up

bench("geglu [512,30720]", nki_geglu, torch_geglu)

# ---- 3. QK-RMSNorm ----
from nki_qk_rmsnorm import nki_qk_rmsnorm
w_qk = wrap_nki(nki_qk_rmsnorm)
Tq, D = 512, 256
xq = torch.randn(Tq, D, dtype=torch.float32).to(dev)
wq = torch.randn(1, D, dtype=torch.float32).to(dev)
wq_flat = wq.squeeze(0)

def nki_qkn():
    return w_qk(xq, wq, 1e-6)

def torch_qkn():
    var = xq.pow(2).mean(-1, keepdim=True)
    return (xq * torch.rsqrt(var + 1e-6)) * wq_flat

bench("qk_rmsnorm [512,256]", nki_qkn, torch_qkn)

# ---- 4. Logit soft-cap ----
from nki_logit_softcap import nki_logit_softcap
w_cap = wrap_nki(nki_logit_softcap)
Tc, V = 256, 4096
xc = (torch.randn(Tc, V, dtype=torch.float32) * 20).to(dev)
cap = 30.0

def nki_cap():
    return w_cap(xc, cap)

def torch_cap():
    return cap * torch.tanh(xc / cap)

bench("logit_softcap [256,4096]", nki_cap, torch_cap)

# ---- 5. Embed scale ----
from nki_embed_scale import nki_embed_scale
w_emb = wrap_nki(nki_embed_scale)
Te, He = 512, 5376
emb = torch.randn(Te, He, dtype=torch.float32).to(dev)
sc = float(He) ** 0.5

def nki_emb():
    return w_emb(emb, sc)

def torch_emb():
    return emb * sc

bench("embed_scale [512,5376]", nki_emb, torch_emb)

# ---- 6. Decode attention hd256 ----
from nki_decode_attention_hd256 import nki_decode_attention_hd256
w_a256 = wrap_nki(nki_decode_attention_hd256)
S = 512
HD = 256
q = torch.randn(1, HD, dtype=torch.float32)
k = torch.randn(S, HD, dtype=torch.float32)
v = torch.randn(S, HD, dtype=torch.float32)
q_t = q.t().contiguous().to(dev)
k_t = k.t().contiguous().to(dev)
v_d = v.to(dev)
q_n = q.unsqueeze(0).to(dev)
k_n = k.unsqueeze(0).to(dev)
v_n = v.unsqueeze(0).to(dev)

def nki_a256():
    return w_a256(q_t, k_t, v_d, 1.0)

def torch_a256():
    return F.scaled_dot_product_attention(q_n, k_n, v_n, scale=1.0, is_causal=False)

bench("decode_attn_hd256 [S=512]", nki_a256, torch_a256)

# ---- 7. Decode attention hd512 ----
from nki_decode_attention_hd512 import nki_decode_attention_hd512
w_a512 = wrap_nki(nki_decode_attention_hd512)
HD2 = 512
q2 = torch.randn(1, HD2, dtype=torch.float32)
k2 = torch.randn(S, HD2, dtype=torch.float32)
v2 = torch.randn(S, HD2, dtype=torch.float32)
q2_t = q2.t().contiguous().to(dev)
k2_t = k2.t().contiguous().to(dev)
v2_d = v2.to(dev)
q2_n = q2.unsqueeze(0).to(dev)
k2_n = k2.unsqueeze(0).to(dev)
v2_n = v2.unsqueeze(0).to(dev)

def nki_a512():
    return w_a512(q2_t, k2_t, v2_d, 1.0)

def torch_a512():
    return F.scaled_dot_product_attention(q2_n, k2_n, v2_n, scale=1.0, is_causal=False)

bench("decode_attn_hd512 [S=512]", nki_a512, torch_a512)

print("\nDone.")
