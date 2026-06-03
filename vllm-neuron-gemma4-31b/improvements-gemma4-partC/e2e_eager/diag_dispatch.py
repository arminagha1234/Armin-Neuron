#!/usr/bin/env python3
"""Diagnose WHY e2e decode shows no win: dispatch overhead + tiny-tensor effect.

Three measurements:
  A) 32-head attention as 32 separate NKI kernel calls  vs  torch batched bmm
  B) a norm at decode shape [1, 5376]                   NKI vs torch
  C) a norm at prefill shape [512, 5376]                NKI vs torch
"""
import sys, os, time
sys.path.insert(0, "/work")
os.environ["NEURON_SKIP_EFA_AFFINITY"] = "1"
import torch, torch.nn.functional as F, torch_neuronx
from torch_neuronx import wrap_nki

dev = torch.device("privateuseone:0")
ITERS, WARMUP = 30, 5

def timeit(fn):
    for _ in range(WARMUP): fn()
    torch_neuronx.synchronize()
    t0 = time.time()
    for _ in range(ITERS): fn()
    torch_neuronx.synchronize()
    return (time.time()-t0)/ITERS*1e3

# A) attention: 32 heads, hd=256, S=512
from nki_decode_attention_hd256 import nki_decode_attention_hd256
wa = wrap_nki(nki_decode_attention_hd256)
NH, HD, S = 32, 256, 512
q = torch.randn(NH, HD).to(dev); K = torch.randn(NH, S, HD).to(dev); V = torch.randn(NH, S, HD).to(dev)
q_t = [q[h].reshape(HD,1).contiguous() for h in range(NH)]
k_t = [K[h].t().contiguous() for h in range(NH)]
v_d = [V[h].contiguous() for h in range(NH)]

def nki_32():
    outs = [wa(q_t[h], k_t[h], v_d[h], 1.0) for h in range(NH)]
    return torch.cat(outs, dim=1)

def torch_32():
    qh = q.unsqueeze(1)  # [NH,1,HD]
    sc = torch.bmm(qh, K.transpose(1,2))
    pr = torch.softmax(sc, -1)
    return torch.bmm(pr, V).reshape(1, NH*HD)

a_nki, a_torch = timeit(nki_32), timeit(torch_32)
print(f"A) 32-head attn:  NKI(32 calls) {a_nki:.3f}ms   torch(batched) {a_torch:.3f}ms"
      f"   -> {'NKI '+format(a_torch/a_nki,'.2f')+'x' if a_nki<a_torch else 'torch '+format(a_nki/a_torch,'.2f')+'x faster'}")

# B) norm at decode shape [1, 5376]
from nki_qk_rmsnorm import nki_qk_rmsnorm
wn = wrap_nki(nki_qk_rmsnorm)
H = 5376
x1 = torch.randn(1, H).to(dev); w1 = torch.randn(1, H).to(dev); w1f = w1.squeeze(0)
def nki_n1(): return wn(x1, w1, 1e-6)
def torch_n1():
    v = x1.pow(2).mean(-1, keepdim=True); return (x1*torch.rsqrt(v+1e-6))*w1f
b_nki, b_torch = timeit(nki_n1), timeit(torch_n1)
print(f"B) norm [1,5376]:    NKI {b_nki:.3f}ms   torch {b_torch:.3f}ms"
      f"   -> {'NKI '+format(b_torch/b_nki,'.2f')+'x' if b_nki<b_torch else 'torch '+format(b_nki/b_torch,'.2f')+'x faster'}")

# C) norm at prefill shape [512, 5376]
x2 = torch.randn(512, H).to(dev)
def nki_n2(): return wn(x2, w1, 1e-6)
def torch_n2():
    v = x2.pow(2).mean(-1, keepdim=True); return (x2*torch.rsqrt(v+1e-6))*w1f
c_nki, c_torch = timeit(nki_n2), timeit(torch_n2)
print(f"C) norm [512,5376]:  NKI {c_nki:.3f}ms   torch {c_torch:.3f}ms"
      f"   -> {'NKI '+format(c_torch/c_nki,'.2f')+'x' if c_nki<c_torch else 'torch '+format(c_nki/c_torch,'.2f')+'x faster'}")
print("Done.")
