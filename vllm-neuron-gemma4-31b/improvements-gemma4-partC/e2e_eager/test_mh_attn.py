#!/usr/bin/env python3
"""Validate + benchmark the multi-head batched attention kernel vs torch + 32-call."""
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

from nki_decode_attention_hd256_mh import nki_decode_attention_hd256_mh
from nki_decode_attention_hd256 import nki_decode_attention_hd256
wmh = wrap_nki(nki_decode_attention_hd256_mh)
w1 = wrap_nki(nki_decode_attention_hd256)

NH, HD, S = 32, 256, 512
q = torch.randn(NH, HD); K = torch.randn(NH, S, HD); V = torch.randn(NH, S, HD)

# batched layout for mh kernel
q_t = q.reshape(NH*HD, 1).contiguous().to(dev)
k_t = K.reshape(NH, S, HD).transpose(1,2).reshape(NH*HD, S).contiguous().to(dev)
v_d = V.reshape(NH*S, HD).contiguous().to(dev)

# correctness
r = wmh(q_t, k_t, v_d, 1.0, NH); torch_neuronx.synchronize()
exp = torch.bmm(torch.softmax(torch.bmm(q.unsqueeze(1), K.transpose(1,2)),-1), V).reshape(NH,HD)
d = (r.cpu()-exp).abs().max().item()
print(f"correctness mh vs torch: diff {d:.6f}  {'PASS' if d<0.05 else 'FAIL'}")

# benchmarks
qd, Kd, Vd = q.to(dev), K.to(dev), V.to(dev)
def torch_32():
    return torch.bmm(torch.softmax(torch.bmm(qd.unsqueeze(1), Kd.transpose(1,2)),-1), Vd).reshape(1,NH*HD)
q_t1 = [q[h].reshape(HD,1).contiguous().to(dev) for h in range(NH)]
k_t1 = [K[h].t().contiguous().to(dev) for h in range(NH)]
v_d1 = [V[h].contiguous().to(dev) for h in range(NH)]
def nki_32():
    return torch.cat([w1(q_t1[h],k_t1[h],v_d1[h],1.0) for h in range(NH)], dim=1)
def nki_mh():
    return wmh(q_t, k_t, v_d, 1.0, NH)

t_torch, t_32, t_mh = timeit(torch_32), timeit(nki_32), timeit(nki_mh)
print(f"torch batched : {t_torch:.3f}ms")
print(f"NKI 32 calls  : {t_32:.3f}ms")
print(f"NKI mh (1 call): {t_mh:.3f}ms")
print(f"  mh vs torch : {'NKI '+format(t_torch/t_mh,'.2f')+'x faster' if t_mh<t_torch else 'torch '+format(t_mh/t_torch,'.2f')+'x faster'}")
print("Done.")
