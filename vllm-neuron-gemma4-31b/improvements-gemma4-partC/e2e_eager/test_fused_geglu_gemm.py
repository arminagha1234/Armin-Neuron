#!/usr/bin/env python3
"""Validate + benchmark the fully-fused GeGLU up-projection vs the torch path.

torch path = gate=x@Wg; up=x@Wu; act=gelu_tanh(gate)*up  (3 ops, intermediate to HBM)
NKI path   = one kernel, intermediate stays in PSUM/SBUF.
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

from nki_fused_geglu_gemm import nki_fused_geglu_gemm
wf = wrap_nki(nki_fused_geglu_gemm)

H, I = 5376, 21504
for M, label in [(1, "decode M=1"), (512, "prefill M=512")]:
    x = torch.randn(M, H)
    Wg = torch.randn(H, I) * 0.02
    Wu = torch.randn(H, I) * 0.02
    x_t = x.t().contiguous().to(dev)
    Wgd, Wud, xd = Wg.to(dev), Wu.to(dev), x.to(dev)

    r = wf(x_t, Wgd, Wud); torch_neuronx.synchronize()
    exp = F.gelu(xd @ Wgd, approximate="tanh") * (xd @ Wud)
    d = (r.cpu()-exp.cpu()).abs().max().item()
    rel = d/(exp.cpu().abs().max().item()+1e-9)
    print(f"\n[{label}] act=[{M},{I}] correctness diff {d:.5f} (rel {rel:.5f}) {'PASS' if rel<0.02 else 'FAIL'}")

    def torch_path():
        return F.gelu(xd @ Wgd, approximate="tanh") * (xd @ Wud)
    def nki_path():
        return wf(x_t, Wgd, Wud)
    t_nki, t_torch = timeit(nki_path), timeit(torch_path)
    tag = f"NKI {t_torch/t_nki:.2f}x faster" if t_nki<t_torch else f"torch {t_nki/t_torch:.2f}x faster"
    print(f"  NKI {t_nki:.3f}ms   torch {t_torch:.3f}ms   -> {tag}")

print("\nDone.")
