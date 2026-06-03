#!/usr/bin/env python3
"""Validate + benchmark NKI GEMM vs torch matmul on Gemma4 projection shapes."""
import sys, os, time
sys.path.insert(0, "/work")
os.environ["NEURON_SKIP_EFA_AFFINITY"] = "1"
import torch, torch_neuronx
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

from nki_gemm import nki_gemm
wg = wrap_nki(nki_gemm)

# Gemma4 projection shapes: gate/up_proj = [M, 5376] @ [5376, 21504]
H, I = 5376, 21504
for M, label in [(1, "decode M=1"), (512, "prefill M=512")]:
    A = torch.randn(M, H)
    B = torch.randn(H, I)   # weight stored as [in, out]
    A_t = A.t().contiguous().to(dev)  # [H, M]
    Bd = B.to(dev)
    Ad = A.to(dev)

    r = wg(A_t, Bd); torch_neuronx.synchronize()
    exp = (Ad @ Bd)
    d = (r.cpu() - exp.cpu()).abs().max().item()
    rel = d / (exp.cpu().abs().max().item()+1e-9)
    print(f"\n[{label}]  C=[{M},{I}]  correctness diff {d:.4f} (rel {rel:.5f}) {'PASS' if rel<0.01 else 'FAIL'}")

    t_nki = timeit(lambda: wg(A_t, Bd))
    t_torch = timeit(lambda: Ad @ Bd)
    sp = t_torch/t_nki
    tag = f"NKI {sp:.2f}x faster" if t_nki<t_torch else f"torch {t_nki/t_torch:.2f}x faster"
    print(f"  NKI {t_nki:.3f}ms   torch {t_torch:.3f}ms   -> {tag}")

print("\nDone.")
