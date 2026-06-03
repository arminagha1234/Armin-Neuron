#!/usr/bin/env python3
"""bf16 throughput: optimized GEMM + fused GeGLU vs torch. bf16 is the serving dtype
and where Trainium's PE array runs ~4x faster than fp32. PSUM stays fp32 (matmul
always accumulates fp32); inputs/outputs bf16."""
import sys, os, time
sys.path.insert(0, "/work")
os.environ["NEURON_SKIP_EFA_AFFINITY"] = "1"
import torch, torch.nn.functional as F, torch_neuronx
from torch_neuronx import wrap_nki

dev = torch.device("privateuseone:0")
ITERS, WARMUP = 20, 5
bf16 = torch.bfloat16

def timeit(fn):
    for _ in range(WARMUP): fn()
    torch_neuronx.synchronize()
    t0 = time.time()
    for _ in range(ITERS): fn()
    torch_neuronx.synchronize()
    return (time.time()-t0)/ITERS*1e3

from nki_gemm_opt import nki_gemm_opt
from nki_fused_geglu_gemm_opt import nki_fused_geglu_gemm_opt
wg_opt = wrap_nki(nki_gemm_opt)
wf_opt = wrap_nki(nki_fused_geglu_gemm_opt)

H, I = 5376, 21504

print("=== bf16 GEMM (hoist-load) vs torch matmul ===")
for M in (128, 512, 1024):
    A = torch.randn(M, H, dtype=bf16); B = (torch.randn(H, I)*0.02).to(bf16)
    A_t = A.t().contiguous().to(dev); Bd = B.to(dev); Ad = A.to(dev)
    r = wg_opt(A_t, Bd); torch_neuronx.synchronize()
    ref = (Ad@Bd)
    rel = (r.float().cpu()-ref.float().cpu()).abs().max().item()/(ref.float().cpu().abs().max().item()+1e-9)
    t_nki = timeit(lambda: wg_opt(A_t, Bd)); t_t = timeit(lambda: Ad@Bd)
    tag = f"NKI {t_t/t_nki:.2f}x" if t_nki<t_t else f"torch {t_nki/t_t:.2f}x"
    print(f"  M={M:5d}: NKI {t_nki:7.3f}ms  torch {t_t:7.3f}ms  -> {tag}  (rel {rel:.4f})")

print("\n=== bf16 fused GeGLU vs torch 3-op ===")
for M in (128, 512, 1024):
    x = torch.randn(M, H, dtype=bf16); Wg=(torch.randn(H,I)*0.02).to(bf16); Wu=(torch.randn(H,I)*0.02).to(bf16)
    x_t = x.t().contiguous().to(dev); Wgd, Wud, xd = Wg.to(dev), Wu.to(dev), x.to(dev)
    r = wf_opt(x_t, Wgd, Wud); torch_neuronx.synchronize()
    exp = F.gelu(xd@Wgd, approximate="tanh")*(xd@Wud)
    rel = (r.float().cpu()-exp.float().cpu()).abs().max().item()/(exp.float().cpu().abs().max().item()+1e-9)
    def tp(): return F.gelu(xd@Wgd, approximate="tanh")*(xd@Wud)
    def np_(): return wf_opt(x_t, Wgd, Wud)
    t_nki, t_t = timeit(np_), timeit(tp)
    tag = f"NKI {t_t/t_nki:.2f}x" if t_nki<t_t else f"torch {t_nki/t_t:.2f}x"
    print(f"  M={M:5d}: NKI {t_nki:7.3f}ms  torch {t_t:7.3f}ms  -> {tag}  (rel {rel:.4f})")

print("\nDone.")
