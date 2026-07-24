"""Prove on-device NKI fwd matches the CPU fp32 oracle (not the bf16-downcast
on-device torch). If NKI-vs-CPUfp32 >> torch-vs-CPUfp32, the kernel is MORE
accurate on silicon than the torch Rank-1 path (coordinator's bf16 thesis)."""
import sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from chunked_gdn import chunked_gdn_forward, l2norm
from gdn_nki_fwd import gdn_chunk_fwd

def cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()

B, T, H, D, BT = 1, 256, 4, 128, 128
torch.manual_seed(0)
q = torch.randn(B, T, H, D); k = torch.randn(B, T, H, D); v = torch.randn(B, T, H, D)
a = torch.randn(B, T, H)
A_log = torch.log(torch.empty(H).uniform_(0, 16)); dt_bias = torch.ones(H)
g = -A_log.exp() * F.softplus(a + dt_bias); beta = torch.randn(B, T, H).sigmoid()

# CPU fp32 oracle
o_cpu, _ = chunked_gdn_forward(q, k, v, g, beta, chunk_size=BT, use_qk_l2norm_in_kernel=True)
o_cpu = o_cpu.float()

# on-device torch Rank-1 (bf16-autocast matmuls)
dev = torch.device("neuron")
o_dev_torch, _ = chunked_gdn_forward(q.to(dev), k.to(dev), v.to(dev), g.to(dev), beta.to(dev),
                                     chunk_size=BT, use_qk_l2norm_in_kernel=True)
o_dev_torch = o_dev_torch.float().cpu()

# on-device NKI (genuine fp32 PSUM)
qn = l2norm(q, dim=-1); kn = l2norm(k, dim=-1)
eye = torch.eye(BT); neg = -torch.tril(torch.ones(BT, BT), -1)
Ld = torch.tril(torch.ones(BT, BT)); ones_row = torch.ones(1, max(BT, D))
o_nki = torch.zeros(B, T, H, D)
for b in range(B):
    for h in range(H):
        args = [x.float().to(dev) for x in
                (qn[b,:,h,:], kn[b,:,h,:], v[b,:,h,:], g[b,:,h].reshape(T,1),
                 beta[b,:,h].reshape(T,1), eye, neg, Ld, ones_row)]
        oo, _ = gdn_chunk_fwd(*args)
        o_nki[b,:,h,:] = oo.float().cpu()

print(f"on-device torch Rank-1  vs CPU-fp32 oracle : cos={cos(o_dev_torch, o_cpu):.6f}")
print(f"on-device NKI  (fp32)   vs CPU-fp32 oracle : cos={cos(o_nki, o_cpu):.6f}")
print(f"on-device NKI           vs on-device torch : cos={cos(o_nki, o_dev_torch):.6f}")
