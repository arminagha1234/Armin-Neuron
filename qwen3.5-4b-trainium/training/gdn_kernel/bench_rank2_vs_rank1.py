"""On-device Rank-2 (NKI fwd + explicit bwd autograd.Function) vs Rank-1
(pure-torch chunked_gdn_forward + autograd) — parity + step/compile timing.

Run inside native_train with NEURON_RT_VISIBLE_CORES=0.
"""
import sys, time, argparse
import torch
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from chunked_gdn import chunked_gdn_forward
from chunked_gdn_nki import gdn_chunked_nki


def cos(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def make_inputs(B, T, H, D, dev, dtype, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, D); k = torch.randn(B, T, H, D); v = torch.randn(B, T, H, D)
    a = torch.randn(B, T, H)
    A_log = torch.log(torch.empty(H).uniform_(0, 16)); dt_bias = torch.ones(H)
    g = -A_log.exp() * F.softplus(a + dt_bias)
    beta = torch.randn(B, T, H).sigmoid()
    return [t.to(dev).to(dtype) if t is not g and t is not beta else t.to(dev).float()
            for t in (q, k, v, g, beta)]


def step_fb(fn, q, k, v, g, beta, BT):
    qa, ka, va, ga, ba = [t.detach().clone().requires_grad_(True) for t in (q, k, v, g, beta)]
    out = fn(qa, ka, va, ga, ba, chunk_size=BT, use_qk_l2norm_in_kernel=True)
    if isinstance(out, tuple):
        out = out[0]
    (out * out).sum().backward()
    return out, [qa.grad, ka.grad, va.grad, ga.grad, ba.grad]


def _sync(x):
    # force materialization on the neuron device
    return float(x.flatten()[0].cpu())


def timeit(fn, iters=5):
    # warm/compile
    t0 = time.time(); out, gr = fn(); _sync(gr[0]); compile_t = time.time() - t0
    t0 = time.time()
    for _ in range(iters):
        out, gr = fn(); _sync(gr[0])
    warm = (time.time() - t0) / iters
    return compile_t, warm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--BT", type=int, default=128)
    ap.add_argument("--iters", type=int, default=5)
    a = ap.parse_args()
    dev = torch.device("neuron")
    B, D = 1, 128
    q, k, v, g, beta = make_inputs(B, a.T, a.H, D, dev, torch.float32)

    print(f"=== device parity+timing B={B} T={a.T} H={a.H} D={D} BT={a.BT} ===")
    # parity
    _, gr1 = step_fb(chunked_gdn_forward, q, k, v, g, beta, a.BT)
    _, gr2 = step_fb(gdn_chunked_nki, q, k, v, g, beta, a.BT)
    o1, _ = chunked_gdn_forward(q, k, v, g, beta, chunk_size=a.BT, use_qk_l2norm_in_kernel=True)
    o2 = gdn_chunked_nki(q, k, v, g, beta, chunk_size=a.BT, use_qk_l2norm_in_kernel=True)
    print(f"forward cos(rank2, rank1) = {cos(o2.cpu(), o1.cpu()):.6f}")
    gr1 = [x.cpu() for x in gr1]; gr2 = [x.cpu() for x in gr2]
    for n, a_, b_ in zip(["dq", "dk", "dv", "dg", "dbeta"], gr2, gr1):
        print(f"  {n:6s} cos={cos(a_, b_):.6f}")

    # timing (eager)
    c1, w1 = timeit(lambda: step_fb(chunked_gdn_forward, q, k, v, g, beta, a.BT), a.iters)
    c2, w2 = timeit(lambda: step_fb(gdn_chunked_nki, q, k, v, g, beta, a.BT), a.iters)
    print(f"\nEAGER fwd+bwd step time (s):")
    print(f"  Rank-1 (pure-torch chunked): compile={c1:.2f}  warm={w1*1000:.1f} ms")
    print(f"  Rank-2 (NKI fwd + expl bwd): compile={c2:.2f}  warm={w2*1000:.1f} ms")


if __name__ == "__main__":
    main()
