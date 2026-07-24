"""End-to-end parity for the Rank-2 autograd.Function (GDNChunkedNKI):
  forward: NKI kernel output vs Rank-1 chunked_gdn_forward
  backward: grads dq,dk,dv,dg,dbeta vs autograd through chunked_gdn_forward

Runs via nki.simulate on CPU (no device needed). --device for real silicon.
"""
import sys, argparse, os
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from chunked_gdn import chunked_gdn_forward

# For CPU simulate, patch the kernel invocation to go through nki.simulate.
import gdn_nki_fwd
import nki
_real = gdn_nki_fwd.gdn_chunk_fwd


def _sim_kernel(*args):
    o, fs = nki.simulate(_real)(*[np.ascontiguousarray(np.asarray(a, np.float32)) for a in
                                  [x.detach().cpu().numpy() if torch.is_tensor(x) else x for x in args]])
    return torch.from_numpy(np.asarray(o)).float(), torch.from_numpy(np.asarray(fs)).float()


def cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def make_inputs(B, T, H, D, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, D); k = torch.randn(B, T, H, D); v = torch.randn(B, T, H, D)
    a = torch.randn(B, T, H)
    A_log = torch.log(torch.empty(H).uniform_(0, 16)); dt_bias = torch.ones(H)
    g = -A_log.exp() * F.softplus(a + dt_bias)
    beta = torch.randn(B, T, H).sigmoid()
    return q, k, v, g, beta


def run(B, T, H, D, BT, on_device):
    print(f"\n=== Rank-2 autograd.Function parity  B={B} T={T} H={H} D={D} BT={BT} ===")
    if not on_device:
        gdn_nki_fwd.gdn_chunk_fwd = staticmethod(_sim_kernel).__func__  # route to simulate
        import chunked_gdn_nki
        chunked_gdn_nki._KERNEL = _sim_kernel
    from chunked_gdn_nki import gdn_chunked_nki

    q, k, v, g, beta = make_inputs(B, T, H, D)
    if on_device:
        import torch_xla.core.xla_model as xm
        dev = xm.xla_device()
        q, k, v, g, beta = [t.to(dev) for t in (q, k, v, g, beta)]

    # ---- oracle fwd+bwd ----
    qa, ka, va, ga, ba = [t.detach().clone().requires_grad_(True) for t in (q, k, v, g, beta)]
    o_ref, _ = chunked_gdn_forward(qa, ka, va, ga, ba, chunk_size=BT, use_qk_l2norm_in_kernel=True)
    grad = torch.randn_like(o_ref)
    o_ref.backward(grad)
    ref = [qa.grad, ka.grad, va.grad, ga.grad, ba.grad]

    # ---- ours fwd+bwd ----
    qb, kb, vb, gb, bb = [t.detach().clone().requires_grad_(True) for t in (q, k, v, g, beta)]
    o_ours = gdn_chunked_nki(qb, kb, vb, gb, bb, chunk_size=BT, use_qk_l2norm_in_kernel=True,
                             backward_mode="explicit")
    o_ours.backward(grad)
    ours = [qb.grad, kb.grad, vb.grad, gb.grad, bb.grad]

    fc = cos(o_ours, o_ref)
    print(f"  forward   cos={fc:.6f}  maxabs={(o_ours-o_ref).abs().max():.3e}")
    ok = fc > 0.9999
    for n, a_, b_ in zip(["dq", "dk", "dv", "dg", "dbeta"], ours, ref):
        c = cos(a_, b_); mx = (a_ - b_).abs().max().item()
        good = c > 0.9999
        ok = ok and good
        print(f"  {n:6s} cos={c:.6f} maxabs={mx:.3e} {'OK' if good else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", action="store_true")
    ap.add_argument("--BT", type=int, default=128)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--H", type=int, default=2)
    a = ap.parse_args()
    ok = run(1, a.T, a.H, 128, a.BT, a.device)
    print("\n" + ("RANK2 AUTOGRAD PARITY PASSED" if ok else "RANK2 AUTOGRAD PARITY FAILED"))
    sys.exit(0 if ok else 1)
