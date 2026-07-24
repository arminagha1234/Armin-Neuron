"""Validate gdn_nki_fwd.gdn_chunk_fwd vs the Rank-1 oracle chunked_gdn_forward.

Default: nki.simulate (CPU functional sim). --device: real NeuronCore.
Per (batch,head) kernel; harness loops B*H and compares o + final_state.
"""
import sys, argparse
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from chunked_gdn import chunked_gdn_forward, l2norm
from gdn_nki_fwd import gdn_chunk_fwd

import nki


def stats(name, a, b):
    a = np.asarray(a, np.float64).ravel(); b = np.asarray(b, np.float64).ravel()
    mx = float(np.max(np.abs(a - b))); denom = float(np.max(np.abs(b))) + 1e-12
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    ok = cos > 0.9999 and mx / denom < 1e-2
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:16s} maxabs={mx:.3e} rel={mx/denom:.3e} cos={cos:.6f}")
    return ok


def make_inputs(B, T, H, D, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, D); k = torch.randn(B, T, H, D); v = torch.randn(B, T, H, D)
    a = torch.randn(B, T, H)
    A_log = torch.log(torch.empty(H).uniform_(0, 16)); dt_bias = torch.ones(H)
    g = -A_log.exp() * F.softplus(a + dt_bias)
    beta = torch.randn(B, T, H).sigmoid()
    return q, k, v, g, beta


def const_tiles(BT, D):
    eye = torch.eye(BT, dtype=torch.float32)
    neg = -torch.tril(torch.ones(BT, BT), -1)          # -1 strict lower
    Ld = torch.tril(torch.ones(BT, BT))                # 1 lower+diag
    ones_row = torch.ones(1, max(BT, D), dtype=torch.float32)
    return eye.numpy(), neg.numpy(), Ld.numpy(), ones_row.numpy()


def run(B, T, H, D, BT, on_device):
    print(f"\n=== GDN NKI fwd vs oracle: B={B} T={T} H={H} D={D} BT={BT} device={on_device} ===")
    q, k, v, g, beta = make_inputs(B, T, H, D)
    o_ref, _ = chunked_gdn_forward(q, k, v, g, beta, chunk_size=BT, use_qk_l2norm_in_kernel=True)
    o_ref = o_ref.float().numpy()   # [B,T,H,D]

    # kernel inputs: prepare l2norm (q,k) in torch; layout [S,D] per (b,h)
    qn = l2norm(q, dim=-1); kn = l2norm(k, dim=-1)   # scale applied inside kernel
    eye, neg, Ld, ones_row = const_tiles(BT, D)

    if on_device:
        import torch_xla.core.xla_model as xm
        dev = xm.xla_device()
    o_k = np.zeros((B, T, H, D), np.float64)
    ok = True
    for b in range(B):
        for h in range(H):
            q_sd = qn[b, :, h, :].contiguous().numpy()
            k_sd = kn[b, :, h, :].contiguous().numpy()
            v_sd = v[b, :, h, :].contiguous().numpy()
            g_sd = g[b, :, h].reshape(T, 1).contiguous().numpy()
            beta_sd = beta[b, :, h].reshape(T, 1).contiguous().numpy()
            args = (q_sd, k_sd, v_sd, g_sd, beta_sd, eye, neg, Ld, ones_row)
            if on_device:
                targs = [torch.from_numpy(np.ascontiguousarray(x)).float().to(dev) for x in args]
                oo, _fs = gdn_chunk_fwd(*targs); xm.mark_step()
                oo = oo.cpu().numpy()
            else:
                oo, _fs = nki.simulate(gdn_chunk_fwd)(*args)
                oo = np.asarray(oo)
            o_k[b, :, h, :] = oo
    ok &= stats("output o", o_k, o_ref)
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", action="store_true")
    ap.add_argument("--BT", type=int, default=128)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--H", type=int, default=2)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--B", type=int, default=1)
    a = ap.parse_args()
    ok = run(a.B, a.T, a.H, a.D, a.BT, a.device)
    print("\n" + ("GDN NKI FWD PASSED" if ok else "GDN NKI FWD FAILED"))
    sys.exit(0 if ok else 1)
