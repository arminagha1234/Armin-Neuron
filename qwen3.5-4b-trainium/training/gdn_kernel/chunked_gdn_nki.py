"""
Rank-2 GDN: torch.autograd.Function wrapping the hand-written NKI forward kernel
(`gdn_nki_fwd.gdn_chunk_fwd`) + a validated backward.

Two backward modes (select via GDN_NKI_BWD env or `backward_mode` arg):
  * "explicit"  (default): our hand-derived explicit backward
    (`chunked_gdn_bwd_ref.gdn_core_backward`), validated cos 1.0 vs autograd.
    Recomputes forward intermediates in torch (recompute-in-backward, standard
    SSD/Mamba2 trick) — no full NKI backward kernel needed to be correct+fast-fwd.
  * "nki": the standalone NKI backward kernel (gdn_nki_bwd) once validated.

The NKI forward runs per (batch, head); this wrapper loops B*H and stacks.
All internal compute fp32 (matches the fp32-internal requirement for full-depth
numerical stability); boundary dtype is preserved.

Mirrors the mamba3_native.py wrapping pattern (wrap the fast kernel as the
forward graph op; provide a correct backward).
"""
from __future__ import annotations

import os
import torch
import torch.nn.functional as F

from chunked_gdn import l2norm
from chunked_gdn_bwd_ref import gdn_core_forward, gdn_core_backward


# --------------------------------------------------------------------------- #
# constant tiles (host) shared by every (b,h) kernel call                      #
# --------------------------------------------------------------------------- #
def _const_tiles(BT, D, device, dtype=torch.float32, blk=16):
    eye = torch.eye(BT, dtype=dtype, device=device)
    neg = -torch.tril(torch.ones(BT, BT, dtype=dtype, device=device), -1)
    Ld = torch.tril(torch.ones(BT, BT, dtype=dtype, device=device))
    ones_row = torch.ones(1, max(BT, D), dtype=dtype, device=device)
    # BLK x BLK block-diagonal mask for the overflow-safe block-LU inverse
    bdiag = torch.zeros(BT, BT, dtype=dtype, device=device)
    for i in range(0, BT, blk):
        bdiag[i:i + blk, i:i + blk] = 1.0
    return eye, neg, Ld, ones_row, bdiag


_KERNEL = None


def _get_kernel():
    global _KERNEL
    if _KERNEL is None:
        from gdn_nki_fwd import gdn_chunk_fwd
        _KERNEL = gdn_chunk_fwd
    return _KERNEL


def _nki_forward_core(qn, kn, v, g, beta, BT):
    """Run the NKI forward per (b,h). Inputs [B,T,H,D] (qn,kn L2-normed), g,beta
    [B,T,H]. Returns core_attn_out [B,T,H,D] (fp32)."""
    B, T, H, D = qn.shape
    dev = qn.device
    kernel = _get_kernel()
    eye, neg, Ld, ones_row, bdiag = _const_tiles(BT, D, dev)
    out = qn.new_zeros(B, T, H, D, dtype=torch.float32)
    for b in range(B):
        for h in range(H):
            q_sd = qn[b, :, h, :].contiguous().float()
            k_sd = kn[b, :, h, :].contiguous().float()
            v_sd = v[b, :, h, :].contiguous().float()
            g_sd = g[b, :, h].reshape(T, 1).contiguous().float()
            beta_sd = beta[b, :, h].reshape(T, 1).contiguous().float()
            o_bh, _fs = kernel(q_sd, k_sd, v_sd, g_sd, beta_sd, eye, neg, Ld, ones_row, bdiag)
            out[b, :, h, :] = o_bh
    return out


class GDNChunkedNKI(torch.autograd.Function):
    """autograd.Function: NKI forward + explicit(recompute) backward.

    forward inputs are the POST-L2norm-able raw q,k,v and raw g,beta (matching
    chunked_gdn_forward's `use_qk_l2norm_in_kernel=True` contract). L2-norm of
    q,k and the 1/sqrt(K) scale happen inside (scale inside the kernel; L2 here).
    """

    @staticmethod
    def forward(ctx, query, key, value, g, beta, chunk_size, use_l2norm, backward_mode):
        qn = l2norm(query, dim=-1, eps=1e-6) if use_l2norm else query
        kn = l2norm(key, dim=-1, eps=1e-6) if use_l2norm else key
        # NKI forward (fast path)
        core = _nki_forward_core(qn, kn, value, g, beta, chunk_size)
        ctx.save_for_backward(query, key, value, g, beta)
        ctx.chunk_size = chunk_size
        ctx.use_l2norm = use_l2norm
        ctx.backward_mode = backward_mode
        return core.to(query.dtype)

    @staticmethod
    def backward(ctx, grad_core):
        query, key, value, g, beta = ctx.saved_tensors
        BT = ctx.chunk_size
        # L2-norm adjoint handled by autograd: recompute the CORE forward under
        # enable_grad from the (post-l2norm) leaves, then autograd the core, and
        # push through l2norm with a second autograd on the norm alone. Simpler:
        # make q,k,v,g,beta leaves, run gdn_core_forward on l2normed inputs.
        with torch.enable_grad():
            q = query.detach().float().requires_grad_(True)
            k = key.detach().float().requires_grad_(True)
            v = value.detach().float().requires_grad_(True)
            gg = g.detach().float().requires_grad_(True)
            bb = beta.detach().float().requires_grad_(True)
            qn = l2norm(q, dim=-1, eps=1e-6) if ctx.use_l2norm else q
            kn = l2norm(k, dim=-1, eps=1e-6) if ctx.use_l2norm else k
            if ctx.backward_mode == "explicit":
                # explicit backward gives grads wrt (qn,kn,v,g,beta); need to also
                # push qn,kn grads through l2norm -> use autograd for the l2norm hop.
                _, saved = gdn_core_forward(qn, kn, v, gg, bb, chunk_size=BT, save=True)
                dqn, dkn, dv, dgg, dbb = gdn_core_backward(grad_core.float(), saved)
                # push dqn,dkn through l2norm via autograd
                if ctx.use_l2norm:
                    dq, dk = torch.autograd.grad(
                        outputs=[qn, kn], inputs=[q, k],
                        grad_outputs=[dqn, dkn], retain_graph=False)
                else:
                    dq, dk = dqn, dkn
                dv_, dgg_, dbb_ = dv, dgg, dbb
            else:  # full autograd through the core forward (fallback / nki-check)
                out = gdn_core_forward(qn, kn, v, gg, bb, chunk_size=BT, save=False)
                dq, dk, dv_, dgg_, dbb_ = torch.autograd.grad(
                    out, [q, k, v, gg, bb], grad_outputs=grad_core.float())
        cast = lambda t, r: t.to(r.dtype)
        return (cast(dq, query), cast(dk, key), cast(dv_, value),
                cast(dgg_, g), cast(dbb_, beta), None, None, None)


def gdn_chunked_nki(query, key, value, g, beta, chunk_size=128,
                    use_qk_l2norm_in_kernel=True, backward_mode=None):
    """Drop-in for chunked_gdn_forward's core (returns only core_attn_out).

    backward_mode: "explicit" (default) or "autograd" or env GDN_NKI_BWD.
    """
    if backward_mode is None:
        backward_mode = os.environ.get("GDN_NKI_BWD", "explicit")
    return GDNChunkedNKI.apply(query, key, value, g, beta, chunk_size,
                               use_qk_l2norm_in_kernel, backward_mode)
