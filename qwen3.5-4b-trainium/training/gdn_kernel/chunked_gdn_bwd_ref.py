"""
Explicit (hand-derived) chunked GatedDeltaNet BACKWARD — the pure-torch math
ORACLE for the Rank-2 NKI backward kernel.

This transcribes the analytic VJP of the EXACT Rank-1 forward
(`chunked_gdn.chunked_gdn_forward`, scalar-per-head decay, Neumann-doubling
(I-A)^-1) as a reverse chunk scan. It is validated to match `torch.autograd`
on the same forward to cos 1.0 (see `test_bwd_ref()`), so it is the correct
blueprint to port to NKI (mamba2_kernel.py-style recompute-forward-in-backward,
reverse-scan dh recurrence, reverse-cumsum dg).

Scope: the CORE chunked scan only. Inputs are POST-L2norm q,k (the L2-norm,
beta=sigmoid, g=-exp(A_log)*softplus gate, and out gated-RMSNorm are trivial
pointwise ops that stay in torch around the core and are handled by autograd /
§1.4 of GDN_BACKWARD_PLAN.md). `scale = 1/sqrt(K)` is applied INSIDE (matches
chunked_gdn.py line 138).

Layout (matches chunked_gdn.py): external tensors are [B,T,H,D] and [B,T,H];
internally transposed to [B,H,T,D] / [B,H,T], padded to a multiple of BT,
chunked to [B,H,NT,BT,D].

Grads returned (all wrt the POST-norm core inputs): dq, dk, dv, dg, dbeta.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# Reuse the overflow-safe block forward-substitution inverse from the Rank-1
# module (and the tiny-block Neumann helper). The old full-chunk Neumann doubling
# formed A^32/A^64 which overflows fp32 with real weights -> full-32L NaN; the
# block method never forms powers above A^(blk/2). Kept API-compatible.
from chunked_gdn import _block_forward_sub_inverse, _neumann_inverse_unit_lower  # noqa: F401


def _reverse_cumsum(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Adjoint of cumsum: dg[t] = sum_{s>=t} dgc[s]."""
    return torch.flip(torch.cumsum(torch.flip(x, dims=[dim]), dim=dim), dims=[dim])


# --------------------------------------------------------------------------- #
# forward CORE (post-norm inputs; scale applied inside) — differentiable, and  #
# also returns the saved intermediates the explicit backward needs.            #
# --------------------------------------------------------------------------- #
def gdn_core_forward(query, key, value, g, beta, chunk_size: int = 128,
                     save: bool = False):
    """Chunked gated-delta forward CORE. Inputs [B,T,H,D] (q,k already L2-normed),
    g,beta [B,T,H] (g = raw per-token log-decay <=0; beta already sigmoid'd).
    Returns core_attn_out [B,T,H,D] (and a `saved` dict if save=True)."""
    B, T, H, K = query.shape
    Vd = value.shape[-1]
    q, k, v, beta_t, g_t = [x.transpose(1, 2).contiguous().float()
                            for x in (query, key, value, beta, g)]
    pad = (chunk_size - T % chunk_size) % chunk_size
    if pad:
        q = F.pad(q, (0, 0, 0, pad)); k = F.pad(k, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        beta_t = F.pad(beta_t, (0, pad)); g_t = F.pad(g_t, (0, pad))
    Tp = T + pad
    scale = 1.0 / (K ** 0.5)
    q = q * scale
    vb = v * beta_t.unsqueeze(-1)
    kb = k * beta_t.unsqueeze(-1)
    NT = Tp // chunk_size
    BT = chunk_size
    rs = lambda x, d: x.reshape(B, H, NT, BT, d)
    q, k, v = rs(q, K), rs(k, K), rs(v, Vd)
    kb, vb = rs(kb, K), rs(vb, Vd)
    g_c = g_t.reshape(B, H, NT, BT)

    mask_incl_diag = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), 0)
    gc = g_c.cumsum(dim=-1)                                    # [B,H,NT,BT]
    gdiff = (gc.unsqueeze(-1) - gc.unsqueeze(-2)).tril()
    decay_mask = gdiff.clamp(max=0.0).exp().tril()            # [B,H,NT,BT,BT]

    Araw = (kb @ k.transpose(-1, -2)) * decay_mask
    A0 = (-Araw).masked_fill(mask_incl_diag, 0.0)            # strictly-lower
    Tinv = _block_forward_sub_inverse(A0, BT, blk=16)        # overflow-safe
    egc = gc.exp().unsqueeze(-1)                              # [B,H,NT,BT,1]
    kg = kb * egc
    u = Tinv @ vb
    w = Tinv @ kg                                            # k_cumdecay

    S = q.new_zeros(B, H, K, Vd)
    outs, S_list, vnew_list = [], [], []
    for i in range(NT):
        q_i, k_i, u_i, w_i = q[:, :, i], k[:, :, i], u[:, :, i], w[:, :, i]
        gc_i, dm_i = gc[:, :, i], decay_mask[:, :, i]
        attn_i = (q_i @ k_i.transpose(-1, -2)) * dm_i
        v_prime = w_i @ S
        v_new = u_i - v_prime
        attn_inter = (q_i * gc_i.unsqueeze(-1).exp()) @ S
        out_i = attn_inter + attn_i @ v_new
        outs.append(out_i)
        S_list.append(S)
        vnew_list.append(v_new)
        g_last = gc_i[:, :, -1]
        state_decay = g_last[:, :, None, None].exp()
        k_decay = (k_i * (g_last[:, :, None] - gc_i).exp().unsqueeze(-1)).transpose(-1, -2)
        S = S * state_decay + k_decay @ v_new

    core = torch.stack(outs, dim=2).reshape(B, H, -1, Vd)[:, :, :T]
    core = core.transpose(1, 2).contiguous()
    if not save:
        return core
    saved = dict(q=q, k=k, v=v, vb=vb, kb=kb, kg=kg, gc=gc, egc=egc,
                 decay_mask=decay_mask, Tinv=Tinv, u=u, w=w,
                 S_list=S_list, vnew_list=vnew_list, mask_incl_diag=mask_incl_diag,
                 scale=scale, BT=BT, NT=NT, T=T, pad=pad, beta_t=beta_t, g_t=g_t)
    return core, saved


# --------------------------------------------------------------------------- #
# explicit BACKWARD — analytic VJP, reverse chunk scan.                        #
# --------------------------------------------------------------------------- #
def gdn_core_backward(grad_core, saved):
    """Explicit backward. grad_core [B,T,H,D]; returns dq,dk,dv,dg,dbeta [B,T,H,*]."""
    q, k, v = saved["q"], saved["k"], saved["v"]
    vb, kb, kg = saved["vb"], saved["kb"], saved["kg"]
    gc, egc, decay_mask = saved["gc"], saved["egc"], saved["decay_mask"]
    Tinv, u, w = saved["Tinv"], saved["u"], saved["w"]
    S_list, vnew_list = saved["S_list"], saved["vnew_list"]
    scale, BT, NT, T, pad = saved["scale"], saved["BT"], saved["NT"], saved["T"], saved["pad"]
    B, H = q.shape[0], q.shape[1]
    K, Vd = q.shape[-1], v.shape[-1]
    dev, dt = q.device, q.dtype

    # incoming grad -> [B,H,T,D] -> pad -> chunks
    do = grad_core.transpose(1, 2).contiguous().float()
    if pad:
        do = F.pad(do, (0, 0, 0, pad))
    do = do.reshape(B, H, NT, BT, Vd)

    egc_full = gc.exp()                       # [B,H,NT,BT]
    glast = gc[:, :, :, -1]                   # [B,H,NT]
    e_state = glast.exp()                     # [B,H,NT]

    # ---- reverse chunk scan for dS (the dh recurrence) ----
    d_attn = torch.zeros(B, H, NT, BT, BT, device=dev, dtype=dt)
    dqg = torch.zeros(B, H, NT, BT, K, device=dev, dtype=dt)
    dkdecay = torch.zeros(B, H, NT, BT, K, device=dev, dtype=dt)
    du = torch.zeros(B, H, NT, BT, Vd, device=dev, dtype=dt)
    dw = torch.zeros(B, H, NT, BT, K, device=dev, dtype=dt)
    dglast_state = torch.zeros(B, H, NT, device=dev, dtype=dt)

    dS = q.new_zeros(B, H, K, Vd)             # dL/dS_{i+1}
    for i in range(NT - 1, -1, -1):
        do_i = do[:, :, i]
        S_i = S_list[i]
        vnew_i = vnew_list[i]
        w_i = w[:, :, i]
        attn_i = (q[:, :, i] @ k[:, :, i].transpose(-1, -2)) * decay_mask[:, :, i]
        qg_i = q[:, :, i] * egc_full[:, :, i].unsqueeze(-1)
        e_i = (glast[:, :, i:i+1] - gc[:, :, i]).exp()             # [B,H,BT]
        kdecay_i = k[:, :, i] * e_i.unsqueeze(-1)

        dvnew_i = attn_i.transpose(-1, -2) @ do_i + kdecay_i @ dS  # [B,H,BT,V]
        d_attn[:, :, i] = do_i @ vnew_i.transpose(-1, -2)
        dqg[:, :, i] = do_i @ S_i.transpose(-1, -2)
        dkdecay[:, :, i] = vnew_i @ dS.transpose(-1, -2)
        dglast_state[:, :, i] = e_state[:, :, i] * (S_i * dS).sum(dim=(-2, -1))
        du[:, :, i] = dvnew_i
        dvprime_i = -dvnew_i
        dw[:, :, i] = dvprime_i @ S_i.transpose(-1, -2)
        dS = (qg_i.transpose(-1, -2) @ do_i
              + e_state[:, :, i][:, :, None, None] * dS
              + w_i.transpose(-1, -2) @ dvprime_i)

    # ---- chunk-parallel backward for the rest ----
    dq = torch.zeros_like(q); dk = torch.zeros_like(k); dv = torch.zeros_like(v)
    dgc = torch.zeros(B, H, NT, BT, device=dev, dtype=dt)
    dbeta = torch.zeros(B, H, NT, BT, device=dev, dtype=dt)

    # step7: attn_i = (q@k^T) * decay_mask
    M = q @ k.transpose(-1, -2)
    draw_attn = d_attn * decay_mask
    dq += draw_attn @ k
    dk += draw_attn.transpose(-1, -2) @ q
    dDmask = d_attn * M

    # step8: qg = q * egc
    dq += dqg * egc_full.unsqueeze(-1)
    dgc += (dqg * q).sum(-1) * egc_full

    # step9: kdecay = k * exp(glast-gc)
    e_all = (glast.unsqueeze(-1) - gc).exp()                       # [B,H,NT,BT]
    dk += dkdecay * e_all.unsqueeze(-1)
    de = (dkdecay * k).sum(-1)                                     # [B,H,NT,BT]
    dglast_kdecay = (de * e_all).sum(-1)                           # [B,H,NT]
    dgc += -de * e_all

    # combine dglast into dgc[...,-1]
    dgc[:, :, :, -1] += dglast_state + dglast_kdecay

    # step11: u=Tinv@vb, w=Tinv@kg
    dTinv = du @ vb.transpose(-1, -2) + dw @ kg.transpose(-1, -2)
    dvb = Tinv.transpose(-1, -2) @ du
    dkg = Tinv.transpose(-1, -2) @ dw

    # step12: Tinv=(I-A0)^-1  => dA0 = Tinv^T dTinv Tinv^T
    TinvT = Tinv.transpose(-1, -2)
    dA0 = TinvT @ dTinv @ TinvT

    # step13: A0 = -Araw on strictly-lower
    strict_lower = torch.tril(torch.ones(BT, BT, dtype=torch.bool, device=dev), -1)
    dAraw = (-dA0) * strict_lower

    # step14: Araw = (kb@k^T) * decay_mask
    M2 = kb @ k.transpose(-1, -2)
    draw_kkT = dAraw * decay_mask
    dDmask += dAraw * M2
    dkb = draw_kkT @ k
    dk += draw_kkT.transpose(-1, -2) @ kb

    # step15: kg = kb * egc
    dkb += dkg * egc_full.unsqueeze(-1)
    dgc += (dkg * kb).sum(-1) * egc_full

    # step16: decay_mask backprop (retained region: c>=j, clamp inactive)
    term = (dDmask * decay_mask).tril()
    dgc += term.sum(-1)          # index c (rows)
    dgc += -term.sum(-2)         # index j (cols)

    # step17: kb=k*beta, vb=v*beta
    beta_chunks = saved["beta_t"].reshape(B, H, NT, BT)
    dk += dkb * beta_chunks.unsqueeze(-1)
    dbeta += (dkb * k).sum(-1)
    dv += dvb * beta_chunks.unsqueeze(-1)
    dbeta += (dvb * v).sum(-1)

    # step18: gc=cumsum(g) -> dg = reverse_cumsum(dgc)
    dg = _reverse_cumsum(dgc, dim=-1)

    # step19: q scaled by `scale`
    dq = dq * scale

    # ---- reshape [B,H,NT,BT,*] -> [B,H,Tp,*] -> slice -> [B,T,H,*] ----
    def unchunk(x, d):
        return x.reshape(B, H, NT * BT, d)[:, :, :T].transpose(1, 2).contiguous()

    def unchunk1(x):
        return x.reshape(B, H, NT * BT)[:, :, :T].transpose(1, 2).contiguous()

    return (unchunk(dq, K), unchunk(dk, K), unchunk(dv, Vd),
            unchunk1(dg), unchunk1(dbeta))


# --------------------------------------------------------------------------- #
# validation vs autograd on the SAME forward core                             #
# --------------------------------------------------------------------------- #
def _cos(a, b):
    a, b = a.flatten().double(), b.flatten().double()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


def test_bwd_ref(B=1, T=256, H=4, K=128, Vd=128, BT=128, seed=0, dtype=torch.float64):
    torch.manual_seed(seed)
    mk = lambda *s: torch.randn(*s, dtype=dtype)
    query = mk(B, T, H, K); key = mk(B, T, H, K); value = mk(B, T, H, Vd)
    a = mk(B, T, H)
    A_log = torch.log(torch.empty(H, dtype=dtype).uniform_(0, 16))
    dt_bias = torch.ones(H, dtype=dtype)
    g = -A_log.exp() * F.softplus(a + dt_bias)
    beta = mk(B, T, H).sigmoid()

    # autograd reference
    qa = query.clone().requires_grad_(True); ka = key.clone().requires_grad_(True)
    va = value.clone().requires_grad_(True); ga = g.clone().requires_grad_(True)
    ba = beta.clone().requires_grad_(True)
    out = gdn_core_forward(qa, ka, va, ga, ba, chunk_size=BT, save=False)
    grad_core = torch.randn_like(out)
    out.backward(grad_core)
    ref = [qa.grad, ka.grad, va.grad, ga.grad, ba.grad]

    # explicit backward
    _, saved = gdn_core_forward(query, key, value, g, beta, chunk_size=BT, save=True)
    dq, dk, dv, dg, dbeta = gdn_core_backward(grad_core, saved)
    ours = [dq, dk, dv, dg, dbeta]

    names = ["dq", "dk", "dv", "dg", "dbeta"]
    print(f"--- explicit-bwd vs autograd  B={B} T={T} H={H} K={K} BT={BT} ---")
    ok = True
    for n, a_, b_ in zip(names, ours, ref):
        c = _cos(a_, b_); mx = (a_ - b_).abs().max().item()
        rel = mx / (b_.abs().max().item() + 1e-30)   # relative roundoff
        good = c > 0.9999999 and rel < 1e-5
        ok = ok and good
        print(f"  {n:6s} cos={c:.8f} maxabs={mx:.3e} rel={rel:.3e} {'OK' if good else 'FAIL'}")
    return ok


if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    ok = True
    ok &= test_bwd_ref(B=1, T=256, H=4, K=128, Vd=128, BT=128, seed=0)
    ok &= test_bwd_ref(B=2, T=384, H=8, K=128, Vd=128, BT=128, seed=3)
    ok &= test_bwd_ref(B=1, T=128, H=2, K=64, Vd=64, BT=64, seed=7)
    print(f"\nBWD_REF_RESULT: {'PASS' if ok else 'FAIL'}")
