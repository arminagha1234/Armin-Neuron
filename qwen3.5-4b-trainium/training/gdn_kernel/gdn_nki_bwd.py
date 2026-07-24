"""
GatedDeltaNet (Qwen3.5-4B) chunked BACKWARD — NKI kernel SCAFFOLD for Trainium2.

STATUS: structural scaffold — DOES NOT COMPILE YET. Two known items:
  (1) the chunk-parallel FINALIZE (decay-mask backprop, the (I-A0)^-1 inverse
      adjoint dA0=N^T dTinv N^T, kb/vb/kg unpacking, reverse-cumsum for dg) is
      not yet wired — see the TODO block at the bottom of the reverse loop.
  (2) per-chunk entering-state caching uses S_all[NT,D,D] as an SBUF tile, which
      puts NT on the partition dim (trace-blocking). FIX: store per-chunk state
      the mamba2 way — a cache indexed on the FREE dim (e.g. (D, NT, D)) or in
      shared_hbm — so each S_i slice keeps D on partition.
Mirrors mamba3_ssd_bwd_kernel_v0.py (a scoped scaffold). The MATH is fully
validated in pure torch (chunked_gdn_bwd_ref, cos 1.0); this is the NKI port
in progress. The SHIPPING backward is the explicit-torch one in
chunked_gdn_nki.GDNChunkedNKI (cos 1.0, faster than Rank-1 on device).

The MATH ORACLE for this kernel is chunked_gdn_bwd_ref.gdn_core_backward,
validated cos 1.0 vs torch.autograd. Every tile below has a named counterpart in
that file; port + validate incrementally via nki.simulate against it.

The SHIPPING Rank-2 path (chunked_gdn_nki.GDNChunkedNKI) does NOT depend on this
kernel: it uses the validated NKI forward + the pure-torch explicit backward
(recompute-in-backward), which is already cos 1.0 and faster than Rank-1 on
device. This kernel is the pure-NKI perf follow-up.

Two-pass structure (mamba2_kernel.py pattern):
  PASS 1  forward replay -> store entering state S_i[NT,D,D] to SBUF/HBM.
  PASS 2  reverse chunk scan i=NT-1..0: recompute chunk intermediates from S_i,
          apply VJP, roll dS.  Finalize dg = reverse_cumsum(dgc).

All fp32 (genuine fp32 PSUM). Per (batch,head). Token-major [S,D].
nc_matmul: dst[M,N] = sum_P stat[P,M]*mov[P,N] (contract = partition dim).
"""
from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl


def _T(data, P, F):
    ps = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=ps, data=data)
    sb = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sb, src=ps)
    return sb


def _mm(stat, mov, M, N):
    """stat[P,M].T @ mov[P,N] -> sbuf (M,N)  (contract P)."""
    ps = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=ps, stationary=stat, moving=mov)
    sb = nl.ndarray((M, N), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sb, src=ps)
    return sb


def _recompute_chunk(k_td, v_c, q_td, g_col, beta_col, I, nL, Ld, LdT, BT, D, m, scale):
    """Recompute forward intermediates for one chunk (matches gdn_core_forward).
    Returns the tiles the backward VJP needs."""
    gc = _mm(LdT, g_col, BT, 1)                                  # cumsum(g)
    egc = nl.ndarray((BT, 1), dtype=nl.float32, buffer=nl.sbuf); nisa.activation(dst=egc, op=nl.exp, data=gc, bias=None, scale=1.0)
    gcrow = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf); nisa.memset(dst=gcrow, value=0.0)
    nisa.tensor_scalar(dst=gcrow, data=gcrow, op0=nl.add, operand0=gc, engine=nisa.vector_engine)
    gccol = _T(gcrow, BT, BT)
    gdiff = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_tensor(dst=gdiff, data1=gcrow, data2=gccol, op=nl.subtract)
    gdiff_cl = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_scalar(dst=gdiff_cl, data=gdiff, op0=nl.minimum, operand0=0.0)
    edm = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf); nisa.activation(dst=edm, op=nl.exp, data=gdiff_cl, bias=None, scale=1.0)
    dmask = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_tensor(dst=dmask, data1=edm, data2=Ld, op=nl.multiply)
    kb = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_scalar(dst=kb, data=k_td, op0=nl.multiply, operand0=beta_col, engine=nisa.vector_engine)
    vb = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_scalar(dst=vb, data=v_c, op0=nl.multiply, operand0=beta_col, engine=nisa.vector_engine)
    kbT = _T(kb, D, BT); kT = _T(k_td, D, BT)
    Araw = _mm(kbT, kT, BT, BT)
    Am = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_tensor(dst=Am, data1=Araw, data2=dmask, op=nl.multiply)
    A0 = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_tensor(dst=A0, data1=Am, data2=nL, op=nl.multiply)
    acc = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_tensor(dst=acc, data1=I, data2=A0, op=nl.add)
    Apow = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_copy(dst=Apow, src=A0)
    for _j in nl.static_range(m - 1):
        ApowT = _T(Apow, BT, BT); Apow = _mm(ApowT, Apow, BT, BT)
        accT = _T(acc, BT, BT); accA = _mm(accT, Apow, BT, BT)
        acc_n = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_tensor(dst=acc_n, data1=acc, data2=accA, op=nl.add); acc = acc_n
    N = acc
    kg = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_scalar(dst=kg, data=kb, op0=nl.multiply, operand0=egc, engine=nisa.vector_engine)
    N_T = _T(N, BT, BT)
    u = _mm(N_T, vb, BT, D); w = _mm(N_T, kg, BT, D)
    qs = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_scalar(dst=qs, data=q_td, op0=nl.multiply, operand0=scale)
    kT2 = kT
    return dict(gc=gc, egc=egc, dmask=dmask, kb=kb, vb=vb, kg=kg, kT=kT2,
                Araw=Araw, Am=Am, A0=A0, N=N, N_T=N_T, u=u, w=w, qs=qs)


@nki.jit
def gdn_chunk_bwd(q, k, v, g, beta, do, eye, negLstrict, Ldiag, ones_row):
    """SCAFFOLD backward. Args (fp32): q,k,v [S,D] (q,k L2-normed, unscaled);
    g,beta [S,1]; do [S,D]; const tiles. Returns dq,dk,dv [S,D], dg,dbeta [S,1].

    PASS 1 + the reverse dS recurrence are wired; the chunk-parallel finalize is
    the remaining TODO (see bottom). Do not use for training yet — use the
    shipping GDNChunkedNKI path (NKI fwd + explicit torch bwd)."""
    S, D = q.shape
    BT = eye.shape[0]; NT = S // BT
    scale = 1.0 / (D ** 0.5)
    m = 0; _bt = BT
    while _bt > 1:
        _bt //= 2; m += 1

    dq_out = nl.ndarray((S, D), dtype=nl.float32, buffer=nl.shared_hbm)
    dk_out = nl.ndarray((S, D), dtype=nl.float32, buffer=nl.shared_hbm)
    dv_out = nl.ndarray((S, D), dtype=nl.float32, buffer=nl.shared_hbm)
    dg_out = nl.ndarray((S, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    dbeta_out = nl.ndarray((S, 1), dtype=nl.float32, buffer=nl.shared_hbm)

    I = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=I, src=eye[0:BT, 0:BT])
    nL = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=nL, src=negLstrict[0:BT, 0:BT])
    Ld = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=Ld, src=Ldiag[0:BT, 0:BT])
    LdT = _T(Ld, BT, BT)
    onesBT = nl.ndarray((1, BT), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=onesBT, src=ones_row[0:1, 0:BT])
    onesD = nl.ndarray((1, D), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=onesD, src=ones_row[0:1, 0:D])

    # ---- PASS 1: forward replay, store entering state per chunk ----
    S_all = nl.ndarray((NT, D, D), dtype=nl.float32, buffer=nl.sbuf)
    St = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.sbuf); nisa.memset(dst=St, value=0.0)
    for i in nl.sequential_range(NT):
        s0 = i * BT
        S_all[i] = nl.copy(St, dtype=nl.float32)
        k_td = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=k_td, src=k[s0:s0 + BT, 0:D])
        v_c = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=v_c, src=v[s0:s0 + BT, 0:D])
        q_td = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=q_td, src=q[s0:s0 + BT, 0:D])
        g_col = nl.ndarray((BT, 1), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=g_col, src=g[s0:s0 + BT, 0:1])
        beta_col = nl.ndarray((BT, 1), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=beta_col, src=beta[s0:s0 + BT, 0:1])
        c = _recompute_chunk(k_td, v_c, q_td, g_col, beta_col, I, nL, Ld, LdT, BT, D, m, scale)
        wT = _T(c["w"], D, BT); wS = _mm(wT, St, BT, D)
        v_new = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_tensor(dst=v_new, data1=c["u"], data2=wS, op=nl.subtract)
        gcT = _T(c["gc"], 1, BT); glast11 = gcT[0:1, BT - 1:BT]
        glastBT = _mm(onesBT, glast11, BT, 1); glastD = _mm(onesD, glast11, D, 1)
        e_stateD = nl.ndarray((D, 1), dtype=nl.float32, buffer=nl.sbuf); nisa.activation(dst=e_stateD, op=nl.exp, data=glastD, bias=None, scale=1.0)
        grel = nl.ndarray((BT, 1), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_tensor(dst=grel, data1=glastBT, data2=c["gc"], op=nl.subtract)
        e_rel = nl.ndarray((BT, 1), dtype=nl.float32, buffer=nl.sbuf); nisa.activation(dst=e_rel, op=nl.exp, data=grel, bias=None, scale=1.0)
        kdec = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_scalar(dst=kdec, data=k_td, op0=nl.multiply, operand0=e_rel, engine=nisa.vector_engine)
        Supd = _mm(kdec, v_new, D, D)
        S_dec = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_scalar(dst=S_dec, data=St, op0=nl.multiply, operand0=e_stateD, engine=nisa.vector_engine)
        St_n = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_tensor(dst=St_n, data1=S_dec, data2=Supd, op=nl.add); St = St_n

    # ---- PASS 2: reverse scan (dS recurrence + intra-chunk VJP) ----
    dS = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.sbuf); nisa.memset(dst=dS, value=0.0)
    for i_rev in nl.sequential_range(NT):
        i = NT - 1 - i_rev
        s0 = i * BT
        S_i = S_all[i]
        k_td = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=k_td, src=k[s0:s0 + BT, 0:D])
        v_c = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=v_c, src=v[s0:s0 + BT, 0:D])
        q_td = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=q_td, src=q[s0:s0 + BT, 0:D])
        g_col = nl.ndarray((BT, 1), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=g_col, src=g[s0:s0 + BT, 0:1])
        beta_col = nl.ndarray((BT, 1), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=beta_col, src=beta[s0:s0 + BT, 0:1])
        do_i = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=do_i, src=do[s0:s0 + BT, 0:D])
        c = _recompute_chunk(k_td, v_c, q_td, g_col, beta_col, I, nL, Ld, LdT, BT, D, m, scale)
        gc, egc, dmask, w, u, qs, kT = c["gc"], c["egc"], c["dmask"], c["w"], c["u"], c["qs"], c["kT"]

        wT = _T(w, D, BT); wS = _mm(wT, S_i, BT, D)
        v_new = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_tensor(dst=v_new, data1=u, data2=wS, op=nl.subtract)

        gcT = _T(gc, 1, BT); glast11 = gcT[0:1, BT - 1:BT]
        glastBT = _mm(onesBT, glast11, BT, 1); glastD = _mm(onesD, glast11, D, 1)
        e_stateD = nl.ndarray((D, 1), dtype=nl.float32, buffer=nl.sbuf); nisa.activation(dst=e_stateD, op=nl.exp, data=glastD, bias=None, scale=1.0)
        grel = nl.ndarray((BT, 1), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_tensor(dst=grel, data1=glastBT, data2=gc, op=nl.subtract)
        e_rel = nl.ndarray((BT, 1), dtype=nl.float32, buffer=nl.sbuf); nisa.activation(dst=e_rel, op=nl.exp, data=grel, bias=None, scale=1.0)
        kdecay = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_scalar(dst=kdecay, data=k_td, op0=nl.multiply, operand0=e_rel, engine=nisa.vector_engine)

        qsT = _T(qs, D, BT); aqk = _mm(qsT, kT, BT, BT)
        attn_i = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_tensor(dst=attn_i, data1=aqk, data2=dmask, op=nl.multiply)
        qg = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_scalar(dst=qg, data=qs, op0=nl.multiply, operand0=egc, engine=nisa.vector_engine)

        # dvnew = attn_i^T@do + kdecay@dS
        t1 = _mm(attn_i, do_i, BT, D)                      # attn_i^T@do (contract BT)
        kdecayT = _T(kdecay, D, BT); t2 = _mm(kdecayT, dS, BT, D)   # kdecay@dS (contract D)
        dvnew = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_tensor(dst=dvnew, data1=t1, data2=t2, op=nl.add)
        dvprime = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_scalar(dst=dvprime, data=dvnew, op0=nl.multiply, operand0=-1.0)

        # roll dS = qg^T@do + e_state*dS + w^T@dvprime
        t_qgdo = _mm(qg, do_i, D, D); t_wdv = _mm(w, dvprime, D, D)
        dS_dec = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_scalar(dst=dS_dec, data=dS, op0=nl.multiply, operand0=e_stateD, engine=nisa.vector_engine)
        dS_a = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_tensor(dst=dS_a, data1=t_qgdo, data2=dS_dec, op=nl.add)
        dS_n = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_tensor(dst=dS_n, data1=dS_a, data2=t_wdv, op=nl.add); dS = dS_n

        # ============================ TODO (finalize) ============================
        # The following chunk-parallel steps from gdn_core_backward remain to be
        # wired (each is a direct matmul/tensor_scalar transcription; validate
        # incrementally via nki.simulate vs gdn_core_backward):
        #   d_attn = do@vnew^T ; dqg = do@S_i^T ; dkdecay = vnew@dS_old^T
        #   dw = dvprime@S_i^T ; du = dvnew
        #   step7:  draw_attn = d_attn*dmask ; dq += draw_attn@k ; dk += draw_attn^T@qs ; dDmask += d_attn*(qs@k^T)
        #   step8:  dq += dqg*egc ; dgc += sum(dqg*q)*egc
        #   step9:  dk += dkdecay*e_rel ; de=sum(dkdecay*k) ; dglast+=sum(de*e_rel); dgc += -de*e_rel
        #   step11: dTinv = du@vb^T + dw@kg^T ; dvb=N^T@du ; dkg=N^T@dw
        #   step12: dA0 = N^T @ dTinv @ N^T                      # matrix-inverse adjoint
        #   step13: dAraw = (-dA0)*strict_lower
        #   step14: draw_kkT=dAraw*dmask ; dDmask += dAraw*(kb@k^T) ; dkb=draw_kkT@k ; dk += draw_kkT^T@kb
        #   step15: dkb += dkg*egc ; dgc += sum(dkg*kb)*egc
        #   step16: term=(dDmask*dmask) tril ; dgc += sum_j term - sum_c term
        #   step17: dk += dkb*beta ; dbeta += sum(dkb*k); dv += dvb*beta ; dbeta += sum(dvb*v)
        #   step18: dg = reverse_cumsum(dgc) = Ld @ dgc  (contract via matmul)
        #   step19: dq *= scale
        # =========================================================================
        # Placeholder stores (NOT correct grads — scaffold only):
        nisa.dma_copy(dst=dq_out[s0:s0 + BT, 0:D], src=qg)
        nisa.dma_copy(dst=dk_out[s0:s0 + BT, 0:D], src=kdecay)
        nisa.dma_copy(dst=dv_out[s0:s0 + BT, 0:D], src=dvnew)
        nisa.dma_copy(dst=dg_out[s0:s0 + BT, 0:1], src=gc)
        nisa.dma_copy(dst=dbeta_out[s0:s0 + BT, 0:1], src=beta_col)

    return dq_out, dk_out, dv_out, dg_out, dbeta_out
