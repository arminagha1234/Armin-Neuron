"""
GatedDeltaNet (Qwen3.5-4B) chunked FORWARD — NKI kernel for Trainium2.

Scalar-per-head-decay specialization of the KDA chunked forward
(kda_chunk_kernel.kda_chunk_fwd), matching the Rank-1 oracle
`chunked_gdn.chunked_gdn_forward` EXACTLY:
  * decay is SCALAR per token (g[BT], broadcast over head-dim), not per-channel
  * BOUNDED decay mask  decay_mask[c,j] = exp(min(gc[c]-gc[j], 0)) for c>=j
    (NOT the split exp(gc)/exp(-gc) factorization — that overflows fp32 at BT=128
    with strong decay; this is the full-32L NaN root cause)
  * intra-chunk (I-A)^-1 via exact Neumann DOUBLING (nilpotent strictly-lower A)
  * scale = 1/sqrt(K) applied to q INSIDE (caller passes L2-normed q,k)
  * ALL internal compute in fp32; caller may pass/collect bf16 at the boundary.

Per (batch, head). Layout: token-major inputs [S, D] (S = NT*BT, D=head_dim=128).
nc_matmul contracts the PARTITION dim: result[M,N] = sum_P stat[P,M]*mov[P,N],
so C = X @ Y (contract inner) => stationary = X^T, moving = Y.
"""
from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl


def _transpose_sb(data, P, F):
    """Transpose SBUF tile (P0,F0) -> (P,F) via nc_transpose -> PSUM -> SBUF."""
    ps = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_transpose(dst=ps, data=data)
    sb = nl.ndarray((P, F), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=sb, src=ps)
    return sb


def _neumann_doubling(M, I, niter, BT):
    """Return sum_j M^j = (I - M)^-1 for a NILPOTENT M, via `niter` doubling
    iterations in the additive/product form  (I+M)(I+M^2)(I+M^4)...(I+M^(2^niter)).
    Exact when M^(2^(niter+1)) = 0. All fp32. The HIGHEST power formed is
    M^(2^niter): callers must size `niter` so this power stays well inside fp32
    range (that's the whole point of the block-LU split below -- the full-chunk
    doubling formed A^32/A^64 and overflowed, causing the full-32L NaN)."""
    acc = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=acc, data1=I, data2=M, op=nl.add)      # I + M
    Apow = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=Apow, src=M)
    for _j in nl.static_range(niter):
        # Apow <- Apow @ Apow   (nc_matmul: stat^T @ mov => stat=Apow^T)
        ApowT = _transpose_sb(Apow, BT, BT)
        Apow2_ps = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=Apow2_ps, stationary=ApowT, moving=Apow)
        Apow_n = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=Apow_n, src=Apow2_ps); Apow = Apow_n
        # acc <- acc + acc @ Apow  = acc @ (I + Apow)
        accT = _transpose_sb(acc, BT, BT)
        accA_ps = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=accA_ps, stationary=accT, moving=Apow)
        accA = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=accA, src=accA_ps)
        acc_n = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=acc_n, data1=acc, data2=accA, op=nl.add); acc = acc_n
    return acc


@nki.jit
def gdn_chunk_fwd(q, k, v, g, beta, eye, negLstrict, Ldiag, ones_row, bdiag):
    """Chunked GDN forward for ONE (batch, head), scalar-per-head decay.

    Args (all fp32):
        q, k, v : [S, D]  token-major. q is L2-normed (NOT yet scaled); k L2-normed.
        g       : [S, 1]  per-token log-decay (<= 0), scalar per head.
        beta    : [S, 1]  per-token write gate (already sigmoid'd).
        eye        : [BT, BT] identity
        negLstrict : [BT, BT] -1 on strictly-lower (c>i), else 0  (for A0 sign/mask)
        Ldiag      : [BT, BT] 1 on lower+diag (c>=j), else 0
        ones_row   : [1, D]  all ones (D>=BT); for broadcasting a scalar over partitions
        bdiag      : [BT, BT] 1 on the BLK x BLK (BLK=16) block-diagonal, else 0.
                     Selects the block-diagonal of A0 for the overflow-safe
                     block-LU inverse (replaces the full-chunk Neumann doubling
                     that formed A^32/A^64 and overflowed fp32 -> full-32L NaN).
    Returns:
        o [S, D], final_state [D, D] (rows=K, cols=V)
    """
    S, D = q.shape
    BT = eye.shape[0]
    NT = S // BT
    scale = 1.0 / (D ** 0.5)
    # log2(BT) as plain arithmetic (BT is a power-of-two compile-time constant);
    # int.bit_length() is not resolvable by the on-device NKI tracer.
    m = 0
    _bt = BT
    while _bt > 1:
        _bt //= 2
        m += 1
    # ---- block-LU sizing (overflow-safe inverse) ----
    # Partition the chunk into BLK x BLK blocks (BLK=16). The block-diagonal
    # inverse needs doubling up to A_D^(BLK/2) (BLK=16 -> A^8, finite); the
    # block-off-diagonal solve needs doubling up to B^(NB/2) where NB=BT/BLK.
    # NEITHER forms A^32/A^64, so no fp32 overflow.
    BLK = 16
    mblk = 0
    _b = BLK
    while _b > 1:
        _b //= 2
        mblk += 1              # log2(BLK) = 4
    NB = BT // BLK
    mnb = 0
    _n = NB
    while _n > 1:
        _n //= 2
        mnb += 1              # log2(NB)
    niter_D = mblk - 1 if mblk >= 1 else 0   # highest power A_D^(2^niter_D)=A_D^8
    niter_B = mnb - 1 if mnb >= 1 else 0     # highest power B^(2^niter_B)

    o_out = nl.ndarray((S, D), dtype=nl.float32, buffer=nl.shared_hbm)
    fs_out = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.shared_hbm)

    I = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=I, src=eye[0:BT, 0:BT])
    nL = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=nL, src=negLstrict[0:BT, 0:BT])
    Ld = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=Ld, src=Ldiag[0:BT, 0:BT])
    Bd = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=Bd, src=bdiag[0:BT, 0:BT])  # block-diag (BLK) mask
    # lower-triangular ones (incl diag) for the cumsum matmul: gc = tril_ones @ g
    # cumsum along tokens: gc[c] = sum_{j<=c} g[j].  As matmul over partition:
    #   gc[c] = sum_j Ldiag[c,j] * g[j]  => stationary = Ldiag^T (=triu), moving = g
    LdT = _transpose_sb(Ld, BT, BT)   # upper-incl-diag: LdT[j,c]=1 iff c>=j

    St = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.sbuf); nisa.memset(dst=St, value=0.0)

    for i in nl.sequential_range(NT):
        s0 = i * BT
        # ---- load chunk token-major [BT,D] / [BT,1] ----
        q_td = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=q_td, src=q[s0:s0 + BT, 0:D])
        k_td = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=k_td, src=k[s0:s0 + BT, 0:D])
        v_c = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=v_c, src=v[s0:s0 + BT, 0:D])
        g_col = nl.ndarray((BT, 1), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=g_col, src=g[s0:s0 + BT, 0:1])
        beta_col = nl.ndarray((BT, 1), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=beta_col, src=beta[s0:s0 + BT, 0:1])

        # ---- cumulative decay gc[BT,1] = sum_{j<=c} g[j]  (matmul: LdT^T? ) ----
        # gc[c] = sum_j LdT[j,c] * g[j]  (LdT[j,c]=1 iff c>=j) -> stat=LdT, mov=g_col
        gc_ps = nl.ndarray((BT, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=gc_ps, stationary=LdT, moving=g_col)
        gc = nl.ndarray((BT, 1), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_copy(dst=gc, src=gc_ps)  # [BT,1]

        # egc[c] = exp(gc[c])  [BT,1]
        egc = nl.ndarray((BT, 1), dtype=nl.float32, buffer=nl.sbuf); nisa.activation(dst=egc, op=nl.exp, data=gc, bias=None, scale=1.0)

        # ---- bounded decay mask: dmask[c,j] = exp(min(gc[c]-gc[j],0)) on c>=j ----
        # gcrow[c,j]=gc[c] (per-partition scalar broadcast over free); gccol[c,j]=gc[j].
        gcrow = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=gcrow, value=0.0)
        nisa.tensor_scalar(dst=gcrow, data=gcrow, op0=nl.add, operand0=gc, engine=nisa.vector_engine)  # gcrow[c,j]=gc[c]
        gccol = _transpose_sb(gcrow, BT, BT)   # gccol[c,j] = gc[j]
        gdiff = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=gdiff, data1=gcrow, data2=gccol, op=nl.subtract)  # gc[c]-gc[j]
        gdiff_cl = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=gdiff_cl, data=gdiff, op0=nl.minimum, operand0=0.0)   # min(.,0) overflow-safe
        edm = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf); nisa.activation(dst=edm, op=nl.exp, data=gdiff_cl, bias=None, scale=1.0)
        dmask = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=dmask, data1=edm, data2=Ld, op=nl.multiply)       # keep c>=j

        # ---- kb = k*beta ; vb = v*beta  (token-major [BT,D]) ----
        kb = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_scalar(dst=kb, data=k_td, op0=nl.multiply, operand0=beta_col, engine=nisa.vector_engine)
        vb = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_scalar(dst=vb, data=v_c, op0=nl.multiply, operand0=beta_col, engine=nisa.vector_engine)

        # ---- Araw = kb @ k^T  [BT,BT] (contract D) => stat=kb^T? : need contract over D on partition
        # kb,k are [BT,D]; want [BT,BT]=kb@k^T contract D. Put D on partition: transpose to [D,BT].
        kbT = _transpose_sb(kb, D, BT)   # [D,BT]
        kT = _transpose_sb(k_td, D, BT)  # [D,BT]
        Araw_ps = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.psum); nisa.nc_matmul(dst=Araw_ps, stationary=kbT, moving=kT)
        Araw = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_copy(dst=Araw, src=Araw_ps)
        # A0 = -(Araw * dmask) on strictly-lower
        Am = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_tensor(dst=Am, data1=Araw, data2=dmask, op=nl.multiply)
        A0 = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_tensor(dst=A0, data1=Am, data2=nL, op=nl.multiply)  # nL = -1 on strict-lower

        # ---- N = (I-A0)^-1 via overflow-safe BLOCK-LU (full-tile matmuls) ----
        # Split A0 = A_D + A_L: A_D = BLK x BLK block-diagonal, A_L the rest.
        # (I-A0)^-1 = (I - B)^-1 (I - A_D)^-1  with  B = (I - A_D)^-1 A_L.
        #   * A_D block-diag & each block strictly-lower nilpotent(idx<=BLK) ->
        #     (I-A_D)^-1 = sum A_D^j exact after niter_D doublings (top power A_D^8).
        #   * B is block-strictly-lower (nilpotent index NB) -> (I-B)^-1 exact
        #     after niter_B doublings (top power B^(NB/2)).
        # HIGHEST matrix power formed is A^8 (never A^32/A^64): no fp32 overflow.
        A_D = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=A_D, data1=A0, data2=Bd, op=nl.multiply)      # block-diagonal part
        A_L = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=A_L, data1=A0, data2=A_D, op=nl.subtract)     # off-block-diagonal part
        # Dinv = (I - A_D)^-1
        Dinv = _neumann_doubling(A_D, I, niter_D, BT)
        # B = Dinv @ A_L   (contract inner => stat = Dinv^T)
        DinvT = _transpose_sb(Dinv, BT, BT)
        Bmat_ps = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.psum); nisa.nc_matmul(dst=Bmat_ps, stationary=DinvT, moving=A_L)
        Bmat = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_copy(dst=Bmat, src=Bmat_ps)
        # Binv = (I - B)^-1
        Binv = _neumann_doubling(Bmat, I, niter_B, BT)
        # N = Binv @ Dinv   (stat = Binv^T)
        BinvT = _transpose_sb(Binv, BT, BT)
        N_ps = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.psum); nisa.nc_matmul(dst=N_ps, stationary=BinvT, moving=Dinv)
        N = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_copy(dst=N, src=N_ps)   # [BT,BT] = Tinv

        # ---- kg = kb * egc  (row scale by egc[c])  [BT,D] ----
        kg = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_scalar(dst=kg, data=kb, op0=nl.multiply, operand0=egc, engine=nisa.vector_engine)

        # ---- u = N @ vb ; w = N @ kg  (contract BT) => stat = N^T ----
        N_T = _transpose_sb(N, BT, BT)
        u_ps = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.psum); nisa.nc_matmul(dst=u_ps, stationary=N_T, moving=vb)
        u = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_copy(dst=u, src=u_ps)
        w_ps = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.psum); nisa.nc_matmul(dst=w_ps, stationary=N_T, moving=kg)
        w = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_copy(dst=w, src=w_ps)

        # ---- scaled q ----
        qs = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_scalar(dst=qs, data=q_td, op0=nl.multiply, operand0=scale)
        qsT = _transpose_sb(qs, D, BT)   # [D,BT]

        # ---- attn_i = (qs @ k^T) * dmask  [BT,BT] ----
        aqk_ps = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.psum); nisa.nc_matmul(dst=aqk_ps, stationary=qsT, moving=kT)
        aqk = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_copy(dst=aqk, src=aqk_ps)
        attn_i = nl.ndarray((BT, BT), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_tensor(dst=attn_i, data1=aqk, data2=dmask, op=nl.multiply)

        # ---- attn_inter = (qs*egc) @ S  [BT,D] (contract D) ----
        qsg = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_scalar(dst=qsg, data=qs, op0=nl.multiply, operand0=egc, engine=nisa.vector_engine)
        qsgT = _transpose_sb(qsg, D, BT)  # [D,BT]
        ai_ps = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.psum); nisa.nc_matmul(dst=ai_ps, stationary=qsgT, moving=St)
        attn_inter = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_copy(dst=attn_inter, src=ai_ps)

        # ---- v_new = u - w@S  [BT,D] (contract D on w? w is [BT,D], S is [D,D]) ----
        wT = _transpose_sb(w, D, BT)  # [D,BT]
        wS_ps = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.psum); nisa.nc_matmul(dst=wS_ps, stationary=wT, moving=St)
        wS = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_copy(dst=wS, src=wS_ps)
        v_new = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_tensor(dst=v_new, data1=u, data2=wS, op=nl.subtract)

        # ---- out = attn_inter + attn_i @ v_new  (contract BT) => stat=attn_i^T ----
        attn_iT = _transpose_sb(attn_i, BT, BT)
        av_ps = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.psum); nisa.nc_matmul(dst=av_ps, stationary=attn_iT, moving=v_new)
        av = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_copy(dst=av, src=av_ps)
        o_chunk = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_tensor(dst=o_chunk, data1=attn_inter, data2=av, op=nl.add)
        nisa.dma_copy(dst=o_out[s0:s0 + BT, 0:D], src=o_chunk)

        # ---- state update: S = S*exp(glast) + (k*exp(glast-gc))^T @ v_new ----
        # glast = gc[BT-1] is a runtime scalar on partition BT-1; move it to
        # partition 0 as [1,1], then broadcast across partitions via a P=1 matmul.
        gcT = _transpose_sb(gc, 1, BT)                 # [1,BT] on partition 0
        glast11 = gcT[0:1, BT - 1:BT]                  # [1,1]
        onesBT = nl.ndarray((1, BT), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=onesBT, src=ones_row[0:1, 0:BT])
        onesD = nl.ndarray((1, D), dtype=nl.float32, buffer=nl.sbuf); nisa.dma_copy(dst=onesD, src=ones_row[0:1, 0:D])
        # glast broadcast to [BT,1]  (grel path) and to [D,1] (state decay path)
        glastBT_ps = nl.ndarray((BT, 1), dtype=nl.float32, buffer=nl.psum); nisa.nc_matmul(dst=glastBT_ps, stationary=onesBT, moving=glast11)
        glastBT = nl.ndarray((BT, 1), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_copy(dst=glastBT, src=glastBT_ps)
        glastD_ps = nl.ndarray((D, 1), dtype=nl.float32, buffer=nl.psum); nisa.nc_matmul(dst=glastD_ps, stationary=onesD, moving=glast11)
        e_stateD = nl.ndarray((D, 1), dtype=nl.float32, buffer=nl.sbuf); nisa.activation(dst=e_stateD, op=nl.exp, data=glastD_ps, bias=None, scale=1.0)

        grel = nl.ndarray((BT, 1), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_tensor(dst=grel, data1=glastBT, data2=gc, op=nl.subtract)  # glast - gc
        e_rel = nl.ndarray((BT, 1), dtype=nl.float32, buffer=nl.sbuf); nisa.activation(dst=e_rel, op=nl.exp, data=grel, bias=None, scale=1.0)
        kdec = nl.ndarray((BT, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_scalar(dst=kdec, data=k_td, op0=nl.multiply, operand0=e_rel, engine=nisa.vector_engine)
        # S += kdec^T @ v_new, contract token dim BT (both operands BT on partition)
        Supd_ps = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.psum); nisa.nc_matmul(dst=Supd_ps, stationary=kdec, moving=v_new)
        Supd = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_copy(dst=Supd, src=Supd_ps)
        S_dec = nl.ndarray((D, D), dtype=nl.float32, buffer=nl.sbuf); nisa.tensor_scalar(dst=S_dec, data=St, op0=nl.multiply, operand0=e_stateD, engine=nisa.vector_engine)
        nisa.tensor_tensor(dst=St, data1=S_dec, data2=Supd, op=nl.add)

    nisa.dma_copy(dst=fs_out[0:D, 0:D], src=St)
    return o_out, fs_out
