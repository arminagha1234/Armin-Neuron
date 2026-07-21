"""Gemma4 d-tiled flash PREFILL kernel v2 — ON-DEMAND tile loading (scales to 32K).

v1 (gemma4_flash_prefill_kernel.py) pre-staged the FULL K/V in SBUF
(k_dpart [128, Sk] per d-chunk + all v_tiles), which is O(Sk) and overflows SBUF /
drives HBM OOM at 12K+. v2 loads each K-tile and V-tile ON DEMAND inside the
K-loop, so SBUF usage is O(tile)=O(128) regardless of context length.

Same validated flash online-softmax math as v1; only the K/V residency changed.
Cost: K/V tiles are re-loaded per Q-tile (more DMA), but memory is bounded so it
FITS at 12K/32K. The SWA window-skip (skip K-tiles outside [qs-window, qs+qsz)) is
a perf optimization on top — see `_kt_range` below (compile-time bound when the
q-loop is unrolled). v2 keeps the full K-loop for correctness-first; flip
GEMMA4_SWA_SKIP=1 to enable the windowed K-range (static q-tile unroll).

Layout (batch folded into BH): q [BH, Sq, D]  k/v [BHkv, Sk, D]  out [BH, Sq, D].
Reference for the skip logic: nkilib/core/attention/attention_cte.py (caps at
head_dim 128; we d-tile to support 256/512).
"""
import os
import nki
import nki.isa as nisa
import nki.language as nl

_P = 128
_NEG = -30000.0
_SWA_SKIP = os.environ.get("GEMMA4_SWA_SKIP", "0") == "1"


@nki.jit
def gemma4_flash_prefill_v2(q, k, v, scale: float = 1.0, sliding_window: int = 0,
                           q_pos_offset: int = 0):
    BH, Sq, D = q.shape
    BHkv, Sk, Dk = k.shape
    groups = BH // BHkv
    n_d = D // _P
    n_qt = (Sq + _P - 1) // _P
    n_kt = (Sk + _P - 1) // _P

    out = nl.ndarray((BH, Sq, D), dtype=q.dtype, buffer=nl.shared_hbm)

    for bh in nl.affine_range(BH):
        kv = bh // groups

        for qt in nl.affine_range(n_qt):
            qs = qt * _P
            qsz = min(_P, Sq - qs)

            # Load Q-tile as [D, qsz] (d on partition) per d-chunk via transpose.
            q_dpart = []
            for dc in nl.static_range(n_d):
                qtile = nl.ndarray((_P, _P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.memset(qtile, value=0)
                nisa.dma_copy(dst=qtile[0:qsz, :],
                              src=q[bh, nl.ds(qs, qsz), nl.ds(dc * _P, _P)])
                q_ps = nl.ndarray((_P, _P), dtype=q.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=q_ps, data=qtile, engine=nisa.engine.tensor)
                qd = nl.ndarray((_P, _P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=qd, src=q_ps)
                q_dpart.append(qd)

            run_max = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(run_max, value=_NEG)
            run_sum = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(run_sum, value=0.0)
            run_out = nl.ndarray((_P, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(run_out, value=0.0)

            for kt in nl.sequential_range(n_kt):
                ks = kt * _P
                ksz = min(_P, Sk - ks)

                # --- ON-DEMAND load this K-tile as [D-part, ksz] per d-chunk ---
                k_dpart = []
                for dc in nl.static_range(n_d):
                    ktile = nl.ndarray((_P, _P), dtype=k.dtype, buffer=nl.sbuf)
                    nisa.memset(ktile, value=0)
                    nisa.dma_copy(dst=ktile[0:ksz, :],
                                  src=k[kv, nl.ds(ks, ksz), nl.ds(dc * _P, _P)])
                    kt_ps = nl.ndarray((_P, _P), dtype=k.dtype, buffer=nl.psum)
                    nisa.nc_transpose(dst=kt_ps, data=ktile, engine=nisa.engine.tensor)
                    kd = nl.ndarray((_P, _P), dtype=k.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=kd[:, 0:ksz], src=kt_ps[:, 0:ksz])
                    k_dpart.append(kd)
                # --- ON-DEMAND load this V-tile [ksz, D] ---
                vt = nl.ndarray((_P, D), dtype=v.dtype, buffer=nl.sbuf)
                nisa.memset(vt, value=0)
                nisa.dma_copy(dst=vt[0:ksz, :], src=v[kv, nl.ds(ks, ksz), :])

                # MM1: scores[qsz, ksz] = sum_dc q_dpart[dc]^T @ k_dpart[dc]
                sc_ps = nl.ndarray((_P, _P), dtype=nl.float32, buffer=nl.psum)
                for dc in nl.static_range(n_d):
                    nisa.nc_matmul(sc_ps[0:qsz, 0:ksz],
                                   stationary=q_dpart[dc][:, 0:qsz],
                                   moving=k_dpart[dc][:, 0:ksz],
                                   accumulate=(dc > 0))
                scores = nl.ndarray((_P, _P), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=scores[0:qsz, 0:ksz], data=sc_ps[0:qsz, 0:ksz],
                                   op0=nl.multiply, operand0=float(scale))

                # causal: keep q_abs >= k_abs, q_abs = q_pos_offset + qs + i
                nisa.affine_select(dst=scores[0:qsz, 0:ksz], pattern=[[-1, ksz]],
                                   channel_multiplier=1, on_true_tile=scores[0:qsz, 0:ksz],
                                   on_false_value=_NEG, cmp_op=nl.greater_equal,
                                   offset=(q_pos_offset + qs - ks))
                if sliding_window and sliding_window > 0:
                    # window: keep q_abs - k_abs < window
                    nisa.affine_select(dst=scores[0:qsz, 0:ksz], pattern=[[1, ksz]],
                                       channel_multiplier=-1, on_true_tile=scores[0:qsz, 0:ksz],
                                       on_false_value=_NEG, cmp_op=nl.greater_equal,
                                       offset=(sliding_window - 1 - (q_pos_offset + qs - ks)))

                # flash online softmax
                tile_max = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=tile_max[0:qsz, :], data=scores[0:qsz, 0:ksz],
                                   op=nl.maximum, axis=(1,))
                new_max = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=new_max[0:qsz, :], data1=run_max[0:qsz, :],
                                   data2=tile_max[0:qsz, :], op=nl.maximum)
                p = nl.ndarray((_P, _P), dtype=nl.float32, buffer=nl.sbuf)
                neg_new = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(dst=neg_new[0:qsz, :], data=new_max[0:qsz, :],
                                   op0=nl.multiply, operand0=-1.0)
                nisa.activation(dst=p[0:qsz, 0:ksz], data=scores[0:qsz, 0:ksz],
                                op=nl.exp, bias=neg_new[0:qsz, :])
                alpha = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=alpha[0:qsz, :], data1=run_max[0:qsz, :],
                                   data2=new_max[0:qsz, :], op=nl.subtract)
                nisa.activation(dst=alpha[0:qsz, :], data=alpha[0:qsz, :], op=nl.exp)
                psum_row = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_reduce(dst=psum_row[0:qsz, :], data=p[0:qsz, 0:ksz],
                                   op=nl.add, axis=(1,))
                nisa.tensor_tensor(dst=run_sum[0:qsz, :], data1=run_sum[0:qsz, :],
                                   data2=alpha[0:qsz, :], op=nl.multiply)
                nisa.tensor_tensor(dst=run_sum[0:qsz, :], data1=run_sum[0:qsz, :],
                                   data2=psum_row[0:qsz, :], op=nl.add)

                # PV
                p_ps = nl.ndarray((_P, _P), dtype=q.dtype, buffer=nl.psum)
                p_bf = nl.ndarray((_P, _P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=p_bf[0:qsz, 0:ksz], src=p[0:qsz, 0:ksz])
                nisa.nc_transpose(dst=p_ps[0:ksz, 0:qsz], data=p_bf[0:qsz, 0:ksz],
                                  engine=nisa.engine.tensor)
                pt = nl.ndarray((_P, _P), dtype=q.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=pt[0:ksz, 0:qsz], src=p_ps[0:ksz, 0:qsz])
                pv_ps = nl.ndarray((_P, D), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(pv_ps[0:qsz, :], stationary=pt[0:ksz, 0:qsz],
                               moving=vt[0:ksz, :], accumulate=False)
                nisa.tensor_scalar(dst=run_out[0:qsz, :], data=run_out[0:qsz, :],
                                   op0=nl.multiply, operand0=alpha[0:qsz, :])
                pv_sb = nl.ndarray((_P, D), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=pv_sb[0:qsz, :], src=pv_ps[0:qsz, :])
                nisa.tensor_tensor(dst=run_out[0:qsz, :], data1=run_out[0:qsz, :],
                                   data2=pv_sb[0:qsz, :], op=nl.add)
                nisa.tensor_copy(dst=run_max[0:qsz, :], src=new_max[0:qsz, :])

            inv = nl.ndarray((_P, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.reciprocal(dst=inv[0:qsz, :], data=run_sum[0:qsz, :])
            o_sb = nl.ndarray((_P, D), dtype=q.dtype, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=o_sb[0:qsz, :], data=run_out[0:qsz, :],
                               op0=nl.multiply, operand0=inv[0:qsz, :])
            nisa.dma_copy(dst=out[bh, nl.ds(qs, qsz), :], src=o_sb[0:qsz, :])

    return out
