"""NKI kernel for BigVGAN's SnakeBeta activation on the NeuronCore.

SnakeBeta (BigVGAN's signature periodic activation, per-channel):

    out = x + (1 / (beta + eps)) * sin(alpha * x) ** 2

with log-scale params (BigVGAN config `snake_logscale: true`): alpha = exp(alpha_param),
beta = exp(beta_param). Effective per-channel scales are precomputed on the host:

    alpha_eff = exp(alpha_param)                 # [C, 1]
    inv_beta  = 1 / (exp(beta_param) + eps)      # [C, 1]
    out[c, t] = x[c, t] + inv_beta[c] * sin(alpha_eff[c] * x[c, t]) ** 2

Two device facts this kernel is built around (found empirically on NKI 0.5.0 / trn2):
  1. Per-channel scales use ``nisa.tensor_scalar`` with a per-partition ``[P,1]`` operand
     (this works; a per-partition ``activation(scale=)`` gives wrong results on device).
  2. ``nl.sin`` is only accurate for |input| <~ 4.18 (±~1.33π) and DIVERGES beyond it
     (max err ~780 at |x|=20). Real ``alpha*x`` reaches ~91 in the vocoder, so we
     RANGE-REDUCE into [-π, π] first: ``r = x - 2π * round(x / 2π)``. There is no floor/mod
     ISA op, but a float→int32→float ``tensor_copy`` rounds (round-half-to-even), which
     gives ``round(x/2π)`` directly. Validated to max_diff ~2.6e-5 over [-500, 500].

Layout: channels C on the partition dim (tiled by 128), time T on the free dim (tiled).
"""
from __future__ import annotations

import math

import nki
import nki.language as nl
import nki.isa as nisa

_INV_2PI = 1.0 / (2.0 * math.pi)
_TWO_PI = 2.0 * math.pi


@nki.jit
def snakebeta_kernel(x, alpha_eff, inv_beta):
    """out = x + inv_beta * sin(alpha_eff * x)**2, per-channel (channels on partition dim).

    Args:
        x:         [C, T] fp32 input (batch already folded/1).
        alpha_eff: [C, 1] fp32, effective alpha (exp(alpha_param) for log-scale).
        inv_beta:  [C, 1] fp32, 1 / (exp(beta_param) + eps).

    Returns:
        [C, T] fp32 output.
    """
    C, T = x.shape
    out = nl.ndarray((C, T), dtype=x.dtype, buffer=nl.shared_hbm)

    TILE_P = 128     # partition dim max
    TILE_F = 4096    # free-dim tile

    n_p = (C + TILE_P - 1) // TILE_P
    n_f = (T + TILE_F - 1) // TILE_F

    for pi in nl.affine_range(n_p):
        p0 = pi * TILE_P
        psz = min(TILE_P, C - p0)

        a_t = nl.ndarray((psz, 1), dtype=nl.float32, buffer=nl.sbuf)
        b_t = nl.ndarray((psz, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=a_t, src=alpha_eff[p0:p0 + psz, 0:1])
        nisa.dma_copy(dst=b_t, src=inv_beta[p0:p0 + psz, 0:1])

        for fi in nl.affine_range(n_f):
            f0 = fi * TILE_F
            fsz = min(TILE_F, T - f0)

            xt = nl.ndarray((psz, fsz), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=xt, src=x[p0:p0 + psz, f0:f0 + fsz])

            # ax = alpha_eff * x   (per-partition scalar multiply)
            ax = nl.ndarray((psz, fsz), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=ax, data=xt, op0=nl.multiply, operand0=a_t)

            # range-reduce ax into [-pi, pi): r = ax - 2pi * round(ax / 2pi)
            q = nl.ndarray((psz, fsz), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(dst=q, data=ax, op0=nl.multiply, operand0=_INV_2PI)
            qi = nl.ndarray((psz, fsz), dtype=nl.int32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=qi, src=q)          # float->int32 rounds (round-half-even)
            qf = nl.ndarray((psz, fsz), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=qf, src=qi)
            nisa.tensor_scalar(dst=qf, data=qf, op0=nl.multiply, operand0=_TWO_PI)
            r = nl.ndarray((psz, fsz), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=r, data1=ax, data2=qf, op=nl.subtract)

            # s = sin(r) = sin(alpha*x)   (r now in nl.sin's accurate range)
            s = nl.ndarray((psz, fsz), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=s, data=r, op=nl.sin)

            # s2 = inv_beta * sin**2
            s2 = nl.ndarray((psz, fsz), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=s2, data1=s, data2=s, op=nl.multiply)
            nisa.tensor_scalar(dst=s2, data=s2, op0=nl.multiply, operand0=b_t)

            # out = x + inv_beta * sin(alpha*x)**2
            ot = nl.ndarray((psz, fsz), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=ot, data1=xt, data2=s2, op=nl.add)

            nisa.dma_copy(dst=out[p0:p0 + psz, f0:f0 + fsz], src=ot)

    return out
