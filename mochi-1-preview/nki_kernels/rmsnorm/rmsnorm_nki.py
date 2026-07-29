"""Fused modulated RMSNorm NKI kernels for the Mochi-1 transformer (trn2).

These kernels replace Mochi's memory-heavy modulated RMS norms. Upstream casts
the whole ``(B, S, 3072)`` tensor to fp32 up to four times per block across 48
blocks; the Python fix (``src/mochi_norm_memory.py``) tiles the sequence axis so
only one fp32 tile is live at a time. A NKI kernel goes further: it fuses the
square -> mean -> rsqrt -> normalize -> scale -> cast chain into one pass over
each 128-row tile, keeping every intermediate in SBUF and casting back to bf16
before the single HBM write. No full-sequence fp32 copy is ever materialised.

--------------------------------------------------------------------------------
Operation (matches ``rmsnorm_ref`` / ``_rms_normalize_tiled`` EXACTLY)
--------------------------------------------------------------------------------
    x_norm = x * rsqrt(mean(x^2, axis=-1) + eps)          # fp32 internal
    out    = x_norm * scale                               # scale optional

``modulated_rmsnorm`` covers norm2/3/4 and their ``_context`` variants. ``scale``
is optional and, when present, is either:
  * ``(B, 1, D)`` -- broadcast over the sequence axis, or
  * ``(B, S, D)`` -- per sequence position.
The broadcast form is resolved from ``scale.shape[1]`` at trace time.

``rmsnorm_zero_core`` covers the norm1 (RMSNormZero) fused core:
    out = rmsnorm(x) * (1 + scale_msa[:, None])
The ``linear``/``silu``/``chunk`` that produce ``scale_msa`` stay in PyTorch; this
kernel takes the already-chunked ``scale_msa`` of shape ``(B, D)`` and folds the
``(1 + .)`` into the scale multiply. The gate/scale_mlp/gate_mlp outputs of
RMSNormZero are passed through unchanged in PyTorch around this kernel.

--------------------------------------------------------------------------------
Shapes / dtypes
--------------------------------------------------------------------------------
    hidden : (B, S, D)   bf16   B in {1, 2} (CFG), S up to ~9796,
                                D = 3072 (visual) or 1536 (context)
    scale  : (B, 1, D) or (B, S, D), bf16, or None       (modulated_rmsnorm)
    scale_msa : (B, D)  bf16                              (rmsnorm_zero_core)
    out    : (B, S, D)   bf16   (same dtype as hidden)

Internal reduction and all arithmetic run in fp32; the output is cast back to the
input dtype -- numerically identical to upstream, not an approximation.

--------------------------------------------------------------------------------
Hardware constraints / mapping (trn2 NeuronCore)
--------------------------------------------------------------------------------
* Partition dim is capped at 128. The reduction is over D (the last axis), so we
  put S on the partition dim (<=128 rows/tile) and D on the free dim, then reduce
  along the free dim per partition (``nl.sum(axis=1)``) -- the classic
  Vector-Engine reduction pattern.
* D = 3072 (or 1536) fits in one SBUF free dim (limit 32767 elems), so no D-axis
  tiling is needed; each row is normalised in a single reduction.
* Batches are host-unrolled so a ``(B, 1, D)`` scale row maps to exactly one batch
  without straddling the S/partition tiling.
* Sequence is tiled in blocks of 128 partitions. The final partial tile is handled
  by min()-sizing the tile (``ap = min(128, S - m)``) so every load/store stays
  in-bounds -- the CLAUDE.md-endorsed boundary-clamp alternative to masking.
  Tiles are addressed with plain ``nl.ds``-style slices (nki 0.5.0 has no
  ``nl.mgrid``).

--------------------------------------------------------------------------------
API / VALIDATION STATUS
--------------------------------------------------------------------------------
Uses the nki 0.5.0 API (``import nki`` / ``nl.load`` / ``nl.store`` / ``nl.sum``
/ ``nl.rsqrt`` + ``nisa`` ops, slice-based tile addressing -- no ``nl.mgrid``).
The math is validated on CPU via ``rmsnorm_ref``; on-device validation on the
Beta-3 DLC (trn2, nki 0.5.0) is pending a free Neuron core. See
``test_rmsnorm.py``.
"""
from __future__ import annotations

# nki 0.5.0 on the Beta-3 DLC: the `neuronxcc.nki` entrypoint routes @nki.jit
# through torch_neuronx.pyhlo (absent in the DLC container), so import the plain
# `nki` namespace, which compiles and runs on-device. Verified against the
# flash-attention kernel's on-device reconciliation.
import nki
import nki.language as nl
import nki.isa as nisa

# trn2 hardware partition dimension (SBUF first-dim cap).
_PMAX = 128


def _normalize_tile(x_dt, D: int, eps: float):
    """Core RMSNorm on one loaded tile ``x_dt`` (P, D); returns fp32 ``x_norm``.

    Upcasts to fp32, computes ``x * rsqrt(mean(x^2, -1) + eps)`` with the mean as
    a free-dim reduction, and broadcasts the (P, 1) inverse-RMS over the free dim.
    Plain python helper (no ``@nki.jit``) so it inlines during tracing.
    """
    P = x_dt.shape[0]

    # Upcast bf16 -> fp32 for the reduction and all arithmetic.
    xf = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=xf, src=x_dt)

    # sum(x^2) over the free (D) axis -> (P, 1).
    sq = nl.ndarray((P, D), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sq, data1=xf, data2=xf, op=nl.multiply)
    ss = nl.sum(sq, axis=1, keepdims=True)  # (P, 1) fp32

    # mean + eps, then rsqrt -> (P, 1). Fused multiply+add via tensor_scalar.
    stat = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=stat, data=ss,
        op0=nl.multiply, operand0=1.0 / float(D),
        op1=nl.add, operand1=eps,
    )
    inv = nl.rsqrt(stat)  # (P, 1)

    # x_norm = xf * inv, broadcasting inv over the free dim.
    return nl.multiply(xf, inv)  # (P, D) fp32


@nki.jit
def modulated_rmsnorm(hidden, scale=None, eps: float = 1e-6):
    """Fused modulated RMSNorm over the last axis.

    Computes ``x * rsqrt(mean(x^2, -1) + eps)`` in fp32, applies the optional
    ``scale``, and casts back to ``hidden``'s dtype -- one pass per tile.

    Args:
        hidden: HBM tensor ``(B, S, D)``, bf16 (or any float). Reduction axis is
            ``D`` (last).
        scale: optional HBM tensor. Either ``(B, 1, D)`` (broadcast over S) or
            ``(B, S, D)`` (per position). ``None`` skips the multiply.
        eps: numerical-stability epsilon (Mochi uses ~1e-6).

    Returns:
        HBM tensor ``(B, S, D)`` with the same dtype as ``hidden``.
    """
    B, S, D = hidden.shape
    dt = hidden.dtype
    out = nl.ndarray((B, S, D), dtype=dt, buffer=nl.shared_hbm)

    has_scale = scale is not None
    # Broadcast form resolved at trace time from the concrete scale shape.
    scale_per_position = has_scale and scale.shape[1] == S and S != 1

    for b in range(B):
        # (B, 1, D) broadcast scale: load the single row once per batch as fp32.
        if has_scale and not scale_per_position:
            s_row = nl.ndarray((1, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=s_row, src=nl.load(scale[b, 0:1, 0:D]))

        for m in range(0, S, _PMAX):
            ap = min(_PMAX, S - m)  # actual partitions in this (possibly partial) tile
            # nki 0.5.0 has no nl.mgrid; address the tile with plain slices.
            x_dt = nl.load(hidden[b, m : m + ap, 0:D])       # (ap, D) input dtype
            x_norm = _normalize_tile(x_dt, D, eps)           # (ap, D) fp32

            if has_scale:
                if scale_per_position:
                    s = nl.ndarray((ap, D), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=s, src=nl.load(scale[b, m : m + ap, 0:D]))
                    x_norm = nl.multiply(x_norm, s)          # elementwise (ap, D)
                else:
                    x_norm = nl.multiply(x_norm, s_row)      # bcast (1, D) over P

            y = nl.ndarray((ap, D), dtype=dt, buffer=nl.sbuf)
            nisa.tensor_copy(dst=y, src=x_norm)              # fp32 -> input dtype
            nl.store(out[b, m : m + ap, 0:D], y)

    return out


@nki.jit
def rmsnorm_zero_core(hidden, scale_msa, eps: float = 1e-6):
    """RMSNormZero (norm1) fused core: ``rmsnorm(x) * (1 + scale_msa[:, None])``.

    The ``(1 + .)`` and the per-batch broadcast over the sequence are folded into
    the kernel. The gate/scale_mlp/gate_mlp chunks of RMSNormZero are handled in
    PyTorch around this call.

    Args:
        hidden: HBM tensor ``(B, S, D)``, bf16 (or any float).
        scale_msa: HBM tensor ``(B, D)`` -- the first chunk of
            ``linear(silu(emb)).chunk(4, dim=1)``. Broadcast over S.
        eps: numerical-stability epsilon.

    Returns:
        HBM tensor ``(B, S, D)`` with the same dtype as ``hidden``.
    """
    B, S, D = hidden.shape
    dt = hidden.dtype
    out = nl.ndarray((B, S, D), dtype=dt, buffer=nl.shared_hbm)

    for b in range(B):
        # (1 + scale_msa[b]) as a (1, D) fp32 row, broadcast across partitions.
        w1 = nl.ndarray((1, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=w1, data=nl.load(scale_msa[b : b + 1, 0:D]),
            op0=nl.add, operand0=1.0,
        )

        for m in range(0, S, _PMAX):
            ap = min(_PMAX, S - m)
            # nki 0.5.0 has no nl.mgrid; address the tile with plain slices.
            x_dt = nl.load(hidden[b, m : m + ap, 0:D])
            x_norm = _normalize_tile(x_dt, D, eps)
            x_norm = nl.multiply(x_norm, w1)                 # bcast (1, D) over P

            y = nl.ndarray((ap, D), dtype=dt, buffer=nl.sbuf)
            nisa.tensor_copy(dst=y, src=x_norm)
            nl.store(out[b, m : m + ap, 0:D], y)

    return out
