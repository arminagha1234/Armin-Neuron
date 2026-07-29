"""CPU reference for the Mochi-1 flash joint-attention NKI kernel.

Steps 1-2 of the incremental NKI pipeline (reference -> numpy). This module is
the ground truth that ``flash_attn_nki.py`` must reproduce and that
``test_flash_attn.py`` validates against.

## What operation this mirrors

It reproduces ``neuron_compat._attention_bmm`` EXACTLY for the non-causal,
joint self-attention Mochi uses (``src/neuron_compat.py``):

    scores = bmm(Q, K^T) * scale          # (P, Sq, Sk)
    scores = scores + key_bias            # additive per-KEY-COLUMN bias
    probs  = softmax(scores, dim=-1)      # over the full key axis
    out    = bmm(probs, V)                # (P, Sq, D)

where ``P = batch*heads`` planes, and everything is computed with an fp32
internal accumulation, matching the port's exact (untiled) path. Because the
softmax denominator is over the *complete* key axis, the query-tiled port is
numerically identical to the untiled form; a flash / online-softmax kernel that
streams the key axis is likewise numerically identical (up to fp rounding).

## The additive key bias (the one Mochi-specific wrinkle)

Mochi's mask is ``(B, 1, 1, Sk)`` -- a function of *key column and batch only*,
broadcast across all heads and all query rows. Value ``0`` for every visual key
and every real text key, ``-10000.0`` (``neuron_compat.MASKED_BIAS``) for padded
text key columns. It is added to the scores before softmax. No full
``(B, H, Sq, Sk)`` mask is ever materialised.

This reference works with the bias in a *canonical per-plane* form ``(P, Sk)``
(one row per ``batch*heads`` plane). ``build_key_bias_from_mask`` expands the
model's ``(B, 1, 1, Sk)`` mask into that form; ``key_bias_from_keep_mask`` builds
it from a ``{0,1}`` keep-mask the way ``neuron_compat.to_additive_mask`` does.

## Two numpy references

* ``flash_attention_ref_np`` -- the naive materialised ``bmm->softmax->bmm``
  (matches ``_attention_bmm`` untiled). This is the primary oracle.
* ``flash_attention_online_np`` -- the *same* result computed with the exact
  online-softmax streaming algorithm the NKI kernel implements (running max +
  running sum + accumulator rescale). Included so the test can prove the online
  algorithm equals the naive one in pure Python, before any device is involved.

## Dtypes

Inputs/outputs are typically bf16; all reductions and intermediate arithmetic
run in fp32, and the result is cast back to the input dtype -- identical to the
port's fp32-internal ``bmm``/``softmax`` path.

License: Apache-2.0 (matches the AWS contrib and the Mochi weights).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:  # torch is optional; the numpy path is self-contained.
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised only in torch-less envs
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


# Mirrors neuron_compat.MASKED_BIAS (NOT -inf: bf16 saturates and softmax
# returns NaN on the compiled lazy backend; NOT 0.0: XLA const-folds it away).
MASKED_BIAS = -10000.0


# ---------------------------------------------------------------------------
# Key-bias builders (canonical per-plane (P, Sk) form)
# ---------------------------------------------------------------------------
def build_key_bias_from_mask(
    additive_mask: np.ndarray,
    num_heads: int,
) -> np.ndarray:
    """Expand a model-side additive mask into the per-plane ``(P, Sk)`` form.

    Args:
        additive_mask: the already-additive bias, shape ``(B, 1, 1, Sk)`` (or
            any shape squeezable to ``(B, Sk)``). Values ``0`` = attend,
            ``MASKED_BIAS`` = masked. This is what ``neuron_compat`` feeds the
            attention path after ``to_additive_mask`` / ``_normalize_mask``.
        num_heads: heads per batch element. ``P = B * num_heads``.

    Returns:
        ``(P, Sk)`` fp32 array: plane ``p`` (batch ``b = p // num_heads``) gets
        that batch's key-column bias. Broadcasting over heads is done by repeat.
    """
    m = np.asarray(additive_mask)
    m = m.reshape(m.shape[0], m.shape[-1])  # (B, Sk)
    # Repeat each batch row across its heads: plane order is (b0h0, b0h1, ...).
    return np.repeat(m.astype(np.float32), num_heads, axis=0)  # (P, Sk)


def key_bias_from_keep_mask(
    keep_mask: np.ndarray,
    num_heads: int,
    masked_value: float = MASKED_BIAS,
) -> np.ndarray:
    """Build the per-plane additive key bias from a ``{0,1}``/bool keep-mask.

    Input convention (diffusers): ``1``/``True`` = ATTEND, ``0``/``False`` =
    MASK OUT. Matches ``neuron_compat.to_additive_mask`` arithmetic
    (``(1 - keep) * masked_value``).

    Args:
        keep_mask: ``(B, Sk)`` (or squeezable to it) of {0,1}/bool.
        num_heads: heads per batch element.
        masked_value: additive value for masked columns (default MASKED_BIAS).

    Returns:
        ``(P, Sk)`` fp32 additive bias.
    """
    keep = np.asarray(keep_mask).astype(np.float32)
    keep = keep.reshape(keep.shape[0], keep.shape[-1])  # (B, Sk)
    additive = (1.0 - keep) * masked_value              # (B, Sk)
    return np.repeat(additive, num_heads, axis=0)       # (P, Sk)


# ---------------------------------------------------------------------------
# NumPy reference #1: naive materialised bmm -> softmax -> bmm
# (matches neuron_compat._attention_bmm untiled path exactly)
# ---------------------------------------------------------------------------
def flash_attention_ref_np(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    key_bias: Optional[np.ndarray] = None,
    scale: Optional[float] = None,
) -> np.ndarray:
    """Non-causal joint self-attention, materialised, fp32 internal.

    Args:
        q: ``(P, Sq, D)`` query, any float dtype. ``P = batch*heads``.
        k: ``(P, Sk, D)`` key.
        v: ``(P, Sk, D)`` value.
        key_bias: optional ``(P, Sk)`` additive per-key-column bias (see
            builders above). Added to scores before softmax, broadcast across
            all query rows. ``None`` = no mask.
        scale: softmax scale; defaults to ``1/sqrt(D)`` (Mochi uses
            ``1/sqrt(128)``).

    Returns:
        ``(P, Sq, D)`` output, cast back to ``q.dtype``.
    """
    in_dtype = q.dtype
    P, Sq, D = q.shape
    Sk = k.shape[1]
    if scale is None:
        scale = 1.0 / np.sqrt(D)

    qf = q.astype(np.float32)
    kf = k.astype(np.float32)
    vf = v.astype(np.float32)

    out = np.empty((P, Sq, D), dtype=np.float32)
    for p in range(P):
        scores = (qf[p] @ kf[p].T) * scale          # (Sq, Sk)
        if key_bias is not None:
            scores = scores + key_bias[p][None, :].astype(np.float32)
        scores = scores - scores.max(axis=-1, keepdims=True)
        e = np.exp(scores)
        probs = e / e.sum(axis=-1, keepdims=True)    # (Sq, Sk)
        out[p] = probs @ vf[p]                       # (Sq, D)
    return out.astype(in_dtype)


# ---------------------------------------------------------------------------
# NumPy reference #2: online-softmax streaming (the exact NKI algorithm)
# ---------------------------------------------------------------------------
def flash_attention_online_np(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    key_bias: Optional[np.ndarray] = None,
    scale: Optional[float] = None,
    q_tile: int = 128,
    k_tile: int = 128,
) -> np.ndarray:
    """Same attention, computed with the flash online-softmax the kernel uses.

    Streams the key axis in tiles, maintaining a running max ``m``, running
    denominator ``l`` and running accumulator ``O`` per query row, rescaling on
    every tile. Mathematically equal to :func:`flash_attention_ref_np` (up to
    fp rounding). Tiling here mirrors the kernel's Q-partition / K-stream tiling
    so this doubles as an algorithm oracle.

    Args mirror :func:`flash_attention_ref_np`; ``q_tile``/``k_tile`` are the
    partition / key-stream tile sizes (default 128 to match the kernel).
    """
    in_dtype = q.dtype
    P, Sq, D = q.shape
    Sk = k.shape[1]
    if scale is None:
        scale = 1.0 / np.sqrt(D)

    qf = q.astype(np.float32)
    kf = k.astype(np.float32)
    vf = v.astype(np.float32)

    out = np.empty((P, Sq, D), dtype=np.float32)
    neg_inf = -1.0e30  # finite sentinel: avoids inf-inf=NaN on fully-masked tiles

    for p in range(P):
        for qs in range(0, Sq, q_tile):
            qe = min(qs + q_tile, Sq)
            qb = qf[p, qs:qe]                        # (q_size, D)
            q_size = qe - qs

            m_i = np.full((q_size, 1), neg_inf, dtype=np.float32)   # running max
            l_i = np.zeros((q_size, 1), dtype=np.float32)           # running sum
            o_i = np.zeros((q_size, D), dtype=np.float32)           # running acc

            for ks in range(0, Sk, k_tile):
                ke = min(ks + k_tile, Sk)
                kb = kf[p, ks:ke]                    # (k_size, D)
                vb = vf[p, ks:ke]                    # (k_size, D)

                s = (qb @ kb.T) * scale              # (q_size, k_size)
                if key_bias is not None:
                    s = s + key_bias[p][None, ks:ke].astype(np.float32)

                tile_max = s.max(axis=-1, keepdims=True)            # (q_size,1)
                m_new = np.maximum(m_i, tile_max)                   # (q_size,1)
                # exp(s - m_new); correction = exp(m_i - m_new)
                probs = np.exp(s - m_new)                           # (q_size,k_size)
                corr = np.exp(m_i - m_new)                          # (q_size,1)

                l_i = corr * l_i + probs.sum(axis=-1, keepdims=True)
                o_i = corr * o_i + probs @ vb                       # (q_size, D)
                m_i = m_new

            out[p, qs:qe] = o_i / l_i
    return out.astype(in_dtype)


# ---------------------------------------------------------------------------
# Torch reference (byte-for-byte the neuron_compat._attention_bmm untiled path)
# ---------------------------------------------------------------------------
if _HAS_TORCH:

    def flash_attention_ref_torch(
        q: "torch.Tensor",
        k: "torch.Tensor",
        v: "torch.Tensor",
        key_bias: "Optional[torch.Tensor]" = None,
        scale: "Optional[float]" = None,
    ) -> "torch.Tensor":
        """Torch twin of :func:`flash_attention_ref_np`.

        Mirrors ``neuron_compat._attention_bmm`` (untiled): ``bmm(Q,K^T)*scale
        (+ per-key-column bias) -> softmax -> bmm(P,V)`` on 3D ``(P, S, D)``
        tensors, fp32 internal. ``key_bias`` is ``(P, Sk)`` (broadcast over
        query rows via ``[:, None, :]``).
        """
        in_dtype = q.dtype
        D = q.shape[-1]
        if scale is None:
            scale = 1.0 / (D ** 0.5)

        qf = q.to(torch.float32)
        kf = k.to(torch.float32)
        vf = v.to(torch.float32)

        scores = torch.bmm(qf, kf.transpose(-1, -2)) * scale     # (P, Sq, Sk)
        if key_bias is not None:
            scores = scores + key_bias.to(torch.float32)[:, None, :]
        probs = scores.softmax(dim=-1)
        out = torch.bmm(probs, vf)                               # (P, Sq, D)
        return out.to(in_dtype)


__all__ = [
    "MASKED_BIAS",
    "build_key_bias_from_mask",
    "key_bias_from_keep_mask",
    "flash_attention_ref_np",
    "flash_attention_online_np",
]
if _HAS_TORCH:
    __all__ += ["flash_attention_ref_torch"]
