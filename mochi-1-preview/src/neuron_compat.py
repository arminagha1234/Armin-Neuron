"""Neuron compatibility shims for Mochi-1 (10B AsymmDiT) native PyTorch.

Derived from the LTX-2 recipe
(`.tmp/Armin-Neuron/ltx2/native-pytorch/src/neuron_compat.py`, itself from
`aws-neuron/neuronx-distributed-inference/contrib/models/ltx2-video-audio`)
with two Mochi-specific changes that the LTX-2 version does NOT handle:

1. **Bool attention masks.** `MochiAttentionPool` (inside
   `MochiCombinedTimestepCaptionEmbedding`, which runs on device) calls
   `F.scaled_dot_product_attention` with a *bool* `attn_mask`:

       attn_mask = mask[:, None, None, :].bool()
       attn_mask = F.pad(attn_mask, (1, 0), value=True)

   Torch SDPA semantics: bool mask means `True = attend`. The LTX-2 shim
   does `scores = scores + attn_mask`, which for a bool tensor adds
   1.0/0.0 to the logits — silently wrong, no error, no NaN. We convert
   bool masks to the additive form first.

2. **Tiled-query attention.** Mochi does full 3D attention over up to
   44,520 visual tokens (163 frames @ 480p). Materialising the whole
   score matrix costs ~96 GB in bf16 across 24 heads, so the plain
   `bmm -> softmax -> bmm` path OOMs well before the model does.
   `_attention_bmm` tiles over the query axis: each tile still attends to
   *all* keys, so the softmax is over the complete key axis and the
   result is **numerically exact**, not an approximation. Memory drops
   from O(Sq x Sk) to O(q_chunk x Sk).

Everything else follows LTX-2: explicit BMM instead of fused SDPA (stock
SDPA miscomputes on Neuron's compiled bf16 lazy backend), and the
`-10000.0` additive mask value rather than `-inf` or `0.0`.

Note: `RankTensor` from the LTX-2 recipe is deliberately absent. Mochi
needs per-rank RoPE slicing too, but we get it by sharding the
`pos_frequencies` *parameter* on its head axis at weight-load time (see
`mochi_tp_plan.shard_pos_frequencies`). That happens outside the traced
graph, so there is no baked-constant-rank hazard to work around.

License: Apache-2.0 (matches the AWS contrib and the Mochi weights).
"""
from __future__ import annotations

import logging
import math
import os

import torch

logger = logging.getLogger(__name__)

# Standard HF/NxDI masked-logit bias. NOT -inf (bf16 saturates to -1e38 and
# softmax returns NaN on the compiled lazy backend) and NOT 0.0 (an all-zero
# mask gets constant-folded away by XLA, so the mask silently vanishes from
# the compiled graph and only bites you once a padded prompt shows up).
MASKED_BIAS = -10000.0

# Auto-tiling budgets the *whole* score tensor, not one plane. The score
# tensor is (batch*heads, Sq, Sk), so plane count matters as much as sequence
# length -- an earlier per-plane threshold picked q_chunk~6656 at 31 frames
# and OOM'd on a 24 GB logical core (peak 23.86 GB) because 12 planes of
# 9796x9796 is 2.3 GB for `scores` and another 2.3 GB for `probs`.
#
# 256 MiB per score tensor keeps scores+probs near 512 MiB regardless of
# geometry, which leaves room for 9.25 GB of TP=4 weights plus the rest of
# the activations.
_AUTO_TILE_BUDGET_BYTES = 256 * 1024 * 1024
_MIN_Q_CHUNK = 512
_DEFAULT_Q_CHUNK = 2048

_sdpa_replaced = False
_sdpa_original = None

# 0 / unset -> automatic (tile only above the threshold).
# >0        -> always tile with this query-chunk size.
# <0        -> never tile.
_q_chunk_override = int(os.environ.get("MOCHI_ATTN_Q_CHUNK", "0"))


def set_attention_chunking(q_chunk: int | None) -> None:
    """Override the query-tile size used by the BMM attention path.

    Args:
        q_chunk: ``None``/``0`` for automatic, a positive int to force that
            tile size, or a negative int to disable tiling entirely.

    Tile size affects the compiled graph shape, so changing it invalidates
    cached NEFFs. Set it once before the first forward.
    """
    global _q_chunk_override
    _q_chunk_override = 0 if q_chunk is None else int(q_chunk)
    logger.info("attention q_chunk override set to %s", _q_chunk_override)


def _resolve_q_chunk(
    seq_q: int, seq_k: int, n_planes: int, elem_bytes: int
) -> int | None:
    """Pick a query-tile size, or None to run untiled.

    Budgets the full `(n_planes, Sq, Sk)` score tensor against
    `_AUTO_TILE_BUDGET_BYTES`. Rounds down to a multiple of 512 so tile
    shapes stay compiler-friendly and recur across calls (each distinct tile
    shape is a separate NEFF).
    """
    if _q_chunk_override < 0:
        return None
    if _q_chunk_override > 0:
        return min(_q_chunk_override, seq_q)

    row_bytes = max(1, n_planes * seq_k * elem_bytes)
    if seq_q * row_bytes <= _AUTO_TILE_BUDGET_BYTES:
        return None

    target = (_AUTO_TILE_BUDGET_BYTES // row_bytes) // 512 * 512
    if target < _MIN_Q_CHUNK:
        target = _MIN_Q_CHUNK
    return min(target, seq_q)


def to_additive_mask(
    mask: torch.Tensor | None,
    dtype: torch.dtype = torch.bfloat16,
    masked_value: float = MASKED_BIAS,
) -> torch.Tensor | None:
    """Convert a {0,1}/bool keep-mask into an additive logit bias.

    Input convention (diffusers): 1/True = ATTEND, 0/False = MASK OUT.
    Output: 0.0 where attending, ``masked_value`` where masked.

    Uses arithmetic rather than ``masked_fill`` so the op lowers cleanly on
    the XLA/Neuron backend.
    """
    if mask is None:
        return None
    keep = mask.to(dtype)
    return ((1.0 - keep) * masked_value).to(dtype)


def _normalize_mask(
    attn_mask: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Coerce any SDPA-style mask into an additive float bias of `dtype`.

    Handles the bool case that the LTX-2 shim gets wrong (see module
    docstring, item 1).
    """
    if attn_mask.dtype == torch.bool:
        # True = attend  ->  0.0 ; False = mask  ->  MASKED_BIAS
        return ((~attn_mask).to(dtype)) * MASKED_BIAS
    return attn_mask.to(dtype)


def _collapse_mask(
    attn_mask: torch.Tensor,
    batch: int,
    heads: int,
    seq_q: int,
) -> torch.Tensor:
    """Reshape a mask so it broadcasts against 3D scores `(batch*heads, Sq, Sk)`.

    Deliberately preserves size-1 axes instead of expanding them: Mochi's
    masks are `(B, 1, 1, Sk)`, and expanding that to a full `(B*H, Sq, Sk)`
    tensor would cost as much memory as the score matrix we are trying to
    keep small.
    """
    if attn_mask.ndim == 2:
        # (Sq, Sk) or (B, Sk) -- treat as broadcast over the batch*head axis.
        return attn_mask.unsqueeze(0)

    if attn_mask.ndim == 3:
        if attn_mask.shape[0] == batch and heads > 1:
            # (B, Sq, Sk) -> (B*H, Sq, Sk) without materialising Sq.
            m = attn_mask.unsqueeze(1).expand(batch, heads, -1, -1)
            return m.reshape(batch * heads, m.shape[-2], m.shape[-1])
        return attn_mask

    if attn_mask.ndim == 4:
        mb, mh, msq, msk = attn_mask.shape
        # Expand only the batch/head axes; leave a size-1 query axis alone.
        if mb == 1 and batch > 1:
            attn_mask = attn_mask.expand(batch, mh, msq, msk)
            mb = batch
        if mh == 1 and heads > 1:
            attn_mask = attn_mask.expand(mb, heads, msq, msk)
            mh = heads
        return attn_mask.reshape(mb * mh, msq, msk)

    raise ValueError(f"unsupported attn_mask ndim={attn_mask.ndim}")


def _attention_bmm(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    scale: float,
    q_chunk: int | None,
) -> torch.Tensor:
    """Explicit-BMM attention on 3D `(batch*heads, S, D)` tensors.

    When `q_chunk` is set, tiles the query axis. Each tile attends to every
    key, so the softmax denominator is complete per tile and the output is
    bit-comparable to the untiled path (no online-softmax rescaling needed).
    """
    key_t = key.transpose(-1, -2)
    seq_q = query.shape[1]

    if q_chunk is None or q_chunk >= seq_q:
        scores = torch.bmm(query, key_t) * scale
        if attn_mask is not None:
            scores = scores + attn_mask
        # Softmax in fp32: `scores` is bf16 (bf16 Q.Kt), and reducing over up
        # to ~9,800 keys with a -10000.0 masked bias in an 8-bit mantissa
        # accumulates real error. Upcasting the reduction (not the matmuls)
        # costs one fp32 tile and makes this path match the fp32-softmax flash
        # NKI kernel in nki_kernels/attention/.
        probs = scores.float().softmax(dim=-1).to(value.dtype)
        return torch.bmm(probs, value)

    chunks = []
    for start in range(0, seq_q, q_chunk):
        stop = min(start + q_chunk, seq_q)
        scores = torch.bmm(query[:, start:stop], key_t) * scale
        if attn_mask is not None:
            m = attn_mask
            if m.shape[-2] > 1:
                # Per-query mask: take the matching row slice.
                m = m[..., start:stop, :]
            scores = scores + m
        probs = scores.float().softmax(dim=-1).to(value.dtype)
        chunks.append(torch.bmm(probs, value))
    return torch.cat(chunks, dim=1)


def neuron_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Drop-in `F.scaled_dot_product_attention` replacement for Neuron.

    CPU tensors fall through to the real SDPA so the T5 text encoder and the
    VAE (both pinned to CPU) keep the optimised kernels.
    """
    if query.device.type == "cpu":
        return _sdpa_original(
            query, key, value,
            attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=is_causal, scale=scale,
        )

    if is_causal:
        # Mochi never uses causal attention; refuse rather than silently
        # dropping the mask.
        raise NotImplementedError(
            "neuron_sdpa: is_causal=True is not implemented (Mochi does not "
            "need it). Pass an explicit additive mask instead."
        )

    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])

    orig_shape = None
    if query.ndim == 4:
        orig_shape = query.shape
        b, h, seq_q, d_head = query.shape
        if attn_mask is not None:
            attn_mask = _normalize_mask(attn_mask, query.dtype)
            attn_mask = _collapse_mask(attn_mask, b, h, seq_q)
        query = query.reshape(b * h, seq_q, d_head)
        key = key.reshape(b * h, -1, key.shape[-1])
        value = value.reshape(b * h, -1, value.shape[-1])
    elif attn_mask is not None:
        attn_mask = _normalize_mask(attn_mask, query.dtype)

    q_chunk = _resolve_q_chunk(
        query.shape[1], key.shape[1], query.shape[0], query.element_size()
    )
    out = _attention_bmm(query, key, value, attn_mask, scale, q_chunk)

    if orig_shape is not None:
        out = out.reshape(orig_shape[0], orig_shape[1], -1, out.shape[-1])
    return out


def install_bmm_sdpa() -> None:
    """Globally replace `F.scaled_dot_product_attention` with `neuron_sdpa`.

    Idempotent. Call once, before building or running any model.
    """
    global _sdpa_replaced, _sdpa_original
    if _sdpa_replaced:
        return
    _sdpa_original = torch.nn.functional.scaled_dot_product_attention
    torch.nn.functional.scaled_dot_product_attention = neuron_sdpa
    _sdpa_replaced = True
    logger.info("BMM-SDPA replacement installed (bool-mask safe, tiled)")


def uninstall_bmm_sdpa() -> None:
    """Restore the original SDPA. Used by the offline test suite."""
    global _sdpa_replaced, _sdpa_original
    if not _sdpa_replaced:
        return
    torch.nn.functional.scaled_dot_product_attention = _sdpa_original
    _sdpa_replaced = False


def is_installed() -> bool:
    return _sdpa_replaced


def print_active_fixes(rank: int = 0) -> None:
    if rank != 0:
        return
    mode = (
        f"auto (budget {_AUTO_TILE_BUDGET_BYTES // (1024*1024)} MiB/score tensor)"
        if _q_chunk_override == 0
        else ("off" if _q_chunk_override < 0 else str(_q_chunk_override))
    )
    print("[neuron_compat] BMM-SDPA installed:", _sdpa_replaced, flush=True)
    print(f"[neuron_compat] query tiling: {mode}", flush=True)
    print(f"[neuron_compat] masked bias : {MASKED_BIAS}", flush=True)
