# SPDX-License-Identifier: Apache-2.0
"""DeepSeek V3.2 weight-loader factories.

The HF DeepSeek V3.2 checkpoint stores 2D matmul weights as native FP8
(``e4m3fn``) with a companion fp32 ``weight_scale_inv`` of shape
``[N/128, K/128]`` — one scale per 128×128 tile, ``ue8m0`` format
(scale is a power of 2). Per-tile dequant rule:

    bf16[i, j] = fp8[i, j].float() * scale[i // 128, j // 128]

Each loader factory in this file detects FP8 by ``len(slices) == 2`` (or
2× the BF16 count, for MoE per-expert loaders). When dequant is needed,
only the rank's slice is read from disk — same I/O footprint as the BF16
path. The dequant happens on the loader thread (CPU) so the device only
ever sees bf16 weights and the model graph is identical for both
checkpoints.

The pre-converted BF16 checkpoint (``DeepSeek-V3.2-bf16``) keeps working
because every loader has a ``len(slices) == 1`` fast path that matches
the original behavior.
"""

from __future__ import annotations

import concurrent.futures as _futures

import torch

from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

__all__ = [
    "transpose_only_loader",
    "sharded_2d_transposed_loader",
    "moe_gate_up_loader",
    "moe_down_loader",
]


_FP8_BLOCK_SIZE = 128

# Concurrent per-expert reads inside the MoE loaders. The HF DeepSeek
# checkpoint stores 256 separate tensors per MoE layer; doing the
# slice/transpose loop sequentially makes the transform-stage thread the
# bottleneck. The PySafeSlice indexing releases the GIL (it's a Rust mmap
# read), so threads make real progress.
_MOE_LOADER_WORKERS = 16


def _fp8_dequant_block(
    weight: torch.Tensor,
    scale: torch.Tensor,
    n_off: int = 0,
    k_off: int = 0,
) -> torch.Tensor:
    """Dequantize a 2D (or N-D with 2D-tile-blocked tail) FP8 weight to BF16.

    ``weight`` shape ``[..., N, K]`` (FP8); ``scale`` shape ``[..., Sn, Sk]``
    (fp32) covering the [N, K] slice's 128-block window. ``n_off``/``k_off``
    is the within-first-block offset (0 if the slice starts at a block
    boundary in the original full tensor).
    """
    bs = _FP8_BLOCK_SIZE
    N, K = weight.shape[-2], weight.shape[-1]
    expanded = scale.repeat_interleave(bs, dim=-2).repeat_interleave(bs, dim=-1)
    full_scale = expanded[..., n_off : n_off + N, k_off : k_off + K]
    return (weight.to(torch.float32) * full_scale.to(torch.float32)).to(torch.bfloat16)


def _scale_window(weight_start: int, weight_size: int) -> tuple[int, int, int]:
    """Compute the scale-block range and within-block offset that covers
    a weight slice ``[weight_start : weight_start + weight_size]``.

    Returns ``(block_start, block_end, within_block_offset)``. Use
    ``scale[block_start:block_end]`` to pick the rows/cols touching the
    rank's slice; pass ``within_block_offset`` to ``_fp8_dequant_block``.
    """
    bs = _FP8_BLOCK_SIZE
    elem_end = weight_start + weight_size
    block_start = weight_start // bs
    block_end = (elem_end + bs - 1) // bs
    return block_start, block_end, weight_start % bs


def _read_2d_dequant_slice(
    w_slice,
    s_slice,
    n_start: int,
    n_size: int,
    k_start: int,
    k_size: int,
) -> torch.Tensor:
    """Read ``[n_start:n_start+n_size, k_start:k_start+k_size]`` of an FP8
    HF weight + its companion ``weight_scale_inv``, and return the
    dequantized bf16 slice. Only the needed weight rows/cols and the scale
    blocks touching them are pulled from disk.
    """
    n_blk_start, n_blk_end, n_off = _scale_window(n_start, n_size)
    k_blk_start, k_blk_end, k_off = _scale_window(k_start, k_size)
    w_part = w_slice[n_start : n_start + n_size, k_start : k_start + k_size]
    s_part = s_slice[n_blk_start:n_blk_end, k_blk_start:k_blk_end]
    return _fp8_dequant_block(w_part, s_part, n_off=n_off, k_off=k_off)


def transpose_only_loader() -> SafetensorsWeightLoader:
    """Loader for replicated 2D weights stored transposed in the checkpoint.

    HF stores ``nn.Linear`` weights as ``[out, in]``; our params are
    ``[in, out]``. No sharding — just transpose. If the checkpoint stores
    the weight as FP8 (signaled by a second slice = ``weight_scale_inv``),
    dequantize to bf16 before transposing.
    """

    def transform(slices, rank):
        if len(slices) == 1:
            return slices[0][:].T
        assert len(slices) == 2, (
            f"transpose_only_loader expected 1 (bf16) or 2 (fp8 + scale) "
            f"slices, got {len(slices)}"
        )
        w_slice, s_slice = slices
        N, K = w_slice.get_shape()
        return _read_2d_dequant_slice(w_slice, s_slice, 0, N, 0, K).T

    return SafetensorsWeightLoader(transform=transform)


def sharded_2d_transposed_loader(
    shard_dim: int,
    shard_size: int,
    num_shards: int,
) -> SafetensorsWeightLoader:
    """Equivalent of ``sharding_weight_loader(is_storage_transposed=True)``
    for the deepseek_v32 model, with optional FP8 dequant.

    Param shape is ``[in, out]`` (transposed from HF's ``[out, in]``):
    - ``shard_dim=1`` → shard the param's output dim → slice HF dim 0.
    - ``shard_dim=0`` → shard the param's input dim → slice HF dim 1.

    For BF16 (``len(slices) == 1``) the read-then-transpose matches the
    standard library loader. For FP8 (``len(slices) == 2``) only the
    rank's slice + the scale blocks touching it are read from disk.
    """
    assert shard_dim in (0, 1), f"shard_dim must be 0 or 1, got {shard_dim}"
    storage_shard_dim = 1 - shard_dim  # transposed storage swaps 0/1

    def transform(slices, rank):
        local_rank = rank % num_shards
        start = local_rank * shard_size

        if len(slices) == 1:
            w_slice = slices[0]
            sl = [slice(None), slice(None)]
            sl[storage_shard_dim] = slice(start, start + shard_size)
            return w_slice[tuple(sl)].T

        assert len(slices) == 2, (
            f"sharded_2d_transposed_loader expected 1 (bf16) or 2 (fp8 + "
            f"scale) slices, got {len(slices)}"
        )
        w_slice, s_slice = slices
        N, K = w_slice.get_shape()
        if storage_shard_dim == 0:
            return _read_2d_dequant_slice(w_slice, s_slice, start, shard_size, 0, K).T
        return _read_2d_dequant_slice(w_slice, s_slice, 0, N, start, shard_size).T

    return SafetensorsWeightLoader(transform=transform)


def moe_gate_up_loader(
    hidden_size: int,
    moe_inter_per_rank: int,
    num_shards: int,
    num_experts: int,
) -> SafetensorsWeightLoader:
    """Loader for fused MoE ``gate_up_proj_weight``.

    Param shape: ``[E, H, 2*I_per_rank]``.

    BF16 checkpoint keys: ``2*E`` entries — ``[gate_e, up_e, ...]``, each
    HF-style ``[I, H]``. FP8 checkpoint keys: ``4*E`` entries —
    ``[gate_e, gate_e_scale_inv, up_e, up_e_scale_inv, ...]``.

    Concat order matches the model forward (``chunk(2, dim=-1)`` →
    gate, up).
    """
    del hidden_size  # kept in signature for caller-side documentation

    def transform(slices, rank):
        if len(slices) == 2 * num_experts:
            stride = 2  # bf16: [gate, up] per expert
            fp8 = False
        elif len(slices) == 4 * num_experts:
            stride = 4  # fp8: [gate_w, gate_s, up_w, up_s] per expert
            fp8 = True
        else:
            raise AssertionError(
                f"moe_gate_up_loader: expected {2 * num_experts} (bf16) or "
                f"{4 * num_experts} (fp8) slices, got {len(slices)}"
            )

        local_rank = rank % num_shards
        start = local_rank * moe_inter_per_rank

        def _read_one(w_slice, s_slice):
            if not fp8:
                return w_slice[start : start + moe_inter_per_rank, :].T.contiguous()
            _, K = w_slice.get_shape()
            return _read_2d_dequant_slice(
                w_slice, s_slice, start, moe_inter_per_rank, 0, K
            ).T.contiguous()

        def _one(e):
            base = stride * e
            if fp8:
                gate = _read_one(slices[base], slices[base + 1])
                up = _read_one(slices[base + 2], slices[base + 3])
            else:
                gate = _read_one(slices[base], None)
                up = _read_one(slices[base + 1], None)
            return torch.cat([gate, up], dim=-1)  # [H, 2*I_per_rank]

        with _futures.ThreadPoolExecutor(max_workers=_MOE_LOADER_WORKERS) as ex:
            per_expert = list(ex.map(_one, range(num_experts)))
        return torch.stack(per_expert, dim=0)  # [E, H, 2*I_per_rank]

    return SafetensorsWeightLoader(transform=transform)


def moe_down_loader(
    hidden_size: int,
    moe_inter_per_rank: int,
    num_shards: int,
    num_experts: int,
) -> SafetensorsWeightLoader:
    """Loader for MoE ``down_proj_weight``.

    Param shape: ``[E, I_per_rank, H]``.

    BF16 checkpoint keys: ``E`` entries, each HF-style ``[H, I]``.
    FP8 checkpoint keys: ``2*E`` entries —
    ``[down_e, down_e_scale_inv, ...]``.
    """
    del hidden_size  # kept in signature for caller-side documentation

    def transform(slices, rank):
        if len(slices) == num_experts:
            stride = 1
            fp8 = False
        elif len(slices) == 2 * num_experts:
            stride = 2
            fp8 = True
        else:
            raise AssertionError(
                f"moe_down_loader: expected {num_experts} (bf16) or "
                f"{2 * num_experts} (fp8) slices, got {len(slices)}"
            )

        local_rank = rank % num_shards
        start = local_rank * moe_inter_per_rank

        def _one(e):
            base = stride * e
            w_slice = slices[base]
            if not fp8:
                return w_slice[:, start : start + moe_inter_per_rank].T.contiguous()
            s_slice = slices[base + 1]
            H, _ = w_slice.get_shape()
            return _read_2d_dequant_slice(
                w_slice, s_slice, 0, H, start, moe_inter_per_rank
            ).T.contiguous()

        with _futures.ThreadPoolExecutor(max_workers=_MOE_LOADER_WORKERS) as ex:
            per_expert = list(ex.map(_one, range(num_experts)))
        return torch.stack(per_expert, dim=0)  # [E, I_per_rank, H]

    return SafetensorsWeightLoader(transform=transform)
