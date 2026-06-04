# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek V3.2 BF16 Implementation
======================================

DeepSeek V3.2 (DeepseekV32ForCausalLM, model_type: deepseek_v32), NOT V3.

Architecture:
- MLA (Multi-Head Latent Attention) with compressed KV cache and weight absorption
- MoE (256 routed experts + 1 shared expert) with group-limited sigmoid routing
- Dense MLP for first 3 layers, MoE for layers 3-60
- YaRN RoPE with interleaved format
- Phase 1: Dense attention (no DSA). DSA Indexer deferred to Phase 2.

Reference implementations:
- reference/DeepSeek-V3.2-Exp/inference/model.py (V3.2 MLA + DSA)
- neuronx-distributed-inference/.../modeling_deepseek.py (NxDI V3 Neuron patterns)
"""

import glob
import logging
import math
import os

import torch
from torch import nn
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF
from vllm_neuron.model.kv_cache import KVSpec, LayerSpec
from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint
from vllm_neuron.utils.weight_loader import (
    expert_parallel_weight_loader,
    set_weight_loader,
)
from transformers import PretrainedConfig
from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.sampler import Sampler
import vllm_neuron.nn as neuron_nn
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding

from nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
)
from nkilib.core.moe.moe_cte.moe_cte import MoECTEImplementation

from .config import DeepseekV32Config
from .weight_loader import (
    moe_down_loader,
    moe_gate_up_loader,
    sharded_2d_transposed_loader,
    transpose_only_loader,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Section 1: RMS Normalization
# =============================================================================


class DeepseekV32RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, dtype=torch.bfloat16):
        super().__init__()
        # HF DeepSeek stores norm weights as fp32; keep param fp32 so weight
        # loading matches checkpoint dtype and forward avoids upcasts.
        # `dtype` arg retained for API compatibility but ignored.
        del dtype
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


# =============================================================================
# Section 2: YaRN Rotary Position Embedding (Interleaved)
# =============================================================================


class DeepseekV32RotaryEmbedding(nn.Module):
    """YaRN RoPE with interleaved format for MLA.

    DeepSeek V3.2 uses interleaved RoPE: pairs (x0, x1), (x2, x3), ...
    This differs from GPT-OSS which uses non-interleaved (split in half).
    """

    def __init__(self, config: DeepseekV32Config):
        super().__init__()
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.rope_theta = config.rope_theta

        rope_scaling = config.rope_scaling or {}
        self.scaling_factor = rope_scaling.get("factor", 1.0)
        self.beta_slow = rope_scaling.get("beta_slow", 1.0)
        self.beta_fast = rope_scaling.get("beta_fast", 32.0)
        self.original_max_pos = rope_scaling.get(
            "original_max_position_embeddings", 4096
        )
        self.mscale = rope_scaling.get("mscale", 1.0)
        self.mscale_all_dim = rope_scaling.get("mscale_all_dim", 0.0)

        self.softmax_scale_correction = self._compute_mscale()

        inv_freq = self._compute_inv_freq("cpu")
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _compute_mscale(self) -> float:
        if self.scaling_factor <= 1.0:
            return 1.0
        mscale = 0.1 * self.mscale * math.log(self.scaling_factor) + 1.0
        return mscale * mscale

    def _compute_inv_freq(self, device) -> torch.Tensor:
        dim = self.qk_rope_head_dim
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, dim, 2, dtype=torch.float, device=device) / dim)
        )

        if self.scaling_factor <= 1.0:
            return inv_freq

        # Match HF `yarn_find_correction_range`: floor(low), ceil(high), clamp
        # to [0, dim-1]. Without this, the ramp band between interpolation and
        # extrapolation frequencies is off by a fraction of an index — measured
        # up to 33% error in the transition region at indices 11-22 for the
        # DS-V3.2 config (qk_rope_head_dim=64, factor=40, beta_fast=32, beta_slow=1).
        low_raw = (
            dim
            / 2
            * math.log(self.original_max_pos / (self.beta_fast * 2 * math.pi))
            / math.log(self.rope_theta)
        )
        high_raw = (
            dim
            / 2
            * math.log(self.original_max_pos / (self.beta_slow * 2 * math.pi))
            / math.log(self.rope_theta)
        )
        low = max(math.floor(low_raw), 0)
        high = min(math.ceil(high_raw), dim - 1)
        if low == high:
            high += 0.001  # HF guards against singularity

        interpolation = inv_freq / self.scaling_factor
        extrapolation = inv_freq

        ramp = (torch.arange(dim // 2, dtype=torch.float32, device=device) - low) / (
            high - low
        )
        mask = 1 - ramp.clamp(0, 1)

        inv_freq = interpolation * (1 - mask) + extrapolation * mask
        return inv_freq

    def forward(
        self, position_ids: torch.Tensor, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :].float()
        position_ids_expanded = position_ids[:, None].float()
        freqs = position_ids_expanded @ inv_freq_expanded
        cos = freqs.cos()
        sin = freqs.sin()
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


def _apply_rotary_emb_interleaved(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Interleaved RoPE: pairs (x0, x1), (x2, x3), ..."""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    r1 = x1 * cos - x2 * sin
    r2 = x2 * cos + x1 * sin
    return torch.stack((r1, r2), dim=-1).flatten(-2)


def _apply_rotary_emb_non_interleaved(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Non-interleaved RoPE: first half paired with second half.

    This matches HF `apply_rotary_emb(x, freqs_cis, interleaved=False)` used by
    the DSA Lightning Indexer. Note that `cos` / `sin` must be shaped so that
    the last dim equals `x.size(-1) // 2` (same convention as the interleaved
    helper above — one freq per pair).

    HF (reference/DeepSeek-V3.2-Exp/inference/model.py:405-425) implements this
    as a complex-number multiply after a transpose of the inner pair layout.
    We inline the equivalent real-valued formula, mirroring Llama-style RoPE.
    """
    half = x.size(-1) // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    r1 = x1 * cos - x2 * sin
    r2 = x2 * cos + x1 * sin
    return torch.cat((r1, r2), dim=-1)


# =============================================================================
# Section 2b: DSA Lightning Indexer
# Phase 2 — selects the top-K prior tokens to attend to, enabling sparse
# attention for seq_len > index_topk. For seq_len <= index_topk (2048) the
# indexer's selection is all tokens; attention is mathematically identical
# to dense.  Reference: reference/DeepSeek-V3.2-Exp/inference/model.py
# class Indexer (lines 435-487).
# =============================================================================


def _build_hadamard_matrix(n: int, dtype: torch.dtype) -> torch.Tensor:
    """Sylvester-construction Walsh-Hadamard matrix of order n (n = power of 2).

    Matches the natural ordering used by ``fast_hadamard_transform`` so the
    matmul ``x @ H / sqrt(n)`` reproduces ``hadamard_transform(x, scale=n^-0.5)``.
    Tested in `reference/ds32/tests/test_t12c_hadamard_cpu.py`.
    """
    H = torch.ones(1, 1, dtype=torch.float32)
    while H.shape[0] < n:
        top = torch.cat([H, H], dim=1)
        bot = torch.cat([H, -H], dim=1)
        H = torch.cat([top, bot], dim=0)
    return H.to(dtype) / math.sqrt(n)


class DeepseekV32Indexer(nn.Module):
    """Lightning Indexer for DeepSeek V3.2 DSA.

    Selects the top-K most relevant prior positions for each query by scoring
    queries (from the MLA ``qr`` latent) against a dedicated compressed key
    cache.  The selected indices produce a sparse attention mask that is
    OR'd with the standard causal mask in MLA.

    Neuron notes:
    - We skip FP8 and run scores in BF16 (reference does FP8 for speed; BF16 is
      numerically a superset).  Guarded by the flag ``use_fp8_scoring`` for
      future swap-in.
    - ``torch.split`` is banned on Neuron → manual slicing.
    - ``torch.topk`` fails on trn2 (HLO sort unsupported — NCC_EVRF029).  The
      ``_topk`` helper delegates to a pluggable implementation so we can swap
      an NKI kernel in later without touching the forward pass.
    - The indexer's own key cache is NOT block-paged; it's a single flat
      buffer ``[max_batch, max_seq_len, index_head_dim]``, because the indexer
      sees the whole sequence contiguously rather than going through vLLM's
      paged KV cache machinery.
    """

    _STORE_CACHE_BF16: bool = True

    def __init__(
        self,
        config: DeepseekV32Config,
        max_batch_size: int = 1,
        max_seq_len: int | None = None,
    ):
        super().__init__()
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.q_lora_rank = config.q_lora_rank
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.index_topk = config.index_topk

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group
        assert self.n_heads % self.world_size == 0, (
            f"index_n_heads={self.n_heads} not divisible by TP world_size={self.world_size}"
        )
        self.n_local_heads = self.n_heads // self.world_size

        self.softmax_scale = self.head_dim**-0.5

        # --- Parameters ---
        # Zero-init (not empty) so meta→device transfer works when indexer
        # weights are missing from the checkpoint.
        # wq_b: column-parallel on index_n_heads  [q_lora_rank, n_local_heads * head_dim]
        self.wq_b_weight = nn.Parameter(
            torch.zeros(
                self.q_lora_rank, self.n_local_heads * self.head_dim, dtype=self.dtype
            )
        )
        # wk: replicated                          [hidden_size, head_dim]
        self.wk_weight = nn.Parameter(
            torch.zeros(self.hidden_size, self.head_dim, dtype=self.dtype)
        )
        # k_norm: LayerNorm on head_dim (NOT RMSNorm — has bias, mean-subtracts)
        # Stored as fp32 in HF checkpoint and used in fp32 in forward.
        self.k_norm_weight = nn.Parameter(
            torch.ones(self.head_dim, dtype=torch.float32)
        )
        self.k_norm_bias = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32))
        self.k_norm_eps = 1e-6
        # weights_proj: replicated in FP32        [hidden_size, n_heads]
        self.weights_proj_weight = nn.Parameter(
            torch.zeros(self.hidden_size, self.n_heads, dtype=self.dtype)
        )

        self._setup_weight_loaders()

        # ----- end of param creation -----

        # Precomputed Hadamard matrix (constant)
        self.register_buffer(
            "hadamard_matrix",
            _build_hadamard_matrix(self.head_dim, self.dtype),
            persistent=False,
        )

        # --- Indexer's own key cache ---
        # Flat per-sequence buffer. Not managed by vLLM's paged KV cache because
        # the head_dim (128) and usage pattern differ from MLA's compressed KV.
        # A test-time caller (CPU tests) can override ``max_seq_len`` to something
        # smaller to save memory.
        if max_seq_len is None:
            max_seq_len = getattr(config, "dsa_max_seq_len", None) or getattr(
                config, "max_position_embeddings", 4096
            )
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        cache_dtype = torch.bfloat16 if self._STORE_CACHE_BF16 else torch.float32
        self.register_buffer(
            "k_cache",
            torch.zeros(max_batch_size, max_seq_len, self.head_dim, dtype=cache_dtype),
            persistent=False,
        )

        # NKI rotational_topk disabled — produces all-zero indices via
        # torch_neuronx.trace (NKI compiler bug in INTEGRATION mode).
        # See reference/ds32/docs/nki_constant_args_bug.md.
        # Using binary-search threshold selection (_topk_mask) instead.
        #
        # WORKAROUND: still call _setup_nki_topk() because
        # prepare_rotational_constants() has a side effect other
        # nkilib NKI kernels depend on at compile time.
        self._setup_nki_topk()

    def _setup_weight_loaders(self):
        """Indexer weights from HF DeepSeek V3.2 checkpoint.

        Only wq_b is sharded across the TP group (column-parallel on
        index_n_heads). wk, k_norm, and weights_proj are replicated.
        """
        # wq_b: param [q_lora, n_local_heads * head_dim] ← HF [n_heads*head_dim, q_lora]
        set_weight_loader(
            self.wq_b_weight,
            sharded_2d_transposed_loader(
                shard_dim=1,
                shard_size=self.n_local_heads * self.head_dim,
                num_shards=self.world_size,
            ),
        )
        # wk: replicated [hidden_size, head_dim] ← HF [head_dim, hidden_size]
        set_weight_loader(self.wk_weight, transpose_only_loader())
        # weights_proj: replicated [hidden_size, n_heads] ← HF [n_heads, hidden_size]
        set_weight_loader(self.weights_proj_weight, transpose_only_loader())
        # k_norm.weight, k_norm.bias: 1D, identity load — no loader needed.

    def _setup_nki_topk(self):
        try:
            from nkilib.core.topk.rotational_topk import (
                rotational_topk,
                create_topk_config,
                create_rotational_topk_config,
                prepare_rotational_constants,
            )
            import nki.language as _nl
            from vllm_neuron.nki.nki_hop import wrap_nki, kernel_registry

            topk_cfg = create_topk_config(
                inp_shape=(self.max_seq_len, self.max_seq_len),
                inp_dtype=_nl.bfloat16,
                k=self.index_topk,
                sorted=False,
                num_programs=2,
            )
            config = create_rotational_topk_config(
                inp_shape=(self.max_seq_len, self.max_seq_len),
                topk_config=topk_cfg,
            )
            config = prepare_rotational_constants(config)
            wrapped = wrap_nki(rotational_topk)
            const_key = hash((self.max_seq_len, self.max_seq_len, self.index_topk))
            kernel_registry.add_constant_args({"config": config}, const_key)
        except ImportError as e:
            # NKI rotational_topk is optional; the indexer falls back to
            # the binary-search _topk_mask path when nkilib is unavailable.
            logger.warning(
                "DSA indexer NKI rotational_topk unavailable, falling back "
                "to binary-search top-k: %s",
                e,
            )

    def _topk_mask(self, scores: torch.Tensor, k: int) -> torch.Tensor:
        """Return boolean mask [*, V] where True = position is in top-k.

        Uses binary search on threshold to avoid torch.topk (HLO sort banned
        on trn2) and NKI rotational_topk (constant_args bug — see
        reference/ds32/docs/nki_constant_args_bug.md).

        All ops (comparison, sum, where, min, max) are Neuron-safe.
        """
        if scores.device.type == "cpu":
            idx = scores.topk(k, dim=-1).indices
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(-1, idx, True)
            return mask

        flat = scores.float().reshape(-1, scores.shape[-1])  # [N, V], fp32
        fill_val = torch.tensor(-1.0e38, dtype=torch.float32, device=scores.device)
        # Replace -inf with a large negative finite value for min/max.
        # Avoid isfinite() which can produce f64 on XLA.
        safe = torch.where(flat > fill_val, flat, fill_val)

        lo = safe.min(dim=-1, keepdim=True).values  # [N, 1]
        hi = safe.max(dim=-1, keepdim=True).values  # [N, 1]
        k_t = torch.tensor(k, dtype=torch.int32, device=scores.device)
        one = torch.tensor(1, dtype=torch.int32, device=scores.device)
        zero = torch.tensor(0, dtype=torch.int32, device=scores.device)
        half = torch.tensor(0.5, dtype=torch.float32, device=scores.device)

        for _ in range(24):
            mid = (lo + hi) * half
            count = torch.where(flat >= mid, one, zero).sum(dim=-1, keepdim=True)
            above = count > k_t
            lo = torch.where(above, mid, lo)
            hi = torch.where(above, hi, mid)

        threshold = (lo + hi) * half
        mask = flat >= threshold  # [N, V]
        return mask.reshape(scores.shape)

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        cos: torch.Tensor,
        sin: torch.Tensor,
        causal_mask_add: torch.Tensor | None,
    ) -> torch.Tensor:
        """Return ``topk_indices`` of shape ``[B, S, min(index_topk, end_pos)]``.

        Parameters mirror the HF reference ``Indexer.forward`` except we pass
        cos/sin directly instead of HF's complex ``freqs_cis``.

        - ``x``: hidden states [B, S, hidden_size]
        - ``qr``: ``q_norm(wq_a(x))`` from the MLA block [B, S, q_lora_rank]
        - ``start_pos``: the absolute position of the first token of ``x`` in the sequence
        - ``cos``, ``sin``: RoPE cos/sin for positions ``start_pos..start_pos+S``,
          shape [S, rope_head_dim//2] — the NON-interleaved helper expects the
          last dim equal to rope_head_dim//2.
        - ``causal_mask_add``: optional additive mask [B, S, end_pos] to apply
          before top-k selection (−inf on positions to ignore, 0 elsewhere).
          Skip at decode (S=1).
        """
        bsz, S, _ = x.shape
        end_pos = start_pos + S

        # --- Q path ---
        # wq_b: [q_lora, Nh_local*head_dim]
        q = torch.matmul(qr, self.wq_b_weight)
        q = q.view(bsz, S, self.n_local_heads, self.head_dim)
        # Reference ref model.py:462: `q_pe, q_nope = split(q, [rope_head_dim, head_dim - rope_head_dim])`
        # → q_pe is the first `rope_head_dim` channels, q_nope is the rest.
        # torch.split is banned on Neuron → manual slice.
        q_pe = q[..., : self.rope_head_dim]
        q_nope = q[..., self.rope_head_dim :]
        # Non-interleaved RoPE on q_pe.  cos/sin shape [S, rope_head_dim//2] →
        # broadcast to [1, S, 1, rope_head_dim//2].
        cos_b = cos.view(1, S, 1, self.rope_head_dim // 2)
        sin_b = sin.view(1, S, 1, self.rope_head_dim // 2)
        q_pe = _apply_rotary_emb_non_interleaved(q_pe, cos_b, sin_b)
        q = torch.cat([q_pe, q_nope], dim=-1)

        # --- K path ---
        k = torch.matmul(x, self.wk_weight)  # [B, S, head_dim]
        # LayerNorm in fp32 (matches HF reference ref model.py:320-321)
        k = torch.nn.functional.layer_norm(
            k.float(),
            (self.head_dim,),
            self.k_norm_weight.float(),
            self.k_norm_bias.float(),
            self.k_norm_eps,
        ).to(k.dtype)
        k_pe = k[..., : self.rope_head_dim]  # [B, S, rope]
        k_nope = k[..., self.rope_head_dim :]  # [B, S, head_dim - rope]
        # Non-interleaved RoPE on k_pe.  The reference treats k as single-head,
        # so shape is [B, S, 1, rope_head_dim] after unsqueeze.
        k_pe_r = _apply_rotary_emb_non_interleaved(
            k_pe.unsqueeze(2),
            cos_b,
            sin_b,
        ).squeeze(2)
        k = torch.cat([k_pe_r, k_nope], dim=-1)  # [B, S, head_dim]

        # --- Hadamard transform ---
        q = torch.matmul(q, self.hadamard_matrix)  # scale baked into matrix
        k = torch.matmul(k, self.hadamard_matrix)

        # --- Store k into the indexer's own cache ---
        # No FP8 yet — BF16 storage.
        self.k_cache[:bsz, start_pos:end_pos] = k.to(self.k_cache.dtype)
        # Read back the whole prefix [0, end_pos) for scoring.
        k_all = self.k_cache[:bsz, :end_pos].to(q.dtype)  # [B, end_pos, head_dim]

        # --- Per-head weights (replicated projection) ---
        # weights_proj runs in FP32 on the reference. [B, S, n_heads]
        weights = torch.matmul(x.float(), self.weights_proj_weight.float())
        weights = weights * (self.n_heads**-0.5)  # [B, S, n_heads]
        # BF16 fallback: the reference multiplies per-block FP8 q_scale into
        # weights before the FP8 matmul.  Without FP8 quantization we simply
        # scale weights by softmax_scale.
        weights = weights.unsqueeze(-1) * self.softmax_scale  # [B, S, n_heads, 1]

        # --- Score matmul (BF16 fallback for fp8_index) ---
        # Ref fp8_index kernel semantics, distilled from context:
        #   index_score[b, s, t] = sum_h weights[b, s, h] * <q[b, s, h], k[b, t]>
        # (Per-head dot product, weighted-summed across heads to produce a
        # single score per (query-position, key-position) pair. Note the
        # indexer has its OWN heads separate from MLA heads.)
        # We compute this in BF16:
        #   raw_scores [B, S, n_local_heads, end_pos] = einsum('bshd,btd->bsht', q, k_all)
        #   index_score [B, S, end_pos] = sum over h of weights[..., h_global, :] * raw_scores[..., h_local, :]
        raw_scores = torch.einsum(
            "bshd,btd->bsht", q, k_all
        )  # [B, S, n_local_heads, end_pos]

        # Select the per-rank slice of weights to match local heads.
        if self.world_size > 1:
            h_start = self.rank * self.n_local_heads
            h_end = h_start + self.n_local_heads
            weights_local = weights[..., h_start:h_end, :]  # [B, S, n_local_heads, 1]
        else:
            weights_local = weights  # [B, S, n_heads, 1]
        index_score_local = (raw_scores.float() * weights_local.float()).sum(
            dim=-2
        )  # [B, S, end_pos]

        # Sum-reduce across ranks so every rank has the same score → same topk.
        if self.world_size > 1:
            self.tp_group.all_reduce(index_score_local)
        index_score = index_score_local  # [B, S, end_pos], fp32

        if causal_mask_add is not None:
            index_score = index_score + causal_mask_add.to(index_score.dtype)

        k_topk = min(self.index_topk, end_pos)
        return self._topk_mask(index_score, k_topk)

    def reset_cache(self) -> None:
        """Zero the key cache. Useful between inference runs/tests."""
        self.k_cache.zero_()


# =============================================================================
# Section 3: MLA Attention
# Phase 1: Dense attention (no DSA Indexer)
# =============================================================================


class DeepseekV32Attention(nn.Module):
    """Multi-Head Latent Attention with compressed KV cache.

    MLA compresses KV into a low-rank representation (kv_lora_rank=512).
    Uses weight absorption to avoid materializing full K/V tensors.

    KV cache stores [k_pe | compressed_kv] = 576 dims with num_kv_heads=1.
    """

    def __init__(self, config: DeepseekV32Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim
        self.max_seq_len = config.max_position_embeddings

        # DeepSeek V3.2 YaRN: softmax_scale is scaled by mscale² when
        # mscale_all_dim is set, matching HF `modeling_deepseek.py` lines 689-695.
        # mscale_all_dim is the rope_scaling field used for the *attention* mscale
        # correction (separate from the cos/sin mscale, which in this config is 1.0).
        # Without this factor our attention logits are compressed by 1/mscale²
        # (≈1/1.874 for the default config) — producing too-flat softmax distributions.
        self.softmax_scale = self.qk_head_dim**-0.5
        rope_scaling = getattr(config, "rope_scaling", None) or {}
        mscale_all_dim = rope_scaling.get("mscale_all_dim", 0)
        scaling_factor = rope_scaling.get("factor", 1.0)
        if mscale_all_dim and scaling_factor > 1.0:
            mscale = 0.1 * mscale_all_dim * math.log(scaling_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.num_heads_per_rank = self.num_heads // self.world_size

        # Query LoRA: hidden → q_lora_rank → num_heads * qk_head_dim
        self.wq_a_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.q_lora_rank, dtype=self.dtype)
        )
        self.q_norm = DeepseekV32RMSNorm(self.q_lora_rank, dtype=self.dtype)
        q_out_size = self.num_heads_per_rank * self.qk_head_dim
        self.wq_b_weight = nn.Parameter(
            torch.empty(self.q_lora_rank, q_out_size, dtype=self.dtype)
        )

        # KV compression: hidden → kv_lora_rank + qk_rope_head_dim
        kv_a_out = self.kv_lora_rank + self.qk_rope_head_dim
        self.wkv_a_weight = nn.Parameter(
            torch.empty(self.hidden_size, kv_a_out, dtype=self.dtype)
        )
        self.kv_norm = DeepseekV32RMSNorm(self.kv_lora_rank, dtype=self.dtype)

        # KV expansion: kv_lora_rank → num_heads * (qk_nope_head_dim + v_head_dim)
        wkv_b_out = self.num_heads_per_rank * (self.qk_nope_head_dim + self.v_head_dim)
        self.wkv_b_weight = nn.Parameter(
            torch.empty(self.kv_lora_rank, wkv_b_out, dtype=self.dtype)
        )

        # Output projection
        o_proj_in = self.num_heads_per_rank * self.v_head_dim
        self.o_proj_weight = nn.Parameter(
            torch.empty(o_proj_in, self.hidden_size, dtype=self.dtype)
        )

        # KV cache: bound externally via bind_kv_cache
        self.k_cache = None
        self.v_cache = None

        # DSA Lightning Indexer (Phase 2). Off by default.
        self.use_dsa = getattr(config, "use_dsa", False)
        if self.use_dsa:
            self.indexer = DeepseekV32Indexer(config)
        else:
            self.indexer = None

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """Attach SafetensorsWeightLoader to each parameter so the unsharded
        HF checkpoint can be loaded directly via load_sharded_pipelined.

        HF stores 2D weights as nn.Linear `[out, in]`; our matmul-style params
        are `[in, out]`, so most loaders pass `is_storage_transposed=True`.
        Replicated params (q_a/kv_a/norms) get no loader — identity load is
        cheap because they're small (q_lora_rank=1536, kv_lora_rank=512).
        """
        # wq_b: [q_lora, Nh_per_rank * qk_head_dim] ← HF [Nh*qk_head, q_lora]
        set_weight_loader(
            self.wq_b_weight,
            sharded_2d_transposed_loader(
                shard_dim=1,
                shard_size=self.num_heads_per_rank * self.qk_head_dim,
                num_shards=self.world_size,
            ),
        )
        # wkv_b: [kv_lora, Nh_per_rank * (qk_nope+v_head)] ← HF [Nh*(qk_nope+v_head), kv_lora]
        set_weight_loader(
            self.wkv_b_weight,
            sharded_2d_transposed_loader(
                shard_dim=1,
                shard_size=self.num_heads_per_rank
                * (self.qk_nope_head_dim + self.v_head_dim),
                num_shards=self.world_size,
            ),
        )
        # o_proj: [Nh_per_rank * v_head, H] ← HF [H, Nh*v_head]
        set_weight_loader(
            self.o_proj_weight,
            sharded_2d_transposed_loader(
                shard_dim=0,
                shard_size=self.num_heads_per_rank * self.v_head_dim,
                num_shards=self.world_size,
            ),
        )
        # wq_a, wkv_a: replicated. HF stores as [out, in] = transposed vs our
        # [in, out]. We need a transpose-only loader so the param shape lines up.
        set_weight_loader(self.wq_a_weight, transpose_only_loader())
        set_weight_loader(self.wkv_a_weight, transpose_only_loader())

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
    ) -> torch.Tensor:
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]
        is_prefill = max_query_len > decode_token_threshold

        if is_prefill:
            # SP: all-gather hidden states before prefill attention
            if self.world_size > 1:
                hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
            return self.forward_prefill(
                hidden_states, positions, position_embeddings, attn_metadata
            )
        else:
            return self.forward_decode(
                hidden_states, positions, position_embeddings, attn_metadata
            )

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
    ) -> torch.Tensor:
        """MLA prefill with weight absorption in BHSD layout (matching NxDI).

        Uses B=1 BHSD tensor layout and matmul/einsum patterns identical to
        NxDI's DeepseekV3Attention.forward to produce the same XLA/HLO graph.
        """
        cos, sin = position_embeddings
        B = 1
        T = hidden_states.shape[0]
        Nh = self.num_heads_per_rank

        # Weight absorption setup (same decomposition as decode path)
        D_per_head = self.qk_nope_head_dim + self.v_head_dim
        wkv_b = (
            self.wkv_b_weight.t().contiguous().view(Nh, D_per_head, self.kv_lora_rank)
        )
        q_absorb = wkv_b[:, : self.qk_nope_head_dim]
        out_absorb = wkv_b[:, self.qk_nope_head_dim :]

        # Query path: hidden → wq_a → q_norm → wq_b
        qr = self.q_norm(torch.matmul(hidden_states, self.wq_a_weight))
        q = torch.matmul(qr, self.wq_b_weight)
        # Reshape to BHSD: [T, Nh*D] → [B, T, Nh, D] → [B, Nh, T, D]
        q = q.view(B, T, Nh, self.qk_head_dim).transpose(1, 2)

        # KV path: hidden → wkv_a → slice compressed_kv/k_pe → kv_norm → RoPE
        # NOTE: torch.split compiles incorrectly on Neuron — use manual slicing
        kv_out = torch.matmul(hidden_states, self.wkv_a_weight)
        compressed_kv = kv_out[..., : self.kv_lora_rank]
        k_pe = kv_out[..., self.kv_lora_rank :]
        compressed_kv = self.kv_norm(compressed_kv)

        # Split Q into nope and pe components (manual slice, not torch.split)
        q_nope = q[..., : self.qk_nope_head_dim]
        q_pe = q[..., self.qk_nope_head_dim :]

        # RoPE on q_pe [B, Nh, T, rope] and k_pe [T, rope] → [B, 1, T, rope]
        k_pe = k_pe.view(B, T, 1, self.qk_rope_head_dim).transpose(1, 2)
        q_pe = _apply_rotary_emb_interleaved(
            q_pe, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
        )
        k_pe = _apply_rotary_emb_interleaved(
            k_pe, cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)
        )

        # Store in KV cache via slot_mapping (same as decode path and gpt_oss)
        layer_name = f"layers.{self.layer_idx}.self_attn"
        if self.k_cache is not None and layer_name in attn_metadata:
            slot_mapping = attn_metadata[layer_name].get("slot_mapping")
            if slot_mapping is not None:
                block_size = self.k_cache.shape[2]
                block_indices = slot_mapping // block_size
                position_indices = slot_mapping % block_size
                k_pe_cache = k_pe.squeeze(1).squeeze(0)  # [T, rope]
                combined = torch.cat([k_pe_cache, compressed_kv], dim=-1)
                head_zeros = torch.zeros_like(block_indices)
                self.k_cache.index_put_(
                    (block_indices, head_zeros, position_indices),
                    combined.to(self.k_cache.dtype),
                )
                self.v_cache.index_put_(
                    (block_indices, head_zeros, position_indices),
                    combined.to(self.v_cache.dtype),
                )

        # Q absorption (NxDI pattern): q_nope [B,Nh,T,nope] @ q_absorb [Nh,nope,kv_lora] → [B,Nh,T,kv_lora]
        q_nope = torch.einsum("hdc,bhqd->bhqc", q_absorb, q_nope)

        # Attention scores in BHTS layout (NxDI pattern)
        # PE: matmul(q_pe [B,Nh,T,rope], k_pe.T [B,1,rope,T]) → [B,Nh,T,T]
        # Nope: einsum('bhqc,blc->bhql', q_nope [B,Nh,T,kv_lora], compressed_kv [B,T,kv_lora]) → [B,Nh,T,T]
        compressed_kv_b = compressed_kv.unsqueeze(0)  # [B, T, kv_lora]
        scores = (
            torch.matmul(q_pe, k_pe.transpose(2, 3))
            + torch.einsum("bhqc,blc->bhql", q_nope, compressed_kv_b)
        ) * self.softmax_scale

        # Causal mask in BHTS layout (NxDI: torch.where with finfo.min)
        causal_mask = torch.tril(
            torch.ones(T, T, device=scores.device, dtype=torch.bool)
        )
        scores = torch.where(causal_mask, scores, torch.finfo(scores.dtype).min)

        # DSA Lightning Indexer (Phase 2). Selects up to index_topk positions to
        # attend to.  When T <= index_topk the selection is the full prefix so
        # the mask is all zeros — skip the indexer entirely (T13a proves this
        # is mathematically identical and it avoids compiling the iterative
        # argmax loop when it would be a no-op).
        if self.indexer is not None and T > self.indexer.index_topk:
            hidden_states_b = hidden_states.unsqueeze(0)  # [B=1, T, H]
            qr_b = qr.unsqueeze(0)  # [B=1, T, q_lora]
            causal_add = torch.zeros(B, T, T, dtype=torch.float32, device=scores.device)
            causal_add.masked_fill_(~causal_mask, float("-inf"))
            topk_mask = self.indexer(
                hidden_states_b,
                qr_b,
                start_pos=0,
                cos=cos,
                sin=sin,
                causal_mask_add=causal_add,
            )  # [B, T, T], bool — True = position selected
            scores = scores.masked_fill(
                ~topk_mask.unsqueeze(1),
                torch.finfo(scores.dtype).min,
            )

        scores = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).to(
            self.dtype
        )

        # Attend in compressed space then V-absorb (NxDI pattern)
        # scores [B,Nh,T,T] × compressed_kv [B,T,kv_lora] → [B,Nh,T,kv_lora]
        x = torch.einsum("bhql,blc->bhqc", scores, compressed_kv_b)
        # V absorption: x [B,Nh,T,kv_lora] × out_absorb [Nh,v,kv_lora] → [B,Nh,T,v]
        attn_output = torch.einsum("bhqc,hdc->bhqd", x, out_absorb)

        # BHSD → BSHD → [T, Nh*v]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(T, Nh * self.v_head_dim)

        # Output projection
        output = torch.matmul(attn_output, self.o_proj_weight)

        # SP: reduce-scatter back to SP-partitioned layout
        if self.world_size > 1:
            output = self.tp_group.reduce_scatter(output, dim=0).contiguous()

        return output

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
    ) -> torch.Tensor:
        """MLA decode with weight absorption and paged KV cache.

        Following the NxDI V3 decode pattern:
        1. Compute q_nope, q_pe for current token
        2. Absorb q_nope via wkv_b weight matrix
        3. Compute active scores from current KV
        4. Compute prior scores from cached KV (via block_table gather)
        5. Combined softmax + V absorption via out_absorb
        """
        cos, sin = position_embeddings
        T = hidden_states.shape[0]  # number of decode tokens
        layer_name = f"layers.{self.layer_idx}.self_attn"

        # Query path
        qr = self.q_norm(torch.matmul(hidden_states, self.wq_a_weight))
        q = torch.matmul(qr, self.wq_b_weight)
        q = q.view(T, self.num_heads_per_rank, self.qk_head_dim)
        q_nope = q[..., : self.qk_nope_head_dim]
        q_pe = q[..., self.qk_nope_head_dim :]
        q_pe = _apply_rotary_emb_interleaved(q_pe, cos.unsqueeze(1), sin.unsqueeze(1))

        # KV path for current token
        kv_out = torch.matmul(hidden_states, self.wkv_a_weight)
        compressed_kv = kv_out[..., : self.kv_lora_rank]
        k_pe = kv_out[..., self.kv_lora_rank :]
        compressed_kv = self.kv_norm(compressed_kv)
        k_pe = _apply_rotary_emb_interleaved(
            k_pe.unsqueeze(1), cos.unsqueeze(1), sin.unsqueeze(1)
        )
        k_pe = k_pe.squeeze(1)

        # Store current token in KV cache
        if self.k_cache is not None and layer_name in attn_metadata:
            slot_mapping = attn_metadata[layer_name].get("slot_mapping")
            if slot_mapping is not None:
                block_size = self.k_cache.shape[2]
                block_indices = slot_mapping // block_size
                position_indices = slot_mapping % block_size
                combined = torch.cat([k_pe, compressed_kv], dim=-1)
                head_zeros = torch.zeros_like(block_indices)
                self.k_cache.index_put_(
                    (block_indices, head_zeros, position_indices),
                    combined.to(self.k_cache.dtype),
                )
                self.v_cache.index_put_(
                    (block_indices, head_zeros, position_indices),
                    combined.to(self.v_cache.dtype),
                )

        # Weight absorption. wkv_b_weight is stored [kv_lora, Nh*(qk_nope+v)]
        # (HF's nn.Linear.weight.t()). To get per-head layout we must transpose
        # first so memory is laid out as [Nh*(qk_nope+v), kv_lora], then view.
        # Doing `self.wkv_b_weight.view(Nh, -1, kv_lora)` directly scrambles
        # the weight matrix and produces wrong q_absorb / out_absorb (verified:
        # the buggy view strides by `kv_lora` instead of `1` per column).
        D_per_head = self.qk_nope_head_dim + self.v_head_dim
        wkv_b = (
            self.wkv_b_weight.t()
            .contiguous()
            .view(self.num_heads_per_rank, D_per_head, self.kv_lora_rank)
        )
        q_absorb = wkv_b[:, : self.qk_nope_head_dim]
        out_absorb = wkv_b[:, self.qk_nope_head_dim :]

        # Absorb q_nope: [T, Nh, qk_nope] @ [Nh, qk_nope, kv_lora] → [T, Nh, kv_lora]
        q_nope_absorbed = torch.einsum("thd,hdc->thc", q_nope, q_absorb)

        # Gather prior KV from paged cache using block_table
        # k_cache shape: [num_blocks, 1, block_size, 576]
        # block_table shape: [B, max_blocks_per_seq]
        block_table = attn_metadata[layer_name].get("block_table_tensor")
        if block_table is not None and self.k_cache is not None:
            B = block_table.shape[0]
            max_blocks = block_table.shape[1]
            block_size = self.k_cache.shape[2]
            S_prior = max_blocks * block_size

            # Gather blocks: [B * max_blocks, 1, block_size, 576]
            flat_indices = block_table.reshape(-1).long()
            gathered = self.k_cache[flat_indices]  # [B*max_blocks, 1, block_size, 576]
            # Reshape to [B, S_prior, 576]
            prior_kv = gathered.squeeze(1).reshape(
                B, S_prior, self.kv_lora_rank + self.qk_rope_head_dim
            )
        else:
            # No cache available — return zero output
            output = torch.zeros(
                T,
                self.hidden_size,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            return output

        # Split prior cache into k_pe and compressed_kv
        k_pe_prior = prior_kv[..., : self.qk_rope_head_dim]  # [B, S_prior, rope_dim]
        compressed_kv_prior = prior_kv[
            ..., self.qk_rope_head_dim :
        ]  # [B, S_prior, kv_lora]

        # Reshape q for batch: [T, Nh, D] → [B, Nh, T/B, D] (T=B for decode)
        q_pe_b = q_pe.view(
            B, -1, self.num_heads_per_rank, self.qk_rope_head_dim
        ).transpose(1, 2)
        q_nope_b = q_nope_absorbed.view(
            B, -1, self.num_heads_per_rank, self.kv_lora_rank
        ).transpose(1, 2)

        # Prior scores: [B, Nh, T/B, S_prior]
        scores_pe = torch.matmul(q_pe_b, k_pe_prior.unsqueeze(1).transpose(-2, -1))
        scores_nope = torch.einsum("bhqc,bsc->bhqs", q_nope_b, compressed_kv_prior)
        prior_scores = (scores_pe + scores_nope) * self.softmax_scale

        # Mask unfilled cache slots: positions >= current_pos are invalid.
        # positions is [T] with T=B for decode; each entry is the sequence
        # position of the decode token. Valid prior slots are 0..pos-1.
        slot_indices = torch.arange(S_prior, device=hidden_states.device).view(
            1, 1, 1, S_prior
        )
        valid_len = positions.view(B, 1, 1, 1)  # [B, 1, 1, 1]
        padding_mask = slot_indices >= valid_len  # True for invalid slots
        prior_scores = prior_scores.masked_fill(
            padding_mask, torch.finfo(prior_scores.dtype).min
        )

        # Active scores (current token attending to itself)
        active_scores = (
            torch.matmul(
                q_pe_b,
                k_pe.view(B, -1, 1, self.qk_rope_head_dim)
                .transpose(1, 2)
                .transpose(-2, -1),
            )
            + torch.einsum(
                "bhqc,bsc->bhqs", q_nope_b, compressed_kv.view(B, -1, self.kv_lora_rank)
            )
        ) * self.softmax_scale

        # Combined softmax over [prior | active]
        all_scores = torch.cat([prior_scores, active_scores], dim=-1)
        all_scores = torch.softmax(all_scores, dim=-1, dtype=torch.float32).to(
            self.dtype
        )

        prior_weights = all_scores[..., :S_prior]
        active_weights = all_scores[..., S_prior:]

        # V absorption: scores @ compressed_kv → wkv_b out_absorb → output
        x_prior = torch.einsum("bhqs,bsc->bhqc", prior_weights, compressed_kv_prior)
        x_active = torch.einsum(
            "bhqs,bsc->bhqc",
            active_weights,
            compressed_kv.view(B, -1, self.kv_lora_rank),
        )
        x = x_prior + x_active

        attn_output = torch.einsum("bhqc,hdc->bhqd", x, out_absorb)
        attn_output = attn_output.transpose(1, 2).reshape(
            T, self.num_heads_per_rank * self.v_head_dim
        )

        output = torch.matmul(attn_output, self.o_proj_weight)

        if self.world_size > 1:
            self.tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 4: Dense MLP (layers 0-2)
# =============================================================================


class DeepseekV32DenseMLP(nn.Module):
    """SiLU-gated MLP for the first 3 layers (before MoE)."""

    def __init__(self, config: DeepseekV32Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.intermediate_per_rank = self.intermediate_size // self.world_size

        self.gate_proj_weight = nn.Parameter(
            torch.empty(
                self.hidden_size, self.intermediate_per_rank, dtype=config.torch_dtype
            )
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(
                self.hidden_size, self.intermediate_per_rank, dtype=config.torch_dtype
            )
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(
                self.intermediate_per_rank, self.hidden_size, dtype=config.torch_dtype
            )
        )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        # gate/up: [H, I_per_rank] ← HF [I, H]; shard intermediate dim
        gate_up_loader = sharded_2d_transposed_loader(
            shard_dim=1,
            shard_size=self.intermediate_per_rank,
            num_shards=self.world_size,
        )
        set_weight_loader(self.gate_proj_weight, gate_up_loader)
        set_weight_loader(self.up_proj_weight, gate_up_loader)
        # down: [I_per_rank, H] ← HF [H, I]; shard intermediate dim (dim 0 in param)
        set_weight_loader(
            self.down_proj_weight,
            sharded_2d_transposed_loader(
                shard_dim=0,
                shard_size=self.intermediate_per_rank,
                num_shards=self.world_size,
            ),
        )

    def forward(
        self, hidden_states: torch.Tensor, is_decode: bool = True
    ) -> torch.Tensor:
        # In prefill, input arrives SP-sharded [T/ws, H] (attention ended in
        # reduce_scatter). all_gather → full [T, H] → compute → reduce_scatter
        # back to [T/ws, H]. Row-parallel all_reduce in this path would sum
        # different token slices across ranks (garbage). Same pattern as MoE
        # and gpt_oss. Decode is not SP-sharded — all_reduce is correct there.
        sp_prefill = (self.world_size > 1) and (not is_decode)
        if sp_prefill:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        gate = torch.matmul(hidden_states, self.gate_proj_weight)
        up = torch.matmul(hidden_states, self.up_proj_weight)
        hidden = torch.nn.functional.silu(gate) * up
        output = torch.matmul(hidden, self.down_proj_weight)

        if sp_prefill:
            output = self.tp_group.reduce_scatter(output, dim=0).contiguous()
        elif self.world_size > 1:
            self.tp_group.all_reduce(output)

        return output


# =============================================================================
# Section 5: MoE (256 routed experts + 1 shared expert)
# =============================================================================


class DeepseekV3Router(nn.Module):
    """Group-limited sigmoid router with e_score_correction_bias (DeepSeek V3/V3.2 noaux_tc).

    Matches HF `MoEGate.forward`:
      logits = Linear(x, W)               fp32
      scores = sigmoid(logits)
      scores_for_choice = scores + bias   # bias AFTER sigmoid
      group_scores = top2-sum per group on scores_for_choice
      top groups → mask non-top-group experts
      topk indices via scores_for_choice (masked)
      topk weights via scores.gather (pre-bias)
      L1 normalize, × routed_scaling_factor
    """

    def __init__(self, config: DeepseekV32Config):
        super().__init__()
        self.router_weight = nn.Parameter(
            torch.empty(
                config.n_routed_experts, config.hidden_size, dtype=config.torch_dtype
            )
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.zeros(config.n_routed_experts, dtype=torch.float32)
        )
        self.n_routed_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.n_group = config.n_expert_groups
        self.topk_group = config.n_limited_groups
        self.routed_scaling_factor = config.routed_scaling_factor
        self.experts_per_group = config.n_routed_experts // config.n_expert_groups

    @staticmethod
    def _iter_topk_values(x: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
        """Iterative top-k values via k successive amax + scatter masks.

        Neuron's trn2 compiler rejects the HLO `sort` op that XLA emits for
        `torch.topk(..., sorted=False)` on shapes like `[T, 8, 32]` with small k
        (NCC_EVRF029). This replacement uses `k` `amax`+`argmax` passes and a
        scatter-based mask, which lower to supported ops.

        Returns a tensor with shape `x.shape` but `dim` replaced by size `k`.
        Values are in descending order (rank 0 = largest).
        """
        vals = []
        cur = x
        for _ in range(k):
            v = cur.amax(dim=dim, keepdim=True)
            idx = cur.argmax(dim=dim, keepdim=True)
            vals.append(v)
            # Mask out the just-picked position with -inf so next amax finds
            # the next largest. Using `-inf` keeps the algorithm correct even
            # when the input is already masked.
            neg_inf = torch.full_like(v, float("-inf"))
            cur = cur.scatter(dim, idx, neg_inf)
        return torch.cat(vals, dim=dim)

    @staticmethod
    def _iter_topk_indices(x: torch.Tensor, k: int, dim: int = -1) -> torch.Tensor:
        """Same as _iter_topk_values but returns indices (in descending order).

        Returns a tensor with shape `x.shape` but `dim` replaced by size `k`,
        dtype `int64`.
        """
        idxs = []
        cur = x
        for _ in range(k):
            idx = cur.argmax(dim=dim, keepdim=True)
            idxs.append(idx)
            # Scatter -inf at the picked position so we don't re-select it.
            v = cur.gather(dim, idx)
            neg_inf = torch.full_like(v, float("-inf"))
            cur = cur.scatter(dim, idx, neg_inf)
        return torch.cat(idxs, dim=dim)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # hidden_states: [T, H] (already RMSNormed by caller)
        T = hidden_states.shape[0]
        logits = torch.nn.functional.linear(
            hidden_states.float(), self.router_weight.float()
        )  # [T, E]
        scores = logits.sigmoid()  # [T, E]
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)

        # Three topk calls below are replaced with iterative amax/argmax. On
        # trn2 the XLA lowering of `torch.topk` for shapes like `[T, 8, 32]`
        # emits HLO `sort`, which the Neuron compiler rejects (NCC_EVRF029).
        # The iterative form uses `amax` + `argmax` + `scatter`, all supported.
        #
        # 1) Sum of top-2 per group  (over [T, 8, 32], k=2/32 → sort in XLA)
        scores_grouped = scores_for_choice.view(
            T, self.n_group, self.experts_per_group
        )  # [T, G, E/G]
        top2_vals = self._iter_topk_values(scores_grouped, k=2, dim=-1)  # [T, G, 2]
        group_scores = top2_vals.sum(dim=-1)  # [T, G]

        # 2) Top-{topk_group} groups  (over [T, 8], k=4/8 → also sort in XLA)
        group_idx = self._iter_topk_indices(
            group_scores, k=self.topk_group, dim=-1
        )  # [T, topk_group]

        group_mask = torch.zeros(
            T, self.n_group, device=hidden_states.device, dtype=scores.dtype
        )
        group_mask.scatter_(1, group_idx, 1.0)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, -1, self.experts_per_group)
            .reshape(T, -1)
        )  # [T, E]
        tmp_scores = scores_for_choice.masked_fill(score_mask == 0, float("-inf"))

        # 3) Final top-{top_k} experts (over [T, 256], k=8/256)
        #    k/n is small (~3%) so XLA should emit TopK here, but we use the
        #    iterative form for consistency and to avoid any surprise.
        topk_idx = self._iter_topk_indices(
            tmp_scores, k=self.top_k, dim=-1
        )  # [T, top_k]

        topk_weight = scores.gather(1, topk_idx)  # pre-bias scores
        topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_weight, topk_idx


class DeepseekV32MoE(nn.Module):
    """Mixture of Experts with group-limited sigmoid routing.

    256 routed experts (8 groups × 32), top-4 groups → top-8 experts total.
    1 shared expert always active.
    Sigmoid activation with e_score_correction_bias and L1 normalization.
    """

    def __init__(self, config: DeepseekV32Config):
        super().__init__()
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        # >>> PARALLELISM: TP + EP configuration <<<
        # EP splits the TP group into ep_degree sub-groups of size tp_degree.
        # When EP is enabled, expert weights are partitioned across EP ranks
        # (linear placement) and intermediate dim is sharded across tp_degree.
        # When EP is disabled (ep_degree=1), this matches the original behavior
        # (all experts on every rank, intermediate sharded across world_size).
        self.ep_degree = 1
        self.ep_rank = 0
        self.tp_degree = self.world_size
        self.ep_tp_group = self.tp_group
        try:
            from vllm.config import get_current_vllm_config

            vllm_config = get_current_vllm_config()
            if (
                vllm_config is not None
                and vllm_config.parallel_config.enable_expert_parallel
            ):
                from vllm_neuron.parallel.neuron_parallel_state import (
                    get_neuron_ep_degree,
                    get_neuron_ep_rank,
                    get_neuron_ep_tp_group,
                )

                self.ep_degree = get_neuron_ep_degree()
                self.ep_rank = get_neuron_ep_rank()
                self.tp_degree = self.world_size // self.ep_degree
                self.ep_tp_group = get_neuron_ep_tp_group()
        except Exception:
            # CPU tests / non-vLLM contexts: fall back to ep_degree=1.
            pass

        # moe_group is the full TP group — the final reduce-scatter/all-reduce
        # after MoE must aggregate across all ranks (both TP and EP).
        self.moe_group = self.tp_group

        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.n_expert_groups = config.n_expert_groups
        self.n_limited_groups = config.n_limited_groups
        self.routed_scaling_factor = config.routed_scaling_factor
        self.moe_intermediate_size = config.moe_intermediate_size

        # >>> PARALLELISM: EP linear placement — rank k owns experts [k*L, (k+1)*L) <<<
        self.num_local_experts = self.n_routed_experts // self.ep_degree
        self.moe_inter_per_rank = self.moe_intermediate_size // self.tp_degree
        moe_inter_per_rank = self.moe_inter_per_rank

        # Router (group-limited noaux_tc). Owns `router_weight` and
        # `e_score_correction_bias` parameters. Replicated on all ranks.
        self.router = DeepseekV3Router(config)

        # Routed experts: gate_up fused [E_local, H, 2*I_per_rank], down [E_local, I_per_rank, H]
        self.gate_up_proj_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.hidden_size,
                moe_inter_per_rank * 2,
                dtype=config.torch_dtype,
            )
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                moe_inter_per_rank,
                self.hidden_size,
                dtype=config.torch_dtype,
            )
        )

        # Shared expert: sharded by the FULL TP group (world_size), NOT
        # tp_degree. This matches attention + dense MLP (both shard by
        # world_size) and makes the final moe_group.all_reduce sum a single
        # set of 64 TP partials — no ep_degree divide needed. The MoE routed
        # path is different (EP-partitioned experts × tp_degree-sharded
        # intermediate), but the shared expert is not EP-specific so we keep
        # it on the attention/dense MLP sharding axis.
        shared_inter = config.n_shared_experts * self.moe_intermediate_size
        self.shared_inter_per_rank = shared_inter // self.world_size
        shared_inter_per_rank = self.shared_inter_per_rank
        self.shared_gate_proj_weight = nn.Parameter(
            torch.empty(
                self.hidden_size, shared_inter_per_rank, dtype=config.torch_dtype
            )
        )
        self.shared_up_proj_weight = nn.Parameter(
            torch.empty(
                self.hidden_size, shared_inter_per_rank, dtype=config.torch_dtype
            )
        )
        self.shared_down_proj_weight = nn.Parameter(
            torch.empty(
                shared_inter_per_rank, self.hidden_size, dtype=config.torch_dtype
            )
        )

        # Post-attention layernorm weight (fused into moe_block_tkg).
        # Stored fp32 in HF checkpoint; used in fp32 in forward.
        self.post_attn_norm_weight = nn.Parameter(
            torch.ones(self.hidden_size, dtype=torch.float32)
        )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self):
        """Attach weight loaders for MoE params.

        Supports EP via the standard ``expert_parallel_weight_loader`` wrapper
        (matches qwen3_moe). The base loader reads all ``n_routed_experts``
        per-expert slices from safetensors and TP-shards each on the
        intermediate dim; the EP wrapper then keeps only this rank's
        experts on dim 0. Result shape: ``[num_local_experts, ..., I_per_rank]``.
        """
        # >>> PARALLELISM: EP linear placement — rank k owns experts [k*L, (k+1)*L) <<<
        local_expert_indices = list(
            range(
                self.ep_rank * self.num_local_experts,
                (self.ep_rank + 1) * self.num_local_experts,
            )
        )

        def _maybe_ep_wrap(loader):
            if self.ep_degree > 1:
                return expert_parallel_weight_loader(local_expert_indices, loader)
            return loader

        # gate_up_proj_weight: TP shard on intermediate dim, then EP-filter experts
        set_weight_loader(
            self.gate_up_proj_weight,
            _maybe_ep_wrap(
                moe_gate_up_loader(
                    hidden_size=self.hidden_size,
                    moe_inter_per_rank=self.moe_inter_per_rank,
                    num_shards=self.tp_degree,
                    num_experts=self.n_routed_experts,
                ),
            ),
        )
        # down_proj_weight: TP shard on intermediate dim, then EP-filter experts
        set_weight_loader(
            self.down_proj_weight,
            _maybe_ep_wrap(
                moe_down_loader(
                    hidden_size=self.hidden_size,
                    moe_inter_per_rank=self.moe_inter_per_rank,
                    num_shards=self.tp_degree,
                    num_experts=self.n_routed_experts,
                ),
            ),
        )

        # Shared expert (sharded by full world_size, transposed storage)
        shared_gate_up_loader = sharded_2d_transposed_loader(
            shard_dim=1,
            shard_size=self.shared_inter_per_rank,
            num_shards=self.world_size,
        )
        set_weight_loader(self.shared_gate_proj_weight, shared_gate_up_loader)
        set_weight_loader(self.shared_up_proj_weight, shared_gate_up_loader)
        set_weight_loader(
            self.shared_down_proj_weight,
            sharded_2d_transposed_loader(
                shard_dim=0,
                shard_size=self.shared_inter_per_rank,
                num_shards=self.world_size,
            ),
        )
        # post_attn_norm_weight: replicated [H], identity load — no loader needed.

    def _shared_expert(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = torch.matmul(hidden_states, self.shared_gate_proj_weight)
        up = torch.matmul(hidden_states, self.shared_up_proj_weight)
        hidden = torch.nn.functional.silu(gate) * up
        return torch.matmul(hidden, self.shared_down_proj_weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        is_decode: bool,
        rank: torch.Tensor,
    ) -> torch.Tensor:
        """Pure-PyTorch MoE forward with correct group-limited noaux_tc routing.

        Replaces the previous `NF.moe_block_tkg` call (which only supports
        standard sigmoid top-k). Dispatches to all-expert einsum like NxDI's
        `ExpertMLPsV2.forward_all_experts`.

        SP consistency: in prefill, inputs arrive as [T/ws, H] (attention ended
        in reduce_scatter). We all_gather → [T, H] for MoE compute, then
        reduce_scatter → [T/ws, H] to keep the SP layout. Same pattern as
        gpt_oss.forward_prefill (see `vllm_neuron/model/gpt_oss/model_bf16.py`).
        In decode the input is not SP-sharded — row-parallel all_reduce is fine.
        """
        # SP gather at entry (prefill only)
        sp_prefill = (self.world_size > 1) and (not is_decode)
        if sp_prefill:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        T = hidden_states.shape[0]
        H = self.hidden_size
        E = self.n_routed_experts
        E_local = self.num_local_experts
        I_local = self.moe_intermediate_size // self.tp_degree

        # RMSNorm (shared expert and router both use the same normed input)
        normed = hidden_states.float()
        variance = normed.pow(2).mean(-1, keepdim=True)
        normed = normed * torch.rsqrt(variance + 1e-6)
        normed = (self.post_attn_norm_weight.float() * normed).to(hidden_states.dtype)

        # Route: [T, top_k] weights and indices (router sees all E experts)
        topk_weight, topk_idx = self.router(normed)

        # Expand to dense [T, E] affinities (zeros outside selected top-k experts)
        expert_affinities = torch.zeros(
            T, E, device=normed.device, dtype=hidden_states.dtype
        )
        expert_affinities.scatter_(1, topk_idx, topk_weight.to(hidden_states.dtype))

        # >>> PARALLELISM: EP — filter global affinities to this rank's local experts <<<
        # Each EP rank owns a contiguous block of `num_local_experts`. When EP
        # is enabled, gather only the affinities for local experts; otherwise
        # expert_affinities is already [T, E_local=E].
        if self.ep_degree > 1:
            if isinstance(rank, torch.Tensor):
                # Neuron: derive ep_rank from the runtime rank tensor so the
                # FX/HLO graph stays rank-agnostic (one NEFF for all ranks).
                ep_rank_t = (rank % (self.ep_degree * self.tp_degree)) // self.tp_degree
                local_expert_indices = (
                    torch.arange(E_local, device=normed.device, dtype=torch.int32)
                    + ep_rank_t.to(torch.int32).view(()) * E_local
                )
            else:
                # CPU tests: use the static ep_rank set at construction.
                start = self.ep_rank * E_local
                local_expert_indices = torch.arange(
                    start,
                    start + E_local,
                    dtype=torch.long,
                    device=normed.device,
                )
            expert_affinities = NF.get_local_expert_affinities(
                expert_affinities, local_expert_indices
            )

        # Expert MLP dispatch: moe_cte kernel for prefill, einsum for decode.
        # moe_cte (CTE = cross-token expert) is a blockwise-matmul NKI kernel
        # designed for large T; at decode T=1 it hits a ZeroDivisionError in
        # build_blockwise_mapping (total_tokens // 16 = 0). At decode T is
        # small and there's no SP, so a single einsum is fine.
        if not is_decode:
            (
                expert_affinities_masked,
                token_position_to_id,
                block_to_expert,
                conditions,
            ) = NF.build_blockwise_mapping(
                expert_affinities=expert_affinities,
                num_local_experts=E_local,
                num_experts_per_token=self.num_experts_per_tok,
                block_size=256,
                moe_group=self.ep_tp_group,
                tp_degree=self.tp_degree,
            )
            routed_output = NF.moe_cte(
                implementation=MoECTEImplementation.shard_on_block,
                conditions=conditions,
                hidden_states=normed,
                expert_affinities_masked=expert_affinities_masked,
                gate_up_proj_weight=self.gate_up_proj_weight.reshape(
                    E_local,
                    self.hidden_size,
                    2,
                    I_local,
                ),
                down_proj_weight=self.down_proj_weight,
                activation_function=ActFnType.SiLU,
                block_size=256,
                token_position_to_id=token_position_to_id.to(dtype=torch.int32),
                block_to_expert=block_to_expert.to(dtype=torch.int32),
                expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
                skip_token=True,
                is_tensor_update_accumulating=True,
            )
        else:
            gate_up_w = self.gate_up_proj_weight.reshape(E_local, H, 2 * I_local)
            gate_up = torch.einsum("th,ehi->eti", normed, gate_up_w)
            gate, up = gate_up.chunk(2, dim=-1)
            intermediate = torch.nn.functional.silu(gate) * up
            down = torch.einsum("eti,eih->eth", intermediate, self.down_proj_weight)
            routed_output = torch.einsum(
                "eth,te->th", down, expert_affinities.to(down.dtype)
            )

        # Shared expert uses same normed input (sharded by full world_size,
        # same as attention/dense MLP). Each rank computes its 1/world_size
        # shard of the intermediate; the final moe_group.all_reduce sums all
        # 64 partials in one shot — no EP-specific scaling needed.
        shared_output = self._shared_expert(normed)

        output = routed_output + shared_output

        # Final collectives aggregate across the full TP group (moe_group).
        # With EP, each rank computed only its local experts' contribution;
        # all_reduce/reduce_scatter sums partial contributions across the
        # full 64-rank group. Both ep_tp (intermediate shards within an EP
        # group) and ep (expert subsets across EP groups) dims are reduced
        # in a single collective since moe_group = full TP group.
        if sp_prefill:
            output = self.moe_group.reduce_scatter(output, dim=0).contiguous()
        elif self.moe_group.world_size > 1:
            self.moe_group.all_reduce(output)

        return output


# =============================================================================
# Section 6: Decoder Layer
# =============================================================================


class DeepseekV32DecoderLayer(nn.Module):
    def __init__(self, config: DeepseekV32Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx

        self.input_layernorm = DeepseekV32RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.self_attn = DeepseekV32Attention(config, layer_idx)

        if layer_idx < config.first_k_dense_replace:
            self.post_attention_layernorm = DeepseekV32RMSNorm(
                config.hidden_size, config.rms_norm_eps, config.torch_dtype
            )
            self.mlp = DeepseekV32DenseMLP(config)
            self.is_moe = False
        else:
            self.mlp = DeepseekV32MoE(config)
            self.is_moe = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: dict,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, positions, position_embeddings, attn_metadata
        )
        hidden_states = residual + hidden_states

        # MLP / MoE
        residual = hidden_states

        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]
        is_decode = max_query_len <= decode_token_threshold

        if self.is_moe:
            hidden_states = self.mlp(hidden_states, positions, is_decode, rank)
        else:
            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states, is_decode=is_decode)

        hidden_states = residual + hidden_states

        return hidden_states


# =============================================================================
# Section 7: Model Backbone
# =============================================================================


class DeepseekV32Model(nn.Module):
    def __init__(self, config: DeepseekV32Config, batch_size: int):
        super().__init__()
        self.config = config

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )

        self.layers = nn.ModuleList(
            [
                DeepseekV32DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )

        self.norm = DeepseekV32RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = DeepseekV32RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=is_prefill, rank=rank
        )

        # Removed: _dump instrumentation moved to external hooks

        if inputs_embeds is not None:
            target_dim = hidden_states.shape[-1]
            current_dim = inputs_embeds.shape[-1]
            if current_dim > target_dim:
                raise ValueError(
                    f"inputs_embeds dim ({current_dim}) exceeds hidden dim ({target_dim})"
                )
            inputs_embeds = torch.nn.functional.pad(
                inputs_embeds, (0, target_dim - current_dim)
            )

        if (
            is_prefill
            and self.world_size > 1
            and inputs_embeds is not None
            and is_token_ids is not None
        ):
            local_len = hidden_states.shape[0]
            start = self.rank * local_len
            inputs_embeds = inputs_embeds[start : start + local_len]
            is_token_ids = is_token_ids[start : start + local_len]

        hidden_states = NF.merge_prompt_embeds(
            hidden_states, inputs_embeds, is_token_ids
        )

        position_embeddings = self.rotary_emb(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )

        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
                rank=rank,
            )

        hidden_states = self.norm(hidden_states)

        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        return hidden_states, []


# =============================================================================
# Section 8: Language Model Head
# =============================================================================


class DeepseekV32ForCausalLM(nn.Module):
    def __init__(self, config: DeepseekV32Config, batch_size: int):
        super().__init__()
        self.config = config
        self.model = DeepseekV32Model(config, batch_size)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size
        self.rank = self.tp_group.rank_in_group

        self.on_device_sampling_config = (
            config.neuron_config.on_device_sampling_config
            if config.neuron_config
            else None
        )
        debug_logits_enabled = (
            config.neuron_config is not None
            and config.neuron_config.debug_logits_dir is not None
        )
        self._gather_logits = (
            config.neuron_config is not None and config.neuron_config.max_logprobs != 0
        ) or debug_logits_enabled

        self.lm_head = neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.tp_group.device_group,
        )

        if self.on_device_sampling_config is not None:
            self.sampler = Sampler(
                self.on_device_sampling_config,
                process_group=self.tp_group.device_group,
            )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        is_token_ids: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        positions = positions.to(torch.int32)

        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name][
            "decode_token_threshold"
        ]
        is_prefill = max_query_len > decode_token_threshold

        T = input_ids.shape[0]
        if is_prefill and ((T <= self.world_size) or (T % self.world_size != 0)):
            raise ValueError(
                f"Prompt Length ({T}) must be > world_size ({self.world_size}) for SP."
            )

        hidden_states, aux_hidden_states = self.model(
            input_ids,
            positions,
            attn_metadata=attn_metadata,
            rank=rank,
            inputs_embeds=inputs_embeds,
            is_token_ids=is_token_ids,
        )

        hidden_states_for_logits = torch.index_select(
            hidden_states, dim=0, index=sampling_positions
        )
        hidden_states_for_logits = hidden_states_for_logits.to(self.config.torch_dtype)
        logits = self.lm_head(hidden_states_for_logits)

        if self.on_device_sampling_config is None:
            if hasattr(self, "_dump_data") and self._dump_data:
                self._dump_data["logits"] = logits.detach().cpu().float().clone()
                _dump_dir = "/tmp/ds32_hidden_dump"
                os.makedirs(_dump_dir, exist_ok=True)
                torch.save(self._dump_data, f"{_dump_dir}/all_hidden.pt")
                logger.warning(
                    "[DS32] Saved %d tensors to %s", len(self._dump_data), _dump_dir
                )
                self._dump_data.clear()
            return logits

        sampled_tokens = self.sampler(
            logits, sampling_params, logit_mask=logit_mask, tp_rank=rank
        )

        gathered_logits = None
        if self._gather_logits:
            if self.tp_group is not None:
                gathered_logits = self.tp_group.all_gather(logits, dim=1)
            else:
                gathered_logits = logits

        return sampled_tokens, gathered_logits

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = DeepseekV32Config.from_configs(hf_config, neuron_config)
        return cls(config, batch_size=1)

    # ── KV Cache Management ──────────────────────────────────────────────

    def get_kv_spec(self):
        layers = []
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            layers.append(
                LayerSpec(
                    name=layer_name,
                    num_kv_heads=1,  # MLA: single compressed KV representation
                    head_size=self.config.head_dim,  # 576 = qk_rope_head_dim + kv_lora_rank
                    dtype=layer.self_attn.dtype,
                    sliding_window_size=None,
                    chunk_size=None,
                )
            )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor, torch.Tensor]]):
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            if layer_name not in kv_caches:
                raise Exception(f"KV cache for layer {layer_name} not initialized")
            layer.self_attn.k_cache = kv_caches[layer_name][0]
            layer.self_attn.v_cache = kv_caches[layer_name][1]

    # ── Weight Loading ───────────────────────────────────────────────────

    def load_weights(
        self, checkpoint_path: str, device: torch.device, cache_dir: str | None
    ) -> None:
        tp_rank = self.rank
        tp_size = self.world_size
        logger.info(
            f"load_weights: tp_rank={tp_rank}, tp_size={tp_size}, "
            f"checkpoint_path={checkpoint_path}"
        )

        # Check for pre-sharded per-rank files (per-layer directory structure)
        global_rank_file = os.path.join(
            checkpoint_path, "global", f"rank_{tp_rank:04d}.safetensors"
        )
        if os.path.exists(global_rank_file):
            logger.info(
                "Loading pre-sharded weights from %s (per-layer format)",
                checkpoint_path,
            )
            from safetensors.torch import load_file

            # Merge per-layer files for this rank (only layers that exist in model)
            state_dict = {}
            expected_subdirs = ["global"] + [
                f"layer_{i:02d}" for i in range(len(self.model.layers))
            ]
            for subdir in expected_subdirs:
                rank_file = os.path.join(
                    checkpoint_path, subdir, f"rank_{tp_rank:04d}.safetensors"
                )
                if os.path.isfile(rank_file):
                    shard = load_file(rank_file, device="cpu")
                    state_dict.update(shard)

            logger.info(
                "Loaded %d tensors from pre-sharded files for rank %d",
                len(state_dict),
                tp_rank,
            )

            # Back-compat key remap: earlier pre-shards stored the MoE router as
            # `layers.{L}.mlp.router_weight` / `layers.{L}.mlp.e_score_correction_bias`.
            # Current code uses `layers.{L}.mlp.router.router_weight` / `.router.e_score_correction_bias`.
            # Rename in-place so old shards keep working without re-sharding.
            old_to_new = []
            for k in list(state_dict.keys()):
                if ".mlp.router_weight" in k:
                    old_to_new.append(
                        (
                            k,
                            k.replace(
                                ".mlp.router_weight", ".mlp.router.router_weight"
                            ),
                        )
                    )
                elif ".mlp.e_score_correction_bias" in k:
                    old_to_new.append(
                        (
                            k,
                            k.replace(
                                ".mlp.e_score_correction_bias",
                                ".mlp.router.e_score_correction_bias",
                            ),
                        )
                    )
            for old_k, new_k in old_to_new:
                state_dict[new_k] = state_dict.pop(old_k)
            if old_to_new:
                logger.info(
                    "Remapped %d pre-sharded keys for rank %d", len(old_to_new), tp_rank
                )

            for name, module in self.named_modules():
                for pname, param in module.named_parameters(recurse=False):
                    full_name = f"{name}.{pname}" if name else pname
                    if full_name in state_dict:
                        setattr(
                            module,
                            pname,
                            torch.nn.Parameter(state_dict[full_name].to(param.dtype)),
                        )
                    elif param.is_meta:
                        setattr(
                            module,
                            pname,
                            torch.nn.Parameter(
                                torch.zeros(
                                    param.shape, dtype=param.dtype, device="cpu"
                                )
                            ),
                        )
                        logger.warning(
                            "Parameter %s not in pre-sharded file, zero-initialized",
                            full_name,
                        )
                for bname, buf in module.named_buffers(recurse=False):
                    if buf.is_meta:
                        if bname == "hadamard_matrix":
                            module.register_buffer(
                                bname,
                                _build_hadamard_matrix(buf.shape[0], buf.dtype),
                                persistent=False,
                            )
                        else:
                            module.register_buffer(
                                bname,
                                torch.zeros(buf.shape, dtype=buf.dtype, device="cpu"),
                                persistent=False,
                            )
            return

        # Dummy-init fallback: only enter when the user pointed at an
        # existing *local directory* with no safetensors. HF model IDs
        # (``"org/model"``) don't pass ``isdir`` and are routed to the
        # HF download path below by ``SafetensorsCheckpoint``.
        if os.path.isdir(checkpoint_path):
            safetensor_files = glob.glob(os.path.join(checkpoint_path, "*.safetensors"))
            if not safetensor_files:
                logger.warning(
                    "No safetensors files found at %s — using randomly initialized weights (dummy mode)",
                    checkpoint_path,
                )
                for name, module in self.named_modules():
                    for pname, param in module.named_parameters(recurse=False):
                        if param.is_meta:
                            materialized = torch.nn.Parameter(
                                torch.randn(
                                    param.shape, dtype=param.dtype, device="cpu"
                                )
                                * 0.02
                            )
                            setattr(module, pname, materialized)
                return

        # Unsharded HF checkpoint path. SafetensorsWeightLoader instances
        # attached to each parameter (via _setup_weight_loaders in each module)
        # extract only the rank's slice from disk — no full tensor materialization.
        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        # Probe the checkpoint to learn which weight keys have a companion
        # ``weight_scale_inv`` (FP8 source). Pair them in the mappings so the
        # loaders see [weight_slice, scale_slice] and dequant inline.
        available_keys = checkpoint.get_tensor_names()
        mappings = self._build_hf_mappings(available_keys=available_keys)
        result = checkpoint.load_sharded_pipelined(
            tp_rank, tp_size, self, mappings, device, strict=False
        )
        if result.missing_keys:
            logger.warning(
                "Missing checkpoint keys for %d params (e.g. %s)",
                len(result.missing_keys),
                result.missing_keys[:5],
            )

        self.load_state_dict(result.state_dict, strict=False, assign=True)

        # Materialize any params still on `meta` (e.g. shared_expert_norm.weight,
        # which has no HF counterpart) with default values, plus buffers that
        # were registered under a meta-device context during construction
        # (Indexer's hadamard_matrix and k_cache).
        for name, module in self.named_modules():
            for pname, param in module.named_parameters(recurse=False):
                if param.is_meta:
                    full_name = f"{name}.{pname}" if name else pname
                    setattr(
                        module,
                        pname,
                        torch.nn.Parameter(
                            torch.zeros(param.shape, dtype=param.dtype, device="cpu")
                        ),
                    )
                    logger.warning(
                        "Parameter %s not in checkpoint, zero-initialized", full_name
                    )
            for bname, buf in module.named_buffers(recurse=False):
                if buf.is_meta:
                    if bname == "hadamard_matrix":
                        module.register_buffer(
                            bname,
                            _build_hadamard_matrix(buf.shape[0], buf.dtype),
                            persistent=False,
                        )
                    else:
                        module.register_buffer(
                            bname,
                            torch.zeros(buf.shape, dtype=buf.dtype, device="cpu"),
                            persistent=False,
                        )

    def _build_hf_mappings(self, available_keys: set[str] | None) -> dict:
        """Build ``{param_name: hf_checkpoint_key | [keys...]}`` for the
        unsharded HuggingFace DeepSeek V3.2 checkpoint format.

        When ``available_keys`` is provided, every 2D matmul weight that has a
        companion ``<key>.weight_scale_inv`` in the checkpoint is paired with
        that scale key, signalling the loaders to dequantize FP8 → BF16 inline.
        For BF16 checkpoints (no companion keys) the mapping degenerates to a
        single key per param — same as before.
        """
        mappings: dict = {}

        def _w(param_name: str, hf_key: str) -> None:
            """Register a 2D weight; pair its ``weight_scale_inv`` if present."""
            scale_key = hf_key + "_scale_inv"
            if available_keys is not None and scale_key in available_keys:
                mappings[param_name] = [hf_key, scale_key]
            else:
                mappings[param_name] = hf_key

        # Top-level (embed/lm_head/norm are bf16/fp32 in HF — never FP8)
        mappings["model.embed_tokens.weight"] = "model.embed_tokens.weight"
        mappings["lm_head.weight"] = "lm_head.weight"
        mappings["model.norm.weight"] = "model.norm.weight"

        for layer_id in range(len(self.model.layers)):
            lp = f"model.layers.{layer_id}"

            # Attention
            mappings[f"{lp}.input_layernorm.weight"] = f"{lp}.input_layernorm.weight"
            _w(f"{lp}.self_attn.wq_a_weight", f"{lp}.self_attn.q_a_proj.weight")
            mappings[f"{lp}.self_attn.q_norm.weight"] = (
                f"{lp}.self_attn.q_a_layernorm.weight"
            )
            _w(f"{lp}.self_attn.wq_b_weight", f"{lp}.self_attn.q_b_proj.weight")
            _w(
                f"{lp}.self_attn.wkv_a_weight",
                f"{lp}.self_attn.kv_a_proj_with_mqa.weight",
            )
            mappings[f"{lp}.self_attn.kv_norm.weight"] = (
                f"{lp}.self_attn.kv_a_layernorm.weight"
            )
            _w(f"{lp}.self_attn.wkv_b_weight", f"{lp}.self_attn.kv_b_proj.weight")
            _w(f"{lp}.self_attn.o_proj_weight", f"{lp}.self_attn.o_proj.weight")

            # DSA Indexer (only when use_dsa=True)
            attn = self.model.layers[layer_id].self_attn
            if attn.indexer is not None:
                ip = f"{lp}.self_attn.indexer"
                _w(f"{ip}.wq_b_weight", f"{ip}.wq_b.weight")
                _w(f"{ip}.wk_weight", f"{ip}.wk.weight")
                mappings[f"{ip}.k_norm_weight"] = f"{ip}.k_norm.weight"
                mappings[f"{ip}.k_norm_bias"] = f"{ip}.k_norm.bias"
                # weights_proj is BF16 in the FP8 checkpoint — no scale companion.
                _w(f"{ip}.weights_proj_weight", f"{ip}.weights_proj.weight")

            if layer_id < self.config.first_k_dense_replace:
                # Dense layers: post-attn norm is a separate module
                mappings[f"{lp}.post_attention_layernorm.weight"] = (
                    f"{lp}.post_attention_layernorm.weight"
                )
                # Dense MLP
                _w(f"{lp}.mlp.gate_proj_weight", f"{lp}.mlp.gate_proj.weight")
                _w(f"{lp}.mlp.up_proj_weight", f"{lp}.mlp.up_proj.weight")
                _w(f"{lp}.mlp.down_proj_weight", f"{lp}.mlp.down_proj.weight")
            else:
                # MoE layers: post-attn norm weight goes into MoE
                mappings[f"{lp}.mlp.post_attn_norm_weight"] = (
                    f"{lp}.post_attention_layernorm.weight"
                )
                # Router
                mappings[f"{lp}.mlp.router.router_weight"] = f"{lp}.mlp.gate.weight"
                mappings[f"{lp}.mlp.router.e_score_correction_bias"] = (
                    f"{lp}.mlp.gate.e_score_correction_bias"
                )
                # Routed experts: register ALL n_routed_experts HF keys. The
                # EP wrapper inside _setup_weight_loaders filters down to the
                # local rank's expert range after the underlying loader stacks
                # all experts. (Same pattern as qwen3_moe.) When the source
                # checkpoint is FP8, each weight key is followed by its
                # ``weight_scale_inv`` in the slice list — the MoE loaders
                # detect the doubled count and dequantize inline.
                moe = self.model.layers[layer_id].mlp
                num_experts = moe.n_routed_experts
                gate_up_keys: list[str] = []
                down_keys: list[str] = []

                def _expert_pair(base: str) -> list[str]:
                    sk = base + "_scale_inv"
                    if available_keys is not None and sk in available_keys:
                        return [base, sk]
                    return [base]

                for e in range(num_experts):
                    gate_up_keys.extend(
                        _expert_pair(f"{lp}.mlp.experts.{e}.gate_proj.weight")
                    )
                    gate_up_keys.extend(
                        _expert_pair(f"{lp}.mlp.experts.{e}.up_proj.weight")
                    )
                    down_keys.extend(
                        _expert_pair(f"{lp}.mlp.experts.{e}.down_proj.weight")
                    )
                mappings[f"{lp}.mlp.gate_up_proj_weight"] = gate_up_keys
                mappings[f"{lp}.mlp.down_proj_weight"] = down_keys
                # Shared expert
                _w(
                    f"{lp}.mlp.shared_gate_proj_weight",
                    f"{lp}.mlp.shared_experts.gate_proj.weight",
                )
                _w(
                    f"{lp}.mlp.shared_up_proj_weight",
                    f"{lp}.mlp.shared_experts.up_proj.weight",
                )
                _w(
                    f"{lp}.mlp.shared_down_proj_weight",
                    f"{lp}.mlp.shared_experts.down_proj.weight",
                )

        return mappings
