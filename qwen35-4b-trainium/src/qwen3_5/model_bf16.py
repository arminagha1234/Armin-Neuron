# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 hybrid (DeltaNet + GQA) model — BF16 implementation.

Layer pattern (32 total): [3 DeltaNet + 1 GQA] x 8

status (this file):
- ✅ RMSNorm, Partial RoPE, full-attention (GQA) layers wired with NF.*
- ✅ Dense SwiGLU MLP with NF.mlp (all 32 layers, TP-sharded intermediate)
- ✅ DeltaNet linear-attention layer (24 layers) — wraps PR #152's
     fused NKI kernel verbatim (kept in `nki_kernels/deltanet_fused.py`)
- 🟡 DeltaNet TP+SP — Phase 4 keeps weights replicated across ranks; head
     sharding is Phase 5
- 🟡 DeltaNet decode path — Phase 5 (single-step recurrent update)
- ✅ DecoderLayer dispatches by config.layer_types[layer_idx]
- ✅ Model + ForCausalLM follow qwen3_moe template (vocab-sharded
     embedding, sequence parallelism, lm_head)

This file mirrors `_reference/qwen3_moe_model_bf16.py` for the GQA layer,
`_reference/llama3/model.py` for the dense MLP, and PR #152's
`_reference/pr152/src/modeling_qwen35.py` for the DeltaNet pipeline.
Differences flagged with `# `.
"""

import logging

import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.distributed.parallel_state import get_tp_group

import vllm_neuron.functional as NF

from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.nn.embedding import VocabDimShardedEmbedding
from vllm_neuron.utils.dtype_utils import FP8_CLAMP_MAX
from vllm_neuron.utils.weight_loader import (
    fused_qkv_weight_loader,
    last_dim_padding_weight_loader,
    set_weight_loader,
    sharding_weight_loader,
    sharding_weight_loader_with_padding,
)

from .config import Qwen3_5Config

logger = logging.getLogger(__name__)


# ============================================================================
# Spliced Q/gate loaders — Qwen3.5 packs the attention output gate into the
# SECOND half of q_proj, PER HEAD interleaved:
#   q_proj.weight on disk = (num_heads * head_dim * 2, hidden)
#   layout: [h0_q(hd), h0_gate(hd), h1_q(hd), h1_gate(hd), ...]
# (HF: `q_proj(x).view(*shape, -1, head_dim*2).chunk(2, dim=-1)`.)
# The 4B parent wrongly assumed no gate existed and disabled it; these
# loaders extract the Q half (for qkv_proj_weight) and the gate half (for
# attn_gate_weight). Ported verbatim from the working Qwen3.6-27B adapter.
# ============================================================================


def _spliced_q_kv_loader(q_size_full, kv_size_full, num_shards, num_kv_replicas,
                         head_dim, num_heads):
    """Loader for `qkv_proj_weight`: Q sub-slice of each head from q_proj
    (Q/gate interleaved per head) + K + V. Returns (hidden, qkv_per_rank)."""
    from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

    assert num_heads % num_shards == 0
    heads_per_rank = num_heads // num_shards
    kv_per_rank = kv_size_full // max(1, (num_shards // max(1, num_kv_replicas)))

    def transform(slices, rank):
        assert len(slices) == 3, "expected (Q, K, V) slices"
        q_slice, k_slice, v_slice = slices
        q_rank = rank % num_shards
        kv_rank = q_rank // max(1, num_kv_replicas)

        first_head = q_rank * heads_per_rank
        q_rows = []
        for h in range(first_head, first_head + heads_per_rank):
            base = h * 2 * head_dim          # start of this head's [q|gate] block
            q_rows.append(q_slice[base : base + head_dim, :])  # the Q half only
        q_t = torch.cat(q_rows, dim=0)

        kv_start = kv_rank * kv_per_rank
        kv_end = kv_start + kv_per_rank
        k_t = k_slice[kv_start:kv_end, :]
        v_t = v_slice[kv_start:kv_end, :]

        cat = torch.cat([q_t, k_t, v_t], dim=0)
        return cat.T.contiguous()

    return SafetensorsWeightLoader(transform=transform)


def _spliced_q_gate_loader(q_size_full, num_shards, head_dim, num_heads):
    """Loader for `attn_gate_weight`: GATE sub-slice of each head from q_proj.
    Gate for head h is rows [h*2*hd + hd : (h+1)*2*hd]. Returns (hidden, q_per_rank)."""
    from vllm_neuron.utils.weight_loader import SafetensorsWeightLoader

    assert num_heads % num_shards == 0
    heads_per_rank = num_heads // num_shards

    def transform(slices, rank):
        assert len(slices) == 1, "expected single q_proj slice"
        q_slice = slices[0]
        q_rank = rank % num_shards
        first_head = q_rank * heads_per_rank
        g_rows = []
        for h in range(first_head, first_head + heads_per_rank):
            base = h * 2 * head_dim
            g_rows.append(q_slice[base + head_dim : base + 2 * head_dim, :])  # gate half
        g_t = torch.cat(g_rows, dim=0)
        return g_t.T.contiguous()

    return SafetensorsWeightLoader(transform=transform)


# ============================================================================
# Section 1: RMSNorm
# Identical to qwen3_moe (standard RMSNorm, no padding).
# ============================================================================


class Qwen3_5RMSNorm(nn.Module):
    """RMS Normalization with the Qwen3.5 `(1 + weight)` convention.

    IMPORTANT: Qwen3.5 (like Gemma) stores RMSNorm weights centered around
    ZERO and applies `output * (1.0 + weight)` at runtime — NOT the
    standard `output * weight`. See HF
    `transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5RMSNorm`:

        output = output * (1.0 + self.weight.float())

    Using plain `weight * x` (with weight init at 1.0) loads the
    checkpoint's ~0-centered norm weights and multiplies directly, making
    every norm in the network ~1.0 too small, which uniformly inflates/
    garbles activations and produces degenerate (whitespace) output. This
    applies to ALL norms: input_layernorm, post_attention_layernorm,
    q_norm, k_norm, and the final model norm.

    This was the root cause of the all-whitespace output. Same bug + fix
    as the Qwen3.6-27B adapter (Qwen3_6RMSNorm).
    """

    def __init__(self, hidden_size: int, eps: float, dtype: torch.dtype) -> None:
        super().__init__()
        # Stored centered at 0 (the checkpoint weights are ~0, not ~1).
        self.weight = nn.Parameter(torch.zeros(hidden_size, dtype=dtype))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        # (1 + weight) convention — the critical Qwen3.5 detail.
        x = x * (1.0 + self.weight.float())
        return x.to(input_dtype)


# ============================================================================
# Section 2: Rotary Position Embedding (Partial RoPE)
# Only the first `partial_rotary_factor * head_dim`
#                      dimensions are rotated (PR #152: 25% = 64 dims).
# ============================================================================


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Interleaved RoPE rotation."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE only to the first `rotary_dim` dimensions of head_dim.

    Qwen3.5 rotates only 25% of head_dim. The remaining
    75% pass through unchanged.

    Args:
        q, k: [..., head_dim]
        cos, sin: [T, rotary_dim/2]
        rotary_dim: number of leading dims of head_dim to rotate
    """
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    cos_full = torch.cat((cos, cos), dim=-1).unsqueeze(0)  # [1, T, rot]
    sin_full = torch.cat((sin, sin), dim=-1).unsqueeze(0)  # [1, T, rot]

    q_rot = (q_rot * cos_full) + (rotate_half(q_rot) * sin_full)
    k_rot = (k_rot * cos_full) + (rotate_half(k_rot) * sin_full)

    return torch.cat((q_rot, q_pass), dim=-1), torch.cat((k_rot, k_pass), dim=-1)


class Qwen3_5RotaryEmbedding(nn.Module):
    """Partial RoPE: produces (cos, sin) for the first `rotary_dim` dims.

    rotary_dim = head_dim * partial_rotary_factor.
    """

    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        # partial RoPE
        self.rotary_dim = int(round(config.head_dim * config.partial_rotary_factor))
        if self.rotary_dim % 2 != 0:
            raise ValueError(
                f"rotary_dim must be even; got {self.rotary_dim} "
                f"(head_dim={self.head_dim}, factor={config.partial_rotary_factor})"
            )

        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float, device="cpu")
                / self.rotary_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        position_ids: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq_expanded = self.inv_freq[None, :].float()  # [1, rot/2]
        position_ids_expanded = position_ids[:, None].float()  # [T, 1]
        freqs = position_ids_expanded @ inv_freq_expanded  # [T, rot/2]
        return freqs.cos().to(dtype=dtype), freqs.sin().to(dtype=dtype)


# ============================================================================
# Section 3: Full-Attention (GQA) Layer  -- the 8 of 32 layers
# Mirrors qwen3_moe Qwen3MoeAttention almost exactly. Differences:
#   - Partial RoPE (only first `rotary_dim` of head_dim)
#   - `attn_output_gate` flag (Qwen3.5 has a sigmoid gate on attn output)
# ============================================================================


class Qwen3_5GQAAttention(nn.Module):
    """GQA attention with TP head sharding and partial RoPE.

    Following Qwen3MoeAttention. Differences flagged with QWEN35-SPECIFIC.
    """

    def __init__(self, config: Qwen3_5Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.rms_norm_eps = config.rms_norm_eps
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = config.head_dim ** -0.5
        # rotary applied only to part of head_dim
        self.rotary_dim = int(round(config.head_dim * config.partial_rotary_factor))
        # Sigmoid gate on attention output. Qwen3.5's HF config flags
        # attn_output_gate=True AND the safetensors DO ship the gate — but
        # it's SPLICED into q_proj's second half (per-head interleaved
        # [q|gate]), not a separate gate_proj.weight. The 4B parent wrongly
        # concluded the gate was missing and forced this False, which
        # dropped the gate on every GQA layer and garbled all output
        # (parity harness: cos 0.39 without gate vs 0.99999 with gate).
        # The spliced loaders in _setup_weight_loaders split q_proj's two
        # halves into qkv_proj_weight (Q + K + V) and attn_gate_weight.
        self.attn_output_gate = bool(getattr(config, "attn_output_gate", False))

        # TP group setup
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        # Head sharding (replicate KV when fewer than world_size)
        self.num_attention_heads_per_rank = (
            self.num_attention_heads // self.world_size
        )
        if self.world_size >= self.num_key_value_heads:
            self.num_key_value_heads_per_rank = 1
            self.num_kv_replicas = self.world_size // self.num_key_value_heads
        else:
            self.num_key_value_heads_per_rank = (
                self.num_key_value_heads // self.world_size
            )
            self.num_kv_replicas = 1
        self.num_key_value_groups = (
            self.num_attention_heads_per_rank // self.num_key_value_heads_per_rank
        )

        # QKV / O weights
        q_size = self.num_attention_heads_per_rank * self.head_dim
        kv_size = self.num_key_value_heads_per_rank * self.head_dim
        qkv_size = q_size + 2 * kv_size
        o_in = (self.num_attention_heads * self.head_dim) // self.world_size

        self.qkv_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, qkv_size, dtype=self.dtype)
        )
        self.o_proj_weight = nn.Parameter(
            torch.empty(o_in, self.hidden_size, dtype=self.dtype)
        )

        # Per-head Q/K layernorm (Qwen3 family)
        self.q_layernorm = Qwen3_5RMSNorm(self.head_dim, self.rms_norm_eps, self.dtype)
        self.k_layernorm = Qwen3_5RMSNorm(self.head_dim, self.rms_norm_eps, self.dtype)

        # gate projection on attention output
        # PR #152: g = sigmoid(linear(hidden)); attn_out = g * attn_out
        # Gate matches Q dimension (one scalar per attention slot).
        if self.attn_output_gate:
            self.attn_gate_weight = nn.Parameter(
                torch.empty(self.hidden_size, q_size, dtype=self.dtype)
            )

        self.q_size = q_size
        self.kv_size = kv_size
        self.qkv_split_indices = [q_size, q_size + kv_size]

        self.k_cache = None
        self.v_cache = None

        # Path D: FP8 KV cache scales. Registered as buffers (not Parameters)
        # because the safetensors checkpoint doesn't ship `k_scale`/`v_scale`
        # keys — registering as Parameters trips strict weight loading.
        #
        # Initial scale = 32. Rationale: BF16 K/V values in attention layers
        # are typically in [-3, +3]. FP8 e4m3 max representable is ~240, so
        # scaling by 32 maps [-3, 3] → [-96, 96] which uses ~40% of FP8's
        # dynamic range without saturating. With scale=1.0 we'd land in FP8's
        # smallest binade and only use ~5 bits of precision — defeating the
        # purpose. 32 is conservative; runtime calibration would tune per
        # layer (TODO follow-up). Float copies avoid Dynamo graph breaks
        # since tensor.item() is host-side.
        self.register_buffer(
            "k_scale",
            torch.tensor(32.0, dtype=torch.float32, device="cpu"),
            persistent=False,
        )
        self.register_buffer(
            "v_scale",
            torch.tensor(32.0, dtype=torch.float32, device="cpu"),
            persistent=False,
        )
        self.k_scale_float = 32.0
        self.v_scale_float = 32.0

        self._setup_weight_loaders()

    def _setup_weight_loaders(self) -> None:
        if self.attn_output_gate:
            # Qwen3.5 spliced layout: q_proj on disk is
            # (num_heads * head_dim * 2, hidden) interleaved PER HEAD as
            # [h0_q, h0_gate, h1_q, h1_gate, ...]. Extract Q half + K + V.
            kv_size_full = self.num_key_value_heads * self.head_dim
            set_weight_loader(
                self.qkv_proj_weight,
                _spliced_q_kv_loader(
                    q_size_full=self.num_attention_heads * self.head_dim,
                    kv_size_full=kv_size_full,
                    num_shards=self.world_size,
                    num_kv_replicas=self.num_kv_replicas,
                    head_dim=self.head_dim,
                    num_heads=self.num_attention_heads,
                ),
            )
        else:
            set_weight_loader(
                self.qkv_proj_weight,
                fused_qkv_weight_loader(
                    q_size=self.q_size,
                    kv_size=self.kv_size,
                    shard_dim=1,
                    num_shards=self.world_size,
                    is_storage_transposed=True,
                    num_kv_replicas=self.num_kv_replicas,
                ),
            )
        set_weight_loader(
            self.o_proj_weight,
            sharding_weight_loader(
                shard_dim=0,
                shard_size=(self.num_attention_heads * self.head_dim)
                // self.world_size,
                num_shards=self.world_size,
                is_storage_transposed=True,
            ),
        )
        if self.attn_output_gate:
            # Spliced gate: load the gate sub-slice of each head from q_proj's
            # second half (per-head interleaved [q|gate]).
            set_weight_loader(
                self.attn_gate_weight,
                _spliced_q_gate_loader(
                    q_size_full=self.num_attention_heads * self.head_dim,
                    num_shards=self.world_size,
                    head_dim=self.head_dim,
                    num_heads=self.num_attention_heads,
                ),
            )

    # ── Forward dispatch ────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ):
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]

        if max_query_len <= decode_token_threshold:
            return self.forward_decode(
                hidden_states, positions, position_embeddings, attn_metadata
            )
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return self.forward_prefill(
            hidden_states, positions, position_embeddings, attn_metadata
        )

    # ── Prefill ─────────────────────────────────────────────────────────

    def forward_prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        hidden_states = hidden_states.to(self.dtype)
        tokens, hidden = hidden_states.shape

        # 1. Fused QKV
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
        ).squeeze(0)

        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)

        q = q.view(tokens, self.num_attention_heads_per_rank, self.head_dim).transpose(0, 1)
        k = k.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(0, 1)
        v = v.view(tokens, self.num_key_value_heads_per_rank, self.head_dim).transpose(0, 1)

        # 2. QK-norm, then partial RoPE
        q = self.q_layernorm(q)
        k = self.k_layernorm(k)

        cos, sin = position_embeddings
        # partial RoPE (only first `rotary_dim` of head_dim)
        q, k = apply_partial_rotary_pos_emb(q, k, cos, sin, self.rotary_dim)

        # 3. Update KV cache (per-rank)
        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]

        block_indices = slot_mapping // block_size
        position_indices = slot_mapping % block_size

        # PATH D: FP8 KV write — quantize K/V if cache is FP8 dtype.
        # If cache is BF16/FP32 (Path C-style), the conditional falls
        # through to a plain dtype cast.
        k_raw = k.reshape(-1, self.head_dim)
        v_raw = v.reshape(-1, self.head_dim)
        if self.k_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            k_flat = (k_raw * self.k_scale).clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX).to(self.k_cache.dtype)
            v_flat = (v_raw * self.v_scale).clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX).to(self.v_cache.dtype)
        else:
            k_flat = k_raw.to(self.k_cache.dtype)
            v_flat = v_raw.to(self.v_cache.dtype)

        head_indices = torch.arange(
            self.num_key_value_heads_per_rank,
            dtype=torch.long,
            device=hidden_states.device,
        ).repeat_interleave(slot_mapping.shape[0])
        block_idx_put = block_indices.repeat(self.num_key_value_heads_per_rank)
        pos_idx_put = position_indices.repeat(self.num_key_value_heads_per_rank)

        self.k_cache.index_put_((block_idx_put, head_indices, pos_idx_put), k_flat)
        self.v_cache.index_put_((block_idx_put, head_indices, pos_idx_put), v_flat)

        # 4. Flash attention (no sinks, no SWA)
        k = k.repeat_interleave(self.num_key_value_groups, dim=0)
        v = v.repeat_interleave(self.num_key_value_groups, dim=0)

        attn_output = NF.flash_attention(
            q.transpose(1, 2),  # [Nh, Dh, T]
            k.transpose(1, 2),  # [Nh, Dh, T]
            v,                  # [Nh, T, Dh]
            scale=self.scaling,
            tp_q=False,
            tp_out=True,
        )

        # sigmoid output gate
        if self.attn_output_gate:
            # gate = sigmoid(hidden @ W_gate)  → shape [tokens, q_size_per_rank]
            gate = torch.sigmoid(hidden_states @ self.attn_gate_weight)
            # attn_output from NF.flash_attention (tp_out=True) is [Nh, Dh, T].
            # Reshape gate [tokens, Nh, Dh] → [Nh, Dh, tokens] to match.
            gate = gate.view(tokens, self.num_attention_heads_per_rank, self.head_dim)
            attn_output = attn_output * gate.permute(1, 2, 0)

        # 5. O-proj + reduce-scatter
        attn_output = attn_output.unsqueeze(0)
        attn_output = NF.o_proj(attn_output, self.o_proj_weight)
        attn_output = attn_output.squeeze(0)

        if self.world_size > 1:
            attn_output = self.tp_group.reduce_scatter(attn_output, dim=0)

        return attn_output.contiguous()

    # ── Decode ──────────────────────────────────────────────────────────

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object,
    ):
        """Decode path — REUSES prefill's NF.qkv_proj + NF.flash_attention.

        The qwen3_moe-style fused `NF.attention_decode` kernel asserts
        `Tensor engine transpose requires shape <= [128, 128]` which fails
        on Qwen3.5's head_dim=256. So we follow the prefill pattern:
        NF.qkv_proj -> RMSNorm -> partial RoPE -> read prior KV from
        cache (gathered via block_table) -> NF.flash_attention with the
        full Q+K context -> NF.o_proj.

        This keeps us inside vllm_neuron's NF kernels (which know how to
        handle the storage layout, TP sharding, etc.) and avoids the
        head_dim=256 limit of the fused decode megakernel.
        """
        if attn_metadata is None:
            return torch.zeros_like(hidden_states)

        layer_name = f"layers.{self.layer_idx}.self_attn"
        slot_mapping = attn_metadata[layer_name]["slot_mapping"]
        block_size = attn_metadata[layer_name]["block_size"]
        max_blocks_per_seq = attn_metadata[layer_name]["max_blocks_per_seq"]
        block_table = attn_metadata[layer_name]["block_table_tensor"]

        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B
        S_ctx = max_blocks_per_seq * block_size
        Nh = self.num_attention_heads_per_rank
        Nkh = self.num_key_value_heads_per_rank
        Dh = self.head_dim

        hidden_states = hidden_states.to(self.dtype)

        # 1) Fused QKV via NF (handles storage layout + TP sharding)
        qkv = NF.qkv_proj(
            hidden=hidden_states.unsqueeze(0),
            qkv_weights=self.qkv_proj_weight,
        ).squeeze(0)

        q, k, v = torch.tensor_split(qkv, self.qkv_split_indices, dim=-1)

        q = q.view(tokens, Nh, Dh).transpose(0, 1)   # [Nh, T, Dh]
        k = k.view(tokens, Nkh, Dh).transpose(0, 1)  # [Nkh, T, Dh]
        v = v.view(tokens, Nkh, Dh).transpose(0, 1)  # [Nkh, T, Dh]

        # 2) Per-head RMSNorm
        q = self.q_layernorm(q)
        k = self.k_layernorm(k)

        # 3) Partial RoPE (first rotary_dim entries only)
        cos, sin = position_embeddings
        q, k = apply_partial_rotary_pos_emb(q, k, cos, sin, self.rotary_dim)

        # 4) Write new K/V into the paged cache.
        block_indices = slot_mapping // block_size
        position_indices = slot_mapping % block_size
        num_tokens = slot_mapping.shape[0]

        k_new_flat = k.reshape(-1, Dh)  # [Nkh*T, Dh]
        v_new_flat = v.reshape(-1, Dh)
        head_indices = torch.arange(
            Nkh, dtype=torch.long, device=hidden_states.device
        ).repeat_interleave(num_tokens)
        block_idx_put = block_indices.repeat(Nkh)
        pos_idx_put = position_indices.repeat(Nkh)
        # PATH D: FP8 KV write — quantize on FP8 cache, plain cast otherwise.
        if self.k_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            k_q = (k_new_flat * self.k_scale).clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX).to(self.k_cache.dtype)
            v_q = (v_new_flat * self.v_scale).clamp(-FP8_CLAMP_MAX, FP8_CLAMP_MAX).to(self.v_cache.dtype)
        else:
            k_q = k_new_flat.to(self.k_cache.dtype)
            v_q = v_new_flat.to(self.v_cache.dtype)
        self.k_cache.index_put_((block_idx_put, head_indices, pos_idx_put), k_q)
        self.v_cache.index_put_((block_idx_put, head_indices, pos_idx_put), v_q)

        # 5) Gather full prior K/V context for each request via block_table.
        # k_cache is [num_blocks, Nkh, block_size, Dh]
        # block_table is [B, max_blocks_per_seq] of block ids
        # PATH C ALT A: keep batch axis separate so per-request masking works.
        # Path B's single-stream flatten of [B*S_ctx] would let request 0
        # cross-attend into request 1's KV — that's the bug max_num_seqs>1
        # could not work around.
        # PATH D: if the cache is FP8, dequantize on read. We use the
        # Llama3 trick of FOLDING the scale into the softmax scaling
        # factor instead of materializing a dequantized K_full tensor.
        # This skips one full-size tensor materialization per layer per
        # token — saves 32 × tokens × Nh × S_ctx × Dh × 2 bytes of
        # short-lived activation memory + DMA traffic.
        #   K_real = K_raw.to(bf16) / k_scale
        #   scores = (Q @ K_real.T) * (1/sqrt(d))
        #          = (Q @ K_raw.to(bf16).T) * (1/sqrt(d) / k_scale)   <- folded
        # The fold for V is harder (would need to bake into o_proj_weight),
        # so we materialize V_full normally — V is read once, K is read
        # twice (mask + attention), so K is the bigger win.
        K_raw = self.k_cache[block_table]   # [B, MB, Nkh, BS, Dh]
        V_raw = self.v_cache[block_table]
        K_raw = K_raw.permute(0, 2, 1, 3, 4).reshape(B, Nkh, S_ctx, Dh)
        V_raw = V_raw.permute(0, 2, 1, 3, 4).reshape(B, Nkh, S_ctx, Dh)
        if self.k_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            # Just cast to bf16; do NOT divide by scale here.
            K_full = K_raw.to(self.dtype)
            V_full = V_raw.to(self.dtype) / self.v_scale  # V dequant still explicit
            # Compute the effective scale: scaling / k_scale folded into one float.
            effective_scale = self.scaling / self.k_scale_float
        else:
            K_full = K_raw.to(self.dtype)
            V_full = V_raw.to(self.dtype)
            effective_scale = self.scaling

        # 6) Repeat KV for GQA grouping
        if self.num_key_value_groups > 1:
            K_full = K_full.repeat_interleave(self.num_key_value_groups, dim=1)
            V_full = V_full.repeat_interleave(self.num_key_value_groups, dim=1)
        # K_full, V_full now [B, Nh, S_ctx, Dh]

        # 7) Manual masked attention (Path C Alt A).
        # We replace NF.flash_attention on decode with explicit
        # Q@K^T -> mask -> softmax -> @V because:
        #   (a) NF.flash_attention has no attention_mask kwarg, only the
        #       built-in single-stream causal triangle which is wrong when
        #       multiple requests share the K/V tensor.
        #   (b) NF.attention_decode (which DOES take a mask) requires
        #       head_dim <= 128 — we have head_dim=256.
        # The compiler still fuses this chain into a single NEFF on the
        # tensor engine. At customer's 20K-in / 1-out shape the decode
        # matmul is ~0.4 GFLOP/layer — small, not the bottleneck.

        # 7) Path E: Gemma4-style split-K + split-V flash attention for head_dim=256.
        # Replaces Path D's full-Q-times-full-K matmul (which forces the
        # Neuron compiler down a slower path because of the 256 reduction
        # dim) with two head_dim=128 matmuls accumulated via PSUM.
        #
        # Reference: ../../../../Gemma4/flash_attn_hd256_nki.py — pure
        # PyTorch split-K kernel, designed to be lowered to NKI by
        # torch.compile. Same math as the original Path D forward_decode
        # but in a layout the Neuron compiler can fuse aggressively.
        #
        # Math:
        #   scores = (Q_lo @ K_lo^T + Q_hi @ K_hi^T) * scale
        #   weights = softmax(scores + mask_bias)
        #   out_lo = weights @ V_lo
        #   out_hi = weights @ V_hi
        #   out    = cat([out_lo, out_hi], dim=-1)
        #
        # The compiler fuses the two QK matmuls into PSUM-accumulating
        # nc_matmul calls (head_dim=128 fits the tensor engine transpose
        # limit). Same trick for AV. Result: a single fused NEFF that
        # parallelizes across requests cleanly, bypassing chunked-prefill.

        # 7a) Reshape Q from [Nh, B*S_decode, Dh] -> [B, Nh, S_decode, Dh]
        q_b = q.transpose(0, 1).reshape(B, S_decode, Nh, Dh).transpose(1, 2)

        # 7b) Per-request causal mask (same as Path D).
        positions_b = positions.view(B, S_decode)              # [B, S_decode]
        k_idx = torch.arange(S_ctx, device=positions.device)   # [S_ctx]
        mask = (k_idx[None, None, None, :] <= positions_b[:, None, :, None])

        # 7c) Split-K matmul: head_dim=256 → two head_dim=128 matmuls.
        # Q_lo @ K_lo^T  +  Q_hi @ K_hi^T  =  Q @ K^T   (mathematically identical)
        # Each individual matmul has K=128 which fits the tensor engine's
        # 128 transpose limit cleanly.
        Dh_half = Dh // 2  # = 128 for head_dim=256
        q_lo = q_b[..., :Dh_half]                 # [B, Nh, S_decode, 128]
        q_hi = q_b[..., Dh_half:]                 # [B, Nh, S_decode, 128]
        k_lo = K_full[..., :Dh_half]              # [B, Nh, S_ctx, 128]
        k_hi = K_full[..., Dh_half:]              # [B, Nh, S_ctx, 128]

        scores = (
            torch.matmul(q_lo, k_lo.transpose(-2, -1))
            + torch.matmul(q_hi, k_hi.transpose(-2, -1))
        )                                          # [B, Nh, S_decode, S_ctx]
        scores = scores * effective_scale

        # Apply mask via additive bias.
        neg_bias = (~mask).to(scores.dtype) * -1e4
        scores = scores + neg_bias
        attn_weights = torch.softmax(scores, dim=-1)

        # 7d) Split-V matmul: weights @ V_lo  cat  weights @ V_hi
        v_lo = V_full[..., :Dh_half]              # [B, Nh, S_ctx, 128]
        v_hi = V_full[..., Dh_half:]              # [B, Nh, S_ctx, 128]
        attn_output_lo = torch.matmul(attn_weights, v_lo)   # [B, Nh, S_decode, 128]
        attn_output_hi = torch.matmul(attn_weights, v_hi)   # [B, Nh, S_decode, 128]
        attn_output = torch.cat([attn_output_lo, attn_output_hi], dim=-1)
        # attn_output: [B, Nh, S_decode, Dh]

        # Re-flatten to [Nh, tokens, Dh] (matching Path B's layout).
        attn_output = attn_output.transpose(0, 1).reshape(Nh, tokens, Dh)

        # 8) Optional sigmoid attention-output gate
        if self.attn_output_gate:
            gate = torch.sigmoid(hidden_states @ self.attn_gate_weight)
            gate = gate.view(tokens, Nh, Dh)
            attn_output = attn_output * gate.transpose(0, 1)

        # 9) O-projection via NF.
        # NF.o_proj expects [B, N, D, S] layout (B=batch, N=heads, D=head_dim, S=seq_len).
        # Our attn_output is [Nh, tokens, Dh] = [N, S, D] — we need to transpose
        # the last two dims and add a leading batch dim.
        # Note: Path B's code at batch=1 had a layout bug here that happened to
        # work because S=1 made the bad reshape harmless. At batch>1 (Path D
        # territory) we need the correct [B, N, D, S] layout.
        attn_output = attn_output.transpose(1, 2).unsqueeze(0)   # [1, Nh, Dh, tokens]
        attn_output = NF.o_proj(attn_output, self.o_proj_weight)  # → [1, tokens, hidden]
        attn_output = attn_output.squeeze(0)                       # [tokens, hidden]

        # 10) All-reduce across TP (NOT reduce_scatter — decode tokens may
        # not be divisible by world_size, e.g. batch=1 with TP=8).
        if self.world_size > 1:
            attn_output = self.tp_group.all_reduce(attn_output)

        return attn_output.contiguous()


# ============================================================================
# Section 4: DeltaNet (linear attention) Layer
# Wraps the validated PR #152 fused NKI kernel with a vllm_neuron-style
# nn.Module. The kernel itself is in `nki_kernels/deltanet_fused.py`
# (verbatim from PR #152 — never edit; fix wrappers instead).
# ============================================================================


def _deltanet_l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalization along `dim`. Q and K are l2-normed before the kernel."""
    return x / torch.linalg.vector_norm(x, dim=dim, keepdim=True).clamp_min(eps)


class Qwen3_5DeltaNetAttention(nn.Module):
    """GatedDeltaNet linear-attention layer.

    Wraps `qwen3_5.nki_kernels.deltanet_fused_chunked_fwd` (PR #152).
    Pipeline mirrors `NeuronGatedDeltaNet.forward` from PR #152:

        in_proj_qkv  → split into q, k, v
        in_proj_z    → output gate input
        in_proj_a    → softplus → exp(-A_log) decay   (g)
        in_proj_b    → sigmoid                         (beta)
        causal conv1d (kernel=4) on concat(q, k, v)    (mixed_post_conv)
        silu(...)
        reshape to (B, H_v, S, head_dim) — k expanded from H_k via repeat
        l2norm + scale by 1/sqrt(k_dim)
        pad S to multiple of 128
        per-(B*H) NKI fused chunked DeltaNet kernel call
        gather outputs
        RMSNorm over head_v_dim
        z gate: out = out * silu(z)
        out_proj  → hidden_size

    State management: PR #152's "side-channel buffer" pattern. Each layer
    holds a `recurrent_state_buffer` and `conv_state_buffer`. NxDI uses
    these as nn.Parameter aliases; we keep that contract and add the
    "+ buffer * 0" residual trick from PR #152 to keep the cache manager
    happy.

    TP strategy (Phase 4 minimum-viable):
        weights replicated across TP ranks (no head sharding).
                 Each rank computes the same DeltaNet output. Functional,
                 not optimal.
        shard num_v_heads across TP. The kernel is per-(b,h)
                 so this is just a question of which heads each rank
                 owns.
    """

    def __init__(self, config: Qwen3_5Config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.dtype = config.torch_dtype
        self.rms_norm_eps = config.rms_norm_eps

        # TP group setup (mirrors GQA layer). DeltaNet weights are
        # replicated across ranks (Phase 4 simplification), but the
        # FORWARD must still understand the sequence-parallel layout
        # vllm-neuron uses: each rank receives tokens/world_size tokens
        # of the residual stream. Without all_gather/reduce_scatter
        # around the DeltaNet recurrence, each rank computes a
        # recurrent state on its tiny shard of the sequence, producing
        # garbage output. See PATH_B_BUG_CONFIRMED.md for evidence.
        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        self.hidden_size = config.hidden_size
        self.num_v_heads = config.deltanet_num_v_heads      # 32
        self.num_k_heads = config.deltanet_num_k_heads      # 16
        self.head_k_dim = config.deltanet_k_head_dim        # 128
        self.head_v_dim = config.deltanet_v_head_dim        # 128
        self.conv_kernel_size = config.deltanet_conv_kernel_size  # 4

        # Derived dims
        self.key_dim = self.head_k_dim * self.num_k_heads       # 2048
        self.value_dim = self.head_v_dim * self.num_v_heads     # 4096
        self.conv_dim = self.key_dim * 2 + self.value_dim       # 8192

        # PR #152 sanity: kernel assumes head_k_dim == head_v_dim == 128
        if self.head_k_dim != 128 or self.head_v_dim != 128:
            raise NotImplementedError(
                f"PR #152 fused DeltaNet kernel hardcodes head_dim=128. "
                f"Got head_k_dim={self.head_k_dim}, head_v_dim={self.head_v_dim}."
            )

        # Input projections — replicated across TP ranks for now
        self.in_proj_qkv_weight = nn.Parameter(
            torch.empty(self.conv_dim, self.hidden_size, dtype=self.dtype)
        )
        self.in_proj_z_weight = nn.Parameter(
            torch.empty(self.value_dim, self.hidden_size, dtype=self.dtype)
        )
        self.in_proj_a_weight = nn.Parameter(
            torch.empty(self.num_v_heads, self.hidden_size, dtype=self.dtype)
        )
        self.in_proj_b_weight = nn.Parameter(
            torch.empty(self.num_v_heads, self.hidden_size, dtype=self.dtype)
        )

        # Causal conv1d on concat(q, k, v) — depthwise, kernel=4
        self.conv1d_weight = nn.Parameter(
            torch.empty(self.conv_dim, 1, self.conv_kernel_size, dtype=self.dtype)
        )

        # Decay parameters
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads, dtype=torch.float32))
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads, dtype=torch.float32))

        # Output: per-head RMSNorm over head_v_dim, then linear back to hidden
        self.norm_weight = nn.Parameter(torch.ones(self.head_v_dim, dtype=self.dtype))
        self.out_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.value_dim, dtype=self.dtype)
        )

        # State buffers (Phase 5 will plumb these into the cache manager).
        # Registered as non-persistent buffers so PyTorch tracks them via
        # named_buffers() — NOT named_parameters(). The vllm_neuron loader
        # iterates named_parameters() so these stay invisible to it,
        # which is what we want (state buffers don't come from the
        # checkpoint; they're zero-initialized at runtime).
        #
        # IMPORTANT: explicit `device="cpu"` overrides the
        # `with torch.device("meta")` context that vllm_neuron uses to
        # build the model skeleton. Without it, these buffers land on
        # `meta` and `model.to(device)` later fails with "Cannot copy out
        # of meta tensor; no data!".
        nc = config.neuron_config
        # vllm-neuron exposes the decode batch-size cap via
        # `num_seqs_buckets` (a list). PR #152's NxDI flavor used
        # `max_batch_size`; we accept both for forward-compat.
        if nc is None:
            max_batch = 1
        elif hasattr(nc, "max_batch_size") and nc.max_batch_size is not None:
            max_batch = int(nc.max_batch_size)
        elif hasattr(nc, "num_seqs_buckets") and nc.num_seqs_buckets:
            max_batch = int(max(nc.num_seqs_buckets))
        else:
            # Last-ditch fallback to dict-style neuron_config.
            max_batch = int(getattr(nc, "max_batch_size", 1) or 1)
        self.register_buffer(
            "recurrent_state_buffer",
            torch.zeros(
                max_batch, self.num_v_heads, self.head_k_dim, self.head_v_dim,
                dtype=self.dtype, device="cpu",
            ),
            persistent=False,
        )
        self.register_buffer(
            "conv_state_buffer",
            torch.zeros(
                max_batch, self.conv_dim, self.conv_kernel_size - 1,
                dtype=self.dtype, device="cpu",
            ),
            persistent=False,
        )

        # Pre-built per-layer DeltaNet masks. Stored as buffers so they
        # move with the module to the Neuron device, and so Dynamo can
        # see them as constants (no global-dict guard failure).
        # device="cpu" is required because the surrounding model is built
        # under `with torch.device("meta")` — without explicit cpu the
        # buffers would be meta tensors and `.to(neuron)` later fails.
        chunk = 128
        self.register_buffer(
            "deltanet_lower_mask",
            torch.tril(torch.ones(chunk, chunk, dtype=torch.float32, device="cpu"), diagonal=-1),
            persistent=False,
        )
        self.register_buffer(
            "deltanet_identity_mat",
            torch.eye(chunk, dtype=torch.float32, device="cpu"),
            persistent=False,
        )
        self.register_buffer(
            "deltanet_lower_mask_diag",
            torch.tril(torch.ones(chunk, chunk, dtype=torch.float32, device="cpu"), diagonal=0),
            persistent=False,
        )

        # Dummy KV cache attrs to satisfy bind_kv_cache contract. We never
        # read these in the DeltaNet forward — the real state lives in the
        # buffers above.
        self.k_cache = None
        self.v_cache = None

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.LongTensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
    ) -> torch.Tensor:
        """Forward pass — dispatches between prefill (CTE, fused NKI kernel)
        and decode (TKG, single-step recurrent update in PyTorch).

        SP boundary handling for PREFILL lives HERE (outside the traced
        graph), not inside _forward_prefill. This mirrors the GQA layer's
        pattern where all_gather/reduce_scatter are Python-level ops that
        happen before/after the NEFF-traced function body.

        DECODE does NOT all_gather: each rank processes its own token(s)
        with the replicated recurrent state already in its buffers, then
        the decoder-layer residual stays consistent because all ranks
        hold the same replicated weights and state. Gathering the batch
        dim in decode breaks the conv_state buffer (sized max_batch=1).
        """
        is_prefill = self._is_prefill(attn_metadata)

        if not is_prefill:
            # Decode: no SP gather. Replicated weights + replicated state
            # mean each rank computes the correct output for its tokens.
            return self._forward_decode(hidden_states)

        # PREFILL SP IN: all_gather scattered tokens → full sequence
        # (Python-level, NOT part of the traced NEFF graph).
        if self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        result = self._forward_prefill(hidden_states)

        # PREFILL SP OUT: replicated output → take this rank's slice.
        # DeltaNet weights are replicated, so every rank computed the same
        # full output. Just slice to restore SP layout.
        if self.world_size > 1:
            tokens = result.shape[0]
            shard_size = tokens // self.world_size
            rank = self.tp_group.rank_in_group
            result = result[rank * shard_size : (rank + 1) * shard_size].contiguous()

        return result

    # ── Prefill (CTE) ────────────────────────────────────────────────────

    def _forward_prefill(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """CTE prefill via fused NKI kernel.

        NOTE: SP boundary (all_gather/slice) is handled in forward(),
        NOT here. This function receives the FULL sequence and returns
        the FULL output. Putting collectives here would bake them into
        the traced NEFF graph, causing massive compile-time blowup.
        """
        # vllm_neuron passes [tokens, hidden]; treat as [B=1, S=tokens, hidden]
        # and follow PR #152's [B, S, H] convention through the rest of the layer.
        tokens, hidden = hidden_states.shape
        batch_size = 1
        seq_len = tokens
        x = hidden_states.view(batch_size, seq_len, hidden).to(self.dtype)

        # 1. Project: QKV (fused), Z, A, B
        qkv = torch.nn.functional.linear(x, self.in_proj_qkv_weight)
        z = torch.nn.functional.linear(x, self.in_proj_z_weight)
        a = torch.nn.functional.linear(x, self.in_proj_a_weight)
        b = torch.nn.functional.linear(x, self.in_proj_b_weight)

        q_raw = qkv[..., : self.key_dim]
        k_raw = qkv[..., self.key_dim : self.key_dim * 2]
        v_raw = qkv[..., self.key_dim * 2 :]

        # 2. Causal Conv1d on concat(q, k, v), then SiLU
        mixed = torch.cat([q_raw, k_raw, v_raw], dim=-1)  # [B, S, conv_dim]
        mixed = mixed.transpose(1, 2)                      # [B, conv_dim, S]
        # Depthwise conv1d via grouped F.conv1d
        conv_out = torch.nn.functional.conv1d(
            mixed,
            self.conv1d_weight,                           # [conv_dim, 1, K]
            bias=None,
            stride=1,
            padding=self.conv_kernel_size - 1,
            groups=self.conv_dim,
        )[:, :, :seq_len]
        mixed_post_conv = torch.nn.functional.silu(conv_out).transpose(1, 2)
        # [B, S, conv_dim]

        # Split back to q, k, v
        q = mixed_post_conv[..., : self.key_dim]
        k = mixed_post_conv[..., self.key_dim : self.key_dim * 2]
        v = mixed_post_conv[..., self.key_dim * 2 :]

        # 3. Reshape to head layout
        q = q.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        k = k.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        v = v.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        # 4. Compute decay g and write-gate beta
        # g = -exp(A_log) * softplus(a + dt_bias)        [B, S, H_v] in fp32
        # beta = sigmoid(b)                              [B, S, H_v]
        a_f32 = a.float()
        g = -self.A_log.exp() * torch.nn.functional.softplus(a_f32 + self.dt_bias)
        beta = b.sigmoid().to(self.dtype)

        # 5. Expand K heads to match V heads (16 -> 32 = repeat 2x)
        kv_repeat = self.num_v_heads // self.num_k_heads  # 2
        if kv_repeat > 1:
            q = (
                q.unsqueeze(3)
                .expand(-1, -1, -1, kv_repeat, -1)
                .reshape(batch_size, seq_len, self.num_v_heads, self.head_k_dim)
            )
            k = (
                k.unsqueeze(3)
                .expand(-1, -1, -1, kv_repeat, -1)
                .reshape(batch_size, seq_len, self.num_v_heads, self.head_k_dim)
            )

        # 6. Transpose to (B, H, S, dim), float32 for the recurrence
        q = q.transpose(1, 2).contiguous().float()
        k = k.transpose(1, 2).contiguous().float()
        v = v.transpose(1, 2).contiguous().float()
        g = g.transpose(1, 2).contiguous().float()
        beta = beta.transpose(1, 2).contiguous().float()

        # 7. l2norm Q and K, scale Q by 1/sqrt(k_dim)
        q = _deltanet_l2norm(q, dim=-1)
        k = _deltanet_l2norm(k, dim=-1)
        scale = 1.0 / (self.head_k_dim ** 0.5)
        q = q * scale

        # 8. Pad S to multiple of 128
        chunk = 128
        pad = (chunk - seq_len % chunk) % chunk
        if pad > 0:
            q = torch.nn.functional.pad(q, (0, 0, 0, pad))
            k = torch.nn.functional.pad(k, (0, 0, 0, pad))
            v = torch.nn.functional.pad(v, (0, 0, 0, pad))
            g = torch.nn.functional.pad(g, (0, pad))
            beta = torch.nn.functional.pad(beta, (0, pad))
        total_seq = seq_len + pad

        # 9. Flatten (B, H) -> per-(b,h) kernel calls.
        BH = batch_size * self.num_v_heads
        q_flat = q.reshape(BH, total_seq, self.head_k_dim).contiguous()
        k_flat = k.reshape(BH, total_seq, self.head_k_dim).contiguous()
        v_flat = v.reshape(BH, total_seq, self.head_v_dim).contiguous()
        g_flat = g.reshape(BH, total_seq).unsqueeze(-1).contiguous()
        beta_flat = beta.reshape(BH, total_seq).unsqueeze(-1).contiguous()

        # 10. Kernel masks (registered buffers — moved with the module).
        from .nki_kernels import call_deltanet_fused
        lower_mask = self.deltanet_lower_mask
        identity_mat = self.deltanet_identity_mat
        lower_mask_diag = self.deltanet_lower_mask_diag

        # 11. Per-(b,h) kernel calls
        outputs = []
        states = []
        for bh in range(BH):
            out_bh, state_bh = call_deltanet_fused(
                q_flat[bh],
                k_flat[bh],
                v_flat[bh],
                g_flat[bh],
                beta_flat[bh],
                lower_mask,
                identity_mat,
                lower_mask_diag,
            )
            outputs.append(out_bh)
            states.append(state_bh)

        # 12. Reassemble: [B, H_v, S+pad, head_v_dim] → [B, S, value_dim]
        output = torch.stack(outputs, dim=0)
        output = output.reshape(batch_size, self.num_v_heads, total_seq, self.head_v_dim)
        output = output[:, :, :seq_len]                              # drop padding
        # [B, H, S, D] -> [B, S, H, D] -> [B, S, value_dim]
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.value_dim)

        # 13. Stash recurrent state for decode path
        final_state = torch.stack(states, dim=0)
        final_state = final_state.reshape(batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim)
        new_rec_state = final_state.to(self.dtype) + self.recurrent_state_buffer * 0
        self.recurrent_state_buffer.data.copy_(new_rec_state)

        # Stash conv state from last (kernel-1) pre-conv mixed tokens.
        if seq_len >= self.conv_kernel_size - 1:
            new_conv_state = mixed[:, :, -(self.conv_kernel_size - 1):].contiguous()
        else:
            new_conv_state = torch.nn.functional.pad(
                mixed, (self.conv_kernel_size - 1 - seq_len, 0)
            )[:, :, -(self.conv_kernel_size - 1):]
        new_conv_state = new_conv_state.to(self.dtype) + self.conv_state_buffer * 0
        self.conv_state_buffer.data.copy_(new_conv_state)

        # 14. RMSNorm over head_v_dim, then z gate (silu), then out_proj
        # output: [B, S, value_dim] -> [B, S, num_v_heads, head_v_dim]
        out_h = output.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        # Per-head RMSNorm: variance over head_v_dim
        x_f32 = out_h.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        out_h = (x_f32 * torch.rsqrt(variance + self.rms_norm_eps)).to(self.dtype)
        out_h = out_h * self.norm_weight  # broadcasts on last dim

        # z gate: [B, S, value_dim] silu
        z_gate = torch.nn.functional.silu(z)
        out_flat = out_h.reshape(batch_size, seq_len, self.value_dim)
        gated = out_flat * z_gate

        # Output projection
        result = torch.nn.functional.linear(gated, self.out_proj_weight)

        # vllm_neuron's residual layout is [tokens, hidden]
        return result.view(seq_len, hidden)

    def _is_prefill(self, attn_metadata) -> bool:
        if attn_metadata is None:
            return True  # default to prefill for unit testing
        layer_name = f"layers.{self.layer_idx}.self_attn"
        if layer_name not in attn_metadata:
            return True
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]
        return max_query_len > decode_token_threshold

    # ── Decode (TKG) ─────────────────────────────────────────────────────

    def _forward_decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """TKG single-step recurrent update.

        NOTE: SP boundary (all_gather/slice) is handled in forward(),
        NOT here. This function receives the FULL batch and returns
        the FULL output.

        Mirrors PR #152's `_recurrent_step`. No NKI kernel — just PyTorch
        elementwise ops on shape [B=1, H_v, 1, head_dim]. Reads/writes
        `recurrent_state_buffer` and `conv_state_buffer`.

        Math (per-(b, h)):
            new_state = state * exp(g_t)
            kv_mem    = sum(new_state * k_t, axis=-2)
            delta     = (v_t - kv_mem) * beta_t
            new_state = new_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            output    = sum(new_state * q_t, axis=-2)
        """
        tokens, hidden = hidden_states.shape
        # Decode path expects exactly one token per sequence; vllm_neuron
        # batches B sequences into [B*1, hidden] = [B, hidden].
        batch_size = tokens
        seq_len = 1
        x = hidden_states.view(batch_size, seq_len, hidden).to(self.dtype)

        # 1. Project (same as prefill but for one token)
        qkv = torch.nn.functional.linear(x, self.in_proj_qkv_weight)
        z = torch.nn.functional.linear(x, self.in_proj_z_weight)
        a = torch.nn.functional.linear(x, self.in_proj_a_weight)
        b = torch.nn.functional.linear(x, self.in_proj_b_weight)

        q_raw = qkv[..., : self.key_dim]
        k_raw = qkv[..., self.key_dim : self.key_dim * 2]
        v_raw = qkv[..., self.key_dim * 2 :]

        # 2. Causal Conv1d using stored state
        # PR #152 pattern: conv_state holds last (kernel-1) tokens; new input
        # is concatenated, then a per-channel weighted sum over the 4-tap window
        # gives one output token per channel.
        mixed_now = torch.cat([q_raw, k_raw, v_raw], dim=-1)  # [B, 1, conv_dim]
        mixed_now = mixed_now.transpose(1, 2)                  # [B, conv_dim, 1]

        conv_state = self.conv_state_buffer[:batch_size]      # [B, conv_dim, 3]
        conv_input = torch.cat([conv_state, mixed_now], dim=-1)  # [B, conv_dim, 4]

        # Depthwise: weight [conv_dim, 1, 4] → [conv_dim, 4]
        w = self.conv1d_weight.squeeze(1)
        # Sum over kernel taps: out = sum_k w[:, k] * conv_input[:, :, k]
        conv_out = (w.unsqueeze(0) * conv_input).sum(dim=-1, keepdim=True)
        # [B, conv_dim, 1]

        mixed_post_conv = torch.nn.functional.silu(conv_out).transpose(1, 2)
        # [B, 1, conv_dim]

        # New conv state: shift left, append latest pre-conv mixed
        new_conv_state = torch.cat([conv_state[:, :, 1:], mixed_now], dim=-1)
        # Buffer-dependency trick
        new_conv_state = new_conv_state.to(self.dtype) + self.conv_state_buffer * 0
        self.conv_state_buffer.data.copy_(new_conv_state)

        # Split q/k/v
        q = mixed_post_conv[..., : self.key_dim]
        k = mixed_post_conv[..., self.key_dim : self.key_dim * 2]
        v = mixed_post_conv[..., self.key_dim * 2 :]

        # 3. Reshape to head layout
        q = q.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        k = k.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        v = v.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        # 4. Decay g + write-gate beta
        a_f32 = a.float()
        g = -self.A_log.exp() * torch.nn.functional.softplus(a_f32 + self.dt_bias)
        beta = b.sigmoid().to(torch.float32)

        # 5. Expand K heads: 16 → 32
        kv_repeat = self.num_v_heads // self.num_k_heads
        if kv_repeat > 1:
            q = (
                q.unsqueeze(3)
                .expand(-1, -1, -1, kv_repeat, -1)
                .reshape(batch_size, seq_len, self.num_v_heads, self.head_k_dim)
            )
            k = (
                k.unsqueeze(3)
                .expand(-1, -1, -1, kv_repeat, -1)
                .reshape(batch_size, seq_len, self.num_v_heads, self.head_k_dim)
            )

        # 6. Transpose to (B, H, S=1, dim), float32 for the recurrence
        q = q.transpose(1, 2).contiguous().float()
        k = k.transpose(1, 2).contiguous().float()
        v = v.transpose(1, 2).contiguous().float()
        g = g.transpose(1, 2).contiguous().float()
        beta = beta.transpose(1, 2).contiguous().float()

        # 7. l2norm + scale
        q = _deltanet_l2norm(q, dim=-1)
        k = _deltanet_l2norm(k, dim=-1)
        scale = 1.0 / (self.head_k_dim ** 0.5)
        q = q * scale

        # 8. Pull recurrent state for these batches
        recurrent_state = self.recurrent_state_buffer[:batch_size].float()
        # Shape: [B, H, head_k_dim, head_v_dim]

        # 9. Single-step recurrent update (PR #152 _recurrent_step)
        q_t = q[:, :, 0]                # [B, H, head_k_dim]
        k_t = k[:, :, 0]                # [B, H, head_k_dim]
        v_t = v[:, :, 0]                # [B, H, head_v_dim]
        g_t = g[:, :, 0].exp().unsqueeze(-1).unsqueeze(-1)
        # [B, H, 1, 1]
        beta_t = beta[:, :, 0].unsqueeze(-1)
        # [B, H, 1]

        new_state = recurrent_state * g_t
        # kv_mem[b, h, v_dim] = sum_k new_state[b, h, k_dim, v_dim] * k_t[b, h, k_dim]
        kv_mem = (new_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        # outer product update: new_state += k_t ⊗ delta
        new_state = new_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # output = sum_k new_state * q_t  →  [B, H, head_v_dim]
        out_one = (new_state * q_t.unsqueeze(-1)).sum(dim=-2)
        # Add a singleton seq dim → [B, H, 1, head_v_dim]
        out_h = out_one.unsqueeze(2)

        # 10. Write back state (with buffer-dependency trick)
        new_state_typed = new_state.to(self.dtype)
        new_rec_state = new_state_typed + self.recurrent_state_buffer * 0
        self.recurrent_state_buffer.data.copy_(new_rec_state)

        # 11. Reshape and finish: RMSNorm, z gate, out_proj
        # out_h: [B, H, 1, head_v_dim] → [B, 1, H, head_v_dim]
        out_h = out_h.transpose(1, 2).contiguous()
        # Per-head RMSNorm
        x_f32 = out_h.float()
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        out_h = (x_f32 * torch.rsqrt(variance + self.rms_norm_eps)).to(self.dtype)
        out_h = out_h * self.norm_weight

        # Flatten to [B, 1, value_dim] then z gate + out_proj
        out_flat = out_h.reshape(batch_size, seq_len, self.value_dim)
        z_gate = torch.nn.functional.silu(z)
        gated = out_flat * z_gate
        result = torch.nn.functional.linear(gated, self.out_proj_weight)

        # Back to vllm_neuron's [tokens, hidden]
        return result.view(tokens, hidden)


# ============================================================================
# Section 5: Dense SwiGLU MLP
# Mirrors vllm_neuron.model.llama3.LlamaMLP. Used for ALL 32 layers.
# ============================================================================


class Qwen3_5MLP(nn.Module):
    """Dense SwiGLU MLP with NF.mlp.

    hidden=2560, intermediate=9216, SiLU activation,
    no bias. Pattern mirrors `vllm_neuron.model.llama3.LlamaMLP` minus
    the optional MLP-DP supergroup (skipped for Phase 3 simplicity;
    re-add in Phase 8 if needed).

    Parallelism:
        - gate/up_proj weights: [hidden, intermediate / TP] (column-parallel)
        - down_proj weight: [intermediate / TP, hidden] (row-parallel)
        - Prefill: all-gather → NF.mlp → reduce-scatter
        - Decode: NF.mlp → all-reduce
    """

    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        self.hidden_size = config.hidden_size
        if config.intermediate_size % self.world_size != 0:
            raise ValueError(
                f"intermediate_size ({config.intermediate_size}) must be "
                f"divisible by TP world_size ({self.world_size})"
            )
        self.intermediate_size_per_rank = config.intermediate_size // self.world_size
        self.dtype = config.torch_dtype

        self.gate_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.intermediate_size_per_rank, dtype=self.dtype)
        )
        self.up_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.intermediate_size_per_rank, dtype=self.dtype)
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(self.intermediate_size_per_rank, self.hidden_size, dtype=self.dtype)
        )

        self._setup_weight_loaders()

    def _setup_weight_loaders(self) -> None:
        gate_up_loader = sharding_weight_loader(
            shard_dim=1,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.world_size,
            is_storage_transposed=True,
        )
        down_loader = sharding_weight_loader(
            shard_dim=0,
            shard_size=self.intermediate_size_per_rank,
            num_shards=self.world_size,
            is_storage_transposed=True,
        )
        set_weight_loader(self.gate_proj_weight, gate_up_loader)
        set_weight_loader(self.up_proj_weight, gate_up_loader)
        set_weight_loader(self.down_proj_weight, down_loader)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        is_decode: bool,
        rank: torch.Tensor | None,
    ) -> torch.Tensor:
        is_prefill = not is_decode

        # SP: all-gather to full sequence before MLP during prefill
        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)

        hidden_states = hidden_states.to(self.dtype)
        # SwiGLU: down(silu(gate(x)) * up(x))
        output = NF.mlp(
            hidden_states,
            self.gate_proj_weight,
            self.up_proj_weight,
            self.down_proj_weight,
        )

        if is_prefill:
            if self.world_size > 1:
                output = self.tp_group.reduce_scatter(output, dim=0)
        else:
            self.tp_group.all_reduce(output)

        return output


# ============================================================================
# Section 6: Decoder Layer (dispatches by layer_types[layer_idx])
# hybrid layer-type dispatch. Otherwise mirrors
#                      qwen3_moe Qwen3MoeDecoderLayer.
# ============================================================================


class Qwen3_5DecoderLayer(nn.Module):
    """One transformer decoder block with attn-type dispatch.

    For Qwen3.5: layer_types[layer_idx] decides whether self_attn is
    a full-attention GQA layer or a linear-attention DeltaNet layer.
    """

    def __init__(self, config: Qwen3_5Config, batch_size: int, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]

        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        # dispatch on layer type
        if self.layer_type == "full_attention":
            self.self_attn = Qwen3_5GQAAttention(config, layer_idx=layer_idx)
        elif self.layer_type == "linear_attention":
            self.self_attn = Qwen3_5DeltaNetAttention(config, layer_idx=layer_idx)
        else:
            raise ValueError(
                f"Unknown layer_type {self.layer_type!r} at layer {layer_idx}"
            )

        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.mlp = Qwen3_5MLP(config)

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

    def _is_decode(self, attn_metadata) -> bool:
        layer_name = f"layers.{self.layer_idx}.self_attn"
        max_query_len = attn_metadata[layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[layer_name]["decode_token_threshold"]
        return max_query_len <= decode_token_threshold

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        is_decode = self._is_decode(attn_metadata)

        # Self-attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            position_embeddings=position_embeddings,
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(
            hidden_states,
            positions=positions,
            is_decode=is_decode,
            rank=rank,
        )
        hidden_states = residual + hidden_states

        return hidden_states


# ============================================================================
# Section 7: Model Backbone
# Mirrors Qwen3MoeModel. Differences: only DeltaNet layers don't manage
# their own KV cache — Phase 5 will return dummy KV for them.
# ============================================================================


class Qwen3_5Model(nn.Module):
    """Qwen3.5 transformer backbone."""

    def __init__(self, config: Qwen3_5Config, batch_size: int) -> None:
        super().__init__()
        self.config = config

        self.tp_group = get_tp_group()
        self.world_size = self.tp_group.world_size

        self.embed_tokens = VocabDimShardedEmbedding(
            vocab_size=config.vocab_size,
            embed_dim=config.hidden_size,
            dtype=config.torch_dtype,
            tp_group=self.tp_group.device_group,
        )

        self.layers = nn.ModuleList(
            Qwen3_5DecoderLayer(config, batch_size, i)
            for i in range(config.num_hidden_layers)
        )

        self.norm = Qwen3_5RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.torch_dtype
        )
        self.rotary_emb = Qwen3_5RotaryEmbedding(config)

        # Embedding sharding
        set_weight_loader(
            self.embed_tokens.weight,
            sharding_weight_loader_with_padding(
                shard_dim=0,
                shard_size=self.embed_tokens.vocab_size_per_rank,
                num_shards=self.world_size,
                pad_dim=1,
                padded_size=config.hidden_size,
                unpadded_size=config.hidden_size,
            ),
        )
        set_weight_loader(
            self.norm.weight, last_dim_padding_weight_loader(config.hidden_size)
        )

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        attn_metadata: object | None = None,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Use any GQA layer's metadata to detect prefill (DeltaNet layers
        # share the same flag because the dispatch is global).
        first_full_idx = next(
            i for i, lt in enumerate(self.config.layer_types) if lt == "full_attention"
        )
        meta_key = f"layers.{first_full_idx}.self_attn"
        max_query_len = attn_metadata[meta_key]["max_query_len"]
        decode_token_threshold = attn_metadata[meta_key]["decode_token_threshold"]
        is_prefill = max_query_len > decode_token_threshold

        hidden_states = self.embed_tokens(
            input_ids, scatter_tokens=is_prefill, rank=rank
        )
        position_embeddings = self.rotary_emb(
            positions, device=hidden_states.device, dtype=hidden_states.dtype
        )

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                positions=positions,
                position_embeddings=position_embeddings,
                attn_metadata=attn_metadata,
                rank=rank,
            )

        hidden_states = self.norm(hidden_states)

        if is_prefill and self.world_size > 1:
            hidden_states = self.tp_group.all_gather(hidden_states, dim=0)
        return hidden_states


# ============================================================================
# Section 8: ForCausalLM (LM head + sampler)
# Minimal Phase 2 — wires backbone + tied embedding lm_head. Phase 6 will
# add the full weight-mapping table. Phase 7 will wire `Sampler` properly.
# ============================================================================


class Qwen3_5ForConditionalGeneration(nn.Module):
    """Qwen3.5 model for conditional generation.

    Mirrors `Qwen3MoeForCausalLM`'s shape: column-parallel lm_head + on-device
    sampler. Tied embeddings handled in the weight loader (lm_head shares
    embed_tokens data).
    """

    def __init__(self, config: Qwen3_5Config, batch_size: int = 1) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3_5Model(config, batch_size=batch_size)

        from vllm.distributed.parallel_state import get_tp_group as _get_tp_group
        import vllm_neuron.nn as _neuron_nn
        from vllm_neuron.nn.sampler import Sampler as _Sampler
        from vllm_neuron.utils.weight_loader import (
            sharding_weight_loader_with_padding as _padding_loader,
            set_weight_loader as _set_loader,
        )

        self.tp_group = _get_tp_group()
        self.world_size = self.tp_group.world_size

        # Sampling config (on-device sampling when configured)
        self.on_device_sampling_config = (
            config.neuron_config.on_device_sampling_config
            if config.neuron_config is not None
            else None
        )
        debug_logits_enabled = (
            config.neuron_config is not None
            and getattr(config.neuron_config, "debug_logits_dir", None) is not None
        )
        max_logprobs = (
            getattr(config.neuron_config, "max_logprobs", 0)
            if config.neuron_config is not None else 0
        )
        self._gather_logits = (max_logprobs != 0) or debug_logits_enabled

        # Column-parallel LM head — vocab dim is sharded across TP ranks
        self.lm_head = _neuron_nn.ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            dtype=config.torch_dtype,
            gather_output=not self.on_device_sampling_config,
            tp_group=self.tp_group.device_group,
        )

        if self.on_device_sampling_config is not None:
            self.sampler = _Sampler(
                self.on_device_sampling_config,
                process_group=self.tp_group.device_group,
            )

        # Shard lm_head on vocab dim. Even though the checkpoint is tied
        # (lm_head loaded from embed_tokens), the loader still wants the
        # explicit sharding spec on this parameter.
        _set_loader(
            self.lm_head.weight,
            _padding_loader(
                shard_dim=0,
                shard_size=config.vocab_size // self.world_size,
                num_shards=self.world_size,
                pad_dim=1,
                padded_size=config.hidden_size,
                unpadded_size=config.hidden_size,
            ),
        )

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None,
    ) -> "Qwen3_5ForConditionalGeneration":
        cfg = Qwen3_5Config.from_configs(hf_config, neuron_config)
        return cls(cfg)

    def get_weight_mappings(self) -> dict[str, list[str]]:
        """Return the {flat_name: [hf_keys]} map for `SafetensorsCheckpoint`.

        deliverable. vllm_neuron's checkpoint loader calls this
        method (or grabs `weight_mappings` attribute) when loading.
        """
        from .weight_loaders_bf16 import build_weight_mappings
        return build_weight_mappings(self.config)

    def get_kv_spec(self):
        """Return the KV cache spec for this hybrid model.

        Mirrors `Qwen3MoeForCausalLM.get_kv_spec` but with hybrid handling:
        - Full-attention (GQA) layers: real KV cache spec
        - DeltaNet layers: dummy zero-head KV spec (PR #152 pattern). The
          real recurrent state lives in side-channel buffers on the layer
          (`recurrent_state_buffer`, `conv_state_buffer`) — Phase 5's
          design from PR #152.
        """
        from vllm_neuron.model.kv_cache import KVSpec, LayerSpec

        layers = []
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            attn = layer.self_attn

            if layer.layer_type == "full_attention":
                # Real KV cache for GQA layers
                layers.append(
                    LayerSpec(
                        name=layer_name,
                        num_kv_heads=attn.num_key_value_heads_per_rank,
                        head_size=attn.head_dim,
                        dtype=attn.dtype,
                        sliding_window_size=None,
                        chunk_size=None,
                    )
                )
            else:
                # DeltaNet: dummy KV with 1 head, head_size=1 to keep the
                # cache manager happy. The actual recurrent state lives
                # in `attn.recurrent_state_buffer` + `attn.conv_state_buffer`.
                layers.append(
                    LayerSpec(
                        name=layer_name,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=self.config.torch_dtype,
                        sliding_window_size=None,
                        chunk_size=None,
                    )
                )
        return KVSpec(layers=layers)

    def bind_kv_cache(self, kv_caches: dict[str, list[torch.Tensor]]):
        """Bind external KV cache tensors to each attention layer.

        Mirrors `Qwen3MoeForCausalLM.bind_kv_cache`. For DeltaNet layers
        the bound K/V tensors are dummies (1 head × 1 head_size) — the
        real state lives in the side-channel buffers.
        """
        for i, layer in enumerate(self.model.layers):
            layer_name = f"layers.{i}.self_attn"
            if layer_name not in kv_caches:
                raise Exception(f"KV cache for layer {layer_name} not initialized")
            attn = layer.self_attn
            # Both layer types accept k_cache/v_cache attributes (full-attn
            # uses them, DeltaNet ignores them — the dummies just satisfy
            # the cache manager).
            attn.k_cache = kv_caches[layer_name][0]
            attn.v_cache = kv_caches[layer_name][1]

    def load_weights(
        self,
        checkpoint_path: str,
        device: torch.device,
        cache_dir: str | None = None,
    ) -> None:
        """Load HF safetensors → our flat-named parameters.

        Mirrors `Qwen3MoeForCausalLM.load_weights`. The mapping covers
        every HF key in the checkpoint; weight loaders (attached to each
        parameter via `set_weight_loader`) handle TP sharding.
        """
        from vllm_neuron.utils.checkpoints import SafetensorsCheckpoint

        tp_rank = self.model.tp_group.rank_in_group
        tp_size = self.model.world_size

        # Build the flat mappings. SafetensorsCheckpoint expects values
        # to be a single string when length-1, list when fused. We get
        # lists from build_weight_mappings; flatten singletons.
        raw = self.get_weight_mappings()
        mappings: dict[str, object] = {}
        for k, v in raw.items():
            mappings[k] = v[0] if len(v) == 1 else v

        checkpoint = SafetensorsCheckpoint(checkpoint_path, cache_dir)
        rank_sharded = checkpoint.load_sharded_pipelined(
            tp_rank,
            tp_size,
            self,
            mappings,
            device,
        ).state_dict

        self.load_state_dict(rank_sharded, strict=False, assign=True)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        attn_metadata: object | None = None,
        sampling_positions: torch.Tensor | None = None,
        sampling_params: torch.Tensor | None = None,
        spec_decode_metadata=None,
        logit_mask: torch.Tensor | None = None,
        rank: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is not None:
            raise ValueError("Input Embedding as Inputs is Not Supported Yet.")

        positions = positions.to(torch.int32)

        first_layer_name = "layers.0.self_attn"
        max_query_len = attn_metadata[first_layer_name]["max_query_len"]
        decode_token_threshold = attn_metadata[first_layer_name]["decode_token_threshold"]
        is_prefill = max_query_len > decode_token_threshold

        T = input_ids.shape[0]
        if is_prefill and ((T <= self.world_size) or (T % self.world_size != 0)):
            raise ValueError(
                f"Prompt Length ({T}) must be > world_size ({self.world_size}) for SP."
            )

        hidden_states = self.model(
            input_ids, positions, attn_metadata=attn_metadata, rank=rank
        )

        # Sampling slice + LM head
        hidden_for_logits = torch.index_select(
            hidden_states, dim=0, index=sampling_positions
        )
        hidden_for_logits = hidden_for_logits.to(self.config.torch_dtype)

        logits = self.lm_head(hidden_for_logits)

        if self.on_device_sampling_config is None:
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

        if spec_decode_metadata is not None:
            from vllm_neuron.nn.rejection_sampler import rejection_sampler
            return rejection_sampler(spec_decode_metadata, sampled_tokens)

        return sampled_tokens, gathered_logits
