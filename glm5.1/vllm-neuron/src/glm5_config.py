# SPDX-License-Identifier: Apache-2.0
"""
GLM 5.1 Configuration for vLLM-Neuron
=======================================
Adapted from DeepSeek V3.2 config. Same MLA + MoE + DSA architecture
with different dimensions.

Key differences from DeepSeek V3.2:
- hidden_size: 6144 (vs 7168)
- num_hidden_layers: 78 (vs 61)
- num_attention_heads: 64 (vs 128)
- q_lora_rank: 2048 (vs 1536)
- qk_nope_head_dim: 192 (vs 128)
- v_head_dim: 256 (vs 128)
- No group-limited routing (n_group=1, topk_group=1)
- vocab_size: 154880 (vs 129280)
"""

import json
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig


@dataclass
class Glm5Config:
    """GLM 5.1 model configuration for vLLM-Neuron deployment."""

    # ── Core transformer ─────────────────────────────────────────────────
    vocab_size: int = 154880
    hidden_size: int = 6144
    num_hidden_layers: int = 78
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    rms_norm_eps: float = 1e-5
    torch_dtype: torch.dtype = torch.bfloat16

    # ── MLA (Multi-Head Latent Attention) ────────────────────────────────
    q_lora_rank: int = 2048
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 192
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256

    # ── MoE ──────────────────────────────────────────────────────────────
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    n_group: int = 1           # No group-limited routing (unlike DeepSeek's 8)
    topk_group: int = 1        # No group selection
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True
    moe_intermediate_size: int = 2048
    intermediate_size: int = 12288  # dense MLP layers (0-2)
    first_k_dense_replace: int = 3  # layers 0-2 use dense MLP

    # ── MLP layer pattern ────────────────────────────────────────────────
    # Explicit pattern: layers 0-2 are "dense", rest are "sparse" (MoE)
    mlp_layer_types: list = None  # populated in __post_init__

    # ── DSA (DeepSeek Sparse Attention) ──────────────────────────────────
    index_n_heads: int = 32
    index_head_dim: int = 128
    index_topk: int = 2048
    use_dsa: bool = False  # Start with dense attention, enable DSA in Phase 2
    dsa_max_seq_len: int = 4096

    # ── Indexer types per layer ──────────────────────────────────────────
    # "full" = layer has its own learned indexer
    # "shared" = layer reuses index from the previous "full" layer
    indexer_types: list = None  # populated in __post_init__ when use_dsa=True

    # ── RoPE ─────────────────────────────────────────────────────────────
    rope_theta: float = 10000.0
    rope_scaling: dict = None  # GLM 5.1 uses standard RoPE (no YaRN by default)
    max_position_embeddings: int = 202752

    # ── Special tokens ───────────────────────────────────────────────────
    pad_token_id: int = None
    bos_token_id: int = 0
    eos_token_id: int = 1
    tie_word_embeddings: bool = False

    # ── Activation ───────────────────────────────────────────────────────
    hidden_act: str = "silu"

    # ── FP8 quantization info ────────────────────────────────────────────
    quantization_config: dict = None

    # ── Framework config ─────────────────────────────────────────────────
    neuron_config: object = None

    # ── Derived fields ───────────────────────────────────────────────────
    num_kv_heads: int = 1          # MLA: single compressed KV
    head_dim: int = 576            # qk_rope_head_dim + kv_lora_rank
    qk_head_dim: int = 256         # qk_nope_head_dim + qk_rope_head_dim

    def __post_init__(self):
        # MLA: single compressed KV representation per head
        self.num_kv_heads = 1
        # head_dim for KV cache = rope_dim + kv_lora_rank
        self.head_dim = self.qk_rope_head_dim + self.kv_lora_rank  # 64 + 512 = 576
        # QK head dim = nope + rope
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim  # 192 + 64 = 256

        # Build MLP layer types if not provided
        if self.mlp_layer_types is None:
            self.mlp_layer_types = (
                ["dense"] * self.first_k_dense_replace +
                ["sparse"] * (self.num_hidden_layers - self.first_k_dense_replace)
            )

        # Build indexer types (if DSA enabled)
        if self.use_dsa and self.indexer_types is None:
            # First layer is "full", then every 4th layer is "full", rest "shared"
            index_topk_freq = 4
            self.indexer_types = []
            for i in range(self.num_hidden_layers):
                if i == 0 or i % index_topk_freq == 0:
                    self.indexer_types.append("full")
                else:
                    self.indexer_types.append("shared")

    @classmethod
    def from_hf_config(cls, hf_config, neuron_config=None):
        """Create from a HuggingFace PretrainedConfig (GlmMoeDsaConfig)."""
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            config_dict = hf_config.to_dict()
        else:
            config_dict = hf_config

        # Map field names
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

        # Handle torch_dtype string → actual dtype
        if "torch_dtype" in filtered_dict and isinstance(filtered_dict["torch_dtype"], str):
            filtered_dict["torch_dtype"] = getattr(torch, filtered_dict["torch_dtype"])

        # Handle rope_parameters → rope_scaling
        if "rope_parameters" in config_dict and "rope_scaling" not in filtered_dict:
            filtered_dict["rope_scaling"] = config_dict["rope_parameters"]

        if neuron_config is not None:
            filtered_dict["neuron_config"] = neuron_config

        return cls(**filtered_dict)

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Returns True if this layer uses MoE routing."""
        return self.mlp_layer_types[layer_idx] == "sparse"

    @property
    def total_expert_params_per_layer(self) -> int:
        """Total expert params per MoE layer (gate + up + down × n_experts)."""
        # Each expert: gate_proj(h→moe_inter) + up_proj(h→moe_inter) + down_proj(moe_inter→h)
        per_expert = 3 * self.hidden_size * self.moe_intermediate_size
        return per_expert * self.n_routed_experts + per_expert * self.n_shared_experts

    @property
    def model_size_estimate_gb(self) -> float:
        """Rough model size estimate in GB (BF16)."""
        # Embedding + LM head
        embed = self.vocab_size * self.hidden_size * 2  # 2 bytes per bf16
        # Per dense layer (attention + MLP)
        attn_per_layer = (
            self.hidden_size * self.q_lora_rank +      # q_a_proj
            self.q_lora_rank * self.num_attention_heads * self.qk_head_dim +  # q_b_proj
            self.hidden_size * (self.kv_lora_rank + self.qk_rope_head_dim) +  # kv_a_proj
            self.kv_lora_rank * self.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim) +  # kv_b_proj
            self.num_attention_heads * self.v_head_dim * self.hidden_size  # o_proj
        ) * 2  # bf16
        dense_mlp = 3 * self.hidden_size * self.intermediate_size * 2
        moe_mlp = self.total_expert_params_per_layer * 2

        total = embed * 2  # embed + lm_head
        for i in range(self.num_hidden_layers):
            total += attn_per_layer
            if self.is_moe_layer(i):
                total += moe_mlp
            else:
                total += dense_mlp

        return total / (1024**3)
