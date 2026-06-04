# SPDX-License-Identifier: Apache-2.0
"""
DeepSeek V3.2 Configuration
================================
Model-specific configuration for DeepSeek V3.2 (DeepseekV32ForCausalLM).
This is V3.2 (model_type: deepseek_v32), NOT V3.

Architecture overview:
- MLA (Multi-Head Latent Attention) with compressed KV cache
- MoE with 256 routed experts + 1 shared expert, group-limited routing
- DSA (DeepSeek Sparse Attention) via Lightning Indexer (Phase 2)
- YaRN RoPE with interleaved format
- 61 layers: 3 dense MLP + 58 MoE
"""

import json
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class DeepseekV32Config:
    # ── Core transformer ─────────────────────────────────────────────────
    vocab_size: int = 129280
    hidden_size: int = 7168
    num_hidden_layers: int = 61
    num_attention_heads: int = 128
    rms_norm_eps: float = 1e-6
    torch_dtype: torch.dtype = torch.bfloat16

    # ── MLA (Multi-Head Latent Attention) ────────────────────────────────
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128

    # ── MoE ──────────────────────────────────────────────────────────────
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 8
    n_expert_groups: int = 8
    n_limited_groups: int = 4
    routed_scaling_factor: float = 2.5
    scoring_func: str = "sigmoid"
    moe_intermediate_size: int = 2048
    intermediate_size: int = 18432  # dense layers
    first_k_dense_replace: int = 3  # layers 0-2 use dense MLP

    # ── DSA (DeepSeek Sparse Attention) — V3.2 specific ──────────────────
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048
    # Enable the Lightning Indexer. Phase 1 (dense attention) ships with
    # this OFF so the existing working model is not regressed. Phase 2 turns
    # it on once the Neuron topk kernel is in place.
    use_dsa: bool = True
    # Indexer cache length. Set to max_model_len by the model constructor.
    dsa_max_seq_len: int = 3072

    # ── RoPE (YaRN) ─────────────────────────────────────────────────────
    rope_theta: float = 10000.0
    rope_scaling: dict = field(
        default_factory=lambda: {
            "type": "yarn",
            "factor": 40.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 0.0,
        }
    )
    max_position_embeddings: int = 163840

    # ── Special tokens ───────────────────────────────────────────────────
    pad_token_id: int | None = None
    bos_token_id: int = 0
    eos_token_id: int = 1

    # ── FP8 quantization info (for reference, not used in BF16 path) ────
    quantization_config: dict | None = None

    # ── Framework config ─────────────────────────────────────────────────
    neuron_config: NeuronConfig | None = None

    # ── Derived fields (computed in __post_init__) ───────────────────────
    num_kv_heads: int = 1
    head_dim: int = 576
    qk_head_dim: int = 192

    def __post_init__(self):
        self.num_kv_heads = 1  # MLA: single compressed KV representation
        self.head_dim = self.qk_rope_head_dim + self.kv_lora_rank  # 64 + 512 = 576
        self.qk_head_dim = (
            self.qk_nope_head_dim + self.qk_rope_head_dim
        )  # 128 + 64 = 192

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            if (
                hasattr(hf_config, "quantization_config")
                and hf_config.quantization_config is None
            ):
                delattr(hf_config, "quantization_config")
                config_dict = hf_config.to_dict()
                hf_config.quantization_config = None
            else:
                config_dict = hf_config.to_dict()
        else:
            config_dict = hf_config

        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in field_names}

        if "torch_dtype" in filtered_dict and isinstance(
            filtered_dict["torch_dtype"], str
        ):
            filtered_dict["torch_dtype"] = getattr(torch, filtered_dict["torch_dtype"])

        filtered_dict["neuron_config"] = neuron_config

        return cls(**filtered_dict)
