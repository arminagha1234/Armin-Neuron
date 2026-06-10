# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-27B (hybrid GatedDeltaNet + GQA) configuration.

Architecture summary (from Qwen/Qwen3.6-27B `config.json`):
- 64 layers total: 48 DeltaNet (linear-attn) + 16 GQA (full-attn)
- Layer pattern: [3 DeltaNet + 1 GQA] x 16
- Hidden 5120 / MLP intermediate 17408 (SwiGLU)
- GQA: 24 Q heads, 4 KV heads, head_dim 256
- DeltaNet: 48 value heads, 16 key heads, k_dim=v_dim=128
- Conv1d kernel 4 (state stores last 3 pre-conv QKV tokens)
- RoPE: partial, 25% of head_dim (= 64 dims rotated), theta=10M
- Vocab 248,320, tie_word_embeddings=False (lm_head is its own tensor)
- Apache 2.0 dense coding model from Alibaba (released April 2026).

The `text_config` block of HF's `Qwen3_5ForConditionalGeneration`
config carries the per-layer values; the top-level config carries
multimodal token IDs we mostly ignore for text-only serving.

NOTE: Qwen3.6-27B's HF architecture class name is still
`Qwen3_5ForConditionalGeneration` (same family) — the version bump
shows up only via the model id and the larger dimensions.
"""

import json
from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class Qwen3_6Config:
    """Configuration for Qwen3.6-27B hybrid (DeltaNet + GQA) model."""

    # Core dims (27B defaults; from_configs() overrides from HF config)
    vocab_size: int = 248320
    hidden_size: int = 5120
    num_hidden_layers: int = 64
    intermediate_size: int = 17408
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 262144
    torch_dtype: torch.dtype = torch.bfloat16
    # 27B ships untied embeddings — lm_head is a separate tensor.
    tie_word_embeddings: bool = False

    # Attention (GQA, full-attn layers)
    num_attention_heads: int = 24
    num_key_value_heads: int = 4
    head_dim: int = 256
    rope_theta: float = 10000000.0
    rope_scaling: dict | None = None
    # Partial RoPE: rotate only `partial_rotary_factor * head_dim` dims.
    # Qwen3.5/3.6 family: 25% of head_dim = 64.
    partial_rotary_factor: float = 0.25
    attn_output_gate: bool = True

    # DeltaNet (linear-attn layers)
    deltanet_num_v_heads: int = 48
    deltanet_num_k_heads: int = 16
    deltanet_k_head_dim: int = 128
    deltanet_v_head_dim: int = 128
    deltanet_conv_kernel_size: int = 4

    # Layer pattern — list of "linear_attention" / "full_attention" of length
    # num_hidden_layers. Loaded from HF config.json `text_config.layer_types`.
    layer_types: list[str] = field(default_factory=list)
    # Convenience: how often a full-attention layer appears.
    # Qwen3.5/3.6: 4 (every 4th layer is full-attn).
    full_attention_interval: int = 4

    # Token IDs
    pad_token_id: int = 248044
    bos_token_id: int | None = None
    eos_token_id: int = 248044

    neuron_config: NeuronConfig | None = None

    # Derived
    def __post_init__(self) -> None:
        # Sanity: layer_types length must match num_hidden_layers when provided
        if self.layer_types and len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(
                f"layer_types has {len(self.layer_types)} entries but "
                f"num_hidden_layers={self.num_hidden_layers}"
            )
        # Sanity: only the two expected layer types
        bad = [lt for lt in self.layer_types
               if lt not in ("linear_attention", "full_attention")]
        if bad:
            raise ValueError(
                f"Unexpected layer_types entries: {set(bad)}. "
                "Expected only 'linear_attention' and 'full_attention'."
            )

    @property
    def num_full_attention_layers(self) -> int:
        return sum(1 for lt in self.layer_types if lt == "full_attention")

    @property
    def num_linear_attention_layers(self) -> int:
        return sum(1 for lt in self.layer_types if lt == "linear_attention")

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig,
    ) -> "Qwen3_6Config":
        """Build from a HF config (path, PretrainedConfig, or dict).

        For HF Qwen3_5ForConditionalGeneration (used by both 3.5 and 3.6),
        the per-layer values live in `text_config`. We pull from there if
        present, then fall back to the top-level dict.
        """
        # Normalize input to a dict
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            # Strip None quantization_config (HF round-trip quirk)
            qc = getattr(hf_config, "quantization_config", "MISSING")
            if qc is None:
                delattr(hf_config, "quantization_config")
                config_dict = hf_config.to_dict()
                hf_config.quantization_config = None
            else:
                config_dict = hf_config.to_dict()
        elif isinstance(hf_config, dict):
            config_dict = hf_config
        else:
            raise TypeError(
                f"Unsupported hf_config type: {type(hf_config).__name__}"
            )

        # Prefer text_config block when present (multimodal HF wrapper)
        text_cfg = config_dict.get("text_config", config_dict)
        merged = {**config_dict, **text_cfg}

        # Pull RoPE from rope_parameters if present (Qwen3.5/3.6 use it)
        rope_params = merged.get("rope_parameters") or {}
        if "rope_theta" in rope_params:
            merged["rope_theta"] = rope_params["rope_theta"]

        # Map HF DeltaNet field names to ours.
        # NOTE: in the 27B `text_config`, the keys are
        # `linear_num_value_heads` / `linear_num_key_heads` /
        # `linear_value_head_dim` / `linear_key_head_dim` /
        # `linear_conv_kernel_dim`. The 4B referenced shorter names; we
        # accept either.
        mapping = {
            # 4B-style names
            "num_v_heads": "deltanet_num_v_heads",
            "num_k_heads": "deltanet_num_k_heads",
            "v_head_dim": "deltanet_v_head_dim",
            "k_head_dim": "deltanet_k_head_dim",
            "conv_kernel_size": "deltanet_conv_kernel_size",
            # 27B-style names
            "linear_num_value_heads": "deltanet_num_v_heads",
            "linear_num_key_heads": "deltanet_num_k_heads",
            "linear_value_head_dim": "deltanet_v_head_dim",
            "linear_key_head_dim": "deltanet_k_head_dim",
            "linear_conv_kernel_dim": "deltanet_conv_kernel_size",
        }
        for src, dst in mapping.items():
            if src in merged and dst not in merged:
                merged[dst] = merged[src]

        # Coerce torch_dtype string -> torch.dtype.
        # 27B's text_config uses "dtype" (not "torch_dtype"); accept either.
        dtype_val = merged.get("torch_dtype") or merged.get("dtype")
        if isinstance(dtype_val, str):
            merged["torch_dtype"] = getattr(torch, dtype_val)

        # Filter to known fields
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in merged.items() if k in field_names}
        filtered["neuron_config"] = neuron_config

        return cls(**filtered)
