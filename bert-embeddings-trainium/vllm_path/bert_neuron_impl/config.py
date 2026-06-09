# SPDX-License-Identifier: Apache-2.0
"""
BERT encoder config for the vllm-neuron backend.

Mirrors the structure of vllm_neuron/model/llama3/config.py but for a
bidirectional BERT encoder (no rotary, no KV cache, learned position
embeddings, post-LN or pre-LN per HF BERT).
"""
import json
from dataclasses import dataclass

import torch
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


@dataclass
class BertNeuronConfig:
    vocab_size: int = 30522
    hidden_size: int = 384
    intermediate_size: int = 1536
    num_hidden_layers: int = 6
    num_attention_heads: int = 12
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    hidden_act: str = "gelu"
    torch_dtype: torch.dtype = torch.bfloat16

    neuron_config: NeuronConfig | None = None

    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.head_dim = self.hidden_size // self.num_attention_heads

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig = None):
        if isinstance(hf_config, (str, bytes)):
            with open(hf_config) as f:
                config_dict = json.load(f)
        elif isinstance(hf_config, PretrainedConfig):
            config_dict = hf_config.to_dict()
            if getattr(hf_config, "torch_dtype", None) is not None:
                config_dict["torch_dtype"] = hf_config.torch_dtype
        else:
            config_dict = hf_config

        field_names = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in config_dict.items() if k in field_names}

        if isinstance(filtered.get("torch_dtype"), str):
            filtered["torch_dtype"] = getattr(torch, filtered["torch_dtype"])

        filtered["neuron_config"] = neuron_config
        return cls(**filtered)
