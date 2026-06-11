# SPDX-License-Identifier: Apache-2.0
"""Factory for Qwen3.5 hybrid model selection based on platform/config.

Mirrors `vllm_neuron.model.qwen3_moe.factory`. Only supports BF16 today;
mxfp4 quantization rejected at validation time.
"""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class Qwen3_5ForConditionalGeneration(nn.Module):
    """Factory that validates config and selects the Qwen3.5 implementation."""

    def __init__(
        self,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None,
    ) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None,
    ) -> nn.Module:
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None,
    ) -> nn.Module:
        cls._validate_config(hf_config, neuron_config)

        # Phase 1: only BF16 implementation exists.
        from .model_bf16 import Qwen3_5ForConditionalGeneration as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls,
        hf_config: PretrainedConfig,
        neuron_config: NeuronConfig | None,
    ) -> None:
        quantization = neuron_config.quantization if neuron_config else None

        if quantization == "mxfp4":
            raise ValueError(
                "quantization='mxfp4' is not yet supported for Qwen3.5. "
                "Please use quantization='bf16' or leave unset."
            )
