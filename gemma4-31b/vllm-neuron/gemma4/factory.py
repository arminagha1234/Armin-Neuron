# SPDX-License-Identifier: Apache-2.0
"""Factory for Gemma4 model selection based on platform and configuration."""

import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class Gemma4ForConditionalGeneration(nn.Module):
    """Factory that validates config and selects the appropriate Gemma4 implementation.

    Registered as Gemma4ForConditionalGeneration to match the HuggingFace
    architecture name. Only the text decoder is implemented; vision encoder
    weights are skipped during loading.

    Implements the VllmModel protocol stubs so that vLLM's ModelRegistry
    recognizes this as a text generation model during architecture validation.
    The actual model implementation lives in model.py; the factory delegates
    to it via from_configs().
    """

    def __init__(
        self,
        hf_config: PretrainedConfig = None,
        neuron_config: NeuronConfig | None = None,
        *,
        vllm_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if hf_config is not None:
            self._model = self._select_implementation(hf_config, neuron_config)
        else:
            self._model = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        *args,
        **kwargs,
    ):
        return self._model(input_ids, positions, *args, **kwargs)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Protocol stub for VllmModel interface."""
        raise NotImplementedError("Use from_configs() to create the model.")

    def compute_logits(self, hidden_states):
        """Protocol stub for VllmModelForTextGeneration interface."""
        raise NotImplementedError("Use from_configs() to create the model.")

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        cls._validate_config(hf_config, neuron_config)

        from .model import Gemma4ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        pass
