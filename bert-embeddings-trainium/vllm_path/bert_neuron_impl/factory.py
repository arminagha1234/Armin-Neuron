# SPDX-License-Identifier: Apache-2.0
"""Factory for the BERT encoder, mirroring vllm_neuron/model/llama3/factory.py."""
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig

from .model import BertEncoderModel


class BertModel(nn.Module):
    """vLLM ModelRegistry entry for HF architecture 'BertModel'."""

    def __init__(self, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None) -> None:
        super().__init__()
        self._model = self._select_implementation(hf_config, neuron_config)

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None) -> nn.Module:
        return cls._select_implementation(hf_config, neuron_config)

    @classmethod
    def _select_implementation(cls, hf_config, neuron_config) -> nn.Module:
        return BertEncoderModel.from_configs(hf_config, neuron_config)
