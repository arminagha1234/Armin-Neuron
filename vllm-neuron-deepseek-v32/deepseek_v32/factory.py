# SPDX-License-Identifier: Apache-2.0
"""Factory for DeepSeek V3.2 model selection.

vLLM-Neuron's model registry calls ``from_configs(...)`` to obtain a model
instance — it does not call this class's ``__init__``/``forward`` directly.
However, vLLM's ``ModelRegistry.register_model`` requires the registered
class to be an ``nn.Module`` subclass (or a string), so we keep the
inheritance even though the ``__init__``/``forward`` overrides are unused.
This is DeepSeek V3.2, NOT V3.
"""

import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig


class DeepseekV32ForCausalLM(nn.Module):
    """Factory that validates config and returns the BF16 implementation."""

    @classmethod
    def from_configs(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> nn.Module:
        cls._validate_config(hf_config, neuron_config)

        from .model import DeepseekV32ForCausalLM as Model

        return Model.from_configs(hf_config, neuron_config)

    @classmethod
    def _validate_config(
        cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig | None
    ) -> None:
        if not hasattr(hf_config, "q_lora_rank"):
            raise ValueError(
                "DeepSeek V3.2 config requires q_lora_rank. "
                "Is this the correct model? Expected model_type='deepseek_v32'."
            )
