# SPDX-License-Identifier: Apache-2.0
"""Qwen3.6-27B (hybrid GatedDeltaNet + GQA) for vllm_neuron.

Adapter for serving Qwen/Qwen3.6-27B via `vllm serve` on Trainium.

Architecture is the same family as Qwen3.5-4B (hybrid linear-attn +
full-attn, [3 lin + 1 full] block pattern, partial RoPE, head_dim 256)
but scaled up:
  hidden 5120, layers 64, heads 24/4 (Q/KV), MLP int 17408,
  deltanet v-heads 48, deltanet k-heads 16.
Crucially, tie_word_embeddings=False — lm_head is its own tensor in
the safetensors index (verified at runtime from index.json).

The HF arch class name is the same `Qwen3_5ForConditionalGeneration`,
so we re-use the same registry slot but our factory builds with the
27B-specific config.

See ../README.md for plan, status, and architecture notes.
"""

from .config import Qwen3_6Config
from .factory import Qwen3_6ForConditionalGeneration

# Register into vllm_neuron's model registry on package import. This makes
# the patch effective in any subprocess that imports qwen3_6 (vLLM workers
# spawn fresh Python processes, each of which re-imports our package via
# PYTHONPATH).
try:
    from .register import register as _register, install_post_plugin_hook
    _register()
    install_post_plugin_hook()
except Exception as _exc:  # pragma: no cover - defensive
    import logging as _log
    _log.getLogger(__name__).warning(
        "qwen3_6: auto-register on import failed: %r", _exc
    )

__all__ = [
    "Qwen3_6Config",
    "Qwen3_6ForConditionalGeneration",
]
