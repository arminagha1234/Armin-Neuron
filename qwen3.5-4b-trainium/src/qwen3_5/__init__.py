# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 (hybrid GatedDeltaNet + GQA) for vllm_neuron.

Path B implementation for serving Qwen3.5-4B / Qwen3.5-9B via
`vllm serve` on Trainium. Mirrors the structure of vllm_neuron's
`qwen3_moe` package so it can be plugged into the model registry
via `register.register()`.

See ../README.md for plan, status, and architecture notes.
"""

from .config import Qwen3_5Config
from .factory import Qwen3_5ForConditionalGeneration

# Register into vllm_neuron's model registry on package import. This makes
# the patch effective in any subprocess that imports qwen3_5 (vLLM workers
# spawn fresh Python processes, each of which re-imports our package via
# PYTHONPATH).
try:
    from .register import register as _register, install_post_plugin_hook
    _register()
    install_post_plugin_hook()
except Exception as _exc:  # pragma: no cover - defensive
    import logging as _log
    _log.getLogger(__name__).warning(
        "qwen3_5: auto-register on import failed: %r", _exc
    )

__all__ = [
    "Qwen3_5Config",
    "Qwen3_5ForConditionalGeneration",
]
