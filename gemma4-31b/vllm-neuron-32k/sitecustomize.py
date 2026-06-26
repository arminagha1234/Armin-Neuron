# SPDX-License-Identifier: Apache-2.0
"""Auto-loaded by Python (when on PYTHONPATH) so `vllm serve` registers Gemma4.

`vllm serve` is its own entrypoint — we can't pass it an import flag. Python
imports `sitecustomize` automatically at startup if it's importable, so dropping
this on PYTHONPATH makes the registration run inside the vllm server process
(and every worker subprocess) without forking vLLM.
"""
import logging

# Teach transformers about the gemma4 config types BEFORE vLLM calls AutoConfig.
try:
    import gemma4_transformers_stub
    gemma4_transformers_stub.install()
    logging.getLogger(__name__).info("[sitecustomize] gemma4 transformers stub installed")
except Exception as exc:
    logging.getLogger(__name__).warning("[sitecustomize] gemma4 stub skipped: %r", exc)

try:
    import gemma4_register
    gemma4_register.install_post_plugin_hook()
    gemma4_register.register()
    logging.getLogger(__name__).info("[sitecustomize] Gemma4 registered")
except Exception as exc:  # never block startup on this
    logging.getLogger(__name__).warning("[sitecustomize] Gemma4 register skipped: %r", exc)
