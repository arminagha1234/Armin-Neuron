# SPDX-License-Identifier: Apache-2.0
"""Auto-loaded by Python (when this dir is on PYTHONPATH) so that ``vllm serve``
recognizes Gemma4 AND has the patched segmented-attention kernel long context
needs -- with no vLLM fork and no manual steps.

``vllm serve`` is its own entrypoint, so we can't pass it an import flag. Python
imports ``sitecustomize`` automatically at interpreter startup if it is
importable, so putting this directory on PYTHONPATH runs the steps below inside
the server process (and every worker subprocess).

Order matters:
  1. Deploy the patched ``vllm_neuron`` ``attention_segmented_cte.py`` (edit A +
     SWA windowed gather) over the installed copy. Runs FIRST -- before vLLM
     imports vllm_neuron -- so the patched module is the one that gets loaded.
  2. Install the transformers gemma4 config stub (so AutoConfig recognizes
     model_type ``gemma4`` before vLLM calls it).
  3. Register ``Gemma4ForConditionalGeneration`` into vLLM's model registry.

Every step is wrapped so a failure logs a warning but never blocks startup.
"""
import logging

_log = logging.getLogger(__name__)

# 1. Deploy patched segmented CTE BEFORE anything imports vllm_neuron.
try:
    import deploy_segmented_cte

    deploy_segmented_cte.deploy()
except Exception as exc:  # never block startup
    _log.warning("[sitecustomize] segmented CTE deploy skipped: %r", exc)

# 2. Teach transformers about the gemma4 config types before vLLM calls AutoConfig.
try:
    import gemma4_transformers_stub

    gemma4_transformers_stub.install()
    _log.info("[sitecustomize] gemma4 transformers stub installed")
except Exception as exc:
    _log.warning("[sitecustomize] gemma4 stub skipped: %r", exc)

# 3. Register the model (+ post-plugin re-register hook, since the vllm_neuron
#    plugin resets the registry after it loads).
try:
    import gemma4_register

    gemma4_register.install_post_plugin_hook()
    gemma4_register.register()
    _log.info("[sitecustomize] Gemma4 registered")
except Exception as exc:
    _log.warning("[sitecustomize] Gemma4 register skipped: %r", exc)
