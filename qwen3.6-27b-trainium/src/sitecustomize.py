# SPDX-License-Identifier: Apache-2.0
"""sitecustomize.py — auto-loaded on Python interpreter startup.

Python imports `sitecustomize` automatically if it's anywhere on
PYTHONPATH (or in site-packages). We use it to ensure the Qwen3.6-27B
registry patch runs in every Python process, including vLLM worker
subprocesses spawned via multiprocessing.

This is the cleanest way to make a registry monkey-patch survive
process forks/spawns without modifying vllm_neuron's installed code.
"""

import os
import logging

logger = logging.getLogger("qwen3_6.sitecustomize")

if os.environ.get("QWEN36_DISABLE_AUTO_REGISTER") != "1":
    try:
        # Triggers __init__.py which runs register() at import-time.
        import qwen3_6  # noqa: F401
        logger.info("qwen3_6 sitecustomize: auto-import + register OK")
    except Exception as exc:
        logger.warning(
            "qwen3_6 sitecustomize: auto-register failed: %r", exc
        )
