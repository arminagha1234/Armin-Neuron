"""Local pytest config for the rmsnorm kernel tests.

Ensures this directory is on ``sys.path`` (so ``rmsnorm_ref`` / ``rmsnorm_nki``
import as top-level modules) and pins ``importlib`` import mode. This keeps
collection self-contained and avoids pytest walking up into unrelated parent
packages that may live above the repo (e.g. a stray ``__init__.py`` in a shared
Downloads/home directory).
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
