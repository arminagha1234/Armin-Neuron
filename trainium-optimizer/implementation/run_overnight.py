#!/usr/bin/env python3
"""
Single entry point for the autonomous overnight run.

    # prove the whole loop end-to-end on the mock backend (synthetic numbers):
    python run_overnight.py --backend mock

    # real run, once the native backend is implemented + the TP=8 gate passes:
    python run_overnight.py --backend native-pytorch-beta3

Sets up sys.path so `src/` modules import cleanly, then hands off to the
driver in src/overnight.py. See ../CLAUDE.md for the rules of the game.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from overnight import main  # noqa: E402

if __name__ == "__main__":
    main()
