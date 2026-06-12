"""Validate Trainium output against the CPU diffusers reference.

Usage:
    # Generate the reference once:
    bash src/run_cpu_ref.sh

    # Run on Trainium:
    bash src/run_compiled_28step.sh

    # Compare:
    python test/test_cosine.py \
        --ref results/output_cpu_ref.png \
        --got results/run_compiled_28step/output.png

Acceptance gate: cosine ≥ 0.95 vs CPU reference (verified at 0.9999
in the canonical run).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True, help="CPU diffusers reference PNG")
    p.add_argument("--got", required=True, help="Trainium output PNG")
    p.add_argument("--threshold", type=float, default=0.95)
    args = p.parse_args()

    ref = np.asarray(Image.open(args.ref).convert("RGB")).astype(np.float32)
    got = np.asarray(Image.open(args.got).convert("RGB")).astype(np.float32)

    if ref.shape != got.shape:
        print(f"[FAIL] shape mismatch: ref {ref.shape} vs got {got.shape}")
        return 1

    rf = ref.flatten()
    gf = got.flatten()
    cos = float(np.dot(rf, gf) / (np.linalg.norm(rf) * np.linalg.norm(gf)))

    print(f"ref: shape={ref.shape}, std={ref.std():.2f}")
    print(f"got: shape={got.shape}, std={got.std():.2f}")
    print(f"cosine(ref, got) = {cos:.6f}")

    if cos >= args.threshold:
        print(f"[PASS] cosine >= {args.threshold}")
        return 0
    print(f"[FAIL] cosine < {args.threshold}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
