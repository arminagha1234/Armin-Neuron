"""Validate the serve smoke-test output against the standalone reference.

Phase 3.0 acceptance gate (Requirement 8 AC#1):
  cosine(serve_test_output, run_compiled_28step) >= 0.9999
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def img_array(path):
    return np.asarray(Image.open(path).convert("RGB")).astype(np.float32)


def main():
    ref_path = Path("results/run_compiled_28step/output.png")
    got_path = Path("results/serve_test_output.png")
    if not ref_path.exists():
        print(f"[error] reference not found: {ref_path}")
        return 2
    if not got_path.exists():
        print(f"[error] serve output not found: {got_path}")
        return 2

    ref = img_array(ref_path)
    got = img_array(got_path)

    print(f"ref: shape={ref.shape} std={ref.std():.2f}")
    print(f"got: shape={got.shape} std={got.std():.2f}")

    if ref.shape != got.shape:
        print(f"[error] shape mismatch")
        return 1

    ref_f = ref.flatten()
    got_f = got.flatten()
    cos = float(np.dot(ref_f, got_f) / (np.linalg.norm(ref_f) * np.linalg.norm(got_f)))

    ref_uniq = len(set(map(tuple, ref.reshape(-1, 3))))
    got_uniq = len(set(map(tuple, got.reshape(-1, 3))))

    # Adjacent pixel diff — a smoothness / detail proxy
    def adj_diff(a):
        return float(np.abs(np.diff(a.reshape(-1, 3), axis=0)).mean())

    print(f"cosine(serve, run_compiled_28step) = {cos:.6f}")
    print(f"unique colors  ref={ref_uniq}  got={got_uniq}")
    print(f"adj-pixel-diff ref={adj_diff(ref):.2f}  got={adj_diff(got):.2f}")

    threshold = 0.9999
    if cos >= threshold:
        print(f"[PASS] cosine >= {threshold}")
        return 0
    print(f"[FAIL] cosine < {threshold}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
