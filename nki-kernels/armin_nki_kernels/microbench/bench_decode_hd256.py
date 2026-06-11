# Copyright Armin Aghaeb. SPDX-License-Identifier: Apache-2.0
"""Microbench for decode_hd256.

Times the eager-PyTorch reference vs the NKI kernel on Qwen3.5-4B's
GQA shapes. Run inside the vllm-neuron container on a Neuron device.

Usage:
    python -m armin_nki_kernels.microbench.bench_decode_hd256

Outputs a table: shape, eager_ms, kernel_ms, speedup.
"""
from __future__ import annotations

import time

import torch

from armin_nki_kernels.attention.decode_hd256_wrap import decode_hd256
from armin_nki_kernels.attention.ref_decode_hd256 import (
    decode_hd256_ref,
    make_test_inputs,
)


SHAPES = [
    # (name, B, Nh, S_q, S_ctx, valid_len)
    ("4B_short_512",   1, 16, 1,  512,  400),
    ("4B_med_2k",      1, 16, 1, 2048, 1500),
    ("4B_long_4k",     1, 16, 1, 4096, 3500),
    ("4B_xlong_20k",   1, 16, 1, 20480, 19500),
    ("27B_typical_2k", 1, 24, 1, 2048, 1500),
]


def _move_to(device, **kw):
    out = {}
    for k, v in kw.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def _bench(fn, kwargs, warmup: int = 3, iters: int = 10) -> float:
    """Return median ms per call."""
    times = []
    # Warmup
    for _ in range(warmup):
        out = fn(**kwargs)
        if hasattr(out, "cpu"):
            out.cpu()  # force materialization
    # Timed
    for _ in range(iters):
        t0 = time.perf_counter()
        out = fn(**kwargs)
        if hasattr(out, "cpu"):
            out.cpu()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def main():
    # Try neuron, fall back to cpu.
    try:
        torch.empty(1, device="neuron")
        device = "neuron"
    except Exception:
        device = "cpu"

    print(f"Running microbench on device: {device}")
    print(f"{'shape':<20} {'eager_ms':>10} {'kernel_ms':>11} {'speedup':>9}")
    print("-" * 55)

    for name, B, Nh, S_q, S_ctx, valid_len in SHAPES:
        inputs = make_test_inputs(B=B, Nh=Nh, S_q=S_q, S_ctx=S_ctx,
                                  valid_len=valid_len)
        inputs = _move_to(device, **inputs)
        try:
            t_eager = _bench(decode_hd256_ref, inputs, warmup=2, iters=5)
        except Exception as e:
            t_eager = float("nan")
            print(f"{name:<20} eager FAILED: {e}")
            continue

        try:
            t_kernel = _bench(decode_hd256, inputs, warmup=2, iters=5)
            speedup = t_eager / t_kernel if t_kernel > 0 else float("nan")
        except NotImplementedError:
            t_kernel = float("nan")
            speedup = float("nan")
        except Exception as e:
            print(f"{name:<20} kernel error: {e}")
            t_kernel = float("nan")
            speedup = float("nan")

        print(f"{name:<20} {t_eager:>10.2f} {t_kernel:>11.2f} {speedup:>8.2f}x")


if __name__ == "__main__":
    main()
