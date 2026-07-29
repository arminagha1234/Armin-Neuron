"""CPU reference for the Mochi-1 fused-SwiGLU feed-forward network.

Ground truth (step 1 + 2 of the CLAUDE.md pipeline: reference -> numpy).
Everything the NKI kernel produces is validated against these functions.

The operation (per transformer block, `diffusers.models.activations.SwiGLU`
wrapped by `MochiFeedForward`):

    up   = x @ W0.T                      # Linear(in -> 2*inner), no bias
    value, gate = up.chunk(2, dim=-1)    # value = up[:, :inner], gate = up[:, inner:]
    hidden = value * silu(gate)          # silu(x) = x * sigmoid(x)
    out  = hidden @ W2.T                 # Linear(inner -> in), no bias  (down proj)

Shapes for full unsharded Mochi `ff` (see NOTES.md / mochi_tp_plan.py):

    in_features = 3072, inner = 8192, out_features = 3072
    W0 : [2*inner, in]  = [16384, 3072]   (ff.net.0.proj.weight)
    W2 : [out, inner]   = [3072, 8192]    (ff.net.2.weight)
    x  : [S, in]                          (S up to ~9796 tokens x CFG batch 2)

The fused-SwiGLU sharding trap (NOTES.md "fused-SwiGLU sharding trap",
mochi_meta_loader._shard_fused_glu):

    W0's global output rows [0:inner] are the value half, [inner:2*inner] the
    gate half. `ColwiseParallel` shards W0 contiguously by output row, so the
    naive per-rank chunk(2) pairs the WRONG value/gate slices. The meta loader
    instead hands rank r  concat(W0[value_slice_r], W0[gate_slice_r])  so the
    local chunk(2) recovers the correct pairing. Per rank r of world_size W
    (per = inner // W):

        value_slice_r = W0[r*per : (r+1)*per, :]           (rows in [0, inner))
        gate_slice_r  = W0[inner + r*per : inner + (r+1)*per, :]
        W0_local      = concat([value_slice_r, gate_slice_r], axis=0)  # [2*per, in]

    W2 is row-sharded on its input (inner) axis (RowwiseParallel):

        W2_local = W2[:, r*per : (r+1)*per]                # [out, per]

    Each rank computes a local down-projection PARTIAL; the sum of partials
    across ranks equals the unsharded output. The all-reduce is OUTSIDE the
    NKI kernel — the kernel produces one rank's local partial.
"""
from __future__ import annotations

import numpy as np


# ── Architecture constants (Mochi-1 `ff`, from mochi_tp_plan.py) ─────────────
IN_FEATURES = 3072
INNER = 8192
OUT_FEATURES = 3072


def silu(x: np.ndarray) -> np.ndarray:
    """SiLU / swish activation: x * sigmoid(x). Computed in fp32."""
    x = x.astype(np.float32)
    return x * (1.0 / (1.0 + np.exp(-x)))


def swiglu_ffn_ref(
    x: np.ndarray,
    w0: np.ndarray,
    w2: np.ndarray,
) -> np.ndarray:
    """Full unsharded fused-SwiGLU FFN.

    Args:
        x:  [S, in_features]        input activations
        w0: [2*inner, in_features]  fused up/gate projection (ff.net.0.proj.weight)
        w2: [out_features, inner]   down projection (ff.net.2.weight)

    Returns:
        [S, out_features] output activations.

    Computed in fp32 regardless of input dtype (matches the intent of running
    the activation in fp32 on device); cast the result yourself if you want
    bf16 outputs.
    """
    two_inner, in_f = w0.shape
    out_f, inner = w2.shape
    assert two_inner == 2 * inner, f"W0 rows {two_inner} != 2*inner {2 * inner}"
    assert x.shape[1] == in_f, f"x in-features {x.shape[1]} != W0 {in_f}"
    assert out_f == in_f or True  # out need not equal in in general

    xf = x.astype(np.float32)
    w0f = w0.astype(np.float32)
    w2f = w2.astype(np.float32)

    up = xf @ w0f.T                       # [S, 2*inner]
    value = up[:, :inner]                 # [S, inner]
    gate = up[:, inner:]                  # [S, inner]
    hidden = value * silu(gate)           # [S, inner]
    out = hidden @ w2f.T                  # [S, out_features]
    return out


def shard_w0_glu(w0: np.ndarray, rank: int, world_size: int) -> np.ndarray:
    """Permuted `[value_slice_r | gate_slice_r]` column shard of W0.

    Mirrors mochi_meta_loader._shard_fused_glu. `w0` is [2*inner, in]; returns
    [2*per, in] with per = inner // world_size, laid out value-then-gate so the
    local chunk(2) splits value from gate correctly.
    """
    two_inner, _ = w0.shape
    assert two_inner % 2 == 0, f"fused GLU output dim {two_inner} is not even"
    inner = two_inner // 2
    assert inner % world_size == 0, (
        f"GLU inner dim {inner} not divisible by world_size {world_size}"
    )
    per = inner // world_size
    value = w0[rank * per:(rank + 1) * per, :]
    gate = w0[inner + rank * per:inner + (rank + 1) * per, :]
    return np.concatenate([value, gate], axis=0)


def shard_w2_row(w2: np.ndarray, rank: int, world_size: int) -> np.ndarray:
    """Row shard (RowwiseParallel) of the down projection on its inner axis.

    `w2` is [out, inner]; returns [out, per] with per = inner // world_size,
    taking the columns matching this rank's value/gate slice.
    """
    out_f, inner = w2.shape
    assert inner % world_size == 0
    per = inner // world_size
    return w2[:, rank * per:(rank + 1) * per]


def swiglu_ffn_tp_local(
    x: np.ndarray,
    w0_local: np.ndarray,
    w2_local: np.ndarray,
) -> np.ndarray:
    """TP-local fused-SwiGLU FFN producing this rank's down-projection PARTIAL.

    This is exactly what the NKI kernel computes per rank.

    Args:
        x:        [S, in_features]      input (replicated across ranks)
        w0_local: [2*per, in_features]  permuted [value_slice|gate_slice] shard
        w2_local: [out_features, per]   row shard of the down projection

    Returns:
        [S, out_features] local partial. Sum across ranks == unsharded output.
    """
    two_per, in_f = w0_local.shape
    assert two_per % 2 == 0
    per = two_per // 2
    out_f, per2 = w2_local.shape
    assert per == per2, f"W0_local per {per} != W2_local per {per2}"
    assert x.shape[1] == in_f

    xf = x.astype(np.float32)
    w0f = w0_local.astype(np.float32)
    w2f = w2_local.astype(np.float32)

    up = xf @ w0f.T                       # [S, 2*per]
    value = up[:, :per]                   # local chunk(2) -> value
    gate = up[:, per:]                    # local chunk(2) -> gate
    hidden = value * silu(gate)           # [S, per]
    partial = hidden @ w2f.T              # [S, out_features]
    return partial


def swiglu_ffn_tp_allreduce(
    x: np.ndarray,
    w0: np.ndarray,
    w2: np.ndarray,
    world_size: int,
) -> np.ndarray:
    """Simulate every rank + the cross-rank all-reduce; returns the summed out.

    Reproduces the NOTES.md correctness check: sum of local partials over all
    simulated ranks must equal the unsharded output.
    """
    inner = w0.shape[0] // 2
    assert inner % world_size == 0
    total = None
    for rank in range(world_size):
        w0_local = shard_w0_glu(w0, rank, world_size)
        w2_local = shard_w2_row(w2, rank, world_size)
        partial = swiglu_ffn_tp_local(x, w0_local, w2_local)
        total = partial if total is None else total + partial
    return total


def _demo() -> None:
    """Self-check: numpy reference internal consistency (step 1/2 validation)."""
    rng = np.random.default_rng(0)

    # Small shape first (per CLAUDE.md: smallest shapes first).
    S, in_f, inner, out_f = 8, 16, 32, 16
    x = rng.standard_normal((S, in_f)).astype(np.float32)
    w0 = rng.standard_normal((2 * inner, in_f)).astype(np.float32) * 0.05
    w2 = rng.standard_normal((out_f, inner)).astype(np.float32) * 0.05

    full = swiglu_ffn_ref(x, w0, w2)

    for ws in (1, 2, 4, 8):
        if inner % ws != 0:
            continue
        summed = swiglu_ffn_tp_allreduce(x, w0, w2, ws)
        err = np.max(np.abs(full - summed))
        assert err < 1e-4, f"TP={ws} partial-sum mismatch: max|err|={err:.2e}"
        print(f"  full vs sum-of-partials TP={ws}: max|err|={err:.2e}  OK")

    # Demonstrate the naive contiguous shard is WRONG (reproduces NOTES.md).
    ws = 4
    per = inner // ws
    naive_total = None
    for rank in range(ws):
        w0_naive = w0[rank * 2 * per:(rank + 1) * 2 * per, :]   # contiguous 4096-block
        val = x @ w0_naive[:per].T
        gt = x @ w0_naive[per:].T
        hid = val * silu(gt)
        w2_local = shard_w2_row(w2, rank, ws)
        part = hid @ w2_local.T
        naive_total = part if naive_total is None else naive_total + part
    naive_err = np.max(np.abs(full - naive_total))
    print(f"  naive contiguous shard TP=4: max|err|={naive_err:.2e}  (expected LARGE)")
    assert naive_err > 1e-2, "naive shard should be visibly wrong"

    # Mochi-like full dims.
    S = 128
    x = rng.standard_normal((S, IN_FEATURES)).astype(np.float32) * 0.1
    w0 = rng.standard_normal((2 * INNER, IN_FEATURES)).astype(np.float32) * 0.02
    w2 = rng.standard_normal((OUT_FEATURES, INNER)).astype(np.float32) * 0.02
    full = swiglu_ffn_ref(x, w0, w2)
    summed = swiglu_ffn_tp_allreduce(x, w0, w2, 4)
    err = np.max(np.abs(full - summed))
    print(f"  Mochi dims full vs sum-of-partials TP=4: max|err|={err:.2e}  OK")
    assert err < 1e-2

    print("swiglu_ref self-check PASSED")


if __name__ == "__main__":
    _demo()
