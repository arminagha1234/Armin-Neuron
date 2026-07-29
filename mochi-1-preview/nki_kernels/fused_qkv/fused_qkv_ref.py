"""CPU reference for the Mochi-1 fused QKV-projection.

Ground truth (step 1 + 2 of the CLAUDE.md pipeline: reference -> numpy). The
NKI kernel in ``fused_qkv_nki.py`` is validated against these functions.

## What is fused (design doc §1.2)

`MochiNeuronAttnProcessor` (src/mochi_neuron_attention.py) runs SIX separate
projection GEMMs per block that share only two inputs:

    visual stream (input = hidden_states, dim 3072):
        to_q       Linear(3072 -> 3072)   no bias
        to_k       Linear(3072 -> 3072)   no bias
        to_v       Linear(3072 -> 3072)   no bias
    text stream   (input = encoder_hidden_states, dim 1536):
        add_q_proj Linear(1536 -> 3072)   no bias
        add_k_proj Linear(1536 -> 3072)   no bias
        add_v_proj Linear(1536 -> 3072)   no bias

Because a Linear with no bias is just ``y = x @ W.T``, three projections that
share one input can be fused into ONE matmul by stacking the weights on the
output axis:

    W_fused = concat([Wq, Wk, Wv], axis=0)          # [3*OUT, IN]
    out     = x @ W_fused.T                           # [S, 3*OUT]
    q, k, v = out[:, :OUT], out[:, OUT:2*OUT], out[:, 2*OUT:]   # static slices

This is exact (not an approximation): ``concat([x@Wq.T, x@Wk.T, x@Wv.T]) ==
x @ concat([Wq,Wk,Wv]).T`` to the bit. The win is fewer, larger GEMMs (better
tensor-engine utilisation, fewer kernel dispatches) versus six small ones.

## Shapes / dtypes (Mochi-1, TP=4, from mochi_tp_plan.py)

    N_HEADS=24, HEAD_DIM=128 -> INNER_DIM (visual dim) = 3072
    TEXT_DIM = 1536
    Each of Q/K/V projects to INNER_DIM = 3072.

    visual: x_v [B, S_v, 3072]      S_v up to 9540, B in {1,2} (CFG)
            W_fused_visual [9216, 3072]  full   / [2304, 3072] per rank at TP=4
    text:   x_t [B, 256, 1536]
            W_fused_text   [9216, 1536]  full   / [2304, 1536] per rank at TP=4

    All bf16 on device; this reference computes in fp32.

## The TP sharding (design doc §1.2)

All six projections are ``ColwiseParallel`` (mochi_tp_plan.py lines 149-152):
each output (inner) axis is column-sharded identically. per = 3072 // W. At
TP=4, per = 768 = 6 heads * 128 head_dim. Rank r sees:

    Wq_local = Wq[r*per:(r+1)*per, :]     (and same slice of Wk, Wv)
    W_fused_local = concat([Wq_local, Wk_local, Wv_local], axis=0)  # [3*per, IN]

so the per-rank fused weight is [2304, IN]. Its output [S, 3*per] splits into
q/k/v-local by the SAME static slice rule (each 768 wide). Unlike the SwiGLU
value/gate trap, there is NO cross-half permutation issue here: q, k and v are
three INDEPENDENT projections, so a rank simply owns rank-r's head-slice of
each. Reassembling the full q across ranks is ``concat_r(q_local_r)`` — an
all-gather that lives OUTSIDE the kernel (in practice the sharded heads feed
straight into the per-rank attention, so no gather is needed at all).
"""
from __future__ import annotations

import numpy as np

# ── Architecture constants (Mochi-1 attn1, from mochi_tp_plan.py) ────────────
INNER_DIM = 3072      # visual stream dim; also each Q/K/V output width
TEXT_DIM = 1536       # encoder_hidden_states dim (text stream input)
N_HEADS = 24
HEAD_DIM = 128
N_QKV = 3             # q, k, v


# ── Weight-fusion helpers (host side) ────────────────────────────────────────
def build_fused_qkv_weight(
    wq: np.ndarray,
    wk: np.ndarray,
    wv: np.ndarray,
) -> np.ndarray:
    """Stack three [OUT, IN] projection weights into one [3*OUT, IN] fused weight.

    Args:
        wq, wk, wv: each [OUT, IN] (e.g. [3072, 3072] visual or [3072, 1536] text).

    Returns:
        [3*OUT, IN] fused weight; row block [0:OUT]=q, [OUT:2*OUT]=k, [2*OUT:]=v.
    """
    assert wq.shape == wk.shape == wv.shape, "q/k/v weights must share shape"
    return np.concatenate([wq, wk, wv], axis=0)


def split_qkv_output(out: np.ndarray, out_width: int) -> tuple[np.ndarray, ...]:
    """Split a fused projection output [S, 3*out_width] back into (q, k, v).

    `out_width` is the per-projection output width: INNER_DIM (=3072) for the
    full/unsharded case, or per (=INNER_DIM // world_size, e.g. 768 at TP=4)
    for a per-rank local output.
    """
    assert out.shape[-1] == N_QKV * out_width, (
        f"output last dim {out.shape[-1]} != 3*out_width {N_QKV * out_width}"
    )
    q = out[..., 0 * out_width:1 * out_width]
    k = out[..., 1 * out_width:2 * out_width]
    v = out[..., 2 * out_width:3 * out_width]
    return q, k, v


def shard_fused_qkv_weight(
    wq: np.ndarray,
    wk: np.ndarray,
    wv: np.ndarray,
    rank: int,
    world_size: int,
) -> np.ndarray:
    """Per-rank fused QKV weight under identical ColwiseParallel sharding.

    Each of the three projections is column-sharded (on its OUT/inner axis) the
    SAME way, then the three rank-local slices are stacked. per = OUT // world.

    Args:
        wq, wk, wv: each [OUT, IN] full projection weights.
        rank, world_size: TP rank / world size.

    Returns:
        [3*per, IN] fused local weight; blocks [0:per]=q_r, [per:2*per]=k_r,
        [2*per:3*per]=v_r.
    """
    out_f, _ = wq.shape
    assert out_f % world_size == 0, f"OUT {out_f} not divisible by world {world_size}"
    per = out_f // world_size
    sl = slice(rank * per, (rank + 1) * per)
    return np.concatenate([wq[sl], wk[sl], wv[sl]], axis=0)


# ── Reference projections ────────────────────────────────────────────────────
def qkv_separate_ref(
    x: np.ndarray,
    wq: np.ndarray,
    wk: np.ndarray,
    wv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The original SIX-GEMM path: three independent no-bias Linears.

    Args:
        x:          [S, IN]
        wq, wk, wv: each [OUT, IN]

    Returns:
        (q, k, v) each [S, OUT], computed in fp32.
    """
    xf = x.astype(np.float32)
    q = xf @ wq.astype(np.float32).T
    k = xf @ wk.astype(np.float32).T
    v = xf @ wv.astype(np.float32).T
    return q, k, v


def fused_projection_ref(x: np.ndarray, w_fused: np.ndarray) -> np.ndarray:
    """General fused no-bias projection: out = x @ w_fused.T (fp32).

    This is exactly what the NKI kernel computes.

    Args:
        x:       [S, IN]
        w_fused: [OUT_TOTAL, IN]   (OUT_TOTAL = 3*OUT for the fused QKV case)

    Returns:
        [S, OUT_TOTAL].
    """
    return x.astype(np.float32) @ w_fused.astype(np.float32).T


def qkv_fused_ref(
    x: np.ndarray,
    wq: np.ndarray,
    wk: np.ndarray,
    wv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fused path: one matmul with stacked weights, then split into (q, k, v)."""
    w_fused = build_fused_qkv_weight(wq, wk, wv)
    out = fused_projection_ref(x, w_fused)
    return split_qkv_output(out, wq.shape[0])


def _demo() -> None:
    """Self-check: prove fused == separate, and TP-local reassembly == full."""
    rng = np.random.default_rng(0)

    for name, IN in (("visual", INNER_DIM), ("text", TEXT_DIM)):
        S = 32
        x = (rng.standard_normal((S, IN)) * 0.1).astype(np.float32)
        wq = (rng.standard_normal((INNER_DIM, IN)) * 0.02).astype(np.float32)
        wk = (rng.standard_normal((INNER_DIM, IN)) * 0.02).astype(np.float32)
        wv = (rng.standard_normal((INNER_DIM, IN)) * 0.02).astype(np.float32)

        qs, ks, vs = qkv_separate_ref(x, wq, wk, wv)
        qf, kf, vf = qkv_fused_ref(x, wq, wk, wv)
        err = max(
            np.max(np.abs(qs - qf)),
            np.max(np.abs(ks - kf)),
            np.max(np.abs(vs - vf)),
        )
        assert err == 0.0, f"{name}: fused != separate, max|err|={err:.2e}"
        print(f"  {name:6s}: fused == separate  max|err|={err:.2e}  OK")

        # TP-local: each rank's fused local weight, outputs reassembled.
        for W in (1, 2, 4, 8):
            per = INNER_DIM // W
            q_parts, k_parts, v_parts = [], [], []
            for r in range(W):
                w_local = shard_fused_qkv_weight(wq, wk, wv, r, W)
                out_local = fused_projection_ref(x, w_local)
                ql, kl, vl = split_qkv_output(out_local, per)
                q_parts.append(ql)
                k_parts.append(kl)
                v_parts.append(vl)
            q_re = np.concatenate(q_parts, axis=1)
            k_re = np.concatenate(k_parts, axis=1)
            v_re = np.concatenate(v_parts, axis=1)
            err = max(
                np.max(np.abs(q_re - qf)),
                np.max(np.abs(k_re - kf)),
                np.max(np.abs(v_re - vf)),
            )
            assert err == 0.0, f"{name} TP={W}: reassembly mismatch {err:.2e}"
            print(f"    TP={W}: reassembled(q|k|v) == fused  max|err|={err:.2e}  OK")

    print("fused_qkv_ref self-check PASSED")


if __name__ == "__main__":
    _demo()
