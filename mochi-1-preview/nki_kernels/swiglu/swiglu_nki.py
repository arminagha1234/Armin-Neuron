"""Fused SwiGLU FFN NKI kernel for the Mochi-1 transformer (Trainium2 / trn2).

Pipeline stage: NKI ISA + tiling + masking (steps 4-6 of CLAUDE.md). Validated
numerically against `swiglu_ref.py` on CPU; on-device validation is PENDING
(no Neuron compiler on this host — see module footer).

Operation (per transformer block; see NOTES.md "fused-SwiGLU sharding trap"):

    up          = x @ W0.T                    # Linear(in -> 2*inner), no bias
    value, gate = up.chunk(2, dim=-1)         # value=up[:,:inner], gate=up[:,inner:]
    hidden      = value * silu(gate)          # silu(g)=g*sigmoid(g)  (diffusers SwiGLU)
    out         = hidden @ W2.T               # Linear(inner -> out), no bias  (down proj)

Full unsharded Mochi `ff`: in=3072, inner=8192, out=3072.
TP-local (this is what a rank sees under ColwiseParallel+RowwiseParallel):
  in=3072, per=inner/world_size (e.g. 2048 at TP=4), out=3072. The kernel
  produces one rank's down-projection PARTIAL; the cross-rank all-reduce is
  OUTSIDE the kernel. `per` and `inner` are the same parameter to the kernel:
  the kernel only ever sees a `[value_slice | gate_slice]`-paired weight, so
  it is agnostic to whether it is running the full or a sharded problem.

── Weight layout (IMPORTANT) ─────────────────────────────────────────────────
The kernel takes weights PRE-TRANSPOSED to contraction-first layout, matching
the `simple_matmul.py` `lhs_T` convention. Weights are static, so transposing
them once at load time is free:

    w0t : [IN, 2*P]   == W0_local.T   (W0_local is [2*P, IN], value-then-gate paired)
    w2t : [P, OUT]    == W2_local.T   (W2_local is [OUT, P])

where P = inner (unsharded) or per (TP-local), and w0t columns [0:P] are the
value half, [P:2*P] the gate half (the permuted pairing from _shard_fused_glu).
See `pretranspose_weights()` for the host-side helper.

── Why this layout: only the activations get transposed on-chip ──────────────
nc_matmul contracts the PARTITION dim: dst[M,N] = sum_K stationary[K,M]*moving[K,N].
Both matmuls contract a dim (IN, then P) that is stored on the FREE axis of the
natural [rows, contraction] tensors, so at least one operand must be transposed
so the contraction lands on the partition dim. By choosing to compute the
transposed intermediate upT[2*P, s] (output index on partition, tokens on free)
the whole chain flows with a SINGLE on-chip transpose:

  MM1  upT[j,s]   : stat = w0t[k_chunk, j_tile]  (direct)  moving = xT[k_chunk, s]  (transposed x)
  act  hiddenT[p,s] = value[p,s] * silu(gate[p,s])         # p on partition, elementwise
  MM2  out[s,o]   : stat = hiddenT[p_chunk, s]   (direct)  moving = w2t[p_chunk, o] (direct)

Output out[s, o] lands with tokens on the partition dim — the natural output
layout, no final transpose.

── Hardware constraints (trn2 NeuronCore) ────────────────────────────────────
  Tensor engine 128x128 systolic: stationary [K<=128, M<=128], moving
  [K<=128, N<=512], result [M, N] in PSUM (fp32 accumulation).
  Partition dim <= 128 everywhere. PSUM free <= 512. Contraction -> partition.
  PSUM cannot DMA to HBM directly: copy PSUM -> SBUF -> HBM.

Shapes / dtypes:
  x      : [S, IN]        bf16   (S up to ~9796 tokens x CFG batch 2)
  w0t    : [IN, 2*P]      bf16
  w2t    : [P, OUT]       bf16
  output : [S, OUT]       bf16   (this rank's local partial)

Author's note on imports: the mandatory NKI 0.3.0 constraint doc uses
`import nki`; the task brief and this repo's NKI_template.py use `neuronxcc.nki`.
We try the 0.3.0 module first and fall back, so the file imports under either
SDK layout present on the Beta-3 DLC.
"""
from __future__ import annotations

try:  # NKI 0.3.0 (SDK 2.29+) — preferred per the mandatory constraint doc.
    import nki
    import nki.isa as nisa
    import nki.language as nl
    _NKI_AVAILABLE = True
except ImportError:
    try:  # Fallback to the neuronxcc namespace (task brief / repo template style).
        import neuronxcc.nki as nki
        import neuronxcc.nki.language as nl
        import neuronxcc.nki.isa as nisa
        _NKI_AVAILABLE = True
    except ImportError:
        # No Neuron SDK on this host (e.g. macOS dev box). Provide a stub so the
        # module still imports — the pure host helpers (pretranspose_weights)
        # remain usable and the test harness can skip device cases cleanly.
        # `@nki.jit` becomes a no-op decorator; calling the kernel without a real
        # SDK raises a clear error rather than compiling.
        _NKI_AVAILABLE = False

        class _NkiStub:
            def jit(self, fn):
                def _guard(*args, **kwargs):
                    raise RuntimeError(
                        "swiglu_ffn_nki requires the Neuron NKI SDK "
                        "(import nki / neuronxcc.nki), which is not installed "
                        "on this host. Run on the Beta-3 DLC."
                    )
                _guard.__wrapped__ = fn
                _guard.__name__ = getattr(fn, "__name__", "swiglu_ffn_nki")
                return _guard

        nki = _NkiStub()
        nisa = None
        nl = None


# ── Self-contained utilities (inlined; nkilib not assumed installed) ─────────
def kernel_assert(condition: bool, error_text: str) -> None:
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def div_ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


# ── Hardware tile constants ──────────────────────────────────────────────────
P_MAX = 128        # partition dim (contraction tile, stationary free tile)
F_STAT_MAX = 128   # stationary free dim max (M)
F_MOV_MAX = 512    # moving free dim max (N)


@nki.jit
def swiglu_ffn_nki(
    x: nl.ndarray,
    w0t: nl.ndarray,
    w2t: nl.ndarray,
) -> nl.ndarray:
    """Fused SwiGLU FFN: out = (value * silu(gate)) @ W2.T, up = x @ W0.T.

    Args:
        x   (nl.ndarray): [S, IN]     @ HBM, input activations (bf16).
        w0t (nl.ndarray): [IN, 2*P]   @ HBM, transposed fused up/gate weight;
                          columns [0:P] value half, [P:2*P] gate half.
        w2t (nl.ndarray): [P, OUT]    @ HBM, transposed down-projection weight.

    Returns:
        nl.ndarray: [S, OUT] @ HBM, this rank's down-projection partial.

    Notes:
        * Contraction dims (IN for MM1, P for MM2) map to the partition dim.
        * Only the input activations are transposed on-chip (xT); weights are
          consumed in their given contraction-first layout.
        * fp32 PSUM accumulation; SiLU computed in fp32; matmul operands bf16.
        * S is tiled by 128 (tokens on partition); 2*P output index tiled by
          128 (MM1 stationary free); OUT tiled by 512 (MM2 moving free); IN and
          P contraction tiled by 128. min()-clamping handles ragged S / OUT.
    """
    kernel_assert(len(x.shape) == 2, f"x must be 2D, got {len(x.shape)}D")
    kernel_assert(len(w0t.shape) == 2, f"w0t must be 2D, got {len(w0t.shape)}D")
    kernel_assert(len(w2t.shape) == 2, f"w2t must be 2D, got {len(w2t.shape)}D")

    S, IN = x.shape
    IN_w, TWO_P = w0t.shape
    P, OUT = w2t.shape

    kernel_assert(IN == IN_w, f"x IN={IN} != w0t IN={IN_w}")
    kernel_assert(TWO_P == 2 * P, f"w0t cols {TWO_P} != 2*P {2 * P} (w2t P={P})")

    output = nl.ndarray((S, OUT), dtype=x.dtype, buffer=nl.shared_hbm)

    n_s_tiles = div_ceil(S, P_MAX)      # token tiles (partition of output)
    n_k_tiles = div_ceil(IN, P_MAX)     # MM1 contraction (IN) tiles
    n_p_tiles = div_ceil(P, P_MAX)      # inner tiles (value/gate index; MM2 contraction)
    n_o_tiles = div_ceil(OUT, F_MOV_MAX)  # MM2 moving-free (OUT) tiles

    for s_idx in nl.affine_range(n_s_tiles):
        s_start = s_idx * P_MAX
        s_size = min(P_MAX, S - s_start)

        # ── Load x tile [s_size, IN] (tokens on partition — natural) ──────────
        x_sb = nl.ndarray((P_MAX, IN), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_sb[0:s_size, 0:IN], src=x[s_start:s_start + s_size, 0:IN])

        # ── Transpose x -> xT chunks: xT_chunks[k] is [in_chunk, s_size] ──────
        # (contraction IN on partition, tokens on free). One transpose per
        # IN-chunk of 128; nc_transpose swaps P<->F on the tensor engine.
        xT_chunks = []
        for k_idx in nl.static_range(n_k_tiles):
            k_start = k_idx * P_MAX
            k_size = min(P_MAX, IN - k_start)
            xt_psum = nl.ndarray((P_MAX, P_MAX), dtype=x.dtype, buffer=nl.psum)
            nisa.nc_transpose(
                dst=xt_psum[0:k_size, 0:s_size],
                data=x_sb[0:s_size, k_start:k_start + k_size],
                engine=nisa.engine.tensor,
            )
            xt_sb = nl.ndarray((P_MAX, P_MAX), dtype=x.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=xt_sb[0:k_size, 0:s_size], src=xt_psum[0:k_size, 0:s_size])
            xT_chunks.append((xt_sb, k_start, k_size))

        # ── MM1 + fused SiLU: build hiddenT tiles [p_size, s_size] ────────────
        # hiddenT[p, s] = value[p, s] * silu(gate[p, s]), p on partition.
        hiddenT_tiles = []
        for p_idx in nl.static_range(n_p_tiles):
            p_start = p_idx * P_MAX
            p_size = min(P_MAX, P - p_start)

            # value half: w0t columns [p_start : p_start+p_size]
            value_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)
            # gate  half: w0t columns [P + p_start : P + p_start + p_size]
            gate_psum = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.psum)

            for k_idx in nl.affine_range(n_k_tiles):
                xt_sb, k_start, k_size = xT_chunks[k_idx]

                w0v_sb = nl.ndarray((P_MAX, P_MAX), dtype=w0t.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=w0v_sb[0:k_size, 0:p_size],
                    src=w0t[k_start:k_start + k_size, p_start:p_start + p_size],
                )
                nisa.nc_matmul(
                    dst=value_psum[0:p_size, 0:s_size],
                    stationary=w0v_sb[0:k_size, 0:p_size],
                    moving=xt_sb[0:k_size, 0:s_size],
                    accumulate=(k_idx > 0),
                )

                g_start = P + p_start
                w0g_sb = nl.ndarray((P_MAX, P_MAX), dtype=w0t.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=w0g_sb[0:k_size, 0:p_size],
                    src=w0t[k_start:k_start + k_size, g_start:g_start + p_size],
                )
                nisa.nc_matmul(
                    dst=gate_psum[0:p_size, 0:s_size],
                    stationary=w0g_sb[0:k_size, 0:p_size],
                    moving=xt_sb[0:k_size, 0:s_size],
                    accumulate=(k_idx > 0),
                )

            # SiLU(gate)*value in fp32, then cast to bf16 for MM2.
            value_sb = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            gate_sb = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=value_sb[0:p_size, 0:s_size], src=value_psum[0:p_size, 0:s_size])
            nisa.tensor_copy(dst=gate_sb[0:p_size, 0:s_size], src=gate_psum[0:p_size, 0:s_size])

            # sig = sigmoid(gate)
            sig_sb = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=sig_sb[0:p_size, 0:s_size], data=gate_sb[0:p_size, 0:s_size], op=nl.sigmoid)
            # silu = gate * sig
            silu_sb = nl.ndarray((P_MAX, P_MAX), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=silu_sb[0:p_size, 0:s_size],
                data1=gate_sb[0:p_size, 0:s_size],
                data2=sig_sb[0:p_size, 0:s_size],
                op=nl.multiply,
            )
            # hidden = value * silu, cast to matmul dtype
            hiddenT_sb = nl.ndarray((P_MAX, P_MAX), dtype=w2t.dtype, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=hiddenT_sb[0:p_size, 0:s_size],
                data1=value_sb[0:p_size, 0:s_size],
                data2=silu_sb[0:p_size, 0:s_size],
                op=nl.multiply,
            )
            hiddenT_tiles.append((hiddenT_sb, p_start, p_size))

        # ── MM2: out[s, o] = sum_p hiddenT[p, s] * w2t[p, o] ──────────────────
        for o_idx in nl.affine_range(n_o_tiles):
            o_start = o_idx * F_MOV_MAX
            o_size = min(F_MOV_MAX, OUT - o_start)

            out_psum = nl.ndarray((P_MAX, F_MOV_MAX), dtype=nl.float32, buffer=nl.psum)
            for p_idx in nl.affine_range(n_p_tiles):
                hiddenT_sb, p_start, p_size = hiddenT_tiles[p_idx]
                w2_sb = nl.ndarray((P_MAX, F_MOV_MAX), dtype=w2t.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=w2_sb[0:p_size, 0:o_size],
                    src=w2t[p_start:p_start + p_size, o_start:o_start + o_size],
                )
                nisa.nc_matmul(
                    dst=out_psum[0:s_size, 0:o_size],
                    stationary=hiddenT_sb[0:p_size, 0:s_size],
                    moving=w2_sb[0:p_size, 0:o_size],
                    accumulate=(p_idx > 0),
                )

            out_sb = nl.ndarray((P_MAX, F_MOV_MAX), dtype=output.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out_sb[0:s_size, 0:o_size], src=out_psum[0:s_size, 0:o_size])
            nisa.dma_copy(
                dst=output[s_start:s_start + s_size, o_start:o_start + o_size],
                src=out_sb[0:s_size, 0:o_size],
            )

    return output


# ── Host-side helpers (numpy/torch, no device) ───────────────────────────────
def pretranspose_weights(w0_local, w2_local):
    """Transpose sharded weights into the kernel's contraction-first layout.

    Accepts numpy arrays or torch tensors.

    Args:
        w0_local: [2*P, IN]  permuted [value_slice | gate_slice] up/gate weight
                  (from swiglu_ref.shard_w0_glu / _shard_fused_glu).
        w2_local: [OUT, P]   row-sharded down weight (from shard_w2_row).

    Returns:
        (w0t, w2t) with shapes [IN, 2*P] and [P, OUT].
    """
    # .T works for both numpy.ndarray and torch.Tensor; make contiguous.
    w0t = w0_local.T
    w2t = w2_local.T
    if hasattr(w0t, "contiguous"):      # torch
        return w0t.contiguous(), w2t.contiguous()
    import numpy as _np                  # numpy
    return _np.ascontiguousarray(w0t), _np.ascontiguousarray(w2t)
