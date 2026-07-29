"""Fused QKV-projection NKI kernel for the Mochi-1 transformer (Trainium2 / trn2).

Pipeline stage: NKI ISA + tiling + masking (steps 4-6 of CLAUDE.md). Validated
numerically against ``fused_qkv_ref.py`` on CPU; on-device validation is PENDING
(no Neuron compiler on this host — another engineer runs it under serialization).

── Operation ─────────────────────────────────────────────────────────────────
A single no-bias projection matmul that replaces three separate GEMMs sharing
one input (see fused_qkv_ref.py and design doc §1.2):

    out = x @ W_fused.T                # W_fused = concat([Wq, Wk, Wv], axis=0)

For the Mochi attention processor this fuses:
  * visual to_q / to_k / to_v : each Linear(3072 -> 3072) on hidden_states
  * text   add_{q,k,v}_proj   : each Linear(1536 -> 3072) on encoder_hidden_states
into ONE matmul per stream. The [S, 3*OUT] result is split back into q/k/v by
static slices (host side, ``fused_qkv_ref.split_qkv_output``). The win is fewer,
larger GEMMs — better tensor-engine utilisation and fewer dispatches — versus
six small ones. This is a matmul-bound op; there is no elementwise epilogue.

── Weight layout (IMPORTANT — matches swiglu_nki convention) ──────────────────
nc_matmul contracts the PARTITION dim:  dst[M,N] = sum_K stationary[K,M]*moving[K,N].
The contraction dim here is IN, which is stored on the FREE axis of both the
natural activation ``x[S,IN]`` and the natural weight ``W_fused[OUT,IN]``. So at
least one operand must be transposed to land IN on the partition dim.

We transpose ONLY the activations on-chip (xT, cheap: tile reused across all
output tiles) and consume the weight PRE-TRANSPOSED to contraction-first layout:

    wt : [IN, OUT]  == W_fused.T        (static weight; transposed once at load,
                                          free — see pretranspose_fused_weight)

Then the matmul flows with a single on-chip transpose per token-tile:

    MM  out[s,o] : stationary = xT[in_chunk, s]  (transposed x)
                   moving     = wt[in_chunk, o]  (direct)
    -> out[s, o] lands with tokens on the partition dim: the natural [S, OUT]
       output layout, no final transpose.

Transposing the weight on-chip instead would be far costlier: the natural weight
slice ``W_fused[o_tile(512), in_chunk(128)]`` has the 512-wide o axis on the
partition, exceeding the 128 partition cap, so it would need four nc_transpose
ops per (in_chunk, o_tile) competing with the matmul. Pre-transposing the static
weight on the host avoids all of that.

── Batch ─────────────────────────────────────────────────────────────────────
The projection is row-independent, so a batched input [B, S, IN] is handled by
the caller folding B into the row axis (reshape to [B*S, IN]); this kernel is 2D.

── Hardware constraints (trn2 NeuronCore) ────────────────────────────────────
  Tensor engine 128x128 systolic: stationary [K<=128, M<=128], moving
  [K<=128, N<=512], result [M, N] in PSUM (fp32 accumulation).
  Partition dim <= 128 everywhere. PSUM free <= 512. Contraction -> partition.
  PSUM cannot DMA to HBM directly: copy PSUM -> SBUF -> HBM.

Shapes / dtypes:
  x   : [S, IN]        bf16   (S = B*S_seq; S_v up to ~9540 x CFG batch 2)
  wt  : [IN, OUT]      bf16   (== W_fused.T; OUT = 3*proj_width, e.g. 9216 full
                              or 2304 per-rank at TP=4)
  out : [S, OUT]       bf16

Tiling:
  S  tiled by 128  (tokens on the output partition; ragged last tile clamped)
  IN tiled by 128  (contraction; accumulated in PSUM across chunks)
  OUT tiled by 512 (moving free dim = tensor-engine moving max; ragged clamped)
"""
from __future__ import annotations

try:  # NKI 0.5.0 on the target box uses the top-level `nki` namespace.
    import nki
    import nki.isa as nisa
    import nki.language as nl
    _NKI_AVAILABLE = True
except ImportError:
    try:  # Fallback to the neuronxcc namespace for older SDK layouts.
        import neuronxcc.nki as nki
        import neuronxcc.nki.language as nl
        import neuronxcc.nki.isa as nisa
        _NKI_AVAILABLE = True
    except ImportError:
        # No Neuron SDK on this host (e.g. macOS dev box). Provide a stub so the
        # module still imports — the pure host helpers remain usable and the test
        # harness can skip device cases cleanly. `@nki.jit` becomes a no-op that
        # raises a clear error if the kernel is actually called without an SDK.
        _NKI_AVAILABLE = False

        class _NkiStub:
            def jit(self, fn):
                def _guard(*args, **kwargs):
                    raise RuntimeError(
                        "fused_qkv_projection_nki requires the Neuron NKI SDK "
                        "(import nki / neuronxcc.nki), which is not installed on "
                        "this host. Run on the target trn2 DLC."
                    )
                _guard.__wrapped__ = fn
                _guard.__name__ = getattr(fn, "__name__", "fused_qkv_projection_nki")
                return _guard

        nki = _NkiStub()
        nisa = None
        nl = None


# ── Self-contained utilities (nkilib not assumed installed) ──────────────────
def kernel_assert(condition: bool, error_text: str) -> None:
    assert condition, (
        f"[INTERNAL_ERROR] [NCC_INKI016] Kernel validation exception: {error_text}"
    )


def div_ceil(n: int, d: int) -> int:
    return (n + d - 1) // d


# ── Hardware tile constants ──────────────────────────────────────────────────
P_MAX = 128        # partition dim (contraction tile; stationary free tile / tokens)
F_MOV_MAX = 512    # moving free dim max (output width tile)


@nki.jit
def fused_qkv_projection_nki(
    x: nl.ndarray,
    wt: nl.ndarray,
) -> nl.ndarray:
    """General fused no-bias projection: out = x @ W_fused.T.

    Args:
        x  (nl.ndarray): [S, IN]   @ HBM, input activations (bf16). S = B*S_seq.
        wt (nl.ndarray): [IN, OUT] @ HBM, PRE-TRANSPOSED fused weight (== W_fused.T,
                         W_fused = concat([Wq,Wk,Wv], axis=0)). bf16.

    Returns:
        nl.ndarray: [S, OUT] @ HBM (bf16). Split into (q,k,v) by static slices of
        width OUT//3 via fused_qkv_ref.split_qkv_output.

    Notes:
        * Contraction dim IN maps to the partition dim; only x is transposed
          on-chip (xT), the weight is consumed in its contraction-first layout.
        * fp32 PSUM accumulation across IN chunks; matmul operands bf16.
        * S tiled by 128 (tokens on output partition), IN tiled by 128
          (contraction), OUT tiled by 512 (moving free). min()-clamping handles
          ragged S / OUT.
    """
    kernel_assert(len(x.shape) == 2, f"x must be 2D, got {len(x.shape)}D")
    kernel_assert(len(wt.shape) == 2, f"wt must be 2D, got {len(wt.shape)}D")

    S, IN = x.shape
    IN_w, OUT = wt.shape
    kernel_assert(IN == IN_w, f"x IN={IN} != wt IN={IN_w}")

    output = nl.ndarray((S, OUT), dtype=x.dtype, buffer=nl.shared_hbm)

    n_s_tiles = div_ceil(S, P_MAX)        # token tiles (partition of output)
    n_k_tiles = div_ceil(IN, P_MAX)       # contraction (IN) tiles
    n_o_tiles = div_ceil(OUT, F_MOV_MAX)  # output-width (moving free) tiles

    for s_idx in nl.affine_range(n_s_tiles):
        s_start = s_idx * P_MAX
        s_size = min(P_MAX, S - s_start)

        # ── Load x tile [s_size, IN] (tokens on partition — natural) ──────────
        x_sb = nl.ndarray((P_MAX, IN), dtype=x.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=x_sb[0:s_size, 0:IN], src=x[s_start:s_start + s_size, 0:IN])

        # ── Transpose x -> xT chunks: xT_chunks[k] is [in_chunk, s_size] ──────
        # (contraction IN on partition, tokens on free). One transpose per IN
        # chunk of 128; reused across every output tile below.
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

        # ── MM: out[s, o] = sum_in xT[in, s] * wt[in, o] ──────────────────────
        for o_idx in nl.affine_range(n_o_tiles):
            o_start = o_idx * F_MOV_MAX
            o_size = min(F_MOV_MAX, OUT - o_start)

            out_psum = nl.ndarray((P_MAX, F_MOV_MAX), dtype=nl.float32, buffer=nl.psum)
            for k_idx in nl.affine_range(n_k_tiles):
                xt_sb, k_start, k_size = xT_chunks[k_idx]
                w_sb = nl.ndarray((P_MAX, F_MOV_MAX), dtype=wt.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=w_sb[0:k_size, 0:o_size],
                    src=wt[k_start:k_start + k_size, o_start:o_start + o_size],
                )
                nisa.nc_matmul(
                    dst=out_psum[0:s_size, 0:o_size],
                    stationary=xt_sb[0:k_size, 0:s_size],
                    moving=w_sb[0:k_size, 0:o_size],
                    accumulate=(k_idx > 0),
                )

            out_sb = nl.ndarray((P_MAX, F_MOV_MAX), dtype=output.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=out_sb[0:s_size, 0:o_size], src=out_psum[0:s_size, 0:o_size])
            nisa.dma_copy(
                dst=output[s_start:s_start + s_size, o_start:o_start + o_size],
                src=out_sb[0:s_size, 0:o_size],
            )

    return output


# ── Host-side helpers (numpy/torch, no device) ───────────────────────────────
def pretranspose_fused_weight(w_fused):
    """Transpose a fused weight [OUT, IN] into the kernel's [IN, OUT] layout.

    The fused weight is static, so this one-time transpose is free. Accepts a
    numpy array or a torch tensor and returns the same type, contiguous.

    Args:
        w_fused: [OUT, IN]  == concat([Wq, Wk, Wv], axis=0) (see
                 fused_qkv_ref.build_fused_qkv_weight / shard_fused_qkv_weight).

    Returns:
        wt : [IN, OUT]  contraction-first layout consumed by the kernel.
    """
    wt = w_fused.T
    if hasattr(wt, "contiguous"):        # torch
        return wt.contiguous()
    import numpy as _np                   # numpy
    return _np.ascontiguousarray(wt)
