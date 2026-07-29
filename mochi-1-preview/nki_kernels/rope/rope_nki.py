"""Fused interleaved-RoPE NKI kernel for the Mochi-1 transformer (trn2).

This kernel replaces the eager interleaved RoPE apply in
``src/mochi_neuron_attention.apply_rotary_emb``. Eager RoPE is memory-bound and
strided: it materialises ``x[..., 0::2]`` and ``x[..., 1::2]`` (two strided
gathers), does two multiplies + one sub and two multiplies + one add, then
``stack(...).flatten(-2)`` (a strided scatter) -- five+ separate passes that each
touch HBM. This kernel fuses the deinterleave -> 2 muls -> sub -> 2 muls -> add
-> re-interleave chain so every element of ``x`` is read once and written once,
with all intermediates living in SBUF and the fp32->bf16 cast folded into the
final store.

--------------------------------------------------------------------------------
Operation (matches ``rope_ref.apply_rotary_emb_np`` / the port EXACTLY)
--------------------------------------------------------------------------------
    x_even = x[..., 0::2].float()          # (B, S, H, D/2)
    x_odd  = x[..., 1::2].float()
    cos_out = x_even * freqs_cos - x_odd * freqs_sin
    sin_out = x_even * freqs_sin + x_odd * freqs_cos
    out = stack([cos_out, sin_out], -1).flatten(-2)      # cos->even, sin->odd

No ``view_as_complex`` -- pure sin/cos, byte-for-byte upstream Mochi. Applied to
visual Q and K only.

--------------------------------------------------------------------------------
Shapes / dtypes  (TP=4)
--------------------------------------------------------------------------------
    x          : (B, S, H, D)     bf16   B in {1, 2}, S up to ~9540,
                                         H = 6 local heads, D = head_dim = 128
    freqs_cos  : (S, H, D//2)     fp32 (default) or bf16   D//2 = 64
    freqs_sin  : (S, H, D//2)     fp32 or bf16
    out        : (B, S, H, D)     bf16 (same dtype as x)

All arithmetic runs in fp32 (matches the port's ``.float()``); the result is cast
back to ``x``'s dtype on the final store. The port verified fp32-internal RoPE
crosses the Neuron compile boundary cleanly, so this is numerically identical to
upstream, not an approximation.

--------------------------------------------------------------------------------
The even/odd interleave -- the crux of this kernel
--------------------------------------------------------------------------------
RoPE's deinterleave/re-interleave is along the *last* (head_dim D) axis. We put
the sequence S on the partition dim (<=128 rows/tile) and head_dim D on the free
dim, iterating heads with a static Python loop (H = 6). For each ``(batch, S-tile,
head)`` we load one contiguous ``(ap, D)`` SBUF tile and use a **strided access
pattern** to separate even and odd columns without any data movement in HBM:

    x_tile.ap(pattern=[[D, ap], [2, D//2]], offset=0)   # even cols 0,2,4,...
    x_tile.ap(pattern=[[D, ap], [2, D//2]], offset=1)   # odd  cols 1,3,5,...

``.ap(pattern=[[part_stride, n_part], [free_stride, n_free]], offset=...)``
describes a strided view over the tile's contiguous SBUF storage in *element*
units: partition stride ``D`` (row length), ``ap`` partitions; free stride ``2``
(every other column), ``D//2`` columns; ``offset`` picks the even (0) or odd (1)
lattice. This is the exact idiom the deepseek_v4 ``fp4_gemm_nki`` kernel uses to
interleave lo/hi nibbles, verified on the same top-level ``nki`` 0.5.0 stack.

We gather even/odd into contiguous fp32 ``(ap, D//2)`` tiles (the ``tensor_copy``
also upcasts bf16->fp32), do the four multiplies + add/sub on contiguous tiles,
then **scatter** cos_out back to the even columns and sin_out to the odd columns
of the output tile via the same ``.ap`` pattern -- reproducing
``stack([cos, sin], -1).flatten(-2)`` exactly. Because the tensor is contiguous
in ``(H, D)``, per-head even/odd gathering matches the flattened ``H*D`` even/odd
ordering, and the ``(S, H, D//2)`` freqs line up column-for-column per head.

--------------------------------------------------------------------------------
Hardware constraints / mapping (trn2 NeuronCore)
--------------------------------------------------------------------------------
* Partition dim capped at 128: sequence tiled in blocks of 128 (``_PMAX``); the
  final partial tile is min()-sized (``ap = min(128, S - m)``) so every DMA stays
  in bounds -- the CLAUDE.md boundary-clamp alternative to masking. Tiles are
  addressed with plain slices (nki 0.5.0 has no ``nl.mgrid`` / ``nl.arange``).
* D = 128 (D//2 = 64) fits trivially in one SBUF free dim, so no D-axis tiling.
* Heads (H=6) are host-unrolled with a static Python loop. Each head is ONE
  multi-partition DMA (S rows loaded together); this is NOT the banned
  per-partition-index single-slice DMA pattern -- the packed axis (S) is loaded
  whole per DMA, and the head offset is a compile-time constant, not a runtime
  scalar index, so no operand aliases to a single index. Every HBM element is
  still touched exactly once (one read, one write): a single pass.
* Batches are host-unrolled too, so a batch-shared ``(S, H, D//2)`` freqs row maps
  cleanly without straddling the S tiling.

--------------------------------------------------------------------------------
API / VALIDATION STATUS
--------------------------------------------------------------------------------
Uses the nki 0.5.0 API: top-level ``import nki`` / ``nl.load`` / ``nl.store`` /
``nl.multiply`` / ``nl.subtract`` / ``nl.add`` + ``nisa.tensor_copy`` and the
``.ap(pattern=..., offset=...)`` strided view. NO ``nl.mgrid`` / ``nl.arange`` /
``nisa.select``. The math is CPU-validated against ``rope_ref`` (and the port's
``apply_rotary_emb`` to zero error); on-device validation on the Beta-3 DLC (trn2,
nki 0.5.0) is pending a free Neuron core -- see ``test_rope.py``.
"""
from __future__ import annotations

# nki 0.5.0 on the Beta-3 DLC: the `neuronxcc.nki` entrypoint routes @nki.jit
# through torch_neuronx.pyhlo (absent in the DLC container), so import the plain
# `nki` namespace, which compiles and runs on-device. Matches rmsnorm_nki.py and
# the deepseek fp4_gemm_nki.py convention.
import nki
import nki.language as nl
import nki.isa as nisa

# trn2 hardware partition dimension (SBUF first-dim cap).
_PMAX = 128


@nki.jit
def apply_rotary_emb(x, freqs_cos, freqs_sin):
    """Interleaved real-arithmetic RoPE, fused into one pass over HBM.

    Computes, per ``(batch, seq, head)``::

        cos_out = x_even * freqs_cos - x_odd * freqs_sin
        sin_out = x_even * freqs_sin + x_odd * freqs_cos
        out[..., 0::2] = cos_out ;  out[..., 1::2] = sin_out

    in fp32 and casts back to ``x``'s dtype on store.

    Args:
        x: HBM tensor ``(B, S, H, D)``, bf16 (or any float). D must be even; the
            RoPE split is over D (the last axis).
        freqs_cos: HBM tensor ``(S, H, D//2)`` -- cos table, shared across batch.
        freqs_sin: HBM tensor ``(S, H, D//2)`` -- sin table, shared across batch.

    Returns:
        HBM tensor ``(B, S, H, D)`` with the same dtype as ``x``.
    """
    B, S, H, D = x.shape
    assert D % 2 == 0, "head_dim D must be even for interleaved RoPE"
    Dh = D // 2
    dt = x.dtype

    out = nl.ndarray((B, S, H, D), dtype=dt, buffer=nl.shared_hbm)

    # Strided access pattern selecting every other free-dim column of a
    # contiguous (ap, D) SBUF tile. offset=0 -> even columns, offset=1 -> odd.
    # (part_stride = row length D; free_stride = 2; D//2 columns.) Rebuilt per
    # tile because the partition count `ap` can shrink on the last tile.
    for b in range(B):
        for m in range(0, S, _PMAX):
            ap = min(_PMAX, S - m)  # actual partitions in this (possibly partial) tile
            # nki forbids inner-function defs; build the even/odd strided
            # access pattern inline (part_stride D, free_stride 2, Dh cols).
            eo = dict(pattern=[[D, ap], [2, Dh]])

            for h in range(H):
                # ---- load one contiguous head tile (ap, D), input dtype -------
                # nki 0.5.0 has no nl.mgrid; address with plain slices. Scalar
                # head index h is a compile-time constant.
                x_tile = nl.load(x[b, m : m + ap, h, :])          # (ap, D)

                # ---- deinterleave even/odd -> contiguous fp32 (ap, Dh) --------
                # tensor_copy from the strided .ap view also upcasts to fp32.
                x_even = nl.ndarray((ap, Dh), dtype=nl.float32, buffer=nl.sbuf)
                x_odd = nl.ndarray((ap, Dh), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=x_even, src=x_tile.ap(offset=0, **eo))
                nisa.tensor_copy(dst=x_odd, src=x_tile.ap(offset=1, **eo))

                # ---- load freqs (ap, Dh) as fp32 (batch-shared tables) --------
                fc = nl.ndarray((ap, Dh), dtype=nl.float32, buffer=nl.sbuf)
                fs = nl.ndarray((ap, Dh), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=fc, src=nl.load(freqs_cos[m : m + ap, h, :]))
                nisa.tensor_copy(dst=fs, src=nl.load(freqs_sin[m : m + ap, h, :]))

                # ---- RoPE arithmetic, all fp32 (ap, Dh) -----------------------
                # cos_out = x_even*fc - x_odd*fs
                cos_out = nl.subtract(nl.multiply(x_even, fc), nl.multiply(x_odd, fs))
                # sin_out = x_even*fs + x_odd*fc
                sin_out = nl.add(nl.multiply(x_even, fs), nl.multiply(x_odd, fc))

                # ---- re-interleave: cos->even cols, sin->odd cols -------------
                # tensor_copy into the strided .ap view casts fp32 -> input dtype
                # (exactly stack([cos, sin], -1).flatten(-2)).
                out_tile = nl.ndarray((ap, D), dtype=dt, buffer=nl.sbuf)
                nisa.tensor_copy(dst=out_tile.ap(offset=0, **eo), src=cos_out)
                nisa.tensor_copy(dst=out_tile.ap(offset=1, **eo), src=sin_out)

                nl.store(out[b, m : m + ap, h, :], out_tile)

    return out
