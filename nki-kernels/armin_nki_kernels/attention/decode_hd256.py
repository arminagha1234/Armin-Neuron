# Copyright Armin Aghaeb. SPDX-License-Identifier: Apache-2.0
"""Fused single-token decode attention NKI kernel for head_dim=256.

Replaces the Python split-K decode path used in Qwen3.5/3.6 GQA layers
(see `Qwen3_5GQAAttention.forward_decode`). Stock vllm-neuron's
`NF.attention_decode` rejects head_dim>128 because the tensor engine's
single-stationary transpose is sized to 128. This kernel does the same
split-K trick internally with PSUM accumulation, plus fuses the
QK + softmax + AV passes into one NEFF.

Reference (math contract): see `ref_decode_hd256.py`. Any change to
this kernel MUST keep cosine > 0.999 vs that reference on the test
shapes in `tests/test_decode_hd256_parity.py`.

Status: STUB. Implementation pending — invoke the
`neuron-nki-writer-agent` to fill in the kernel body using the plan
in the docstring below.
"""
from __future__ import annotations

# NOTE: do NOT import nki at module level. The vllm-neuron container
# provides `nki`, `nki.isa`, `nki.language`. The CPU sim path also
# provides them via the simulator. But this file is also imported by
# the wrapper module on a CPU-only dev host where the package isn't
# installed — guard the import.
try:
    import nki
    import nki.isa as nisa
    import nki.language as nl
    _NKI_AVAILABLE = True
except ImportError:
    _NKI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Kernel constants
# ---------------------------------------------------------------------------
P_MAX = 128                # tensor engine partition dim
HEAD_DIM = 256             # full head dim (Qwen3.5/3.6)
HEAD_DIM_HALF = 128        # split-K halves
NEG_BIAS = -65504.0        # bf16 min — used on causal-violated slots


# ---------------------------------------------------------------------------
# Kernel signature (the actual @nki.jit definition will go here)
# ---------------------------------------------------------------------------
#
# Input shapes (per (batch, head_q) pair — the wrapper iterates B*Nh):
#   q       : (S_q=1, 256)         bf16  — query for one decode token
#   k_full  : (S_ctx, 256)         bf16  — gathered K cache (already
#                                          GQA-repeated by caller)
#   v_full  : (S_ctx, 256)         bf16  — gathered V cache
#   mask    : (S_q=1, S_ctx)       bool  — causal/padding mask
#   scale   : float                       — pre-softmax scaling
#
# Output:
#   out     : (S_q=1, 256)         bf16
#
# Constraints / ranges:
#   - S_ctx is a multiple of P_MAX (128). Caller pads if needed.
#   - S_q == 1 in the standard decode case. Generalizing to S_q > 1
#     (speculative decode) is a follow-up.
#   - Dtype is bf16 in/out, fp32 internal for QK and softmax.
#
# Internal tiling plan (high level — agent fills in the @nki.jit body):
#   1. Load Q (1, 256) into SBUF: split into Q_lo (1, 128) and Q_hi (1, 128).
#   2. For each S_ctx chunk of 128 tokens:
#        a. Load K_chunk_lo (128, 128) and K_chunk_hi (128, 128).
#        b. PSUM accumulator s_chunk (1, 128) ← Q_lo @ K_chunk_lo.T
#                                              + Q_hi @ K_chunk_hi.T
#           (two nc_matmul calls accumulating into the same PSUM tile)
#        c. Multiply s_chunk by scale.
#        d. Add neg_bias for masked positions (mask is partition-broadcast).
#   3. Concat all chunk scores into a single (1, S_ctx) tile in SBUF.
#   4. Run softmax in fp32: tensor_scalar(exp), tensor_reduce(sum),
#      tensor_scalar(divide). Cast back to bf16.
#   5. For each S_ctx chunk of 128 tokens:
#        a. Load V_chunk_lo (128, 128) and V_chunk_hi (128, 128).
#        b. Accumulate out_lo (1, 128) ← weights_chunk @ V_chunk_lo
#                  out_hi (1, 128) ← weights_chunk @ V_chunk_hi
#           (two nc_matmul calls per chunk into separate PSUM tiles)
#   6. Concat [out_lo | out_hi] into out (1, 256) and DMA back.
#
# The reference NKI patterns to crib from:
#   - vllm_neuron/functional/attention/attention_decode.py (the
#     megakernel for head_dim<=128)
#   - vllm_neuron/functional/attention/attention_segmented_cte.py (the
#     chunked-prefill flash kernel — same Q@K → softmax → @V structure
#     but for full sequence)
#   - aws-neuron neuron-nki-samples (the canonical flash-attention
#     example uses the exact split-K pattern over 128-D halves)


# ---------------------------------------------------------------------------
# STUB — to be filled in by the writer agent
# ---------------------------------------------------------------------------

if _NKI_AVAILABLE:

    @nki.jit
    def decode_hd256_kernel(q, k_full, v_full, mask, scale):
        """STUB. Will be filled in by neuron-nki-writer-agent.

        See the tiling plan above. The wrapper in `decode_hd256_wrap.py`
        is what model code calls; that wrapper iterates over (batch,
        head) and feeds per-(b, h) tiles into this kernel.
        """
        raise NotImplementedError(
            "decode_hd256_kernel: NKI body not yet implemented. "
            "See the tiling plan in the module docstring."
        )

else:

    def decode_hd256_kernel(*args, **kwargs):
        raise RuntimeError(
            "nki is not available in this environment — this kernel "
            "must run inside the vllm-neuron container."
        )
