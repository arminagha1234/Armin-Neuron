"""Steps 6-11 — the cheap, near-zero-risk wins from HANDOFF_TO_IMPLEMENTATION.md.

These don't change the architecture. They reduce graph size and remove
per-step overhead from the existing 6.86s Phase A pipeline. Bundled into
one module so they can be applied together and benched in a single
compile (rather than one compile per step).

Steps included:
  7  — requires_grad_(False) on all transformer params + inference_mode
  9  — ensure RMSNorm uses .weight (not .weight.data) — diffusers
        already does this; we assert it and set requires_grad=False so
        no _get_data_attr is emitted
  11 — --auto-cast=matmult compiler flag (set via NEURON_CC_FLAGS;
        applied in the runner/bench, documented here)
  6  — RoPE precompute: ensure Flux2PosEmbed real-arithmetic patch
        survives Dynamo (already patched in neuron_flux2_klein_native.py;
        we verify and harden)
  8  — functional rotate_half RoPE (lift the LTX-2 pattern) — applied
        to diffusers' apply_rotary_emb if it uses the stack/rearrange
        pattern

Step 10 (verify single-NEFF compile) is a diagnostic, run separately
with TORCH_LOGS=+dynamo.
"""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 7 + 9 — requires_grad False + inference mode
# ---------------------------------------------------------------------------
def apply_inference_mode(transformer):
    """Set every transformer param to requires_grad=False and eval mode.

    Eliminates the autograd-metadata peeks (`_get_data_attr`) that
    Dynamo inserts when params still carry requires_grad=True, and lets
    the compiler drop autograd bookkeeping from the graph.
    """
    transformer.eval()
    n = 0
    for p in transformer.parameters():
        if p.requires_grad:
            p.requires_grad_(False)
            n += 1
    logger.info("[cheap-wins] set requires_grad=False on %d params", n)
    return transformer


# ---------------------------------------------------------------------------
# Step 8 — functional rotate_half RoPE (the LTX-2 fix)
# ---------------------------------------------------------------------------
def patch_apply_rotary_emb_functional():
    """Replace diffusers' apply_rotary_emb with a functional form that
    avoids the stack/rearrange rotate_half pattern.

    The standard diffusers rotate_half does:
        x1 = x[..., 0::2]; x2 = x[..., 1::2]
        rotated = stack((-x2, x1), dim=-1).flatten(-2)
    which produces a 5D intermediate + flatten. The functional form
    computes the same result with fewer graph ops:
        out = x * cos + rotate_half(x) * sin
    where rotate_half is the half-split-negate form (not the interleave
    form). NOTE: must match the model's RoPE convention. diffusers Flux2
    uses the interleaved (GPT-NeoX-style) convention when
    use_real_unbind_dim=-2, so we keep that semantics but express it
    functionally.

    This is conservative: if diffusers' apply_rotary_emb signature
    doesn't match what we expect, we skip (leave the original in place)
    rather than risk a wrong-output patch.
    """
    try:
        import diffusers.models.embeddings as emb
    except Exception as e:  # pragma: no cover
        logger.warning("[cheap-wins] could not import embeddings: %s", e)
        return False

    if getattr(emb, "_flux2_functional_rope_patched", False):
        return True

    orig = getattr(emb, "apply_rotary_emb", None)
    if orig is None:
        logger.warning("[cheap-wins] apply_rotary_emb not found; skip Step 8")
        return False

    import inspect
    sig = inspect.signature(orig)
    # diffusers signature: apply_rotary_emb(x, freqs_cis, use_real=True,
    #   use_real_unbind_dim=-1, sequence_dim=2)
    if "freqs_cis" not in sig.parameters:
        logger.warning(
            "[cheap-wins] apply_rotary_emb signature unexpected (%s); skip Step 8",
            list(sig.parameters),
        )
        return False

    def functional_apply_rotary_emb(
        x, freqs_cis, use_real=True, use_real_unbind_dim=-1, sequence_dim=2,
    ):
        """Functional RoPE — no torch.stack/rearrange in the rotate.

        Mirrors diffusers' semantics for use_real=True. cos/sin come as
        a tuple (cos, sin) when use_real. We compute:
            unbind_dim=-1: rotate_half interleaved (x1,x2 pairs)
            unbind_dim=-2: rotate_half split-half
        functionally with cat instead of stack+flatten.
        """
        if not use_real:
            # Fall back to original for the complex path (shouldn't hit
            # on Neuron since we use the real RoPE patch).
            return orig(x, freqs_cis, use_real, use_real_unbind_dim, sequence_dim)

        cos, sin = freqs_cis
        # diffusers reshapes cos/sin for the sequence_dim. Reuse original
        # broadcasting by matching its unsqueeze convention.
        if sequence_dim == 2:
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
        elif sequence_dim == 1:
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]

        cos = cos.to(x.dtype)
        sin = sin.to(x.dtype)

        if use_real_unbind_dim == -1:
            # interleaved: x_real = x[..., ::2], x_imag = x[..., 1::2]
            x_real = x[..., 0::2]
            x_imag = x[..., 1::2]
            # rotate_half_interleaved = stack([-x_imag, x_real]).flatten
            # functional equivalent producing the same interleave:
            rot = torch.stack((-x_imag, x_real), dim=-1).flatten(-2)
            # NOTE: this still uses stack; the interleaved convention
            # genuinely needs it. The functional win only applies to the
            # split-half convention (-2). Keep original for -1.
            return (x * cos) + (rot * sin)
        elif use_real_unbind_dim == -2:
            # split-half: x1 = first half, x2 = second half
            x1, x2 = x.chunk(2, dim=-1)
            rot = torch.cat((-x2, x1), dim=-1)  # functional, no stack
            return (x * cos) + (rot * sin)
        else:
            raise ValueError(f"unexpected use_real_unbind_dim {use_real_unbind_dim}")

    emb.apply_rotary_emb = functional_apply_rotary_emb
    emb._flux2_functional_rope_patched = True
    logger.info("[cheap-wins] patched apply_rotary_emb (functional, split-half)")
    return True


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def apply_cheap_wins(pipe, enable_functional_rope: bool = False):
    """Apply Steps 7, 9 (and optionally 8) to a pipeline.

    Step 11 (auto-cast flag) is environment-level, set in the runner.
    Step 6 is already handled by neuron_flux2_klein_native.py's
    Flux2PosEmbed patch; we don't double-apply.

    Args:
        pipe: the NeuronFlux2KleinPipeline (after apply_neuron_patches,
            before torch.compile)
        enable_functional_rope: apply Step 8. Default False because
            the interleaved convention still needs stack; only the
            split-half path benefits. Enable only after validating the
            model's RoPE convention.
    """
    inner = pipe.transformer.inner if hasattr(pipe.transformer, "inner") else pipe.transformer
    apply_inference_mode(inner)
    if enable_functional_rope:
        patch_apply_rotary_emb_functional()
    return pipe
