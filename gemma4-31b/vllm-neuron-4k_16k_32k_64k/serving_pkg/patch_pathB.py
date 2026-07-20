# SPDX-License-Identifier: Apache-2.0
"""Path B — TTFT-optimization patches for the custom Gemma4 model.

Honest scope: most of the win we can VALIDATE on the serve side is config-level
(exact-match buckets, greedy on-device sampling, segmented prefill kept on,
TP=16). The fused NKI kernels in Gemma4/optimized_forward.py are mostly stubs;
`flash_attn_hd256_nki` is a pure-torch split-K reference, not a true NKI kernel.

This module wires the *safe* drop-in pieces and leaves the rest gated behind a
flag so we can A/B them on-device and only keep what actually helps:

  GEMMA4_PB_FLASH=1   -> route SWA (head_dim=256) prefill attn through split-K
  GEMMA4_PB_SOFTCAP=1 -> fused lm-head + tanh softcap

Both default OFF. Turn on individually, re-measure, keep only measured wins.
Everything here is monkey-patched onto the loaded model class, no fork.
"""
import logging
import os

logger = logging.getLogger(__name__)


def apply_pathB_patches() -> None:
    """Monkey-patch the Gemma4 model classes per the GEMMA4_PB_* flags."""
    flags = {
        "flash": os.environ.get("GEMMA4_PB_FLASH", "0") == "1",
        "softcap": os.environ.get("GEMMA4_PB_SOFTCAP", "0") == "1",
    }
    if not any(flags.values()):
        logger.info("[pathB] no GEMMA4_PB_* flags set; serving stock model (config-only Path B)")
        return

    try:
        import gemma4.model as gm
    except Exception as exc:
        logger.warning("[pathB] could not import gemma4.model: %r", exc)
        return

    if flags["flash"]:
        _patch_swa_flash(gm)
    if flags["softcap"]:
        _patch_softcap(gm)


def _patch_swa_flash(gm) -> None:
    """Route SWA (head_dim=256) full-prefill attention through split-K reference.

    Only safe for the non-segmented, non-sliding full-prefill path on SWA layers
    (head_dim=256). Global layers (head_dim=512) and sliding/segmented paths fall
    back to the original forward.
    """
    try:
        from gemma4.flash_attn_hd256_nki import gemma4_flash_attention
    except Exception as exc:
        logger.warning("[pathB] flash kernel import failed: %r", exc)
        return

    orig_prefill = gm.Gemma4Attention.forward_prefill

    def patched_prefill(self, hidden_states, positions, position_embeddings, attn_metadata=None):
        # Only intercept the simple SWA full-prefill case; everything else
        # uses the validated original path.
        if getattr(self, "head_dim", None) != 256 or getattr(self, "sliding_window", None) is not None:
            return orig_prefill(self, hidden_states, positions, position_embeddings, attn_metadata)
        # Fall back to original — the safe full-kernel swap requires the same
        # QKV/norm/RoPE/KV-cache plumbing the original does. We intentionally do
        # NOT duplicate that here; this hook is a placeholder kept OFF by default
        # until a true NKI hd256 kernel is wired and equivalence-checked.
        return orig_prefill(self, hidden_states, positions, position_embeddings, attn_metadata)

    gm.Gemma4Attention.forward_prefill = patched_prefill
    logger.info("[pathB] SWA split-K flash hook installed (currently pass-through; needs equivalence check)")


def _patch_softcap(gm) -> None:
    """No-op placeholder: fused lm-head+softcap needs the model's exact lm_head
    layout. Kept gated until validated. The stock _apply_logit_softcapping is
    already correct, so this is purely a latency experiment."""
    logger.info("[pathB] softcap fusion requested but left as stock (correctness-first)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    apply_pathB_patches()
    print("pathB patch module OK")
