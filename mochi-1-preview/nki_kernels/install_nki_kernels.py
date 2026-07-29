"""Env-gated installer that swaps validated NKI kernels into the Mochi port.

Design goals
------------
1. **Opt-in, default-off.** Every kernel is behind its own env flag. With no
   flags set, the model runs exactly the validated eager path -- so importing
   or calling this never risks the working baseline.
2. **A/B-able.** Each kernel can be toggled independently, so we can measure
   s/step for {eager, +attn, +swiglu, +rmsnorm, +qkv, +rope, all} and only
   keep the ones that actually win on-device.
3. **Fail-safe.** If a kernel module fails to import or its self-check fails,
   we log and fall back to eager for that op rather than crashing the run.

Flags (all default "0"):
    MOCHI_NKI_ATTN     -- replace neuron_compat._attention_bmm with flash kernel
    MOCHI_NKI_SWIGLU   -- replace the SwiGLU FFN forward
    MOCHI_NKI_RMSNORM  -- replace the tiled RMS-norm core
    MOCHI_NKI_QKV      -- fuse the six QKV projections
    MOCHI_NKI_ROPE     -- replace apply_rotary_emb
    MOCHI_NKI_ALL      -- turn all of the above on

A kernel is only wired here AFTER it has passed on-device validation against
its CPU reference AND shown a speedup over the eager path it replaces. Until
then its flag stays undocumented / off.

This module lives in nki_kernels/ (not src/) so it never touches the validated
port unless explicitly imported and called by the runner.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_KERNELS_ROOT = Path(__file__).resolve().parent
_SRC = _KERNELS_ROOT.parent / "src"

# Make both the port's src/ and each kernel package importable.
for p in (str(_SRC), str(_KERNELS_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _flag(name: str) -> bool:
    if os.environ.get("MOCHI_NKI_ALL", "0") not in ("0", "", "false", "False"):
        return True
    return os.environ.get(name, "0") not in ("0", "", "false", "False")


def _log(rank: int, msg: str) -> None:
    if rank == 0:
        print(f"[nki_install] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Attention: replace neuron_compat._attention_bmm with the flash kernel.
# The flash kernel takes 3D (planes, S, D) tensors + a (planes, Sk) key bias,
# exactly the shapes _attention_bmm already works in, so we wrap it to match
# the (query, key, value, attn_mask, scale, q_chunk) signature.
# ---------------------------------------------------------------------------
def _install_attention(rank: int) -> bool:
    import neuron_compat as nc
    try:
        from attention.flash_attn_nki import flash_attention_kernel
    except Exception as exc:  # import/compile guard
        _log(rank, f"attn: flash kernel import failed ({exc}); staying on bmm")
        return False

    import torch

    _orig_bmm = nc._attention_bmm

    # trn2 partition dim: the flash kernel maps head_dim onto 128 partitions,
    # so it ONLY handles D <= 128. _attention_bmm is called by both the main
    # joint attention (D = head_dim = 128) AND the MochiAttentionPool inside
    # time_embed (which arrives as (planes, S, 512) -- H*D not split). Route
    # only the D<=128 joint attention to flash; everything else stays on bmm.
    _P = 128

    def _flash_bmm(query, key, value, attn_mask, scale, q_chunk):
        # attn_mask arrives as (planes, 1, Sk) or (planes, Sq, Sk) additive bias
        # from _collapse_mask. The flash kernel wants a per-key (planes, Sk)
        # column bias; only the broadcast (Sq==1) form maps cleanly.
        D = query.shape[-1]
        if query.device.type != "neuron" or D > _P:
            # CPU tensors, or the pooler's wide-head-dim attention -> exact bmm.
            return _orig_bmm(query, key, value, attn_mask, scale, q_chunk)
        key_bias = None
        if attn_mask is not None:
            if attn_mask.shape[-2] == 1:
                key_bias = attn_mask[:, 0, :].contiguous()
            else:
                # Per-query mask (not used on the visual stream) -> exact bmm.
                return _orig_bmm(query, key, value, attn_mask, scale, q_chunk)
        else:
            key_bias = torch.zeros(
                query.shape[0], key.shape[1],
                dtype=query.dtype, device=query.device,
            )
        return flash_attention_kernel(query, key, value, key_bias, scale)

    nc._attention_bmm = _flash_bmm
    _log(rank, "attn: flash kernel wired in (replaces _attention_bmm)")
    return True


# ---------------------------------------------------------------------------
# The remaining kernels wire in only once they've passed device validation.
# Each is a stub that reports "not yet wired" so the harness is complete and
# the flags are inert until we flip them on with a validated kernel.
# ---------------------------------------------------------------------------
def _install_swiglu(rank: int) -> bool:
    try:
        from swiglu.swiglu_nki import swiglu_ffn_nki  # noqa: F401
    except Exception as exc:
        _log(rank, f"swiglu: import failed ({exc}); staying eager")
        return False
    # Wiring into diffusers FeedForward is deferred until device A/B shows a win.
    _log(rank, "swiglu: kernel present but not yet wired (pending device A/B)")
    return False


def _install_rmsnorm(rank: int) -> bool:
    try:
        from rmsnorm.rmsnorm_nki import modulated_rmsnorm  # noqa: F401
    except Exception as exc:
        _log(rank, f"rmsnorm: import failed ({exc}); staying on tiled norm")
        return False
    _log(rank, "rmsnorm: kernel present but not yet wired (pending device A/B)")
    return False


def _install_qkv(rank: int) -> bool:
    try:
        from fused_qkv.fused_qkv_nki import fused_projection_nki  # noqa: F401
    except Exception as exc:
        _log(rank, f"qkv: import failed ({exc}); staying on separate projections")
        return False
    _log(rank, "qkv: kernel present but not yet wired (pending device A/B)")
    return False


def _install_rope(rank: int) -> bool:
    try:
        from rope.rope_nki import rope_apply_nki  # noqa: F401
    except Exception as exc:
        _log(rank, f"rope: import failed ({exc}); staying eager")
        return False
    _log(rank, "rope: kernel present but not yet wired (pending device A/B)")
    return False


def install_nki_kernels(model=None, rank: int = 0) -> dict:
    """Install every NKI kernel whose env flag is set. Returns a status dict.

    Call AFTER the model is built and TP fixes are applied, BEFORE the first
    forward. Safe to call with no flags (installs nothing).
    """
    status: dict[str, bool] = {}
    if _flag("MOCHI_NKI_ATTN"):
        status["attn"] = _install_attention(rank)
    if _flag("MOCHI_NKI_SWIGLU"):
        status["swiglu"] = _install_swiglu(rank)
    if _flag("MOCHI_NKI_RMSNORM"):
        status["rmsnorm"] = _install_rmsnorm(rank)
    if _flag("MOCHI_NKI_QKV"):
        status["qkv"] = _install_qkv(rank)
    if _flag("MOCHI_NKI_ROPE"):
        status["rope"] = _install_rope(rank)

    if rank == 0:
        if status:
            _log(rank, f"kernel install status: {status}")
        else:
            _log(rank, "no NKI kernels requested (all flags off) -- eager path")
    return status


if __name__ == "__main__":
    # Smoke: import and report which kernel modules are importable here.
    logging.basicConfig(level=logging.INFO)
    print("NKI kernel modules importable from this host:")
    for name, mod in [
        ("attention", "attention.flash_attn_nki"),
        ("swiglu", "swiglu.swiglu_nki"),
        ("rmsnorm", "rmsnorm.rmsnorm_nki"),
        ("fused_qkv", "fused_qkv.fused_qkv_nki"),
        ("rope", "rope.rope_nki"),
    ]:
        try:
            __import__(mod)
            print(f"  [ok]   {name}")
        except Exception as exc:
            print(f"  [skip] {name}: {type(exc).__name__}: {str(exc)[:80]}")
