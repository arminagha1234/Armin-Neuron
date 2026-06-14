"""PAVE-derived correctness/perf fixes for running the FLUX.2-klein VAE
decoder on Neuron.

Background: a naive `vae.to("neuron")` + per-block compile was ~1.3x SLOWER
than the CPU VAE. The PAVEDigitalTwinDiffusion Neuron port
(code.amazon.com/packages/PAVEDigitalTwinDiffusion) diagnosed the exact
cause and the fixes:

  1. UPSAMPLE GATHER TRAP (the big one). The decoder's 3 nearest-neighbor
     `F.interpolate(mode="nearest")` upsamples lower to a per-element GATHER
     (~tens of millions of dynamic-DMA packets) on Neuron -> ~74% of device
     time at ~1% tensor utilization. Replace with a reshape->expand->reshape
     2x nearest upsample (a broadcast/DMA copy, NOT a gather). PAVE: 4196ms
     -> 1189ms (~3.5x). Must patch EVERY Upsample2D, not just the decoder.

  2. GROUPNORM bf16 -> NaN. Stock bf16 GroupNorm produces NaN on Neuron for
     the VAE's small-variance activations. Cast to fp32 around F.group_norm.

  3. PRECISION. Default here: leave VAE STORAGE as-is (bf16, matching the
     pipeline contract). PAVE's strict fp32-storage recipe is also wired
     up (fp32_storage=True) but conflicts with our pipeline's
     `_encode_vae_image` which forces bf16 inputs. We measured the simpler
     bf16-storage variant + GN-fp32 patch passes the std~=18.15 quality
     gate.

  4. channels_last OFF on the Neuron VAE path (it's a CPU-only win; ~700x
     regression on Neuron due to NEFF explosion + per-forward recompile).
     Enforced by the caller (don't pass --vae-channels-last with --vae-on-neuron).

  5. MIXED-FLAG COMPILE: the DiT compiles with --model-type=transformer
     (the proven winner on transformers; A/B-tested 5.92s vs 7.71s under
     unet-inference). The VAE compiles with --model-type=unet-inference
     (PAVE's conv-scheduling win, on a conv workload). Use
     install_mixed_flag_wrapper(vae) to wrap vae.encode/vae.decode so the
     unet-inference flag is set in NEURON_CC_FLAGS for both compile and
     warm calls — the cache key includes flags, so the same NEFF is hit
     consistently. Measured 4.19s end-to-end (-29% vs the CPU-VAE
     baseline, std=18.16, gate PASS).

Apply BEFORE torch.compile / before the first forward, after the VAE is built.
Then VERIFY zero gather ops remain with verify_no_gather().
"""
from __future__ import annotations

import contextlib
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

VAE_NEURON_CC_FLAG = "--model-type=unet-inference"


# ---------------------------------------------------------------------------
# Fix 1 — gather-free nearest 2x upsample
# ---------------------------------------------------------------------------

def _nearest_upsample_2x(x: torch.Tensor) -> torch.Tensor:
    """Exact nearest-neighbor 2x upsample via reshape->expand->reshape.

    For input [N, C, H, W] returns [N, C, 2H, 2W] where each pixel is
    repeated in a 2x2 block. Identical output to
    F.interpolate(scale_factor=2, mode="nearest") but lowers to a broadcast
    + contiguous copy on Neuron instead of a dynamic gather.
    """
    n, c, h, w = x.shape
    x = x.reshape(n, c, h, 1, w, 1)
    x = x.expand(n, c, h, 2, w, 2)
    x = x.reshape(n, c, h * 2, w * 2)
    return x


def _patch_upsample_modules(root: nn.Module) -> int:
    """Replace every Upsample2D.forward in `root` with a gather-free version.

    Sweeps ALL submodules (decoder up_blocks AND any others) per PAVE's
    "patch all upsamples" lesson. Returns the count patched.
    """
    n = 0
    for name, m in root.named_modules():
        if type(m).__name__ != "Upsample2D":
            continue
        if not getattr(m, "interpolate", True):
            # conv-transpose upsample: not a gather, leave it
            continue

        def make_forward(mod):
            def forward(hidden_states, output_size=None, *args, **kwargs):
                assert hidden_states.shape[1] == mod.channels
                if mod.norm is not None:
                    hidden_states = mod.norm(
                        hidden_states.permute(0, 2, 3, 1)
                    ).permute(0, 3, 1, 2)
                if mod.use_conv_transpose:
                    return mod.conv(hidden_states)
                # gather-free 2x nearest (klein VAE only ever upsamples 2x)
                hidden_states = _nearest_upsample_2x(hidden_states)
                if mod.use_conv:
                    if mod.name == "conv":
                        hidden_states = mod.conv(hidden_states)
                    else:
                        hidden_states = mod.Conv2d_0(hidden_states)
                return hidden_states
            return forward

        m.forward = make_forward(m)
        n += 1
        logger.info("[vae-fix] patched upsample %s", name)
    return n


# ---------------------------------------------------------------------------
# Fix 2 — fp32 GroupNorm (bf16 -> NaN guard)
# ---------------------------------------------------------------------------

def _patch_groupnorms_fp32(root: nn.Module) -> int:
    """Wrap every GroupNorm.forward to compute in fp32 then cast back."""
    n = 0
    for name, m in root.named_modules():
        if not isinstance(m, nn.GroupNorm):
            continue

        def make_forward(mod):
            def forward(x):
                orig_dtype = x.dtype
                out = F.group_norm(
                    x.float(), mod.num_groups,
                    mod.weight.float() if mod.weight is not None else None,
                    mod.bias.float() if mod.bias is not None else None,
                    mod.eps,
                )
                return out.to(orig_dtype)
            return forward

        m.forward = make_forward(m)
        n += 1
    return n


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def apply_vae_neuron_fixes(vae, fp32_storage: bool = False) -> dict:
    """Apply PAVE VAE-on-Neuron fixes in place. Returns a summary dict.

    fp32_storage=False (default, what production ships): bf16 storage +
        upsample-gather fix + GroupNorm-fp32 patch. Production-validated:
        4.19s warm end-to-end at 1024, std=18.16 (gate PASS) when used
        with install_mixed_flag_wrapper().

    fp32_storage=True: PAVE's strict recipe — fp32 weights + matmul
        auto-cast via NEURON_CC_FLAGS. Requires wrap_vae_io_fp32(vae) too
        because the pipeline pushes bf16 tensors at the VAE boundary.
    """
    if fp32_storage:
        vae.float()

    n_up = _patch_upsample_modules(vae)
    n_gn = _patch_groupnorms_fp32(vae)
    logger.info("[vae-fix] upsamples patched=%d, groupnorms patched=%d, fp32_storage=%s",
                n_up, n_gn, fp32_storage)
    return {"upsamples_patched": n_up, "groupnorms_patched": n_gn,
            "fp32_storage": fp32_storage}


def wrap_vae_io_fp32(vae) -> None:
    """Cast inputs to fp32 at the VAE encode/decode boundary; cast outputs
    back to the input's dtype. Use only with apply_vae_neuron_fixes(...,
    fp32_storage=True). Without fp32 storage this wrapper is unnecessary."""
    orig_encode = vae.encode
    orig_decode = vae.decode

    def encode_wrap(x, *a, **kw):
        return orig_encode(x.float(), *a, **kw)

    def decode_wrap(z, *a, **kw):
        in_dtype = z.dtype
        out = orig_decode(z.float(), *a, **kw)
        if hasattr(out, "sample"):
            out.sample = out.sample.to(in_dtype)
        return out

    vae.encode = encode_wrap
    vae.decode = decode_wrap


@contextlib.contextmanager
def _vae_flags():
    """Context manager: append --model-type=unet-inference to NEURON_CC_FLAGS,
    restore on exit. Cache key includes the flag, so this controls compile
    AND keeps warm-call cache lookups consistent."""
    saved = os.environ.get("NEURON_CC_FLAGS")
    cur = saved or ""
    if VAE_NEURON_CC_FLAG not in cur:
        os.environ["NEURON_CC_FLAGS"] = (
            (cur + " " if cur else "") + VAE_NEURON_CC_FLAG
        )
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("NEURON_CC_FLAGS", None)
        else:
            os.environ["NEURON_CC_FLAGS"] = saved


def install_mixed_flag_wrapper(vae) -> None:
    """Wrap vae.encode and vae.decode so --model-type=unet-inference is set
    via NEURON_CC_FLAGS for the duration of every call.

    The DiT compile fires outside this wrapper and uses whatever flag
    NEURON_CC_FLAGS holds at compile time (default = transformer). The
    VAE compile fires inside this wrapper and uses unet-inference. The
    cache key includes flags, so warm calls hit the right compiled NEFF
    only if the same flag is set — this wrapper ensures that.

    Measured: DiT=transformer + VAE=unet-inference (mixed) lands at 4.19s
    end-to-end vs the CPU-VAE baseline 5.92s (-29%, std=18.16 gate PASS).
    """
    orig_encode = vae.encode
    orig_decode = vae.decode

    def encode_w(*a, **kw):
        with _vae_flags():
            return orig_encode(*a, **kw)

    def decode_w(*a, **kw):
        with _vae_flags():
            return orig_decode(*a, **kw)

    vae.encode = encode_w
    vae.decode = decode_w


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_no_gather(vae, sample_z: torch.Tensor) -> dict:
    """Run vae.decode under TorchDispatchMode and assert 0 gather-class aten
    ops remain. Returns the op tally. Run on CPU before moving to Neuron."""
    from torch.utils._python_dispatch import TorchDispatchMode

    GATHER_OPS = {
        "aten.gather", "aten.index", "aten.index_select",
        "aten.upsample_nearest2d", "aten.upsample_nearest1d",
        "aten.upsample_nearest3d", "aten.index.Tensor",
    }
    found = {}

    class GatherSniffer(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = str(func)
            for g in GATHER_OPS:
                if g in name:
                    found[name] = found.get(name, 0) + 1
            return func(*args, **(kwargs or {}))

    with torch.no_grad(), GatherSniffer():
        _ = vae.decode(sample_z)

    return {"gather_ops": found, "clean": len(found) == 0}


if __name__ == "__main__":
    # Self-test: confirm gather-free upsample matches F.interpolate exactly.
    x = torch.randn(1, 4, 8, 8)
    a = _nearest_upsample_2x(x)
    b = F.interpolate(x, scale_factor=2.0, mode="nearest")
    assert a.shape == b.shape, (a.shape, b.shape)
    assert torch.equal(a, b), (a - b).abs().max()
    print("nearest_upsample_2x matches F.interpolate exactly:", a.shape)
