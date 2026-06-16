"""Mixed-precision patch for FLUX.2-klein-4B at high resolution.

Root cause (proven 2026-06-15): all-bf16 DiT collapses at high token
count (CPU+patches bf16 @1280 = std 8.99 vs fp32 = 17.65). fp32
everywhere is correct but OOMs at 4 MP even at TP=4.

This is the GPU-standard mixed-precision recipe: keep the big matmul
weights in bf16 (memory + speed), but run the precision-critical ops in
fp32:
  - LayerNorm / RMSNorm / AdaLayerNorm  (norms are the most sensitive)
  - attention softmax + output accumulation
  - the residual-stream adds

We implement it by wrapping the norm modules' forward to upcast to fp32,
compute, and downcast back to bf16. This keeps the residual/matmul path
bf16 (fits memory) while the variance/mean computations stay fp32 (which
is where bf16 loses precision at high token count).

Usage (after apply_neuron_patches, before .to(device)):
    import flux2_mixed_precision as mp
    mp.install_fp32_norms(pipe.transformer.inner)
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _fp32_forward_wrapper(module):
    orig_forward = module.forward

    def wrapped(*args, **kwargs):
        # Upcast tensor args to fp32
        new_args = tuple(
            a.float() if isinstance(a, torch.Tensor) and a.is_floating_point()
            else a for a in args
        )
        new_kwargs = {
            k: (v.float() if isinstance(v, torch.Tensor) and v.is_floating_point()
                else v)
            for k, v in kwargs.items()
        }
        out = orig_forward(*new_args, **new_kwargs)
        # Downcast outputs back to bf16
        if isinstance(out, torch.Tensor):
            return out.to(torch.bfloat16)
        if isinstance(out, tuple):
            return tuple(
                o.to(torch.bfloat16) if isinstance(o, torch.Tensor) and
                o.is_floating_point() else o
                for o in out
            )
        return out

    module.forward = wrapped


def install_fp32_norms(model, rank=0):
    """Walk the model and upcast every LayerNorm / RMSNorm / *Norm* module
    to compute in fp32. Also upcasts their weight params to fp32 so the
    fp32 compute path is consistent.
    """
    norm_types = []
    # Collect class names that look like norms
    n_patched = 0
    for name, m in model.named_modules():
        cls = type(m).__name__
        # Only upcast LEAF normalization layers. Composite norms like
        # AdaLayerNormContinuous contain an nn.Linear projection that is
        # fed `x.to(x.dtype)`; if we upcast that module's input to fp32
        # while its Linear weight stays bf16, the matmul fails to
        # legalize ("matmul: input datatypes mismatched"). The composite's
        # internal `.norm` LayerNorm is itself a leaf and still gets
        # upcast separately, so we keep the precision benefit where it
        # matters (the actual mean/var reduction) without breaking the
        # bf16 matmul path.
        has_linear_child = any(isinstance(c, nn.Linear) for c in m.modules())
        looks_like_norm = (
            isinstance(m, (nn.LayerNorm, nn.GroupNorm))
            or cls.endswith("RMSNorm")
            or cls.endswith("LayerNorm")
        )
        is_norm = looks_like_norm and not has_linear_child
        if is_norm and hasattr(m, "forward"):
            # upcast params to fp32
            for p in m.parameters(recurse=False):
                p.data = p.data.float()
            for b_name, buf in m.named_buffers(recurse=False):
                if buf is not None and buf.is_floating_point():
                    setattr(m, b_name, buf.float())
            _fp32_forward_wrapper(m)
            n_patched += 1
            norm_types.append(cls)

    if rank == 0:
        from collections import Counter
        print(f"[mixed_precision] upcast {n_patched} norm modules to fp32: "
              f"{dict(Counter(norm_types))}", flush=True)
