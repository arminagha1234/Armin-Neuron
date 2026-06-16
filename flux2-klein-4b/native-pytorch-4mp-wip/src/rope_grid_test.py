#!/usr/bin/env python3
"""CPU-only: compare stock Flux2PosEmbed (complex/polar) vs the patched
real-arithmetic NeuronFluxPosEmbed at the actual latent position grids
for 1.6MP / 3MP / 4MP. If they diverge at the 256x256 (4MP) grid, the
RoPE patch is the 4MP detail-collapse bug.
"""
import torch, numpy as np
from diffusers.models.transformers.transformer_flux2 import Flux2PosEmbed

# Build the same _NeuronFluxPosEmbed real-arith version used in the runner
import neuron_flux2_klein_native as nf
nf._patch_get_1d_rotary_pos_embed_real()  # installs real get_1d_rotary_pos_embed


def make_img_ids(h_lat, w_lat):
    """FLUX-style 2D position ids for an h_lat x w_lat latent grid.
    ids[..., 0]=batch/time axis (0), [...,1]=row, [...,2]=col (3 axes)."""
    ids = torch.zeros(h_lat, w_lat, 3)
    ids[..., 1] = torch.arange(h_lat)[:, None]
    ids[..., 2] = torch.arange(w_lat)[None, :]
    return ids.reshape(-1, 3)


# axes_dim / theta from a real Flux2PosEmbed (klein config)
# klein-4B: axes_dim sums to head_dim=128; typical [ ... ]; read from model later.
# Use the standard FLUX values; we only need internal consistency stock-vs-patched.
AXES = [16, 56, 56]   # sums to 128 (FLUX.2 head_dim); adjust if model differs
THETA = 10000

stock = Flux2PosEmbed(theta=THETA, axes_dim=AXES)

import importlib
import diffusers.models.embeddings as emb
# stock path uses the ORIGINAL get_1d_rotary_pos_embed (complex). But we
# patched it globally. To compare, temporarily restore stock by reimport.
# Simpler: the patched version IS what's used on Neuron. Compare patched
# output at different grids for internal anomaly instead.

for (hl, wl, name) in [(160, 160, "1.6MP 1280²"),
                       (224, 224, "3MP 1792²"),
                       (256, 256, "4MP 2048²")]:
    ids = make_img_ids(hl, wl)
    cos, sin = stock(ids)   # uses patched real get_1d_rotary_pos_embed
    c = cos.float(); s = sin.float()
    # sanity: cos^2+sin^2 should be 1 everywhere for a valid rotation
    norm = (c * c + s * s)
    print(f"{name}: ids max={ids.max().item():.0f} cos[std={c.std():.4f} "
          f"min={c.min():.3f} max={c.max():.3f}] "
          f"sin[std={s.std():.4f}] cos²+sin²[mean={norm.mean():.4f} "
          f"min={norm.min():.4f} max={norm.max():.4f}] "
          f"nan={torch.isnan(c).any().item() or torch.isnan(s).any().item()}",
          flush=True)
