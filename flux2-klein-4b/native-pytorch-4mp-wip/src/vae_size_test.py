#!/usr/bin/env python3
"""CPU-only: decode a normal latent through the FLUX.2 VAE at several
latent grid sizes to see if the VAE decode collapses at the 4MP size
(256x256 latent -> 2048x2048 image). No Neuron, no compile.

If output std stays healthy at 256 -> VAE is fine, the 4MP bug is the DiT
latent. If output std collapses at 256 -> the VAE (likely a tiling /
groupnorm / numerical issue) is the 4MP bug.
"""
import os, torch, numpy as np

MODEL = "black-forest-labs/FLUX.2-klein-4B"
tok = os.environ.get("HF_TOKEN")

from diffusers import DiffusionPipeline
print("loading VAE only...", flush=True)
# Load the full pipe on CPU fp32 then grab vae
pipe = DiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.float32, token=tok)
vae = pipe.vae.eval()
sf = getattr(vae.config, "scaling_factor", 1.0) or 1.0
shift = getattr(vae.config, "shift_factor", 0.0) or 0.0
ch = vae.config.latent_channels
print(f"vae latent_channels={ch} scaling_factor={sf} shift_factor={shift}", flush=True)

torch.manual_seed(0)
for grid in [80, 112, 128, 224, 256]:
    z = torch.randn(1, ch, grid, grid, dtype=torch.float32)
    # undo scaling like the pipeline does before decode
    zz = z / sf + shift
    with torch.no_grad():
        img = vae.decode(zz).sample
    a = img.float()
    res = grid * 16
    print(f"grid {grid}x{grid} (~{res}x{res} img): "
          f"decode std={a.std().item():.4f} mean={a.mean().item():.4f} "
          f"min={a.min().item():.2f} max={a.max().item():.2f}", flush=True)
