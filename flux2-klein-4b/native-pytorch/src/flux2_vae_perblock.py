"""Phase B retry — per-block VAE decoder compile for FLUX.2-klein-4B.

The earlier Phase B attempt compiled the whole VAE as one graph and hit
NCC_IXTP002 (11.4M instructions > 10M limit). The VAE decoder is the
culprit: 4 UpDecoderBlock2D blocks at increasing spatial resolution
(up to 1024×1024) unroll into a huge conv graph.

Fix: compile each up_block (and mid_block) SEPARATELY with
torch.compile(backend="neuron"). Each sub-block stays well under the
10M-instruction ceiling. The conv_in / conv_norm_out / conv_out stay
eager (tiny). This keeps the VAE on Neuron without the monolithic
compile.

Usage (after pipe.apply_neuron_patches(..., vae_on_neuron=True) and
pipe.vae.to(neuron)):

    from flux2_vae_perblock import compile_vae_decoder_per_block
    compile_vae_decoder_per_block(pipe.vae)
"""
from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def compile_vae_decoder_per_block(vae, backend: str = "neuron") -> int:
    """torch.compile each decoder up_block + mid_block separately.

    Returns the number of submodules compiled.
    """
    decoder = getattr(vae, "decoder", None)
    if decoder is None:
        logger.warning("[vae-perblock] vae.decoder not found")
        return 0

    n = 0

    # mid_block (ResnetBlock + optional attention) — moderate size
    mid = getattr(decoder, "mid_block", None)
    if mid is not None:
        decoder.mid_block = torch.compile(mid, backend=backend, dynamic=False)
        n += 1
        logger.info("[vae-perblock] compiled decoder.mid_block")

    # up_blocks — the conv-heavy resolution-doubling stages
    up_blocks = getattr(decoder, "up_blocks", None)
    if up_blocks is not None:
        for i in range(len(up_blocks)):
            up_blocks[i] = torch.compile(up_blocks[i], backend=backend, dynamic=False)
            n += 1
        logger.info("[vae-perblock] compiled %d decoder.up_blocks", len(up_blocks))

    # Leave conv_in, conv_norm_out, conv_act, conv_out EAGER — they're
    # tiny and compiling them separately would add boundary overhead.
    return n
