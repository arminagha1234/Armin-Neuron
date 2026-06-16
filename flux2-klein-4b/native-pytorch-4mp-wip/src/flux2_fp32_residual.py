"""fp32 residual-stream patch for FLUX.2-klein-4B at high resolution.

Root cause (proven 2026-06-15): fp32 softmax + fp32 norms are NOT enough
(std 4.74 @1280 TP=2). The bf16 collapse is in the RESIDUAL-STREAM
accumulation: `hidden_states = hidden_states + attn_output` repeated over
25 blocks x 4 steps in bf16. At high token count the residual magnitudes
grow (the stock code even clips at +-65504, the bf16 max) and bf16
rounding collapses the output toward its mean.

Full fp32 fixes it (TP=4 fp32 @1280 = std 16.99) but OOMs above 1.6MP
because fp32 activations don't fit and head-parallel TP can't shard the
sequence-dim activation.

This patch is the memory-safe middle ground: keep ONLY the residual
accumulators (`hidden_states`, `encoder_hidden_states`) in fp32 — one
[B,S,D] tensor each — while every matmul/sub-block still runs in bf16
(weights + activations bf16, so the big activation footprint stays bf16
and FITS). The normed+modulated input to each attn/ff sub-block is
explicitly cast to bf16 so the bf16 Linear weights match.

Combine with flux2_mixed_precision.install_fp32_norms for fp32 norm
reductions. Usage (after apply_neuron_patches, before .to(device)):
    import flux2_fp32_residual as fr
    fr.install_fp32_residual(pipe.transformer.inner)
"""
from __future__ import annotations

import torch
from diffusers.models.transformers.transformer_flux2 import (
    Flux2Modulation,
    Flux2SingleTransformerBlock,
    Flux2TransformerBlock,
)

BF16 = torch.bfloat16
FP32 = torch.float32


def install_fp32_residual(model, rank: int = 0):
    """Monkeypatch both Flux2 block forwards to keep the residual stream
    in fp32 while running sub-blocks in bf16."""

    if getattr(Flux2TransformerBlock, "_fp32_residual_installed", False):
        if rank == 0:
            print("[fp32_residual] already installed", flush=True)
        return

    # ---- Double-stream block ----
    def double_forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb_mod_img,
        temb_mod_txt,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
    ):
        joint_attention_kwargs = joint_attention_kwargs or {}

        (shift_msa, scale_msa, gate_msa), (shift_mlp, scale_mlp, gate_mlp) = \
            Flux2Modulation.split(temb_mod_img, 2)
        (c_shift_msa, c_scale_msa, c_gate_msa), (c_shift_mlp, c_scale_mlp, c_gate_mlp) = \
            Flux2Modulation.split(temb_mod_txt, 2)

        # fp32 residual accumulators
        hs = hidden_states.float()
        ehs = encoder_hidden_states.float()

        # Img stream norm + modulation (norm wrapper computes fp32, returns
        # bf16; multiply/add in fp32 then cast to bf16 for the bf16 attn)
        norm_hs = self.norm1(hs).float()
        norm_hs = (1 + scale_msa) * norm_hs + shift_msa
        norm_ehs = self.norm1_context(ehs).float()
        norm_ehs = (1 + c_scale_msa) * norm_ehs + c_shift_msa

        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hs.to(BF16),
            encoder_hidden_states=norm_ehs.to(BF16),
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        # residual adds in fp32
        hs = hs + (gate_msa * attn_output.float())

        norm_hs = self.norm2(hs).float()
        norm_hs = norm_hs * (1 + scale_mlp) + shift_mlp
        ff_output = self.ff(norm_hs.to(BF16))
        hs = hs + gate_mlp * ff_output.float()

        ehs = ehs + (c_gate_msa * context_attn_output.float())
        norm_ehs = self.norm2_context(ehs).float()
        norm_ehs = norm_ehs * (1 + c_scale_mlp) + c_shift_mlp
        context_ff_output = self.ff_context(norm_ehs.to(BF16))
        ehs = ehs + c_gate_mlp * context_ff_output.float()

        # hand back bf16 to keep the inter-block interface unchanged;
        # the next block re-upcasts. (The fp32 work happened where it
        # matters: inside the residual adds at full precision.)
        return ehs.to(BF16), hs.to(BF16)

    # ---- Single-stream (parallel) block ----
    def single_forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb_mod,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
        split_hidden_states=False,
        text_seq_len=None,
    ):
        if encoder_hidden_states is not None:
            text_seq_len = encoder_hidden_states.shape[1]
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        mod_shift, mod_scale, mod_gate = Flux2Modulation.split(temb_mod, 1)[0]

        hs = hidden_states.float()
        norm_hs = self.norm(hs).float()
        norm_hs = (1 + mod_scale) * norm_hs + mod_shift

        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hs.to(BF16),
            image_rotary_emb=image_rotary_emb,
            **joint_attention_kwargs,
        )

        hs = hs + mod_gate * attn_output.float()
        hs_out = hs.to(BF16)

        if split_hidden_states:
            return hs_out[:, :text_seq_len], hs_out[:, text_seq_len:]
        return hs_out

    Flux2TransformerBlock.forward = double_forward
    Flux2TransformerBlock._fp32_residual_installed = True
    Flux2SingleTransformerBlock.forward = single_forward
    Flux2SingleTransformerBlock._fp32_residual_installed = True

    if rank == 0:
        print("[fp32_residual] patched double + single block forwards "
              "(residual stream in fp32, sub-blocks bf16)", flush=True)
