"""Static-shape attention processor for Mochi-1 on Neuron.

## The blocker this file removes

`diffusers.models.attention_processor.MochiAttnProcessor2_0` strips prompt
padding with a data-dependent gather:

    mask = attention_mask[idx][None, :]
    valid_prompt_token_indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
    valid_encoder_query = encoder_query[idx:idx+1, :, valid_prompt_token_indices, :]
    ...
    attn_output = F.pad(attn_output, (0, 0, 0, total_length - valid_sequence_length))

`torch.nonzero` has an output shape that depends on tensor *values*, so a
compiled Neuron graph cannot express it. Every downstream shape
(`valid_sequence_length`, the `F.pad` amount) inherits that dynamism.

## The replacement

Keep all `max_sequence_length` (256) text tokens unconditionally and mask
the padded ones with an additive `-10000.0` bias on their **key** columns.
After the softmax those keys carry ~0 weight, so the visual stream sees
exactly what it saw before, with entirely static shapes. The `F.pad`
dance disappears because nothing was ever dropped.

Fidelity detail: upstream zero-fills the *output* rows belonging to
dropped text positions (that is what the `F.pad` does) before `to_add_out`
runs. Those rows only ever feed the next block's masked-out K/V, so they
cannot influence the video, but we reproduce the zero-fill anyway
(`zero_padded_context=True`) to keep the encoder stream bit-comparable
against a CPU reference while debugging.

The rotary helper is copied verbatim from upstream so RoPE numerics match
exactly. Note it is real sin/cos arithmetic -- no `torch.view_as_complex`
-- so unlike Z-Image and FLUX.2-klein, Mochi needs no RoPE rewrite.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from neuron_compat import MASKED_BIAS


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    """Interleaved real-arithmetic RoPE, identical to upstream Mochi.

    Args:
        x: `(B, S, H, D)` queries or keys.
        freqs_cos / freqs_sin: `(S, H, D//2)`.

    Kept byte-for-byte equivalent to
    `MochiAttnProcessor2_0.__call__.apply_rotary_emb` so that any numerical
    divergence on device is attributable to the backend, not to a rewrite.
    """
    x_even = x[..., 0::2].float()
    x_odd = x[..., 1::2].float()
    cos = (x_even * freqs_cos - x_odd * freqs_sin).to(x.dtype)
    sin = (x_even * freqs_sin + x_odd * freqs_cos).to(x.dtype)
    return torch.stack([cos, sin], dim=-1).flatten(-2)


def build_joint_attention_bias(
    encoder_attention_mask: torch.Tensor | None,
    seq_visual: int,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Additive bias over the joint `[visual | text]` key axis.

    Visual keys are always attended (bias 0). Text keys get `MASKED_BIAS`
    wherever the prompt mask says padding.

    Returns `(B, 1, 1, seq_visual + seq_text)`, broadcast over heads and
    query positions -- deliberately not expanded, since a full
    `(B, H, Sq, Sk)` mask would cost as much memory as the scores.
    """
    if encoder_attention_mask is None:
        return None

    keep = encoder_attention_mask
    if keep.ndim > 2:
        keep = keep.reshape(keep.shape[0], -1)

    batch = keep.shape[0]
    text_bias = (1.0 - keep.to(dtype)) * MASKED_BIAS
    visual_bias = torch.zeros(
        (batch, seq_visual), dtype=dtype, device=text_bias.device
    )
    joint = torch.cat([visual_bias, text_bias], dim=1)
    return joint[:, None, None, :]


class MochiNeuronAttnProcessor:
    """Drop-in replacement for `MochiAttnProcessor2_0`, static shapes only.

    Args:
        zero_padded_context: reproduce upstream's zero-fill of encoder
            output rows at padded prompt positions. Costs one multiply;
            keep it on unless you are chasing a perf regression.

    Requires no TP awareness: `attn.heads` is patched to the per-rank head
    count by `mochi_tp_plan.apply_tp_fixes`, and the QK-norm weights are
    shaped `[head_dim]` (128) so they stay replicated and valid on a
    sharded head axis. That is a real simplification over LTX-2, whose
    `rms_norm_across_heads` needed a cross-rank all-reduce inside the norm.
    """

    def __init__(self, zero_padded_context: bool = True) -> None:
        self.zero_padded_context = zero_padded_context

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # ---- visual stream -------------------------------------------------
        query = attn.to_q(hidden_states).unflatten(2, (attn.heads, -1))
        key = attn.to_k(hidden_states).unflatten(2, (attn.heads, -1))
        value = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1))
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # ---- text stream (asymmetric: 1536 -> 3072 projections) -----------
        enc_query = attn.add_q_proj(encoder_hidden_states).unflatten(2, (attn.heads, -1))
        enc_key = attn.add_k_proj(encoder_hidden_states).unflatten(2, (attn.heads, -1))
        enc_value = attn.add_v_proj(encoder_hidden_states).unflatten(2, (attn.heads, -1))
        if attn.norm_added_q is not None:
            enc_query = attn.norm_added_q(enc_query)
        if attn.norm_added_k is not None:
            enc_key = attn.norm_added_k(enc_key)

        # RoPE applies to the visual stream only.
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, *image_rotary_emb)
            key = apply_rotary_emb(key, *image_rotary_emb)

        # (B, S, H, D) -> (B, H, S, D)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        enc_query = enc_query.transpose(1, 2)
        enc_key = enc_key.transpose(1, 2)
        enc_value = enc_value.transpose(1, 2)

        seq_visual = query.shape[2]
        seq_text = enc_query.shape[2]

        # Joint multi-modal self-attention over [visual | text]. Static.
        query = torch.cat([query, enc_query], dim=2)
        key = torch.cat([key, enc_key], dim=2)
        value = torch.cat([value, enc_value], dim=2)

        bias = build_joint_attention_bias(attention_mask, seq_visual, query.dtype)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=bias, dropout_p=0.0, is_causal=False
        )

        # (B, H, S, D) -> (B, S, H*D)
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)

        # Explicit slices rather than split_with_sizes -- lowers to two
        # static slices with no shape inference on the backend.
        visual_out = hidden_states[:, :seq_visual]
        context_out = hidden_states[:, seq_visual:]

        if self.zero_padded_context and attention_mask is not None:
            keep = attention_mask
            if keep.ndim > 2:
                keep = keep.reshape(keep.shape[0], -1)
            context_out = context_out * keep.to(context_out.dtype).unsqueeze(-1)

        visual_out = attn.to_out[0](visual_out)
        visual_out = attn.to_out[1](visual_out)

        # Absent on the final block (context_pre_only=True), matching upstream.
        if hasattr(attn, "to_add_out"):
            context_out = attn.to_add_out(context_out)

        return visual_out, context_out


def install_neuron_attn_processor(
    model,
    zero_padded_context: bool = True,
    verbose: bool = True,
) -> int:
    """Swap every block's processor for the static-shape version.

    Returns the number of processors replaced (expect 48 for Mochi-1).
    """
    replaced = 0
    for block in model.transformer_blocks:
        attn = getattr(block, "attn1", None)
        if attn is None:
            continue
        attn.processor = MochiNeuronAttnProcessor(
            zero_padded_context=zero_padded_context
        )
        replaced += 1

    if verbose:
        print(
            f"[mochi_attn] installed MochiNeuronAttnProcessor on {replaced} "
            f"blocks (removes torch.nonzero; zero_padded_context="
            f"{zero_padded_context})",
            flush=True,
        )
    return replaced
