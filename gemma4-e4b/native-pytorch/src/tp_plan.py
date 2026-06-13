"""TP plan for Google Gemma 4 E4B-it on Trainium2 / Inferentia2.

E4B is a multimodal "Effective 4B" model with ~7.94B raw parameters,
~16 GB BF16 footprint, 42 decoder layers. Notable architectural features
that the plan must respect:

  * **KV-sharing across layers.** 24 of 42 layers are "owners" with full
    q_proj / k_proj / v_proj / o_proj. The remaining 18 layers are
    "shared" and only have q_proj / o_proj — they reuse a prior owner
    layer's KV cache. The plan can only target k_proj / v_proj on
    layers that actually have them.
  * **2 KV heads.** ``num_key_value_heads = 2`` caps practical
    ColwiseParallel sharding at TP=2. TP > 2 with naïve KV sharding
    triggers ``RuntimeError: shape '[B, S, -1, 256]' is invalid for
    input of size <local>`` because 2 KV heads do not divide N>2 ranks.
  * **Per-Layer Embeddings (PLE).** Every layer has
    ``per_layer_input_gate`` (Linear 2560 -> 256) and
    ``per_layer_projection`` (Linear 256 -> 2560) used for an
    element-wise multiply against ``hidden_states``. The HF forward
    expects full-rank tensors here; sharding either projection breaks
    the broadcast. Leave them replicated (they total only ~84 MB
    across all 42 layers, so the memory cost is irrelevant).
"""

from __future__ import annotations

from typing import Any

from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
)


def build_e4b_tp_plan(model: Any) -> tuple[dict[str, Any], Any, str, int, int]:
    """Build a TP plan for a HF Gemma4ForConditionalGeneration model.

    Returns (plan, decoder_module, decoder_module_dotted_name,
             owner_attn_count, shared_attn_count).
    """
    decoder = None
    decoder_name = None
    for name, mod in model.named_modules():
        if name.endswith("language_model") and hasattr(mod, "layers"):
            decoder = mod
            decoder_name = name
            break
    if decoder is None:
        raise RuntimeError(
            "Could not locate `language_model` decoder in the HF model. "
            f"Top-level children: "
            f"{[n for n, _ in model.named_children()]}"
        )

    plan: dict[str, Any] = {}
    owners = 0
    shareds = 0

    for i, layer in enumerate(decoder.layers):
        prefix = f"{decoder_name}.layers.{i}"
        attn_keys = {n for n, _ in layer.self_attn.named_children()}

        # q_proj and o_proj are present on every layer.
        plan[f"{prefix}.self_attn.q_proj"] = ColwiseParallel()
        plan[f"{prefix}.self_attn.o_proj"] = RowwiseParallel()

        # k_proj / v_proj only on "owner" layers (the layers that
        # actually compute KV; others share KV with a prior owner).
        if "k_proj" in attn_keys and "v_proj" in attn_keys:
            plan[f"{prefix}.self_attn.k_proj"] = ColwiseParallel()
            plan[f"{prefix}.self_attn.v_proj"] = ColwiseParallel()
            owners += 1
        else:
            shareds += 1

        # MLP — present on every layer.
        plan[f"{prefix}.mlp.gate_proj"] = ColwiseParallel()
        plan[f"{prefix}.mlp.up_proj"] = ColwiseParallel()
        plan[f"{prefix}.mlp.down_proj"] = RowwiseParallel()

        # Per-Layer Embeddings (PLE) — DO NOT shard. The HF forward
        # path multiplies hidden_states (full-rank, 2560-dim) against
        # per_layer_input (also expected full-rank). Sharding either
        # the gate or the projection breaks the broadcast at runtime.
        # Leave replicated — they're only ~2 MB per layer, ~84 MB total.

    return plan, decoder, decoder_name, owners, shareds
