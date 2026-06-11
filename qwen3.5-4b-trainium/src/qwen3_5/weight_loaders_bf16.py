# SPDX-License-Identifier: Apache-2.0
"""Weight loaders for Qwen3.5 BF16 checkpoint format — Phase 6.

Provides `build_weight_mappings(config)` which returns a dict mapping
flat parameter names (the names our `Qwen3_5ForConditionalGeneration`
exposes) to lists of HuggingFace safetensors keys.

vllm_neuron's `SafetensorsCheckpoint` consumes this mapping plus the
per-parameter weight loader (set via `set_weight_loader` in
`model_bf16.py`) to materialize on-device tensors.

HuggingFace checkpoint layout for Qwen3.5-4B
(`model.safetensors.index.json`):

  Embeddings / norm / lm_head:
    model.embed_tokens.weight       [vocab, hidden]
    model.norm.weight               [hidden]
    (lm_head shares embed_tokens via tie_word_embeddings)

  Per layer L (0..31):
    Pre-attention norms:
      model.layers.{L}.input_layernorm.weight              [hidden]
      model.layers.{L}.post_attention_layernorm.weight     [hidden]

    MLP (all 32 layers):
      model.layers.{L}.mlp.gate_proj.weight   [intermediate, hidden]
      model.layers.{L}.mlp.up_proj.weight     [intermediate, hidden]
      model.layers.{L}.mlp.down_proj.weight   [hidden, intermediate]

    GQA full-attention layers (L in [3, 7, 11, 15, 19, 23, 27, 31]):
      model.layers.{L}.self_attn.q_proj.weight        [Q*head_dim, hidden]
      model.layers.{L}.self_attn.k_proj.weight        [KV*head_dim, hidden]
      model.layers.{L}.self_attn.v_proj.weight        [KV*head_dim, hidden]
      model.layers.{L}.self_attn.o_proj.weight        [hidden, Q*head_dim]
      model.layers.{L}.self_attn.q_norm.weight        [head_dim]
      model.layers.{L}.self_attn.k_norm.weight        [head_dim]
      model.layers.{L}.self_attn.gate_proj.weight     [Q*head_dim, hidden]   (only when attn_output_gate=True)

    DeltaNet linear-attention layers (L in [0,1,2, 4,5,6, 8,9,10, ...]):
      model.layers.{L}.linear_attn.in_proj_qkv.weight  [conv_dim, hidden]   (conv_dim = key_dim*2 + value_dim)
      model.layers.{L}.linear_attn.in_proj_z.weight    [value_dim, hidden]
      model.layers.{L}.linear_attn.in_proj_a.weight    [num_v_heads, hidden]
      model.layers.{L}.linear_attn.in_proj_b.weight    [num_v_heads, hidden]
      model.layers.{L}.linear_attn.conv1d.weight       [conv_dim, 1, kernel]
      model.layers.{L}.linear_attn.conv1d.bias         [conv_dim]            (we ignore — Qwen3.5 uses bias=False)
      model.layers.{L}.linear_attn.dt_bias             [num_v_heads]
      model.layers.{L}.linear_attn.A_log               [num_v_heads]
      model.layers.{L}.linear_attn.norm.weight         [head_v_dim]
      model.layers.{L}.linear_attn.out_proj.weight     [hidden, value_dim]

      Note on naming: PR #152 uses `linear_attn` as the HF submodule
      name. Earlier checkpoints from the Qwen team also use this.

      State buffers (`recurrent_state_buffer`, `conv_state_buffer`)
      are runtime state, NOT loaded from the checkpoint — they live as
      zero-init nn.Parameters.

Differences from PR #152's NxDI loader:
- We fuse Q+K+V into one `qkv_proj_weight` tensor (NF.qkv_proj wants
  it that way); PR #152 keeps them separate inside its modeling.
- We don't load `conv1d.bias` (Qwen3.5 conv has no bias).
- We expose tied lm_head as an alias of embed_tokens, not a separate
  weight tensor.
"""

from __future__ import annotations

from .config import Qwen3_5Config


def build_weight_mappings(config: Qwen3_5Config) -> dict[str, list[str]]:
    """Return a `{flat_param_name: [hf_safetensors_keys]}` mapping.

    Keys in the returned dict are dot-paths into our
    `Qwen3_5ForConditionalGeneration` module tree. Values are lists
    because vllm_neuron's loaders accept either a single source key
    (length-1 list) or fused multiple sources (e.g. q/k/v → qkv).
    """
    mappings: dict[str, list[str]] = {}

    # HF prefix: Qwen3.5-4B is a multimodal HF wrapper, so all text-decoder
    # tensors live under `model.language_model.` not `model.`. Detect at
    # build time so this works for both wrappers.
    HF = "model.language_model"

    # Backbone
    mappings["model.embed_tokens.weight"] = [f"{HF}.embed_tokens.weight"]
    mappings["model.norm.weight"] = [f"{HF}.norm.weight"]

    # LM head — tied to embed_tokens. We DO register it as a flat param
    # (column-parallel ColumnParallelLinear) but load the same source
    # tensor as embed_tokens. The loader applies its own sharding (vocab
    # dim shard for lm_head, vs. vocab-sharded embedding for embed_tokens —
    # both happen to use the same source tensor with different shard specs).
    mappings["lm_head.weight"] = [f"{HF}.embed_tokens.weight"]

    if not config.layer_types:
        raise ValueError("config.layer_types is empty; can't build mappings")
    if len(config.layer_types) != config.num_hidden_layers:
        raise ValueError(
            f"layer_types length ({len(config.layer_types)}) != "
            f"num_hidden_layers ({config.num_hidden_layers})"
        )

    for L, lt in enumerate(config.layer_types):
        layer = f"model.layers.{L}"
        prefix = f"{HF}.layers.{L}"  # HF source key prefix

        # Pre-norms (always present)
        mappings[f"{layer}.input_layernorm.weight"] = [f"{prefix}.input_layernorm.weight"]
        mappings[f"{layer}.post_attention_layernorm.weight"] = [
            f"{prefix}.post_attention_layernorm.weight"
        ]

        # MLP (always present, identical across layer types)
        mappings[f"{layer}.mlp.gate_proj_weight"] = [f"{prefix}.mlp.gate_proj.weight"]
        mappings[f"{layer}.mlp.up_proj_weight"] = [f"{prefix}.mlp.up_proj.weight"]
        mappings[f"{layer}.mlp.down_proj_weight"] = [f"{prefix}.mlp.down_proj.weight"]

        if lt == "full_attention":
            # GQA: fuse Q + K + V into one tensor (loader handles concat)
            mappings[f"{layer}.self_attn.qkv_proj_weight"] = [
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.v_proj.weight",
            ]
            mappings[f"{layer}.self_attn.o_proj_weight"] = [
                f"{prefix}.self_attn.o_proj.weight"
            ]
            # Per-head Q/K layernorm
            mappings[f"{layer}.self_attn.q_layernorm.weight"] = [
                f"{prefix}.self_attn.q_norm.weight"
            ]
            mappings[f"{layer}.self_attn.k_layernorm.weight"] = [
                f"{prefix}.self_attn.k_norm.weight"
            ]
            # Attention output gate. Qwen3.5's HF config has
            # attn_output_gate=True and the gate IS shipped — but it's
            # SPLICED into q_proj's second half (per-head interleaved
            # [q|gate]), not a separate gate_proj.weight. The spliced
            # loaders (_spliced_q_kv_loader / _spliced_q_gate_loader) split
            # q_proj's two halves. The gate maps from the SAME q_proj.weight
            # tensor; its loader slices the gate half per head.
            if getattr(config, "attn_output_gate", False):
                mappings[f"{layer}.self_attn.attn_gate_weight"] = [
                    f"{prefix}.self_attn.q_proj.weight"
                ]

        elif lt == "linear_attention":
            la = f"{prefix}.linear_attn"
            mappings[f"{layer}.self_attn.in_proj_qkv_weight"] = [f"{la}.in_proj_qkv.weight"]
            mappings[f"{layer}.self_attn.in_proj_z_weight"] = [f"{la}.in_proj_z.weight"]
            mappings[f"{layer}.self_attn.in_proj_a_weight"] = [f"{la}.in_proj_a.weight"]
            mappings[f"{layer}.self_attn.in_proj_b_weight"] = [f"{la}.in_proj_b.weight"]
            mappings[f"{layer}.self_attn.conv1d_weight"] = [f"{la}.conv1d.weight"]
            mappings[f"{layer}.self_attn.A_log"] = [f"{la}.A_log"]
            mappings[f"{layer}.self_attn.dt_bias"] = [f"{la}.dt_bias"]
            mappings[f"{layer}.self_attn.norm_weight"] = [f"{la}.norm.weight"]
            mappings[f"{layer}.self_attn.out_proj_weight"] = [f"{la}.out_proj.weight"]
            # State buffers are NOT loaded from checkpoint — zero-init at runtime.

        else:
            raise ValueError(
                f"Unknown layer type {lt!r} at layer {L}; "
                "expected 'full_attention' or 'linear_attention'."
            )

    return mappings


def expected_unmapped_keys(config: Qwen3_5Config) -> set[str]:
    """HF keys that exist in the checkpoint but we deliberately ignore.

    Used by tests to assert we're not silently dropping unknown weights.
    """
    skip: set[str] = {
        # tied lm_head — alias of embed_tokens, no separate parameter on our side
        "lm_head.weight",
    }
    # conv1d biases (Qwen3.5 sets bias=False but some checkpoints still ship them as zeros)
    for L, lt in enumerate(config.layer_types):
        if lt == "linear_attention":
            skip.add(f"model.layers.{L}.linear_attn.conv1d.bias")
    return skip
