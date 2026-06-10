# SPDX-License-Identifier: Apache-2.0
"""Per-rank weight shape audit for the Qwen3.6-27B adapter.

Hooks into vllm_neuron's `Qwen3_6ForConditionalGeneration` AFTER weights
load on each TP rank, dumps {flat_name: (shape, dtype, mean, std, l2)}
for the parameters most likely to be mis-sharded:

  - lm_head.weight                                           (untied head)
  - model.embed_tokens.weight                                (vocab-sharded embed)
  - model.layers.{full_attn[0]}.self_attn.qkv_proj_weight    (fused QKV, GQA)
  - model.layers.{full_attn[0]}.self_attn.o_proj_weight      (output proj)
  - model.layers.{full_attn[0]}.self_attn.q_layernorm.weight (per-head norm)
  - model.layers.{linear_attn[0]}.self_attn.in_proj_qkv_weight (deltanet QKV)
  - model.layers.{linear_attn[0]}.self_attn.out_proj_weight    (deltanet output)
  - model.layers.0.mlp.gate_proj_weight                      (SwiGLU gate)
  - model.layers.0.mlp.down_proj_weight                      (SwiGLU down)

Run inside the vllm_neuron container after sitecustomize loads:

  PYTHONPATH=/workspace/qwen36_adapter \\
  python -m qwen3_6.test.dump_weight_shapes \\
      --model /root/models/Qwen3.6-27B \\
      --tp 4

Output goes to stdout per worker and is also written to
/tmp/weight_audit_tp{tp}.json so we can diff TP=4 vs TP=8.
"""

import argparse
import json
import os
import sys


def _stats(t) -> dict:
    import torch
    if t is None:
        return {"shape": None}
    if hasattr(t, "is_meta") and t.is_meta:
        return {"shape": list(t.shape), "dtype": str(t.dtype), "is_meta": True}
    try:
        f = t.detach().to(torch.float32).cpu()
        return {
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "mean": float(f.mean().item()),
            "std": float(f.std().item()),
            "l2": float(f.norm().item()),
            "absmax": float(f.abs().max().item()),
        }
    except Exception as exc:
        return {"shape": list(t.shape) if hasattr(t, "shape") else None,
                "error": repr(exc)}


def _walk_named(model) -> dict:
    """Walk through possibly-wrapped model to collect named_parameters + buffers."""
    out = {}
    for n, p in model.named_parameters():
        out[n] = p
    for n, b in model.named_buffers():
        if n not in out:
            out[n] = b
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/Qwen3.6-27B")
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--out", default=None,
                    help="JSON output path (default: /tmp/weight_audit_tp{tp}.json)")
    args = ap.parse_args()

    out_path = args.out or f"/tmp/weight_audit_tp{args.tp}.json"

    # Defer all heavy imports until after argparse so --help is fast.
    # Also ensure our adapter is registered.
    import qwen3_6  # noqa: F401  triggers register()

    # Build a NeuronConfig that mirrors what `vllm serve` would build,
    # then bring up the model the same way the runner does. Easier path:
    # use the public-ish factory from vllm_neuron tests.
    print(f"[dump] model={args.model} tp={args.tp}", flush=True)
    print(f"[dump] writing -> {out_path}", flush=True)

    # We use vllm_neuron's NeuronConfig and the adapter factory.
    from vllm_neuron.model.neuron_config import NeuronConfig

    nc = NeuronConfig(
        tp_degree=args.tp,
        seq_len=4096,
        batch_size=1,
        torch_dtype="bfloat16",
        on_device_sampling_config=None,
    )

    import json as _json
    with open(os.path.join(args.model, "config.json")) as f:
        hf_cfg = _json.load(f)

    from qwen3_6.factory import Qwen3_6ForConditionalGeneration
    model = Qwen3_6ForConditionalGeneration.from_configs(hf_cfg, neuron_config=nc)

    # We do NOT call load_weights here — this dump runs OUTSIDE of the
    # full distributed runner (no TP group is initialized in this single
    # process). Goal: show the unloaded parameter SHAPES so we can
    # eyeball whether the 27B Q/K/V split + lm_head spec is correct
    # before paying for another full compile.
    named = _walk_named(model)

    # Layer types to find first full-attn / first linear-attn layer.
    cfg = model.config if hasattr(model, "config") else None
    layer_types = list(getattr(cfg, "layer_types", []) or [])
    first_full = next((i for i, lt in enumerate(layer_types) if lt == "full_attention"), None)
    first_lin = next((i for i, lt in enumerate(layer_types) if lt == "linear_attention"), None)

    targets = [
        "lm_head.weight",
        "model.embed_tokens.weight",
        "model.norm.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj_weight",
        "model.layers.0.mlp.up_proj_weight",
        "model.layers.0.mlp.down_proj_weight",
    ]
    if first_full is not None:
        for k in (
            f"model.layers.{first_full}.self_attn.qkv_proj_weight",
            f"model.layers.{first_full}.self_attn.o_proj_weight",
            f"model.layers.{first_full}.self_attn.q_layernorm.weight",
            f"model.layers.{first_full}.self_attn.k_layernorm.weight",
        ):
            targets.append(k)
    if first_lin is not None:
        for k in (
            f"model.layers.{first_lin}.self_attn.in_proj_qkv_weight",
            f"model.layers.{first_lin}.self_attn.in_proj_z_weight",
            f"model.layers.{first_lin}.self_attn.in_proj_a_weight",
            f"model.layers.{first_lin}.self_attn.in_proj_b_weight",
            f"model.layers.{first_lin}.self_attn.conv1d_weight",
            f"model.layers.{first_lin}.self_attn.A_log",
            f"model.layers.{first_lin}.self_attn.dt_bias",
            f"model.layers.{first_lin}.self_attn.norm_weight",
            f"model.layers.{first_lin}.self_attn.out_proj_weight",
        ):
            targets.append(k)

    audit = {
        "tp": args.tp,
        "model": args.model,
        "first_full_attn_layer": first_full,
        "first_linear_attn_layer": first_lin,
        "params": {},
    }
    for name in targets:
        t = named.get(name)
        audit["params"][name] = _stats(t) if t is not None else {"shape": None, "missing": True}

    with open(out_path, "w") as f:
        json.dump(audit, f, indent=2, default=str)

    print(f"[dump] {len(audit['params'])} params probed; first full-attn layer = {first_full}, "
          f"first linear-attn layer = {first_lin}")
    for name, info in audit["params"].items():
        print(f"  {name:64s}  {info}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
