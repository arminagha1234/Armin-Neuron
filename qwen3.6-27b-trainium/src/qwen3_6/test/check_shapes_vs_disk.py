# SPDX-License-Identifier: Apache-2.0
"""Compare expected sharded shapes vs the on-disk safetensors shapes.

Doesn't need a TP group — just opens the safetensors metadata and the
HF config, computes what each parameter SHOULD look like at TP=N for
both 4B-style and 27B configs, and prints a clean side-by-side table.

Usage:
    PYTHONPATH=/workspace/qwen36_adapter \\
    python -m qwen3_6.test.check_shapes_vs_disk \\
        --model /root/models/Qwen3.6-27B --tp 8

What we look for:
- `lm_head.weight` shape on disk: should be (vocab, hidden) = (248320, 5120).
  Our model's lm_head.weight is sharded as (vocab/N, hidden) on each rank
  (last_dim_padding_weight_loader with shard_dim=0). At TP=8 that is
  (31040, 5120) per rank. At TP=4: (62080, 5120).

- `model.language_model.layers.{full_attn}.self_attn.q_proj.weight`
  shape on disk: (Q*head_dim, hidden) = (24*256, 5120) = (6144, 5120).
- `...k_proj.weight`: (4*256, 5120) = (1024, 5120).
- `...v_proj.weight`: (4*256, 5120) = (1024, 5120).
- Fused qkv at TP=8: q_per_rank=3, kv_per_rank=1 (KV replicated 2x).
  qkv_size = 3*256 + 2*(1*256) = 1280. Storage layout in our model is
  transposed: (hidden, qkv_size) = (5120, 1280) per rank.
- Fused qkv at TP=4: q_per_rank=6, kv_per_rank=1 (KV replicated 1x).
  qkv_size = 6*256 + 2*(1*256) = 2048. Stored: (5120, 2048) per rank.

Mismatches with model.parameter shapes flag a loader spec bug.
"""

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/Qwen3.6-27B")
    ap.add_argument("--tp", type=int, default=8)
    args = ap.parse_args()

    cfg_path = os.path.join(args.model, "config.json")
    with open(cfg_path) as f:
        full_cfg = json.load(f)
    text_cfg = full_cfg.get("text_config", full_cfg)

    hidden = int(text_cfg["hidden_size"])
    n_q = int(text_cfg["num_attention_heads"])
    n_kv = int(text_cfg["num_key_value_heads"])
    head_dim = int(text_cfg["head_dim"])
    vocab = int(text_cfg["vocab_size"])
    intermediate = int(text_cfg["intermediate_size"])
    layer_types = text_cfg.get("layer_types", [])
    first_full = next((i for i, lt in enumerate(layer_types) if lt == "full_attention"), None)
    first_lin = next((i for i, lt in enumerate(layer_types) if lt == "linear_attention"), None)

    # DeltaNet
    dn_v = int(text_cfg.get("linear_num_value_heads", text_cfg.get("num_v_heads", 0)))
    dn_k = int(text_cfg.get("linear_num_key_heads", text_cfg.get("num_k_heads", 0)))
    dn_v_dim = int(text_cfg.get("linear_value_head_dim", text_cfg.get("v_head_dim", 0)))
    dn_k_dim = int(text_cfg.get("linear_key_head_dim", text_cfg.get("k_head_dim", 0)))

    print(f"=== Qwen3.6-27B config (text) ===")
    print(f"  hidden={hidden}  n_q={n_q}  n_kv={n_kv}  head_dim={head_dim}  vocab={vocab}")
    print(f"  intermediate={intermediate}  layers={text_cfg.get('num_hidden_layers')}")
    print(f"  first full-attn layer = {first_full}  first linear-attn layer = {first_lin}")
    print(f"  DeltaNet: v_heads={dn_v} k_heads={dn_k} v_dim={dn_v_dim} k_dim={dn_k_dim}")

    # On-disk safetensors shapes
    print()
    print(f"=== Reading actual safetensors header for key shapes (TP-agnostic) ===")
    from safetensors import safe_open
    HF = "model.language_model"
    suspects = [
        "lm_head.weight",
        f"{HF}.embed_tokens.weight",
        f"{HF}.norm.weight",
    ]
    if first_full is not None:
        suspects += [
            f"{HF}.layers.{first_full}.self_attn.q_proj.weight",
            f"{HF}.layers.{first_full}.self_attn.k_proj.weight",
            f"{HF}.layers.{first_full}.self_attn.v_proj.weight",
            f"{HF}.layers.{first_full}.self_attn.o_proj.weight",
            f"{HF}.layers.{first_full}.self_attn.q_norm.weight",
            f"{HF}.layers.{first_full}.self_attn.k_norm.weight",
        ]
    if first_lin is not None:
        suspects += [
            f"{HF}.layers.{first_lin}.linear_attn.in_proj_qkv.weight",
            f"{HF}.layers.{first_lin}.linear_attn.in_proj_z.weight",
            f"{HF}.layers.{first_lin}.linear_attn.in_proj_a.weight",
            f"{HF}.layers.{first_lin}.linear_attn.out_proj.weight",
            f"{HF}.layers.{first_lin}.linear_attn.A_log",
            f"{HF}.layers.{first_lin}.linear_attn.dt_bias",
            f"{HF}.layers.{first_lin}.linear_attn.norm.weight",
            f"{HF}.layers.{first_lin}.linear_attn.conv1d.weight",
        ]
    suspects += [
        f"{HF}.layers.0.mlp.gate_proj.weight",
        f"{HF}.layers.0.mlp.up_proj.weight",
        f"{HF}.layers.0.mlp.down_proj.weight",
    ]

    # Find which file each tensor lives in
    idx_path = os.path.join(args.model, "model.safetensors.index.json")
    with open(idx_path) as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]

    by_file: dict[str, list[str]] = {}
    for k in suspects:
        f = weight_map.get(k)
        if f is None:
            print(f"  MISSING in index: {k}")
            continue
        by_file.setdefault(f, []).append(k)

    for st_name, keys in by_file.items():
        st_path = os.path.join(args.model, st_name)
        with safe_open(st_path, framework="pt") as f:
            for k in keys:
                t = f.get_slice(k)
                print(f"  {k:80s}  shape={tuple(t.get_shape())}  dtype={t.get_dtype()}")

    # What we expect at this TP
    print()
    print(f"=== What our model EXPECTS at TP={args.tp} (rank 0) ===")
    tp = args.tp
    n_q_per_rank = n_q // tp
    if tp >= n_kv:
        n_kv_per_rank = 1
        kv_replicas = tp // n_kv
    else:
        n_kv_per_rank = n_kv // tp
        kv_replicas = 1
    q_size = n_q_per_rank * head_dim
    kv_size = n_kv_per_rank * head_dim
    qkv_size = q_size + 2 * kv_size

    print(f"  n_q_per_rank={n_q_per_rank}  n_kv_per_rank={n_kv_per_rank}  kv_replicas={kv_replicas}")
    print(f"  q_size={q_size}  kv_size={kv_size}  qkv_size={qkv_size}")
    print(f"  qkv_proj_weight  expected shape (hidden, qkv_size) = ({hidden}, {qkv_size})  [storage transposed]")
    o_in = (n_q * head_dim) // tp
    print(f"  o_proj_weight    expected shape ({o_in}, {hidden})  [storage transposed]")
    print(f"  lm_head.weight   expected (per rank) shape ({vocab // tp}, {hidden})  [shard_dim=0]")
    print(f"  embed_tokens.weight expected shape ({vocab // tp}, {hidden})  [vocab-sharded]")
    print(f"  mlp.gate/up_proj_weight expected ({hidden}, {intermediate // tp})  [shard_dim=1, transposed]")
    print(f"  mlp.down_proj_weight expected ({intermediate // tp}, {hidden})  [shard_dim=0, transposed]")

    if dn_v and dn_k:
        # in_proj_qkv: dim_in=hidden, dim_out = 2*key_dim + value_dim
        #   key_dim = dn_k * dn_k_dim ; value_dim = dn_v * dn_v_dim
        key_dim = dn_k * dn_k_dim
        value_dim = dn_v * dn_v_dim
        qkv_dim = 2 * key_dim + value_dim
        print(f"  deltanet.in_proj_qkv     expected (hidden, qkv_dim) = ({hidden}, {qkv_dim})")
        print(f"  deltanet.in_proj_z       expected (hidden, value_dim) = ({hidden}, {value_dim})")
        print(f"  deltanet.in_proj_a/b     expected (hidden, n_v_heads) = ({hidden}, {dn_v})")
        print(f"  deltanet.out_proj        expected (value_dim, hidden) = ({value_dim}, {hidden})")
        print(f"  deltanet.A_log/dt_bias   expected (n_v_heads,) = ({dn_v},)")
        print(f"  deltanet.norm.weight     expected (v_head_dim,) = ({dn_v_dim},)")
        print(f"  deltanet.conv1d.weight   expected (qkv_dim, 1, kernel) = ({qkv_dim}, 1, "
              f"{text_cfg.get('linear_conv_kernel_dim', text_cfg.get('conv_kernel_size', 4))})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
