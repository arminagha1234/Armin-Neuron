"""Meta-init weight loader for Qwen-Image transformer under TP.

Mirrors `neuron/examples/wan_training/meta_init_loader.py`. The pattern:

    with torch.device("meta"):
        model = LargeModel(config)
    parallelize_module(model, mesh, plan)
    load_weights_sharded(model, model_path, tp_local_rank, dtype)

Loading streams safetensors shards from disk and slices each tensor
along the right dim per parameter-name pattern, installing only this
rank's piece on the Neuron device. Avoids OOM by never holding the full
model on one device.

Per-tensor sharding rules for QwenImageTransformer2DModel:

    - {to_q,to_k,to_v}.weight              → split dim 0 (output)
    - {to_q,to_k,to_v}.bias                → split dim 0
    - to_out.0.weight                      → split dim 1 (input)
    - to_out.0.bias                        → replicate (post-allreduce)
    - ff.net.0.proj.weight                 → split dim 0
    - ff.net.0.proj.bias                   → split dim 0
    - ff.net.2.weight                      → split dim 1
    - ff.net.2.bias                        → replicate
    - norm_q.weight, norm_k.weight         → split dim 0 (sharded RMSNorm)
    - All other params (norms, embeds, etc.) → replicate

The rules below pattern-match parameter names. Any param not matching a
shard rule is treated as replicate.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn


# Patterns: (regex, shard_dim or None for replicate)
SHARD_RULES: list[tuple[re.Pattern[str], int | None]] = [
    # Image-side attention QKV — column shard (output dim)
    (re.compile(r"\.attn\.to_[qkv]\.weight$"), 0),
    (re.compile(r"\.attn\.to_[qkv]\.bias$"), 0),
    # Image-side attention output — row shard (input dim)
    (re.compile(r"\.attn\.to_out\.0\.weight$"), 1),
    (re.compile(r"\.attn\.to_out\.0\.bias$"), None),  # replicate (post-allreduce)
    # Text-side attention QKV (added tokens path)
    (re.compile(r"\.attn\.add_[qkv]_proj\.weight$"), 0),
    (re.compile(r"\.attn\.add_[qkv]_proj\.bias$"), 0),
    # Text-side attention output
    (re.compile(r"\.attn\.to_add_out\.weight$"), 1),
    (re.compile(r"\.attn\.to_add_out\.bias$"), None),
    # Image-side FFN
    (re.compile(r"\.img_mlp\.net\.0\.proj\.weight$"), 0),
    (re.compile(r"\.img_mlp\.net\.0\.proj\.bias$"), 0),
    (re.compile(r"\.img_mlp\.net\.2\.weight$"), 1),
    (re.compile(r"\.img_mlp\.net\.2\.bias$"), None),
    # Text-side FFN
    (re.compile(r"\.txt_mlp\.net\.0\.proj\.weight$"), 0),
    (re.compile(r"\.txt_mlp\.net\.0\.proj\.bias$"), 0),
    (re.compile(r"\.txt_mlp\.net\.2\.weight$"), 1),
    (re.compile(r"\.txt_mlp\.net\.2\.bias$"), None),
    # Sharded RMSNorm weights — but ONLY if shape == inner_dim. In
    # Qwen-Image-Edit-2511 the norm_q/k/added_q/added_k weights are
    # head_dim-sized [128] (per-head normalization, applied after
    # reshape to (B, S, H, D)), so they DO NOT need sharding. We
    # special-case this in load_weights_sharded by inspecting the
    # actual tensor shape against world_size; if shape[0] == inner_dim
    # Per-head RMSNorm weights — these are [head_dim]=[128] in 2511,
    # applied AFTER reshape to (B, S, H, D), so they MUST NOT be sharded.
    # The earlier rules said S0 + a runtime check; switching to None
    # here makes intent explicit and fixes a case where the runtime
    # check missed (head_dim=128 IS divisible by world=4 → 32, which
    # is wrong because the dim being sharded would be the head_dim
    # axis, not a model-dim axis).
    (re.compile(r"\.attn\.norm_[qk]\.weight$"), None),
    (re.compile(r"\.attn\.norm_added_[qk]\.weight$"), None),
]


def _find_shard_dim(param_name: str) -> int | None:
    for pat, dim in SHARD_RULES:
        if pat.search(param_name):
            return dim
    return None


def _shard_tensor(
    full: torch.Tensor,
    *,
    dim: int,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    if full.shape[dim] % world_size != 0:
        raise ValueError(
            f"shard dim {dim} of shape {full.shape} not divisible by world_size {world_size}"
        )
    per = full.shape[dim] // world_size
    start = rank * per
    return full.narrow(dim, start, per).contiguous()


def _read_safetensors_index(model_path: str) -> dict[str, str]:
    """Read the sharded safetensors index file.

    Returns a dict mapping parameter name → shard filename.
    """
    index_path = Path(model_path) / "diffusion_pytorch_model.safetensors.index.json"
    single_path = Path(model_path) / "diffusion_pytorch_model.safetensors"
    if index_path.exists():
        with index_path.open() as f:
            data = json.load(f)
        return data["weight_map"]
    elif single_path.exists():
        # Single-shard model
        with safe_open(single_path, framework="pt", device="cpu") as f:
            return {key: single_path.name for key in f.keys()}
    else:
        raise FileNotFoundError(
            f"No safetensors index or single-file at {model_path}"
        )


def load_weights_sharded(
    model: nn.Module,
    model_path: str,
    *,
    tp_local_rank: int,
    world_size: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
) -> None:
    """Stream weights from disk into a meta-init model, sharded per rank.

    Pre:
        - `model` was constructed under `torch.device("meta")`
        - `parallelize_module` already applied (sets DTensor placements)
        - `model_path` contains a safetensors index + shards
    Post:
        - Every parameter has materialised storage for this rank's slice
        - Replicated params are copied as-is

    Notes on DTensor:
        After parallelize_module, parameters that are sharded are
        DTensors. We use distribute_tensor / from_local to install the
        local shard with the right placement. Replicated params get
        Replicate() placement.
    """
    try:
        from torch.distributed.tensor import DTensor
        from torch.distributed.tensor.placement_types import (
            Replicate,
            Shard,
        )
    except ImportError:
        DTensor = None  # type: ignore[misc, assignment]
        Replicate = None  # type: ignore[misc, assignment]
        Shard = None  # type: ignore[misc, assignment]

    weight_map = _read_safetensors_index(model_path)
    print(
        f"[meta_loader rank{tp_local_rank}] {len(weight_map)} params "
        f"in index, world_size={world_size}, dtype={dtype}"
    )

    # Group params by shard file so we open each .safetensors once.
    by_shard: dict[str, list[str]] = {}
    for param_name, shard_file in weight_map.items():
        by_shard.setdefault(shard_file, []).append(param_name)

    state_dict = dict(model.state_dict())
    materialized: list[str] = []
    missing: list[str] = []

    for shard_file, names in by_shard.items():
        shard_path = Path(model_path) / shard_file
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for name in names:
                if name not in state_dict:
                    missing.append(name)
                    continue
                full = f.get_tensor(name).to(dtype)
                shard_dim = _find_shard_dim(name)
                # Per-head RMSNorm sanity check: if the rule says shard
                # but the actual tensor is head_dim-sized (not divisible
                # by world_size or smaller than world_size), it's a
                # per-head norm — replicate instead.
                if shard_dim is not None:
                    target_dim_size = full.shape[shard_dim]
                    if target_dim_size < world_size or target_dim_size % world_size != 0:
                        # Per-head or other non-shardable; replicate
                        shard_dim = None
                if shard_dim is None:
                    piece = full
                else:
                    piece = _shard_tensor(
                        full, dim=shard_dim, rank=tp_local_rank, world_size=world_size
                    )
                # Install
                target = state_dict[name]
                # If target is a DTensor (post-parallelize_module), use
                # the DTensor.from_local path.
                if DTensor is not None and isinstance(target, DTensor):
                    placements = list(target.placements)
                    mesh = target.device_mesh
                    new_dt = DTensor.from_local(
                        piece.to(device),
                        device_mesh=mesh,
                        placements=placements,
                    )
                    _set_module_param(
                        model,
                        name,
                        nn.Parameter(new_dt) if not name.endswith(".bias") or new_dt.requires_grad else nn.Parameter(new_dt, requires_grad=False),
                    )
                elif target.is_meta:
                    new_param = nn.Parameter(piece.to(device), requires_grad=False)
                    _set_module_param(model, name, new_param)
                else:
                    target.data.copy_(piece.to(target.device, target.dtype))
                materialized.append(name)

    if missing:
        print(f"[meta_loader rank{tp_local_rank}] WARN: {len(missing)} keys in checkpoint not in model: {missing[:3]}")
    print(f"[meta_loader rank{tp_local_rank}] materialized {len(materialized)} params")


def _set_module_param(model: nn.Module, name: str, new_param: nn.Parameter) -> None:
    """Replace a parameter on a nested module given its dotted name."""
    parts = name.split(".")
    *parent_path, leaf = parts
    parent = model
    for p in parent_path:
        parent = getattr(parent, p)
    # Use setattr — works for nn.Parameter
    setattr(parent, leaf, new_param)
