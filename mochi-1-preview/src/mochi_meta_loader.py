"""Meta-init sharded weight loader for Mochi-1's `MochiTransformer3DModel`.

Usage:

    with torch.device("meta"):
        model = MochiTransformer3DModel.from_config(cfg)
    parallelize_module(model, mesh, mochi_tp_plan(world_size))
    load_weights_sharded(model, weights_dir, tp_local_rank=rank,
                         world_size=world_size, variant="bf16")

Streams one checkpoint shard at a time and keeps only this rank's slice, so
peak host memory stays near `20 GB / world_size` instead of staging the full
20 GB per process.

## The fused-SwiGLU trap

`ff.net.0.proj` is a single `Linear(3072 -> 16384)` whose output is split at
runtime by `diffusers.models.activations.SwiGLU`:

    hidden_states, gate = self.proj(x).chunk(2, dim=-1)
    return hidden_states * self.activation(gate)

So global output rows `[0:8192]` are the value half and `[8192:16384]` are
the gate half. A plain contiguous column shard hands rank *r* rows
`[r*4096 : (r+1)*4096]`, and the local `chunk(2)` then splits *that* block
in half -- pairing global rows `[0:2048]` with `[2048:4096]` on rank 0,
when the correct pairing is `[0:2048]` with `[8192:10240]`.

The failure is silent: shapes all check out, the model runs, the video is
just wrong. `_shard_fused_glu` fixes it by giving each rank
`concat(value_slice_r, gate_slice_r)`, so the local `chunk(2)` recovers the
right pairing.

The resulting DTensor's *global* view is a permutation of the true weight.
That is fine and intentional -- nothing ever reconstructs the global
tensor, and every consumer only touches the local shard.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

# Sharding rules, first match wins.
#   0    -> column shard (split output features)
#   1    -> row shard (split input features)
#   None -> replicate
#   "glu"-> fused [value|gate] column shard, see _shard_fused_glu
SHARD_RULES: list[tuple[re.Pattern[str], object]] = [
    # Fused SwiGLU projections MUST come before the generic rules.
    (re.compile(r"\.ff\.net\.0\.proj\.weight$"), "glu"),
    (re.compile(r"\.ff_context\.net\.0\.proj\.weight$"), "glu"),

    # Visual + text QKV: produce inner_dim -> column shard. (No biases in
    # the checkpoint: bias=False and added_proj_bias=False.)
    (re.compile(r"\.attn1\.to_[qkv]\.weight$"), 0),
    (re.compile(r"\.attn1\.add_[qkv]_proj\.weight$"), 0),

    # Attention output projections: consume inner_dim -> row shard.
    (re.compile(r"\.attn1\.to_out\.0\.weight$"), 1),
    (re.compile(r"\.attn1\.to_out\.0\.bias$"), None),
    (re.compile(r"\.attn1\.to_add_out\.weight$"), 1),
    (re.compile(r"\.attn1\.to_add_out\.bias$"), None),

    # FFN second layer consumes the sharded inner dim -> row shard.
    (re.compile(r"\.ff\.net\.2\.weight$"), 1),
    (re.compile(r"\.ff_context\.net\.2\.weight$"), 1),

    # QK norms are [head_dim] = [128]; valid on any head sharding.
    (re.compile(r"\.attn1\.norm_(added_)?[qk]\.weight$"), None),

    # AdaLN modulation drives the unsharded hidden states -> replicate.
    (re.compile(r"\.norm1\.linear\.(weight|bias)$"), None),
    (re.compile(r"\.norm1_context\.linear(_1)?\.(weight|bias)$"), None),
]

# Params expected to remain replicated at the top level.
_TOP_LEVEL_REPLICATED = (
    "patch_embed.", "time_embed.", "norm_out.", "proj_out.", "pos_frequencies",
)


def _find_shard_rule(param_name: str) -> object:
    for pattern, rule in SHARD_RULES:
        if pattern.search(param_name):
            return rule
    return None


def _shard_contiguous(
    full: torch.Tensor, dim: int, rank: int, world_size: int
) -> torch.Tensor:
    if full.shape[dim] % world_size != 0:
        raise ValueError(
            f"dim {dim} of {tuple(full.shape)} not divisible by {world_size}"
        )
    per = full.shape[dim] // world_size
    return full.narrow(dim, rank * per, per).contiguous()


def _shard_fused_glu(
    full: torch.Tensor, rank: int, world_size: int
) -> torch.Tensor:
    """Column-shard a fused `[value | gate]` projection, preserving chunk(2).

    `full` is `(2 * inner, in_features)`. Returns
    `(2 * inner / world_size, in_features)` laid out as
    `[value_slice_r | gate_slice_r]`.
    """
    total_out = full.shape[0]
    if total_out % 2 != 0:
        raise ValueError(f"fused GLU output dim {total_out} is not even")
    inner = total_out // 2
    if inner % world_size != 0:
        raise ValueError(
            f"GLU inner dim {inner} not divisible by world_size {world_size}"
        )
    per = inner // world_size
    value = full.narrow(0, rank * per, per)
    gate = full.narrow(0, inner + rank * per, per)
    return torch.cat([value, gate], dim=0).contiguous()


def shard_tensor(
    full: torch.Tensor, rule: object, rank: int, world_size: int
) -> torch.Tensor:
    """Apply a sharding rule. Exposed so the offline tests can verify it."""
    if world_size == 1 or rule is None:
        return full
    if rule == "glu":
        return _shard_fused_glu(full, rank, world_size)
    return _shard_contiguous(full, int(rule), rank, world_size)


def _read_index(model_path: str | Path, variant: str | None) -> dict[str, str]:
    """Resolve the safetensors weight map for the requested variant.

    Mochi puts the variant *after* `.index`, i.e.
    `diffusion_pytorch_model.safetensors.index.bf16.json`, which is not the
    usual diffusers `...bf16.safetensors.index.json` layout -- hence the
    explicit candidate list rather than a generic pattern.
    """
    base = Path(model_path)
    candidates = []
    if variant:
        candidates.append(base / f"diffusion_pytorch_model.safetensors.index.{variant}.json")
        candidates.append(base / f"diffusion_pytorch_model.{variant}.safetensors.index.json")
    candidates.append(base / "diffusion_pytorch_model.safetensors.index.json")

    for path in candidates:
        if path.exists():
            weight_map = json.loads(path.read_text())["weight_map"]
            return weight_map

    single = base / "diffusion_pytorch_model.safetensors"
    if single.exists():
        with safe_open(single, framework="pt", device="cpu") as f:
            return {k: single.name for k in f.keys()}

    raise FileNotFoundError(
        f"no safetensors index found under {base} (tried: "
        f"{[c.name for c in candidates]})"
    )


def _resolve(model: nn.Module, dotted: str):
    """Walk a dotted param path. Returns `(parent_module, leaf_name)`."""
    *parents, leaf = dotted.split(".")
    module = model
    for part in parents:
        if not hasattr(module, part):
            return None, None
        module = getattr(module, part)
    return module, leaf


def load_weights_sharded(
    model: nn.Module,
    model_path: str | Path,
    *,
    tp_local_rank: int,
    world_size: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str | torch.device = "cpu",
    variant: str | None = "bf16",
    strict: bool = True,
    verbose: bool = True,
) -> dict:
    """Stream checkpoint weights into a meta-init, TP-sharded model.

    Args:
        strict: raise if any parameter is still on the meta device
            afterwards. Leave this on -- a stray meta parameter otherwise
            surfaces as an inscrutable device error deep in the first
            forward.

    Returns a summary dict (also handy for the offline tests).
    """
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        DTensor = None

    weight_map = _read_index(model_path, variant)
    by_shard: dict[str, list[str]] = {}
    for name, shard_file in weight_map.items():
        by_shard.setdefault(shard_file, []).append(name)

    if verbose and tp_local_rank == 0:
        print(
            f"[mochi_loader] {len(weight_map)} tensors across "
            f"{len(by_shard)} files, world_size={world_size}, dtype={dtype}",
            flush=True,
        )

    loaded: list[str] = []
    unexpected: list[str] = []
    rule_counts: dict[str, int] = {}

    for shard_file in sorted(by_shard):
        shard_path = Path(model_path) / shard_file
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for name in by_shard[shard_file]:
                parent, leaf = _resolve(model, name)
                if parent is None or not hasattr(parent, leaf):
                    unexpected.append(name)
                    continue
                target = getattr(parent, leaf)
                if target is None:
                    unexpected.append(name)
                    continue

                rule = _find_shard_rule(name)
                full = f.get_tensor(name).to(dtype)

                # Guard: if a rule would not divide evenly, fall back to
                # replication rather than crashing on an odd dim.
                if rule is not None and rule != "glu":
                    axis_len = full.shape[int(rule)]
                    if axis_len % world_size != 0:
                        rule = None

                piece = shard_tensor(full, rule, tp_local_rank, world_size)
                key = "replicate" if rule is None else str(rule)
                rule_counts[key] = rule_counts.get(key, 0) + 1

                if DTensor is not None and isinstance(target, DTensor):
                    piece = DTensor.from_local(
                        piece.to(device),
                        device_mesh=target.device_mesh,
                        placements=list(target.placements),
                    )
                    setattr(parent, leaf, nn.Parameter(piece, requires_grad=False))
                elif getattr(target, "is_meta", False):
                    setattr(
                        parent, leaf,
                        nn.Parameter(piece.to(device), requires_grad=False),
                    )
                else:
                    target.data.copy_(piece.to(target.device, target.dtype))
                loaded.append(name)

    still_meta = [
        n for n, p in model.named_parameters() if getattr(p, "is_meta", False)
    ] + [
        n for n, b in model.named_buffers() if getattr(b, "is_meta", False)
    ]

    summary = {
        "loaded": len(loaded),
        "unexpected": unexpected,
        "still_meta": still_meta,
        "rule_counts": rule_counts,
    }

    if verbose and tp_local_rank == 0:
        print(f"[mochi_loader] loaded {len(loaded)} tensors {rule_counts}", flush=True)
        if unexpected:
            print(
                f"[mochi_loader] WARNING: {len(unexpected)} checkpoint keys "
                f"absent from the model: {unexpected[:5]}",
                flush=True,
            )
        if still_meta:
            print(
                f"[mochi_loader] ERROR: {len(still_meta)} params still on "
                f"meta: {still_meta[:5]}",
                flush=True,
            )

    if strict and still_meta:
        raise RuntimeError(
            f"{len(still_meta)} parameters left on the meta device after "
            f"loading; first few: {still_meta[:8]}"
        )

    # A checkpoint key the model has no home for is a silent-corruption risk on
    # the first real-weight load (the model runs, but a projection may be
    # unshifted or a norm missing) -- exactly the "structurally correct but
    # wrong video" failure mode. Under strict loading, refuse rather than warn.
    if strict and unexpected:
        raise RuntimeError(
            f"{len(unexpected)} checkpoint keys have no matching module in the "
            f"model; first few: {unexpected[:8]}. This usually means the "
            f"checkpoint layout drifted from MochiTransformer3DModel. Pass "
            f"strict=False to downgrade to a warning."
        )

    return summary
