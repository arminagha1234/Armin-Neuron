"""Tensor-parallel plan and Neuron fixes for Mochi-1 `MochiTransformer3DModel`.

Architecture (verified against the published checkpoint header, not inferred
from the docs -- see NOTES.md for the shape dump):

    48 blocks, 24 heads x 128 head_dim, inner_dim 3072
    text ("context") stream dim 1536  <- the "asymmetric" in AsymmDiT
    ff inner 8192, ff_context inner 4096
    block 47 is context_pre_only: no `to_add_out`, no `ff_context`
    10.028 B params total, 20.06 GB bf16

Per block:
    attn1.to_{q,k,v}        Linear(3072 -> 3072)   no bias
    attn1.add_{q,k,v}_proj  Linear(1536 -> 3072)   no bias
    attn1.to_out.0          Linear(3072 -> 3072)   bias
    attn1.to_add_out        Linear(3072 -> 1536)   bias   (absent on block 47)
    attn1.norm_{q,k,added_q,added_k}   [128]       replicated
    ff.net.0.proj           Linear(3072 -> 16384)  fused SwiGLU [value|gate]
    ff.net.2                Linear(8192 -> 3072)
    ff_context.net.0.proj   Linear(1536 -> 8192)   fused SwiGLU (absent on 47)
    ff_context.net.2        Linear(4096 -> 1536)   (absent on 47)
    norm1.linear            Linear(3072 -> 12288)  AdaLN, replicated
    norm1_context.linear    Linear(3072 -> 6144)   AdaLN, replicated

Two things worth flagging versus the LTX-2 recipe:

* **No adaptive QK-norm needed.** `MochiAttention` builds
  `MochiRMSNorm(dim_head, ...)`, so the norm weight is `[128]` and stays
  valid on a sharded head axis. LTX-2's `rms_norm_across_heads` normalised
  over the full inner_dim and therefore needed an all-reduce inside the
  norm. Mochi does not.

* **No `RankTensor`.** Mochi's RoPE frequencies live in a `pos_frequencies`
  parameter of shape `[3, 24, 64]`, i.e. indexed by head. Sharding that
  parameter on its head axis at load time gives each rank the correct
  per-rank frequencies for free, entirely outside the traced graph. That
  sidesteps the "Python rank bakes in as constant 0 under SPMD tracing"
  hazard that `RankTensor` exists to work around.

* **AdaLN is 27% of the model and stays replicated.** `norm1.linear` +
  `norm1_context.linear` are 2.705 B of the 10.028 B parameters. They
  modulate the unsharded hidden states, so sharding them would require an
  all-gather per block. Consequence: TP reduces weight memory sub-linearly.
  See `estimate_rank_weight_bytes`.
"""
from __future__ import annotations

import torch

# ── Architecture constants (from transformer/config.json) ───────────────────
N_LAYERS = 48
N_HEADS = 24
HEAD_DIM = 128
INNER_DIM = N_HEADS * HEAD_DIM          # 3072
TEXT_DIM = 1536                          # pooled_projection_dim
FF_INNER = (4 * INNER_DIM * 2) // 3      # 8192
FF_CONTEXT_INNER = (4 * TEXT_DIM * 2) // 3  # 4096
CONTEXT_PRE_ONLY_LAYER = N_LAYERS - 1    # 47
MAX_TEXT_TOKENS = 256

# Dims that must divide evenly for the plan to be valid.
_DIVISIBILITY = {
    "num_heads": N_HEADS,
    "ff_inner": FF_INNER,
    "ff_context_inner": FF_CONTEXT_INNER,
}

# TP degrees that both (a) evenly divide Mochi's dims and (b) form a valid
# collective group on the trn2 4x4 NeuronLink torus.
#
# Divisibility alone would allow {1, 2, 4, 8} (24 heads, 8192/4096 FF).
# But the collective layer only builds an algorithm for groups that tile the
# torus: measured on i-03a587c283fffb075, TP=2/4/16 init cleanly while TP=8
# fails with "no_hier no_mesh replica-group: [0..7]" -- 8 ranks are neither a
# ring nor a 2D sub-mesh of a 4x4 torus. So the usable intersection is {1,2,4}.
#
# TP=16 *does* init collectively (it is the full torus) but 24 heads do not
# divide by 16; using it needs the heads padded to 32 (see NOTES.md
# "pushing past the ceiling"). Not implemented, so 16 is not offered here.
VALID_WORLD_SIZES = (1, 2, 4)

# 8 divides the dims but deadlocks on this topology. Fail fast with the reason
# rather than hanging after a 20 GB weight load. Override only if you are on a
# topology where an 8-way group tiles (e.g. a differently wired instance).
_TORUS_BROKEN = {8}


def validate_world_size(world_size: int) -> None:
    """Raise if `world_size` cannot shard the architecture *and* form a
    collective group on this hardware.

    Two independent constraints:
      1. Divisibility: 24 heads + 8192/4096 FF inner dims.
      2. Collective topology: the trn2 4x4 NeuronLink torus builds algorithms
         for group sizes {2, 4, 16} but not 8 (empirically "no_hier no_mesh").

    TP=8 passes (1) but fails (2), which is the trap this guards: without the
    check you load 20 GB of sharded weights and only then hit the collective
    init failure.
    """
    import os

    bad = {
        name: dim for name, dim in _DIVISIBILITY.items()
        if dim % world_size != 0
    }
    if bad:
        raise ValueError(
            f"world_size={world_size} does not evenly divide {bad}. "
            f"Mochi-1 supports TP in {VALID_WORLD_SIZES} on the trn2 torus."
        )

    if world_size in _TORUS_BROKEN and not os.environ.get("MOCHI_ALLOW_TP8"):
        raise ValueError(
            f"world_size={world_size} divides Mochi's dims but does NOT form a "
            f"valid collective group on the trn2 4x4 NeuronLink torus: an "
            f"{world_size}-rank group yields 'no_hier no_mesh' at "
            f"init_process_group (measured). Valid TP on this topology is "
            f"{VALID_WORLD_SIZES}; TP=16 works collectively but needs the 24 "
            f"heads padded to 32. Note that raising TP would not relieve the "
            f"long-sequence memory ceiling anyway -- the OOMing tensor is the "
            f"full-width residual-stream norm activation, which is replicated "
            f"across head-parallel ranks, not sharded. See NOTES.md. Set "
            f"MOCHI_ALLOW_TP8=1 to override on a differently wired instance."
        )


def local_heads(world_size: int) -> int:
    return N_HEADS // world_size


def mochi_tp_plan(world_size: int) -> dict:
    """`parallelize_module` plan keyed by submodule path.

    Colwise on every projection that produces the attention/FF inner dim,
    Rowwise on every projection that consumes it. Both default to
    `use_local_output=True`, so intermediate tensors are plain local
    tensors with per-rank shapes -- which is why `apply_tp_fixes` must
    patch `attn.heads`.
    """
    validate_world_size(world_size)
    from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

    plan: dict[str, object] = {}
    for layer in range(N_LAYERS):
        prefix = f"transformer_blocks.{layer}"
        is_last = layer == CONTEXT_PRE_ONLY_LAYER

        # Visual QKV and text QKV both produce inner_dim -> column shard.
        for proj in ("to_q", "to_k", "to_v",
                     "add_q_proj", "add_k_proj", "add_v_proj"):
            plan[f"{prefix}.attn1.{proj}"] = ColwiseParallel()

        # Output projections consume inner_dim -> row shard + all-reduce.
        plan[f"{prefix}.attn1.to_out.0"] = RowwiseParallel()
        if not is_last:
            plan[f"{prefix}.attn1.to_add_out"] = RowwiseParallel()

        # SwiGLU FFN. net.0.proj is a *fused* [value|gate] projection; the
        # meta loader compensates with a permuted shard so that the local
        # chunk(2) still splits value from gate. See mochi_meta_loader.
        plan[f"{prefix}.ff.net.0.proj"] = ColwiseParallel()
        plan[f"{prefix}.ff.net.2"] = RowwiseParallel()
        if not is_last:
            plan[f"{prefix}.ff_context.net.0.proj"] = ColwiseParallel()
            plan[f"{prefix}.ff_context.net.2"] = RowwiseParallel()

    return plan


def _is_sharded(linear) -> bool:
    """True iff `linear.weight` is a Shard-placed DTensor."""
    if linear is None or not hasattr(linear, "weight"):
        return False
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return False
    w = linear.weight
    if isinstance(w, DTensor):
        return any(p.__class__.__name__ == "Shard" for p in w.placements)
    return False


def apply_tp_fixes(model, world_size: int, rank: int = 0, verbose: bool = True) -> int:
    """Patch `attn.heads` to this rank's head count.

    After ColwiseParallel, `to_q(hidden_states)` returns `inner_dim/N`
    features. The processor then does
    `unflatten(2, (attn.heads, -1))`, so `attn.heads` must be the *local*
    head count or the inferred head_dim comes out wrong (silently -- the
    unflatten still succeeds, it just produces garbage groupings).

    Returns the number of attention modules patched.
    """
    if world_size == 1:
        if verbose and rank == 0:
            print("[mochi_tp] world_size=1, no head patching needed", flush=True)
        return 0

    new_heads = local_heads(world_size)
    patched = 0
    for layer in range(N_LAYERS):
        attn = getattr(model.transformer_blocks[layer], "attn1", None)
        if attn is None:
            continue
        if _is_sharded(getattr(attn, "to_q", None)):
            attn.heads = new_heads
            patched += 1

    if verbose and rank == 0:
        print(
            f"[mochi_tp] patched attn.heads -> {new_heads} on {patched}/"
            f"{N_LAYERS} blocks (world_size={world_size})",
            flush=True,
        )
        if patched != N_LAYERS:
            print(
                f"[mochi_tp] WARNING: expected {N_LAYERS} sharded blocks, saw "
                f"{patched}. Check that parallelize_module ran first.",
                flush=True,
            )
    return patched


def shard_pos_frequencies(
    model,
    rank: int,
    world_size: int,
    device: torch.device | str = "cpu",
    verbose: bool = True,
) -> None:
    """Shard `pos_frequencies` `[3, 24, 64]` on its head axis.

    This is how each rank gets the RoPE frequencies for *its* heads. Done
    at load time on a plain parameter, so no traced-rank machinery is
    needed. Also stashes a CPU fp32 copy on the model for
    `patch_rope_cpu_precompute` to build the cos/sin tables from without a
    device-to-host copy.
    """
    pf = model.pos_frequencies
    if getattr(pf, "is_meta", False):
        raise RuntimeError(
            "shard_pos_frequencies called before pos_frequencies was "
            "materialised; run the weight loader first."
        )

    full = pf.detach().to("cpu", torch.float32)
    if full.shape[1] != N_HEADS:
        raise ValueError(
            f"expected pos_frequencies head axis {N_HEADS}, got {tuple(full.shape)}"
        )

    if world_size > 1:
        per = N_HEADS // world_size
        local = full.narrow(1, rank * per, per).contiguous()
    else:
        local = full.contiguous()

    # CPU fp32 master copy, already sharded -- the RoPE tables are built
    # from this so the per-step forward never touches the device parameter.
    model._mochi_cpu_pos_frequencies = local.clone()
    model.pos_frequencies = torch.nn.Parameter(
        local.to(device=device, dtype=pf.dtype if pf.dtype != torch.float32 else local.dtype),
        requires_grad=False,
    )

    if verbose and rank == 0:
        print(
            f"[mochi_tp] pos_frequencies sharded {tuple(full.shape)} -> "
            f"{tuple(local.shape)} (heads {N_HEADS} -> {local.shape[1]})",
            flush=True,
        )


def patch_rope_cpu_precompute(
    model,
    rope_dtype: torch.dtype = torch.float32,
    rank: int = 0,
    verbose: bool = True,
) -> None:
    """Compute the RoPE cos/sin tables on CPU once, then cache on device.

    Upstream `MochiRoPE.forward` runs `torch.arange`, `torch.linspace`,
    `torch.meshgrid`, an `einsum`, and `torch.autocast(device.type, fp32)`
    on whatever device the latents are on. Two problems on Neuron:

    * `torch.autocast("neuron", ...)` is not a registered autocast backend,
      so it either errors or silently no-ops.
    * The whole grid is a pure function of `(num_frames, height, width)` and
      a frozen parameter, so recomputing it inside the denoising loop puts
      a pile of shape-construction ops in the compiled graph for nothing.

    We build the tables on CPU in fp32 (upstream's exact math, so numerics
    match), move them to the device once, and memoise per geometry.

    Args:
        rope_dtype: dtype of the cos/sin tables handed to the model.
            fp32 matches upstream exactly. If device testing shows fp32
            tensors misbehaving across the compile boundary -- the failure
            mode LTX-2 fix #5/#8 addressed -- retry with `torch.bfloat16`.
            This is the first knob to try if output is structured but wrong.
    """
    rope = model.rope
    cpu_pf = getattr(model, "_mochi_cpu_pos_frequencies", None)
    if cpu_pf is None:
        cpu_pf = model.pos_frequencies.detach().to("cpu", torch.float32)

    cache: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    def _forward(pos_frequencies, num_frames, height, width, device=None, dtype=None):
        # `pos_frequencies` (a device tensor) is intentionally ignored in
        # favour of the CPU master copy; they hold identical values.
        key = (int(num_frames), int(height), int(width), str(device), rope_dtype)
        cached = cache.get(key)
        if cached is None:
            pos = rope._get_positions(
                num_frames, height, width,
                device=torch.device("cpu"), dtype=torch.float32,
            )
            freqs = torch.einsum("nd,dhf->nhf", pos, cpu_pf)
            cos = torch.cos(freqs).to(rope_dtype)
            sin = torch.sin(freqs).to(rope_dtype)
            if device is not None:
                cos = cos.to(device)
                sin = sin.to(device)
            cached = (cos, sin)
            cache[key] = cached
        return cached

    rope.forward = _forward
    model._mochi_rope_cache = cache

    if verbose and rank == 0:
        print(
            f"[mochi_tp] RoPE precompute patched: CPU fp32 grid, cached per "
            f"geometry, emitted as {rope_dtype}",
            flush=True,
        )


def estimate_rank_weight_bytes(world_size: int, dtype_bytes: int = 2) -> dict:
    """Per-rank transformer weight footprint under this plan.

    Exists because the naive "20 GB / TP" estimate is wrong for Mochi: the
    AdaLN modulation layers are replicated, so they do not shrink with TP.

    Checks divisibility only, not collective topology -- this is illustrative
    arithmetic, so it will happily estimate TP=8 or TP=16 to show the AdaLN
    replication effect even though `validate_world_size` refuses to actually
    run them on the trn2 torus.
    """
    bad = {n: d for n, d in _DIVISIBILITY.items() if d % world_size != 0}
    if bad:
        raise ValueError(
            f"world_size={world_size} does not evenly divide {bad}."
        )

    per_block_sharded = (
        3 * INNER_DIM * INNER_DIM          # to_q/k/v
        + 3 * INNER_DIM * TEXT_DIM         # add_{q,k,v}_proj
        + INNER_DIM * INNER_DIM            # to_out.0
        + 2 * INNER_DIM * FF_INNER         # ff.net.0.proj (fused: value+gate)
        + FF_INNER * INNER_DIM             # ff.net.2
    )
    per_block_sharded_ctx = (
        INNER_DIM * TEXT_DIM               # to_add_out
        + 2 * TEXT_DIM * FF_CONTEXT_INNER  # ff_context.net.0.proj
        + FF_CONTEXT_INNER * TEXT_DIM      # ff_context.net.2
    )
    per_block_replicated = (
        4 * INNER_DIM * INNER_DIM          # norm1.linear (3072 -> 12288)
        + 4 * INNER_DIM * TEXT_DIM         # norm1_context.linear (3072 -> 6144)
        + 4 * HEAD_DIM                     # qk norms
    )

    n_ctx_blocks = N_LAYERS - 1  # block 47 has no context path
    sharded = N_LAYERS * per_block_sharded + n_ctx_blocks * per_block_sharded_ctx
    replicated = N_LAYERS * per_block_replicated

    # Top-level: patch_embed, time_embed (incl. the 63M attention pooler),
    # norm_out, proj_out -- all replicated.
    top_level = (
        3072 * 12 * 2 * 2 + 3072           # patch_embed
        + 3072 * 256 + 3072                # timestep_embedder.linear_1
        + 3072 * 3072 + 3072               # timestep_embedder.linear_2
        + 8192 * 4096 + 8192               # pooler.to_kv
        + 4096 * 4096 + 4096               # pooler.to_q
        + 3072 * 4096 + 3072               # pooler.to_out
        + 1536 * 4096 + 1536               # caption_proj
        + 6144 * 3072 + 6144               # norm_out.linear
        + 48 * 3072 + 48                   # proj_out
        + 3 * N_HEADS * (HEAD_DIM // 2)    # pos_frequencies
    )

    total = sharded + replicated + top_level
    per_rank = sharded // world_size + replicated + top_level
    return {
        "world_size": world_size,
        "total_params": total,
        "total_gb": total * dtype_bytes / 1e9,
        "per_rank_params": per_rank,
        "per_rank_gb": per_rank * dtype_bytes / 1e9,
        "replicated_params": replicated + top_level,
        "replicated_frac": (replicated + top_level) / total,
    }


def visual_token_count(num_frames: int, height: int, width: int) -> int:
    """Visual tokens for a given output geometry.

    VAE compresses 8x8 spatially and 6x temporally; the DiT then applies a
    patch_size=2 embedding. 480x848x163 gives 44,520, matching the figure
    published on the model card -- a useful check that this arithmetic is
    right.
    """
    latent_frames = (num_frames - 1) // 6 + 1
    latent_h = height // 8
    latent_w = width // 8
    return latent_frames * (latent_h // 2) * (latent_w // 2)


def print_plan_summary(world_size: int) -> None:
    est = estimate_rank_weight_bytes(world_size)
    print(
        f"[mochi_tp] TP={world_size}: {est['total_gb']:.2f} GB total -> "
        f"{est['per_rank_gb']:.2f} GB/rank "
        f"({est['replicated_frac']*100:.0f}% replicated, "
        f"local_heads={local_heads(world_size)})",
        flush=True,
    )
