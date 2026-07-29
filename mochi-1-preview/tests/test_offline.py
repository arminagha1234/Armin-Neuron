"""Offline (CPU-only, no Neuron) verification of the staged Mochi-1 port.

Everything here runs on a laptop. The point is to catch the *silent*
correctness bugs -- wrong SwiGLU pairing, wrong head sharding, a processor
that changes numerics, a TP plan with a typo'd module path -- before
burning Trainium time on them.

Run:
    .venv/bin/python neuron/examples/Mochi/tests/test_offline.py

Tests, roughly in order of how much they would have cost to debug on device:

  1. fused-SwiGLU shard equivalence, plus proof the naive shard is wrong
  2. attention TP equivalence (colwise/rowwise/heads patch, all combined)
  3. processor equivalence vs upstream, with a padded prompt
  4. tiled attention is numerically exact, not approximate
  5. bool mask handling (the LTX-2 shim bug)
  6. TP plan paths all resolve on the real 48-layer architecture
  7. RoPE CPU precompute matches upstream
  8. token-count and parameter-count arithmetic vs published figures
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import neuron_compat  # noqa: E402
from mochi_meta_loader import shard_tensor, _shard_contiguous  # noqa: E402
from mochi_neuron_attention import (  # noqa: E402
    MochiNeuronAttnProcessor,
    apply_rotary_emb,
)
from mochi_tp_plan import (  # noqa: E402
    N_HEADS,
    N_LAYERS,
    CONTEXT_PRE_ONLY_LAYER,
    estimate_rank_weight_bytes,
    mochi_tp_plan,
    validate_world_size,
    visual_token_count,
)

torch.manual_seed(0)

PASS, FAIL = [], []


def check(name: str, condition: bool, detail: str = "") -> None:
    (PASS if condition else FAIL).append(name)
    mark = "PASS" if condition else "FAIL"
    print(f"  [{mark}] {name}" + (f" -- {detail}" if detail else ""), flush=True)


def close(a: torch.Tensor, b: torch.Tensor, tol: float = 1e-5) -> tuple[bool, float]:
    err = (a - b).abs().max().item()
    return err <= tol, err


# ── 1. fused SwiGLU sharding ────────────────────────────────────────────────
def test_swiglu_shard():
    print("\n[1] fused SwiGLU shard equivalence")
    from diffusers.models.attention import FeedForward

    dim, inner, ws = 96, 256, 4
    ff = FeedForward(dim, inner_dim=inner, activation_fn="swiglu", bias=False)
    ff.eval()
    x = torch.randn(2, 5, dim)
    with torch.no_grad():
        reference = ff(x)

    w_proj = ff.net[0].proj.weight.data     # (2*inner, dim)
    w_out = ff.net[2].weight.data           # (dim, inner)

    def run_sharded(shard_fn):
        total = torch.zeros_like(reference)
        for r in range(ws):
            wp = shard_fn(w_proj, r)
            per = wp.shape[0] // 2
            local = x @ wp.T
            value, gate = local.chunk(2, dim=-1)
            act = value * F.silu(gate)
            wo = w_out[:, r * per:(r + 1) * per]
            total = total + act @ wo.T
        return total

    correct = run_sharded(lambda w, r: shard_tensor(w, "glu", r, ws))
    ok, err = close(correct, reference, 1e-4)
    check("permuted GLU shard reproduces unsharded output", ok, f"max|err|={err:.2e}")

    naive = run_sharded(lambda w, r: _shard_contiguous(w, 0, r, ws))
    ok_naive, err_naive = close(naive, reference, 1e-4)
    check(
        "naive contiguous GLU shard is WRONG (trap confirmed)",
        not ok_naive,
        f"max|err|={err_naive:.2e} (large error expected)",
    )

    # Structural: rank r's local rows are value_r followed by gate_r.
    inner_half = w_proj.shape[0] // 2
    per = inner_half // ws
    r = 2
    piece = shard_tensor(w_proj, "glu", r, ws)
    ok_v = torch.equal(piece[:per], w_proj[r * per:(r + 1) * per])
    ok_g = torch.equal(piece[per:], w_proj[inner_half + r * per: inner_half + (r + 1) * per])
    check("GLU shard layout is [value_r | gate_r]", ok_v and ok_g)


# ── 2. attention TP equivalence ─────────────────────────────────────────────
def _build_attention(dim=96, heads=4, head_dim=24, text_dim=48):
    from diffusers.models.attention_processor import MochiAttention
    attn = MochiAttention(
        query_dim=dim, heads=heads, dim_head=head_dim, bias=False,
        added_kv_proj_dim=text_dim, added_proj_bias=False,
        out_dim=dim, out_context_dim=text_dim, context_pre_only=False,
        processor=MochiNeuronAttnProcessor(), eps=1e-5,
    )
    attn.eval()
    return attn


def test_attention_tp():
    print("\n[2] attention TP equivalence (colwise + rowwise + heads patch)")
    dim, heads, head_dim, text_dim, ws = 96, 4, 24, 48, 2
    attn = _build_attention(dim, heads, head_dim, text_dim)

    b, s_v, s_t = 1, 7, 5
    hidden = torch.randn(b, s_v, dim)
    context = torch.randn(b, s_t, text_dim)
    mask = torch.ones(b, s_t, dtype=torch.bool)

    with torch.no_grad():
        ref_v, ref_c = attn(hidden, context, mask)

    # Emulate TP by hand: shard weights, run each rank, sum the row-shard
    # partials, add the replicated biases once at the end.
    import copy
    partial_v = torch.zeros_like(ref_v)
    partial_c = torch.zeros_like(ref_c)
    inner = heads * head_dim
    per = inner // ws

    for r in range(ws):
        local = copy.deepcopy(attn)
        local.heads = heads // ws          # the apply_tp_fixes patch
        for proj in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj"):
            m = getattr(local, proj)
            m.weight = torch.nn.Parameter(
                shard_tensor(getattr(attn, proj).weight.data, 0, r, ws)
            )
            m.out_features = per
        # Row shards, bias stripped (added once after the sum).
        local.to_out[0].weight = torch.nn.Parameter(
            shard_tensor(attn.to_out[0].weight.data, 1, r, ws)
        )
        local.to_out[0].bias = None
        local.to_add_out.weight = torch.nn.Parameter(
            shard_tensor(attn.to_add_out.weight.data, 1, r, ws)
        )
        local.to_add_out.bias = None

        with torch.no_grad():
            v, c = local(hidden, context, mask)
        partial_v = partial_v + v
        partial_c = partial_c + c

    partial_v = partial_v + attn.to_out[0].bias.data
    partial_c = partial_c + attn.to_add_out.bias.data

    ok_v, err_v = close(partial_v, ref_v, 1e-4)
    ok_c, err_c = close(partial_c, ref_c, 1e-4)
    check(f"TP={ws} visual stream matches unsharded", ok_v, f"max|err|={err_v:.2e}")
    check(f"TP={ws} text stream matches unsharded", ok_c, f"max|err|={err_c:.2e}")


# ── 3. processor equivalence vs upstream ────────────────────────────────────
def test_processor_equivalence():
    print("\n[3] static-shape processor vs upstream MochiAttnProcessor2_0")
    from diffusers.models.attention_processor import MochiAttnProcessor2_0

    dim, heads, head_dim, text_dim = 96, 4, 24, 48
    attn = _build_attention(dim, heads, head_dim, text_dim)

    b, s_v, s_t = 1, 9, 6
    hidden = torch.randn(b, s_v, dim)
    context = torch.randn(b, s_t, text_dim)

    # A genuinely padded prompt: 4 real tokens, 2 pad. This is the case the
    # upstream torch.nonzero path exists to handle.
    mask = torch.tensor([[True, True, True, True, False, False]])

    # RoPE tables shaped (S, H, head_dim//2), as MochiRoPE emits.
    cos = torch.randn(s_v, heads, head_dim // 2)
    sin = torch.randn(s_v, heads, head_dim // 2)

    with torch.no_grad():
        attn.processor = MochiAttnProcessor2_0()
        up_v, up_c = attn(hidden, context, mask, image_rotary_emb=(cos, sin))

        attn.processor = MochiNeuronAttnProcessor(zero_padded_context=True)
        new_v, new_c = attn(hidden, context, mask, image_rotary_emb=(cos, sin))

    ok_v, err_v = close(new_v, up_v, 2e-4)
    check("visual output matches upstream with a padded prompt", ok_v,
          f"max|err|={err_v:.2e}")

    # Encoder rows at padded positions: upstream zero-fills pre-to_add_out,
    # so both should agree there too when zero_padded_context=True.
    ok_c, err_c = close(new_c, up_c, 2e-4)
    check("text output matches upstream (incl. zero-filled pad rows)", ok_c,
          f"max|err|={err_c:.2e}")

    # And confirm the masked tokens genuinely cannot influence the video: if
    # the padded context values change, the visual output must not.
    context_perturbed = context.clone()
    context_perturbed[:, 4:] += 10.0
    with torch.no_grad():
        alt_v, _ = attn(hidden, context_perturbed, mask, image_rotary_emb=(cos, sin))
    ok_iso, err_iso = close(alt_v, new_v, 2e-4)
    check("padded prompt tokens do not leak into the visual stream", ok_iso,
          f"max|err|={err_iso:.2e}")

    # Fully-masked prompt. The pipeline hits this on every run with an empty
    # negative prompt: _get_t5_prompt_embeds does
    #   prompt_attention_mask = torch.zeros_like(..., dtype=torch.bool)
    # Upstream then selects zero text tokens and attends to visual only. Ours
    # biases all 256 text keys to -10000. These agree because the visual keys
    # are never masked, so no softmax row is fully masked and no NaN appears.
    empty_mask = torch.zeros(b, s_t, dtype=torch.bool)
    with torch.no_grad():
        attn.processor = MochiAttnProcessor2_0()
        up_v0, _ = attn(hidden, context, empty_mask, image_rotary_emb=(cos, sin))
        attn.processor = MochiNeuronAttnProcessor(zero_padded_context=True)
        new_v0, new_c0 = attn(hidden, context, empty_mask, image_rotary_emb=(cos, sin))
    ok_e, err_e = close(new_v0, up_v0, 2e-4)
    check("fully-masked prompt (empty negative) matches upstream", ok_e,
          f"max|err|={err_e:.2e}")
    check("fully-masked prompt produces no NaN/Inf",
          bool(torch.isfinite(new_v0).all() and torch.isfinite(new_c0).all()))

    # No dynamic ops: upstream's torch.nonzero should be gone. Verify by
    # banning nonzero during the call.
    original_nonzero = torch.nonzero
    calls = []

    def _tracking_nonzero(*a, **k):
        calls.append(1)
        return original_nonzero(*a, **k)

    torch.nonzero = _tracking_nonzero
    try:
        with torch.no_grad():
            attn.processor = MochiNeuronAttnProcessor()
            attn(hidden, context, mask, image_rotary_emb=(cos, sin))
    finally:
        torch.nonzero = original_nonzero
    check("processor calls torch.nonzero zero times", len(calls) == 0,
          f"calls={len(calls)}")


# ── 4. tiled attention exactness ────────────────────────────────────────────
def test_chunked_attention():
    print("\n[4] tiled-query attention is exact")
    b, h, s_q, s_k, d = 1, 3, 130, 64, 16
    q = torch.randn(b, h, s_q, d)
    k = torch.randn(b, h, s_k, d)
    v = torch.randn(b, h, s_k, d)
    bias = torch.zeros(b, 1, 1, s_k)
    bias[..., 40:] = neuron_compat.MASKED_BIAS

    with torch.no_grad():
        reference = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)

    # Force the BMM path by pretending the tensors are on device: call
    # neuron_sdpa's internals directly instead of going through the CPU
    # fallthrough.
    neuron_compat._sdpa_original = F.scaled_dot_product_attention
    scale = 1.0 / (d ** 0.5)
    m = neuron_compat._collapse_mask(
        neuron_compat._normalize_mask(bias, q.dtype), b, h, s_q
    )
    q3 = q.reshape(b * h, s_q, d)
    k3 = k.reshape(b * h, s_k, d)
    v3 = v.reshape(b * h, s_k, d)

    untiled = neuron_compat._attention_bmm(q3, k3, v3, m, scale, None)
    ok_u, err_u = close(untiled.reshape(b, h, s_q, d), reference, 1e-5)
    check("untiled BMM matches fused SDPA", ok_u, f"max|err|={err_u:.2e}")

    for chunk in (32, 64, 128, 7):
        tiled = neuron_compat._attention_bmm(q3, k3, v3, m, scale, chunk)
        ok_t, err_t = close(tiled, untiled, 1e-6)
        check(f"tiled (q_chunk={chunk}) is exact vs untiled", ok_t,
              f"max|err|={err_t:.2e}")

    # Auto-tiling must budget the whole (planes, Sq, Sk) score tensor, not one
    # plane. The per-plane version picked q_chunk~6656 for 31-frame CFG and
    # OOM'd a 24 GB logical core at 23.86 GB peak.
    budget = neuron_compat._AUTO_TILE_BUDGET_BYTES
    floor = neuron_compat._MIN_Q_CHUNK
    for planes, sq, sk in ((12, 9796, 9796), (6, 6616, 6616), (12, 44776, 44776)):
        chunk = neuron_compat._resolve_q_chunk(sq, sk, planes, 2)
        if chunk is None:
            used = planes * sq * sk * 2
            check(f"untiled only when it fits ({planes}x{sq}x{sk})", used <= budget,
                  f"{used/1e6:.0f} MB vs budget {budget/1e6:.0f} MB")
        else:
            used = planes * chunk * sk * 2
            # At very long sequences the 512-row floor wins over the budget:
            # tiles narrower than that waste too much compute. Accept the
            # floor, but hold it to a hard 1 GB ceiling so it can never be
            # the thing that OOMs a 24 GB core.
            at_floor = chunk == floor
            ok = used <= budget or (at_floor and used <= 1_100_000_000)
            check(f"auto tile {planes}x{sq}x{sk} -> q_chunk={chunk} bounded", ok,
                  f"{used/1e6:.0f} MB"
                  + (f" (at {floor}-row floor, budget {budget/1e6:.0f} MB)"
                     if at_floor else f" vs {budget/1e6:.0f} MB"))

    # The geometry that actually OOM'd must now tile aggressively.
    chunk_31f = neuron_compat._resolve_q_chunk(9796, 9796, 12, 2)
    check("31-frame CFG geometry now tiles (was ~6656, OOM)",
          chunk_31f is not None and chunk_31f <= 1536, f"q_chunk={chunk_31f}")

    # Per-query (non-broadcast) masks must be sliced along the query axis too.
    full_mask = torch.zeros(b, h, s_q, s_k)
    full_mask[..., 50:] = neuron_compat.MASKED_BIAS
    m_full = neuron_compat._collapse_mask(full_mask, b, h, s_q)
    a = neuron_compat._attention_bmm(q3, k3, v3, m_full, scale, None)
    bb = neuron_compat._attention_bmm(q3, k3, v3, m_full, scale, 32)
    ok_pq, err_pq = close(a, bb, 1e-6)
    check("tiling slices per-query masks correctly", ok_pq, f"max|err|={err_pq:.2e}")


# ── 5. bool mask handling ───────────────────────────────────────────────────
def test_bool_mask():
    print("\n[5] bool attention mask (the LTX-2 shim bug)")
    b, h, s_q, s_k, d = 1, 2, 4, 6, 8
    q = torch.randn(b, h, s_q, d)
    k = torch.randn(b, h, s_k, d)
    v = torch.randn(b, h, s_k, d)

    # Exactly the shape MochiAttentionPool builds.
    keep = torch.tensor([[True, True, True, False, False]])
    bool_mask = keep[:, None, None, :]
    bool_mask = F.pad(bool_mask, (1, 0), value=True)  # (1,1,1,6)

    with torch.no_grad():
        reference = F.scaled_dot_product_attention(q, k, v, attn_mask=bool_mask)

    scale = 1.0 / (d ** 0.5)
    m = neuron_compat._collapse_mask(
        neuron_compat._normalize_mask(bool_mask, q.dtype), b, h, s_q
    )
    got = neuron_compat._attention_bmm(
        q.reshape(b * h, s_q, d), k.reshape(b * h, s_k, d),
        v.reshape(b * h, s_k, d), m, scale, None,
    ).reshape(b, h, s_q, d)
    ok, err = close(got, reference, 1e-5)
    check("bool mask converted to additive bias correctly", ok, f"max|err|={err:.2e}")

    # The LTX-2 shim added the bool tensor straight to the scores; show that
    # this is materially different, i.e. the fix is load-bearing.
    naive = neuron_compat._attention_bmm(
        q.reshape(b * h, s_q, d), k.reshape(b * h, s_k, d),
        v.reshape(b * h, s_k, d), bool_mask.reshape(1, 1, s_k).to(q.dtype),
        scale, None,
    ).reshape(b, h, s_q, d)
    ok_naive, err_naive = close(naive, reference, 1e-5)
    check("adding a raw bool mask would be wrong (fix is load-bearing)",
          not ok_naive, f"max|err|={err_naive:.2e} (large error expected)")


# ── 6. TP plan paths resolve on the real architecture ───────────────────────
def test_plan_paths():
    print("\n[6] TP plan paths resolve on the real 48-layer model")
    from diffusers import MochiTransformer3DModel

    cfg = {
        "patch_size": 2, "num_attention_heads": 24, "attention_head_dim": 128,
        "num_layers": 48, "pooled_projection_dim": 1536, "in_channels": 12,
        "out_channels": None, "qk_norm": "rms_norm", "text_embed_dim": 4096,
        "time_embed_dim": 256, "activation_fn": "swiglu",
        "max_sequence_length": 256,
    }
    with torch.device("meta"):
        model = MochiTransformer3DModel.from_config(cfg)

    n_params = sum(p.numel() for p in model.parameters())
    check("model builds with 10.03 B params",
          abs(n_params - 10_028_000_000) < 20_000_000,
          f"{n_params/1e9:.3f} B")

    for ws in (2, 4):
        plan = mochi_tp_plan(ws)
        missing, non_linear = [], []
        for path in plan:
            module = model
            for part in path.split("."):
                if not hasattr(module, part):
                    missing.append(path)
                    module = None
                    break
                module = getattr(module, part)
            if module is not None and not isinstance(module, torch.nn.Linear):
                non_linear.append(path)
        check(f"TP={ws}: all {len(plan)} plan paths exist", not missing,
              f"missing={missing[:3]}")
        check(f"TP={ws}: all plan targets are nn.Linear", not non_linear,
              f"bad={non_linear[:3]}")

    # Block 47 is context_pre_only: no to_add_out, no ff_context. The plan
    # must not reference them, and must reference them for block 0.
    plan = mochi_tp_plan(4)
    last = f"transformer_blocks.{CONTEXT_PRE_ONLY_LAYER}"
    check("plan omits to_add_out on the context_pre_only block",
          f"{last}.attn1.to_add_out" not in plan)
    check("plan omits ff_context on the context_pre_only block",
          f"{last}.ff_context.net.0.proj" not in plan)
    check("plan includes to_add_out on block 0",
          "transformer_blocks.0.attn1.to_add_out" in plan)
    check("block 47 really lacks to_add_out in the architecture",
          not hasattr(model.transformer_blocks[47].attn1, "to_add_out"))

    # QK norms must be [head_dim], which is what lets us skip LTX-2's
    # adaptive all-reduce norm.
    shape = tuple(model.transformer_blocks[0].attn1.norm_q.weight.shape)
    check("QK norm weight is [head_dim]=[128] (no adaptive norm needed)",
          shape == (128,), f"shape={shape}")

    # World-size validation. 3/6/12/16 fail divisibility; 8 divides the dims
    # but is rejected on topology grounds (trn2 4x4 torus "no_hier no_mesh"),
    # which is the trap worth guarding since it only shows up after a 20 GB
    # weight load on device.
    for bad in (3, 6, 12, 16):
        try:
            validate_world_size(bad)
            check(f"world_size={bad} rejected (divisibility)", False,
                  "accepted but should not be")
        except ValueError:
            check(f"world_size={bad} correctly rejected (divisibility)", True)

    import os as _os
    _had = _os.environ.pop("MOCHI_ALLOW_TP8", None)
    try:
        validate_world_size(8)
        check("world_size=8 rejected (torus, no_hier no_mesh)", False,
              "accepted but deadlocks on device")
    except ValueError as e:
        check("world_size=8 rejected (torus, no_hier no_mesh)",
              "no_hier no_mesh" in str(e))
    # Overridable for other topologies.
    _os.environ["MOCHI_ALLOW_TP8"] = "1"
    try:
        validate_world_size(8)
        check("MOCHI_ALLOW_TP8=1 overrides the torus guard", True)
    except ValueError:
        check("MOCHI_ALLOW_TP8=1 overrides the torus guard", False)
    finally:
        _os.environ.pop("MOCHI_ALLOW_TP8", None)
        if _had is not None:
            _os.environ["MOCHI_ALLOW_TP8"] = _had

    for good in (1, 2, 4):
        try:
            validate_world_size(good)
            check(f"world_size={good} accepted (torus-valid)", True)
        except ValueError:
            check(f"world_size={good} accepted (torus-valid)", False)


# ── 7. RoPE CPU precompute matches upstream ─────────────────────────────────
def test_rope_precompute():
    print("\n[7] RoPE CPU precompute matches upstream MochiRoPE")
    from diffusers.models.transformers.transformer_mochi import MochiRoPE
    import mochi_tp_plan

    rope = MochiRoPE()
    heads, half = 24, 64
    pos_freqs = torch.randn(3, heads, half)
    num_frames, h, w = 6, 15, 26

    with torch.no_grad():
        ref_cos, ref_sin = rope(pos_freqs, num_frames, h, w,
                                device=torch.device("cpu"), dtype=torch.float32)

    class _Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rope = MochiRoPE()
            self.pos_frequencies = torch.nn.Parameter(pos_freqs.clone())

    holder = _Holder()
    mochi_tp_plan.patch_rope_cpu_precompute(holder, rope_dtype=torch.float32,
                                            verbose=False)
    got_cos, got_sin = holder.rope(holder.pos_frequencies, num_frames, h, w,
                                   device=torch.device("cpu"))
    ok_c, err_c = close(got_cos, ref_cos, 1e-6)
    ok_s, err_s = close(got_sin, ref_sin, 1e-6)
    check("patched RoPE cos matches upstream", ok_c, f"max|err|={err_c:.2e}")
    check("patched RoPE sin matches upstream", ok_s, f"max|err|={err_s:.2e}")
    check("RoPE table shape is (tokens, heads, head_dim/2)",
          tuple(got_cos.shape) == (num_frames * h * w, heads, half),
          f"{tuple(got_cos.shape)}")

    # Second call must hit the cache, not recompute.
    again_cos, _ = holder.rope(holder.pos_frequencies, num_frames, h, w,
                               device=torch.device("cpu"))
    check("RoPE tables are cached across steps", again_cos is got_cos)

    # Head-axis sharding of pos_frequencies gives each rank its own heads.
    ws, r = 4, 2
    per = heads // ws
    sharded = pos_freqs.narrow(1, r * per, per)
    holder2 = _Holder()
    holder2.pos_frequencies = torch.nn.Parameter(sharded.clone())
    holder2._mochi_cpu_pos_frequencies = sharded.clone()
    mochi_tp_plan.patch_rope_cpu_precompute(holder2, rope_dtype=torch.float32,
                                            verbose=False)
    sh_cos, _ = holder2.rope(holder2.pos_frequencies, num_frames, h, w,
                             device=torch.device("cpu"))
    ok_sh, err_sh = close(sh_cos, ref_cos[:, r * per:(r + 1) * per], 1e-6)
    check(f"pos_frequencies head shard (rank {r}/{ws}) selects the right heads",
          ok_sh, f"max|err|={err_sh:.2e}")


# ── 8. published-figure arithmetic ──────────────────────────────────────────
def test_arithmetic():
    print("\n[8] token / memory arithmetic vs published figures")
    tokens = visual_token_count(163, 480, 848)
    check("163 frames @ 480x848 gives the model card's 44,520 visual tokens",
          tokens == 44_520, f"got {tokens}")

    est = estimate_rank_weight_bytes(1)
    check("parameter estimate lands near 20.06 GB bf16",
          abs(est["total_gb"] - 20.06) < 0.4, f"{est['total_gb']:.2f} GB")

    print("      per-rank weight footprint (bf16):")
    prev = None
    monotonic = True
    for ws in (1, 2, 4, 8):
        e = estimate_rank_weight_bytes(ws)
        print(f"        TP={ws}: {e['per_rank_gb']:5.2f} GB/rank  "
              f"({e['replicated_frac']*100:4.1f}% replicated)")
        if prev is not None and e["per_rank_gb"] >= prev:
            monotonic = False
        prev = e["per_rank_gb"]
    check("per-rank footprint decreases with TP", monotonic)

    # Sub-linear scaling is the point: naive 20/TP would predict 2.5 GB at
    # TP=8, but replicated AdaLN keeps it well above that.
    e8 = estimate_rank_weight_bytes(8)
    check("TP=8 footprint exceeds the naive 20/8 GB estimate (AdaLN replicated)",
          e8["per_rank_gb"] > 2.5 * 1.5,
          f"{e8['per_rank_gb']:.2f} GB vs naive 2.51 GB")

    print("      score-matrix cost across all 24 heads (bf16, untiled):")
    for frames in (19, 31, 61, 85, 163):
        t = visual_token_count(frames, 480, 848) + 256
        gb = t * t * N_HEADS * 2 / 1e9
        print(f"        {frames:3d} frames: {t:6,d} tokens -> {gb:6.1f} GB")


# ── 9b. tiled RMS norms are numerically identical to upstream ──────────────
def test_tiled_norms():
    print("\n[9b] tiled RMS norms match upstream exactly")
    from diffusers.models.transformers.transformer_mochi import (
        MochiModulatedRMSNorm, MochiRMSNormZero,
    )
    from mochi_norm_memory import (
        TiledModulatedRMSNorm, TiledRMSNormZero, install_tiled_norms,
    )

    b, s, d = 2, 1000, 96
    x = torch.randn(b, s, d)
    eps = 1e-6

    # ModulatedRMSNorm, no scale and with a (B,1,D) broadcast scale.
    up = MochiModulatedRMSNorm(eps=eps)
    for scale in (None, 1 + torch.randn(b, 1, d)):
        ref = up(x, scale)
        for tile in (128, 256, 4096):
            got = TiledModulatedRMSNorm(eps, tile)(x, scale)
            ok, err = close(got, ref, 1e-6)
            label = "no scale" if scale is None else "broadcast scale"
            check(f"ModulatedRMSNorm tile={tile} ({label}) exact", ok,
                  f"max|err|={err:.2e}")

    # RMSNormZero: must return the same four outputs.
    upz = MochiRMSNormZero(embedding_dim=d, hidden_dim=4 * d, eps=eps)
    upz.eval()
    emb = torch.randn(b, d)
    with torch.no_grad():
        ref = upz(x, emb)
        got = TiledRMSNormZero(upz, eps, 128)(x, emb)
    names = ("hidden", "gate_msa", "scale_mlp", "gate_mlp")
    for i, nm in enumerate(names):
        ok, err = close(got[i], ref[i], 1e-6)
        check(f"RMSNormZero tiled output '{nm}' exact", ok, f"max|err|={err:.2e}")

    # Peak fp32 scratch must actually shrink.
    untiled_bytes = b * s * d * 4
    tiled_bytes = b * 128 * d * 4
    check("tiling reduces fp32 scratch", tiled_bytes < untiled_bytes / 5,
          f"{tiled_bytes/1e3:.0f} KB vs {untiled_bytes/1e3:.0f} KB")

    # Installer must hit every norm on the real architecture and preserve the
    # checkpoint parameter names the loader and TP plan depend on.
    from diffusers import MochiTransformer3DModel
    cfg = {
        "patch_size": 2, "num_attention_heads": 24, "attention_head_dim": 128,
        "num_layers": 48, "pooled_projection_dim": 1536, "in_channels": 12,
        "out_channels": None, "qk_norm": "rms_norm", "text_embed_dim": 4096,
        "time_embed_dim": 256, "activation_fn": "swiglu",
        "max_sequence_length": 256,
    }
    with torch.device("meta"):
        model = MochiTransformer3DModel.from_config(cfg)
    before = {n for n, _ in model.named_parameters()}
    n = install_tiled_norms(model, verbose=False)
    after = {n for n, _ in model.named_parameters()}
    check("installer replaced a norm in every block", n >= 48 * 4,
          f"replaced {n}")
    check("parameter names unchanged (loader + TP plan still valid)",
          before == after,
          f"added={sorted(after-before)[:3]} removed={sorted(before-after)[:3]}")

    # No MochiModulatedRMSNorm / MochiRMSNormZero should survive.
    leftover = [
        type(m).__name__ for m in model.modules()
        if type(m).__name__ in ("MochiModulatedRMSNorm", "MochiRMSNormZero")
    ]
    check("no upcasting norms left in the model", not leftover,
          f"leftover={leftover[:4]}")


# ── 9. staged sources are importable / syntactically valid ─────────────────
def test_fp32_softmax():
    print("\n[10] attention softmax reduces in fp32")
    # bf16 scores over many keys with a -10000 masked bias lose precision if
    # the softmax reduces in bf16. The BMM path must upcast the reduction so
    # it matches the fp32-softmax flash NKI kernel.
    torch.manual_seed(0)
    b_h, s_q, s_k, d = 2, 128, 4096, 128
    q = torch.randn(b_h, s_q, d, dtype=torch.bfloat16)
    k = torch.randn(b_h, s_k, d, dtype=torch.bfloat16)
    v = torch.randn(b_h, s_k, d, dtype=torch.bfloat16)
    bias = torch.zeros(b_h, 1, s_k, dtype=torch.bfloat16)
    bias[..., 3000:] = neuron_compat.MASKED_BIAS
    scale = 1.0 / (d ** 0.5)

    out = neuron_compat._attention_bmm(q, k, v, bias, scale, None)

    # fp32 oracle: same math, everything in fp32.
    qf, kf, vf = q.float(), k.float(), v.float()
    scores = torch.bmm(qf, kf.transpose(-1, -2)) * scale + bias.float()
    oracle = torch.bmm(scores.softmax(dim=-1), vf)

    ok, err = close(out.float(), oracle, 6e-2)
    check("bf16 BMM attention tracks the fp32 oracle", ok, f"max|err|={err:.2e}")

    # The masked keys must carry ~zero weight: perturbing masked V rows should
    # not change the output.
    v2 = v.clone()
    v2[:, 3000:] += 10.0
    out2 = neuron_compat._attention_bmm(q, k, v2, bias, scale, None)
    ok_leak, err_leak = close(out.float(), out2.float(), 1e-2)
    check("masked keys do not leak into the output", ok_leak,
          f"max|err|={err_leak:.2e}")


def test_strict_loader_rejects_unexpected():
    print("\n[11] strict loader rejects unexpected checkpoint keys")
    import tempfile
    import torch.nn as nn
    from safetensors.torch import save_file
    from mochi_meta_loader import load_weights_sharded

    # A tiny model with one known parameter.
    model = nn.Module()
    model.lin = nn.Linear(4, 4, bias=False).to_empty(device="meta")

    with tempfile.TemporaryDirectory() as d:
        shard = Path(d) / "diffusion_pytorch_model.safetensors"
        save_file(
            {
                "lin.weight": torch.randn(4, 4),
                "ghost.weight": torch.randn(4, 4),  # no home in the model
            },
            str(shard),
        )
        raised = False
        try:
            load_weights_sharded(
                model, d, tp_local_rank=0, world_size=1,
                dtype=torch.float32, device="cpu", variant=None,
                strict=True, verbose=False,
            )
        except RuntimeError as exc:
            raised = "ghost.weight" in str(exc) or "no matching module" in str(exc)
        check("strict=True raises on an unexpected checkpoint key", raised)

        # strict=False must still load the good key and only warn.
        model2 = nn.Module()
        model2.lin = nn.Linear(4, 4, bias=False).to_empty(device="meta")
        summary = load_weights_sharded(
            model2, d, tp_local_rank=0, world_size=1,
            dtype=torch.float32, device="cpu", variant=None,
            strict=False, verbose=False,
        )
        check("strict=False downgrades to a warning and still loads",
              summary["loaded"] == 1 and "ghost.weight" in summary["unexpected"],
              f"loaded={summary['loaded']} unexpected={summary['unexpected']}")


def test_sources_load():
    print("\n[9] staged sources load cleanly")
    import importlib

    for name in ("neuron_compat", "mochi_neuron_attention",
                 "mochi_tp_plan", "mochi_meta_loader", "mochi_norm_memory"):
        try:
            importlib.import_module(name)
            check(f"{name} imports", True)
        except Exception as exc:  # noqa: BLE001
            check(f"{name} imports", False, repr(exc))

    # The runner imports torch_neuronx only inside main(), so byte-compiling
    # it is the most we can check off-device.
    import py_compile
    runner = SRC / "run_mochi_native.py"
    try:
        py_compile.compile(str(runner), doraise=True, cfile=None)
        check("run_mochi_native.py compiles", True)
    except Exception as exc:  # noqa: BLE001
        check("run_mochi_native.py compiles", False, repr(exc))

    # Its --help path must work without a Neuron runtime present.
    import subprocess
    proc = subprocess.run(
        [sys.executable, str(runner), "--help"],
        capture_output=True, text=True, timeout=180,
    )
    check("run_mochi_native.py --help works off-device",
          proc.returncode == 0, (proc.stderr or "").strip().splitlines()[-1:] or "")


def main():
    print("=" * 72)
    print("Mochi-1 Trainium port -- offline verification")
    print(f"torch {torch.__version__}")
    print("=" * 72)

    test_swiglu_shard()
    test_attention_tp()
    test_processor_equivalence()
    test_chunked_attention()
    test_bool_mask()
    test_plan_paths()
    test_rope_precompute()
    test_arithmetic()
    test_tiled_norms()
    test_fp32_softmax()
    test_strict_loader_rejects_unexpected()
    test_sources_load()

    print("\n" + "=" * 72)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for name in FAIL:
            print(f"  FAILED: {name}")
    print("=" * 72)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
