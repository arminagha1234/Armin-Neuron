# FLUX.2-klein-4B 4 MP — Fix Plan (think-hard, cross-repo)

Written after deep-diving the NxDI FLUX implementation, the dedicated
FLUX NKI flash kernel, and the Qwen-Image-Edit MMDiT contrib. This plan
turns the 503 s / corrupted 4 MP result into a credible customer story.

## The single most important realization

**We never tested the winning combination: `attention_cte` + v2 plan.**
→ **NOW TESTED (2026-06-15):** v2 + attention_cte at 1024² TP=2 gives
**std=18.13 (correct!) at 59.7 s** — 36% faster than manual flash's 93 s.
So the winning combo is confirmed correct. BUT see the 4 MP caveat below.

### 4 MP caveat discovered after (critical)

v2 + attention_cte at **2048² (4 MP) TP=2 does NOT finish** — each step
takes 12-18 min and it never gets past step 2/4. Root cause: **v2 SKIPS
sharding the 20 single-stream blocks.** At 1 MP that's fine (they fit),
but at 4 MP each unsharded single-stream block runs full 66K-token
attention on every rank → enormous compile + no activation-memory relief
where it's needed most. **=> v2 is a 1-MP correctness fix, NOT a 4-MP
solution. The single-stream blocks MUST be sharded for 4 MP, using
NxDI's two-proj_out + one-all-reduce pattern (Phase 2).**



| TP plan | Attention compute | Tested | Result |
|---|---|---|---|
| v1 (shards SwiGLU FFN) | attention_cte[2] | ✅ | std=45 ❌ (SwiGLU bug) |
| v1 (shards SwiGLU FFN) | manual flash | ✅ | std=45 ❌ (SwiGLU bug) |
| v2 (attention-only) | manual flash (Python) | ✅ | **std=18.12 ✅** but 93 s (slow) |
| **v2 (attention-only)** | **attention_cte[2]** | ✅ | **1024²: std=18.13 ✅ 59.7 s. 4MP: does NOT finish (single-stream wall)** |

Why this should work:
- v2 fixed the SwiGLU corruption (proven: manual flash → std 18.12)
- `attention_cte[2]` is the **same kernel NxDI uses in production** and
  it's called the RIGHT way (whole `[B*H,S,D]` in ONE call, NOT the
  per-head Python loop that stalled `nki_flash_attn_flux.py`)
- Confirmed by reading NxDI `attention_wrapper_sharded_without_swap`:
  ```python
  q = query.reshape((bs * n_head, q_len, d_head))   # one call
  attn_output = attention_cte[2](q, k, v, scale,
      causal_mask=False, tp_q=True, tp_k=True, tp_out=False)
  ```
  Our `flux2_attention_cte.py` already does exactly this. The std=45 was
  100% the SwiGLU FFN sharding, NOT the kernel.

**=> First action next session: run v2 plan + attention_cte at 1024² TP=2.
Expect std~18 and a big speedup over the 93 s manual-flash number.**

## Phase 0 — confirm the winning combo (½ day)

1. `run_flux2_tp_v2.py` currently imports `flux2_attention_manual_flash`.
   Make `run_flux2_tp_v2_cte.py` that imports `flux2_attention_cte`
   instead (the CTE installer already exists and is correct).
2. Run TP=2 @ 1024². Expect: std~18, warm << 93 s (probably 10-25 s).
3. If correct + fast → we have a shippable TP=2 path immediately.
4. Then TP=2 @ 2048² (4 MP). This is the headline number.

## Phase 1 — fix TP=4 corruption (1 day)

v2 works at TP=2 but corrupts at TP=4 (std=4.99). Root cause unknown.
Approach:

1. **Per-block tensor-norm bisection.** Add a debug hook that captures
   the L2 norm of each block's output at TP=2 and TP=4 for the same
   seed/input. The first block where TP=4 norm diverges from TP=2 is the
   culprit.
2. **Most likely suspects:**
   - The encoder+image token concat: with 24 heads/4 = 6 heads/rank, the
     joint sequence concat may misalign. Check `to_add_out` RowwiseParallel.
   - `inner_dim` not divisible cleanly: inner_dim=3072, /4 = 768 per rank,
     768/head_dim(128) = 6 heads. Divides fine, so not this.
   - The single-stream blocks are SKIPPED entirely in v2 (not sharded).
     At TP=4 the unsharded single-stream may be the divergence if the
     DTensor placement of its inputs is wrong. **Borrow NxDI's
     single-stream sharding** (next phase) instead of skipping.

## Phase 2 — adopt NxDI's single-stream sharding (1-2 days)

The single-stream blocks (20 of them) are the bulk of the compute and we
currently DON'T shard them (v2 skips them). NxDI shows exactly how:

`NeuronFluxSingleTransformerBlock` (modeling_flux.py:535):
```python
self.proj_mlp     = ColumnParallelLinear(dim, mlp_hidden_dim, gather_output=False)
self.proj_out_attn = RowParallelLinear(dim, dim, input_is_parallel=True,
                                       reduce_output=False, skip_bias_add=True)
self.proj_out_mlp  = RowParallelLinear(mlp_hidden_dim, dim, bias=False,
                                       input_is_parallel=True, reduce_output=False)
# forward:
mlp_hidden = act_mlp(proj_mlp(norm_hidden))      # column-parallel MLP in
attn_out   = attn(norm_hidden)                    # sharded attention
out_attn, bias = proj_out_attn(attn_out)          # row-parallel, no reduce
out_mlp        = proj_out_mlp(mlp_hidden)          # row-parallel, no reduce
proj_out = reduce_from_tensor_model_parallel_region(out_attn + out_mlp)  # ONE all-reduce
hidden = gate * (proj_out + bias)
```

Key tricks:
- **Two separate `proj_out` (attn + mlp), `reduce_output=False`, summed,
  then ONE all-reduce.** Avoids the all-gather after QKV AND merges two
  all-reduces into one. This is the clean MMDiT single-stream shard.
- **bias added AFTER the all-reduce** (correctness).

For FLUX.2-klein (SwiGLU, not GELU), the analog:
- `to_qkv_mlp_proj` is fused `[3*inner_dim ; mlp_hidden*2]`. Split it
  into a QKV ColumnParallel and a **SwiGLU-aware** MLP shard:
  shard `gate` and `value` projections SEPARATELY as two
  ColumnParallelLinears (each `[dim -> mlp_hidden]`), so each rank gets a
  consistent (gate_i, value_i) pair. Then `silu(gate_i)*value_i` is
  correct per-rank, and the down-proj is RowParallel.
- This is the proper fix for ROOT CAUSE #1 that lets us shard the FFN
  too (instead of replicating it), which is needed for real 4 MP speed.

## Phase 3 — the batched FLUX NKI flash kernel (1-2 days)

We have a purpose-built kernel: `.tmp/autocomp/sols/trn-advanced-nki1/9_flux_attn_ref.py`.
It loops `for head_idx in nl.affine_range(num_heads)` **inside** the
kernel (device-side, no per-head recompile — fixes ROOT CAUSE #3).

Plan:
1. Wrap `9_flux_attn_ref.solution` with `@nki_op` / `wrap_nki` taking
   `[B, H, S, D]` directly (kernel signature wants q `(H,S,D)`,
   k `(H,D,S)` transposed, v `(H,S,D)`).
2. Validate numerics vs the manual flash (must hit std~18).
3. Compare speed vs `attention_cte[2]`. The autocomp ref is a starting
   point; the optimized autocomp output (if any) or the production
   `attention_cte[2]` may be faster. Use whichever wins on the FLUX.2
   shape (seq~4608 @ 1 MP, ~66K @ 4 MP).
4. NOTE: at 4 MP the seq is ~66K — the kernel's flash sectioning
   (`_FLASH_ATTENTION_SECTION_LENGTH = 8K`) matters. `attention_cte`
   already handles >10K via flash sections (per its docstring). The
   autocomp ref loads full K/V per tile (`LARGE_TILE_SZ = seq_len`),
   which will OOM at 66K — so **at 4 MP, `attention_cte[2]` is the right
   kernel** (it flash-sections), and the autocomp ref is only good up to
   ~8-10K seq (≈1-2 MP).

## Phase 4 — context/sequence parallelism for 4 MP (2-3 days)

TP alone shards HEADS (24→6 at TP=4). But attention activation is
O(seq²) and seq doesn't shard under pure TP. NxDI's FLUX has a
**context-parallel** path:
`attention_wrapper_context_parallel_single_transformer` (line 140) +
`global_cp_deg` / `cp_offset` args in `attention_cte`.

Context parallel (CP) shards the SEQUENCE across ranks — each rank does
attention for its slice of Q with full K/V (gathered). This is what
actually unlocks 4 MP at low latency, because it cuts the seq² activation
per rank. The `attention_cte` kernel already supports CP natively
(`global_cp_deg > 1, cp_offset`).

Plan: adopt NxDI's CP attention wrapper. This is the highest-leverage
item for 4 MP latency. Combine CP (shard seq) with TP (shard heads) for
maximum activation reduction.

## Phase 5 — FP8 weights (1 day)

`black-forest-labs/FLUX.2-klein-4b-fp8` exists. FP8 weights halve HBM,
freeing room for bigger activations (helps 4 MP fit at lower TP). Adapt
the FP8 loader from the GLM 5.1 / Qwen3-235B FP8 work
(`vllm-omni-beta1/Qwen3-235B-FP8-PR1987/weight_loaders_fp8.py`).

## Recommended order of attack (next session)

1. ~~**Phase 0** — v2 + attention_cte~~ ✅ **DONE.** Correct at 1024²
   (std 18.13, 59.7 s) but **does NOT finish at 4 MP** because the 20
   single-stream blocks are unsharded and run full 66K attention per
   rank. **This makes Phase 2 the critical path, not optional.**
2. **Phase 2 (NOW #1 PRIORITY)** (1-2 days) — shard the single-stream
   blocks using NxDI's two-proj_out + one-all-reduce pattern. This is
   THE blocker for 4 MP. Without it, single-stream attention at 66K
   tokens never compiles in reasonable time.
3. **Phase 4** (2-3 days) — context parallelism (shard the sequence, not
   just heads). The real 4 MP latency unlock — `attention_cte` already
   supports it (`global_cp_deg`, `cp_offset`).
4. **Phase 5** (1 day) — FP8 weights for memory headroom.
5. **Phase 1/3** — TP=4 debug + kernel choice as needed.

Realistic outcome after Phases 2+4: **4 MP at ~15-40 s on trn2 TP=4**,
correct output, which combined with 8 concurrent pipelines per box is a
genuine $/image story vs A100/H100 for batch workloads.

## The cross-repo reference index (so we don't re-find these)

| What | Where |
|---|---|
| **Production FLUX TP model** (the template) | `neuron/external/pr-117-nxdi-diffusion-models/src/neuronx_distributed_inference/models/diffusers/flux/modeling_flux.py` |
| Sharded attention wrapper (the right `attention_cte[2]` call) | same file, `attention_wrapper_sharded_without_swap` L82 |
| **Context-parallel attention** (seq sharding for 4 MP) | same file, `attention_wrapper_context_parallel_single_transformer` L140 |
| Single-stream block TP (2 proj_out, 1 all-reduce) | same file, `NeuronFluxSingleTransformerBlock` L535 |
| FFN/GELU column+row parallel | same file, `NeuronFeedForward` L762 |
| **Batched FLUX flash NKI kernel** (device-side head loop) | `.tmp/autocomp/sols/trn-advanced-nki1/9_flux_attn_ref.py` |
| Qwen-Image-Edit MMDiT CTE attention (another worked example) | `neuron/external/pr-117-nxdi-diffusion-models/contrib/models/Qwen-Image-Edit/src/attention_cte_qie_hoisted_q.py` |
| `attention_cte` kernel source + CP/flash-section docs | `nkilib/core/attention/attention_cte.py` (in the container) |
| Our working v2 TP plan (attention-only, correct @ TP=2) | `/mnt/data/work/flux2_latest/flux2_tp_plan_v2.py` |
| FP8 weight loader pattern | `neuron/examples/vllm-omni-beta1/Qwen3-235B-FP8-PR1987/weight_loaders_fp8.py` |
