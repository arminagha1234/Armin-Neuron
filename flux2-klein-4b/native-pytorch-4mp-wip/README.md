# FLUX.2-klein-4B — High-Resolution (3MP / 4MP) on Trainium2 — WIP

Native PyTorch, Beta 3 stack. This folder is the work-in-progress effort
to push FLUX.2-klein-4B above the shipped 1 MP (1024²) baseline up to
4 MP (2048²) on a single trn2.48xlarge using tensor parallelism.

## Status (2026-06-15)

| Resolution | Status | Recipe | Quality (std, ref ~18) | Warm |
|---|---|---|---:|---:|
| 1024² (1 MP) | ✅ shipped | single-core bf16 | 18.16 | 4.2 s |
| 1280² (1.6 MP) | ✅ **correct** | fp32 + v3 full-shard, TP=2 | **16.93** | 227 s |
| 1792² (3.2 MP) | ✅ **correct (new)** | fp32 + v3 full-shard, TP=4 | **13.30** | 429 s |
| 2048² (4 MP) | 🟡 **fits + runs, output collapses** | fp32 + v3 full-shard, TP=8 | 2.60 | 579 s |

See `results/` for the generated PNGs and the progress dashboard. The
1.6 MP and 3 MP images are genuine (242 / 222 unique colors); the 4 MP
image collapses to a near-flat field (19 unique colors).

> Warm times are with a **correctness-first pure-Python tile flash**
> attention. They are not optimized — the next step is a batched fp32
> NKI/CTE attention kernel, which should cut these by 10–40×.

## The core problem and the solution

**Problem:** above ~1 MP the all-bf16 DiT collapses (precision loss in
the residual/softmax reductions as token count grows). fp32 fixes
correctness but OOMs above 1.6 MP — because the original TP plan only
sharded attention heads, leaving the 20 single-stream blocks + all FFNs
**replicated** on every core.

**Solution (v3 full sharding):** split the fused SwiGLU FFN
(`linear_in` → `gate_proj` + `value_proj`) and the single-stream fused
`to_qkv_mlp_proj` / `to_out` into separately-shardable linears, then
shard the whole model (not just attention). This drops the per-core fp32
activation ~world_size×, so fp32 (the correct recipe) now FITS:

- 1.6 MP fits at just TP=2 (was TP=4)
- 3 MP fits at TP=4 (was OOM at TP=4 *and* TP=8)
- 4 MP fits + runs end-to-end at TP=8 (was OOM everywhere)

The weight splits are weight-preserving — verified on CPU before any
device run (`src/flux2_v3_selftest.py`, max|Δ| ~1e-6).

## Remaining issue — 4 MP correctness cliff

4 MP now *fits and runs* (the hard infra problem), but the output
collapses (std 2.60). Localized by elimination — **ruled out:** VAE,
flash-tiling, TP core count, bf16 precision, memory/OOM, RoPE, and
block-level magnitude collapse (per-block hidden-state std is healthy
and nearly identical to the working 3 MP). The collapse is post-block:
the 4 MP tokens appear to converge to a shared direction in 3072-space,
so the per-token `norm_out` LayerNorm yields a spatially uniform latent.
Next diagnostic: per-token directional diversity (cosine similarity) at
the last block. Full reasoning in `PROGRESS_AND_FINDINGS.md`.

## Files

| File | Role |
|---|---|
| `src/flux2_tp_plan_v3.py` | v3 full-shard plan + `restructure_for_tp` (splits fused linears) + head fixes |
| `src/flux2_v3_selftest.py` | CPU weight-split equivalence check (all PASS) |
| `src/run_flux2_tp_v3.py` | v3 runner — `--dtype fp32`, `--attn manual\|sdpa`, `--flash-tile`, `--probe-blocks` |
| `src/flux2_fp32_residual.py` | fp32 residual-stream patch (memory-safe precision fix) |
| `src/flux2_mixed_precision.py` | fp32 leaf-norm upcast (skips composite AdaLN) |
| `src/flux2_attention_manual_flash.py` | pure-Python tile flash (correct, slow) — the current scaffold |
| `src/flux2_attention_sdpa.py` | SDPA attention (fast, but OOMs at fp32 high-res — see notes) |
| `src/flux2_attention_cte.py` | CTE NKI-kernel attention wrapper |
| `src/flux2_tp_plan_v2.py` | v2 plan (attention-only shard) — superseded by v3 |
| `src/neuron_flux2_klein_native.py` | pipeline subclass + Neuron patches (real-RoPE, fp32 pos-embed) |
| `src/vae_size_test.py`, `src/rope_grid_test.py` | CPU diagnostics (VAE / RoPE ruled out) |
| `PROGRESS_AND_FINDINGS.md` | full diagnostic log (the source of truth) |
| `SINGLE_STREAM_SHARDING_PLAN.md` | the v3 sharding design |
| `FIX_PLAN.md` | cross-repo reference index (NxDI FLUX, FLUX NKI kernel) |
| `results/` | generated PNGs + progress dashboard |

## Reproduction (on a trn2.48xlarge, Beta 3 DLC)

```bash
# env
export NEURON_RT_VIRTUAL_CORE_SIZE=2 NEURON_LOGICAL_NC_CONFIG=2 \
       NEURON_SKIP_EFA_AFFINITY=1 HF_TOKEN=<your_token>

# 3 MP (1792²), correct, TP=4
torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
    run_flux2_tp_v3.py --dtype fp32 --height 1792 --width 1792 --runs 2

# 4 MP (2048²), fits + runs (output collapse WIP), TP=8
torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
    run_flux2_tp_v3.py --dtype fp32 --height 2048 --width 2048 --runs 1
```

## Next steps

1. **Speed:** replace the pure-Python tile flash with a batched fp32
   NKI/CTE attention kernel for the working 1.6 MP + 3 MP paths.
2. **4 MP correctness:** per-token directional-diversity diagnostic →
   fix the long-sequence attention behavior that flattens the latent.

## Validation

trn2.48xlarge (us-east-2), Beta 3 DLC, native PyTorch + `torch_neuronx`,
`torch.device("neuron")`, distributed backend `neuron`, 2026-06-15.

## License

Apache-2.0 (matches the repo).
