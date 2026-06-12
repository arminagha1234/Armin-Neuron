# LTX-2 19B on Trainium2 — Native PyTorch (Beta 3) — WORK IN PROGRESS

⚠️ **This is a WIP port, not yet producing output.** It is committed to
preserve substantial infrastructure progress and document the exact
remaining blocker. See `LTX2_BETA3_STATUS.md` for the detailed state.

## Goal

Port `Lightricks/LTX-2` (18.88B audio-video DiT) to Trainium2 using
**native PyTorch + torch_neuronx on the Beta 3 stack** (no NxDI, no
vLLM), TP=4, mirroring the approach that shipped for Qwen-Image-Edit.

This is the workload we expect to close the Trainium-vs-H100 gap the
most (large model + heavy per-step video diffusion compute amortizes
the TP collective overhead — see `customers/fal/ACCOUNT_PLAN.md`).

## What works (verified on trn2.48xlarge, Beta 3 DLC)

- ✅ Beta 3 DLC + driver install, `torch.device("neuron")` ops
- ✅ Meta-init build of the 18.88B transformer
- ✅ TP=4 `parallelize_module` (1152-entry plan)
- ✅ Sharded weight loader (~5s) — `ltx2_meta_loader.py`
- ✅ `attn.heads` patched to heads/N for all 6 attention types
- ✅ All 4 RoPE modules sliced by rank
- ✅ Adaptive QK norm for `rms_norm_across_heads` (resolved the
  sharded-norm shape mismatch)
- ✅ Runs end-to-end through text encoder (CPU) → connectors (CPU) →
  into the denoising transformer forward
- ✅ CPU↔Neuron transfer wrapper on the transformer input

## What's blocking (one fix away)

The `norm_q`/`norm_k` weights stay on `meta` after `parallelize_module`
because the loader's module-walk resolver can't reach them post-
parallelization. The fix is a final "materialize all remaining meta
params" pass. See `LTX2_BETA3_STATUS.md` → "CURRENT blocker".

## Files

| File | Purpose |
|---|---|
| `ltx2_run.py` | TP=4 runner: meta-init → parallelize → load → swap into pipeline → CPU/Neuron patches → generate |
| `ltx2_tp_plan.py` | TP plan + attn.heads patch + adaptive QK norm + 4-RoPE rank slice |
| `ltx2_meta_loader.py` | Sharded weight loader (module-walk resolver) |
| `ltx2_beta3.py` | Single-core smoke test (verified the Beta 3 stack) |
| `ltx2_beta3_fsdp.py` | Early FSDP attempt (superseded by ltx2_run.py) |
| `ltx2_naive_trn2.py` | Naive single-core attempt (OOMs — documents the need for TP) |
| `setup_beta3.sh` | Beta 3 host setup (driver from DLC artifacts) |
| `LTX2_BETA3_STATUS.md` | Detailed status + exact remaining blocker + next steps |

## Repro

See `LTX2_BETA3_STATUS.md` → "Repro". Runs inside the Beta 3 `beta3`
container with `torchrun --nproc_per_node=4`.

## Why committed as WIP

This captures ~8 distinct architectural fixes (meta-init, TP plan,
RoPE slicing for 4 modules, adaptive RMSNorm for across-heads norm,
CPU/Neuron boundary handling) that are reusable for any LTX-family or
similar audio-video DiT port. The remaining blocker is well-understood
and documented. Better to preserve this than lose it on a scratch box.
