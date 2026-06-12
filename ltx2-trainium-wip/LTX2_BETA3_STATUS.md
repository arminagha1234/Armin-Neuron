# LTX-2 19B native PyTorch on Trainium2 (Beta 3) — WIP status

**Date:** 2026-06-12
**Stack:** Beta 3 DLC (`concourse-release-0461d3b:latest`), torch 2.11.0,
torch_neuronx 2.11.3.0.1278, `torch.device("neuron")`, TP=4 via
`parallelize_module` + `backend="neuron"` PG.

## What works

1. ✅ Beta 3 DLC pulled, runtime driver installed on host, `neuron-ls`
   shows 16 devices × 4 cores.
2. ✅ `beta3` container running privileged with Neuron access.
3. ✅ `torch.device("neuron")` tensor ops verified.
4. ✅ `LTX2Pipeline` + `LTX2VideoTransformer3DModel` import in diffusers (git main).
5. ✅ Meta-init build of the 18.88B LTX-2 transformer.
6. ✅ TP=4 `parallelize_module` with 1152-entry plan (`ltx2_tp_plan.py`).
7. ✅ Sharded weight loader streams weights per-rank in ~5s
   (`ltx2_meta_loader.py`).
8. ✅ `attn.heads` patched to heads/N (8 video, 8 audio).
9. ✅ All FOUR RoPE modules (`rope`, `audio_rope`, `cross_attn_rope`,
   `cross_attn_audio_rope`) sliced by rank.
10. ✅ Pipeline runs end-to-end through: text encoder (CPU) → connectors
    (CPU) → into the denoising transformer's forward.
11. ✅ Transformer-input CPU→Neuron wrapper catches all boundary tensors.

## CURRENT blocker (2026-06-12, latest) — norm_q/norm_k not materialized

After fixing the RMSNorm shape issue (adaptive QK norm) and the
operation order (load weights → then install adaptive norm), the run
gets all the way INTO the denoising transformer forward but fails with:

```
RuntimeError: Tensor on device meta is not on the expected device neuron:0!
```

Root cause (confirmed): the meta loader reports
`576 keys in checkpoint not in model` for all the
`transformer_blocks.*.{attn1,attn2,audio_attn1,...}.norm_{q,k}.weight`
keys. These ARE present in the meta-init model's state_dict (verified:
`transformer_blocks.0.attn1.norm_q.weight (4096,)` exists), but the
loader's module-walk resolver (`_resolve`) cannot reach them after
`parallelize_module` ran. So the norm weights stay on `meta`, and the
first multiply against them in the attention forward raises the
meta-device error.

### Next step to try

The issue is `parallelize_module` + the `norm_q`/`norm_k` module
identity. Two things to investigate:
1. Whether `parallelize_module` replaced the parent attention module
   such that `getattr(block.attn1, "norm_q")` no longer returns the
   original RMSNorm (DTensor wrapping can rebind submodules).
2. Whether diffusers' LTX2 `norm_q` weight is registered under a
   different attribute after the model is parallelized.

Quickest robust fix: materialize ALL remaining meta params in one pass
AFTER parallelize_module + load, by walking `model.named_parameters()`
and `.to_empty()`-ing or filling any param still `.is_meta` with the
checkpoint tensor looked up by its full dotted name. This sidesteps the
resolver mismatch entirely.

## EARLIER blocker (resolved) — TP-aware RMSNorm for `rms_norm_across_heads`

LTX-2 uses `qk_norm = "rms_norm_across_heads"`. The norm weights are
full inner_dim:

| Module | norm_q / norm_k shape |
|---|---|
| attn1, attn2 (video) | 4096 |
| audio_attn1, audio_attn2, audio_to_video_attn, video_to_audio_attn | 2048 |

Current failure:
```
RuntimeError: Attempting to broadcast a dimension of length 512 at -1!
  had torch.Size([512]); but expected shape broadcastable to [2, 26, 2048]
```

Root cause: `ltx2_meta_loader.py` shards `norm_[qk].weight` on dim 0
(4096→1024 video, 2048→512 audio), but `rms_norm_across_heads`
normalizes over the FULL head dim. The norm is applied to a tensor that
is NOT consistently sharded the way the weight is — the audio norm in
particular sees a full 2048-dim tensor but its weight is now 512.

### The fix (next step)

Two correct options:

**Option A — replicate norm weights + TP-aware norm forward.**
- In `ltx2_meta_loader.SHARD_RULES`, change `norm_[qk]` from dim 0 to
  `None` (replicate full weight on every rank).
- Replace each `LTX2Attention.norm_q/norm_k` with a `TPRMSNorm` that:
  1. computes local sum-of-squares over the rank's head slice,
  2. all-reduces across TP ranks to get the global RMS,
  3. divides by full_dim,
  4. applies the rank's slice of the (replicated) weight.
- This mirrors the LTX-2 TP recipe in
  `.kiro/steering/neuron-tp-on-beta2.md` "fix #2".

**Option B — don't shard the QK projection at all (simpler, slower).**
- Remove `attn*.to_q/to_k` from the ColwiseParallel plan so q/k stay
  full-dim; only shard `to_v` + `to_out`. Then norm_q/norm_k stay full
  and the norm "just works". Costs more memory per rank (q/k not
  sharded) but avoids the TP-aware norm machinery. May OOM.

Recommend Option A. The `TPRMSNorm` class already exists in
`customers/fal/path_c/tprmsnorm.py` (built for Qwen-Image-Edit) and can
be adapted — the key change is the across-heads reduction dim.

## Files (this WIP)

- `ltx2_run.py` — main TP=4 runner (meta-init → parallelize → load →
  pipeline swap → CPU/Neuron patches → generate)
- `ltx2_tp_plan.py` — TP plan + attn.heads patch + 4-RoPE rank slice
- `ltx2_meta_loader.py` — sharded weight loader (SHARD_RULES need the
  norm_[qk] → None change per Option A)
- `ltx2_beta3.py` — single-core smoke (verified Beta 3 stack)
- `setup_beta3.sh` — Beta 3 host setup (driver install from DLC artifacts)

## Repro

```bash
# On the box, inside the beta3 container:
sudo docker exec -e HF_HOME=/opt/dlami/nvme/ltx2/hf_cache \
    -e HF_TOKEN=<token> \
    -e NEURON_RT_VIRTUAL_CORE_SIZE=2 -e NEURON_RT_NUM_CORES=4 \
    beta3 bash -c 'source /opt/torch-neuronx/.venv/bin/activate && \
    cd /workspace && torchrun --nproc_per_node=4 --rdzv_backend c10d \
    --rdzv_endpoint localhost:29500 ltx2_run.py \
    --num-steps 4 --num-frames 25 --no-compile'
```

## Hardware

trn2.48xlarge `i-02a51e30b3a33408d`, container `beta3`,
HF cache + weights at `/opt/dlami/nvme/ltx2/`.
