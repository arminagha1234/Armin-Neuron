# Clay-on-Trainium — Performance Roadmap & trn2 Plan

Baseline measured on **trn1, single NeuronCore, bf16, manual attention**:
~9 img/s (base 128px, B16, compile) / ~5.7 img/s (large 256px, B8) / **~8% MFU**
(95 BF16 TFLOPS/core peak). ~8% is an un-tuned floor with named ~3–5× headroom.

## Improvement levers, ordered by expected impact

### 1. Fused attention — fix the SDPA-backward blocker (BIGGEST lever)
- Today we run `fused_attn=False` (manual matmul+softmax) because SDPA **backward**
  crashes the trn1 Beta-3 runtime (`tensor_set_slice` assertion). Manual attention is
  the slow path.
- **On trn2 / newer SDK: first just retest `fused_attn=True`** — the bug may already be
  gone. That alone should lift MFU noticeably.
- If still broken: write a bidirectional (non-causal) flash-attention NKI kernel with a
  working backward, or try FlexAttention (guide lists it as forward-only/causal — Clay
  attention is non-causal, so a custom bwd is likely needed). File with Neuron team.

### 2. Multi-core scale-out (near-linear images/s)
- Clay is small → **data parallel** is the easy multiplier: replicate the model,
  shard the batch, world-group all-reduce grads (pattern already verified in-sync at
  2 cores in `clay_ddp_train.py`).
- On trn1, >2-core collectives failed (`Failed to verify MLA indices` at global-comm
  init). **Retest on trn2** — likely trn1-specific / needs a newer collectives build.
- trn2.48xlarge = 16 Trainium2 chips (64 NeuronCores). Target: DP across all cores →
  ~50–60× the single-core img/s if collectives cooperate.

### 3. Bigger batch (memory-bound)
- batch=1 starves the core (2.7–3.8% MFU). Batching to 8–16 ~2× throughput.
- trn2 has more HBM/core → push batch higher. Add **gradient checkpointing** to fit
  large batch at 256px/large.

### 4. torch.compile + fused attn together
- Compile gave 1.4× at small batch on trn1 (converges with eager at large batch).
- Combine with fused attention; retune on the trn2 compiler. Note: compile can't
  reshape in-process (no dynamic shapes) — one compile per (batch, resolution) bucket.

### 5. bf16-mixed autocast (numerics + minor perf)
- We use a blanket bf16 cast + fp32→bf16 casts on pos/wave embeddings (`# bf16-safe`).
- Move to Clay's real **bf16-mixed autocast** so sensitive reductions stay fp32.
  Test `torch.autocast(device_type=..., dtype=bf16)` support on the neuron backend.

### 6. Amortize the frozen DINOv2 teacher (~1/3 of hot-path FLOPs)
- Teacher (304M) runs every step just to produce the representation-loss target on
  fixed inputs. **Precompute/cache teacher targets offline** (dataset is fixed), or run
  the teacher on a separate core/stream. Removes a big chunk of per-step FLOPs.

### 7. Make the whole step compile (remove the eager island)
- `mask_out` is a `@torch._dynamo.disable` eager island (argsort→AwsNeuronTopK fails,
  on-graph RNG). Refactor: generate shuffle indices outside the graph (host/CPU) and
  pass them in; do masking via `torch.gather`/`torch.scatter`. Vectorize the
  channel-drop python loop; move RNG on-device. Then the full step compiles with no
  graph breaks → better utilization.

### 8. Compiler/runtime flags
- Runtime suggested: `NEURON_CC_FLAGS="--hbm-scratchpad-page-size=2048"` +
  `NEURON_SCRATCHPAD_PAGE_SIZE=2048`. Sweep these.
- Try `NEURON_RT_VIRTUAL_CORE_SIZE=2` (LNC2) on trn2.

### 9. Profile before optimizing further
- Capture a neuron-explorer profile (NEFF+NTFF), find the bottleneck engine
  (PE array vs DMA vs manual-softmax), and target that. Don't guess past ~step 8.

## trn2 bring-up checklist (when the box lands)
1. Same Beta-3 DLC works on trn2 (guide: "TRN3, TRN2, TRN1, or INF2"). Pull image,
   install runtime debs, reload driver, `neuron-ls` (expect 64 cores / LNC).
2. Copy this `clay/` folder to the box; `pip install transformers` in the DLC container.
3. **Retest `fused_attn=True`** (SDPA backward) — first thing, it's the top lever.
4. Re-run `clay_full_train.py` (large, 256px) fp32 + bf16 to confirm parity.
5. Re-run `clay_bench.py` sweep → new single-core baseline (trn2 ≈ 4× trn1/chip).
6. Re-run `clay_ddp_train.py` at 2 → 4 → 8 → 64 cores; confirm collectives scale.
7. If SDPA-backward works: drop `fused_attn=False`, re-sweep, recompute MFU.
8. Then: batch scaling + gradient checkpointing + teacher caching + compiler flags.
9. Profile with neuron-explorer; iterate on the real bottleneck.

## Expected trajectory
trn1 single-core ~8% MFU (manual attn) → trn2 + fused attn + batch + compile should
push single-core MFU up several-fold; multi-core DP then multiplies images/s roughly
linearly with core count. Report **images/s** as the headline; MFU once tuned.
