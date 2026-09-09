# Fusing DeepSeek-V4-Flash's two-source decode attention into one flash NKI kernel

This folder is the record of chasing a **batch-16 decode regression** to its root and fixing
most of it with a single fused attention kernel. Everything here is self-measured on a
Trainium2 instance in native PyTorch mode (no XLA), 43 layers, tensor-parallel 32,
generation length 128. Every number is observed. Nothing is projected. The correctness gate
is a fixed golden first-token id (`4256`); a run that does not reproduce it is reported as a
failure, not smoothed over.

> Measurement note: absolute decode tok/s carries a few-percent run-to-run spread across
> compiler/DLC snapshots (e.g. baseline batch 8 has been observed at 31.4 and 33.0 on
> different runs). The load-bearing claims here are same-session A/B deltas and the golden
> token, not any single absolute number.

**Headline:** the model's two-source attention — a sliding-window KV source and a compressed
KV source that share **one** softmax (max + denominator) plus a per-head sink term — is fused
into a **single online-softmax ("flash") NKI kernel** that never materialises the full
`[batch, heads, context]` score tensor. At batch 16 it lifts decode from **23.56 -> 29.83
tok/s (+26.6%)** with the golden token preserved. At batch 8 it is neutral. That gap between
"+26.6% at 16" and "neutral at 8" is the whole story, and it is explained below.

![DeepSeek-V4-Flash native decode progress: batch-16 regression fixed by the flash dual-attn kernel, and the scaling ceiling that explains the residual](progress.png)

---

## The regression

Doubling the decode batch should raise aggregate throughput. It did the opposite:

| batch | aggregate decode tok/s | per-step time (batch / tok_s) |
|------:|-----------------------:|------------------------------:|
|     8 | 33.02                  | 0.242 s                       |
|    16 | 23.56                  | 0.679 s                       |

Batch 16 was **slower in aggregate** than batch 8. A decode step at 16 took 2.80x the time of
a step at 8, when a well-behaved decode should be closer to 2x (or less, if weight bandwidth
is shared across the batch). Something was scaling worse than linearly with the batch.

## Locating it

The head budget rules out a compute wall. This model has 64 attention heads at `head_dim=512`;
sharded over TP=32 that is **2 heads per rank**. The score and output matmuls therefore have an
output partition dimension of 2 — the 128x128 systolic array runs at roughly 1.5% utilisation.
Decode attention here is **not** compute-bound; it is bound by moving KV out of HBM and by
per-op launch overhead.

A skip-MoE decomposition (run the whole decode path with the expert FFN turned off, timing
only; the golden intentionally does not hold) isolates the non-MoE work:

| non-MoE decode work | batch 8 | batch 16 | 16/8 step ratio |
|---|---:|---:|---:|
| baseline attention | 43.55 tok/s (0.184 s) | 35.41 tok/s (0.452 s) | **2.46x** |

2.46x for a 2x batch, with the expert FFN removed — so the super-linear part lived in
attention, not the MoE. The remaining suspect that scales with batch is the **score tensor**:
the unfused path materialises `[batch, heads, context]` (and a second one for the compressed
source), and at batch 16 that pushed the working set past what stays resident, so it spilled.

## The kernel

`nki_dual_sparse_attn.py` fuses the entire two-source attention body into one kernel and
removes the score tensor from the picture:

- **Online softmax.** KV is processed block-by-block with a running max, running denominator,
  and a running output accumulator. The full score row is never held; only one
  `[heads, kv_block]` tile exists at a time. Order-independence of the online update lets the
  **two sources share one max/denominator** — run source A's blocks, then source B's blocks,
  into the same running state.
- **Per-head sink** is folded into the shared denominator once, after both sources.
- **Batch loop on the outside.** The kernel loops over batch and does each sequence's
  attention independently, so its on-chip (SBUF) footprint is **independent of batch**. That
  is the property that removes the spill.

The math it implements, exactly:

```
M      = max(  max(qA·scale + maskA),  max(qB·scale + maskB),  sink )
denom  = sum(exp(A - M)) + sum(exp(B - M)) + exp(sink - M)
out    = ( sum(exp(A - M)·kvA) + sum(exp(B - M)·kvB) ) / denom
```

It was validated against a CPU reference to ~6.5e-9 max abs error before ever going on device.

## Result

| batch | baseline | fused flash kernel | golden |
|------:|---------:|-------------------:|:------:|
|     8 | 33.02    | ~neutral (-1%)     | PASS   |
|    16 | 23.56    | **29.83 (+26.6%)** | PASS   |

The spill hypothesis holds: at batch 8 there is no spill, so removing a spill that is not
there does nothing (neutral). At batch 16 the spill was real, and the flash kernel recovers
most of it. The non-MoE step scaling improves in step with that — from 2.46x to 2.22x:

| non-MoE decode work | batch 8 | batch 16 | 16/8 step ratio |
|---|---:|---:|---:|
| baseline attention | 43.55 (0.184 s) | 35.41 (0.452 s) | 2.46x |
| **flash attention** | 58.69 (0.136 s) | 52.76 (0.303 s) | **2.22x** |

## The honest ceiling

The fix does **not** make batch 16 beat batch 8 in aggregate throughput (29.83 is still under
33.02). Even with the expert FFN removed, flash attention still scales 2.22x per 2x batch —
super-linear. That residual is the 2-heads-per-rank reality above: with attention DMA/launch
bound, doubling the batch roughly doubles the step, so aggregate throughput stays about flat
across batch instead of rising. Closing that last gap is not a kernel problem — it needs the
attention sharded over fewer ranks than the rest of the model, which is an architecture change.

So the kernel's real value is making the **higher-concurrency (batch 16) regime viable at
near-parity** with batch 8, instead of the 30%-slower regression it was.

## Batch 32 is a different wall

Batch 32 does not run at all on this configuration — it is a hard **HBM out-of-memory** at
device warm-up, on every rank: 24 GiB/core, ~23.4 GiB already allocated, and it fails asking
for the next 8 MiB. That is the KV cache plus activations for 32 concurrent sequences
exceeding on-device memory. The flash kernel shrinks the on-chip (SBUF) working set, not the
HBM KV cache, so it cannot help here — confirmed: the kernel reaches warm-up and OOMs
identically. Batch 32 needs lower-precision KV, more ranks, or paged KV, none of which is a
kernel change.

## What didn't move the needle

The kernel's batch loop can be built two ways, selectable with `V4_ATTN_B_SEQ`:
`affine_range` (default; the compiler is free to unroll the batch) or `sequential_range`
(no unroll). At batch 16 the two are a wash (29.75 vs 29.83, golden PASS both) — the batch
loop is not where the time goes. The sequential form compiles a smaller graph, but it does
**not** rescue batch 32: same HBM OOM at warm-up. Recorded because "we tried the obvious
scheduling knob and it was neutral" is worth as much as a win.

## How it's wired

The kernel is engaged by a batch-gated switch:

- unset (default): **AUTO** — engages at batch >= `VLLM_NEURON_V4_NKI_ATTN_MIN_BATCH`
  (default 16), where it is a win; stays off at low batch where it is neutral-to-slightly-negative.
- `VLLM_NEURON_V4_NKI_ATTN=1`: force on.
- `VLLM_NEURON_V4_NKI_ATTN=0`: force off.

## Files

- `nki_dual_sparse_attn.py` — the fused dual-source online-softmax attention kernel.
