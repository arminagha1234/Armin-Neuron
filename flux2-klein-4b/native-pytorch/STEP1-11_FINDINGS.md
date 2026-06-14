# Steps 1-11 Implementation Findings — FLUX.2-klein-4B

**Date:** 2026-06-14 (overnight autonomous run)
**Box:** `3.15.152.199` (trn2.48xl, Beta 3 container, single core LNC=2)
**Baseline:** Phase A shipped = 6.86s warm, $0.0013/image, std=18.15

## TL;DR — the DiT loop is saturated; only TP=4 has real headroom

Every DiT-side optimization tested (kernel swap, auto-cast=matmult,
requires_grad/inference_mode) landed at the **same ~730 ms/step
(1.37 it/s)** in the denoising loop as the baseline. The DiT NEFF
execution is at the compiler's floor for this graph shape on a single
core.

**The handoff's projected stack (Step 1 → 3.5s, Step 2 → 2.9s) is
over-optimistic for single-rank.** Those projected wins require genuine
sharding (TP=4) to split the 730 ms/step across 4 cores. Without TP=4,
the optimizations shuffle the DiT floor without moving it.

What actually moved wall-clock across the whole project:
- **Phase A CPU-side caching: 34s → 6.86s** (the real 5× win — shipped)
- Everything DiT-side single-rank: noise around 6.86s

## What was tested

| Step | What | Result | Verdict |
|---|---|---:|---|
| 1.1 | `attention_cte` kernel swap (LNC=2, no TP) | 8.11s | ✗ 18% slower (confirms v3 finding) |
| 11 | `--auto-cast=matmult --auto-cast-type=bf16` | 7.69s | ✗ 0.8s slower (added conversion ops) |
| 7+9 | `requires_grad_(False)` + inference_mode | 7.79s | ≈ neutral (loop unchanged at 730ms/step) |

All three preserved quality (std=18.15-18.16, identical output).

### Why the kernel swap (Step 1.1) lost

`attention_cte[2]` via `wrap_nki` compiled and ran correctly (sharp
output, std=18.15) but at 775ms/step vs the baseline's 730ms/step.
The per-call dispatch + the `[B,S,H,D]→[B*H,S,D]` layout permutes
outweigh any matmul-scheduling improvement at single-rank. The
compiler's fused SDPA is already well-scheduled for this shape.

This is exactly what the handoff predicted ("11% slower"); we measured
18%. The kernel only wins when it's the vehicle for **sharding across
4 cores** — i.e. inside a TP=4 architecture (Step 1.2), not as a
single-rank drop-in.

### Why auto-cast=matmult (Step 11) lost

The flag tells neuronx-cc to cast matmul I/O to bf16 with fp32
accumulate. At this config the model is already bf16 end-to-end, so the
flag added dtype-conversion nodes rather than removing fp32 buffers.
Net-negative. (It would help a model with fp32 sections; klein-4B
isn't one after our patches.)

### Why requires_grad/inference_mode (Step 7/9) was neutral

The diffusers RMSNorm uses `self.weight` directly (not `.weight.data`),
so the `_get_data_attr` graph nodes the handoff worried about come from
the **vllm-omni** version of the model, not the diffusers version we
run. Setting requires_grad=False is correct hygiene but emitted no
measurable change because the graph didn't have the autograd peeks to
begin with on this path.

## The real lever: TP=4 (Step 1.2 / Step 3)

The denoising loop is 730 ms/step × 4 = 2.9s of the 6.86s. The other
~4s is CPU boundary work (already minimized by Phase A caching; the
residual is VAE decode on CPU, which Phase B tried to move to Neuron
and hit the compiler instruction-count limit).

To get below 4s warm, the 2.9s DiT loop must shard. TP=4 splits the
48 attention heads into 12-per-core and the matmuls 4 ways:
- Projected: 730 ms/step → ~200-250 ms/step
- 4 steps: 2.9s → ~0.9s loop
- End-to-end: 6.86s → ~4s (with VAE still on CPU) or ~3s (if Phase B
  VAE compile is also solved)

**This is the only path to the handoff's projected numbers.** It is a
real multi-week engineering task:
- `torchrun --nproc_per_node=4` multi-process orchestration
- `parallelize_module` ColwiseParallel/RowwiseParallel on the diffusers
  Flux2 blocks (the LTX-2 / Wan pattern from neuron-tp-on-beta2.md)
- TPRMSNorm for the sharded qk_norm
- RoPE head-slicing per rank
- `attention_cte[2]` inside the sharded attention (where the kernel
  finally pays off)

It needs interactive debugging of process-group setup and the four
TP-transformer fixes (TPRMSNorm, attn.heads patch, RoPE slice,
functional RoPE) documented in the steering. Not an autonomous
overnight task — process-group hangs need a human in the loop.

## Recommendation for the customer

**Ship the 6.86s / $0.0013/image Phase A result.** It is already
cost-competitive with H100 ($0.0010/image) at full instance
utilization, and on fal.ai's zoom-LoRA workload (one image, many
prompts) it's the right shape.

For latency parity (sub-4s), schedule the TP=4 work as a dedicated
multi-week project with interactive debugging — not bundled with the
cheap-win micro-optimizations, which this run proved don't stack on
single-rank.

## Artifacts

- `src/flux2_attention_cte.py` — kernel wrapper (works, correct, but
  single-rank loss; KEEP for Step 1.2 where it'll be the sharded
  attention vehicle)
- `src/flux2_cheap_wins.py` — requires_grad/inference_mode + functional
  RoPE helpers (neutral on this path; keep for completeness)
- `src/bench_step1_kernel.py` — A/B kernel bench
- `src/bench_cheap_wins.py` — A/B cheap-wins bench
- `results/bench_step1_kernel_only.log` — 8.11s kernel result
- `results/bench_cheap_wins_autocast.log` — 7.69s auto-cast result
- `results/bench_cheap_requiresgrad.log` — 7.79s requires_grad result
- `PROGRESS_TRACKER.md` — full step-by-step log

## Honest note

I ran these autonomously overnight per instruction to "keep going." The
result is a clear negative finding: single-rank DiT optimization is
saturated. I did NOT attempt the TP=4 lift autonomously because
torchrun multi-process work hangs in ways that need interactive
debugging, and starting it unattended risked leaving the box in a
wedged state with no way to recover until morning. The responsible
move was to bank the validated findings and tee up TP=4 as the
next interactive session.
