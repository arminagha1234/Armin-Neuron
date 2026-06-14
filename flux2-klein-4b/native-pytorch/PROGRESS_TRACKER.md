# FLUX.2-klein-4B Optimization Progress Tracker

**Started:** 2026-06-14
**Goal:** push from 6.86s warm (Phase A shipped) toward H100 latency
parity (~0.9s) by executing Steps 1-11 from `HANDOFF_TO_IMPLEMENTATION.md`.

**Current shipped state:** 6.86s warm, $0.0013/image at full instance
utilization (1.3× cost vs H100 $0.0010)

## Step status

| Step | Title | Status | Result | Time spent | Commit |
|---|---|---|---|---:|---|
| **1** | Lift NxDI FLUX architecture | 🟡 1.1 done (neg); 1.2 BLOCKED (no collective comms) | kernel-only 8.11s loss; TP=4 blocked by EFA/ENC | — | 7582ae1 |
| 2 | FP8 weights (OCP→Neuron rescale) | ⏸️ deferred | needs TP=4 first (bandwidth win only matters when sharded) | — | — |
| 3 | Context parallelism (TP=4 × CP=2) | ⛔ blocked | same collective-comms blocker as 1.2 | — | — |
| 4 | Fused MLP/o_proj/qkv kernels | ⏸️ deferred | DiT loop saturated single-rank; needs TP=4 | — | — |
| 5 | NKI RoPE replacement | ⏸️ deferred | ~30ms, not worth it pre-TP | — | — |
| 6 | RoPE precompute outside graph | ✅ already done | handled by Flux2PosEmbed patch in pipeline | — | — |
| 7 | requires_grad_(False) | ✅ tested | neutral (7.79s, loop unchanged) | — | 7582ae1 |
| 8 | Functional rotate_half | ⏸️ skipped | interleaved RoPE still needs stack; minimal win | — | — |
| 9 | RMSNorm `.weight` | ✅ verified n/a | diffusers already uses .weight not .weight.data | — | 7582ae1 |
| 10 | Verify single-NEFF compile | ⏸️ deferred | diagnostic; lower priority than TP blocker | — | — |
| 11 | --auto-cast=matmult flag | ✅ tested | net-negative (7.69s, added conversions) | — | 7582ae1 |

## Cumulative wall-clock target

| Milestone | Target | Actual | Delta from baseline |
|---|---:|---:|---:|
| Baseline (Phase A shipped) | — | **6.86 s** | 0 |
| After Step 1 (NxDI lift) | ~3.5 s | TBD | -3.36 s |
| After Step 2 (FP8) | ~2.9 s | TBD | -3.96 s |
| After Step 3 (CP) | ~2.0 s | TBD | -4.86 s |
| After Step 4 (fused kernels) | ~1.7 s | TBD | -5.16 s |
| H100 reference | ~0.9 s | — | — |

## Step 1 — Lift NxDI FLUX architecture

**Status:** in progress (started 2026-06-14)
**Effort estimate (handoff):** 1 week
**Expected win:** 30-50% on 6.86s baseline → ~3.5s

### Re-scoping insight (2026-06-14 15:00)

Reading NxDI source revealed **the 30-50% win comes from TP=4 sharding,
not from `attention_cte` alone.** Previous v3 test (`wrap_nki(attention_cte)`
standalone) was 11% slower at FLUX scale because the compiler's fused
SDPA already wins when not sharded.

Splitting Step 1 into two sub-steps:
- **Step 1.1** — kernel-only swap: replace SDPA with `attention_cte[2]`
  (LNC=2 sharded, no TP=4). Test whether the kernel alone wins at
  klein-4B's specific shape (seq=4608, head_dim=128). 1 day.
- **Step 1.2** — full TP=4 lift: native-PyTorch `parallelize_module` +
  `attention_cte` inside sharded blocks. The 30-50% win lives here.
  ~1-2 weeks.

Only doing 1.2 if 1.1 confirms the kernel works; that gates whether
the kernel matters at this shape.

### Sub-tasks

- [x] Confirm NxDI source on disk (`.tmp/nxdi-ltx/...`)
- [x] Confirm klein-4B refs on disk (`refs/upstream_flux2_klein_*.py`)
- [x] Read NxDI `modeling_flux.py` end-to-end
- [x] Map klein-4B's actual block structure onto NxDI's FLUX.1 classes
  - klein-4B: 8 double + 48 single, heads=48, head_dim=128, joint_dim=15360
  - FLUX.1: 19 double + 38 single, heads=24, head_dim=128, joint_dim=4096
  - Critically: diffusers Flux2 attention layout is `[B, S, H, D]`
    (sequence_dim=1), NxDI/v3 was tested with `[B, H, S, D]`. Different
    permute pattern.
- [x] Identify exact `attention_cte` flags for klein-4B's QKVParallelLinear
  - `tp_q=True, tp_k=True, tp_out=False, causal_mask=False`
  - Use `attention_cte[2](...)` for LNC=2 grid (matches `NEURON_RT_VIRTUAL_CORE_SIZE=2`)
- [x] **Step 1.1**: write `flux2_attention_cte.py` (kernel wrapper)
- [x] **Step 1.1**: write `bench_step1_kernel.py` (A/B bench)
- [x] **Step 1.1**: push to box, kick off compile + bench (running)
- [x] **Step 1.1**: report numbers — **CONFIRMED LOSS: 8.11s vs 6.86s baseline (18% slower)**
  - Quality identical (std=18.15 matches), so kernel is numerically correct
  - tqdm 1.29 it/s = ~775ms/step (vs Phase A 730ms) — loop slightly slower
  - Per-call dispatch + layout permutes outweigh matmul savings at single-rank
  - Matches handoff prediction ("11% slower"); we measured 18%
  - DECISION: don't ship kernel-only. The win needs TP=4 (Step 1.2).
- [ ] **Step 1.2**: full TP=4 lift (the actual 30-50% win)

### Step 1.2 plan (TP=4 native-PyTorch)

The kernel only wins when the attention compute is sharded across 4
cores. Single-rank, the compiler's fused SDPA already wins. To get the
30-50%:

1. `torchrun --nproc_per_node=4 --rdzv_backend c10d` (Beta 3 pattern)
2. `init_process_group` — Beta 3 uses standard c10d (NOT backend="neuron"
   per beta3-only.md steering)
3. `parallelize_module` with ColwiseParallel on to_qkv, RowwiseParallel
   on to_out (the LTX-2 / Wan pattern from neuron-tp-on-beta2.md)
4. attention_cte[2] inside the sharded blocks (heads sharded → each
   rank does heads/4 = 12 heads)
5. TPRMSNorm for the sharded qk_norm (the LTX-2 fix #2)
6. RoPE head-slice per rank (the LTX-2 fix #4)

This is the multi-day task. Starting the scaffold now.
- [ ] Validate cosine ≥ 0.9999 vs canonical 4-step bf16 reference
- [ ] Bench warm timed; confirm ≤ 4.0s

### Notes / decisions

- 2026-06-14 14:30: starting task. Reading NxDI source first.
- 2026-06-14 15:00: split into 1.1 (kernel swap) + 1.2 (TP=4 lift).
- 2026-06-14 15:30: 1.1 code on box, compile starting.

## Steps 6-11 — cheap wins (TESTED)

**Status:** tested, net-negative or neutral on single-rank.

### Step 11 (--auto-cast=matmult) + Step 7 (requires_grad) bundled
- Config B: 7.69s avg (vs 6.86s baseline) → **0.83s SLOWER**
- Quality fine (std=18.16)
- The auto-cast=matmult flag added conversion overhead rather than
  removing it. Net-negative at this config.
- Re-testing Step 7 alone (no auto-cast) to isolate.

### CRITICAL ANALYSIS (2026-06-14 ~17:00)

**Every DiT-side optimization lands at ~730ms/step (1.37 it/s) in the
denoising loop — the same as baseline.** The DiT NEFF execution is
SATURATED at the single-rank floor for this graph shape. Evidence:
- Phase A baseline:        1.37 it/s (730ms/step)
- Kernel swap (Step 1.1):  1.29 it/s (775ms/step) — slightly slower
- auto-cast (Step 11):     1.37 it/s (730ms/step) — same loop
- requires_grad (Step 7):  testing now

**Implication:** the handoff's projected stack (Step 1→3.5s, Step
2→2.9s) is over-optimistic for SINGLE-RANK. Those wins require genuine
sharding (TP=4) to split the 730ms/step across 4 cores → ~200ms/step.

The ONLY remaining lever with real headroom is **TP=4 (Step 1.2 /
Step 3 context parallelism)**. Everything else is shuffling the
~730ms/step DiT floor.

What actually moved wall-clock this whole project:
- Phase A CPU-side caching: 34s → 6.86s (the 5× win)
- Everything DiT-side: noise around 6.86s

### Recommendation revision

For the customer (fal.ai), the shipped 6.86s / $0.0013/image is already
cost-competitive with H100. The path to latency parity requires TP=4,
which is a real multi-week engineering project with torchrun
multi-process orchestration. Not an overnight autonomous task — it
needs interactive debugging of process-group hangs.

**Steps 2-11 do NOT stack to the projected numbers on single-rank.**
The honest customer story: 6.86s shipped, TP=4 needed for sub-4s,
and TP=4 is the next major investment.

## Boxes used

- `3.15.152.199` — test gemma4 box, trn2.48xl, beta3 container,
  Phase A NEFF cache at `/mnt/data/work/flux2/neff_cache_4step`
- `i-0cf5d3577220d6091` (Mel) — trn2.3xlarge, ap-southeast-4, used
  by other chat for canonical 4-step bench

## GitHub commits

- `b101647` re-bench at distilled config (4 steps, guidance=1.0)
- `a955dc5` image-latent caching cuts wall-clock 5× (34.6s → 6.86s)
- `9036322` Phase B (VAE on Neuron) blocked by compiler instr-count limit
- TBD: Step 1 — NxDI lift
