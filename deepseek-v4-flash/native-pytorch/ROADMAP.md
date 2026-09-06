# Serving DeepSeek-V4-Flash on native PyTorch — the improvement roadmap

This is the "what next and how far can it go" plan, grounded in what the decode
work in this folder actually measured. It is deliberately opinionated about
order, because the levers are not equal and several dead ends look attractive.

## The one equation that orders everything

For an inference server the number that matters is aggregate tokens/second across
all in-flight sequences:

```
throughput  ~=  batch_size  x  per_sequence_rate  x  tokens_per_step
                ^^^^^^^^^^      ^^^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^
                (1) batching    (2) step latency      (3) speculation
```

Everything below is one of those three factors, plus the infrastructure needed
to measure them honestly. The measured starting point:

- **Batch 1 decode: ~6.2 tok/s, ~160 ms/step, TTFT ~217 ms**, greedy token 4256
  matching the XLA path.
- The reference XLA path on the same model does **1.35 tok/s at batch 1 and 37.81
  at batch 128** -- a **28x** swing that lives almost entirely in factor (1).

That 28x is the headline. It says plainly: **batch_size is the dominant term, and
this native path forfeits it by being stuck at batch 1.** Every batch-1
micro-optimisation is a rounding error against it.

## What we already know, so we don't relearn it

Three findings from the decode work bound the search space:

1. **Batch-1 decode is not compute-kernel-bound.** Removing ~3,440 small copy
   kernels per step (the transpose-free Sinkhorn) moved throughput by ~0%. So the
   step is bound by collective dispatch and fixed per-request / per-layer host
   overhead, not by the amount of small compute.
2. **The two wins that landed were operator-level, not kernel-level.** A single
   `topk` replacing six sequential argmax passes, and `pow(2)`->`x*x`, together
   worth +20.7%. Neither is in any kernel library -- they were in the model's hot
   path. That is where cheap wins hide.
3. **Batch>1 is blocked by one specific thing:** the model compiles with
   `fullgraph=True`, so the vocab-sharded lm-head's collective sits inside a
   single 32-rank graph that deadlocks at the compile barrier. The wheel exposes
   `VLLM_NEURON_DEBUG_MODE=1` -> `fullgraph=False`, which lets dynamo graph-break
   and keep collectives eager between compiled regions. That experiment is in
   flight.

---

## Tier 1 -- Unlock batching (the 28x prize)

This is the whole game. Nothing else in this document matters as much.

### 1.1 Break the fullgraph lm-head deadlock  ·  effort: low  ·  risk: low
`VLLM_NEURON_DEBUG_MODE=1` disables fullgraph. If graph-break alone lets batch=8
compile and run, batching is unblocked with a one-line env change. Cost: some lost
fusion (measure it). **In flight now.** If it works, immediately sweep batch
in {8, 16, 32, 64} to find where HBM or the collective cost caps it.

### 1.2 If graph-break isn't enough: take the lm-head collective out by hand
Two fallbacks, in order of preference:
- **Eager logits / explicit graph break** right before the lm-head reduction, so
  the decoder stack stays compiled and only the vocab reduction runs eager. Zero
  HBM cost, preserves the greedy token.
- **Replicate the lm-head** instead of vocab-sharding it -- removes the collective
  entirely, at the cost of `vocab x hidden x 2 bytes` per rank. At vocab~129k,
  hidden 4096, bf16 that is ~1 GB/rank; check it against the ~21/24 GB already
  resident before committing.

### 1.3 Then tune the batch/bucket grid  ·  effort: medium
Batch 32 previously failed at prefill warmup with an HBM allocation error, so
batching has to be co-tuned with the prefill bucket and `num_gpu_blocks_override`,
not raised blindly. Build the bucket grid the way a real server would: a few
prefill buckets, a few decode batch sizes, warm each.

**Expected payoff:** if native tracks the XLA batch curve even loosely, this is
the difference between ~6 tok/s and tens of tok/s aggregate. It is the only tier
that changes the order of magnitude.

---

## Tier 2 -- Cut per-step overhead (factor 2)

Only worth doing *after* batching, because batching amortises fixed per-step cost
and changes which of these still matter. Ordered by evidence strength.

### 2.1 Reduce collective count  ·  effort: medium  ·  measured ~21% of step
A decode token issues ~150 collectives. The MoE routing's `all_gather` +
`all_reduce MAX` is ~86 of them and is avoidable: if the router is replicated,
each rank computes the argmax redundantly and both collectives disappear. This is
the single largest identified per-step term and it needs no kernel work.

### 2.2 Overlap the remaining collectives with compute  ·  effort: high
The measured overlap of collectives with the tensor engine was ~0%, because the
runtime has one compute queue and one collectives queue per core and every
collective is dispatched as its own graph. Reaching the collectives queue
independently needs the collective on a non-default stream (what host-CC enables
but this port doesn't use). This is the real fix for the collective share, and the
natural vehicle is a decode megakernel or a hierarchical collective schedule.

### 2.3 Keep pruning operator-level overhead  ·  effort: low  ·  but nearly tapped
The audit surfaced more bit-exact micro-opts (RMSNorm weight upcast hoisting,
constant-tensor caching done *without* the recompile trap, redundant f32 round
trips). Individually small; batch-1 already showed this class has limited headroom.
Do them opportunistically, not as a focus. **Lesson banked:** never cache into a
module attribute inside a compiled forward -- it flips a dynamo guard and vLLM's
`fail_on_recompile` makes it fatal.

---

## Tier 3 -- Speculative decode (factor 3) -- the second multiplier

DeepSeek-V4-Flash ships a Multi-Token-Prediction (MTP) head, and the weights are
present in the checkpoint. Accepted speculation multiplies tokens-per-step directly
-- a 2-3x decode multiplier on top of whatever batching achieves.

**Status: hard, possibly infeasible on this stack today.** The earlier assessment
found spec-decode is gated to EAGLE3-only here, and wiring MTP would mean rewinding
the compressed-attention recurrence across 41 layers -- estimated weeks. So this is
a **research bet, not a quick win**, and it is correctly sequenced *after* batching
because a spec-decode multiplier on a batch-1 base is still a batch-1 base. Revisit
once (a) batching works and (b) the serving stack's spec-decode path is understood.

---

## Tier 4 -- Per-token compute efficiency (factor 2, compute side)

Batch-1 evidence says compute is not the batch-1 bottleneck, but at high batch the
matmuls dominate again, so these matter *at scale*:

- **Quantization floor.** FP8 is the compression floor on trn2 (MXFP4 is Trn3-only).
  The routed experts are already handled; auditing whether attention and the shared
  expert use the cheapest legal path at high batch is worth a pass.
- **MoE kernel choice at batch.** `moe_tkg` was throughput-neutral at batch 1 but
  is structurally better at batch>1 (it walks local experts once; the CTE path pays
  block-padding that only amortises with tokens). Re-measure `tkg` vs `cte` at
  batch 8/16/32 -- the ranking may flip, and it's a ready env toggle.
- **Kernel fusion where the profile points.** The hyper-connection NKI fusion
  kernels in this repo are built and bit-exact but not yet wired in; a decode
  megakernel (2.2) is the higher-value fusion. Neither is worth doing before a
  real device profile says so.

---

## Tier 5 -- Infrastructure that unblocks everything else

These are not throughput themselves; they are the instruments, and two are
currently broken in ways that cost real time.

- **Device profiling is blocked on this SDK.** `neuron-explorer` is in the
  container and the capture env is correct, but the session trace only flushes on
  clean NRT teardown and the vLLM workers hard-crash on teardown, so no trace ever
  lands. Until teardown is clean (or a newer SDK), attribution stays evidence-based
  rather than trace-based. Fixing worker teardown is the unlock for all
  profile-guided work in Tiers 2 and 4.
- **Measure over long generations, always.** Short generations are biased, not just
  noisy: the same build read 8 / 0.8 / 3.6 tok/s at 4-token and 128-token windows,
  and the truth was the long one. The run-to-run floor is ~+/-6% at 32 tokens and
  ~+/-1% at 128. Any A/B below the floor is unresolvable -- establish the floor
  before comparing.
- **Fail loud, never fall back to a plausible default.** Three separate silent
  fallbacks cost runs and conclusions here: a stale shared-storage copy winning
  over the launch payload, a defaults line clobbering passed-in values, and a
  128 KB command-size limit failing with no log. Each now fails loudly or is
  guarded. This is the cheapest reliability investment there is.

---

## The honest bottom line

- **Do Tier 1 first and almost exclusively** until batching works. It is the only
  order-of-magnitude lever, it is close (a graph-break experiment is running), and
  everything else is a rounding error until it lands.
- **Then Tier 2.1** (delete ~86 collectives/step) as the cleanest post-batching win.
- **Treat Tier 3 (MTP) as a research bet**, sequenced after batching, not before.
- **Fix worker teardown (Tier 5)** in parallel -- it is what turns the profiling
  from blocked into a guide for Tiers 2 and 4.

Batch 1 at ~6 tok/s is a correctness demo. Batching is the serving story, and it is
one deadlock away.
