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

That 28x is the headline: **batch_size is the dominant term.** This native path was
stuck at batch 1; it now scales to a **5x aggregate at batch 8** (Tier 1 below), the single
biggest change in this document. It does not yet reach the full 28x -- the peak is at batch 8
and batch 32 is HBM-bound -- but the order of magnitude has moved.

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
3. **Batch>1 is now unblocked -- and the cause was not the lm-head.** The symptom
   (a 32-rank hang at a vocab-sharded lm-head `all_reduce`) pointed the wrong way.
   The real blocker was a dynamic-shape lowering failure: `S_decode = tokens // B`
   in the decode attention produced an unbacked SymInt the compiler cannot lower at
   batch>1. Forcing static shapes (`VLLM_NEURON_V4_STATIC_SHAPES=1`) plus `ep_degree=8`
   fixed it -- batch 8 runs at 30.84 tok/s aggregate (5x over batch 1), golden token
   matched. Replicating the lm-head, the originally-planned fix, was a dead end that
   regressed batch 1.

---

## Tier 1 -- Unlock batching  ·  DONE (5x at batch 8), ceiling found

This was the whole game, and it is now unblocked. Measured curve (full 43 layers, TP=32,
EP=8, `moe_cte`, 128 tokens, golden 4256 verified on every row that runs):

| batch | decode tok/s (aggregate) | vs batch 1 |
|---|---|---|
| 1  | 6.2   | 1.0x |
| 8  | 30.84 | 5.0x |
| 16 | 22.89 | 3.7x |
| 32 | does not fit (OOM) | -- |

### What actually unblocked it -- not the lm-head

The symptom -- a 32-rank hang at a vocab-sharded lm-head `all_reduce`, with
`ShardIndexInjection: conflicting shard indices` in the log -- pointed at the collective, and
the original plan chased it there. Both attempts were wrong:

- **`VLLM_NEURON_DEBUG_MODE=1` (fullgraph=False) -- tried, did not help.** Still hung at the
  same point.
- **Replicating the lm-head -- tried, dead end.** It removed the vocab collective but
  *regressed batch 1*: the same hang reappeared at batch 1, because removing one collective
  just moved the compile skew to the next.

The real cause, found by dropping to a 4-layer slice (loads in minutes) that turned the hang
into a clean error, was a **dynamic-shape lowering failure**. `S_decode = tokens // B` in the
decode attention is a floordiv of two symbolic sizes, so it lowers to an unbacked SymInt
(`INT64_MIN`) the Neuron compiler cannot resolve at batch>1. The fix keeps shapes static:

1. `automatic_dynamic_shapes = False` + `assume_static_by_default = True` so the tracer never
   marks the batch/token dim symbolic (one NEFF per enumerated bucket).
2. `view(B, -1, hidden)` instead of `view(B, tokens // B, hidden)` at the decode reshapes --
   identical value, no floordiv SymInt.

Both behind `VLLM_NEURON_V4_STATIC_SHAPES=1` (default off). `ep_degree=8` is required
alongside: it avoids `NCC_ITIN902` and drops resident HBM enough to fit batch>1 at all.

### What still limits it

- **Aggregate throughput peaks at batch 8 and regresses at batch 16.** The context-encoding
  MoE kernel does super-linear work as tokens-per-step rise (block count grows with a
  `+ (E_local - 1)` floor). Pushing the peak out needs the batch-aware token-generation
  kernel (`VLLM_NEURON_V4_DECODE_MOE=tkg`), correct at batch 1 and untested at batch > 1.
- **Batch 32 does not fit** at 43 layers -- OOM (`NRT_RESOURCE`, ~25.5 of 25.8 GB per core) at
  every KV-block count tried. A memory-capacity wall; the levers are weight quantization or a
  different parallelism layout, not a larger bucket.

The order-of-magnitude change this tier promised is realised in part (5x, not the full 28x
the XLA path reaches at batch 128), and the two things blocking the rest are named and
measured rather than mysterious.

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
