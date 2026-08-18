# Optimization Stages

The ordered pipeline the optimizer walks for each model. Cheapest and safest
first; most speculative last.

Two principles govern the ordering:

1. **Cheap before expensive.** A config change costs one compile. A novel
   kernel costs design, implementation, debugging, and carries real risk of
   subtle numerical error. Exhaust config space first.
2. **Borrow before invent.** Reference implementations encode years of
   collective tuning. Steal that first. Only write something new where
   nothing exists to steal, or where the stolen version underperforms.

```
  Stage 0    BASELINE        autoport output, verified + measured
                |
                v
  Stage 0.5  HARVEST         mine the aws-neuron corpus + bank for existing
                |            working kernels. Read-only, no compiles.
                |            Emits the candidate inventory + unmatched-op queue.
                v
  Stage 1    CONFIG          no new code — TP/dtype/batching/kernel selection
                |
                v
  Stage 2    KNOWN KERNELS   swap in harvested + bank kernels
                |
                v
  Stage 3    BORROW          port patterns from vLLM/SGLang/TRT-LLM/FA
                |
                v
  Stage 4    INVENT          write novel NKI guided by profile + roofline
                |             (only wins if it beats Stage 3 by margin)
                v
  Stage 5    GRAPH REWRITE   fusion, reordering, layout — highest risk
```

**Stage 0.5 is free and it front-loads everything else.** Full design in
[`harvest-corpus.md`](./harvest-corpus.md). The short version: `nki-library`
already ships production kernels for attention (including KV-parallel
segmented attention with online softmax merging — i.e. context parallelism),
MoE prefill/decode, fused QKV, RoPE, RMSNorm-quant, and Mamba-style scan
kernels. Plus `nki-moe` is a corpus of competition-optimized MoE kernels.

The `auto_research` run hand-built context parallelism in hour eleven. The
library already had it. Harvest exists so that does not happen to us.

Each stage is gated: equivalence must pass, guardrails must hold, and the
result must beat the incumbent before it is promoted. A stage that produces
nothing better leaves the incumbent in place and emits an `anti_pattern`
lesson recording what was tried.

---

## Stage 0 — Baseline

**Input**: HuggingFace model id
**Output**: a working, equivalence-verified NxDI or vLLM-Neuron implementation
**Cost**: one autoport run + one compile + one measurement

Produced by the existing autoport agent. This is the floor — every later
stage is measured as a delta against it.

If Stage 0 fails, stop. There is nothing to optimize until the model runs
correctly.

Record: baseline measurements on the search-time probe shape, full toolchain
stamp, HBM peak.

## Stage 1 — Config search (no new code)

**What changes**: only configuration. No source is written or modified.
**Cost**: one compile per candidate
**Risk**: low — every axis here is a supported, tested path

Axes:

| Axis | Values |
|------|--------|
| `tp_degree` | 1, 2, 4, 8, 16, 32 |
| `cp_degree` | 1, 2, 4 |
| `weights_dtype` | fp32, bf16, fp8, int8-w8a8 |
| `activations_dtype` | bf16, fp8 |
| `kv_cache_dtype` | bf16, fp8 |
| `attention_kernel` | whichever variants the stack already ships |
| `batching` | static, dynamic, continuous |
| `sequence_layout` | contiguous, paged, prefix-cached |

This is where bank `config_prior` lessons do the most work — a good prior can
land within a few percent of the Stage-1 optimum on the first candidate.
Anti-patterns prune here too, before any compile happens.

**Exit when**: no-improvement streak on config axes, or all single-axis moves
tried.

**Calibration, corrected against real data.** An earlier version of this doc
claimed most of the total speedup lands here. The
`internal-prior-optimization-run` results contradict that: param
tuning gave **+19%**, model-code changes gave **+405%**, and structural TP
changes (Local-Q, Context Parallel) gave **+193%** on the same model. Config
is the cheapest stage and worth doing first, but on an unoptimized baseline it
is *not* the biggest contributor. Expect single-digit-to-low-double-digit
percentages here and plan for the real wins in Stages 3-5.

Caveat on that data: their config phase had a 1-hour budget against a very
unoptimized baseline (0.28% MFU), so it is not a controlled comparison. But it
is enough to retire the "config carries most of it" assumption.

## Stage 2 — Known kernel substitution

**What changes**: swap already-proven NKI kernels in for hot ops.
**Cost**: one compile per swap
**Risk**: low-medium — kernels are proven, but not necessarily in *this*
model's shape regime

Profile the Stage-1 winner. For each hot op, query the bank for
`nki_kernel` lessons whose op signature and shape constraints match. Try
each applicable one.

Critical detail: a kernel proven at `head_dim=128, seq=4k` is not
automatically correct or fast at `head_dim=64, seq=64k`. The bank's shape
constraints must be checked, not assumed. A kernel applied outside its
validated shape range is a Stage-4 experiment, not a Stage-2 substitution.

## Stage 3 — Borrow ("steal")

**What changes**: port an optimization pattern from a reference implementation
into NKI.
**Cost**: agent implementation time + compile + equivalence + measure
**Risk**: medium — new code, but the *algorithm* is battle-tested

### Procedure

1. Profile the Stage-2 winner. Identify the top-N hot ops and classify each
   bottleneck: compute-bound, DMA/bandwidth-bound, sync-bound, or
   layout/transpose overhead.
2. For each hot op, search the reference corpus for how others solved it:

   | Source | License | Strong at |
   |--------|---------|-----------|
   | vLLM | Apache 2.0 | Paged attention, KV management, continuous batching, prefix caching |
   | SGLang | Apache 2.0 | Radix cache, structured decoding, batching heuristics |
   | TensorRT-LLM | Apache 2.0 + Nvidia terms (**review first**) | Fused kernels, quantization-aware fusion |
   | FlashAttention | BSD-3-Clause | Block-wise attention, softmax numerics |
   | HF transformers | Apache 2.0 | Reference semantics for equivalence |

3. Port the pattern to NKI. Direct code borrowing is permitted for
   Apache-2.0/BSD sources — see `open-questions.md` Q4 for the required
   `THIRD_PARTY_NOTICES` entry and per-file provenance header.
4. Emit a `reference_translation` lesson recording the mapping, with
   `source_references` populated. Mandatory, not optional.

### Why this comes before invention

A CUDA paged-attention kernel represents thousands of engineer-hours and
millions of production runs. The *algorithm* is sound even though the
*implementation* is CUDA-specific. Porting the algorithm and re-expressing it
in NKI captures that work. Writing something novel from scratch and hoping to
beat it is a bad first bet.

## Stage 4 — Invent

**What changes**: a novel NKI kernel, designed from profile data and hardware
characteristics rather than copied from a reference.
**Cost**: highest — design, implement, debug, validate
**Risk**: highest — novel numerics are the most likely source of subtle,
hard-to-detect error

### When Stage 4 triggers

Only under one of these conditions:

- **No reference exists.** The op is Neuron-specific, or the pattern has no
  analog in the reference corpus (e.g. something specific to Gated DeltaNet
  on this hardware).
- **The borrowed version underperforms.** Stage 3 produced a working kernel,
  but profile analysis shows it is still far from the roofline bound.
- **A reference pattern does not map.** The CUDA approach depends on hardware
  features Neuron does not have (warp shuffles, specific shared-memory
  behavior), so a faithful port is impossible and a different algorithm is
  required.

### How invention is directed (not random)

Invention is guided by three inputs, not guesswork:

1. **Profile data** — which engine is the bottleneck, what the gap is
   (idle time, inefficiency, excess DMA traffic, transposes). This says
   *what* to fix.
2. **Roofline bound** — computed from hardware characteristics (PE array
   dimensions, SBUF capacity, DMA bandwidth). This says *how much room
   exists*. If measured is already at 85% of roofline, invention has little
   headroom and Stage 4 should skip.
3. **Bank aggregates** — which tiling strategies, loop orders, and buffer
   allocation patterns have historically worked on Neuron for this op class.

Candidate variants are generated along known axes: tiling strategy, loop
order, buffer allocation, instruction selection, DMA batching. Each attempt
records *why* it was tried, so the resulting lesson is meaningful rather than
"this random thing was faster."

### The promotion rule

**An invented kernel only replaces the borrowed one if it beats it by a
margin.**

```
promote_invented = (
    invented.equivalence_passed
    and invented.perf > borrowed.perf * (1 + INVENTION_MARGIN)
)
```

`INVENTION_MARGIN = 0.05` (5%).

Two reasons the margin exists, and only one is about noise:

1. **Measurement noise.** Our stopping criteria treat <2% as noise. A margin
   below that is meaningless.
2. **Risk-adjusted preference.** Borrowed code has been exercised by
   thousands of users across many shapes and edge cases. Our invented kernel
   has been tested by us, on our benchmark shapes, today. On a near-tie,
   prefer the better-tested artifact. The 5% margin is the price of
   accepting the higher maintenance and correctness risk of bespoke code.

If the invented kernel loses, it is **not** discarded silently — it becomes a
`provisional` lesson recording the attempt and why it lost. That is
information the next model benefits from.

## Stage 5 — Graph rewrites

**What changes**: the computation graph itself — operator fusion, reordering,
memory layout changes.
**Cost**: medium-high
**Risk**: highest for equivalence, because changing operation order changes
floating-point accumulation

Examples: RMSNorm→attention fusion, RoPE folded into QKV projection,
eliminating redundant transposes, layout changes to avoid DMA.

Last because these interact with everything before them. A fusion that helps
at TP=8/bf16 may hurt at TP=16/fp8, so it must be evaluated against the
already-settled config rather than searched jointly.

---

## Tournament structure

Within each stage, candidates compete. Between stages, the incumbent carries
forward.

```
incumbent = stage_0_baseline

for stage in [1, 2, 3, 4, 5]:
    candidates = stage.generate(incumbent, bank, profile)
    candidates = bank.prune_antipatterns(candidates)      # zero-cost filter

    for cand in candidates:
        if not guardrails.check(cand):                    # HBM, compile timeout
            bank.add_negative(cand, reason); continue
        neff = compile(cand)
        if not equivalence.check(neff):                   # HARD gate
            bank.add_negative(cand, "equivalence"); continue
        perf = measure(neff, probe_shape)

        margin = INVENTION_MARGIN if stage == 4 else 0.0
        if perf > incumbent.perf * (1 + margin):
            incumbent = cand
            bank.add_positive(cand, perf)
        else:
            bank.add_negative(cand, "no improvement")

    if stage.no_improvement_streak >= 5: break

full_sweep(incumbent)     # all shapes x full batch sweep — publish this
```

Note that the invention margin applies **only at Stage 4**. Everywhere else,
any improvement above the noise floor promotes.

---

## Measuring whether it actually invents anything

The open question worth instrumenting: does the optimizer ever produce
something genuinely new, or does it only ever recombine borrowed work?

Early rounds will be borrow-dominated, and that is expected — the bank is
empty and the reference corpus is rich. Invention should rise over time as:

- easy borrows get exhausted
- the bank accumulates Neuron-specific knowledge that references lack
- profile analysis gets more precise about where the real headroom is

Metrics to track per run and in aggregate:

| Metric | Definition | What it tells us |
|--------|-----------|------------------|
| `borrow_rate` | Stage-3 kernels promoted / total kernels promoted | How much we are standing on others' work |
| `invention_rate` | Stage-4 kernels promoted / total kernels promoted | How often novel work wins |
| `invention_attempt_rate` | Stage-4 attempted / models optimized | How often we even try |
| `invention_win_rate` | Stage-4 promoted / Stage-4 attempted | Whether invention is productive or flailing |
| `invention_margin_actual` | Mean speedup of promoted Stage-4 vs. the Stage-3 it beat | Whether wins are marginal or substantial |
| `roofline_attainment` | measured / roofline bound for the final kernel | Whether headroom remains at all |

Expected trajectory, stated as a falsifiable prediction:

- **Models 1-5**: `invention_rate` near zero. Borrowing dominates.
- **Models 5-20**: `invention_attempt_rate` rises as easy borrows run out.
  `invention_win_rate` likely low at first.
- **Models 20+**: if `invention_win_rate` climbs, the bank is teaching the
  optimizer something the references do not contain. That is the interesting
  result.

If `invention_win_rate` stays flat near zero past ~30 models, that is a real
finding too: it would mean the reference corpus is sufficient and Stage 4 is
not earning its cost. Worth knowing either way — and either outcome is
publishable.

---

## What this means for the knowledge bank

`nki_kernel` lessons need a provenance discriminator so the metrics above are
computable:

```yaml
type: nki_kernel
provenance:
  origin: borrowed | invented | hybrid
  # if borrowed or hybrid:
  source_references:
    - repo: https://github.com/vllm-project/vllm
      commit: a1b2c3d4
      license: Apache-2.0
      what_was_taken: "block-wise KV indexing scheme"
  # if invented:
  invention_rationale: >
    No reference handles Gated DeltaNet recurrent state on a systolic
    array. Designed from the roofline bound: DMA-bound at 34% of peak
    bandwidth, so restructured to batch state reads across 4 timesteps.
  beat_borrowed_by: 0.12          # required if origin=invented and a
                                  # borrowed alternative existed
```

`origin: hybrid` covers the common real case — a borrowed algorithm with
substantial Neuron-specific restructuring. Being able to distinguish these
three is what makes the invention metrics honest rather than
self-congratulatory.

---

# Execution discipline

Adopted from `internal-prior-optimization-run`, which ran this loop
autonomously for 12 hours and produced a large (multiple-x). These are not stylistic
preferences; each one prevents an observed failure mode.

## Phase budgets with scoped write permissions

Each stage gets a time budget **and** an explicit set of files it may modify.
The scoping is the important half — during config tuning the agent physically
cannot start rewriting kernels, which is what stops rabbit-holing.

| Stage | Budget | May modify |
|-------|--------|-----------|
| 0.5 Harvest | 30 min | **nothing** — read-only, emits a manifest |
| 1 Config | 1 h | serving config, env vars, launch scripts |
| 2 Known kernels | 1 h | kernel selection only (which existing kernel, not its source) |
| 3 Borrow | 3 h | the vLLM-Neuron / NxDI fork — any file |
| 4 Invent | 6 h | the NKI kernel fork — any file |
| 5 Graph rewrite | 2 h | fork-level graph passes, fusion config |

Auto-advance on budget exhaustion. Do not stop to ask.

## Never modifiable

- **The benchmark harness, its configs, and the baseline reference outputs.**
  The agent must not be able to modify its own grader. This is the
  reward-hacking guard, and the ADAS survey names reward hacking as a live
  safety problem in exactly this class of system.
- The container / environment definition.
- The agent's own instruction file.

## Git as the state machine

```
KEEP     if metric improved AND correctness >= threshold   → branch advances
DISCARD  if correctness < threshold                        → git reset --hard HEAD~1
DISCARD  if metric equal or worse                          → git reset --hard HEAD~1
ALWAYS   append a row to results.tsv, keep or discard
```

The branch head *is* the incumbent. History is the trace. Every ledger row
carries its commit hash, so every point on the trajectory chart links to a
real diff.

## Three measurement tiers

Refines the two-tier scheme in `guardrails.md`. The middle tier exists
specifically because it exposes long-context attention behavior at
fast-compile cost — the regime where the real bottlenecks live.

| Tier | Use | Cost |
|------|-----|------|
| **fast** | Param-only changes, quick iteration | Minimal compile |
| **medium** | **The loop default.** Long-context behavior, fast compile | Moderate |
| **full** | End-of-run validation only | Expensive |

Score on the **average over the last 50% of turns/positions**, not the overall
mean. This deliberately weights long-context steady-state; averaging over all
turns lets strong early-turn numbers mask attention-scaling collapse.

## Prompt rules that earned their place

Taken close to verbatim, because each addresses an observed failure:

- **Never stop.** Run the full budget autonomously. The human may be asleep.
  Ask nothing mid-loop.
- **Mix small and large.** Alternate quick parameter tweaks with bigger
  structural changes. Small wins compound, but do not get stuck only tuning
  knobs. *(Observed failure: the reference agent fixated on MoE for 12 hours
  before a human redirected it to attention, where the real wins were.)*
- **Learn from CUDA.** Named techniques to port, not a vague instruction:
  FlashAttention (tiling + online softmax), FlashDecoding (KV-parallel split
  for long context), PagedAttention (block-sparse KV), Ring Attention
  (sequence-parallel), MegaBlocks MoE, Triton patterns. Where a technique is
  well-proven on GPU, work out how the principle maps onto NeuronCore's
  SBUF / PSUM / DMA model.
- **Question framework defaults.** Much of the stack was designed for GPUs.
- **Deleting code for equal performance is the best kind of win.**
- **Think harder when out of ideas.** Re-read the architecture, read the
  source to find bottlenecks, and ask what is fundamentally different about
  Neuron versus GPU that the current code fails to account for.
- **Kill any experiment exceeding 30 minutes** (compile + run) and record it
  as a crash.

## Expectation calibration for Stage 4

From the reference repo's own stated limitations:

> AI agents excel at parameter tuning, calling existing modules, and writing
> small patches. Writing 4000-line flash attention from scratch in a single
> session is beyond current capability.

So Stage 4 should target **patch-scale invention** — restructure a tiling
loop, change a buffer allocation strategy, fuse two ops — not
subsystem-scale rewrites. Set the roofline gate accordingly: if closing the
gap requires a new 4000-line kernel, Stage 4 should decline and record why.

This is also the strongest available argument for borrow-before-invent. It is
not a philosophical preference; it is a capability observation from someone
who ran the experiment.
