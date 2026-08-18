# Open Questions

Load-bearing decisions that block phase 1 or shape everything after. Each
has a proposed default so the plan is actionable, but confirm before
building.

## Q1. Search strategy for phase 1 — RESOLVED

**Decision: hand-priored greedy.**

Perturb one config axis at a time, ordered by expected impact from the
bank's priors. Cheap, interpretable, debuggable — you can read the decision
log and understand why it went where it went.

Bayesian optimization and learned policies stay in Phase 4, once the
leaderboard has generated enough measurements to train a cost model
against. The proposer sits behind a single interface
(`proposer.next_k(current, k)`) so swapping it later is contained.

## Q2. Budget representation — RESOLVED

**Decision: no compute budget cap.** Capacity is not the constraint.

Termination is instead governed by search-quality criteria (no-improvement
streak, marginal-improvement threshold, max iterations, search-space
exhaustion) — see [`guardrails.md`](./guardrails.md#stopping-criteria).

The available compute goes to **horizontal parallelism** — many models
optimized concurrently, candidates compiled in parallel within a model —
rather than unbounded depth on any single model. Depth past the plateau
produces noise, not wins.

## Q3. Which seed models for phase 1 — MOSTLY RESOLVED

**Decision: three-model ladder, in this order.**

| Order | Model | Role | Params | bf16 weights | Instance |
|-------|-------|------|--------|--------------|----------|
| 1 | **Gemma 4 31B** | Prove the loop | 31B dense | ~62 GB | trn2.3xlarge (tight) |
| 2 | **Qwen3.8-27B** | Hybrid-attention adapter | 27B | ~54 GB | trn2.3xlarge |
| 3 | **MiniMax M2** | Large MoE at scale | 230B total / 10B active | ~460 GB | trn2.48xlarge |

Weight sizes are rough (params x 2 bytes); verify against actual configs
before sizing instances.

Rationale for the ordering: Gemma 4 31B is a standard transformer
architecture under Apache 2.0, with strong public documentation and a #3
Arena ranking among open models. If the loop misbehaves on it, the loop is
at fault — not the model. That is exactly what we want from model #1.

### Important: Qwen version disambiguation

These are **separate releases**, not variants of one model:

| Release | Date | Sizes | License | Architecture note |
|---------|------|-------|---------|-------------------|
| Qwen3.5 | Feb 2026 | 0.8B / 2B / 4B / 9B + 397B-A17B MoE | Apache 2.0 (small) | Gated DeltaNet + sparse MoE hybrid |
| Qwen3.6 | Apr 2026 | 27B dense, 35B-A3B MoE | Apache 2.0 | 1M-token context |
| Qwen3.7 Max | May 2026 | — | **Closed weights** | Unusable for us |
| Qwen3.8 | Jul-Aug 2026 | 27B + 2.4T-A95B Max | Apache 2.0 | Built on Qwen3.5 foundation |

Two consequences:

1. **"Qwen3.8" the flagship is 2.4T params / 95B active.** It does not fit
   on a trn2.3xlarge or anything close to it. Use `Qwen3.8-27B`.
2. **Qwen3.5/3.8 use Gated DeltaNet** — a linear-attention variant
   alternating with full attention at roughly 3:1. This is *not* standard
   attention. Consequences:
   - The `dense_causal_lm` adapter will not cover it
   - Linear attention manages state differently than a KV cache, so memory
     modeling and the `stress` shape behave differently
   - Needs its own adapter: `hybrid_attention_causal_lm`

   This makes Qwen a genuinely valuable generality test — and a poor choice
   for model #1.

### Still open

- **"Muse"** — needs disambiguation. Microsoft's Muse (game world model)?
  A music-generation model? Something else? Not yet slotted into the ladder.
- **MiniMax version pinning** — they ship fast (M2 -> M2.1 -> M2.5 -> M2.7,
  plus M3, H3 video, Music 3). Decide: pin a version, or track "latest text
  model" dynamically via the discovery job. Note H3 uses a custom Community
  License Agreement rather than Apache 2.0 — the license filter must catch
  that class of model.
- **`stress` shape coverage** — at least one seed must genuinely support
  128k+ positions, or the 64k/64k shape reports `not_applicable` on all
  three and never actually gets exercised. Gemma 4's 256K context covers
  this.

## Q4. Licensing stance for pattern borrowing — RESOLVED (with mechanics)

**Decision: direct code borrowing permitted, with attribution mechanics.**

Important correction to the reasoning that got us here: **running on
Trainium has no bearing on software license obligations.** Direct borrowing
is fine because the *licenses* permit it, not because of the target
hardware. That distinction matters because it tells us which sources are
actually safe.

### Source-by-source

| Source | License | Direct borrowing | Obligations |
|--------|---------|-----------------|-------------|
| vLLM | Apache 2.0 | Yes | Retain copyright notices, include license text, state changes, propagate NOTICE |
| SGLang | Apache 2.0 | Yes | Same as above |
| FlashAttention | BSD-3-Clause | Yes | Retain copyright + disclaimer |
| TensorRT-LLM | Apache 2.0 + some Nvidia terms | **Review first** | Some components carry terms differing from the top-level license |
| HF transformers | Apache 2.0 | Yes | Standard Apache attribution |

### Required mechanics (build these in from day one)

1. **`THIRD_PARTY_NOTICES` file** at repo root, updated on every borrow.
   Lists source, license, and what we took.
2. **Provenance header** on any file containing borrowed code:

```python
# Portions adapted from vLLM (https://github.com/vllm-project/vllm)
# Commit: a1b2c3d4
# License: Apache-2.0 (see THIRD_PARTY_NOTICES)
# Changes: ported CUDA paged-attention indexing to NKI; replaced
#          warp-level reduction with nl.matmul-based accumulation.
```

3. **`source_references` field** on every `reference_translation` lesson in
   the knowledge bank (already in the schema — now mandatory, not optional).

Cost: roughly 10 minutes per borrow. Benefit: we never have to untangle
provenance retroactively, which is a genuinely expensive exercise once a
codebase is large and public.

### Flagged for human review

**TensorRT-LLM specifically.** Nvidia ships some components under terms
that differ from the repo's top-level Apache 2.0. Worth ~20 minutes from
whoever owns open-source compliance *before* we borrow from that repo. All
other sources listed above are clean.

### Separate issue: model licenses != code licenses

For the leaderboard we also *run* models and publish benchmark numbers.
Some model licenses restrict benchmarking or publication (e.g. MiniMax H3's
custom Community License Agreement). The discovery job's license filter
handles this — see `plan.md` phase 3 filters. Do not conflate the two.

## Q5. Compile-time reduction strategy

Every candidate compile costs ~5-20 min on Trn2. If phase 1's optimizer
tries 30 candidates per model, that's 2.5-10 hours per model just in
compiles.

**Options:**
- (a) NEFF cache keyed on config subgraph hash — reuse when possible
- (b) Compile in parallel across multiple trn2 instances
- (c) Cost-model prediction to skip compiles we know will be worse
- (d) All of the above eventually

**Proposed default: (a) and (b) for phase 1; (c) is phase 4.** Cache
handles the "small delta, most graph unchanged" case; parallelism handles
the "unrelated candidates" case.

## Q6. Equivalence tolerance per family — NEEDS NUMERICS REVIEW

### What this question actually is

When the optimizer makes a model faster, it does so by **changing the
math**: lower precision (bf16 instead of fp32), a kernel that sums values
in a different order, a fused op that skips an intermediate rounding step.

All of those produce *slightly different numbers* than the original. Not
wrong — just not bit-identical. So we have to answer, for every candidate:
**how different is too different?**

Concrete example. The original model, given a prompt, assigns probability
0.847 to the next token being "the". Our optimized version says 0.848.
Nobody cares — same token wins, same output.

Now suppose it says 0.6 and a *different* token wins. That model has
different behavior. It may still sound fine in casual testing, and it may
be 3x faster, but we would be publishing a recipe that silently degrades
output quality. That is the failure this gate prevents.

An **equivalence tolerance** is where we draw that line. Two parts:

- **Numerical** — how far raw values may drift. `rtol=1e-3` means "within
  0.1% relative difference."
- **Behavioral** — does it actually *do* the same thing. For an LLM: given
  the same prompt, does greedy decoding select the same tokens across 100+
  positions?

Per-family, because "same thing" means something different per model type:

| Model type | "Same behavior" means |
|------------|----------------------|
| LLM | Same tokens chosen |
| Diffusion | Perceptually identical image (LPIPS — pixel-exact is impossible and unnecessary) |
| Speech | Same transcription, or WER within a small delta |
| Embeddings | Vectors point the same direction (cosine similarity > 0.999) |

### Proposed defaults

Checked into each adapter (`leaderboard/adapters/*.py`), explicit rather
than buried inside the equivalence agent:

| Family | Numerical | Behavioral |
|--------|-----------|------------|
| Dense causal LM | `rtol=1e-3, atol=1e-5` | Top-1 token match on greedy, 100+ positions |
| MoE causal LM | Same, on final logits | Allow routing divergence if logits agree |
| Hybrid attention (Gated DeltaNet) | `rtol=1e-3, atol=1e-5` | Top-1 match; watch recurrent state drift over long sequences |
| Diffusion | — | LPIPS / FID window vs. reference image |
| Speech | — | WER within delta on a fixed benchmark set |
| Encoder-only | `rtol=1e-4, atol=1e-6` | Cosine similarity > 0.999 |

### What still needs a human

The numbers above are conservative enough to **build against** — this does
not block starting. But get a numerics review before publishing results
publicly, because the failure is asymmetric:

- **Too loose** → we publish a "fast" recipe that quietly produces worse
  output. Nobody notices until someone else does, in public.
- **Too tight** → we reject valid configs and the optimizer plateaus early
  for no real reason. Wasteful but visible and fixable.

Two rows I am least confident about:

1. **MoE** — expert routing is inherently somewhat stochastic. Some
   divergence is normal and expected. "How much divergence before the model
   is genuinely different" is a judgment call needing someone who knows the
   numerics.
2. **Hybrid attention** — linear-attention recurrent state can accumulate
   drift over long sequences in a way standard attention does not. At the
   `stress` shape (64k output) this could compound. Needs empirical
   characterization, not a guessed constant.

## Q7. Definition of "top 100" — RESOLVED

**Decision: multi-source weekly ingestion.** A discovery job pulls ranked
model lists from several sources every week, unions them, and re-ranks.

Sources (all machine-readable):
- **LMArena / LMSYS Chatbot Arena** leaderboard — community quality signal
- **HuggingFace trending** — momentum signal
- **HuggingFace downloads, last 30 days** — actual-usage signal
- **HF Open LLM Leaderboard** — benchmark-score signal
- **AWS customer signal** — manually maintained list of what customers ask
  for (this is the one that keeps us commercially relevant)

Ranking: weighted borda count across sources, with the customer-signal list
given an override flag so a specifically-requested model always makes the
cut regardless of public popularity.

Filters applied after ranking:
- Open weights (we must be able to download and run it)
- License permits benchmarking + publishing results
- Inference-oriented checkpoint (skip training-only artifacts)

Weekly cadence means the list churns. Handle it by keeping entries "warm"
— a model that drops off the list keeps its published recipe, just stops
getting refreshed. See `plan.md` phase 3 for the discovery component.

## Q8. Leaderboard baseline comparison

To be trustworthy, our numbers need a comparison point. Candidates:

- (a) H100 with vLLM 0.6+ (public, easy to reproduce)
- (b) H200 with TensorRT-LLM (Nvidia's best, harder to reproduce)
- (c) A100 with vLLM (older, wider deployment)
- (d) All three

**Proposed default: (a) for every entry, (b) as a stretch goal per entry.**
H100+vLLM is the fair fight because tooling is comparable. Report perf,
perf-per-dollar, perf-per-watt. If Trainium loses on some models, publish
that too — it's what makes the numbers trustworthy.

## Q9. Where does the bank live?

Options:
- (a) In this repo, part of the same codebase
- (b) Separate repo (public, so it can be a citable artifact)
- (c) A database + service (queryable API)

**Proposed default: (a) initially, (b) once it stabilizes.** Static-site
render from markdown, git as the store. Move to a service only if
retrieval speed becomes a bottleneck (unlikely for lesson counts under
10k).

## Q10. Provisional vs. verified lesson gating

Auto-generated lessons from optimizer runs are cheap and unreliable.
Human-authored are expensive and trusted.

**Proposed policy:**
- All optimizer-generated lessons land in `lessons/provisional/`
- Weekly triage: a human reviews the queue, promotes to `lessons/verified/`,
  edits schema fields as needed, or drops
- Proposer only reads `verified/` in v0
- In phase 4, the proposer can consult `provisional/` with low weight

This keeps the bank trustworthy while still letting the optimizer
contribute.

## Q11. Human-in-the-loop checkpoints

Fully autonomous is a nice narrative but a bad phase-1 default. Where do
humans intervene?

**Proposed checkpoints:**
- After every optimizer run: review the top-3 configs and top-3 emitted
  lessons before publishing
- On any measured regression > 20% from a prior best: alert, don't
  auto-publish
- On equivalence failure with novel error pattern: alert, don't discard
  silently

## Q12. What do we release publicly first?

Options:
- (a) Just the optimizer, "here's how we got X on Qwen 3.5"
- (b) Optimizer + bank, "here's the framework"
- (c) Leaderboard first, "we built this thing, look at the results"

**Proposed default: (c).** The leaderboard is the compelling story for
external audiences. The optimizer + bank are means to that end. Sequencing:

1. Build phase 1 internally
2. Build phase 2 (bank) internally
3. Run phase 3 leaderboard internally against 20-30 models
4. **First public release**: leaderboard + recipes + reproducibility scripts
5. Later: open-source the optimizer + bank once they're stable

Different orderings are defensible — this one prioritizes "impressive
public artifact" over "credit for the framework."

---

## Q13. Build on Autocomp, or build our own? — RESOLVED

**Decision: build our own, taking Autocomp's architecture rather than its
code.**

What we adopt from it (see `references-analysis.md`): beam search over
LLM-generated plans, the plan-then-implement two-phase prompt, the
optimization menu with dropout, and hardware-in-the-loop at every iteration.
Those are the load-bearing ideas and they are proven on Trainium.

Why not build *on* it:

- **Different unit of work.** Autocomp optimizes a single kernel against a
  fixed benchmark problem. We optimize a whole model's serving path, where
  Stage 1 (config: TP, dtype, batching) is a large cheap win Autocomp has no
  concept of.
- **We already own better Neuron-specific agents.** The NAD package's
  `neuron-nki-writer-agent`, `neuron-nki-debugger-agent`, and
  `neuron-nki-profile-analysis-agent` encode internal Neuron knowledge that a
  doc-generated agent will not have.
- **The knowledge bank is our differentiator.** Autocomp demonstrated schedule
  reuse is worth up to 24%; it did not productionize a durable, curated,
  staleness-managed store. That is the part we want to own.
- **Coupling cost.** Adopting their search loop means adopting their
  abstractions for hardware config, eval backends, and problem definitions —
  none of which fit a model-level leaderboard cleanly.

Still worth doing: **run Autocomp once on a single kernel from one of our seed
models**, purely as a calibration baseline. If their `trn2-nki2` agent beats
our NKI agents on the same kernel, that is a signal our Stage 3/4 prompting
needs work. Cheap experiment, useful either way.

## Q14. Serving backend: native PyTorch or vLLM-Neuron? — NEEDS A DECISION

You lean native PyTorch. I think the honest answer is **both, with different
jobs** — and that using native PyTorch as the *leaderboard* backend would
undermine the leaderboard's credibility.

### The case for native PyTorch (real, and stronger than it first looks)

- **It attacks our single worst cost.** Compile time dominates the search
  loop — 5-20 min per candidate. Eager mode needs no compile at all for many
  changes. A correctness-iteration loop that takes seconds instead of minutes
  is transformative for Stage 3/4 development velocity.
- **Debuggability.** No XLA graph opacity. Errors surface where they happen.
- **`torch.compile(backend="neuron")`** gives a compiled path when you want
  performance, so it is not eager-only.
- **It is the strategic direction** for the stack.

### The case against it as the leaderboard backend

- **Nobody serves production LLM inference in PyTorch eager.** A leaderboard
  that says "Gemma 4 31B does N tok/s on Trainium" has to reflect how the
  model would actually be served, or the number misleads. This is the decisive
  argument.
- **The optimization surface is thinner.** vLLM-Neuron already has continuous
  batching, paged attention, prefix caching, and a scheduler — all of which
  are things to *tune*. Native PyTorch has none of that; we would be building
  serving infrastructure rather than optimizing it.
- **The one proven result used vLLM-Neuron.** `auto_research` hit a large (multiple-x) on it.
  That is the only end-to-end evidence we have that this whole approach works
  on Trainium.
- **Known TP limitations.** Per this workspace's own beta notes, verified on
  Trn1: cross-chip TP (TP>=4) fails at init with `Failed to execute the device
  barrier 1`; only intra-chip TP=2 works. Teardown SIGSEGVs. The container
  must be restarted between TP runs. Our seed models need TP=8.
- **Eager performance does not predict compiled performance.** So the fast
  iteration advantage applies to *correctness* iteration, not to performance
  search — and performance search is the actual job.

### Proposed split

| Backend | Role | Why |
|---------|------|-----|
| **vLLM-Neuron** | **Primary — the leaderboard backend** | Production-representative, proven at a large (multiple-x), rich tunable surface, TP works |
| **Native PyTorch** | Fast-iteration research track + Stage 4 kernel dev environment | Eager mode removes the compile wait for correctness work; better debugging; also the right home for architectures vLLM does not support |
| NxDI | Autoport baseline producer (Stage 0) | Already what `neuron-framework-autoport` targets |

Concretely: develop and validate a kernel in native PyTorch eager where the
loop is fast, then land and measure it in vLLM-Neuron where the number counts.

### Blocking unknown

**Does cross-chip TP work in native PyTorch on Trn2?** The documented failure
is Trn1-specific and the note says a 13B needing TP>=4 is "blocked here → use
a Trn2." If TP>=8 works cleanly on Trn2, native PyTorch becomes far more
viable as a primary and this question should be reopened.

That is a half-day experiment: bring up one seed model at TP=8 on a
trn2.48xlarge under the native-PyTorch DLC and see whether
`init_process_group(backend="neuron")` survives. Worth doing before locking
this in.

### If you want native PyTorch as primary anyway

It is a defensible choice if the framing changes from "leaderboard of serving
performance" to "leaderboard of achievable Trainium performance." That is a
different and more research-flavored product — arguably more interesting, and
it sidesteps the vLLM-version-churn problem entirely. But it needs to be a
deliberate reframing, published as such, not a quiet substitution. Say the
word and I will rewrite the plan around it.
