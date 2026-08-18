# Reference Analysis: Autocomp + AutoResearch

Two prior systems shape this design substantially. Reading them changed
several of our decisions — including one I had gotten wrong.

## 1. Autocomp (UC Berkeley BAR)

- Repo: https://github.com/ucb-bar/autocomp
- Paper: [arXiv 2505.18574](https://arxiv.org/abs/2505.18574) (NeurIPS 2025)
- Authors: Charles Hong, Sahil Bhatia, Alvin Cheung, Yakun Sophia Shao
- Tech report: [EECS-2026-167](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2026/EECS-2026-167.html)

**This is a working, published implementation of a large part of what we
designed — and it already supports Trainium with NKI.** Agents exist for
`trn1-nki1`, `trn2-nki1`, `trn2-nki2`, and `trn3-nki2`. Other targets: TPU
v5e/v6e, CUDA, Gemmini, RISC-V Vector, Apple Metal.

### Their three stated contributions

1. Each optimization pass is a structured **two-phase prompt** — a *planning*
   phase and a *code generation* phase.
2. Domain knowledge is injected during planning via a concise, adaptable
   **optimization menu**.
3. Correctness and performance metrics **from real hardware** feed back at
   every search iteration.

### What we adopt

| Autocomp mechanism | Why we take it |
|--------------------|----------------|
| **Beam search** (`beam_size=4`) | Proven on this exact task across 6 hardware targets. See the revision note below. |
| **Plan-then-implement split** | Plans are cheap text; implementations are expensive code. Generate 8 plans, implement selectively. Also lets a stronger model plan and a cheaper one code (`models` vs `code_models`). |
| **Optimization menu with dropout** (`dropout_menu_options=0.25`) | A curated strategy list, randomly sub-sampled per prompt to force diversity instead of always reaching for the same trick. |
| **`menu_strategy="one-shot"`** | Dynamically generates *new* strategies via an LLM call rather than only picking from the static menu. This is their invention mechanism. |
| **Hardware-in-the-loop every iteration** | No surrogate models, no predicted scores. Real compile, real measurement, real equivalence. |
| **`use_edits`** (structured old_str/new_str JSON) | More reliable than whole-file rewrites once kernels get large. |
| **`early_stop_iters`** | Same idea as our no-improvement streak. |
| **`translate_perf_threshold=15`** | During an initial translation phase, keep candidates up to 15x *worse* if they are structurally correct. Correctness first, speed after — good discipline. |
| **Trace visualizer** | Their VS Code extension does exactly what you asked for. See `trajectory-reporting.md`. |

### The finding that validates our knowledge bank

From the abstract:

> optimization schedules generated from Autocomp can be reused across similar
> tensor operations, improving speedups by up to **24%** under a fixed sample
> budget.

That is third-party empirical support for the core knowledge-bank thesis —
that lessons transfer across models and make later optimizations cheaper.
We are not speculating; someone measured it.

### Revision: beam search, not greedy

**I previously recommended hand-priored greedy for Q1. That was too
conservative and I am revising it.**

The reasoning I gave was "a good greedy with human priors beats a bad
Bayesian on a small budget." That is still true, but it set up a false
choice — the actual alternative is not Bayesian optimization, it is **beam
search over LLM-generated plans**, which is:

- Proven on Trainium/NKI specifically, not just in principle
- Barely more complex than greedy (keep top-k instead of top-1)
- Substantially more robust to local optima, which matters because our config
  axes interact (a fusion that helps at TP=8/bf16 can hurt at TP=16/fp8)

Revised recommendation:

```
search_strategy   = beam
beam_size         = 4        # candidates surviving each iteration
num_plan_candidates = 8      # plans per parent per iteration
num_code_candidates = 1      # implementations per plan
early_stop_iters  = 5        # matches our no-improvement streak
```

We keep the hand-priored part — bank `config_prior` lessons seed the initial
beam and bias the menu. Priors plus beam search, rather than priors plus
greedy.

### Where we differ from Autocomp deliberately

| Dimension | Autocomp | Us | Why |
|-----------|----------|-----|-----|
| **Unit of optimization** | A single kernel | A **whole model** end to end | Autocomp optimizes `nki1` tutorial problems. We start from an autoported model and optimize the full inference path. Different scope. |
| **Config-space search** | Not a focus — it optimizes code | **Stage 1 is config-only** (TP, dtype, batching) | Model-level work has a large cheap config space that kernel-level work does not. Highest ROI, and Autocomp does not cover it. |
| **Persistent cross-run memory** | Schedule reuse shown in the paper; not a durable curated store | **Knowledge bank** with provisional/verified tiers, staleness policy, SDK stamping | They proved reuse works. We productionize it as a durable, human-curated artifact. |
| **Anti-patterns** | Not a first-class concept | Own lesson type, prunes candidates **before compile** | At 5-20 min per compile, zero-cost pruning is a large win. |
| **Invention gating** | Menu one-shot generates new strategies; no explicit borrow-vs-invent accounting | Stage 3 vs Stage 4 with a **5% promotion margin** and provenance tracking | Makes "did it invent anything" measurable rather than anecdotal. |
| **Equivalence** | Correctness checks per iteration | Full **8-stage equivalence pipeline** per family with per-family tolerances | We publish public recipes; a silently-degraded model is our worst outcome. |
| **Scope of publishing** | Research artifact | **Public leaderboard** across top-100 models, refreshed on SDK release | Different product. |

### Open question this raises

**Should we build on Autocomp rather than beside it?** It is Apache-2.0-ish
open source (verify license), already has Trainium NKI agents, and has an
Agent Builder that generates hardware agents from docs.

Two readings:

- **Build on it.** Use Autocomp as the Stage-3/4/5 kernel optimizer and put
  our config search, knowledge bank, and leaderboard around it. Saves months.
- **Build beside it.** Our existing NAD agents (`neuron-nki-writer-agent`,
  `neuron-nki-debugger-agent`, `neuron-nki-profile-analysis-agent`) already
  cover kernel work with Neuron-team-specific knowledge Autocomp lacks.

My read: **evaluate Autocomp on one of our seed models before deciding.** If
its `trn2-nki2` agent performs comparably to our NKI agents on the same
kernel, adopting it is a large accelerant. If our agents win because they
encode internal Neuron knowledge, keep ours and take Autocomp's *search
architecture* rather than its code. Either way, take the beam search and the
plan/implement split. Added as Q13.

## 2. `internal-prior-optimization-run` (private, cloned + read)

**This is the most directly useful reference of the set.** A working 12-hour
autonomous optimization loop for LLM prefill throughput on Trn2, built on
vLLM-Neuron + NKI. It is the closest existing thing to what we are building,
and it has real measured results.

### Its results

| Model | Type | Official config | 128K context | MFU (base→opt) | Correctness |
|-------|------|----------------|--------------|----------------|-------------|
| Tongyi-30B-A3B | MoE | **a large (multiple-x)** | **a large** | 0.28% → 4.93% | 100% |
| GPT-OSS-20B | MoE | **a large** | **a large** | 0.36% → 5.89% | 100% |
| GPT-OSS-120B | MoE | **a large** | ~9.2x | 0.60% → 6.88% | 100% |
| Qwen3-VL-32B | Dense | **2.8x** | 2.4x | 10.36% → 28.87% | 100% |

Two conclusions we should internalize:

1. **MoE gains dwarf dense gains** (11-17x vs 2.8x). The MoE wins came from
   eliminating `all_gather` of activations. Our seed set is currently three
   dense-ish models — we may be picking the *hardest* case to show value on.
2. **Even after a large (multiple-x), MFU is ~5%.** The headroom remaining is enormous.
   That reframes what "optimized" means and argues strongly for tracking
   roofline attainment rather than just speedup multiples.

### Its round structure (real numbers)

| Round | Focus | Medium 32K | Full 128K | MFU |
|-------|-------|-----------|-----------|-----|
| 1 | Param tuning (segment size, KV dtype, block size) | 712 → 845 (+19%) | 258 → 365 (+41%) | 0.33% |
| 2 | Model code (GQA broadcast, BF16 attn) + NKI flash_attention | 845 → 4,269 (**+405%**) | 365 → 1,533 (+320%) | 1.68% |
| 3 | Context Parallel + Local-Q | 4,269 → 12,503 (+193%) | 1,533 → 6,200 (+304%) | 4.93% |

**This corrects an assumption I had wrong.** I wrote in
`optimization-stages.md` that "most models will get the majority of their
total speedup from Stage 1 [config]." Their data says otherwise: param tuning
gave +19%, model code gave +405%, and structural TP changes gave +193%. Config
tuning was the *smallest* contributor.

Caveat: their Phase 1 had only a 1-hour budget and their baseline was
extremely unoptimized (0.28% MFU), so it is not a clean controlled comparison.
But the claim "config carries most of the win" is not supported and I have
removed it.

### Phase structure with budgets and file scoping

The single best structural idea in the repo:

| Phase | Budget | Editable |
|-------|--------|----------|
| 1: Params | 1 h | `run/config.env`, `run/serve_*.bash` |
| 2: Model code | 3 h | Any file in the vLLM-Neuron fork |
| 3: Kernels | 8 h | Any file in the nkilib fork |

**Read-only, never modifiable**: `benchmark/` (the judge program, configs, and
baseline logits), the container config, and `program.md` itself.

Two things this buys:

- **The agent cannot modify its own grader.** That is a reward-hacking
  prevention we did not have explicitly. Add it.
- **Phase-scoped write permissions prevent rabbit-holing.** During param
  tuning the agent physically cannot start rewriting kernels.

Their stated insight confirms it: *"A clean, constrained directory structure
(explicit editable vs read-only) is critical for productive auto-research."*

### Keep/discard, with git as the state machine

```
KEEP     if tok_per_s improved AND correctness >= 99.0%
DISCARD  if correctness < 99.0%  (correctness violation)
DISCARD  if tok_per_s equal or worse
DISCARD  = git reset --hard HEAD~1
```

Plus: **log every experiment to `results.tsv` regardless of outcome.** 184
rows for one model. Each row carries the commit hash, so every point on their
chart links to a real diff.

Using git reset as the discard mechanism is elegant — the branch *is* the
incumbent, and history is the trace.

### Three measurement tiers (better than my two)

| Tier | Config | Purpose |
|------|--------|---------|
| fast | TP=4, 16K, ~13 turns | Quick iteration on param-only changes |
| medium | TP=4, 32K, 10×3000 tok | **The loop default** — shows long-context attention with fast compile |
| full | TP=8, 128K, ~42 turns | End-of-run validation only |

I had specified two tiers (search probe + final sweep). Three is better: the
middle tier exists specifically because it exposes long-context attention
behavior at fast-compile cost. Adopt.

### The scoring metric

`avg_prefill_tok_per_s` **averaged over the last 50% of turns**. Deliberately
weights long-context steady-state rather than the easy early turns. One number
decides keep/discard.

Correctness: top-5 logit comparison vs. baseline, pass if >99% of positions
match top-1.

### Their prompt engineering worth stealing verbatim

From `program.md`'s Critical Rules:

- **"NEVER STOP"** — *"Once the loop begins, do NOT pause to ask the human.
  Run autonomously for the full 12 hours (1+3+8). The human may be asleep."*
- **"Mix small and large"** — *"Alternate between quick parameter tweaks and
  bigger structural changes. Small wins compound, but don't get stuck only
  tuning knobs."* Explicit anti-rabbit-hole pressure.
- **"Learn from CUDA"** — names the specific techniques to port:
  FlashAttention (tiling + online softmax), FlashDecoding (KV-parallel split),
  PagedAttention (block-sparse KV), Ring Attention (sequence-parallel),
  MegaBlocks MoE, Triton patterns. *"If a technique is well-proven on GPU,
  consider how the same principle applies to NeuronCores' SBUF/PSUM/DMA
  model."* **This is our Stage 3, already written as a prompt.**
- **"Think harder"** — the out-of-ideas fallback: re-read the architecture,
  read the source to find bottlenecks, *"think about what's fundamentally
  different about Neuron vs GPU that the code isn't accounting for."*
- **"Question framework defaults — many were designed for GPUs, not Neuron."**
- **"If you can DELETE code and get the same performance, that's the best
  kind of win."**
- Timeout: a single experiment over 30 min (compile + run) is killed and
  treated as a crash. Matches our compile-timeout guardrail.

### Their honest limitations (calibrate Stage 4 with these)

Stated in the README:

> AI agents excel at parameter tuning, calling existing modules, and writing
> small patches. **Writing 4000-line flash attention from scratch in a single
> session is beyond current capability.**

> Long-context optimization (attention/CP) **required manual steering** — the
> agent initially fixated on MoE for 12 hours before being redirected.

Both matter for us:

- The first is direct empirical support for **borrow before invent**, and it
  says our Stage 4 expectations should be modest — patch-scale invention, not
  subsystem-scale.
- The second is the rabbit-hole failure mode, observed in the wild for 12
  hours. It argues for beam search (diversity by construction) over greedy,
  and for the "mix small and large" instruction as an explicit prompt rule.

### The three real optimizations they found

Worth seeding into our knowledge bank as verified `op_rewrite` lessons,
because they are concrete, Neuron-specific, and measured. All three share one
pattern: **replace a large `all_gather` of activations with local compute on
fewer tokens plus a small collective at the end.**

- **Local-Q** — standard TP all-gathers full hidden states, then each rank
  computes QKV on a shard. Local-Q inverts it: each rank computes full QKV on
  its local tokens (`seq/TP`), then all-gathers only the small K/V. Saves the
  hidden all_gather and cuts QKV compute by TP×.
- **Context Parallel (CP)** — prior KV cache split across ranks; each rank
  attends to `1/TP` of prior context, merged via online softmax reduction
  (exact, not approximate). Cuts prior-attention compute by TP×.
- **Local-MoE / Local-MLP** — each rank keeps full MoE/MLP weights and
  processes only local tokens, then all-reduces the output. Eliminates the
  MoE/MLP input all_gather entirely.

## 3. AutoResearch (the genre)

The lineage the above repo descends from:

### [karpathy/autoresearch](https://github.com/karpathy/autoresearch)

The originating pattern, described plainly:

> give an AI agent a small but real LLM training setup and let it experiment
> autonomously overnight. It modifies the code, trains for 5 minutes, checks
> if the result improved, keeps or discards, and repeats. You wake up in the
> morning to a log of experiments and (hopefully) a better model.

Two things to take from this framing:

1. **The loop is deliberately dumb and that is the point.** Modify → verify →
   keep/discard → repeat. No clever search needed to get value.
2. **The artifact is the log.** What you wake up to is a *trace of
   experiments*, not just a final number. That is the thing you asked for.

### [WecoAI/awesome-autoresearch](https://github.com/WecoAI/awesome-autoresearch)

A curated list of autoresearch use cases, and its selection criterion is the
important part:

> Every entry includes a link to the actual optimization trajectory so you can
> see **what the agent tried, not just the final result**.

This is the norm we should meet. A leaderboard entry that says "4,200 tok/s"
is worth much less than one that says "4,200 tok/s, and here is the 23-step
path from 1,100, including the six things that failed."

### What we adopt

- **The trajectory is a first-class deliverable**, not a debug log. Every
  leaderboard entry links its full optimization trace.
- **Failures are published**, not hidden. The pruned branches are half the
  information.
- **Overnight autonomy as the target UX** — start it, walk away, read the
  trace in the morning.

## Consolidated changes to our plan

| Change | Where |
|--------|-------|
| Beam search (size 4) with 8 plans/parent, replacing pure greedy | `open-questions.md` Q1 revised |
| Plan-then-implement two-phase prompt per candidate | `optimization-stages.md` |
| Optimization menu with dropout; one-shot generation for Stage 4 | `optimization-stages.md` |
| Trajectory report as a required per-run artifact | `trajectory-reporting.md` (new) |
| Publish failed branches alongside wins | `trajectory-reporting.md` |
| Evaluate Autocomp as a component before building Stage 3/4/5 from scratch | `open-questions.md` Q13 (new) |
| Cite the 24% schedule-reuse result as prior support for the bank | `knowledge-bank.md` |

---

## 4. The three papers

### AFlow — Automating Agentic Workflow Generation

- [arXiv 2410.10762](https://arxiv.org/abs/2410.10762), **ICLR 2025 Oral**
- Repo: https://github.com/FoundationAgents/AFlow

Represents workflows as **code**, then searches that space with **Monte Carlo
Tree Search**, refining through three signals: code modification,
**tree-structured experience**, and execution feedback.

What is relevant to us:

| AFlow idea | Application here |
|------------|-----------------|
| **MCTS over a code-represented space** | A third search option alongside greedy and beam. MCTS handles deep sequential dependencies better than beam — relevant because our stages compound (a Stage-5 fusion's value depends on the Stage-1 config beneath it). |
| **Tree-structured experience** | Their experience store is organized as a *tree* mirroring the search, not a flat list. Our bank is currently flat. A tree lets us answer "given this path so far, what worked next" rather than only "what works for this model class". |
| **Code as the representation** | Matches our reality — NKI kernels and config YAML are both code. No DSL to invent. |

Note the related [VFlow](https://arxiv.org/abs/2504.03723) applies the same
MCTS-over-workflows approach to *Verilog generation* — evidence the pattern
transfers to hardware-adjacent code generation, which is our setting too.

**Recommendation**: keep beam search for phase 1 (simpler, proven on Trainium
by Autocomp), but note MCTS as the phase-4 upgrade path rather than jumping
straight to a learned cost model. AFlow's tree-structured experience is the
more immediately actionable idea — it changes how we index the bank.

### AgentArch — a benchmark, not a design system

- [arXiv 2509.10769](https://arxiv.org/abs/2509.10769) (ServiceNow)
- Repo: https://github.com/ServiceNow/AgentArch
- Full title: *"AgentArch: A Comprehensive Benchmark to Evaluate Agent
  Architectures in Enterprise"*

**Small correction to the title as given**: this is a *benchmark for
evaluating* agent architectures, not a system that automatically designs them.
Worth knowing before citing it.

Its motivating observation is the useful part:

> While individual components of agentic architectures have been studied in
> isolation, there remains limited empirical understanding of **how different
> design dimensions interact** within complex multi-agent systems.

Directly applicable: our own design has many interacting dimensions — search
strategy × stage ordering × bank-on/off × plan-implement split × beam width.
We should not assume the combination we picked is good simply because each
piece is individually defensible.

**Recommendation**: run a small **ablation** on the first 3-5 models rather
than only measuring end-to-end speedup:

| Ablation | Question it answers |
|----------|--------------------|
| bank on vs. off | Is the knowledge bank actually earning its complexity? |
| beam 4 vs. greedy | Does beam width justify 4x the compiles? |
| plan-implement split vs. single-shot | Is the two-phase prompt worth the extra LLM call? |
| stages 3+4 on vs. config-only | How much do kernels add over config tuning? |
| anti-pattern pruning on vs. off | How many compiles does it really save? |

That converts "we built a thing and it was fast" into "we know which parts
made it fast" — which is both better engineering and a far stronger paper.

### ADIAS — Automated Design of Interactive Agentic Systems

- [arXiv 2608.06410](https://arxiv.org/html/2608.06410v1)

Its critique of prior work is the insight worth taking:

> Automated agent design improves agent harnesses through iterative revision,
> evaluation, and feedback summarization. Existing methods are largely
> **candidate-centric**: cross-round experience is organized around candidate
> agents, which leaves the **repair progress implicit**.

Read that against our knowledge bank. Our lessons are currently organized
around **interventions** — "use TP=8", "swap in this kernel". That is
candidate-centric in exactly the sense ADIAS criticizes: it records *what was
changed*, and leaves *what problem was being solved* implicit.

**The fix is a second index on the bank: by symptom, not just by
intervention.**

```yaml
# Current: intervention-indexed (keep)
lesson_id: local-q-replaces-hidden-allgather
type: op_rewrite
intervention: { ... }

# Add: symptom index
symptoms_addressed:
  - bottleneck: collective_bound
    signature: "all_gather of hidden states dominates; >30% of step time in CC"
    observed_via: "profile shows CC engine busy, PE idle"
  - bottleneck: redundant_compute
    signature: "QKV computed on full hidden after gather rather than local shard"
```

Then the proposer can query in the direction it actually needs to:

> "Profile says I am collective-bound with the CC engine at 40% and PE idle.
> What has fixed *that* before?"

rather than only:

> "This is a 30B MoE. What configs are good for 30B MoEs?"

The symptom query is strictly more useful during Stages 3-5, because by then
we have profile data and the bottleneck is *known*. The intervention index
stays useful for Stage 1, where we have no profile yet and are picking a
starting point from model class alone.

### Also relevant: ADAS and the 2026 survey

- **ADAS** — [arXiv 2408.08435](https://arxiv.org/abs/2408.08435), ICLR 2025,
  https://github.com/ShengranHu/ADAS. The parent field. Its framing —
  *"inventing novel building blocks and/or combining them in new ways"* — is
  precisely our borrow (combine existing) vs. invent (novel building block)
  split, which is reassuring convergent design.
- **[ADAS survey, 2026](https://www.preprints.org/manuscript/202606.0238)** —
  classifies 33 methods on four axes. Useful for positioning ours:

| Axis | Our choice |
|------|-----------|
| **Optimization target** | Full code + compound-system params (config *and* kernels) |
| **Search strategy** | LLM-as-optimizer with beam search; MCTS as upgrade path |
| **Representation** | Code (NKI) + structured config |
| **Feedback signal** | Scalar (tok/s) gated by a hard correctness predicate, plus natural-language critique from profile analysis |

The survey names two tensions that we should expect to hit:

1. **Expressiveness vs. searchability** — our config space plus arbitrary NKI
   is highly expressive and therefore hard to search. That is exactly why
   stage ordering and anti-pattern pruning matter.
2. **Feedback richness vs. credit assignment** — when a candidate bundles a
   config change *and* a kernel swap and gets 12% faster, which one earned it?
   Our single-axis-at-a-time greedy/beam perturbation is the mitigation; worth
   stating explicitly as a design constraint rather than an accident.

The survey also flags **reward hacking** as an open safety problem in this
area — which the reference repo already guards against by making `benchmark/`
read-only. We should adopt that literally.

---

## Consolidated changes (updated)

| Change | Source | Where |
|--------|--------|-------|
| Beam search (size 4), 8 plans/parent, replacing pure greedy | Autocomp | `open-questions.md` Q1 |
| Plan-then-implement two-phase prompt | Autocomp | `optimization-stages.md` |
| Optimization menu with dropout; one-shot generation for invention | Autocomp | `optimization-stages.md` |
| **Phase budgets + phase-scoped editable-file permissions** | auto_research | `optimization-stages.md` |
| **`benchmark/` read-only — agent cannot modify its own grader** | auto_research + ADAS survey | `guardrails.md` |
| **Git as state machine: DISCARD = `git reset --hard HEAD~1`** | auto_research | `optimization-stages.md` |
| **Three measurement tiers (fast / medium / full)** | auto_research | `guardrails.md` |
| **Score on last 50% of turns, not the mean** | auto_research | `guardrails.md` |
| **MFU as a normalizing secondary metric (peak 380 TFLOPS/core BF16)** | auto_research | `trajectory-reporting.md` |
| **"NEVER STOP" / "mix small and large" / "learn from CUDA" prompt rules** | auto_research | agent system prompt |
| Trajectory chart + report + ledger as required artifacts | auto_research + awesome-autoresearch | `trajectory-reporting.md` |
| Publish failed branches, with counts, on the chart itself | auto_research | `trajectory-reporting.md` |
| **Symptom index on the bank, not only intervention index** | ADIAS | `knowledge-bank.md` |
| **Tree-structured experience, not flat lessons** | AFlow | `knowledge-bank.md` |
| **Ablation study on first 3-5 models** | AgentArch | `plan.md` phase 1 |
| MCTS noted as phase-4 search upgrade | AFlow | `plan.md` phase 4 |
| Evaluate Autocomp as a component before building Stage 3/4/5 | Autocomp | `open-questions.md` Q13 |
| Seed bank with Local-Q / CP / Local-MoE as verified lessons | auto_research | `knowledge-bank.md` |
| Corrected: config tuning is **not** the dominant win source | auto_research data | `optimization-stages.md` |
| Reconsider seed set — MoE showed 11-17x vs dense 2.8x | auto_research data | `open-questions.md` Q3 |
