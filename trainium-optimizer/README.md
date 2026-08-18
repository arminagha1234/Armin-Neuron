# Autonomous Trainium Model Optimizer + Leaderboard

An ambitious extension of the current Neuron autoport/NKI tooling. The idea in
one paragraph:

> Take our existing autoport + NKI writer/debugger + equivalence agents and
> wrap them in an outer loop that autonomously optimizes any HuggingFace
> model on Trainium along two objectives (max throughput, min latency),
> steals patterns from vLLM/SGLang/TensorRT-LLM under license-compliant
> terms, writes its own NKI kernels when nothing off-the-shelf fits, and
> writes every proven trick to a **knowledge bank** so the N+1'th model is
> cheaper to optimize than the N'th. Then point it at the top 100 open
> models and publish a **Neuron-optimized leaderboard** that refreshes on
> every SDK release.

## Why this is worth doing

- Nobody owns "best perf-per-dollar on Trainium for open models" as a
  visible public artifact. MLPerf covers a handful; vLLM benchmarks aren't
  Neuron-focused; HF Open LLM is accuracy, not perf.
- We already have the atomic pieces (autoport, NKI writer/debugger,
  equivalence, profile-analysis agents). The gap is an orchestrator + a
  memory of what has worked.
- The knowledge bank is the flywheel: leaderboard runs generate lessons,
  lessons make future runs cheaper, which makes the leaderboard cheaper to
  maintain, which drives more lessons.

## Docs in this folder

| File | What it covers |
|------|----------------|
| [`plan.md`](./plan.md) | Phased milestones with concrete deliverables. Read first. |
| [`guardrails.md`](./guardrails.md) | Benchmark shapes per track, batch sweeps, HBM/compile limits, stopping criteria, SDK stamping |
| [`model-landscape.md`](./model-landscape.md) | Researched top open models per modality (Aug 2026 snapshot) |
| [`optimization-stages.md`](./optimization-stages.md) | The 6-stage pipeline: config → known kernels → **borrow** → **invent** → graph rewrites, plus execution discipline |
| [`references-analysis.md`](./references-analysis.md) | What we took from Autocomp, auto_research (a large (multiple-x) on Trn2), AFlow, AgentArch, ADIAS |
| [`harvest-corpus.md`](./harvest-corpus.md) | The aws-neuron org survey + **Stage 0.5: Harvest** — mine working kernels before optimizing |
| [`agent-topology.md`](./agent-topology.md) | Multi-agent design (worker vs. watcher agents) and what host this runs on |
| [`trajectory-reporting.md`](./trajectory-reporting.md) | The ledger + chart + report showing **how** it improved, not just the final number |
| [`architecture.md`](./architecture.md) | How the components fit — optimizer loop, knowledge bank, leaderboard runner |
| [`knowledge-bank.md`](./knowledge-bank.md) | Lesson schema, types of lessons, seed content, staleness handling |
| [`open-questions.md`](./open-questions.md) | Load-bearing decisions — several now resolved |
| [`references.md`](./references.md) | References we'll consult for kernel patterns + prior art |
| `websites_check` | Original reference-checking note (kept as-is) |

## Decisions locked in so far

- **Four benchmark shapes**: `chat` (1k/512), `rag` (10k/512), `generate`
  (512/10k), `stress` (64k/64k — from a real customer ask)
- **Batch sweep** 1→32 powers of two, shape-aware (`stress` caps at 4)
- **HBM ceiling 85%** measured at peak KV occupancy, **30 min compile
  timeout** — OOM avoidance is a hard gate
- **No compute budget cap.** Termination is by search-quality criteria;
  spare capacity goes to parallelism across models, not depth on one
- **Leaderboard first**, framework open-sourced later
- **Weekly multi-source model discovery** (LMArena + HF trending + HF
  downloads + Open LLM Leaderboard + AWS customer signal)
- **Provisional vs. verified** knowledge-bank tiers with human triage
- **Per-family adapters** define the search space; the outer loop is shared
- **Anti-patterns are first-class** — own folder, read by the proposer to
  prune candidates before compiling
- **Hand-priored greedy** search for v0; learned policies deferred to Phase 4
- **Seed ladder**: Gemma 4 31B → Muse Glimmer 30B → Qwen3.8-27B (all Apache
  2.0, all fit trn2.3xlarge). MiniMax dropped — H3 is video-gen with reported
  territory restrictions; M-series deferred to phase 3.
- **V1 scope is Track A (LLMs) only.** Image / video / ASR / TTS tracks are
  fully designed but deferred — keeping them specified forces the outer loop
  to stay track-agnostic, so adding one later is writing an adapter rather
  than refactoring the core.
- **Six optimization stages**, cheapest and safest first: baseline → config →
  known kernels → **borrow** → **invent** → graph rewrites.
- **Borrow before invent.** Steal patterns from vLLM / SGLang / TRT-LLM /
  FlashAttention first. A novel kernel only replaces a borrowed one if it
  beats it by **5%** — noise floor plus a risk premium for bespoke code.
  Losing inventions are recorded, not discarded.
- **Invention is instrumented**, not assumed: `invention_rate`,
  `invention_win_rate`, `roofline_attainment` tracked from run 1 so "does it
  ever create something new" gets an empirical answer.
- **Direct code borrowing** permitted from Apache-2.0 / BSD sources, with
  `THIRD_PARTY_NOTICES` + per-file provenance headers. TensorRT-LLM needs a
  compliance look before we borrow from it.
- **Full toolchain stamped on every result** (`neuronx_cc` version
  especially), with a re-verification pass on every SDK release

## TL;DR of the plan

1. Pick 3 seed models across families (dense LLM, MoE, diffusion). Build
   the optimizer's outer loop with a hand-priored greedy search + hard
   compute budget. Get end-to-end wins on all three.
2. Ship the recipes as a blog post + repo. Get external signal early.
3. Design the knowledge bank schema, bootstrap it with 30-50 lessons we
   already know from the Neuron team, integrate into the optimizer's
   proposer.
4. Pick a "top 100 open models" definition and start a rolling leaderboard.
   Publish per-model recipes + evidence.
5. Only after that: replace hand-priored greedy with a learned search
   policy.

## Non-goals for V1

- Not solving general AutoML — we're doing model→Neuron optimization only.
- Not building a training-time optimizer — inference only.
- Not chasing state-of-the-art search algorithms in phase 1 — a good greedy
  with human-designed priors beats a bad Bayesian on a small budget.
- Not competing with Nvidia's leaderboard numbers — we publish honest
  perf-per-dollar comparisons and let the numbers speak.

## Prior art this builds on

Five systems studied in detail; full analysis in
[`references-analysis.md`](./references-analysis.md).

| System | What we took |
|--------|-------------|
| [**Autocomp**](https://github.com/ucb-bar/autocomp) (UC Berkeley, NeurIPS 2025) | Beam search over LLM-generated plans; plan-then-implement two-phase prompt; optimization menu with dropout; hardware-in-the-loop every iteration. Already supports Trainium NKI. Their paper also measured **24% improvement from reusing optimization schedules across similar ops** — direct empirical support for the knowledge bank. |
| **internal-prior-optimization-run** (private) | The closest existing system — a 12-hour autonomous loop hitting **a large (multiple-x) on Tongyi-30B-A3B**. Took: phase budgets with scoped write permissions, read-only grader, git-as-state-machine, three measurement tiers, MFU normalization, the trajectory chart format, and its prompt rules. |
| [**AFlow**](https://arxiv.org/abs/2410.10762) (ICLR 2025 Oral) | MCTS over code-represented workflow space; **tree-structured experience** rather than a flat lesson store. |
| [**AgentArch**](https://arxiv.org/abs/2509.10769) (ServiceNow) | Design dimensions interact in ways component-level study misses → run an **ablation** on the first 3-5 models, not just end-to-end speedup. |
| [**ADIAS**](https://arxiv.org/html/2608.06410v1) | Candidate-centric experience stores leave repair progress implicit → index the bank **by symptom**, not only by intervention. |

Two things that changed because of this reading:

1. **Search strategy revised from greedy to beam** (Autocomp proved beam on
   Trainium specifically; greedy was too conservative).
2. **The "config carries most of the speedup" assumption was wrong** — real
   data shows config +19%, model code +405%, structural TP changes +193%.
   Corrected in `optimization-stages.md`.

## Harvest before optimize

A **Stage 0.5: Harvest** runs before any optimization: read-only, 30-minute
budget, mines the `aws-neuron` corpus and the knowledge bank for kernels that
already work, and emits a ranked candidate inventory plus an
`unmatched_ops` queue that becomes the evidence-derived Stage 3/4 work list.

Motivating case: the `auto_research` run hand-built Context Parallel in its
third round. [`nki-library`](https://github.com/aws-neuron/nki-library)
already ships *Attention KV Parallel Segmented CTE* — "KV-parallel segmented
prefill attention with online softmax merging for context parallelism." Plus
`FGCC` (fused all-gather + matmul) for the collective bottleneck they also
attacked by hand. Harvest exists so we try the shelf before the workshop.

Provenance is now four-valued, and reported separately so the invention metric
stays honest:

| `origin` | Meaning | Risk |
|----------|---------|------|
| `harvested` | Existing AWS-maintained kernel, used as-is | Low — maintained upstream |
| `borrowed` | Pattern ported from vLLM / SGLang / TRT-LLM / FlashAttention | Medium |
| `hybrid` | Borrowed algorithm, substantially restructured for Neuron | Medium-high |
| `invented` | Novel, from profile + roofline | Highest |

A run that is 90% `harvested` is a **good outcome** — the ecosystem had the
answer. It is just not invention, and conflating the two would flatter us.

## Backend strategy

vLLM-Neuron (XLA) is the V1 target. Native PyTorch / TorchNeuron is expected to
become the serving substrate later, so the backend sits behind an adapter from
day one and every bank lesson is **layer-tagged**:

| `layer` | Survives an XLA → native-PyTorch migration? | `migration_risk` |
|---------|---------------------------------------------|------------------|
| `kernel` (NKI) | **Yes** — NKI is below the framework boundary | low |
| `collective` (TP/CP/EP patterns) | Mostly | low-medium |
| `config` | Concepts yes, knob names no | medium |
| `framework` (vLLM/NxDI internals) | Often not | high |
| `graph` (XLA passes) | Likely not | high |

Two consequences:

- When the migration happens, the re-verification scope is a single query
  (`migration_risk: high`) rather than "re-test everything."
- **The proposer prefers lower layers on ties.** A NKI kernel worth +15% beats a
  framework patch worth +15%, because one survives. Encoded as a ranking
  tiebreaker, not left to instinct.

Roughly 80% of this framework is backend-independent — bank, search loop,
guardrails, ledger, reporting, discovery, leaderboard, and the NKI kernels
themselves. Waiting for the beta would protect the other 20%, which is not
worth an unbounded delay.

## Agents

Worker agents come from the existing NAD package. Two **watcher** agents are
new, and both target a specific place where the hardware oracle is weak:

| Watcher | Weakness it covers | Budget |
|---------|-------------------|--------|
| **Adversarial equivalence** | The gate proves "matches on these inputs," not "correct in general." This agent tries to *break* winning candidates — boundary shapes, ragged batches, numerical edges. Findings become permanent test cases. | ≤10% of stage time, winners only |
| **Supervisor** | The oracle scores candidates but not *direction*. Reads the ledger every 30 min and redirects when the search concentrates somewhere the profile says is not the bottleneck. | ≤2% (reads a TSV) |

The supervisor exists because of a documented failure: the reference
implementation's agent "fixated on MoE for 12 hours before being redirected" by
a human.

Deliberately **not** built: committee voting, agent debate, duplicate reviewers.
We have a hardware oracle — compile it and measure it. Those patterns
approximate a missing oracle and mostly add cost when you have one.

## Where it runs

Two layers, because a 12-hour autonomous run on a headless trn2 cannot have an
IDE in the path:

- **Core loop** — host-agnostic Python package, provider-agnostic LLM interface
  (Bedrock / Anthropic / OpenAI / local vLLM). Runs on the instance. This is
  what gets open-sourced, because public reproducibility cannot require a
  specific editor.
- **Interactive shell** — Kiro / Claude Code, for bank triage, trajectory
  review, and debugging stalled runs.

Phase 1 prototypes as NAD-style agents and skills (fastest path, reuses the 8
existing agents), then phase 2 extracts the core into the package with NAD
agents behind a `KernelWriter` interface. That interface is also how the
Autocomp calibration experiment (Q13) plugs in.
