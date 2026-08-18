# Agent Topology and Host

Two questions: should agents watch each other, and what does this run on.

---

# Part 1: Multi-agent, and agents watching each other

## Framing: we already have multiple agents

The NAD package ships 8. The optimizer orchestrates them:

| Agent | Role in our stages |
|-------|-------------------|
| `neuron-framework-autoport-agent` | Stage 0 baseline |
| `neuron-nki-writer-agent` | Stages 3, 4 |
| `neuron-nki-debugger-agent` | Fixes failed compiles |
| `neuron-nki-profile-analysis-agent` | Bottleneck classification for Stages 2-5 |
| `experimental-neuron-framework-equivalence-agent` | The correctness gate |
| `neuron-nki-agent` | General kernel lifecycle |

So the real question is not "should there be multiple agents" but **"should
agents also check each other, and where does that pay off."**

## The key constraint: we have a hardware oracle

Most multi-agent cross-checking exists because there is no ground truth — you
cross-examine because nobody can just *look up* whether the answer is right.

We can. Compile it and measure it. The hardware is the arbiter.

**That makes generic agent-watching-agent patterns low-value here.** Committee
voting, debate, and consensus mechanisms are ways to approximate a missing
oracle. When you have one, they mostly add cost and latency for no accuracy.

But the oracle is weak in exactly two places, and that is where watchers earn
their keep.

## Where the oracle is weak (build watchers here)

### Weakness 1: Equivalence is sampled, not proven

The equivalence gate says "matches the reference on *these* inputs." It does
not say "correct in general." A kernel can pass 100 greedy-decode positions
and be wrong on a shape, a padding case, or a numerical edge the test set never
touched.

**→ Adversarial Equivalence Agent.** Its job is to *break* the candidate:

- Hunt for input shapes that trigger divergence (boundary sequence lengths,
  non-power-of-2 batch, ragged batches)
- Probe numerical edges (denormals, inf/NaN propagation, extreme logit ranges)
- Attack padding and masking behavior
- Test the shape regime *just outside* the kernel's declared constraints

It is adversarial by construction: it succeeds when it finds a failure. That
is a genuinely different objective from the writer agent's, which is why
separating them works — the same agent trying to both write and break its own
kernel is a weak check.

Failures it finds become `anti_pattern` lessons *and* new permanent test cases.
The test set grows monotonically, so the same class of bug cannot recur.

This is the highest-value watcher because a silently-wrong published recipe is
the worst outcome the whole system can produce.

### Weakness 2: The oracle scores candidates, not direction

Measurement tells you candidate N was 3% faster. It does not tell you that
you have spent six hours in the wrong subsystem.

This failure is documented in the reference implementation:

> Long-context optimization (attention/CP) required manual steering — the
> agent initially fixated on MoE for 12 hours before being redirected.

Twelve hours of compute burned because nothing was watching the *shape* of the
search.

**→ Supervisor Agent.** Runs on a timer (say every 30 min), reads only the
ledger and profile history, never touches code:

- Is the win rate collapsing? (many candidates, no promotions)
- Is the search concentrated in one op while the profile says the bottleneck
  is elsewhere?
- Are we below the stage's budget burn rate with no progress?
- Has roofline attainment plateaued, meaning the remaining headroom is
  somewhere we are not looking?

Output is a redirect written into the loop's context, not a code change:

```
SUPERVISOR: 14 candidates in the last 90 min, 0 promoted, all targeting
moe_dispatch (8% of profile time). Profile shows attention_prefill at 47%
and rising with context length. Redirect Stage 3 to attention. Deprioritize
MoE for the remainder of this stage.
```

Cheap — it reads a TSV and a profile summary. It replaces the human who had to
intervene at hour twelve.

### Weakness 3 (partial): the reward can be gamed

The ADAS survey names reward hacking as a live safety problem for
self-improving systems. Our primary defense is structural, not agentic: the
benchmark harness, its configs, and the baseline reference outputs are
**read-only** (see `optimization-stages.md`). The agent cannot edit its own
grader.

A light watcher adds a second layer: flag any diff that touches
measurement-adjacent code, or any result that improves the metric while
correctness confidence *drops*. Suspiciously good results deserve scrutiny —
a 40% jump from a one-line change is more likely a broken benchmark than a
breakthrough.

## Where watchers are NOT worth it

Being explicit, because these are the patterns people reach for by default:

| Pattern | Why not here |
|---------|-------------|
| Committee voting on which plan is best | The hardware votes. Compile the top-k and measure. Cheaper and correct. |
| Debate / argumentation between agents | Expensive, slow, and resolves questions the oracle answers directly. |
| Duplicate writer agents cross-reviewing code | Beam search already gives parallel diversity, and measurement already ranks them. Review adds latency without adding signal. |
| A "manager" agent decomposing work | The stage pipeline *is* the decomposition, and it is fixed. Nothing to decide. |

## Proposed topology

```
                    ┌──────────────────────────────┐
                    │       ORCHESTRATOR           │
                    │  stage pipeline, beam,       │
                    │  tournament, promotion       │
                    └──────┬────────────────┬──────┘
                           │                │
              ┌────────────┴──────┐    ┌────┴─────────────────┐
              │   WORKER AGENTS   │    │   WATCHER AGENTS     │
              │  (existing NAD)   │    │      (new)           │
              ├───────────────────┤    ├──────────────────────┤
              │ autoport          │    │ adversarial-         │
              │ nki-writer        │    │   equivalence        │
              │ nki-debugger      │    │ supervisor           │
              │ profile-analysis  │    │ (reward-hack flag)   │
              │ equivalence       │    │                      │
              └───────────────────┘    └──────────────────────┘
                           │                │
                    ┌──────┴────────────────┴──────┐
                    │      HARDWARE ORACLE         │
                    │  compile → equivalence →     │
                    │  measure → profile           │
                    └──────────────────────────────┘
```

Two watchers, both cheap, both targeting a specific documented weakness. Not a
committee.

## Cost discipline

Watchers must not dominate the budget. Caps:

| Watcher | Trigger | Budget |
|---------|---------|--------|
| Adversarial equivalence | Only on candidates that already passed the standard gate *and* beat the incumbent | ≤ 10% of stage time |
| Supervisor | Timer, every 30 min | ≤ 2% (reads a TSV) |
| Reward-hack flag | On any suspiciously large single-step gain (> 25%) | negligible |

Adversarial equivalence runs only on *winners*. Running it on every candidate
would double the cost of the whole loop for no benefit — a candidate that lost
on performance does not need a deeper correctness audit.

## Worth measuring, not assuming

Per the AgentArch point about interacting design dimensions, include watchers
in the phase-1 ablation:

- watchers on vs. off → do they change final quality or just cost?
- adversarial equivalence on vs. off → how many real bugs does it actually
  catch, and how many were caught by the standard gate anyway?
- supervisor on vs. off → does it measurably reduce time-in-wrong-subsystem?

If the adversarial agent catches nothing across 5 models, it is not earning its
10% and should be cut.

---

# Part 2: What does this run on?

## The requirement that settles it

The loop runs **12 hours autonomously on a remote trn2 instance**. Headless.
The reference implementation runs inside a Docker container on the instance,
driven by a `program.md` the agent reads.

An IDE cannot be in that path. So the core must be a **CLI / daemon**, not an
editor extension.

## Two layers, different hosts

| Layer | What it is | Host |
|-------|-----------|------|
| **Core loop** | Python package, host-agnostic, calls model APIs directly | Runs on the trn2 box. No IDE. |
| **Interactive shell** | Human-in-the-loop work: triage the bank, review trajectories, debug a stuck run | Kiro / Claude Code / whatever the engineer uses |

This is Autocomp's architecture and it is the right one. Autocomp is a plain
Python package that supports OpenAI, Anthropic, AWS Bedrock, Google Vertex,
Together, and local vLLM. `python -m autocomp.search.run_search` and walk away.

## Why host-agnostic for the core

- **Headless requirement.** Above.
- **Public reproducibility.** The leaderboard's credibility depends on outsiders
  reproducing numbers. "Install Kiro first" is a non-starter for that.
- **Model flexibility.** We will want to try different planner models, and
  possibly a cheaper model for implementation than for planning (Autocomp
  separates `models` from `code_models` for exactly this). Being tied to one
  host's model selection is a constraint we do not need.
- **Bedrock access.** Running planner models through Bedrock keeps everything
  inside the AWS account that already holds the Trainium capacity.

## Why keep a Kiro/Claude Code layer anyway

- **The NAD agents already exist there.** `neuron-nki-writer-agent`,
  `neuron-nki-debugger-agent`, and the profile-analysis agent are real, tested
  assets with Neuron-specific knowledge. Reimplementing them as raw API calls
  throws that away.
- **Human-in-the-loop tasks are genuinely interactive.** Weekly bank triage,
  reading a trajectory and deciding whether a lesson generalizes, debugging why
  a stage stalled — all better in an editor with file context.
- **Development speed.** Building the first version as Kiro skills/agents is
  much faster than building a framework from scratch.

## Recommended path

**Phase 1 — prototype as NAD-style agents + skills.** Fastest path to a working
loop. Reuses the 8 existing agents. Validates the stage pipeline, the bank
schema, and the trajectory format on 3 seed models. Accept the host coupling as
temporary.

**Phase 2 — extract the core into a host-agnostic package.** Once the loop's
shape is settled, lift the orchestrator, bank, guardrails, ledger, and
reporting into a Python package with a provider-agnostic LLM interface. Keep
NAD agents as one possible *worker* implementation behind an interface:

```python
class KernelWriter(Protocol):
    def write(self, spec: KernelSpec, context: Profile) -> Kernel: ...

# implementations:
#   NADAgentKernelWriter      — invokes the existing Kiro/Claude Code agent
#   DirectAPIKernelWriter     — calls Bedrock/Anthropic/OpenAI directly
#   AutocompKernelWriter      — delegates to Autocomp, for calibration
```

That interface is also how we run the Q13 calibration experiment — swap in
Autocomp as a worker and compare against our agents on identical kernels.

**Phase 3 — the public release is the package**, with the agents as an optional
interactive frontend.

## Practical near-term shape

```
Instructor laptop (Kiro)          trn2.48xlarge (headless)
─────────────────────────         ────────────────────────────────
kick off a run          ──ssh──▶  docker run optimizer-daemon
review trajectory       ◀──────   writes results.tsv + charts
triage bank weekly      ◀──────   provisional/ lessons queue up
debug a stalled stage   ──ssh──▶  attach to the container
```

The overnight run does not need anyone present. Morning review happens in the
editor, against artifacts the run produced.
