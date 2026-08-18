# Implementation

Working scaffold of the optimizer core. Backend-independent by design — the
whole thing runs today against a mock backend with zero Trainium hardware, and
a real backend is an added file behind `backends/base.py`.

See the design docs one level up (`../plan.md`, `../optimization-stages.md`,
`../architecture.md`, etc.) for the why. This README is the what-and-how.

## What is built and tested

| Module | Purpose | Status |
|--------|---------|--------|
| `src/ledger.py` | Append-only experiment record; git-as-state-machine; metrics | ✅ tested |
| `src/trajectory_chart.py` | The "how it improved" chart, generated from the ledger | ✅ tested |
| `src/bank.py` | Knowledge bank: lessons, intervention + symptom retrieval, anti-pattern pruning, layer tagging, staleness, tiers | ✅ tested |
| `src/guardrails.py` | HBM ceiling, compile timeout, measurement quality, stopping criteria, invention margin | ✅ tested |
| `src/proposer.py` | Beam search over config axes, seeded by bank priors | ✅ (via orchestrator tests) |
| `src/orchestrator.py` | Walks the stage pipeline, runs the tournament, keep/discard, records everything | ✅ tested |
| `src/backends/base.py` | The backend interface (the ~20% that is backend-specific) | ✅ |
| `src/backends/mock.py` | Hardware-free stand-in modeling compile cost + config effects | ✅ tested |

**31 tests, all passing.** Run them:

```bash
cd implementation/src
python -m pytest -q
```

## What is NOT built yet

- **Real backends.** `backends/vllm_neuron_xla.py`, `nxdi_xla.py`,
  `native_pytorch.py` — these wrap the actual Neuron toolchain and NAD agents.
  The mock proves the core; the real ones are the next milestone.
- **Stages 2-5 candidate generators.** The tournament shell exists in the
  orchestrator; the kernel-authoring stages delegate to the NAD worker agents,
  which are not wired in yet.
- **Stage 0.5 Harvest.** Designed in `../harvest-corpus.md`; the corpus matcher
  is not implemented.
- **Watcher agents** (adversarial equivalence, supervisor). Designed in
  `../agent-topology.md`.
- **Discovery job + leaderboard runner** (phase 3).
- **The report generator** (`optimization_report.md`); the chart and ledger
  exist, the prose report does not.

## Run the demos

Both are hardware-free and take seconds.

```bash
cd implementation

# 1. Real Stage-1 beam search on the mock backend, charted from its output:
python examples/run_mock_search.py

# 2. A hand-authored ledger mirroring auto_research's Round 2 (845 -> 4269),
#    to show the chart format on a realistic multi-stage trajectory:
python examples/make_sample_run.py
```

Each writes a `results.tsv` and a trajectory PNG under `examples/<name>/`.

## How the pieces connect

```
ModelSpec ─▶ Orchestrator ──────────────────────────────────┐
                │                                            │
                │ establish_baseline (Stage 0, not gated)    │
                │                                            ▼
                │ run_stage1_config:                    Ledger (results.tsv)
                │   BeamProposer.seed  ◀── KnowledgeBank.query_interventions
                │   BeamProposer.expand                      │
                │   KnowledgeBank.prune ◀── anti-patterns    │ every attempt,
                │   for each survivor:                       │ keep or discard
                │     Backend.compile                        │
                │     equivalence gate (HARD)                │
                │     Backend.measure                        │
                │     Guardrails (HBM, noise, timeout)       │
                │     keep/discard vs incumbent              │
                ▼                                            ▼
           incumbent config                          trajectory_chart
                                                     (reads the ledger)
```

## Design invariants the tests lock in

- **Equivalence is a hard gate.** A faster config that fails equivalence is
  never kept (`test_equivalence_failure_blocks_promotion`).
- **The baseline is the reference**, not equivalence-gated; an incumbent always
  exists even if every change fails.
- **Anti-patterns prune before compile**, at zero cost
  (`test_no_tp16_was_ever_compiled`).
- **Kept metrics are monotonic** — a KEEP only happens on improvement
  (`test_incumbent_monotonic_in_ledger`).
- **Stage 4 clears a higher bar** — the 5% invention margin
  (`guardrails.is_improvement`).
- **Lower layers win ties** — kernel over framework, for migration durability
  (`test_layer_durability_ordering`, proposer `select`).
- **Provenance is four-valued** — harvested / borrowed / hybrid / invented,
  so the invention metric stays honest.

## Dependencies

Standard library plus `pyyaml` (bank) and `matplotlib` (chart). No Neuron SDK,
no AWS, no network — the mock backend makes the whole core runnable anywhere.

## Next milestone

Wire `backends/vllm_neuron_xla.py` to a real trn2 and run Stage 1 on Gemma 4
31B. That is the first point where a number means something. Everything else
here is validated logic waiting for a real oracle.

---

## Autonomous overnight run

The whole loop runs unattended, over all three seed models, building the bank
as it goes. Backend-agnostic: `mock` proves it in minutes (synthetic numbers);
`native-pytorch-beta3` produces real numbers once that backend is implemented.

```bash
cd implementation

# Prove the loop end-to-end (synthetic — labeled as such in the output):
python run_overnight.py --backend mock

# Real run (after the native backend is implemented + TP=8 gate passes):
python run_overnight.py --backend native-pytorch-beta3
```

Outputs land in `implementation/artifacts/`:
- `LEADERBOARD.md` — cross-model summary (self-labels synthetic runs)
- `OVERNIGHT_LOG.md` — timestamped running log
- `optimization_runs/<slug>/` — per-model trace + trajectory chart
- `optimized_models/<slug>/` — the deliverable recipe bundle

Seed the bank first so the proposer starts smart, not empty:

```bash
cd src && python seed_bank.py     # 7 verified lessons from real findings
```
