# CLAUDE.md — Handoff & Rules of the Game

Read this first. It tells you what is built, what is stubbed, what to build
next, and the rules you must follow. Then read `implementation/README.md`
(the code map) and `plan.md` (the design).

This is an autonomous optimizer that makes HuggingFace models faster on AWS
Trainium, one model at a time, and records how it did it. The full design is in
the `*.md` files at this level; the working code is in `implementation/`.

---

## Honest state (do not assume more than this)

**Built and tested (31 tests pass, no hardware needed):**
- The optimization core: ledger, knowledge bank, guardrails, beam-search
  proposer, orchestrator, trajectory chart. All backend-independent.
- A mock backend that proves the loop end-to-end.

**Stubbed — this is your job:**
- `implementation/src/backends/native_pytorch.py` — the real backend. Every
  method raises `NotImplementedError` with the exact Beta 3 pattern in its
  docstring. Nothing here has touched real Trainium.
- Stages 2-5 (kernel work) — the tournament shell exists; the kernel-authoring
  candidate generators are not wired to the NAD agents yet.
- Stage 0.5 Harvest — designed in `harvest-corpus.md`, not implemented.

**If you try to optimize a real model right now, you will hit
`NotImplementedError` immediately.** That is expected. Finishing the native
backend is task 1.

---

## Your first tasks, in order

### Task 0 — sanity check the harness (10 min, no hardware)
```bash
cd implementation/src && python -m pytest -q          # expect 31 passed
cd .. && python examples/run_mock_search.py            # expect ~18x on mock + a chart
```
If those pass, the core logic is intact and the problem is purely the backend.

### Task 1 — set up the Beta 3 environment
Follow `implementation/ENVIRONMENT.md`. It pulls the Beta 3 DLC, installs the
driver, and verifies the device. **Beta 3 only — never Beta 2** (hard rule,
see `internal Neuron Beta 3 setup docs`).

### Task 2 — the TP=8 gate (do this BEFORE finishing the backend)
There is one unknown that decides everything: **does cross-chip tensor
parallelism (TP>=4) work in native PyTorch on Trn2?** It is documented failing
on Trn1 (`Failed to execute the device barrier 1`). Our seed models need TP=8.

Run the smoke test in `ENVIRONMENT.md`. Three outcomes:
- **TP=8 works** → native PyTorch is viable as primary. Proceed.
- **TP=8 fails like Trn1** → STOP and report. Native PyTorch becomes a
  kernel-dev tool only, and the plan needs the vLLM-Neuron backend instead.
  Do not spend a day building a backend that cannot load the seed models.
- **Works but slow** → proceed, but note it.

### Task 3 — finish `native_pytorch.py`
Implement in this order (each unlocks the next stage):
1. `build_baseline` + `compile` + `measure` → enables Stage 1 (config search)
2. `profile` → enables Stages 2-5 (kernel work)
3. `kernel_swap_points` → enables kernel substitution

Run Stage 1 on the smallest seed first (see below).

### Task 4 — first real run
```bash
# from implementation/, once the backend is real:
python -m optimizer.run --model google/gemma-4-31B --backend native-pytorch-beta3 \
    --objective throughput --stage 1
```
Then publish the recipe (see Output below) and read the trajectory chart.

---

## Seed models (start small, escalate)

| Order | Model | Why | Notes |
|-------|-------|-----|-------|
| 1 | Gemma 4 31B | Standard transformer, Apache 2.0, control case | If the loop breaks here, it's the loop's fault |
| 2 | Muse Glimmer 30B | Second dense, Apache 2.0, has a perception encoder | |
| 3 | Qwen3.8-27B | Gated DeltaNet (linear attention) — needs a new adapter | Hardest; do last |

All fit a trn2.3xlarge in bf16. Do NOT start on a 70B+ or MoE flagship.

---

## Rules of the game (non-negotiable)

These come from `optimization-stages.md` and the reference implementation that
hit a large (multiple-x). Each prevents an observed failure.

1. **Equivalence is a hard gate.** A faster config that fails equivalence is a
   bug, not a win. Never keep it. The baseline defines correctness; everything
   is measured against it.
2. **Keep/discard, git as the state machine.**
   - KEEP if metric improved AND correctness >= threshold → commit stays
   - DISCARD otherwise → `git reset --hard HEAD~1`
   - Log EVERY attempt to the ledger regardless — keeps and discards both.
3. **Borrow before invent.** Try harvested (nkilib) and borrowed (vLLM/SGLang/
   FlashAttention) kernels before writing a novel one. A novel kernel only
   wins if it beats the borrowed one by >= 5%. Record losing inventions too.
4. **Anti-patterns prune before compile.** Never compile a config the bank
   already knows is bad. Compiles cost 5-20 min; pruning is free.
5. **The benchmark/grader is READ-ONLY.** You may never modify the measurement
   harness, its configs, or the baseline reference outputs. This is the
   reward-hacking guard. Modifying your own grader is cheating, not optimizing.
6. **Phase-scoped edits.** In Stage 1 you edit config only. Don't start
   rewriting kernels during config tuning.
7. **Never stop mid-loop to ask.** Run the budget autonomously; the human may
   be away. Only stop for a HARD blocker (TP=8 fails, missing dependency,
   hardware unavailable). Report those, don't churn.
8. **Mix small and large.** Alternate quick config tweaks with bigger
   structural changes. Don't get stuck only tuning knobs (the reference agent
   burned 12 hours fixated on one subsystem — don't).
9. **Stamp the full toolchain on every result** (`neuronx_cc` version
   especially). A result without its toolchain stamp is not reproducible.
10. **Kill any experiment over 30 min** (compile + run) and treat as a crash.

---

## Output layout — where things go

```
optimization_runs/<slug>/       the SEARCH TRACE (process)
    results.tsv                 append-only ledger, one row per experiment
    optimization_timeline.png   the trajectory chart
    run_NNN/                    per-candidate: config, neff, profile, measurements
    OPTIMIZATION_LOG.md         human-readable running log

optimized_models/<slug>/        the DELIVERABLE (product)  ← generate with publish.py
    recipe.json                 winning config + kernels + toolchain (machine-readable)
    RECIPE.md                   human-readable summary
    reproduce.sh                exact commands to rebuild the result
    backend.diff                the fork changes that produced it
    results.tsv + chart         copied evidence
```

- `<slug>` is the model id's last segment, lowercased: `gemma-4-31b`.
- Scratch/throwaway files go in `.tmp/`, never in the source tree.
- The optimized "model" is a RECIPE (config + kernels + diff), not new weights.
  We do not retrain.

Generate the deliverable after a run:
```python
from publish import publish
publish(run_dir="optimization_runs/gemma-4-31b",
        out_root="optimized_models", model_id="google/gemma-4-31B",
        backend="native-pytorch-beta3", toolchain=backend.toolchain_stamp())
```

---

## Guardrails (enforced in code — `implementation/src/guardrails.py`)

| Guardrail | Value |
|-----------|-------|
| HBM ceiling | 85% at peak (measure at full KV occupancy, not step 0) |
| Compile timeout | 30 min per candidate |
| Warmup / measured iters | 3 / 10 minimum |
| No-improvement stop | 5 rounds |
| Max iterations | 100 |
| Invention margin | 5% over the borrowed alternative |
| Compute budget | uncapped — spend it on parallelism across models, not depth |

Benchmark shapes (track A, text): `chat` 1k/512, `rag` 10k/512, `generate`
512/10k, `stress` 64k/64k. During search, probe with `chat` @ batch 1 and 32;
run the full sweep only on the winner. See `guardrails.md`.

---

## What NOT to do

- Don't optimize image/video/speech models. V1 is text LLMs only (track A).
- Don't start on large MoE flagships (Kimi K3, GLM-5.2). Seeds first.
- Don't use Beta 2 patterns (`privateuseone`, `backend="neuron"` PG init).
- Don't modify the benchmark harness or baseline references.
- Don't skip the TP=8 gate. It is cheap and it prevents a wasted day.
- Don't hand-edit `results.tsv` — append via the `Ledger` API.

---

## When you're stuck

- Re-read the model architecture and the profile. What is Neuron-specific that
  the code isn't accounting for?
- Check `harvest-corpus.md` — nkilib may already ship the kernel you need.
- Query the bank by symptom, not just model class: "profile says
  collective_bound, what has fixed that?"
- The design rationale for any decision is in the `*.md` files here. If
  something seems arbitrary, it probably has a documented reason.
