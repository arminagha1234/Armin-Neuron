# RUN — the single instruction for Claude Code

Copy the block below into Claude Code, running from this directory, on a
machine with the Beta 3 Trainium environment set up (see
`implementation/ENVIRONMENT.md`). It is written to run autonomously with no
human in the loop.

---

## The prompt

> Read `CLAUDE.md` fully, then `implementation/README.md`. Follow the rules of
> the game in `CLAUDE.md` exactly. Do not stop to ask me questions — run
> autonomously and only halt for a HARD blocker (TP=8 gate fails, hardware
> unavailable, missing dependency you cannot install). I am asleep.
>
> Execute in this order:
>
> 1. **Sanity**: `cd implementation/src && python -m pytest -q` — expect 35
>    passed. Then `cd .. && python run_overnight.py --backend mock` to confirm
>    the loop works end to end. This produces synthetic numbers — that is
>    expected, it is just proving the harness.
>
> 2. **Seed the bank**: `cd src && python seed_bank.py` — loads the initial
>    verified lessons (Local-Q, Context Parallel, Local-MoE, config priors,
>    anti-patterns).
>
> 3. **Environment**: follow `implementation/ENVIRONMENT.md` sections 2-3 to
>    stand up the Beta 3 native-PyTorch DLC and install the driver.
>
> 4. **TP=8 gate** (ENVIRONMENT.md section 3): run the smoke test. If it fails
>    with the device-barrier error, STOP and write a report — do not build the
>    backend, because it cannot load the 30B seed models. If it passes,
>    continue.
>
> 5. **Implement `implementation/src/backends/native_pytorch.py`**: fill in the
>    stubbed methods in the documented order (build_baseline + compile +
>    measure first, then profile, then kernel_swap_points). The Beta 3 patterns
>    are in each docstring and in `internal Neuron Beta 3 setup docs`. Wire the real
>    equivalence checker into `overnight.py`'s `_equivalence_for`.
>
> 6. **Run for real**:
>    `python run_overnight.py --backend native-pytorch-beta3`
>    This optimizes gemma-4-31b, muse-glimmer-30b, and qwen3-8-27b in turn,
>    within phase budgets, building the knowledge bank as it goes.
>
> 7. **Leave a clean morning artifact**: the run writes
>    `implementation/artifacts/LEADERBOARD.md`, per-model trajectory charts and
>    recipes, and `OVERNIGHT_LOG.md`. Append a short `MORNING_SUMMARY.md` to the
>    repo root: what worked, what failed, the real speedups if any, and what you
>    would try next.
>
> Honor every rule in `CLAUDE.md`: equivalence is a hard gate, keep/discard via
> git, borrow before invent, benchmark is read-only, never modify your own
> grader, stamp the toolchain on every result, kill any experiment over 30 min.

---

## What "success" looks like in the morning

Realistic outcomes, best to worst — all are fine to wake up to:

1. **Best case**: native backend works, TP=8 passes, one-to-three models got
   real Stage-1 (and maybe Stage-2) speedups, bank has real lessons, leaderboard
   has real numbers. → post to GitHub with real results.
2. **Likely case**: backend partly implemented, TP=8 outcome known, one model
   through baseline + partial config search. → post the framework, mark real
   results "in progress", include the honest MORNING_SUMMARY.
3. **Blocker case**: TP=8 failed, or the DLC/driver fought back. → post the
   framework + the mock demo + a written account of the hardware blocker. Still
   a strong artifact.

In every case the **framework itself** is the postable thing. Do not publish
mock numbers as real — the leaderboard file self-labels synthetic runs; keep
that honesty in the GitHub post.

## Before you push to GitHub

Run `scripts/prepare-github.sh` (from this directory). It stages a clean copy,
scans for secrets, and refuses to proceed if it finds any. Never `git init` in
the parent workspace — it contains credentials.
