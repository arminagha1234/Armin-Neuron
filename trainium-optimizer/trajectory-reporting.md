# Trajectory Reporting

The optimization trace is a **first-class deliverable**, not a debug log. A
leaderboard entry saying "4,269 tok/s" is worth far less than one saying
"4,269 tok/s, and here is the 23-step path from 845, including the ~40
experiments that failed."

Modeled directly on `internal-prior-optimization-run`'s
`generate_chart.py` + `results.tsv` + `optimization_report_en.md` trio, with
the manual step automated away.

## The three artifacts

Every optimization run produces all three. They serve different audiences.

| Artifact | Audience | Format |
|----------|----------|--------|
| `results.tsv` | The loop itself, and aggregate analysis | Append-only TSV, one row per experiment |
| `optimization_timeline.png` / `.html` | Humans skimming "did it work" | Chart |
| `optimization_report.md` | Humans who want the why | Markdown with embedded chart |

## 1. `results.tsv` — the experiment ledger

Append-only. **Every experiment is logged regardless of outcome.** This is
the source of truth the chart and report are generated from.

Schema (extending the reference implementation's):

```
commit    phase  stage  origin    tok_per_s  mfu   correctness  compile_s  status   description
a616588   1      config  -         570.6      0.2   100.0        391.0      keep     baseline (BF16 experts, FP8 KV, TP4, 4096 segment)
96f6839   1      config  -         570.6      0.2   100.0        391.0      discard  explicit O2 compile (570.585 vs 570.601 — noise)
2082e41   1      config  -         1078.1     0.4   100.0        331.0      keep     2048-token prefill segment (+88.9%)
5fea7e9   1      config  -         1005.0     0.4   100.0        290.0      discard  1024-token segment (-6.78% vs 2048)
d4e5f6a   3      borrow  vllm      4071.1     1.7   100.0        612.0      keep     NKI flash attention (FlashAttention tiling + online softmax)
e5f6a7b   4      invent  -         4102.0     1.7   100.0        740.0      discard  novel 6-way SBUF split (+0.8%, under 5% margin)
```

New columns beyond the reference:

- **`stage`** — which of our six stages produced it (`baseline`, `config`,
  `known_kernel`, `borrow`, `invent`, `graph_rewrite`)
- **`origin`** — for borrowed work, the source (`vllm`, `sglang`, `trtllm`,
  `flashattn`); empty for config and invented work

These two columns are what make the borrow-vs-invent metrics computable
straight off the ledger.

**`commit` is load-bearing.** Each row links to the actual diff, so the chart
and report can hyperlink every point to the code that produced it. Git is
also the state machine: `DISCARD = git reset --hard HEAD~1`.

## 2. The chart

### What the reference does well (and we copy)

From `generate_chart.py`:

- **Dark GitHub palette** — `#0d1117` figure, `#161b22` axes, `#c9d1d9` text.
  Reads well in a README and in dark-mode docs.
- **One subplot per round**, so rounds with different benchmarks are never
  plotted on a shared axis (they are not comparable).
- **Line + scatter**, points colored **by phase** — blue `#58a6ff` params,
  green `#3fb950` model code, orange `#f0883e` kernels.
- **Vertical dashed phase separators** with phase labels above.
- **Gain annotations** (`+38%`, `+150%`) on the meaningful jumps only, not
  every point. Big jumps get larger, bolder, colored text.
- **A star annotation on the single largest gain** — `★ Largest single gain
  (nkilib kernel integration)`.
- **A red callout box for failures**: `~40 MoE kernel experiments: all <1%`.
  Failures are on the chart, not hidden.
- **Final result badge** — the headline number in a filled rounded box.
- **Subtitle carries the conditions**: hardware, TP degree, correctness,
  context length. Without these the number is meaningless.

### What we change

| Change | Why |
|--------|-----|
| **Auto-generate from `results.tsv`** | The reference hardcodes its arrays (`r1_toks = [570.6, 1078.1, ...]`, "simplified from ~100 experiments"). Hand-curated charts drift from the ledger and do not scale to 100 models. |
| **Add a roofline ceiling line** | A horizontal dashed line at the computed bound, so the chart answers "how much headroom is left" — not just "how far did we come". |
| **Add MFU on a secondary axis** | The reference reports MFU in tables but not on the chart. Seeing tok/s and MFU together shows whether gains came from real efficiency or from doing more work. |
| **Mark pruned candidates** | Faded X markers for discarded experiments at their measured score, so the beam's width is visible rather than implied. |
| **Distinguish borrow vs invent by marker shape** | Circle = borrowed, diamond = invented, square = config. Makes the provenance story legible at a glance. |
| **Emit HTML alongside PNG** | The reference ships both `optimization_chart.html` and PNGs. Interactive hover showing commit + description + diff link is worth a lot when reviewing 180 experiments. |

### Layout

```
┌──────────────────────────────────────────────────────────────────────┐
│  Gemma 4 31B — Prefill Throughput Optimization                       │
│  trn2.48xlarge │ TP=8 │ 100% correctness │ chat 1k/512 │ SDK 2.28.0  │
├──────────────────────────────────────────────────────────────────────┤
│ tok/s                                    ····· roofline bound ·····  │
│  5000┤                                                               │
│      │                                            ╭──●  4,269  ★     │
│  4000┤                                            │      ┌─────────┐ │
│      │                                       ╭────╯      │4,269 t/s│ │
│  3000┤                                  ╭────╯ +150%     └─────────┘ │
│      │                            ╭─────╯                            │
│  2000┤                     ╭──────╯  +38%      ✗ ✗   ← 40 kernel     │
│      │        ╭────●───────╯                          experiments,   │
│  1000┤  ●─────╯    +15%          ✗                    all <1%        │
│      │  845                                                          │
│       └──┬─────────┬──────────┬──────────┬──────────┬────────────┬── │
│       baseline  │ config   │  known   │  borrow  │  invent   │ graph │
│          ┊      ┊  (blue)  ┊ (teal)   ┊ (green)  ┊ (orange)  ┊ (red) │
│      ────┴──────┴──────────┴──────────┴──────────┴───────────┴────── │
│                        ● kept   ✗ discarded                          │
│                     ○ borrowed  ◆ invented  ■ config                 │
└──────────────────────────────────────────────────────────────────────┘
```

### Stage colors

Reusing the reference's palette, extended to our six stages:

| Stage | Color | Hex |
|-------|-------|-----|
| baseline | grey | `#8b949e` |
| config | blue | `#58a6ff` |
| known_kernel | teal | `#39c5cf` |
| borrow | green | `#3fb950` |
| invent | orange | `#f0883e` |
| graph_rewrite | red | `#f85149` |

## 3. The report

Structure adapted from `optimization_report_en.md`, which is a good template:

```markdown
# <Model> Optimization Report

## 1. Results
   Headline table: baseline → final, per shape, MFU before/after, correctness
   The chart

## 2. Core Optimizations
   Each winning change explained with mechanism, not just name.
   Diagram where the change is structural.

## 3. Trajectory
### Stage 1: Config
### Stage 2: Known kernels
### Stage 3: Borrow
### Stage 4: Invent
### Stage 5: Graph rewrite
   Per stage: what was tried, what won, what failed and why.

## 4. What Failed
   Explicit section. The ~40 experiments that went nowhere, grouped by
   hypothesis, with the reason each was abandoned.

## 5. Borrow vs. Invent
   How much came from where. invention_win_rate for this run.

## 6. MFU Methodology
   The formula, stated, so numbers are reproducible and auditable.

## 7. Remaining Bottleneck & Next Directions
   Roofline attainment. What is still on the table.

## 8. Reproduction
   Exact commands. Toolchain stamp. Commit hash of the final config.
```

Section 4 is non-negotiable. The reference's insight that "~40 MoE kernel
experiments: all <1%" is arguably more valuable to the next engineer than
knowing which one worked — it tells them where *not* to spend a day.

## MFU as a secondary metric

The reference uses:

```
MFU = 2 × active_params × tok/s / (peak_TFLOPS_per_core × TP_cores)
```

with `peak = 380 TFLOPS/core` BF16 on Trn2.

Two reasons to adopt it:

1. **It normalizes across models.** 4,269 tok/s on a 3B-active MoE and 4,269
   tok/s on a 31B dense model are wildly different achievements. MFU says so.
2. **It shows absolute headroom.** The reference took Tongyi-30B from 0.28%
   to 4.93% MFU — a a large (multiple-x) speedup that still leaves ~95% of the hardware on
   the table. Publishing that honestly is more useful than a bare multiplier,
   and it sets expectations for how much further Stage 4 could go.

Note the caveat they state: MoE models have structurally low MFU because
active params are a small fraction of total. Compare MFU within an
architecture class, not across.

## Per-turn / per-position curve

For long-context shapes, a single averaged number hides the shape of the
problem. The reference scores on the **average across the last 50% of turns**
specifically to weight long-context steady-state, and plots tok/s per turn.

We do the same: for `rag` and `stress`, emit a secondary chart of throughput
vs. context position. That curve is where attention-scaling problems become
visible — a config that looks fine at 4k and collapses at 60k is invisible in
the average.

## Implementation notes

- `matplotlib` with `Agg` backend, `dpi=180`, `bbox_inches='tight'` (matches
  the reference).
- Read `results.tsv`, group by stage, order by commit order. No hardcoded
  arrays.
- Annotate the top-3 gains automatically by delta magnitude, plus a star on
  the max. Suppress annotations below a threshold (say 5%) so the chart stays
  readable.
- Failure callouts auto-generated: group `discard` rows by stage, count them,
  emit "N experiments in <stage>, all below threshold".
- Roofline line requires the roofline calculator (see
  `optimization-stages.md` Stage 4) — if unavailable, omit the line rather
  than guessing.
- Emit `trajectory.json` next to the chart for machine consumption by the
  leaderboard aggregator.

## Aggregate view (leaderboard-level)

Beyond per-model trajectories, the leaderboard needs a cross-model view:

- **Speedup distribution** across all optimized models, split by architecture
  family. The reference's own data shows MoE (11-17x) versus dense (2.8x) —
  a large and useful difference.
- **Where gains come from**, stacked by stage across all models. Answers "is
  config tuning or kernel work carrying this?"
- **`invention_win_rate` over time**, model index on X. This is the chart that
  answers whether the system ever learns to create rather than copy.
- **Bank hit rate over time.** Should climb. If flat, the bank is not
  earning its keep.
