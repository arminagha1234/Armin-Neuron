# Plan

Phased so we ship visible artifacts early and don't scope-creep the outer
loop into a research project. Each phase has a concrete deliverable and a
"why we stop here" checkpoint.

## Phase 0 — Scope-check (1 week)

Agree on the answers in `open-questions.md` before writing code.

**Resolved:**
- ~~Q1 search strategy~~ → hand-priored greedy
- ~~Q2 compute budget~~ → no cap; terminate on search-quality criteria,
  spend capacity on parallelism (`guardrails.md`)
- ~~Q3 seed models~~ → Gemma 4 31B, then Qwen3.8-27B, then MiniMax M2
- ~~Q4 licensing~~ → direct borrowing OK for Apache-2.0/BSD sources with
  `THIRD_PARTY_NOTICES` + per-file provenance headers
- ~~Q7 "top 100" definition~~ → multi-source weekly ingestion
- ~~Benchmark shapes / ceilings / SDK stamping~~ → all of `guardrails.md`
- ~~Q12 public release ordering~~ → leaderboard first

**Still open (none block starting):**
- Q6: per-family equivalence tolerances — defaults are usable; needs a
  numerics review before publishing results. MoE and hybrid-attention rows
  are the uncertain ones.
- "Muse" needs disambiguation before it can be slotted into the ladder
- MiniMax version pinning (pin one vs. track latest dynamically)
- TensorRT-LLM licensing review — only blocks borrowing from *that* repo

**Deliverable**: signed-off design doc.

### Seed model ladder

| # | Model | Role | Adapter needed | Instance |
|---|-------|------|----------------|----------|
| 1 | Gemma 4 31B | Prove the loop | `dense_causal_lm` (exists) | trn2.3xlarge |
| 2 | Qwen3.8-27B | Generality test | `hybrid_attention_causal_lm` (**new**) | trn2.3xlarge |
| 3 | MiniMax M2 | MoE at scale | `moe_causal_lm` | trn2.48xlarge |

Start with Gemma 4 31B: standard architecture, Apache 2.0, 256K context
(so the `stress` shape is actually exercised), #3 open model on Arena. If
the loop breaks on it, that is the loop's fault — which is what we want
from the first model.

Note that Qwen3.8-27B needs a **new adapter** because Qwen3.5/3.8 use Gated
DeltaNet (linear attention alternating with full attention ~3:1). Budget
extra time for model #2 accordingly — see `open-questions.md` Q3.

## Phase 1 — Three-model proof (~4 weeks)

Build the optimizer end-to-end on three seed models.

### Seed set (resolved — see the ladder table in Phase 0)

| Model | Why |
|-------|-----|
| Gemma 4 31B | Standard transformer, Apache 2.0, 256K context, well-documented. The control case. |
| Qwen3.8-27B | Gated DeltaNet hybrid attention — forces a new adapter, proves the loop generalizes. |
| MiniMax M2 | 230B/10B-active MoE — routing bottlenecks, expert parallelism, needs trn2.48xl. |

### Optimization stages

The loop walks an ordered pipeline per model — cheapest and safest first,
most speculative last. Full detail in
[`optimization-stages.md`](./optimization-stages.md).

| Stage | What changes | Risk | Notes |
|-------|-------------|------|-------|
| 0 Baseline | autoport output | — | The floor. Everything measured against it. |
| 1 Config | nothing (config only) | Low | Highest ROI. Most speedup lands here. |
| 2 Known kernels | proven NKI swaps from bank | Low-med | Check shape constraints, don't assume |
| 3 **Borrow** | port patterns from vLLM/SGLang/TRT-LLM/FA | Med | Steal first — references encode years of tuning |
| 4 **Invent** | novel NKI from profile + roofline | High | Only wins if it beats Stage 3 by **5%** |
| 5 Graph rewrite | fusion, reordering, layout | High | Last, because it interacts with everything prior |

**Borrow before invent** is the governing principle. A CUDA paged-attention
kernel represents thousands of engineer-hours and millions of production
runs — the algorithm is sound even where the implementation is CUDA-specific.
Port that first. Write something novel only where nothing exists to steal, or
where the stolen version leaves real headroom against the roofline bound.

The 5% invention margin is not only about measurement noise (we treat <2% as
noise). It is a risk-adjusted preference: borrowed code has been exercised by
thousands of users across many shapes; our invented kernel has been tested by
us, today, on our shapes. On a near-tie, prefer the better-tested artifact.

Losing invented kernels are **not discarded** — they become `provisional`
lessons recording the attempt and why it lost.

### Phase-1 instrumentation: does it ever invent anything?

Worth answering empirically rather than asserting. Track from run 1:

- `borrow_rate`, `invention_rate` — what fraction of promoted kernels came
  from where
- `invention_attempt_rate`, `invention_win_rate` — is Stage 4 productive or
  flailing
- `roofline_attainment` — is there headroom left at all

Falsifiable prediction: models 1-5 near-zero invention (bank empty, corpus
rich); models 5-20 rising attempt rate as easy borrows exhaust; models 20+
rising win rate *if* the bank is teaching something references lack. If
`invention_win_rate` stays flat past ~30 models, Stage 4 is not earning its
cost — which is a real finding, and publishable either way.

### Optimizer outer loop (v0)

```
initial_config = autoport_baseline(model)
best = measure(initial_config)
budget = 100 trn2-hours

while budget > 0:
    candidates = proposer.next_k(current=best, k=8)
    for cfg in candidates:
        neff = compile(cfg)                          # dominant cost
        if not equivalence_check(neff, reference):
            log_negative_lesson(cfg, "equivalence failed")
            continue
        perf = measure(neff, objective="throughput")
        if perf > best.perf:
            best = cfg
            log_positive_lesson(cfg, perf)
    if no_improvement_streak >= 3: break            # stopping criterion
```

Proposer in v0: hand-priored greedy. Priors come from what the Neuron team
already knows (TP degrees that make sense for Trn2, quantization safe
zones, attention kernel choices, etc.). Not learned yet.

Objective knob: throughput or latency, picked per run. Never optimize both
at once in v0 — we don't need Pareto fronts yet.

All benchmark shapes, batch sweeps, HBM/compile ceilings, and stopping
criteria come from [`guardrails.md`](./guardrails.md). The pseudocode above
elides them for readability; they are hard gates in the real loop.

Note the two-tier measurement strategy — during search we probe with `chat`
at batch 1 and 32 only; the full 4-shape × full-batch sweep runs once on the
winner. Without this, measurement time swamps compile time.

### Deliverables

- [ ] Optimizer CLI: `optimize --model <hf-id> --objective <throughput|latency> --budget-hours N`
- [ ] Per-model recipes committed to repo (config YAML + generated NKI
      kernels + measurements)
- [ ] Blog post: "How we got X tok/s on Qwen 3.5 on Trn2, autonomously"
- [ ] Verification: every recipe reproducible from `main` with one command

### Why we stop here

Three models is enough to prove the loop works and generalizes across
architectures. Anything less risks overfitting. Anything more risks
building v0 as our permanent shape.

## Phase 2 — Knowledge bank (~4 weeks, can overlap Phase 1)

The bank is what makes phase 3 tractable, so start designing it during
phase 1 and populate it from phase 1 runs.

See [`knowledge-bank.md`](./knowledge-bank.md) for the schema and the seed
content.

### Deliverables

- [ ] Lesson schema (structured YAML) checked in
- [ ] 30-50 seed lessons hand-authored from existing team knowledge
- [ ] Bank retrieval API the optimizer's proposer calls
- [ ] Static-site render of the bank (markdown + metadata → browsable HTML)
- [ ] Every phase 1 run automatically emits lessons

## Phase 3 — Top-100 leaderboard (~8 weeks)

Now the flywheel spins. The knowledge bank means model 100 optimizes in
~1/5 the compute of model 1.

### Scope: Track A (text-to-text) only

**V1 is LLMs only.** Image, video, and speech tracks are designed (see
[`guardrails.md`](./guardrails.md#per-track-benchmark-shapes) and
[`model-landscape.md`](./model-landscape.md)) but deliberately deferred.

| Track | Modality | Status |
|-------|----------|--------|
| **A** | **Text-to-text** | **V1 — build this** |
| B | Text-to-image | Designed, deferred |
| C | Text-to-video | Designed, deferred |
| D | Speech (ASR) | Designed, deferred |
| E | Text-to-speech | Designed, deferred |

Why keep the other four specified but unbuilt: it forces the outer loop to
stay track-agnostic from day one. Shapes, metric, and equivalence live in the
per-family adapter, so adding Track B later is writing one adapter — not
refactoring the core. Designing for five and shipping one is cheap; designing
for one and retrofitting four is not.

Track B is the natural second, because image generation forces non-token
shapes and non-token equivalence — the real test of whether the framework
generalizes past LLMs.

Current top models per track are in
[`model-landscape.md`](./model-landscape.md) as an Aug-2026 hand-curated
baseline; the discovery job replaces it with live weekly data.

### Model discovery (weekly, multi-source)

A scheduled job assembles the ranked list. No single source is
authoritative — each has a bias, so we union and re-rank.

| Source | Signal it provides | Track coverage | Bias to correct for |
|--------|-------------------|----------------|---------------------|
| LMArena text leaderboard | Human-preference quality | A | Skews chat-tuned, English |
| LMArena text-to-image leaderboard | Human preference on images | B | Aesthetic-preference bias |
| HuggingFace trending | Momentum / what's new | all | Spiky, hype-sensitive |
| HF downloads (30d) | Actual deployed usage | all | Lags; favors older models |
| HF Open LLM Leaderboard | Benchmark scores | A | Gameable, accuracy-only |
| Artificial Analysis Intelligence Index | Composite capability score | A | Single-vendor methodology |
| ASR leaderboards (WER-based) | Transcription accuracy | D | English-heavy |
| AWS customer signal (manual) | What customers pay for | all | Small sample, highest business value |

```
discovery/
  sources/
    lmarena.py           # scrape/API the Arena leaderboard
    hf_trending.py       # HF API, sort=trending
    hf_downloads.py      # HF API, downloads last 30d
    hf_open_llm.py       # Open LLM Leaderboard
    customer_signal.py   # reads a checked-in YAML list, human-maintained
  rank.py                # weighted borda count across sources
  filters.py             # open weights / license / inference-ready
  snapshot.py            # writes dated top-100.yaml, version-stamped
```

Rules:
- Runs weekly. Each run writes a dated snapshot (`top-100-2026-08-14.yaml`)
  so the leaderboard is always traceable to a list version.
- `customer_signal` entries carry an **override flag** — a specifically
  requested model always makes the list regardless of public ranking.
- Models that drop off the list keep their published recipe but stop
  getting refreshed. Mark them `archived`, don't delete. Nothing published
  ever silently disappears.
- New arrivals get queued by rank; the runner works down the queue.

### Per-architecture-family adapters

The outer loop is universal; the search space isn't. Ship adapters for:

**Track A (V1 — build these):**
- `dense_causal_lm` — Llama-family, Mistral, Phi, Gemma, Muse Glimmer
- `hybrid_attention_causal_lm` — Qwen3.5/3.8 Gated DeltaNet, Kimi K3 KDA, and
  anything else mixing linear + full attention. A growing category.
- `moe_causal_lm` — Mixtral, DeepSeek V4, GLM-5.2, MiniMax M-series
- `encoder_only` — BERT-family, embeddings (cheap to add, quick wins)

**Deferred (designed, not built):**
- `diffusion` — Stable Diffusion, FLUX, Qwen-Image, Z-Image
- `video_diffusion` — LTX-2.5, Wan 2.2, HunyuanVideo
- `asr` — Whisper, Canary-Qwen, Parakeet
- `tts` — Kokoro

Each adapter defines: what config axes exist, what kernel patterns are
worth trying, what equivalence tolerance to use.

### Deliverables

- [ ] Public leaderboard site — **Track A only** (perf, cost, recipe per model)
- [ ] Multi-source weekly discovery job + dated list snapshots
- [ ] Per-family adapters for Track A: `dense_causal_lm`,
      `hybrid_attention_causal_lm`, `moe_causal_lm`
- [ ] Reproducibility script per leaderboard entry
- [ ] Weekly list refresh + SDK-release re-verification loop
- [ ] All four Track-A shapes reported per model, with `not_applicable` /
      `oom` distinguished from `failed` (see `guardrails.md`)
- [ ] Borrow/invent metrics published alongside perf — this is a
      differentiated story no other leaderboard tells

## Phase 4 — Learned search policy (later)

Only after we have ~50 leaderboard entries. Data volume matters more than
algorithm choice here.

Options in rough order of complexity:
1. Learned cost model (predict measured perf from config without
   recompiling) — highest ROI, standard AutoTVM pattern
2. Bayesian optimization with a learned prior over configs
3. Full RL over the search space (last resort, high engineering cost)

### Deliverables

- [ ] Cost model trained on leaderboard measurements
- [ ] A/B: hand-priored greedy vs. learned proposer on next 10 leaderboard
      entries. Ship the winner.

## Explicit non-goals per phase

- **Phase 1**: no learned search, no leaderboard, no bank populated from
  external corpora. Prove the loop on 3 models.
- **Phase 2**: no auto-populating the bank from vLLM/SGLang scraping.
  Human-curated only until we understand the schema.
- **Phase 3**: no chasing "we beat Nvidia" narratives. Publish honest numbers
  with perf-per-dollar. If we lose on some models, publish that too — it's
  what makes the leaderboard trustworthy.
- **Phase 4**: no full-blown NAS. We're optimizing existing models, not
  designing new ones.

## Risk register (see `open-questions.md` for the resolution plan)

| Risk | Mitigation |
|------|------------|
| Compile time dominates → search too slow | NEFF caching, subgraph reuse, per-iter compile budget |
| Autonomous loop makes silent equivalence-breaking changes | Equivalence agent is a hard gate; positive results without pass = discarded |
| Seed-model bias in the bank | Diversify seeds early, weight confidence by architecture diversity |
| Neuron SDK regressions invalidate lessons | SDK-version stamp on every lesson, spot-recheck top-N on each release |
| Compute cost balloons on the leaderboard | Per-model budget cap, warm-start from bank aggressively |
| Licensing on "borrowed" kernel patterns | Semantic borrowing only; attribute in-lesson; never copy code verbatim |
