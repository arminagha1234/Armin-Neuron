# Guardrails

Fixed limits and benchmark shapes. These make measurements comparable
across models and runs, and stop one pathological config from eating a
whole run.

Everything here is a **hard constraint on the optimizer**, not a
suggestion. Violating a guardrail means the candidate is rejected, not
warned about.

## Benchmark shapes

Throughput is meaningless without a fixed input/output shape. Long input is
prefill-bound (compute-heavy, big matmuls). Long output is decode-bound
(memory-bandwidth-heavy, one token at a time). A config that wins one can
lose the other, so we measure all four.

| Shape | Input tokens | Output tokens | Real-world analog | Primarily stresses |
|-------|-------------|---------------|-------------------|--------------------|
| `chat` | 1,024 | 512 | Chatbot turn | Balanced — the common case |
| `rag` | 10,240 | 512 | Doc summarization, RAG | Prefill / compute |
| `generate` | 512 | 10,240 | Agentic loops, code gen | Decode / memory bandwidth |
| `stress` | 65,536 | 65,536 | Long-context customer ask | Both, plus KV-cache pressure |

`stress` is deliberately extreme. A real customer asked for 64k/64k, so we
test it even though most models won't fit it at any useful batch size. When
it doesn't fit, that's a **result**, not an error — see "When a shape
doesn't fit" below.

### Sequence-length note for `stress`

64k in + 64k out means the KV cache must hold up to 128k positions at peak
(end of generation). For a 32B GQA model that is tens of GB before you
count weights. Expect:

- Most models to require `tp_degree >= 8` just to fit `stress` at batch 1
- Paged / block-sparse attention to be mandatory rather than optional
- Some models to be structurally incapable (max_position_embeddings < 128k)
  — record as `not_applicable`, not `failed`

## Batch sweep

Full power-of-2 sweep, but **shape-aware** because the big shapes won't fit
at high batch:

| Shape | Batch sizes swept |
|-------|-------------------|
| `chat` | 1, 2, 4, 8, 16, 32 |
| `rag` | 1, 2, 4, 8, 16, 32 |
| `generate` | 1, 2, 4, 8, 16, 32 |
| `stress` | 1, 2, 4 (higher will OOM on most models; attempt and record) |

Batch 1 is the latency number. Batch 32 is the throughput number. The
middle points show us the scaling curve, which is what tells us whether a
config is bandwidth-bound or compute-bound.

### Two-tier measurement (important for search speed)

Measuring all 4 shapes × 6 batches = up to 24 measurements per candidate.
At a couple of minutes each that is ~45+ min of measurement on top of every
compile. Too slow to do for every candidate during search.

So:

- **During search** — measure a cheap probe: `chat` at batch 1 and batch 32.
  Two measurements, enough signal to rank candidates.
- **On the final winner** — run the full 4-shape × full-batch sweep. This
  is what gets published to the leaderboard.

Same final data quality, roughly 10x less measurement time during search.

## Resource ceilings

### HBM ceiling: 85% of available

Reject any config whose **peak** HBM exceeds 85%.

Critical detail: peak HBM for a generation workload happens at the *end*,
when the KV cache is fully populated — not during prefill. So the check
must be performed at full sequence occupancy (`input + output` positions),
not at step 0. A config that looks fine at token 1 and OOMs at token 65,536
is the exact failure we're preventing.

Why 85% and not 95%: fragmentation, allocator overhead, and the fact that a
config sitting at 94% will OOM on a slightly longer request in production.
Headroom is not waste.

### Compile timeout: 30 minutes per candidate

Kill any NEFF compile exceeding 30 min. Configurable per family — some
large MoE models legitimately need longer, and the adapter can raise it.

Distinguish two outcomes:
- **Timeout** → candidate rejected, emit an `anti_pattern` lesson noting
  the config shape that blew up compile time
- **Slow but completed** (e.g. 25 min) → valid candidate, but record
  `compile_time_seconds` so we can prefer faster-compiling equivalents

### Measurement stability

Each measurement must run enough tokens to be trustworthy:
- Minimum 3 warmup iterations (discard)
- Minimum 10 measured iterations, or 30 seconds of sustained generation,
  whichever is longer
- Report p50 and p99, not just mean — p99 is what a user actually feels
- Reject a measurement whose p99/p50 ratio exceeds 3.0 (too noisy, rerun)

## Stopping criteria

Compute budget is **not** a constraint here — we have the capacity. But the
search still needs termination conditions, for two reasons that have
nothing to do with cost:

1. **Diminishing returns are real.** Past a point the optimizer is
   measuring noise, not finding wins. Continuing produces false "gains"
   that don't reproduce.
2. **Throughput across the leaderboard matters more than depth on one
   model.** An optimizer stuck on model 1 forever means models 2-100 never
   get done. With lots of compute, the right move is **parallelism across
   models**, not infinite search per model.

### Termination conditions (any one triggers stop)

| Condition | Default | Rationale |
|-----------|---------|-----------|
| No-improvement streak | 5 rounds | Greedy has plateaued |
| Marginal improvement | < 2% for 3 consecutive rounds | Measuring noise |
| Max iterations | 100 candidates | Hard backstop |
| Search space exhausted | all single-axis moves tried | Greedy is done by definition |

### Use the compute on parallelism instead

Given unlimited capacity, the scaling lever is horizontal:

- Run **N models concurrently**, each on its own instance
- Within a model, compile **candidates in parallel** across instances
  (compile is the bottleneck, and candidates are independent)
- Keep a NEFF cache keyed on config-subgraph hash so repeated subgraphs
  across candidates and across models are never recompiled

This turns "unlimited compute" into "the whole leaderboard refreshes fast,"
which is the actual goal.

## When a shape doesn't fit

Not every model can run every shape. Three distinct outcomes, recorded
differently:

| Outcome | Meaning | Recorded as |
|---------|---------|-------------|
| `ok` | Ran, measured, equivalence passed | Normal result |
| `not_applicable` | Model structurally can't do it (e.g. `max_position_embeddings` = 8k, asked for 64k) | Leaderboard shows "—", not a failure |
| `oom` | Model could in principle, but no config found that fits under the HBM ceiling | Leaderboard shows "OOM at batch N", plus an `anti_pattern` lesson |
| `failed` | Something broke — compile error, equivalence failure, crash | Investigate; do not publish |

Keeping `not_applicable` and `oom` distinct from `failed` matters. A model
that can't do 64k context isn't broken, and the leaderboard shouldn't imply
it is.

## Equivalence gate (recap)

Equivalence is a guardrail too — the hardest one. A faster config that
produces different outputs is not a win, it is a bug.

Per-family tolerances live in the adapter (see `architecture.md`). Defaults:

| Family | Numerical | Behavioral |
|--------|-----------|------------|
| Dense causal LM | `rtol=1e-3, atol=1e-5` | Top-1 token match on greedy decode, 100+ positions |
| MoE causal LM | `rtol=1e-3, atol=1e-5` on final logits | Allow routing divergence if logits agree |
| Diffusion | — | LPIPS / FID window vs. reference image |
| Speech | — | WER within delta on a fixed benchmark set |
| Encoder-only | `rtol=1e-4, atol=1e-6` | Cosine similarity > 0.999 on embeddings |

No candidate is ever promoted on performance alone. Equivalence passes
first, then performance is considered.

## Summary table (the numbers to remember)

| Guardrail | Value |
|-----------|-------|
| Shapes | `chat` 1k/512, `rag` 10k/512, `generate` 512/10k, `stress` 64k/64k |
| Batch sweep | 1-32 powers of 2 (1-4 for `stress`) |
| Search-time probe | `chat` @ batch 1 and 32 only |
| HBM ceiling | 85% at peak (full KV occupancy) |
| Compile timeout | 30 min per candidate (family-overridable) |
| Warmup / measured | 3 / 10 iterations minimum |
| No-improvement stop | 5 rounds |
| Max iterations | 100 |
| Compute budget | Uncapped — use it on parallelism, not depth |

## Neuron SDK version tracking

Every optimization result is stamped with the exact toolchain that produced
it, and every SDK release triggers a re-verification pass. Without this the
knowledge bank silently rots — see `knowledge-bank.md` for why stale lessons
are the highest-severity quality risk.

### Stamped on every result

Emitted at the end of every optimization run, alongside the measurements:

```yaml
toolchain:
  neuron_sdk: 2.28.0              # umbrella release version
  neuronx_cc: 2.26.6360.0         # compiler — the one that matters most
  torch_neuronx: 2.11.3.0.1419
  nki: 0.5.0
  neuron_driver: 2.29.0.0         # aws-neuronx-dkms on the host
  instance_type: trn2.48xlarge
  ami_id: ami-0abc123
  measured_at: 2026-08-14T20:34:53Z
```

`neuronx_cc` is called out because compiler changes are the most common
cause of a previously-valid lesson going stale. Two runs on the "same" SDK
but different compiler builds are not comparable.

### On every SDK release

**Trigger source.** The pass below is kicked off by the
[Neuron *What's New* / release-notes feed](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/whats-new.html),
checked at the start of each run. A newer `neuron_sdk` / `neuronx_cc` version
(or a dated changelog entry we haven't seen) starts the pass. The changelog
also says *what* changed, which focuses the work: a compiler/kernel entry means
re-verify kernel lessons first; a new fusion or attention kernel is a candidate
new config axis (step 3 below).

A re-verification pass, in priority order:

1. **Re-verify the top-N most-consulted bank lessons** (N ~ 50). Re-run the
   measurement that justified each one. Outcomes:
   - Still holds → extend `neuron_sdk_versions`, bump `last_reverified`
   - No longer holds → mark stale, move to `provisional/` for triage,
     open an investigation
   - Now unnecessary (compiler does it automatically) → mark superseded,
     record which SDK version obsoleted it
2. **Re-run the leaderboard's published configs** for the current top ~20
   models. Publish a delta: what got faster, what regressed.
3. **Check for new compiler capabilities** worth adding as config axes — a
   new fusion pass or attention kernel may open search space that did not
   exist before.

Regressions are as newsworthy as improvements. If SDK 2.29 makes a
published recipe 15% slower, that goes in the delta report and gets filed
against the compiler team. Silently re-tuning around a regression hides
signal they need.

### Bank staleness policy

| Lesson age (SDK versions since last verified) | Proposer treatment |
|---|---|
| 0 (current) | Full confidence weight |
| 1 minor bump | Full weight |
| 2 minor bumps | Down-weighted |
| 3+ minor bumps, or any major bump | Not used until re-verified |

This makes staleness *visible* rather than silent. A lesson that stops being
consulted because it aged out shows up in the bank metrics
(`stale_ratio`), which is the trigger to go re-verify it.

---

# Per-track benchmark shapes

Everything above this line is **Track A (text-to-text)**. Token counts are
meaningless for image, video, and audio generation, so each modality track
needs its own shapes, its own primary metric, and its own equivalence method.

The **outer loop is identical across tracks**. Only these three things change
per track — which is exactly what the per-family adapter owns.

## Track A: Text-to-text

| | |
|---|---|
| **Shape axis** | input tokens x output tokens |
| **Shapes** | `chat` 1k/512, `rag` 10k/512, `generate` 512/10k, `stress` 64k/64k |
| **Batch sweep** | 1, 2, 4, 8, 16, 32 (`stress`: 1, 2, 4) |
| **Primary metric** | tokens/sec (throughput), TTFT + per-token p50/p99 (latency) |
| **Equivalence** | Top-1 token match on greedy decode, 100+ positions |

## Track B: Text-to-image

| | |
|---|---|
| **Shape axis** | resolution x denoising steps |
| **Primary metric** | images/sec (throughput), seconds/image at batch 1 (latency) |
| **Equivalence** | LPIPS vs. reference image, plus FID over a fixed prompt set |

| Shape | Resolution | Steps | Analog |
|-------|-----------|-------|--------|
| `standard` | 1024x1024 | 28 | Default quality generation |
| `turbo` | 1024x1024 | 4 | Distilled / Turbo checkpoints (Z-Image-Turbo etc.) |
| `highres` | 2048x2048 | 28 | Print / upscale workflows |

Batch sweep: 1, 2, 4, 8. Latent tensors are large — 8 is often the practical
ceiling at 1024x1024 and `highres` may cap at 1-2.

Note: step count is *both* a shape axis and a config axis. Fix it per shape
for benchmarking, but the optimizer may separately explore step-reduction
(distillation, better schedulers) as an optimization — those results go in a
different column, because fewer steps changes output quality, not just speed.

## Track C: Text-to-video

| | |
|---|---|
| **Shape axis** | resolution x frame count x denoising steps |
| **Primary metric** | seconds per generated video (the number users feel) |
| **Secondary** | generated-frames/sec, HBM peak |
| **Equivalence** | Per-frame LPIPS + temporal-consistency check (adjacent-frame delta distribution) |

| Shape | Resolution | Frames | Steps | Analog |
|-------|-----------|--------|-------|--------|
| `short` | 480p | 121 (5s @ 24fps) | 30 | The common case, low-VRAM entry |
| `hd` | 720p | 121 (5s @ 24fps) | 30 | Production quality |
| `long` | 480p | 361 (15s @ 24fps) | 30 | Duration stress — memory scales with frames |

Batch sweep: 1, 2. Video latents are enormous; batch >2 is unrealistic for
most models and instances.

Temporal consistency matters for equivalence in a way it does not for images:
a config could produce individually-plausible frames that flicker. Per-frame
LPIPS alone would pass that. The adjacent-frame delta check catches it.

## Track D: Speech (ASR / STT)

| | |
|---|---|
| **Shape axis** | audio clip duration |
| **Primary metric** | Real-Time Factor (RTF) = audio_duration / processing_time — higher is better |
| **Secondary** | latency to first token (for streaming), throughput in audio-hours/hour |
| **Equivalence** | Word Error Rate delta vs. reference on a fixed benchmark set |

| Shape | Duration | Analog |
|-------|----------|--------|
| `utterance` | 30 s | Voice command, dictation snippet |
| `meeting` | 5 min | Meeting segment |
| `longform` | 1 hr | Podcast / lecture transcription (chunking behavior matters here) |

Batch sweep: 1, 8, 32. Audio features are small so high batch is realistic.

`longform` specifically tests chunking and context-carryover logic, which is
where ASR implementations most often diverge from each other.

## Track E: Text-to-speech

| | |
|---|---|
| **Shape axis** | input text length |
| **Primary metric** | RTF (generated audio duration / wall time) |
| **Equivalence** | Mel-spectrogram distance vs. reference, plus a speaker-similarity check if voice cloning is involved |

| Shape | Input | Analog |
|-------|-------|--------|
| `sentence` | ~20 words | UI notification, short response |
| `paragraph` | ~200 words | Article read-aloud |
| `document` | ~2000 words | Audiobook chapter |

Batch sweep: 1, 8, 32.

## What stays constant across all tracks

These are track-independent and enforced everywhere:

| Guardrail | Value |
|-----------|-------|
| HBM ceiling | 85% at peak |
| Compile timeout | 30 min per candidate (family-overridable) |
| Warmup / measured iterations | 3 / 10 minimum |
| p99/p50 noise rejection | reject and rerun above 3.0 |
| No-improvement stop | 5 rounds |
| Max iterations | 100 |
| Two-tier measurement | cheap probe during search, full sweep on winner |
| Equivalence gate precedence | equivalence passes *before* performance is considered |
| Toolchain stamping | full SDK/compiler versions on every result |
| Outcome taxonomy | `ok` / `not_applicable` / `oom` / `failed` |

The **two-tier measurement** probe differs per track — pick the cheapest
representative shape:

| Track | Search-time probe |
|-------|------------------|
| A (text) | `chat` @ batch 1 and 32 |
| B (image) | `standard` @ batch 1 and 4 |
| C (video) | `short` @ batch 1 |
| D (ASR) | `utterance` @ batch 1 and 8 |
| E (TTS) | `sentence` @ batch 1 and 8 |
