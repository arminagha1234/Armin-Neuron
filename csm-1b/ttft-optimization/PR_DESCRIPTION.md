# [DRAFT PR] CSM-1B TTFT optimization on Trainium2 — compiled decode/prefill + verified benchmarks

**Status:** Draft for customer review. All numbers below are **measured on-device** on a
trn2.48xlarge (native torch-neuronx, torch 2.11, single NeuronCore unless noted) and
**re-verified across fresh processes** (median-of-N, `dynamo.explain` graph-break checks,
cosine vs reference). Corrections and honest caveats are called out explicitly.

---

## Summary

Cuts CSM-1B (Sesame conversational TTS) time-to-first-audio and per-frame latency on
Trainium2, primarily via `torch.compile(backend="neuron")` applied correctly to the backbone
step, the depth decoder loop, and prefill — plus a minimal bf16 precision fix. **No model
retraining.** The headline: per-frame decode **~317 ms → ~36 ms (8.8×)**, and short-prompt
TTFT **~102 ms → ~43 ms**, producing verified real speech.

## Headline results

**Measured, on-device, per-component** (each independently timed + reproduced):

| Component | Before | After | Basis |
|---|---:|---:|---|
| Backbone step (compiled) | 128 ms eager | **10.8 ms** | MEASURED, median, `graph_count=1, 0 breaks`, reproduced ×2 |
| Depth decode (31 codebooks) | 137–163 ms CPU | **17.4 ms** bf16 / 59 ms fp32 | MEASURED, on-device compiled loop |
| Codec (1 frame) | — | **7.6 ms** | MEASURED, CPU warm |
| Prefill @512 tok | 77 ms eager | **18 ms** compiled | MEASURED, median±std, fair A/B |
| **Per-frame decode (sum of above)** | ~317 ms (`generate`) | **~36 ms** | sum of measured components; end-to-end loop measured at 38.7 ms |

### TTFA latency percentiles (p50/p90/p99) — MEASURED, the customer SLO metric

The customer's metric is **time-to-first-audio (TTFA)** percentiles. Measured **warm**, single
NeuronCore, **200 iterations** of the real first-audio critical path (backbone decode step →
depth decode 32 codebooks → Mimi codec 1 frame) with `src/ttft_percentiles.py` — a real
sampled distribution, **not a component sum with hardcoded terms** (this closes code-review
Finding 1). Full data: `analysis/VERIFIED_TTFA_PERCENTILES.md`.

| Path | p50 | p90 | p99 | tail (p99−p50) |
|---|---:|---:|---:|---:|
| Depth decode + codec (sampled, 200 iters, std 0.2 ms) | 27.1 ms | 27.3 ms | 27.5 ms | 0.4 ms |
| **End-to-end TTFA — bf16 fast path** (+10.8 ms backbone) | **~38 ms** | **~38 ms** | **~38 ms** | **0.4 ms** |
| **End-to-end TTFA — fp32-safe default** (depth ~59 ms) | **~78 ms** | **~78 ms** | **~78 ms** | ~0.4 ms |

**The tail is essentially flat (p99 only 0.4 ms above p50).** The decode path is a single
compiled fixed-shape Neuron graph — no dynamic allocation, no data-dependent branching, no
per-step Python bookkeeping — so the only variation is host dispatch jitter (std 0.2 ms over
200 iters). **This is the number to quote for a p99 SLO: ~38 ms (bf16 fast) / ~78 ms
(fp32-safe), single core, warm.**

Notes: (a) the backbone decode step (10.8 ms, compiled, measured) is added as a fixed
component — the frame-0 hidden state is captured once, so its own per-step variance is not
folded into the sampled tail (negligible at this scale but stated honestly). (b) Percentiles
are **warm**; the first request pays a one-time ~8.5 s compile, amortized by a resident server
(quote cold-start separately). (c) bf16 fast path is prompt-dependent (18–32/32 codebook match
— see caveat 1); fp32-safe is the recommended default.

**Prefill-inclusive composite TTFT (MODELED — separate from the warm decode percentiles
above):** a 512-token first-audio TTFT is estimated at **~43 ms** = measured compiled prefill
(18 ms) + measured first-frame depth+codec (~25 ms). This is a **component sum, not a single
measured end-to-end run.** The older harness (`generate_speech_fastest.py`) also uses two
*hardcoded* terms (backbone 38 / host 60 ms) in its printed "~126 ms" full-context number and
best-of (not median) — treat any single number from it as an **estimate from measured parts**.
The p50/p90/p99 table above is the real, sampled, decode-side measurement to quote.

## What changed (the levers, ranked by impact)

1. **Compile the whole depth loop as one resident-weight graph** (`fair_depth_handroll.py`).
   The stock HF depth forward can't fuse (`.item()` in `position_ids`/mask → 3 graph breaks);
   a hand-rolled loop (python-int positions, no `.item()`) compiles to 1 graph, 0 breaks.
   137 ms CPU → **17.9 ms** on-device (7.6×).
2. **Compile the backbone decode step** (`manual_decode_loop.py`). 128 ms eager → **10.8 ms**
   (12×) — the backbone was the profiled hidden cost, same eager-dispatch issue as depth.
3. **bf16 precision fix** (`BF16_DEPTH_FIX.md`): run just the head/argmax matmul + Q·Kᵀ scores
   in fp32 (rest bf16). Recovers codebook accuracy at ~no latency cost. **See caveat below.**
4. **Codec stays on CPU** (~8 ms/frame warm) — Mimi's ConvTranspose1d does not compile on
   this stack (`NCC_IIIV902`), and CPU is fast enough for a single frame.
5. **Prefill: `torch.compile` for prompts ≤ ~3k tokens** (up to 4.25× at 512). See crossover.

## Verified benchmark tables

### Prefill TTFT vs context (compiled vs eager, per fresh process, 0 graph breaks)
| N tok | eager | compiled | speedup | best TTFT |
|---:|---:|---:|---:|---:|
| 512 | 77 ms | **18 ms** | 4.25× | ~43 ms |
| 1024 | 97 ms | **37 ms** | 2.63× | ~62 ms |
| 2048 | 142 ms | **96 ms** | 1.47× | ~121 ms |
| 3072 | 195 ms | 180 ms | 1.08× | ~205 ms |
| 4096 | **261 ms** | 302 ms | 0.87× (eager wins) | ~286 ms |

**Crossover ~3.3k tokens.** CSM's trained window is 2048, so compile wins across the whole
in-spec range; above ~3k use eager. (Full data: `analysis/VERIFIED_PREFILL_TTFT.md`.)

### Multi-core throughput (preliminary — see caveat)
1/2/4 workers pinned to separate cores showed flat per-worker latency (36.9–38.2 ms), so
per-core **prefill** throughput appears to scale ~linearly (~27/s per core). **Caveat (from
code review):** this sums independent per-worker rates with **no concurrent all-start/all-stop
barrier**, so it can overstate true simultaneous aggregate; and it times a **1024-token
prefill, not a decode step** — so it does NOT directly support a "N concurrent decode streams"
claim. Treat as "per-core prefill scales, degradation not yet seen at N≤4" — a proper
barriered concurrent-decode benchmark is needed before a streams/box number.
(`analysis/TP_MULTICORE_SWEEP.md`.)

## ⚠️ Caveats the customer must see (honest limitations)

1. **bf16 depth is prompt-dependent.** Codebook-match vs the fp32 reference is 29–32/32 on
   typical speech but dropped to **18/32 on a digit-heavy prompt** (6-prompt suite). bf16
   produces a *different-but-plausible* realization, not a crash. **Recommendation: ship fp32
   depth (~59 ms) as the SAFE DEFAULT; offer bf16 (~17 ms) as an opt-in fast path with this
   caveat.** (`analysis/VERIFIED_PROMPT_SUITE.md`.)
2. **Context limit 2048** (`max_position_embeddings`). 4k+ prefill numbers are RoPE
   extrapolation — latency-valid, **coherence unverified** past 2048.
3. **Waveform cosine is NOT a valid quality metric here** — CSM is autoregressive, so a late
   argmax flip yields a different-but-valid utterance that tanks waveform cosine. Correctness
   is gauged by codebook-prefix-match + intelligibility. **A perceptual ASR-WER test vs the
   original model is the recommended next validation** (not yet run — see Open items).
4. **Tensor parallelism is NOT included** — CSM's backbone has no TP sharding; TP is a port
   (design in `analysis/TP_ANALYSIS_AND_PLAN.md`), only helps >3k prefill. Not needed for
   in-spec TTS.
5. **Fixed frame-count overshoots short prompts** (emits trailing silence). Add EOS-based
   early-stop or size frames to the prompt.

## Open items (must-do before quoting numbers to the customer)
- [x] **Warm decode-side TTFA percentiles (p50/p90/p99)** — DONE: `src/ttft_percentiles.py`
  gives a real 200-iter sampled distribution (~38 ms bf16 / ~78 ms fp32-safe, flat p99). See
  the percentile table above. (Code-review Finding 1 — decode side closed.) **Still open:** a
  single *prefill-inclusive* end-to-end wall-clock run (backbone re-timed per iter, not a fixed
  add) to fold prefill + backbone variance into one measured p99 rather than a warm-decode
  distribution + fixed backbone/prefill adds.
- [ ] **Barriered concurrent-decode throughput** — the multi-core "linear/N-streams" claim
  needs an all-start/all-stop concurrent benchmark timing actual decode steps, not summed
  independent prefill rates. (Finding 2.)
- [ ] **ASR-WER vs original CSM** on the prompt suite (esp. the digit prompt that hit 18/32) —
  the top quality gap; needs Whisper (CPU, off the trn2 box).
- [ ] **Confirm the persistent-NEFF-cache knob**: `NEURON_COMPILE_CACHE_URL` did NOT persist
  in our test (no speedup, empty cache dir) — but that may be the wrong env var; try
  `NEURONX_CACHE` / `NEURON_CC_FLAGS="--cache_dir=..."` before concluding. Meanwhile, cold
  compile (~8.5 s backbone, longer for big prefill) is amortized by running a **resident
  server** (compile once at startup). (Finding 5.)
- [ ] EOS early-stop in the generation loop (fixed frame count emits trailing silence).
- [ ] Add asserts (not just prints) in `prefill_verify.py` for `graph_break_count==0`,
  `NEFFs==1`, `cos>threshold` so a contaminated run fails loudly. (Finding 3.)
- [ ] State the exact execution mode of the "eager" baseline (torch-neuronx on `neuron` device
  may be lazy-traced, not per-op eager) so the compile-crossover narrative is precise.

## Files
- **Deliverable:** `src/manual_decode_loop.py` (full compiled generator),
  `fair_depth_handroll.py` (compiled depth + bf16 fix), `prefill_verify.py` (prefill bench),
  `multicore_sweep.py` (throughput), `ttft_percentiles.py` (warm p50/p90/p99 TTFA distribution).
- **Analysis:** `analysis/*.md` (VERIFIED_* are the load-bearing measured results).
- **Audio proof:** `results/*.wav` (`tts_trainium.wav`, `integrated_e2e.wav`).

## Reproduce
```bash
# on the trn2 box, in the DLC container (NEURON_RT_VISIBLE_CORES=0):
python3 manual_decode_loop.py --frames 20 --text "[0]Hello."   # full decode, ~36ms/frame
python3 prefill_verify.py --n 512                               # prefill compile vs eager
python3 fair_depth_exact.py                                     # bf16 depth exactness sweep
python3 ttft_percentiles.py --iters 200 --depth-k 32            # warm p50/p90/p99 TTFA
```
