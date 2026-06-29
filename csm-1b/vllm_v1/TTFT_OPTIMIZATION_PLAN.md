# CSM-1B on Trainium — TTFT / Latency Optimization Plan

**Goal:** drive CSM-1B text-to-speech latency down on a **single NeuronCore**.
Customer states **<500 ms TTFT**, really wants **<100 ms**. This plan defines what
TTFT means for TTS, where the time goes, and a ranked, staged path — with an honest
read on what's achievable single-core vs. with a few cores.

**Owner:** Armin · **Status:** PLAN (brainstorm) · **Created:** 2026-06-28

---

## 0. What "TTFT" means for a TTS model (this reframes everything)

CSM is autoregressive over **audio frames**. Mimi runs at **12.5 frames/s**, so each
frame ≈ **80 ms of audio**. Per frame the model does:
- **1 backbone step** (16-layer Llama, predicts codebook 0), then
- **31 sequential depth-decoder steps** (codebooks 1..31), then
- the frame's 32 codes go to the **Mimi decoder** → waveform.

For an interactive voice agent, the metric that matters is **time-to-first-audio
(TTFA)** — when the speaker starts producing sound — not time to the *whole*
utterance. So:

```
TTFT(audio) = T_prefill            (backbone over the text prompt, once)
            + T_frame0             (1 backbone step + 31 depth steps)
            + T_codec(1 frame)     (Mimi decode of the first frame, streaming)
```

**This is the single biggest insight:** if we **stream** (emit frame 0's audio as soon
as it's ready) instead of generating the full clip then decoding once (what HF
`generate(output_audio=True)` does today), TTFT drops from "whole-utterance time" to
"~one-frame time." Everything below assumes we move to streaming generation.

---

## 1. Where the time goes today (and what to measure first)

Current shipped path (`generate_speech.py`): backbone + Mimi offloaded to NeuronCore
(fp32), depth decoder + generate loop on CPU, **per-shape recompiles** (cold 24-frame
run was ~740 s — almost all compile, not compute).

**We have never measured a clean WARM per-frame breakdown** — that's Stage 0 and gates
every decision. Expected warm decomposition on a single core (to be confirmed):

| Component | Per call | Per frame | Notes |
|---|---|---|---|
| Backbone step (16L, h2048, 1 token, KV-cached) | ~5–15 ms | ×1 | fp32; ~halves in bf16 |
| Depth decoder step (4L, tiny) | ~1–3 ms | **×31** | **the inner-loop killer** — 31 serial steps |
| Mimi decode (1 frame) | ~5–15 ms | ×1 (streaming) | conv stack |
| Host↔device sync overhead | ~0.2–1 ms | **×~32** | offload does ~32 mark_steps/frame |

The **31-step depth loop + ~32 host syncs per frame** is almost certainly the
dominant cost, not the backbone matmul. That focuses the optimization.

---

## 2. Is single-core <100 ms TTFT possible? Honest read

- **<500 ms TTFT, single core: yes, very achievable** with streaming + bf16 +
  fixed-shape compiled graphs (no recompiles).
- **<100 ms TTFT, single core: borderline / aggressive.** It hinges entirely on the
  31-step depth loop. If each depth step (compiled, on-device, low-overhead) is ~1–2 ms
  and host sync is hidden, frame 0 ≈ prefill(~15 ms) + backbone(~8 ms) + 31×~1.5 ms
  (~47 ms) + codec(~10 ms) ≈ **~80 ms** — under 100 ms, but with little margin.
- **<100 ms with comfortable margin: use 2–4 cores (TP).** Tensor-parallel splits the
  backbone/depth matmuls and the host loop overhead amortizes → frame 0 well under
  100 ms. This trades the "single core" constraint for headroom.

**Verdict to give the customer:** <500 ms is safe single-core; **<100 ms is reachable
but tight single-core and comfortable at TP=2–4.** The depth decoder's 31 serial steps
are the fundamental floor — most of the plan attacks that.

---

## 3. The optimization ladder (ranked by impact on TTFT)

### Tier 1 — mandatory, biggest wins
1. **Streaming generation (emit frame 0 immediately).** Reimplement the generate loop
   to yield each frame's audio as it's produced (incremental Mimi decode), instead of
   `generate(output_audio=True)` decoding the whole clip at the end. Turns TTFT from
   whole-utterance → one-frame. **The #1 lever.** Mimi supports streaming decode.
2. **Fixed-shape compiled graphs + persistent NEFF cache.** Eliminate the per-step
   recompiles (today's 740 s cold). Bucket the backbone (prefill bucket + decode
   bucket) and compile the depth-decoder step once; reuse the warm NEFFs. Without this
   there is no stable latency at all.
3. **bf16 / mixed precision.** We run fp32 only to dodge the massive-activation norm
   collapse (layers ~24–25 hit ~1e16). Isolate fp32 to just those norms / affected
   layers and run the rest (all the matmuls) in **bf16** → ~2× on backbone + depth.
   Validate parity with the teacher-forced harness.

### Tier 2 — attack the 31-step depth loop (the floor)
4. **Depth decoder on-device, compiled as a tight loop.** Today it's on CPU (the
   on-device attempt hit `NRT_EXEC_OOB` on the codebook-index embedding path — fixable
   by correcting the index/offset handling). A compiled on-device 31-step routine with
   minimal host syncs removes ~31 CPU↔device round-trips per frame.
5. **Cut host↔device syncs.** The offload approach does ~32 `mark_step`s/frame. Fuse
   the per-frame compute into fewer graphs (ideally one backbone-step graph + one
   fused depth graph), or drive the whole frame loop on-device, so host overhead
   stops dominating at batch 1.
6. **Fused kernels for the backbone** (QKV+norm+RoPE, GeGLU/MLP), à la the gemma4
   fused kernels already in the omni beta — fewer ops/dispatches per layer.

### Tier 3 — heavier / model-level
7. **Weight quantization (int8 / fp8) on the backbone.** Single-stream decode is
   weight-bandwidth-bound; int8 weights ≈ halve the DMA → faster steps. Validate audio
   quality.
8. **TP=2–4 (multi-core).** The deterministic way to buy margin under 100 ms: split
   backbone + depth matmuls across cores. Trades the single-core constraint.
9. **Parallel / fewer codebook steps (research).** The 31-step serial depth loop is the
   hard floor. Options: predict multiple codebooks per step (model change), or a
   distilled/shallower depth decoder. High risk, biggest ceiling.
10. **Speculative audio decoding** — draft cheap frames, verify in parallel. Unproven
    for CSM; flag as exploratory.

### Tier 4 — config / serving
11. **Prefill bucketing to the real prompt length** (avoid padding short prompts).
12. **Warm-up at server start** so the first real request isn't paying compile.
13. **vLLM-Omni `CsmPipeline` bucketed serving** (the artifact already registered) —
    gives continuous batching + the `/v1/audio/speech` endpoint; per-request TTFT plus
    aggregate throughput for many concurrent callers.

---

## 4. Staged experiment plan (each gated by a warm measurement)

| Stage | Action | Target signal |
|---|---|---|
| **0** | Instrument a **warm** per-frame + per-component latency breakdown (fixed shapes, cache warm). Confirm the depth loop / host sync is the bottleneck. | a real ms breakdown |
| **1** | **Streaming generate** (emit frame 0). Measure TTFA. | TTFA ≪ full-clip time |
| **2** | **bf16 + isolate fp32 norms**; re-validate parity (teacher-forced cos). | ~2× backbone/depth, cos≈1.0 |
| **3** | **Fixed-shape compile + warm NEFF cache**; kill recompiles. | stable warm per-frame |
| **4** | **Depth decoder on-device + compiled** (fix the OOB index path); cut syncs. | depth step ms ↓, frame ms ↓ |
| **5** | If still >100 ms single-core: **TP=2 then TP=4**. | TTFA <100 ms |
| **6** | **int8 weights** (optional squeeze) + warm-up + bucket. | margin |
| **7** | Wire into the **`CsmPipeline`** for streamed serving + concurrency. | productionized |

**Stop conditions:** report the warm TTFA after each stage. Lock <500 ms early
(Stages 1–3), then push toward <100 ms (Stages 4–6).

---

## 5. Realistic latency budget (single core, warm, post-Stages 1–4)

```
T_prefill (backbone, ~15 tok, bf16)        ~10–20 ms
T_frame0 backbone step (bf16, KV-cached)    ~4–8 ms
T_frame0 depth (31 steps, compiled device)  ~30–60 ms   <- the swing factor
T_codec (1 frame, streaming)                ~5–12 ms
-------------------------------------------------------
TTFA (first audio)                          ~50–100 ms  (single core, optimistic)
```
- Hits **<100 ms** if the depth loop lands at the fast end; **safely <100 ms at TP=2–4**.
- **<500 ms** is comfortable even before the depth-loop work.
- **Real-time factor:** each frame = 80 ms of audio; sustaining <80 ms/frame warm =
  faster-than-real-time streaming (no audio underrun). That's the streaming-stability
  bar in addition to TTFA.

---

## 6. Open questions for the customer
1. **Streaming or full-clip?** TTFA only matters for streaming. If they need the whole
   clip returned at once, "TTFT" = full-utterance time and the bar is different.
2. **Concurrency?** Single low-latency stream vs. many concurrent callers changes the
   single-core-vs-TP and the `CsmPipeline`-serving decision.
3. **Quality bar for bf16/int8?** Confirms how aggressive we can be on precision.
4. **Is single-core a hard constraint, or a cost target?** If cost, TP=2 on a 3xl-class
   slice is the clean way to <100 ms.

---

## 7. First action
Stage 0: build a warm per-frame latency harness (`src/bench_ttft.py`) — fixed shapes,
cache warmed — that reports `T_prefill`, `T_backbone_step`, `T_depth_step`,
`T_depth_total`, `T_codec`, and end-to-end TTFA. Everything else is gated on that
breakdown. (Highest-information, lowest-effort next step.)
