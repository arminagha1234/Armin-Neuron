# CSM-1B TTFT Exploration (vllm_explore) — learning from gemma4 vllm_32k_explorev2

Active TTFT optimization for CSM-1B. Baseline = `../vllm_v1` (streaming warm TTFA
241ms; steady ~295ms/frame; depth decoder 156ms = bottleneck). This folder applies the
TTFT levers proven on gemma4 (`gemma4-31b/vllm_32k_explorev2`) to CSM.

**Owner:** Armin · **Status:** EXPLORE · **Created:** 2026-06-28

## What we learned from gemma4 vllm_32k_explorev2 (and how it maps to CSM)

| gemma4 lever | gemma4 result | CSM mapping |
|---|---|---|
| **Prefix caching (APC)** | TTFT 0.83→0.42s on repeated context (3.2× @90% hit) | **CSM is conversational** — multi-turn dialogue reuses prior turns + speaker audio context. Cache the backbone prefill of the conversation history → per-turn TTFA collapses to ~one-new-turn prefill. **Likely the #1 CSM win.** |
| **Small prefill seg/buckets** | seg=512 → lowest fixed per-chunk floor | CSM prefill is a short text prompt (+ optional audio context). Use the smallest prefill bucket; multi-bucket for varied prompt lengths. |
| **EAGLE3 / spec decode** | amortizes fixed ~356ms/step decode | CSM has TWO AR loops: backbone (per frame) + depth (31 steps/frame). Spec/parallel decoding of codebooks or frames amortizes the per-step floor — maps to the 156ms depth loop + backbone step. |
| **TP=32 best latency** (prefill compute-bound, scales 1/TP) | TP=16 1.45s vs TP=32 0.93s | CSM backbone matmul is compute; TP=2–4 cuts the 38ms backbone step + helps prefill. |
| **FP8 on trn2** (plain e4m3, not MX) | verified working | FP8 the backbone matmuls (2× via double-FP8 on safe layers). |
| **async scheduling / gather cap** | NO win (floor is collective/dispatch) | Confirms: don't chase host-overlap; reduce per-step work (fusion/TKG) + collectives (TP). |

## Key insight CSM adds beyond gemma4
gemma4's bottleneck was the backbone decode step (collective/dispatch floor). **CSM's
bottleneck is the 31-step DEPTH decoder (156ms/frame)** — gemma4's levers don't touch it.
So CSM needs BOTH the gemma4 transferable levers (prefix cache, buckets, TP, fp8) AND a
CSM-specific depth-loop fix (on-device fused / TKG megakernel / parallel codebooks).

## Experiment ladder (this folder)
### Tier A — transferable from gemma4 (cheap, high ROI)
- **A1. Prefix caching for multi-turn conversation.** Measure per-turn TTFA when the
  conversation context is reused (the CSM-native pattern). Build a multi-turn harness:
  turn 1 (cold) vs turn 2+ (cached prefix). Expected: turn-2+ TTFA ≪ turn-1.
- **A2. Small/multi prefill buckets** for the backbone prefill — cut padding for short
  text prompts.
- **A3. TP=2–4** for the backbone step + prefill (compute-bound, scales with cores).

### Tier B — CSM-specific depth-loop attack (the 156ms elephant)
- **B1. Depth decoder on Neuron + StaticCache** (fix the NRT_EXEC_OOB codebook-index path).
  - **DONE (diagnosis, 2026-06-28, `results/B1_DEPTH_ON_DEVICE.md`):** the OOB index-range
    hypothesis is FALSE (all 31 gather indices in-bounds on CPU). Per-step forward-wrapper
    offload is the wrong shape — 31 host↔device round-trips/frame would cost more sync than
    the 156ms compute saved. **Reframe: fuse the whole 31-step loop into ONE device graph.**
- **B1' (revised, NEXT). Trace the whole 31-step depth loop as a single device callable**
  (fixed shapes, on-device cache, one round-trip/frame). Needs the static-index rewrite of
  `CsmCodebooksHead` (python-int slice instead of `weight[cache_position-1]`).
  - **DONE (2026-06-28, `results/B1P_FUSED_DEPTH.md`):** built + validated (32/32 EXACT in
    fp32). OOB root cause confirmed = dynamic gather with a runtime *device-tensor* offset;
    fixed with static python-int offsets + `inputs_embeds`. **Speed is a wash in bf16:
    177.9→166.8ms = 1.07×** (fp32 1.68×). The loop is latency-bound on 31 serial tiny
    steps, so fusing host round-trips doesn't help in bf16.
- **B2. NKI TKG megakernel** (`attention_block_tkg`) on backbone + the fused depth step —
  **DOWNGRADED:** collapses per-op dispatch *within* a step, but the bottleneck is the
  31-deep serial chain, not dispatch. Expected marginal. Park unless B3 needs it.
- **B3. Parallel/speculative codebook decoding — PRIMARY depth lever (PROMOTED).** Breaking
  the serial dependency is the only thing that moves the 156–178ms floor. Directions:
  (a) multi-codebook-per-step parallel head + cheap verify/correct; (b) codebook-group
  factorization; (c) distilled single-shot RVQ predictor. Research risk, highest payoff.
- A3 (TP=2–4 backbone) remains a secondary lever (shaves the 38ms backbone step).

### Tier C — squeeze
- **C1. FP8 backbone matmuls** (e4m3, trn2-safe).
- **C2. Warm NEFF cache + AOT** so cold compiles never hit a request.

## Method
Each experiment gated by the `vllm_v1` harnesses (`bench_ttft.py`, `stream_speech.py`).
Report warm TTFA + steady per-frame. Don't touch `../vllm_v1` (frozen reference).

## First action
A1 — multi-turn prefix-caching harness: does CSM's conversational context reuse collapse
per-turn TTFA? This is the highest-ROI transferable lever and matches CSM's actual use.

## A1 RESULT (2026-06-28): prefix caching does NOT help CSM — pivot to Tier B
Ran `src/multiturn_ttft.py` (see `results/A1_PREFIX_CACHING.md`). First-frame latency is
**flat at ~250ms** as context grows 45→125 tokens (≈3×); the only bump is a one-time
prefill-bucket step at 25→45 tokens. CSM's TTFA floor is **per-frame compute (backbone
step + 31-step depth loop), not prefill** — structurally unlike gemma4. So:
- **Tier A dropped** (prefix caching + prefill buckets don't move TTFA; prefill is
  already negligible — one small static bucket suffices).
- **Active focus is now Tier B — the 156ms depth decoder.** Order: B1 (depth on Neuron +
  StaticCache, fix `NRT_EXEC_OOB`) → B3 (parallel codebook decode) → B2 (NKI TKG).
- A3 (TP=2–4) kept as a secondary lever (shaves the 38ms backbone step).
