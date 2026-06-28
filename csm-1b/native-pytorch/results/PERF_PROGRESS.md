# CSM-1B Performance Progress (2026-06-28)

Measured optimization progress toward <500ms / <100ms TTFA. See
`../KERNEL_AND_PERF_PLAN.md` for the full roadmap.

## bf16 — validated + applied
- **Backbone correctness:** bf16 (CsmRMSNorm self-upcasts variance to fp32) vs fp32:
  **cosine 0.999968–0.999990, argmax 100%.** No collapse (that was a different model).
- **Backbone direct forward:** 36.4 ms → 25.9 ms (**1.40×**).
- **In-loop per-frame (warm):**
  - backbone step: **48 ms → 35 ms** (1.4×)
  - depth total (31 steps): **291 ms → 156 ms** (1.9×)
  - per-frame compute: **~339 ms → ~191 ms** (~1.8×)
- **Config that works:** whole model bf16 EXCEPT `codec_model` fp32 (bf16 breaks the
  Mimi convs; codec is fed int codes so the boundary is dtype-agnostic). Applied to
  `src/generate_speech.py` and `src/bench_ttft.py`.

## Still to do (ranked, from KERNEL_AND_PERF_PLAN.md)
1. **Streaming** (emit frame 0) — turns TTFA into one-frame time. Biggest lever.
2. **Warm NEFF cache / fixed-shape compile** — the prefill (546ms) and codec (273ms)
   are one-shot compile/dispatch overhead, not compute; a persistent warm cache + the
   compiled-graph path removes them. (This run showed 45s/20s prefill/codec = cold
   compile of new bf16 shapes — they amortize once cached.)
3. **NKI TKG megakernel** (`experimental/transformer/transformer_tkg.py`) on the
   backbone decode step + depth steps — collapses per-op dispatch into one kernel; the
   structural fix for the overhead and the depth-loop floor.
4. **Depth decoder on Neuron** (fix the NRT_EXEC_OOB index path) — crush the 156ms.
5. **Multi-core TP=2–4** — split backbone/depth matmuls; the margin for a hard <100ms.
6. **MXFP8** squeeze on the matmuls.

## Read
- bf16 gives a clean ~1.8× on per-frame compute — real and applied.
- The dominant remaining costs are (a) one-shot compile/dispatch overhead (fix: warm
  cache + TKG megakernel) and (b) the 156ms depth loop (fix: on-device + fused).
- <500ms is in reach with streaming + warm cache + bf16; <100ms needs the TKG
  megakernels + on-device depth, with TP=2–4 as the safety margin.
