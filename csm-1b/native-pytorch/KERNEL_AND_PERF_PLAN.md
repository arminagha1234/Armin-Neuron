# CSM-1B — Kernel & Performance Roadmap to <500ms / <100ms TTFA

Synthesis of measured findings + the concrete path, including NKI kernels and
multi-core. Companion to `TTFT_OPTIMIZATION_PLAN.md`. **Created 2026-06-28.**

## Measured ground truth (single NeuronCore, this session)
- **TTFA today ≈ 1158 ms** (fp32, offload). Breakdown (warm):
  prefill **546 ms**, backbone step **48 ms/frame**, **depth 291 ms/frame**
  (31 serial steps), codec **273 ms**.
- **Backbone real compute is ~36 ms** (warm, direct forward) — so the 546 ms
  "prefill" and 273 ms "codec" are **dispatch/sync/recompile overhead, not math.**
- **bf16 (norms kept fp32) = 1.40× on the backbone, cosine 0.999990, argmax 100%**
  — validated; the earlier collapse was bf16 *norms*, fixed by fp32 norms.

## The two elephants (attack these, not the 36 ms backbone)
1. **Depth decoder: 291 ms/frame** — 31 sequential steps on CPU (~9.4 ms each). This
   is the single biggest per-frame cost and the hard AR floor.
2. **Per-call overhead** — generate machinery + ~32 host↔device syncs/frame + per-shape
   recompiles inflate prefill (546 ms) and codec (273 ms) far above compute.

## The roadmap (ranked by measured impact)

### Lever 1 — Streaming (TTFA = one frame, not the whole clip) [biggest, mandatory]
Reimplement the loop to emit frame 0's audio as soon as it's ready (incremental Mimi
decode). Turns the 3.5 s full-clip wall into ~one-frame TTFA. Nothing reaches sub-second
without this.

### Lever 2 — Kill the dispatch/sync overhead via NKI TKG megakernels [the <100ms key]
The overhead is the dominant cost, and the NKI library has the exact tool:
- **`experimental/transformer/transformer_tkg.py` — Transformer TKG megakernel**: a full
  transformer forward-pass *megakernel* for token generation. Fuses RMSNorm+QKV+RoPE+
  attention+MLP+oproj into **one kernel** → collapses ~per-layer dispatches and host
  syncs into a single device call. This is the structural fix for the 546 ms-type
  overhead on the **backbone decode step**.
- **`experimental/transformer/attention_block_tkg.py` — Attention Block TKG**: fused
  RMSNorm+QKV+RoPE+attention+oproj — same idea, attention block granularity.
- **`core/attention/attention_tkg.py`**, **`core/qkv/qkv.py`**, **`core/mlp/mlp.py`**,
  **`core/embeddings/rope.py`**, **`core/rmsnorm/rmsnorm_quant.py`** — building blocks if
  the megakernel needs adaptation to CSM's dims (h=2048, 16 layers).
- **Apply the same TKG megakernel to the depth decoder** (4 layers) so each of the 31
  steps is one fused kernel instead of a Python+dispatch round-trip — directly attacks
  the 291 ms elephant.
- Install path: `pip install nki-library` (matches our neuronx-cc branch) replaces the
  bundled nkilib; reachable via `NF.*` like the gemma4 flash kernels.

### Lever 3 — Depth decoder on Neuron, compiled [attacks the 291 ms elephant]
Today depth runs on CPU (on-device attempt hit `NRT_EXEC_OOB` on the codebook-index
embedding path — fixable by correcting the index/offset gather). On-device + fused
(Lever 2) turns 31×9.4 ms CPU steps into a tight compiled device loop.

### Lever 4 — bf16 (norms fp32) [validated 1.4×]
Apply the validated bf16+fp32-norm scheme to backbone (and codec). Integration note:
keep a clean dtype boundary — bf16 backbone internals, fp32 norms, cast the backbone
output to match the lm_head/depth dtype (the offloaded path needs the boundary cast;
`bench_bf16.py` proves the numerics).

### Lever 5 — Fixed-shape compile + warm NEFF cache
Eliminate per-step recompiles (today's 740 s cold). Bucket the backbone (prefill +
decode buckets) and compile the depth step once; ship a warm cache. Removes the
recompile component of prefill/codec overhead.

### Lever 6 — Multi-core (TP=2–4) [the deterministic margin for <100ms]
Tensor-parallel the backbone (and depth) matmuls across 2–4 NeuronCores. CSM's HF
backbone isn't TP-sharded, so this needs column/row-parallel linears (NxD-style) or the
vLLM-Omni engine's TP. Splits the per-step compute and amortizes host overhead →
comfortably <100 ms when stacked on Levers 1–5. (The trn2.48xl has 16 cores; the omni
plugin already runs Wan at TP=8.)

### Lever 7 — Codec: depthwise-conv NKI + streaming decode
Mimi decode (273 ms measured, mostly overhead) → use the NKI
**`experimental/conv/depthwise_conv1d.py`** kernel for the SEANet convs and decode
**one frame at a time** (streaming) instead of the whole clip. The codec already
compiles on Neuron at cosine 1.0; this cuts its per-call overhead + enables Lever 1.

### Lever 8 — FP8 / MXFP8 (squeeze)
`experimental/matmul_mxfp8/` + `core/rmsnorm/rmsnorm_quant.py` (RMSNorm→fp8) for the
backbone matmuls once bf16 is in — another ~2× on the dense matmuls, validate audio.

## Projected budget after the ladder (single core, warm)
```
T_prefill   (TKG megakernel, bf16, bucketed)   ~10–25 ms   (from 546 ms)
T_backbone0 (TKG megakernel step, bf16)          ~5–10 ms
T_depth0    (31 fused on-device steps)          ~20–50 ms   (from 291 ms)  <- swing
T_codec0    (streaming depthwise-conv, 1 frame)  ~5–15 ms   (from 273 ms)
-----------------------------------------------------------------------
TTFA (streaming)                                ~40–100 ms  single core (optimistic)
```
- **<500 ms: reached early** (Levers 1+4+5 alone — streaming + bf16 + no recompiles).
- **<100 ms: needs Levers 2+3 (TKG megakernels + on-device depth)**, with **TP=2–4
  (Lever 6) the safety margin.** The depth loop is the swing factor.

## Staged execution (each gated by `bench_ttft.py`)
1. **Streaming** generate (emit frame 0) → measure TTFA. [Lever 1]
2. **bf16+fp32-norm** clean integration (boundary cast) → re-measure. [Lever 4]
3. **Fixed-shape compile + warm cache** → kill recompiles. [Lever 5]
4. **NKI TKG megakernel** on the backbone decode step (`pip install nki-library`,
   wire via NF) → measure the overhead drop. [Lever 2]
5. **Depth decoder on Neuron + TKG-fused** (fix the OOB) → crush the 291 ms. [Lever 3]
6. **TP=2 → TP=4** if still >100 ms single-core. [Lever 6]
7. **Streaming depthwise-conv codec** + **MXFP8** squeeze. [Levers 7, 8]

## Honest verdict
- **<500 ms single-core: high-confidence** (streaming + bf16 + bucketed compile).
- **<100 ms: reachable** with the NKI TKG megakernels collapsing dispatch overhead +
  on-device fused depth loop; **TP=2–4 makes it comfortable.** The 31-step depth decoder
  is the fundamental floor and is the main reason multi-core is likely needed for a hard
  <100 ms guarantee.

## Tools in this folder
- `src/bench_ttft.py` — warm per-component latency harness (TTFA breakdown).
- `src/bench_bf16.py` — bf16-vs-fp32 backbone correctness + speed (the 1.40× / cos
  0.99999 result).
- `src/generate_speech.py` — working offload generate (the integration target).
