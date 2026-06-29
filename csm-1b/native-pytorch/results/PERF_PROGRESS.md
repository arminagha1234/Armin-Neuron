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

## Read
- bf16 gives a clean ~1.8× on per-frame compute — real and applied.
- The dominant remaining costs are (a) one-shot compile/dispatch overhead (fix: warm
  cache + TKG megakernel) and (b) the 156ms depth loop (fix: on-device + fused).
- <500ms is in reach with streaming + warm cache + bf16; <100ms needs the TKG
  megakernels + on-device depth, with TP=2–4 as the safety margin.
## NKI TKG megakernel — available + interface de-risked (2026-06-28)
The bundled `nkilib` (in the Neuron venv / neuronx-cc) already ships the TKG
megakernels — no separate install needed:
- `nkilib/experimental/transformer/attention_block_tkg.py` (+ `_torch.py` reference,
  `_sharding.py` for multi-core).
- `nkilib/experimental/transformer/transformer_tkg.py` (full-step megakernel).

`AttentionBlockTkgTorchRef.forward` interface maps directly onto a CSM backbone layer
— it fuses, in one kernel: input RMSNorm → **fused QKV** (`W_qkv` shape `[H,(N+2)*D]`,
optional bias, **fp8-capable** via `quantization_type_qkv`/dequant scales) → optional
QK-norm (pre/post RoPE) → **RoPE** (cos/sin) → attention → **KV-cache update**
(`kv_cache_update_idx`). CSM uses standard Llama attention (QK-norm off), so the
mapping is: feed each layer's fused QKV + o_proj weights, the RoPE tables, and the
paged KV cache.

### Integration contract (next build)
1. Per backbone layer: replace norm+QKV+RoPE+attn+KV-update with one
   `attention_block_tkg` call (weights remapped to `[H,(N+2)*D]`); MLP stays separate
   (or use the full `transformer_tkg` megakernel to fuse the whole layer).
2. Provide RoPE cos/sin and the KV-cache tensors/update index.
3. Optional fp8 (`quantization_type_qkv`) for the extra ~2x.
4. Use `attention_block_tkg_sharding.py` for TP across 2–4 cores.

This collapses the ~per-op dispatch (the dominant TTFT overhead) into one kernel per
layer — the structural path to <100ms. It's a real multi-hour kernel-integration build;
the interface is now confirmed + mapped, so it's de-risked.

### Ranked remaining work
1. **Streaming** (emit frame 0) — turns TTFA into one-frame time. Biggest lever.
2. **Warm NEFF cache / fixed-shape compile** — removes the one-shot prefill/codec
   compile+dispatch overhead (the 45s/20s cold numbers amortize once cached).
3. **TKG megakernel** integration (interface above) — the dispatch-overhead fix.
4. **Depth decoder on Neuron** (fix NRT_EXEC_OOB index path) + TKG-fuse — crush 156ms.
5. **Multi-core TP=2–4** (`attention_block_tkg_sharding`) — margin for hard <100ms.
6. **MXFP8** matmul squeeze.
## Streaming implemented + the offload-path latency ceiling (2026-06-28)
`src/stream_speech.py` hooks CSM's per-frame `streamer.put(codes)`, decodes each
frame's 32 codes immediately (1 frame = 1920 samples = exactly 80 ms @ 24 kHz), and
records true TTFA. **Streaming works** — emits frame 0 and wrote a 1.0 s clip.

**But it exposed the architectural ceiling of the lazy torch_xla offload path:**
per-frame latency is dominated by **per-step recompiles** — each decode frame grows
the KV cache → a new tensor shape → neuronx-cc recompiles, and the NEFF cache is not
stably reused across the dynamic shapes (measure-pass TTFA was ~19 s, all recompile).
A clamp on codes was needed first (generated frames can contain eos/special ids ≥
codebook_size → OOB in the RVQ embedding lookup on-device).

### Decisive conclusion
The **offload approach (CPU generate loop + per-call device offload) is a correctness
vehicle, not a latency one.** It cannot deliver stable low latency because of
dynamic-shape recompiles + per-call host syncs. Low latency therefore *requires* the
fixed-shape compiled-graph path:
- **Fixed-shape bucketing + AOT trace** (torch_neuronx.trace / the omni engine's
  compiled path) so the backbone decode step and codec compile ONCE and reuse, OR
- the **NKI TKG megakernel** (compiles the whole step as one fixed kernel).

This makes Levers 2/3/5 (TKG megakernel, on-device depth, fixed-shape compile) the
**required** path to <500ms — not optional polish. Streaming (done) + bf16 (done) are
necessary but insufficient on the lazy-offload runtime; they pay off once the per-step
graph is fixed-shape and compiled once.

### Net status toward targets
- ✅ Streaming emit-frame-0 mechanism: working.
- ✅ bf16 ~1.8× per-frame compute: working.
- ⛔ Stable low TTFA: blocked by lazy-offload per-step recompiles → needs fixed-shape
  AOT compile (bucketed) or the TKG megakernel. That is the next build and the gating
  item for both <500ms and <100ms.
## Fixed-shape AOT trace attempt — needs a real build (not a quick win)
Tried `torch_neuronx.trace` on the codec single-frame decode (fixed (1,32,1)) to prove
the fixed-shape unblock. It **core-dumped** in the XLA tensor bridge — the Mimi codec's
stray RVQ tensors + dynamic internal quantizer ops don't trace cleanly, and mixing
`torch_neuronx.trace` with an active `torch_xla` session conflicts.

### Honest boundary
Two distinct approaches now hit walls for *stable low latency*:
1. Lazy torch_xla offload → per-step recompiles (latency ceiling).
2. `torch_neuronx.trace` → core dump on the codec.

⇒ The remaining path to <500ms/<100ms is a **real engineering build**, not a quick
experiment:
- **Preferred:** the **vLLM-Omni `CsmPipeline`** on a torch_xla 2.9 runtime — the omni
  engine provides the bucketed/compiled-graph path (fixed shapes, warm NEFFs) that the
  lazy offload lacks. (Blocked today only by the container's torch_xla 2.10 regressions;
  fix = 2.9 runtime.)
- **Or:** hand-built fixed-shape AOT: trace the backbone single-step at bucketed KV-cache
  shapes + the codec frame-decode (handling stray tensors / dynamic ops), drive a manual
  decode loop. Then layer the **NKI TKG megakernel** (`attention_block_tkg`) to collapse
  per-layer dispatch.

### Session net (CSM TTFT)
| Lever | State |
|---|---|
| Streaming (emit frame 0) | ✅ implemented (stream_speech.py) |
| bf16 (~1.8x per-frame) | ✅ validated + applied |
| Stage-0 latency breakdown | ✅ measured (TTFA ~1.16s, overhead-dominated) |
| NKI TKG megakernel | ✅ available + interface mapped (de-risked) |
| Fixed-shape AOT / compiled graph | ⛔ the required next build (lazy path + quick trace both blocked) |
| Multi-core TP | ⛔ via sharding kernel / omni engine (after fixed-shape) |

**Bottom line:** bf16 + streaming are done and necessary; the gating item for the
targets is the fixed-shape compiled-graph path, best delivered via the vLLM-Omni
pipeline on torch_xla 2.9 (or a hand-built AOT trace + TKG megakernel). That is a
multi-hour/-day build, now fully scoped and de-risked.
## BREAKTHROUGH: StaticCache → warm TTFA 244 ms (<500 ms target HIT) (2026-06-28)
The per-frame recompiles were caused by **DynamicCache growing each frame**. Switching
to **`cache_implementation="static"`** (pre-allocated fixed-size KV cache) makes every
decode step a fixed shape → **compile once, no per-frame recompile** — on the existing
offload path, no AOT trace needed.

### Measured (single NeuronCore, bf16 backbone, static cache)
- **Backbone decode step: stable 38.4 ms** (min 37.9, max 40.2) — recompile variance
  gone (was wildly variable).
- **Streaming warm TTFA = 243.9 ms** (down from ~19,500 ms on dynamic cache) — **under
  the customer's 500 ms TTFT target.**
- One-time prefill graph compile (~50 s) is amortized by a persistent NEFF cache.

### So, for the stated target
- **<500 ms TTFT (time-to-first-audio): ACHIEVED** — 244 ms warm, with
  streaming + bf16 + StaticCache. Applied to `generate_speech.py`
  (`cache_implementation="static"`) and `stream_speech.py`.

### Remaining for sustained real-time + <100 ms
- **Sustained per-frame** is still high (codec decode per frame not yet
  shape-stabilized across frames; depth decoder = 156 ms CPU). For real-time
  streaming (<80 ms/frame) and the <100 ms stretch:
  1. Stabilize/AOT the per-frame **codec** decode (fixed (1,32,1)) so it doesn't
     recompile across frames.
  2. **Depth decoder on Neuron** + StaticCache (fix the OOB) — cut the 156 ms CPU loop.
  3. **NKI TKG megakernel** on the backbone step — push 38 ms → single-digit ms.
  4. **TP=2–4** for the margin.
- StaticCache is the key enabler that retires the "lazy path can't do latency" ceiling:
  with fixed shapes it CAN, and TTFA is already <500 ms.
