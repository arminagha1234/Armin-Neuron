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
