# Qwen3.5-4B Production Serving on Trainium — Plan

**Goal:** Production-ready vLLM-Neuron serving of `Qwen/Qwen3.5-4B`
(hybrid GatedDeltaNet + GQA) for Scaledown. Batched HTTP, paged KV,
OpenAI-compatible API. 20K-in / short-out workload.

**Why vLLM-Neuron (not native PyTorch eager):** production needs
continuous batching, paged attention, and an HTTP server. Native
PyTorch eager (`torch_neuronx`) is a research/validation harness, not a
serving stack. So Path 1 (vLLM-Neuron) is mandatory for this goal.

---

## What we KNOW works (established 2026-06-09)

1. ✅ Model + weights + tokenizer correct (HF CPU ref → "Paris")
2. ✅ gdn_neuron reference math correct on Neuron container (CPU → "Paris")
3. ✅ Path B custom adapter **compiles and serves end-to-end** via vLLM
   (HTTP up, /v1/completions responds)
4. ✅ `vllm_neuron.nki.nki_hop.wrap_nki` importable (NKI bridge live)
5. ✅ Root cause of garbage output found + fixed: DeltaNet layers
   didn't handle sequence-parallel input. Fix = all_gather/slice in
   `forward()` outside the traced graph.

## What's BLOCKING us

**The compile, not the serving.** Two coupled problems:

- **Compiler host-RAM OOM.** `neuronx-cc` (walrus_driver) needs
  25-30 GB RAM per bucket-rank. The DeltaNet recurrence unrolls 4096
  sequential state updates into one huge graph. On trn2.3xl (124 GB
  host) with TP=2-4 × parallel buckets, this OOMs.
- **tp_barrier timeout.** When one rank's compile is slow (or got
  OOM-killed), the fast rank's 30-min barrier expires → engine init
  dies.

Both are HOST-side compile problems. The model RUNS fine on-device
(8.8 GB weights, fits 96 GB HBM easily). We just can't COMPILE the
NEFFs on this box.

---

## THE PLAN — 3 phases

### Phase 1: Get a correct, compiled vLLM serve (unblock the OOM)

**Approach A — compile on big RAM, serve on the 3xl (RECOMMENDED)**

NEFFs are portable. Compile-time RAM is the only constraint; runtime
is cheap.

1. Provision a **trn2.48xl** (2 TB host RAM) OR reuse the Ohio 48xl.
2. Deploy the SP-fixed Path B code there, compile all buckets at TP=4.
   2 TB host RAM has no trouble with the DeltaNet graph.
3. **rsync the compile_cache** (`/root/.cache/vllm/neuron/compile_cache`)
   from the 48xl to the 3xl.
4. Start the serve on the 3xl — it cache-hits every NEFF, no compile,
   instant startup. Runtime memory is fine on a 3xl.
5. Validate "Paris" output, then run the bench.

Cost: ~1 hr of 48xl ($21.50/hr) for the compile, then serve on the
cheap 3xl ($2.23/hr). One-time compile cost amortizes across all
future restarts (cache persists if we mount it on a host volume).

**Approach B — make the graph smaller so it compiles on the 3xl**

Add a graph break at the DeltaNet 128-token chunk boundary so each NEFF
is a single chunk's recurrence, not the whole 4096-step unroll. The NKI
kernel ALREADY chunks internally — we need vllm-neuron's graph extractor
to not inline the whole Python chunk-loop into one graph.

More elegant (compiles anywhere, smaller NEFFs, faster cold start) but
requires deeper surgery on `_forward_prefill` + understanding the
graph-capture boundary. Higher risk, ~half-day.

**Approach C — serial compile + raised barrier timeout on the 3xl**

`VLLM_NEURON_PARALLEL_COMPILE_WORKERS=1` (already tried, helps RAM) +
patch/raise the tp_barrier timeout so the slow rank isn't killed.
We got 7/~10 NEFFs this way with cache-resume. Might limp to completion
across a few restart cycles. Fragile, not production-clean.

**Decision: Approach A.** Cleanest, lowest-risk, gives a real
production artifact (a portable NEFF cache) we can redeploy.

### Phase 2: Validate correctness + lock the config

1. "Paris" smoke on the compiled serve (greedy, single prompt)
2. Logit parity vs HF CPU reference (cosine ≥ 0.999, top-1 match)
3. Confirm batched serving works (max_num_seqs > 1) — Path B's o_proj
   layout fix + the SP fix should make batch>1 correct now
4. Lock the production serve config (TP, bucket list, max_model_len,
   KV dtype)

### Phase 3: Production hardening + benchmark

1. **Persistent NEFF cache** — mount compile_cache on a host volume so
   container restarts don't recompile (the lesson from tonight)
2. **Bucket set for the workload** — customer is 20K-in. Use chunked
   prefill: `num_batched_tokens_buckets=[512,1024,2048,4096]`,
   `max_model_len=24576` (or higher). 20K inputs stream through the 4K
   kernel via chunked-prefill scheduler.
3. **Benchmark** with Jim's neuronx-benchmark-tool (per benchmarking
   steering rule) — TTFT + throughput sweep, multiple batch sizes
4. **$/M-token** at the customer's shape, vs the p4d A100 baseline
5. **Reproducible deploy** — Dockerfile/compose + serve.sh + the cached
   NEFFs, so Scaledown can stand it up themselves

---

## The SP fix (already applied, needs validation on 48xl)

In `qwen3_5/model_bf16.py`, `Qwen3_5DeltaNetAttention.forward()`:
- all_gather scattered tokens → full sequence BEFORE the traced
  `_forward_prefill`/`_forward_decode` (Python-level, outside graph)
- slice replicated output → this rank's shard AFTER (Python-level)
- Collectives are OUTSIDE the NEFF graph (critical — putting them
  inside exploded the graph and made compile even heavier)

This mirrors exactly how the GQA layer already handles SP.

---

## Immediate next action

Provision/recover a trn2.48xl, deploy SP-fixed Path B, compile at TP=4,
rsync cache to the 3xl, serve + validate "Paris". That's the unblock.

## Open questions for the customer / Armin

- Target TTFT and throughput SLA? (decides bucket config + TP)
- Max context they actually send? (20K confirmed, but ceiling?)
- Acceptable to ship as a private vllm-neuron branch, or do they need
  to wait for upstream #2087?
