# CSM-1B Tensor Parallelism — analysis, honest scope, and build plan

**Context:** the customer asked to "increase TP and sweep." Important honest fact up front:
**CSM's backbone (HF `CsmForConditionalGeneration`) has NO tensor-parallel plumbing** — no
`parallelize_module`, no `torchrun`/`init_process_group`, no sharded weights (verified: grep
of all CSM scripts finds TP machinery only in unrelated NKI-kernel/probe files). So TP is not
a runtime knob; it requires a *port*. This doc = (c) scoping + (a) design/plan. The measured
multi-core-worker sweep is (b), in `TP_MULTICORE_SWEEP.md`.

## Where TP would even help (regime analysis)

CSM has two very different compute phases:

| Phase | Shape | Bound | Does TP help? |
|---|---|---|---|
| Backbone **decode step** (per frame) | batch=1, seq=1, 16 layers | dispatch/latency-bound (compiled = 10.6 ms) | **NO** — nothing to shard at seq=1; per-layer all-reduce comms would dominate the tiny per-core compute. Matches the general "TP loses at seq-len-1" result. |
| Backbone **prefill** (TTFT) | batch=1, seq=N | compute-bound for N≳1k (O(N)·MLP + O(N²)·attn) | **YES** — real matmul work to split. This is the ONLY regime TP helps, and it's exactly where `torch.compile` LOSES (>3k, see VERIFIED_PREFILL_TTFT.md). |
| Depth decoder | 31 serial tiny steps | latency-bound (compiled 17.9 ms) | **NO** — serial + tiny, comms-bound. |
| Mimi codec | conv stack, CPU | — | N/A (won't compile on Neuron). |

**Conclusion:** TP is worth building ONLY for long-context prefill (N ≳ 3k), and only if the
customer needs contexts past CSM's 2048 trained window. Below 2k, `torch.compile` already
gives up to 4.25× and TP-comms would erase the small per-core savings. **For the in-spec
(≤2048) TTS use case, TP is not the right lever — compile is.**

## Expected TP prefill gain (estimate, to be measured if built)

Prefill at N is ~compute-bound; sharding the 16-layer backbone across TP ranks splits the
per-layer matmuls ~linearly, minus per-layer all-reduce comms (2 collectives/layer × 16).
At N=4096 (eager 261 ms, compiled 302 ms):
- **TP=2:** ideal 2× → ~130 ms; realistic ~1.6–1.7× after comms → **~155–165 ms**.
- **TP=4:** ideal 4× → ~65 ms; realistic ~2.5–3× after comms → **~90–105 ms**.
- TP=8: comms overhead likely dominates for a 1B model at these N; diminishing.
These are estimates; a built sweep would confirm. The all-reduce cost is the swing factor
(on-chip collectives via `TORCH_NEURONX_ENABLE_HOST_CC` — the same path Mochi used).

## Build plan for real TP (a) — Mochi-style TP port of the backbone

The Mochi port (`neuron/examples/Mochi`) is the exact template — it did TP=2/4/8 on a
DiT via `torch.distributed.tensor.parallel`. Reuse that recipe:

1. **TP plan** (`ColwiseParallel`/`RowwiseParallel` per layer):
   - Llama attention: `q/k/v_proj` → Colwise (split heads across ranks), `o_proj` → Rowwise
     (+ all-reduce). CSM backbone: 32 q-heads / 8 kv-heads (GQA), head_dim 64 → valid TP ∈
     {2,4,8} (8 kv-heads divide by 2/4/8; 32 q-heads too).
   - Llama MLP (SwiGLU): `gate_proj`/`up_proj` → Colwise, `down_proj` → Rowwise (+ all-reduce).
   - RMSNorms + embeddings: replicate.
2. **Meta-init + sharded weight loader** (stream shards per rank), like Mochi's
   `load_weights_sharded`.
3. **Patch head count** per rank (`apply_tp_fixes` analog) so attention reshapes use local heads.
4. **Launch** `torchrun --nproc_per_node={2,4,8}` + `dist.init_process_group(backend="neuron")`
   + `TORCH_NEURONX_ENABLE_HOST_CC=1` (host collectives — mandatory, else EFA path hangs, per
   the Mochi runbook).
5. **Compile the sharded backbone** per-layer (dodge the 5M-instr NEFF ceiling) — but note
   the prefill-crossover finding: at the N where TP helps, compile may hurt, so measure
   TP×{eager,compiled} both.
6. **Validate** per-rank output vs single-core (cosine ≥0.999) before timing.

**Effort:** multi-day-to-weeks (the depth decoder + codec + generate loop must also cope with
a sharded backbone; Mochi shows it's tractable). **ROI gate:** only if the customer needs
>3k context AND the accuracy of RoPE-extrapolation past 2048 is acceptable. Otherwise skip.

## Recommendation
- **In-spec (≤2048) TTS:** do NOT build TP. Use `torch.compile` (≤3k it wins up to 4.25×).
- **If long context (>3k) is a hard requirement:** build TP per the plan above; TP=4 is the
  likely sweet spot (~2.5–3× prefill), pair with flash/tiled attention for the O(N²) term
  and fp8 weights for the MLP term. Measure TP×compile×dtype — don't assume.
