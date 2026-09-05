# DeepSeek-V4-Flash decode on native PyTorch (no XLA)

Full 43-layer DeepSeek-V4-Flash decode, compiled and executed through
`torch.compile` -> torch-mlir -> StableHLO -> `neuronx-cc`, with **no torch-xla anywhere in
the stack**, on a single `trn2.48xlarge`.

**It works, and the output is correct.** 15 blockers stood between a model that imports and
a model that decodes. The last one looked for days like a compiler bug that only the
compiler team could fix. It was a single missing configuration flag.

---

## Measured

43 layers, `backend=neuron_native`, TP=32, EP=8, batch 1, prefill 512, GEN=32.
Three independent runs, same payload:

| run | decode tok/s | TTFT | golden token | `NCC_ITIN902` |
|---|---|---|---|---|
| 1 | 8.92 | 219.9 ms | PASS (4256) | 0 |
| 2 | 6.93 | 221.6 ms | PASS (4256) | 0 |
| 3 | 8.56 | 220.2 ms | PASS (4256) | 0 |

TTFT is stable to about +/- 1 ms while decode spreads 6.9-8.9, so **~8 tok/s +/- 1** is the
honest figure at batch 1; the spread is host-side noise, not model variance.

The golden token is the greedy first token for a fixed prompt, and **4256 is the same value
the XLA path produces**. That is the correctness gate: the native path is not merely
producing tokens, it is producing the same tokens.

### What this number is not

- **Not comparable to the XLA path's 22.25 tok/s.** That is batch 8; this is batch 1. Decode
  here is weight-DMA-bound, so batching amortises a fixed per-step weight read. For scale:
  the XLA path on this same model measures 1.35 tok/s at batch 1 and 37.81 at batch 128.
- **Not tuned.** The MoE compute dtype has not been swept.
- **Not batch-scaled, and there is a ceiling.** Batch 32 does not run at 43 layers: it
  compiles clean (zero `NCC_`) and then fails to *allocate*, at prefill warmup ---
  `NRT EXECUTION FAILED: Failed to load model with collectives, Failed to allocate resource`
  for prefill bucket 512. That is device HBM exhaustion, not a graph defect. The same
  signature appears with an over-large `num_gpu_blocks_override`. Batch scaling on this path
  therefore has to be co-tuned with the prefill bucket and KV block count, not raised alone.
- **Not a served endpoint.** This is offline `LLM(...)` + `generate(...)`, not an HTTP server
  with continuous batching. Standing that up is a further step.

## The blocker that mattered

`neuronx-cc` failed the decode graph with:

```
[INTERNAL_ERROR] [NCC_ITIN902] TensorInitialization error: idx i_shard_68001:
<class 'neuronxcc.pelican.ir.AffineIV'> doesn't appear in params or loopnest
```

Prefill compiled cleanly every time. Only decode failed.

`i_shard_68001` is an **LNC shard induction variable** — it exists only because the LNC=2
sharding passes create it. The chain: an in-graph indirect **write** becomes an LNC scatter
DMA; the shard-axis pass attaches an *equality* predicate relating a loop IV to the shard IV;
predicate projection can widen inequalities but leaves equalities intact; the shard IV
survives into the ISL lowering, which can only represent loop-nest dimensions and SPMD
parameters; it is in neither, so the lowering raises.

That indirect write lives in the MoE blockwise mapping. Its decode path loops
`ceil(num_local_experts / 2)` times, one indirect DMA per iteration, into a buffer allocated
full-size on every core — so the store has no shard dimension for the predicate to attach to.

`num_local_experts = n_routed_experts / ep_degree`:

```
no expert parallelism   256 experts  ->  128 loop iterations   ->  fails
ep_degree=8              32 experts  ->   16 loop iterations   ->  compiles
```

The fix is `ep_degree=8`. One flag.

| configuration | outcome |
|---|---|
| TP=32, **EP=8** | **3/3 SUCCEEDED**, golden matched, zero `ITIN902` |
| TP=32, no EP | **0/3** — `ITIN902` x226, always the same `i_shard_68001` |

Side effects, both free: per-rank host memory fell from roughly 780 GB to about 300 GB, and
weight load from ~45 min to ~25 min, because each rank now loads only its own 32 experts.

## Three theories that were wrong, and the tell

Each was disproved by the same observation: **the failing index never moved.**

- Rewrote every in-graph `scatter_` — all 186 live indirect writes per decode step converted
  to `arange == idx` broadcast-compare plus `torch.where`, verified bit-exact on CPU across
  five shape configurations per site. Index unchanged.
- Upgraded the compiler — `neuronx-cc` 2.27.5334.0 over 2.27.2878.0, installed and verified.
  Index unchanged.
- Forced the sharded mapping branch on decode. 0/3, and that line turned out to be a
  deliberate, measured optimisation rather than a defect.

If an index like `i_shard_68001` is **bit-stable** across changes that substantially alter
graph structure, the thing you changed is not the cause. Trusting that signal earlier would
have saved a lot of time.

## Where the 125 ms actually goes

At ~8 tok/s the decode step is about 125 ms, or 2.9 ms per layer. A weight-bandwidth
roofline accounts for only 1-20% of that, so most of the step is unexplained by the thing
that is supposed to dominate memory-bound decode. Two concrete causes have since been
identified by reading the kernel library and the runtime, rather than by guessing.

### 1. Decode is running the prefill MoE kernel

The MoE library exposes two different expert-compute entry points: a **context-encoding**
kernel, which takes a blockwise token-to-expert mapping and walks blocks, and a
**token-generation** kernel, which takes per-token affinities directly. This port called the
context-encoding one for both phases, differing only in a `tp_degree` argument to the mapping
builder.

That is the wrong pairing, and the block loop is why. The context-encoding kernel's loop is a
plain Python `range()` unrolled at trace time over `len(token_position_to_id) // block_size`.
It is not data-dependent. The `conditions` vector that marks which blocks hold real tokens is
accepted by the wrapper and then **never forwarded to the kernel at all** --- the kernel has
no such parameter. So at decode, with roughly one real token, the kernel still executes the
full static block count for every layer. The mapping builder's own block count is
`ceil((T*k - (E_local-1)) / block_size) + E_local - 1`, and that `+ E_local - 1` term is
structural: with 32 local experts you cannot get below 32 blocks regardless of how few tokens
you have.

The token-generation kernel needs no blockwise mapping, no `token_position_to_id`, no
`block_to_expert`, and no `conditions`. It takes `expert_affinities [T, E]` and
`expert_index [T, K]` and does its own expert-local slicing from a rank id. That deletes the
mapping call from the decode path entirely --- which is also where the `NCC_ITIN902` compiler
bug documented above lives.

There is a wrinkle specific to this model. The fused variant that also folds in the RMSNorm
and the router cannot express this router: its activation enum has no `sqrt(softplus(x))`,
it has no notion of a bias that shifts *selection* without shifting the returned affinities,
and it has no routed-scaling-factor argument. So the correct split here is a PyTorch router
feeding the token-generation expert kernel --- which is exactly the split the unfused entry
point exists for. The router is a `[T, 4096] x [4096, 256]` matmul plus elementwise work; it
is not where the time is.

A second, smaller thing in the router: top-k was implemented as `k` sequential
`argmax` + mask passes over `[T, 256]`. At `k=6` that is six full passes to compute what one
`topk` computes.

### 2. The step is collective-dispatch-bound, and the queue depth says so

With 43 layers and 3-4 collectives per layer, a decode token issues roughly 130-170
collectives. On the native path each one compiles to its own small collective-only graph and
is dispatched separately, so there is no compute in the same graph for the compiler to overlap
it against. And the runtime's queue budget makes overlap structurally impossible on that path:
there is **one** compute queue and **one** collectives queue per logical core. Reaching the
collectives queue independently requires issuing the collective on a non-default stream, which
is the thing host CC actually enables and which this port does not do.

125 ms over ~150 dispatches is ~0.8 ms each, which is the right order of magnitude for a
host round trip per collective. A profiled graph of comparable shape showed collectives at
about 43% of the device window with essentially zero overlap against the tensor engine ---
a share that a weight-bandwidth roofline cannot see at all.

Both of these are software-structural, not hardware limits. Neither was visible from the
roofline, which is the general lesson: a 5-100x unexplained gap is usually a
bottleneck-*class* error, and cost models that assume the class only tell you the floor.

### Measuring it rather than arguing about it

For the record, since this cost real time: capturing a device profile on this stack needs
`NEURON_RT_INSPECT_DEVICE_PROFILE=session`. The value `1` selects a per-graph *model* mode
that, on this build, emits the compiled graphs and no trace file at all --- which reads
exactly like profiling being unsupported. It is not. Also worth knowing before you start:
profiling costs roughly 20-25% throughput, so the step time must come from an unprofiled
run and the profile used only for attribution; and with 32 workers there is no rank-scoped
variable, so the env has to be gated on local rank or you get 32 concurrent captures.

## Contents

- [`path-analysis/`](path-analysis/) — all 15 blockers, each with the exact error it produces
  and the fix, in the order they surface. Read this before attempting a similar port.
- [`hyper-connection-fusion/`](hyper-connection-fusion/) — seven NKI fusion increments for
  the hyper-connection boundary, validated against float64 references and bit-identical to
  the unfused path. Built and measured, **not yet wired into the model's forward pass**;
  see the honesty note in that README about what the ratios do and do not mean.

## Reproducing

The configuration that works:

```
TP=32                       tensor parallel across 32 NeuronCores
ep_degree=8                 REQUIRED -- see above
LNC=2                       logical NeuronCore config (the default)
VLLM_NEURON_BACKEND=neuron_native
NEURON_EXECUTION_BACKEND=native
TORCH_NEURONX_DYNAMO_BACKEND_ONLY=1
NEURON_PLATFORM_TARGET_OVERRIDE=trn2
VLLM_NEURON_DISABLE_PARALLEL_TRACE=1
VLLM_NEURON_PARALLEL_COMPILE_WORKERS=1
VLLM_NEURON_MOE_BLOCK_SIZE=128
TORCH_NEURONX_ENABLE_HOST_CC=1
```

Two of those are worth calling out. `VLLM_NEURON_PARALLEL_COMPILE_WORKERS=1` is not a
preference: the setting is per-worker, so the default multiplies by the worker count and can
put hundreds of concurrent compiler processes on one host. And
`TORCH_NEURONX_ENABLE_HOST_CC=1` is needed because without it collective init looks for a
device transport that cannot initialise in this container and blocks indefinitely.

It is worth being precise about what that flag does, because the obvious reading is wrong.
It does **not** move collective payloads through host memory. Reading the runtime source:
the flag calls `nrt_cc_create_stream`, which sets a global permitting host-orchestrated
collectives to coexist with NEFF-embedded ones. The reduce still runs on the device DMA
engines over HBM addresses. What it changes is *scheduling*, and it only changes execution
at all for collectives issued on a **non-default** stream. A plain `dist.all_reduce` is on
the default stream, so on that path the flag is inert for dispatch.

It is not free, though. With host CC on, the runtime reserves a second collective context
and splits the collective TopSPs and DMA queue-bundles between them, defaulting to `3:1`
(`NEURON_RT_CC_STREAM_SPLIT`). So enabling it and then never using a side stream hands
about a quarter of the collective resources to a context that goes unused. The likely
reason it fixed the hang is unrelated to data movement: the barrier-semaphore reservation
grows from one context to the maximum when the flag is set.

A useful trick for iteration: `hf_overrides={"num_hidden_layers": 4}` compiles a 4-layer
slice that still covers `compress_ratios` `[0, 0, 4, 128]`, so it exercises every distinct
layer type and still reproduces the `ITIN902` trigger (the failing loop count depends on
expert count, not layer count). That turns a ~90 minute debug cycle into about 15.
