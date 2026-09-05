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
- **Not tuned.** No batch scaling, and the MoE compute dtype has not been swept.
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
`TORCH_NEURONX_ENABLE_HOST_CC=1` routes collectives through host collective-communication;
without it, collective init looks for a device path that cannot initialise in a container and
blocks indefinitely.

A useful trick for iteration: `hf_overrides={"num_hidden_layers": 4}` compiles a 4-layer
slice that still covers `compress_ratios` `[0, 0, 4, 128]`, so it exercises every distinct
layer type and still reproduces the `ITIN902` trigger (the failing loop count depends on
expert count, not layer count). That turns a ~90 minute debug cycle into about 15.
