# DeepSeek-V4-Flash native-PyTorch decode on Trainium2 — path analysis

Field notes from clearing the native-PyTorch (no-XLA) decode path for DeepSeek-V4-Flash
through vLLM-Neuron, on a `trn2.48xlarge`. Everything below is self-measured.

**Headline: it works.** All **15 blockers** are cleared and the full 43-layer decode graph
now compiles *and executes* on the native-PyTorch path, with the golden argmax matching the
XLA reference. Blocker 15 looked for a long time like a compiler bug that only the compiler
team could fix. It was not. It was a **single missing configuration flag**: `ep_degree=8`.

Measured, 43 layers, `backend=neuron_native`, no XLA, TP=32, EP=8, batch=1, prefill 512,
GEN=32 — reproduced on three independent runs:

| run | decode tok/s | TTFT | golden token | `NCC_ITIN902` |
|---|---|---|---|---|
| 1 | 8.92 | 219.9 ms | PASS (4256) | 0 |
| 2 | 6.93 | 221.6 ms | PASS (4256) | 0 |
| 3 | 8.56 | 220.2 ms | PASS (4256) | 0 |

TTFT is stable to ±1 ms while decode spreads 6.9–8.9 tok/s, so treat **~8 tok/s ± 1** as the
batch=1 figure; the spread is host-side noise, not model variance.

**This number is not comparable to the 22.25 tok/s XLA baseline** — that baseline is batch=8
and this is batch=1. Decode is weight-bandwidth-bound, so batching amortises the per-step
weight read and moves throughput by more than an order of magnitude. A batch=8 native
measurement is the correct comparison and is not included here yet. No tuning has been
applied either: sliding-window attention, CTE block-size, and the bf16-matmul/f32-accumulator
work are all still on the table.

This document is the writeup so nobody has to walk the same 15 doors again.

For reference the XLA path is done and published — 43 layers, TP=32, batch 8, prefill
512 tokens, GEN=32 → **22.25 tok/s decode, golden argmax matched.** See the sibling
folder's README.

---

## The 15 blockers, in the order they surfaced

Each was cleared. The fix is in the payload; the "why" is why anyone else pointing this
model at native mode will meet the same wall.

| # | symptom | root cause | fix |
|---|---|---|---|
| 1 | `ModuleNotFoundError: No module named 'uvloop'` on `import vllm` | vLLM's lazy `__getattr__` masks missing deps at import; the concrete probe (`from vllm import LLM, SamplingParams, TokensPrompt`) exposes it | install `uvloop` (and the transitive `--no-deps` graph) |
| 2 | `RuntimeError: Tried to instantiate class 'neuron.Runtime', but it does not exist` inside vllm-neuron's platform auto-detect | the native image does not register `torch.classes.neuron.Runtime` | `NEURON_PLATFORM_TARGET_OVERRIDE=trn2` — the code short-circuits the NRT class query when this is set |
| 3 | `Tried to register multiple backend fallbacks for the same dispatch key PrivateUse1` (native vs libtorch-neuronx-lite both claim it) | native torch_neuronx registers PrivateUse1 in NeuronBindings.cpp; libtorch-neuronx-lite tries to register it again in storage.cpp | `NEURON_EXECUTION_BACKEND=native` — lite's C++ side reads this via `getenv()` and yields the dispatch key |
| 4 | `libtorch_neuronx_lite/__init__.py: import torch_xla` fails (native image has no XLA) | lite's default import path pulls its XLA implementation | `TORCH_NEURONX_DYNAMO_BACKEND_ONLY=1` — makes lite skip the XLA-dependent imports |
| 5 | `ModuleNotFoundError: No module named 'libtorch_neuronx_lite.distributed'` from vllm-neuron 0.21's import redirector | 0.21 rewrites `torch_neuronx.*` → `libtorch_neuronx_lite.*` unconditionally | use vllm-neuron **0.24** — the redirector was removed upstream |
| 6 | `ImportError: Support for Transformers v4 is deprecated and was removed in vLLM v0.24.0` | vLLM 0.24 needs transformers >= 5.5.3; installing dep names without versions leaves the image's shipped v4 in place | honour version specifiers for `transformers` at install time |
| 7 | `FileNotFoundError: vllm_neuron/overrides/__init__.py` | absent in 0.24; collectives moved to `vllm_neuron/functional/collectives` | make the patch that touches this path non-fatal on 0.24 |
| 8 | log/artifact loss across regions (checkpoint only exists in one region; other regions have no path to fetch results) | the launcher has no `--region` flag; artifacts and checkpoint are region-local | embed the model port inside the command payload (~100 KB base64 tarball); write logs to a region-independent workload-output artifact bucket |
| 9 | `SyntaxError: invalid syntax` in a patched model.py — one of my own patches emitted a comment line without the `#` | my patch step E injected a comment-shaped marker without the leading `#` | add `_verify_syntax(path)` to every patch step and fail loudly rather than silently continuing |
| 10 | `ModuleNotFoundError: No module named 'vllm_neuron.nki.nki_hop'` from the model port | vllm-neuron 0.24 moved `nki_hop` under `libtorch_neuronx_lite.nki` | rewrite the port's import to the new location |
| 11 | `ModuleNotFoundError: No module named 'libtorch_neuronx_lite.compile.native_backend'` on step "compile model with vllm_neuron backend" | vllm-neuron 0.24 imports this module during dynamo backend registration; the current `libtorch-neuronx-lite` wheel on the package index does not ship it (a separate "native lite wheel" apparently does, but is not on the index) | shim `native_backend.py` in-place at startup — aliases torch_neuronx's real native `neuron` dynamo backend under a distinct name (docstring forbids `"neuron"`). Documented caveat: the shim may bypass lite's artifact caching and parallel-compile |
| 12 | `TypeError: parallel_trace() got an unexpected keyword argument 'native_compile_only'` — all 64 workers die identically after loading all 43 layers | vllm-neuron 0.24 calls `parallel_trace(..., native_compile_only=True)`; the installed lite wheel's `parallel_trace` signature does not accept that kwarg (same "native lite wheel" gap as blocker 11) | `VLLM_NEURON_DISABLE_PARALLEL_TRACE=1` — this is the **documented native fallback** per vllm-neuron's own source: `_run_parallel_trace_jobs` returns False before ever calling `parallel_trace`, and `_extract_graphs` takes the sequential path that's explicitly labeled "native fallback deliberately does no extraction here — parent warmups retrace each bucket, compile StableHLO sequentially, and execute on NRT" |
| 13 | `RuntimeError: Model warmup failed for {prefill,decode} ... when making fake tensor call` — dynamo graph break `gb4315` in three sites in `vllm_neuron/functional/moe/moe_blockwise.py` | `torch.tensor([python_scalar], device=device)` cannot be lowered under fake-tensor tracing (Dynamo can't materialise a real tensor from Python data in fake mode); `_build_blockwise_mapping_kernel` has this call in **at least three branches**, each surfacing on a different warmup path (prefill / decode sharded / decode unsharded) | AST-based patch that rewrites **every** `torch.tensor([scalar])` call in the file to `torch.full((1,), scalar)`, which has an equivalent FX lowering rule that respects fake tensors. Local self-test confirmed the rewriter is scoped: it does not touch multi-element lists, and does not touch non-`torch.tensor(...)` calls |
| 14 | `WorkloadInterrupted-137-OOMKilled` — pod SIGKILLed 60+ min into compile with 95%+ memory | `VLLM_NEURON_PARALLEL_COMPILE_WORKERS` is **per-worker**; at TP=64 with default 6 this yields up to 384 concurrent `neuronx-cc` processes on the host; peaked at 1.9 TB of 2 TB | `PARALLEL_COMPILE_WORKERS=1` **and** either drop to TP=32 (halves worker count so ~32 cc procs, ~820 GB peak, comfortable) or accept TP=64 with the memory headroom being tighter |
| 15 | `[NCC_ITIN902] TensorInitialization error: idx i_shard_68001: AffineIV doesn't appear in params or loopnest — Please open a support ticket` | **Not a compiler bug.** `build_blockwise_mapping`'s decode path loops `ceil(num_local_experts / 2)` times, issuing one indirect DMA per iteration into a buffer that carries no shard dimension. With no expert parallelism `num_local_experts = 256`, so that is 128 iterations — deep enough for the LNC shard-axis pass to attach an *equality* predicate on a shard induction variable, which the predicate-widening step cannot represent, so the ISL lowering fails | **`ep_degree=8`.** `num_local_experts` becomes 32, the loop becomes 16 iterations, and the predicate never forms |

### The deterministic-failure abort loop

Aside from the fixes themselves, one meta-change was worth its weight during this work:
the workload script's retry loop was rewritten to detect **deterministic** failure
signatures (`TypeError: parallel_trace`, `fake tensor call`, `Model warmup failed`,
`NCC_ITIN902`, and the native_backend `ModuleNotFoundError`) and abort after attempt 1
rather than replaying attempts 2 and 3. Each attempt costs ~30-45 min of weight-load;
skipping two of them on a deterministic failure recovered ~1.5h per bad run and made
debugging tolerable.

---

## How blocker 15 was actually cleared

`ITIN902` reports a *shard* induction variable — `i_shard_68001`. Those exist only because
the LNC=2 sharding passes create them. The chain is:

1. an in-graph indirect **write** becomes an LNC scatter DMA;
2. the shard-axis pass attaches an **equality** predicate relating a loop IV to the shard IV;
3. predicate projection can widen inequalities but leaves equalities intact, so the shard IV
   survives into the ISL lowering, which can only represent loop-nest dimensions and SPMD
   parameters;
4. it is in neither, so the lowering raises.

The indirect write is in the MoE blockwise mapping. Its decode path loops
`ceil(num_local_experts / 2)` times, one indirect DMA per iteration, into a buffer allocated
full-size on every core — so the store has no shard dimension to hang the predicate on.

`num_local_experts = n_routed_experts / ep_degree`. With no expert parallelism that is
256 experts and **128 iterations**. With `ep_degree=8` it is 32 experts and **16 iterations**,
and the failing predicate never gets created.

That is the whole fix. One flag.

### The controlled result

| configuration | outcome |
|---|---|
| TP=32, **EP=8**, 43 layers | **3 / 3 SUCCEEDED**, golden token matched, zero `ITIN902` |
| TP=32, **no EP** | **0 / 3** — `ITIN902` ×226, always the same `i_shard_68001` |

### What was ruled out first, and why it is worth knowing

Three plausible theories were tested and each was disproved by the *same* piece of evidence:
the failing index never moved.

- **Rewriting every in-graph `scatter_`.** All 186 live indirect writes per decode step were
  converted to `arange == idx` broadcast-compare plus `torch.where`, verified bit-exact
  against the originals on CPU across five shape configurations per site. The failing index
  was unchanged. Worth keeping as a cleanliness win; it was not this bug.
- **Upgrading the compiler.** `neuronx-cc` 2.27.5334.0 installs and verifies cleanly over
  2.27.2878.0. The failing index was unchanged.
- **Forcing the sharded mapping branch on decode** (`tp_degree` instead of `1`). 0/3, and that
  line turns out to be a deliberate, measured optimisation rather than a defect.

If an index like `i_shard_68001` is *bit-stable* across changes that substantially alter graph
structure, the thing you changed is not the cause. That signal would have saved several hours
had it been trusted earlier.

### Side benefits of EP=8

Not just a compile fix. Each rank holds 1/8 of the expert weights, so per-rank peak host
memory dropped from roughly 780 GB to about 300 GB, and weight load fell from ~45 min to
~25 min. It also cuts decode weight-bandwidth per rank, which is the dominant term at
batch=1.

### One caveat that cost a run

At batch=8 an over-provisioned `num_gpu_blocks_override=1024` failed at prefill warmup with
`NRT_RESOURCE`. The graph **compiled**; this was purely allocation. KV cache, the prefill
NEFF, the decode NEFF and the weights must all coexist in HBM, and batch=8 only needs
~4.4k token slots. Leave the block count unset and let the serving stack size it.

## What is *not* the wall

Some things are worth explicitly ruling out because they *look* like natural suspects.

- **The `torch.compile` / dynamo path is fine.** Blocker 13's AST patch cleared every graph
  break; after applying it, warmup traced both prefill and decode successfully.
- **`torch_neuronx` native dispatch is fine.** Every one of the 43 layers loaded, and
  compile started — this is well past the point where a bad dispatch would have failed.
- **The model port is fine.** It ran at TP=32 through warmup and at TP=64 through the
  entire compile window without generating any code-level error.
- **HBM was fine.** Reported free 1535 GiB at startup on both good and bad runs.
- **The compile OOM was solved.** Post-blocker-14, memory peaked at ~820 GB of 2 TB at
  TP=32 and ~1729 GB at TP=64 — the latter is closer to the ceiling but stayed under it.
  If TP=64 becomes the target, halving `PARALLEL_COMPILE_WORKERS` was the correct move.

---

## What ships regardless of the wall

- **XLA baseline.** 22.25 tok/s at 43L TP=32 B=8, golden matched, published in the sibling
  folder. Independent of native.
- **Hyper-connection kernel fusion.** Seven fusion increments were built and validated on
  hardware in native mode (no XLA), against float64 references, with a paired repeated
  benchmark. All in `../hyper-connection-fusion/` — bit-identical to the unfused path,
  ratios up to 1.40x when compared to the multi-kernel baseline, and the whole
  `hc_pre` + RMSNorm boundary (and separately the whole `hc_post`+`hc_pre`+RMSNorm
  inter-block transition) runs in a single kernel. **Not wired into the model's forward
  yet** — those are validated as kernels, not as a live speedup.
- **This document.** Any next attempt at this port picks up cleanly from blocker 15
  instead of starting at blocker 1.
