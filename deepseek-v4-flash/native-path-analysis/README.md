# DeepSeek-V4-Flash native-PyTorch decode on Trainium2 — path analysis

Field notes from clearing the native-PyTorch (no-XLA) decode path for DeepSeek-V4-Flash
through vLLM-Neuron, on a `trn2.48xlarge`. Everything below is self-measured.

**Headline:** we cleared **15 concrete blockers** in the software stack and reached actual
compilation of the whole 43-layer decode graph. The last blocker is a compiler internal
error that only the compiler team can fix; the workaround (different TP shape) ran clean
but was killed by a platform-level active-deadline before finishing. So the native path is
now a *known-shape* problem, not a mystery. This document is the writeup so nobody has to
walk the same 15 doors again.

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
| 15 | `[NCC_ITIN902] TensorInitialization error: idx i_shard_68001: AffineIV doesn't appear in params or loopnest — Please open a support ticket` | **Neuron compiler internal invariant violation** during decode graph compile. Not user code — same model compiles fine in XLA mode | (open) partial workaround below |

### The deterministic-failure abort loop

Aside from the fixes themselves, one meta-change was worth its weight during this work:
the workload script's retry loop was rewritten to detect **deterministic** failure
signatures (`TypeError: parallel_trace`, `fake tensor call`, `Model warmup failed`,
`NCC_ITIN902`, and the native_backend `ModuleNotFoundError`) and abort after attempt 1
rather than replaying attempts 2 and 3. Each attempt costs ~30-45 min of weight-load;
skipping two of them on a deterministic failure recovered ~1.5h per bad run and made
debugging tolerable.

---

## Where the wall is

Blocker 15 is a `neuronx-cc` internal error. `AffineIV doesn't appear in params or loopnest`
is a compiler IR invariant, not user-facing. The compiler emits this on the DECODE graph at
TP=32 (128-way sharded expert tensors, high shard indices), and the error message itself
tells you to file a support ticket. The same model compiles cleanly in XLA mode, so the bug
is specific to the graph shape produced by the native backend.

Two workarounds were tried:

1. **TP=64.** Different sharding pattern → different graph shape → *might miss the compiler
   pass that triggers the bug*. Result: **compile ran completely clean for 2h50m with zero
   errors of any kind** — no `ITIN902`, no OOM, no traceback. But the runner's
   active-deadline TTL killed the workload before compilation finished. At TP=64 there
   are twice as many
   subgraphs (128 vs 64 for TP=32), so the compile takes ~2x longer, and it exceeded the
   platform's per-workload wall clock.

2. **TP=32 with `NEURON_CC_FLAGS="--enable-saturate-infinity --disable-internal-io-dge"`.**
   Attempted after run 9 was TTL'd. Not verifiable in this session because subsequent
   workload starts in the same region hung immediately at container step 1 (`import torch`)
   for over 30 minutes with no output; three consecutive relaunches showed identical
   symptoms. This was a platform / regional issue that emerged after run 9 and coincided
   with a service-side `ExpiredTokenException` on the platform's exec API. Not a code
   issue, not reproducible in the pattern of user errors.

### What would actually unblock it

- **Fix the compiler bug.** File a support ticket with the reproducer (this is a small
  shape variation; the graph that hits ITIN902 is a straightforward TP=32 decode of a
  256-expert MoE with sharded expert weights).
- **Or extend the platform active-deadline** on the workload runner so TP=64 compile has
  time to finish. Its progress was on-track and error-free; it wants ~4-5h of runtime
  rather than the ~3h it got.
- **Or ship the "native lite wheel"** referenced by vllm-neuron's own source (blockers 11
  and 12). Its `parallel_trace` supports `native_compile_only` and would enable the fork
  pool, which means compile can parallelise across workers rather than run sequentially in
  each worker's process — potentially bringing the compile time inside the current TTL.
  Right now the wheel on the package index does not include this.

---

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
