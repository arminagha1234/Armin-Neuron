# `torch.compile(backend="neuron")` on the Mochi-1 DiT — research report

Design-doc "step 0": the untried lever that resets the baseline for every NKI
kernel decision. This is pure research (local files + AWS docs); nothing was
run on device. Commands below are meant to be executed under device
serialization.

**Stack (from the live box):** torch 2.11.0, torch_neuronx 2.11.3,
neuronx-cc 2.26, Python 3.12, Beta-3 DLC. TP=4 via
`torch.distributed.tensor.parallel`. Transformer forward is fully static-shape
(no data-dependent ops; `torch.nonzero` was removed in the port). RoPE tables
cross the compile boundary as fp32.

> **Note on status vs the README.** The Mochi README/NOTES already record a
> *completed* on-device compile run (2026-07-29): 31f warm 589 s (~11–15 s/it),
> 61f cold 3059 s with ~8 s/step warm, persistent cache at
> `/data/neuron_cache` surviving `docker restart`, and `--rope-bf16` *not*
> needed. So compile is no longer strictly "untried" — but the design doc's
> `KERNEL_PLAN.md` still frames it as step 0, and the numbers below both
> explain *why* those README results look the way they do and give the exact
> recipe to reproduce/extend them cleanly. Where the README's measured result
> and a doc-derived expectation differ, I call it out.

---

## TL;DR / ranking

1. **Compile's headline win for Mochi is memory, not throughput.** It already
   took the envelope from 31 → 61 frames on a 24 GB/rank box by replacing the
   eager caching allocator (which fragmented: 23.25 GB allocated, largest free
   chunk only 221 MB) with whole-graph buffer assignment. Warm per-step is in
   the *same ballpark as eager* (~8–15 s/step vs ~6.3 s/step eager), sometimes
   slower. This is the single most important and slightly counter-intuitive
   finding, and it is already confirmed on device.
2. **Is compile a bigger win than the NKI kernels?** For *reach/memory*, yes —
   it is the cheapest way to run >31 frames and it is done. For *raw
   throughput at the validated 31f/64-step config*, no — compile roughly
   matches eager there. The one NKI kernel that beats compile on both axes is
   the **flash joint-attention kernel (KERNEL_PLAN #1)**, because the compiler
   provably cannot produce it (stock SDPA miscomputes on the Neuron bf16
   backend, and there is no flash/online-softmax rewrite for the custom
   joint+additive-bias layout). Everything else in the plan (#2 FFN, #3 QKV,
   #4 norms, #5 RoPE) is gated behind a compile measurement precisely because
   compile is expected to absorb most of their value.
3. **Recommended order:** (a) lock the persistent NEFF cache and profile a
   warm compiled run at 31f to get the real per-op cost table; (b) build the
   flash joint-attention kernel (#1) — the only "build now"; (c) only build
   #2 (FFN) if the compiled profile still shows it top-2; (d) fold #4/#5/#3
   into #1/#2 as prologues, never standalone.

---

## Q1 — What does `backend="neuron"` actually do vs eager, for a static-shape TP model? Expected speedup?

**Mechanism (AWS "TorchNeuron Native Backend Overview",
`frameworks/torch/pytorch-native-overview.html`):** TorchNeuron implements a
**custom TorchDynamo backend** — it does *not* use Inductor. The pipeline is:
Dynamo captures Python bytecode into an FX graph → AOT Autograd splits
forward/backward → the Neuron backend lowers the FX graph(s) to Neuron IR
(which "may include NKI kernels passed as custom ops") → neuronx-cc generates
Trainium instructions. Each Dynamo-captured region compiles to a compiled
graph artifact (a NEFF); TorchDynamo's caching and **graph breaks are
supported** ("Yes, the Neuron TorchDynamo backend supports graph breaks").

**Eager path, for contrast:** ops dispatch immediately, but TorchNeuron runs
an **Adaptive Eager Execution** layer that enqueues operators asynchronously
and *fuses* runs of them "into single operators based on fusion heuristics,"
preserving order and numerics. So eager on Neuron is already doing some
runtime fusion — which is part of why compile's throughput win over it is
modest here.

**What this means for a static-shape DiT:** with `dynamic=False` and no
data-dependent ops, one Mochi transformer forward traces into essentially a
**single large graph region** (modulo any graph breaks — see Q3). The 48
blocks are captured together and buffer-assigned as a whole, which is the
mechanism behind the memory win: neuronx-cc plans one contiguous allocation
for the whole forward instead of the eager allocator servicing thousands of
independent per-op allocations and stranding memory in small chunks (the
README's fragmentation OOM). Note: the doc does **not** state NEFF
granularity explicitly (whole-graph vs sub-graph); the observed behavior
(one ~25 min compile for 61f, one persisted NEFF that all steps reuse) is
consistent with one dominant NEFF for the forward.

**Expected speedup range for a diffusion DiT:** compile-vs-eager throughput
speedup on Neuron is highly workload-dependent. Two internal reference points:

- **Where compile crushes eager:** the mamba3 probe
  (`knowledge/nki-kernels/mamba3/probe_kernel_vs_compile.py`,
  `PHASE4_PERF_FINDINGS.md`) and `pytorch_32k_LEARNINGS.md` both show eager
  per-op dispatch pins host CPU at 100% building the graph at high op counts;
  compile removes that. The 32K learning is explicit: *"eager per-op dispatch
  at 32K pins ranks at 100% CPU building the graph … `torch.compile(backend=
  "neuron", dynamic=False, fullgraph=True)` is the right call."* That regime
  is dispatch/CPU-bound.
- **Where compile ≈ eager (Mochi's regime):** Mochi's forward is large,
  compute-heavy matmuls (48 blocks × big GEMMs) with relatively *few, large*
  ops per block. The eager Adaptive-Eager fusion already covers most of the
  cheap elementwise fusion, and the matmuls are already efficient. So the
  measured result — warm compiled ~8–15 s/step vs ~6.3 s/step eager — is the
  expected outcome: **near-parity, memory-dominated, not throughput-dominated.**

Bottom line for Q1: expect **little-to-no per-step speedup** for Mochi from
compile alone; expect a **large effective-capacity gain** from buffer
assignment. Treat compile as a memory/envelope lever, and get throughput from
the flash-attention kernel.

---

## Q2 — Cold-compile cost and making it a one-time cost (persistent NEFF cache)

**Cold cost observed:** README reports ~770 s NEFF build for 31f, and ~25 min
for the single 61f graph. This is paid once *per distinct graph geometry*.

**Persistent cache — how it works (AWS `neuron-caching.html`):** Neuron has a
two-level cache. The in-memory JIT cache prevents duplicate compiles *within*
a run; the **Neuron Persistent Cache** prevents duplicate compiles *across*
runs by writing compiled graphs to disk (or S3). On a fresh process the JIT
cache is empty, so the graph is "sent for compilation," but neuronx-cc checks
the persistent cache first and **returns the stored NEFF if the key matches**,
skipping the build. Only *compute graphs* are cached — weights are treated as
inputs, so parameter values do not invalidate the cache.

**Exact env vars:**

| Var / flag | Effect | Notes |
|---|---|---|
| `NEURON_COMPILE_CACHE_URL=<path or s3://…>` | Sets the persistent cache location | Default is `/var/tmp/neuron-compile-cache`. This is the one the Mochi README uses (`/data/neuron_cache`) and it survived `docker restart`. |
| `NEURON_CC_FLAGS="--cache_dir=<path>"` | Also sets cache location; **takes priority** over `NEURON_COMPILE_CACHE_URL` if both set | Same flag channel as all other compiler flags. |
| `NEURON_CC_FLAGS="--no_cache"` | Disables caching | For forcing a clean compile when debugging. |
| `NEURON_CC_FLAGS="--retry_failed_compilation"` | Recompiles graphs previously marked failed | **Important gotcha:** a failed/interrupted compile is cached *as a failure* and skipped on rerun unless you set this. |

**Cache key = hash of (compiler flags + the graph).** It invalidates on:
compiler-version change, and any graph change — which for Mochi includes
**`--num-frames`, `--guidance-scale` (CFG batch 1 vs 2), `--q-chunk`, and
`--norm-tile`**. NOTES.md already flags this: "Tile size affects the compiled
graph shape, so changing it invalidates cached NEFFs — set it once." So each
distinct `(frames, CFG, q_chunk, norm_tile)` tuple pays the cold build once.

**How to warm it:**
1. Put the cache on a persistent, large volume (**not** the root disk — see the
   32K learning: neuronx-cc also writes scratch to `/tmp/neuron_backend`
   hardcoded and can fill a small root disk with `LLVM ERROR: No space left`).
   Use `NEURON_COMPILE_CACHE_URL=/data/neuron_cache` and, if root disk is
   small, `ln -sfn /data/neuron_backend /tmp/neuron_backend` and
   `TMPDIR=/data/tmp`.
2. Run the target geometry once (pays the cold build), then all subsequent
   runs of the *same* geometry skip it. The README's warm 31f run (589 s vs
   1360 s cold) is exactly this.
3. There is no separate "warmup API" for the native backend — warming = run
   the real forward once. In practice the first denoise step absorbs the whole
   NEFF build; steps 2..N reuse the in-process NEFF.

> **Caution from `LEARNINGS_V1_V9.md`:** *always measure warm, and beware the
> cache masking latent failures.* On Gemma4 every "working" run was silently
> served from a cached NEFF; a fresh compile exposed an SBUF overflow the cache
> had been hiding. For Mochi, before claiming a config "compiles," force one
> clean compile with `--no_cache` (or a fresh `NEURON_COMPILE_CACHE_URL`) so a
> latent overflow can't hide behind a stale NEFF.

---

## Q3 — Failure modes with `fullgraph=False` + DTensor/TP + collectives inside the compiled region

**What the port does today:** `torch.compile(transformer, backend="neuron",
dynamic=False, fullgraph=False)` (run_mochi_native.py:311–315). TP is DTensor
via `parallelize_module`; the `RowwiseParallel` blocks all-reduce block outputs
back to full 3072 width inside the forward.

**Does the all-reduce compile cleanly or graph-break?** Key facts:

- The native backend **supports `torch.distributed` including DTensor and
  Tensor Parallelism** (native-overview page), and **supports graph breaks**.
  So `fullgraph=False` is the safe/forgiving setting: if a collective or a
  DTensor dispatch can't be captured into the graph, Dynamo inserts a graph
  break and runs that region eagerly rather than erroring. The README's
  working compiled runs are the evidence this holds for Mochi.
- **`fullgraph=True` is riskier** and is *not* what you want first here. It
  forbids graph breaks, so any DTensor op or collective Dynamo can't trace
  becomes a hard error. The 32K learning recommends `fullgraph=True` *only*
  after switching to **functional collectives**
  (`torch.distributed._functional_collectives`), which are traceable and
  "avoid graph breaks." Mochi's TP all-reduce comes from DTensor
  `RowwiseParallel`, not hand-written functional collectives, so `fullgraph=
  True` may graph-break or error on the collective. **Stay at
  `fullgraph=False`** unless/until you convert TP to functional collectives.
- **Overlap:** per the overview, compute/communication overlap "is supported by
  the Neuron Compiler itself, and not by the TorchDynamo backend" — so whether
  the all-reduce ends up inside the NEFF or on a graph-break boundary, overlap
  is the compiler's job, not something you tune in Dynamo.

**`TORCH_NEURONX_ENABLE_HOST_CC=1` interaction with compile.** This is
mandatory for the port (README: without host collective communication the
all-reduce takes the OFI/EFA device path and hangs on the barrier). It selects
*where the collective executes* (host CC path) and is orthogonal to whether the
collective is *captured* into the compiled graph. Keep it set for compiled runs
exactly as for eager — the README's compiled runs used the same env. The
benign `aws-ofi-nccl initialization failed` warning appears in working
compiled runs too. Also keep `TORCH_NEURONX_ENABLE_ASYNC_NRT=1`.

**Practical failure modes to expect (ranked):**
1. **Cold-compile time explodes** (already known: 10–30 min). Mitigate with the
   persistent cache (Q2). Not a correctness issue.
2. **Graph break around the DTensor all-reduce** → a chunk of each block runs
   eager. Harmless with `fullgraph=False`; costs some of compile's potential
   throughput. Detect with `TORCH_LOGS="graph_breaks"` (see Q6).
3. **`fullgraph=True` hard error on the collective / DTensor** — don't use it
   yet.
4. **Cache-masked latent SBUF overflow** — force one clean compile (Q2).
5. **Barrier/`init_process_group` breakage between runs** — routine, restart
   the container (README).

---

## Q4 — The fp32-RoPE-across-compile-boundary risk (LTX-2 fix #5/#8)

**What the risk is.** In the LTX-2 port, RoPE cos/sin tables computed in fp32
and passed *into* the compiled region could be mishandled at the compile
boundary — the compiled graph would consume them at the wrong precision or the
autocast/downcast interacted badly — producing output that is *structured but
wrong* (not a crash, not NaN — the worst kind: looks plausible, is incorrect).
The mitigation was `--rope-bf16`: emit the tables as bf16 so there is no
fp32→bf16 reconciliation across the boundary.

**In Mochi's code:** `patch_rope_cpu_precompute(model, rope_dtype=torch.bfloat16
if args.rope_bf16 else torch.float32, …)` (run_mochi_native.py:207–211). RoPE is
CPU-precomputed and the tables are attributes the forward reads; with
`--rope-bf16` they are bf16, otherwise fp32.

**What actually happened on Mochi (resolved on device, NOTES.md §"Open
questions", README):** the failure mode **did not reproduce**. *"fp32 RoPE
tables cross the compile boundary fine here … `--rope-bf16` was never needed;
eager and compiled both produce correct output with fp32 tables."* Two
structural reasons Mochi is safer than LTX-2 (NOTES "Why this port is
tractable"): Mochi's RoPE is **real sin/cos arithmetic with no
`view_as_complex`**, and the QK norms are `[head_dim]`-shaped so they stay
replicated on the sharded head axis — none of LTX-2's complex-freq / adaptive-
QK-norm machinery that made the boundary fragile.

**Recommendation:** run compile with default fp32 RoPE. Keep `--rope-bf16` as
the *first* knob to try **only if compiled output is structured-but-wrong**
(and note it changes the graph → new cache key). It is a mitigation kept in the
kit, not a required flag for Mochi.

---

## Q5 — Does compile subsume the RMSNorm / RoPE / fused-QKV NKI kernels? Where does a hand kernel still beat the compiler?

**Short answer:** compile is expected to absorb the *elementwise/GEMM-fusion*
kernels (#3 QKV, #4 norms, #5 RoPE) and take a big bite out of #2 (FFN). It
**cannot** produce #1 (flash joint attention), which is the only "build now."
This is exactly how `design/KERNEL_PLAN.md` ranks them; the reasoning:

- **#4 Modulated RMSNorm / AdaLN — compile wins.** RMSNorm (upcast → square →
  mean → rsqrt → mul → downcast → modulate) is *the* canonical fusible pattern
  for an inductor-class backend, and the Neuron backend has first-class support
  for it. The *memory* problem (fp32 full-sequence upcast, the real
  long-sequence wall — 436 MB @ 61f, 550 MB @ 163f) is **already solved** in
  PyTorch by `mochi_norm_memory.py` (tiled, bit-exact). A compiled fused norm
  would recover the elementwise fusion *and* the buffer assignment without the
  fp32-boundary worry. **Do not build a standalone norm kernel** — fold norm
  modulation into #1/#2 prologues only if those kernels get built.
- **#5 RoPE apply — compile wins standalone.** Cheap interleaved elementwise;
  compile fuses it trivially and elides the two rotated `(2,9540,6,128)`
  intermediates. Value *only* as a fused prologue inside #1, where rotated Q/K
  never leave SBUF.
- **#3 Fused QKV — compile likely wins.** Plain GEMMs; the only NKI edge is
  dispatch count (small vs attention) and const-folding the stacked weight,
  which compile can also do. Fold into #1 at most.
- **#2 Fused SwiGLU FFN — bench compile first.** SwiGLU is the canonical
  matmul→SiLU·gate→matmul pattern compilers fuse well, so compile is expected
  to recover the wide-intermediate-HBM saving on its own. FFN is the largest
  FLOP consumer (~138 TFLOP/step), so *if* the compiled profile still shows it
  top-2, a kernel that also overlaps the down-proj all-reduce and fuses the
  pre-norm could add a residual win. Decision gated on the profile.

**Where a hand NKI kernel STILL beats the compiler — #1, flash joint
attention (BUILD NOW).** Two independent reasons compile cannot match it:

1. **Correctness forces a custom path already.** Stock
   `F.scaled_dot_product_attention` "miscomputes on Neuron's compiled bf16 lazy
   backend" — this is *why* the port replaced it with the explicit
   `bmm→(+mask)→softmax→bmm` shim (`neuron_compat.py`, install_bmm_sdpa
   monkeypatches `F.scaled_dot_product_attention` globally). So the compiler's
   fused SDPA is off the table; the realistic compile baseline is the
   *materialized* BMM path, which torch.compile **cannot** rewrite into flash
   attention — there is no online-softmax rewrite in the Neuron backend for
   this custom joint (`[visual|text]`) + additive-key-bias layout.
2. **It eliminates score/prob materialization the compiler must keep.** The
   materialized path round-trips `scores` and `probs` through HBM per block
   (~240 MB each at 31f CFG; 7.5 GB+ at 61f, which is what forces tiny
   `q_chunk`). A flash kernel keeps a `≤128×512 fp32` score block (~256 KB) +
   running stats in SBUF; HBM traffic for scores/probs → zero. `d=128` lands
   exactly on the 128×128 tensor-engine partition dim, so QK^T needs no padding.
   Template: nkilib `attention_cte` with `causal_mask=False` + the key-bias
   fold.

This matches the durable NKI principle in the plan: **correctness-forced custom
paths and memory-blowup eliminations are where hand NKI pays for itself; pure
elementwise/GEMM fusion is where the compiler has caught up.** For Mochi that is
exactly one "build now" kernel.

---

## Q6 — Concrete step-by-step recipe (exact env vars, warmup, cache, measurement, A/B)

All commands are for the trn2.48xlarge, Beta-3 DLC, TP=4, run under device
serialization. Base env block (matches README + the cache/scratch hardening
from the 32K learning):

```bash
# --- persistent NEFF cache + scratch on a big persistent volume ---
export NEURON_COMPILE_CACHE_URL=/data/neuron_cache        # survives docker restart
mkdir -p /data/neuron_cache /data/neuron_backend /data/tmp
ln -sfn /data/neuron_backend /tmp/neuron_backend          # neuronx-cc hardcodes /tmp/neuron_backend
export TMPDIR=/data/tmp

# --- collectives + runtime (mandatory for the TP all-reduce; same as eager) ---
export NEURON_RT_NUM_CORES=4
export TORCH_NEURONX_ENABLE_HOST_CC=1        # without this the all-reduce hangs on the barrier
export TORCH_NEURONX_ENABLE_ASYNC_NRT=1

# --- CPU threads: torchrun sets OMP_NUM_THREADS=1, which cripples the CPU VAE ---
export OMP_NUM_THREADS=48 MKL_NUM_THREADS=48   # 192 vCPU / 4 ranks
export TOKENIZERS_PARALLELISM=false
```

**Step 0 — pin geometry.** Choose ONE `(frames, CFG, q_chunk, norm_tile)` and
keep it fixed across the whole A/B, because each distinct tuple is a separate
cache key and a separate ~10–30 min cold build. Recommended A/B geometry:
**31 frames, CFG on (guidance 4.5), 64 steps, q_chunk auto, default norm_tile**
— the validated reference config.

**Step 1 — eager baseline (A).** Get a clean warm eager s/step:

```bash
torchrun --nnodes 1 --nproc_per_node 4 --rdzv_backend c10d \
  --rdzv_endpoint localhost:29500 \
  src/run_mochi_native.py --num-frames 31 --num-steps 64 --guidance-scale 4.5
# read results/run_report.json -> "s_per_step" ; expect ~6.3 s/step
```

**Step 2 — force ONE clean cold compile (expose latent overflows).** Do the
first compiled run with caching *disabled* so a cache-masked SBUF overflow
(the Gemma4 trap) can't hide:

```bash
NEURON_CC_FLAGS="--no_cache" \
torchrun ... src/run_mochi_native.py --num-frames 31 --num-steps 4 --guidance-scale 4.5 --compile
#  ^ 4 steps is enough to prove it compiles + runs; step 1 absorbs the ~770 s build
```
If this OOMs/overflows, that is the *real* compile status of the geometry — not
a cache artifact.

**Step 3 — warm the persistent cache.** Re-run the identical geometry *with*
the cache (drop `--no_cache`); it should skip the ~770 s build:

```bash
torchrun ... src/run_mochi_native.py --num-frames 31 --num-steps 64 --guidance-scale 4.5 --compile
# read "s_per_step"; README warm result was ~11-15 s/it at 31f (i.e. compile ~= or slower than eager here)
```

**Step 4 — compiled measurement (B).** The `run_report.json` `s_per_step` from
Step 3 is B. Compare A (Step 1) vs B (Step 3). **A/B cleanly by:** same
geometry, same seed (`--seed 42` default), warm cache for B, and read
`elapsed_s / num_steps` from `results/run_report.json` (the runner already
computes `s_per_step`). Ignore step-1 of the compiled run (it carries the NEFF
build). For per-step granularity, watch the progress-bar it/s on steps 2..N.

**Step 5 — profile per-op cost (the payoff for the kernel decision).** With a
warm NEFF, capture where time actually goes so KERNEL_PLAN #2/#3/#4/#5 can be
decided on data, not guesswork. Use the fallback-op tracker (already documented
in `docs/native-pytorch-op-tracking.md`) to confirm nothing silently fell back
to CPU:

```bash
export TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1
# in a one-off harness around one forward:
#   torch_neuronx.clear_op_tracking(); <forward>;
#   print(torch_neuronx.get_fallback_ops(), torch_neuronx.get_executed_ops())
```
For graph-break visibility during compile, set `TORCH_LOGS="graph_breaks,recompiles"`
before the run — this tells you whether the DTensor all-reduce broke the graph
(Q3) and whether anything is recompiling (a sign of an unpinned dynamic shape).

**Step 6 — memory envelope check (compile's actual win).** Re-run at **61
frames** with compile to reproduce the README's headline (61f eager OOMs before
step 0; compiled it runs). This is the concrete demonstration that compile's
value here is buffer assignment:

```bash
torchrun ... src/run_mochi_native.py --num-frames 61 --num-steps 8 --guidance-scale 4.5 --compile
```

**Expected pitfalls (checklist):**
- Cold build 10–30 min per geometry — expected, cached after first.
- Failed compile is cached as a failure → add `NEURON_CC_FLAGS=
  "--retry_failed_compilation"` if a fixed config still "fails" instantly.
- Changing `--q-chunk` / `--num-frames` / `--guidance-scale` / `--norm-tile`
  = new cache key = new cold build. Pin them.
- Root-disk fill from `/tmp/neuron_backend` → use the symlink above.
- Don't use `fullgraph=True` yet (the code uses `False`; keep it — Q3).
- If output is structured-but-wrong under compile, first try `--rope-bf16`
  (Q4), though it was not needed in practice.
- Restart the container between TP runs (stale runtime state breaks
  `init_process_group`).

---

## Is compile a bigger win than the NKI kernels, and in what order?

**Bigger win on memory/reach: yes, and it's the cheapest.** Compile is the
reason >31 frames run at all on 24 GB/rank, with zero kernel-authoring effort.
That alone justifies making it the default for any long-clip run and locking the
persistent cache.

**Bigger win on throughput: no.** Warm compiled per-step ≈ eager (sometimes
slower) for Mochi's compute-bound, few-large-ops-per-block forward, because
eager's Adaptive Eager Execution already fuses the cheap stuff and the matmuls
are already efficient. The throughput lever is the flash attention kernel.

**Order of operations:**
1. **Compile + persistent cache + profile (days, done-ish).** Lock
   `NEURON_COMPILE_CACHE_URL`, force one clean compile, warm the cache, and get
   the per-op profile at 31f warm. This *resets the baseline* every kernel is
   judged against and confirms the memory envelope. It likely absorbs #3 (QKV),
   #4 (norms), #5 (RoPE) outright and dents #2 (FFN).
2. **Build KERNEL_PLAN #1 — flash joint-attention** (`attention_cte`,
   `causal_mask=False`, + additive key-bias fold). The only kernel the compiler
   *cannot* produce, the biggest throughput lever, and correctness-forced. This
   is where hand NKI pays for itself.
3. **Build #2 (fused SwiGLU FFN) only if the compiled profile shows it top-2**,
   preferably fusing `norm3`/`norm4` modulated-RMS as its pre-norm.
4. **Fold #5 (RoPE) and #4 (norm1 modulation) into #1's prologue** as a second
   pass — never standalone.
5. **Skip #3 standalone;** fold QKV into #1 only if projection shows up in the
   profile after compile.

---

## Sources

**Local files (repo):**
- `neuron/examples/Mochi/README.md` — measured eager/compiled numbers, memory
  envelope, fragmentation OOM, TP/collective story, warm-vs-cold cache result.
- `neuron/examples/Mochi/NOTES.md` — op triage, RoPE tractability, fp32-RoPE
  resolved-on-device note, cache survives `docker restart`.
- `neuron/examples/Mochi/src/run_mochi_native.py` — the `--compile`,
  `--rope-bf16`, `--q-chunk`, `--norm-tile` paths; TP setup; env expectations.
- `neuron/examples/Mochi/src/neuron_compat.py` — BMM-SDPA shim (why stock SDPA
  is off the table); tile→graph-shape/cache-key coupling.
- `neuron/examples/Mochi/nki_kernels/design/KERNEL_PLAN.md` — the kernel ranking
  this report's Q5 is grounded in.
- `neuron/docs/native-pytorch-op-tracking.md` — fallback-op tracking APIs.
- `neuron-knowledge-pack-public/knowledge/learnings/pytorch_32k_LEARNINGS.md` —
  eager-dispatch-CPU-bound at high op count; `torch.compile(dynamic=False,
  fullgraph=True)` + functional collectives; `/tmp/neuron_backend` disk-fill
  fix.
- `.../learnings/LEARNINGS_V1_V9.md` — cache-masked latent SBUF overflow ("always
  measure warm, beware the NEFF cache"); check persistent cache for the good
  NEFF.
- `.../nki-kernels/mamba3/PHASE4_PERF_FINDINGS.md`,
  `probe_kernel_vs_compile.py` — compile-vs-kernel crossover; dispatch overhead
  is real in standalone but gone in a compiled graph.

**AWS Neuron docs (public):**
- TorchNeuron Native Backend Overview —
  `https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/pytorch-native-overview.html`
  (custom TorchDynamo backend, no Inductor; graph breaks supported; DTensor/TP/
  FSDP support; SimpleFSDP recommended under compile; overlap is the compiler's
  job; compile requires Trainium hardware; closed Beta).
- Neuron Persistent Cache —
  `https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-features/neuron-caching.html`
  (`NEURON_COMPILE_CACHE_URL`, `NEURON_CC_FLAGS --cache_dir` priority, default
  `/var/tmp/neuron-compile-cache`, S3 support, `--no_cache`,
  `--retry_failed_compilation`, cache key = flags+graph, failed-compile
  caching).
- PyTorch on Neuron index —
  `https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/index.html`
  (TorchNeuron Native = eager + torch.compile; torch-neuronx = XLA).

**Gaps / not found in public docs:** the public pages do **not** document
`NEURONX_CACHE` (only `NEURON_COMPILE_CACHE_URL` / `--cache_dir` are documented;
`NEURONX_CACHE` appears in older XLA-flow references but is not the native-
backend var), nor do they state NEFF granularity, nor `dynamic`/`fullgraph`
semantics for the native backend — those were inferred from the general
TorchDynamo contract plus the on-device behavior recorded in the Mochi README
and the internal learnings. Verify `NEURONX_CACHE` behavior empirically if you
rely on it; prefer `NEURON_COMPILE_CACHE_URL`.
