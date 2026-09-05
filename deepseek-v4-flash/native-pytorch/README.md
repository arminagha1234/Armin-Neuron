# DeepSeek-V4-Flash decode on native PyTorch (no XLA)

Full 43-layer DeepSeek-V4-Flash decode, compiled and executed through
`torch.compile` -> torch-mlir -> StableHLO -> `neuronx-cc`, with **no torch-xla anywhere in
the stack**, on a single `trn2.48xlarge`.

**It works, and the output is correct.** 15 blockers stood between a model that imports and
a model that decodes. The last one looked for days like a compiler bug that only the
compiler team could fix. It was a single missing configuration flag.

---

## Measured

43 layers, `backend=neuron_native`, TP=32, EP=8, batch 1, prefill 512, **128 tokens
requested and 128 generated** (`gen_actual=128/128`, 127 decode steps, `ignore_eos=True`).
Golden token 4256 and zero `NCC_` on every row.

| run | model.py | decode MoE | host CC | decode tok/s | ms/step |
|---|---|---|---|---|---|
| 1 | baseline | `moe_cte` | on | 5.19 | 193 |
| 2 | baseline | `moe_cte` | on | 5.07 | 197 |
| 3 | baseline | `moe_cte` | **off** | 5.42 | 185 |
| 4 | + router/pow2 | `moe_cte` | on | 6.08 | 165 |
| 5 | + router/pow2 | `moe_cte` | on | **6.30** | **159** |
| 6 | + router/pow2 | `moe_tkg` | on | 6.26 | 160 |

**Decode is ~6.2 tok/s at batch 1, about 160 ms per step.** TTFT is ~217 ms.

### What moved it

**Two small rewrites are worth +20.7%** (rows 1-2 vs 4-5: same kernel, same collective
config, differing only in these edits, against a +/- 1.2% run-to-run spread):

- The router selected its top-6 experts with **six sequential `argmax` + suppress passes**
  over a `[T, 256]` score tensor. That is six full reductions per MoE layer, 43 layers deep,
  to compute what one `topk` computes. Replaced with a single `torch.topk` plus the same
  arange-comparison one-hot the hash-routing branch already used. Verified bit-identical to
  the old mask on 200 random cases across five batch sizes, and identical in the downstream
  affinities.
- `x.pow(2)` -> `x * x` at three RMSNorm sites. `pow` lowers to HLO `power`, which consumes
  one of eight activation-lookup-table slots; `multiply` does not. Bit-exact by construction
  and verified over 18 shape/dtype combinations.

Neither is clever. Both were sitting in the hot path for 43 layers.

**`moe_tkg` is correct and throughput-neutral** (row 6 at 6.26 against a `moe_cte` mean of
6.19, inside the +/- 1.8% spread). It produces the same golden token through an entirely
different expert kernel with no blockwise mapping, which is worth having, but it does not
move batch-1 throughput. Kept behind `VLLM_NEURON_V4_DECODE_MOE=tkg|cte`, default `cte`.

**Host CC off looks like +5.7%** (row 3 at 5.42 against 5.19/5.07 with it on). Only one
sample, so suggestive rather than established -- but it is above both on-runs and it agrees
in direction with an isolated all-reduce probe, so it is probably real. Repeats running.

### Measure over a long generation, or do not measure

The same build, same box, differing only in how many tokens were generated:

| tokens generated | decode tok/s | run-to-run spread |
|---|---|---|
| 4 (EOS, uncontrolled) | 0.67 - 0.93 | +/- 16% |
| 32 | 3.37 - 3.77 | +/- 6.0% |
| 128 | 5.07 - 5.19 | +/- 1.2% |

A short window inside one `generate()` call is dominated by fixed per-request overhead, so
it is **biased low as well as noisy** -- 32 tokens understated the rate by 42%, and 4 tokens
by a factor of six. And with a +/- 6% floor at 32 tokens, several of the A/Bs run early in
this work could not have resolved their own effect even in principle. Establish the noise
floor before deciding what is worth comparing.

Two figures on this page were published and retracted before this was understood. An early
revision reported ~8 tok/s, from dividing by the *requested* token count while the model hit
EOS after 4 tokens. A later one reported `moe_tkg` as 9% slower, from runs that turned out
to be byte-identical because the deploy step preferred a stale copy of the model source over
the one shipped with the launch. Both are written up below rather than quietly removed; the
second in particular has a lesson worth more than the number.

The golden token is the greedy first token for a fixed prompt, and 4256 is the same value
the XLA path produces -- across every configuration above, including a completely different
expert kernel.

### What this number is not

- **Not comparable to the XLA path's 22.25 tok/s.** That is batch 8; this is batch 1. The
  like-for-like pair is batch 1 against batch 1: XLA measures **1.35 tok/s** there, and this
  native path now measures **~6.2**. Read in that direction the native path is ahead at batch
  1; the XLA path wins overall because it scales with batch (37.81 at batch 128) and this one
  cannot yet leave batch 1.
- **Batch 8 does not run on this path at all today.** It deadlocks; see below.
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

## Where the 160 ms goes

At 6.2 tok/s the decode step is about 160 ms, or 3.7 ms per layer. Measured collectives
account for ~34 ms (~21%) and a weight-bandwidth roofline for a further modest share, so
roughly half the step is still unattributed by the things that are supposed to dominate
memory-bound decode.

Two structural causes were identified by reading the kernel library and the runtime. The
first has an implementation but no valid measurement yet, for a reason that is itself the
most transferable lesson on this page -- see the failure note at the end.

### 1. Decode runs the prefill MoE kernel

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

**Measured result: correct, throughput-neutral.** 6.26 tok/s against a `moe_cte` mean of
6.19 over the same code, inside the +/- 1.8% spread. An earlier revision reported it as 9%
slower; that was withdrawn because the deploy step had preferred a stale copy of `model.py`
and the compared runs were byte-identical (failure note below).

Two things temper the expected gain even so, and both are worth stating before measuring
rather than after. The context-encoding call already passes `skip_token=True`, which is
exactly the mechanism for skipping padded token slots -- so the padded work was probably
already being elided and the "128x waste" framing above overstates it. And the
token-generation path runs `is_all_expert=True`, walking all 32 local experts for every
token, so at batch 1 it may trade one form of fixed work for another. The clearer wins are
structural rather than arithmetic: the blockwise mapping disappears from decode, and with it
the call site of the `NCC_ITIN902` compiler bug.

Selected by `VLLM_NEURON_V4_DECODE_MOE=tkg|cte`, default `cte` until it is measured.


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

### 2. Collectives are ~21% of the step, and the queue depth explains why they don't overlap

With 43 layers and 3-4 collectives per layer, a decode token issues roughly 130-170
collectives. On the native path each one compiles to its own small collective-only graph and
is dispatched separately, so there is no compute in the same graph for the compiler to overlap
it against. And the runtime's queue budget makes overlap structurally impossible on that path:
there is **one** compute queue and **one** collectives queue per logical core. Reaching the
collectives queue independently requires issuing the collective on a non-default stream, which
is the thing host CC actually enables and which this port does not do.

Measured directly on this image, a 32-rank all-reduce costs ~0.23 ms and that cost is flat
from 4 KiB to 2 MiB -- so these are alpha-bound, and ~150 of them is **~34 ms, about 21% of
the 160 ms step**. That makes collective count the largest identified term, and it is
attackable: the MoE mapping's `all_gather` + `all_reduce MAX` pair is already avoided at
decode by computing the mapping redundantly per rank, and the remaining per-layer pair is
what a decode megakernel or a hierarchical schedule would target.

Both are software-structural rather than hardware limits, and neither was visible from a
roofline. The honest summary is that **roughly half the 160 ms step is still
unattributed**: ~34 ms of collectives plus a modest weight-DMA share does not close it, and
the structural MoE-kernel fix that reasoning from source predicted would be large turned out
to be throughput-neutral. Meanwhile the two changes that actually moved the number by +20%
were a redundant reduction loop and an operator choice -- neither of which any amount of
kernel-library reading would have surfaced, because neither is in the kernel library. The
remaining gap needs a device profile, not another argument from first principles.

### Measuring it rather than arguing about it

For the record, since this cost real time: capturing a device profile on this stack needs
`NEURON_RT_INSPECT_DEVICE_PROFILE=session`. The value `1` selects a per-graph *model* mode
that, on this build, emits the compiled graphs and no trace file at all --- which reads
exactly like profiling being unsupported. It is not. Also worth knowing before you start:
profiling costs roughly 20-25% throughput, so the step time must come from an unprofiled
run and the profile used only for attribution; and with 32 workers there is no rank-scoped
variable, so the env has to be gated on local rank or you get 32 concurrent captures.

## Batch > 1 deadlocks in the lm-head, not the MoE

Five independent runs at `max_num_seqs` 8 sat for 5.5 hours each and produced nothing.
They were not slow; they were wedged. All five stopped at the identical log line:

```
WARNING vocab_sharding_spmd.py:316 ShardIndexInjection: conflicting shard indices {0..31}
INFO    backend.py:125            Detected collective operation: all_reduce.default
```

and then emitted only `shm_broadcast: No available shared memory broadcast block found in
60 seconds` once a minute, forever, with 32 workers spinning at ~11% CPU and **zero**
compiler processes alive.

That is a 32-rank compile barrier deadlock on a traced graph that contains a collective.
At batch 1 the graph either isn't produced or doesn't contain one; at batch>1 the
vocab-sharded lm-head puts an `all_reduce` inside the traced region and the ranks cannot
all get through. So the batch>1 blocker is in the **lm-head sharding**, not in the MoE,
which is where I had been looking.

Batch 32 fails differently and earlier -- it compiles clean and then cannot allocate
(`NRT EXECUTION FAILED: ... Failed to allocate resource` at prefill warmup), i.e. device
HBM exhaustion. Two distinct blockers, not one.

**A monitoring lesson that cost 27 machine-hours.** A phase classifier that reports
"tracing" on the absence of an error will call a deadlock healthy indefinitely. The signal
that separates them is `shm_broadcast` starvation **together with no live compiler
process** -- the conjunction matters, because starvation messages also appear transiently
during a perfectly healthy compile (observed: 4 starvation messages while 160 `neuronx-cc`
processes were running, on a run that went on to succeed).

## The failure that invalidated two rounds of results

Worth more than any number on this page.

The deploy step resolved the model source like this:

```bash
if [ -f "$SHARED/port/model.py" ]; then
  SRC="$SHARED/port"          # "it is the same content"
elif [ -f "$PAYLOAD/model.py" ]; then
  SRC="$PAYLOAD"
fi
```

The comment was true when written. It stopped being true the moment the payload was edited,
and nothing anywhere said so. Three launches shipped a modified `model.py` inside the command
payload, the shared copy stayed at an older timestamp, and all three ran the old code. Two
rounds of A/B conclusions were drawn from runs that were byte-identical to each other --
including a published "9% regression" that was simply variance.

What made it invisible is that every *other* signal looked right. The env var reached the
container. The launch payload contained the new file. The run completed, passed the golden
token, and produced a plausible number. The only way to notice was to ask what the deployed
file actually contained, which nothing prompted.

Three changes, all cheap, any one of which would have caught it:

- **The payload wins.** A launch-supplied artifact takes precedence over cached shared state,
  and a divergence is reported rather than silently resolved.
- **Fingerprint the thing that determines the result.** The log now prints the md5 and byte
  count of the deployed `model.py`, plus a grep count for the specific markers the run is
  supposed to be testing. A run whose result depends on a source file should say which source
  file, unprompted.
- **Echo the knobs after they are derived, not before.** The config line printed
  `V4_DECODE_MOE=unset` because it ran above the export that sets it -- so the one diagnostic
  that would have flagged this was itself misleading.

The same class of bug appeared twice more in this work: a defaults line that assigned `V4_*`
unconditionally and so ignored every value passed in from the launch (a `GEN=128` run
measured 32 tokens, silently), and a phase classifier that inferred "healthy" from the
absence of an error and so reported a five-hour deadlock as progress. All three share a
shape: **a silent fallback to a plausible default.** In an experiment harness, a wrong default
that runs is more expensive than a hard failure, because it costs the run *and* the
conclusion.


## Host collective-communication is not required at TP=32

An isolated 32-rank `all_reduce` probe on this image, no model:

| payload | host CC off | host CC on | host CC + `CC_STREAM_SPLIT=7:1` |
|---|---|---|---|
| 4 KiB | 0.235 ms | 0.284 ms | 0.284 ms |
| 64 KiB | 0.241 ms | 0.286 ms | 0.287 ms |
| 512 KiB | 0.230 ms | 0.299 ms | 0.296 ms |
| 2048 KiB | 0.225 ms | 0.316 ms | 0.311 ms |

Three things come out of this. The device path **works at TP=32 with host CC off**, so the
claim in an earlier revision of this page that host CC is what makes native tensor
parallelism work here is wrong -- the hang it originally "fixed" was environmental, most
likely orphaned ranks from a killed `torchrun` still holding the cores. Raising the
collective stream split does not recover host CC's cost, so that cost is not the resource
split. And latency is flat across a 512x payload range, i.e. these collectives are
alpha-bound, not bandwidth-bound, so reducing collective *count* is the lever and tuning
chunk size is not (a 256 KiB chunk measured slightly worse, and
`INTRA_LATENCY_OPT_MESH_ALG` is rejected outright by the runtime).

**End-to-end, turning it off measures ~5.7% faster** -- but only once the measurement was
good enough to see it. An earlier revision of this page claimed the opposite, that the full
model got *slower* with host CC off; that comparison had the two host-CC-off runs at 0.68
and 0.93 tok/s bracketing the on-run at 0.93, i.e. entirely inside variance. At ~34 ms of collectives in
a ~160 ms step the whole collective budget is ~21%, and at a 128-token generation the noise
floor drops to ~1.2% -- which is finally fine enough to see a host-CC effect. The first such
measurement puts host CC off at **+5.7%** (5.42 vs 5.19/5.07), agreeing in direction with
the isolated probe. One sample; repeats running.

The transferable point is about method rather than about this flag. An isolated
microbenchmark can rank a primitive cleanly; it cannot rank a whole-system configuration
whose effect is smaller than the system's measurement noise. Establish the noise floor
first, then decide what is worth A/B-ing at all.

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
