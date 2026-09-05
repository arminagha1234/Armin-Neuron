# DeepSeek-V4-Flash on Trainium2 — batched decode via vLLM-Neuron

Getting **DeepSeek-V4-Flash** (284B MoE, 43 layers) to *decode* on a single
`trn2.48xlarge`, and measuring it honestly.

Prefill (TTFT) on this model was already reachable. **Batched decode was not**, and
that is the part that actually determines serving throughput. This folder is the
field notes from making decode work: four ceilings that are easy to mistake for bugs in
your own code, and one architectural dead end that looked promising and wasn't.

Everything below is **self-measured** on the public `pytorch-inference-vllm-neuronx`
container. Every number is an observed measurement — there are no projections left in
this document. Where an earlier extrapolation existed it is kept only to compare
against the value that replaced it.

**Headline:** full 43-layer batched decode at **22.25 tok/s** (TP=32, batch 8), with the
greedy first token matching the golden argmax. Four distinct ceilings had to be cleared
to get there, and each is written up below with the error it produces, because all four
are easy to mistake for bugs in your own code.

---

## Why decode is the whole problem

V4-Flash decode is **weight-DMA-bound**: each step streams expert weights out of HBM
and does almost no math per token. So:

- at batch 1 you pay a full weight read for a single token — hopeless for throughput
- batching amortises that same read across many tokens

Measured, that is a **6.6x** difference between batch 1 and batch 8. Decode work that
doesn't enable batching is not throughput work.

A useful corollary: because cost tracks the *weights*, decode step time barely moves
as you grow the batch until you saturate bandwidth. If your step time is nearly flat
in batch size, you are bandwidth-bound, not compute-bound — which tells you
immediately which optimisations are pointless.

---

## Measured results

`trn2.48xlarge`, LNC=2, 512-token prompt, `GEN=32`, greedy, bf16 weights.
Decode tok/s is prefill-excluded.

### Batch scaling (4 layers, TP=8)

| batch | result | decode tok/s | TTFT | load+compile |
|---|---|---|---|---|
| 1 | pass | **38.91** | 38.9 ms | 378 s |
| 2 | **compile fail** | — | — | — |
| 4 | **compile fail** | — | — | — |
| 8 | pass | **258.00** | 280.6 ms | 383 s |
| 16 | **compile fail** | — | — | — |
| 32 | **compile fail** | — | — | — |

Batch 1 reproduced at `GEN=8` (39.14) and `GEN=32` (38.91), so the figure is stable.

### Depth scaling (batch 8)

| layers | TP | result | decode tok/s | per-step |
|---|---|---|---|---|
| 4 | 8 | pass | 258.00 | 31.0 ms |
| 12 | 8 | pass | 75.80 | 105.5 ms |
| 12 | 16 | pass | **82.54** | — |
| 24 | 8 | **HBM OOM** | — | — |
| 43 | 32 | pass | **22.25** | 449.7 ms |

Depth scaling is **slightly superlinear** — 3x the layers costs 3.4x the time
(~9.3 ms/layer fitted). Extrapolating from 4 and 12 layers predicted ~20 tok/s at full
depth; the measured value is **22.25**, so the extrapolation was ~11% pessimistic.

### Full depth (43 layers, TP=32, batch 8)

```
ttft = 6921.5 ms    t_gen = 18.067 s    load+compile = 3020.9 s
decode      = 22.25 tok/s   (prefill-excluded, steady state)
aggregate   = 12.62 tok/s   (includes prefill, GEN=32)
first token = 4256          golden argmax MATCHED
```

Both figures are reported deliberately. `22.25` is the steady-state decode rate, which
is what serving throughput tracks. `12.62` folds the 6.9 s prefill into only 32
generated tokens; it converges toward the decode rate as generation lengthens. Quoting
only the aggregate understates decode, and quoting only decode ignores that prefill is
real work — so both are here.

**The correctness gate.** Every shallower number above comes from a truncated model
whose output distribution is degenerate, so greedy argmax flips under floating-point
noise and cannot be validated. At full depth the first generated token is `4256`,
matching the golden argmax on every sequence in the batch. That is the only correctness
claim in this folder that covers the whole network.

---

## Ceiling 1 — the compiler's activation-table limit

Every compile failure above is the same error:

```
[NCC_INLA001] Instruction LoadActFuncSet I-4926-0-PWP:
              the number of activation tables must be <= 8
```

The Scalar Engine holds at most **8 activation-function tables per instruction set**.
V4-Flash decode uses many distinct activation functions — softmax, a `sqrtsoftplus`
router, clamped SwiGLU, `rsqrt`, an `exp` for power-of-two scale decode,
Hadamard-related transforms — and at some shapes the compiler fuses more than 8 into
one set.

**The important detail: it is not monotonic in batch size.** Batch 1 passes, 2 and 4
fail, 8 passes, 16 and 32 fail. It is driven by *how the compiler fuses activations at
a given shape*, not by how big the batch is. Practical consequence:

> Viable decode batch buckets are a **discrete set that must be discovered
> empirically**. Do not assume that if 8 works, 16 will.

Mitigations, in increasing order of effort:

1. Reduce the number of *distinct* activation functions on the decode path. Several
   are expressible with functions already present, or as plain arithmetic — a
   power-of-two scale decode can be a multiply instead of an `exp`.
2. Break the fusion so tables load across multiple instruction sets.
3. Move activations into **NKI** kernels, which bypass the activation-table path
   entirely. This is the only mitigation that attacks the cause rather than working
   around it.

---

## Ceiling 2 — a grouped output projection capping tensor parallelism

The model uses a **grouped low-rank output projection** (8 groups). The natural
sharding gives each rank whole groups:

```python
n_local_groups = o_groups // world_size      # 8 // 16 == 0  -> ZeroDivisionError
```

That caps TP at 8. A `trn2.48xlarge` under LNC=2 exposes **64 logical cores** (16
devices x 4), so capping at 8 leaves you a small fraction of total HBM — and **that is
what made 24 layers die with `nrt_tensor_allocate`** (HBM exhaustion), which in turn
makes 43 layers impossible.
So it looks like an efficiency nit and is actually a does-it-fit blocker.

### The fix, and why it is valid

Let several ranks **share** one group. A group's latent is a *sum* over that group's
heads:

```
latent_g = sum_h ( o_h @ wo_a_h )
```

If ranks A and B each own half of group `g`'s heads, each produces a *partial* latent.
Because the second projection `wo_b` is linear:

```
wo_b(latent_A) + wo_b(latent_B) == wo_b(latent_A + latent_B)
```

and the output path already ends in an all-reduce across all ranks, those partials sum
correctly **for free**. No new collective. Only the weight slicing changes:

```python
n_local_groups  = max(1, o_groups // world_size)
ranks_per_group = max(1, world_size // o_groups)
```

- first projection: index by **group** (`rank // ranks_per_group`), and slice the input
  dim by the within-group head range (`rank % ranks_per_group`)
- second projection: index by **group** as well, so ranks sharing a group load the
  **same** rows and their partial outputs sum in the existing all-reduce

Indexing the second projection by *rank* instead is a subtle trap: at TP=16, rank 15
asks for row 15360 of an 8192-row axis and reads off the end.

**Measured:** TP=16 works and is slightly faster than TP=8 (82.54 vs 75.80 tok/s at 12
layers, batch 8). Correctness signal: the first generated token is **identical** at
TP=8 and TP=16, and that token comes from the prefill path, which exercises the same
projection weights.

---

## Ceiling 3 — the KV pool crowds out collective buffers

Reaching full depth surfaced a failure that looks like a model-too-big problem and
isn't:

```
Could not load the model status=4 message=Allocation Failure
NRT:nrt_load_collectives   Failed to load collectives for model.
```

Weights loaded fine — all 43 layers. It died allocating **collective buffers**. The
cause is that vLLM sizes the KV cache to fill whatever device memory is free, and at
full depth there is very little slack left after weights:

| what | tokens of KV |
|---|---|
| auto-sized (1464 blocks x 32) | 45,768 |
| actually needed (8 seqs x 552) | **4,416** |
| capped (256 blocks) | 7,437 |

Roughly **10x more KV than the workload uses**, and the excess left nothing for the
collectives. Capping the block count fixes it and costs nothing, because the pool was
never being used:

```python
num_gpu_blocks_override = 256      # blocks, not tokens; block_size=32 here
```

**One trap worth naming.** The obvious companion knob is
`gpu_memory_utilization`, and lowering it makes things *worse*:

```
Computed KV cache budget is below minimum threshold.
effective=0.00 GiB, minimum=1.00 GiB
```

At 0.6 the budget computed to **zero**, because weights alone already exceed 60% of
device memory. The two knobs act at different stages — utilization gates a *budget
check* that runs first, while the block override caps the *actual allocation* later.
So the working combination is **default (high) utilization together with an explicit
block cap**: the check passes, and the pool still stays small.

---

## Ceiling 4 — a cold compile at high TP dies at the collective barrier

At TP=32, a cold compile reliably fails near the end:

```
RuntimeError: Operation timed out!
[gloo/transport/tcp/pair.cc] Connection closed by peer
RuntimeError: Engine core initialization failed
```

It got to 64 of 65 graphs before dying — over two hours of compilation, discarded.
The mechanism is compile-time skew: individual graphs take 350–530 s and ranks do not
finish together, so a rank that arrives early at a collective barrier waits past the
deadline while its peers are still compiling.

There is no timeout knob that fixes this (the engine-startup wait is not
env-tunable in this version). What works is accepting it and **treating bring-up as
two passes**:

> **Pass 1** compiles and is *expected* to fail at the barrier.
> **Pass 2** starts with a warm compile cache, every rank skips compilation, all ranks
> reach the barrier together, and the run completes.

Both full-depth successes here followed exactly that pattern. Once you know it, it is
a 30-second retry rather than a two-hour mystery. Worth scripting the retry.

---

## A measurement footnote that changes the numbers

The 43-layer figure above is at TP=32, and TP=32 does **not** use the whole machine:

```
distinct processes holding Neuron devices: 32
devices with processes:  0 1 2 3 4 5 6 7
devices 8-15:            idle
```

Four ranks land on each of the first eight devices, so only **half the devices** — and
therefore roughly half the aggregate HBM bandwidth — are in play. Since decode is
weight-DMA-bound, that matters directly: **22.25 tok/s is a half-machine number.**

It also reframes Ceiling 3. The `Allocation Failure` was not really "the KV pool is
greedy"; it was "we are fitting a ~568 GB model into the ~768 GB belonging to eight
devices instead of the full complement." The block cap is a valid fix, but raising TP
attacks the cause.

Worth stating plainly rather than burying: **checking which devices your ranks
actually occupy is a two-second command, and not doing it cost several hours of
misdiagnosis here.**

---

## Sub-projects

- **[`decode-static-shapes/`](decode-static-shapes/)** — making a decode step whose
  graph shape is *independent of sequence position*, so one compiled graph serves every
  token instead of recompiling per step. Includes a parity harness that proves the
  rewrite bit-exact against the reference over 130 decode steps.
- **[`fp4-expert-gemm/`](fp4-expert-gemm/)** — a negative result worth knowing before
  you spend a week on it: an FP4 expert GEMM that is numerically correct on device and
  **85x slower** than bf16, for an architectural reason.

- **[`hyper-connection-fusion/`](hyper-connection-fusion/)** — collapsing the model's
  hyper-connection boundary (which runs **172 times per decoded token**) from several
  small ops into **one NKI kernel**, one stage at a time, measuring at every step. Ends
  with all of `hc_pre` plus its RMSNorm in a single kernel, bit-identical to the unfused
  path. Also contains a worked example of a benchmarking mistake: a single-sample
  measurement that produced a confident conclusion the repeated measurement then
  retracted.

---

## Reproduction notes

Public container:

```
public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.21.0.1.0.0-neuronx-py313-sdk2.31.0-ubuntu24.04
```

Two things that will cost you time:

**The checkpoint ships a quantisation config the serving stack refuses.** Build a
shadow checkpoint directory — symlink every file, then write a `config.json` with
`quantization_config` removed. Weights dequantise to bf16 at load.

**Killed runs leak Neuron cores.** vLLM renames its worker processes, so a `pkill`
pattern matching your launcher script name does *not* match them. They survive, hold
every core, and the next launch dies with `The PyTorch Neuron Runtime could not be
initialized`. Kill workers by their actual process name, and confirm with `neuron-ls`
that the PID column is empty before relaunching.

**Cap the KV pool, and leave `gpu_memory_utilization` alone.** See Ceiling 3 — the
default pool is ~10x oversized at full depth and starves the collectives, but lowering
utilization drives the computed budget to zero.

**Expect the first high-TP compile to fail.** See Ceiling 4 — budget two passes, and
script the retry so a warm-cache relaunch is automatic.

**Confirm your rank-to-device placement.** `neuron-ls` shows which devices hold
processes. A TP degree below the logical-core count can silently leave half the devices
idle, which halves both capacity and bandwidth.

---

## Honest status

- Decode **works and batches**, measured at 4, 12 and **43 layers**.
- Full depth: **22.25 tok/s** decode at TP=32, batch 8 — and the golden argmax matches,
  so this is a validated full-network result, not just a throughput figure.
- That number is a **half-machine** result (TP=32 occupies 8 of 16 devices). TP=64 is
  the obvious next step and is untested here; expect the memory pressure behind
  Ceiling 3 to ease, and the barrier skew behind Ceiling 4 to get worse with 2x the
  ranks.
- Correctness: the static-shape decode rewrite is bit-exact against the reference over
  130 steps; the sharding fix agrees on first token across TP degrees; and full-depth
  greedy decode reproduces the golden first token.
- **No ceiling here is *solved*.** Ceiling 1 is worked around by discovering viable
  batch buckets empirically. Ceiling 2 is fixed. Ceilings 3 and 4 are worked around
  with a block cap and a retry. Only Ceiling 1 has a real fix available — moving
  activations into NKI kernels — and that work has now **started** but is not finished:
  see [`hyper-connection-fusion/`](hyper-connection-fusion/), where one whole
  hyper-connection boundary is now a single validated kernel. Those kernels are
  validated **standalone** and are not yet wired into the model's forward pass, so no
  ceiling has moved and no token/s figure here changes because of them.
- Expert parallelism is **off** (`ep_degree=1`), so all experts sit on every rank with
  only the intermediate dimension sharded. At TP=32 that is 2048/32 = **64 columns per
  rank**, a very thin GEMM. Enabling EP would widen it substantially and is the most
  promising untested lever after TP.
