# DeepSeek-V4-Flash on Trainium2 — batched decode via vLLM-Neuron

Getting **DeepSeek-V4-Flash** (284B MoE, 43 layers) to *decode* on a single
`trn2.48xlarge`, and measuring it honestly.

Prefill (TTFT) on this model was already reachable. **Batched decode was not**, and
that is the part that actually determines serving throughput. This folder is the
field notes from making decode work: two hard ceilings that are easy to mistake for
bugs in your own code, and one architectural dead end that looked promising and
wasn't.

Everything below is **self-measured** on the public `pytorch-inference-vllm-neuronx`
container. Nothing is projected unless labelled `PROJECTION`.

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
| 43 | 32 | in progress | — | — |

Depth scaling is **slightly superlinear** — 3x the layers costs 3.4x the time
(~9.3 ms/layer fitted). That gives `PROJECTION: ~20 tok/s at 43 layers, batch 8`,
above a 15 tok/s target. It is an extrapolation from two points; the full-depth run
is the real number.

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

That caps TP at 8. On a `trn2.48xlarge` (32 logical cores under LNC=2) you then get
only 8 cores' worth of HBM — and **that is what made 24 layers die with
`nrt_tensor_allocate`** (HBM exhaustion), which in turn makes 43 layers impossible.
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

## Sub-projects

- **[`decode-static-shapes/`](decode-static-shapes/)** — making a decode step whose
  graph shape is *independent of sequence position*, so one compiled graph serves every
  token instead of recompiling per step. Includes a parity harness that proves the
  rewrite bit-exact against the reference over 130 decode steps.
- **[`fp4-expert-gemm/`](fp4-expert-gemm/)** — a negative result worth knowing before
  you spend a week on it: an FP4 expert GEMM that is numerically correct on device and
  **85x slower** than bf16, for an architectural reason.

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

---

## Honest status

- Decode **works and batches**, with real measurements at 4 and 12 layers.
- Full-depth (43-layer) throughput is **not yet measured**. The figure above is an
  extrapolation from two points.
- Correctness so far: the static-shape decode rewrite is bit-exact against the
  reference (130 steps), and the sharding fix agrees on first token across TP degrees.
  Full-depth golden-token validation is pending the 43-layer run.
- Neither ceiling is *solved*. Ceiling 1 is worked around by picking viable batch
  buckets; ceiling 2 is fixed for TP, but ceiling 1 still bounds how far batching goes.
