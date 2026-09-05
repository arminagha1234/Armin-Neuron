# FP4 expert GEMM on Trainium2 — a correct kernel that is 85x too slow

A negative result, published because it is expensive to rediscover.

**Idea.** V4-Flash decode is weight-DMA-bound, and the expert weights are FP4. So
instead of dequantising weights to bf16 and doing a normal GEMM, write a NKI kernel
that streams the *packed FP4* weights and dequantises inside SBUF. That reads
**3.76x fewer weight bytes** (7.80 MB vs 29.36 MB per expert GEMM), which on a
bandwidth-bound step should be close to a 3.76x win.

**Outcome.** The kernel is numerically correct on device (cosine `0.999999` at every
shape tested) and **85x slower** than a bf16 `F.linear`. The reason is architectural,
not a tuning miss.

---

## Measured

Real V4-Flash expert dims, `trn2` (LNC=2):

| M | K | N | NKI FP4 | bf16 `F.linear` | ratio | cosine |
|---|---|---|---|---|---|---|
| 128 | 256 | 256 | 0.417 ms | 0.127 ms | 0.30x | 0.999999 |
| 1 | 7168 | 2048 | 15.628 ms | 0.179 ms | 0.01x | 0.999999 |
| 8 | 7168 | 2048 | 15.633 ms | 0.178 ms | 0.01x | 0.999999 |
| 32 | 7168 | 2048 | 16.143 ms | 0.187 ms | 0.01x | 0.999999 |
| 128 | 7168 | 2048 | 16.823 ms | 0.230 ms | 0.01x | 0.999999 |

Latency is **flat in M** — the tell that cost tracks the *weights*, not the batch.

## The profile says it all

`neuron-explorer`, M=1 K=7168 N=2048:

| metric | value |
|---|---|
| total time | 15.39 ms (matches wall clock) |
| VectorE busy | **88.7%** (159,315 instructions) |
| ScalarE busy | **92.8%** (76,264 instructions) |
| TensorE busy | 3.3% |
| transpose FLOPs / total FLOPs | **99.2%** |
| HBM read | 8.03 MB |
| memory-bandwidth utilisation | **0.0007%** |
| DMA packets | 258,048, averaging **33 bytes** |
| SBUF read / write | 2.24 GB / 1.78 GB (for an 8 MB problem) |
| NEFF size | 8.76 MB — comparable to the weight data itself |

So it is not bandwidth-bound at all. It is bound by **vector/scalar instruction count**
doing the dequantisation, and almost all tensor-engine work is layout shuffling.

## Root cause — no hardware MX matmul on this generation

The NKI API on a `trn2` install exposes `nc_matmul_mx` (native MXFP8/MXFP4 matmul with
integrated dequantisation) and a packed `float4_e2m1fn_x4` dtype. They **trace without
complaint**. On device:

```
[NCC_IBIR530] MatmultMx is not supported
```

Native MX matmul is a **next-generation** feature. On this generation, FP4 weights must
be dequantised in *software* on the vector/scalar engines. That work is proportional to
the **weight element count** (14.7M values per call here) and is paid every call — about
0.8 ms of unavoidable vector work, against a 0.179 ms bf16 GEMM.

> An API being importable is **not** evidence of hardware support. Check availability on
> the target generation *before* designing a quantised-weight kernel.

Better tiling would recover maybe 10-50x of the 85x, but the floor is the dequantisation
itself, and the floor still loses. So: **do not ship this on this generation.** The same
kernel design collapses to a single `nc_matmul_mx` on hardware that has it, and becomes
the right answer there.

---

## Two anti-patterns worth naming

Both were in the first version and both are easy to write by accident.

**Tiny strided DMAs in the inner loop.** A loop nest of 16 x 56 x 4 = 3584 iterations,
each DMA-ing a `[128, 16]` uint8 weight slice (**16 bytes per partition row**) and a
`[128, 1]` scale (**1 byte per row**), produced 258,048 DMA packets averaging 33 bytes
for 8 MB of real data. Rule of thumb: a DMA under ~512 bytes per partition row, or a
vector op on a free dim under ~512 elements, is a red flag inside an inner loop. Hoist
the transfer and dequantise over a wide free dimension.

**Per-tile weight transposes.** Dequantising as `[N, K]` then transposing to `[K, N]`
per K-tile so the contraction lands on the partition dim burned **99.2% of all device
FLOPs** on data movement. Pre-pack the quantised weight along the other axis at load
time instead. `transpose_flops / hardware_flops` is a good first-order health metric for
any dequantise-then-matmul kernel.

---

## Debugging notes that saved the day

**The dtype gate that hides behind an opaque error.** The kernel first failed to compile
with only `COMPILATION FAILED` on the custom call. The real error:

```
[NCC_EVRF051] Data type F8E4M3FN is not supported on TRN1/TRN2
```

Any `float8_e4m3fn` tensor you pass in puts that dtype in the **module signature** and
the verifier rejects it — even though NKI tracing succeeded and the kernel binary was
already cached. Note the `UNSAFE_FP8FNCAST` escape hatch only silences the *tracing*
complaint; it never reaches the compiler.

The clean fix is to keep the dtype off the boundary: pass the raw quantised activation as
**bf16**. `e4m3 -> bf16` is **lossless**, so this is bit-identical, and it is free when
the kernel already casts on the DMA — in which case FP8 was only ever an HBM container
and buys nothing at decode, where activations are tiny next to weights. `float8_e4m3`
(without `fn`) *is* supported and can be used freely *inside* a kernel; the verifier only
inspects the signature.

**Getting the real compiler error at all.** The artifact directory named in the error is
deleted during compile teardown, *before* Python raises — so copying it after the run
gets nothing, or a log truncated mid-write. Start a **background poller** before the run
that snapshots the directory every ~0.1 s. Even then the retained log may stop at the
failing pass, but it contains the exact sub-tool command line: **replay that by hand with
`--verbose=debug`** to get the complete message. Do not reach for `--save-temps` /
`--dump-dir`; they are rejected as unknown arguments, and because they fail as an
*argument* error they mask the error you were trying to capture.

**Passing pass-level flags.** Error text sometimes names a flag the top-level compiler
driver does not accept. Use the documented passthrough
(`--internal-hlo2tensorizer-options=<flags>`) rather than guessing spellings; a
`--internal-`-prefixed spelling may parse and then be silently ignored, which looks
exactly like "the flag doesn't work".

**Profiling.** `neuron-profile` is gone — use `neuron-explorer capture` / `view
--output-format=summary-json`. Kernel NEFFs are not left in the working directory; they
are cached, and a run that *hits* the cache never invokes the compiler at all (so an
artifact poller catches nothing and you wrongly conclude capture failed). Perturb a shape
to force a miss.

**Benchmarking.** Device execution is asynchronous: ops return before the device
finishes. Force completion inside the timed loop by pulling the result to host, or you
will measure enqueue cost and report an absurd speedup. Two tells that you are timing
dispatch: latency nearly flat as batch grows by orders of magnitude, and implied
throughput above the chip's FLOP ceiling.
