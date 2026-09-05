# Fusing DeepSeek-V4-Flash's hyper-connection boundary into one NKI kernel

DeepSeek-V4-Flash wraps every layer in **hyper-connections**: instead of one residual
stream it carries `hc_mult = 4` copies, mixes them down before attention and the FFN, and
expands them back out afterwards. That mixing (`hc_pre` / `hc_post`) runs **4 times per
layer**, so at 43 layers it is **172 boundaries per decoded token**.

Each boundary is cheap arithmetic spread over several small ops. That shape is exactly where
per-launch overhead, not math, sets the cost. This folder is the record of collapsing one
whole boundary into a **single NKI kernel**, one stage at a time, with a measurement at every
step.

Everything here is self-measured on a Trainium2 instance in native PyTorch mode (no XLA).
Every number below is observed. Nothing is projected.

**Headline:** the entire inter-block transition — `hc_post`, then all of `hc_pre`
(statistic, projection, 20-iteration sinkhorn, weighted combine), then RMSNorm — runs as
**one kernel**, bit-identical to the unfused path and validated against a float64 reference.
That transition happens **86 times per decoded token**, and the fused kernel is
**1.20–1.23x** faster than the two-kernel path it replaces.

---

## Why fusion and not micro-optimisation

An earlier attempt optimised a single kernel's *internals*. The routing kernel for this same
model was rewritten to cut its instruction count roughly 4x, using an ISA primitive that
returns the 8 largest values per partition in one instruction instead of a six-round
reduce/mark/suppress loop.

It bought about 7%:

```
T=1   v1 183.8us   v2 172.3us   (1.07x)
T=8   v1 184.9us   v2 171.8us   (1.08x)
```

and the giveaway is that latency was **flat** across an 8x change in problem size (−0.3%).
At 8 KB the kernel is launch-bound, so nothing you do *inside* it matters. The conclusion was
that the remaining win is amortising launches — i.e. fusion — and the rest of this folder
tests that claim.

**A caveat kept deliberately visible:** that kernel is ~17x faster than eager op-by-op torch,
and that number is not worth quoting. In the real model these ops sit inside a compiled graph
that already fuses some of them. An eager comparison is context, never an end-to-end claim.

---

## The boundary, and what depends on what

```
x_flat ──▶ statistic ──▶ projection ──▶ mixes ──▶ sinkhorn ──▶ pre ──▶ combine ──▶ RMSNorm
```

Written out:

```python
var    = x_flat.square().mean(-1, keepdim=True)      # over hc*D = 16384
rsqrt  = torch.rsqrt(var + eps)
mixes  = matmul(x_flat, hc_fn.t()) * rsqrt           # [P, 24]
pre, post, comb = split_sinkhorn(mixes, ...)         # 20 iterations on a 4x4
y      = sum_m pre[:, m] * x_flat.view(P, 4, D)[:, m, :]
out    = y * rsqrt(mean(y^2) + eps) * weight
```

Two facts about this graph drove every decision:

1. **Nothing external is consumed after `x_flat`.** Every step feeds only the next. So the
   entire chain is legally fusable into one kernel.
2. **The only real barriers are attention and the FFN.** Those are large separate kernels and
   nothing fuses across them. Everything *between* two blocks is fusable.

I got (2) wrong twice before getting it right, and both errors were of the same kind —
mistaking *where a stage belongs* for *what it depends on*:

* First I wrote that the sinkhorn was a hard barrier splitting every boundary in two. It
  splits *across* boundaries, not *within* one. Noticing that produced increment 5.
* Then I wrote that `hc_post` could not join, because it *produces* the tensor `hc_pre`
  consumes. But the layer's forward pass is `hc_post` → (residual carry) → `hc_pre` → norm,
  with nothing external in between, so it fuses forward perfectly well. That produced
  increment 7, which is the biggest win in this folder.

The reachable unit is therefore the whole transition between two blocks:

```
attn/ffn out ──▶ hc_post ──▶ hidden ──▶ hc_pre ──▶ RMSNorm ──▶ next block
              \________________ one kernel ___________________/
```

---

## Results

Trainium2, native PyTorch (no XLA), fp32, `hc_mult=4`, `D=4096`, `K=hc*D=16384`,
`mix_hc=24`, 20 sinkhorn iterations. `P = batch x seq`, so decode is P=1..8.

Every timing pulls its result to host **inside** the timed region under
`torch.inference_mode()`. Neuron dispatch is asynchronous, so a loop that skips this measures
enqueue cost and reports impossible speedups.

| increment | stages in the kernel | P=1 | P=8 | vs unfused |
|---|---|---|---|---|
| 2 | combine + RMSNorm | 238.3 us | 243.4 us | **1.21x** / **1.34x** |
| 3 | + sinkhorn | 372.7 us | 380.7 us | **1.40x** / **1.45x** |
| 5 | + statistic + projection (**all of `hc_pre`**) | 443.0 +- 6.6 us | 438.7 +- 5.0 us | **1.154 +- 0.044x** / **1.177 +- 0.015x** |
| 6 | `hc_post` on its own (the expand side) | 254.2 us | 313.1 us | — (baseline for 7) |
| 7 | **`hc_post` + `hc_pre` + RMSNorm** | 516.8 +- 12.3 us | 528.4 +- 10.7 us | **1.201 +- 0.032x** / **1.229 +- 0.051x** |

Increment 5 is reported with error bars because a single run of it is not trustworthy — see
"the measurement that nearly fooled me" below. Those figures are 7 repetitions of 20 iterations,
with the fused and unfused variants **interleaved within each repetition** so drift hits both
arms of the comparison equally. The fused kernel was faster in **7 of 7** repetitions at both
batch sizes.

Every increment is **bit-identical** to the multi-kernel path it replaces (max abs difference
exactly 0), and every one is checked against a float64 reference.

### Correctness

| increment | error vs float64 | margin below bf16 resolution |
|---|---|---|
| 2 | rel 4.8e-06 / 6.8e-06 | 811x / 574x |
| 3 | rel 6.2e-07 / 2.7e-05 | 6286x / 146x |
| 4 | rel 3.3e-05 / 3.7e-05 | 120x / 106x |
| 5 | rel 2.1e-06 / 1.5e-05 | 1849x / 254x |

The sinkhorn is supposed to produce a doubly-stochastic matrix, and it does: `comb` row sums
**and** column sums both come out 1.0000.

### The measurement that nearly fooled me

I first ran increment 5 once and got P=1 443.6 us, P=8 473.2 us — a +6.7% growth with problem
size — and concluded the kernel had crossed out of the launch-bound regime into being
compute-sensitive. That would have been a satisfying result: it is what absorbing a
K=16384 contraction *ought* to do.

Then a re-run gave P=1 504.6 us, P=8 487.9 us, i.e. **−3.3%**. The verdict had inverted. The
P=1 figure alone had moved 13.7% between runs, which is larger than the effect either run was
trying to detect.

Measured properly — 7 repetitions, interleaved:

```
P=1   443.0 +- 6.6 us
P=8   438.7 +- 5.0 us
difference  -1.0%,  pooled noise  +-1.9%
```

**Verdict: not resolvable at this precision.** No claim either way. All three increments remain
consistent with being launch-bound, and the earlier "regime change" conclusion was an artifact
of trusting one sample.

The paired ratio survives this scrutiny where the scaling claim does not, and the reason is
structural: the ratio compares two variants measured microseconds apart under identical
conditions, whereas the scaling comparison spans separately-timed blocks and so absorbs every
source of drift. If you take one thing from this folder, take that — **compare paired, and
repeat before concluding.**

### Reading the ratio honestly

The fused-vs-unfused ratio *drops* from 1.40x (increment 3) to ~1.16x (increment 5). That is
not a regression:

* the **baseline changed** — increment 3 was measured against three kernels, increment 5
  against two;
* the **denominator grew** — the projection is now the bulk of the kernel, so a fixed saving is
  a smaller share of a bigger total.

In absolute terms the fusion saves ~70-77 us per boundary (511 → 443 at P=1, 516 → 439 at P=8).
What that does **not** license is multiplying by 172 boundaries and quoting a per-token figure,
for the compiled-graph reason above.

---

## Things that cost me time, written down

**A 340x disagreement with torch that is not a bug.** `torch.mean` reduces **pairwise**
(error ~ log2(N)·eps); the hardware's scalar engine reduces **sequentially** (~N·eps). At
D=4096 the kernel is *expected* to differ from torch by roughly N/log2(N) ≈ 340x. I wrote a
float64 check that flagged this as "genuinely wrong" — the check was wrong, not the kernel.
The right gate is the sequential-fp32 bound, cross-checked against bf16 resolution, since
these activations are bf16 in the real model. Simply loosening the tolerance would have hidden
a real bug later.

**Fused activation+reduce is a trap for softmax.** The scalar engine can apply an activation
and reduce its result in one instruction, which gives `sum(x^2)` for free — genuinely 1
instruction instead of 2, and used for every RMS statistic here. But it discards the pointwise
intermediate, so anything needing *both* `f(x)` and `sum(f(x))` has to recompute. Softmax needs
both. Used there it measures ~19% **slower**, so the sinkhorn's row-softmax deliberately keeps
a separate reduce.

**I predicted the wrong transpose path.** The projection contracts over K=16384, and
`nc_matmul` only ever contracts over the partition axis, so `x` must be transposed in 128-wide
chunks. I expected transposing during the DMA to win, since it moves work off the tensor engine
and skips a PSUM round-trip. Measured, it loses:

```
P=1   nc_transpose 287.5us   vs   dma_transpose 514.8us   (1.79x)
P=8   nc_transpose 273.4us   vs   dma_transpose 300.1us   (1.10x)
```

128 small DMA transposes cost more than 128 tensor-engine transposes plus their PSUM copies,
and it is worst at P=1 where each DMA moves 128x1 elements — per-descriptor overhead dominates.
Both are bit-identical, so it is purely a scheduling choice. Both paths are kept, `nc_transpose`
is the default.

**Sinkhorn without a single transpose.** A sinkhorn alternates row and column normalisation,
and the column direction looks like it needs a transpose. Keeping `comb` **flat** as
`[P, hc*hc]` row-major means column *j* lives at flat positions *j*, *hc+j*, *2hc+j*, … so all
`hc` column sums fall out of `hc` strided elementwise adds on `[P, hc]` slices. The CPU
reference needs a stack of `.transpose().contiguous()` calls to express the same thing.

**Two API constraints worth knowing.** Both `nc_matmul` operands must be in SBUF
(`moving must be in [sbuf], got shared_hbm`), and since the weight is `[16384, mix_hc]` it
exceeds the 128-partition limit and has to be streamed one contraction chunk at a time.
Separately, putting the **activation** in the stationary slot rather than the weight makes the
result come out as `[P, mix_hc]` directly, avoiding a final transpose.

**Partition-dim broadcast only works from HBM.** Broadcasting the partition dimension of an
on-chip tensor is rejected (`Cannot broadcast partition dim (dim 0) for on-chip tensors`);
broadcasting the HBM source during the DMA is allowed, and is fewer instructions than a
shuffle-group loop.

**PSUM tiles get allocated once, outside the chunk loop.** Allocating per iteration is a good
way to exhaust the allocator. The error it produces is the compiler's general allocator error,
which is the *same code* reported by an unrelated activation-table ceiling — so the code alone
does not tell you which problem you have.

---

## Files

| file | what it is |
|---|---|
| `nki_hc_combine_norm.py` | increment 2 — combine + RMSNorm, plus the unfused kernels used as the A/B baseline |
| `nki_hc_sinkhorn_fused.py` | increment 3 — sinkhorn + combine + RMSNorm |
| `nki_hc_matmul_head.py` | increment 4 — RMS statistic + projection (tensor engine), both transpose paths |
| `nki_hc_pre_full.py` | increment 5 — all of `hc_pre` + RMSNorm in one kernel |
| `nki_hc_post.py` | increment 6 — `hc_post`, the expand side |
| `nki_hc_block_transition.py` | increment 7 — `hc_post` + `hc_pre` + RMSNorm, the full transition |
| `test_*.py` | per-increment validation and A/B benchmarks |
| `repeat_hc_pre_full.py` | repeatability harness — interleaved, repeated, reports mean +- stdev |
| `sim_gate_hc.py` | simulator gate, run at both logical-core configurations |

Run a test directly on a Trainium2 instance:

```bash
python3 test_hc_pre_full.py
```

The simulator gate is run once per logical-core setting, because the configuration is read at
import time:

```bash
NEURON_LOGICAL_NC_CONFIG=1 python3 sim_gate_hc.py
NEURON_LOGICAL_NC_CONFIG=2 python3 sim_gate_hc.py
```

Both pass with **identical** numbers, which is the point — it means no stage depends on a
particular logical-core pairing.

---

## What is not claimed

* **No end-to-end model speedup.** These kernels are validated standalone. They are not yet
  wired into the model's forward pass, so there is no token/s number attributable to them here.
* **No multi-device result.** Every measurement is single-core. A collective at the real serving
  world size, with per-rank sharded weights, is a separate gate that these runs do not satisfy —
  and passing simulator and small-scale tests is not a substitute for it.
* **The whole hyper-connection path is now NKI, but only as kernels.** Every stage
  (`hc_post`, statistic, projection, sinkhorn, combine, norm) has a validated kernel, and the
  full transition is fused into one. None of it is wired into the model yet.
