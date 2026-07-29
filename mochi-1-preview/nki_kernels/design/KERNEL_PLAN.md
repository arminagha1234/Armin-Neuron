# Mochi-1 on Trn2 — NKI Kernel Plan

Design-only ranking document. No kernel code here — this decides *what* to
build, *why*, and *in what order*, and calls out where `torch.compile` almost
certainly wins so we don't burn weeks hand-writing a kernel the compiler would
match.

## Baseline being replaced

- Port is native PyTorch (TorchNeuron), **eager**, TP=4, bf16, ~6.3 s/step at
  31 frames / 64 steps with CFG. `torch.compile` is **untried** — so every
  number below is measured against an eager BMM baseline, and the honest
  comparison for each kernel is "NKI vs *compiled* eager", not "NKI vs today".
- All attention today goes through `neuron_compat._attention_bmm`: explicit
  `bmm → (+mask) → softmax → bmm`, query-tiled to cap the score tensor at
  ~256 MiB. Exact, but it round-trips `scores` and `probs` through HBM and
  launches a pile of ops per block.
- FFN, norms, QKV projections, RoPE apply are all stock diffusers / plain
  PyTorch, sharded by the TP plan.

## Fixed geometry (TP=4, the recommended config)

Derived from `mochi_tp_plan.py` constants and the NOTES.md token tables.

Global: 48 blocks, 24 heads × 128 head_dim, inner_dim 3072, text dim 1536,
ff inner 8192, ff_context inner 4096. TP=4 → **6 local heads**, local inner
768, local ff inner 2048, local ff_context inner 1024.

Batch: **B=1 no-CFG, B=2 with CFG** (CFG is the validated path). Attention
works on `batch*local_heads` planes → **12 planes** at B=2, TP=4.

Token counts (visual + 256 text = joint Sk), per NOTES.md:

| Frames | Latent f | Visual tok | Joint S (=Sq=Sk) | scores/plane bf16 | 12-plane scores |
|---:|---:|---:|---:|---:|---:|
| 19 (diffusers default) | 4 | 6,360 | 6,616 | 87.5 MB | 1.05 GB |
| **31 (validated)** | 6 | 9,540 | **9,796** | 191.8 MB | 2.30 GB |
| 61 | 11 | 17,490 | 17,746 | 629 MB | 7.55 GB |
| 85 | 15 | 23,850 | 24,106 | 1.16 GB | 13.9 GB |
| 163 (model card) | 28 | 44,520 | 44,776 | 4.01 GB | 48.1 GB |

Per-block linear shapes at TP=4 (what a kernel sees locally):

| Op | Global | Local (TP=4) | dtype |
|---|---|---|---|
| `attn1.to_{q,k,v}` | 3072→3072 | 3072→768 (colwise) | bf16 |
| `attn1.add_{q,k,v}_proj` | 1536→3072 | 1536→768 (colwise) | bf16 |
| `attn1.to_out.0` | 3072→3072 | 768→3072 (rowwise, +AR) | bf16, has bias |
| `ff.net.0.proj` (fused SwiGLU) | 3072→16384 | 3072→4096 = [2048 val \| 2048 gate] | bf16 |
| `ff.net.2` (down) | 8192→3072 | 2048→3072 (rowwise, +AR) | bf16 |
| `ff_context.net.0.proj` | 1536→8192 | 1536→2048 = [1024\|1024] | bf16 (absent blk 47) |
| `ff_context.net.2` | 4096→1536 | 1024→1536 (rowwise) | bf16 (absent blk 47) |
| `norm1.linear` (AdaLN) | 3072→12288 | replicated 3072→12288 | bf16, has bias |
| `norm{1..4}` RMS | over 3072 | replicated, weight-free RMS + fp32 | in bf16 / calc fp32 |

QKV output tensors (31f, per stream, before joint cat), layout `(B, H, S, D)`:
- visual q/k/v: `(2, 6, 9540, 128)` bf16
- text q/k/v: `(2, 6, 256, 128)` bf16
- joint q/k/v after cat: `(2, 6, 9796, 128)` bf16
- additive key bias: `(B, 1, 1, 9796)` bf16 — 0 on visual keys, 0/-10000 on text keys

Head_dim = 128 lands **exactly** on the 128×128 tensor-engine partition
dimension. That is the single most important fact for the attention kernel:
the QK^T contraction dim is a full, unpadded 128.

---

## Ranked candidates

### #1 — Flash non-causal joint attention with additive key-bias  ⭐ BUILD NOW

**Replaces:** `neuron_compat._attention_bmm` + `neuron_compat.neuron_sdpa`
(the whole `bmm→softmax→bmm` path, lines ~191–281), invoked from
`mochi_neuron_attention.MochiNeuronAttnProcessor.__call__` line 164
(`F.scaled_dot_product_attention`). Also subsumes the auto-tiling machinery
(`_resolve_q_chunk`, `_attention_bmm` chunk loop) — flash makes it obsolete.

**Shapes/dtypes (31f, TP=4, CFG):** per plane `q,k,v = (9796, 128)` bf16,
12 planes (B=2 × 6 heads). Non-causal, full joint `[visual | text]` key axis.
Additive bias vector length 9796 per batch (broadcast over heads and query
rows), bf16, values in {0, −10000}. RoPE already applied to visual q/k
(see #5 for folding it in). Softmax accumulation in fp32. Output `(9796, 128)`
bf16 per plane.

**Why NKI beats PyTorch/torch.compile here — this is the strong case:**
1. *Correctness forces a custom path anyway.* NOTES.md item under Op triage:
   stock `F.scaled_dot_product_attention` "miscomputes on Neuron's compiled
   bf16 lazy backend", which is exactly why the port hand-rolled the BMM shim.
   So the compiler's fused SDPA is off the table; the realistic compile
   baseline is the *materialized* `bmm→softmax→bmm`, which torch.compile
   cannot turn into flash attention (no online-softmax rewrite in the Neuron
   backend for this custom joint+bias layout).
2. *Kills score/prob materialization.* Today at 31f CFG the tiled path holds
   ~240 MB `scores` + ~240 MB `probs` and round-trips both through HBM per
   block. A flash kernel keeps the live score block (`≤128×512 fp32`, ~256 KB)
   and running stats in SBUF — HBM traffic for scores/probs drops to zero.
   At 61f+ the materialized path's memory (7.5 GB+) is the thing forcing tiny
   `q_chunk`; flash removes the constraint entirely.
3. *Perfect engine mapping.* d=128 contraction = full partition dim, no
   padding waste on QK^T. This is the geometry nkilib `attention_cte` is tuned
   for.
4. *One kernel launch replaces O(Sq/q_chunk × ops) dispatches* per block × 48
   blocks × 2 (CFG) × 64 steps — eager dispatch overhead is real at this op
   count.

**Tiling strategy on trn2:**
- Model on nkilib `attention_cte` (`causal_mask=False`), which already does
  the flash structure: Q in groups of 128, K/V in ~8K-token sections, running
  (max, sum) flash rescale across sections. Reuse that skeleton.
- **QK^T:** contraction = d = 128 → partition dim. stationary = Q group
  `[128(d), ≤128 q]`, moving = K block `[128(d), ≤512 k]`, PSUM result
  `[q≤128, k≤512]` (PSUM free ≤512 on trn2/gen3). Sq=9796 → 77 Q-groups of
  128; Sk=9796 → 20 K-blocks of 512.
- **Additive key-bias:** the bias is `(B,1,1,Sk)` — a function of key column
  and batch only, constant across query rows and heads. Fold it into the
  score block right before the running-max/exp: `S += bias[k_block]` as a
  `[1, k_block]` broadcast add (or fold into the `nisa.activation` bias
  operand on the exp step, free). This is the *only* deviation from a stock
  flash kernel and it is cheap — no full `(B,H,Sq,Sk)` mask is ever formed.
- **PV:** contraction = k ≤ 128 (nc_matmul K≤128), so the 512-wide K-block is
  consumed as 4×128 sub-blocks for `P @ V`; V laid out `[k≤128, d=128]`,
  P `[q≤128, k≤128]`, accumulate O `[q≤128, d=128]` in PSUM with flash rescale.
- **Non-causal:** skip all causal masking / triangle compute-skipping — every
  Q group attends every K block. Simpler than CTE's causal path (no upper-
  triangle skip), just more full blocks.
- **SBUF budget:** Q group 128×128 bf16 = 32 KB; K/V block 512×128 bf16 =
  128 KB each; score block 128×512 fp32 = 256 KB; running O+stats `128×128`
  fp32 + `128×1` = ~66 KB. Total live ≪ 24 MB SBUF. The whole point: seq
  length no longer touches HBM-resident intermediates.
- **LNC/planes:** 12 planes (B·H) map to the SPMD grid the way `attention_cte`
  shards batch across the 2 NeuronCores of a logical core; secondary shard on
  Sq for the odd counts.

**Correctness plan:** CPU/numpy reference = the exact current path
(`_attention_bmm` untiled, fp32) — it is already the port's ground truth and
the offline suite validates it against upstream at `max|err|=0`. Validate
incrementally per the SKILL workflow: (a) QK^T+scale+bias block vs numpy;
(b) online-softmax stats vs a full-materialized fp32 softmax; (c) PV; (d) full
kernel. Tolerances: bf16 atol/rtol 1e-2 (CLAUDE.md), plus per-head cosine sim
≥ 0.999 vs an fp32 oracle. Must reproduce the three masked-prompt cases the
offline suite already pins (padded / fully-masked / leak-isolation).

**Risk/difficulty:** Medium-high. Flash + custom bias + joint layout is a real
kernel, but `attention_cte` is a close, proven template and the bias fold is
localized. **Verdict: build now.** It is both the biggest perf lever and a
correctness-forced custom path, so it is not competing against a compiler
alternative — the compiler can't produce it.

---

### #2 — Fused SwiGLU FFN (gate/up + SiLU + down)  ⚠ BENCH torch.compile FIRST

**Replaces:** `ff` and `ff_context` forward — the diffusers `SwiGLU` +
`net.2` down-proj. The value|gate split is the `_shard_fused_glu` permuted
shard from `mochi_meta_loader`; a kernel must honor that the *local* weight is
already `[value_shard | gate_shard]` so `chunk(2)` on the local 4096 (or 2048
for context) is correct.

**Shapes/dtypes (31f, TP=4):** visual FFN input `(2, 9540, 3072)` bf16;
`net.0.proj` local `3072→4096` → split `[2048 value | 2048 gate]`;
`h = value * silu(gate)` → `(2, 9540, 2048)`; `net.2` local `2048→3072` with
row-parallel all-reduce. Context FFN: `(2, 256, 1536)` → `1536→2048` →
`[1024|1024]` → `1024→1536`. FFN is the **largest FLOP consumer**: ~151
MFLOP/token, ~138 TFLOP/denoise-step across visual tokens × 48 blocks × CFG,
edging out attention at 31f.

**Why NKI could beat the current path:** fuses the two matmuls with the
SiLU·gate elementwise so the wide intermediate (`(2,9540,4096)` local ≈ 150 MB
bf16) never goes to HBM. nkilib `mlp` kernel does exactly this (gate/up/down +
activation fusion, optional norm fusion). Could also fuse `norm3`/`norm4`
modulated-RMS as its pre-norm (see #4) to eat a norm pass for free.

**Why torch.compile probably gets most of this:** SwiGLU is the canonical
matmul→elementwise→matmul pattern the inductor/Neuron backend fuses well. The
elementwise `value*silu(gate)` is exactly what compilers exist to fuse, and
the two matmuls are plain, large, and already efficient as bmm. Expect
torch.compile to recover the intermediate-HBM savings on its own. The residual
NKI win is mostly the down-proj + all-reduce overlap and avoiding op-dispatch,
which is smaller here than for attention.

**Tiling strategy:** hidden H=3072 → 24 partition tiles of 128; local I=2048
→ PSUM free tiles of 512 (4 tiles). gate/up as two `[128,512]`-moving matmuls
accumulating over the H=3072 contraction (24×128 K-steps, `affine_range`),
result to PSUM, copy to SBUF, `nisa.activation(op=SiLU)` on gate, tensor_tensor
multiply with value, then down-proj `2048→3072` (contraction 2048 = 16×128).
Row-parallel all-reduce stays outside the kernel (collective). Column-tiling
per nkilib mlp. SBUF: weights dominate — value+gate `3072×4096` bf16 = 25 MB
local *exceeds* a single SBUF residency, so weights stream in H-tiles (this is
standard mlp-kernel behavior).

**Correctness plan:** numpy/torch fp32 reference of `down(value*silu(gate))`
with the permuted-shard weight; assert equivalence to the port's existing
`test_swiglu_shard` (permuted matches unsharded to 2e-7). bf16 atol/rtol 1e-2.

**Risk/difficulty:** Medium (nkilib mlp is a strong template). **Verdict:**
Build the compile baseline first. If `torch.compile` gets within ~10-15% of a
hand kernel on the FFN, skip it — the engineering cost isn't worth it. Only
commit to the kernel if profiling shows FFN as a top-2 cost *and* compile
leaves intermediate-HBM or dispatch on the table.

---

### #3 — Fused QKV projection (visual + text)  ⚠ torch.compile LIKELY WINS

**Replaces:** the six separate projections in `MochiNeuronAttnProcessor`
lines 124–135 (`to_q/to_k/to_v`, `add_{q,k,v}_proj`).

**Shapes/dtypes (31f, TP=4):** visual `(2,9540,3072) → 3× (2,9540,768)`;
text `(2,256,1536) → 3× (2,256,768)`. Asymmetric contraction (3072 vs 1536),
so it's really two fused matmuls (one per stream), not one.

**Why NKI could help:** one matmul with a stacked `[to_q|to_k|to_v]` weight
`3072→2304` local avoids three separate dispatches and weight reloads; nkilib
`qkv` kernel is built for this and can fold the `[128]` QK-RMSNorm.

**Why compile likely wins:** these are plain GEMMs with no fusion opportunity
beyond concatenating weights — torch.compile can const-fold the weight
concatenation and batch the matmul itself. The only NKI edge is dispatch
count, which matters far less than in attention. **Verdict:** don't build
standalone. Fold the QKV+QK-norm+RoPE into #1's prologue if #1 profiling shows
projection/RoPE as a visible pre-attention cost; otherwise leave to compile.

---

### #4 — Modulated fp32 RMSNorm + AdaLN  ⚠ torch.compile PROBABLY WINS; already mitigated

**Replaces:** `mochi_norm_memory.TiledModulatedRMSNorm` /
`TiledRMSNormZero` (the whole file) — the `norm1..4` + `_context` modulated
RMS norms and the AdaLN scale/gate application.

**Shapes/dtypes (31f, TP=4, CFG):** hidden `(2, 9796, 3072)`; RMS reduces over
3072 in **fp32**; modulation `scale`/`gate` are `(2,1,3072)` broadcast over S.
AdaLN linear `norm1.linear` is `3072→12288` replicated, run **per-sample not
per-token** (input is the timestep embedding, `(B,3072)`), so it is cheap; the
expensive part is the per-token fp32 RMS + modulate over 9796×3072.

**Why the port already tiles it:** `mochi_norm_memory` exists precisely because
the fp32 upcast — not attention — is the binding memory constraint at long S
(e.g. the 436 MB / 550 MB allocations that OOM'd at 61f/163f). It already
tiles the sequence and keeps fp32 scratch to ~48 MB. So the *memory* problem is
solved in PyTorch.

**Why torch.compile probably wins the *perf*:** RMSNorm (upcast → square →
mean → rsqrt → mul → downcast → modulate) is the single most fusible pattern
for an inductor-class backend, and the Neuron backend has first-class support
for it. A compiled fused RMSNorm would recover the memory win *and* the
elementwise fusion without any of the fp32-upcast-across-compile-boundary risk
the port already worries about (`--rope-bf16` failure mode). nkilib
`rmsnorm_quant` / the RMSNorm subkernel exists, but there's no quant here, so
we'd only use the plain norm path.

**Where a kernel *does* pay off:** as a **fused pre-norm inside #1 and #2** —
i.e. don't build a standalone norm kernel, fold `norm1`/`norm3` modulation into
the attention QKV prologue and the FFN prologue (nkilib mlp/qkv both support
norm fusion). That eats an entire full-sequence read/write per block.

**Correctness plan:** numpy fp32 reference identical to
`_rms_normalize_tiled`; assert bit-parity with the existing tiled path
(itself numerically identical to upstream). fp32 atol/rtol 1e-5 for the norm
math, bf16 1e-2 on the modulated output.

**Risk/difficulty:** Low as a standalone kernel, but **low value standalone**.
**Verdict:** do not build standalone — try `torch.compile` on the norms first
(very likely sufficient), and only realize norm gains by *fusing* them into #1
/ #2 if those kernels get built.

---

### #5 — Fused RoPE-apply folded into attention QK  ⚠ ONLY AS AN EXTENSION OF #1

**Replaces:** `mochi_neuron_attention.apply_rotary_emb` (lines 45–64) calls at
lines 143–144, for the visual stream only.

**Shapes/dtypes (31f, TP=4):** q,k visual `(2, 9540, 6, 128)`; cos/sin tables
`(9540, 6, 64)` fp32 (CPU-precomputed per geometry, cached — see
`patch_rope_cpu_precompute`). Interleaved real arithmetic (no
`view_as_complex`), done in fp32 then cast to bf16.

**Why fold it in:** applied standalone it materializes two rotated
`(2,9540,6,128)` bf16 tensors (~140 MB each) just to feed QK^T. Folding the
rotation into #1's Q/K load (rotate the 128-wide head vector in SBUF right
before the QK^T matmul) removes that materialization. cos/sin stream in as
`[S,64]` per head — small.

**Why NOT standalone:** RoPE apply is cheap elementwise; torch.compile fuses
it trivially, and standalone it saves only the two intermediates that compile
would also elide. It has value *only* as a prologue fused into the attention
kernel, where the rotated Q/K never leaves SBUF.

**Correctness plan:** reference = upstream `apply_rotary_emb` in fp32 (kept
byte-exact in the port on purpose). Validate the fused QK^T against
`rotate → bmm` in fp32; bf16 1e-2. Watch the fp32-table-across-compile-boundary
failure mode (`--rope-bf16`), the port's known LTX-2 fix #5/#8 risk.

**Risk/difficulty:** Low-medium *as part of #1*. **Verdict:** build only when
#1 is correct and profiling shows the RoPE apply + rotated-tensor traffic is a
measurable pre-attention cost. Otherwise leave to compile.

---

## Recommended build order

1. **First, spend a day on `torch.compile(backend="neuron")`** end-to-end
   (README step 5, currently untried). It is the cheapest possible win and it
   *resets the baseline* for every kernel decision below. Specifically it will
   likely absorb #3 (QKV), #4 (norms), and #5 (RoPE) outright, and take a big
   bite out of #2 (FFN). Measure per-op cost with the profiler after compiling.
   Caveat: cold compile is 10-30 min of NEFF build, and `--q-chunk` / tile
   sizes change graph shape (cache invalidation) — pin them.

2. **Build #1, the flash joint-attention kernel.** This is the one the compiler
   *cannot* produce (stock SDPA miscomputes on the Neuron bf16 backend, and
   there's no flash rewrite for the custom joint+additive-bias layout). It is
   the biggest single lever, and the only pure "build now". Template:
   nkilib `attention_cte` with `causal_mask=False` + the key-bias fold.
   Validate against the port's existing exact BMM path (already upstream-
   verified).

3. **Then, only if compile-baseline profiling still shows FFN top-2: build
   #2** (fused SwiGLU), preferably with `norm3`/`norm4` modulated-RMS fused in
   as its pre-norm (folds in #4's value for the FFN half).

4. **Fold #5 (RoPE) and #4 (norm1 modulation) into #1's prologue** as a
   second pass on the attention kernel, once it's correct and profiled — not
   as standalone kernels.

5. **Skip #3 standalone.** Fold QKV into #1's prologue only if projection
   shows up in the profile after compile.

### "Build now" vs "torch.compile probably wins"

| Kernel | Call | Rationale |
|---|---|---|
| #1 Flash joint attention | **BUILD NOW** | Compiler can't produce it (SDPA miscomputes; no flash rewrite); biggest lever; d=128 maps perfectly |
| #2 Fused SwiGLU FFN | **Build if profile says so** | Largest FLOPs, but compile fuses SwiGLU well; decide after compile baseline |
| #4 Modulated RMSNorm/AdaLN | **Compile wins** | Canonical fusible pattern; memory already mitigated by tiled norms; only fold into #1/#2 |
| #3 Fused QKV | **Compile likely wins** | Plain GEMMs, only dispatch-count upside; fold into #1 at most |
| #5 Fused RoPE | **Compile wins standalone** | Cheap elementwise; value only as #1 prologue |

Guiding principle (from the NKI development principles): correctness-forced
custom paths and memory-blowup eliminations are where hand-written NKI pays
for itself; pure elementwise/GEMM fusion is where the compiler has caught up.
For Mochi, that puts exactly one kernel — the joint flash attention — firmly in
"build now", with everything else gated behind a `torch.compile` measurement
that hasn't been taken yet.
