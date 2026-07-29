# Making the Mochi Flash-Attention NKI Kernel Fast on Trn2

**Goal:** beat `torch.bmm`-based tiled attention (18.7 ms) with a NKI kernel that is
currently **2.3x slower (43.9 ms)** at `S=6616, planes=P=6, D=head_dim=128, bf16,
non-causal, per-key-column additive bias`.

**Scope:** pure research (no device access). This feeds the engineer optimizing
`attention/flash_attn_nki.py` on-device. All hardware numbers are cited to the Trn2
architecture guide and `nc_matmul` API; all tiling claims are cross-checked against the
on-device-proven kernels in the knowledge pack.

Sources:
- Trn2 arch guide: `awsdocs-neuron.readthedocs-hosted.com/.../architecture/trainium2_arch.html`
- Trn1/Inf2 arch guide (PSUM banks + matmul cost model): `.../architecture/trainium_inferentia2_arch.html`
- `nki.isa.nc_matmul` API: `.../nki/api/generated/nki.isa.nc_matmul.html`
- nki-samples flash attention: `github.com/aws-neuron/nki-samples/blob/main/src/nki_samples/tutorials/attention_fwd_performance/attention_kernels.py`
- Proven local kernels: `neuron-knowledge-pack-public/knowledge/nki-kernels/attention_decode_kernel.py`, `attention_tkg_raw.py`
- Current kernel under review: `neuron/examples/Mochi/nki_kernels/attention/flash_attn_nki.py`

---

## TL;DR — the five things costing you the 2.3x

The current kernel (`flash_attn_nki.py`) is correct but does almost everything in the
**128-wide, transpose-per-tile, online-softmax** shape — the "simplest provably-correct"
tiling its own docstring calls a "step-7 optimisation left for on-device tuning" (lines
48-51). That step is the whole ballgame. In priority order:

1. **QK^T moving operand is 128 wide, must be 512.** 4x too many matmul instructions on
   the one engine that dominates runtime.
2. **K is `nc_transpose`d every k-tile** (line 234) and **P is `nc_transpose`d every
   k-tile** (line 349). Pre-lay-out K as `[D, Sk]` in HBM to eliminate the K transpose
   entirely; the P transpose can be avoided with the PV layout trick (§3).
3. **Online softmax over `sequential_range`** (line 223) serializes the k-loop and pays a
   running-max/sum rescale every tile. The whole key row fits in SBUF at these sizes — use
   **2-pass** and drop the correction math and the sequential dependency (§4).
4. **Per-k-tile bias reload + `stream_shuffle_broadcast`** (lines 260-269) runs a DMA and a
   32-lane shuffle *inside* the hot loop. Load the bias row once per plane; broadcast via
   `activation` bias arg / zero-stride `.ap()`, not a shuffle.
5. **No LNC2 sharding and `affine_range` only on the outer plane loop.** 6 planes should be
   spread across both NeuronCores; the k-loop should be parallel, not sequential.

None of these change the math — they change the instruction mix so the Tensor Engine runs
512-wide and the Vector/Scalar work rides alongside it instead of blocking it.

---

## 1. Why bmm wins at this size, and the roofline

### The shape is emphatically compute-bound (for a fused kernel)
Per head, attention is `4·S²·D` FLOP (QK^T is `2·S²·D`, PV is `2·S²·D`):

```
FLOP  = 4 × 6616² × 128 × 6 planes ≈ 1.35e11 FLOP   (135 GFLOP)
```

A **fused** kernel only moves Q,K,V,O through HBM (scores never spill):

```
bytes = 4 tensors × S × D × 2 B × 6 planes ≈ 4.06e7 B   (40.6 MB)
arithmetic intensity ≈ 1.35e11 / 4.06e7 ≈ 3,300 FLOP/byte
```

Trn2 ridge point: `79 BF16 TFLOP/s per NeuronCore ÷ ~375 GB/s/core ≈ 210 FLOP/byte`
(per chip: `632 TFLOP/s ÷ 3 TB/s ≈ 211`). At **3,300 ≫ 210** you are ~15x past the ridge:
**strongly compute-bound**. The compute-bound floor is

```
135 GFLOP ÷ 632 TFLOP/s (full chip) ≈ 213 µs   (one core: ~1.7 ms)
```

Both bmm (18.7 ms) and the NKI kernel (43.9 ms) are **1-2 orders of magnitude above the
floor** — neither is near peak. This is an **overhead / instruction-issue-bound** regime,
not a FLOP-bound one. That is *good news*: the gap is closable by fixing the instruction
mix, not by needing more FLOPs.

> Note: a *non-fused* kernel that spills the `6616×6616` score matrix to HBM would move
> ~525 MB/head (bf16) and flip memory-bound. That is exactly why you must keep scores
> on-chip (§4). The current kernel already does — good — but it pays for it with the
> online-softmax serialization.

### Why the compiler's bmm is hard to beat here
- It emits the QK^T and PV matmuls at the **full 512 moving free dim** automatically, and
  keeps the Tensor Engine near-continuously fed.
- It **fuses** the `*scale`, bias-add, max, exp, and sum onto the Vector/Scalar engines so
  they overlap the matmuls instead of stalling them.
- It picks **optimal DMA** (large contiguous, double-buffered) and shards across both cores.
- The score/prob matrices at S=6616 still *tile* cheaply for bmm — it is not yet forced into
  the pathological HBM-spill regime where it loses (that is the long-S crossover, §6).

The current NKI kernel loses all four of those advantages simultaneously.

---

## 2. The moving-operand fix — structure QK^T and PV so the free dim is 512

**`nc_matmul` contract** (from the API + `simple_matmul.py`):
`dst[M,N] = stationary[K,M].T @ moving[K,N]`, where `K` = contraction = **partition (≤128)**,
`M` = stationary free (**≤128**), `N` = moving free (**≤512** on gen2/3). Internal accum is
FP32. `LoadStationary` is a pure data move and is **up to 4x cheaper than MultiplyMoving**,
so the "more-tiles matrix should be stationary" rule = *put the operand you re-load most, or
the one with the larger free axis, on the stationary side to minimize total MultiplyMoving
time.* (The CLAUDE.md "load stationary is 4x faster" is this; the separate 4x is FP32 being
~4x slower than BF16 — keep matmul inputs bf16.)

### MM1 — QK^T with k as the 512-wide moving dim
Contraction is `D=128` → partition. This is the ideal geometry (`D==128` fills the array).

```
stationary = Qᵀ   [D=128, q_tile=128]     # q on stationary free (≤128)
moving     = Kᵀ   [D=128, k_blk=512]      # k on moving free  (=512)  ← THE FIX
dst        = scores [q=128, k_blk=512]  in PSUM (fp32)
```

- `k_blk = 512` cuts the QK^T MultiplyMoving instruction count by **4x** vs the current
  `k_size=128` (line 244-248). Each 512-wide matmul amortizes the ~64-cycle `MM_INIT_LATENCY`
  far better than a 128-wide one.
- Q is stationary and loaded **once per q-tile** (it is re-used across all k-blocks) — this is
  the correct stationary choice (Q has 1 tile, K sweeps many; but Q is the reused operand so it
  stays resident). This matches nki-samples: Q stationary `[d=128,q=128]`, K moving
  `[d=128,k=FMAX_MOVING=512]`, `num_kv_tiles = seqlen_kv // 512`.
- PSUM budget: `scores[128,512]` fp32 = exactly **one PSUM bank** (512 fp32/partition). Fine.

### MM2 — PV, two options; prefer the 512-wide `outᵀ` form
The current PV (lines 347-367) does `stationary=Pᵀ[k,q], moving=V[k,d=128]` → `out[q,d]` with
moving free = `d = 128`. With `D=128` fixed, that free dim can't grow while keeping q on the
partition. Two ways forward:

**Option A (simplest, keep as-is at 128 moving):** accept PV at 128-wide but still fix MM1 and
kill the transposes. PV is `2·S²·D` = half the FLOPs; even leaving it 128-wide, fixing MM1 +
transposes + softmax gets most of the win.

**Option B (widen PV to 512 too):** compute the **transpose of the output** so `q` becomes the
moving free dim:

```
stationary = V    [k=128, d=128]          # d on stationary free (≤128)
moving     = Pᵀ   [k=128, q_blk=512]      # q on moving free (=512)
dst        = outᵀ [d=128, q_blk=512]  in PSUM
```

This requires processing a **q-block of 512** (four 128 softmax sub-rows) and produces `outᵀ`
`[d, q]`, which you store transposed (or transpose once at the end per q-block, amortized over
the whole k-sweep — cheap). Pᵀ `[k, q]` is exactly the layout you already build for MM2, just
wider. Net: PV MultiplyMoving count also drops 4x. Do Option A first, measure, then B.

### Concrete tile loop (2-pass, pre-transposed K, Option A PV)

```python
PMAX, FSTAT, FMOV = 128, 128, 512          # nl.tile_size.{pmax, gemm_stationary_fmax, gemm_moving_fmax}
Q_TILE = 128                                # q rows on partition (softmax lanes)
K_BLK  = 512                                # k moving free for MM1
n_kb   = div_ceil(Sk, K_BLK)

for p in nl.affine_range(P):                # planes → also shard across LNC2 (see §7)
    # ---- load bias row ONCE per plane (was per k-tile) ----
    bias = load key_bias[p, :]  -> SBUF [1, Sk]          # broadcast later, not reloaded

    for qt in nl.affine_range(n_q_tiles):
        # Q loaded once, transposed once → Qᵀ [D=128, q=128] resident, stationary
        Qt = load+transpose q[p, qt] -> SBUF [D, 128]

        # ===== PASS 1: scores + row max (scores kept resident in SBUF) =====
        scores = SBUF [128, Sk]  (fp32; bf16 to halve if Sk large — see §6)
        for kb in nl.affine_range(n_kb):
            Kt_blk = Kt[p, :, kb*512 : kb*512+512]        # Kᵀ pre-transposed in HBM → [D,512], NO transpose
            s_ps = PSUM [128, 512]
            nc_matmul(dst=s_ps, stationary=Qt, moving=Kt_blk)          # k moving = 512
            # scale + bias(add, broadcast over q via zero-stride .ap) fused into the copy-out
            tensor_scalar(scores[:, kb_slice], s_ps, mul=scale)         # or activation w/ bias
            tensor_tensor(scores[:, kb_slice], +, bias_bcast[kb_slice])
        rowmax = tensor_reduce(scores, max, axis=free)                 # [128,1] ONE reduce over full row

        # ===== PASS 2: exp (fused sum) + PV =====
        probs = SBUF [128, Sk] (bf16)
        rowsum= activation(dst=probs, data=scores, op=exp, bias=-rowmax,
                           reduce_op=add, reduce_res=rowsum)            # exp+sum fused, one pass
        out_ps = PSUM [128, D]
        for kt in nl.affine_range(Sk // PMAX):                          # PV still 128-contraction tiles
            Pt = transpose(probs[:, kt_slice]) -> [k=128, q=128]        # (Option B removes this)
            V  = load v[p, kt_slice]           -> [k=128, d=128]
            nc_matmul(dst=out_ps, stationary=Pt, moving=V, accumulate=(kt>0))
        out = out_ps * reciprocal(rowsum)                              # single final normalize
        store out -> out[p, qt]
```

Key differences from the current kernel: **k moving=512**, **no online correction**, **one
max-reduce and one exp+sum over the whole row**, **bias loaded once**, **K never transposed**,
**PV accumulates in PSUM across k-tiles** instead of rescaling `o_run` every tile.

---

## 3. Transpose minimization

The current kernel does, **per (plane, q-tile)**: 1× `nc_transpose(Q)` (fine, hoisted) plus,
**per k-tile**: 1× `nc_transpose(K)` (line 234) + 1× `nc_transpose(P)` (line 349). At
`Sk=6616` that is **~52 K-transposes + ~52 P-transposes per q-tile**, each a Tensor-Engine op
competing with the matmuls.

### Kill the K transpose entirely — pre-lay-out K as `[D, Sk]` in HBM
This is the single most proven trick in the pack. `attention_decode_kernel.py` does exactly
this: its adapter passes `k_cache` **pre-transposed** as `[B, Hkv, D, S_total]`
(`_nki_attention_decode`, line 438: `k.transpose(-1,-2).contiguous()`), and the kernel then
loads K directly `[D, S]` with **one multi-partition DMA and zero `nc_transpose`** (lines
216-220). The MM1 there is `stationary=q_packed [D,heads], moving=k_full[:, k_off:k_off+512]`
— K is *already* in `[D, k]` moving layout.

For Mochi, K comes from a RoPE step upstream. **Fold the K→`[D,Sk]` transpose into the RoPE
kernel's output write** (or a one-shot transpose kernel) so the attention kernel receives
`Kᵀ [P, D, Sk]` and never transposes K in the hot loop. `attention-tkg.md` calls this the
`tp_k_prior` optimization: "Reduces transpose operations during computation." The
`attention_tkg_raw` SOTA config sets `tp_k_prior=True` (line 256).

Cost moved: ~52 transposes/q-tile → 0. The pre-transpose is `O(S·D)` done once, off the
critical loop.

### Reduce the P transpose
- **Option B (§2)** makes `Pᵀ[k,q]` the moving operand you already need — but you still
  transpose scores→Pᵀ once per k-tile. That transpose is unavoidable in the online form
  because contraction flips from `D` (MM1) to `k` (MM2). It is inherent to attention.
- Mitigations: (a) do the exp **into the transposed layout** by transposing `scores` (or the
  pre-exp tile) with the Tensor Engine while the Scalar Engine is otherwise idle, overlapping
  it; (b) downcast Pᵀ to **bf16** before the transpose (v7 in nki-samples does this) — bf16
  transpose is cheaper and MM2 wants bf16 anyway; (c) batch the transpose to 512-wide
  (`nc_transpose` PSUM free ≤512 on gen3) so it is one op per 512 keys, not per 128.

---

## 4. Online-softmax overhead vs 2-pass

### Is the online rescale a bottleneck here? Yes, indirectly.
The online form (lines 286-384) runs, **per k-tile**, on the Vector/Scalar engines:
`max`, `m_new`, `neg_m_new`, `exp`, `correction=exp(diff)`, `l = corr·l + sum`,
`o = corr·o + tile` (two ops on the full `[128,D]` accumulator). That is ~8 vector ops per
k-tile *with loop-carried dependencies* — the `sequential_range` (line 223) forbids the
compiler from overlapping k-iterations or the matmuls across them. So the Tensor Engine
stalls waiting on the scalar rescale chain. The rescale itself is cheap in FLOPs; the
**serialization** is what hurts.

### When you can use 2-pass: whenever the key row fits in SBUF — it does here
2-pass = (pass 1) compute the full score row `[128, Sk]` and its max; (pass 2) `exp` (fused
sum) then PV. No running-max, no per-tile correction, no sequential dependency — the k-blocks
in pass 1 are **independent** (`affine_range`), and PV accumulates in one PSUM bank.

SBUF cost of holding one q-tile's full score row:

| Frames | Sk    | fp32 row `[128,Sk]` /partition | bf16 /partition | Fits? (224 KB/part) |
|--------|-------|-------------------------------|-----------------|---------------------|
| Mochi (this bug) | 6,616  | 26 KB   | 13 KB  | Easily |
| 61f    | ~17,000 | 68 KB  | 34 KB  | Easily |
| 163f   | ~44,000 | 176 KB | 88 KB  | fp32 tight, **bf16 fine** |

So **2-pass is valid across every Mochi shape** (use bf16 scores at 163f). This is exactly
what `attention_cte` does: it is a plain 2-pass (max, then exp+sum) kernel and only switches
to *sectioned* flash (running stats across 8K-token sections) when **KV length > 10K**
(attention-cte.md, "Flash Attention: For K/V length > 10K tokens, divides into 8K-token
sections"). The nki-samples performance kernels are likewise **2-pass, not online**, and v6
fuses max-subtract+exp+sum into one Scalar activation "at no extra cost."

**Recommendation:** use 2-pass for Mochi and 61f. For 163f, either bf16 scores (2-pass) or
adopt attention_cte's 8K-section flash (online only *across sections*, i.e. ~5 corrections
total, not ~344). Either way, **drop per-128-tile online rescaling.**

**Tradeoff:** 2-pass reads the score row twice from SBUF (once for max, once for exp). That is
on-chip bandwidth (cheap, ~0.96-1.2 GHz × 128 lanes) and fully overlaps the Tensor Engine.
Online saves that second read but blocks the Tensor Engine — a bad trade when compute-bound.

---

## 5. PSUM budget and the right `q_tile × k_tile`

PSUM = **8 banks × 512 fp32/partition** (2 KB/bank/partition, 16 KB/partition total), up to 8
outstanding accumulation groups.

- **MM1 score tile `[128, 512]` fp32 = exactly 1 bank.** Use 2 banks to double-buffer
  (compute k-block `i+1`'s scores while copying `i` out) — 2 banks.
- **MM2 `out[128, D=128]` fp32 = 128/512 of a bank.** Accumulates across all k-tiles in **one
  bank** (that is the win of PSUM accumulate vs the current SBUF `o_run` rescale).
- **Transpose scratch** (Qᵀ, Pᵀ) uses PSUM as the `nc_transpose` destination: `[D,128]` and
  `[128,512]` ≤ 1 bank each. Budget 2 banks.

Total ~5-6 of 8 banks — comfortable, leaves room for pipelining. **Right tiling:**

- `q_tile = 128` (partition = softmax lanes; must be ≤128, use all 128).
- MM1 `k_blk = 512` (moving free = PSUM bank width; the whole point of §2).
- MM2 `k_tile = 128` (contraction ≤128), or `q_blk = 512` moving in Option B.
- Score row resident in **SBUF** `[128, Sk]` (§4), not PSUM.

Do **not** shrink q_tile below 128 or k_blk below 512 — that is precisely the current
kernel's mistake (`_K_TILE = 128`, line 99).

---

## 6. Where NKI simply won't beat the compiler for THIS shape — the honest crossover

### For Mochi's own S=6616, D=128, non-causal, 6 planes: the win is marginal and hard.
This shape is compute-bound but tiny in absolute FLOPs (135 GFLOP, ~213 µs chip floor) and the
compiler's bmm is already a well-tuned 512-wide, fused, dual-core, double-buffered pipeline.
A correctly-optimized NKI kernel (all of §1-§5 + LNC2) should be able to **match and modestly
beat** bmm here — the 2.3x deficit is entirely self-inflicted overhead, not a fundamental
wall — but expect the ceiling to be roughly **bmm-parity to ~1.2-1.4x faster**, not a
blowout. The reason to expect only modest gains: with only **6 planes** there is limited
outer parallelism to hide latency, and softmax's inherent P-transpose + reduction chain caps
Tensor-Engine duty cycle. If after §1-§5 you are within ~10-20% of bmm, that is the expected
result for this shape — do not chase diminishing returns here.

### The real NKI win is at long S, where bmm is forced to spill.
The score/prob matrices grow as `S²`. bmm (and any non-fused path) must materialize them:

| Frames | Sk    | scores `Sk²` bf16 /plane | ×6 planes | bmm behavior |
|--------|-------|--------------------------|-----------|--------------|
| Mochi  | 6,616  | 87 MB   | 525 MB  | tiles OK, competitive |
| 61f    | 17,000 | 578 MB  | 3.5 GB  | heavy HBM spill of scores/probs |
| 163f   | 44,000 | 3.9 GB  | 23 GB   | **OOM or catastrophic tiling** |

At 61f/163f the arithmetic intensity of a *fused* kernel is unchanged (~3,300 FLOP/byte,
still compute-bound), but **bmm's effective intensity collapses** because it re-reads/writes
the `S²` score and prob tensors through HBM. That is where a fused NKI flash kernel — which
**never** materializes `S²` to HBM (scores live in the `[128, Sk]` SBUF row, §4; sectioned
flash beyond 10K per attention_cte) — pulls decisively ahead. **Crossover reasoning:**

- Below ~8-10K tokens: bmm's score tiling still fits comfortably; NKI parity-to-modest-win.
  Fixing the kernel is worthwhile (parity at least, and it composes into a fused pipeline),
  but do not expect >1.5x.
- Above ~10K tokens (61f, 163f): bmm pays growing `S²` HBM traffic and eventually OOMs; the
  fused NKI kernel stays on-chip and compute-bound. **This is where NKI wins big (and where
  bmm may not run at all).** Prioritize the kernel for these shapes.

**Strategic recommendation:** ship the optimized kernel primarily for the long-sequence
Mochi configs (61f, 163f), where it is both faster and *necessary* (bmm OOMs). For the 6616
shape, target parity — treat beating bmm there as a stretch goal, not the acceptance bar.

---

## Prioritized optimization list (most impact first)

1. **[Biggest] Widen MM1 moving to `k_blk=512`** (§2). 4x fewer QK^T MultiplyMoving ops on the
   dominant engine. Change `_K_TILE=128` → a 512-wide k-block for MM1. Cross-ref: nki-samples
   `num_kv_tiles = seqlen_kv // 512`.
2. **Pre-transpose K to `[D, Sk]` in HBM; remove the per-k-tile `nc_transpose(K)`** (§3).
   ~52 Tensor-Engine ops/q-tile → 0. Fold into RoPE output. Cross-ref: `attention_decode_kernel.py`
   pre-transposed `k_cache [B,Hkv,D,S]`; `tp_k_prior=True`.
3. **Replace online softmax with 2-pass; make the k-loop `affine_range`** (§4). Removes the
   loop-carried rescale chain that stalls the Tensor Engine. Valid at all Mochi sizes (bf16
   scores at 163f). Cross-ref: attention_cte 2-pass, flash only >10K.
4. **Hoist the bias: load `key_bias[p,:]` once per plane; broadcast via `activation` bias /
   zero-stride `.ap()`, not `nc_stream_shuffle` per k-tile** (§1.4, lines 260-269).
5. **Accumulate PV in one PSUM bank across k-tiles** instead of `o_run = corr·o_run + tile`
   every iteration (§5, replaces lines 369-381).
6. **LNC2 shard the 6 planes across both NeuronCores** (§1.5); dispatch with a program grid
   like `attention_tkg_raw`'s `wrapped[2]` (grid=2). 6 planes → 3 per core.
7. **Downcast transposes/exp/Pᵀ to bf16** in the hot path (nki-samples v7). Halves transpose
   and MM2 cost; matmul inputs must be bf16 anyway (FP32 matmul is ~4x slower).
8. **[Option B, after 1-5] Widen MM2 to 512** via the `outᵀ[d, q_blk=512]` layout with
   `stationary=V[k,d], moving=Pᵀ[k,q=512]` (§2). 4x fewer PV ops; transpose output once per
   q-block.
9. **Double-buffer** MM1 scores across 2 PSUM banks and prefetch K/V blocks (software
   pipelining) once 1-8 are in and profiled.

Expected trajectory: items 1-5 should collapse most of the 2.3x (they remove the 4x
instruction inflation, the ~104 transposes/q-tile, and the serialization). Items 6-9 push from
parity toward a win. Honest ceiling at S=6616: **~bmm-parity to ~1.4x**. The decisive,
easy win is reserving this kernel for **61f/163f**, where bmm spills `S²` through HBM or OOMs
and the fused NKI kernel is both faster and the only option that runs.

---

## Appendix — exact tiling of the proven kernels (for reference)

**`attention_decode_kernel.py`** (Gemma4 decode, S_q=1, d>128 tiled):
- K **pre-transposed** in HBM `[B,Hkv,D,S]`; loaded `[D=128, S]` with one DMA per d-chunk,
  **no `nc_transpose` on K** (lines 216-220).
- MM1: `stationary=q_packed[dc] [D,heads]`, `moving=k_full[dc][:, k_off:k_off+K_TILE]`,
  `accumulate=(dc>0)` over d-chunks (lines 293-299). K_TILE=128 here only because S_q=1
  makes the moving free = keys and heads are tiny; for prefill you widen this to 512.
- Softmax: **negated-max** trick (`running_max_neg`, +1e30 sentinel), **fused exp+sum**
  (`reduce_cmd.reset_reduce`, lines 346-352). Online *only because decode streams the KV
  cache*; s_active=1.
- MM2: transpose `tile_exp → p_t [k,heads]` (one transpose/k-tile), `stationary=p_t`,
  `moving=v_tile[kt*n_d+dc]`, V **pre-tiled** `[128,128]` (lines 355-367).
- DMA discipline: **one multi-partition DMA** per operand, never per-index scalar loops
  (the wrap_nki aliasing rule).

**`attention_tkg_raw.py` / attention-tkg.md** (paged decode): SOTA config `tp_k_prior=True`
(pre-transposed K), `strided_mm1=False`, `use_pos_id=True`, `qk_in_sb=True`, grid=2 (LNC2).
Confirms: pre-transposed K + in-SBUF QK + dual-core is the fast path.

**nki-samples `attention_fwd_performance`** (prefill, the closest analog to Mochi):
`PMAX=128, FMAX_STATIONARY=128, FMAX_MOVING=512`. Q **stationary** `[d=128,q=128]`, K
**moving** `[d=128,k=512]`, `num_kv_tiles=seqlen_kv//512`. PV tiled 128×128 with V
pre-transposed. **2-pass softmax** (not online); v6 fuses max/exp/sum; v7 bf16-downcasts
transposes/exp; asserts `seqlen_q>=512` for the tiled path. v7 is "the fastest attention
kernel we have thus far." **This is the template to match for Mochi prefill attention.**
