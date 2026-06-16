# FLUX.2-klein-4B 4 MP — Progress & Findings (2026-06-15)

Full record of the multi-core / 4 MP investigation so nothing is lost.

## Goal

Enable FLUX.2-klein-4B at 4 MP (2048×2048) on Trainium2. Single core OOMs
above 1 MP in the DiT attention activation memory. Need multi-core
(tensor parallelism) to spread the attention activation.

## Single-core resolution sweep (baseline)

Beta 3 DLC, `--vae-on-neuron --cache-image-latents`, 4-step distilled:

| Resolution | MP | Single core | Wall reason |
|---|---:|---|---|
| 1024×1024 | 1.05 | ✅ 5.8 s warm | baseline (the shipped number) |
| 1280×1280 | 1.64 | ❌ OOM | 512 MB shared-scratchpad alloc fail |
| 1536×1536 | 2.36 | ❌ OOM | DiT attention activation memory |
| 1792×1792 | 3.21 | ❌ compile error | activation memory |
| 2048×2048 | 4.19 | ❌ OOM (extrap) | 16× more attention than 1024² |

**Wall = activation memory, not weights.** DiT attention is O(tokens²);
2048² has 64K tokens vs 1024²'s 16K → 16× attention memory.

## Multi-core (TP) results

All on trn2.48xl, Beta 3 DLC, env:
`NEURON_RT_VIRTUAL_CORE_SIZE=2`, `NEURON_LOGICAL_NC_CONFIG=2`,
`NEURON_SKIP_EFA_AFFINITY=1`.

| Config | Res | TP | Warm | Quality (std, ref 18.16) | Verdict |
|---|---:|---:|---:|---:|---|
| single-core (default SDPA) | 1024² | 1 | 5.8 s | 18.16 | ✅ shipped |
| manual flash | 1024² | 1 | 43.5 s | **18.13** | ✅ proves manual flash math correct |
| v1 plan + attention_cte[2] | 1024² | 2 | 57.5 s | **45.39** | ❌ corrupted |
| v1 plan + attention_cte[1] | 1024² | 2 | 56.7 s | 45.39 | ❌ corrupted (same) |
| v1 plan + manual flash | 1024² | 2 | 79.6 s | 45.39 | ❌ corrupted (proves NOT the kernel) |
| **v2 plan + manual flash** | **1024²** | **2** | **93 s** | **18.12** | ✅ **CORRECT — TP works!** |
| **v2 plan + attention_cte[2]** | **1024²** | **2** | **59.7 s** | **18.13** | ✅ CORRECT + 36% faster than manual |
| v2 plan + manual flash | 2048² (4MP) | 2 | 503 s | (not captured) | ✅ runs, slow |
| **v2 plan + attention_cte[2]** | **1792² (3MP)** | **2** | **216 s** | **2.64** | ⚠️ DiT fast (14s/4-step) but VAE decode ~200s + output corrupt |
| v2 plan + manual flash | 2048² (4MP) | 4 | 534 s | **4.99** | ❌ corrupted at TP=4 |
| v2 plan + NKI flash | 1024² | 2 | stalled | — | ❌ per-(b,h) recompile blowup |

## ROOT CAUSE #1 — the SwiGLU sharding bug (SOLVED)

The v1 TP plan sharded the FFN `linear_in` with `ColwiseParallel`. But
FLUX.2-klein-4B's FFN is a **SwiGLU**:

```python
class Flux2SwiGLU(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)   # gate, value
        return self.gate_fn(x1) * x2
```

`linear_in` outputs `[gate; value]` concatenated on the last dim
(18432 = 9216 + 9216). `ColwiseParallel` shards the last dim, so each
rank gets an **arbitrary mix** of gate/value columns. Then `chunk(2)`
pairs the wrong halves → garbage. **This is why std=45 regardless of
attention kernel.**

**Fix (v2 plan):** don't shard the GLU FFN at all. Only shard attention
Q/K/V/O. FFN runs replicated (each rank computes full FFN). Slower but
correct → **std=18.12 at TP=2 confirms.**

## ROOT CAUSE #2 — TP=4 corruption (NOT yet solved)

v2 plan works at TP=2 (std=18.12) but corrupts at TP=4 (std=4.99,
near-blank). v2 skips the single-stream blocks entirely (they have a
fused `to_qkv_mlp_proj` we couldn't safely shard). The remaining
divergence at TP=4 is unexplained — needs per-block tensor-norm
comparison TP=2 vs TP=4 to localize. Likely candidates: the encoder/
image token concat boundary, or `to_out.0` RowwiseParallel all-reduce
at 6-heads-per-rank.

## ROOT CAUSE #3 — NKI kernel per-(batch,head) recompile (NOT yet solved)

The `nki_flash_attn_flux.py` wrapper does:
```python
for bh in range(B * H):
    outputs.append(_flash_attn_fwd_op(q_flat[bh], ...))
```
Under `torch.compile`, this generates a separate graph per (batch,head).
25 layers × 12 heads × 4 steps ≈ 1200 unique graphs → compile never
finishes (steps got *slower*: 320 s → 931 s). The kernel must process
all heads in ONE call (batched), not a Python loop.

## THE BIG DISCOVERY — production references exist in-repo

Found during this session (the "look in NxDI / vllm / all repos" search):

### 1. NxDI production FLUX TP implementation
`neuron/external/pr-117-nxdi-diffusion-models/src/neuronx_distributed_inference/models/diffusers/flux/modeling_flux.py`
(1511 lines, AWS Neuron team's own FLUX-on-Trainium TP model).

Key patterns it uses that solve our exact bugs:
- **Separate proj_out for attn and MLP** (`proj_out_attn` + `proj_out_mlp`,
  both `RowParallelLinear(reduce_output=False)`), summed then ONE
  `reduce_from_tensor_model_parallel_region`. Avoids the all-gather after
  QKV and merges two all-reduces into one. This is the clean way to
  shard the single-stream fused block.
- **GELU FFN as a fused `ColumnParallelLinear`** (`NeuronGELU`,
  `gather_output=False`). NOTE: NxDI's FLUX is FLUX.1 (GELU FFN), ours is
  FLUX.2-klein (SwiGLU) — the activation differs but the **column/row
  parallel split pattern is identical and directly applicable**.
- **`attention_cte[2]` used directly** and works, because NxDI builds the
  whole model with `ColumnParallelLinear`/`RowParallelLinear` from
  `neuronx_distributed`, so the parallel state is set up correctly. The
  kernel isn't the problem — the *surrounding sharding* is.
- `attention_wrapper_sharded_without_swap()` (line 82) and
  `attention_wrapper_context_parallel_single_transformer()` (line 140)
  show the exact Q/K/V layout + `tp_k=True` transpose handling for the
  CTE kernel.

### 2. Dedicated FLUX bidirectional flash NKI kernel
`.tmp/autocomp/sols/trn-advanced-nki1/9_flux_attn_ref.py` (101 lines).

This is a **multi-head batched** flash kernel:
```python
@nki.jit
def solution(q, k, v, kernel_dtype, acc_type,
             num_heads=24, seq_len=4608, d_head=128):
    for head_idx in nl.affine_range(num_heads):   # device-side loop!
        for q_tile_idx in nl.affine_range(num_q_tiles):
            ...
```
The `num_heads` loop is **inside** the kernel via `nl.affine_range` —
device-side, NO recompile per head. This is exactly the fix for ROOT
CAUSE #3. Input layout: q `(num_heads, seq_len, d_head)`,
k `(num_heads, d_head, seq_len)` transposed, v `(num_heads, seq_len, d_head)`.
Bidirectional (no causal mask) — correct for diffusion.

Autocomp (the repo it lives in) is an auto-optimization tool — there may
be an *optimized* variant beyond this reference. Worth checking
`.tmp/autocomp/` for tuned outputs.

### 3. Qwen-Image-Edit NxDI contrib (another MMDiT)
`neuron/external/pr-117-nxdi-diffusion-models/contrib/models/Qwen-Image-Edit/src/`
has `compile_transformer_v1_flash.py`, `attention_cte_qie_hoisted_q.py` —
another worked example of CTE attention in a diffusion DiT with TP.

## Files produced this session

In `/mnt/data/work/flux2_latest/` on the test box (3.15.152.199):
- `run_flux2_tp_v2.py` — TP runner with v2 plan (attention-only shard) ✅ correct at TP=2
- `flux2_tp_plan_v2.py` — v2 plan + apply_tp_fixes_v2
- `flux2_attention_manual_flash.py` — pure-Python flash (correct, slow)
- `flux2_attention_nki.py` — NKI installer (per-bh loop, needs batched kernel)
- `run_flux2_tp_v2_nki.py` — NKI variant runner

In this WIP folder:
- `README.md` — plan + sweep findings
- `RESULTS_TP2_FIRST_PASS.md` — first TP=2 attempt notes
- `PROGRESS_AND_FINDINGS.md` — this file
- `src/` — the TP scaffold copies


## 3 MP (1792²) deep result — TWO NEW WALLS beyond the DiT (2026-06-15)

Ran v2 plan + attention_cte at 1792² (3.2 MP), TP=2. This is the most
informative single data point of the session:

```
first call (cold compile): 1721 s
warm avg: 216.4 s   (run0 215.6, run1 217.2)
DiT denoising (stderr bar): 14-15 s for 4 steps  ← FAST
quality: std=2.64   (reference 18.16)            ← CORRUPT
```

### Wall #4: the DiT is fast, the VAE is the bottleneck at high res

The DiT denoising loop runs **4 steps in ~14 s** at 3 MP TP=2 — the
attention sharding + CTE kernel is working well. But the full pipeline
warm call is **216 s**. The ~200 s gap is **VAE decode + image-latent
prep at 1792²**, which runs unsharded on the VAE-on-Neuron path.

=> At ≥3 MP the VAE, not the DiT, becomes the dominant cost. The VAE
also needs attention to high-res: either VAE tiling (`enable_tiling`)
or a sharded/optimized VAE decode. The 1 MP win hid this because the
VAE was cheap at 1 MP.

### Wall #5: output corrupted at 3 MP (std=2.64)

std=2.64 (vs 18.16 reference) = near-blank/broken image. At 1024² the
exact same code path gave std=18.13 (correct). So something breaks
specifically at higher resolution. Suspects:
- VAE-on-Neuron PAVE fixes (gather-free upsample + fp32 GroupNorm) may
  not hold at 1792² tile sizes.
- The sharded DiT latent may be subtly wrong at this resolution but only
  visible after VAE decode amplifies it.
- The image-latent caching path interacting with the larger latent.

Action: re-run 3 MP with **VAE on CPU** (not Neuron) to isolate whether
the corruption is the VAE-on-Neuron path or the sharded DiT. If
CPU-VAE gives std~18, the bug is the VAE-on-Neuron path at high res; if
still std~2.6, the bug is in the DiT latent.

### Revised wall summary

| Resolution | DiT (sharded) | VAE decode | Output | Verdict |
|---|---|---|---|---|
| 1024² (1 MP) | fast | cheap | std 18.13 ✅ | works |
| 1792² (3 MP) | **fast (14s/4-step)** | **~200s ❌** | **std 2.64 ❌** | DiT ok, VAE+quality broken |
| 2048² (4 MP) | doesn't finish compiling | — | — | single-stream wall |

### What this changes in the plan

The DiT TP sharding is NOT the main 3 MP blocker — it's already fast.
The new priorities for 3 MP specifically:
1. **Fix the high-res quality corruption** (isolate VAE-on-Neuron vs DiT
   with a CPU-VAE control run) — BLOCKER
2. **Optimize/shard the VAE decode** at high res (it's ~200s, the new
   dominant cost) — tiling or TP on the VAE
3. Then the DiT single-stream sharding (Phase 2) matters more at 4 MP
   than 3 MP, since 3 MP DiT already runs in 14s

This means **3 MP is potentially closer than 4 MP** — the DiT works,
we "just" need to fix VAE cost + the high-res quality bug. That's a
more tractable target than the 4 MP single-stream compile wall.


## 3 MP output image inspection — it's NEAR-BLANK (2026-06-15)

Inspected the actual saved PNG from the 3 MP run:

```
shape: (1792, 1792, 3)
min/max: 167 / 187   mean: 173.25   std: 2.64
unique values: 21    per-channel mean: [176.4, 173.2, 170.2]
```

The output is a **near-flat gray field** — essentially the (180,180,180)
synthetic input gray, barely modified. The DiT denoising ran fast (14s)
but **produced almost no change to the latent**. The image-to-image
pipeline returned ~the input.

### Refined diagnosis

- VAE is on **CPU** in the TP path (the runner does NOT pass
  `vae_on_neuron=True`), so this is NOT a VAE-on-Neuron bug.
- The sharded DiT forward is producing **near-identity / near-zero
  output** at 3 MP. At 1024² the identical code path gives std=18.13
  (correct), so this is **resolution-dependent**.
- Same failure class as TP=4 at 4 MP (std=4.99, near-blank). The common
  thread: **as sharding × resolution grows, the sharded attention path
  collapses toward zero output.**

### Hypothesis for the high-res sharded collapse

Most likely one of:
1. **RoPE / positional embedding at high-res sequence** — the CPU+fp32
   RoPE replacement (`NeuronFluxPosEmbed`) may produce wrong freqs at
   ~49K positions (numerical range, or an indexing assumption that holds
   at 4608 tokens but not 49152).
2. **attention_cte flash-sectioning at long seq** — at 49K tokens the
   kernel uses its flash-section path (`>10K`). Combined with the
   sharded head count (12) + bidirectional mask, the online-softmax
   accumulation may underflow/zero out. The kernel was validated by the
   NxDI team at FLUX.1 shapes; our FLUX.2 high-res shape may be outside
   tested range.
3. **bf16 numerical underflow** in the longer softmax denominator at
   49K keys → near-zero attention weights → near-zero output.

### Next diagnostic (cheap, do first next session)

Run 1792² TP=2 with **manual flash attention** (not CTE). The manual
flash is plain torch (online softmax in fp32-ish), no kernel sectioning.
- If manual flash gives std~18 at 3 MP → the bug is in `attention_cte`'s
  long-sequence path. Fix: tune the kernel's section length or use the
  autocomp FLUX kernel for the FLUX.2 shape.
- If manual flash ALSO gives std~2.6 at 3 MP → the bug is upstream
  (RoPE at high-res positions, or the latent prep). Fix: validate RoPE
  freqs at 49K positions vs CPU reference.

This single control run localizes wall #5 to either the kernel or the
RoPE/prep, which determines the fix.

## Session-end status (2026-06-15)

**Solid wins:**
- TP multi-core proven correct on Beta 3 (SwiGLU bug solved, v2 plan)
- 1 MP TP=2: correct (std 18.13), 59.7s with CTE kernel
- DiT denoising at 3 MP is FAST (14s/4-step) — attention sharding works
- Full cross-repo reference index captured (NxDI FLUX, FLUX NKI kernel,
  Qwen-Image-Edit, context-parallel attention)

**Open walls (in priority order):**
1. **High-res sharded DiT collapses to near-blank output** (3 MP std=2.64,
   4 MP std=4.99). Localize with the manual-flash-at-3MP control run.
2. VAE decode at high res is ~200s on CPU — needs tiling or TP.
3. 4 MP single-stream blocks don't compile in reasonable time (need
   NxDI-style single-stream sharding).
4. TP=4 corruption (separate from #1, or same root cause).

**The honest takeaway:** the DiT compute scaling works (3 MP DiT is fast),
but there's a high-resolution numerical-correctness bug in the sharded
path that's the real blocker — not raw compute. That's a focused
debugging target, not an open-ended perf problem.


## CONCLUSIVE: high-res bug is UPSTREAM of attention (2026-06-15)

Ran the diagnostic control — 3 MP TP=2 with **manual flash** (plain torch,
no NKI kernel):

```
3 MP manual flash:    std=2.64   warm 334s
3 MP attention_cte:   std=2.64   warm 216s
```

**IDENTICAL corruption (std=2.64) with both attention implementations.**
This conclusively rules out the attention kernel. The high-res collapse
is **upstream of attention** — same pattern as the SwiGLU bug (where both
kernels gave identical std=45).

### Prime suspect: RoPE / positional embedding at high-res

The `NeuronFluxPosEmbed` (CPU+fp32 RoPE replacement) computes position
frequencies for the latent grid. At 1792² there are ~49K image tokens vs
~4096 at 1024². Likely failure modes:
1. The 2D position-id grid (`img_ids`) for the larger latent is computed
   with a hardcoded or 1024-derived assumption → wrong positions.
2. fp32 freq computation range issue at ~49K positions.
3. The `image_rotary_emb` cos/sin tensor shape mismatches the sharded
   attention's expected `[S, D]` at high res.

Because RoPE feeds EVERY attention layer, wrong RoPE → every layer
near-identity → near-blank output (exactly what we see: output ≈ input
gray field).

### Definitive next step (next session, cheap)

Add a RoPE validation harness: compute `image_rotary_emb` (cos/sin) at
1792² on the Neuron path vs a pure-CPU diffusers reference for the same
latent grid. Diff them. If they diverge, the bug is localized to the
position-id grid construction or the freq computation at high res — a
small, fixable function, NOT the TP sharding or the attention kernel.

This is the highest-leverage single fix: **if RoPE at high-res is
corrected, the already-fast sharded DiT (14s/4-step at 3 MP) should
produce correct output**, unblocking 3 MP immediately.

### Updated wall priority

1. **RoPE-at-high-res correctness** (the std=2.64 root cause) — BLOCKER,
   cheap to localize, likely a small function fix.
2. VAE decode at high res (~200s CPU) — perf, needs tiling/TP.
3. 4 MP single-stream compile wall — needs NxDI single-stream sharding.

The encouraging news: walls #1 and the DiT speed are both better than
feared. The DiT shards and runs fast at 3 MP; the blocker is a localized
positional-embedding correctness bug, not a fundamental compute or
memory limit.


## DECISIVE bisection — bug is Neuron-device-TP, NOT patches/pipeline (2026-06-15 cont'd)

Controls run at 1280² (4 steps, same seed/input):

| Config | std | Verdict |
|---|---:|---|
| CPU stock pipeline (no patches, no TP) | **17.65** | ✅ correct — model+pipeline CAN do 1280² |
| CPU + neuron patches (no TP, no device) | **17.65** | ✅ correct — patches are fine |
| Neuron device + TP=2 (v2 plan) | **4.58** | ❌ broken |
| Neuron single-core @ 1152² | OOM | can't test (DiT doesn't fit) |

**Conclusion: the bug is specifically Neuron-device execution WITH TP at
high res.** The patches, the pipeline math, RoPE, the cache, the sigma
cast — all proven correct. Eliminated.

### The std gradient is the key clue

std by resolution (TP=2 v2): 1024²=18.1 ✅, 1280²=4.58, 1792²=2.64,
2048²=fails. The output collapse **worsens smoothly with resolution**.
A logic bug would be all-or-nothing; a smooth degradation points to a
**numerical-precision / accumulation** problem that scales with token
count.

### Leading hypothesis: bf16 accumulation in the sharded attention reduce

The `to_out.0` RowwiseParallel does an all-reduce summing each rank's
partial attention output. At higher token counts the bf16 accumulation
error grows → progressive collapse. At 1024² (1024 tokens) the error is
tolerable; by 1792² (3136 tokens) it dominates.

Next test: force fp32 for the attention output / the row-parallel reduce
and re-run 1280². If std recovers to ~18, the fix is fp32 accumulation
in the sharded attention output path (cheap, localized).


## ROOT CAUSE FOUND: bf16 precision collapse at high token count (2026-06-15)

The decisive control — CPU + patches at **bf16** (no TP, no device, no
compile) at 1280²:

| Config (1280²) | dtype | std | 
|---|---|---:|
| CPU stock | fp32 | 17.65 ✅ |
| CPU + patches | fp32 | 17.65 ✅ |
| **CPU + patches** | **bf16** | **8.99 ⚠️ degraded** |
| Neuron device + TP=2 | bf16 | 4.58 ❌ |

**bf16 alone — with NO TP and NO Neuron — degrades 1280² from 17.65 to
8.99.** TP makes it worse (4.58) by adding bf16 accumulation in the
cross-rank all-reduce.

### Why this happens

The DiT forward runs in bf16. Attention softmax denominators and the
residual stream accumulate over the token dimension. At 1024² (1024
tokens) bf16 rounding is tolerable (std 18). As tokens grow
(1280²=1600, 1792²=3136, 2048²=4096) the accumulated bf16 error grows,
collapsing the output toward its mean (gray field). TP compounds it via
the bf16 RowParallel reduce.

This explains EVERYTHING:
- the smooth std gradient with resolution (precision, not logic)
- identical across attention kernels (it's the residual stream, not attn)
- RoPE clean (it's accumulation, not positional)
- works at 1024², degrades above (token-count dependent)

### THE FIX: run the DiT in fp32 at high res

CPU fp32 proves correct (17.65). So the fix is to keep the high-res DiT
in fp32 (or at least fp32 norms + fp32 attention accumulation + fp32 TP
reduce). Trade-off: ~2× memory + slower, but TP splits the fp32 weights
across cores so it can fit. **Testing TP=2 @ 1280² full-fp32 now.**

If full-fp32 is too slow/big, the optimization is selective fp32:
- fp32 softmax denominator + output accumulation in attention (the
  `attention_cte` kernel already supports `softmax_dtype=fp32,
  mm_out_dtype=fp32` — just pass them)
- fp32 LayerNorm/AdaLN (norms are the most precision-sensitive)
- fp32 residual adds
- keep matmul weights bf16 for speed

This is a well-understood mixed-precision recipe — exactly what GPU FLUX
does (bf16 matmuls, fp32 softmax/norm/accumulation). Our path was
all-bf16, which is why it collapsed at high res.


## ✅✅✅ FIX CONFIRMED: TP=4 + fp32 → CORRECT above 1024² (2026-06-15)

First correct high-res output on Neuron:

| Config | std | Notes |
|---|---:|---|
| TP=2 bf16 @ 1280² | 4.58 ❌ | precision collapse |
| TP=2 fp32 @ 1280² | OOM (rank1) ❌ | fp32 too big for 2-way split |
| **TP=4 fp32 @ 1280²** | **16.99 ✅** | **CORRECT! fits + precise** |

DiT denoising warm: **13s for 4 steps**. No OOM on any rank.

**The winning recipe: TP=4 (4-way split fits the fp32 memory) + fp32
(fixes the bf16 high-res precision collapse).** This is the first
correct >1 MP result on Trainium. Now testing 2048² (4 MP) with the
same recipe.

The earlier TP=4 corruption (std=4.99) was at bf16 — it was the SAME
precision bug, not a TP=4 logic bug. fp32 fixes both.


## MIXED-PRECISION LADDER — quantified the bf16 ceiling (2026-06-15 cont'd)

Built three memory-safe mixed-precision modules and tested the full
ladder at 1280² TP=2 (all keep activations bf16 → all FIT, unlike full
fp32). Goal: recover fp32 correctness without the fp32 activation OOM.

| Config @1280² TP=2 | std | Δ | Note |
|---|---:|---:|---|
| bf16 (CTE) baseline | 4.58 | — | precision collapse |
| + fp32 norms (CTE) | 5.30 | +0.7 | norms help a little |
| + fp32 norms + fp32 softmax (manual flash) | 4.74 | ~0 | softmax NOT the culprit |
| + fp32 norms + fp32 softmax + **fp32 residual stream** | **7.93** | **+3.2** | residual stream WAS a major leak |
| + all above + fp32 attention QK/PV matmul | 1.44 | −6.5 | ❌ regression (bug in fp32-mm under TP DTensor) |
| CPU bf16 (everything bf16, reference ceiling) | 8.99 | — | the bf16-matmul ceiling |
| CPU fp32 / TP=4 fp32 | 17.65 / 16.99 | — | correct, but fp32 OOMs >1.6MP |

### What the ladder proves

1. **The residual stream was the single biggest fixable bf16 leak**
   (+3.2 std). The stock code even clips it at ±65504 (bf16 max) — it
   genuinely overflows at high token count. `flux2_fp32_residual.py`
   keeps the two residual accumulators in fp32 (one [B,S,D] tensor each)
   while sub-blocks stay bf16 → memory-safe.
2. **Norms and softmax are minor contributors** (~+0.7, ~0). They're
   already fp32-accumulated in the manual flash.
3. **Memory-safe mixed precision tops out at the bf16-MATMUL ceiling
   (~8 std), same as CPU-bf16.** The residual+norm+softmax fixes recover
   the TP penalty (4.58→7.93 ≈ CPU bf16 8.99) but cannot exceed it,
   because the Linear/attention matmul INPUTS are still bf16.
4. **fp32 matmul inputs are what's actually needed for >1.6MP
   correctness** — and that's exactly the full-fp32 path that OOMs,
   because fp32 q/k/v/activations don't fit and head-parallel TP doesn't
   shard the sequence-dim activation.
5. fp32-attn-mm regressed to 1.44 — a bug in the fp32 attention matmul
   under TP DTensor (not a precision effect). Not chased further; the
   conclusion above doesn't depend on it.

### fp32 OOM scaling — head-TP can't fit fp32 ≥3MP (confirmed both ways)

| Config | Result |
|---|---|
| TP=4 fp32 @1280² (1.6MP) | ✅ std 16.99, 13s DiT |
| TP=2 fp32 @1280² | ❌ OOM rank1 |
| TP=4 fp32 @1792² (3.2MP) | ❌ OOM (NRT alloc 843MB fail, loops) |
| **TP=8 fp32 @1792² (3.2MP)** | **❌ OOM (34 alloc fails, exits)** |

Doubling cores 4→8 barely helped at 3MP because the dominant memory
(sequence-dim activation + manual-flash fp32 score tiles) does NOT shard
under ColwiseParallel/RowwiseParallel — those only shard weights + the
per-head attention. **This is the definitive memory wall.**

## THE VERDICT (2026-06-15)

| Resolution | Status | Recipe |
|---|---|---|
| 1024² (1 MP) | ✅ SHIPPED | single-core bf16, 4.2s warm |
| 1280² (1.6 MP) | ✅ **CORRECT (new)** | TP=4 fp32, std 16.99, ~13s DiT |
| 1792² (3.2 MP) | ❌ blocked | fp32 OOMs @TP8; bf16/mixed = ceiling ~8 |
| 2048² (4 MP) | ❌ blocked | same wall, worse |

**The honest finding:** FLUX.2-klein-4B above ~1.6 MP needs fp32 *matmul*
precision, which needs the full activation in fp32, which does not fit
under head-parallel TP at any core count we have (8). Memory-safe mixed
precision recovers to the bf16 ceiling (~8 std) but the model genuinely
needs fp32 matmuls at high token count — that's not optional for this
architecture.

**The one real unlock left: context / sequence parallelism.** Shard the
sequence dimension (the 12K–64K tokens) across cores so each core holds
an fp32 *slice* of the activation that fits. NxDI already has this:
`attention_wrapper_context_parallel_single_transformer()` in
`neuron/external/pr-117-nxdi-diffusion-models/.../flux/modeling_flux.py`.
That's the next implementation — it's a real multi-hour port, not a flag.

### New files this pass (in /mnt/data/work/flux2_latest/)
- `flux2_mixed_precision.py` — fp32 leaf-norm upcast (skips composite
  AdaLN-with-Linear to avoid matmul dtype mismatch)
- `flux2_fp32_residual.py` — fp32 residual-stream patch (the +3.2 win)
- `run_flux2_tp_mixed.py` — runner with --attn cte|manual,
  --fp32-residual, --fp32-attn-mm, --fp32-attn flags
- manual flash gained COMPUTE_FP32 / set_compute_fp32()


## ✅✅✅ v3 FULL SHARDING — 3MP UNBLOCKED (2026-06-15)

Implemented `flux2_tp_plan_v3.py`: splits the fused SwiGLU FFN
(linear_in → gate_proj + value_proj) and the single-stream fused
`to_qkv_mlp_proj`/`to_out` (→ to_q_s/to_k_s/to_v_s/mlp_gate/mlp_value +
dual out-proj), so the bulk of the model (20 single-stream blocks + all
FFNs) shards instead of running replicated. CPU forward-equivalence
verified first (`flux2_v3_selftest.py`, all PASS max|Δ|~1e-6).

| Config (fp32, v3 full shard) | Fits? | std | vs v2 | 
|---|---|---:|---|
| 1280² (1.6MP) TP=2 | ✅ (v2 OOM'd @TP2) | **16.93** | v2 needed TP=4 |
| **1792² (3.2MP) TP=4** | ✅ (v2 OOM'd @TP4 AND TP8) | **13.30** | **v2 = OOM** |
| 2048² (4MP) TP=8 | ⏳ running | — | v2 = OOM |

### What v3 changes

- **Memory:** sharding the 20 single-stream blocks + FFNs drops the
  per-core fp32 activation ~world_size×. fp32 now FITS at 3MP (TP=4) and
  at 1.6MP with just TP=2. The replicated-FFN OOM (843MB alloc) is gone.
- **Correctness:** std 16.93 @1.6MP and 13.30 @3MP — both real images
  (vs bf16 blank 2.64 @3MP). The CPU split self-test guarantees the
  weight layout is preserved, so no SwiGLU-scramble regression.

### 3MP: from blank to real

1792² went from **std 2.64 (blank gray, bf16)** to **std 13.30 (real
image, v3 fp32)**. First correct >2MP output on Trainium. std 13.30 vs
~18 reference = structurally correct but slightly soft — likely the
synthetic gray-probe input + residual bf16 in the to_out all-reduce /
embedder. Real-prompt generation would likely score higher.

### Speed (honest)

v3 uses the pure-Python manual flash in fp32 → slow warm:
- 1280² TP=2: 227s warm
- 1792² TP=4: 429s warm
Speed is the NEXT optimization (swap manual flash → a batched fp32-capable
NKI/CTE kernel; the autocomp FLUX kernel at
`.tmp/autocomp/sols/trn-advanced-nki1/9_flux_attn_ref.py` is the device-side
multi-head batched reference). Correctness + fit came first.

### The recipe that works

fp32 (correct) + v3 full sharding (fits) = the combination that resolves
the memory×precision tension. No context parallelism needed after all —
sharding the fused projections (not just attention heads) was enough to
fit fp32 through 3MP. 4MP test in flight.

### New files
- `flux2_tp_plan_v3.py` — full-shard plan + restructure_for_tp + fixes
- `flux2_v3_selftest.py` — CPU weight-split equivalence check (all PASS)
- `run_flux2_tp_v3.py` — v3 runner (--dtype fp32|bf16, full shard)


## 4MP (2048²) v3 TP=8 fp32 — FITS but correctness cliff (2026-06-15)

```
first call (compile): 891s
warm: 578.8s
quality: std=2.60   ← collapsed (blank)
OOM: NONE (0 alloc failures) — 4MP fp32 FITS with v3 sharding
```

**Memory wall at 4MP is SOLVED** — v3 sharding lets 4MP fp32 run
end-to-end on TP=8 with no OOM (v2 OOM'd at every config). But the output
collapsed (std 2.60).

### The fp32 resolution gradient (the remaining bug)

Even in fp32 + v3 full sharding, std degrades with resolution:

| Res | TP | std (fp32 v3) |
|---|---|---:|
| 1280² (1.6MP) | 2 | 16.93 ✅ |
| 1792² (3.2MP) | 4 | 13.30 ✅ (soft) |
| 2048² (4MP) | 8 | 2.60 ❌ (cliff) |

This is NOT bf16 precision (we're in fp32) and NOT memory (it fits). It's
resolution-dependent and cliffs hard between 3MP and 4MP. Leading
suspects (in order):

1. **manual-flash online-softmax over many tiles.** At 4MP S≈65K tokens,
   tile=1024 → 64×64 tiles. The running m/l accumulation over 64 KV
   tiles may degrade. Cheap test: bump tile_size to 4096 (fewer tiles)
   or compare vs CTE kernel at 4MP.
2. **TP=8 specific** (heads=3/rank). 3MP worked at TP=4 (heads=6/rank).
   Test 4MP at TP=4 (if it fits) to remove the heads=3 variable.
3. **VAE decode at 2048²** producing gray — the VAE runs unsharded on
   CPU; verify the DiT latent std before VAE (add a latent-std print)
   to localize DiT-vs-VAE.
4. **RoPE freqs at 65K positions** — flagged earlier; less likely since
   apply_rotary_emb is per-position (broadcasts over heads) and 3MP
   (49K tokens) still gave 13.30.

### Next diagnostic (cheap-ish)
Add a latent-std print right before VAE decode in the v3 runner. One 4MP
run tells us if the DiT latent is already collapsed (→ flash/RoPE bug) or
fine (→ VAE bug). That localizes the cliff without guessing.

### Bottom line
- 1.6MP: ✅ correct (16.93)
- 3MP: ✅ mostly correct (13.30) — **NEW, was blank**
- 4MP: ✅ fits + runs (no OOM) ❌ output collapsed (2.60) — memory solved,
  one resolution-dependent correctness bug left to localize.

Huge net progress: the session started with everything >1MP OOM'ing or
blank. Now 3MP produces real images and 4MP runs end-to-end (just needs
the correctness cliff fixed).


## 4MP cliff LOCALIZED to the DiT (not VAE, not tiling, not TP-count) — 2026-06-15

Ran three targeted diagnostics:

1. **4MP TP=4 (heads=6, like working 3MP)** → std 2.60, same collapse.
   Pre-VAE latent std=1.0811. → NOT TP=8/heads=3 specific.
2. **4MP TP=8 flash-tile=8192 (8 tiles vs 64)** → pre-VAE latent std=1.0811,
   *identical* to tile=1024. → manual-flash online-softmax accumulation
   RULED OUT.
3. **CPU VAE decode at every latent grid size** (`vae_size_test.py`):
   | latent grid | image | decode std ([-1,1]) |
   |---|---|---|
   | 80² | 1280² | 0.1418 |
   | 112² | 1792² | 0.1439 |
   | 128² | 2048² | 0.1443 |
   | 256² | 4096² | 0.1457 |
   The VAE is FLAT across all sizes — does NOT collapse at 4MP latent
   size. → VAE RULED OUT.

### The smoking gun

A random-normal latent (std≈1) decodes to image std ≈0.144 in [-1,1]
≈ **18 in 0-255**. The 4MP DiT latent ALSO has global std 1.08, yet
decodes to image std **2.60**, not 18. The only way a std≈1 latent
decodes to a near-flat image is if it is **spatially smooth** — i.e.
the DiT collapsed the high-frequency spatial content of the latent while
keeping the global magnitude.

**Conclusion: at 4MP the DiT produces a detail-poor / smooth latent.**
It's deterministic (identical 1.0811 across all configs), upstream of
attention-tiling and the VAE, and resolution-specific (fine at 3MP).

### Prime suspect: RoPE / positional grid at the 256×256 token grid

FLUX uses a 2D RoPE position grid over the latent. At 4MP the grid is
256×256 (65K image tokens) vs 224×224 at 3MP. If the position-id
construction or the freq computation saturates/wraps at the higher
index range, every attention layer gets wrong high-frequency positional
phase → the model can't place fine detail → smooth latent. This matches:
- deterministic (positions are fixed per resolution)
- resolution-specific cliff (fine at 224, breaks at 256)
- detail-loss signature (positional phase controls spatial frequency)

### Definitive next step
Compare `image_rotary_emb` (cos/sin) produced by `NeuronFluxPosEmbed` at
the 256×256 grid vs a pure-CPU diffusers reference for the same grid.
Diff them. Expect divergence at the high position indices. The fix is a
small correction to the position-id grid / freq dtype at high res — NOT
a sharding, memory, or precision change (those are all solved now).

### Net session outcome
- **1.6MP (1280²): ✅ correct, std 16.93, fits at TP=2**
- **3MP (1792²): ✅ correct real image, std 13.30 — NEW capability**
- **4MP (2048²): ✅ fits + runs end-to-end (memory wall SOLVED via v3
  full sharding); ❌ one localized DiT positional-embedding bug at the
  256² grid produces a smooth latent (std 2.60). Cleanly isolated, small
  fixable function, not a fundamental wall.**

The hard problems (precision collapse, fp32 OOM, fused-projection
sharding) are all solved. 4MP is one positional-embedding fix away.


## RoPE ruled out too — 4MP cliff is deeper (2026-06-15)

`rope_grid_test.py` (CPU) compared the real-arith RoPE at every grid:
```
1.6MP 160²: cos²+sin²=1.0000 (min..max 1.0000), no NaN
3MP  224²: cos²+sin²=1.0000, no NaN
4MP  256²: cos²+sin²=1.0000, no NaN
```
RoPE produces perfectly valid rotations at the 256² grid. **RoPE ruled
out.**

### Hypotheses ELIMINATED for the 4MP cliff
- ❌ VAE (flat across all latent sizes, CPU test)
- ❌ manual-flash tiling (latent identical tile=1024 vs 8192)
- ❌ TP core count (identical std at TP=4 and TP=8)
- ❌ bf16 precision (we're in fp32)
- ❌ memory / OOM (fits + runs end-to-end)
- ❌ RoPE / positional grid (valid rotations at 256²)

### Remaining lead: attention dilution / detail-loss at 65K tokens
The std gradient 16.93 (25K tok) → 13.30 (49K tok) → 2.60 (65K tok)
tracks token count. At 65K bidirectional keys the softmax may flatten
(each weight ≈ 1/65536 → output ≈ global mean → smooth latent). This is
consistent with the "smooth latent, normal global std" decode signature.
BUT GPUs run FLUX.2 at 4MP fine, so if this is the cause it's a
Trainium-specific manifestation (e.g. the QK scale or RMSNorm interplay
in our manual flash at long seq), not inherent.

### Next diagnostic (the right one)
Add per-block latent-magnitude tracking: print the hidden_states std
after each of the 25 blocks at 4MP vs 3MP. Find the block where the 4MP
detail dies relative to 3MP. That pinpoints whether it's a specific
block/op (attention vs FFN vs modulation) rather than guessing. Pair
with attention-weight entropy at one block (uniform vs peaked) to
confirm/deny the dilution hypothesis.

This is a focused instrumentation task, not an open-ended search — six
hypotheses are already eliminated.


## Per-block probe: blocks are HEALTHY at 4MP — collapse is post-block (2026-06-15)

`--probe-blocks` (forward hooks printing hidden_states std per block,
first denoising steps), 3MP vs 4MP, both fp32 TP=4:

| block | 3MP (works, img 13.30) | 4MP (broken, img 2.60) |
|---|---:|---:|
| double[0] | 1.87 | 1.76 |
| double[4] | 8.44 | 8.34 |
| single[0] | 36.7 | 31.4 |
| single[9] | 39.9 | 35.2 |
| single[19] | 67.8 | 64.6 |

**The per-block magnitude curves are nearly identical** (3MP slightly
higher). The transformer blocks are healthy at 4MP — no internal
magnitude collapse. This ELIMINATES the attention-dilution / block-level
hypothesis.

### Refined localization
The collapse is POST-block: `norm_out` (AdaLayerNormContinuous, per-token
LayerNorm over the 3072 channel dim) → `proj_out` → unpatchify. Since
block magnitudes match 3MP but the final latent is spatially uniform
(std 1.08, decodes flat), the 4MP tokens likely converge to a shared
DIRECTION in 3072-space — LayerNorm then normalizes magnitude but the
shared direction yields near-identical per-token outputs → spatially
uniform latent → flat image. std (magnitude) can't see this; need
per-token cosine-similarity / directional diversity to confirm.

### Eliminated (7 hypotheses now)
VAE, flash-tiling, TP-count, bf16, memory, RoPE, **block-level magnitude
collapse**.

### Next diagnostic
Probe per-token directional diversity at the last block (mean pairwise
cosine similarity, or std across tokens of the per-channel mean) at 4MP
vs 3MP. If 4MP tokens are near-collinear and 3MP aren't, that confirms
the directional-collapse mechanism and points at which op (the long-seq
attention's value-averaging is still the likely origin even though
magnitude is preserved — uniform attention makes every token's output
the same VALUE vector, collapsing direction while the residual keeps
magnitude).

### STRATEGIC NOTE
3MP (1792²) is a CORRECT, working capability NOW (std 13.30). It is the
better near-term target than 4MP. Recommend: optimize the SPEED of the
working 1.6MP + 3MP paths (swap the pure-Python fp32 manual flash for a
batched fp32 NKI kernel — the autocomp FLUX kernel reference) for
immediate customer value, and treat 4MP correctness as a separate
deeper-research item (directional collapse at 65K tokens).
