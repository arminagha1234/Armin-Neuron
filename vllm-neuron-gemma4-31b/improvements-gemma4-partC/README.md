# Gemma4 31B — Part C: NKI Kernel Improvements

Custom NKI kernels targeting the bottlenecks identified in Part B / PR #1552,
plus the per-layer elementwise fusions that eat HBM bandwidth.

**All kernels below are compiled + validated on device** (Beta 2 DLC,
`torch_neuronx 2.11.3` eager build) against CPU / SDPA references.

---

## TL;DR — What we learned (read this first)

This effort went all the way to a **real NxDI end-to-end measurement on trn2**,
and the honest bottom line reversed the promising microbenchmarks:

1. **Per-op, the kernels look fast** — 4-16× vs a single PyTorch/SDPA call, and
   2.4× vs a hand-written decode fallback (`FALLBACK_RESULTS.md`).
2. **In NxDI, compiled into the production decode NEFF and run on device, the
   split-K decode kernel is ~42% SLOWER** than stock NxDI decode (TPOT 43.5 ms
   vs 30.7 ms; 23 vs 32.6 tok/s/seq). See `e2e_eager/NXDI_INTEGRATION_RESULTS.md`.
3. **Root cause:** the earlier wins were measured against the wrong baseline (a
   single SDPA call, then a hand-written unfused fallback). NxDI's *actual*
   decode path is a **compiler-scheduled batched** attention that parallelizes
   all 32 heads across the systolic array. The kernel loops heads sequentially
   inside the NEFF and adds reshape/transpose overhead, so it can't keep up.

**The number reconciliation (why measuring beat estimating):**
- vs 1 SDPA call → "16×" (wrong unit)
- vs hand-written decomposed fallback → "2.4×" (wrong baseline)
- **in NxDI, compiled, on device → 0.7× (42% slower) — the truth**

**Recommendation:** for Gemma4 decode on this stack, **NxDI's compiled path is
the right answer.** A hand-written eager NKI decode kernel is not a speedup —
beating the compiler's batched flash-decode would require a full batched-matmul
kernel rewrite (no per-head loop), not a monkey-patch. The elementwise/GEMM
kernels (below) are correct and validated but, as the e2e layer analysis showed,
target <5% of layer time or lose to the vendor matmul. **Eager NKI is not the
throughput lever for Gemma4; the levers are in the compiled serving graph.**

> The sections below document the kernels and the earlier per-op/standalone
> results. They are accurate for what they measured, but the **NxDI end-to-end
> result above is the one that reflects production.** Don't quote the per-op
> speedups without the e2e caveat.

---

## Repo sweep — NKI references mined for this work

Searched the workspace + Amazon internal code search (`InternalCodeSearch`) for
NKI kernels that could help. Most useful finds:

| Source | What it gave us |
|---|---|
| **AutoFixer `real_generation.py`** (NeuronAutoFixerAIM) | Confirms Gemma4 / Qwen3.6-35B (head_dim 256) **must** run with `attn_kernel_enabled=False` — the NKI flash-attn kernel rejects head_dim>128 (`NCC_INKI016`). External validation of the exact gap our split-K kernels fill. |
| **NxDI `attention_base.py` + `nxdi_nki_kernels_reference.md`** | Production kernel inventory + flash-attn constraints (`D_k <= 512`, `S_k % 128 == 0`). Confirms decode uses a hand-optimized NKI kernel and prefill uses flash — and neither covers head_dim>128 cleanly. |
| **nkilib `core/subkernels/rmsnorm_tkg.py`** | The fused `activation(rsqrt, scale=1/H, bias=eps)` trick (3 ops → 1) and the reduce-over-partition-via-ones-matmul pattern. **Applied** to our norm kernels (validated, see below). |
| **Official `matrix_multiplication_nki_kernels.py`** | `hoist_load` / `block_free_dimension` / `fully_optimized` matmul tiling. **Applied** in `nki_gemm_opt.py` — closed the fp32 prefill gap from 2.2× to 1.96× (but bf16 still loses to vendor). |
| **Jim's `fused_silu_gate_mlp_nki.py`, `fused_rmsnorm_linear_nki.py`** | The "keep x in SBUF, reuse for both gate+up projections" fusion pattern. **Applied** in `nki_fused_geglu_gemm_opt.py`. |
| **NxDI `flash_fwd` (pr-117 diffusion)** | Online-softmax flash-attention structure (q-tile × kv-tile, running max/sum) — reference for the split-K decode softmax. |
| **AgiSweNeuralRL `nki_flash_attn_2`** | FA2 forward / split-KV / backward task specs — split-KV is the long-context decode pattern; informs future multi-token-gen work. |

**Improvement applied from the sweep:** the nkilib fused-rsqrt activation trick
(`scale`+`bias` inside `nisa.activation(op=rsqrt)`) is now used in
`nki_fused_rmsnorm_residual.py` and `nki_qk_rmsnorm.py`. It collapses the
mean-divide + eps-add + rsqrt from 3 ops to 1. Verified on device (diff 0.00019)
— timing is within noise (the scalar ops on `[P,1]` were already negligible), so
it's kept for instruction-count / code-cleanliness, not a claimed speedup.

---

## The Problem (from Part B + PR #1552)

| Bottleneck | Current | Root Cause |
|---|---|---|
| **Decode attention** | ~350ms/token (SDPA fallback) | head_dim=256/512 exceeds NKI decode megakernel limit of 128 |
| **Norm + residual** | 240 extra HBM DMAs per forward | 4 norms + 2 residuals per layer × 60 layers, each a separate HBM read/write |
| **GeGLU gate** | extra [T, 30720] HBM round-trip/layer | gelu(gate)*up materialized to HBM before down_proj |
| **QK-norm, soft-cap, embed-scale** | separate framework ops | each a standalone elementwise HBM pass |

## NKI Kernels (all validated on device)

| # | Kernel | What it fuses | Max abs diff | Status |
|---|---|---|---|---|
| 1 | `nki_decode_attention_hd256.py` | decode attention, head_dim=256 (SWA layers, 49/60) via 2-way split-K | **0.000001** vs SDPA | ✅ PASS |
| 2 | `nki_decode_attention_hd512.py` | decode attention, head_dim=512 (global layers, 11/60) via 4-way split-K | **0.000000** vs SDPA | ✅ PASS |
| 3 | `nki_fused_rmsnorm_residual.py` | `residual + RMSNorm(x) * w` in one HBM pass | **0.000037** vs CPU | ✅ PASS |
| 4 | `nki_geglu_mlp.py` | `gelu_tanh(gate) * up` in one HBM pass | **0.000003** vs CPU | ✅ PASS |
| 5 | `nki_qk_rmsnorm.py` | QK-RMSNorm over head_dim `* w` in one HBM pass | **0.000256** vs CPU | ✅ PASS |
| 6 | `nki_logit_softcap.py` | `cap * tanh(x / cap)` in one HBM pass | **0.000004** vs CPU | ✅ PASS |
| 7 | `nki_embed_scale.py` | `embeds * sqrt(hidden)` in one HBM pass | **0.000000** vs CPU | ✅ PASS |
| 8 | `nki_decode_attention_hd256_mh.py` | all-heads-in-one-dispatch decode attn, hd=256 (fixes 32× dispatch overhead) | **0.000010** vs torch | ✅ PASS |
| 9 | `nki_decode_attention_hd512_mh.py` | all-heads-in-one-dispatch decode attn, hd=512 (global layers) | **0.000000** vs torch (fp32) | ✅ PASS |
| 10 | `nki_gemm.py` / `nki_gemm_opt.py` | tiled GEMM (naive + hoist-load) | **rel 0.00000** vs torch | ✅ PASS (slower than vendor) |
| 11 | `nki_fused_geglu_gemm.py` / `_opt.py` | `gelu(x@Wg)*(x@Wu)` fully fused (proj+act+mul) | **rel 0.00000** vs torch | ✅ PASS (1.04× decode) |

The decode/elementwise kernels (1-7) are the validated drop-ins. The
multi-head attention kernels (8-9) cover the **full Gemma4 decode attention**
(49 SWA hd256 + 11 global hd512) and beat the real fallback 1.8-3.1× (see
`e2e_eager/FALLBACK_RESULTS.md`). The GEMM/fused-GEMM kernels (10-11) came out of
the throughput investigation — see `e2e_eager/` for why they do/don't help.

### 1 & 2. Split-K Decode Attention (head_dim 256 / 512)

Replaces PyTorch SDPA for decode (single-token). The standard NKI decode
megakernel only supports head_dim ≤ 128; Gemma4's 256 (SWA) and 512 (global)
exceed it, forcing the slow SDPA fallback that #1552 flagged as ~350ms/token.

**Pattern:** split head_dim into 128-dim chunks (2 for hd256, 4 for hd512):
- Score: `sum_chunks(Q_c^T @ K_c)` via `nisa.nc_matmul` accumulating into one PSUM
- Softmax over cached S (free dim)
- Output: `weights @ V` per chunk, accumulated and stored

Q/K are passed **head_dim-major (transposed)** so the 128-partition DMA
constraint always holds regardless of cached sequence length S.

### 3. Fused RMSNorm + Residual

Fuses `residual + RMSNorm(module_output) * weight` into one HBM pass.
Saves 2 HBM round-trips per fusion point × 2 points/layer × 60 layers =
**240 eliminated DMAs**.

### 4. Fused GeGLU

Fuses `gelu_tanh(gate) * up` (Gemma4 uses `gelu_pytorch_tanh`). The
intermediate is [T, 30720] — fusing the gate avoids one full HBM
round-trip of that tensor per layer × 60 layers.

### 5. QK-RMSNorm

Gemma4 RMSNorms the query and key projections (over head_dim=256) before
RoPE. One reusable fused pass for both q_norm and k_norm.

### 6. Logit Soft-Cap

Gemma4 soft-caps attention logits (cap=50) and final logits (cap=30) with
`cap * tanh(x / cap)`. Fuses the divide + tanh + multiply (3 passes → 1).

### 7. Embed Scale

Gemma scales token embeddings by `sqrt(hidden_size)` (≈73.32 for 31B).
Trivial, but fused so it chains with the embedding DMA without a framework op.

## How they were validated (the working recipe)

The vLLM-Neuron v5 beta serving image **lacks `torch_neuronx`**, but the
**Beta 2 DLC** (`concourse-release-0461d3b:latest`) ships `torch_neuronx
2.11.3` (eager build) with `nki_kernel.py` + `nki_hop.py`. The eager build
runs `@nki.jit` kernels via the `wrap_nki` bridge (NOT raw `kernel(...)`,
which needs the missing `pyhlo` trace path):

```python
import torch, torch_neuronx
from torch_neuronx import wrap_nki
from nki_geglu_mlp import nki_geglu_mlp

dev = torch.device("privateuseone:0")   # neuron device in the eager build
wrapped = wrap_nki(nki_geglu_mlp)        # bridge @nki.jit -> torch dispatch
result = wrapped(gate.to(dev), up.to(dev))
torch_neuronx.synchronize()              # only sync needed
result.cpu()
```

Test drivers: `test_nki_wrap.py` (kernels 1, 3), `test_nki_hd512.py`
(kernel 2), `test_nki_more.py` (kernels 4-7) — all in
`customers/Hippocratic/gemma4_vllm/`.

### Key NKI fixes found during on-device bring-up

1. **`nisa.rsqrt` doesn't exist** in this build → use
   `nisa.activation(dst, data, op=nl.rsqrt)`.
2. **`tensor_tensor` / `tensor_scalar` do NOT auto-broadcast.**
   - Per-partition scalar ([P,1] × [P,F]): use `nisa.tensor_scalar(...,
     operand0=rsqrt_tile)`.
   - Per-free weight ([1,H] × [P,H]): physically replicate the weight to
     [P,H] via a broadcast-DMA access pattern
     `weight.ap(pattern=[[0, P], [1, H]])` (partition stride 0).
3. **DMA partition alignment:** the outer (partition) dim of an HBM→SBUF
   copy must be ≤128. For decode attention, load K **transposed**
   (head_dim-half=128 as partition, S as free) so arbitrary S doesn't
   break alignment.
4. **`nc_transpose` is limited to ≤[32,32]** (vector engine). To transpose
   a [1,S] weight row to [S,1], use an **identity matmul**
   (`nc_matmul(stationary=w[1,S], moving=ident[1,1])`) — works for any size.
5. **GeLU is `nl.gelu_apprx_tanh`** (not `nl.gelu_tanh`) in this build —
   matches Gemma4's `gelu_pytorch_tanh`. Plain `nl.gelu`, `nl.silu`,
   `nl.tanh`, `nl.sigmoid` are also present.
6. **PSUM → HBM is illegal directly** → copy PSUM→SBUF with
   `nisa.tensor_copy` first, then `nisa.dma_copy` from SBUF.

## Integration Plan

These kernels need to be wired into `gemma4/model.py`. Integration points:

```python
# Gemma4DecoderLayer.forward — norm+residual fusion
from .nki_fused_rmsnorm_residual import nki_fused_rmsnorm_residual
hidden = nki_fused_rmsnorm_residual(residual, attn_out, self.post_attention_layernorm.weight, 1e-6)

# Gemma4MLP.forward — GeGLU gate fusion
from .nki_geglu_mlp import nki_geglu_mlp
act = nki_geglu_mlp(self.gate_proj(x), self.up_proj(x))
y = self.down_proj(act)

# Gemma4Attention.forward_decode — split-K decode (per head)
from .nki_decode_attention_hd256 import nki_decode_attention_hd256   # SWA layers
from .nki_decode_attention_hd512 import nki_decode_attention_hd512   # global layers
```

## Measured Speedups (on-device microbenchmark)

Each kernel timed head-to-head vs the PyTorch-on-Neuron path that produces the
same result, same device, 50 iters after warmup. Full detail + caveats in
[`RESULTS.md`](./RESULTS.md). Driver: `bench_nki_vs_torch.py`.

| Kernel | NKI (ms) | torch-on-Neuron (ms) | speedup |
|---|---|---|---|
| decode_attn_hd256 (S=512) | 0.083 | 1.371 | **16.5×** |
| decode_attn_hd512 (S=512) | 0.086 | 1.348 | **15.8×** |
| qk_rmsnorm [512,256] | 0.074 | 0.622 | **8.5×** |
| logit_softcap [256,4096] | 0.072 | 0.463 | **6.5×** |
| rmsnorm_residual [512,5376] | 0.159 | 0.768 | **4.8×** |
| embed_scale [512,5376] | 0.079 | 0.308 | **3.9×** |
| geglu [512,30720] | 0.493 | 0.584 | **1.2×** |

The decode-attention win (16×) is the headline — that's the SDPA-fallback path
#1552 flagged as the per-token bottleneck.

> ⚠️ **Read [`e2e_eager/RESULTS.md`](./e2e_eager/RESULTS.md) before quoting these
> numbers.** When we built a full Gemma4 decoder layer and timed the real decode
> path (torch vs NKI, same weights), the end-to-end win was **~1.01×, not 16×**.
> The per-op speedups are real but measured at the wrong granularity: a layer
> has 32 heads (torch batches them into one `bmm`; 32 separate kernel dispatches
> lose), and the norms — while 5-7× faster per-op — are <5% of layer time, which
> is dominated by the projection GEMMs the kernels don't touch. Batching all
> heads into one dispatch closes the attention gap from 19.6× slower to 1.24×
> slower, but torch's batched matmul still wins for decode where SDPA works. The
> kernels are correct, validated building blocks — useful *if* the serving path
> degrades to an SDPA fallback, not a free end-to-end speedup where it doesn't.

## Honest Caveats

1. **Microbenchmarks, not end-to-end serving throughput.** A 16× op-level
   speedup does NOT mean 16× tokens/min — per-token latency also includes QKV/O
   projections, MLP, norms, scheduler, and sampler. The end-to-end gain depends
   on the attention fraction of per-token time (large, per Part B, but not
   100%). See `RESULTS.md` for the full reasoning.

2. **Validated + benchmarked, not yet served.** All 7 kernels compile, are
   numerically correct, and beat the eager baseline on device in the Beta 2 DLC.
   They are **not yet wired into the vLLM serving path** — the vLLM-Neuron v5
   serving image lacks `torch_neuronx`. Serving integration is blocked on a
   platform dependency, not on the kernels. The next achievable step for an
   end-to-end number is a native-PyTorch eager Gemma4 forward in this container,
   benchmarked with vs without the kernels.

3. **Decode kernel sequence length.** The decode kernels tile cached S in
   128-chunks with a Python loop. For very long contexts (>2K cached
   tokens) the loop length grows; correctness holds (validated at S=64) but
   the speedup vs SDPA at long S hasn't been characterized.

## Files

```
improvements-gemma4-partC/
├── README.md                          # this file
├── RESULTS.md                         # per-op microbench speedups
├── bench_nki_vs_torch.py              # per-op microbench driver
│
├── nki_decode_attention_hd256.py      # decode attn, head_dim=256 (2-way split-K)
├── nki_decode_attention_hd512.py      # decode attn, head_dim=512 (4-way split-K)
├── nki_decode_attention_hd256_mh.py   # all-heads-in-one-dispatch decode attn (SWA)
├── nki_decode_attention_hd512_mh.py   # all-heads-in-one-dispatch decode attn (global)
├── nki_fused_rmsnorm_residual.py      # residual + RMSNorm*w (fused rsqrt activation)
├── nki_geglu_mlp.py                   # gelu_tanh(gate) * up (tiled P+F)
├── nki_qk_rmsnorm.py                  # QK-RMSNorm over head_dim (fused rsqrt)
├── nki_logit_softcap.py               # cap * tanh(x/cap)
├── nki_embed_scale.py                 # embeds * sqrt(hidden)
├── nki_gemm.py                        # tiled GEMM (naive)
├── nki_gemm_opt.py                    # tiled GEMM (hoist-load)
├── nki_fused_geglu_gemm.py            # fully-fused GeGLU GEMM (naive)
├── nki_fused_geglu_gemm_opt.py        # fully-fused GeGLU GEMM (hoist-load)
│
└── e2e_eager/                         # the end-to-end investigation
    ├── RESULTS.md                     # full-layer torch-vs-NKI: ~1.01x + why
    ├── GEMM_RESULTS.md                # GEMM + fused-GeGLU: where the win is
    ├── THROUGHPUT_GEMM_RESULTS.md     # bf16 + hoist-load: vendor matmul wins 4-6x
    ├── FALLBACK_RESULTS.md            # NKI vs REAL decode fallback: 1.8-3.1x win
    ├── gemma4_eager_layer.py          # one faithful Gemma4 decoder layer
    ├── bench_e2e_layer.py             # layer torch-vs-NKI + full-model projection
    ├── diag_dispatch.py               # dispatch-overhead + tiny-tensor diagnostic
    ├── test_mh_attn.py                # batched multi-head attention bench
    ├── test_fallback_vs_nki.py        # NKI vs REAL decode fallback (SWA + global)
    ├── test_gemm.py / test_gemm_opt.py / test_gemm_bf16.py
    └── test_fused_geglu_gemm.py
```
