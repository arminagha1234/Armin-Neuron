# Part C — NxDI End-to-End Integration Results (the real measurement)

This is the measurement that supersedes every prior Part C number: the split-K
decode kernel **wired into NxDI's `compute_for_token_gen`, compiled into the
production decode NEFF, and run on a real trn2 NeuronCore.** Not a microbenchmark,
not a hand-written reproduction of the fallback — the actual serving framework.

## Setup (all on the existing trn2.48xlarge, no new instance)

- Replicated the Neuron DLAMI software stack on the AL2023 host: a Python 3.11
  venv with **NxDI 0.8.16251 + NxD 0.17.26814 + torch-neuronx 2.5.1** (trace-
  capable) + neuronx-cc 2.25 + Neuron runtime lib. (The eager DLC containers
  lacked `torch_neuronx.trace` and NxD; the public pip stack has a glibc-2.35
  dep that AL2023's 2.34 can't meet — solved by pinning the manylinux_2_28
  SDK-2.20-era wheels.)
- Model: `google/gemma-4-31b-it`, TP=4, LNC=2, bf16, seq_len=512.
- NxDI Gemma4 example (`gemma4-31b-it-nxdi`) as the model implementation.
- Kernel wired via monkey-patch of `NeuronAttentionBase.compute_for_token_gen`
  (`nki_decode_patch.py`): reshape NxDI's 4D prior/active KV → head-major
  layout → call `nki_decode_attention_hd256_mh` / `hd512_mh` → reshape back.
- Baseline and NKI compiled into **separate** NEFF dirs; both decode NEFFs
  compiled clean (neuronx-cc, ~203 s each).

## Results (1 batch, 32 warm decode steps, greedy)

| Config | TTFT | TPOT (mean) | decode tok/s/seq |
|---|---|---|---|
| **Baseline** (stock NxDI decode) | 174.7 ms | **30.68 ms** | 32.6 |
| **NKI split-K** (this kernel) | 174.0 ms | **43.53 ms** | 23.0 |

**The NKI kernel made decode ~42% SLOWER end-to-end (30.7 → 43.5 ms).**
TTFT is unchanged (expected — the patch only touches the decode path).

(Baseline matches the example's documented ~165 ms TTFT / ~33 tok/s, confirming
the environment is sound. Output text is gibberish in both runs because
on-device sampling was disabled and inputs were dummy-padded — irrelevant to the
*timing*, which exercises the full attention+MLP per step either way.)

## Why the kernel LOSES end-to-end (the honest root cause)

The standalone Part C benchmark (`FALLBACK_RESULTS.md`) showed the kernel 2.4×
**faster** than a decode fallback. That was real, but measured against the
**wrong baseline**:

1. **The standalone "fallback" was a hand-written torch reproduction** of the
   decomposed attention (per-head bmm + softmax). NxDI's *actual*
   `compute_for_token_gen` is a **compiler-optimized batched** path — all 32
   heads as fused tensor ops the neuronx-cc scheduler parallelizes across the
   systolic array. That real baseline is much faster than my reproduction.
2. **The kernel loops heads sequentially** (`affine_range(NH)` inside the NEFF —
   32 serial iterations). The compiled batched path does them in parallel.
   Serial-per-head loses to the compiler's batched matmul.
3. **The patch adds real graph ops**: `q.reshape`, `K.transpose(1,2)`, the
   prior+active `cat`, and the head-major repack all become NEFF nodes that the
   stock path never pays.

So the per-op "2.4×" never had a chance to translate: it beat an unfused
reference, but inside NxDI the competition is the compiler's fused batched
decode, which is simply better than a hand-written per-head NKI loop in eager
trace form.

## What this means (consistent with the rest of Part C)

This closes the loop honestly:
- **Microbench (1 op vs 1 SDPA call):** kernel 16× — wrong unit.
- **vs hand-written decomposed fallback:** kernel 2.4× — wrong baseline.
- **In NxDI, compiled, on device:** kernel **0.7× (42% slower)** — the truth.

The eager hand-written NKI decode kernel does **not** beat NxDI's compiled
decode path for Gemma4. The framework's batched, compiler-scheduled attention —
even on the head_dim>128 "fallback" path — outperforms a per-head NKI loop.

## What would actually be needed to win

To beat the compiled batched path, the kernel would need to:
1. **Process all heads in a single batched matmul** (no per-head Python loop) —
   i.e. lay out Q/K/V so one `nc_matmul` covers all heads, matching what the
   compiler already does. This is a substantial rewrite, essentially
   reimplementing the batched flash-decode the compiler emits.
2. **Avoid the reshape/transpose overhead** by consuming NxDI's native 4D
   prior/active layout directly in the kernel.
3. Realistically, only pay off if it also fuses something the compiler can't
   (e.g. the QK-norm + softcap + the attention in one kernel) — pure attention
   is already well-served.

That is real flash-decode-kernel engineering (cf. nkilib `attention_block_tkg`),
not a monkey-patch. The honest conclusion: **for Gemma4 decode on this stack,
NxDI's compiled path is the right answer; a hand-written eager NKI decode kernel
is not a speedup.**

## Value delivered regardless

- Proved the **full integration path works**: DLAMI stack on an existing box,
  kernel traces into the production NEFF, compiles, and runs on device.
- Produced a **real, measured** with/without number instead of a projection.
- Caught that the promising microbenchmark did **not** survive production — which
  is exactly why we measured instead of estimating.

## Reproduce

```bash
# on the trn2.48xl, ~/nxdi_v25 venv, LD_LIBRARY_PATH=/opt/aws/neuron/lib
NKI_DECODE=0 TP=4 SEQ_LEN=512 bash ~/nxdi_gemma4/run_nxdi.sh   # baseline
NKI_DECODE=1 TP=4 SEQ_LEN=512 bash ~/nxdi_gemma4/run_nxdi.sh   # NKI patched
```
