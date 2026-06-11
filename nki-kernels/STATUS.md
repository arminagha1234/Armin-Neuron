# NKI kernels — current status

## decode_hd256 (single-token decode attention, head_dim=256)

**Status:** parity validated, integrated into qwen3.5 model with feature flag,
device-perf benchmark pending.

### What's done

- **Math contract:** `armin_nki_kernels/attention/ref_decode_hd256.py`
  Pure-PyTorch reference — split-K + split-V matmul pair, fp32 softmax.
  This is the parity oracle.
- **NKI kernel:** `armin_nki_kernels/attention/decode_hd256.py`
  Fused QK + softmax + AV in one NEFF. Tiles ctx into 128-token chunks,
  PSUM-accumulates the split-K halves, fp32 softmax with two-pass reduce,
  PSUM-accumulates the split-V halves across all chunks.
- **Wrapper:** `armin_nki_kernels/attention/decode_hd256_wrap.py`
  vllm-neuron-style adapter — uses `vllm_neuron.nki.nki_hop.wrap_nki` on
  Neuron, falls back to the PyTorch reference on CPU. Pre-computes the
  fp32 mask bias on CPU before moving to device (the bool→fp32 cast
  is quirky on neuron device).
- **Tests:**
  - `tests/test_decode_hd256_parity.py` — pytest sweep (CPU fallback).
  - Direct simulate sweep — 6 shapes from S_ctx=128 to S_ctx=4096:

    | shape          | S_ctx | valid | cosine vs ref |
    |----------------|-------|-------|---------------|
    | smoke          | 128   | 64    | 0.999991      |
    | 4B_short       | 128   | 100   | 0.999988      |
    | 4B_typical     | 512   | 400   | 0.999992      |
    | 27B_typical    | 512   | 400   | 0.999992      |
    | 4B_2K          | 2048  | 1500  | 0.999987      |
    | 4B_chunked     | 4096  | 2048  | 0.999986      |

    All > 0.999 threshold.

### Integration into Qwen3.5-4B

The kernel is wired into `qwen3.5-4b-trainium/src/qwen3_5/model_bf16.py`
behind the env flag `QWEN35_NKI_DECODE=1`. When enabled (and S_decode=1,
head_dim=256, S_ctx % 128 == 0), `forward_decode` calls
`call_decode_hd256(q, k_full, v_full, mask_bias, scale)` per (batch, head)
instead of the eager split-K + split-V matmul pair. Default behavior
(flag unset) is unchanged — the eager path the existing benchmarks use.

### What's NOT done yet

- **Real Neuron device parity:** the wrap_nki HOP only resolves during
  vllm-neuron's FX trace, not in eager mode. Direct calls from a vanilla
  Python module produce
  `NotImplementedError: could not find kernel for HigherOrderOperator
  nki_kernel_wrapper at DispatchKey.PrivateUse1`. End-to-end model
  generation is the only way to actually test on hardware.
- **End-to-end perf benchmark:** to measure speedup vs. the eager path,
  need to recompile the 4B model with `QWEN35_NKI_DECODE=1` and re-run
  the bench scripts that produced `BENCHMARK_TRN2_48XL.md`. Compile is
  ~10 min on trn2.48xl. Once done, expected speedup is most pronounced
  at long context (S_ctx ≥ 4K) where the fused NEFF cuts intermediate
  HBM traffic from 3 buffers (scores, weights, out_lo, out_hi) down
  to just the final out.
- **Quantitative claims:** currently we have parity. Speedup is
  hypothetical until the bench runs. Will update this file once data
  is in.

### How to A/B benchmark

```bash
# A — eager (existing baseline)
QWEN35_NKI_DECODE=0 ./serve.sh
python qwen3.5-4b-trainium/test/bench_qwen35_long.py
# → record TTFT, decode tok/s

# B — NKI (new path)
QWEN35_NKI_DECODE=1 ./serve.sh
python qwen3.5-4b-trainium/test/bench_qwen35_long.py
# → record TTFT, decode tok/s

# Diff and update BENCHMARK_TRN2_48XL.md
```

Or (cleaner) split into two compile artifacts so the A/B is one bench
script with the URL switched.

## Roadmap

- [ ] End-to-end perf benchmark (4B + 27B)
- [ ] Wire into Qwen3.6-27B `customers/Makora_27B/.../model_bf16.py`
      (same shape contract — head_dim=256, GQA-repeated K/V, single decode token)
- [ ] If speedup is real and material → graduate to upstream
      `vllm-neuron/vllm_neuron/functional/attention/`
      and make a PR alongside the Path B PR (#2104)
- [ ] Multi-token decode variant (S_q > 1) — currently kernel asserts S_q==1
- [ ] FP8 KV variant (Path D) — fold v_dequant_scale into the kernel scale
