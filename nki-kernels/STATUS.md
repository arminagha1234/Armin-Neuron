# NKI kernels — current status

## decode_hd256 (single-token decode attention, head_dim=256)

**Status:** parity validated, integrated into qwen3.5 model, benchmarked
end-to-end on real Trainium2 hardware. **v1 is correct but ~20% slower
than the eager baseline that neuronx-cc auto-fuses.** Not upstream-ready.

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
  Neuron, falls back to the PyTorch reference on CPU.
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

- **End-to-end on hardware:** kernel compiles and runs cleanly inside
  vLLM-Neuron via `wrap_nki[2](...)`. Generates coherent factual text.
  See `qwen3.5-4b-trainium/BENCHMARK_NKI_VS_EAGER.md` for the full A/B.

### v1 perf result — slower than eager (do not upstream)

```
trn2.48xlarge, TP=4, MAX_LEN=2048, Qwen3.5-4B (head_dim=256), conc=1:

  EAGER (compiler-fused split-K)   79.6 tok/s decode
  NKI v1 (this kernel)             63.7 tok/s decode
  ratio                            0.80×   (NKI is 20% SLOWER)
```

Identical TTFT (5.44s for both — kernel only runs in decode).
Identical correctness on the probe battery.

### Why v1 is slower (hypothesized)

`aws-neuron/nki-library/core/attention/attention_tkg.py` uses several
techniques v1 of this kernel does not:

1. **Online flash-attention softmax** — running max + running sum
   maintained across chunks, no two-pass reduce. v1 does a two-pass
   reduce (sum across F dim, then transpose+reduce across P dim) which
   is the wrong pattern for streaming attention.

2. **No explicit K transpose** — v1 has an extra `nc_transpose` per
   chunk to flip K from (128 ctx, 128 dim) to (128 dim, 128 ctx) so the
   `nc_matmul` partition contraction is on the right axis. The reference
   `attention_tkg` uses a different stationary/moving layout that avoids
   this transpose.

3. **Aggressive inter-iteration fusion** — neuronx-cc compiling the
   eager Python likely fuses the chunked QK+softmax+AV passes into one
   pipelined inner loop without the explicit per-chunk SBUF
   materialization v1 forces.

### v2 plan

To beat eager, v2 needs the flash-attention pattern. Steps:
1. Drop the two-pass softmax → online running max+sum (one pass over
   chunks)
2. Drop the explicit K transpose → use the `attention_tkg` stationary/
   moving layout pattern
3. Pre-allocate output accumulator at the start, run AV multiply
   in-loop with the weights from the same chunk (no separate AV pass)
4. Re-bench. If v2 is faster than eager, file the upstream PR. If not,
   accept that the compiler is already optimal and document the finding.

### Reference materials for v2

- `aws-neuron/nki-library/src/nkilib_src/nkilib/core/attention/attention_tkg.py`
  — flash attention reference (`_MAX_D_HEAD = 128`, doesn't help us
  directly but the pattern transfers)
- `aws-neuron/nki-library/src/nkilib_src/nkilib/core/attention/attention_tkg_design_spec.md`
  — design spec with diagrams of the LNC2 sharding + FA loop
- vllm-neuron internal `vllm_neuron/functional/attention/attention_decode.py`
  — production decode kernel for head_dim ≤ 128

### Integration (current state)

The kernel is wired into `qwen3.5-4b-trainium/src/qwen3_5/model_bf16.py`
behind `QWEN35_NKI_DECODE=1`. When set (and S_decode=1, head_dim=256,
S_ctx % 128 == 0), `forward_decode` calls `call_decode_hd256(...)` per
(batch, head). Default behavior (flag unset) is unchanged — uses the
eager path that all currently-published benchmarks measure.

This means: with the v1 kernel committed but the flag default off,
existing benchmarks remain valid and the NKI kernel is opt-in for
experimentation.

## Roadmap

- [x] Write v1 kernel (correctness, no perf focus)
- [x] Validate parity via nki.simulate (cosine > 0.99998)
- [x] Wire into model + verify end-to-end on hardware
- [x] A/B bench against eager → confirmed v1 is SLOWER
- [ ] Write v2 with flash-attention pattern
- [ ] A/B bench v2; if faster, prepare upstream PR
- [ ] Long-context bench (MAX_LEN=20480 customer shape)
- [ ] Concurrency stress with `MAX_NUM_SEQS=8` (current bench was
      single-stream because serve.sh defaults to MAX_NUM_SEQS=1)
- [ ] Wire into Qwen3.6-27B (same shape contract)
- [ ] If v2 wins → submit to `aws-neuron/nki-library/experimental/attention/`
- [ ] FP8 KV variant (Path D) — fold v_dequant_scale into the kernel scale

## Honest takeaway

The compiler is good. Hand-writing NKI is not automatically a win. The
right question to ask before writing a kernel is "what is the compiler
producing that's suboptimal?" — answering that requires reading the
emitted MLIR / NEFF and spotting concrete inefficiencies. We didn't
do that step before writing v1, so v1 ended up implementing a layout
that compiles cleanly but is no better than what neuronx-cc already
does for the eager path.

For v2 we should:
1. Capture a profile of the eager decode (with neuron-profile)
2. Look at where the time actually goes — DMA traffic, matmul
   utilization, softmax cost
3. Write the v2 kernel to specifically beat that bottleneck

This is the methodology the AWS Neuron team uses for the kernels in
nki-library. It's also exactly what the
`/Users/aghaebra/Downloads/test_kiro/.kiro/skills/neuron-nki-profile-querying`
skill is designed for.
