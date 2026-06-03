# Part C — On-Device Microbenchmark Results

**Question asked:** are the NKI kernels actually faster than letting PyTorch run
the unfused ops on Neuron?

**Answer:** yes, for every kernel. Measured on the Neuron device in the Beta 2
DLC (`torch_neuronx 2.11.3` eager build), `privateuseone:0`.

## Method

Each NKI kernel is timed head-to-head against the **PyTorch-on-Neuron** path
that produces the same result (the same eager path the SDPA fallback uses in
serving). Both run on the same device. 5 warmup iters (first call compiles the
NEFF), then 50 timed iters with a single `torch_neuronx.synchronize()` at the
end of the batch. Times are ms/iter.

Driver: `customers/Hippocratic/gemma4_vllm/bench_nki_vs_torch.py`

Shapes mirror Gemma4 31B (hidden=5376, intermediate=30720, head_dim=256/512).

## Results

| Kernel | shape | NKI (ms) | torch-on-Neuron (ms) | speedup |
|---|---|---|---|---|
| decode_attn_hd256 | S=512 | 0.083 | 1.371 | **16.5×** |
| decode_attn_hd512 | S=512 | 0.086 | 1.348 | **15.8×** |
| qk_rmsnorm | [512,256] | 0.074 | 0.622 | **8.5×** |
| logit_softcap | [256,4096] | 0.072 | 0.463 | **6.5×** |
| rmsnorm_residual | [512,5376] | 0.159 | 0.768 | **4.8×** |
| embed_scale | [512,5376] | 0.079 | 0.308 | **3.9×** |
| geglu | [512,30720] | 0.493 | 0.584 | **1.2×** |

(All numerically validated separately — diffs in README table, all ≤ 0.000256.)

## Reading the results

- **Decode attention is the headline (16×).** This is the SDPA-fallback path
  PR #1552 flagged at ~350ms/token for Gemma4's head_dim=256/512. The split-K
  kernel keeps the whole op on the matmul + vector engines with no fallback.
  At S=512 cached tokens it's ~16× the eager SDPA path.
- **Elementwise fusions land 4-8×** because they collapse several framework
  dispatches and HBM round-trips into one pass. `qk_rmsnorm`, `logit_softcap`,
  `rmsnorm_residual`, `embed_scale` are all bandwidth/dispatch bound — exactly
  what fusion fixes.
- **GeGLU is the smallest win (1.2×)** and that's expected: the [512,30720]
  intermediate is so wide the op is dominated by *moving* the tensor, not by
  dispatch. Fusing the gelu+multiply saves the dispatch + one logical HBM
  round-trip but the byte movement dominates, so the ceiling is low. Still net
  positive.

## Honest caveats

1. **These are isolated-op microbenchmarks, not end-to-end serving
   throughput.** A 16× speedup on the decode-attention op does NOT mean 16×
   tokens/min — decode also does QKV/O projections, MLP, norms, and the
   scheduler/sampler overhead. The end-to-end gain depends on what fraction of
   per-token latency is attention. From Part B, decode attention is the single
   largest component (~350ms of the per-token time was the SDPA fallback), so
   the attention win should translate to a meaningful — but not 16× — token/min
   improvement.

2. **Still not wired into vLLM serving.** The vLLM-Neuron v5 serving image lacks
   `torch_neuronx`, so these can't load in the serving path yet. To get an
   end-to-end number we either (a) wait for a serving image with `torch_neuronx`
   matching the serving torch, or (b) stand up a native-PyTorch eager Gemma4
   forward in this Beta 2 container and benchmark per-token latency with vs
   without the kernels. Option (b) is the next achievable step.

3. **torch-on-Neuron eager is the baseline, not a hand-optimized graph.** The
   comparison is against the same eager path serving falls back to — which is
   the right baseline for "does the kernel help vs what we have today" — but a
   fully compiled torch.compile graph could narrow some of the elementwise gaps.
   The decode-attention win is robust regardless because the alternative there
   is a genuine fallback, not a compiled kernel.

## Reproduce

```bash
# in the beta2_nki container, /work mounted from /data/work
docker exec beta2_nki bash -lc 'cd /work && python3 bench_nki_vs_torch.py'
```
