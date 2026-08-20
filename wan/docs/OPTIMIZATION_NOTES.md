# Optimization notes — Wan 2.2 TI2V-5B on Trainium2

## Starting point
The original path used trace-based weight-baked graphs (XLA-style). We ported to **native PyTorch**
(`torch.device("neuron")` + `torch.compile(backend="neuron", dynamic=False)`) for a cleaner,
shim-free baseline to optimize on.

Model: 30 layers, 24 heads, head_dim 128 (dim 3072), in/out 48 channels, text_dim 4096,
ffn 14336 (GELU-tanh), patch (1,2,2), qk_norm=rms_norm_across_heads, interleaved RoPE. DiT seq ≈ 5070.

## The wins, in order of impact

### 1. RoPE scatter-free rewrite (the unlock)
The interleaved rotary embedding wrote results with strided index-assignment
(`out[...,0::2]=...; out[...,1::2]=...`), which lowered to an int32 scatter that crashed the compiler
*and* forced a pathologically slow eager path. Rewriting it as
`out = torch.stack([o1, o2], dim=-1).flatten(-2)` (identical interleave via concat+reshape) is
**bit-identical** and unblocked everything:
- eager per-forward 9869 → 1773 ms
- `torch.compile` then works: → **943 ms/forward**

### 2. Tensor parallelism (TP=4) via DTensor
- ColwiseParallel on `to_q/to_k/to_v` and `ffn.net.0.proj`; RowwiseParallel on `to_out.0` and `ffn.net.2`.
- `attn.heads → heads/world`.
- **Adaptive across-heads QK-RMSNorm**: all-reduce the sum-of-squares so the normalization denominator
  is the full 3072 even when heads are sharded → numerically exact. Parity vs single-core: **cos 0.9991**.
- On a trn2.3xl under LNC2, the chip exposes 4 logical cores → **TP=4** uses all of them.
- Per-forward: 943 (TP=1) → 575 (TP=2) → **331 ms (TP=4, isolated microbenchmark)**.

### 3. bf16 VAE decode
The causal-conv3d VAE decodes in bf16 at ~14 s, **3× faster than fp32**, parity **PSNR 56.8 dB** vs fp32
(no visible difference). Approaches that did **not** help: a rolling feat-cache (block-wise whole-decoder
compile was 8× slower and a ~34-min non-caching compile) and routing convs through a hand-tuned conv3d
kernel (≈ parity with the compiler's own lowering). bf16 is the right answer.

### 4. TeaCache (step-skipping) — a real but sign-off-gated lever
Diffusion denoises over 50 steps (×2 CFG = 100 DiT forwards). Consecutive steps change the output very
little, so TeaCache estimates the change (from the timestep-modulated first-block input, rescaled by a
model-specific polynomial) and **skips the full forward** when it's below a threshold, reusing a cached
residual. Training-free. Tradeoff: higher skip = faster but drifts from the un-skipped output.
- The runs in `results/` used an **un-calibrated (identity) rescale**, so fidelity drops with skip
  (54% skip → coherent/softer; 74% → hazy). A **calibrated polynomial** recovers fidelity-per-skip and
  is the recommended next step before deploying any TeaCache setting.

## The real-pipeline gap (important)
Isolated per-forward microbenchmarks (331 ms at TP=4) do **not** equal the glued-pipeline cost. The real
glued run is **511 ms/forward** (un-batched) plus per-step overhead (CPU scheduler round-trips, multi-rank
coordination) and T5-on-CPU. Always measure the real pipeline — the honest e2e is **~134 s**, not the
~45 s a microbenchmark would suggest.

## Correctness methodology
- Per-stage parity vs single-core / CPU-fp32 (cosine, PSNR/SSIM).
- Decode-level "no visible difference" gate: **PSNR > 40 dB, SSIM > 0.98** (achieved 56.7 / 0.9994).
- Frame sanity: 49/49 frames, no NaN, dynamic range preserved.
- A `--golden <video>` hook is included for a full frame-level PSNR/SSIM/LPIPS comparison against a
  reference video once available.

## Where the remaining time goes (optimization frontier)
1. **T5-on-CPU = 65 s** — a single-chip memory workaround; disappears on a larger instance (T5 on-device < 1 s).
2. **Per-step pipeline overhead** — CPU scheduler round-trips + multi-rank coordination.
3. **TeaCache calibration** — to make step-skipping usable at high fidelity.
4. **Multi-chip (trn2.48xl)** — the clean path to H100 parity/beat.
