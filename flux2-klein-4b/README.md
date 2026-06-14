# FLUX.2-klein-4B on AWS Trainium2

**Latest (2026-06-14): 4.19 s warm per image at 1024², lossless,**
via VAE-on-Neuron + PAVE fixes + mixed-flag compile. **~25% cheaper
$/image than H100** at full-instance throughput utilization on
trn2.48xlarge (32 logical cores, apples-to-apples 4-step comparison;
H100 4-step latency extrapolated from measured 28-step at 0.218 s/step).

[FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
is Black Forest Labs' fastest distilled 4B DiT — designed for 4-step
sub-second generation on GPU. On Trainium2, it runs end-to-end via
`torch.compile(backend="neuron")` with native PyTorch (no NxDI, no
vLLM). The cost story is competitive at scale.

See
[`native-pytorch/MIXED_FLAG_VAE_NEURON_WIN.md`](native-pytorch/MIXED_FLAG_VAE_NEURON_WIN.md)
for the A/B record (4 measured configurations) and the recipe.

## Cost comparison vs H100 — apples-to-apples 4-step

| Path | Wall-clock | $/image | vs H100 4-step ($0.00105) |
|---|---:|---:|---:|
| H100 single GPU (4-step extrapolated) | 0.87 s | $0.00105 | baseline |
| trn2.48xl + prior shipped (5.92 s) | 5.92 s | $0.00110 | 5% more expensive |
| **trn2.48xl + this PR (4.19 s)** | **4.19 s** | **$0.00078** | **~25% CHEAPER** ✅ |

(H100: single GPU at $4.326/hr. Trainium: trn2.48xl at $21.50/hr,
$/image divided by 32 logical cores at full throughput utilization.)

A measured 4-step H100 baseline is on the followups list — we're
quoting the linear extrapolation from a measured 28-step run, which
slightly *understates* H100's edge because some H100 fixed overhead
amortizes over more steps. So 25% cheaper is the optimistic Trainium
read; the true number after a measured H100 4-step run will likely
be a few points lower.

> The earlier 28-step legacy bench is preserved at the bottom of this
> doc for historical context, but the percentages in those tables
> were computed against the 28-step H100 cost ($0.0073), not the
> apples-to-apples 4-step cost ($0.00105). Use the table above.

---

### Legacy table (28-step bench, 28-step H100 baseline) — historical only

The numbers below were computed against the **28-step H100 cost
($0.0073/image)**, not the apples-to-apples 4-step cost ($0.00105).
The %-cheaper claims are therefore overstated and should not be used
in customer conversations — refer to the 4-step table above instead.
Kept for traceability only.

| Configuration | $/image | vs H100 28-step ($0.0073) |
|---|---:|---:|
| **trn2.48xl + prompt cache + 4 steps** (model's intended use) | **$0.0016** | **78% CHEAPER** ✅ |
| **trn2.48xl × 32 cores, 4 steps (full pipeline)** | **$0.0061** | **16% CHEAPER** ✅ |
| **trn2.48xl + prompt cache + 12 steps** | **$0.0026** | **64% CHEAPER** ✅ |
| **trn2.48xl + prompt cache + 28 steps** | **$0.0060** | **18% CHEAPER** ✅ |
| trn2.48xl × 16 cores (28 steps, no caching) | $0.0068 | 7% cheaper ✅ |
| trn2.48xl × 4 cores (28 steps) | $0.0091 | 1.24× more expensive |
| trn2.3xl batch parallel (28 steps) | $0.024 | 3.3× more expensive |

(H100: single GPU at $4.326/hr = $0.0073/image at 28 steps.
Trainium2: trn2.48xlarge $21.50/hr, trn2.3xlarge $2.23/hr.)

**Key insight:** FLUX.2-klein-4B is already a distilled model — it's
designed for 4-step generation. Each Trainium2 logical core runs an
independent pipeline. On a trn2.48xlarge (32 logical cores), amortizing
the instance cost across concurrent workloads beats H100 on $/image.
Adding prompt caching (encoding text once, reusing embeddings) eliminates
the CPU overhead and makes the gap even wider.

## TL;DR — which path do I want?

```
                    ┌─────────────────────────────────┐
                    │ FLUX.2-klein-4B on Trainium2    │
                    └────────────────┬────────────────┘
                                     │
              ┌──────────────────────┴──────────────────────┐
              │                                             │
   "Just FLUX.2-klein, fastest"                "FLUX inside multi-modal omni"
              │                                             │
              ▼                                             ▼
       native-pytorch/                                vllm-omni/
       trn2.48xl + prompt caching                     needs omni engine
       4.19 s/img, $0.00078 (~25% cheaper than H100)  shared scheduler/KV
                                                      + other modalities
```

## Validated results (measured on hardware)

| What we tested | Result |
|---|---|
| Single-core compile (28 steps, 1024²) | 56.8 s/img, 2030 ms/step |
| 4-core batch parallel on trn2.48xl | 60.7 s for 4 imgs, $0.0091/img |
| 16-core batch parallel on trn2.48xl | ~183 s for 16 imgs, **$0.0068/img** |
| Steady-state (5 consecutive, no reload) | 56.6-57.6 s/img (rock-stable) |
| NEFF execution per step (tqdm) | **960 ms** (CPU overhead is the rest) |
| neuron-profile MFU | **51.8%** (balanced compute/bandwidth) |
| Step sweep (4/8/12/16/20/28 steps) | All working, 4-step = 42.5 s |
| Resolution sweep (256-1280²) | All working, scales O(n²) as expected |
| inf2 test | ❌ BLOCKED (DMA transpose not supported on inf2 hardware) |
| TP=2 test | ❌ BLOCKED (LNC architecture prevents sharing cores) |

## What we learned from profiling

The DiT forward takes 960 ms on Neuron. Breakdown:
- **Tensor engine (matmuls): 258 ms** — 51.8% MFU, well-utilized
- **DMA (weight loading): 272 ms** — memory-bandwidth bound
- **Vector engine (norms, activations): 170 ms**
- **Scalar (control/scheduling): ~200 ms**

**Conclusion:** Model is balanced between compute and bandwidth. FP8
quantization (halving 22.7 GB weight traffic) is the #1 lever for
further improvement. NKI custom kernels are NOT worth pursuing
(compiler already achieves good MFU). See
[`PROFILING_RESULTS.md`](native-pytorch/PROFILING_RESULTS.md).

## Repository layout

```
flux2-klein-4b/
├── README.md                          # this file
├── native-pytorch/                    # the recommended path
│   ├── README.md                      # usage + architecture + results
│   ├── BENCHMARK_VS_H100.md          # full cost analysis + scaling data
│   ├── PROFILING_RESULTS.md           # neuron-profile analysis
│   ├── src/
│   │   ├── neuron_flux2_klein_native.py    # 10-patch pipeline subclass
│   │   ├── run_flux2_klein_native.py       # single-core CLI runner
│   │   └── run_batch_parallel.py           # multi-core batch runner
│   └── results/
│       ├── flux_example1_neuron.png        # sample outputs
│       ├── flux_compiled_cached.png
│       ├── flux_batch_core0.png
│       └── flux_batch_core1.png
└── vllm-omni/                         # for multi-modal omni serving
    ├── README.md
    ├── BENCHMARK.md
    └── src/ + results/
```

## Next steps (optimization roadmap)

| Priority | Optimization | Expected gain | Effort |
|---|---|---|---|
| 1 | **Prompt caching in serving** | Eliminates 30s CPU overhead | Done (diffusers supports `prompt_embeds` kwarg) |
| 2 | **Use 4 steps** (model's intended operating point) | 7× fewer steps = 7× less Neuron time | Zero code changes |
| 3 | **FP8 quantization** (`FLUX.2-klein-4b-fp8` from BFL) | 15-20% per-step speedup | 1-2 days (if neuronxcc supports FP8 matmul) |
| 4 | **FLUX.2-klein-9B** (larger variant) | More compelling story (doesn't fit consumer GPU) | Same pipeline code, just swap model ID |
| 5 | Compiler flag tuning | 5-10% | Experimental |

## Validation

- **Native PyTorch:** trn2.3xlarge + trn2.48xlarge, Beta 3 DLC, 2026-06-13
- **vLLM-Omni:** trn2.48xlarge, vllm-omni 0.19.0rc1, 2026-06-13
- **H100:** single GPU at $4.326/hr, torch 2.12, CUDA 13.0
- **inf2:** tested and blocked (DMA transpose limitation)

## License

Apache-2.0 (contrib code). Model weights:
[FLUX.2 Community License](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).
