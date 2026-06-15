# FLUX.2-klein-4B on AWS Trainium2

**Latest (2026-06-14, MEASURED):** 4.19 s warm per image at 1024²,
lossless, via VAE-on-Neuron + PAVE fixes + mixed-flag compile. That's
a **−29% Trainium-internal speedup** over the prior shipped 5.92 s
path (still real, still ships).

**Honest H100 comparison (measured):** at 4-step 1024², a single H100
on p5.4xl Capacity Blocks runs the same model in **0.49 s warm /
$0.00059 per image** (measured 2026-06-14 on diffusers 0.39.0.dev),
which is ~8.5× faster latency and ~25% cheaper $/image than our
saturated trn2.48xl. The earlier "Trainium ~25% cheaper" headline
came from a 6.1 s × 4/28 ≈ 0.87 s extrapolation that overcosted H100;
the measured 4-step number is 1.8× faster than that, and that flips
the comparison.

[FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
is Black Forest Labs' fastest distilled 4B DiT — designed for 4-step
sub-second generation on GPU. On Trainium2, it runs end-to-end via
`torch.compile(backend="neuron")` with native PyTorch (no NxDI, no
vLLM). The customer story is honest: H100 is the cheaper / faster
option for FLUX.2-klein-4B at 4-step today; Trainium is competitive
on hourly entry pricing (trn2.3xl at $2.23/hr is the cheapest hourly
box here) and on inventory / availability where H100 capacity is
constrained.

See
[`native-pytorch/MIXED_FLAG_VAE_NEURON_WIN.md`](native-pytorch/MIXED_FLAG_VAE_NEURON_WIN.md)
for the A/B record (4 measured configurations) and the recipe.

## Cost comparison vs H100 — apples-to-apples 4-step (MEASURED)

| Path | Wall-clock | $/image | vs p5.4xl ($0.00059) |
|---|---:|---:|---:|
| **H100 1× p5.4xl Capacity Blocks (MEASURED 4-step)** | **0.49 s** | **$0.00059** | baseline |
| trn2.3xl single-core (single-stream entry) | 4.19 s | $0.00260 | ~4.4× more expensive |
| trn2.48xl + prior shipped (5.92 s, 32-core saturated) | 5.92 s | $0.00110 | ~1.9× more expensive |
| **trn2.48xl + this PR (4.19 s, 32-core saturated)** | **4.19 s** | **$0.00078** | **~1.3× more expensive** |

(H100: p5.4xlarge Capacity Blocks at $4.326/hr.
Trainium: trn2.3xlarge at $2.23/hr (single-stream entry,
1 logical core);
trn2.48xlarge at $21.50/hr, $/image divided by 32 logical
cores at full throughput utilization.)

**Reading the table:**
- **Lowest latency**: H100 (~8.5× faster per image at 1024²).
- **Lowest hourly**: trn2.3xl at $2.23/hr — useful for getting started
  before you hit the volume that justifies a larger box.
- **Cheapest unit cost at scale**: H100 at p5.4xl is ~25% cheaper
  per image than the saturated trn2.48xl. (See
  [`native-pytorch/BENCHMARK_VS_H100.md`](native-pytorch/BENCHMARK_VS_H100.md)
  for the full measurement methodology.)

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

(H100: p5.4xlarge Capacity Blocks at $4.326/hr.
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
       4.19 s/img, $0.00078 (Trainium-internal -29%)  shared scheduler/KV
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
- **H100:** p5.4xlarge Capacity Blocks at $4.326/hr, torch 2.12, CUDA 13.0
- **inf2:** tested and blocked (DMA transpose limitation)

## License

Apache-2.0 (contrib code). Model weights:
[FLUX.2 Community License](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).
