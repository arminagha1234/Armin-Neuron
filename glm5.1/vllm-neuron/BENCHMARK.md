# GLM 5.1 — TTFT Benchmark on Trainium2 (vLLM-Neuron)

> ⚠️ **THIS IS A BASELINE — WE CAN IMPROVE IT SUBSTANTIALLY.**
> The numbers below are a first-light bring-up of GLM 5.1 on Trainium2
> using BF16-dequantized weights and no MoE-specific optimizations.
> The single biggest lever (FP8-on-device weight storage) is not yet
> applied. Expect large gains once the optimization roadmap below is
> executed. Treat these as a floor, not a representative result.

**Date:** 2026-06-14
**Instance:** trn2.48xlarge (us-east-2)
**Container:** `concourse-release-28ce3c3:...vllm-neuron-private-beta-trn10-v5`
**Stack:** vLLM-Neuron 0.19.0, transformers 5.12.0, neuronx-cc 2.25.3371
**Model:** `mconcat/GLM-5.1-FP8-Dynamic` (754B FP8, 695 GB on disk)
**dtype:** bfloat16 (FP8 weights dequant to BF16 during load — see blocker)
**Harness:** identical to `vllm-neuron-gemma4-31b/bench_ttft.py` (same
unique-random-token prompts, streaming TTFT, median of N, same seq-len
scan). Re-used byte-for-byte so numbers are directly comparable.

## Headline (baseline — improvable)

| Path | Layers | TP | TTFT | How measured |
|---|---:|---:|---:|---|
| Offline `LLM.generate` smoke | 2 | 2 | **320 ms** | direct generate (max_model_len 128) |
| Offline `LLM.generate` | 30 | 32 | **3,259 ms** | direct generate (max_model_len 128) |
| Served HTTP (gemma harness) | 30 | 32 | **blocked** | server won't start — see below |
| Served HTTP (gemma harness) | 78 | 32 | **blocked** | weights don't fit at all |

The 30-layer offline TTFT of **3,259 ms** is the comparable baseline
number. Extrapolated to the full 78 layers it implies roughly **8.5 s**
TTFT — well over a 500 ms production target, which is exactly why the
optimization roadmap matters.

## Why the served HTTP benchmark is blocked (important)

We tried to run the *exact* Gemma 4 31B served harness (vLLM
`vllm serve` + `bench_ttft.py` against the OpenAI endpoint). The server
fails to finish compiling at **every** bucket size — including the
smallest single `[256]` bucket — with:

```
[NCC_EOOM002] Maximum peak HBM usage of 28.44GB exceeds HBM limit of
24.00GB for Trn2. This consists of 1.50MB model constants,
43.00GB I/O tensors, 6.89GB internal (scratchpad) tensors ...
```

Key insight: the **43 GB I/O-tensor figure is identical at bucket 256
and bucket 4096**. It does NOT scale with sequence length, so it is
not activation memory — it is the **BF16-expanded MoE expert weights**
being materialized as graph I/O on each core. At 30 layers × 256
experts, dequantized to BF16, that's ~43 GB/core — far past the 24 GB
Trn2 budget. No bucket-size tuning can fix this.

The offline `LLM.generate` path *does* run at 30 layers because it uses
a smaller `max_model_len` (128) and a different graph-capture path that
doesn't materialize all expert weights as simultaneous graph I/O.

**Conclusion: the served path needs FP8-on-device weights to fit at all.
The 3,259 ms offline number is our honest baseline until then.**

## Gemma 4 31B comparison (for context — NOT apples-to-apples yet)

| Model | Served TTFT @ ~256 | Notes |
|---|---:|---|
| Gemma 4 31B (dense, served, multi-bucket) | ~102 ms | dense 31B, fits cleanly |
| GLM 5.1 (754B MoE, offline, 30/78 layers) | 3,259 ms | partial model, BF16 dequant, no EP |

Gemma 4 is a 31B dense model that fits and serves cleanly. GLM 5.1 is a
754B MoE — a fundamentally heavier model — and these baseline numbers
reflect a not-yet-optimized bring-up, not the achievable steady state.

## Why it's slow / why it doesn't fit (decomposition)

| Issue | Cause | Fix |
|---|---|---|
| Weights don't fit served | FP8 → BF16 dequant on load, 2× HBM | **FP8-on-device storage** |
| TTFT high | MoE routing replicated on every rank | **Expert parallelism (EP)** |
| TP capped at 32 | `index_n_heads=32` not divisible by 64 | indexer-aware sharding |
| Routing overhead | unoptimized top-8 over 256 experts | NKI selection-bias router |

## Optimization roadmap (what "improve" means)

| # | Optimization | Effort | Expected effect |
|---|---|---|---|
| 1 | **FP8-on-device weights** | 1–2 days | Unblocks served path + full 78 layers (halves weight HBM) |
| 2 | **Expert parallelism EP=8** | 1 day | 3–4× on the MoE share of TTFT |
| 3 | **NKI selection-bias router** | 2–3 days | 1.5–2× on routing |
| 4 | **Multi-bucket compile** | 1 day | Cuts effective TTFT on a real payload mix (Gemma saw −41%) |
| 5 | **DSA full pipeline** | 2 days | Big win at ≥4K context |

Realistic post-#1–#3 target: ~1.0 s TTFT @ 1K tokens — a 3×+ improvement
over this baseline, and the starting point for hitting production
latency goals.

## Reproduction (offline 30-layer baseline — known working)

```bash
sudo docker exec -e NEURON_RT_VIRTUAL_CORE_SIZE=2 -e NEURON_SKIP_EFA_AFFINITY=1 \
  vllm_glm5 python /work/serve_glm5.py --num-hidden-layers 30 --tp 32
# → loads in ~13 min, TTFT ~3,259 ms, generates tokens
```

The served HTTP harness (`bench/bench_ttft.py`) is included and is the
exact Gemma 4 31B script; it will produce comparable numbers once the
FP8-on-device weight fix lets the server start.
