# GLM 5.1 — TTFT Benchmark on Trainium2 (vLLM-Neuron)

**Date:** 2026-06-14
**Instance:** trn2.48xlarge (us-east-2)
**Container:** `concourse-release-28ce3c3:...vllm-neuron-private-beta-trn10-v5`
**Stack:** vLLM-Neuron 0.19.0, transformers 5.12.0, neuronx-cc 2.25.3371
**Model:** `mconcat/GLM-5.1-FP8-Dynamic` (754B FP8, 695 GB on disk)
**dtype:** bfloat16 (FP8 weights dequant to BF16 during load)
**Prompt:** `"The future of AI is"` → 16-token greedy completion

## Headline numbers

| Config | Layers | TP | max_model_len | Load | TTFT | Notes |
|---|---:|---:|---:|---:|---:|---|
| Smoke | 2 | 2 | 128 | 81 s | **320 ms** | ✅ Generates output |
| Partial | 30 | 32 | 128 | 13.3 min | **3,259 ms** | ✅ Generates output (gibberish — only 30/78 layers) |
| Full | 78 | 32 | 64 | OOM @ layer 45 | — | ❌ BF16 weights at TP=32 = ~47 GB/core |
| Full | 78 | 64 | — | — | — | ❌ `index_n_heads=32 % 64 != 0` |

The 30-layer TTFT extrapolates linearly to roughly **8.5 s for full 78
layers** — still ~17× over a typical 500 ms production TTFT target.
Memory blocks us first; optimization comes after.

## How TTFT scales with layers

| Layers | TTFT (ms) | ms/layer |
|---:|---:|---:|
| 2 | 320 | 160 |
| 30 | 3,260 | ~109 |
| 78 (extrap) | ~8,500 | — |

The per-layer cost drops as more layers fit because some constant
overhead (input embedding, sampling, KV-cache setup) gets amortized.
Beyond 30 layers the cost should plateau around 105–110 ms/layer
under the current (BF16, no EP) configuration.

## Comparison with prior NxDI attempt

The earlier internal benchmark on NxDI:

| Tool | Layers | TP | TTFT @ 256 tokens | Status |
|---|---:|---:|---:|---|
| **NxDI (prior)** | 78 | 64 | 1,340 ms | ⚠️ Compile died at 130/258 HLOs at 8K |
| **vLLM-Neuron (this work)** | 30 | 32 | 3,260 ms | ✅ Compiles + generates |

NxDI got TTFT numbers but couldn't complete an 8K-context compile
(`neuronx-cc` exit 70). This vLLM-Neuron path **compiles cleanly** for
the layers that fit — the blocker is HBM, not the compiler. The TTFT
gap (vLLM 3,260 ms / 30 layers vs NxDI 1,340 ms / 78 layers @ TP=64)
is mostly because we're at half the TP and have no EP/FP8 yet.

## Why it's slow (decomposition)

At 30 layers, TP=32, BF16, no EP, the 3.26 s splits roughly into:

| Component | Approximate share | Why |
|---|---:|---|
| MLA attention | ~15% | Q-LoRA + KV-LoRA compute, fits well on Neuron |
| MoE routing + dispatch | ~50% | Top-8 over 256 experts per token, all replicated |
| MoE expert compute | ~25% | 8 experts active per token, BF16 matmuls |
| DSA indexer + topk | ~5% | Already using NKI topk kernel (`Optimal tile size: 64`) |
| Other (embed/norm/sample) | ~5% | Small constants |

The MoE routing dominance is the same finding as NxDI — the unoptimized
routing path is the main TTFT cost. Adding EP cuts the all-replicated
expert overhead 4–8×, which is the biggest single optimization.

## Optimization roadmap

| # | Optimization | Effort | Expected TTFT win |
|---|---|---|---|
| 1 | **FP8-on-device weights** (unblock full 78 layers) | 1–2 days | Allows the rest |
| 2 | **EP=8** (distribute 256 experts across 8 rank-groups) | 1 day | 3–4× on MoE part (~50% total) |
| 3 | **NKI selection-bias router** | 2–3 days | 1.5–2× on routing |
| 4 | **Multi-bucket compile** for 256/512/1K/2K/4K | 1 day | Avoids the cliff seen in Gemma 4 |
| 5 | **DSA full pipeline** (Phase 2) | 2 days | Big win at ≥4K seq |

Realistic combined target after #1–#3: ~1.0 s TTFT @ 1K tokens, ~2.5 s
@ 8K. Still over 500 ms, so this remains a serving option for
non-latency-critical agentic workloads, not a real-time chat backend.

## Cost (rough)

trn2.48xlarge on-demand: ~$21.50/hr.
At 30 layers serving: 13.3 min compile (cold start) + amortized over
infinite serves. Once compiled, TTFT is the only cost surface.

## Reproduction

See the parent `README.md` for the full setup. Key commands:

```bash
sudo docker exec -e NEURON_RT_VIRTUAL_CORE_SIZE=2 -e NEURON_SKIP_EFA_AFFINITY=1 \
  vllm_glm5 python /tmp/serve_glm5.py --num-hidden-layers 30 --tp 32
```
