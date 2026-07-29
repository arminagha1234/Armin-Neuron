# Sesame CSM-1B (Text-to-Speech) on Trainium

[Sesame CSM-1B](https://huggingface.co/sesame/csm-1b) is a conversational text-to-speech
model: a **Llama-3.2-1B backbone** + a small **depth decoder** (32 residual codebooks per
frame) + the **Mimi** audio codec that turns codes into 24 kHz speech. This folder covers
running it on **AWS Trainium2** and cutting its latency, with every performance number
measured on-device.

There are two workstreams here:

| Folder | What it delivers | Status |
|---|---|---|
| **[native-pytorch/](native-pytorch/)** | Get CSM-1B running **correctly** on Trainium (XLA offload of backbone + Mimi codec), plus a one-command `generate_speech.py` and a vLLM-Omni `CsmPipeline`. | Working. Validated **cosine 1.000000** vs CPU (teacher-forced, argmax 100%). |
| **[ttft-optimization/](ttft-optimization/)** | Cut time-to-first-audio and per-frame latency with `torch.compile(backend="neuron")` on the backbone step, the depth loop, and prefill, plus a minimal bf16 precision fix. **No retraining.** | Benchmarked. Per-frame decode **~317 ms → ~36 ms (8.8×)**; warm TTFA **p50/p99 ~38 ms** (bf16) / **~78 ms** (fp32-safe). |

If you just want speech out of the model, start at **[native-pytorch/README.md](native-pytorch/README.md)**.
If you care about latency, read on.

---

## Correctness first (native-pytorch port)

The functional port runs the heavy compute (16-layer backbone transformer + Mimi codec) on a
NeuronCore and keeps the generate loop on CPU. It is proven correct by **teacher-forced logit
match (cosine 1.000000, argmax 100%)** and **Mimi decode cosine 1.0** vs the CPU reference.

Note on determinism: CSM is autoregressive, so the exact waveform differs run-to-run and
CPU-vs-Neuron. A single argmax flip from sub-ULP floating-point differences cascades into a
*different but equally valid* speech realization. Correctness is therefore gauged by
teacher-forced logit/codebook match, **not** waveform cosine. Full details in
[native-pytorch/README.md](native-pytorch/README.md).

---

## Latency: the TTFT / TTFA optimization

All numbers below are **measured on-device** on a `trn2.48xlarge` (native torch-neuronx,
torch 2.11, single NeuronCore unless noted) and re-verified across fresh processes
(median-of-N, `dynamo.explain` graph-break checks, cosine vs reference). Measured results are
kept separate from modeled composites, and caveats are stated plainly.

The core win: the backbone decode step and the 31-codebook depth loop were assumed to be
eager/CPU costs, but each compiles to a **single fixed-shape resident-weight Neuron graph**
(`graph_count=1, 0 breaks`) once the data-dependent `.item()` bookkeeping is hand-rolled away.
That collapses per-frame decode from ~317 ms (stock `model.generate`) to ~36 ms.

### Time-to-first-audio (TTFA) percentiles — the SLO metric

Measured **warm**, single NeuronCore, **200 iterations** of the real first-audio critical path
(backbone decode step → depth decode 32 codebooks → Mimi codec 1 frame). This is a real
sampled distribution, not a component sum.

| Path | p50 | p90 | p99 | tail (p99−p50) |
|---|---:|---:|---:|---:|
| Depth decode + codec (sampled, 200 iters, std 0.2 ms) | 27.1 ms | 27.3 ms | 27.5 ms | 0.4 ms |
| **End-to-end TTFA — bf16 fast path** (+10.8 ms backbone) | **~38 ms** | **~38 ms** | **~38 ms** | 0.4 ms |
| **End-to-end TTFA — fp32-safe default** (depth ~59 ms) | **~78 ms** | **~78 ms** | **~78 ms** | ~0.4 ms |

The tail is essentially flat (p99 only 0.4 ms above p50): the decode path is one compiled
fixed-shape graph, so the only variation is host dispatch jitter (std 0.2 ms). **Quote ~38 ms
(bf16 fast) or ~78 ms (fp32-safe), single core, warm.** Full data:
[ttft-optimization/analysis/VERIFIED_TTFA_PERCENTILES.md](ttft-optimization/analysis/VERIFIED_TTFA_PERCENTILES.md).

### Per-frame decode breakdown (measured, on-device)

| Component | Before | After | Notes |
|---|---:|---:|---|
| Backbone step (compiled) | 128 ms eager | **10.8 ms** | `graph_count=1, 0 breaks`, reproduced ×2 |
| Depth decode (31 codebooks) | 137–163 ms CPU | **17.4 ms** bf16 / 59 ms fp32 | one compiled resident-weight loop |
| Codec (1 frame) | — | **7.6 ms** | CPU, warm |
| **Per-frame decode** | **~317 ms** (`generate`) | **~36 ms** | end-to-end loop measured at 38.7 ms |

Details: [ttft-optimization/analysis/VERIFIED_DECODE.md](ttft-optimization/analysis/VERIFIED_DECODE.md)
and [DEPTH_ON_DEVICE_FAIR.md](ttft-optimization/analysis/DEPTH_ON_DEVICE_FAIR.md).

### Prefill vs context: compile-vs-eager crossover

Per fresh process, 0 graph breaks, median±std, cosine(compiled, eager) ≥ 0.9998.

| Prompt (tokens) | eager | compiled | speedup | best TTFT |
|---:|---:|---:|---:|---:|
| 512 | 77 ms | **18 ms** | 4.25× | ~43 ms |
| 1024 | 97 ms | **37 ms** | 2.63× | ~62 ms |
| 2048 | 142 ms | **96 ms** | 1.47× | ~121 ms |
| 3072 | 195 ms | 180 ms | 1.08× | ~205 ms |
| 4096 | **261 ms** | 302 ms | 0.87× (eager wins) | ~286 ms |

`torch.compile(backend="neuron")` wins for prompts up to ~3.3k tokens (up to **4.25×** at 512);
above that, eager is faster. CSM's trained window is 2048, so compile wins across the entire
in-spec range. Full data:
[ttft-optimization/analysis/VERIFIED_PREFILL_TTFT.md](ttft-optimization/analysis/VERIFIED_PREFILL_TTFT.md).

> **Measured vs modeled:** the TTFA percentile table is a real sampled measurement. A single
> *prefill-inclusive* end-to-end TTFT (e.g. ~43 ms for a 512-token prompt) is a **component sum
> of measured parts**, not one wall-clock run — treat those single composite numbers as
> estimates.

### What made it fast (levers, ranked by impact)

1. **Compile the whole depth loop as one resident-weight graph.** The stock HF depth forward
   can't fuse (`.item()` in positions/mask forces 3 graph breaks); a hand-rolled loop with
   python-int positions compiles to 1 graph, 0 breaks. 137 ms CPU → **17.9 ms** on-device (7.6×).
2. **Compile the backbone decode step.** 128 ms eager → **10.8 ms** (12×) — same eager-dispatch
   issue as depth.
3. **bf16 precision fix.** Run just the head/argmax matmul + Q·Kᵀ scores in fp32 (rest bf16) to
   recover codebook accuracy at ~no latency cost. See
   [BF16_DEPTH_FIX.md](ttft-optimization/analysis/BF16_DEPTH_FIX.md).
4. **Codec stays on CPU** (~8 ms/frame) — Mimi's ConvTranspose1d does not compile on this stack,
   and CPU is fast enough for a single frame.
5. **Prefill: compile for prompts ≤ ~3k tokens** (up to 4.25× at 512).

---

## Caveats and honest limitations

1. **bf16 depth is prompt-dependent.** Codebook match vs the fp32 reference is 29–32/32 on
   typical speech but dropped to **18/32 on a digit-heavy prompt**. bf16 produces a
   *different-but-plausible* realization, not a crash. **Ship fp32 depth (~59 ms) as the safe
   default; offer bf16 (~17 ms) as an opt-in fast path.** See
   [VERIFIED_PROMPT_SUITE.md](ttft-optimization/analysis/VERIFIED_PROMPT_SUITE.md).
2. **Context limit 2048** (`max_position_embeddings`). The 4k prefill numbers are RoPE
   extrapolation — latency-valid, coherence unverified past 2048.
3. **Waveform cosine is not a valid quality metric here** (autoregressive divergence, above). A
   perceptual ASR-WER test vs the original model is the recommended next validation and has not
   been run yet.
4. **Tensor parallelism is not included.** CSM's backbone has no TP sharding; it would be a port
   and only helps >3k prefill, which is out of the in-spec TTS range. Design in
   [TP_ANALYSIS_AND_PLAN.md](ttft-optimization/analysis/TP_ANALYSIS_AND_PLAN.md).
5. **Fixed frame-count overshoots short prompts** (emits trailing silence). Production needs an
   EOS-based early-stop.
6. **Cold start:** the first request pays a one-time ~8.5 s `torch.compile`; a persistent NEFF
   cache via `NEURON_COMPILE_CACHE_URL` did not engage on this stack. Amortize with a resident
   server (compile once at startup). See
   [COLDSTART_CACHE.md](ttft-optimization/analysis/COLDSTART_CACHE.md).

A full pre-PR code review (one real precision bug found and fixed, over-claims corrected to
measured-vs-modeled framing) is in
[ttft-optimization/CODE_REVIEW.md](ttft-optimization/CODE_REVIEW.md).

---

## Reproduce

**Generate speech (native-pytorch, `trn2.3xlarge`):** follow the copy-paste guide in
[native-pytorch/README.md](native-pytorch/README.md).

**Latency benchmarks (`trn2.48xlarge`, DLC container, single core):**

```bash
# NEURON_RT_VISIBLE_CORES=0
python3 manual_decode_loop.py --frames 20 --text "[0]Hello."   # full decode, ~36 ms/frame
python3 ttft_percentiles.py  --iters 200 --depth-k 32          # warm p50/p90/p99 TTFA
python3 prefill_verify.py    --n 512                           # prefill compile vs eager
python3 fair_depth_exact.py                                    # bf16 depth exactness sweep
```

Scripts live in [ttft-optimization/src/](ttft-optimization/src/); the full benchmark writeup is
[ttft-optimization/PR_DESCRIPTION.md](ttft-optimization/PR_DESCRIPTION.md).

---

## Directory layout

```
csm-1b/
├── README.md                 # you are here
├── native-pytorch/           # functional port: run CSM-1B TTS on Trainium (cosine 1.0)
│   ├── README.md             # copy-paste guide
│   ├── src/                  # generate_speech.py, csm_pipeline.py, offload + CPU-ref harnesses
│   └── results/              # validation writeup + sample .wav
└── ttft-optimization/        # latency work: compiled decode/prefill + verified benchmarks
    ├── PR_DESCRIPTION.md      # full benchmark writeup
    ├── CODE_REVIEW.md         # pre-PR review findings
    ├── src/                   # decode loop, depth loop, prefill/percentile/throughput harnesses
    ├── analysis/             # VERIFIED_* measured results + design docs
    └── results/              # audio proof (.wav)
```

---

## Hardware and software

- **Functional port (native-pytorch/):** single Trainium2 chip — a `trn2.3xlarge` is enough.
  Requires the **native-PyTorch Neuron beta** (torch_xla 2.9); the public beta's older
  torch_xla breaks CSM's int64 RoPE/mask casts.
- **Latency benchmarks (ttft-optimization/):** `trn2.48xlarge`, native torch-neuronx (torch
  2.11), `torch.compile(backend="neuron")`, single NeuronCore per measurement.

## Credits and license

Model: `eustlb/csm-1b` (canonical HF conversion). Original: Sesame CSM
([sesame/csm-1b](https://huggingface.co/sesame/csm-1b)), Apache-2.0.
