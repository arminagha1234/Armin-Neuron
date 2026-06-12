# fal Qwen-Image-Edit-2511 — Trainium2 vs H100 final results

**Date:** 2026-06-12
**Owner:** Armin Agha-Ebrahim
**Status:** Phase 1 + Phase 2 (bench) complete. Phase 3 HTTP serving wrap working end-to-end.

## TL;DR

For Qwen-Image-Edit-2511 + fal's `Multiple-Angles-LoRA` on the workload
sizes we tested (512–1024², 1–2 input images, 4–28 steps), **H100
wins decisively on both latency and $/image**:

- **Latency:** H100 is ~11× faster at 28 steps (8.5s vs 92.6s @ 512×512)
- **Cost:** H100 is ~25× cheaper per image at full-box DP ($0.0092 vs $0.229)
- **3-image edit:** doesn't fit on Trainium2 today (OOMs after first request);
  fits trivially on H100 (60.3 GB peak vs Trainium2's 24 GB user budget per core)

This is the workload class H100 was built for: a small (~12B) dense DiT
that fits in 80 GB HBM, no sharding tax, mature CUDA kernels. Trainium2
needs TP=4 → collective overhead floors per-step latency at ~1.5s and
the CPU-bound flat tax (encoder + VAE encode/decode + scheduler glue)
adds another ~70s per request.

**Conclusion: don't pitch fal Trainium for small image edit workloads.**
Pitch the workloads where Trainium has a real shot — see
`ACCOUNT_PLAN.md` "Trainium-vs-H100 benchmark prioritization" for the
ranked target list (TL;DR: video diffusion + LoRA training queues).

## Setup

| | |
|---|---|
| Model | `Qwen/Qwen-Image-Edit-2511` |
| LoRA | `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA` (merged into base) |
| Pipeline | `diffusers.QwenImageEditPlusPipeline` |
| Precision | bf16 |
| Trainium hardware | trn2.48xlarge (`i-02a51e30b3a33408d`, us-east-2) |
| Trainium config | TP=4, 4 cores, native PyTorch + `torch_neuronx`, eager + `torch.compile(backend="neuron")` + real-valued RoPE |
| H100 hardware | p5.48xlarge, 8× H100 80GB |
| H100 config | stock diffusers, attention slicing ON, no compile, no FP8 |
| On-demand pricing used | trn2.48xl $35.7608/hr; p5.48xl $31.464/hr (Jun 2026) |

Both platforms run the **same merged-LoRA bf16 weights** and the **same
diffusers pipeline class**, so comparisons are software-stack apples-to-apples.

Trainium HTTP serving: FastAPI + uvicorn front end + persistent torchrun
worker over Unix socket. Validated cosine 1.000000 vs standalone
`run_compiled_28step.png` reference (zero numerical drift introduced by
HTTP transport).

## Headline numbers

### Resolution sweep (28 steps, single input image, warm)

| Resolution | H100 latency | Trainium latency | latency gap |
|---|---:|---:|---:|
| 512×512   | 8.53 s  | **92.63 s**  | 10.9× |
| 768×768   | 10.89 s | **113.87 s** | 10.5× |
| 1024×1024 | 14.39 s | **149.50 s** | 10.4× |

Trainium2 1024² **cold first call: 434.4 s** because `dynamic=False`
forces a shape-specific torch.compile recompile (~5 min). Persistent
NEFF cache across container restarts is a Phase 3 production
requirement.

### Step count sweep (512×512, single input image, warm)

| Steps | H100 | Trainium | gap | Trainium per-step (transformer only) |
|---:|---:|---:|---:|---:|
| 4  | 1.43 s | 73.83 s | 51.6× | 6.3 s/step |
| 8  | 2.62 s | 76.92 s | 29.3× | 3.5 s/step |
| 16 | 5.00 s | 83.22 s | 16.6× | 2.1 s/step |
| 28 | 8.56 s | 91.86 s | 10.7× | 1.5 s/step |

Latency gap narrows as steps increase because Trainium's ~70 s flat
tax (encoder + VAE encode/decode + scheduler) gets amortized across
more denoising steps. Pure per-step transformer cost on Trainium
asymptotes to ~1.5 s vs H100's 305 ms (5× slower per step at scale).

### Multi-input-image sweep (28 steps, 512×512, warm)

| Input images | H100 latency | Trainium latency | gap |
|---:|---:|---:|---:|
| 1 | 8.56 s  | 93.46 s   | 10.9× |
| 2 | 17.15 s | **227.17 s** | 13.2× |
| 3 | 27.64 s | **DOES NOT FIT** ❌ | — |

**3-image OOMs on Trainium2 today.** First sample took 17 minutes of
torch.compile + denoise then errored; subsequent samples failed with
HTTP 500 (worker process is in a degraded state and can't accept new
work). The Plus pipeline concatenates per-image latents into the
attention sequence, ~tripling token count vs single-image; activations
exceed the ~24 GB user budget per core after ~14 GB of TP-sharded
transformer weights are already resident.

This is a real customer-facing limitation. fal's Multiple-Angles-LoRA
is explicitly built around 2–3 input images (subject from different
angles); Trainium can't serve the 3-input case today without the
deferred Phase 2.5 work (two-phase VAE/encoder loading to free
activation budget).

### Per-stage breakdown (canonical 28-step 512² 1-image, warm)

| Stage | Time | % of total |
|---|---:|---:|
| encoder (CPU, Qwen2.5-VL 8.4B) | 7.5 s   | 8.0% |
| **vae_encode (CPU)**           | **32.1 s** | **34.4%** |
| denoise (Neuron transformer)   | 42.8 s  | 45.9% |
| vae_decode (CPU)               | 10.9 s  | 11.7% |
| postprocess                    | 0.005 s | 0.0% |
| **total**                      | **93.5 s** | 100% |

**The flat tax is real and quantified.** ~50 s of every Trainium
request is CPU-side work that doesn't shrink with step count or
resolution. Closing this is Phase 2.5 (two-phase model loading); we
attempted simple co-residence and it OOMs on the 24 GB-per-core
budget. Multi-day engineering effort, deferred until customer asks for
sub-90s latency.

H100 stage breakdown for the same workload (from their bench report):

| Stage | Time | % of total |
|---|---:|---:|
| encoder    | 0.10 s | 1.2% |
| vae_encode | 0.05 s | 0.6% |
| transformer| 8.12 s | 97.9% |
| vae_decode | 0.02 s | 0.3% |

H100 has effectively no flat tax — 98% of the time is in the
transformer because everything fits and runs on the GPU.

## $/image at full-box data-parallel

The cost story is the most important number for fal.

### 512×512, 28 steps, 1 image

| Platform | Box hourly | Imgs/min/box | $/image |
|---|---:|---:|---:|
| **trn2.48xl (4× DP, TP=4 each)** | **$35.76**  | 156 (extrapolated) | **$0.229** |
| **p5.48xl  (8× H100 data-parallel)**  | **$31.46**  | 56.9 (measured)    | **$0.0092** |

H100 is **~25× cheaper per image** at full box throughput.

### Cost ratio at every resolution we measured

| Workload | trn2 $/image | p5 $/image | trn2 multiplier vs p5 |
|---|---:|---:|---:|
| 28-step 512² 1-img | $0.229 | $0.0092 | 25× |
| 28-step 768² 1-img | $0.283 | $0.0119 | 24× |
| 28-step 1024² 1-img | $0.372 | $0.0157 | 24× |
| 28-step 512² 2-img | $0.564 | $0.0187 | 30× |
| 28-step 512² 3-img | DOES NOT FIT | $0.0302 | — |

Even with aggressive 1-yr Trainium reservations (~50% off on-demand),
trn2 stays ~12× more expensive per image than H100 on-demand. Spot
H100 (~30% off) widens the gap further.

This is **not** a workload where Trainium pricing wins.

## TTFI (full cold start, spawn → first image)

Customer-relevant for any cold-start scenario: serverless ramp,
container restart, autoscale-up.

| Platform | spawn → /health | spawn → first image |
|---|---:|---:|
| Trainium2 | 121 s (worker boot + weights + ProcessGroup) | **331 s** |
| H100 | n/a (single-process bench) | **140 s** |

Trainium TTFI is **2.4× slower** than H100. Of Trainium's 331 s, ~169 s
is the cold compile of the transformer for the new shape — caching the
NEFF across container restarts could cut TTFI to ~250 s; still slower
than H100 but closer.

## What was optimized vs not

Path C as benchmarked uses:

- ✅ Native PyTorch + `torch_neuronx` (no NxDI/NxDT/vLLM)
- ✅ TP=4 via `parallelize_module` + DTensor on Beta 2 `'neuron'` PG
- ✅ Meta-init weight loader (streams weights from disk per rank)
- ✅ `torch.compile(backend="neuron", dynamic=False, fullgraph=False)` —
  shaved 12% off eager (3.8 → 3.35 s/step)
- ✅ Real-valued RoPE replacement (compile-compatible — stock complex64
  RoPE crashes MLIR lowering)
- ✅ bf16 collectives (verified — DTensor `RowwiseParallel` doesn't
  upcast to fp32)

Tried and abandoned for this workload:

- ❌ NKI flash attention (`nkilib.core.attention.attention_cte`) — slower
  than stock SDPA at this seq_len/head count (4.61 vs 3.35 s/step at
  512²; would flip to a win at higher resolutions)
- ❌ TP=8 — 3.28 s/step vs 3.35 s at TP=4. Collectives are the floor;
  more cores don't help.
- ❌ VAE on Neuron (replicate-per-rank) — OOMs alongside transformer
- ❌ VAE on Neuron via NKI conv2d — works but slower than CPU due to
  per-conv dispatch overhead
- ❌ Phased VAE loading (load → free → reload) — OOMs because
  transformer weights stay resident; cleanup requires full reload

Not yet attempted (deferred):

- 🟡 **Two-phase model loading** (encoder OFF Neuron during transformer,
  swap in for VAE encode/decode) — multi-day engineering work; estimated
  ~30 s savings (~17% latency reduction). Required to make 3-input edits
  fit at all.
- 🟡 **NEFF cache persistence across container restarts** — operationally
  important; not a code change so not a Phase 2 task per se.
- 🟡 **FP8 / FP4 quantization on Trainium2** — Trainium2 supports it;
  haven't enabled. Could 2× our perf and is the most plausible Trainium-
  side optimization but won't close the gap to H100 on this workload.

## Files

- Bench harness: `customers/fal/path_c/serve/bench_full.py`,
  `bench_report.py`, `bench_dp_box.py`, `bench_dp_box.sh`
- Production runner: `customers/fal/path_c/run_compiled.py`
- HTTP serving: `customers/fal/path_c/serve/{worker.py,server.py}`
- Bench results JSON:
  - `customers/fal/path_c/results/bench/canonical_steps/results.json`
    (canonical 28-step 512² + step sweep, no stages — older bench)
  - `customers/fal/path_c/results/bench/full_sweep/results.json`
    (resolution + step + 1-image multi-image, with stages)
  - `customers/fal/path_c/results/bench/multi_image_only/results.json`
    (1-/2-image multi-image; 3-image failed)
- Bench reports (rendered):
  - `customers/fal/path_c/results/bench/full_sweep/REPORT.md`
  - `customers/fal/path_c/results/bench/multi_image_only/REPORT.md`
- H100 reference: `customers/fal/h100_bench/results_20260611-193415/report.md`
- Account plan with prioritization: `customers/fal/ACCOUNT_PLAN.md`

## Honest framing for the fal conversation

What we **can** say truthfully:

- "We have a working native-PyTorch port of Qwen-Image-Edit-2511 + your
  LoRA running on Trainium2 with cosine 0.9999 vs CPU diffusers reference."
- "On a small dense DiT image-edit workload like this one, today H100 is
  faster and cheaper per image. We're not pitching this specific workload
  for cost savings."
- "Trainium2 hits a 24 GB-per-core ceiling that means **3-input edits
  don't fit**. We can solve this with multi-day engineering work
  (two-phase VAE loading), but only if 3-input is in your top requests."
- "Where Trainium does have a story for fal: video diffusion (Wan, LTX),
  LoRA training queues, and high-volume open-weights workloads where
  reservation pricing matters more than per-call latency. See the
  prioritized target list in ACCOUNT_PLAN.md."

What we **should not** say:

- ~~"Trainium beats H100 on $/image."~~ — we don't (this workload, today)
- ~~"Phase 2 will close the gap."~~ — Phase 2.5 saves ~30 s, gap stays 5-7×
- ~~"Just turn on FP8 and we'll match H100."~~ — would help, won't close it

## Next steps (per priority)

1. **LTX-2 19B image-to-video bench** — same harness, existing pipeline
   working. Hours of work. Best chance at a competitive Trainium-vs-H100
   number.
2. **LoRA training $/job cost model** — no benchmark; just price the
   FLUX/Wan/Qwen training endpoints at fal volume assumptions on a trn1.32xl
   reserved instance vs their current spot-H100 spend.
3. **Wan-2.2 a14b image-to-video bench** — ~1 week port; second
   strongest candidate.
4. **DEFERRED until fal asks**: two-phase VAE/encoder loading on
   Qwen-Image-Edit-2511 (saves ~30 s, unlocks 3-input edits).
