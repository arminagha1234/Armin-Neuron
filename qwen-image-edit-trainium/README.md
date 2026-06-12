# Qwen-Image-Edit-2511 on AWS Trainium2 — Native PyTorch

Working reference implementation of `Qwen/Qwen-Image-Edit-2511`
(the diffusers `QwenImageEditPlusPipeline`) on Trainium2, including
the `fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA` adapter merged
into the base. End-to-end image editing pipeline running on a
`trn2.48xlarge` with **TP=4 native PyTorch + `torch_neuronx`** —
no NxDI, no NxDT, no vLLM.

This is a customer-readable port: the runtime entry point is one
~230-line Python file (`src/run_simple.py`), and the supporting
scaffolding (TP plan, meta-init weight loader, real-valued RoPE
replacement) is split into small files that map to the eight
Beta 2 TP fixes documented in the AWS Neuron customer guidance.

## Status

✅ **Correctness validated end-to-end** on `trn2.48xlarge` (TP=4)
with cosine **0.999897** vs CPU diffusers reference output at the
canonical 28-step / 512×512 / 1-image config. The HTTP serving wrap
preserves bit-exact semantics: cosine **1.000000** vs the standalone
runner output.

| Test | Result |
|---|---|
| 28-step 512×512 1-image vs CPU diffusers ref | cosine **0.999897** ✅ |
| HTTP `/edit` round-trip vs standalone runner | cosine **1.000000** ✅ |
| Run-to-run determinism (same seed, n=5) | cosine ≥ 0.9999 ✅ |
| 2-input-image edit | works (227s warm) ✅ |
| 3-input-image edit | OOMs on 24 GB user budget per core ❌ |

## Performance — verified on trn2.48xlarge, TP=4, BF16

Two execution modes are wired up:

1. **Eager** (`run_simple.py`) — 3.8s/step warm at 512×512
2. **`torch.compile(backend="neuron")` + real-valued RoPE**
   (`run_compiled.py`) — 3.35s/step warm (12% faster than eager)

The default ship is the compile path. Real-valued RoPE is required
because the stock complex64 RoPE in diffusers crashes MLIR lowering
during compile (`DecomposeComplexOps pass crashed unexpectedly`).

### Canonical workload (28 steps, 512×512, 1 input image, n=5 warm)

| Metric | Value |
|---|---:|
| Cold (1st request after worker boot) | 168.05 s |
| Warm p50 | 93.83 s |
| Warm p95 | 95.05 s |
| Warm p99 | 95.19 s |
| Warm stdev | 0.85 s |
| Cosine vs CPU diffusers ref | 0.999897 |

### Per-stage breakdown (warm canonical)

| Stage | Time | % of total |
|---|---:|---:|
| Text encoder (CPU, Qwen2.5-VL 8.4B) | 7.5 s | 8.0% |
| **VAE encode (CPU)** | **32.1 s** | **34.4%** |
| Denoising (Neuron transformer, 28 steps) | 42.8 s | 45.9% |
| VAE decode (CPU) | 10.9 s | 11.7% |
| Postprocess | 0.005 s | 0.0% |
| **Total** | **93.5 s** | 100% |

The CPU-side flat tax (encoder + VAE encode + decode + scheduler
glue) is ~50 s and doesn't shrink with step count or resolution.
Closing this is where the next 30-40% of latency lives — see
"What's deferred" below.

### Resolution sweep (28 steps, single input image, warm)

| Resolution | Cold | Warm p99 |
|---|---:|---:|
| 512×512 | 92.6 s | 92.98 s |
| 768×768 | 114.5 s | 114.01 s |
| 1024×1024 | 434.4 s¹ | 149.68 s |

¹ The 1024×1024 cold first call includes `torch.compile`'s
shape-specific recompile (~5 min). NEFF cache persistence across
container restarts is a Phase 3 production requirement.

### Step count sweep (512×512, single input image, warm)

| Steps | Warm p99 | Per-step (warm transformer) |
|---:|---:|---:|
| 4 | 74.0 s | 6.3 s |
| 8 | 77.1 s | 3.5 s |
| 16 | 83.3 s | 2.1 s |
| 28 | 92.1 s | 1.5 s |

Per-step transformer cost asymptotes to **1.5 s** at 28 steps —
the apparent slope is dominated by the fixed CPU flat tax getting
amortized across more denoising steps.

### Multi-input-image sweep (28 steps, 512×512, warm)

| Input images | Warm p99 |
|---:|---:|
| 1 | 92.24 s |
| 2 | 227.27 s |
| 3 | OOM ❌ |

3-input edits exceed the ~24 GB user budget per Neuron core
because the Plus pipeline concatenates per-image latents
(~3× the token count). This is a hard customer-facing limit
today; closing it requires the deferred two-phase loading work.

### TTFI (full cold start)

Wall-clock from worker process spawn to first image returned over HTTP:

| Stage | Time |
|---|---:|
| Worker spawn → torchrun → weights → ProcessGroup ready | 121.0 s |
| Server start + uvicorn boot + first POST sent | 41.2 s |
| First `/edit` POST → image returned (cold pipeline) | 168.7 s |
| **Total spawn → first image** | **330.9 s** |

Of the 168.7 s first-image cost, ~169 s is `torch.compile`'s cold
shape-specific compile. Persisting the NEFF cache across restarts
would cut TTFI to roughly 250 s.

Full benchmark JSON + rendered reports:
[results/bench/full_sweep/REPORT.md](./results/bench/full_sweep/REPORT.md),
[results/bench/multi_image_only/REPORT.md](./results/bench/multi_image_only/REPORT.md),
and the consolidated narrative + framing for the customer in
[BENCHMARK_TRN2_48XL.md](./BENCHMARK_TRN2_48XL.md).

## Architecture — what's on Neuron, what's on CPU

| Component | Location | Notes |
|---|---|---|
| Transformer (Qwen-Image DiT, ~14B with merged LoRA) | **Neuron** (TP=4) | Sharded across 4 cores, ~6 GB/rank resident weights |
| `QwenEmbedRopeReal` replacement | **Neuron** | Drop-in for stock complex64 RoPE; required for `torch.compile` |
| Qwen2.5-VL 8.4B text encoder | CPU | Doesn't fit alongside transformer's 24 GB user budget |
| AutoencoderKLQwenImage VAE | CPU | Standalone VAE-on-Neuron benchmarks: 0.09s encode, 0.29s decode (344× and 34× faster than CPU) — but co-residence OOMs the transformer |
| Scheduler (FlowMatchEulerDiscreteScheduler) | CPU | Stock diffusers; tiny |

Both encoder and VAE could move to Neuron via two-phase model
loading (load VAE → encode → free → load transformer → denoise →
free → load VAE → decode), but that's deferred multi-day work —
see "What's deferred" below.

## TP plan and the eight Beta 2 fixes

Native PyTorch TP on Trainium2 Beta 2 needs eight specific fixes
to compile and produce correct output. All applied here:

| # | Fix | Where it lives |
|---|---|---|
| 1 | `init_process_group(backend='neuron')` (NOT 'xla', NOT 'gloo') | `run_simple.py` |
| 2 | `init_device_mesh('neuron', (N,))` | `run_simple.py` |
| 3 | Meta-init + slice-from-disk weight loader (avoids OOM at `.to()`) | `qwen_edit_meta_loader.py` |
| 4 | TP-aware RMSNorm for sharded inner_dim | `qwen_edit_tp_plan.py` |
| 5 | Patch `attn.heads` to `heads/N` after ColwiseParallel | `qwen_edit_tp_plan.py` |
| 6 | Slice 3D RoPE outputs by rank | `qwen_edit_tp_plan.py` |
| 7 | Functional RoPE (no in-place ops on views) | `rope_real.py` |
| 8 | Real-valued RoPE for compile compatibility | `rope_real.py` |

## Quickstart

### Prerequisites

```bash
# On a trn2.48xlarge with the Beta 2 DLC container:
sudo docker run -d --name fal_beta2 \
    --network host --ulimit core=-1 --device=/dev/neuron0 ... \
    421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest \
    sleep infinity
sudo docker exec -it fal_beta2 bash
source /opt/torch-neuronx/.venv/bin/activate

# Download model + LoRA, merge once
huggingface-cli download Qwen/Qwen-Image-Edit-2511 \
    --local-dir /root/.cache/huggingface/...
huggingface-cli download fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA \
    --local-dir /tmp/fal_lora
python src/merge_lora.py \
    --base /root/.cache/huggingface/.../snapshots/<sha>/transformer \
    --lora /tmp/fal_lora \
    --out  /opt/dlami/nvme/fal/merged_lora/transformer
```

The full setup script is `src/setup_box.sh` (NVMe RAID0 + cache redirects)
+ `src/setup_beta2.sh` (Beta 2 DLC pull + container start).

### Eager run (4 steps, smoke test)

```bash
NEURON_RT_NUM_CORES=4 \
torchrun --nproc_per_node=4 --standalone src/run_simple.py \
    --base-model-path /root/.cache/huggingface/.../snapshots/<sha> \
    --merged-transformer /opt/dlami/nvme/fal/merged_lora/transformer \
    --images results/test_input.png \
    --prompt "show the subject from a different camera angle" \
    --num-steps 4 \
    --height 512 --width 512 \
    --output results/output_first.png
```

### Compile run (28 steps, production)

```bash
bash src/run_compiled_28step.sh
# 28-step warm run: ~93 s on hot NEFF cache.
# Expect ~170 s for the very first call (cold compile).
```

### HTTP serving (FastAPI + persistent torchrun worker)

```bash
# Terminal 1 — start worker (5 min cold to compile transformer)
bash serve/launch_worker.sh
# Wait for: "[worker r0] pipeline ready, entering serve loop"

# Terminal 2 — start FastAPI front-end
bash serve/launch_server.sh

# Terminal 3 — smoke test
curl http://localhost:8000/health
python serve/test_client.py --output results/serve_test.png
```

The worker is one persistent torchrun process (4 ranks). The server
is a separate uvicorn process; they communicate over a Unix socket
at `/tmp/fal_pipeline.sock`. Single in-flight request enforced by
asyncio lock + the all-ranks-same-pipeline invariant required by
Beta 2's `'neuron'` ProcessGroup.

### Run the bench yourself

```bash
# After worker + server are up:
python bench/bench_full.py --all --n-warm 6 \
    --out results/bench/<your_run>/results.json
python bench/bench_report.py results/bench/<your_run>/results.json \
    > results/bench/<your_run>/REPORT.md
```

## What's deferred

Work that didn't fit in the initial port and is the next ~30%
latency win:

1. **Two-phase VAE/encoder loading** — load VAE only during encode/decode,
   transformer only during denoising. Estimated ~30 s saved per request
   (~17% latency reduction); also unlocks 3-input-image edits which OOM
   today. Engineering: 1-2 weeks.
2. **NEFF cache persistence across container restarts** — operationally
   important; cuts cold TTFI from ~330 s to ~250 s.
3. **FP8/FP4 quantization on Trainium2** — supported by hardware but
   not yet wired up for diffusion. Probable 1.5-2× speedup in the
   transformer compute, but won't change the CPU-bound flat tax.

## What was tried and abandoned

Documented so future contributors don't re-tread:

| Attempt | Outcome |
|---|---|
| NKI flash attention (`nkilib.core.attention.attention_cte`) | Compiles and is numerically correct, but at this seq_len (~5K tokens) and head count (6 heads/rank after TP=4) it's **slower** than stock SDPA: 4.61 s/step vs 3.35 s/step. Banked for higher resolutions or longer sequences where it should flip to a win. |
| TP=8 instead of TP=4 | 3.28 s/step vs 3.35 s. Collectives are the floor; more cores didn't help. |
| VAE on Neuron (replicate-per-rank) | OOMs alongside transformer (24 GB user budget per core, transformer is ~23.8 GB resident). |
| VAE on Neuron via NKI conv2d | Works numerically but slower than CPU due to per-conv dispatch overhead. |
| Phased VAE loading (load → free → reload) | Same OOM — transformer weights stay resident; cleanup requires full reload. |

## Files

```
qwen-image-edit-trainium/
├── README.md (this file)
├── BENCHMARK_TRN2_48XL.md      # Full customer narrative + framing
├── src/
│   ├── run_simple.py           # Eager TP=4 entry point (~230 lines)
│   ├── run_compiled.py         # torch.compile + real RoPE entry point
│   ├── run_cpu_ref.py          # CPU diffusers reference for cosine validation
│   ├── qwen_edit_tp_plan.py    # TP layout + 5 architectural fixes
│   ├── qwen_edit_meta_loader.py # Meta-init weight loader
│   ├── rope_real.py            # Real-valued RoPE (compile-compatible)
│   ├── merge_lora.py           # One-time LoRA merge helper
│   └── setup_*.sh              # Box and container setup scripts
├── serve/
│   ├── worker.py               # Persistent torchrun worker
│   ├── server.py               # FastAPI /edit + /health
│   ├── launch_worker.sh        # torchrun launcher
│   ├── launch_server.sh        # uvicorn launcher
│   ├── test_client.py          # Smoke test client
│   ├── validate_serve.py       # Cosine validator vs reference
│   └── README.md               # Serve-specific architecture doc
├── bench/
│   ├── bench_full.py           # Customer-grade bench harness
│   ├── bench_report.py         # JSON → markdown report renderer
│   ├── bench_dp_box.py         # 4-worker data-parallel bench
│   └── bench_dp_box.sh         # 4-worker launch helper
├── test/
│   └── (validation scripts)
└── results/
    └── bench/                   # Captured bench JSON + rendered reports
```

## Compatibility Matrix

| Instance | TP | SDK / Container | Status |
|---|:-:|---|---|
| trn2.48xlarge | 4 | Beta 2 DLC (`concourse-release-0461d3b`) | VALIDATED |
| trn2.48xlarge | 8 | Beta 2 DLC | VALIDATED (2% faster than TP=4 — collectives are floor) |
| trn2.3xlarge | 4 | Beta 2 DLC | NOT TESTED (should work; smaller box, less memory headroom) |

## License

Apache 2.0 (this port). The base model and LoRA carry their own licenses
(see Hugging Face model cards: `Qwen/Qwen-Image-Edit-2511`,
`fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA`).
