# PixelDiT-XL (~1B) on AWS Neuron — Native PyTorch (Beta 3)

Train a **1-billion-parameter pixel-space Diffusion Transformer** on AWS
Neuron using the **native PyTorch** path — `torch.device("neuron")` +
`torch.compile(backend="neuron")`. No NxD, no vLLM. Single-core, multi-core
data-parallel, and **weight-sharded FSDP** all verified working.

---

## 🖥️ Instance & stack (what this was validated on)

| | |
|---|---|
| **EC2 instance** | `inf2.24xlarge` (6 × Inferentia2 devices = **12 NeuronCores**, 16 GB/core) |
| **Also supported** | `trn1`, `trn2`, `trn3`, `inf2` (per the Beta 3 guide — Trainium recommended for production training) |
| **Container** | Beta 3 DLC `concourse-release-0461d3b:latest` |
| **Versions** | torch **2.11.0**, torch-neuronx **2.11.3**, neuronx-cc **2.25**, nki **0.4.0** |
| **Device API** | `torch.device("neuron")` (native PrivateUse1 backend) |
| **Compile** | `torch.compile(backend="neuron", dynamic=False)` — static shapes only |

> ℹ️ **inf2 vs Trainium:** inf2 is an inference-class part. This example proves
> the port runs and trains on inf2, but for production training throughput use
> **Trainium (trn1/trn2/trn3)** — the same scripts run there unchanged with more
> cores and higher memory bandwidth.

---

## 🚀 Quick start

### 1. Launch + enter the Beta 3 container
```bash
# On the inf2/trn instance, the Beta 3 container should be running:
sudo docker ps                       # look for image concourse-release-0461d3b:latest
sudo docker exec -it beta3 bash      # 'beta3' = container name
cd /work
```
Models/scripts live in `/work`; copy the three `.py` files + `sweep.sh` there.

### 2. Pick free cores
`neuron-ls` shows which cores are busy. **Pin every run to free cores** with
`NEURON_RT_VISIBLE_CORES` so you don't disturb other jobs (e.g. a live vLLM
server on cores 0–3). All examples below use cores **4+**.

---

## 📜 The scripts

| Script | What it does |
|---|---|
| `pixeldit_xl_train.py` | Model + **single-core** training, `--check` parity test, `--data-dir` (real ImageFolder data), `--save-dir/--resume` checkpointing |
| `pixeldit_xl_dist.py` | **Multi-core data-parallel** (replicate + grad all-reduce). Fastest/step when the model fits on one core |
| `pixeldit_xl_fsdp.py` | **Multi-core FSDP** (weight-sharded). Use when the model is too big to replicate per core |
| `sweep.sh` | Single-core batch/resolution **memory-ceiling** sweep |

### A. Correctness check (CPU vs Neuron, ~1 min)
```bash
NEURON_RT_VISIBLE_CORES=6 python3 pixeldit_xl_train.py \
    --check --hidden 384 --depth 4 --image-size 64 --patch 8
# → [check] PARITY PASS (fp32 CPU vs Neuron)
```

### B. Single-core training (full 1B)
```bash
NEURON_RT_VISIBLE_CORES=6 python3 pixeldit_xl_train.py \
    --steps 3 --batch 1 --image-size 256 --patch 16
# first step pays a one-time NEFF compile (~3.5 min); warm steps ~200 ms
```

### C. Train on real images + checkpoints
```bash
NEURON_RT_VISIBLE_CORES=6 python3 pixeldit_xl_train.py \
    --data-dir /path/to/imagefolder \
    --save-dir /work/ckpts --save-every 500 --steps 2000 --batch 4
# resume:
NEURON_RT_VISIBLE_CORES=6 python3 pixeldit_xl_train.py \
    --resume /work/ckpts/pixeldit_step1999.pt --steps 1000 ...
```

### D. Multi-core data-parallel (2 cores → 2× throughput)
```bash
python3 pixeldit_xl_dist.py --world-size 2 --core-offset 4 \
    --steps 4 --batch 1 --image-size 256 --patch 16
# ranks pin to cores 4,5 ; prints GRAD-SYNC PASS
```

### E. Multi-core FSDP (weight-sharded)
```bash
python3 pixeldit_xl_fsdp.py --world-size 2 --core-offset 4 \
    --hidden 1152 --depth 12 --steps 3
# → FSDP wrap OK via FSDP2+DeviceMesh(neuron) ; each core holds 1/world_size of the weights
```

### F. Memory-ceiling sweep
```bash
CORE=6 bash sweep.sh        # results in /work/sweep_results.txt
```

---

## 📊 Verified results (inf2.24xlarge, 1.01B params, bf16)

| Test | Result |
|---|---|
| **CPU↔Neuron parity (fp32)** | ✅ PASS — max abs diff 4.3e-5 |
| **Single-core train** | ✅ ~200 ms/step warm (after ~3.5 min one-time compile) |
| **Memory ceiling @256px** | ✅ batch **1, 2, 4, 8** all fit on one 16 GB core |
| **Data-parallel (2 cores)** | ✅ ~2× throughput, GRAD-SYNC PASS (spread 0.0) |
| **FSDP2 (`DeviceMesh("neuron")`)** | ✅ shards weights (10.3M of 20.6M per core, ws=2) |
| **Checkpoint save + resume** | ✅ round-trips model + AdamW + step |

### ⚠️ Known limitation — keep attention tokens ≤ 256
512px @ patch 16 (= 1024 tokens) **fails to compile** with:
```
COMPILATION FAILED: nkilib.core.attention.attention_cte ...
DMACopy ... transpose only supported for HBM->SB
```
This is a neuronx-cc flash-attention-kernel limitation (not OOM, not a model bug).
**Workaround:** keep ≤256 tokens — e.g. **512px @ patch 32** works (verified). For
longer sequences, file an issue at https://github.com/aws-neuron/aws-neuron-sdk.

---

## Model

Pixel-space DiT-XL: patch-embed conv → N adaLN-Zero transformer blocks (timestep +
class conditioning) → unpatchify. Trained with a rectified-flow velocity objective
(`target = noise − x0`). Configurable via `--hidden/--depth/--heads/--image-size/--patch`.
Default 1.01B = hidden 1408 / depth 28 / heads 16.

See [`RESULTS.md`](RESULTS.md) for full logs and numbers.
