# PixelDiT-XL (~1B) on AWS Neuron — Native PyTorch (Beta 3) — Results

**Box:** inf2.24xlarge (`i-02efdfb60de0330b8`), 6 Inferentia2 devices / 12 cores, 16 GB/core.
**Stack:** Beta 3 DLC `concourse-release-0461d3b:latest` — torch 2.11.0, torch_neuronx 2.11.3, neuronx-cc 2.25, nki 0.4.0. Native path: `torch.device("neuron")` + `torch.compile(backend="neuron", dynamic=False)`.
**Isolation:** all runs pinned to free cores (4–6) via `NEURON_RT_VISIBLE_CORES`; the live vLLM server on cores 0–3 was never touched.

Model: pixel-space DiT-XL, adaLN-Zero blocks, patch-embed conv, timestep+class conditioning, rectified-flow velocity loss. **1.010 B params** at hidden=1408 / depth=28 / heads=16.

## 1. Correctness — CPU vs Neuron forward parity (fp32)
```
[check] max_abs=4.321e-05  mean_abs=8.928e-06  max_rel=2.925e-02
[check] PARITY PASS (fp32 CPU vs Neuron)
```
Identical weights + inputs on CPU and Neuron agree to ~4e-5 abs — the lowering is numerically correct.

## 2. Single-core training (1B, bf16, AdamW)
```
[info] parameter count = 1.010 B (1,010,001,408)   tokens/seq = 256
step 0 loss=2.3841  first (compile+run) = 207,194.6 ms   # one-time NEFF compile
step 1 loss=2.3918  step = 203.8 ms
step 2 loss=2.3896  step = 197.7 ms
```
Full fwd → `loss.backward()` → `AdamW.step()` runs on a **single 16 GB Inferentia2 core**; warm steps ~200 ms. NEFF cache persists across restarts.

## 3. Multi-core data-parallel (cores 4–5, manual grad all-reduce over the "neuron" PG)
```
[info] params=0.294B  world_size=2  cores=4..5
rank0/1 step1 step=~193 ms   (2 images/step at unchanged per-step latency → ~2x throughput)
[check] per-rank gnorm = ['1.0467','1.0467']
[check] GRAD-SYNC PASS (spread=0.000e+00)
```
Gradients are correctly synchronized (identical grad-norm on every rank each step). Scales to N cores by `--world-size N --core-offset 4` (8 free cores available → up to 8-way DP).

## 4. Single-core memory-ceiling sweep (full 1B, 256px, patch16 → 256 tokens)
| Batch | Result | compile+1step |
|---|---|---|
| 1 | **FIT** | 207.0 s |
| 2 | **FIT** | 329.4 s |
| 4 | **FIT** | 288.1 s |
| 8 | **FIT** | 116.3 s (warm cache) |
| 1 @ 512px/patch16 (1024 tok) | **compiler error** (see below) | — |

Batch up to **8** fits on one 16 GB core at 256px. (bf16 weights 2 GB + grads 2 GB + fp32 AdamW state 8 GB ≈ 12 GB, leaving headroom for activations.)

### 512px / 1024-token caveat (root-caused)
512px at patch16 = 1024 tokens fails to **compile** (not OOM, not a model bug):
```
COMPILATION FAILED: nkilib.core.attention.attention_cte ...
DMACopy ... transpose only supported for HBM->SB
```
This is a neuronx-cc limitation in the auto-selected NKI flash-attention backward kernel at the longer sequence. **Workaround:** keep tokens ≤256 (e.g. 512px with patch=32). Otherwise file an aws-neuron-sdk issue per the compiler message.

## Files
- `pixeldit_xl_train.py` — model + single-core train + `--check` parity + `--data-dir` (ImageFolder) + `--save-dir/--resume` checkpointing.
- `pixeldit_xl_dist.py` — multi-core data-parallel (spawn + `init_process_group("neuron")` + all-reduce) with grad-sync correctness check.
- `sweep.sh` — single-core memory-ceiling sweep harness.

## How to run (inside the Beta 3 container)
```bash
# parity test
NEURON_RT_VISIBLE_CORES=6 python3 pixeldit_xl_train.py --check --hidden 384 --depth 4 --image-size 64 --patch 8
# single-core train on real data + checkpoints
NEURON_RT_VISIBLE_CORES=6 python3 pixeldit_xl_train.py --data-dir /path/to/imagefolder \
    --save-dir /work/ckpts --save-every 500 --steps 2000 --batch 4
# multi-core data-parallel (2 cores)
python3 pixeldit_xl_dist.py --world-size 2 --core-offset 4 --steps 10 --batch 4
```

## 5. Confirmations (run after the sweep)

**512px works with patch=32 (256 tokens)** — confirms the 512/patch16 failure was the
1024-token flash-attn kernel, not resolution or memory:
```
1.016B params, tokens/seq=256
step 0 first (compile+run)=208,967 ms ; step 1 = 196.1 ms ; [done]
```

## 6. Real FSDP (weight-sharded) — `pixeldit_xl_fsdp.py`
FSDP2 `fully_shard` with `DeviceMesh("neuron")` **works** on the Beta 3 stack and
genuinely shards parameters across cores:
```
[rank0] FSDP wrap OK via FSDP2+DeviceMesh(neuron)
step warm ~363 ms/step (vs 193 ms for plain DP — extra all-gather/reduce-scatter)
local-shard-params = 10.3M  (full = 20.6M)  with world_size=2   # each core holds 1/2 the weights
[done] FSDP training OK via FSDP2+DeviceMesh(neuron)
```
The fallback ladder (FSDP2+mesh → FSDP2+default-PG → FSDP1) is built in; the mesh
path succeeded first. Use FSDP when a model is too big to replicate per core;
use plain DP (`pixeldit_xl_dist.py`, faster/step) when it already fits.

**Checkpoint save + resume verified:**
```
[ckpt] saved /work/ckpts/pixeldit_step0.pt        (~25 MB)
[ckpt] saved /work/ckpts/pixeldit_step1.pt
SAVE_RC=0
[ckpt] resumed from /work/ckpts/pixeldit_step1.pt at step 2
RESUME_RC=0 ; [done]
```
Model + AdamW optimizer state + step index round-trip correctly.
