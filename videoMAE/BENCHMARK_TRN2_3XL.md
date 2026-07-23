# VideoMAE v2 pretraining on Trainium2 — throughput benchmark

**Hardware:** trn2.3xlarge (1 Neuron device, 4 NeuronCores, LNC2, 96 GB device mem)
**Stack:** Native PyTorch Beta 3 DLC — torch 2.11 + torch-neuronx 2.11.3 + neuronx-cc 2.25
**Model:** `OpenGVLab/VideoMAEv2-Base` pretraining (encoder 86.2M + decoder 14.8M = 101.3M params)
**Workload:** one training step = fwd + bwd + AdamW, tube masking 0.9, 16×224×224 clips
**Method:** steady-state — `warmup` steps skip the step-0 NEFF compile, then median of `iters`
timed steps; each step force-synced via `loss.item()` (beta runs async). Fixed batch reused
so the graph is static. Data-loading is excluded (measures device train-step throughput).

## Eager mode (`torch.device("neuron")`), single core

| dtype | batch | step (ms) | videos/s | peak mem (GB) |
|---|---:|---:|---:|---:|
| fp32 | 1 | 220 | 4.54 | 2.81 |
| fp32 | 2 | 330 | 6.07 | 3.83 |
| fp32 | 4 | 541 | 7.39 | 6.07 |
| fp32 | 8 | 1128 | 7.09 | 10.13 |
| bf16 | 1 | 192 | 5.21 | 1.40 |
| bf16 | 2 | 303 | 6.61 | 1.98 |
| bf16 | 4 | 460 | 8.69 | 3.04 |
| bf16 | 8 | 800 | 10.00 | 5.28 |

Observations:
- **bf16 + batch is the first lever:** fp32/batch-1 (4.54 v/s) → bf16/batch-8 (10.0 v/s)
  = **2.2× with no kernels**, and bf16 roughly halves memory.
- fp32 throughput saturates ~batch 4 and regresses at 8; bf16 keeps scaling and still uses
  only 5.3 GB of 96 GB at batch 8 — lots of headroom for larger batch.

## torch.compile (`backend="neuron"`), single core

`model = torch.compile(model, backend="neuron", dynamic=False)`. Measured the same way
(steady-state, compile excluded). `dynamic=False` + **one process per shape** is required —
see the caveat below.

| dtype | batch | step (ms) | videos/s | peak mem (GB) | vs eager |
|---|---:|---:|---:|---:|---:|
| bf16 | 1 | 114.0 | 8.77 | 1.83 | **1.68×** |
| bf16 | 2 | 191.8 | 10.43 | 2.61 | **1.58×** |
| bf16 | 4 | 316.7 | 12.63 | 4.05 | **1.45×** |
| bf16 | 8 | 513.8 | **15.57** | 7.09 | **1.56×** |
| fp32 | 1 | 124.7 | 8.02 | 3.66 | **1.77×** |
| fp32 | 4 | 325.5 | 12.29 | 8.09 | **1.66×** |

Observations:
- **torch.compile beats eager 1.45–1.77×** across the grid — cross-op fusion by `neuronx-cc`
  on a captured graph vs eager's per-op dispatch. It's the single biggest lever measured here.
- **Best config: bf16 + batch 8 + compile = 15.57 videos/s** — **3.4×** over the original
  eager/fp32/batch-1 baseline (4.54). And it keeps scaling with batch where eager plateaued.
- With compile, the fp32↔bf16 throughput gap narrows (compile fp32-b4 12.29 ≈ bf16-b4 12.63);
  bf16's remaining edge is mostly memory (enables bigger batches).

**Caveat (beta):** compiling >1 input shape *in one process* fails — after a second batch
size, TorchDynamo recompiles with dynamic shapes, which the beta rejects
(`BackendCompilerFailed: ... Neuron backend`). Workaround: `dynamic=False` **and** a fresh
process per shape (or shape bucketing in a real training loop, which uses one fixed shape).
First-compile per shape is slow (host-bound `walrus_driver`, minutes) but hits the persistent
NEFF cache after.

## Multi-core (FSDP2, 2 NeuronCores) — directional

The FSDP2 pretraining run (batch 2/rank) held ~0.7 s/step including host data-gen ≈
**~5.7 videos/s total** — *below* single-core big-batch. Expected: FSDP shards params and
adds all-gather / reduce-scatter every step, which is not worth it for a 101M-param model on
2 cores. For scaling a model this size, **DDP** (replicate + gradient all-reduce) is the right
tool and should approach ~2×; FSDP earns its keep when the model does not fit on one core.

## Where the compute goes (for the kernel question)

Per-layer compute splits into linear/matmul work (QKV+proj+MLP ≈ `12·N·C²`) and attention
(`2·N²·C`), so attention's share is `N/6C`:

| block | N (tokens) | C | attention share of block |
|---|---:|---:|---:|
| encoder | 160 (visible) | 768 | ~3.5% |
| decoder | 1568 (all)    | 384 | ~40% |

The decoder dominates total FLOPs (processes 1568 tokens vs the encoder's 160), so overall
**attention ≈ 29% of training FLOPs, almost all in the decoder's 1568-token full attention**;
matmuls (QKV/proj/MLP) are ~70%.

## Would custom NKI kernels speed up training?

Modestly, and **not first.** Grounded in the numbers above and this repo's own precedent:

- **Matmuls (~70%)** already map near-optimally to the tensor engine via `neuronx-cc`. Hand
  kernels rarely beat the compiler here — see `nki-kernels/STATUS.md` and the qwen
  `BENCHMARK_NKI_VS_EAGER.md`, where a hand-written decode-attention kernel was **0.80×
  (20% slower)** than the eager path the compiler auto-fuses.
- **Attention (~29%, decoder)** is the only real target. A **flash-style fused attention
  kernel (forward *and* backward, non-causal)** that avoids materializing the 1568×1568
  scores could recover part of it — realistic end-to-end gain ~**1.15–1.3×**, *if* it beats
  the compiler (not guaranteed, per the qwen result). The beta's `FlexAttention` is
  forward-only + causal, so it does not cover this training path — it would be custom NKI work.

**Order of levers (cheapest first):**
1. **bf16 + batch** — measured **2.2×**, free. Also enables `torch.compile` (see above).
2. **`torch.compile(backend="neuron")`** — cross-op fusion; measured delta in the table above.
3. **VideoMAE v2 dual masking** — algorithmically shrinks decoder tokens → directly cuts the
   ~29% attention + decoder MLP.
4. **THEN** a profile-driven NKI flash-attention kernel for the decoder — only after a
   `neuron-profile` capture shows a concrete inefficiency the compiler is leaving on the table
   (the methodology in `nki-kernels/STATUS.md`).
