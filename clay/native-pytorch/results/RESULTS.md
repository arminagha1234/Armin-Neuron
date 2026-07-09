# Clay MAE — Native PyTorch Beta-3 EAGER training on Trainium (trn1)

**Date:** 2026-07-08
**Instance:** `<redacted-instance>` (trn1.32xlarge, us-east-2c), 32 NeuronCores
**Stack:** Native PyTorch Beta-3 DLC (`<internal-beta-image>`)
- torch 2.11.0, torch-neuronx 2.11.3.0.1278, neuronx-cc 2.25.1280, nki 0.4.0
- driver aws-neuronx-dkms **2.28.0.0** (installed from DLC runtime_artifacts; the
  public DLAMI's 2.29 driver is not compatible with the beta userspace)
- `device = torch.device("neuron")`, eager mode, single core (`NEURON_RT_NUM_CORES=1`)

## Bottom line
**Clay's MAE trains end-to-end on Trainium in eager mode** — real
DynamicEmbedding (DOFA runtime-generated conv/linear weights) + Encoder ViT +
random masking + Decoder ViT + scatter reconstruction + Clay's verbatim
channel-drop augmentation + `per_pixel_loss`, run through forward → backward →
AdamW.step on the Neuron device. Loss decreases, gradients are finite, weights
update.

(The only omitted component is the frozen DINOv2 teacher — a standard ViT that
contributes 10% of the loss and needs a network download. It is not part of the
encoder/decoder architecture and does not affect the training-path result.)

## The one blocker we hit — and the one-line fix
| Attention path | Forward | Backward |
|---|---|---|
| `fused_attn=True`  (F.scaled_dot_product_attention) | works | **crashes** the Neuron runtime: `KaenaRuntime/tdrv/tensor.c:185: tensor_set_slice: Assertion (tensor_source->_size) >= (offset + size)` |
| `fused_attn=False` (manual matmul+softmax) | works | **works** |

Clay already exposes `fused_attn` as a flag in `claymodel/backbone.py`. Flipping
it to `False` is the entire change needed to train on Beta-3. No architectural
surgery. (SDPA *forward* is fine, so inference/embeddings are unaffected — this
is strictly an SDPA-backward limitation in the beta runtime. Consistent with the
guide's "attention bias gradient not supported in SDPA" note.)

## Measured run (fused_attn=false, overfit/deterministic-mask, lr=1e-3, small, 15 steps)
```
ClayMAE(small) params=28.9M
[step 0]  loss=0.798060  grad_norm=1.53e-02  47.98s  (incl. compile)
[step 1]  loss=0.810574  grad_norm=5.75e-02   1.72s
[step 2]  loss=0.796765  ...                  0.15s
...
[step 14] loss=0.790410  grad_norm=8.97e-03   0.16s
[check] max |Δweight| encoder.layer0.to_qkv = 1.12e-02 (UPDATED)
loss: 0.7981 -> 0.7904 (monotonic after warmup)
```
- First step 48s (NEFF compile); steady-state ~0.15 s/step (small, L=64, single core).

## How to reproduce (inside the DLC container on the box)
```bash
docker exec claytest bash -lc "cd /work && \
  NEURON_RT_NUM_CORES=1 python -u clay_eager_train.py \
  --device neuron --size small --steps 15 --grid 8 --overfit --lr 1e-3 --fused-attn false"
```
Flip `--fused-attn true` to reproduce the SDPA-backward crash.

## Files (in this dir, mounted to /work in the container)
- `claymodel/` — Clay's real `Encoder`/`Decoder`/`DynamicEmbedding`/`backbone`/`utils`
  (verbatim; only `fused_attn` threaded as a constructor arg, teacher/Lightning omitted)
- `clay_eager_train.py` — the eager training smoke test (fwd+bwd+AdamW, grad-norm &
  weight-update checks, `--overfit`, `--fused-attn`, `--lr`)
- `clay_probe*.py` — the bisection probes used to localize the crash to SDPA backward

## Talking points for the meeting
1. Native PyTorch Beta-3 stands up on trn1 (driver swap from DLC artifacts, then
   `device="neuron"` eager).
2. Clay's **forward runs as-is** on Trainium (SDPA included) — inference/embeddings
   are clean.
3. **Training works in eager** with a one-flag change (`fused_attn=False`): full MAE
   fwd/bwd/AdamW, loss decreasing, weights updating.
4. The only friction was a beta-runtime limitation in **SDPA backward**, which we
   localized precisely and worked around with Clay's existing config flag. Worth
   filing as beta feedback so SDPA-backward is fixed before GA (then the fused path
   trains too).


---

# UPDATE — WHOLE ClayMAE (with DINOv2 teacher) trains on trn1

Answering "can we train the whole Clay model on trn": **yes, verified.**

## What ran
The genuine `ClayMAE.forward` — encoder + decoder + dynamic embedding + masking +
channel-drop + **reconstruction loss (90%)** + **frozen DINOv2 teacher +
representation loss (10%)** — through fwd → bwd → AdamW on `device="neuron"`, eager.

Teacher: real **DINOv2** weights via HuggingFace `transformers`
(`facebook/dinov2-large`, `Dinov2Model`, frozen, forward-only). We swapped Clay's
timm teacher for the transformers one because `timm` hard-imports `torchvision`, and
installing torchvision forces torch 2.12 which breaks the Beta-3 neuron torch 2.11
build. transformers needs no torchvision and keeps the real DINOv2 weights.
(`torchvision v2.Resize` also replaced with `F.interpolate`.)

## Runs (single NeuronCore, eager, fused_attn=false, fp32)
| Config | Params (train / frozen) | Tokens | loss traj | repr-loss traj | step time |
|---|---|---|---|---|---|
| base, 128px, p8  | 108.7M / 304.4M | 256  | 0.8228 → 0.8007 | 1.036 → 0.821 | 82s then ~0.32s |
| **large, 256px, p8 (real cfg)** | **328.4M / 304.4M** | **1024** | 0.8159 → 0.7905 | 0.956 → 0.705 | 116s then ~0.5s |

- The **representation loss falling** proves the frozen-teacher forward + cosine +
  backprop into the encoder/proj all work on device.
- The full 633M-param model (large + teacher) fits and trains on **one** trn1 core.

## Caveats / notes for the meeting
- `fused_attn=False` still required (SDPA-backward blocker, as before). Teacher's SDPA
  is forward-only so it's unaffected.
- **bf16 blanket-cast** hit a dtype-mismatch in matmul lowering (Clay's fp32 sincos
  pos-encoding mixing with bf16 activations). Real Clay uses **bf16-mixed autocast**,
  not a blanket `.to(bf16)`; fp32 runs clean. Wiring autocast is a follow-up.
- Data is synthetic random; recon loss stays ~flat at lr 5e-6 (expected). This proves
  the *compute path* trains, not convergence on real imagery.
- Single-core here; multi-core FSDP (guide's path) is the next step for throughput.

## Repro
```bash
# base
docker exec claytest bash -lc "cd /work && NEURON_RT_NUM_CORES=1 python -u clay_full_train.py \
  --device neuron --size base  --img 128 --patch 8 --steps 6 --teacher facebook/dinov2-large --fused-attn false"
# large (real pretraining config)
docker exec claytest bash -lc "cd /work && NEURON_RT_NUM_CORES=1 python -u clay_full_train.py \
  --device neuron --size large --img 256 --patch 8 --steps 4 --teacher facebook/dinov2-large --fused-attn false"
```


---

# UPDATE 2 — bf16, multi-core, and torch.compile

## bf16 mixed precision — ✅ WORKS
Blanket `.to(bfloat16)` initially failed (`matmul: input datatypes mismatched`)
because Clay computes the sincos **pos-encoding** and **wave-embedding** in fp32,
which then met bf16 activations in a matmul. Fix = cast those to the activation
dtype (3 one-line edits in `model.py`/`factory.py`, marked `# bf16-safe`).
After that, **large trains in bf16**: loss 0.812→0.808, repr 0.955→0.913,
~0.40 s/step (vs 0.50 fp32). (Clay's own config uses bf16-*mixed* autocast; the
dtype-cast approach is equivalent for this purpose and avoids needing autocast on
the PrivateUse1 backend.)

## Multi-core data-parallel — ✅ WORKS at 2 cores, ⚠️ blocked at 4
Recipe from the DLC's own FSDP example: `dist.init_process_group("neuron")`,
launched with `torchrun --nproc_per_node N --rdzv_backend c10d`, gradients synced
with a **world-group all-reduce** (`dist.all_reduce`, the collective that works on
this stack), AdamW after.

- **world=2:** WHOLE ClayMAE(base)+teacher trained data-parallel across 2
  NeuronCores; **cross-rank weight divergence = 0.000 (replicas IN SYNC)** — real
  data-parallel training. ~0.42 s/step after compile.
- **world=4:** collective init fails — `CCOM WARN Failed to verify MLA indices` →
  `neuronInitGlobalComm failed` → `failed to setup global communicator`. A minimal
  2-rank all-reduce works (result 3.0 = expected); a 4-rank all-reduce fails the
  same way. So this is a **stack/topology limitation at >2 cores on this trn1 beta**,
  not a Clay issue. (EFA/OFI warnings are non-fatal; the failure is MLA-index/global
  comm verification.) → escalate to Neuron team; likely needs specific core-grouping
  or a newer collectives build.

## torch.compile(backend="neuron") — ⚠️ fails on argsort masking (eager is fine)
`COMPILATION FAILED: ... custom-call ... AwsNeuronTopK ... source_line=99` — i.e.
`torch.argsort(noise)` in `Encoder.mask_out`. The compile backend can't lower the
argsort-based random masking. This is the concrete version of the earlier point:
**eager runs Clay as-is; torch.compile needs the masking refactored** (e.g. replace
argsort-shuffle with a compile-friendly index selection, plus vectorizing the
channel-drop loop and moving RNG on-device).

## Net status matrix
| Front | Status |
|---|---|
| Core MAE eager (single core) | ✅ |
| WHOLE ClayMAE + DINOv2 teacher, base | ✅ |
| WHOLE ClayMAE + teacher, **large (real cfg)** | ✅ |
| bf16 | ✅ (after dtype-cast fix) |
| Multi-core data-parallel, 2 cores | ✅ (replicas in sync) |
| Multi-core, 4 cores | ⚠️ stack collective-init blocker (MLA indices) |
| SDPA attention backward (`fused_attn=True`) | ❌ beta runtime crash → use `fused_attn=False` |
| torch.compile | ❌ argsort masking (AwsNeuronTopK) → needs refactor; eager unaffected |

## Files added this round
- `clay_full_train.py` — real ClayMAE + transformers DINOv2 teacher, both losses
- `clay_ddp_train.py` — multi-core data-parallel (torchrun + world all-reduce)
- `allreduce_smoke.py` — minimal collective smoke test (2 ok / 4 fails)
- `--compile` flag added to `clay_eager_train.py`


---

# UPDATE 3 — torch.compile now WORKS (and is faster)

## Why it failed
`torch.compile(backend="neuron")` failed lowering `torch.argsort(noise)` in
`Encoder.mask_out` — it maps to an `AwsNeuronTopK` custom-call that the compiler
rejected (`COMPILATION FAILED ... AwsNeuronTopK ... source_line=99`). The random
masking also does on-graph RNG (`torch.randn`). Neither is math worth compiling —
it's data-prep that produces gather/scatter indices.

## The fix (one line)
Mark `mask_out` as an **eager island** so dynamo doesn't try to compile the
argsort/RNG, and compiles the transformer-heavy subgraphs around it:
```python
@torch._dynamo.disable
def mask_out(self, patches):
    ...
```
dynamo inserts graph breaks at the mask_out boundary, runs it eagerly (argsort works
fine in eager, as we'd already shown), and hands the surrounding compute to the
neuron compile backend.

## Results — compile works end-to-end AND speeds up the step
| Config (single core) | eager step | torch.compile step | speedup |
|---|---|---|---|
| core MAE, small, grid8, fp32 | 0.15 s | **0.07 s** | ~2.0× |
| WHOLE ClayMAE base + DINOv2 teacher, 128px, bf16 | 0.26 s | **0.17 s** | ~1.5× |

Both compiled runs trained correctly (fwd+bwd+AdamW, weights updated, loss moving).
First step is longer (compile: ~65s core / ~153s full incl. teacher download).

## Updated status matrix
| Front | Status |
|---|---|
| WHOLE ClayMAE + teacher, large (real cfg), fp32 & bf16, eager | ✅ |
| bf16 | ✅ |
| Multi-core DP, 2 cores | ✅ (replicas in sync) |
| Multi-core, 4 cores | ⚠️ stack collective-init blocker (MLA indices) |
| SDPA backward (`fused_attn=True`) | ❌ → use `fused_attn=False` |
| **torch.compile** | ✅ (after `@torch._dynamo.disable` on mask_out) — ~1.5–2× faster |

Only remaining open item: 4-core+ collectives (Neuron-team escalation).
