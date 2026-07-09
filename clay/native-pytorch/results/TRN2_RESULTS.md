# Clay on trn2 — Steps 1–6 (Melbourne, ap-southeast-4)

Instance `<redacted-instance>` (**trn2.3xlarge**, Trainium2, LNC2), 4 NeuronCores,
**96 GB device mem (24 GB/core, 1.5× trn1)**. Beta-3 native PyTorch DLC, driver
aws-neuronx-dkms 2.28 (swapped from public 2.29). torch 2.11, transformers 5.13.

## Step-by-step

### 1–2. Setup ✅
Same Beta-3 DLC works on trn2. Pulled image (cross-region from us-east-1 ECR),
installed runtime debs, reloaded driver → `neuron-ls` shows 4 cores / 96 GB.
Copied `clay/` folder, `pip install transformers` in the container.

### 3. SDPA-backward retest ❌ (still broken on trn2)
`fused_attn=True` → **same `tensor_set_slice` runtime assertion** as trn1. So the
SDPA-backward crash is a **Beta-3 runtime bug, not trn1-specific**. `fused_attn=False`
(manual attention) remains required. → escalate to Neuron team; still the top MFU lever.

### 4. Whole model, large/256px ✅ (faster than trn1)
`ClayMAE(large)` + DINOv2-large teacher, bf16, single core:
loss 0.812→0.803, repr 0.954→0.861, weights updated.
**0.31 s/step** vs trn1's 0.40 s (~1.3× faster/core).

### 5. Single-core throughput sweep (base, 128px, bf16, eager) ⚠️ partial
| batch | step_s | samp/s | notes |
|--:|--:|--:|--|
| 1 | 0.218 | 4.58 | |
| 2 | 0.465 | 4.30 | noisy (beta compiler variance) |
| 4 | 0.376 | **10.64** | best; > trn1 peak (8.9) |
| 8 | — | — | **compile error** `aten::add.Tensor` (trn2 beta compiler quirk) |
- Best single-core so far **10.6 samp/s @ batch 4** (trn1 peaked 8.9). batch-8 compile
  failure and batch-1/2 noise are trn2 beta-compiler issues to chase (scratchpad flags,
  or file with Neuron team).

### 6. Multi-core data-parallel ✅ — scales past 2 (trn1's blocker is GONE)
| world | step_s | replicas | aggregate img/s (batch1/rank) |
|--:|--:|--|--:|
| 2 | 0.35 | IN SYNC (Δ=0.000) | 5.7 |
| **4** | 0.35 | **IN SYNC (Δ=0.000)** | **11.4** |
- **trn1 failed at world=4** (`Failed to verify MLA indices` / global-comm init).
  **trn2 runs 4-core data-parallel cleanly** — collectives scale. This was the big open
  question; trn2 answers it.
- ~2.5× aggregate throughput from 1→4 cores at batch1/rank (sub-linear because batch=1
  starves each core + teacher/collective overhead; higher per-rank batch will scale
  better).

## Net trn2 vs trn1
| | trn1 (16 GB/core) | trn2 (24 GB/core) |
|--|--|--|
| Whole model large bf16 step | 0.40 s | **0.31 s** |
| Single-core best (base) | 8.9 samp/s | **10.6 samp/s** |
| SDPA backward | ❌ crash | ❌ crash (same bug) |
| Multi-core DP | 2 ✅ / 4 ❌ | 2 ✅ / **4 ✅** |

## Open items on trn2
- SDPA-backward crash (Beta bug) — top lever, escalate.
- base batch-8 compile error (`aten::add.Tensor`) + batch-1/2 timing noise — beta
  compiler; try `NEURON_CC_FLAGS="--hbm-scratchpad-page-size=2048"` +
  `NEURON_SCRATCHPAD_PAGE_SIZE=2048`, and re-sweep.
- Push per-rank batch on 24 GB/core; combine with 4-core DP for real aggregate img/s.
- LNC2 tuning (`NEURON_RT_VIRTUAL_CORE_SIZE=2`) not yet explored.
