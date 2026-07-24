# Qwen3.5-4B trn2 training THROUGHPUT — final measured table (2026-07-24)
Full-32L, bf16, NKI GDN kernel (fixed, finite), + compiler flags. All ALLFIN=True (loss finite+decreasing).

| Config | world×LNC | phys cores | seq | bs | ckpt | s/step | agg tok/s | MFU |
|--------|-----------|-----------|-----|----|----- |--------|-----------|-----|
| single-core NKI | 1×LNC2 | 2 | 512 | 1 | no | 2.50 | 205 | ~3% |
| A_seq256_2r | 2×LNC2 | 4 | 256 | 1 | no | 2.91 | 176 | 1.4% |
| A_ckpt_2r | 2×LNC2 | 4 | 512 | 1 | yes | 4.47 | 229 | 1.8% |
| D_4r_lnc2 | 4×LNC2 | 8 | 512 | 1 | yes | 5.14 | 399 | 1.6% |
| **C_ckpt_8r** | **8×LNC1** | **8** | 512 | 1 | yes | 4.64 | **883.8** | **3.5%** |
| (8r bs8 / seq1024 / seq2048) | 8×LNC1 | 8 | — | — | — | OOM | — | — |

## HEADLINE: 883.8 tok/s (8-rank LNC1 + activation-ckpt + NKI kernel), full-32L, finite.
= ~10.6x the 83 tok/s single-core-stock starting point. Best MFU 3.5%.

## KEY FINDINGS
- Activation checkpointing is what let the fast (memory-hungry) NKI kernel FIT at 8-rank LNC1 (8.6GB/rank).
  Without ckpt, 8-rank full-32L OOMs. bs>1 and seq>512 still OOM at 8-rank (8.6GB/rank too tight).
- MFU stays LOW (1.4-3.5%). Root causes: (a) bs=1 starves tensor engine — THE MFU lever, but blocked by
  the 8.6GB/rank memory ceiling at LNC1; (b) GDN linear-attn is inherently low arithmetic intensity;
  (c) beta stack. Bigger batch (the real MFU lever) needs more memory/core than LNC1 gives.
- Throughput ceiling on THIS stack ~= 8 phys cores (16/32 ranks fail collective topology). Full 64-core
  node would ~8x this (needs AWS collective-topology fix — out of our hands).

## WALL-CLOCK (per token count; plug in customer's real number — tokens ÷ tok/s @ 883.8):
| dataset tokens | @883.8 tok/s (8-rank today) |
|---|---|
| 1B  | ~19 min |
| 5B  | ~1.6 hr |
| 10B | ~3.1 hr |
| 30B | ~9.4 hr |
(customer "1-3 days on GPUs" ~= 3-30B tokens -> ~1-9 hr on trn2 at this config)

## BATCH/MFU SWEEP — FINAL (2026-07-24, squeeze agent)
Full-32L LoRA, NKI kernel, compiler flags, activation-ckpt. agg tok/s / MFU%:
| config | agg tok/s | MFU% |
|--------|-----------|------|
| 2-rank LNC2 seq512 bs1 | 229.1 | 1.81 |
| 2-rank LNC2 seq512 bs2 | 331.4 | 2.62 |
| 2-rank LNC2 seq512 bs4 | 401.5 | 3.17 |
| 4-rank LNC2 seq512 bs1 | 398.9 | 1.57 |
| 4-rank LNC2 seq512 bs2 | 641.6 | 2.53 |
| **8-rank LNC1 seq512 bs1** | **883.8** | **3.49** ← HEADLINE |
| 8-rank LNC1 seq256 bs2 | 742.9 | 2.93 |

KEY: core count = biggest tok/s lever; batch = biggest MFU-per-core lever (2-rank bs1→bs4: 1.81→3.17%).
But batch/seq scaling needs LNC2's ~48GB/core; 8-rank LNC1 (24GB/core) is hard-capped at seq512 bs1
(larger → fragmentation OOM). Peak MFU ~3.5%, below the 20-25% good-trn2 target — remaining overhead is
the kernel's per-(b,h) Python loop, eager attention, and checkpoint recompute. Two mechanism findings:
(1) ≥8-rank REQUIRES FSDP2 fully_shard + explicit init_device_mesh (FSDP1 wrapper dies no_hier no_mesh);
(2) concurrent 8-rank jobs on disjoint cores crash (db_physical_core assertion) — run 8-rank serially.
