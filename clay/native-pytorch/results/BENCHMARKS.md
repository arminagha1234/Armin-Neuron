# Clay on Trainium — throughput sweep (trn1, single NeuronCore, bf16)

Instance `<redacted-instance>` (trn1.32xlarge), Beta-3 native PyTorch, eager & compile,
`fused_attn=False` (manual attention — SDPA-backward blocked), 1 NeuronCore-v2.
MFU denominator = **95 BF16 TFLOPS/core** (190 per 2-core chip).
MFU is a **rough** estimate: `FLOPs/step ≈ 6·(P_enc·tok_enc + P_dec·tok_dec)·B + 2·P_teacher·tok_teacher·B`
(fwd+bwd for trainable, fwd-only for frozen teacher). Throughput (samp/s, tok/s) is exact/measured.

## base model, 128px, patch 8 (L=256), teacher=dinov2-large
| batch | eager step_s | eager samp/s | eager MFU% | compile step_s | compile samp/s | compile MFU% | compile speedup |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1  | 0.243 | 4.11 | 3.8 | 0.172 | 5.82 | 5.4 | 1.42× |
| 2  | 0.370 | 5.40 | 5.0 | 0.282 | 7.09 | 6.6 | 1.31× |
| 4  | 0.609 | 6.57 | 6.1 | 0.488 | 8.19 | 7.6 | 1.25× |
| 8  | 0.950 | 8.42 | 7.8 | 0.909 | 8.81 | 8.2 | 1.05× |
| 16 | 1.791 | 8.93 | 8.3 | 1.754 | 9.12 | 8.5 | 1.02× |
| 32 | 3.657 | 8.75 | 8.1 | — | — | — | — |

## large model, 256px, patch 8 (L=1024) — real v1.5 pretraining config, eager
| batch | step_s | samp/s | tok/s | MFU% |
|--:|--:|--:|--:|--:|
| 1 | 0.542 | 1.85 | 1890 | 2.7 |
| 2 | 0.505 | 3.96 | 4055 | 5.7 |
| 4 | 0.849 | 4.71 | 4822 | 6.8 |
| 8 | 1.399 | 5.72 | 5856 | 8.2 |

## Takeaways
1. **batch=1 starves the core** (2.7–3.8% MFU). Batching to 8–16 roughly doubles
   throughput and lifts MFU to ~8%. Sweet spot: batch 8–16.
2. **torch.compile helps most at small batch** (~1.4× at B=1) and converges with eager
   at large batch (~1.0× at B=16) — it removes per-op dispatch overhead, which stops
   mattering once matmuls dominate. Note: compile can't change batch shape in-process
   (guide's "no dynamic shapes"); each batch = a fresh compile.
3. **~8% MFU is the honest un-tuned floor**, not a ceiling, because:
   - manual attention (`fused_attn=False`) — the fast fused kernel is blocked by the
     SDPA-backward beta bug; fixing it is the biggest single MFU lever.
   - single core, batch-limited by 16 GB.
   - the frozen DINOv2 teacher runs every step (counted in FLOPs but it's overhead you
     could amortize / cache).
   - no autocast/kernel tuning, no scratchpad page-size tuning (runtime suggested
     `--hbm-scratchpad-page-size=2048`).
4. **Levers to raise MFU, in expected impact order:** SDPA-backward fix (fused attn) →
   multi-core (once >2-core collectives work) → larger batch → autocast-mixed →
   scratchpad/compiler flags.

## MFU recommendation
Report **throughput (samp/s, tok/s)** as the headline now; treat MFU (~8%) as a labeled
"un-tuned beta floor, manual-attention, batch-limited, single-core." Revisit MFU as the
primary metric only after SDPA-backward + multi-core land.

## Repro
```bash
# eager sweep
python clay_bench.py --size base --img 128 --patch 8 --dtype bfloat16 --batches 1,2,4,8,16,32
# compiled (per-batch, fresh process each — compile can't reshape in-process)
for b in 1 2 4 8 16; do python clay_bench.py --size base --img 128 --patch 8 \
  --dtype bfloat16 --batches $b --compile; done
# real-config large
python clay_bench.py --size large --img 256 --patch 8 --dtype bfloat16 --batches 1,2,4,8
```
