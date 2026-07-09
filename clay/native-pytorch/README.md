# Clay Foundation Model — native PyTorch on AWS Trainium

Port + training of the [Clay v1.5](https://github.com/Clay-foundation/model)
Earth-observation Masked Autoencoder on AWS Trainium using the **AWS Neuron native
PyTorch** path (`torch.device("neuron")`, eager + `torch.compile`).

> Infra identifiers (instance IDs, IPs, account, private-beta image) are redacted.
> Runs used AWS Neuron's native-PyTorch **private beta** — setup specifics omitted;
> contact the AWS Neuron team for access.

## What works
- **The whole Clay model trains on Trainium** — Encoder + Decoder + DynamicEmbedding
  (DOFA) + random masking + channel-drop + reconstruction loss + **frozen DINOv2
  teacher + representation loss** — fwd/bwd/AdamW, in **eager and torch.compile**,
  **fp32 and bf16**, at the real `large` / 256px / patch-8 config (633M params).
- **Multi-core data-parallel** verified with replicas provably in sync
  (world-group all-reduce): 2 cores on trn1; **2 and 4 cores on trn2**.

## Measured throughput (single NeuronCore, bf16, whole model incl. teacher, manual attention)
| Platform | Config | images/s |
|---|---|---|
| trn1 | base, 128px, batch16 (compile) | ~9.1 |
| trn1 | large, 256px, batch8 | ~5.7 |
| trn2 | base, 128px, batch4 | ~10.6 |
| trn2 | large, 256px | 0.31 s/step (~1.3× faster/core than trn1) |

~8% MFU — an **un-tuned floor**, not a ceiling (see `results/NEXT_STEPS.md`).

## Known beta limitations (localized)
1. **SDPA backward crashes** the runtime → use `fused_attn=False` (manual attention).
   Confirmed on both trn1 and trn2, so it's a runtime bug, not device-specific.
   Forward/inference with SDPA is fine. This is the top perf lever.
2. **>2-core collectives** fail on trn1 (`MLA indices`); **work on trn2** (2 and 4).
3. **torch.compile** can't lower the argsort masking → `mask_out` runs as an eager
   island (`@torch._dynamo.disable`); the rest compiles (~1.4× at small batch).

## Layout
- `src/claymodel/` — Clay's real Encoder/Decoder/DynamicEmbedding/backbone/utils,
  verbatim except: `fused_attn` threaded, teacher pluggable (timm→HF transformers),
  `# bf16-safe` dtype casts, `@torch._dynamo.disable` on `mask_out`.
- `src/clay_full_train.py` — whole ClayMAE + DINOv2 teacher, both losses.
- `src/clay_eager_train.py` — core MAE trainer (eager/compile smoke test).
- `src/clay_ddp_train.py` — multi-core data-parallel (torchrun + world all-reduce).
- `src/clay_bench.py` — throughput sweep.
- `src/clay_probe*.py`, `src/allreduce_smoke.py` — the probes that localized the
  SDPA-backward crash and the collective limits.
- `results/` — `RESULTS.md`, `BENCHMARKS.md`, `TRN2_RESULTS.md`, `NEXT_STEPS.md`.

## Teacher note
Uses `facebook/dinov2-large` via HuggingFace `transformers` (Clay's timm teacher pulls
torchvision, which conflicts with the beta torch build). Do **not** `pip install timm`
in the beta env.
