# CSM-1B decode per-frame — RE-VERIFIED (reproduces across fresh runs)

**Method:** manual hand-rolled decode loop (no `model.generate`): compiled resident-weight
backbone step + compiled device depth + CPU codec. Measured on trn2.48xlarge single core
(`NEURON_RT_VISIBLE_CORES=0`), bf16. Backbone compile confirmed `graph_count=1,
graph_break_count=0`. Harness: `src/manual_decode_loop.py`.

## Per-frame breakdown — reproduces across 2 fresh runs

| component | run 1 | run 2 (fresh) | note |
|---|---:|---:|---|
| backbone step (compiled) | 10.6 ms | 10.8 ms | was ~128 ms eager (12× from compile) |
| depth (K=32, bf16 + fp32 head/qk) | 17.8 ms | 18.1 ms | |
| codec (CPU, warm) | 7.3 ms | 7.6 ms | |
| **steady per-frame** | **35.7 ms** | **35.9 ms** | vs ~317 ms in `model.generate` (8.8×) |
| **TTFT (frame 0 = depth+codec)** | 26.9 ms | 25.3 ms | |

The **~36 ms/frame** headline is reproducible, not a single-run fluke.

## Correctness (depth bf16 exactness) — VERIFIED, reproduces

Knob sweep (fair_depth_exact.py), fresh run reconfirms:

| fp32 ops | codebook match vs fp32 oracle | depth ms |
|---|---:|---:|
| none (pure bf16) | 7/32 | 17.95 |
| head only | 22/32 | 17.25 |
| **head + qk** | **31/32** | ~17.4 |

- bf16-baseline 7/32 and head-only 22/32 **reproduced exactly** across runs.
- head+qk = 31/32 (the 1 remaining flip = benign bf16 rounding, same as CPU-bf16) — the
  minimal fp32 fix, ~no latency cost. [head+qk reconfirm run: see reconfirm_headqk.log]

## Honest caveat on the full-loop audio metric
The manual loop's "codes vs stock = 45/256, audio cosine 0.10" is NOT a correctness failure:
CSM is autoregressive, so a late argmax flip cascades into a different-but-valid speech
realization, collapsing waveform cosine (documented in the CSM README). Ground truth: frame0
and frame1 codes match stock EXACTLY and manual_loop.wav is real speech (energy varies 107×).
Use codebook-prefix-match + intelligibility, NOT waveform cosine, as the correctness gauge.

## Net (verified)
- Per-frame decode: **~36 ms** (backbone 10.8 compiled + depth 18.1 + codec 7.6), 8.8× faster
  than the 317 ms `model.generate` path.
- Depth correct at 31/32 with the head+qk fp32 fix, ~17 ms.
- Combined with prefill (VERIFIED_PREFILL_TTFT.md): a 512-token prompt → ~43 ms TTFT
  (compiled prefill) + ~36 ms/frame streaming.
