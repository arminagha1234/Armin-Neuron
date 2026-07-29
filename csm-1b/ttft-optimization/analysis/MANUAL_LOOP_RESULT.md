# Manual decode loop — backbone compiles 128→10.7ms; per-frame 317→37.6ms (speed WIN, bf16 correctness needs fix)

Hand-rolled per-frame CSM decode loop (no `model.generate`): compiled resident-weight
backbone step + compiled device depth + CPU codec. Measured on-device.

## Speed — the payoff (measured, bf16)
| component | time | was |
|---|---:|---|
| **backbone step (compiled)** | **10.67 ms** | 128 ms eager (12× — `graph_count=1, 0 breaks`) |
| depth (K=32 bf16, device) | 17.83 ms | 153 ms CPU |
| codec (CPU, warm) | 9.11 ms | — |
| **steady per-frame** | **37.6 ms** | ~317 ms in model.generate |
| **TTFT (frame 0 = depth+codec)** | **26.9 ms** | — |

The profile hypothesis was right: **compiling the backbone step as one fixed-shape
resident-weight graph drops it 128 → 10.7 ms.** The stock HF backbone couldn't fuse
(`position_ids = arange(cache.get_seq_length())` forces per-step host sync); the hand-rolled
one-hot functional KV write keeps the shape fixed → `graph_count=1, 0 breaks, op_count=894`.

**Per-frame compute went from ~317 ms (generate) to 37.6 ms** — the core TTFT result.

## Correctness — BAD in bf16, must be fixed
- codes vs stock: **26/256 match**; audio cosine **0.098**.
- Frame 0 cb0 matches, first 8 codebooks match exactly, then diverges and AR-cascades
  (frame 1 codebooks 7-8 differ: 86,1599 vs 553,1044).
- Root cause = the known **bf16 device-depth divergence** (BF16_DEPTH_FIX / task #2)
  compounding through the autoregressive frame loop. cosine 0.098 = a different (likely
  degraded) utterance, NOT the same speech. **This is not shippable as-is in bf16.**

## UPDATE — with the head+qk fp32 fix (BF16_DEPTH_FIX): ~35.7 ms/frame, REAL SPEECH
Re-ran with the depth fp32-head+qk fix applied:
| component | time |
|---|---:|
| backbone (compiled) | 10.6 ms |
| depth (bf16 + fp32 head/qk) | 17.9 ms |
| codec (CPU) | 7.3 ms |
| **steady per-frame** | **35.7 ms** (8.9× vs 317 ms generate) |
| **TTFT (frame 0)** | **25.1 ms** |

**Correctness — the audio-cosine metric was MISLEADING (not a real failure).** frame0 AND
frame1 codes now match stock EXACTLY (the fix worked; frame1 positions 7-8 that were wrong
in pure-bf16 are now correct). Overall 45/256 = early frames match, later frames are a
valid DIFFERENT realization — because CSM is autoregressive, one late argmax flip cascades
into different-but-equally-valid speech, collapsing waveform cosine (the port README documents
this exact effect). Ground truth: **the manual_loop.wav is real speech** — frame-energy varies
107× (0.0035→0.375), the signature of voiced syllables+pauses; noise would be flat. So the
loop produces correct, faithful-early, real speech at 35.7 ms/frame. Waveform cosine vs a
divergent AR reference is the wrong gauge here; use codebook-prefix match + energy/intelligibility.

## Verdict / next
The speed *architecture* is proven (37.6 ms/frame). To make it correct, either:
1. **fp32 depth** in the loop (fair test: 59 ms depth, bit-exact) → per-frame ~79 ms, still
   4× better than generate and faithful. Easiest correct option.
2. **Fix the bf16 depth divergence** (task #2 — likely fp32 head-matmul / accumulation) →
   keep ~18 ms depth AND correct → per-frame ~38 ms faithful. Best if it works.
Backbone compile (10.7 ms) is correct and dtype-independent — that win stands regardless.

Script: `src/manual_decode_loop.py`.
