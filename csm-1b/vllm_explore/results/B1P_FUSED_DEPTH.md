# B1' — Fuse the depth loop into one device graph (RESULT: ~wash in bf16; B3 is the only real lever)

Follow-up to B1. The depth decoder (31 serial codebook steps/frame, ~156–178ms) is CSM's
TTFA floor. B1 showed per-step offload is the wrong shape (31 host↔device round-trips).
B1' fuses the whole loop into a **single device graph** (one round-trip/frame) and measures
whether that wins. Harness: `src/b1p_fused_depth.py`.

## What we built
A hand-driven greedy depth decode that runs as one device graph:
- keeps a `StaticCache` resident on device, drives the 31-step loop manually with **no
  host sync** mid-loop (no `.item()`), one `mark_step` + one transfer at the end;
- uses the model's **real** `CsmDepthDecoderModel.forward` (correct mask/rotary/layers/cache)
  but feeds it precomputed `inputs_embeds` so it skips its dynamic embed gather;
- applies the codebooks head as a **static** weight-slice matmul (`hidden @ head.weight[k-1]`).

## NRT_EXEC_OOB — root cause confirmed and fixed
Reusing the stock forward with `input_ids` reproduced `NRT_EXEC_OOB (status 1006)`. B1 had
shown the gather *indices* are in-range, which was puzzling. B1' resolves it: the OOB is the
**indirect copy whose offset is a runtime device tensor** —
`embed_tokens(input_ids + cache_position*vocab)` and `head.weight[cache_position-1]`. Neuron
needs the indirect-copy index to be **compile-time bounded**. Replacing the offsets with
**python-int constants** (precompute `inputs_embeds`; static head slice) removes the OOB
entirely. So the earlier "NRT_EXEC_OOB" was never an index-range bug — it was a
dynamic-vs-static index-provenance issue.

## Correctness
CSM's depth decoder **samples by default** (stochastic; hence the `temperature` warning and
a reference that changed every run). Forcing it greedy (`do_sample=False`):

| dtype | fused vs CPU-reference (greedy) |
|---|---|
| **fp32** | **32/32 codebooks EXACT** ✅ (math validated) |
| bf16 | 11/32 — bf16 device-vs-CPU argmax-flip cascade (one flip propagates AR), not a bug; vllm_v1 already accepts bf16 at cos 0.999968 vs fp32 |

## Speed — the headline
| dtype | CPU stock | fused 1-graph device | speedup |
|---|---|---|---|
| bf16 (production) | 177.9 ms | **166.8 ms** | **1.07×** |
| fp32 | 309.6 ms | 184.0 ms | 1.68× |

In the production dtype (bf16) **fusing into one device graph is essentially a wash.**

## Why fusing doesn't win — the structural truth
The depth decode is **latency-bound on 31 serial, mutually-dependent tiny steps**:
- the 31 steps are **serial** (codebook k conditions on codebook k-1) — they cannot be
  parallelized across steps, so the chip's throughput advantage doesn't apply;
- each step is **tiny** (seq len 1, a small depth-decoder transformer) — the device is
  underutilized and latency-bound; per-step bf16 device latency ≈ bf16 CPU latency;
- removing host round-trips (the point of fusing) only removes overhead that wasn't the
  dominant cost in bf16 — hence ~1.07×.

## Decision — re-prioritize Tier B
- **B1' (fused 1-graph): done, ~wash in bf16.** Keep the harness as the correctness/latency
  gate and the OOB-free static-index recipe (reusable for any on-device depth work).
- **B2 (NKI TKG megakernel): downgrade.** It collapses per-op dispatch *within* a step, but
  the bottleneck is the 31-deep **serial chain**, not per-op dispatch. Expected marginal.
- **B3 (parallel / speculative codebook decoding): PROMOTE to the primary depth lever.**
  Breaking the serial dependency is the only thing that can move the 156–178ms floor.
  Directions: (a) predict multiple codebooks per step with a small parallel head + a
  cheap verify/correct pass; (b) codebook-group factorization; (c) a distilled
  single-shot RVQ predictor. Research risk, but it's the only path to a real win.

## Repro
```bash
# correctness (deterministic): forces greedy, expect 32/32 in fp32
CSM_MODEL=<csm_1b path> python src/b1p_fused_depth.py --dtype fp32 --frames 5
# production latency:
CSM_MODEL=<csm_1b path> python src/b1p_fused_depth.py --dtype bf16 --frames 10
```
