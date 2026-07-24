# Segmented >16k kernel — smoke test result (PARKED)

**Date:** 2026-07-24. **Verdict: attempted, does NOT engage on the public image. No >16k speedup. PARKED as future work.**

## The A/B (same serve config, same session, 32k conc1, median-of-5, warmup)
| | 32k conc1 TTFT |
|---|---|
| OFF (torch SDPA baseline) | 10.672 s |
| ON (`GEMMA4_CTE_SEGMENTED=1`) | 10.665 s |
| **delta** | **~0% — identical** |

## Why: the kernel fell back to torch on EVERY call
- `runtime_fallback` on all 64 attention calls; `kernel_ran_ok=0`.
- The lever's load-time message fired ("SEGMENTED_CTE_LEVER: enabled"), but each per-call invocation of `NF.segmented_attention` raised an exception and degraded to the correct-but-slow torch path.
- The exact exception was swallowed by the generic fallback (not logged verbosely) — so root cause is undiagnosed. Candidates: `seqlen_q == kv_segment_size` wrapper constraint unmet at real buckets; hd512 assert on the segmented path; a shape/contract mismatch the CPU/numpy math-proof could not catch.

## What this confirms (honest)
- The CPU/numpy proof validated the MASKING MATH (cos ≥ 0.9999), NOT the live NKI call on neuronx-cc 2.26. Math-correct ≠ runs-on-device.
- **The safety design worked perfectly:** env-gated (default off), clean torch fallback, coherent output ("a city of romance, art, and culture"), and ZERO impact on the proven ≤16k path.
- **>16k stays on the segmented torch path:** 32k ~10.7s, 64k ~48s (conc1, TP32) — the honest customer number.

## Status
- Patch reverted; box left in the proven config (≤16k CTE kernel + bf16 fallback intact).
- Segmented-kernel lever = documented FUTURE work, alongside CP, for long-context. Not shipped.
- To resume: add verbose exception logging to `patch_segmented_cte.py`'s fallback branch, re-run ON-only at 32k, capture the actual error → then decide quick-fix vs real-blocker.

## Does NOT affect the real wins
The ≤16k NKI prefill kernel (7/23/40% pure-kernel, 15/33/52% stacked) is untouched and remains the headline.
