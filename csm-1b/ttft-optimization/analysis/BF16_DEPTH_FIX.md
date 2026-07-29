# bf16 depth exactness — FIXED: fp32 on head+qk → 31/32 at 17.4ms (no speed cost)

The compiled bf16 device depth diverged from the fp32 serial oracle (7/32, cascading after
~6 codebooks) while CPU bf16 got 31/32. A knob sweep (fair_depth_exact.py, each op flippable
to fp32 independently, measured on-device) found the minimal fix.

## Result

| config (fp32 ops) | codebook match | depth ms |
|---|---:|---:|
| bf16 baseline (all bf16) | 7/32 | 17.83 |
| head | 22/32 | 17.39 |
| head + rope | 7/32 | 17.69 |
| **head + qk** | **31/32** | **17.43** |
| av + head + qk | 31/32 | 17.45 |

## The fix
Run just **two ops in fp32**: the **head/argmax matmul** (`out @ head.weight[k]`, a 2051-way
argmax — precision-sensitive) and the **Q·Kᵀ attention scores**. Everything else stays bf16.
- Recovers **31/32** — the single remaining flip (position 26: 1841 vs oracle 1893) is the
  same benign bf16-rounding flip that CPU-bf16 also shows, i.e. NOT a device defect.
- Costs **nothing measurable**: 17.43 ms vs 17.83 ms pure-bf16 (the head is one small
  1024×2051 matmul/step; qk is tiny at seq-len ≤32).
- `head` alone → 22/32; `qk` is the other necessary piece; `rope` fp32 doesn't help; `av`
  adds nothing.

## Why
The divergence was localized to reduced-precision accumulation in exactly the two
argmax-deciding reductions (the vocab head and the attention scores). fp32-accumulating
those two removes the flips that were cascading through the autoregressive chain.

## Impact on the manual loop
Combined with the compiled backbone (10.7 ms) + CPU codec (9.1 ms):
- **~37 ms/frame, faithful (31/32)** — vs the pure-bf16 loop's cosine 0.098 (broken), and
  vs 317 ms in model.generate.
- No need for the fp32-depth fallback (59 ms) — bf16+fp32-head+qk gets exactness at ~17 ms.

Apply the `head`+`qk` fp32 knobs in the manual loop's depth stage. Script:
`fair_depth_exact.py` (EXACT_CONFIGS knob sweep).
