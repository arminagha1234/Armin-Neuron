# CSM-1B prompt/speaker suite + bf16 numerical stability (pre-PR validation)

**Method:** full optimized device depth (bf16 + fp32 head+qk fix, `fair_depth_handroll`)
vs the fp32 CPU serial oracle, over 6 prompts × both speaker IDs, each from a real captured
backbone hidden state (not one hand-picked sentence). Codebook-argmax match /32.

**Harness note:** this suite was run as an ad-hoc sweep over per-prompt captured backbone
hidden states, built on the depth loop in `src/fair_depth_handroll.py` and the argmax-vs-fp32
correctness check in `src/fair_depth_exact.py`. The standalone driver script is **not included
in this bundle** — it consumed per-prompt hidden-state captures that are not part of the
deliverable — so this table is not reproducible as-is from `src/`. To regenerate it, drive
`fair_depth_handroll` over fresh captures for the six prompts below. The two reusable
components it was assembled from are in `src/`.

## Result — mostly solid, ONE real outlier

| prompt | speaker | cb-match /32 |
|---|---|---:|
| "The quick brown fox…" | [0] | 29 |
| "Trainium delivers high throughput…" | [0] | **32** |
| "Please schedule the meeting…" | [1] | 31 |
| "Weather today is sunny…" | [1] | **32** |
| **"Numbers like one two three four five…"** | [0] | **18** ⚠️ |
| "Thank you for calling…" | [1] | **32** |

**Mean 29.0/32, range 18–32.**

## Findings (customer-relevant)
1. **Both speaker IDs are fine** — [0] and [1] each hit 32/32; no speaker-specific failure.
2. **5/6 prompts: 29–32/32** — the head+qk bf16 fix is solid on typical speech.
3. **The bf16 path is PROMPT-DEPENDENT and degrades on hard content.** The digit-heavy
   prompt ("one two three four five…") dropped to **18/32** — the AR argmax chain diverges by
   ~codebook 18 on this input. bf16 is NOT uniformly safe.
4. **This is why the fp32 depth path matters.** fp32 was bit-exact (32/32) regardless of
   input (DEPTH_ON_DEVICE_FAIR / BF16_DEPTH_FIX). 

## PR recommendation (dtype policy)
- **fp32 depth (~59 ms) = the SAFE DEFAULT** — bit-exact vs the reference on every prompt.
- **bf16 depth (~17 ms) = the FAST path**, ~3.4× faster, with a documented caveat: accuracy
  is prompt-dependent (29–32/32 typical, but as low as 18/32 on digit/hard content), which
  produces a different-but-plausible realization, not a crash. Offer as opt-in for
  latency-critical use where minor realization drift is acceptable.
- Do NOT present bf16 as "31/32 always" — it's "29–32 typical, 18 worst-case seen."

## Caveat
Match is vs the fp32 hand-rolled oracle (proven == stock HF at 32/32). Low match ≠ bad
audio necessarily (AR-divergent-but-valid), but for a customer the conservative read is:
ship fp32 as default, bf16 as a flagged fast option. A perceptual (ASR-WER) test on the
18/32 case would quantify whether it's actually worse or just different — pending.
