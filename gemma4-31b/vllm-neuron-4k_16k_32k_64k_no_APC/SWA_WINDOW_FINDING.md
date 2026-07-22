# Long-context lever: SWA windowed prior (33% but not correct yet)

**Status: promising, measured, but NOT shippable — reverted. Needs a kernel change.**

## The idea
In segmented (>16k) prefill, `_segmented_prefill_attention` gathers the **full** prior-KV span
(`padded_kv_len = max_blocks * block_size`, sized for `max_model_len`) and passes it as `k_prior`
for **every** layer, masking only via the kernel's `sliding_window` flag. The **50 SWA layers**
(of 60) only ever attend to the last `sliding_window` (1024) keys, so they carry the entire prior
history for nothing. Windowing the SWA prior to its trailing ~1024 tokens should cut the dominant
prior-gather cost on 50/60 layers.

## What we measured (on-device, trn2, TP32, clean single-tenant box, conc=1)
Implemented `patches/patch_swa_window_prior.py` — for SWA layers, gather only the trailing
`w_blocks = ceil(window/bs)+1` blocks at a dynamic offset (static shape → trace-safe); global
layers keep the full span. It **compiled cleanly** (no trace/SIGSEGV) and was **fast**:

| input | full-span (correct) | SWA-windowed | Δ |
|---|---|---|---|
| 32k | 3.06 s | **2.06 s** | **−33%** |
| 64k | 6.10 s | **4.11 s** | **−33%** |

## Why it's NOT shippable (correctness failure)
Token-parity gate (fixed 18k prompt, seed 42, greedy) **FAILED**:
- full-span: `' lulub bib kiyizev lulit coqic ...'`
- windowed: `' luv luv luv luv ...'` (degenerate)

(A short chat prompt still returns 'Paris' — it's single-chunk, so the cross-chunk window is never
exercised. That's why a quick sanity check misses the bug.)

**Root cause:** `NF.flash_attention` masks prior keys by their **absolute tensor index**, not
relative to `prior_used_len`. Compacting the SWA prior into a window slice (labeled indices
`[0, 1056)`) makes the current chunk — at true position ~8192 — see those keys as ~7000 tokens away,
i.e. **outside** the sliding window → **all prior masked** → SWA layers lose cross-chunk context →
degenerate output. The "prior sits immediately before the current chunk (relative)" assumption is
wrong for this kernel.

## What a correct fix needs — two viable paths
The naive `k_prior` compaction fails because the kernel masks prior keys by absolute index. Two
correct options:

1. **Kernel change (cleanest):** add a `prior_start_offset` arg to `NF.flash_attention` so the
   sliding-window mask on the prior uses `(offset + i)` as the key's absolute position. Then the
   caller passes a compact window slice + its true offset. nkilib/functional change — escalate to
   the Neuron kernel owners.

2. **model.py-only (no kernel change), via square attention over the window:** for SWA layers, drop
   `k_prior` entirely and instead attend the current chunk over a `[window_prior ++ current]`
   **contiguous** k/v with the queries also extended by `window` dummy rows (so `q_len == k_len` and
   the standard causal+sliding_window mask applies with correct RELATIVE positions); slice off the
   first `window` output rows. Gather only the trailing `w_blocks` prior blocks (static shape,
   dynamic offset — the same trace-safe gather this patch already does). Cost: SWA attention becomes
   `(window+chunk)²` instead of `chunk × (full_prior+chunk)` — a big net win at long context because
   it removes the full-prior gather/scan (window≈1056 ≪ prior at 32k/64k), at the price of ~1.2–1.3×
   the current-chunk attention and `window` wasted query rows. Needs implementation + the same
   token-parity gate. This is the recommended model.py-only route; it was NOT implemented here
   (out of the ≤8k customer scope, and it's a non-trivial restructure warranting careful validation).

Either path keeps the global (full-causal) layers on the full prior — only the 50 SWA layers change.

**Caution for anyone quoting "just slice k_prior caller-side":** validate **token parity on a
multi-chunk prompt** first. The naive caller-side slice is ~33% faster but silently wrong.

## Reproduce
```bash
# apply (on a segmented serve), then compare:
python3 patches/patch_swa_window_prior.py
# baseline (revert first) vs windowed — must match, but they DON'T:
python3 patches/parity_check.py      # deterministic 18k prompt, prints the 40 greedy tokens
```
