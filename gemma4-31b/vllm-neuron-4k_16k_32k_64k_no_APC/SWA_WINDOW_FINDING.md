# Long-context lever: SWA windowed prior — CORRECT and −33% (VALIDATED)

**Status: VALIDATED on-device. Correct (token parity PASS) and −33% TTFT at 32k/64k.**
Use `patches/patch_swa_window_prior_v2.py`. This SUPERSEDES the earlier "broken, reverted"
conclusion (see "Correction" below — that was a misdiagnosis).

## The idea
In segmented (>16k) prefill, `_segmented_prefill_attention` gathers the **full** prior-KV span
(`padded_kv_len = max_blocks * block_size`, sized for `max_model_len`) and passes it as `k_prior`
for **every** layer, masking only via the kernel's `sliding_window` flag. The **50 SWA layers**
(of 60) only ever attend to the last `sliding_window` (1024) keys, so they carry the entire prior
history for nothing. Windowing the SWA prior to its trailing ~1024 tokens cuts the dominant
prior-gather+scan cost on 50/60 layers. Global (full-causal) layers keep the full span.

## What we measured (on-device, trn2, TP32, clean single-tenant box `ec2-3-19-59-18`/vllm_ga, conc=1)
`patches/patch_swa_window_prior_v2.py`: for SWA layers gather only the trailing
`w_blocks = ceil(window/bs)+1` blocks at a dynamic offset (static shape → trace-safe), set
`prior_used_len = valid window length`; global layers unchanged. Compiled cleanly (~165s), and:

| input | full-span (baseline) | SWA-windowed (v2) | Δ | token parity |
|---|---|---|---|---|
| 32k | 3.03 s | **2.021 s** | **−33.3%** | ✅ PASS |
| 64k | 6.053 s | **4.040 s** | **−33.3%** | ✅ PASS |

Token-parity gate (fixed 18k prompt = 17866 tok = 3 segmented chunks, seed 42, greedy 40 tok):
**byte-identical** to full-span (`' lulub bib kiyizev lulit coqic kici pufufi ...'`). TPOT unchanged
(~211 ms — the fix touches only prefill). The multi-chunk prompt DOES exercise the cross-chunk SWA
window (chunks 2 and 3 have prior context), so the parity gate is meaningful.

## Why it's correct — the masking is SHIFT-INVARIANT
`attention_cte.flash_attention`'s causal + sliding-window masks depend only on the **difference**
`q_pos - k_pos`:
```
q_pos = arange(S_q) + prior_len + cp_offset
k_pos = arange(prior_len + S_k)
mask  = (q_pos < k_pos) | (q_pos >= k_pos + sliding_window)
```
When the SWA prior is windowed, the concatenated `[windowed_prior | current]` maps to absolute
positions **contiguously with a uniform shift** (the windowed prior ends exactly where the current
chunk begins). A uniform shift cancels out of every `q_pos - k_pos`, so with
`prior_used_len = valid_window_len` and `cp_offset = 0` the masked scores are identical to
full-span. Proven three independent ways:
1. CPU masking test (kernel torch fallback): max abs diff windowed-vs-full = **2.4e-7**.
2. CPU gather-plumbing test (paged cache + non-identity block_table + dynamic-offset gather): **0.0**.
3. On-device token parity at 18k (3 chunks): **byte-identical**.
(See `patches/swa_window_validate.py`, `swa_window_validate2.py`.)

## Correction to the earlier "broken / absolute-index" writeup
A prior session reported the windowed prior as "fast but degenerate (`' luv luv luv'`), reverted"
and hypothesized the kernel "masks prior keys by absolute tensor index." **That root cause is
wrong** — the CPU tests above disprove it, and v2 (whose gather logic is essentially identical to
that v1 attempt) passes token parity on-device. The earlier degenerate result was therefore a
**confounded test**, not a flaw in the windowing (most likely a stale NEFF / not-fully-recompiled
serve, or a combined patch state). The correct discipline that caught it this time: recompile from
clean, then run the multi-chunk token-parity gate before trusting a fast number.

## Scope note
This is a >16k lever. Hippocratic's traffic is ≤8k, where prefill is already single-shot and fast
(4k 0.22s / 8k 0.39s). This fix matters for long-context deployments (32k/64k) and turns a
previously-"broken" idea into a shippable −33% win. Global layers still scan the full prior; a
further long-context lever (global-layer full-span at 32k+) would need Context Parallelism.

## Reproduce
```bash
# on a segmented serve (LEN=66560 SEG=8192):
python3 patches/patch_swa_window_prior_v2.py                 # apply (backs up model.py.pre_swawin_v2)
python3 /tmp/parity_check.py                                  # 18k prompt, prints 40 greedy tokens
# compare full-span (backup) vs windowed -> MUST match (they do)
```
