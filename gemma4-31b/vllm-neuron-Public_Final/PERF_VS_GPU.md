# Gemma4-31B TTFT — Trainium2 vs H100 (measured, same methodology)

Both cold, random prompts, no-APC, bf16, conc=1. H100 = 2×H100 80GB, TP2 (comparable-scale GPU config).

| input | Trn2 TP32 (public) | H100 2×80GB TP2 | notes |
|---|---:|---:|---|
| 4k  | `<FILL>` | 0.240 | `<FILL>` |
| 8k  | `<FILL>` | 0.461 | |
| 16k | `<FILL>` | 1.008 | |
| 32k | `<FILL>` | 2.377 | |
| 64k | `<FILL>` | 5.778 | |

H100 concurrency + TP2×DP4 (throughput/density): see charts. `<FILL honest read: where Trn2 wins/loses>`

> Honest note: on the current public-image fp32/bf16-fallback prefill path, Trn2 TTFT vs H100 is `<FILL>`.
> The d-tiled flash-prefill kernel (if it compiles on cc-2.26) is expected to close ≤16k; not yet confirmed.
