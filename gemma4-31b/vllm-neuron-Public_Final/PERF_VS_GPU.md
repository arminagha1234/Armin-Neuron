# Gemma4-31B TTFT — Trainium2 vs H100 (measured, same methodology)

Both cold, unique random prompts, no-APC, bf16, conc=1. Trn2 = trn2.48xlarge on the public image + NKI
prefill kernel (**median-verified**). H100 = 2×H100 80GB, TP2 (a comparable *quarter-box* GPU config,
prior-session medians).

| input (actual tokens) | Trn2 TP32 (public) | H100 2×80GB TP2 | read |
|---|---:|---:|---|
| 4k (~3.7–4.2k) | **0.239** | 0.240 | **tie** — Trn2 matches H100 |
| 8k (~8.1k) | **0.422** | 0.461 | **Trn2 faster** (−8%) |
| 16k (~15.1k) | **0.806** | 1.008 | **Trn2 faster** (−20%) |
| 32k (~31k) | 9.98 | 2.377 | H100 faster (Trn2 segmented path) |
| 64k (~63k) | 44.79 | 5.778 | H100 faster (Trn2 segmented path) |

## Honest read
- **≤16k: Trn2 is competitive-to-better than H100** — ties at 4k and is *faster* at 8k/16k, on the **public**
  image with the NKI prefill kernel. This is the sweet spot for chat/RAG-style prompts.
- **32k/64k: H100 wins clearly** — Trn2 falls to the segmented prefill path above 16k (see `ROADMAP.md`),
  while H100 scales smoothly. If long-context latency is critical near-term, that workload favors GPU today.

## Fair-comparison caveats
- H100 here is a **quarter-box TP2** (2 of 8 GPUs). A full-box GPU config (TP8) would post lower H100
  latencies — so this is "Trn2 full box vs H100 quarter-box," not "box vs box." Treat ≤16k as "Trn2 is in the
  same class as a comparable-scale H100 slice," not "beats a full H100 node."
- Trn2 numbers are cold / no-APC / random (worst case). The Trn2 throughput/$ story (TP16/TP8 density,
  multiple replicas/box) is separate — see the concurrency charts in the README.
- FP8 (excluded here) would further help both long-context and decode throughput on Trn2.
