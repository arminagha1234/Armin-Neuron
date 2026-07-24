# Gemma4-31B (BASE) — full measured TTFT grid (public image, bf16, no-APC, cold random)

Model: **google/gemma-4-31B** (text-only). Image: public vLLM-Neuron 0.21 (SDK 2.31 / neuronx-cc 2.26).
NKI prefill kernel ON + bf16 fallback. Unique random prompt per request (APC cannot serve → true cold prefill).
conc=1 is median-of-10 (±0.003s, reproduced 3×); conc≥2 warmup+bench. **P99 = tail latency, indicative (modest N/cell), NOT an SLA guarantee.**
Token counts are ACTUAL measured (~4k=3.7-4.2k, ~16k=15.1k, ~32k=31k, ~64k=63k), not round labels.


## TP32 (8 chips / whole box)

**TTFT median (s):**

| input | c1 | c2 | c4 | c8 | c16 | c32 |
|---|---|---|---|---|---|---|
| 4k | 0.239 | 0.535 | 0.586 | 6.378 | 20.816 | 50.018 |
| 8k | 0.422 | 0.631 | 1.621 | 8.238 | 23.319 | 54.431 |
| 16k | 0.806 | 1.212 | 2.408 | 9.229 | 25.416 | 58.588 |
| 32k | 10.664 | 16.003 | 26.661 | — | — | — |
| 64k | 47.981 | 71.97 | 119.785 | — | — | — |


**TTFT P99 (s):**

| input | c1 | c2 | c4 | c8 | c16 | c32 |
|---|---|---|---|---|---|---|
| 4k | 0.239 | 0.645 | 0.916 | 22.939 | 46.562 | 112.937 |
| 8k | 0.422 | 0.828 | 2.782 | 26.216 | 51.447 | 121.258 |
| 16k | 0.806 | 1.595 | 3.694 | 27.71 | 54.781 | 129.44 |
| 32k | 10.664 | 21.299 | 42.56 | — | — | — |
| 64k | 47.981 | 95.874 | 191.496 | — | — | — |


## TP16 (4 chips / 2 replicas)

**TTFT median (s):**

| input | c1 | c2 | c4 | c8 | c16 | c32 |
|---|---|---|---|---|---|---|
| 4k | 0.315 | 0.475 | 0.779 | 2.668 | 6.064 | 14.028 |
| 8k | 1.17 | 1.751 | 2.477 | 4.859 | 11.295 | 24.996 |
| 16k | 1.178 | 1.768 | 2.937 | 6.169 | 13.178 | 27.389 |
| 32k | 16.538 | 24.805 | 41.336 | — | — | — |
| 64k | 69.104 | 103.659 | 172.785 | — | — | — |


**TTFT P99 (s):**

| input | c1 | c2 | c4 | c8 | c16 | c32 |
|---|---|---|---|---|---|---|
| 4k | 0.315 | 0.626 | 1.227 | 6.857 | 12.829 | 30.039 |
| 8k | 1.17 | 2.323 | 4.05 | 10.756 | 22.062 | 51.424 |
| 16k | 1.178 | 2.338 | 4.65 | 13.063 | 26.084 | 55.953 |
| 32k | 16.538 | 33.041 | 66.045 | — | — | — |
| 64k | 69.104 | 138.152 | 276.266 | — | — | — |


## TP8 (2 chips / 4 replicas)

**TTFT median (s):**

| input | c1 | c2 | c4 | c8 | c16 | c32 |
|---|---|---|---|---|---|---|
| 4k | 1.2 | 1.204 | 4.049 | 9.092 | 19.468 | 40.301 |
| 8k | 1.201 | 1.801 | 4.127 | 10.279 | 20.707 | 43.382 |
| 16k | 2.461 | 3.695 | 7.217 | 14.733 | 30.08 | 60.866 |
| 32k | 30.448 | 45.674 | 76.093 | — | — | — |
| 64k | — | — | — | — | — | — |


**TTFT P99 (s):**

| input | c1 | c2 | c4 | c8 | c16 | c32 |
|---|---|---|---|---|---|---|
| 4k | 1.2 | 1.796 | 9.034 | 18.049 | 40.383 | 80.724 |
| 8k | 1.201 | 2.39 | 9.137 | 19.392 | 41.582 | 86.878 |
| 16k | 2.461 | 4.907 | 14.066 | 28.044 | 60.325 | 120.565 |
| 32k | 30.448 | 60.867 | 121.673 | — | — | — |
| 64k | — | — | — | — | — | — |


## H100 ×2 80GB (TP2) — reference sweep (prior session)

Comparable-scale GPU baseline: 2× H100 80GB, TP=2. conc=1 medians match the conc=1 line in the README table
exactly (4k 0.240 / 8k 0.461 / 16k 1.008 / 32k 2.377 / 64k 5.778). This is the H100 line drawn in the
"vs concurrency" charts. **P99 was not captured for H100**, so on the P99 chart the H100 line is this median.

**TTFT median (s):**

| input | c1 | c2 | c4 | c8 | c16 | c32 |
|---|---|---|---|---|---|---|
| 4k | 0.240 | 0.359 | 0.637 | 1.139 | 2.075 | 3.964 |
| 8k | 0.461 | 0.694 | 1.197 | 2.157 | 4.100 | 8.021 |
| 16k | 1.008 | 1.510 | 2.569 | 4.651 | 8.812 | 16.894 |
| 32k | 2.377 | 3.565 | 5.927 | 10.714 | 20.192 | 42.746 |
| 64k | 5.778 | 8.699 | 14.594 | 26.003 | 48.894 | 94.487 |


## Notes
- TP8 fit **up to 32k** with no OOM on public (better than prior beta reports of TP8 OOM ≥16k). 64k@TP8 not run.
- 32k/64k use segmented prefill (SEG=2048) — the 16k→32k jump is the single-shot(NKI)→segmented(torch) path swap, not smooth scaling.
- H100 2×80GB TP2 conc=1 medians (prior session): 4k 0.240 / 8k 0.461 / 16k 1.008 / 32k 2.377 / 64k 5.778 s.
