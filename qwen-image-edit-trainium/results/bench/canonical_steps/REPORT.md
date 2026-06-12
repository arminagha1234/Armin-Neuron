# fal Qwen-Image-Edit — Trainium2 benchmark report

_Generated 2026-06-12T02:40:38.363001_

- Machine: **trn2.48xlarge**
- On-demand price: **$21.5/hr**
- Workers per box (data-parallel ceiling): **4**
- Tool: `customers/fal/path_c/serve/bench_full.py`


## Canonical workload (512×512, 28 steps, 1 image)

- Cold (1st request): 168.05s
- Warm n: 5
- Warm mean: 93.97s
- Warm p50: 93.83s
- Warm p95: 95.05s
- Warm p99: 95.19s
- Warm stdev: 850ms

### Throughput & cost (p99 latency)

- Single worker: 37.8 img/hr → **$0.5685 / image**
- Box (4 workers data-parallel): 151.3 img/hr → **$0.1421 / image** (extrapolated)


## Step count sweep

| Step count | Cold | Warm mean | Warm p99 | $/image (1 worker) | $/image (4× DP) |
|---|---:|---:|---:|---:|---:|
| 4 | 76.03s | 76.57s | 76.74s | $0.4583 | $0.1146 |
| 8 | 79.83s | 79.07s | 79.55s | $0.4751 | $0.1188 |
| 16 | 84.71s | 84.99s | 85.32s | $0.5096 | $0.1274 |
| 28 | 94.10s | 93.65s | 93.92s | $0.5609 | $0.1402 |


## Per-core memory (before bench)

_unavailable: []_

## Per-core memory (after bench)

_unavailable: []_

