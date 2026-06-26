# Raw validated results — 2026-06-26 (trn2.48xlarge, us-east-2)

Container: vLLM-Neuron beta v5 (sdk2.30, vLLM 0.19.0, neuronx-cc 2.25.3371.0),
driver 2.x.9291.0 (matched). Model: gemma-4-31b-it, TP=32, on-device greedy.
Serve: `bash launch.sh 36864 4096 4096 32 1` (max-model-len 36864, chunked seg=4096, mns=1).
Fresh compile, I-485 = 0 throughout.

## Correctness (test_serving.py)
```
[PASS] health endpoint 200
[PASS] model listed
[PASS] short factual generation  -> ' Paris.\n\nThe capital of France is'
[PASS] simple arithmetic  -> ' 4.'
[PASS] needle@5%  (ctx~27028 tok) -> 'BANANA-7731' (9.1s)
[PASS] needle@50% (ctx~27028 tok) -> 'BANANA-7731.' (8.8s)
[PASS] needle@95% (ctx~27028 tok) -> 'BANANA-7731.' (6.3s)
==== 7/7 checks passed ====
```

## TTFT sweep — SWA windowed gather (bench_ttft.py)
```
{"in_target": 1024,  "prompt_tokens": 931,   "ttft_median_s": 0.832, "ttft_min_s": 0.832}
{"in_target": 2048,  "prompt_tokens": 1849,  "ttft_median_s": 0.834, "ttft_min_s": 0.833}
{"in_target": 4096,  "prompt_tokens": 3694,  "ttft_median_s": 0.840, "ttft_min_s": 0.840}
{"in_target": 8192,  "prompt_tokens": 7384,  "ttft_median_s": 0.850, "ttft_min_s": 0.850}
{"in_target": 16384, "prompt_tokens": 14755, "ttft_median_s": 0.873, "ttft_min_s": 0.872}
{"in_target": 32000, "prompt_tokens": 28813, "ttft_median_s": 0.918, "ttft_min_s": 0.918}
```

## TTFT sweep — BEFORE windowing (full-span torch gather, baseline)
```
{"in_target": 1024,  "prompt_tokens": 931,   "ttft_median_s": 1.638}
{"in_target": 4096,  "prompt_tokens": 3694,  "ttft_median_s": 1.646}
{"in_target": 16384, "prompt_tokens": 14755, "ttft_median_s": 1.687}
{"in_target": 32000, "prompt_tokens": 28813, "ttft_median_s": 1.744}
```
=> SWA windowed gather: ~1.9× TTFT improvement, pure-PyTorch (no I-485 risk), 7/7 correctness held.

## Decode throughput
~2.9 tok/s batch-1 (engine log, head_dim>128 decode path). 32K-in/500-out ≈ 0.92s TTFT + ~170s decode.

## NEFF caches tarred (on /scratch)
- cache_4k_working.tar.gz (42M) — canonical 4K
- cache_32k_working.tar.gz (120M) — 32K full-gather
- cache_32k_swa_windowed.tar.gz (149M) — 32K + SWA windowing (this result)
