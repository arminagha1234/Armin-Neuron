# Long-context roadmap (32k / 64k)

≤16k is fast (single-shot NKI kernel). 32k/64k currently use a **segmented** prefill path that is correct
but slow (32k ~10.7s, 64k ~48s at conc=1, TP32, from the segmented A/B smoke session; the
median-verified headline figures in RESULTS.md are **9.98 s / 44.79 s**). Two levers were investigated to close this:

## 1. Segmented NKI kernel (near-term)
Extend the ≤16k NKI `attention_cte` kernel to the >16k segmented path (replacing the torch fallback).
- **Status:** the masking/scaling math is validated on CPU (cosine ≥ 0.9999), but on the current public
  image (neuronx-cc 2.26) the kernel does not yet engage at serve time — it falls back cleanly to torch
  (correct output, no ≤16k impact, no speedup). See `SEGMENTED_SMOKE_RESULT.md`.
- **Next:** surface the on-device fallback exception, fix the shape/contract mismatch, re-test. Estimated a
  few hours of on-device work once the exception is captured.

## 2. Context Parallelism (CP) — the real long-context fix
Sharding the sequence dimension across chips would parallelize long-context prefill.
- **Status:** not available for this model on the public image. CP for inference ("DCP") is wired only for a
  different model family and requires disaggregated serving plus a new attention kernel — a multi-week port,
  not a config flag.
- **Positioning:** the strategic long-context solution; scope it as a dedicated project.

## Today's honest guidance
For ≤16k, Trn2 is fast and competitive with H100. For 32k/64k on the public image, expect the segmented
latencies above; if long-context latency is critical near-term, that workload is better on GPU until one of
the levers above lands. FP8 (excluded here) would also help long-context KV pressure and decode throughput.
