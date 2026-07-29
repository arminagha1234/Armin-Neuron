# CSM-1B optimization — pre-PR code review findings

Two independent static code reviews of the shipped scripts (generator+depth loop, and the
benchmark harnesses). Findings ranked; the load-bearing math was verified correct, and one
real correctness bug was found and FIXED.

## Verified CORRECT (not bugs — so reviewers don't re-chase)
- Depth codebook loop indexing, the `(k-1)*vocab` embed offset, `head[k-1]` indexing (no off-by-one).
- GQA head expand/reshape matches HF `repeat_kv`; RoPE is standard Llama; positions in-bounds.
- Manual-loop KV-cache growth/positions + causal mask correct; the one-hot functional write.
- **torch.compile "compile-once" claim holds** — no `.item()`, python-int positions passed as
  fixed-shape tensors, fixed KV buffers; genuinely single-graph.
- int8 dequant path is correct symmetric per-channel quant (only used with `--quant int8`).
- Depth head+qk fp32 fix is present in the shipped path and matches the 31/32-validated config.
- prefill_verify A/B methodology is sound (fair, correct `.cpu()` sync, compile-time separated).
- fair_depth_exact correctness gate is a real argmax-vs-fp32 check.

## FIXED before PR
- **[HIGH, correctness] Backbone step ran bf16 QK + bf16 codebook-0 argmax** while the depth
  decoder and the frame-0 prefill used the fp32 head/QK fix — an asymmetry that could flip
  codebook-0 (which seeds depth AND the next frame's backbone input) → silent autoregressive
  drift. **FIXED**: `manual_decode_loop.py` backbone step now does fp32 QK scores + fp32
  `lm_head` argmax (matching the depth fix). **Re-verified on-device: codes-vs-stock
  divergences dropped 73/1536 → 37/1536 (~halved)** at the same ~37.7 ms/frame (fp32 on those
  two tiny ops is free). Confirms the bug was real and the fix works.

## Corrected in the PR writeup (not code)
- **[HIGH] "Measured TTFT ~126ms" was a modeled sum** (hardcoded backbone 38 + host 60, best-of
  timing). PR now separates MEASURED components from the MODELED composite; open item: run one
  real end-to-end p50.
- **[HIGH] "Linear scaling / N streams"** unsubstantiated (summed independent per-worker rates,
  no concurrency barrier, times a prefill not a decode). PR reworded; open item: barriered
  concurrent-decode benchmark.
- **[MED] Cold-start cache**: `NEURON_COMPILE_CACHE_URL` may be the wrong knob; open item to try
  `NEURONX_CACHE`/`--cache_dir`. Meanwhile use a resident server.

## Documented as benchmark-only limitations (productionization TODOs, not blockers)
- **[HIGH] No EOS/early-stop** — fixed `--frames` emits trailing silence past the utterance;
  add EOS detection for production.
- **[MED] B=1 only** — codec reshape + validation assume batch 1; add `assert B==1` or fix codec.
- **[MED] Short-prompt/cache-kwarg fragility** — capture spies assume kwargs present & prompt
  len>1; harden for production.
- **[LOW] code<->codec clamp mismatch** (codec clamps 0..2047, backbone embed doesn't); doc why.
- **[LOW] "fp32 oracle" is bf16-weights upcast** — relabel as kernel-math check; the
  hand-rolled-vs-stock-HF check is the true correctness anchor.
- **[LOW] hygiene**: hardcoded `/host` paths + `set_num_threads(24)` → make CLI args /
  `os.cpu_count()`; remove dead code (`MAX=0`, `frames0_seed`, redundant `.float()`).

## Net
Load-bearing math is correct; one real correctness bug (backbone bf16 precision asymmetry) was
found and fixed + re-verified; the over-claimed numbers (composite TTFT, multicore streams) are
corrected to honest measured-vs-modeled framing. Remaining items are benchmark-script hygiene
and production-hardening, listed as TODOs — appropriate for a first customer PR of
research-grade optimization scripts.
