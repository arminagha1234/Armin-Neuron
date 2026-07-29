# CSM-1B cold-start + NEFF cache — measured (pre-PR)

**Method:** compile the backbone (N=1024 prefill) via `torch.compile(backend="neuron")` in
two SEPARATE fresh processes sharing `NEURON_COMPILE_CACHE_URL=/host/neuron_cache_test`.
run1 = cold, run2 should hit the on-disk cache. Harness: `src/coldstart.py`.

## Result — persistent NEFF cache did NOT engage via NEURON_COMPILE_CACHE_URL

| run | compile+first-run |
|---|---:|
| run1 (cold) | 8.5 s |
| run2 (fresh process, same cache dir) | 8.6 s — **no speedup** |

And the cache dir was **empty** afterward; no NEFF cache dir exists at the custom path or the
usual defaults (`/var/tmp/neuron-compile-cache`, `/root/.cache/neuron`, `/tmp/...`).

## Finding (honest, customer-relevant)
**On this torch-neuronx stack, `NEURON_COMPILE_CACHE_URL` does NOT produce a cross-process
persistent NEFF cache for the `torch.compile(backend="neuron")` path.** The ~8.5 s compile is
paid every fresh process. So the cold-start story is:
- **The compile cost is NOT auto-amortized across restarts** by that env var (contradicts the
  assumption in earlier notes / the PAVE-derived `NEURON_COMPILE_CACHE_URL` suggestion — that
  may apply to a different backend/version).
- **The working mitigation is a long-lived server process:** compile once at startup (warmup
  call), keep the process resident, serve many requests — the ~8.5 s is a one-time startup
  cost, not per-request. This is the standard serving pattern and is what PAVEDigitalTwinDiffusion's
  TileEnhancer/service does (workers stay up).
- Finding the correct persistent-NEFF mechanism for this exact torch-neuronx version (so a
  cold restart skips compile) is an **open item** — the `NEURON_COMPILE_CACHE_URL` knob tested
  here is not it. Candidates to try: `NEURONX_CACHE`/`NEURON_CC_FLAGS="--cache_dir=..."`, or
  `torch_neuronx.trace`-based AOT save/load, or the DLC's documented cache path.

## Numbers for the PR
- Backbone compile (N=1024): ~8.5 s cold, per fresh process (small graph; larger N compiles
  longer — 4k took ~50 s in the prefill sweep).
- Warm (in-process, after compile): the measured 18–96 ms/prefill (VERIFIED_PREFILL_TTFT).
- **Recommendation: run CSM as a resident server** (warmup at startup) so compile is one-time;
  do NOT rely on `NEURON_COMPILE_CACHE_URL` for cross-restart caching until the correct knob
  is confirmed.
