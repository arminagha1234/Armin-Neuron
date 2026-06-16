# TP=2 first-pass — runs but quality + perf both wrong

**Date:** 2026-06-15
**Box:** trn2.48xl, Beta 3 DLC `concourse-release-0461d3b:latest`
**Config:** FLUX.2-klein-4B, 1024×1024, 4 distilled steps, TP=2

## What worked

- ✅ Beta 3 `torch.distributed.init_process_group(backend='neuron')` smoke
  test passes (12.6 s init, all_reduce returns the right sum)
- ✅ FLUX.2 TP=2 runner reaches the end of warm runs without crashing
- ✅ Required env vars discovered:
  - `NEURON_RT_VIRTUAL_CORE_SIZE=2`
  - `NEURON_LOGICAL_NC_CONFIG=2` (must match prebuilt NEFF)
  - `NEURON_SKIP_EFA_AFFINITY=1`

## What did NOT work

| Metric | Single core (baseline) | TP=2 | Verdict |
|---|---:|---:|---|
| Warm wall clock | 5.8 s | **57.5 s** | **10× SLOWER** ❌ |
| Output std (quality gate) | 18.16 | **45.39** | **WRONG output** ❌ |

```
[tp run] === FLUX.2-klein-4B TP=2 SUMMARY ===
  warm avg: 57.50s   min: 57.47s
  quality: std=45.39
  vs Phase A single-rank baseline 6.86s: -50.64s
```

## Likely root causes (to investigate)

1. **`attention_cte` flash kernel** is installed at TP=2 but not at TP=1.
   The single-core 5.8 s number uses default SDPA. The kernel at TP=2
   may be:
   - producing wrong output (std=45 evidence)
   - running unnecessary or duplicated work per step
2. **TP plan correctness** — the sharded constants (24 heads, 5 double +
   20 single blocks) match the model. Patching `attn.heads -> 12` is
   correct math. But the **interaction between `parallelize_module`,
   `apply_tp_fixes`, and the FLUX.2 attention processor** may double-
   shard or miss a fix.
3. **All-reduce overhead at every block** — even with correct sharding,
   the cross-rank communication at every transformer block can dominate
   if (a) the sharded chunks are too small to amortize, or (b) the
   collective is being executed via TCP fallback instead of Neuron
   collectives. The smoke test logged `OFI plugin initNet() failed is
   EFA enabled?` — so collectives may be falling back to a slower path.

## Next steps to debug (not in this session)

1. **Run TP=2 WITHOUT the `attention_cte` install** to isolate whether the
   issue is the kernel or the sharding plan.
2. **Add per-block tensor norm assertions** — compare TP=1 and TP=2
   intermediate tensors at the same input to localize where divergence
   begins.
3. **Bench the all_reduce throughput** on this box with a real-sized
   tensor (e.g. `[1, 24, 4096, 128]` bf16) to see if collectives are on
   the slow fallback path.
4. **Check if `tp_smoke_test.py` was used at TP=2 with a non-trivial
   tensor** — if the smoke test was only validating `[4]` allreduce, the
   collectives at production tensor sizes might still be fundamentally
   slow on this box.

## What we DON'T need to redo

- The Beta 3 distributed setup recipe — that part works
- The TP plan structure (`flux2_tp_plan.py`) — the constants match the
  model
- The runner scaffolding (`run_flux2_tp.py`) — runs end-to-end, just
  produces wrong numbers

## Honest customer-story implication

**Right now we cannot quote a 4 MP TP number.** The single-core 1 MP
number (5.8 s) is solid. Multi-core scaling needs at least 2-3 days of
debugging before we have a credible 4 MP / TP=4 result for the customer.

Two paths from here:

1. **Debug the existing TP scaffold** — keep the meta-init + custom
   kernel approach, fix the bug in attention_cte or the TP plan
   interaction. ETA: 2-3 days.
2. **Restart TP=2 from scratch** with the simplest possible plan (no
   custom kernel, just diffusers default attention with sharded heads),
   prove correctness first, optimize later. ETA: 1-2 days for a
   correct-but-slow baseline, then iterate.

Recommendation: option 2. The 5.8 s single-core number proves diffusers
+ default SDPA is quality-correct. We get a known-correct TP=2 number,
even if slow, and incrementally add the `attention_cte` kernel back with
quality assertions at each step.
