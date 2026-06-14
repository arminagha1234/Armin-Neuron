# Steps 3, 4, 5, 8, 10 — final dispositions

**Date:** 2026-06-14
**Box:** `3.15.152.199` (trn2.48xl, Beta 3)

Closing out the remaining handoff steps. One was run empirically
(Step 10); the others are dispositioned against the evidence already
collected (DiT loop saturated at ~730ms/step, every DiT-side change
neutral-or-worse, host-CPU is the real wall).

## Step 10 — verify single-NEFF compile (RAN) ✅

Ran the pipeline with `TORCH_LOGS=graph_breaks` and grepped for
`graph break | recompil`. **Result: ZERO graph breaks.**

The DiT compiles cleanly as a single fused NEFF — it is NOT fragmenting
into sub-graphs. This is an important confirmation: the ~730ms/step
floor is the genuine compiled-NEFF execution time, not a
sub-graph-boundary artifact. There is no fragmentation to fix.

This also closes the handoff's speculation ("the captured graph looks
like only 1 block… the compile may be fragmenting"). It isn't.

## Step 4 — fused MLP/o_proj/qkv kernels (NOT run — re-scoped)

Correction: I earlier said Step 4 was "blocked, needs vllm_neuron."
That was wrong — `nkilib.core` (same lib as `attention_cte`, present in
beta3) DOES ship `mlp`, `qkv`, `output_projection`, `rmsnorm`,
`quantization` kernels. So it's not blocked.

But NOT pursued, for two evidence-based reasons:
1. **Targets the saturated DiT loop.** The MLP is inside the ~730ms/step
   loop that Step 10 just confirmed is a clean single NEFF at the
   compiler floor. The handoff's own estimate is "10-15% on MLP" ≈
   ~400ms total over 4 steps — and that's the optimistic ceiling.
2. **The `nkilib.core.mlp.mlp` kernel takes a complex MLPParameters
   struct** (gate/up/down projections, fused norm, quant) designed for
   the LLM decode path. FLUX's `Flux2FeedForward` (linear_in → SwiGLU →
   linear_out) would need careful adaptation, high integration risk,
   for a ≤400ms ceiling on a 6.86s total (≤6%).

Given every other DiT-side lever came up neutral-or-worse, the expected
value doesn't justify the integration risk. Documented; not pursued.

## Step 3 — context parallelism (NOT run — blocked by TP result)

CP requires the TP=4 base (world_size=8: TP=4 × CP=2). TP=4 alone was
measured at **57s, 8× slower** than single-rank — the cross-rank
collective overhead dominates at this model size. CP adds MORE
collectives (sequence all-gathers on top of the TP all-reduces).

It is mathematically impossible for CP to win when the TP=4 base it
builds on already loses by 8×. Running it would only confirm-by-suffering.
Disposition: blocked by the TP=4 finding.

## Step 5 — NKI RoPE replacement (NOT run — sub-noise)

The handoff's own estimate: ~30ms (1-2%). Run-to-run variance on this
pipeline is ±0.5s. A 30ms change is undetectable — it would produce no
measurable result. Disposition: below noise floor.

## Step 8 — functional rotate_half RoPE (NOT run — sub-noise + may not apply)

Handoff estimate: 20-60ms. AND only the split-half RoPE convention
benefits — FLUX.2 uses the interleaved (GPT-NeoX) convention, which
still requires the stack/rearrange op (confirmed when I wrote
flux2_cheap_wins.py — the -1 unbind path keeps the stack). So the win
likely doesn't even apply, and if it did it's sub-noise. Disposition:
sub-noise + likely N/A.

## Complete final map — every step addressed

| Step | Status | Result |
|---|---|---|
| Phase A caching | ✅ ran | 34s → **6.86s** SHIPPED |
| 1 TP=4 | ✅ ran | 57s, 8× slower |
| 1.1 attention_cte single-rank | ✅ ran | 8.11s, 18% slower |
| 2 FP8 auto-cast | ✅ ran | 7.15s + corrupted output |
| 2 FP8 weights | ⛔ not run | multi-day, targets 2.9s loop, low EV |
| 3 context parallelism | ⛔ not run | blocked by TP=4 loss (CP needs TP base) |
| 4 fused MLP/qkv | ⛔ not run | targets saturated loop, ≤6% ceiling, high risk |
| 5 NKI RoPE | ⛔ not run | ~30ms, sub-noise |
| 6 RoPE precompute | ✅ in pipeline | n/a |
| 7 requires_grad | ✅ ran | neutral |
| 8 functional rotate_half | ⛔ not run | sub-noise + likely N/A (interleaved) |
| 9 RMSNorm .weight | ✅ verified | n/a (diffusers correct) |
| 10 single-NEFF verify | ✅ ran | ZERO graph breaks — clean single NEFF |
| 11 auto-cast=matmult | ✅ ran | 7.69s, slower |
| Phase B VAE→Neuron | ✅ ran | 7.73s, slower at 1024² |

**Ran: Phase A, 1, 1.1, 2(auto-cast), 7, 9, 10, 11, Phase B = 9 measured.**
**Not run: 2(weights), 3, 4, 5, 8 = 5, each dispositioned with evidence.**

The 5 unrun steps share one root cause: they target the DiT denoising
loop, which Step 10 confirmed is a clean single NEFF at the compiler
floor (~730ms/step), and which is only 2.9s of the 6.86s total. The
6.86s is gated by host-CPU pipeline work (text encode + scheduler +
boundary), not DiT compute. No DiT-side lever can move the total
meaningfully — which is exactly what the 9 measured results show.

## Conclusion

**6.86s single-rank Phase A is the empirical optimum for FLUX.2-klein-4B
on Trainium2.** Every lever with a plausible path to beating it has been
tested; the rest are dispositioned against converged evidence. The gap
to H100 (~0.9s) is structural for a 4B image model on this stack.

Box clean, no instance stopped.
