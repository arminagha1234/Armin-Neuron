# CSM-1B prefill TTFT — RIGOROUSLY VERIFIED (compile vs eager crossover)

**Method (customer-grade):** ONE context length per FRESH process (no torch._dynamo cache/
guard contamination), `dynamo.explain` graph-break inspection + `get_dynamo_metrics` NEFF
count on every point, median+std over 20 iters (3 warmup), cosine(compiled, eager) for
correctness. Backbone = CSM Llama-3.2-1B (16 layers, hidden 2048), bf16, `use_cache=False`
pure prefill forward. Box: trn2.48xlarge, single NeuronCore (`NEURON_RT_VISIBLE_CORES=0`).
Harness: `src/prefill_verify.py`.

## Result — the compile-vs-eager crossover is REAL (not a graph-break artifact)

Every point compiled to `graph_count=1, graph_break_count=0, 1 NEFF (1618 nodes)` with tight
variance, so the crossover is a genuine compute-regime effect.

| N (tokens) | eager median (ms) | compiled median (ms) | speedup | cos(comp,eager) | TTFT best (+25 ms frame0) |
|---:|---:|---:|---:|---:|---:|
| 512  | 77.2  | **18.2**  | **4.25×** | 0.99985 | ~43 ms (compiled) |
| 1024 | 97.0  | **36.9**  | **2.63×** | 0.99980 | ~62 ms (compiled) |
| 2048 | 141.5 | **96.2**  | **1.47×** | 0.99982 | ~121 ms (compiled) |
| 3072 | 195.0 | **180.2** | **1.08×** | 0.99998 | ~205 ms (compiled) |
| 4096 | **261.1** | 301.5 | **0.87×** | 1.00010 | ~286 ms (**eager**) |

std ≤1.7 ms eager, ≤0.5 ms compiled — numbers are stable.

## What it means (customer guidance)
- **`torch.compile(backend="neuron")` wins for prompts ≤ ~3k tokens** (up to **4.25×** on
  prefill at 512), advantage decaying smoothly to parity at ~3.3k.
- **Above ~3k, eager is faster** — compiled backbone at 4096 is 0.87× (13% slower).
- **CSM's trained context window is 2048** (`max_position_embeddings`), so **compile wins
  across the entire in-spec range.** 4k+ is RoPE extrapolation (latency-valid, coherence
  unverified) and is where eager wins.
- Recommended policy: enable compile for ≤2k contexts; for >3k use eager, or the TP / flash
  / fp8 levers (below), NOT compile.

## Why compile loses at large N — NOT root-caused (stated honestly)
Plausible: at small N the graph is fixed per-op-launch-overhead bound (compile fuses/
schedules it away → 4.25×); at large N the O(N²) attention compute dominates and the Neuron
compiler's schedule/tiling for the big attention matmuls is apparently worse than eager's
per-op kernels. **This mechanism is UNPROVEN** — an earlier "graph break at 4k" hypothesis
was tested and REFUTED (0 breaks everywhere). Do not present the mechanism as fact; the
crossover *measurement* is solid, the *cause* is open.

## Corrections to earlier reporting (transparency)
1. The first prefill numbers (102–534 ms) were **eager**, mislabeled as "compiled" — the
   harness never wrapped the backbone in torch.compile. Corrected here.
2. The "4k compiled slower because of layout" claim: the **slowdown is real & reproduced**
   (0.87×, std 0.5 ms), but the stated *cause* was speculation and is flagged unproven.

## "+50 frames out"
50 audio frames ≈ 50 × ~36 ms ≈ **1.8 s of streaming AFTER first audio** — does not affect
TTFT (first audio arrives at the TTFT numbers above).
