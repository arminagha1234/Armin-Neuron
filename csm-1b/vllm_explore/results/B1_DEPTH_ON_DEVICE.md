# B1 — Depth decoder on device: diagnosis (the OOB hypothesis was wrong; reframing Tier B)

The depth decoder (31 serial codebook steps/frame, ~156ms on CPU) is CSM's TTFA floor
(A1 confirmed prefill/prefix-caching can't touch it). B1 tried to move it onto the
NeuronCore. Diagnostic harness: `src/b1_depth_on_device.py --mode diagnose`.

## Finding 1 — the `NRT_EXEC_OOB` index-range hypothesis is FALSE
Instrumented the two data-dependent gather sites in `modeling_csm.py`:
- `CsmDepthDecoderModel.forward`: `embed_tokens(input_ids + clamp(cache_position-1)*vocab_size)`
- `CsmCodebooksHead.forward`: `self.weight[cache_position - 1]`

CPU reference run, all 31 steps of one frame:

```
[embed] cache_pos=[0,1] index range=[0,420]    table=65632  ok
[embed] cache_pos=[2]   index range=[3840,3840] table=65632  ok
...
[embed] cache_pos=[31]  index range=[63575,63575] table=65632  ok
```

Every index is in-bounds (max 63,575 < 65,632 across all steps). **The depth decoder's
dynamic gathers are mathematically valid** — the earlier `NRT_EXEC_OOB` was not an
index-out-of-range bug in this path.

## Finding 2 — per-step forward-wrapper offload is the WRONG SHAPE
Offloading `depth_decoder` with the same forward-wrapper that works for the backbone
fails immediately with:

```
RuntimeError: Expected XLA tensor. Got: CPUBFloat16Type
```

The depth decoder's per-step KV cache (StaticCache) and the `backbone_last_hidden_state`
it consumes cross the CPU↔device boundary; the wrapper that marshals a single clean
forward (backbone) doesn't compose with generate's depth-specific cache, whose tensors
are allocated CPU-side and fed into a device forward.

### The deeper reason this approach can't win (even if marshalling were fixed)
The backbone offload wins because it is **one** forward per frame — a single host↔device
round-trip. The depth loop is **31 tiny sequential forwards** per frame. Wrapping each
step means **31 host↔device round-trips + 31 `mark_step` syncs per frame**. That fixed
sync/transfer latency (tens of µs–ms each, ×31, ×every frame) would plausibly **exceed**
the 156ms of CPU compute it's trying to replace. Per-step offload trades compute for
sync overhead — the wrong trade.

## Reframed Tier B: fuse the loop, don't offload the steps
The depth loop must execute as **a single device graph per frame** (one round-trip), not
31 wrapped steps. Concrete paths, in order of effort/payoff:

- **B1' (revised) — trace the whole 31-step depth loop as one callable.** Build a
  standalone module that takes `backbone_last_hidden_state` and runs all 31 codebook
  steps internally (fixed shapes, on-device cache), compiled once via
  `torch_neuronx.trace` / a single `mark_step` boundary. One round-trip/frame. This is
  the cleanest first win and de-risks B2.
- **B3 — parallel / speculative codebook decoding.** The 31 steps are serial only because
  each codebook conditions on the previous. Investigate decoding codebooks in parallel
  (or in groups) with a correction pass — amortizes the serial floor. Highest payoff,
  most research risk.
- **B2 — NKI TKG megakernel** (`attention_block_tkg`) for the fused depth (and backbone)
  step — collapses per-op dispatch inside the graph once B1' gives a single-graph target.

The static-index work (replacing `self.weight[cache_position-1]` with a python-int static
slice, since the loop visits codebooks in order) is still needed to make the fused graph
shape-stable — but it's a means to the fused graph, not a fix on its own.

## Repro
```bash
CSM_MODEL=<csm_1b path> python src/b1_depth_on_device.py --mode diagnose
```
