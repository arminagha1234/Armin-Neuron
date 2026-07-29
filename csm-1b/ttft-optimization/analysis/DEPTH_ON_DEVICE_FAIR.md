# CSM-1B depth decoder on-device — the FAIR test (measured 2026-07-29)

**Bottom line: YES. The fair config beats CPU decisively — measured, not projected.**
Run the whole 31-step depth loop as ONE compiled graph with weights resident:

- **bf16, full 31 codebooks: 17.9 ms/frame = 7.6× faster than the 137 ms CPU baseline**,
  and only **1.28× above the ~14 ms HBM floor** (vs 12–80× for every prior device run).
- **fp32, full 31 codebooks: 59.1 ms = 2.3× CPU, argmax bit-EXACT** vs the CPU fp32 serial
  reference AND vs stock HuggingFace (32/32). This proves the graph math is correct.
- The prior "CPU wins" verdict was an artifact of **per-step eager dispatch/host-sync**, now
  quantified: the *identical hand-rolled math* run eager = **612 ms** (0.22× CPU, 43.7× above
  floor); compiled as one graph = **17.9 ms**. **~34× is pure dispatch overhead, not compute.**
  The device hardware was never the loser.

All numbers: trn2.48xlarge, 1 NeuronCore (`NEURON_RT_VISIBLE_CORES=0`), torch 2.11 native
torch-neuronx 2.11.3, transformers 5.13, `torch.compile(backend="neuron", dynamic=False)`.
Warm = best of 20 iters, single host boundary (`.cpu()` on the final `[B,K]` tensor) per call.

## Results table

| config | depth ms/frame | vs 137 ms CPU | vs ~14 ms floor | argmax vs fp32 oracle |
|---|---:|---:|---:|:--|
| CPU int8 (shipped baseline) | 137 | 1.0× | 6–11× above | (reference) |
| device eager, same math, bf16 | 612.4 | **0.22× (loses)** | 43.7× above | drifts (bf16) |
| **device compiled, bf16, K=31** | **17.9** | **7.6×** | **1.28× above** | drifts after cb≈10 (bf16 AR rounding) |
| device compiled, bf16 + int8 wt-only, K=31 | 18.9 | 7.2× | 1.35× above | drifts (bf16) |
| **device compiled, fp32, K=31** | **59.1** | **2.3×** | 4.2× above | **EXACT 32/32** |
| device compiled, bf16, K=16 (partial) | 7.97 | 17.2× | **0.57× (below floor)** | drifts after cb≈10 |
| device compiled, fp32, K=16 (partial) | 29.9 | 4.6× | 2.1× above | **EXACT 16/16** |

Prior device numbers for context (from earlier measured runs): eager per-step 1106 ms
(~80× floor), torch_xla fused bf16 166 ms (~12× floor). The fair compiled-resident bf16
path at **17.9 ms** is **9–62× faster than those** and sits at the floor.

## Why the fair test required a hand-rolled loop (the graph-break finding)

**Ladder step 1a — compile the STOCK HF depth forward loop (`fair_depth_neuronx.py`): FAILED
to fuse.** `dynamo.explain` = **graph_count=4, graph_break_count=3**. Every step breaks at:

```
Data dependent operator: aten._local_scalar_dense.default   (a .item() call)
```

Root cause is in `transformers/models/csm/modeling_csm.py::CsmDepthDecoderModel.forward`:
`past_seen_tokens = past_key_values.get_seq_length()` then
`position_ids = torch.arange(past_seen_tokens, past_seen_tokens + inputs_seq_length)` — the
StaticCache seq length is a device scalar, so `arange` forces a `.item()` host sync + a
`create_causal_mask` build **every step**. Result: 4 subgraphs, per-step host sync, and
recompiles → **138 ms warm, 0.99× CPU** (12857 ms max-iter from recompilation). This is the
exact "unfair" per-step-sync wall, just with compiled kernels. Setting
`torch._dynamo.config.capture_scalar_outputs=True` does not rescue it — it pushes the failure
deeper into `PendingUnbackedSymbolNotFound {u0,u1}` (unbacked symints from the same `arange`),
which the neuron backend cannot lower.

**Ladder step 1b — hand-rolled clean loop (`fair_depth_handroll.py`): FUSES to ONE graph.**
Reuse the real weights but replace the HF forward's cache/mask/position plumbing with:
- **python-int positions** (compile-time constants) so the K-step loop fully **unrolls**;
- a **plain-list KV cache** (`torch.cat` per layer), no `Cache` object, no `.item()`;
- no `create_causal_mask` (single-token decode against a growing K/V is inherently causal);
- argmax on-device, **exactly one** `.cpu()` at the very end.

`dynamo.explain` = **graph_count=1, graph_break_count=0**, op_count 6580 → one NEFF of
~12.6k nodes (fp32) / 13.7k (bf16). Weights are read once into HBM and reused across all 31
serial steps. **This is THE fair config, and it wins.**

## Correctness

The hand-rolled fp32 loop is **bit-exact** vs both the CPU fp32 serial oracle (32/32) and
the shipped stock HF depth decoder (32/32, after aligning stock's leading position-0 token).
So the single-graph rewrite is a faithful reimplementation, not an approximation.

bf16 matches for the first ~10 codebooks then diverges — this is ordinary bf16 rounding
compounding through an autoregressive argmax chain, **not** a kernel bug (the fp32 run on the
identical graph is exact). For deployment this is the same accuracy posture as the existing
bf16 backbone; if bit-exactness on all 31 codebooks is required, fp32 at 59 ms still beats
CPU by 2.3×. Practically, partial-depth K=16 (the shipped TTFT strategy) only needs the first
16 codebooks and the perceptual tolerance there is high.

## int8 weight-only quant (ladder step 2): no win on this path — honest negative

Per-output-channel symmetric int8 on q/k/v/o/gate/up/down (dequant in-graph) gave **18.9 ms
vs 17.9 ms bf16 — no speedup.** Reason: after `torch.compile`, the in-graph
`int8→bf16 dequant` is fused/constant-folded and the compiled GEMV at batch=seq=1 is no
longer bandwidth-starved — bf16 already sits at just 1.28× above the HBM floor, so there is
almost no bandwidth headroom left for int8 to recover. int8's premise (halve 230→115 MB of
per-step weight reads) is real for an *un-compiled per-step* path, but the single-graph
compile already removed the overhead that made bandwidth dominant. **Do not ship int8 for
this path; it adds quant error for ~0 gain.** (It would matter again only for a batched or
much larger depth model where the GEMV re-saturates HBM.)

## Ladder step 3 (hand NKI depth-step kernel): NOT NEEDED

The task said to attempt NKI only if compile left >2× above floor. The compiled bf16 path is
**1.28× above floor** (K=16 is *below* the nominal floor at 0.57×). There is <30% headroom to
the roofline; a hand kernel would chase single-digit-percent gains at large cost. The compiled
single-graph path already achieves the goal. NKI is explicitly deprioritized by the measured
result.

## Scripts (in `src/`)

- **`fair_depth_handroll.py`** — THE fair test. Hand-rolled single-graph depth loop; the
  winning path. Flags: `--mode {eager,compile}`, `--dtype {bf16,fp32}`, `--quant {none,int8}`,
  `--k K`, `--iters N`. Prints `dynamo.explain` graph/break counts, neuron compile metrics,
  warm ms/frame, and argmax match vs the CPU fp32 oracle.
- **`fair_depth_neuronx.py`** — the step-1a attempt that compiles the *stock* HF loop and
  surfaces the 3 graph breaks (`aten._local_scalar_dense`) with `dynamo.explain`. Kept as the
  documented evidence of *why* the stock path cannot fuse.

### Reproduce (one device process at a time)

```bash
scp fair_depth_handroll.py ubuntu@<box>:/home/ubuntu/
# winning path, full frame, bf16:
docker exec -e NEURON_RT_VISIBLE_CORES=0 mochi \
  python3 /host/fair_depth_handroll.py --mode compile --dtype bf16 --k 32 --iters 20
# correctness proof, fp32 (argmax EXACT 32/32):
docker exec -e NEURON_RT_VISIBLE_CORES=0 mochi \
  python3 /host/fair_depth_handroll.py --mode compile --dtype fp32 --k 32 --iters 20
# overhead proof, same math eager (612 ms):
docker exec -e NEURON_RT_VISIBLE_CORES=0 mochi \
  python3 /host/fair_depth_handroll.py --mode eager --dtype bf16 --k 32 --iters 5
```
Note: NRT defaults to requesting all 64 logical cores; scope to one with
`NEURON_RT_VISIBLE_CORES=0` or `nrt_init` fails with "Logical Neuron Core(s) not available".
First compile takes ~5 min (300 s NEFF compile) — cache it for warm runs.

## What this changes for CSM TTFT

The depth decoder — 65% of TTFT and long assumed a CPU job — **belongs on the NeuronCore**
when compiled as one resident graph. Full-31 depth drops **137 ms (CPU) → 17.9 ms (device
bf16), 7.6×**; partial K=16 → **8.0 ms**. Folding the device depth path into the TTFT model
(backbone ~38 + depth 8–18 + codec ~23 + host ~60) puts warm TTFT in the **~130–140 ms** range
with the full 31-codebook frame on-device, or lower with partial-depth — and removes the CPU
depth decoder as the critical-path bottleneck entirely. Next step: wire
`fair_depth_handroll`'s compiled loop into `generate_speech_fast.py` in place of the CPU
`_depth_decode`, keeping the fp32 option for bit-exact frames.
