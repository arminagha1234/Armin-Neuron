# gemma-4-31B: the NKI prefill kernel is SLOWER than the torch fallback at ~3.5k tokens

Short version: 31B prefill runs a score-materializing PyTorch SDPA fallback
rather than the custom NKI kernel, because `GEMMA4_V2_PREFILL` defaults to off.
That looks like an obvious win to flip. **It is not — flipping it costs 33%
throughput.** Measured A/B below. Leave it off at Indeed's shape.

## Measured A/B

One trn2.48xlarge, TP16 (= 4 of 16 chips, `logical-neuroncore-config: 2`), bf16,
`max_model_len=4096`, `max_num_batched_tokens=4096` (single-shot prefill, no
segmentation), `max_num_seqs=32`, prefix caching off, greedy. Prompt 3461 tokens
/ 50 output. Servers benchmarked **one at a time** (idle co-resident servers are
known to perturb neighbours), identical flags, only the env var differs.

| | `GEMMA4_V2_PREFILL=0` (torch SDPA) | `=1` (NKI d-tiled kernel) |
|---|---|---|
| prefill-only p50 | **0.364 s -> 9,509 tok/s** | 0.613 s -> 5,643 tok/s |
| conc 4 / 8 / 16 / 32 (RPS) | 0.81 / 1.26 / 1.75 / **1.91** | 0.67 / 0.96 / 1.22 / **1.28** |
| peak, 4 chips | **1.91 RPS** | 1.28 RPS |
| per box (4 replicas) | **7.66 RPS** | 5.14 RPS |
| MFU | **16.0%** | 10.7% |

Zero errors either arm. **V2 is 1.68x slower on prefill and delivers 0.67x the
throughput.**

### The control reproduces across nodes

The `v2off` arm was measured on two different trn2.48xlarge instances a day
apart: 9,449 tok/s / 1.91 RPS peak, then 9,509 tok/s / 1.91 RPS peak. Agreement
within 0.6%, identical peak. The 31B baseline is stable.

### V2 demonstrably engaged

Not an inert flag. The compiled prefill graph changes:

| variant | prefill NEFF | decode NEFF |
|---|---|---|
| `v2off` | `graph_27e203e7...` 12.7 MB | `graph_70d38651...` 3.56 MB |
| `v2on` | `graph_c9ea9123...` **51.5 MB** | `graph_70d38651...` 3.56 MB |

4x larger prefill graph (the d-tiled kernel unrolls), identical decode graph
(V2 is prefill-only). Both arms produced coherent, character-identical output at
ptok=3461, so this is a pure performance regression, not a correctness break.

## Why the flag exists and why it is off

`head_dim` exceeds what the NKI attention kernels support, on both attention
variants: SWA layers are `head_dim=256` (16 KV heads, `sliding_window=1024`),
global layers are `head_dim=512` (4 KV heads). `NF.flash_attention` and the
decode megakernel cap at `MAX_HEAD_DIM=128`. `model.py:23-27` states it plainly:
both variants exceed the limit, so "the functional layer fallbacks (PyTorch
attention) are used automatically."

`gemma4_flash_prefill_v2.py` was written to close that gap — d-tiled, O(tile)
memory, causal + sliding window inside the kernel, "Replaces the
score-materializing SDPA". The dispatch at `model.py:662`:

```python
if _V2_PREFILL is not None and (_v2_can_run is None or _v2_can_run(q)):
    attn_output = _V2_PREFILL[2](...)          # d-tiled flash NKI
else:
    # Fallback: expand KV for GQA + torch SDPA (materializes scores).
    attn_output = self._manual_sdpa(q, k, v, attn_mask)
```

`_v2_can_run` is *not* a head-dim check — it is
`vllm_neuron.nki.nki_hop.can_run_kernel`, which only tests
`VLLM_NEURON_DISABLE_NKI_KERNELS`, CPU mode, and `device != "cpu"`. On device it
returns `True`. The only gate is the env var:

```python
_USE_V2_PREFILL = _os.environ.get("GEMMA4_V2_PREFILL", "0") == "1"   # model.py:51
```

The source contradicts itself about the default. `model.py:45` says "Gated by
GEMMA4_V2_PREFILL (default on)" and reports "d512 cosine 1.000000"; `model.py:49`
says "OFF by default"; the code says off. Our coherence check confirms the
numerics comment — V2 is *correct*. The measurement above explains why the
default was nonetheless flipped to off.

Plausible mechanism for the regression at this length: at T=3461 the SDPA score
matrix is still small enough to be cheap, while the d-tiled kernel pays fixed
per-tile overhead across `head_dim` 256/512 tiles and 60 layers. The kernel's
asymptotic advantage (O(tile) instead of O(T^2) memory) should only pay off at
longer contexts. Untested here — see below.

## Two claims in the repo that this does not support

1. **`verified_kernel_ab.json` is not an A/B.** Its `kernel_on` block has real
   numbers (TP32: 3719 tok -> 0.237 s, 8123 -> 0.424 s, 13950 -> 0.811 s) but
   `kernel_off` is `null`, marked "not captured in this run — kernel-OFF (torch
   fallback) baseline pending". Nothing in `launch_serve_public.sh` sets
   `GEMMA4_V2_PREFILL`, and the file's own `serve_config` string does not mention
   it, so the run labelled `kernel_on` was most likely *also* the torch path.
2. **The README's "NKI prefill kernel cuts <=16k TTFT by up to 40%"** is not
   reproducible at 3.5k tokens. We measure the kernel 68% *slower*. The claim may
   hold at 8k/14k; it does not hold here.

## `GEMMA4_SWA_SKIP` is a no-op in this package

Setting `GEMMA4_SWA_SKIP=1` on top of `V2_PREFILL=1` produced a **byte-identical
graph hash** (`graph_c9ea9123...` in both), so it changes nothing to compile and
was not benchmarked separately. This package ships the stubbed version: the flag
is read at `gemma4_flash_prefill_v2.py:26` but the K-loop remains full-range. A
real windowed K-range would be worth something — at `sliding_window=1024` and
T=3461 it could skip ~70% of K tiles on SWA layers (5 of every 6 layers) — but it
is not implemented here.

## Correction to an earlier E2B note

An earlier experiment recorded "+V2 prefill: byte-identical output, so V2 didn't
engage" for gemma-4-E2B, and treated that as evidence about V2. It is not
evidence: `GEMMA4_V2_PREFILL` was unset, so `_V2_PREFILL` was `None` and the V2
branch was unreachable. Byte-identical output is the expected result of patching
a dead branch. The E2B V2 prefill test was never actually run.

## Where 31B stands

7.66 RPS/box against a 50 RPS target, at 16.0% MFU — roughly 6.5 boxes. The MFU
gap is real but the prefill kernel is not the way to close it, and it is now
measured rather than assumed. Open leads, in rough order of expected value:

- Implement the real SWA K-range skip (~70% of K tiles are dead at T=3461 on 5/6
  of layers). This attacks the same cost SDPA is paying, without adopting V2.
- Find the V2 crossover length. If V2 wins at 8k/14k, the repo's claim is
  salvageable and long-context configs should flip the flag. Needs `LEN=16384`
  and a fresh NEFF cache per arm.
- FP8: the measured gate on Llama-3.1-8B was **1.36x, not 2x** (see
  `../FP8_FEASIBILITY.md`), which moves 31B from ~6.5 to ~5 boxes, not to ~3.
