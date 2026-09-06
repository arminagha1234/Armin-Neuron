# Gemma-4-31B-it on vLLM-Neuron: the decode ceiling is real

31B serves correctly (coherence 3/3, TTFT 0.62 s) but its throughput is bounded
by decode in a way that neither concurrency nor a custom attention kernel fixes.
Recording it so the same two experiments are not repeated.

## Measured: already saturated at concurrency 16

TP=32, `max_num_seqs=64`, ~3500 prompt tokens, chat endpoint, coherence 3/3.

| conc | RPS | avg latency | decode |
|---:|---:|---:|---|
| 1 | 0.102 | 9.78 s | 1.58 tok/s |
| 4 | 0.383 | 10.44 s | |
| 8 | 0.707 | 11.31 s | |
| 16 | 1.224 | 13.07 s | |
| 32 | 1.226 | 17.82 s | |
| 48 | 1.227 | 22.57 s | |
| 64 | 1.498 | 27.75 s | |
| 96 | 1.396 | 37.88 s | |
| 128 | 1.501 | 47.93 s | |

RPS is flat from 16 onward while latency grows linearly — a queue, not headroom.
Raising `max_num_seqs` from 32 to 64 and sweeping concurrency to 128 moved
per-replica RPS from ~1.22 to ~1.50, and the best figure is **lower** than an
earlier measurement capped at concurrency 16 (1.75). Treat 1.5 RPS/replica as the
ceiling for this shape.

At TP=32 a `trn2.48xlarge` holds two replicas, so ~3 RPS/box.

## Per-request decode is host-dispatch-bound, not attention-bound

A d-tiled NKI decode kernel supporting `head_dim` 256/512 already exists in the
serving package (`attention_decode_kernel.py`, flash-style online softmax, GQA
head packing). It is **dead code** — nothing imports it.

That is deliberate. When it was wired in it produced **0 speedup**, and a
separate branch measured it **3x slower** than the TKG megakernel path
(210 vs 625 tok/s at equal concurrency). Four independent write-ups in this repo
reach the same conclusion: per-request decode sits near ~2.9 tok/s because of
host dispatch overhead, so replacing the attention math changes nothing.

Do not write a new head_dim>128 decode kernel expecting a per-request win.

## Where the remaining upside actually is

- The **TKG** decode backend rather than SDPA or the custom NKI kernel. The
  public nkilib TKG kernel rejects `kv_heads > 1` and enforces `d_head <= 128`,
  so this requires the modified `attention_tkg` that adds `n_d_tiles=2` for
  `d_head=256` (see `decode_kernel_dhwanw/` elsewhere in this repo). Reported
  1249 -> 2486 tok/s aggregate combined with `mns=128` and a larger KV pool.
- **MTP / speculative decode** for per-request latency (2-3x), which is the right
  lever when the bound is dispatch rather than compute.

## Native PyTorch is blocked

Every native attempt fails during `init_process_group`, before any weight loads:

```
RuntimeError: Failed to execute the device barrier 2
  torch_neuronx._C._nrt_barrier(0, rank, size)
  in _neuron_runtime_setup
```

Observed at TP=8, 16 and 32. Not memory: 31B bf16 is 62 GB, which is 7.8 GB/core
at TP=8 against 24 GB available, and the failure precedes loading. TP=2 would
avoid the cross-chip barrier but cannot hold the model (31 GB/core).

Note this is model-specific in practice: Qwen3-8B at TP=4 and Qwen3.5-4B at TP=16
both initialise fine in the same image.

## Latency is the thing to tell a customer

TTFT is good (0.62 s). But at 1.58 tok/s per request, 50 output tokens take
~30 s. Adding instances buys throughput, not latency. Any capacity plan built on
RPS alone will understate how slow an individual 31B request is on this stack.

## TKG decode kernel — wired, but does not land on Kaizen

The container ships a raw `nkilib.core.attention.attention_tkg` kernel whose RoPE
asserts only `d_head % 64 == 0`, so unlike the block megakernel (stuck at 128) it
*supports* Gemma-4's head_dim 256/512. A 262-line wrapper (`attention_tkg_raw`,
retargeted from dhwanw's branch by changing 3 import lines) plus a decode
dispatcher were spliced into the model behind `GEMMA4_DECODE_BACKEND=tkg`. The
wiring is correct — patch, import, dispatch, and a clean local dry-run all pass —
but no serving number was obtained, across four attempts:

1. missing import (patcher bug) — fixed.
2. SWA layers: vLLM 0.21's page-size unification doubles SWA `block_size` 16→32,
   which breaks the kernel's `s_prior % 128` alignment. dhwanw's fix lives in
   `neuron_model_runner.py::_compute_swa_num_blocks` (an 8291-line file, 5 call
   sites) — deep framework surgery.
3. global-only + **fp8 KV**: internal NKI compile assertion. The wrapper has no
   fp8 handling (no `k_scale`/`v_scale`/`fp8_packed`/packed-swizzle), and fp8 KV
   is exactly what enabled the batching win, so the two don't compose without
   more work.
4. global-only + **bf16 KV**: compiled *past* the trace stage (the fp8 assertion
   was gone) but was still compiling at the ~31-min pod wall.

Bounded upside anyway: dhwanw measured SWA-TKG at **+5.5%** and fused-mask at
**+4.1%** — small next to the **+67%** batching win (17→10 boxes) that config
tuning already banked. The wrapper and installer are preserved for a longer wall
or an EC2 box, but the honest conclusion is that on the public 0.21 image + Kaizen,
config tuning (KV cache sizing, fp8 KV, right `max_model_len`) is where the 31B
throughput actually came from — not the attention kernel.
