# NKI kernels — current status

## decode_hd256 (single-token decode attention, head_dim=256)

**Status:** parity validated, integrated into qwen3.5 model, benchmarked
end-to-end on real Trainium2 hardware. **v1 is correct but ~20% slower
than the eager baseline that neuronx-cc auto-fuses.** Not upstream-ready.

### What's done

- **Math contract:** `armin_nki_kernels/attention/ref_decode_hd256.py`
  Pure-PyTorch reference — split-K + split-V matmul pair, fp32 softmax.
  This is the parity oracle.
- **NKI kernel:** `armin_nki_kernels/attention/decode_hd256.py`
  Fused QK + softmax + AV in one NEFF. Tiles ctx into 128-token chunks,
  PSUM-accumulates the split-K halves, fp32 softmax with two-pass reduce,
  PSUM-accumulates the split-V halves across all chunks.
- **Wrapper:** `armin_nki_kernels/attention/decode_hd256_wrap.py`
  vllm-neuron-style adapter — uses `vllm_neuron.nki.nki_hop.wrap_nki` on
  Neuron, falls back to the PyTorch reference on CPU.
- **Tests:**
  - `tests/test_decode_hd256_parity.py` — pytest sweep (CPU fallback).
  - Direct simulate sweep — 6 shapes from S_ctx=128 to S_ctx=4096:

    | shape          | S_ctx | valid | cosine vs ref |
    |----------------|-------|-------|---------------|
    | smoke          | 128   | 64    | 0.999991      |
    | 4B_short       | 128   | 100   | 0.999988      |
    | 4B_typical     | 512   | 400   | 0.999992      |
    | 27B_typical    | 512   | 400   | 0.999992      |
    | 4B_2K          | 2048  | 1500  | 0.999987      |
    | 4B_chunked     | 4096  | 2048  | 0.999986      |

    All > 0.999 threshold.

- **End-to-end on hardware:** kernel compiles and runs cleanly inside
  vLLM-Neuron via `wrap_nki[2](...)`. Generates coherent factual text.
  See `qwen3.5-4b-trainium/BENCHMARK_NKI_VS_EAGER.md` for the full A/B.

### v1 perf result — SUPERSEDED: v1 is 4.77x FASTER than eager at realistic max_num_seqs

```
trn2.48xlarge, TP=4, MAX_LEN=2048, Qwen3.5-4B (head_dim=256), conc=1:

  EAGER (compiler-fused split-K)   79.6 tok/s decode
  NKI v1 (this kernel)             63.7 tok/s decode
  ratio                            0.80×   (NKI is 20% SLOWER)
```

Identical TTFT (5.44s for both — kernel only runs in decode).
Identical correctness on the probe battery.

**Terminology first, because "eager" is overloaded three ways in this stack and
the word is misleading here.** In the vLLM-Neuron serving path there is *no*
op-by-op eager execution: `neuron_model_runner.py` calls `torch.compile` with a
registered `"vllm_neuron"` backend (line ~1385, backend in
`vllm_neuron/compile/backend.py`), and `NeuronConfig` exposes **no** compile
knob — it is unconditional. Both arms below are
`torch.compile(backend="vllm_neuron")` -> HLO -> neuronx-cc -> NEFF, and the
`QWEN35_NKI_DECODE=0` arm compiles **5 NEFFs** including a 5,075,250-byte decode
graph. The only difference is *what the compiler lowers*: plain PyTorch split-K
ops, or a hand-written NKI kernel injected through `wrap_nki`.

So "eager" in this file means **"the plain-PyTorch source path, which
torch.compile + neuronx-cc then fuse"** — not eager execution. The three senses
in play across the repo:

| "eager" | means |
|---|---|
| here / `QWEN35_NKI_DECODE=0` | PyTorch-source path, still fully torch.compile'd |
| native TorchNeuron | genuine op-by-op `torch.device("neuron")`, no compile |
| `attn_implementation: "eager"` in trainium-optimizer recipes | HuggingFace's attention selector (eager vs sdpa vs flash), orthogonal to compilation |

This matters for how the result reads: the kernel is not beating an interpreter,
it is beating `torch.compile` with full optimisation opportunity. Note the
compiler produced the *smallest* decode graph (5.07 MB vs v1's 7.28 MB and v2's
7.82 MB) and the *slowest* one.

**The original measurement was single-stream at `MAX_NUM_SEQS=1`** (the roadmap
item "Concurrency stress with MAX_NUM_SEQS=8" was never done, and `serve.sh`
defaults to 1). Re-measured at `MAX_NUM_SEQS=16` — the configuration the
published capacity numbers actually use — the ordering **reverses**:

```
trn2.48xlarge, TP=4, MAX_LEN=2048, blocks=200, MNS=16, 1811-tok prompt,
sustained windows, one server at a time:

  torch.compile'd PyTorch split-K  295.01 ms/token  0.165 RPS/chip  2.6 RPS/box
    (QWEN35_NKI_DECODE=0, decode NEFF 5,075,250 B)
  NKI v1                            61.85 ms/token  0.514 RPS/chip  8.2 RPS/box
    (QWEN35_NKI_DECODE=1, decode NEFF 7,281,631 B)
  ratio                              4.77x FASTER   3.1x throughput
```

Decode ms/token was measured at both 50 and 200 output tokens and agreed to
0.06 ms, so this is not noise. The reason for the reversal is that the compiled
PyTorch split-K materialises `[MNS x S_ctx]` score tensors every decode step,
so its cost scales with the `max_num_seqs` bucket; the kernel's does not. At
MNS=1 that penalty is invisible.

**Consequences:** v1 should be default-ON, not "do not upstream". Anyone running
Qwen3.5 with `QWEN35_NKI_DECODE=0` is losing ~3.1x decode throughput. And the v2
premise below ("v2 needs the flash pattern to beat eager") was built on a
baseline that was wrong by 4.77x.

### Why v1 is slower (hypothesized)

`aws-neuron/nki-library/core/attention/attention_tkg.py` uses several
techniques v1 of this kernel does not:

1. **Online flash-attention softmax** — running max + running sum
   maintained across chunks, no two-pass reduce. v1 does a two-pass
   reduce (sum across F dim, then transpose+reduce across P dim) which
   is the wrong pattern for streaming attention.

2. **No explicit K transpose** — v1 has an extra `nc_transpose` per
   chunk to flip K from (128 ctx, 128 dim) to (128 dim, 128 ctx) so the
   `nc_matmul` partition contraction is on the right axis. The reference
   `attention_tkg` uses a different stationary/moving layout that avoids
   this transpose.

3. **Aggressive inter-iteration fusion** — neuronx-cc compiling the
   eager Python likely fuses the chunked QK+softmax+AV passes into one
   pipelined inner loop without the explicit per-chunk SBUF
   materialization v1 forces.

### v2 result — built, parity-clean, 2.7% faster than v1

`attention/decode_hd256_v2.py`. Single-pass restructure. Gated into the qwen3_5
package by `QWEN35_DECODE_V2=1` (see `deploy_v2_into_qwen35.py`); default off.

**What v1 was actually doing** — read off the code rather than guessed. v1 makes
**three separate loops over S_ctx** plus a global reduction chain:

| v1 stage | per chunk |
|---|---|
| loop 1 (QK) | 2 DMAs (`k_lo`, `k_hi` as separate `[128,128]` halves), 2 `nc_transpose` + 2 PSUM->SBUF copies, 2 `nc_matmul`, 1 `tensor_scalar` |
| loop 2 (mask) | 1 `memset` of a **full `[128,128]`** tile, 1 DMA, 1 **`nc_transpose` of a `[128,128]` tile purely to turn a 128-element row into a column**, 1 `tensor_tensor` |
| softmax | exp; `tensor_reduce` axis=1; then memset + copy + `nc_transpose` + copy + reduce **again** just to finish a partition-dim sum; reciprocal; `broadcast_to`; multiply; bf16 cast over the whole score tile |
| loop 3 (AV) | 2 DMAs (`v_lo`, `v_hi`), 2 `nc_matmul` |

The root cause of all of it is one layout choice: v1 puts **ctx on the partition
axis**, which forces (a) a transpose to place the mask, (b) a partition-dim
reduction for the softmax denominator, and (c) the full score tile to be
materialised before AV can begin.

**What v2 changes.** Scores live as `[1, ctx]` — partition 1, free ctx:

- `mask_bias[0, chunk]` loads **directly** as `[1,128]`. No memset, no transpose.
- the softmax sum is a **free-axis** `tensor_reduce` accumulated into a running
  `[1,1]`. The entire partition-reduction chain disappears.
- K and V are each loaded **once per chunk at full width `[128,256]`**, halving
  DMA instructions and doubling the contiguous free dimension per descriptor
  (256B -> 512B for bf16).
- AV becomes **one** matmul instead of two:
  `nc_matmul(stationary=w_t[ctx,1], moving=v_chunk[ctx,256]) -> [1,256]`.
  Both head-dim halves in one instruction, no V split, and the result lands in
  the final `[1,256]` layout so there is **no closing transpose**.
- one fused loop, so chunk `c+1`'s DMA can overlap chunk `c`'s compute.

No running max and no rescale: like v1, the additive mask saturates to -65504 so
masked slots underflow to 0, which leaves the unnormalised weights **linear** in
`u` — so both the denominator and the AV product accumulate across chunks with no
flash-style correction term. That linearity is what makes the single pass possible.

Per chunk: DMAs 5 -> 3, `nc_matmul` 4 -> 3, `memset` 1 -> 0, `nc_transpose` 3 -> 3
(the two K transposes are unavoidable — `nc_matmul` contracts over the partition
axis, QK contracts over the head dim, and K arrives as `[ctx, d]`; V needs none
because AV contracts over ctx), plus v1's ~10-instruction global reduction chain
goes to zero.

**Parity** (`tests/test_decode_hd256_v2_parity.py`, direct `nki.simulate`, first
attempt):

| shape | S_ctx | fp32 cosine | bf16 cosine |
|---|---:|---:|---:|
| smoke | 128 | 1.000000 | 0.999994 |
| 4B_short | 128 | 1.000000 | 0.999993 |
| 4B_typical | 512 | 1.000000 | 0.999993 |
| 4B_2K | 2048 | **1.000000** (max_abs 0.0) | 0.999980 |

**Device A/B**, same node, same session, sequential, coherence-gated:

```
  torch.compile'd PyTorch  295.01 ms/token   0.165 RPS/chip
  v1       61.85 ms/token   0.514 RPS/chip
  v2       60.20 ms/token   0.522 RPS/chip     <- 1.027x vs v1
```

**Honest verdict: v2 is a 2.7% decode win.** Reproducible (60.26 at 50 output
tokens, 60.20 at 200) but small, and ~1.6% on throughput. It is not the
multiple the instruction-count reduction suggests.

**Why it is small — the structural ceiling.** Qwen3.5-4B is 32 layers: **24
`linear_attention` (GDN) + 8 `full_attention`**. `decode_hd256` only runs on the
**8** full-attention layers. Even a perfect attention decode kernel can only
touch a quarter of the model, and the GDN recurrent step dominates the rest.
This is consistent with two other measurements on the same model: the transposed
recurrent-state rewrite (which targeted the 24 GDN layers) was null, and the
`max_num_seqs` sweep was flat.

So the remaining decode headroom on Qwen3.5 is in the **GDN recurrence**, not in
attention. And since the model is prefill-bound at Indeed's shape anyway
(prefill 0.906 s of ~1.28 s service time), neither is the lever that moves
capacity — that is the 768 per-`(b,h)` prefill kernel launches.

### Reference materials for v2

- `aws-neuron/nki-library/src/nkilib_src/nkilib/core/attention/attention_tkg.py`
  — flash attention reference (`_MAX_D_HEAD = 128`, doesn't help us
  directly but the pattern transfers)
- `aws-neuron/nki-library/src/nkilib_src/nkilib/core/attention/attention_tkg_design_spec.md`
  — design spec with diagrams of the LNC2 sharding + FA loop
- vllm-neuron internal `vllm_neuron/functional/attention/attention_decode.py`
  — production decode kernel for head_dim ≤ 128

### Integration (current state)

The kernel is wired into `qwen3.5-4b-trainium/src/qwen3_5/model_bf16.py`
behind `QWEN35_NKI_DECODE=1`. When set (and S_decode=1, head_dim=256,
S_ctx % 128 == 0), `forward_decode` calls `call_decode_hd256(...)` per
(batch, head). Default behavior (flag unset) is unchanged — uses the
eager path that all currently-published benchmarks measure.

This means: with the v1 kernel committed but the flag default off,
existing benchmarks remain valid and the NKI kernel is opt-in for
experimentation.

## Roadmap

- [x] Write v1 kernel (correctness, no perf focus)
- [x] Validate parity via nki.simulate (cosine > 0.99998)
- [x] Wire into model + verify end-to-end on hardware
- [x] A/B bench against eager → confirmed v1 is SLOWER
- [x] Write v2 (single-pass restructure; flash rescale proved unnecessary
      because the unnormalised weights are linear in u)
- [x] A/B bench v2 -> 1.027x vs v1, 4.9x vs eager
- [ ] Upstream v1+v2 (v1 is 4.77x eager at MNS=16 -- the earlier
      'do not upstream' verdict was measured at MNS=1 and is withdrawn)
- [ ] Long-context bench (MAX_LEN=20480 customer shape)
- [ ] Concurrency stress with `MAX_NUM_SEQS=8` (current bench was
      single-stream because serve.sh defaults to MAX_NUM_SEQS=1)
- [ ] Wire into Qwen3.6-27B (same shape contract)
- [ ] If v2 wins → submit to `aws-neuron/nki-library/experimental/attention/`
- [ ] FP8 KV variant (Path D) — fold v_dequant_scale into the kernel scale

## Honest takeaway

The compiler is good. Hand-writing NKI is not automatically a win. The
right question to ask before writing a kernel is "what is the compiler
producing that's suboptimal?" — answering that requires reading the
emitted MLIR / NEFF and spotting concrete inefficiencies. We didn't
do that step before writing v1, so v1 ended up implementing a layout
that compiles cleanly but is no better than what neuronx-cc already
does for the eager path.

For v2 we should:
1. Capture a profile of the eager decode (with neuron-profile)
2. Look at where the time actually goes — DMA traffic, matmul
   utilization, softmax cost
3. Write the v2 kernel to specifically beat that bottleneck

This is the methodology the AWS Neuron team uses for the kernels in
nki-library. It's also exactly what the
`/Users/aghaebra/Downloads/test_kiro/.kiro/skills/neuron-nki-profile-querying`
skill is designed for.
