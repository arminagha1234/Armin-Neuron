# Qwen3.5-4B throughput: 0.054 -> 0.775 RPS/chip, and why it is now prefill-bound

Qwen3.5 dominates this study's fleet estimate: its target is 500 RPS, 10x the
other three models, so it drives most of the box count. This is the ledger of
what moved it, what did not, and where the ceiling now sits.

All numbers: `trn2.48xlarge`, LNC=2, TP=4 (**4 logical cores = 1 Trainium2
chip**, so RPS/replica == RPS/chip and a box holds 16 replicas), bf16,
`max_model_len=2048`, `max_num_seqs=16`, prefix caching off, greedy, 1811-token
prompt / 50 output tokens. Coherence checked on every configuration. Servers
benchmarked **one at a time**.

## Ledger

| # | Change | RPS/chip | vs stock | Boxes for 500 RPS |
|---|---|---:|---:|---:|
| 0 | stock (e2e 18.87 s) | 0.054 | 1.0x | 579 |
| 1 | `QWEN35_NKI_DECODE=1` | 0.136 | 2.5x | 230 |
| 2 | `--num-gpu-blocks-override 140` | 0.172 | 3.2x | 182 |
| 3 | blocked triangular inverse (PR #98) | 0.414 | 7.7x | 76 |
| 4 | `--num-gpu-blocks-override 200` | 0.515 | 9.5x | 61 |
| 5 | clean rebuild, current DLC | **0.775** | **14.4x** | **40** |
| 6 | transposed recurrent state | 0.775 | — | 40 | 

Rows 0-4 are prior sessions. Rows 5-6 are this one.

## Row 4: the `num_gpu_blocks` ceiling is 200

DeltaNet's recurrent state is stored in the paged KV block pool (the port hijacks
`k_cache`), so block count gates concurrency rather than just context length.
Sweeping it:

| blocks | result |
|---|---|
| 140 | 0.414 RPS/chip — flatlines at conc=2 |
| **200** | **0.515 RPS/chip** — climbs to conc=8 (0.413 / 0.500 / 0.515) |
| 230 | server dies, `NRT_RESOURCE` |
| 260 | server dies, `NRT_RESOURCE` |

Going 140 -> 200 changes the *shape* of the concurrency curve, not just its
height: at 140 the scheduler cannot keep more than two sequences resident, so
added concurrency does nothing. **200 is the practical maximum** — zero code, 1.24x.

## Row 5: 0.775 RPS/chip, and a 4x decode discrepancy worth flagging

Rebuilt from scratch on a fresh node and the current public DLC, with the blocked
inverse and blocks=200:

| | value |
|---|---|
| prefill-only p50 (`max_tokens=1`) | 0.905 s -> 2,000 tok/s |
| e2e p50 (50 tokens) | 1.670 s |
| decode | **15.60 ms/token** (45% of e2e) |
| conc 1 / 2 / 4 / 8 | 0.599 / 0.559 / 0.775 / 0.770 RPS |
| peak | **0.775 RPS/chip = 12.4 RPS/box -> 40 boxes** |

**Prefill reproduces the blocked-inverse result exactly** (0.905 s vs 0.906 s;
2,000 tok/s both), which is the anchor that says this is the same workload.
**Decode does not:** 15.60 ms/token here against 61.8 ms/token previously — 4x
faster, unexplained by any change we made.

We are not claiming credit for that 4x. The earlier 61.8 ms was corroborated
twice (host measurement 61.82 ms, device profile 62.15 ms on the decode NEFF), so
it was real *on that build*. The difference is most likely the image: the previous
node's DLC shipped `qwen3_5` preinstalled and this one does not, so the
`vllm_neuron` build differs. Anyone reproducing should re-measure decode rather
than assume either figure. Two independent baseline runs on this node agree
(0.775 / 0.775, conc=8 0.770 / 0.769), including one with an idle 31B server
co-resident on other devices — which turned out not to perturb it.

## Row 6: transposing the recurrent state is a null result

**Hypothesis.** The decode recurrence writes two matrix-vector products as
broadcast-multiply plus a reduction over `dim=-2`:

```python
kv_mem  = (new_state * k_t.unsqueeze(-1)).sum(dim=-2)
out_one = (new_state * q_t.unsqueeze(-1)).sum(dim=-2)
```

`new_state` is `[B, 32, 128, 128]` fp32 = 2.1 MB per sequence per layer. Each
broadcast materialises a full fp32 copy, and `dim=-2` maps to the **partition**
axis on trn2, so the reduce goes through the PE array as a transpose. The device
profile of the *old* build attributed 86% of decode FLOPs (796 of 924 GFLOP) to
transposes, with DMA 100.4% active, arithmetic intensity 9.6, and 12.0 GB HBM
read per token against ~2.2 GB of weights.

**Change.** Store the state as `[B, 32, head_v_dim, head_k_dim]` instead. Both
reductions then run over the **last** (free) axis, which the vector engine does
directly, with no PE transpose:

```python
kv_mem  = (st * k_t.unsqueeze(-2)).sum(dim=-1)
st      = st + delta.unsqueeze(-1) * k_t.unsqueeze(-2)
out_one = (st * q_t.unsqueeze(-2)).sum(dim=-1)
```

Algebraically identical, since `st[v,k] == state[k,v]`. `head_k_dim ==
head_v_dim == 128` and the state is kept flat in `k_cache` with an explicit
reshape on read, so the layout swap needs no cache resize. Prefill writes the
state transposed **once per request**; decode then reads and writes it transposed,
so no per-token transpose survives. See
[`patches/patch_state_transpose.py`](patches/patch_state_transpose.py), gated by
`QWEN35_STATE_T=1`.

**Result — null.**

| | baseline | `QWEN35_STATE_T=1` |
|---|---:|---:|
| prefill p50 | 0.905 s | 0.905 s |
| decode | 15.60 ms/token | 15.33 ms/token (-1.7%) |
| e2e p50 | 1.670 s | 1.656 s |
| peak | 0.775 RPS/chip | **0.775 RPS/chip** |

It genuinely engaged — all four prefill NEFF hashes differ from baseline — and
output stayed coherent. It simply had nothing to grab: the transpose mountain
that motivated it belongs to the old build's decode graph, and on this build
decode is already 4x cheaper. Recorded so nobody re-derives it.

## Where the ceiling is now: prefill, not decode

Both arms cap at 0.775 RPS/chip at conc=4 **and** conc=8. Prefill alone is
0.905 s, so:

- prefill-only ceiling = 1 / 0.905 = **1.105 RPS/chip** (17.7 RPS/box, 28 boxes)
- measured peak is 0.775 = **70% of that ceiling**
- therefore **a free decode would buy at most 1.43x**, and no decode change can
  do better

That is a hard reframe. Decode was 77% of e2e on the old build and the obvious
next target; here it is 45% of e2e at conc=1 and cannot move the peak at all.
Prefill work is the only thing that raises this number.

The prefill profile (previous session, same kernel) says where the room is: MFU
13.06% against a **64.80%** compiler-reported achievable ceiling (~5x), DMA 95%
active, 14% of FLOPs in transposes, arithmetic intensity 142.

Two unattempted prefill levers, in order of expected value:

1. **Batch the per-`(b,h)` kernel launches.** `_forward_prefill` calls the
   DeltaNet NKI kernel once per (batch, head) from a serial Python loop:
   32 heads x 24 linear layers = **768 launches per request**. At 0.905 s that is
   1.18 ms per launch on a `[1811,128]` single-head slice, which is still
   plausibly dispatch-dominated. The kernel has no batch axis at all (no
   `nl.program_id` / `nl.spmd` anywhere), so this is a kernel change: promote
   `(b,h)` to a real SPMD/grid axis and call once per layer. Independent of, and
   composable with, the blocked inverse. Validate with `nki.simulate` on CPU
   before paying a compile.
2. **The fp32 upcast and 10 `.contiguous()` copies** at `model_bf16.py`
   1092-1147: five `.transpose(1,2).contiguous().float()` on
   `[1,32,1811,128]` tensors (~31 MB each in fp32) plus five more
   `reshape(...).contiguous()` to flatten `BH`, per layer, x24 layers. With DMA
   95% active, deleting copied bytes is a direct attack on the binding resource.
   Also pads S 1811 -> 1920 (+6% wasted work).

## Reproducing

```bash
# 1. install the qwen3_5 package into the DLC, then make from_configs accept
#    vLLM's text_neuron_config kwarg (needed after ANY source overlay)
python3 patches/q35_compat_patch.py

# 2. optional: the null-result state layout change
python3 patches/patch_state_transpose.py     # then QWEN35_STATE_T=1

# 3. serve
QWEN35_BLOCK_INV=1 QWEN35_NKI_DECODE=1 \
vllm serve <model> --served-model-name q35 \
  --tensor-parallel-size 4 --max-model-len 2048 --max-num-seqs 16 \
  --num-gpu-blocks-override 200 --no-enable-prefix-caching \
  --additional-config '{"neuron_config":{"kv_segment_size_buckets":[2048],
      "on_device_sampling_config":{"all_greedy":true}}}'
```

`q35_compat_patch.py` exists because the registry resolves
`Qwen3_5ForConditionalGeneration` from `factory.py`, while only `model_bf16.py`
carries the `text_neuron_config` fix. Overlaying patched sources without also
patching `factory.py` fails at worker startup with
`TypeError: from_configs() got an unexpected keyword argument
'text_neuron_config'`, surfacing as `Engine core initialization failed` — dig
past that cascade for the real exception.
