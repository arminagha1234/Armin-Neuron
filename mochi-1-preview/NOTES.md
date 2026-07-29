# Mochi-1 port notes — verified facts and reasoning

Everything here was read off the published checkpoint or the diffusers
source, not inferred from the model card. Where a number comes from
arithmetic rather than measurement, it says so.

## Architecture, from the checkpoint headers

Read by range-requesting the safetensors headers of
`transformer/diffusion_pytorch_model.bf16-*.safetensors` (first 8 bytes give
the JSON header length, then the header lists every tensor's shape).

```
10.028 B parameters, 20.06 GB bf16, 1071 tensors across 3 shards
48 blocks, 24 heads x 128 head_dim, inner_dim 3072
text ("context") stream dim 1536          <- the asymmetry in AsymmDiT
ff inner 8192, ff_context inner 4096
```

Per-block parameters and the sharding rule applied to each:

| Parameter | Shape | Bias | TP rule |
|---|---|---|---|
| `attn1.to_{q,k,v}.weight` | `[3072, 3072]` | no | column (dim 0) |
| `attn1.add_{q,k,v}_proj.weight` | `[3072, 1536]` | no | column (dim 0) |
| `attn1.to_out.0.weight` | `[3072, 3072]` | yes | row (dim 1) |
| `attn1.to_add_out.weight` | `[1536, 3072]` | yes | row (dim 1) |
| `attn1.norm_{q,k,added_q,added_k}.weight` | `[128]` | — | replicate |
| `ff.net.0.proj.weight` | `[16384, 3072]` | no | **fused GLU** |
| `ff.net.2.weight` | `[3072, 8192]` | no | row (dim 1) |
| `ff_context.net.0.proj.weight` | `[8192, 1536]` | no | **fused GLU** |
| `ff_context.net.2.weight` | `[1536, 4096]` | no | row (dim 1) |
| `norm1.linear.weight` | `[12288, 3072]` | yes | replicate |
| `norm1_context.linear.weight` | `[6144, 3072]` | yes | replicate |

Top level, all replicated: `patch_embed.proj` `[3072, 12, 2, 2]`,
`time_embed.*` (including a 63 M-parameter attention pooler),
`norm_out.linear` `[6144, 3072]`, `proj_out` `[48, 3072]`,
`pos_frequencies` `[3, 24, 64]` (sharded on the head axis — see below).

### Block 47 is different

`context_pre_only=True` on the last block. It has **no** `to_add_out` and
**no** `ff_context`, and its `norm1_context` is a `MochiLayerNormContinuous`
with `linear_1` `[1536, 3072]` rather than the `MochiRMSNormZero` `linear`
`[6144, 3072]` the other 47 blocks have. The TP plan omits the missing
modules; `test_plan_paths` asserts both the omission and that the
architecture really lacks them.

### Checkpoint file naming quirk

The bf16 index is `diffusion_pytorch_model.safetensors.index.bf16.json` —
variant suffix **after** `.index`, not the usual diffusers
`...bf16.safetensors.index.json`. `_read_index` tries both plus the fp32
fallback rather than pattern-matching.

The fp32 transformer is 5 shards / ~40 GB; the runner's `--variant bf16`
path passes `ignore_patterns` to `snapshot_download` so you do not pull 40 GB
you will not use.

## Op triage

| Concern | Verdict |
|---|---|
| `torch.view_as_complex` in RoPE | **absent.** Mochi's processor defines its own real sin/cos `apply_rotary_emb`. No RoPE rewrite, unlike Z-Image / FLUX.2-klein. |
| `torch.nonzero` in the processor | **present, fatal.** Fixed. See below. |
| `F.scaled_dot_product_attention` | present. Replaced with explicit BMM per the LTX-2 finding that stock SDPA miscomputes on Neuron's compiled bf16 lazy backend. |
| Bool SDPA mask | **present** in `MochiAttentionPool`. Needed a shim fix the LTX-2 version lacks. |
| `torch.autocast(device.type, fp32)` in `MochiRoPE._create_rope` | present. Not a registered autocast backend off CUDA — reproduced the warning locally on CPU ("target dtype is not supported. Disabling autocast"). Sidestepped by CPU precompute. |
| `meshgrid` / `linspace` / `arange` on device | present in `MochiRoPE._get_positions`. Moved to CPU. |
| Dynamic shapes in `pipeline_mochi.py` | none. `num_frames = (num_frames - 1) // 6 + 1` is static Python. |
| `.nonzero()` in the scheduler | present in `FlowMatchEulerDiscreteScheduler.index_for_timestep`. Avoided by keeping the scheduler on CPU. |
| VAE `Conv3d` / `repeat_interleave` / chunked GroupNorm | present, unexamined — VAE stays on CPU. |

### The `torch.nonzero` blocker in detail

Upstream `MochiAttnProcessor2_0`:

```python
mask = attention_mask[idx][None, :]
valid_prompt_token_indices = torch.nonzero(mask.flatten(), as_tuple=False).flatten()
valid_encoder_query = encoder_query[idx:idx+1, :, valid_prompt_token_indices, :]
valid_query = torch.cat([query[idx:idx+1], valid_encoder_query], dim=2)
attn_output = F.scaled_dot_product_attention(valid_query, valid_key, valid_value, ...)
valid_sequence_length = attn_output.size(2)
attn_output = F.pad(attn_output, (0, 0, 0, total_length - valid_sequence_length))
```

`torch.nonzero`'s output shape depends on tensor *values*, and
`valid_sequence_length` plus the `F.pad` amount inherit that. Nothing here
can be traced to a static graph.

The replacement keeps all 256 text tokens and biases padded text **key**
columns by `-10000.0`. Padded keys get ~0 softmax weight, so the visual
stream is unchanged, and the `F.pad` disappears because nothing was dropped.

Equivalence is exact in three cases the tests cover:

- partially padded prompt (4 real / 2 pad): `max|err| = 0.00e+00` visual
- fully-masked prompt: `max|err| = 1.79e-07`, no NaN
- perturbing padded context values leaves the visual output bit-identical,
  confirming masked tokens genuinely cannot leak

The fully-masked case is not hypothetical: `_get_t5_prompt_embeds` sets
`prompt_attention_mask = torch.zeros_like(..., dtype=torch.bool)` whenever
the negative prompt is empty, which is the default. No softmax row ends up
fully masked because visual keys are never masked, so there is no NaN risk.

Upstream also zero-fills the encoder output rows at dropped positions (that
is what the `F.pad` does) before `to_add_out`. Those rows only feed the next
block's masked-out K/V, so they cannot influence the video — but
`zero_padded_context=True` reproduces the zero-fill anyway so the encoder
stream stays comparable against a CPU reference while debugging.

### The fused-SwiGLU sharding trap

`diffusers.models.activations.SwiGLU`:

```python
hidden_states, gate = self.proj(x).chunk(2, dim=-1)
return hidden_states * self.activation(gate)
```

So for `ff.net.0.proj` `[16384, 3072]`, global output rows `[0:8192]` are the
value half and `[8192:16384]` are the gate half. `ColwiseParallel` shards
contiguously, handing rank *r* rows `[r*4096 : (r+1)*4096]`; the local
`chunk(2)` then pairs global `[0:2048]` with `[2048:4096]` on rank 0 instead
of `[0:2048]` with `[8192:10240]`.

Every shape checks out. The model runs. The video is wrong.

`_shard_fused_glu` gives each rank `concat(value_slice_r, gate_slice_r)`, so
the local `chunk(2)` recovers the correct pairing. The DTensor's notional
*global* view is then a permutation of the true weight, which is harmless:
nothing reconstructs the global tensor and every consumer touches only the
local shard.

Measured in `test_swiglu_shard`: permuted shard matches unsharded to
2.09e-07; naive contiguous shard is off by 4.79e-01.

### The bool-mask shim bug

`MochiAttentionPool.forward` builds

```python
attn_mask = mask[:, None, None, :].bool()
attn_mask = F.pad(attn_mask, (1, 0), value=True)
x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, ...)
```

Torch SDPA treats a bool mask as `True = attend`. The LTX-2 shim does
`scores = scores + attn_mask`, which for bool adds 1.0/0.0 to the logits —
no error, no NaN, just wrong. `_normalize_mask` converts bool to
`(~mask) * -10000.0` first. Measured difference between the two: 3.97e-01.

This runs on device: the pooler lives inside `time_embed`, which is part of
the transformer.

### Why no `RankTensor`

LTX-2 needs a traced per-rank index because a Python `rank` baked into a
slice compiles to `constant 0` for every rank under XLA SPMD tracing, so all
ranks silently apply rank 0's RoPE shard.

Mochi's RoPE frequencies live in a `pos_frequencies` parameter of shape
`[3, 24, 64]` — indexed by head. Sharding that *parameter* on its head axis
at load time gives each rank its own frequencies before tracing ever starts,
so there is no baked-constant hazard to work around.
`test_rope_precompute` verifies rank 2 of 4 selects exactly
`ref_cos[:, 12:18]`.

### Why no adaptive QK norm

`MochiAttention.__init__` builds `MochiRMSNorm(dim_head, eps, True)`, giving
`[128]`-shaped norm weights applied over the last axis of
`(B, S, H, 128)`. Sharding the head axis leaves the norm valid unchanged, so
the weight replicates and needs no cross-rank reduction. LTX-2's
`rms_norm_across_heads` normalised over the full inner_dim and therefore
needed an all-reduce inside the norm; Mochi does not.

## Memory arithmetic

All computed, none measured.

### Per-rank weights

AdaLN modulation (`norm1.linear` + `norm1_context.linear`) is 2.705 B
parameters, 27% of the model. It modulates unsharded hidden states, so
sharding it would need a per-block all-gather. It stays replicated, and TP
therefore scales sub-linearly.

| TP | GB/rank | naive 20/TP | local heads |
|---:|---:|---:|---:|
| 1 | 20.08 | — | 24 |
| 2 | 12.86 | 10.04 | 12 |
| 4 | 9.25 | 5.02 | 6 |
| 8 | 7.44 | 2.51 | 3 |

Valid TP is **{1, 2, 4, 8}** only. TP=3/6/12 divide the 24 heads but not the
8192 FF inner dim; TP=16 divides neither. `validate_world_size` rejects the
rest rather than failing later inside the loader.

### Attention scores

Latent grid at 480×848: `/8` spatially → 60×106, `patch_size=2` → 30×53 =
1590 tokens per latent frame. Latent frames = `(frames-1)//6 + 1`.

| Frames | Latent frames | Visual tokens | +text | Scores, 24 heads, bf16 |
|---:|---:|---:|---:|---:|
| 19 (diffusers default) | 4 | 6,360 | 6,616 | 2.1 GB |
| 31 | 6 | 9,540 | 9,796 | 4.6 GB |
| 61 | 11 | 17,490 | 17,746 | 15.1 GB |
| 85 | 15 | 23,850 | 24,106 | 27.9 GB |
| 163 (model card) | 28 | 44,520 | 44,776 | 96.2 GB |

163 frames reproducing the model card's published 44,520 is a useful check
that this arithmetic is right; `test_arithmetic` asserts it.

Multiply by ~2 for `scores` and `probs` being live together, and by 2 again
for CFG batching. Divide by TP. So 163 frames at TP=4 with CFG would want
roughly 96.2 × 2 × 2 / 4 ≈ 96 GB per rank untiled — hence tiling.

### Tiled attention

Each query tile attends to all keys, so the softmax denominator is complete
within the tile and no online-softmax rescaling is needed. Memory becomes
O(q_chunk · Sk) instead of O(Sq · Sk). Verified exact against the untiled
path at chunk sizes 32, 64, 128, and a deliberately non-dividing 7, and
against fused SDPA at 3.58e-07.

Auto-tiling engages above 2**26 score elements per plane (128 MiB bf16) and
picks `q_chunk ≈ 2**26 / Sk` rounded to a multiple of 512. Tile size affects
the compiled graph shape, so changing it invalidates cached NEFFs — set it
once via `--q-chunk`.

## Offline test coverage

`tests/test_offline.py`, 50 checks, CPU only, ~40 s. Grouped by what they
would have cost to debug on device:

1. fused-SwiGLU shard equivalence + proof the naive shard is wrong
2. attention TP equivalence at TP=2 (colwise + rowwise + heads patch
   together, biases added once after the partial sum)
3. processor equivalence vs upstream: padded prompt, fully-masked prompt,
   leak isolation, and a `torch.nonzero` call counter asserting zero
4. tiled attention exactness, including per-query mask slicing
5. bool mask handling, including a demonstration the fix is load-bearing
6. all 573 TP plan paths resolve to real `nn.Linear` modules on a meta-init
   48-layer model, at TP=2/4/8, plus block-47 asymmetry and world-size
   rejection
7. RoPE precompute bit-exactness, caching, and head-axis shard selection
8. token count vs the model card, parameter count vs the checkpoint,
   monotonic TP scaling
9. all sources import, runner byte-compiles and `--help` runs off-device

What this does **not** establish: that the port produces correct video. Each
transformation is verified numerically neutral in isolation, which is a
weaker claim than end-to-end output matching. The loader has never read real
Mochi weights.

## Open questions — resolved on device (2026-07-29)

Run on `i-03a587c283fffb075` (trn2.48xlarge, us-east-2b), Beta-3 DLC, TP=4.

1. **fp32 RoPE across the compile boundary?** Fine. `--rope-bf16` was never
   needed; eager and compiled both produce correct output with fp32 tables.
   The LTX-2 fix #5/#8 failure mode did not reproduce for Mochi.
2. **Does `use_local_output=True` hold through the processor chain?** Yes.
   The loader reports `{replicate: 498, colwise: 288, rowwise: 190, glu: 95}`
   and `attn.heads` patches to 6 on all 48 blocks; output is correct, so the
   DTensor→local plumbing survives `unflatten`/`transpose`/`cat` on device.
3. **Attention pooler with the F.pad-ed bool mask?** Compiles and runs. The
   bool-mask shim (`_normalize_mask`) was the reason this works; without it
   the pooler would have been silently wrong.
4. **Latency and CPU share.** Eager ~6.3 s/step denoise; compiled warm
   ~8–15 s/step. CPU VAE decode is ~40% of wall clock and every rank
   duplicates it — the biggest remaining optimisation, exactly as LTX-2 and
   Cosmos predicted.
5. **TP=8?** No, and it turns out not to matter. `tools/collective_probe.py`
   sweep on device: TP=2/4/16 init cleanly, TP=8 fails `no_hier no_mesh`. The
   16 devices are a 4×4 NeuronLink torus (each links to 4 neighbours per
   `neuron-ls`); collectives tile it as a link (2), ring (4), or full mesh
   (16), and 8 is neither. Mochi needs TP | 24 heads → torus-valid ∩
   divisible = {2, 4}, so TP=4 is the hard max for pure head-parallelism.
   TP=16 is torus-valid but needs 24 heads padded to 32.

   The important part: **raising TP would not lift the memory ceiling.** The
   61f eager OOM was 436,125,696 bytes = 2·17746·3072·4 — the full inner dim
   in fp32, i.e. the residual-stream norm activation. `RowwiseParallel`
   all-reduces block outputs back to full 3072 width, so the residual stream
   and its norms are replicated across head-parallel ranks, not sharded.
   TP=8/16 would leave that tensor identical. The levers that actually shard
   it are sequence/context parallelism or bigger cores (LNC=4, 48 GB/rank).

## Accuracy verification (2026-07-29)

Closed the "no CPU reference" gap. Ran identical config (19f, 8 steps,
guidance 1.0, seed 777) on Neuron (bf16) and CPU (fp32), same bf16 checkpoint
weights, comparing pre-VAE latents so the VAE is excluded.

- **Latents:** cosine 0.99913, correlation 0.99911, relative L2 4.20%,
  PSNR 50.2 dB, per-channel rel L2 2.6–5.9% (uniform, not structured).
- **Decoded frames (same VAE):** whole-clip PSNR 51.95 dB, flat ~52 dB across
  all 19 frames, MAE 0.52 on 0–255 (half a level). Side-by-side visually
  indistinguishable.

The ~4% latent divergence is bf16 rounding accumulated over 8 denoising
steps, not a bug — structured error (wrong shard/RoPE/mask) would collapse
cosine and blow up relative error. Tools: `tests/compare_latents.py`,
`tests/decode_and_compare.py`. Point estimate at one geometry, not swept.

## New findings that only surfaced on device

- **The long-sequence memory wall is the RMS norms, not attention.** Mochi's
  `MochiModulatedRMSNorm`/`MochiRMSNormZero` upcast the full `(B, S, 3072)`
  tensor to fp32, four times per block. Failures matched
  `batch × tokens × 3072 × 4` to the byte (436,125,696 at 61f CFG;
  550,281,216 at 163f). `mochi_norm_memory.py` tiles this, bit-exact. This
  was not visible from source reading — it only showed up as an OOM whose
  size did not match any attention tensor.

- **`torchrun` sets `OMP_NUM_THREADS=1`.** First run's CPU VAE decode crawled
  on 1 thread/rank. Pass `OMP_NUM_THREADS=48` explicitly.

- **Eager hits an allocator-fragmentation ceiling at ~31 frames**, not a
  capacity limit: 23.25 GB allocated of 24, 2.72 GB free, but largest free
  chunk only 221 MB. `torch.compile` (whole-graph XLA buffer assignment)
  clears it and reaches 61 frames. Compile's value here is memory, not speed.

- **24 GB per logical core** at `logical-neuroncore-config: 2` (confirmed via
  `neuron-ls` + the OOM `total_hbm=25769803776`). `NEURON_LOGICAL_NC_CONFIG=4`
  for 48 GB cores is accepted by `neuron-ls` but rejected at runtime once the
  driver is initialised at config 2.

- **Persistent NEFF cache** at `/data/neuron_cache` survives `docker restart`;
  a warm 31f compiled run skipped the ~770 s build (589 s vs 1360 s cold).

## Sources

- [genmo/mochi-1-preview](https://huggingface.co/genmo/mochi-1-preview) —
  model card, config, checkpoint headers
- [diffusers `transformer_mochi.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_mochi.py)
- [diffusers `pipeline_mochi.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/mochi/pipeline_mochi.py)
- [diffusers `attention_processor.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py) —
  `MochiAttention`, `MochiAttnProcessor2_0`
- [diffusers `activations.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/activations.py) — `SwiGLU`
- [aws-neuron LTX-2 contrib](https://github.com/aws-neuron/neuronx-distributed-inference/tree/main/contrib/models/ltx2-video-audio) —
  origin of the BMM-SDPA and additive-mask fixes
- local: `.tmp/Armin-Neuron/ltx2/native-pytorch/src/` — the recipe this
  derives from

Content from external sources was paraphrased; code excerpts are quoted only
as needed to identify the specific lines being patched.
