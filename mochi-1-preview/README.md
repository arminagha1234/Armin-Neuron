# Mochi-1 preview (10B AsymmDiT) on AWS Trainium2 — native PyTorch

Port of [genmo/mochi-1-preview](https://huggingface.co/genmo/mochi-1-preview)
to Trainium using the **native PyTorch backend (TorchNeuron)** — eager plus
`torch.compile(backend="neuron")`, no XLA, not NxD Inference.

## Status: working on device

Validated end-to-end on trn2.48xlarge, eager mode, TP=4.

![Mochi-1 on Trainium2, frame 15 of 31](results/frame_15.png)

*"Close-up of a chameleon's eye, with its scaly skin changing color. Ultra
high resolution 4k." — 848×480, 31 frames, 64 steps, cfg 4.5, TP=4 eager.*

| Stage | State |
|---|---|
| Architecture analysis against the real checkpoint | done |
| `torch.nonzero` removal (the one hard blocker) | done, verified vs upstream |
| Fused-SwiGLU TP sharding | done, verified |
| TP plan (573 entries, TP ∈ {2,4,8}) | done, all paths resolve |
| Tiled attention for long clips | done, verified exact |
| RoPE CPU precompute + head-axis sharding | done, verified bit-exact |
| Sharded meta-init weight loader | **working** — 1071 tensors in 3.0 s |
| End-to-end generation | **working** — sharp, prompt-adherent output |
| Tiled RMS norms (long-sequence memory) | **working**, bit-exact |
| `torch.compile` path | **working** — extends envelope to 85 frames |
| Accuracy vs a CPU reference | **measured — cosine 0.9991, rel L2 4.2%** |

Offline suite: **54/54 passing**, both on macOS CPU (torch 2.12) and inside
the Beta-3 DLC (torch 2.11).

### Measured results

| Config | Denoise | Total | Notes |
|---|---:|---:|---|
| 19f, 4 steps, no CFG | 29 s (7.3 s/step) | 204 s | undersampled (Mochi wants ~64 steps) |
| 31f, 64 steps, cfg 4.5 | ~400 s (6.3 s/step) | 662 s | **the validated result above** |

### Validated envelope, and where it stops

| Frames | Tokens | CFG | eager | compiled |
|---:|---:|:---:|---|---|
| 19 | 6,360 | off | works | — |
| 31 | 9,540 | **on** | **works (ref)** | works |
| 61 | 17,490 | on | OOM (fragmentation) | **works** |
| 85 | 23,850 | on | OOM | **works** (~2.8 s of video) |
| 163 | 44,520 | off | OOM | compiler error (NCC_IBTN020) |

The 85-frame compiled clip is the validated ceiling. **163 frames — the
model card's headline geometry — does not compile:** after a 64-minute
build it fails with `[NCC_IBTN020] convertAccessPattern: step out of int16
range`. At 44,520 tokens some tensor access pattern has a stride exceeding
the int16 limit (32,767). This is a Neuron-compiler constraint at scale,
**distinct from the memory ceiling** that caps eager at 31 frames — 163f gets
through TP sharding, norm tiling, and attention tiling, then dies inside
`neuronx-cc`. Localizing the offending op needs `XLA_HLO_DEBUG` plus a fresh
~1 h compile per iteration; not pursued. Was CFG-off (batch 1) with
`--q-chunk 512`, so the int16 overflow is in a reshape/transpose stride, not
the attention tile.

The eager ceiling is **allocator fragmentation**, not raw capacity. The
61-frame eager failure reports `allocated peak = 23.25 GB` of 24 GB with
`total_free = 2.72 GB` but `largest_cached_free = 221 MB`, across
`segments=189 free_chunks=62`. There is enough free memory for the 436 MB
request; there is no contiguous chunk big enough.

**`torch.compile` clears this** — see below. 61 frames with CFG, which OOM'd
in eager mode before step 0, runs to completion compiled and produces correct
output. So the eager envelope is 31 frames; the compiled envelope is at least
61 frames (~2 s of video).

Three levers, none of which worked in this environment:

- **`torch.compile`** is the most promising lever precisely because XLA does
  buffer assignment and would not fragment the way the eager allocator does.
  See the compile results below.
- **More tensor-parallel ranks would NOT help**, for two independent reasons
  detailed in "The collective / TP=8 story" below.

So: 31 frames (~1 s of video at 30 fps) with the reference 64-step CFG config
is the validated envelope in eager mode on a shared 24 GB/rank configuration.

### torch.compile

`--compile` works and produces correct output at 31 frames.

| Run | Total | NEFF build | Warm per-step (progress bar) |
|---|---:|---:|---:|
| cold (first call) | 1360 s | ~770 s | 36–51 s/it (compile bleed) |
| warm (NEFF cached) | 589 s | 0 | **11–15 s/it** |

The persistent NEFF cache (`NEURON_COMPILE_CACHE_URL=/data/neuron_cache`)
survives `docker restart` — the warm run skipped the entire ~770 s build.

The step-1 timer absorbs the entire NEFF build; subsequent steps reuse the
in-process NEFF. Measured cleanly at 61 frames: step 1 took ~25 min
(compile), then steps 2–8 ran at **~8 s/step** (progress-bar elapsed went
41:42 → 41:50 → 41:58 across steps 6→7→8).

**Compile clears the eager fragmentation ceiling.** This is the headline
result: 61 frames with CFG, which eager OOM'd before step 0
(`largest_cached_free = 221 MB` vs a 436 MB request), runs to completion
under `torch.compile` and produces a correct, sharp 61-frame clip. XLA does
whole-graph buffer assignment, so it does not strand memory in small chunks
the way the eager allocator does. Confirmed end to end:

| Frames | CFG | compiled total | warm per-step | output |
|---:|:---:|---:|---:|---|
| 31 | on | 589 s (warm) | 11–15 s | correct |
| 61 | on | 3059 s (cold, incl. ~25 min compile) | ~8 s | correct, sharp |
| 85 | on | ~2900 s (cold, incl. ~53 min compile) | ~12 s | correct, sharp |

Cold compile time grows with the graph: ~13 min at 31f, ~25 min at 61f,
~53 min at 85f. Warm per-step stays roughly flat (~8–15 s). Each geometry
pays the compile once, then the persisted NEFF cache amortises it.

Two honest caveats:

- Compiled warm per-step is in the same ballpark as eager (~6.3 s/step on the
  64-step run), sometimes slower. Compile's value for this model is **memory
  (buffer assignment), not throughput** — it is what makes >31 frames
  possible at all on a 24 GB/rank box, not what makes them fast.
- The 61-frame cold compile is ~25 min for a single graph. Persisted NEFF
  cache amortises that across runs of the same geometry, but each new
  (frames, CFG, tile) combination pays it once.

Eager mode, no `torch.compile`. Roughly 40% of wall clock is the CPU VAE
decode, as expected from the LTX-2 and Cosmos ports.

Sharding landed exactly as designed — the loader reports
`{replicate: 498, colwise: 288, rowwise: 190, glu: 95}`, and 95 GLU shards is
precisely 48 `ff` + 47 `ff_context` (block 47 has no context FFN).

Output quality across step counts, as a correctness signal:

| Config | Gradient energy | Std | Range | Levels |
|---|---:|---:|---|---:|
| 4 steps | 0.28 | 17.0 | 15–110 | 95 |
| 64 steps | **8.72** | 47.9 | 0–255 | 256 |

The 31× jump in gradient energy and the move to full dynamic range is what
distinguishes "undersampled" from "broken". A corrupted port does not
converge to detail as steps increase.

Notably, `--rope-bf16` was **not** needed: fp32 RoPE tables cross the compile
boundary fine here, so the LTX-2 fix #5/#8 failure mode did not reproduce.

## Why this port is tractable

Mochi's RoPE is real sin/cos arithmetic with no `torch.view_as_complex`, so
it needs none of the RoPE rewriting that Z-Image and FLUX.2-klein required.
`pipeline_mochi.py` has no data-dependent shapes. 24 heads and 8192/4096 FF
dims divide cleanly for TP ∈ {2,4,8}.

The QK norms are `[head_dim]`-shaped (`MochiRMSNorm(dim_head, ...)`), so they
stay replicated and valid on a sharded head axis. That removes LTX-2's
adaptive-QK-norm-with-all-reduce entirely.

## The four real problems, and what was done

**1. `torch.nonzero` in the attention processor.** Upstream strips prompt
padding with a value-dependent gather, which a compiled graph cannot
express. Replaced with a static path that keeps all 256 text tokens and
applies a `-10000.0` additive bias to padded text *key* columns. Verified
identical to upstream (`max|err| = 0.00e+00` on the visual stream) for a
padded prompt, a fully-masked prompt, and with RoPE active.

**2. Fused SwiGLU under column sharding — a silent-corruption trap.**
`ff.net.0.proj` is one `Linear(3072 → 16384)` whose output diffusers splits
at runtime as `[value | gate]`. A contiguous column shard makes the local
`chunk(2)` pair the wrong halves: rank 0 gets global rows `[0:2048]` with
`[2048:4096]` when it should get `[0:2048]` with `[8192:10240]`. Shapes all
check out and the model runs — the video is just wrong. Fixed with a
permuted shard; the test confirms the naive version is off by 0.48 while the
permuted one matches to 2e-07.

**3. Bool attention masks.** `MochiAttentionPool` (on device, inside
`time_embed`) passes a **bool** mask to SDPA. The LTX-2 BMM shim does
`scores = scores + attn_mask`, which for a bool tensor adds 1.0/0.0 to the
logits — wrong, with no error and no NaN. The shim now converts bool masks
to additive form first; the test shows the difference is 0.40.

**4. Quadratic attention memory.** Full 3D attention over 44,520 visual
tokens costs ~96 GB of bf16 score matrix across 24 heads (192 GB counting
`probs` alongside `scores`), doubled again by CFG batching. Added
tiled-query attention: each tile still attends to every key, so the softmax
is complete per tile and the result is **numerically exact**, not an
approximation. Memory drops from O(Sq·Sk) to O(q_chunk·Sk).

## Layout

```
Mochi/
├── README.md                     # this file
├── NOTES.md                      # verified architecture facts, memory tables, gotchas
├── requirements.txt
├── src/
│   ├── neuron_compat.py          # BMM-SDPA (bool-safe, tiled) + additive mask
│   ├── mochi_neuron_attention.py # static-shape processor (kills torch.nonzero)
│   ├── mochi_tp_plan.py          # TP plan, heads patch, RoPE precompute, sizing
│   ├── mochi_meta_loader.py      # sharded streaming loader + fused-GLU shard
│   └── run_mochi_native.py       # end-to-end TP runner
└── tests/
    └── test_offline.py           # 50 CPU-only checks, no Neuron needed
```

## Device runbook

Needs a **Trn2**. 20 GB of bf16 DiT weights exceed what the Trn1 cross-chip
TP ceiling (TP=2 max, per `native-pytorch-trn` steering) can host.

```bash
# 0. Beta-3 DLC, privileged for /dev/neuron*
aws ecr get-login-password --region us-east-1 \
  | sudo docker login --username AWS --password-stdin \
      421672808698.dkr.ecr.us-east-1.amazonaws.com
sudo docker pull 421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest
sudo docker run -it --privileged \
  -v /home/ubuntu/Mochi:/work \
  421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest bash

# 1. Re-run the offline suite inside the container (catches version drift)
cd /work && python tests/test_offline.py

# 2. Pre-stage weights (~30 GB: 20 GB bf16 DiT + 9.5 GB T5 + 1.8 GB VAE)
python src/run_mochi_native.py --download-only

# 3. Smallest possible bring-up: 19 frames, 4 steps, no CFG, eager, TP=4
NEURON_RT_NUM_CORES=4 TORCH_NEURONX_ENABLE_HOST_CC=1 \
TORCH_NEURONX_ENABLE_ASYNC_NRT=1 \
torchrun --nnodes 1 --nproc_per_node 4 \
  --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
  src/run_mochi_native.py --num-frames 19 --num-steps 4 --guidance-scale 1.0

# 4. Then scale up: CFG on, more frames/steps
... src/run_mochi_native.py --num-frames 31 --num-steps 16 --guidance-scale 4.5

# 5. Then compile for throughput (cold call adds 10-30 min of NEFF build)
... src/run_mochi_native.py --num-frames 31 --num-steps 16 --compile
```

`TORCH_NEURONX_ENABLE_HOST_CC=1` is mandatory — without host collective
communication the all-reduce tries the OFI/EFA device path and hangs on the
barrier. The `aws-ofi-nccl initialization failed` warning is benign and
appears in working runs too.

Restart the container between TP runs; a crashed run leaves the Neuron
runtime in a state that breaks the next `init_process_group`.

### Component placement

| Component | Size (bf16) | Device |
|---|---:|---|
| `MochiTransformer3DModel` | 20.06 GB | **Neuron**, TP-sharded |
| `T5EncoderModel` (T5-XXL) | 9.5 GB | CPU |
| `AutoencoderKLMochi` | 1.84 GB | CPU |
| `FlowMatchEulerDiscreteScheduler` | — | CPU |

Execution device is deliberately **CPU**, with the device hop confined to
the transformer wrapper. `FlowMatchEulerDiscreteScheduler.step` resolves its
timestep index with `.nonzero()`, so leaving the scheduler on device buys a
data-dependent op for no benefit. Per-step latent round-trip is ~2 MB.

### Per-rank weight footprint

TP scales **sub-linearly** here: the AdaLN modulation layers
(`norm1.linear`, `norm1_context.linear`) are 2.705 B parameters — 27% of the
model — and they modulate unsharded hidden states, so sharding them would
need a per-block all-gather. They stay replicated.

| TP | GB/rank | vs naive 20/TP |
|---:|---:|---|
| 1 | 20.08 | — |
| 2 | 12.86 | 10.04 |
| 4 | 9.25 | 5.02 |
| 8 | 7.44 | 2.51 |

TP=4 is the recommended starting point: 9.25 GB of weights leaves roughly
15 GB of a ~24 GB core budget for activations.

## Problems actually hit on device, and their fixes

Both were environmental/sizing, not correctness. Recorded because they cost
real time.

**1. `torchrun` sets `OMP_NUM_THREADS=1`.** The first run finished denoising
in 29 s then sat in the CPU VAE decode for over 10 minutes, with all four
ranks pinned to a single thread each — 4 of 192 vCPUs. Fix: pass
`OMP_NUM_THREADS=48 MKL_NUM_THREADS=48` explicitly (192 vCPUs / 4 ranks).
Decode dropped to ~170 s.

Also worth knowing: every rank redundantly runs the CPU VAE decode, since
only the transformer is collective. Gating decode to rank 0 would free CPU
for it, and is the obvious next optimisation.

**2. Auto-tiling budgeted per-plane instead of per-tensor, and OOM'd.** The
original threshold looked at `Sq × Sk` alone and picked `q_chunk ≈ 6656` for
31-frame CFG — barely any tiling. The score tensor is
`(batch × heads, Sq, Sk)`, so with CFG batch 2 and 6 local heads that is 12
planes of 9796², i.e. 2.3 GB for `scores` and another 2.3 GB for `probs`.
Peak hit 23.86 GB against a 24 GB logical core and a 740 MB allocation
failed. `_resolve_q_chunk` now budgets the whole tensor to 256 MiB, including
plane count and element size, which picks `q_chunk = 1024` for that geometry.
Two new offline tests pin the behaviour.

Useful constants confirmed by `neuron-ls`: `logical-neuroncore-config: 2`,
16 devices × 96 GB, 4 logical cores per device, so **24 GB per logical core**
(`total_hbm=25769803776`).

**3. The real long-sequence constraint is the norms, not attention.** Pushing
past 31 frames kept failing on allocations that matched
`batch × tokens × 3072 × 4 bytes` exactly — 436,125,696 for 61-frame CFG and
550,281,216 for 163-frame no-CFG. That is **fp32**, and its source is
`MochiModulatedRMSNorm` / `MochiRMSNormZero`, which upcast the whole
`(B, S, 3072)` tensor:

```python
hidden_states = hidden_states.to(torch.float32)   # full-sequence fp32 copy
hidden_states = self.norm(hidden_states)
```

Each block does this four times (`norm1`, `norm2`, `norm3`, `norm4`, plus the
`_context` variants) — 382 such norms across the model. Attention tiling
cannot help because these tensors are outside attention.

`mochi_norm_memory.py` tiles the norms over the sequence axis, upcasting one
tile at a time. The arithmetic stays in fp32 so results are **bit-identical**
(0.00e+00 across every offline case), and the parameter names are preserved
so the loader and TP plan are unaffected.

That was necessary but not sufficient. See the envelope below.

## The collective / TP=8 story

The obvious "give each rank less to hold" move is more tensor-parallel ranks.
It does not work here, and understanding why took a topology probe
(`tools/collective_probe.py`, which inits a group and does one all_reduce so
you can sweep world sizes in seconds instead of 20-minute model runs).

**TP=8 fails the collective, TP=2/4/16 pass.** Measured on
`i-03a587c283fffb075`:

| TP | all_reduce | note |
|---:|---|---|
| 2 | OK (got 3.0) | single NeuronLink hop |
| 4 | OK (got 10.0) | ring of 4 — the reference config |
| 8 | **`no_hier no_mesh`** | does not tile the torus |
| 16 | OK (got 136.0) | the full 4×4 torus |

`neuron-ls` shows the 16 devices wired as a **4×4 NeuronLink torus** — each
device links to exactly four neighbours (device 0 → {12, 3, 4, 1}). The
collective layer builds an algorithm only for groups that tile that torus: a
link (2), a ring/row (4), or the whole 2D mesh (16). Eight ranks are neither
a ring nor a clean 2×4 sub-mesh of a 4×4 torus, so the runtime reports
"no_hier no_mesh" — no hierarchical and no mesh algorithm — and
`init_process_group` fails. This is independent of which cores you pick
(`NEURON_RT_VISIBLE_CORES=0-15` fails identically) and of the virtual core
size.

**Mochi's bind:** a valid Mochi TP degree must divide 24 heads and the
8192/4096 FF dims → {1, 2, 4, 8}. Intersect with torus-valid {2, 4, 16} and
you get **{2, 4}**. TP=4 is not a conservative choice, it is the maximum pure
head-parallelism this model can do on this topology. TP=16 is torus-valid but
would need the 24 heads padded to 32 (not implemented).
`validate_world_size` now rejects 8 up front with this explanation rather than
letting you discover it after a 20 GB weight load (`MOCHI_ALLOW_TP8=1`
overrides on differently-wired hardware).

**The deeper point: more TP would not lift the memory ceiling anyway.** The
tensor that OOMs at 61 frames eager is 436,125,696 bytes =
`2 (CFG) × 17,746 tokens × 3072 × 4` — the **full** inner dim in fp32, i.e.
the residual-stream norm activation. The TP plan shards q/k/v/ff, but
`RowwiseParallel` all-reduces each block output back to the full 3072-wide
residual stream, and the modulated RMS norms run on *that*. So the OOMing
tensor is **replicated across head-parallel ranks, not sharded** — TP=8 or
TP=16, even if the collective allowed them, would leave it exactly the same
size. Confirmed by the byte math (3072, not 3072/TP) and by the plan
(`to_out`/`ff.net.2` are Rowwise → full-width output).

### So how do you actually push past 61 frames?

Ranked by payoff vs effort, given the residual stream is the wall:

1. **Bigger logical cores — `logical-neuroncore-config: 4` → 48 GB/rank.**
   Simplest by far (no code change, ~2× the memory that is actually binding).
   Blocked *on this shared box only*: the driver is initialised at LNC=2 and
   two other teams' containers (`v4_native`, `vllm_ga`) are resident, so
   reloading it is off-limits. Needs a dedicated instance. This is the
   recommended next step.
2. **Sequence / context parallelism.** Shard the 44k-token sequence across
   ranks — this shards the residual stream and the norm activations, which
   head-TP cannot. It composes with the torus-valid TP=4 group and mirrors
   what Genmo's own repo does ("efficient context parallel implementation").
   The correct long-term answer; a real implementation effort.
3. **`torch.compile`** (done) — buffer assignment already took 31 → 61
   frames without any parallelism change.
4. **Smaller norm tile / lower-precision norm.** `MOCHI_NORM_TILE` is already
   wired; combined with compile it may squeeze a little further. Cheap to try.
5. **NKI flash-attention.** Removes the score-matrix scratch entirely, but
   that scratch is already tiled and is not the current binding constraint.

## Remaining failure modes and levers

0. **`NCC_IBTN020` int16 access-pattern overflow at 163 frames.** A compiler
   limit, not a memory or code issue. The 44,520-token sequence produces a
   reshape/transpose whose stride exceeds int16. Levers to try (each is a
   fresh ~1 h compile): break the offending reshape into smaller chunks so no
   single stride exceeds 32,767; or file it upstream with `XLA_HLO_DEBUG=1`
   output, which is the compiler team's suggested path. Not pursued here.
1. **OOM at high frame counts.** Force smaller tiles with `--q-chunk 512`,
   drop CFG via `--guidance-scale 1.0`, or move to TP=8 (halves both the
   local head count and the weight footprint).
2. **Output structured but wrong.** Try `--rope-bf16`. Did not occur in
   practice, but it remains the LTX-2 fix #5/#8 failure mode.
3. **`Failed to execute the device barrier`.** Stale runtime state from a
   previous run. `docker restart mochi` between runs; this is routine, not a
   bug.
4. **Compile time explodes with `--compile`.** Eager correctness is the gate;
   compile is the optimisation. Untried so far.

## Accuracy vs CPU fp32 reference

The DiT was checked directly against a CPU fp32 run at identical seed and
config (19f, 8 steps, guidance 1.0, seed 777), comparing the pre-VAE latents
so the VAE is out of the picture. Both runs use the *same* bf16 checkpoint
weights, so this isolates Neuron bf16 compute vs CPU fp32 compute.

Measured at two guidance settings (19f, seed 777), comparing pre-VAE latents:

| Config | cosine | rel L2 | frame PSNR | reading |
|---|---:|---:|---:|---|
| guidance 1.0 (no CFG), 8 steps | 0.99913 | 4.2% | 51.95 dB | faithful; pure bf16 rounding |
| guidance 4.5 (CFG), 12 steps | 0.98725 | 16.1% | 32.63 dB | CFG-amplified rounding, still faithful |

**The no-CFG number is the clean per-pass fidelity.** 4.2% relative L2,
cosine 0.9991, per-channel error a uniform 2.6–5.9% (not structured). A wrong
shard, RoPE, or dropped mask would collapse cosine and push error into the
tens of percent — this doesn't.

**The CFG number looks worse but isn't a regression.** Classifier-free
guidance computes `out = uncond + g·(text − uncond)`, which amplifies per-pass
bf16 rounding by up to `2g − 1` (≈8× at g=4.5). The divergence scales almost
linearly with guidance (4.2% → 16.1% for a 4.5× guidance increase) — exactly
the CFG-amplification signature, not the flat offset a bug would produce.
Confirmed perceptually: decoded frames match at **32.6 dB** and the
side-by-side (`results/cmp_cfg_sidebyside.png`) is two sharp, correct,
artifact-free chameleon eyes differing only in the finest scale texture.

Reproduce with `tests/compare_latents.py` (guidance-aware verdict) and
`tests/decode_and_compare.py`, against `results/lat_*_19f.pt` (no-CFG) and
`results/lat_*_cfg.pt` (CFG).

## Honest gaps

- **Accuracy measured at two guidance settings, one geometry (19f).** Verified
  at guidance 1.0 (8 steps) and 4.5 (12 steps); not swept across frame counts
  or step counts. The figures are point estimates, and the CFG number is
  understood as guidance-amplified rounding rather than bounded independently.
- **`torch.compile` untried.** All numbers above are eager. The 6.3 s/step is
  therefore an upper bound, not the achievable latency.
- **No H100 cost comparison**, unlike the LTX-2 and FLUX write-ups.
- **CPU VAE decode is ~40% of wall clock** and every rank duplicates it.
  Moving it to Neuron means checking `CogVideoXCausalConv3d` with
  `pad_mode="replicate"` and `repeat_interleave` upsampling, neither of which
  has been examined.
- The download pulled ~87 GB rather than ~30 GB: `ignore_patterns` excludes
  the fp32 transformer shards but not the root-level original-format
  `dit.safetensors` / `encoder.safetensors` / `decoder.safetensors` (~42 GB),
  which the diffusers pipeline never reads.

## License

Apache-2.0, matching the Mochi weights and the AWS Neuron contrib code this
derives from.
