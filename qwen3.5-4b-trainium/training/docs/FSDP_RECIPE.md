# Qwen3.5-4B Native-Neuron Multi-Core FSDP Recipe (Trainium2 / box2)

Working recipe for multi-core FSDP LoRA training of Qwen3.5-4B in the NATIVE
PyTorch Neuron beta (`torch.device("neuron")`, NOT torch_xla) on trn2.48xlarge.

Script: `/work/qwen35_fsdp_bench.py` (container) == `/data/qwen35_fsdp_bench.py` (host).

## Key mechanics (reverse-engineered from /opt/torch-neuronx)

- The native stack registers a c10d ProcessGroup backend named **"neuron"** in
  `torch_neuronx/distributed/backend.py`. It is registered on import of
  `torch_neuronx.distributed`. You MUST do BOTH imports before init:
      import torch_neuronx
      import torch_neuronx.distributed          # <-- registers "neuron" PG backend + collectives
  Without the second import, init_process_group(backend="neuron") + collectives fail
  with `ENC ... no_mesh` (confirmed independently by the flux2_klein team).
- Init: `torch.distributed.init_process_group(backend="neuron")`. torchrun provides
  RANK / WORLD_SIZE / LOCAL_RANK / LOCAL_WORLD_SIZE / MASTER_ADDR / MASTER_PORT.
- The backend AUTO-MAPS each rank to a NeuronCore: inside init it reads LOCAL_RANK and
  sets `NEURON_RT_VISIBLE_CORES = <that core>` (one logical core per rank), sets up the
  NRT root comm id, pins NUMA. So you do NOT set NEURON_RT_VISIBLE_CORES yourself;
  just set NEURON_RT_NUM_CORES (or NEURON_RT_VISIBLE_CORES as a range) at the launcher.
- Flush / step boundary: the native beta runs EAGERLY (no xm.mark_step). Use
  `torch_neuronx.synchronize()` (== torch.neuron.synchronize()) to make sure kernels
  finished before timing.
- FSDP: upstream `torch.distributed.fsdp.FullyShardedDataParallel` works directly on
  neuron (device_id=torch.device("neuron")). Same idiom as gemma4_31b_training.

## Qwen3.5-4B specifics (HYBRID model — important)

- Model class `Qwen3_5ForCausalLM`; text_config has 32 decoder layers.
- Attention is HYBRID: 24 `linear_attention` layers (modules `in_proj_qkv/in_proj_z/
  in_proj_a/in_proj_b/out_proj`) + 8 `full_attention` layers (`q/k/v/o_proj`).
  Layer 0-2 are linear_attention, so `--layers 2` truncation exposes NO q/k/v/o_proj.
- LoRA target_modules MUST be filtered to modules that ACTUALLY EXIST in the (possibly
  truncated) model, else PEFT raises "Target modules not found". The script auto-detects
  present nn.Linear leaf names and intersects with the candidate list.
- After get_peft_model, cast the whole model to bf16 (`model.to(torch.bfloat16)`):
  PEFT creates adapters in fp32 while base is bf16, and FSDP's flat-param requires a
  UNIFORM dtype (else "Must flatten tensors with uniform dtype but got bf16 and fp32").
- FSDP auto_wrap_policy: wrap on ALL distinct decoder-layer classes found in the
  `...layers` ModuleList (hybrid model may have >1 class). Decoder class is
  `Qwen3_5DecoderLayer`.
- use_orig_params=True (needed so LoRA's mixed frozen/trainable params coexist in a flat param).

## WORKING launch (2-rank and 4-rank VALIDATED GREEN)

```bash
# inside container: sudo docker exec native_train bash -c '...'
cd /work
export NEURON_RT_VIRTUAL_CORE_SIZE=2         # LNC2 (2 physical cores fused per logical) -- REQUIRED for world>2 collectives
export NEURON_RT_NUM_CORES=4                  # 2*nproc? no -> = nproc (logical cores). Set == nproc_per_node.
export TORCH_NEURONX_ENABLE_HOST_CC=1
export TORCH_NEURONX_ENABLE_ASYNC_NRT=1
export TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=1
export NEURON_COMPILE_CACHE_URL=/work/neuron_compile_cache   # persist NEFFs on big /data disk
# NEFF /tmp guard (neuronx-cc hardcodes /tmp/neuron_backend, can fill root disk):
#   ln -sfn /work/neuron_backend /tmp/neuron_backend

torchrun --nproc_per_node 4 --rdzv_backend c10d --rdzv_endpoint localhost:29524 \
    /work/qwen35_fsdp_bench.py --lora --layers 4 --seq 128 --bs 1 --steps 4
```

Full 32-layer LoRA seq512 (matches single-core baseline seq/mode): drop `--layers`,
use `--seq 512 --steps 6`. First step compiles (~30-90s truncated, several min for full 32L).

## RESULTS (box2, trn2.48xlarge, LoRA, eager, bf16)

| ranks | config              | warm s/step | aggregate tok/s | loss (decreasing) |
|-------|---------------------|-------------|-----------------|-------------------|
| 1     | full-32L seq512 (baseline) | 6.16  | 83              | yes               |
| 2     | 2L seq128           | 0.827       | 309.5           | 12.888 -> 12.804  |
| 4     | 4L seq128           | 1.632       | 313.7           | 12.970 -> 12.900  |

2-rank and 4-rank FSDP are GREEN with decreasing loss. The distributed init, per-rank
core assignment, FSDP shard/allgather/reduce-scatter collectives, LoRA, and bf16 all work.

## BLOCKER at >=8 ranks

8-rank (and the earlier 8-rank attempt) fail at collective init:
`ENC:enc_init_comm ... failed (2) to init a collective algorithm. reason: no_hier no_mesh
replica-group: [0,1,2,3,4,5,6,7]` -> `RuntimeError: Failed to execute the device barrier 2`.
This is a collective-topology/mesh selection issue at world>=8 on this beta3 container
(NOT a model bug; 2 and 4 ranks work). The flux2_klein team also only ever validated 2
and 4 ranks on this stack. Tried (did not help): NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE
/ NEURON_RT_DBG_MESH_CHANNEL_BUFFER_SIZE = 256MB, NEURON_PLATFORM_TARGET_OVERRIDE=trn2.
NEXT to try: NEURON_SKIP_EFA_AFFINITY=1 + FI_PROVIDER=efa + explicit NEURON_RT_ROOT_COMM_ID
(the flux2 collective env), LNC1 (VIRTUAL_CORE_SIZE=1), or a 2D device mesh / FSDP2
fully_shard with an explicit init_device_mesh.

## Hard warnings baked in / to watch
- seq >= ~5000: eager-FSDP SDPA backward produces all-NaN grads (LeVo finding). The 8
  full_attn layers use SDPA -> keep FSDP validation at seq <= 2048.
- NEFF cache: point at /work (=/data, 1.6T free); /tmp/neuron_backend is hardcoded by
  neuronx-cc and can fill the root disk.
- NET/OFI aws-ofi-nccl init-failed warnings are NON-FATAL (intra-node transport fallback).

## CRITICAL REGIME FINDING (2026-07-24) — FSDP is for FULL-FT, not LoRA

Full-32L **LoRA** FSDP=4 seq512 measured: **17.08 s/step, 119.9 agg tok/s** (loss 14.16→13.30).
That is only **1.44× over single-core** (83 tok/s) — SUBLINEAR. Why: LoRA freezes the base (~3M
trainable). FULL_SHARD shards the FROZEN base and all-gathers it every forward = pure comm overhead,
with ~zero optimizer-state to shard (the thing FSDP saves). The 4L-smoke's 313 tok/s was misleading
(tiny model, comm hidden).

⟹ Regime split:
- **LoRA** → use **DDP / NO_SHARD** (replicate weights, all-reduce only the tiny adapter grads) +
  data-parallel for throughput. FULL_SHARD is the wrong tool. (bench currently hardcodes FULL_SHARD
  at line ~140 — change to ShardingStrategy.NO_SHARD for the LoRA throughput number.)
- **Full fine-tune** → FSDP FULL_SHARD is ESSENTIAL: 4.21B weights+grads+AdamW fp32 ≈ 50GB > 24GB/core,
  OOMs single-core; sharding optimizer state across ranks is what makes full-FT fit at all. THIS is
  where FSDP's ~linear scaling shows up. (Measuring full-FT FSDP=4 seq512 now → fsdp_fullft_4rank.log.)

The customer's choice (LoRA vs full-FT) picks the parallelism strategy. Need their recipe to quote the
real wall-clock.

## ✅ FULL-FT FSDP=4 RESULT (2026-07-24) — the capability unlock
Full fine-tune (ALL 4.21B params, no LoRA), FSDP=4, seq512, eager:
**26.83 s/step, 76.3 agg tok/s, loss 14.16 → 10.74** (steep decrease — real full-weight training).
- Single-core CANNOT run full-FT (OOM: 50GB > 24GB/core). FSDP sharding AdamW state across 4 ranks is
  what makes full fine-tuning FIT AND RUN. For full-FT, FSDP is the ENABLER, not just a speedup.
- Warm step 14s (vs 27s avg incl. the reduce-scatter-heavy early steps). Compile 233s cold.

## Two-regime summary table (box2, trn2.48xl, full-32L, eager, bf16, seq512)
| mode      | parallelism | s/step | agg tok/s | loss        | note |
|-----------|-------------|--------|-----------|-------------|------|
| LoRA      | single core | 6.16   | 83        | 14.1→14.0   | fits 1 core |
| LoRA      | FSDP=4      | 17.08  | 119.9     | 14.2→13.3   | FULL_SHARD wrong tool (shards frozen base); use NO_SHARD/DDP |
| full-FT   | single core | —      | OOM       | —           | 50GB>24GB, cannot run |
| full-FT   | FSDP=4      | 26.83  | 76.3      | 14.2→10.7   | FSDP = the enabler; near-linear headroom to 8/16 (blocked >=8) |

## LoRA NO_SHARD(DDP) result (2026-07-24) — correct LoRA multi-core tool
LoRA FSDP-container NO_SHARD (=DDP: replicate + all-reduce adapter grads), world=4, seq512:
**13.64 s/step, 150.2 agg tok/s, loss 14.16 → 13.30.** Compile only 23.6s (vs 439s FULL_SHARD — no
base-weight sharding graph).
- Beats LoRA FULL_SHARD (119.9 tok/s) as predicted → NO_SHARD IS the right LoRA strategy. But still only
  ~1.8× single-core (83), and per-step 13.6s > single-core 6.16s. The aggregate gain is from 4× data/step;
  the elevated per-step time shows comm overhead layered on an already-slow GDN-fallback step.

## FINAL two-regime table (box2, trn2.48xl, full-32L, eager, bf16, seq512, world=4)
| mode    | parallelism        | s/step | agg tok/s | vs 1-core | loss      |
|---------|--------------------|--------|-----------|-----------|-----------|
| LoRA    | single core        | 6.16   | 83        | 1.0×      | 14.1→14.0 |
| LoRA    | FSDP=4 FULL_SHARD  | 17.08  | 119.9     | 1.44×     | 14.2→13.3 | (wrong tool)
| LoRA    | FSDP=4 NO_SHARD/DDP| 13.64  | 150.2     | 1.81×     | 14.2→13.3 | (right tool)
| full-FT | single core        | —      | OOM       | —         | —         | (50GB>24GB)
| full-FT | FSDP=4 FULL_SHARD  | 26.83  | 76.3      | enabler   | 14.2→10.7 | (FSDP = the enabler)

## KEY CONCLUSION: GDN kernel swap is the #1 lever (3 independent signals)
1. torch.compile FAILS on the GDN fallback (SBUF overflow / neuronx-cc vectorize error) at any depth.
2. GDN fallback graph is huge → ~7min cold compile, heavy per-step.
3. Multi-core per-step (13.6s) >> single-core (6.16s): comm overhead stacks on an already-slow GDN step.
Swapping GDN torch-fallback → clean chunked NKI kernel (fwd+bwd autograd.Function) cuts base step time,
unblocks torch.compile, and makes multi-core scaling clean. THEN scale DP/FSDP wider.

## >=8-rank blocker CONFIRMED (2026-07-24) — 2 independent attempts failed
- Plain FSDP=8: failed at collective init (no_hier no_mesh → device barrier failure).
- FSDP=8 + flux2 EFA env (NEURON_SKIP_EFA_AFFINITY=1 FI_PROVIDER=efa): ALSO failed, root cause
  "NET/OFI Failed to initialize rdma protocol" on all ranks → mesh barrier fails.
- VERDICT: genuine platform/collective-topology limit at world>=8 on THIS beta3 container. 2 and 4
  ranks are solid. Defer to Beta 4 (newer build may fix the collective mesh selection). Do NOT keep
  brute-forcing container collective flags — diminishing returns. Next real options: newer DLC, LNC1,
  or FSDP2 fully_shard with explicit init_device_mesh (2D mesh).
