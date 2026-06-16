# Sequence-Length Scaling — Honest Findings

Where the current setup actually works, and where it walls.

## TL;DR

| Workload | Current limit |
|---|---|
| Real long-content training (sequences mostly filled with real tokens) | **seq=1024** |
| Padded short content (seq=N but real tokens stay ~hundreds) | seq ≥ 32K (no observed wall) |

**The "32K validated" number is misleading on its own** — at those lengths the
toy data is being padded to fill the sequence, and many ops short-circuit on
padding. The honest customer-facing number is the real-content one.

## Method

Two scans against `Qwen/Qwen3.6-27B` LoRA SFT, FSDP=16, batch=1, bf16:

1. **Padded scan** — `full_sft_fsdp.py` with the original 3 short toy examples
   (~120 tokens each), padded to `max_length=SEQ`. Tests whether the FSDP /
   activation memory plumbing handles bigger tensors.
2. **Real-long scan** — `phase3_long_seq_scan.py` with each example built by
   concatenating a 114-token passage until the tokenized sequence reaches
   85% of `max_length`. Tests actual long-context compute.

5 steps per rung. Logged peak HBM, per-step time, loss trajectory.

## Data

### Padded scan (toy short data, padded to SEQ)

| SEQ | real tok / sample | result | warm s/step | first→last loss |
|---:|---:|:---:|---:|:---:|
| 500 | ~380 | ✅ | 49 | 3.91 → 0.32 |
| 2,048 | ~380 | ✅ | 37 | 3.91 → 1.59 |
| 4,096 | ~380 | ✅ | 37 | 3.91 → 1.64 |
| 8,192 | ~380 | ✅ | 37 | 3.91 → 1.57 |
| 16,384 | ~380 | ✅ | 38 | 3.91 → 1.62 |
| 32,768 | ~380 | ✅ | 38 | 3.91 → 1.51 |

Per-step time stays flat because actual content is short — most attention
work is over padding positions and gets skipped.

### Real-long scan (sequences actually filled with real tokens)

| SEQ | real tok / sample | result | warm s/step | first→last loss |
|---:|---:|:---:|---:|:---:|
| 1,024 | 791 | ✅ | 162 | 0.594 → 0.076 |
| 2,048 | 1,695 | ❌ OOM | — | — |
| 4,096 | 3,390 | ❌ OOM | — | — |

OOM is `neuron::alloc::lazy NRT_RESOURCE` on a ~2.5 GB allocation when peak
HBM hits ~18.8 GB / 25.7 GB. Fragmentation contributes — `largest_cached_free`
sits just under the requested size at the moment of failure.

## Why the toy-data scan misled

The trainer pads short text up to `max_length=SEQ`, but `attn_implementation=
"eager"` honors the attention mask, so most of the padded positions don't
generate meaningful work for full attention. Activation tensors are allocated
at full SEQ, but the **active** working set is small. So the FSDP / Neuron
plumbing handled large allocations, while the actual long-context compute
path was never exercised.

The real-content scan stresses the actual long-context path (real Q/K/V
matmuls over real tokens, real grad propagation through the sequence).

## What walls at seq ≥ 2K with real content

- 64 transformer layers on a 27 B model = lots of layer activations even
  with FSDP sharding the params.
- `fsdp_activation_checkpointing: true` is already on — every `Qwen3_5DecoderLayer`
  is recomputed in backward — so the ceiling is the per-layer peak during
  recompute, not the sum of all boundary activations.
- One ~2.5 GB allocation is the killer. Likely candidates: a full
  attention-score buffer in some intermediate fp32 cast inside the GQA layers,
  or a fp32 working buffer the compiler picks for a cast/transpose.

## Mitigations (NOT yet implemented)

In rough order of effort:

1. **Cast attention working buffers to bf16 instead of fp32.** Some HF eager
   attention paths do an internal fp32 cast for stability. For LoRA SFT, bf16
   attention is usually fine.
2. **More aggressive activation recompute.** Today: per-decoder-layer. Could
   recompute at sub-layer granularity (q_proj/k_proj/v_proj separately,
   attention separately, MLP separately). Recovers ~50% activation memory at
   the cost of ~30 % step time.
3. **Sequence parallelism on top of FSDP.** Shard the activation along the
   seq dim across the 16 cores. Would 16× the addressable real-content seq
   length but is real engineering work — needs the model class to be
   sequence-aware.
4. **Drop FSDP ranks, lift TP=2 inside the parallelism config.** Current is
   FSDP=16 / TP=1; FSDP=8 / TP=2 splits each layer across 2 cores → ~half
   the per-rank activation. Doesn't help if model parameters were the bottleneck,
   but here activations are, so it should help.

## Customer-facing line

> Native PyTorch + TRL `SFTTrainer` + LoRA + FSDP=16 fine-tuning of
> Qwen3.6-27B is validated end-to-end on Trainium2 today at seq=1024 with
> real long-content data. The training loop converges (loss 0.59 → 0.08 in
> 5 steps), runs at ~162s/step warm. Going to longer sequences (>2K real
> content) is the next engineering gate — the levers are bf16 attention
> casts, finer-grained activation recompute, or sequence parallelism. We
> can target the customer's actual sequence-length need from there.

## Reproduce

```bash
# Real-long scan (in-container):
sudo docker exec neuron_grpo bash -lc 'bash /mnt/data/launch_long_sft.sh'
# Edit SEQ in launch_long_sft.sh to scan.
```

The scan script is `src/phase3_long_seq_scan.py`. The launcher is
`src/launch_long_sft.sh` (same FSDP config as the validated full SFT,
just points at the long-seq entry script and writes its own log).
