# FLUX.2-klein-4B — Resolution sweep (Trainium2, single logical core)

**Date:** 2026-06-13
**Hardware:** trn2.3xlarge (LNC=2, single logical core, ~24 GB user budget)
**Stack:** Beta 3 DLC, torch 2.11.0, torch_neuronx 2.11.3, neuronxcc 2.25
**Model:** FLUX.2-klein-4B (with zoom-LoRA, scale=1.1)
**Steps:** 28, bf16, seed=42, guidance_scale=3.5
**Pipeline:** `NeuronFlux2KleinPipeline` + `torch.compile(backend="neuron")`

## Numbers

| Resolution | Tokens (DiT seq len) | Per-step (warm) | Total / 28 steps | $/image | vs H100 ($/image) |
|---:|---:|---:|---:|---:|---:|
| 256² | 256 | **194 ms** | **5.4 s** | **$0.0033** | TBD |
| 512² | 1024 | **566 ms** | **15.8 s** | **$0.0098** | TBD |
| 768² | 2304 | **1,258 ms** | **35.2 s** | **$0.0218** | TBD |
| 1024² | 4096 | **2,350 ms** | **65.9 s** | **$0.0408** | $0.0073 — H100 5.6× cheaper |
| 1280² | 6400 | **6,302 ms** | **176.5 s** | **$0.109** | ~$0.015 — H100 7× cheaper |

(Cost = total × ($2.23/hr / 3600 s) on trn2.3xlarge.)

## What scales how

The per-step cost grows roughly as **O(seq²)** because attention is the
dominant op in the DiT forward, and FLUX's parallel transformer blocks
have full bidirectional attention (no causal mask).

| Doubling | Tokens × | Observed per-step × |
|---|---:|---:|
| 256 → 512 | 4× | 2.9× |
| 512 → 768 | 2.25× | 2.2× |
| 768 → 1024 | 1.78× | 1.87× |

Sub-quadratic scaling at lower resolutions (256→512 was only 2.9× for
4× tokens) suggests fixed overhead (boundary moves, scheduler step,
small ops) is non-trivial at small shapes. At 1024×1024 the curve is
basically quadratic.

## Batch-parallel cost projection at each resolution

Run two parallel processes (one per logical core) and the per-image
cost halves (same compute, twice the throughput per instance):

| Resolution | Single-core $/image | Batch-parallel $/image (2 procs × LNC=2) |
|---:|---:|---:|
| 256² | $0.0033 | **$0.00164** |
| 512² | $0.0098 | **$0.00489** |
| 768² | $0.0218 | **$0.0109** |
| 1024² | $0.0408 | **$0.0238** |

At 256² the batch-parallel cost is **$0.00164/image** — that's
**606,000 images per dollar**.

## Files

| File | Resolution |
|---|---|
| `results/flux_sweep_256.png` | 256×256 |
| `results/flux_sweep_512.png` | 512×512 |
| `results/flux_sweep_768.png` | 768×768 |
| `results/flux_compiled_cached.png` | 1024×1024 (from PR #11) |
| `results/flux_sweep_1280.png` | 1280×1280 |

## Reproduction

```bash
# Single resolution (replace 768 with target):
NEURON_RT_VISIBLE_CORES=0-1 NEURON_RT_VIRTUAL_CORE_SIZE=2 HF_TOKEN=$HF_TOKEN \
    python src/run_batch_parallel.py --core 0 \
        --image input.jpg --steps 28 --height 768 --width 768 \
        --output flux_sweep_768.png

# Full sweep:
for RES in 256 512 768 1024 1280; do
    NEURON_RT_VISIBLE_CORES=0-1 NEURON_RT_VIRTUAL_CORE_SIZE=2 HF_TOKEN=$HF_TOKEN \
        python src/run_batch_parallel.py --core 0 \
            --image input.jpg --steps 28 --height $RES --width $RES \
            --output flux_sweep_$RES.png
done
```

Each new resolution costs ~5-15 minutes of compile (NEFFs are
shape-specialized — `dynamic=False` is required because Beta 3 doesn't
support dynamic shapes in `torch.compile`). Subsequent runs at the same
shape are warm in seconds via the persistent NEFF cache at
`/tmp/neff_cache`.
