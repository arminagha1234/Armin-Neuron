# FLUX.2-klein-4B on AWS Trainium2

Two production-ready paths to run [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
(text-to-image and image-to-image, with optional LoRA fusion) on AWS
Trainium2. Pick the path that matches your serving need:

| Path | Best for | 1024² × 28 steps | Per-image | $/image (Trn2) | vs H100 ($0.0073) |
|---|---|---:|---:|---:|---:|
| [`native-pytorch/`](native-pytorch/) — single core | **Lowest latency standalone** | 65.9 s | 65.9 s | $0.041 | H100 5.6× cheaper |
| [`native-pytorch/`](native-pytorch/) — batch parallel | **Full-instance throughput** | 77 s for 2 imgs | 38.5 s aggregate | $0.024 | H100 3.3× cheaper |
| [`vllm-omni/`](vllm-omni/) | Multi-modal serving stack | extrapolated >800 s | ~150 s/step | extrapolated $4.7 | n/a |

(H100 baseline: single H100 GPU at $4.326/hr = $0.0073/image.
Trainium2: trn2.3xlarge at $2.23/hr.)

**Current state:** H100 wins on $/image for this workload due to the
10.8× per-step latency gap. Trainium2's value here is functional
validation of the native PyTorch stack (torch.compile on DiT models
works end-to-end) and throughput scaling via batch parallelism. The
cost story requires closing the per-step gap via split-aware TP=2 and
future compiler improvements. See
[`BENCHMARK_VS_H100.md`](native-pytorch/BENCHMARK_VS_H100.md) for
the full analysis and break-even math.

## TL;DR — which path do I want?

```
                    ┌─────────────────────────────────┐
                    │ FLUX.2-klein-4B on Trainium2    │
                    └────────────────┬────────────────┘
                                     │
              ┌──────────────────────┴──────────────────────┐
              │                                             │
   "Just FLUX.2-klein, fastest"                "FLUX inside multi-modal omni"
              │                                             │
              ▼                                             ▼
       native-pytorch/                                vllm-omni/
       single core or batch-parallel                  needs omni engine
       65.9s single / 38.5s batch                     shared scheduler/KV
       $0.041 single / $0.024 batch                   + other modalities
       (H100 still cheaper today;
        working to close the gap)
```

## Highlights

### Native PyTorch path
- **65.9 s** for 28-step 1024×1024 with `torch.compile(backend="neuron")` (single core)
- **38.5 s/image aggregate** with batch parallelism (2 procs × LNC=2 on 4-core trn2.3xl)
- H100 single GPU at $4.326/hr is still **3.3-5.6× cheaper per image** at current latency gap
- Functional validation: native PyTorch + torch.compile works end-to-end on a 4B DiT
- Single Trainium2 logical core (no TP needed; ~24 GB user budget; 4B model uses ~8 GB)
- LoRA support via `pipe.fuse_lora()` before compile
- Beta 3 stack (torch 2.11, torch_neuronx 2.11.3, neuronxcc 2.25)

### vLLM-Omni path
- Working end-to-end as of 2026-06-13 (NEFF compiles, denoising loop
  runs, latents unpack, VAE decodes, real PNGs produced)
- **18.44 s/step** at 256×256, **36.35 s/step** at 512×512 (TP=1)
- Auto-registered via `PIPELINE_REGISTRY` — drops into the
  vllm-omni-neuron plugin's `diffusion/models/` folder
- 8-fix sequence (RoPE real-arithmetic, fp32 pinning, contiguity,
  scheduler bf16, VAE-on-CPU, unpack-on-CPU)
- Mirror of the LTX-2 omni pattern — same architecture, same
  customizations

## Repository layout

```
flux2-klein-4b/
├── README.md                          # this file (path picker)
├── native-pytorch/                    # the recommended standalone path
│   ├── README.md
│   ├── BENCHMARK_VS_H100.md
│   ├── src/
│   │   ├── neuron_flux2_klein_native.py    # 10-patch pipeline subclass
│   │   ├── run_flux2_klein_native.py       # single-core CLI runner
│   │   └── run_batch_parallel.py           # parallel-launch CLI (LNC=2 × 2 procs)
│   └── results/
│       ├── flux_example1_neuron.png        # eager 1024² output
│       ├── flux_compiled_cached.png        # compiled 1024² output
│       ├── flux_batch_core0.png            # batch parallel, core 0
│       └── flux_batch_core1.png            # batch parallel, core 1
└── vllm-omni/                         # for multi-modal omni serving
    ├── README.md
    ├── BENCHMARK.md
    ├── src/
    │   ├── neuron_flux2_klein_pipeline.py  # vllm-omni pipeline subclass
    │   ├── run_flux2_klein_omni.py         # Omni runner
    │   ├── flux2_klein_stage.yaml          # stage config
    │   ├── upstream_patch.py               # real-arithmetic RoPE patch
    │   ├── patch_unpack.py                 # CPU unpack patch
    │   └── merge_lora.py                   # offline LoRA fuse tool
    └── results/
        ├── flux2_klein_256x256.png         # 256² / 4-step output
        └── flux2_klein_512x512.png         # 512² / 8-step output
```

## Validation

Both paths validated 2026-06-13 on AWS Trainium2.

- Native PyTorch: trn2.3xlarge `i-0cf5d3577220d6091` (ap-southeast-4),
  Beta 3 DLC.
- vLLM-Omni: trn2.48xlarge container `vllm_omni`, vllm-omni 0.19.0rc1
  + vllm-omni-neuron plugin.

Both paths produce visually-identical outputs given the same prompt +
seed (verified within bf16 precision against a CPU reference).

## License

Apache-2.0 (contrib code in this folder). Model weights:
[FLUX.2 Community License](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).
LoRA license: see the LoRA's HF repo.
