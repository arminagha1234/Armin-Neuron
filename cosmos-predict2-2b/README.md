# NVIDIA Cosmos-Predict2-2B on Trainium2

NVIDIA's Cosmos World Foundation Model (Predict2 2B variant) running on
AWS Trainium2, native PyTorch, Beta 3 stack. Both **Text-to-Image** and
**Image-to-Video (Video2World)** generation work end-to-end with
**numerically correct output** (CPU-reference-matching std), no TP yet.

## Path picker

| Path | Resolution × steps | Warm | Notes |
|---|---:|---:|---|
| **native-pytorch** (Text2Image) | 1024² × 20 | ~42 s | DiT on Neuron, T5+VAE on CPU |
| **native-pytorch** (Video2World) | 480×832 × 25f × 12 | **245 s** | DiT 142 s, CPU side 102 s |
| **vllm-omni** | — | — | WIP — see subfolder |

```
                     ┌──────────────────────────────────┐
                     │  Cosmos-Predict2-2B on Trainium2 │
                     └────────────────┬─────────────────┘
                                      │
                ┌─────────────────────┴─────────────────────┐
                │                                           │
       "Lowest-latency standalone"               "Multi-modal omni serving"
                │                                           │
                ▼                                           ▼
         native-pytorch/                              vllm-omni/
```

## Layout

```
cosmos-predict2-2b/
├── README.md                       # this file
├── native-pytorch/
│   ├── README.md                   # status, repro, files, known issues
│   ├── BENCHMARK.md                # numbers + breakdown
│   ├── src/
│   │   ├── cosmos_cpu_smoke.py     # CPU reference + porting shims
│   │   ├── cosmos_neuron.py        # Text2Image runner (DiT on neuron)
│   │   └── cosmos_video_neuron.py  # Video2World runner (DiT on neuron)
│   └── results/
│       ├── cosmos_t2i_1024.png
│       ├── cosmos_t2i_512_cpu_reference.png
│       └── cosmos_video_480x832_25f.mp4
└── vllm-omni/
    ├── README.md                   # WIP stub
    └── ...
```

## What works today

- **Text2Image:** 512² (~16 s warm) and 1024² (~42 s warm), output
  numerically matches the CPU reference (std 76.25 on Neuron vs 76.61 on
  CPU at 512² / 12 steps).
- **Video2World:** 256² × 17f (~22 s warm), 480×832 × 17f (~131 s warm),
  **480×832 × 25f (~245 s warm)** — real video clip, MP4 produced.
- Persistent NEFF cache works across runs.

## Architecture (per the model_index.json)

| Component | Class | Where it runs |
|---|---|---|
| Text encoder | `T5EncoderModel` | CPU |
| Tokenizer | `T5TokenizerFast` | CPU |
| Transformer (DiT) | `CosmosTransformer3DModel` | **Neuron** |
| VAE | `AutoencoderKLWan` (the WAN VAE) | CPU |
| Scheduler | `FlowMatchEulerDiscreteScheduler` | — |

The DiT is the hot path; T5 and the WAN VAE stay on CPU for now. Moving
the VAE onto Neuron is the obvious next optimization for video — the
480×832 × 25f benchmark spends ~102 s on the CPU side (T5 + VAE decode)
versus ~142 s on the Neuron DiT.

## Validation

trn2.48xlarge (us-east-2), Beta 3 DLC, native PyTorch + `torch_neuronx`,
`torch.device("neuron")`, 2026-06-17.

## License

Apache-2.0 (matches the repo). Cosmos model weights are subject to
NVIDIA's license terms — accept the license on the Hugging Face model
page before downloading.
