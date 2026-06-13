# LTX-2 on AWS Trainium2

Two paths to run [Lightricks/LTX-2](https://huggingface.co/Lightricks/LTX-2)
(18.88B audio-video diffusion transformer) on AWS Trainium2. Pick the path
that matches your serving need:

| Path | Best for | 384×512 / 25f / 8 steps (warm) | Per-step | vs H100 |
|---|---|---:|---:|---:|
| [`native-pytorch/`](native-pytorch/) | **Lowest latency standalone** | 165.4 s | 20.68 s | 19.4× per-step gap (CPU flat tax) |
| [`vllm-omni/`](vllm-omni/) | Multi-modal omni serving | — (WIP) | — | — |

(Cost based on trn2.48xlarge $35.76/hr; H100 baseline 2.84 s/clip on
p5.48xlarge single GPU.)

The native PyTorch path is validated end-to-end with correct (sharp) output.
The 58× warm gap is dominated by CPU flat tax (Gemma-3 text encoder + VAE
on CPU: ~115 s of 165 s). Moving those onto Neuron is the primary
optimization lever — see the roadmap in
[`native-pytorch/BENCHMARK_VS_H100.md`](native-pytorch/BENCHMARK_VS_H100.md).

## TL;DR — which path do I want?

```
                ┌─────────────────────────────────┐
                │ LTX-2 18.88B on Trainium2       │
                └────────────────┬────────────────┘
                                 │
          ┌──────────────────────┴──────────────────────┐
          │                                             │
"Just LTX-2, fastest"                    "LTX inside multi-modal omni"
          │                                             │
          ▼                                             ▼
   native-pytorch/                                vllm-omni/
   TP=4, torchrun                                 needs omni engine
   165.4s warm (CPU-bottlenecked)                 WIP / blocked
   validated, sharp output ✅                      container contention ⚠️
```

## Highlights

### Native PyTorch path
- **165.4 s warm** for 8-step 384×512 25-frame generation (n=5, σ=0.74)
- **6.33 s/step** transformer-only latency on Neuron (TP=4, ~10 GB/rank)
- **Correct output** validated against CPU reference: sharpness ratio 6.4×
  (Neuron sharper than 4-step CPU; at 8 steps both converge)
- **Ten Neuron correctness fixes** (the production recipe from the
  [AWS Neuron LTX-2 contrib](https://github.com/aws-neuron/neuronx-distributed-inference/tree/main/contrib/models/ltx2-video-audio)):
  BMM-SDPA, RankTensor RoPE, adaptive QK-norm, additive -10000 mask, bf16
  cast, meta-init build, TP plan covering all 6 attention paths, etc.
- Beta 3 stack (torch 2.11, torch_neuronx 2.11.3, `torch.device("neuron")`)

### vLLM-Omni path
- Working pipeline load + TP=4 sharding
- **Blocked**: stock `F.scaled_dot_product_attention` on Neuron miscomputes
  (confirmed root cause: BMM-SDPA fix resolves); additionally the omni
  container's shared-memory diffusion-stage dispatcher conflicts with other
  model benchmarks in the same container
- See [`vllm-omni/README.md`](vllm-omni/README.md) for status + blockers

## Repository layout

```
ltx2/
├── README.md                          # this file (path picker)
├── native-pytorch/                    # the validated standalone path
│   ├── README.md
│   ├── BENCHMARK_VS_H100.md
│   ├── src/
│   │   ├── neuron_compat.py           # BMM-SDPA + RankTensor + mask conversion
│   │   ├── run_ltx2_native.py         # TP=4 end-to-end runner
│   │   ├── ltx2_tp_plan.py            # TP plan + attn fixes + QK-norm + RoPE
│   │   └── ltx2_meta_loader.py        # sharded weight loader
│   └── results/
│       ├── ltx2_bmmsdpa_test.png      # frame 0 (384×512)
│       └── ltx2_bmmsdpa_test.mp4      # 25-frame video @ 24fps
└── vllm-omni/                         # WIP — blocked
    └── README.md                      # status + blockers
```

## Validation

- **Date:** 2026-06-13
- **Instance:** trn2.48xlarge (us-east-2, `i-0c2806a95b490e26e`)
- **Stack:** Beta 3 DLC (`concourse-release-0461d3b:latest`), torch 2.11.0,
  torch_neuronx 2.11.3.0.1278
- **Correctness:** Neuron output matches CPU reference to within bf16 rounding
  after all 10 Neuron compatibility fixes (most critically: BMM-SDPA replacing
  stock `F.scaled_dot_product_attention`). See
  [`neuron/examples/LTX/BLUR_INVESTIGATION.md`](../../neuron/examples/LTX/BLUR_INVESTIGATION.md)
  for the full root-cause analysis.

## License

Apache-2.0 (contrib code). Model weights:
[Lightricks License](https://huggingface.co/Lightricks/LTX-2).
