# Gemma 4 E4B-it on Trainium2

Google's [gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) (~7B params) served on Trainium2 via vLLM-Neuron.

E4B is a Gemma4-family model with Per-Layer Embeddings (PLE), heterogeneous attention (head_dim=256 SWA / 512 global), QK-norm, and KV-sharing.

| Path | Status | TP | Output Quality | Notes |
|---|---|---|---|---|
| vLLM-Neuron | **Working (TP=2)** | 2 | Partial English | PLE enabled, scale=1.0 |
| Native PyTorch | Stub | — | — | From prior PR #8 |

```
              ┌─────────────────────────────┐
              │ gemma-4-E4B-it on Trainium2 │
              └──────────────┬──────────────┘
                             │
          ┌──────────────────┴──────────────────┐
          │                                     │
"vLLM-Neuron serving"                  "Native PyTorch standalone"
          │                                     │
          ▼                                     ▼
    vllm-neuron/                         native-pytorch/
```

## Key Findings (2026-06-13)

1. **PLE is structurally required** — without it, ALL output is garbage regardless of hardware/TP
2. **TP=2 produces coherent English** — `num_kv_replicas=1` path is correct
3. **TP=4 produces garbage** — `num_kv_replicas=2` path has a bug in QKV weight replication
4. **v_norm stabilizes output** — E4B checkpoint has no v_norm weights but the parameter-free RMSNorm helps
5. **5 base patches required** (config None-handling, GeLU inline, force-torch MLP, cache wipe, transformers stub)

## Layout

```
gemma4-e4b/
├── README.md                    # this file
├── native-pytorch/              # stub (from prior PR #8)
│   └── README.md
└── vllm-neuron/                 # working path
    ├── README.md
    ├── src/
    │   ├── model_e4b_ple.py     # patched model.py with PLE (1657 lines)
    │   ├── config_e4b_ple.py    # config with E4B fields
    │   └── patches.md           # the 5 base patches
    └── results/
        └── output_samples.md    # actual model outputs
```

## Validation

- Date: 2026-06-13
- Instance: trn2.48xlarge (`i-0c2806a95b490e26e`, us-east-2)
- Container: vllm-neuron private-beta-trn10-v5
- TP=2, max_model_len=128, scale=1.0

## License

Model: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
