# SETUP — Gemma4-31B on Trainium2 (public image)

## Prerequisites
- **trn2.48xlarge** instance (16 Neuron devices / 32 LNC2 cores) on a Neuron DLAMI.
- Docker with the Neuron runtime.
- Hugging Face access to the gated `google/gemma-4-31B` weights (`hf auth login`).

## 1. Pull the public image (no beta, no ECR allowlist)
```bash
docker pull public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.21.0.1.0.0-neuronx-py313-sdk2.31.0-ubuntu24.04
```

## 2. Start the container (mounts all 16 Neuron devices)
```bash
bash scripts/run_container.sh
```
This runs the image with `--device /dev/neuron0..15`, your model dir mounted at `/root/models`, and drops
you into a shell. See `scripts/run_container.sh` for the exact `docker run`.

## 3. Download weights + apply the text-only fix (REQUIRED)
```bash
hf download google/gemma-4-31B --local-dir /root/models/gemma-4-31b
python3 make_textonly.py          # -> /root/models/gemma-4-31b-text
bash install_public.sh            # register Gemma4ForCausalLM in the public plugin
```
`gemma-4-31B` ships as a multimodal checkpoint; `make_textonly.py` strips the vision config and emits a
`Gemma4ForCausalLM` text-only model dir that serves cleanly on the public plugin.

## Orphan-core note (important)
Neuron workers (`VLLM::Worker`) are NOT always reaped by `pkill "vllm serve"`. If a serve fails to
initialize the Neuron runtime ("Runtime could not be initialized"), orphan workers are holding the cores.
Reliable clear: **restart the container** (`docker restart <name>`). The sweep runner relaunches serves
between sizes — if you hit this mid-sweep, restart and resume from the last size.
