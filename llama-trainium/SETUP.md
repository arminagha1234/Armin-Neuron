# Setup — Native PyTorch (TorchNeuron) on Trainium

These examples run inside the **TorchNeuron** container (native PyTorch backend,
no XLA). TorchNeuron is a **closed beta** — request access and the container
image through your AWS account team. Official overview:
https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/pytorch-native-overview.html

## 1. Launch a Trainium instance

Any Trn1 / Trn2 / Inf2 instance on an Ubuntu AMI with Docker works. The examples
here were developed on a `trn1.32xlarge` (16 Trainium chips, 32 NeuronCores).
For the single-core Llama-1 example a `trn1.2xlarge` (1 chip) is enough.

Confirm your AWS CLI is configured:
```bash
aws sts get-caller-identity
```

## 2. Get the TorchNeuron container

Pull the TorchNeuron beta image you were granted (image URI comes from your AWS
account team — it is not public), then run it with the Neuron devices exposed:

```bash
# Log in to the beta ECR registry (URI provided by your account team)
aws ecr get-login-password --region us-east-1 \
  | sudo docker login --username AWS --password-stdin <BETA_REGISTRY>

sudo docker pull <BETA_IMAGE>:latest

# Run interactively with the Neuron devices attached
sudo docker run -it --privileged <BETA_IMAGE>:latest bash
```

The current beta image ships the whole native stack pre-installed
(`torch`, `torch-neuronx`, `neuronx-cc`, `nki`) — no wheel/venv setup needed.

## 3. Verify the device is visible

Inside the container:
```bash
neuron-ls        # should list your NeuronCores
```

Then confirm native PyTorch can run an op on the device (expect `16.0`):
```python
import torch, torch_neuronx
d = torch.device("neuron")
x = torch.ones(8, device=d)
print((x + x).sum().item())
```

On Trn1 you'll see a benign warning that asynchronous IO requires Trn2 — the op
still runs correctly (it falls back to synchronous IO).

## 4. Install example dependencies (HF models)

```bash
pip install transformers
```

For gated models (Llama-3.1-8B), accept the license on Hugging Face and log in:
```bash
hf auth login   # paste your HF token
```

## 5. Run the examples

See the per-example READMEs:
- [`training-smoke-test/`](./training-smoke-test/) — verify training works
- [`llama-7b/`](./llama-7b/) — single-core Llama-1 7B inference
- [`llama-3.1-8b/`](./llama-3.1-8b/) — tensor-parallel Llama-3.1-8B (WIP)

## Notes

- **First run compiles a NEFF** (tens of seconds to a few minutes). A persistent
  cache makes later runs fast.
- Each Trn1 NeuronCore has **16 GB HBM** — a 7B model fits on one core, an 8B does
  not (see the top-level README's memory section).
- Do **not** commit tokens, credentials, or private image URIs to this repo.

## Multi-core (tensor parallelism)

Models bigger than one core's 16 GB HBM (e.g. an 8B or larger) must be sharded
across cores. Native TP uses `torch.distributed` with the `neuron` backend + HF
`tp_plan="auto"`. **Host collective communication is required** — without it the
all-reduce hangs:

```bash
NEURON_RT_NUM_CORES=<N> TORCH_NEURONX_ENABLE_HOST_CC=1 TORCH_NEURONX_ENABLE_ASYNC_NRT=1 \
  torchrun --nnodes 1 --nproc_per_node=<N> --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
  your_tp_script.py ...
```

Gotchas:
- **`TORCH_NEURONX_ENABLE_HOST_CC=1` is essential.** Without it, the intra-node
  collective tries the OFI/EFA device path (fails to init in-container) and hangs
  on the barrier. The `aws-ofi-nccl init failed / is EFA enabled?` warning is benign.
- **Restart the container between TP runs** — teardown leaves stale runtime state
  that breaks the next `init_process_group`.
- On **Trn1**, TP=2 (both cores on one chip) works; cross-chip TP (TP≥4) currently
  fails at the device-barrier init on the beta backend. Use **Trn2** for models
  that need TP≥4 (e.g. a 13B).

See [`llama-3.1-8b/`](./llama-3.1-8b/) for a validated 2-core example.
