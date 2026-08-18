# Environment Setup

Two environments. The **core** runs anywhere (laptop, CI). The **backend**
runs on a Trainium instance with the Beta 3 native-PyTorch DLC.

## 1. Core (no hardware) — run the harness + tests

```bash
cd implementation
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd src && python -m pytest -q          # expect 31 passed
```

That is enough to run the mock demos and develop backend-independent logic.

## 2. Backend — Beta 3 native PyTorch on Trainium

**Hard rule: Beta 3, never Beta 2.** Full detail in
`internal Neuron Beta 3 setup docs`. This is the authoritative source; the summary
below is for convenience.

### 2a. Pull the Beta 3 DLC

```bash
# ECR login (cross-account). Works with an EC2 Neuron instance role or AWS creds.
aws ecr get-login-password --region us-east-1 | sudo docker login \
    --username AWS --password-stdin \
    <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

sudo docker pull <BETA3_NEURON_DLC_IMAGE>
```

### 2b. Extract artifacts + install the Beta 3 driver

The public DLAMI driver is **incompatible** with Beta 3 — you must install the
one shipped in the image.

```bash
imageID=$(sudo docker images -q --filter \
    reference=<BETA3_NEURON_DLC_IMAGE>)
cd $HOME
sudo docker create --name tmp $imageID && sudo docker cp tmp:/workspace . && sudo docker rm tmp

sudo apt-get update && sudo apt-get install -y dkms build-essential
sudo dpkg -i $HOME/workspace/runtime_artifacts/*.deb

neuron-ls   # trn2.48xl: expect 16 devices x 4 cores = 64 total
```

### 2c. Either run in the container, or a host venv

```bash
# Option A — interactive container
sudo docker run -it --privileged $imageID /bin/bash

# Option B — host venv
cd $HOME/workspace && sudo apt install -y python3.12-venv
python3.12 -m venv native_venv && source native_venv/bin/activate
pip install uv
uv pip install $HOME/workspace/nki_wheels/nki-0.4.0*-cp312-cp312-linux_x86_64.whl
uv pip install $HOME/workspace/neuronx_cc_wheels/neuronx_cc-2.*-cp312-cp312-linux_x86_64.whl
cd $HOME/workspace/torch_neuron_eager && uv pip install -e .[dev]
```

### 2d. Verify native PyTorch works (expect 16.0)

```python
import torch, torch_neuronx
d = torch.device("neuron")            # Beta 3 device string, NOT privateuseone
x = torch.ones(8, device=d)
print((x + x).sum().item())           # -> 16.0
```

## 3. The TP=8 gate — RUN THIS BEFORE finishing the backend

Decides whether native PyTorch can be the primary backend at all. Cross-chip
TP is documented failing on Trn1; Trn2 is untested. Seed models need TP=8.

Save as `tp8_smoke.py`:

```python
import os, sys, torch, torch.distributed as dist
from transformers import AutoModelForCausalLM

dist.init_process_group(backend="neuron")   # per beta3-only PG backend note
dev = torch.device("neuron")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-31B", dtype=torch.bfloat16,
    attn_implementation="eager", tp_plan="auto",
).to(dev)
ids = torch.randint(0, 32000, (1, 128), device=dev)
out = model(ids)                            # one forward at TP=8
print("TP=8 forward OK, logits:", tuple(out.logits.shape))
sys.stdout.flush(); os._exit(0)             # clean exit; teardown can SIGSEGV
```

Run it:

```bash
NEURON_RT_NUM_CORES=8 TORCH_NEURONX_ENABLE_HOST_CC=1 TORCH_NEURONX_ENABLE_ASYNC_NRT=1 \
  torchrun --nnodes 1 --nproc_per_node=8 \
  --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
  tp8_smoke.py
```

- **Prints the logits shape** → TP=8 works. Native PyTorch is viable. Proceed.
- **`Failed to execute the device barrier 1`** → cross-chip TP is broken here
  too. STOP, report, and switch the plan to a vLLM-Neuron backend. Do not
  build a backend that cannot load the seed models.

## 4. Where the pieces from Downloads fit

The user's `~/Downloads` has related material. Note for whoever runs this:
- `Native PyTorch User Guide - Beta 3 (5_15_26).pdf` — the authoritative user
  guide. Read it if the steering summary is not enough.
- `torch_neuronx-2.11.3.0.19138+...whl` — a native PyTorch wheel (a build newer
  than the steering's `.1254`; prefer the DLC's bundled version unless you have
  a reason to override).
- `internal-prior-optimization-run.zip` — the reference
  implementation that hit a large (multiple-x). Worth extracting for its `program.md` and
  kernel patterns.

**NOTE:** the user referenced a Downloads folder "neuron beta" for DLC
instructions. No folder by that exact name was found — the authoritative
instructions used here are from `internal Neuron Beta 3 setup docs`. If the user has
a different/newer beta image in mind, confirm the ECR URI before pulling.

## Gotchas (from internal Beta 3 setup docs)

- `dynamic=True` in `torch.compile` raises → use static shapes / bucketing.
- float64 → silently fp32; int64 → silently int32. Cast explicitly.
- float8 (e4m3/e5m2) not in Beta 3 → the fp8 config axes are XLA-backend only
  for now.
- Restart the container between TP runs — a crashed run leaves the runtime in a
  state that breaks the next `init_process_group`.
- Expect a benign teardown SIGSEGV after results print; `os._exit(0)` avoids it
  masking a success.
