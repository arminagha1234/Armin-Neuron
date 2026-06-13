#!/bin/bash
# Beta 3 setup on Trainium2 host (per Native PyTorch User Guide - Beta 3, 5/15/26).
# Runs detached.
set -uo pipefail

LOG=/opt/dlami/nvme/beta3_setup.log
exec >>"$LOG" 2>&1

echo "============================================"
echo "[$(date)] setup_beta3.sh starting"
echo "============================================"

IMAGE_REPO=421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b
# Step 1 — make sure image is pulled. The 0461d3b tag has been seen
# to alias to a different repo on the host (concourse-release-1cb0647)
# after a CodePipeline refresh. Find the image by digest if the tag
# isn't present.
echo "[$(date)] step 1: ensure image pulled"
imageID=$(sudo docker images -q --filter reference="$IMAGE_REPO" | head -1)
if [ -z "$imageID" ]; then
    echo "[$(date)] tag $IMAGE_REPO not present, falling back to first ECR image we have"
    imageID=$(sudo docker images --format "{{.Repository}} {{.ID}}" | grep '421672808698' | head -1 | awk '{print $2}')
fi
if [ -z "$imageID" ]; then
    echo "[$(date)] ERROR: no Neuron DLC image found"
    exit 1
fi
echo "[$(date)] imageID=$imageID"

# Step 2 — extract /workspace from container to host (NVMe)
echo "[$(date)] step 2: extract /workspace from container"
mkdir -p /opt/dlami/nvme/beta3
cd /opt/dlami/nvme/beta3
if [ ! -d workspace ]; then
    sudo docker create --name tmp_beta3 "$imageID" 2>&1 | tail -3
    sudo docker cp tmp_beta3:/workspace . 2>&1 | tail -3
    sudo docker rm tmp_beta3 2>&1 | tail -3
    sudo chown -R ubuntu:ubuntu workspace
fi
ls workspace/

# Step 3 — install driver
echo "[$(date)] step 3: install driver from runtime_artifacts/"
sudo apt-get update -qq 2>&1 | tail -3
sudo apt-get install -qq -y dkms build-essential python3.12-venv 2>&1 | tail -3
sudo dpkg -i workspace/runtime_artifacts/*.deb 2>&1 | tail -10

# Step 4 — verify
echo "[$(date)] step 4: verify"
neuron-ls 2>&1 | head -10

# Step 5 — host venv (Option B from the guide)
echo "[$(date)] step 5: create host venv"
cd /opt/dlami/nvme/beta3/workspace
if [ ! -d native_venv ]; then
    python3.12 -m venv native_venv
fi
source native_venv/bin/activate
pip install -q uv 2>&1 | tail -3
export UV_PROJECT_ENVIRONMENT=/opt/dlami/nvme/beta3/workspace/native_venv

# Install nki + neuronx-cc + torch_neuronx + diffusers
uv pip install /opt/dlami/nvme/beta3/workspace/nki_wheels/nki-0.4.0*-cp312-cp312-linux_x86_64.whl 2>&1 | tail -5
uv pip install /opt/dlami/nvme/beta3/workspace/neuronx_cc_wheels/neuronx_cc-2.*-cp312-cp312-linux_x86_64.whl 2>&1 | tail -5
cd /opt/dlami/nvme/beta3/workspace/torch_neuron_eager
uv pip install -e ".[dev]" 2>&1 | tail -10

# Diffusers from git main (LTX-2 needs dev version per PR57 README)
uv pip install git+https://github.com/huggingface/diffusers.git 2>&1 | tail -5
uv pip install -q transformers accelerate Pillow imageio imageio-ffmpeg 2>&1 | tail -3

# Verify the stack
echo "[$(date)] verify torch + torch_neuronx + neuron device"
python -c "
import torch, torch_neuronx
print('torch:', torch.__version__)
print('torch_neuronx:', torch_neuronx.__version__)
d = torch.device('neuron')
print('device created:', d)
x = torch.randn(4, 4, device=d)
y = x @ x.T
print('neuron tensor probe ok:', y.shape, 'on', y.device)
" 2>&1 | tail -10

echo "============================================"
echo "[$(date)] setup_beta3.sh DONE"
echo "venv: /opt/dlami/nvme/beta3/workspace/native_venv"
echo "============================================"
