#!/usr/bin/env bash
# Get the Beta 2 DLC running on the fal-ai trn2.48xl, ready for Path C.
#
# Phase 1 Day 2 (continued): we found that stock SDK 2.9 DLAMI is missing
# the 'neuron' ProcessGroup backend that DTensor + parallelize_module
# need. The fix is the Beta 2 DLC image — that's what wan_training and
# LTX-2.3 v3 actually ran on.
#
# Steps:
#   1. Move docker data-root to NVMe (root disk only has 37 GB free, image is ~42 GB)
#   2. Pull Beta 2 DLC image
#   3. Create persistent container with /data mount + path_c bind mount
#   4. Install diffusers + peft + transformers + accelerate inside (no-deps)
#
# Idempotent: re-runnable.

set -euo pipefail

NVME_MOUNT=/opt/dlami/nvme
DOCKER_DATA="$NVME_MOUNT/docker"
PATH_C_DIR="$NVME_MOUNT/fal/path_c"
HF_CACHE="$NVME_MOUNT/hf_cache"
IMAGE="421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-0461d3b:latest"
CONTAINER=fal_beta2

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ─── 1. Move docker data-root to NVMe ────────────────────────────────
if [ ! -d "$DOCKER_DATA" ]; then
    log "Moving docker data-root to $DOCKER_DATA"
    sudo systemctl stop docker docker.socket containerd 2>/dev/null || true
    sudo mkdir -p "$DOCKER_DATA"
    if [ -d /var/lib/docker ]; then
        sudo cp -a /var/lib/docker/. "$DOCKER_DATA/" 2>/dev/null || true
    fi
    sudo mkdir -p /etc/docker
    echo '{"data-root": "'"$DOCKER_DATA"'"}' | sudo tee /etc/docker/daemon.json >/dev/null
    sudo systemctl start docker
    sleep 3
fi
sudo docker info 2>&1 | grep -i "data root" || true

# ─── 2. ECR login ────────────────────────────────────────────────────
log "ECR login (us-east-1, account 421672808698)"
aws ecr get-login-password --region us-east-1 \
    | sudo docker login --username AWS --password-stdin \
        421672808698.dkr.ecr.us-east-1.amazonaws.com

# ─── 3. Pull Beta 2 image ────────────────────────────────────────────
log "Pulling Beta 2 image (this may take a while — ~40 GB)"
sudo docker pull "$IMAGE"

# ─── 4. Create persistent container ──────────────────────────────────
if sudo docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    log "Container $CONTAINER already exists; not re-creating"
else
    log "Creating $CONTAINER"
    sudo docker run -d --name "$CONTAINER" --privileged \
        -v /dev:/dev \
        -v "$NVME_MOUNT":"$NVME_MOUNT" \
        -v "$PATH_C_DIR":/work/path_c \
        -v "$HF_CACHE":/root/.cache/huggingface \
        -e HF_HOME=/root/.cache/huggingface \
        --network host \
        --shm-size 32g \
        "$IMAGE" sleep infinity
fi

# ─── 5. Install diffusers stack inside container ─────────────────────
log "Installing diffusers stack inside $CONTAINER"
sudo docker exec "$CONTAINER" bash -c '
    source /opt/torch-neuronx/.venv/bin/activate
    pip install --index-url=https://pypi.org/simple/ --no-cache-dir --no-deps \
        diffusers==0.38.0 peft==0.19.1 einops ftfy
    pip install --index-url=https://pypi.org/simple/ --no-cache-dir \
        transformers==4.46.3 tokenizers \
        Pillow numpy huggingface_hub safetensors accelerate==1.0.1
    python -c "import diffusers; print(\"diffusers\", diffusers.__version__)"
    python -c "from diffusers import QwenImageEditPlusPipeline; print(\"QwenImageEditPlusPipeline OK\")"
    python -c "import torch_neuronx; print(\"torch_neuronx\", torch_neuronx.__version__)"
'

log "Beta 2 setup complete. Container: $CONTAINER"
log "To use:"
log "  sudo docker exec -it $CONTAINER bash"
log "  source /opt/torch-neuronx/.venv/bin/activate"
log "  cd /work/path_c"
