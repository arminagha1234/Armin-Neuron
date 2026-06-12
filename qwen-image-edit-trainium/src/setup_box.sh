#!/usr/bin/env bash
# Phase 1 Day 1: box setup for the fal-ai trn2.48xlarge.
#
# Runs on the box itself (not from the dev machine). Does:
#   1. Mount the 4 NVMe instance-store drives as /opt/dlami/nvme (RAID0)
#   2. Install diffusers ≥ 0.38 + transformers + accelerate into the
#      native-PyTorch Neuron venv (aws_neuronx_venv_pytorch_2_9)
#   3. Cache Qwen-Image-Edit-2511 + the fal Multiple-Angles LoRA to NVMe
#
# Usage (on box):
#   curl -O https://raw.githubusercontent.com/.../setup_box.sh   # or scp
#   bash setup_box.sh
#
# Idempotent: re-running is safe.

set -euo pipefail

NVME_MOUNT=/opt/dlami/nvme
VENV=/opt/aws_neuronx_venv_pytorch_2_9
HF_CACHE="$NVME_MOUNT/hf_cache"
PROJECT_DIR="$NVME_MOUNT/fal"

log() { echo "[$(date +%H:%M:%S)] $*"; }

# ─── 1. Mount instance-store NVMe ─────────────────────────────────────
if mountpoint -q "$NVME_MOUNT"; then
    log "NVMe already mounted at $NVME_MOUNT"
else
    log "Mounting 4× NVMe drives as RAID0 at $NVME_MOUNT"
    sudo mkdir -p "$NVME_MOUNT"
    # Find the 4 instance-store drives (1.7 TB each, NOT the root)
    DRIVES=$(lsblk -dn -o NAME,SIZE | awk '$2 == "1.7T" {print "/dev/" $1}')
    DRIVE_COUNT=$(echo "$DRIVES" | wc -l)
    if [ "$DRIVE_COUNT" -ne 4 ]; then
        log "WARN: expected 4× 1.7T NVMe, found $DRIVE_COUNT — falling back to single drive"
        FIRST=$(echo "$DRIVES" | head -1)
        sudo mkfs.ext4 -F "$FIRST"
        sudo mount "$FIRST" "$NVME_MOUNT"
    else
        sudo apt-get install -y mdadm >/dev/null 2>&1 || true
        # shellcheck disable=SC2086
        sudo mdadm --create --verbose /dev/md0 --level=0 --raid-devices=4 $DRIVES --force --run
        sudo mkfs.ext4 -F /dev/md0
        sudo mount /dev/md0 "$NVME_MOUNT"
    fi
    sudo chown -R ubuntu:ubuntu "$NVME_MOUNT"
fi
df -h "$NVME_MOUNT" | tail -1

# ─── 2. Project + cache dirs ──────────────────────────────────────────
mkdir -p "$HF_CACHE" "$PROJECT_DIR/path_c" "$PROJECT_DIR/path_c/results"
log "HF cache: $HF_CACHE"
log "Project:  $PROJECT_DIR"

# ─── 3. Install diffusers stack into native-PyTorch venv ──────────────
log "Activating $VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

log "Upgrading pip + installing diffusers stack"
pip install --quiet --upgrade pip
pip install --quiet \
    "diffusers>=0.38" \
    "transformers>=4.45" \
    "accelerate>=0.34" \
    "huggingface_hub>=0.25" \
    "safetensors>=0.4" \
    "Pillow" \
    "peft>=0.13"

python -c "import diffusers; print('diffusers', diffusers.__version__)"
python -c "from diffusers import QwenImageEditPlusPipeline; print('QwenImageEditPlusPipeline available')"

# ─── 4. Cache HF artifacts to NVMe ────────────────────────────────────
export HF_HOME="$HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$HF_CACHE"

log "Downloading Qwen-Image-Edit-2511 base model (this may take a while)"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
hf_cache = os.environ["HF_CACHE"] if "HF_CACHE" in os.environ else os.path.expanduser("~/.cache/huggingface")
path = snapshot_download(
    repo_id="Qwen/Qwen-Image-Edit-2511",
    cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE"),
    local_dir_use_symlinks=False,
)
print("base snapshot:", path)
PY

log "Downloading fal Multiple-Angles LoRA"
python - <<'PY'
import os
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA",
    cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE"),
    local_dir_use_symlinks=False,
)
print("lora snapshot:", path)
PY

log "Setup complete."
log "Next step: scp Path C code to $PROJECT_DIR/path_c/ from the dev machine."
