#!/usr/bin/env bash
# Run the prefill benchmark end-to-end inside a Beta 3 DLC container.
#
# Usage:
#   COMPILE=1 ./serve.sh   # eager + torch.compile sweep
#   ./serve.sh             # eager-only sweep
#
# Assumes:
#   * Beta 3 DLC at /opt/torch-neuronx/.venv (installed by the DLC)
#   * Gemma 4 E4B-it cached at /root/.cache/huggingface (run
#     `huggingface-cli download google/gemma-4-E4B-it` first)
#   * This script is invoked from inside the Beta 3 container
#   * NEFF cache at /tmp/neff_cache should be bind-mounted from a host
#     dir so it persists across container restarts (else cold-start =
#     ~70 s per bucket)

set -euo pipefail

VENV=/opt/torch-neuronx/.venv
MODEL_DIR=${MODEL_DIR:-/root/models/gemma-4-E4B-it}
SEQ_LENS=${SEQ_LENS:-64,128,256,512,1024,2048}
WARMUP=${WARMUP:-1}
RUNS=${RUNS:-3}
COMPILE_FLAG=""
[[ "${COMPILE:-0}" == "1" ]] && COMPILE_FLAG="--compile"

# Materialize local model dir if it doesn't exist
if [[ ! -d "$MODEL_DIR" ]]; then
  echo ">> building local model dir (patched tokenizer)"
  "$VENV/bin/python" "$(dirname "$0")/build_local_model.py" --dst "$MODEL_DIR"
fi

OUT="results_${COMPILE:-eager}.json"

NEURON_RT_VIRTUAL_CORE_SIZE=2 NEURON_RT_NUM_CORES=2 \
  "$VENV/bin/torchrun" \
    --nproc_per_node=2 --rdzv_backend=c10d --rdzv_endpoint=localhost:29500 \
    "$(dirname "$0")/run_e4b.py" \
    --model "$MODEL_DIR" \
    --seq-lens "$SEQ_LENS" \
    --warmup "$WARMUP" --runs "$RUNS" \
    $COMPILE_FLAG \
    --out "$OUT"
