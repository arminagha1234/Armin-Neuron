#!/bin/bash
# Launch a Gemma4-31B server on the PUBLIC vLLM-Neuron v0.21 DLC for one input-size
# config, then WAIT (in-process) until it is ready. Mirrors the repo's launch_serve.sh
# but for the public stack: the Gemma4 model is already installed/registered in
# vllm_neuron/model/gemma4 (no PYTHONPATH/sitecustomize/deploy needed).
#
#   LEN=5120 SEG=512 BUCKETS=512 MNS=32 KV_CACHE_DTYPE=auto APC=1 bash launch_serve_public.sh
set -u
MODEL=${MODEL:-/root/models/gemma-4-31b-it}
SERVED_NAME=${SERVED_NAME:-gemma4}
TP=${TP:-32}
LEN=${LEN:-5120}
SEG=${SEG:-512}
BUCKETS=${BUCKETS:-512}
MNS=${MNS:-32}
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}
APC=${APC:-1}
PORT=${PORT:-8000}
LOGDIR=${LOGDIR:-/root}
LOG="$LOGDIR/serve_len${LEN}_seg${SEG}_tp${TP}.log"

echo "[launch] stopping any existing vllm serve ..."
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true
pkill -9 -f "multiproc_executor" 2>/dev/null || true
sleep 6

export NEURON_SKIP_EFA_AFFINITY=${NEURON_SKIP_EFA_AFFINITY:-1}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/scratch/neff_public}
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-2400}
export VLLM_ENGINE_ITERATION_TIMEOUT_S=${VLLM_ENGINE_ITERATION_TIMEOUT_S:-2400}
export VLLM_RPC_TIMEOUT=${VLLM_RPC_TIMEOUT:-2400000}
export NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE=$(( LEN * 5376 * 2 ))

ADD="{\"neuron_config\":{\"num_batched_tokens_buckets\":[${BUCKETS}],\"num_seqs_buckets\":[${MNS}],\"on_device_sampling_config\":{\"all_greedy\":true}}}"
KVDT_ARG=""
[ "$KV_CACHE_DTYPE" != "auto" ] && KVDT_ARG="--kv-cache-dtype $KV_CACHE_DTYPE"
APC_ARG="--no-enable-prefix-caching"
[ "$APC" = "1" ] && APC_ARG="--enable-prefix-caching"

echo "[launch] MODEL=$MODEL TP=$TP LEN=$LEN SEG=$SEG BUCKETS=$BUCKETS MNS=$MNS KV=$KV_CACHE_DTYPE APC=$APC PORT=$PORT"
echo "[launch] serve log -> $LOG"
nohup vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$LEN" \
  --max-num-seqs "$MNS" \
  --max-num-batched-tokens "$SEG" \
  $KVDT_ARG \
  $APC_ARG \
  --additional-config "$ADD" \
  --port "$PORT" --host 0.0.0.0 \
  >> "$LOG" 2>&1 < /dev/null &
disown

echo "[launch] waiting for server to be ready (first launch compiles — can take 10-20 min)..."
for i in $(seq 1 360); do
  if curl -s --max-time 3 "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q "$SERVED_NAME"; then
    echo "[launch] server READY after ~$((i * 5))s"
    exit 0
  fi
  # bail early if the process died
  if ! pgrep -f "vllm serve" >/dev/null 2>&1; then
    echo "[launch] ERROR: vllm serve process exited. Last log lines:"; tail -40 "$LOG"; exit 1
  fi
  sleep 5
done
echo "[launch] ERROR: server not ready after 30 min. Last serve log lines:"; tail -40 "$LOG"; exit 1
