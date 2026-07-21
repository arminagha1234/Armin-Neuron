#!/bin/bash
# Launch a Gemma4-31B vLLM-Neuron server for one input-size config, then wait until it is ready.
# Called by run_benchmark.sh (once per input size), or run standalone. Configure via env vars.
#
#   LEN=5120 SEG=512 BUCKETS=512 KV_CACHE_DTYPE=fp8_e4m3 APC=1 bash launch_serve.sh
#
# All settings have defaults; override any via env var (see run_benchmark.sh for per-size values).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"

MODEL=${MODEL:-google/gemma-4-31B-it}          # HF id (or local path) of the Gemma4-31B checkpoint
SERVED_NAME=${SERVED_NAME:-gemma4}
TP=${TP:-32}                                   # tensor-parallel size
LEN=${LEN:-5120}                               # --max-model-len (caps input+output)
SEG=${SEG:-512}                                # --max-num-batched-tokens (chunked-prefill segment). 512 = lowest TTFT floor + enables APC.
BUCKETS=${BUCKETS:-512}                         # num_batched_tokens_buckets — with segmented prefill this MUST equal SEG (single bucket)
MNS=${MNS:-32}                                 # --max-num-seqs (>= max concurrency you will test)
KV_CACHE_DTYPE=${KV_CACHE_DTYPE:-auto}         # 'auto' = bf16 (default). Set 'fp8_e4m3' for fp8 KV cache (raises concurrency ceiling at long ctx).
APC=${APC:-1}                                  # 1 = enable prefix caching. Big TTFT win when the context repeats (RAG / fixed prompt); requires SEG<LEN.
PORT=${PORT:-8000}
SERVING_PKG=${SERVING_PKG:-$HERE/serving_pkg}  # Gemma4 registration package (bundled here); on PYTHONPATH so `vllm serve` recognizes Gemma4
LOGDIR=${LOGDIR:-.}
LOG="$LOGDIR/serve_len${LEN}_tp${TP}.log"

echo "[launch] stopping any existing vllm serve ..."
pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true
pkill -9 -f "multiproc_executor" 2>/dev/null || true
sleep 6

if [ -n "$SERVING_PKG" ]; then
  export PYTHONPATH="$SERVING_PKG:${PYTHONPATH:-}"   # sitecustomize.py here registers Gemma4 + deploys the segmented kernel
  echo "[launch] SERVING_PKG on PYTHONPATH: $SERVING_PKG"
  # Deploy the patched segmented-attention kernel over the container's vllm_neuron
  # copy (edit A + SWA windowed gather). Required for Gemma4 chunked prefill
  # (head_dim 256/512). Idempotent + backs up the original. sitecustomize also
  # does this defensively; running it here makes it visible in the launch log.
  if [ -f "$SERVING_PKG/deploy_segmented_cte.py" ]; then
    python3 "$SERVING_PKG/deploy_segmented_cte.py" \
      || echo "[launch] WARN: segmented CTE deploy failed — long-context prefill may not work"
  fi
fi
# Long-context (32K/64K) compile + execute can exceed vLLM's default timeouts.
export NEURON_SKIP_EFA_AFFINITY=${NEURON_SKIP_EFA_AFFINITY:-1}
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-2400}
export VLLM_ENGINE_ITERATION_TIMEOUT_S=${VLLM_ENGINE_ITERATION_TIMEOUT_S:-2400}
export VLLM_RPC_TIMEOUT=${VLLM_RPC_TIMEOUT:-2400000}
export NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE=$(( LEN * 5376 * 2 ))
ADD="{\"neuron_config\":{\"num_batched_tokens_buckets\":[${BUCKETS}],\"num_seqs_buckets\":[${MNS}],\"on_device_sampling_config\":{\"all_greedy\":true}}}"
KVDT_ARG=""
[ "$KV_CACHE_DTYPE" != "auto" ] && KVDT_ARG="--kv-cache-dtype $KV_CACHE_DTYPE"
APC_ARG=""
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

echo "[launch] waiting for server to be ready (first launch compiles the model — can take 10-20 min)..."
for i in $(seq 1 300); do
  if curl -s --max-time 3 "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q "$SERVED_NAME"; then
    echo "[launch] server READY after ~$((i * 5))s"
    exit 0
  fi
  sleep 5
done
echo "[launch] ERROR: server not ready after 25 min. Last serve log lines:"
tail -30 "$LOG" 2>/dev/null
exit 1
