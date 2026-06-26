#!/bin/bash
# explore_v2 launch. Usage: bash launch.sh LEN BUCKETS SEG TP MNS
#   LEN     = max-model-len (caps input+output combined)
#   BUCKETS = num_batched_tokens_buckets, comma list, last must == SEG
#   SEG     = max-num-batched-tokens
#   TP      = tensor-parallel-size
#   MNS     = max-num-seqs
# Stage 1 (canonical 4K):  bash launch.sh 4096 512,1024,2048,4096 4096 32 4
# Stage 1 fallback (2K):   bash launch.sh 2048 512,1024,2048 2048 32 4
# Stage 2 (32K chunked):   bash launch.sh 36864 4096 4096 32 1
LEN=${1:-4096}; BUCKETS=${2:-512,1024,2048,4096}; SEG=${3:-4096}; TP=${4:-32}; MNS=${5:-4}
MODEL=/root/models/gemma-4-31b-it
LOG=/work/serve_len${LEN}_tp${TP}.log

pkill -9 -f "vllm serve" 2>/dev/null || true
pkill -9 -f "EngineCore" 2>/dev/null || true
pkill -9 -f "multiproc_executor" 2>/dev/null || true
sleep 6

export PYTHONPATH=/work/pkg
export NEURON_SKIP_EFA_AFFINITY=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=2400
export VLLM_ENGINE_ITERATION_TIMEOUT_S=2400
export VLLM_RPC_TIMEOUT=2400000
export NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE=$(( LEN * 5376 * 2 ))

ADD="{\"neuron_config\":{\"num_batched_tokens_buckets\":[${BUCKETS}],\"num_seqs_buckets\":[${MNS}],\"on_device_sampling_config\":{\"all_greedy\":true}}}"
echo "=== explore_v2 len=$LEN buckets=$BUCKETS seg=$SEG tp=$TP mns=$MNS @ $(date -u) ===" | tee "$LOG"
nohup vllm serve "$MODEL" \
    --served-model-name gemma4 \
    --tensor-parallel-size "$TP" \
    --max-model-len "$LEN" \
    --max-num-seqs "$MNS" \
    --max-num-batched-tokens "$SEG" \
    --additional-config "$ADD" \
    --port 8000 --host 0.0.0.0 \
    >> "$LOG" 2>&1 < /dev/null &
disown
echo "PID=$! LOG=$LOG"
