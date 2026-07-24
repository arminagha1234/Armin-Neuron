# Profile your own workload (Neuron profiler on the public image)

The public vLLM-Neuron image ships the built-in Neuron profiler — you can capture device + system profiles
on a live serve and view them in Neuron Explorer. Verified working on the public image (SDK 2.31).

## 1. Start the serve with profiling enabled
Add `--profiler-config` (reuse the `"cuda"` kind — it's reinterpreted as Neuron profiling) and a
`neuron_profiler` block in `--additional-config`:
```bash
GEMMA4_CTE_PREFILL=1 GEMMA4_BF16_FALLBACK=1 \
vllm serve /root/models/gemma-4-31b-text --served-model-name gemma4 \
  --tensor-parallel-size 16 --max-model-len 16384 --max-num-seqs 32 \
  --max-num-batched-tokens 16384 --no-enable-prefix-caching --async-scheduling \
  --profiler-config '{"profiler":"cuda"}' \
  --additional-config '{"neuron_config":{"num_batched_tokens_buckets":[256,512,1024,2048,4096,8192,16384],"num_seqs_buckets":[32],"on_device_sampling_config":{"all_greedy":true}},"neuron_profiler":{"activities":["device_profile","system_profile"],"neuron_cores":[0,1,2,3],"output_dir":"/root/nrt_profile"}}' \
  --port 8000 --host 0.0.0.0
```

## 2. Capture around representative traffic
```bash
curl -X POST http://localhost:8000/start_profile
# ... send your representative requests here ...
curl -X POST http://localhost:8000/stop_profile
```
Tip: to profile steady-state only, add `"delay_iterations":50,"max_iterations":20` to `--profiler-config`
(auto-starts after 50 engine steps, auto-stops after 20 more).

## 3. Inspect
```bash
ls /root/nrt_profile/                              # i-<instance>_pid_<pid>/<ts>/*.ntff + neffs/
neuron-profile show-session -s <path>/*.ntff       # quick engine-activity summary (CLI)
neuron-explorer view -d /root/nrt_profile/         # full device timeline (GUI)
```

## What we found (TP16, 16k, conc=8)
`show-session` engine counts per core: TENSOR (matmul) ~1.59M vs SYNC (collective/barrier) ~57K — a ~28:1
compute-to-sync ratio, evenly balanced across cores. This is the measured reason TP16 (4-chip replica) has
a strong compute-to-communication ratio and scales well under concurrent load. See the deployment guidance
in the README ("TP32 for latency, TP16 for concurrent throughput").

## Common issues
- **No output:** ensure `--profiler-config '{"profiler":"cuda"}'` was passed at startup (endpoints only mount then).
- **Explorer can't render device profile:** confirm `neffs/` exists in the output dir (auto-copied from the compile cache).
- **`.ntff` files are large** (~5 GB/core for a busy window) — capture short windows and clean up after.
