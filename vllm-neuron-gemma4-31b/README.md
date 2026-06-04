# Gemma4 31B IT on vLLM-Neuron (AWS Trainium2)

Serve Google's Gemma 4 31B IT on a trn2.48xlarge using the vLLM-Neuron plugin.

**172 ms weighted-average TTFT** on a real customer payload distribution
(TP=32, multi-bucket `[512, 1024, 2048, 4096]`).

---

## Headline Results

All numbers measured on `trn2.48xlarge` (us-east-2), vLLM-Neuron v5 beta, bf16,
on-device greedy sampling. Raw JSONs in [`results/`](results/).

### Distribution-aware TTFT (matches a real customer payload mix)

Customer's payload distribution: 24.8% ≤0.5K, 53.1% ≤1K, 9.5% ≤2K, 12.7% ≤4K.

| Bucket | Share of traffic | Multi-bucket TTFT | Single-bucket [4096] TTFT |
|---:|---:|---:|---:|
| ≤0.5K | 24.8% | **102.1 ms** | 290 ms |
| ≤1K | 53.1% | **153.8 ms** | 290 ms |
| ≤2K | 9.5% | **288.3 ms** | 290 ms |
| ≤4K | 12.7% | **295.9 ms** | 290 ms |
| **Weighted average** | 100% | **🎯 172.0 ms** | 290.5 ms |

**Multi-bucket cuts effective TTFT by 41% on this customer's traffic mix.**


### Single-bucket configs (when context is fixed)

| Input | TP | Bucket | TTFT (median) | Status |
|---:|---:|---:|---:|---|
| ≤1K | 32 | `[1024]` | **102 ms** | ✅ best for short prompts |
| 4K | 32 | `[4096]` | 293 ms | ✅ 41% under 500 ms target |
| 4K | 16 | `[4096]` | 452 ms | passes target |
| 8K | 32 | `[8192]` | 659 ms | ❌ 32% over target |

### Throughput (TP=32, single-bucket [4096], in=1024 / out=256)

> TPOT is bound at ~343 ms/token (≈2.9 tok/s per request) by the head_dim>128
> SDPA decode fallback. Aggregate throughput scales with `max_num_seqs` until
> that ceiling. Above concurrency=4, requests just queue.

| Concurrency | Aggregate tok/s | Per-req tok/s | Per-req latency |
|---:|---:|---:|---:|
| 1 | 2.9 | 2.9 | 87.8 s |
| **4** | **11.6** | **2.9** | **88.6 s** |
| 8 | 11.6 (queued) | 1.6 | 155 s |
| 16 | 11.6 (queued) | 0.9 | 288 s |

Throughput plateaus at concurrency 4 because `max_num_seqs=4`. Concurrency 8
and 16 don't add throughput — they just queue, and per-request latency grows
linearly. Raise `max_num_seqs` to lift the ceiling at the cost of more
KV-cache HBM.

### Generation Proof

```
Prompt:  "The capital of France is"
TTFT:    292.6 ms
Output:  " Paris.\n\nThe capital of France is Paris.\n\n..."
TPOT:    343 ms (the head_dim>128 SDPA-fallback decode rate)
```


---

## What You Need

- A **trn2.48xlarge** instance (or any Trn2 with ≥32 NeuronCores)
- The **vLLM-Neuron Private Beta** container image (get from your AWS Neuron contact)
- A **HuggingFace token** with access to `google/gemma-4-31b-it` (gated model)
- This repo cloned somewhere accessible

## Step-by-Step

### Step 1 — Set environment variables

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx
export IMAGE=<your-vllm-neuron-beta-image-uri>
```

### Step 2 — Pull the image and install the matched Neuron driver

```bash
aws ecr get-login-password --region us-east-1 \
  | sudo docker login --username AWS --password-stdin 421672808698.dkr.ecr.us-east-1.amazonaws.com
sudo docker pull "$IMAGE"

# Extract and install the matched driver from the image
TMP=$(mktemp -d)
sudo docker create --name extract-driver "$IMAGE"
sudo docker cp extract-driver:/opt/aws/neuron/driver/. "$TMP/"
sudo docker rm extract-driver
# Ubuntu/Debian:
sudo dpkg -i $TMP/aws-neuronx-dkms_*.deb
# Amazon Linux:
# sudo dnf install -y $TMP/aws-neuronx-dkms-*.rpm
modinfo neuron | grep "^version"
```

### Step 3 — Start the container

```bash
sudo mkdir -p /data/{hf_cache,neff_cache,work}
sudo docker run -d --privileged --name vllm_neuron \
  -v /data/hf_cache:/root/.cache/huggingface \
  -v /data/neff_cache:/root/.cache/vllm \
  -v /data/work:/work \
  --env "HF_TOKEN=$HF_TOKEN" \
  --env "NEURON_SKIP_EFA_AFFINITY=1" \
  -p 8000:8000 --ipc=host \
  $(for i in $(seq 0 15); do echo --device /dev/neuron$i; done) \
  "$IMAGE" sleep infinity
```

### Step 4 — Copy this repo's code into the container

```bash
sudo docker cp gemma4/                 vllm_neuron:/work/pkg/gemma4/
sudo docker cp gemma4_register.py      vllm_neuron:/work/pkg/
sudo docker cp gemma4_transformers_stub.py vllm_neuron:/work/pkg/
sudo docker cp sitecustomize.py        vllm_neuron:/work/pkg/
sudo docker cp make_local_model.py     vllm_neuron:/work/pkg/
sudo docker cp bench_ttft.py           vllm_neuron:/work/pkg/
sudo docker cp bench_throughput.py     vllm_neuron:/work/pkg/
sudo docker cp bench_distribution.py   vllm_neuron:/work/pkg/
```

### Step 5 — Download model and patch tokenizer

```bash
sudo docker exec vllm_neuron python3 -c "
from huggingface_hub import snapshot_download
import os
print(snapshot_download('google/gemma-4-31b-it', token=os.environ['HF_TOKEN']))
"
sudo docker exec vllm_neuron python3 /work/pkg/make_local_model.py
```

This creates `/root/models/gemma-4-31b-it` with patched `tokenizer_config.json`
(checkpoint ships `extra_special_tokens` as a list; transformers 4.x needs a dict).


### Step 6 — Start the vLLM server

**Recommended config — multi-bucket, distribution-optimized (172 ms weighted avg):**

```bash
sudo docker exec -d \
  -e NEURON_SKIP_EFA_AFFINITY=1 \
  -e PYTHONPATH=/work/pkg \
  vllm_neuron \
  bash -c 'vllm serve /root/models/gemma-4-31b-it \
    --tensor-parallel-size 32 \
    --max-model-len 4096 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 4096 \
    --additional-config '"'"'{"neuron_config":{"num_batched_tokens_buckets":[512,1024,2048,4096],"num_seqs_buckets":[4],"on_device_sampling_config":{"all_greedy":true}}}'"'"' \
    2>&1 | tee /work/serve.log'
```

**Alternative — single-bucket [4096] (flat 290 ms regardless of input size):**

Use this if all your prompts are around 4K tokens. Replace
`"num_batched_tokens_buckets":[512,1024,2048,4096]` with
`"num_batched_tokens_buckets":[4096]`.

**Alternative — single-bucket [1024] (102 ms for short prompts only):**

Use this if all your prompts are ≤1K. Set `--max-model-len 1024
--max-num-batched-tokens 1024` and `"num_batched_tokens_buckets":[1024]`.

First launch compiles the model (~5-8 min per bucket). Watch the log:

```bash
sudo docker exec vllm_neuron tail -f /work/serve.log
# wait for: "Application startup complete."
```

### Step 7 — Test it

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/root/models/gemma-4-31b-it","prompt":"The capital of France is","max_tokens":20,"temperature":0}'
```

Expected `"text": " Paris.\n\n..."`.

### Step 8 — Benchmark TTFT

```bash
# TTFT scan (single-bucket or multi-bucket — works for either)
sudo docker exec vllm_neuron python3 /work/pkg/bench_ttft.py \
  --model /root/models/gemma-4-31b-it \
  --seq-lens 256,512,1024,2048,3900 --runs 5 \
  --tag gemma4-tp32 --out /work/results.json

# Distribution-aware (matches the customer's payload mix)
sudo docker exec vllm_neuron python3 /work/pkg/bench_distribution.py
```

### Step 9 — Throughput sweep

```bash
sudo docker exec vllm_neuron python3 /work/pkg/bench_throughput.py \
  --model /root/models/gemma-4-31b-it \
  --concurrency 1,4,8,16 \
  --input-tokens 1024 --output-tokens 256 \
  --reqs-per-level 2
```


---

## Configuration Tips

### Why multi-bucket beats single-bucket on this workload

With single-bucket `[4096]`, every prompt is padded to 4096 tokens before the
kernel runs — a 256-token prompt pays the same 290 ms as a 4K prompt. With
multi-bucket `[512, 1024, 2048, 4096]`, each request lands in its smallest
fitting bucket: a 256-token prompt runs through the 512-bucket NEFF at 102 ms
instead of the 4K NEFF at 290 ms.

For workloads with mixed prompt lengths (most production traffic),
**multi-bucket is the right answer.** Use single-bucket only when context is
known and fixed.

### TP options

Gemma4 has 32 attention heads. Valid TP values: 1, 2, 4, 8, 16, 32.
TP=64 is impossible (32 not divisible by 64).

| TP | 4K TTFT |
|---:|---:|
| 16 | 452 ms |
| **32** | **293 ms** |

### Config constraints (vLLM-Neuron beta)

1. `max-num-batched-tokens` must be one of `[512, 1024, 2048, 4096]` OR equal to `max-model-len`.
2. The last entry in `num_batched_tokens_buckets` must equal `max-num-batched-tokens`.
3. `NEURON_SKIP_EFA_AFFINITY=1` is required on instances without EFA.

---

## How It Works (custom model registration)

Gemma4 is not in the vLLM-Neuron beta's built-in model list. This example
self-registers the model so `vllm serve` can use it:

1. **`sitecustomize.py`** — Python auto-imports this at startup. It runs
   `gemma4_transformers_stub.install()` (teaches `AutoConfig` about
   `model_type: gemma4`) and `gemma4_register.register()` (injects our model
   class into vLLM's `ModelRegistry`).
2. **`gemma4_register.py`** — Forces `Gemma4ForConditionalGeneration` into both
   `vllm_neuron.model.registry` and `vllm.ModelRegistry`. Includes a
   post-plugin hook that re-applies the registration after vLLM's plugin
   loader resets the registry.
3. **`gemma4/model.py`** — The model implementation with TP-sharded attention,
   heterogeneous SWA+Global layers, KV cache, and weight loading.

No vLLM fork needed. Everything runs via `PYTHONPATH`.

---

## Model Architecture

| | SWA Layers (49) | Global Layers (11) |
|---|---|---|
| head_dim | 256 | 512 |
| KV heads | 16 | 4 |
| Attention | Sliding window (1024) | Full causal |
| RoPE θ | 10,000 | 1,000,000 |
| RoPE coverage | 100% | 25% (partial) |
| V projection | Separate | K=V (copies K) |

Other features: QK normalization, V normalization, per-layer scalar, 4 norms
per layer, GeGLU activation, logit softcapping (30.0), tied word embeddings,
vocab 262144.

---

## Files

```
├── README.md                        # This file
├── gemma4/                          # Model package
├── gemma4_register.py               # Runtime model registration
├── gemma4_transformers_stub.py      # Config stub for transformers 4.x
├── sitecustomize.py                 # Auto-registration on import
├── make_local_model.py              # Build patched local model dir
├── bench_ttft.py                    # TTFT benchmark
├── bench_throughput.py              # Throughput benchmark
├── bench_distribution.py            # Distribution-aware bench (customer mix)
└── results/                         # Raw measurement JSONs
    ├── ttft_single_bucket_4k.json   # Flat 290 ms scan
    ├── ttft_multi_bucket.json       # Per-bucket TTFT scan
    ├── ttft_8k_clean.json           # 8K bucket measurement
    ├── ttft_distribution.json       # Weighted-avg summary
    ├── throughput.json              # Concurrency sweep
    └── generation_proof.json        # End-to-end gen sample
```

## License

Apache-2.0
