# Gemma4 31B IT on vLLM-Neuron (AWS Trainium2)

Serve Google's Gemma 4 31B IT on a trn2.48xlarge using the vLLM-Neuron plugin.
**293 ms TTFT @ 4K input** with TP=32 single-bucket — 41% under the 500 ms target.

---

## Results

All numbers measured on `trn2.48xlarge` (us-east-2), vLLM-Neuron v5 beta,
bf16, on-device greedy sampling.

| Input | TP | Bucket | TTFT (median) | Throughput | Status |
|---:|---:|---:|---:|---:|---|
| 4K | 16 | `[4096]` | 452 ms | — | passes 500 ms target |
| **4K** | **32** | **`[4096]`** | **293 ms ✅** | **693 tok/min @ conc=4** | **best 4K config (41% under target)** |
| 8K | 32 | `[8192]` | 659 ms ❌ | (not measured) | misses 500 ms target by 32% |

**Generation proof** (TP=32, single-bucket `[4096]`):
> "The capital of France is" → " Paris.\n\nThe capital of France is Paris..."
> TTFT 292.6 ms · TPOT 343 ms · 32 tokens in 10.94 s

Full benchmark detail (TTFT scan, throughput sweep, 8K, generation):
[`improvements-gemma4-partB/RESULTS.md`](improvements-gemma4-partB/RESULTS.md)

NKI kernel investigation (kernels validated on device, then honestly
characterized vs the compiled NxDI baseline):
[`improvements-gemma4-partC/`](improvements-gemma4-partC/)

---

## What You Need

- A **trn2.48xlarge** instance (or any Trn2 with ≥32 NeuronCores)
- The **vLLM-Neuron Private Beta** container image (get from your AWS Neuron contact)
- A **HuggingFace token** with access to `google/gemma-4-31b-it` (gated model)
- This repo cloned somewhere accessible

---

## Step-by-Step

### Step 1 — Set your environment variables

```bash
# Your HuggingFace token (needs access to google/gemma-4-31b-it)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxx

# The vLLM-Neuron beta image URI (get from your Neuron team contact)
export IMAGE=<your-vllm-neuron-beta-image-uri>
```

### Step 2 — Pull the image and install the Neuron driver

```bash
# ECR login (instance role must have ECR pull access)
aws ecr get-login-password --region us-east-1 \
  | sudo docker login --username AWS --password-stdin 421672808698.dkr.ecr.us-east-1.amazonaws.com

# Pull
sudo docker pull "$IMAGE"

# Extract and install the matched Neuron driver from the image
TMP=$(mktemp -d)
sudo docker create --name extract-driver "$IMAGE"
sudo docker cp extract-driver:/opt/aws/neuron/driver/. "$TMP/"
sudo docker rm extract-driver

# Ubuntu/Debian:
sudo dpkg -i $TMP/aws-neuronx-dkms_*.deb
# Amazon Linux:
# sudo dnf install -y $TMP/aws-neuronx-dkms-*.rpm

# Verify
modinfo neuron | grep "^version"
```

### Step 3 — Start the container

```bash
# Create persistent directories for model weights + compiled NEFFs
sudo mkdir -p /data/{hf_cache,neff_cache,work}

# Start container with all 16 Neuron devices
sudo docker run -d --privileged \
  --name vllm_neuron \
  -v /data/hf_cache:/root/.cache/huggingface \
  -v /data/neff_cache:/root/.cache/vllm \
  -v /data/work:/work \
  --env "HF_TOKEN=$HF_TOKEN" \
  --env "NEURON_SKIP_EFA_AFFINITY=1" \
  -p 8000:8000 \
  --ipc=host \
  --device /dev/neuron0 --device /dev/neuron1 --device /dev/neuron2 --device /dev/neuron3 \
  --device /dev/neuron4 --device /dev/neuron5 --device /dev/neuron6 --device /dev/neuron7 \
  --device /dev/neuron8 --device /dev/neuron9 --device /dev/neuron10 --device /dev/neuron11 \
  --device /dev/neuron12 --device /dev/neuron13 --device /dev/neuron14 --device /dev/neuron15 \
  "$IMAGE" \
  sleep infinity
```

### Step 4 — Copy this repo's code into the container

```bash
# From the directory where you cloned this repo:
sudo docker cp gemma4/              vllm_neuron:/work/pkg/gemma4/
sudo docker cp gemma4_register.py   vllm_neuron:/work/pkg/
sudo docker cp gemma4_transformers_stub.py vllm_neuron:/work/pkg/
sudo docker cp sitecustomize.py     vllm_neuron:/work/pkg/
sudo docker cp make_local_model.py  vllm_neuron:/work/pkg/
sudo docker cp bench_ttft.py        vllm_neuron:/work/pkg/
```

### Step 5 — Download model weights and patch the tokenizer

```bash
sudo docker exec vllm_neuron python3 -c "
from huggingface_hub import snapshot_download
import os
p = snapshot_download('google/gemma-4-31b-it', token=os.environ['HF_TOKEN'])
print('Downloaded to:', p)
"

# Build local model dir with patched tokenizer_config.json
sudo docker exec vllm_neuron python3 /work/pkg/make_local_model.py
```

This creates `/root/models/gemma-4-31b-it` with symlinked weight files
and a fixed `tokenizer_config.json` (the checkpoint ships `extra_special_tokens`
as a list; transformers 4.x needs a dict).

### Step 6 — Start the vLLM server

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
    --additional-config '"'"'{"neuron_config": {"num_batched_tokens_buckets": [4096], "num_seqs_buckets": [4], "on_device_sampling_config": {"all_greedy": true}}}'"'"' \
    2>&1 | tee /work/serve.log'
```

**First launch compiles the model (~5-8 minutes).** Watch the log:

```bash
sudo docker exec vllm_neuron tail -f /work/serve.log
```

Wait until you see:
```
INFO:     Application startup complete.
```

### Step 7 — Test it

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/root/models/gemma-4-31b-it",
    "prompt": "The capital of France is",
    "max_tokens": 20,
    "temperature": 0
  }'
```

Expected output includes `"text": " Paris.\n\n..."`.

### Step 8 — Benchmark TTFT

```bash
sudo docker exec vllm_neuron python3 /work/pkg/bench_ttft.py \
  --model /root/models/gemma-4-31b-it \
  --seq-lens 256,512,1024,2048,3900 \
  --runs 7 \
  --tag gemma4-tp32 \
  --out /work/results.json
```

Expected output:
```
seq_len=3900: median 294 ms  PASS (target 500ms)
```

---

## Configuration Tips

### Why single-bucket?

With `num_batched_tokens_buckets: [4096]` (one bucket), TTFT at 4K is **294 ms**.
With `[256, 512, 1024, 2048, 4096, 8192, 10240]` (many buckets), the same 4K
request takes **993 ms** — 3.4× slower. The Neuron compiler produces a more
optimized graph when it only has one bucket to target.

**Use single-bucket when your workload has a known context length.**

### TP options

Gemma4 has 32 attention heads. Valid TP values: 1, 2, 4, 8, 16, 32.
TP=64 is not possible (32 is not divisible by 64).

| TP | Cores used | 4K TTFT |
|---:|---:|---:|
| 16 | 16 of 64 | 452 ms |
| 32 | 32 of 64 | 294 ms |

### Serving 8K input

To serve 8K input, change the config:
```bash
--max-model-len 8192 --max-num-batched-tokens 8192
--additional-config '{"neuron_config": {"num_batched_tokens_buckets": [8192], ...}}'
```

8K TTFT at TP=32: **662 ms** (linear from 4K).

### Config constraints (vLLM-Neuron beta)

1. `max-num-batched-tokens` must be one of `[512, 1024, 2048, 4096]` OR
   equal to `max-model-len`. Other values are rejected.
2. The last entry in `num_batched_tokens_buckets` must equal `max-num-batched-tokens`.
3. `NEURON_SKIP_EFA_AFFINITY=1` is required on instances without EFA.

---

## How It Works (custom model registration)

Gemma4 is not in the vLLM-Neuron beta's built-in model list. This example
self-registers the model so `vllm serve` can use it:

1. **`sitecustomize.py`** — Python auto-imports this at startup. It runs
   `gemma4_transformers_stub.install()` (teaches `AutoConfig` about
   `model_type: gemma4`) and `gemma4_register.register()` (injects our
   model class into vLLM's `ModelRegistry`).

2. **`gemma4_register.py`** — Forces `Gemma4ForConditionalGeneration` into
   both `vllm_neuron.model.registry` and `vllm.ModelRegistry`. Includes a
   post-plugin hook that re-applies the registration after vLLM's plugin
   loader resets the registry.

3. **`gemma4/model.py`** — The actual model implementation with TP-sharded
   attention, heterogeneous SWA+Global layers, KV cache, and weight loading.

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

Other features: QK normalization, V normalization, per-layer scalar,
4 norms per layer, GeGLU activation, logit softcapping (30.0),
tied word embeddings, vocab 262144.

---

## Files

```
├── README.md                        # This file
├── gemma4/                          # Model package
│   ├── __init__.py
│   ├── config.py                    # Config dataclass
│   ├── factory.py                   # ModelRegistry factory
│   ├── model.py                     # Full model implementation
│   ├── flash_attn_hd256_nki.py      # Split-K attention kernel (head_dim=256)
│   ├── fused_geglu.py               # Fused GeGLU MLP kernel
│   ├── fused_qk_norm_rope.py        # Fused QK-norm + RoPE
│   ├── fused_norm_residual.py       # Fused norm-residual
│   ├── fused_embed_scale.py         # Fused embedding + scale
│   ├── fused_logit_softcap.py       # Fused LM head + softcap
│   └── optimized_forward.py         # Kernel integration guide
├── gemma4_register.py               # Runtime model registration
├── gemma4_transformers_stub.py      # Config stub for transformers 4.x
├── sitecustomize.py                 # Auto-registration on import
├── make_local_model.py              # Build patched local model dir
├── bench_ttft.py                    # TTFT benchmark
└── throughput_bench.py              # Throughput benchmark
```

## License

Apache-2.0
