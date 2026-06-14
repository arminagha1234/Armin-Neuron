# GLM 5.1 on Trainium2 — vLLM-Neuron

Production serving path for GLM 5.1 (754B MoE+MLA+DSA) on trn2.48xlarge
using vLLM-Neuron's private-beta image. End-to-end inference is
**validated for partial models** (2 and 30 layers). Full 78 layers is
blocked on FP8-on-device weight storage.

## Status

| Config | Layers | TP | Load | TTFT | Status |
|---|---:|---:|---:|---:|---|
| Smoke | 2 | 2 | 81 s | 320 ms | ✅ working |
| Partial | 30 | 32 | 13.3 min | 3,259 ms | ✅ working |
| Full | 78 | 32 | — | — | ❌ OOM at layer ~45 |
| Full | 78 | 64 | — | — | ❌ `index_n_heads=32` not divisible by TP=64 |

## Architecture (where things run)

```
                       Host CPU
              ┌────────────────────────┐
              │ vLLM scheduler / API   │
              │  + tokenizer           │
              │  + KV-cache mgmt       │
              └──────────┬─────────────┘
                         │
                  TP=32 dispatch
                         │
   ┌──────────┬──────────┴──────────┬──────────┐
   ▼          ▼                     ▼          ▼
 Core 0    Core 1   ........     Core 30   Core 31
 ┌────┐   ┌────┐                ┌────┐   ┌────┐
 │MLA │   │MLA │                │MLA │   │MLA │
 │MoE │   │MoE │  (each rank    │MoE │   │MoE │
 │DSA │   │DSA │   holds 1/32   │DSA │   │DSA │
 └────┘   └────┘   of weights)  └────┘   └────┘
```

## Patches applied (in this folder)

1. **`src/patch_registry.py`** — registers `GlmMoeDsaForCausalLM` in
   vLLM-Neuron's model registry, pointing at the DeepSeek V3.2 model
   class (architectures match).
2. **`src/patch_get_tensor_names.py`** — fixes an API drift between
   the DeepSeek V3.2 model code and the current
   `SafetensorsCheckpoint` (`get_tensor_names()` → `_tensor_name_to_file.keys()`).
3. **`config.json` quantization stripping** — vLLM-Neuron rejects
   `compressed-tensors` weight quantization; the loader auto-detects FP8
   from filenames and dequants correctly without the config block.
4. **`transformers` upgrade** to 5.12 — needed for `glm_moe_dsa` model
   type recognition.

The two new env vars:

```
NEURON_RT_VIRTUAL_CORE_SIZE=2   # standard for trn2
NEURON_SKIP_EFA_AFFINITY=1      # NEW: bypass EFA setup
                                # (lets vLLM-Neuron run on hosts/containers
                                #  without EFA passthrough)
```

The EFA-skip flag is a **hard-won finding**. The default vllm-neuron
worker init unconditionally probes `/sys/bus/pci/devices/{bdf}/infiniband`
and crashes if no EFA device is present (single-host containers,
non-EFA test instances, etc.). Setting `NEURON_SKIP_EFA_AFFINITY=1`
short-circuits both EFA-affinity and CPU-affinity setup. Workers fall
back to Gloo backend for distributed comm — works fine for single-host TP.

## Files

| File | Role |
|---|---|
| `src/glm5_config.py` | GLM 5.1 dataclass adapter (matches DeepSeek V3.2 shape with GLM dims) |
| `src/serve_glm5.py` | CLI runner (load + 1 prompt + TTFT) |
| `src/test_2layer.py` | 2-layer smoke validator |
| `src/patch_registry.py` | Adds GLM 5.1 + DeepSeek V3.2 to vLLM-Neuron registry |
| `src/patch_get_tensor_names.py` | API drift fix for SafetensorsCheckpoint |
| `BENCHMARK.md` | TTFT numbers + analysis |

## Reproduction (30 layers — known working)

```bash
# 1. SSH to a trn2.48xlarge with the vLLM-Neuron private-beta image pulled
ssh ubuntu@<your-trn2.48xl>

# 2. Start a container with /sys, /dev, and the model dir mounted
sudo docker run -d --name vllm_glm5 --privileged --net=host \
  --shm-size=64g \
  -v /mnt/data:/mnt/data -v /sys:/sys -v /dev:/dev \
  -e HF_HOME=/mnt/data/hf_cache \
  -e NEURON_RT_VIRTUAL_CORE_SIZE=2 \
  421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-28ce3c3:pytorch-2.10-inference-neuron-py312-sdk2.x.x-ubuntu24.04-neuron-ops-release-2.30-vllm-neuron-private-beta-trn10-v5 \
  sleep infinity

# 3. Upgrade transformers
sudo docker exec vllm_glm5 pip install --upgrade transformers

# 4. Install the DeepSeek V3.2 model code
#    (one-off — copy the deepseek_v32 folder from PR #2025 into
#    /opt/conda/lib/python3.12/site-packages/vllm_neuron/model/)

# 5. Apply the patches
sudo docker cp src/patch_registry.py vllm_glm5:/tmp/
sudo docker exec vllm_glm5 python /tmp/patch_registry.py
sudo docker cp src/patch_get_tensor_names.py vllm_glm5:/tmp/
sudo docker exec vllm_glm5 python /tmp/patch_get_tensor_names.py

# 6. Strip the compressed-tensors block from config.json
sudo docker exec vllm_glm5 python -c "
import json
p = '/mnt/data/hf_cache/models--mconcat--GLM-5.1-FP8-Dynamic/snapshots/<hash>/config.json'
c = json.load(open(p)); c.pop('quantization_config', None)
json.dump(c, open(p, 'w'), indent=2)"

# 7. Run inference (30 layers smoke — known working)
sudo docker cp src/serve_glm5.py vllm_glm5:/tmp/
sudo docker exec -e NEURON_RT_VIRTUAL_CORE_SIZE=2 -e NEURON_SKIP_EFA_AFFINITY=1 \
  vllm_glm5 python /tmp/serve_glm5.py --num-hidden-layers 30 --tp 32
```

Expected on a fresh box: ~13 minute first-time compile, then a TTFT
around 3.3 seconds for "The future of AI is" → 16-token completion.

## Known issues

1. **OOM at full 78 layers, TP=32, BF16.** Per-core HBM budget is ~24 GB.
   Full 78-layer BF16 weights at TP=32 = ~47 GB/core. Loads up to layer
   ~45 then crashes with `nrt_tensor_allocate status=4`.

2. **TP=64 blocked by `index_n_heads=32`.** GLM 5.1's DSA indexer has
   only 32 heads (vs DeepSeek V3.2's 64). Cannot evenly shard across 64
   ranks. Need a config override or model-side adapter.

3. **TTFT is high.** Even at 30 layers we see 3.3s — far over the
   500ms target. Most of the time is in MoE routing + DSA indexer
   compute. Optimization roadmap: EP, FP8 matmul kernels, NKI
   selection-bias router.

## Next steps

1. **FP8-on-device weight storage.** Patch the weight loader so FP8
   weights stay FP8 in HBM (don't auto-cast to BF16). Reference: PR
   #1987 (Qwen3-235B FP8). Halves the per-core memory and unblocks the
   full 78 layers.
2. **Expert parallelism (EP=4 or EP=8).** Distribute the 256 experts
   across ranks instead of replicating. Reduces per-rank MoE compute
   by 4×–8×.
3. **TP=64 unblocking.** Either accept that some indexer ranks are
   idle, or upstream a config flag that scales the indexer differently
   from attention.
4. **Multi-bucket compile** for 256 / 512 / 1K / 2K / 4K / 8K seq lens.

## License

Apache 2.0 for our code. GLM 5.1 weights are governed by the upstream
Z.ai / Tencent Hunyuan license — see the model card.
