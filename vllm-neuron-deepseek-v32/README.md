# DeepSeek V3.2 (671B MoE) on vLLM-Neuron — Status Report

Deploy notes, working steps, four required patches, and the open
decode-runtime failure when serving **DeepSeek-V3.2** (deepseek-ai/DeepSeek-V3.2 —
671B total, ~37B active per token, MLA + DSA Lightning Indexer + 256+1-expert MoE)
on AWS Trainium2.

**Status as of 2026-06-04: Model loads, prefill NEFFs execute (bucket=128
and bucket=1024 verified), but the decode NEFF fails to schedule on the
Neuron runtime at every sequence length tested. Documented for upstream follow-up.**

---

## What works (4 of 5 phases)

| Step | Result |
|---|---|
| 1. Code injection (PR #2025 model code) | ✅ Verified, all imports clean |
| 2. SafetensorsCheckpoint shim (`get_tensor_names()`) | ✅ Method added to v5 beta image |
| 3. Model registry wiring (`DeepseekV32ForCausalLM`) | ✅ Listed alongside Llama, Qwen, GptOss |
| 4. FP8 checkpoint download (192 files, 643 GB) | ✅ ~42 min from HF Hub |
| 5. **2-layer smoke** (TP=64, NUM_LAYERS=2, max_model_len=128) | ✅ **Loads in 1.7 min, generates 10 tokens** |
| 6. **8-layer FX graph + HLO compile** | ✅ 4-hour compile completed across 64 ranks |
| 7. **Prefill NEFF execution** (bucket=128, bucket=1024) | ✅ "Successfully warmed up for prefill bucket 1024" |
| 8. **Decode NEFF execution** | ❌ `status=1 Unknown Failure` at every seq tested |

## What we did to get this far

The deploy required four runtime patches against the v5 beta image to even
reach the decode warmup step. Each is a real upstream issue:

### Patch 1 — `SafetensorsCheckpoint.get_tensor_names()` shim
PR #2025 calls `checkpoint.get_tensor_names()` to enumerate FP8/BF16 keys
for inline dequant. The v5 beta image's `vllm_neuron.utils.checkpoints`
predates this method. We add it back in 5 lines:

```python
def get_tensor_names(self) -> list[str]:
    self._ensure_indexed()
    return list(self._tensor_name_to_file.keys())
```

See `apply_shim_local.py`.

### Patch 2 — `InplaceRewritePass` O(N²) → O(K)
The FX pass `vllm_neuron/fx_passes/inplace_rewrite_pass.py` has
`_update_subsequent_ops` doing a linear scan through `gm.graph.nodes`
per inplace op. For DeepSeek V3.2 (61L × 256 experts → ~100k+ FX nodes,
hundreds of inplace ops), this is **O(N×K)** and takes hours.

`original_input.users` is an O(1)-lookup dict that PyTorch FX maintains
automatically. We use it instead:

```python
# was: linear scan of all subsequent nodes
nodes_list = list(gm.graph.nodes)
start_idx = nodes_list.index(modified_node) + 1
for later_node in nodes_list[start_idx:]:  # O(N) per call
    ...

# is: direct lookup of users
node_index = {n: i for i, n in enumerate(gm.graph.nodes)}
modified_idx = node_index[modified_node]
for later_node in original_input.users.keys():  # O(K) per call
    if node_index.get(later_node, -1) > modified_idx:
        ...
```

Reduces the FX rewrite pass from hours to seconds. See `patch_fx_pass.py`.

### Patch 3 — `tp_barrier()` no-op
Even with the FX pass fixed, the 64-rank TP barrier was deadlocking after
all ranks reached it. Two layers of timeout:
1. Python `timedelta(1800s)` — bumped to 14400s, didn't help
2. **gloo TCP `unbound_buffer` internal 1800ms cap** — gloo's `barrier().wait()`
   with all 64 peers issuing the op repeatedly desyncs internal state

A polled retry loop didn't recover. **No-opping the barrier worked**:
the next inference op (TP all-reduce in forward) naturally synchronizes
all ranks, so the explicit barrier is not strictly required.

```python
def tp_barrier(timeout=...):
    return  # no-op
```

See `patch_barrier_noop.py`.

### Patch 4 — Cap `max_model_len` at 1024
PR #2025's `dsa_max_seq_len = 3072`, but at `max_model_len=4096` the model's
DSA Lightning Indexer's `slice_scatter` errors with shape mismatch:

```
RuntimeError: expand: attempting to expand a dimension of length 4096 -> 3072!
```

Cap `max_model_len <= 3072` (we use 1024 for HBM safety, see Patch 5 below).

### "Patch 5" — HBM at seq=2048 needs more cores
Even at 8 layers, prefill bucket=2048 OOMs:
```
RuntimeError: Could not load the model status=4 message=Allocation Failure
```
At TP=64, the per-core HBM budget can't fit weights + KV cache + activations
at seq=2048. We drop to seq=1024 to stay under the cliff.

## The wall: decode NEFF scheduling

After all 4 patches and the seq=1024 cap, prefill execution succeeds:

```
Successfully warmed up for prefill bucket 128 with kv_segment_size 0
Successfully warmed up for prefill bucket 1024 with kv_segment_size 0
```

But decode warmup then fails with a **Neuron Runtime error**, not a Python error:

```
RuntimeError: Model warmup failed for decode (batch=1, seq=1024).
Error: Failed to schedule neff execution. status=1 message=Unknown Failure
```

Same failure at seq=512. This is below the Python layer where our patches
operate — the NEFF is loaded onto the NeuronCore but the runtime's `execute()`
syscall returns status=1.

We can't fix this from outside the runtime. See `logs/decode_failure.txt`
for the full investigation.

## Cost of this investigation

| Phase | Duration | Cost @ $21.50/hr |
|---|---:|---:|
| Setup (code, shim, registry, swap) | ~15 min | $5 |
| Download FP8 checkpoint (643 GB) | 42 min | $15 |
| 2L smoke pass | 2 min | $1 |
| 61L stall investigation (FX pass) | ~70 min | $25 |
| FX pass patch + 8L compile retry | ~4 hr | $86 |
| Barrier debugging + no-op + retry | ~3 hr | $65 |
| Decode NEFF failure across multiple sizes | ~30 min | $11 |
| **Total** | **~9.3 hr** | **~$210** |

## What to try next (not done in this run)

1. **Try a newer vLLM-Neuron beta image** — the `v5` we tested may
   predate fixes for this exact decode path. Newer images may also
   incorporate a fix for the FX rewrite O(N²).
2. **Capture decode NEFF execution profile** with
   `NEURON_RT_INSPECT_DEVICE_PROFILE=1` to get a stack trace from
   the Neuron runtime side instead of just `status=1`.
3. **Test at TP=32** (half the world size) to see if the schedule
   failure is per-core resource exhaustion vs a real bug.
4. **Use a pre-converted BF16 checkpoint** (skip the FP8 dequant
   path entirely). PR #2025 supports both — the BF16 path may
   compile a different decode NEFF.
5. **Disable the DSA Lightning Indexer** (set `dsa_max_seq_len=0`
   in config) — the Indexer is the most novel piece of V3.2. If it's
   the cause of the decode failure, the model would still serve
   without it (just no long-context advantage).

## Files

```
.
├── README.md                      # This file
├── smoke.py                       # 2L/61L entry (NUM_LAYERS env var)
├── smoke_2k.py                    # max_model_len=2048 entry (capped to 3072 DSA limit)
├── apply_shim_local.py            # Patch 1: SafetensorsCheckpoint.get_tensor_names
├── patch_fx_pass.py               # Patch 2: InplaceRewritePass O(N^2) -> O(K)
├── patch_barrier_noop.py          # Patch 3: tp_barrier / world_barrier -> no-op
├── registry_patched.py            # Drop-in registry.py with DeepseekV32ForCausalLM added
├── model_init.py                  # Drop-in vllm_neuron/model/__init__.py
├── deepseek_v32/                  # PR #2025 model code (5 files)
│   ├── __init__.py
│   ├── config.py                  # 131 lines
│   ├── factory.py                 # 39 lines
│   ├── model.py                   # 2185 lines
│   └── weight_loader.py           # 276 lines
└── logs/
    ├── 2l_smoke_pass.txt          # The 2-layer success
    ├── 61l_stall_pyspy.txt        # Original FX pass stall (now fixed by Patch 2)
    ├── 61l_log_excerpt.txt        # Compile timing data
    └── decode_failure.txt         # The wall we hit (status=1 Unknown Failure)
```

## Reproduction

```bash
# Inside the vLLM-Neuron v5 beta container on a trn2.48xlarge:

# 1. Download model code (assumes you have GitHub access)
SHA=9b41195e16d1a7467511a78a4912d703d79ce780
git clone https://github.com/aws-neuron/private-vllm-neuron.git
cd private-vllm-neuron && git checkout $SHA
sudo docker cp vllm_neuron/model/deepseek_v32 vllm_neuron:$PYTHONPATH/vllm_neuron/model/
sudo docker cp vllm_neuron/model/__init__.py vllm_neuron:$PYTHONPATH/vllm_neuron/model/

# 2. Apply all four patches
sudo docker cp apply_shim_local.py vllm_neuron:/tmp/
sudo docker cp patch_fx_pass.py vllm_neuron:/tmp/
sudo docker cp patch_barrier_noop.py vllm_neuron:/tmp/
sudo docker cp registry_patched.py vllm_neuron:$PYTHONPATH/vllm_neuron/model/registry.py
sudo docker exec vllm_neuron python3 /tmp/apply_shim_local.py
sudo docker exec vllm_neuron python3 /tmp/patch_fx_pass.py
sudo docker exec vllm_neuron python3 /tmp/patch_barrier_noop.py

# 3. Download model weights (642 GB, ~42 min on 10Gbit network)
sudo docker exec vllm_neuron python3 -c "
from huggingface_hub import snapshot_download
import os
snapshot_download('deepseek-ai/DeepSeek-V3.2',
                   token=os.environ['HF_TOKEN'], max_workers=8)
"

# 4. Smoke test (2L) — works
NUM_LAYERS=2 python3 smoke.py
# Expect: "MODEL LOAD: 105s", "SMOKE TEST PASSED (2 layers)"

# 5. Larger compile (8L, max_model_len=1024)
NUM_LAYERS=8 MAX_LEN=1024 python3 smoke_2k.py
# Expect: prefill warmups succeed, decode warmup fails with status=1
```

## References

- **PR #2025** — https://github.com/aws-neuron/private-vllm-neuron/pull/2025
  ("feat: DeepSeek V3.2 BF16 (MLA + DSA + MoE) for trn2") — closed (not merged),
  head SHA `9b41195e16d1a7467511a78a4912d703d79ce780`
- **PR #2062** — https://github.com/aws-neuron/private-vllm-neuron/pull/2062
  ("Deepseek ref") — module-level pytest tests, depends on #2025
- **Model on HF** — https://huggingface.co/deepseek-ai/DeepSeek-V3.2 (not gated)
- **Test instance** — `i-0d55a7514c80f5075` (us-east-2 trn2.48xlarge,
  vllm_neuron container running v5 beta image, hostname ec2-52-15-96-92)

---

## License

Apache-2.0 (model code from PR #2025 is upstream Apache-2.0)
