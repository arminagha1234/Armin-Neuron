# DeepSeek V3.2 (685B MoE) on vLLM-Neuron — Status Report

Deploy notes, working steps, and the open compile bottleneck for serving
**DeepSeek-V3.2** (deepseek-ai/DeepSeek-V3.2 — 671B total, ~37B active per token,
MLA + DSA Lightning Indexer + 256+1-expert MoE) on AWS Trainium2.

**Status as of 2026-06-04: Model loads, 2-layer smoke generates tokens
end-to-end, but the full 61-layer compile gets stuck for >50 minutes in
the FX `InplaceRewritePass`. Documented for upstream follow-up.**

---

## What works

| Step | Result |
|---|---|
| 1. Code injection (PR #2025 model code) | ✅ Verified, all imports clean |
| 2. SafetensorsCheckpoint shim (`get_tensor_names()`) | ✅ Method added to v5 beta image |
| 3. Model registry wiring (`DeepseekV32ForCausalLM`) | ✅ Listed alongside Llama, Qwen, GptOss |
| 4. FP8 checkpoint download (192 files, 643 GB) | ✅ ~42 min from HF Hub |
| 5. **2-layer smoke** (TP=64, NUM_LAYERS=2) | ✅ **Loads in 1.7 min, generates 10 tokens** |
| 6. **Full 61-layer compile** | ❌ **Stalls in FX `_update_subsequent_ops` after weight load** |

## 2-layer smoke result

```
DeepSeek V3.2 — 2-layer smoke (TP=64)
MODEL LOAD: 105s (1.7 min)
'The capital of France is' -> 'oint054 Rund959arki survivor-angopisnickelsen'  (10 tok, 311 ms)
SMOKE TEST PASSED (2 layers)
```

Output is gibberish because 2 layers can't form coherent text — that's expected
and matches AWS's docstring on `run.py`. The point of the 2-layer test is to
prove the model class registers, FP8 weights load, and the compiler emits a
working NEFF for the MLA + DSA + MoE block. **All three pass.**

## The 61-layer compile bottleneck

After ~7 minutes of weight loading (BF16 dequant from FP8 + per-rank sharding,
~207 sec/worker), all 64 workers enter graph capture and stall in:

```
_update_subsequent_ops (vllm_neuron/fx_passes/inplace_rewrite_pass.py:528)
_convert_inplace_ops (vllm_neuron/fx_passes/inplace_rewrite_pass.py:111)
run (vllm_neuron/fx_passes/inplace_rewrite_pass.py:44)
run_passes (vllm_neuron/fx_passes/pass_manager.py:60)
_run_fx_passes (vllm_neuron/compile/capture_backend.py:180)
```

After >50 minutes of pegged 150% CPU per worker (multi-thread Python GIL
contention), **no `neuronx-cc` subprocess has spawned and the compile cache
remains at "Local cache miss"**. The InplaceRewritePass is iterating over
the FX graph nodes with what appears to be O(N²) scaling — `_update_subsequent_ops`
does a linear scan through `gm.graph.nodes[start_idx:]` per inplace operation,
and there are tens to hundreds of thousands of nodes for a 61-layer × 256-expert MoE.

```python
# vllm_neuron/fx_passes/inplace_rewrite_pass.py:520-540
def _update_subsequent_ops(self, gm, modified_node, original_input):
    nodes_list = list(gm.graph.nodes)
    start_idx = nodes_list.index(modified_node) + 1   # O(N) lookup
    for later_node in nodes_list[start_idx:]:          # O(N) scan
        ...
```

Called once per inplace op — for a graph with ~K inplace ops and ~N total
nodes, total cost is O(K × N). At 61L × 256 experts, both K and N are very
large.

## Verified state at the time of stall

- **Host RAM:** 203 GiB used out of 2 TB (1.7 TB free, no swap touched, no OOM)
- **Container memory:** healthy, all 64 workers responding to py-spy probes
- **Neuron devices:** all 16 chips × 4 cores claimed by the workers (no driver issues)
- **No exceptions in the log** — workers are stuck in user code, not crashed
- **Last log activity:** 00:29:59 UTC — workers all reached `Local cache miss` then
  silenced into the FX pass loop. No log lines for the next 50+ minutes.

The bottleneck is NOT host memory, NOT NEFF compile, NOT shard memory.
It's a **single-threaded Python algorithm in the FX graph rewrite pass**.

## What to try next (not done in this run)

1. **Disable `enable_chunked_prefill`** in vLLM args. The `splitting_ops` list
   adds dispatch nodes that may multiply the inplace-rewrite work. Run with
   `--enable-chunked-prefill=False`.

2. **Set `enforce_eager=True`** to bypass `torch.compile` entirely. We'd lose
   compile-time fusion, but get a measurable TTFT for the MLA+DSA+MoE forward
   pass.

3. **Try the newer vllm-neuron beta image** (we tested on `v5`, AWS team
   mentioned newer revisions). The InplaceRewritePass may have been
   optimized post-PR-#2025-merge.

4. **Patch `_update_subsequent_ops`** to use a `dict[Node, set[Node]]` index
   from node to its uses, then update only those uses — converts O(N) per
   call to O(1). This is the right algorithmic fix.

## Cost incurred for this run

| Phase | Duration | Cost @ $21.50/hr |
|---|---:|---:|
| Setup (code sync, registry patch, swap) | ~10 min | $4 |
| Download FP8 checkpoint (643 GB) | 42 min | $15 |
| 2L smoke (load + run) | 2 min | $1 |
| 61L stall + investigation | ~70 min | $25 |
| **Total** | **~125 min** | **~$45** |

## Reproduction

Inside the vLLM-Neuron v5 beta container on a `trn2.48xlarge`:

### 1. Inject PR #2025 model code

```bash
# Pull canonical files from PR #2025 head SHA 9b41195e16d1a7467511a78a4912d703d79ce780
# Files needed (192 LOC sum across model files):
# - vllm_neuron/model/deepseek_v32/__init__.py        (8 lines)
# - vllm_neuron/model/deepseek_v32/config.py          (131 lines)
# - vllm_neuron/model/deepseek_v32/factory.py         (39 lines)
# - vllm_neuron/model/deepseek_v32/model.py           (2185 lines)
# - vllm_neuron/model/deepseek_v32/weight_loader.py   (276 lines)
# - vllm_neuron/model/__init__.py                     (modified +1 -1, adds 'deepseek_v32' to lazy-import tuple)

sudo docker cp deepseek_v32/ vllm_neuron:/opt/conda/lib/python3.12/site-packages/vllm_neuron/model/
sudo docker cp model_init.py vllm_neuron:/opt/conda/lib/python3.12/site-packages/vllm_neuron/model/__init__.py
```

### 2. Patch registry.py (add DeepseekV32ForCausalLM to get_models())

```python
# In /opt/conda/lib/python3.12/site-packages/vllm_neuron/model/registry.py
# Add: from .deepseek_v32 import DeepseekV32ForCausalLM
# Add: ("DeepseekV32ForCausalLM", DeepseekV32ForCausalLM) to the models list
```

### 3. Apply SafetensorsCheckpoint shim

The v5 beta's `vllm_neuron.utils.checkpoints.SafetensorsCheckpoint` is missing
`get_tensor_names()` which PR #2025 calls. Add it (see `apply_shim_local.py`):

```python
def get_tensor_names(self) -> list[str]:
    self._ensure_indexed()
    return list(self._tensor_name_to_file.keys())
```

### 4. Add NVMe swap (defensive — prior DeepSeek V3-0324 attempts hit 2.08 TB peak)

```bash
sudo fallocate -l 256G /swap.img
sudo chmod 600 /swap.img
sudo mkswap /swap.img
sudo swapon /swap.img
```

PR #2025's streaming weight loader doesn't actually need this for V3.2 — peak
host RAM during load is only ~200 GB. The swap is insurance.

### 5. Smoke test (2 layers)

```bash
sudo docker exec -d vllm_neuron bash -c "
  NUM_LAYERS=2 python3 smoke.py 2>&1 | tee /work/smoke_2l.log
"
```

Should output: `MODEL LOAD: ~105s` and `SMOKE TEST PASSED (2 layers)`.

### 6. Full 61L (currently blocked)

```bash
sudo docker exec -d vllm_neuron bash -c "
  NUM_LAYERS=61 python3 smoke.py 2>&1 | tee /work/full_61l.log
"
```

Will stall in `vllm_neuron/fx_passes/inplace_rewrite_pass.py`. See
`logs/61l_stall_pyspy.txt` for the reproducible stack trace.

## Files

```
.
├── README.md                      # This file
├── smoke.py                       # 2L/61L entry point (NUM_LAYERS env var)
├── apply_shim_local.py            # SafetensorsCheckpoint.get_tensor_names patch
├── registry_patched.py            # registry.py with DeepseekV32 added
├── deepseek_v32/                  # PR #2025 model code (192 LOC sum)
│   ├── __init__.py
│   ├── config.py                  # 131 lines
│   ├── factory.py                 # 39 lines
│   ├── model.py                   # 2185 lines (MLA + DSA + MoE)
│   └── weight_loader.py           # 276 lines (FP8 → BF16 streaming dequant)
└── logs/
    ├── 2l_smoke_pass.txt          # The successful 2L run
    ├── 61l_stall_pyspy.txt        # py-spy dump of the stuck worker
    └── 61l_log_excerpt.txt        # Last useful log lines before silence
```

## References

- **PR #2025** — https://github.com/aws-neuron/private-vllm-neuron/pull/2025
  ("feat: DeepSeek V3.2 BF16 (MLA + DSA + MoE) for trn2") — closed, head SHA
  `9b41195e16d1a7467511a78a4912d703d79ce780`
- **PR #2062** — https://github.com/aws-neuron/private-vllm-neuron/pull/2062
  ("Deepseek ref") — module-level pytest tests, depends on #2025
- **Model on HF** — https://huggingface.co/deepseek-ai/DeepSeek-V3.2 (not gated)
- **Customer instance** — `i-0d55a7514c80f5075` (us-east-2 trn2.48xlarge,
  vllm_neuron container running v5 beta image, hostname ec2-52-15-96-92)

---

## License

Apache-2.0 (model code from PR #2025 is upstream Apache-2.0)
