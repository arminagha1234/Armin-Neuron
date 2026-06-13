# Gemma 4 E4B on vLLM-Neuron

## Status: Working at TP=2, partial quality

Server serves end-to-end with PLE enabled. Produces coherent English sentences
for some prompts at TP=2. Garbage at TP=4 (KV replication bug).

## Architecture

E4B is NOT Gemma3n (no AltUp/Laurel). It IS the Gemma4 architecture with:

| Component | Value | Notes |
|---|---|---|
| hidden_size | 2560 | |
| num_hidden_layers | 42 | |
| num_attention_heads | 8 | |
| num_key_value_heads | 2 | |
| head_dim (SWA) | 256 | sliding_window=512 |
| head_dim (Global) | 512 | partial_rotary_factor=0.25 |
| hidden_size_per_layer_input | 256 | PLE (required!) |
| num_kv_shared_layers | 18 | layers 24-41 |
| attention_k_eq_v | **False** | (31B has True) |
| num_global_key_value_heads | **None** | falls back to num_key_value_heads |
| altup_num_inputs | None | NO AltUp |
| laurel_rank | None | NO Laurel |

## Patches Required (5 total)

1. **Config None-handling** — `get_layer_num_kv_heads` returns `num_key_value_heads` when `num_global_key_value_heads` is None
2. **GeLU inline** — replace `F.gelu(x, approximate="tanh")` with explicit tanh formula (Dynamo can't trace torch._C._nn.gelu)
3. **Force-torch MLP** — `_can_use_kernel()` returns False (NKI MLP kernel hits ISA bug at E4B's shapes)
4. **Transformers stub** — register `gemma4` model_type in AutoConfig
5. **vllm.ModelRegistry hook** — register `Gemma4ForConditionalGeneration` as text-gen model

## The `num_kv_replicas` Bug (TP=4 failure)

At TP=4 with only 2 KV heads:
- `world_size (4) > num_kv_heads (2)` → `num_kv_replicas = 4/2 = 2`
- Ranks 0,1 share KV head 0; ranks 2,3 share KV head 1
- The fused QKV weight loader correctly replicates K/V weights (verified)
- But something in the attention forward or reduce_scatter/all_reduce is wrong for this config

At TP=2:
- `world_size (2) == num_kv_heads (2)` → `num_kv_replicas = 1`
- Clean 1:1 mapping — works correctly

**Workaround:** serve at TP=2.

## Reproduction (trn2.48xlarge)

```bash
# In the vllm_neuron container (v5 image):
export PYTHONPATH=/work
export NEURON_SKIP_EFA_AFFINITY=1

# Apply patches (see src/patches.md)
# Copy model_e4b_ple.py → /opt/conda/.../vllm_neuron/model/gemma4/model.py
# Copy config_e4b_ple.py → /opt/conda/.../vllm_neuron/model/gemma4/config.py

vllm serve /root/models/gemma-4-E4B-it \
    --tensor-parallel-size 2 \
    --max-model-len 128 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 128 \
    --additional-config '{"neuron_config":{"num_batched_tokens_buckets":[128],"num_seqs_buckets":[1],"on_device_sampling_config":{"all_greedy":true}}}'
```

## Known Issues

1. **TP=4 garbage** — `num_kv_replicas=2` path produces wrong attention output
2. **PLE memory** — embed_tokens_per_layer is 5.3GB; needs TP≥2 to fit
3. **No NKI prefill kernel for head_dim>128** — uses torch fallback SDPA
4. **Some prompts still produce garbage at TP=2** — likely related to global attention layers (head_dim=512) or KV-sharing layers 24-41

## Files

| File | Role |
|---|---|
| `src/model_e4b_ple.py` | Full patched model.py (1657 lines) with PLE |
| `src/config_e4b_ple.py` | Config with E4B fields + k_eq_v fix |
| `src/patches.md` | The 5 base patches (copy-pasteable) |
| `results/output_samples.md` | Actual model outputs at TP=2 |
