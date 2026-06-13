# Required Patches for E4B on vLLM-Neuron v5

Apply these to the container's site-packages BEFORE serving.

## Patch 1: Config None-handling

File: `/opt/conda/lib/python3.12/site-packages/vllm_neuron/model/gemma4/config.py`

Change `num_global_key_value_heads: int = 4` to `int | None = 4`

In `get_layer_num_kv_heads`:
```python
# Before:
return self.num_global_key_value_heads

# After:
v = self.num_global_key_value_heads
return v if v is not None else self.num_key_value_heads
```

Also set `attention_k_eq_v: bool = False` (was True for 31B).

## Patch 2: GeLU inline

File: `/opt/conda/lib/python3.12/site-packages/vllm_neuron/functional/mlp.py`

Line ~274, replace:
```python
return lambda x: torch.nn.functional.gelu(x, approximate="tanh")
```
with:
```python
return lambda x: 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
```

## Patch 3: Force-torch MLP

Same file, add before `_can_use_kernel`:
```python
def _can_use_kernel(*args, **kwargs):
    return False
```
(rename the original to `_ORIG_can_use_kernel`)

## Patch 4: Transformers stub

File: `/work/gemma4_transformers_stub.py` (loaded via sitecustomize)

Registers `gemma4` and `gemma4_text` model types in transformers CONFIG_MAPPING.

## Patch 5: Model registration

File: `/work/register_gemma4.py` (loaded via sitecustomize)

Registers `Gemma4ForConditionalGeneration` in both vllm_neuron and vLLM registries
with `is_text_generation_model=True`.
