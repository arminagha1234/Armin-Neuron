# Qwen3.5-4B on vLLM-Neuron 0.21

Serving Qwen3.5-4B (hybrid GatedDeltaNet + attention: 24 `linear_attention` +
8 `full_attention` layers) took eleven attempts and six distinct fixes. All six
are listed with their exact error text, because none of them were guessable from
the symptom.

Final state: **coherence 3/3** — *"The capital of France is **Paris**."*,
*"The largest planet in our solar system is **Jupiter**."*, *"4"*.

## The model package

`install_qwen35.sh` installs a `qwen3_5` package into `vllm_neuron/model/` and
registers it in `registry.py` **on disk**.

The on-disk part matters. An in-process `registry.get_models()` monkeypatch does
not survive: vLLM-Neuron spawns `Worker_TP0..N` via `multiproc_executor`, each of
which re-imports `vllm_neuron.model.registry` fresh and never sees the patch.
The workers then resolve vLLM's built-in stub and fail with
`AttributeError: type object 'Qwen3_5ForConditionalGeneration' has no attribute
'from_configs'`. A source-level registration is inherited by every subprocess.

The installer also disables the auto-`register()` call in the *installed*
`__init__.py`, because `registry.py` imports the package directly and calling
`register()` during that import is circular.

## The six blockers

### 1. Use the ORIGINAL config, not a text-only rewrite

Promoting `text_config` to top level makes `AutoConfig` build a
`Qwen3_5TextConfig`, and vLLM's platform check rejects it:

```
TypeError: Invalid type of HuggingFace config.
  Expected: vllm.transformers_utils.configs.qwen3_5.Qwen3_5Config
  Found:    transformers.models.qwen3_5.configuration_qwen3_5.Qwen3_5TextConfig
```

The rewrite was not merely unnecessary — `qwen3_5/config.py:from_configs`
already prefers the nested `text_config` block. Feed it the untouched config.

### 2. `from_configs` must accept `text_neuron_config`

With the wrapper config, vLLM treats the model as multimodal:

```
TypeError: Qwen3_5ForConditionalGeneration.from_configs() got an unexpected
keyword argument 'text_neuron_config'. Did you mean 'neuron_config'?
```

Accept it and treat it as the decoder's `neuron_config`, in both `factory.py`
and `model_bf16.py`.

### 3. Do not verify a patch by importing the patched module

Confirming the patch with an import after
`sys.path.insert(0, ".../site-packages/vllm_neuron")` shadowed the real `vllm`
package with `vllm_neuron/vllm/`:

```
ModuleNotFoundError: No module named 'vllm.logger'
```

All three patches had applied correctly; the *check* aborted the run. Verify by
AST instead — no imports, no side effects.

### 4. `forward` must tolerate the runner's wrapper-config kwargs

```
TypeError: Unexpected keyword arguments:
  ['rotary_position_ids', 'vision_embedding_blocks', 'vision_positions']
```

Qwen3.5 is text-only and uses standard RoPE keyed off `positions`, so accept and
ignore all three.

### 5. Implement `SupportsMRoPE`

The wrapper config makes vLLM set `uses_mrope=True`, and it then demands the
protocol at inference time — after a ~1400 s compile:

```
TypeError: Model Qwen3_5ForConditionalGeneration sets uses_mrope=True but does
not implement the SupportsMRoPE protocol.
```

For a text-only model this is trivial: return `[3, seq_len]` with all three
sections equal to the position ramp. `forward` ignores it anyway.

### 6. Do not let a patch steal the `@torch.no_grad()` decorator

The subtlest one. Anchoring an insertion on `def forward(` places the new method
**between** the decorator and the function, silently transferring
`@torch.no_grad()` to the inserted method and leaving `forward` undecorated. The
KV cache then carries a `grad_fn` and the in-place paged write fails to trace:

```
Dynamo failed to run FX node with fake tensors: call_method index_put_(
  FakeTensor(size=(70, 1, 32, 256), dtype=bfloat16, grad_fn=<AsStridedBackward0>), ...)
  at model_bf16.py:505  self.v_cache.index_put_(...)
```

Include the decorator in the anchor, and assert afterwards that exactly one
`@torch.no_grad()`-decorated `forward` remains.

### Also: `kv_segment_size_buckets` is required

Omitting it yields `kv_segment_size=0`, the KV cache is allocated degenerate
(`v_cache` came out `(70, 1, 32, 256)`), and the prefill write cannot trace.

## Serving config that works

```
--tensor-parallel-size 4 --max-model-len 2048 --max-num-seqs 4
--max-num-batched-tokens 2048 --no-enable-prefix-caching
--additional-config '{"neuron_config":{
    "num_batched_tokens_buckets":[2048],
    "kv_segment_size_buckets":[2048],
    "num_seqs_buckets":[4],
    "on_device_sampling_config":{"all_greedy":true}}}'
```

Compile is ~1450 s and the NEFF cache reaches 2.3 GB.

## Measured — 1877 prompt tokens / 50 output, TP=4

| conc | RPS | avg latency |
|---:|---:|---:|
| 1 | 0.157 | 6.38 s |
| 2 | 0.157 | 9.56 s |
| 4 | 0.157 | 15.93 s |
| 8 | 0.157 | 28.68 s |

Flat, because `max_num_seqs=4` caps batching — beyond 4 requests simply queue.
Decode is ~7.8 tok/s, so 50 output tokens cost 6.4 s against prefill's 0.12 s.

**`max_num_seqs=32` does not fit the compile budget.** It was still actively
compiling at 1561 s (NEFF cache 2.3 GB -> 9.1 GB, compiler activity lines still
climbing) when the ~31 min pod wall hit. Not an error — a compile-time limit.
Anyone with a longer wall should raise `max_num_seqs`; decode is
memory-bandwidth-bound and batching is the lever that matters here.

## Gotcha: this model false-negatives coherence gates

Qwen3.5 emits a `Thinking Process:` preamble. At `max_tokens=32` all three
probes were truncated before the answer and scored **0/3** on a model that was
completely correct. Pass `chat_template_kwargs: {"enable_thinking": false}` or
raise the budget well past the preamble.
