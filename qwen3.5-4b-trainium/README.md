# Qwen3.5-4B on AWS Trainium2 via vLLM-Neuron

Working reference implementation of `Qwen/Qwen3.5-4B` on Trainium2,
serving over `vllm serve` through the vllm-neuron container. The
model is Alibaba's hybrid **GatedDeltaNet + GQA** architecture
(8 full-attention layers + 24 linear-attention layers, head_dim=256,
`attn_output_gate=True`, `(1+weight)` RMSNorm) and is not stock-supported
by vllm-neuron.

This adapter builds on top of vllm-neuron Beta v5 and adds:

- A native `Qwen3_5ForConditionalGeneration` registration that swaps
  vllm's lazy stub for our implementation.
- A weight loader that handles Qwen3.5's per-head spliced `q_proj`
  (interleaved `[h_q | h_gate]` per head, with `attn_output_gate=True`).
- The `(1+weight)` RMSNorm convention used throughout the network.
- A tile-based DeltaNet kernel (PR #152's NKI fused chunked forward)
  for prefill, and a recurrent-step decode path.
- **State persistence for DeltaNet via non-zero-init nn.Buffer** —
  see "The state-persistence fix" below.

## Status

✅ **Validated correctness** at TP=4, MAX_LEN=512 on `trn2.48xlarge`
(vllm-neuron Beta v5 container). Output is coherent and factually
correct across varied prompts.

| Test | Result |
|------|--------|
| "The capital of France is" → first 200 tok | structured `<think>` analysis acknowledging Paris |
| "Calculate 17 times 23" | "= 391", with two methods shown |
| "1 to 30, one per line" | perfect 1→30, no drift |
| "Why is the sky blue?" (480 tok) | accurate Rayleigh-scattering essay with the `1/λ⁴` formula and correct wavelengths |
| "Largest country / planet / ..." | correct: Russia, Jupiter, etc. |

### Baseline performance (TP=4, MAX_LEN=512, single-shot prefill, decode batch=1)

| Input tokens | TTFT (s) | Decode tok/s |
|---:|---:|---:|
| 100 | 1.83 | 18.4 |
| 200 | 1.83 | 18.4 |
| 300 | 1.83 | 18.4 |

TTFT is flat across input lengths up to ~500 tokens — vllm-neuron's
chunked-prefill scheduler has a fixed floor cost for the prefill NEFF.
Decode throughput is independent of input length at this scale.

Numbers for MAX_LEN=4096 (single-shot 4k prefill) and the customer
20k-input shape (chunked prefill at 4k) are pending — recompile
takes ~30 min and benchmarks will be added in a follow-up commit.

## Quickstart

Inside the vllm-neuron Beta v5 container, with this folder on
`PYTHONPATH`:

```bash
# Short context (single-shot prefill, fastest compile)
TP=4 MAX_LEN=512 PORT=8000 \
  MODEL=/path/to/Qwen3.5-4B \
  ./src/serve.sh

# Medium context (single-shot 4K prefill)
TP=4 MAX_LEN=4096 PORT=8000 \
  MODEL=/path/to/Qwen3.5-4B \
  ./src/serve.sh

# Long context (chunked prefill on 4K chunks — customer shape)
TP=4 MAX_LEN=32768 BUCKET=4096 \
  MODEL=/path/to/Qwen3.5-4B \
  ./src/serve.sh
```

## The state-persistence fix

The first cut of this adapter (the same pattern used in the related
27B port and in NxDI's PR #152) stored DeltaNet's recurrent state and
conv1d state in side-channel `nn.Buffer`s **initialized to zeros**, and
mutated each step via `buffer.data.copy_(new_value)` plus a
`+ buffer * 0` residual trick.

**This silently fails on Neuron.** Symptom: the first generated token
was correct (prefill OK), tokens 2–3 were plausible, then the model
dropped into a 3-token loop:

```
"The capital of France is" → " Paris, the capital of the capital of
                              the capital of the capital of the same
                              of the same..."
```

### Diagnosis

Replacing the buffer reads with explicit `torch.zeros_like` in
`_forward_decode` produced **bit-identical** output to the original
broken case. That meant the buffer reads were ALREADY zero — Neuron's
compiler had constant-folded the zero-initialized buffer, so subsequent
`.data.copy_()` writes never propagated between forward calls. DeltaNet
was operating as a stateless linear layer.

### Fix

Initialize the state buffers with **non-zero epsilon** (`1e-30`)
instead of zeros:

```python
eps = 1e-30
self.register_buffer(
    "recurrent_state_buffer",
    torch.full(
        (max_batch, num_v_heads, head_k_dim, head_v_dim),
        eps,
        dtype=torch.float32,
        device="cpu",
    ),
    persistent=False,
)
```

The non-zero init prevents the compiler from constant-folding the
buffer (it's no longer "all zeros, must be a constant"), so it stays
in the graph as a real input tensor. Mutations via `.copy_()` now
propagate between forward calls, exactly like the KV cache.

The `1e-30` value rounds to exactly `0` in bf16 arithmetic, so the
math is unchanged from a "zero start" — only the float32 storage
representation differs.

### Why simpler is better

We tried a more invasive fix first: store DeltaNet state in the
vllm-tracked KV cache via flat `index_put_(arange(N))` over 524k
floats per layer per step. This worked at MAX_LEN=512 but generated
graphs with 4M+ instructions, blowing up compile time at MAX_LEN=4096
to over 90 minutes per HLO graph. The epsilon-init Buffer approach
is functionally equivalent, has no graph-size impact, and compiles
in normal time.

## Layout

```
src/
  qwen3_5/
    __init__.py              # plugin entry: registers the model class
    config.py                # Qwen3_5Config (parses HF text_config + RoPE)
    factory.py               # vllm-neuron model factory glue
    model_bf16.py            # Qwen3_5ForConditionalGeneration + layers
    register.py              # vllm registry override (force-replace stub)
    weight_loaders_bf16.py   # HF→our-flat-name weight mapping
    nki_kernels/
      deltanet_fused.py      # PR #152 NKI fused chunked DeltaNet kernel
  _serve_main.py             # `vllm serve` entrypoint with our registry
  serve.sh                   # convenient launcher with sensible defaults

test/
  test_phase1_skeleton.py    # config + factory smoke
  test_paris_smoke.py        # weight-mapping + register smoke
  test_logits_parity.py      # CPU prefill parity vs HF
```

## Required container patches (Beta v5)

The vllm-neuron Beta v5 container ships with default 30-minute
distributed-init timeouts that are too short for full Qwen3.5
compile (which can take 20-40+ minutes per HLO). Patch before the
first `serve.sh` run:

```bash
# 4-hour torch distributed timeout
sudo docker exec vllm_neuron sed -i \
  's|^default_pg_timeout: timedelta = _DEFAULT_PG_TIMEOUT|default_pg_timeout: timedelta = timedelta(hours=4)|g' \
  /opt/conda/lib/python3.12/site-packages/torch/distributed/constants.py

# 4-hour vllm-neuron tp_barrier timeout
sudo docker exec vllm_neuron sed -i \
  's|timedelta(seconds=1800)|timedelta(hours=4)|g' \
  /opt/conda/lib/python3.12/site-packages/vllm_neuron/parallel/neuron_parallel_state.py

# Clear pyc cache so changes take effect
sudo docker exec vllm_neuron find /opt/conda/lib/python3.12/site-packages/torch/distributed -name __pycache__ -exec rm -rf {} +
```

## Other things worth knowing

- The `(1+weight)` RMSNorm bug (Gemma-style) is fixed throughout;
  weights are zero-centered on disk and `output * (1 + weight)` is
  applied at runtime. This was the bug that caused the 4B's first-token
  output to look like garbage in the parent code.
- `attn_output_gate=True` is honored. The gate weight is spliced into
  `q_proj`'s second half on disk (per-head `[h_q | h_gate]` interleaving);
  custom loaders (`_spliced_q_kv_loader`, `_spliced_q_gate_loader`) split
  it into `qkv_proj_weight` (Q+K+V) and `attn_gate_weight`.
- `tie_word_embeddings=True` for 4B (lm_head is aliased to
  `embed_tokens`).
- `attn_output_gate` is applied as `sigmoid(gate)` after attention,
  before `o_proj`.
- The mask uses bf16 min (`-65504`) on causal-violating positions and
  softmax runs in fp32 — matches NxDI's reference for QK-mask leakage
  control.

## Next steps / not yet done

- Path D-style FP8 KV cache. Plumbed but not yet enabled by default.
- Fused decode attention NKI kernel for head_dim=256 (replace the
  current Python split-K matmul in `Qwen3_5GQAAttention.forward_decode`).
- Fused DeltaNet recurrent-step NKI kernel (replace the current
  ~10 elementwise ops per token).
- TP=8 sweep (currently TP=4 is what we benchmark).
- Speculative decode via the `mtp.*` head (currently skipped during
  weight loading; the head is in the checkpoint).

## License

This adapter is Apache 2.0. The Qwen3.5-4B weights are © Alibaba,
also Apache 2.0. See `NOTICE`.
