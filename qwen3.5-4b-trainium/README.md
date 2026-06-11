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
- **State persistence for DeltaNet via non-zero-init `nn.Buffer`** —
  see "The state-persistence fix" below.

## Status

✅ **Correctness validated end-to-end** on `trn2.48xlarge` (TP=4) at
three serving configurations: single-shot prefill at MAX_LEN=512 and
MAX_LEN=4096, and chunked prefill at MAX_LEN=20480 with 4K chunks
(the customer 20K-input pattern). Output is coherent and factually
correct across varied prompts.

| Test | Result |
|------|--------|
| "The capital of France is" → 200 tok | structured `<think>` analysis acknowledging Paris |
| "Calculate 17 times 23" | "= 391", with two methods shown |
| "1 to 30, one per line" | perfect 1→30, no drift |
| "Why is the sky blue?" (480 tok) | accurate Rayleigh-scattering essay with `1/λ⁴` formula and correct wavelengths |
| "Largest country / planet / ..." | correct (Russia, Jupiter, etc.) |
| 20K-token AI history → summarize | coherent numbered milestone list, factually accurate |

## Performance — verified on trn2.48xlarge, TP=4, BF16 KV

Three serving configs, end-to-end timings, single greedy request:

| Mode | MAX_LEN | BUCKET | TTFT (s) | Decode tok/s | Use case |
|---|---:|---:|---:|---:|---|
| Short single-shot | 512 | 512 | **1.83** (≤300 tok in) | 18.4 | small prompts (<500 tok) |
| Medium single-shot | 4096 | 4096 | **8.61** (200-4000 tok in) | 18.35 | 1-4K prompts |
| Long chunked | 20480 | 4096 (×5) | **43.04** (20K input) | 18.08 | 20K customer pattern |

Cost at trn2.48xl on-demand ($21.50/hr):

| Workload | $/M input | $/M output |
|---|---:|---:|
| 20K input, 200 output | $12.85 | $330 |
| 200 input, 32 output | $1.85 | $325 |

Full benchmark with sample completions, reproduction steps, and the
break-down of where time goes:
[BENCHMARK_TRN2_48XL.md](./BENCHMARK_TRN2_48XL.md).

A separate trn2.3xl benchmark is in
[BENCHMARK_TRN2_3XL.md](./BENCHMARK_TRN2_3XL.md) — note that the
earlier `$1.63/M-input` 3xl number reported elsewhere was measured
on a broken model (DeltaNet state was being constant-folded — see
"The state-persistence fix" below) and is not honest. Post-fix 3xl
numbers are in that doc.

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

# Long context — customer 20K input shape (chunked prefill, 5 × 4K chunks)
TP=4 MAX_LEN=20480 BUCKET=4096 \
  MODEL=/path/to/Qwen3.5-4B \
  ./src/serve.sh
```

First compile takes ~25-40 min on a 48xl (`walrus_driver` is
host-CPU-bound). Subsequent runs hit the NEFF cache. The
`MAX_LEN=20480` config is fast to recompile because the 4K prefill
graph is shared with `MAX_LEN=4096` and only the decode graph
changes.

To reproduce the benchmark numbers in
[BENCHMARK_TRN2_48XL.md](./BENCHMARK_TRN2_48XL.md):

```bash
# Inside the container after serve is up:
python3 bench_qwen35_sweep.py --input-lengths 200,500,1000,2000,4000 --max-new 64
python3 bench_qwen35_long.py  --input-tokens 20000 --max-new 200
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

The same fix has been pushed to the upstream
[private-vllm-neuron PR #2104](https://github.com/aws-neuron/private-vllm-neuron/pull/2104)
on commit `8d7e2109`.

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

bench_qwen35_sweep.py        # short/medium-context throughput sweep
bench_qwen35_long.py         # customer 20K × 200 long-context bench
BENCHMARK_TRN2_48XL.md       # full 48xl benchmark with reproduction
BENCHMARK_TRN2_3XL.md        # 3xl benchmark + 3xl-vs-48xl tradeoffs
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

- The `(1+weight)` RMSNorm convention (Gemma-style) is fixed
  throughout; weights are zero-centered on disk and `output * (1 + weight)`
  is applied at runtime.
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
- **NEFFs are portable across Trainium instance types**: a NEFF
  compiled on a 48xl runs identically on a 3xl as long as the runtime
  is the same vllm-neuron Beta v5 image. Customers shipping a fixed
  configuration can compile once on a 48xl (faster compile box) and
  serve from a 3xl ($2.23/hr vs $21.50/hr).

## Roadmap (not yet done)

The decode throughput (~18 tok/s) is the bottleneck for the customer's
20K-input shape. Two NKI optimizations expected to land 2-4× on this:

1. **Fused decode-attention NKI kernel for `head_dim=256`** — replaces
   the current Python split-K matmul in `Qwen3_5GQAAttention.forward_decode`
   (stock `NF.attention_decode` rejects head_dim>128). Expected
   1.3-2× decode on the 8 GQA layers.
2. **Fused DeltaNet recurrent-step NKI kernel** — replaces ~10
   elementwise ops per token across the 24 GDN layers. Expected
   1.5-2× on those layers.
3. **FP8 KV cache** (Path D, half-built in
   `customers/Scaledown/pathD/`) — 2× memory budget → bigger batch →
   linear $/M improvement.
4. **TP=8 sweep** (currently TP=4 is what we benchmark).
5. **Speculative decode via the `mtp.*` head** (currently skipped
   during weight loading; the head is in the checkpoint).

## License

This adapter is Apache 2.0. The Qwen3.5-4B weights are © Alibaba,
also Apache 2.0. See `NOTICE`.
