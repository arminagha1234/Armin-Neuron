# armin_nki_kernels

Hand-written NKI kernels for Trainium2, used by the model adapters in this
repo. Each kernel here has:

- A **pure-PyTorch reference** implementation (the `ref_*.py` file next to it)
  that defines the exact math, dtype, and broadcast contract.
- A **parity test** that verifies cosine similarity > 0.999 between the
  kernel and the reference on representative shapes.
- A **microbench** under `microbench/` that times the kernel vs the
  reference under realistic shapes and reports the speedup.
- Optional **vllm-neuron wrapper** (`*_wrap.py`) that calls the kernel via
  `vllm_neuron.nki.nki_hop.wrap_nki(...)` so the same kernel works inside
  a `vllm serve` graph.

## Why this folder exists

Initially: keep iteration fast. Each kernel rev cycle is `edit → CPU sim
parity → microbench → on-device run` — order of minutes. Putting them in a
shared NKI library would slow this loop to days-per-rev.

Eventually: graduate kernels that are general-purpose (correct API, no
model-specific knobs, dtype-flexible) into the upstream NKI library. This
folder always stays the customer-facing entry point so adapters don't
need to update import paths when a kernel moves upstream — we just
re-export it from upstream.

## Kernels

### Attention

- **`attention/decode_hd256.py`** — fused single-token decode attention
  for `head_dim=256`. Replaces the Python split-K
  (`Q_lo·K_lo + Q_hi·K_hi → softmax → ·V_lo|V_hi`) currently used in
  Qwen3.5/3.6 GQA decode. Stock `NF.attention_decode` rejects `head_dim>128`
  on the tensor engine transpose path; this kernel does the same split-K
  internally with PSUM accumulation + fused softmax + AV in one NEFF.

  Status: in development. Reference impl in `ref_decode_hd256.py`.

### DeltaNet (linear attention)

- **`deltanet/recurrent_step.py`** — fused single-token recurrent update
  for the GatedDeltaNet decode path. Replaces the ~10 elementwise +
  reduction PyTorch ops per token (currently in
  `Qwen3_5DeltaNetAttention._forward_decode`) with a single fused kernel
  per `(batch, head)`.

  Status: planned (after `decode_hd256` lands).

## Layout

```
armin_nki_kernels/
  attention/
    decode_hd256.py        ← NKI kernel
    ref_decode_hd256.py    ← pure-torch reference
    decode_hd256_wrap.py   ← wrap_nki() wrapper for vllm-neuron
  deltanet/
    recurrent_step.py
    ref_recurrent_step.py
    recurrent_step_wrap.py
  microbench/
    bench_decode_hd256.py
    bench_recurrent_step.py
tests/
  test_decode_hd256_parity.py
  test_recurrent_step_parity.py
```

## Install

```bash
pip install -e .
```

This makes `import armin_nki_kernels` work from anywhere. Adapters import
specific kernels:

```python
from armin_nki_kernels.attention.decode_hd256_wrap import decode_hd256
attn_out = decode_hd256(q, k_full, v_full, mask, scale)
```

## Running tests

```bash
# Parity (CPU sim, ~seconds)
pytest tests/

# Microbench on Neuron (requires neuron device, ~minutes)
python -m armin_nki_kernels.microbench.bench_decode_hd256
```

## License

Apache 2.0.
