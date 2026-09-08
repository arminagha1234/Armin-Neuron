# FP8 on Trn2 via vLLM-Neuron: measured 1.36x, not 2x

Read this before planning any FP8 quantization work. The theoretical 2x from
FP8 tensor-engine "double performance mode" does **not** materialize end to end
in this stack today.

## Measurement

Llama-3.1-8B, **same model both sides**, TP8, 3500 in / 1 out, one
trn2.48xlarge. Each configuration was benchmarked **alone on the box** (the other
server stopped) because an idle cross-chip TP8 server is known to degrade
neighbours by up to 45%.

| | prefill tok/s | peak RPS (2 chips) | RPS/chip | single p50 |
|---|---|---|---|---|
| BF16 | 24,592 | 7.78 | 3.89 | 0.141 s |
| **FP8 (ModelOpt static)** | **32,195** | **10.54** | **5.27** | **0.108 s** |
| ratio | 1.31x | **1.36x** | 1.36x | 1.31x |

Both coherent, zero errors, both saturate at concurrency 2-4.

FP8 outputs: `"The capital of France is"` -> `"Paris. The capital of France is
Paris"`; `"2 plus 2 equals"` -> `"4"`; `"The largest planet..."` -> `"Jupiter. It
is a gas giant,"`.

## Why only 1.36x

At the 31B-equivalent operating point this corresponds to about **10% of the
box's FP8 peak** (2.18 of 20.8 PFLOP/s). So the limit is implementation
efficiency, not hardware. FP8 accelerates tensor-engine matmuls while norms,
attention and DMA are unchanged, so Amdahl caps the win well below 2x.

There is real headroom left, but capturing it needs more than a dtype switch.

## Consequence for gemma4-31B

| | RPS/box | boxes for 50 RPS | % of target |
|---|---|---|---|
| bf16 today | 7.4 | 6.8 | 15% |
| with FP8 at 1.36x | 10.0 | 5.0 | 20% |

FP8 saves roughly **1.8 boxes**, and does **not** bring 31B near 50 RPS/box.
Earlier estimates of "FP8 -> 3-4 boxes" assumed 2x and should be retracted.

Even a hypothetical FP4 at similar efficiency lands around 3.5-4 boxes, so
**31B at 50 RPS/box is not reachable by quantization alone at this stack's
efficiency.** Treat the 31B target as a ~5-box workload, or renegotiate the
shape.

## Practical notes for anyone wiring FP8 up

1. **Only NVIDIA ModelOpt static FP8 is supported.** `llama3/quantization.py`
   raises on anything else; compressed-tensors checkpoints (neuralmagic,
   RedHatAI) are explicitly rejected.

2. **`nvidia/Llama-3.1-8B-Instruct-FP8` works after a one-line config
   transform.** ModelOpt puts the quant info in `hf_quant_config.json`, but the
   port reads `config.json["quantization_config"]`. Copy the inner dict across:

   ```python
   cfg["quantization_config"] = {"quant_method": "modelopt",
                                 "quantization": hq["quantization"]}
   ```
   The parser requires `quant_algo == "FP8"` **and**
   `kv_cache_quant_algo == "FP8"` **and** an `exclude_modules` list containing
   `lm_head`. No re-quantization is needed; the checkpoint already carries 224
   `F8_E4M3` weight tensors plus 512 `F32` scales.

3. **The FP8 decode path requires `kv_heads == 1` per rank.** Otherwise
   `attention_decode` silently picks the torch fallback and dies with:

   ```
   NotImplementedError: Attention block torch fallback does not support QKV quantization
   ```
   Llama-3.1-8B has 8 KV heads, so TP4 (2 per rank) fails and TP8 (1 per rank)
   works. Check this before choosing a TP degree.

4. FP8 here means weights **and** activations **and** KV cache, since the port
   requires `kv_cache_quant_algo=FP8`.

## Reproducing

Scripts under `.tmp/indeed-bench/fp8-gate/` in the working environment:
`fp8_prep.sh` (download + config graft), `fp8_tp8.sh` (serve both at TP8),
`seq_bench.sh` (sequential benchmark + the gate arithmetic).
