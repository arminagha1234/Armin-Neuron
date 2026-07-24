# LLaMA-1 7B on Trainium — native PyTorch, single core

Runs Meta's original **LLaMA-1 7B** (`huggyllama/llama-7b`, a public re-upload of
the original weights) on a **single Trn1 NeuronCore** with native PyTorch. No XLA,
no tensor parallelism needed — a 7B model in bf16 (~13.5 GB) fits in the core's
16 GB HBM.

## Status

✅ **Validated.** Per-position top-1 agreement of Neuron bf16 vs CPU fp32 is
**100.0% (39/39 positions)** — the ported model predicts the exact same tokens as
the reference.

| Check | Result |
|---|---|
| Weights fit on 1 Trn1 core (16 GB) | ✅ ~13.5 GB bf16 |
| Neuron bf16 vs CPU fp32 top-1 agreement | ✅ **100.0%** (39 positions) |
| First forward (incl. NEFF compile) | ~200 s |
| Cached forward | fast |

## Run

Generate text:
```bash
python3 run_native.py --model huggyllama/llama-7b \
    --prompt "The capital of France is" --max-new-tokens 30
```

Validate numerics against a CPU reference:
```bash
python3 validate.py huggyllama/llama-7b
```

Expected tail of `validate.py`:
```
[llama-7b] per-position top-1 agreement (neuron bf16 vs cpu fp32): 100.0%  (39 positions)
PORT_OK
```

## Files

- `run_native.py` — load model, `.to("neuron")`, greedy-generate. The minimal
  "how do I run Llama on Trainium" example.
- `validate.py` — teacher-forced per-position agreement vs a CPU fp32 reference.

## Notes

- Uses `attn_implementation="eager"` (the supported attention path for this beta).
- `use_cache=False` keeps the example simple; each new sequence length compiles a
  NEFF on first use, then caches.
- On Trn1 you'll see a benign "async IO requires Trn2 → falling back to synchronous
  IO" log. It doesn't affect correctness.
- Want a bigger model? An **8B won't fit on one Trn1 core** — see
  [`../llama-3.1-8b/`](../llama-3.1-8b/) for the tensor-parallel approach.
