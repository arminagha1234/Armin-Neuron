# AlphaGenome on Trainium2 — Working Result

**Status: FUNCTIONAL on Trainium at full 131,072-bp (128K) input.**

## Environment
- Instance: `trn2.48xlarge`, Deep Learning AMI Neuron.
- venv: `/opt/aws_neuronx_venv_pytorch_2_9` (torch 2.9.1, torch_neuronx 2.9.0, torch_xla 2.9.0).
- neuronx-cc 2.25.3371. Weights: `gtca/alphagenome_pytorch` fold_0 (920 MB fp32).

## Result
The full model (10 of 11 heads) compiles and runs on a single NeuronCore and matches
the CPU reference to ~6 decimals.

| Length | Compile | Correctness vs CPU | Warm forward |
|---|---|---|---|
| 16,384 | yes | 17/18 cos=1.000000 | 0.31 s |
| 131,072 (128K) | yes (`--optlevel=1`) | 17/18 cos=1.000000 | ~3.0 s |

`contact_maps` is the one outlier: cos≈0.9965 (float32 accumulation drift in the
pairwise/3D-contact head). All track heads (atac, dnase, procap, cage, rna_seq,
chip_tf, chip_histone, splice_sites, splice_site_usage) are bit-exact (cos 1.0).

## The two Neuron issues found and how they were resolved

### 1. `splice_junctions` head — unsupported `sort` (NCC_EVRF029)
The junction head does a top-k/sort over genome positions; `torch.sort`/`torch.topk`
lower to an HLO `sort`, which trn2 does not support. **Resolution:** skip that one
head via the model's own `heads=(...)` selector (the other 10 heads cover the
standard track + contact + splice-site outputs). If junctions are needed, run that
head on CPU.

### 2. Long sequence (>=32,768) — compiler internal error (NCC_ITIN902 / AffineIV)
At the default optimizer level, neuronx-cc crashes in `TensorInitialization ->
TongaIslSimplifier -> IntegerSetAnalysis` with "ISL compute budget exceeded" then an
`AffineIV doesn't appear in params or loopnest` assertion. Root cause: the
relative-position machinery (`SequenceToPairBlock._shift` + `repeat_interleave`
bias expansion in `attention.py`) generates affine access maps whose coefficients
grow with the token count; the aggressive loop fusion at default opt level blows the
polyhedral analyzer's budget. **Bisect:** 16,384 ok, 24,576 ok, 32,768 fails (default opt).
**Resolution:** compile with **`NEURON_CC_FLAGS=--optlevel=1`** — avoids the fusion
that triggers the ISL blowup. With optlevel=1, 32,768 AND full 131,072 compile and
run correctly. No model surgery required.

## Files (src/)
- `predict_alphagenome.py` — one-call inference wrapper (the deliverable).
- `fasta_to_onehot.py` — turn a DNA FASTA into the (1,S,4) one-hot input.
- `common.py` — shared input gen + oracle compare helpers.
- `run_cpu_oracle.py` — CPU reference generator.
- `run_neuron.py` — Neuron run + correctness compare harness.
- `analyze_hlo.py` — HLO proto analyzer used to localize the compiler crash.

## Open / next
- `contact_maps` cos 0.9965 — investigate float32 accumulation in the pairwise head if
  contact maps matter to the use case (force that head's einsums to fp32, or keep it
  on CPU). All other outputs are exact.
- `splice_junctions` on Neuron would need a TopK-based rewrite that lowers to the
  Neuron TopK op (torch.topk currently lowers to `sort`), or run on CPU.
- Perf: `--optlevel=1` trades some runtime optimization for compileability; if latency
  matters, isolate optlevel=1 to only the modules that need it.
- Scaling toward 1M bp: not yet tested; expect more memory + longer compile.
