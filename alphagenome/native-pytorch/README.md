# Running AlphaGenome on AWS Trainium — Step-by-Step Guide

This is a complete, copy-paste guide to run **AlphaGenome** (Google DeepMind's DNA
sequence model) on an **AWS Trainium2** instance. It predicts hundreds of genomic
tracks (ATAC, DNase, CAGE, RNA-seq, ChIP, contact maps, splice sites) at
single-base resolution from DNA sequences up to 131,072 bp.

**You do not need to know anything about Trainium or Neuron.** Follow the steps in
order. Every command is copy-paste.

> **Status:** ✅ Working on Trainium2 at the full 131,072-bp window. All track heads
> match a CPU reference to ~6 decimal places. (One head, `contact_maps`, is within
> ~0.4%; the `splice_junctions` head is skipped — see [Notes](#notes--limitations).)

---

## What you get

| Output head | What it predicts | Shape (at 131,072 bp) |
|---|---|---|
| `atac` | Chromatin accessibility | (1, 131072, 256) @1bp, (1, 1024, 256) @128bp |
| `dnase` | DNase-seq | (1, 131072, 384) / (1, 1024, 384) |
| `procap` | Transcription initiation | (1, 131072, 128) / (1, 1024, 128) |
| `cage` | 5'-cap RNA | (1, 131072, 640) / (1, 1024, 640) |
| `rna_seq` | RNA expression | (1, 131072, 768) / (1, 1024, 768) |
| `chip_tf` | TF binding | (1, 1024, 1664) @128bp |
| `chip_histone` | Histone marks | (1, 1024, 1152) @128bp |
| `contact_maps` | 3D chromatin contacts | (1, 64, 64, 28) |
| `splice_sites` | Splice-site classes | (1, 131072, 5) |
| `splice_site_usage` | Splice-site usage | (1, 131072, 734) |

Speed after the one-time compile: **~3 seconds** per 131,072-bp sequence.

---

## Step 0 — Launch a Trainium instance (one time)

1. In the AWS Console, launch an **EC2 instance** of type **`trn2.48xlarge`**
   (Trainium2). A smaller `trn1`/`inf2` also works for shorter sequences, but this
   guide assumes trn2.
2. For the AMI, pick the **"Deep Learning AMI Neuron (Ubuntu 22.04/24.04)"** — search
   "Neuron" in the AMI catalog. This AMI comes with the Neuron drivers and Python
   environments pre-installed, so you skip all driver setup.
3. Give it a key pair you have (e.g. `mykey.pem`) and at least 200 GB of disk.
4. Launch, and note the instance's **Public DNS** (looks like
   `ec2-XX-XX-XX-XX.us-east-2.compute.amazonaws.com`).

> If your AMI does **not** have Neuron pre-installed, follow the official setup once:
> https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/neuron-setup/

---

## Step 1 — Connect to the instance

From your laptop terminal:

```bash
ssh -i /path/to/mykey.pem ubuntu@ec2-XX-XX-XX-XX.us-east-2.compute.amazonaws.com
```

You're now on the Trainium machine. All remaining steps run **on the instance**.

---

## Step 2 — Make a working folder

```bash
# Use fast local disk if available, otherwise your home folder is fine.
mkdir -p ~/alphagenome
cd ~/alphagenome
```

---

## Step 3 — Turn on the Neuron Python environment

The Neuron AMI ships ready-made Python environments. Activate the PyTorch one:

```bash
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
```

Your prompt should now start with `(aws_neuronx_venv_pytorch_2_9)`.

Check it sees the Trainium chips (you should see a table of NeuronCores):

```bash
neuron-ls
```

---

## Step 4 — Install AlphaGenome

We install **only** the AlphaGenome code (`--no-deps`) so it does not disturb the
Neuron PyTorch that's already set up:

```bash
pip install alphagenome-pytorch --no-deps
pip install huggingface_hub numpy   # small helpers, safe to install
```

Verify it imports:

```bash
python -c "from alphagenome_pytorch import AlphaGenome; print('AlphaGenome OK')"
```

---

## Step 5 — Download the model weights

```bash
python - <<'PY'
from huggingface_hub import hf_hub_download
p = hf_hub_download("gtca/alphagenome_pytorch", "model_fold_0.safetensors",
                    local_dir="./hf")
print("weights at:", p)
PY
```

This downloads ~920 MB to `./hf/model_fold_0.safetensors`.

---

## Step 6 — Get the run scripts

Copy `src/predict_alphagenome.py` from this repo into your working folder (e.g. with
`git clone` of this repo, or `scp` it up). Then tell it where the weights are:

```bash
export AG_WEIGHTS=$HOME/alphagenome/hf/model_fold_0.safetensors
```

> The script automatically handles the two Trainium-specific details for you
> (see [Why this just works](#why-this-just-works)). You don't need to set anything else.

---

## Step 7 — Run it!

**Quick demo** (random 131,072-bp sequence — proves everything works):

```bash
python predict_alphagenome.py
```

The **first run compiles the model for the Trainium chip and takes a few minutes.**
This is normal and happens only once — the result is cached, so later runs are ~3 s.
You'll see each output head and its shape printed at the end.

**Run on your own DNA sequence** (see Step 8 to make the input file):

```bash
python predict_alphagenome.py --input my_seq.npy --organism 0 --out predictions.pt
# --organism 0 = human, 1 = mouse
```

Predictions are saved to `predictions.pt`. Load them later with:

```python
import torch
out = torch.load("predictions.pt")
print(out["atac"][128].shape)   # ATAC-seq prediction at 128bp resolution
```

---

## Step 8 — Turn a DNA sequence (FASTA) into model input

AlphaGenome wants a one-hot array of shape `(1, S, 4)` where `S` is a multiple of
128. Use the helper:

```bash
python fasta_to_onehot.py --fasta my_region.fa --length 131072 --out my_seq.npy
```

Then feed `my_seq.npy` to Step 7. The helper pads/crops to the requested length and
encodes A,C,G,T → channels 0,1,2,3.

---

## Understanding the output

`predict_alphagenome.py` returns a dictionary:

```python
out["atac"][1]      # (1, 131072, 256)  ATAC-seq at single-base resolution
out["atac"][128]    # (1, 1024,   256)  ATAC-seq at 128bp resolution
out["rna_seq"][128] # (1, 1024,   768)  RNA expression at 128bp
out["contact_maps"] # (1, 64, 64, 28)   3D chromatin contact maps
out["splice_sites"]["probs"]  # (1, 131072, 5)  splice-site probabilities
```

Track channels are zero-padded (e.g. ATAC has 167 real human tracks inside 256
channels). To get named, padding-stripped tracks, see the upstream
[named outputs guide](https://github.com/genomicsxai/alphagenome-pytorch).

---

## Why this just works

Two Trainium-specific quirks are handled **for you** by `predict_alphagenome.py`,
explained here so you understand what's happening:

1. **It compiles with `--optlevel=1`.** Trainium compiles the model to a chip-specific
   binary the first time you run. At the compiler's default optimization level it
   crashes on long sequences (an internal compiler bug in its array-index analysis).
   Optimization level 1 avoids it, with no change to the math. The script sets this
   automatically.
2. **It skips the `splice_junctions` head.** That head needs a `sort` operation the
   Trainium chip doesn't support. The other 10 heads (all the standard tracks +
   contact maps + splice-site classification/usage) run fine. If you specifically
   need splice *junctions*, compute that one head on CPU.

You don't have to do anything for either — just run the script.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `neuron-ls` shows nothing / "No neuron devices" | You're not on a Trainium/Inferentia instance, or the AMI lacks Neuron drivers. Use a Neuron DLAMI on a `trn2`/`trn1`/`inf2` instance. |
| `ModuleNotFoundError: torch_xla` | You didn't activate the env. Run `source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate`. |
| First run "hangs" for minutes | That's the one-time compile. Wait — it caches and is fast afterward. |
| `sort is not supported on trn2` | You re-enabled the `splice_junctions` head. Leave it skipped (default), or run it on CPU. |
| `NCC_ITIN902` / `AffineIV` compiler error | The optlevel flag isn't being applied. Make sure you run via `predict_alphagenome.py` (it sets `NEURON_CC_FLAGS=--optlevel=1`), or set that env var yourself before running. |
| Out of memory at very long sequences | Use a shorter `--length`, or a bigger instance. 131,072 bp fits comfortably on trn2. |
| Want to start fresh | Clear the compile cache: `rm -rf /var/tmp/neuron-compile-cache` (next run recompiles). |

---

## Ways to optimize the model

The model is correct and runs at ~3 s/sequence. If you want it **faster** or to
handle **more data**, here are the highest-value levers, roughly in order of effort:

### Easy (config only)
1. **Reuse the compile cache.** The first run compiles; every run after is fast. Keep
   `/var/tmp/neuron-compile-cache` around (back it up) so you never recompile. If you
   always use the same sequence length, you compile exactly once.
2. **Fix the sequence length.** Compile for the one length you actually use (e.g.
   `--length 131072`). Mixing many lengths means many compiles. Pick one.
3. **Pre-compile ahead of time (AOT).** Run the demo once on the target length during
   setup so the researcher's first real run is already fast.
4. **Only compute the heads you need.** Each head adds work. If the researcher only
   wants `atac` and `rna_seq`, restrict to those — fewer outputs, faster run, smaller
   memory. (In the script, pass a shorter `heads=(...)` list to `predict`.)

### Medium (precision / batching)
5. **bfloat16 compute.** The model supports a mixed-precision mode
   (`DtypePolicy.mixed_precision()`), which matches the original JAX model and is
   ~2× faster on Trainium's matrix engines than float32. Validate outputs still meet
   your tolerance, then use it for inference.
6. **Batch multiple sequences.** Instead of one sequence at a time, stack N sequences
   into shape `(N, S, 4)` and run once. Trainium is far more efficient at batch >1 —
   near-linear throughput gains until you fill the chip.
7. **Use more than one NeuronCore.** A trn2.48xlarge has many cores. Run several
   independent sequences in parallel (data parallelism) — one model copy per core —
   to multiply throughput. Simplest with a small launcher that pins each process to a
   core.

### Advanced (kernel / graph level)
8. **Tune the optimizer level per-module.** We use `--optlevel=1` globally to dodge a
   compiler bug in the relative-position code. Most of the model compiles fine at the
   default (higher) level. Isolating optlevel=1 to only the attention/pair modules
   would let the rest run at full optimization — faster, same correctness.
9. **Improve `contact_maps` precision.** It's the one head at ~0.4% off (float32
   accumulation in the pairwise attention). Forcing that head's einsums to float32,
   or running just that head on CPU, makes it exact if contact maps matter to you.
10. **Replace the unsupported `sort` with a Neuron TopK kernel.** To bring the
    `splice_junctions` head on-device, rewrite its top-k selection so it lowers to
    Neuron's supported TopK op (or a small NKI kernel) instead of `sort`.
11. **Trim the global KV / sequence work for shorter inputs.** If you routinely use
    sequences shorter than 131,072, compile for that exact length so the model doesn't
    pad up to the maximum — less compute, lower latency.

---

## Notes & limitations

- **All track heads match a CPU reference to ~6 decimals.** `contact_maps` is within
  ~0.4% (float32 drift). `splice_junctions` is skipped (unsupported `sort`).
- Tested at lengths 16,384 and 131,072 bp. Scaling toward 1,000,000 bp is untested and
  will need more memory and a longer first compile.
- This guide uses the single-fold checkpoint (`model_fold_0`). The repo also has an
  all-folds ensemble for higher accuracy at higher cost.

## Credits & license
- AlphaGenome PyTorch port: [genomicsxai/alphagenome-pytorch](https://github.com/genomicsxai/alphagenome-pytorch)
  (Apache-2.0). Weights: [gtca/alphagenome_pytorch](https://huggingface.co/gtca/alphagenome_pytorch).
- Original model © Google DeepMind, subject to the
  [AlphaGenome Model Terms](https://deepmind.google.com/science/alphagenome/model-terms).
