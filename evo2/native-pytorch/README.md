# Running Evo 2 on AWS Trainium — Step-by-Step Guide

This is a complete, copy-paste guide to run **Evo 2** (Arc Institute's StripedHyena2
DNA language model) on **AWS Trainium2**. Evo 2 produces embeddings and next-token
predictions for DNA sequences and is used for variant-effect scoring, embedding
extraction, and generation.

**You do not need to know anything about Trainium or Neuron.** Follow the steps in
order. Every command is copy-paste.

> **Status:** ✅ **Evo2-1B working on Trainium** — cosine **1.000000**, top-1 byte
> agreement **100%** vs a CPU reference (prefill). The 1B model fits on a single
> Trainium chip.

---

## ⚠️ Two important prerequisites (read first)

### 1. Instance: launch a `trn2.3xlarge` in **Melbourne (ap-southeast-4)**
The 1B model is small and runs on a **single Trainium2 chip**, so a **`trn2.3xlarge`**
(the smallest trn2 slice) is the right, cheap choice — you do **not** need a
`trn2.48xlarge`. Launch it in the **Asia Pacific (Melbourne) `ap-southeast-4`**
region.

> A 48xlarge is only needed for the **40B** model (which must be sharded across many
> cores). For 1B (and for AlphaGenome), a 3xlarge is plenty.

### 2. Software: you need the **native-PyTorch Neuron beta** — NOT the public beta
This port was developed and validated against the **native-PyTorch Neuron beta**
(Beta 3) toolchain, not the public Neuron SDK release. The public beta does **not**
have everything this model needs to compile and run. Make sure the instance/AMI is
provisioned with the **native-PyTorch beta** Neuron stack before you start. If you
are unsure which build you have, check with whoever provisioned the box — the public
release will not work for this model.

---

## What you get
- **Embeddings** — `(1, T, 1920)` hidden states for any DNA string (used for
  downstream classifiers / variant-effect work).
- **Logits** — `(1, T, 512)` next-byte predictions (byte-level DNA LM).
- Single Trainium chip; first call compiles (~30 s), then fast.

---

## Step 0 — Launch the instance (one time)
1. In the AWS Console, region **Asia Pacific (Melbourne) `ap-southeast-4`**, launch an
   EC2 **`trn2.3xlarge`**.
2. Use an AMI provisioned with the **native-PyTorch Neuron beta** (see prerequisite 2).
3. Give it a key pair and ≥ 150 GB disk. Note the **Public DNS**.

---

## Step 1 — Connect
```bash
ssh -i /path/to/mykey.pem ubuntu@<your-instance-public-dns>
```
All remaining steps run on the instance.

## Step 2 — Working folder
```bash
mkdir -p ~/evo2 && cd ~/evo2
```

## Step 3 — Activate the native-PyTorch Neuron environment
```bash
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
neuron-ls          # should print a table with at least one NeuronCore
```
(If your beta uses a differently-named venv, activate that one — the key is it's the
**native-PyTorch beta** environment, not the public release.)

## Step 4 — Install the HuggingFace runtime
We use the clean HuggingFace port of Evo 2 (pure PyTorch — no `vortex`, no CUDA, no
TransformerEngine). Install `transformers` without disturbing the Neuron PyTorch:
```bash
pip install "transformers==4.48.3"
```

## Step 5 — Download the model (1B)
```bash
python - <<'PY'
from huggingface_hub import snapshot_download
p = snapshot_download("Taykhoom/Evo2-1B-8K", local_dir="./evo2_1b_8k")
print("model at:", p)
PY
export EVO2_MODEL=$HOME/evo2/evo2_1b_8k
```
(~2.2 GB. Larger variants exist — `Taykhoom/Evo2-7B-8K`, etc. — but start with 1B.)

## Step 6 — Get the run scripts
Copy these two files from this repo into `~/evo2`:
- `src/predict_evo2.py` — the one-command runner.
- `src/evo2_neuron_patch.py` — the Neuron compatibility patches (imported automatically).

## Step 7 — Run it!
**Embeddings:**
```bash
python predict_evo2.py --seq ACGTACGTACGTACGTACGTACGT --mode embed --out emb.pt
```
**Next-token logits:**
```bash
python predict_evo2.py --seq ACGT --mode logits --out logits.pt
```
The **first run compiles for the Trainium chip (~30 s)** — normal, and cached for
later runs. Load results later:
```python
import torch
emb = torch.load("emb.pt")      # (1, T, 1920)
```

---

## Why this just works (what the wrapper handles for you)
Evo 2's StripedHyena2 architecture has several pieces that don't run on Trainium
out of the box. `predict_evo2.py` + `evo2_neuron_patch.py` handle all of them
automatically — you don't need to do anything:

1. **No TransformerEngine / FP8.** The config requests FP8 input projections (needs an
   NVIDIA Hopper GPU). We set `use_fp8_input_projections=False`, which uses the port's
   pure-PyTorch fallback (same weights).
2. **Pure-PyTorch attention** (`attn_implementation="eager"`) — no CUDA flash-attn.
3. **FFT-convolution → conv1d.** Evo 2's Hyena operators do long convolutions with the
   FFT (`torch.fft`, complex numbers), and Trainium's compiler does not support complex
   numbers. We replace them with a mathematically-identical real `conv1d` (validated to
   match the original to ~6 decimals on CPU before running on Trainium).
4. **fp32.** A couple of layers produce enormous activations (~1e16) that the final
   normalization is designed to absorb; in half-precision this collapses to zero on
   Trainium. Running in fp32 fixes it (the 1B model is small, so fp32 is fine).

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `neuron-ls` shows nothing | Not on a Trainium instance, or wrong AMI. Use a `trn2` instance with the native-PyTorch beta. |
| `Complex data types are not supported` (NCC_EVRF004) | The FFT→conv1d patch isn't applied. Run via `predict_evo2.py` (it imports `evo2_neuron_patch`). |
| `FP8 requires Transformer Engine` | `use_fp8_input_projections` wasn't disabled. The wrapper sets it; if calling the model yourself, set `config.use_fp8_input_projections=False`. |
| Output is all zeros / garbage | You're running in bf16. Use fp32 (`model.float()`); the wrapper does this. |
| First run is slow | One-time compile. It caches; later runs are fast. |
| `Got a cached failed neff` | A previous compile was interrupted. Clear it: `rm -rf /var/tmp/neuron-compile-cache` and rerun. |
| Public-beta compile errors | You're likely on the public Neuron release. This model needs the **native-PyTorch beta** (prerequisite 2). |

---

## Ways to optimize / extend

### Easy
1. **Keep the compile cache** (`/var/tmp/neuron-compile-cache`) so you compile once.
2. **Fix one sequence length** to avoid recompiles for each new length.
3. **Embeddings only.** If you only need embeddings, the recommended Evo 2 embedding is
   the pre-norm of a middle block (`blocks[12]` for 1B) — cheaper than full logits.

### Medium
4. **Mixed precision.** Full fp32 is used only because two layers' massive activations
   break the norm in bf16. Isolating fp32 to just those norms (and keeping the rest
   bf16) would cut memory/time roughly in half — the path to fitting bigger models.
5. **Batch sequences.** Stack N DNA strings into one call for near-linear throughput.
6. **Multiple NeuronCores.** Run independent sequences in parallel, one per core.

### Advanced (scaling to 7B / 40B)
7. **7B** uses the identical recipe; just download `Taykhoom/Evo2-7B-8K`. Expect a
   larger compile and more memory.
8. **40B** (50 layers, ~76 GB bf16) must be **sharded across NeuronCores** (tensor /
   sequence parallel) and likely needs the mixed-precision approach (#4) so it fits —
   full fp32 would be ~152 GB. This needs a `trn2.48xlarge`, not a 3xlarge.
9. **Longer context (8K → 1M).** The Hyena long-conv kernel length grows with sequence
   length; for very long inputs, tile/window the long convolution. Test memory at 8192
   first.
10. **Decode / generation.** Only the prefill forward is validated. The KV-cache +
    Hyena-recurrence decode path (`step_fir`/`step_iir`) is untested on Trainium.

---

## Notes & limitations
- Validated: **Evo2-1B**, prefill, fp32, seqlen up to a few hundred bp — cosine
  1.000000, top-1 100% vs CPU.
- Decode/generation and long-context (8K+) are not yet validated on Trainium.
- 7B/40B are the same recipe + sharding (see `results/RESULTS.md`).

## Credits & license
- HF port: [`Taykhoom/Evo2-1B-8K`](https://huggingface.co/Taykhoom/Evo2-1B-8K)
  (bit-exact re-implementation of the Arc Institute model).
- Original Evo 2: [arcinstitute/evo2](https://github.com/ArcInstitute/evo2),
  [Zymrael/vortex](https://github.com/Zymrael/vortex). Apache-2.0.
