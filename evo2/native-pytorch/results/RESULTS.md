# Evo2 (StripedHyena2) on Trainium2 — Working Result

**Status: Evo2-1B FUNCTIONAL on Trainium — cosine 1.000000, top-1 agreement 100%
vs CPU reference (prefill).** Runs on a single Trainium2 chip (trn2.3xlarge).

## Environment
- Instance: `trn2.3xlarge` (1 Trainium2 chip is enough for the 1B model).
- Software: **native-PyTorch Neuron beta** (Beta 3). The public Neuron release is
  not sufficient for this model.
- Base: `Taykhoom/Evo2-1B-8K` (HuggingFace `trust_remote_code` port — pure PyTorch,
  no `vortex` / CUDA / TransformerEngine).

## What worked
Ported the **HuggingFace port**, NOT the stock `evo2`/`vortex` package (CUDA + TE
bound, cannot run on Neuron). Recipe (all in `src/`):

1. **`use_fp8_input_projections=False`** — engages the pure-PyTorch `TELinear`
   fallback (no TransformerEngine / no Hopper).
2. **`attn_implementation="eager"`** — pure-PyTorch attention (no flash-attn).
3. **FFT → conv1d (the real port work, `evo2_neuron_patch.py`).** neuronx-cc rejects
   complex dtypes (`NCC_EVRF004`); StripedHyena2's Hyena operators do long
   convolutions via FFT (complex64). Both FFT paths convolve a real, finite filter,
   so each equals a real depthwise causal `conv1d` (FFT is just an O(L log L) form):
   - `engine.fftconv_func` (FIR / hcm) → conv1d. Validated vs FFT: max abs diff ~1e-5.
   - `HyenaInferenceEngine.parallel_iir` (hcl long-conv) → conv1d. NOTE: the port's
     built-in `long_fir_threshold` conv branch is buggy (correlates instead of
     convolving — no filter flip, cosine 0.69), so we replace the method with a
     correct flipped-kernel causal conv. Validated vs FFT on CPU.
4. **fp32 (`model.float()`).** Layers 24–25 produce massive activations (~1.8e16);
   the final RMSNorm absorbs them, but in bf16 on Neuron the norm collapses to
   exactly 0. fp32 fixes it and makes parity exact.

## Validation
- CPU patched-conv1d vs original-FFT (parity of the port): cosine 0.999999.
- Neuron (patched, fp32) vs CPU reference (fp32): **cosine 1.000000, top-1 100%**.
- First compile ~30 s; single NeuronCore.

## Files (src/)
- `predict_evo2.py` — one-command embeddings / logits runner (the deliverable).
- `evo2_neuron_patch.py` — FFT→conv1d patches (FIR + IIR), CPU-validated.
- `run_evo2.py` — CPU-oracle / Neuron-compare harness.

## Open / next
- **7B**: identical recipe, just the larger checkpoint; bigger compile + memory.
- **40B**: same recipe + tensor/sequence sharding across NeuronCores (needs a
  trn2.48xlarge). Full fp32 ≈152 GB, so isolate fp32 to just the norm/massive-
  activation layers (mixed precision) to fit.
- **Long context (8K → 1M)**: the hcl conv kernel length grows with sequence length;
  tile/window the long conv for very long inputs. Validate memory at 8192 first.
- **Decode/generation**: only prefill validated; the KV-cache + Hyena-recurrence
  decode path (`step_fir`/`step_iir`) is untested on Neuron.
