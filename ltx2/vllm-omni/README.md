# LTX-2 18.88B on Trainium2 — vLLM-Omni path

**Status: Output quality bug — blurry output despite BMM-SDPA fix**

The vLLM-Omni `NeuronLTX2Pipeline` (registered as `model_arch="LTX2Pipeline"`)
runs end-to-end but produces **washed-out video** (pixel std=16 vs CPU ref std=68).

## What works

- NeuronLTX2Pipeline loads and initializes (text encoder + connectors +
  transformer on Neuron TP=4, VAE on CPU)
- BMM-SDPA replacement is installed and active
- Functional RoPE (no in-place `addcmul_` on views) is installed
- RoPE precomputation on CPU + caching works
- Full denoising loop runs (245s cold, ~165s warm)
- Produces 25 frames of output (correct shapes)
- **No container contention** (earlier failures were caused by a FLUX bench
  auto-restarting in the same container, not by code bugs)

## Root cause of remaining blur

The omni `vllm_neuron` compile backend captures the **entire transformer
forward** as a single FX graph that runs on Neuron in bf16. Inside this
compiled graph, several preprocessing operations produce wrong results:

1. **Attention mask conversion** (`(1 - mask) * -10000` at line 1625) —
   when compiled in bf16, the all-zeros result gets constant-folded by XLA
   and DROPPED from the graph entirely (AWS team docstring confirms this).
   The net effect: cross-attention runs without any mask constraint.

2. **Time embedding MLP** — runs in bf16 inside the compiled graph; precision
   loss in the SiLU + linear produces slightly different modulation values
   than CPU fp32.

3. **Caption projection** — same bf16 precision issue when compiled.

The **native PyTorch path** (validated, sharp output) solves this by running
ALL preprocessing on CPU and only sending pre-processed tensors to Neuron.
The omni path needs the same architectural split.

## The fix (not yet implemented)

Refactor `_NeuronTransformerWrapper.forward()` to do the FULL preprocessing
on CPU (matching Jim's NxDI `NeuronTransformerWrapper` pattern):

```python
def forward(self, hidden_states, audio_hidden_states, encoder_hidden_states, ...):
    # 1. Run proj_in, time_embed, caption_projection ON CPU (eager, fp32)
    hs = self.inner.proj_in(hidden_states)       # CPU
    temb, emb_ts = self.inner.time_embed(...)    # CPU
    enc_hs = self.inner.caption_projection(...)  # CPU
    # ... (cross-attn conditioning, RoPE, mask conversion)

    # 2. Move 22 tensors to Neuron
    # 3. Call a "blocks-only" compiled forward that skips preprocessing
    video_out, audio_out = self._compiled_blocks_forward(*22_tensors)
    return video_out, audio_out
```

This requires:
- Splitting the transformer's forward into "preprocessing" (stays eager/CPU)
  and "blocks + output" (compiled for Neuron)
- Changing the compile boundary so only the blocks subgraph gets traced
- ~2-3 hours of surgery on `neuron_ltx2_pipeline.py`

## For now: use `native-pytorch/`

The native PyTorch path with `torchrun` + the `neuron_ltx2_native.py` script
produces correct video output. The optimized version (`neuron_ltx2_optimized.py`)
achieves ~30s generation using pre-compiled NEFFs.

## Files (patches applied in container, not persisted to repo)

| File | Patch | Status |
|---|---|---|
| `ltx2_transformer.py` | BMM-SDPA injected at module level | Applied ✅ |
| `ltx2_transformer.py` | Functional RoPE (no in-place addcmul_) | Applied ✅ |
| `sdpa.py` | Original `torch.all` check → `pass` (keeps mask) | Baseline |
| `neuron_ltx2_pipeline.py` | RoPE precompute + coord CPU→Neuron | Applied ✅ |

## Key learnings for the refactor

From `neuron/examples/LTX/BLUR_INVESTIGATION.md`:
- Step-0 noise_pred: Neuron std=0.84 vs CPU std=1.06, max ±11 vs ±5
- Encoder_hidden_states identical (text encoder on CPU is correct)
- Hidden_states differ due to RNG (not a quality issue, just different seed)
- The 20% std damping + outlier spikes are the compiled SDPA + mask interaction
