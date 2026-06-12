# LTX-2 on vLLM-Omni — Progress Status

**Branch:** `ltx2-omni-bringup`
**Box:** `i-02a51e30b3a33408d` @ `3.150.135.217` (us-east-2 trn2.48xlarge)
**Container:** `vllm_omni`
**Image:** `concourse-release-1cb0647:pr-655cd3ee9b8d69818e52f37fbe1bb2a445bfbd60`

## Where we are

**12 of 13 issues cleared.** Stuck on a deep cross-attn shape bug in vllm-omni's LTX-2 transformer code.

### The breakthrough (this session)

Adopted Jim Burtoft's NxDI-port pattern (`neuron/external/pr-117-nxdi-diffusion-models/contrib/models/ltx2-video-audio/src/pipeline.py::NeuronTransformerWrapper`): wrap the DiT in an EAGER `nn.Module` that sits OUTSIDE the `torch.compile` boundary and does coord/contiguity prep before calling the inner compiled DiT.

**This decisively unblocked the catch-22** that defeated the previous session:
- coords-on-CPU (needed for pipeline's `.repeat()` at line 1090) ✓
- coords-on-Neuron (needed inside compiled graph) ✓ — done eagerly in wrapper

After this fix:
- Weight loading: ✓ (re-prefix `transformer.inner.*` in `load_weights`)
- All 4 TP workers initialize: ✓
- Transformer compile starts: ✓
- Encoder + connectors + RoPE + mask flow: ✓
- Reaches the dummy-run forward pass through the LTX-2 DiT.

### The new wall (issue 13)

Crashes inside the FX trace of the cross-attn (text-to-video) on:
```
Dynamo failed: scaled_dot_product_attention(...,
  attn_mask=FakeTensor(size=(2, 256, 8, 128), bf16))
'Attempting to broadcast a dimension of length 128 at -1!
Mismatching argument at index 1 had torch.Size([2, 256, 8, 128]);
but expected shape should be broadcastable to [2, 8, 256, 1024]'
```

The mask shape `[2, 256, 8, 128]` is the QUERY pre-permute shape, not a mask. Walking the LTX2 attn processor on paper produces the right 4D mask `[2, 8, 1, 1024]` at every step. So the bug is somewhere subtle in the FX trace inside vllm-omni's LTX-2 path — possibly Dynamo aliasing the query into the mask slot, or a cross-attn code path that wasn't tested on Neuron.

This is **deeper than issues 1-11**: those were eager-mode pipeline glue. This is inside the compiled graph itself.

## The two paths forward

### Path 1 — Stay on vllm-omni, surgery on `ltx2_transformer.py`

Patch the cross-attn processor in vllm-omni's `ltx2_transformer.py` to avoid the shape collision. We have the file in `.tmp/ltx2_transformer.py` to study. Could compare against diffusers' reference LTX2 attention to find the deviation.

**Cost:** Hours of debugging + container-side patches not in git.
**Risk:** May hit more issues behind it (we're 13 deep now).

### Path 2 — Pivot to diffusers' own LTX2Pipeline.__call__ (RECOMMENDED)

Override `NeuronLTX2Pipeline.forward(req)` to delegate to `diffusers.LTX2Pipeline.__call__()` with our Neuron-resident DiT plugged in via `pipe.transformer = self._NeuronTransformerWrapper`. This bypasses ALL of vllm-omni's `pipeline_ltx2.forward()` AND `ltx2_transformer.forward()` — diffusers' reference forward goes through **diffusers-native** `LTX2VideoTransformer3DModel` (without vLLM-Omni's parallel-layer port).

Trade-off: diffusers' DiT isn't vLLM-parallel-aware, so we'd run un-sharded on a single 96GB Neuron core (LTX-2 19B might just fit at bf16 ≈ 38GB weights + activations).

**This is exactly what Jim's NxDI port does.** They use diffusers' pipeline + a `NeuronTransformerWrapper` swap-in. We get to copy that pattern more closely.

### Path 3 — Wait for vllm-omni next image

Beta 1 LTX-2 path is clearly untested on Neuron. The team will fix; we lose customer-facing time but eliminate the surgery cost.

## Fixes baked into `neuron_ltx2_pipeline.py` (committed on this branch)

All 11 prior fixes + the new wrapper:

1. ✅ Weight loading via framework `(name, tensor)` iterable — strip + re-prefix `transformer.inner.`
2. ✅ Gemma3 encoder stays on CPU (`_get_gemma_prompt_embeds` override)
3. ✅ encode_prompt return-tuple `(embeds, mask)`
4. ✅ Attention mask dtype pre-cast on CPU before Neuron move
5. ✅ Connectors API mismatch (`additive_mask` kwarg) via `_ConnectorsCompatWrapper`
6. ✅ Connectors dtype trap — wrapper round-trips inputs to CPU, runs, moves outputs back
7. ✅ RoPE `prepare_video_coords` on CPU
8. ✅ RoPE `prepare_audio_coords` on CPU
9. ✅ Video latent dtype forced bf16 (positional dtype @ index 6)
10. ✅ Audio latent dtype forced bf16 (kwarg)
11. ✅ SDPA data-dependent branch (`if torch.all(...)`) — in-container `pass` patch
12. ✅ **`_NeuronTransformerWrapper`** — eager wrapper around the DiT for CPU↔Neuron coord move outside `torch.compile`

## Files

- `customers/fal/ltx2-omni/neuron_ltx2_pipeline.py` — current pipeline (~830 lines)
- `customers/fal/ltx2-omni/run_ltx2_omni.py` — runner with `--bench-runs` flag
- `customers/fal/ltx2-omni/ltx2_stage.yaml` — Omni stage config (TP=4)
- `customers/fal/ltx2-omni/DECISIONS.md` — full decision log

## Reference: Jim Burtoft's NxDI port (the working LTX-2 on Trainium)

`neuron/external/pr-117-nxdi-diffusion-models/contrib/models/ltx2-video-audio/`

Validated end-to-end at 22s warm @ 384×512/25-frame/8-step on trn2.3xl TP=4. The `NeuronTransformerWrapper` pattern in that file's `pipeline.py::NeuronTransformerWrapper` is what we adopted in `_NeuronTransformerWrapper` here.

If we pivot to Path 2 (diffusers pipeline), the work amounts to:
1. Take `pipeline.py::NeuronTransformerWrapper` as the template
2. Re-init in our `NeuronLTX2Pipeline.__init__` with a diffusers-loaded LTX2Pipeline
3. Swap in the wrapper as `pipe.transformer`
4. Override our `forward(req)` to call `pipe(prompt=req.prompts[0], ...)`

Can be done in a single session.
