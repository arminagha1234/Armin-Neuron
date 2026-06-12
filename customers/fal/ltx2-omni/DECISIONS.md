# LTX-2 on vLLM-Omni — Decision Log

A running log of what we tried, why, and what we decided. Newest at top.

---

## 2026-06-12 — Option 2 in progress: clearing Neuron-specific bugs in the LTX path

After deciding on Option 2 (keep vllm-omni's `forward()`, wrap only the
broken bits), we've been clearing a sequence of Neuron-specific failures.
Each fix moves the failure deeper — we are now well inside the DiT
transformer forward, past the entire encoder + connectors + prompt path.

Progress ladder (each one fixed, revealing the next):

4. **Connectors API mismatch** (`additive_mask` kwarg) — wrapped
   `self.connectors` with `_ConnectorsCompatWrapper` that translates the
   vllm-omni-style call `(embeds, additive_mask, additive_mask=True)` to
   the diffusers 0.38 signature `(hidden_states, attention_mask,
   padding_side, scale_factor)` and converts additive→binary mask.

5. **Connectors dtype trap** — connectors module is on CPU but received
   Neuron tensors. Wrapper now moves inputs to CPU, runs, moves outputs
   back to the Neuron device (two-step dtype/device throughout).

6. **`prepare_video_coords` contiguity** (CURRENT) — at
   `ltx2_transformer.py:1069`, `Expected self.is_contiguous() to be true`.
   The RoPE coordinate prep does `meshgrid → stack → flatten → repeat`
   on Neuron tensors, producing a non-contiguous tensor a downstream op
   rejects. This is INSIDE vllm-omni's transformer, not our pipeline.
   Next: either patch the transformer to add `.contiguous()` or run the
   coordinate prep on CPU and move the result to Neuron.

The pattern is clear: vllm-omni's LTX-2 transformer code was never
exercised on Neuron, so it hits Neuron-specific quirks (contiguity,
combined .to(), unsupported ops) one at a time. Each is a ~5-minute
fix+recompile cycle.

---

## 2026-06-12 — Decision: implement Option 2 (override/wrap, don't rewrite)



**Context:** vLLM-Omni Beta 1's base `LTX2Pipeline.forward()` is broken
against the diffusers 0.38 shipped in the same image. It calls
`self.connectors(prompt_embeds, additive_attention_mask, additive_mask=True)`
but diffusers 0.38 `LTX2TextConnectors.forward()` has no `additive_mask`
kwarg (signature is `forward(hidden_states, attention_mask,
padding_side='left', scale_factor=8)`). This is a vendor bug we cannot
fix by subclass overrides of the encoder/mask paths — the break is in
the middle of `forward()` itself.

**Decision:** Override `forward()` entirely in `NeuronLTX2Pipeline` so it
does NOT call vllm-omni's broken `pipeline_ltx2.forward()`. Instead, drive
the denoising loop ourselves (or call diffusers' own `LTX2Pipeline.__call__`
with our Neuron-resident transformer plugged in as `pipe.transformer`).

**Why this is acceptable:**
- It produces a real LTX-2 video on Trainium TODAY, which is the
  customer-facing deliverable.
- The hard part (Neuron-resident transformer + TP weight loading +
  encoder/VAE on CPU) is already done and reused.
- Trade-off: we partially bypass vllm-omni's serving orchestration for
  the forward path. We keep the Omni stage/worker/registration plumbing,
  but the actual generation logic is ours, not vllm-omni's.

**What we keep for when the vendor bug is fixed:**
- The `NeuronLTX2Pipeline` subclass (init, to, encode_prompt, load_weights)
- The Gemma3-CPU encoder bridge
- The mask dtype pre-cast pattern
- The PIPELINE_REGISTRY drop-in registration

When AWS-Neuron ships a Beta 2/3 Omni image with the connectors API
fixed, we can delete our `forward()` override and inherit the base
again.

---

## 2026-06-12 — Fixes that worked (climbing the ladder)

In order, each fix moved the failure deeper into the pipeline:

1. **Gemma3 encoder stays on CPU** (`_get_gemma_prompt_embeds` override).
   Fixed: `RuntimeError: Expected self.dtype() == dst.dtype()` at
   pipeline_ltx2.py:315 (encoder forward on Neuron inputs while model
   on CPU). Now: encode on CPU, move embeds to Neuron after final cast.

2. **Return (embeds, mask) tuple** from `_get_gemma_prompt_embeds`.
   Fixed: `ValueError: not enough values to unpack (expected 2, got 1)`
   at pipeline_ltx2.py:364 (base destructures two values).

3. **Pre-cast attention mask to bf16 on CPU** before moving to Neuron.
   Fixed: dtype trap at pipeline_ltx2.py:947 (the base does
   `(1 - mask.to(embeds.dtype)) * -1e6`, which trips on Neuron int→fp).

After all three: stage initializes, all 4 TP workers ready, transformer
loaded — then hit the vendor connectors bug (the wall that triggered the
Option 2 decision above).

---

## 2026-06-12 — Infra brought up (all working)

- Pulled vLLM-Omni Beta 1 image (`concourse-release-1cb0647:pr-655cd...`)
  to fal trn2.48xl (`i-02a51e30b3a33408d`, `3.150.135.217`).
- Moved docker + containerd storage to `/opt/dlami/nvme` (root was too
  small for the 44 GB image).
- `NEURON_SKIP_EFA_AFFINITY=1` is mandatory — workers die at init without
  it (no EFA sysfs on this host).
- HelloWorldPipeline + Wan2.2-T2V dev mode both validated end-to-end.
  Real Wan2.2 .mp4 generated → `customers/fal/results/wan22_dev_first.mp4`.
  This proves the vLLM-Omni runtime/scheduler/worker plumbing is healthy;
  the LTX block is specifically the vllm-omni LTX pipeline glue.
- Downloaded `Lightricks/LTX-2` weights (~291 GB, all variants).

---

## Reusable knowledge

- vLLM-Omni Neuron plugin auto-registers any module under
  `vllm_omni_neuron/diffusion/models/` that exposes a `PIPELINE_REGISTRY`
  list. Drop a file in, it registers, overriding base pipelines of the
  same `model_arch`.
- The framework weight loader walks `self.weights_sources`
  (`DiffusersPipelineLoader.ComponentSource`) and calls
  `pipeline.load_weights(iterable_of_name_tensor)`. For a Neuron subclass
  that only puts the transformer on device, set `weights_sources` to just
  the transformer subfolder and strip/re-prefix `transformer.` in the
  load_weights forwarder.
- Neuron's lazy backend rejects combined `.to(device, dtype)` — always
  split into `.to(dtype)` then `.to(device)`, and prefer doing the dtype
  cast on CPU before the device move.


---

## 2026-06-12 (later) — coord-prep patch worked; now whack-a-mole in forward()

The `_patch_coord_prep_to_cpu` fix WORKED — but I had to target the right
object. The methods live on `transformer.rope` / `transformer.audio_rope`
(NOT `transformer` directly), and `device` is passed POSITIONALLY:
    transformer.rope.prepare_video_coords(B, F, H, W, latents.device, fps=...)

After fixing that, the error moved from `pipeline_ltx2.py:1069`
(prepare_video_coords) to `pipeline_ltx2.py:1090`:
    video_coords = video_coords.repeat((2,) + (1,) * (video_coords.ndim - 1))
→ same `Expected self.is_contiguous()` class, now on the CFG-duplication
`.repeat()`.

### The real finding

This is whack-a-mole. vllm-omni's LTX-2 `forward()` + transformer were
never exercised on Neuron, so nearly every tensor-shape op
(`.repeat()`, `.flatten()`, `meshgrid`, in-place index assignment) hits a
Neuron lazy-backend quirk one line at a time. Cleared 4 so far
(connectors API, connectors dtype, video-coord prep, audio-coord prep),
now on the 5th, with likely 10+ more in the denoising loop + VAE decode.

### Cross-reference that confirms the diagnosis

`customers/Makora_27B/pr/src/qwen3_6/model_bf16.py` →
`Qwen3_6RotaryEmbedding` is the Neuron-SAFE RoPE pattern:
  - `inv_freq` built on `device="cpu"`
  - pure functional ops: `@`, `.cos()`, `.sin()`, `.to(dtype)`
  - NO in-place assignment, NO meshgrid/stack/repeat

vllm-omni's LTX coord prep violates every one of these. Same family of
model (Qwen3.6 is also head_dim=256, partial RoPE) — the contrast makes
it clear that vllm-omni's LTX path just wasn't written Neuron-aware.

### Decision for next session

Two options, both keep the NeuronLTX2Pipeline scaffolding we built:
  (a) Override the WHOLE `forward()` in NeuronLTX2Pipeline — copy
      vllm-omni's forward(), insert `.contiguous()` after every reshape/
      repeat and replace in-place index assignment with functional
      equivalents. Multi-hour but keeps vllm-omni serving.
  (b) Override `forward()` to drive diffusers' own `LTX2Pipeline.__call__`
      with our Neuron transformer as a swap-in. diffusers' reference impl
      may be more Neuron-tolerant than vllm-omni's half-port.

Recommend trying (a) first for ~30 min (we've already cleared 4 ops, the
remaining ones in forward() before the denoise loop are countable). If
the denoise loop turns out to be equally dense, switch to (b).

All fixes so far are committed on branch `ltx2-omni-bringup`.


---

## 2026-06-12 (even later) — reached the compiled attention backend

Continued clearing the ladder. Big milestone: we are now INSIDE the
torch.compile trace of the DiT transformer's attention. Fixes since the
last entry:

8. **Video latent dtype** (`pipeline_ltx2.py:1162`,
   `latent_model_input.to(prompt_embeds.dtype)`) — overrode
   `prepare_latents` to force bf16. The base calls it POSITIONALLY
   (dtype is arg index 6), so we rewrite `args[6]` rather than passing a
   kwarg (which collided → "got multiple values for argument 'dtype'").
   ✅ FIXED

9. **Audio latent dtype** (`pipeline_ltx2.py:1166`,
   `audio_latent_model_input.to(prompt_embeds.dtype)`) — overrode
   `prepare_audio_latents` (called with `dtype=` kwarg, so simpler).
   ✅ FIXED

10. **`video_coords.repeat()` contiguity** (`pipeline_ltx2.py:1090`) —
    keep coords on CPU through the pipeline's `.repeat()` (runs on CPU,
    fine); the patched `transformer.forward` moves them to the Neuron
    device + `.contiguous()` at the actual point of use. ✅ FIXED

### Current wall: data-dependent branch in the SDPA attention backend

```
File ".../vllm_omni/diffusion/attention/backends/sdpa.py", line 34,
  in _maybe_reshape_attn_mask
    if attn_mask is not None and torch.all(attn_mask != 0):
torch._dynamo.exc: Data-dependent branching — Dynamo does not support
tracing dynamic control flow.
```

`torch.all(attn_mask != 0)` produces a tensor; branching on it is
data-dependent control flow that torch.compile (Dynamo, fullgraph)
cannot trace. This is in vLLM-Omni's SDPA backend, reached via the
plugin's own `vllm_omni_neuron/diffusion/attention/sdpa.py:19`.

This is the 11th distinct issue and the first one INSIDE the compiled
graph (all prior ones were eager-mode pipeline glue). It confirms the
LTX path in vllm-omni Beta 1 was never compiled for Neuron — even the
attention backend has an untraceable branch.

### Honest assessment at this depth

We've cleared 10 issues and reached the compiled attention kernel. The
remaining work is now compiler-level (Dynamo graph breaks), which is a
different and deeper class than the eager dtype/contiguity fixes. Options:
  - Patch the plugin's `sdpa.py` to avoid the `torch.all(...)` branch
    (make mask handling static / unconditional). Plausible 1-fix.
  - But there may be more graph-break branches behind it in the same
    backend + the DiT blocks.

This is genuinely deep vendor-code surgery now. Each fix is real and
moving us forward, but we're patching vllm-omni's untested-on-Neuron
LTX path one issue at a time, and we've moved from "pipeline glue"
(fixable in our subclass) into "compiled-kernel internals" (requires
patching vllm-omni's own attention backend files in the container).

### Files patched in-container so far (NOT portable; container-local)
- `vllm_omni_neuron/diffusion/models/neuron_ltx2_pipeline.py` — OUR file
  (all 10 fixes live here, committed to ltx2-omni-bringup branch)

### Next: patch sdpa.py attention-mask branch
The `if torch.all(attn_mask != 0)` needs to become branchless. Likely
the intent is "if the mask is all-ones (no masking), skip applying it."
On a compiled graph we must always apply (or always skip) — make it
static based on whether a mask was passed at all, not its values.


---

## 2026-06-12 (latest) — Wrapper pattern works; hit DEEPER vllm-omni LTX bug (cross-attn mask shape)

### What worked — `_NeuronTransformerWrapper` resolved the catch-22

Adopted Jim Burtoft's NxDI port pattern (from `neuron/external/pr-117-
nxdi-diffusion-models/contrib/models/ltx2-video-audio/src/pipeline.py`):
wrap the DiT transformer in an eager `nn.Module` (`_NeuronTransformer
Wrapper`) that sits **outside** the `torch.compile` boundary and does
the CPU→Neuron coord move + contiguity right before calling the inner
compiled DiT.

This decisively unblocked the catch-22 between coords-on-CPU
(needed for the pipeline's `.repeat()` at line 1090) and coords-on-Neuron
(needed inside the compiled graph). With the wrapper:
- `_patch_coord_prep_to_cpu` now LEAVES coords on CPU (not moves to
  Neuron) — pipeline's `.repeat()` is a CPU op, fine.
- The wrapper's `forward()` then `.contiguous().to(device=neuron)` the
  coords eagerly, before the inner compiled call.
- `wrapper.compile(...)` only compiles the inner; wrapper stays eager.

Bonus side-effect: the framework's strict-load check goes through cleanly
because we re-prefix loaded names with `transformer.inner.` (the wrapper
introduces a `.inner` namespace in `named_parameters()`).

Result: weight loading, transformer init, all 4 TP workers ready, dummy
run started compiling — **passed every issue 1-11**.

### New issue 13 — vllm-omni LTX-2 cross-attn produces wrong-shape mask

Crashed in the dummy compile run on:
```
Dynamo failed: scaled_dot_product_attention(...,
  attn_mask=FakeTensor(size=(2, 256, 8, 128), bf16))
got: 'Attempting to broadcast a dimension of length 128 at -1!
Mismatching argument at index 1 had torch.Size([2, 256, 8, 128]);
but expected shape should be broadcastable to [2, 8, 256, 1024]'
```

Site: `vllm_omni/diffusion/models/ltx2/ltx2_transformer.py:469`
(LTX2AudioVideoAttnProcessor.__call__ → attn.attn → SDPA at backend
sdpa.py:108).

The mask shape `[2, 256, 8, 128]` looks like the QUERY tensor
pre-permute (B, S_q, H_local, head_dim), not a real attention mask.
Expected shapes for cross-attn:
- query: `[2, 8, 256, 128]` (B, H, S_q, head_dim) ✓
- key: `[2, 8, 1024, 128]` ✓
- attn_mask: `[2, 1, 1, 1024]` or `[2, 8, 1, 1024]` (broadcast)
- expected SDPA output dims: `[2, 8, 256, 128]` (broadcasts over scores `[2,8,256,1024]`)

We got attn_mask=`[2, 256, 8, 128]` instead. That's the unpermuted
query shape leaking into the mask slot.

This is **deeper** than issues 1-11 — those were eager-mode pipeline
glue. This one is in the LTX-2 attention processor / cross-attn path
that vllm-omni inherited from a non-Neuron-tested origin. Tracing
through `_prepare_attention_mask` -> `prepare_attention_mask` ->
`.view(B, heads, -1, S_k)` — every step on paper produces the right
4D mask shape. So the bug is somewhere subtle: maybe the FX trace has
the call to `attention_mask=...` and `key=...` swapped on a
particular block, or the SP-plan hooks introduce a phantom rebind.

### What this means

We are now squarely in vendor-bug territory inside a code path that
needs deep diffing against the upstream LTX-2 reference. Three options:

1. **Patch the cross-attn processor** to avoid the shape collision.
   We have local copies of `pipeline_ltx2.py` and `ltx2_transformer.py`
   in `.tmp/`. Could trace through diffusers' reference LTX2 attn
   processor to see what's different and apply targeted patches in
   the in-container `ltx2_transformer.py`.

2. **Pivot to the NxDI-style approach using diffusers' OWN
   LTX2Pipeline.__call__**. This is the same pattern Jim's contrib
   port uses: keep our `NeuronLTX2Pipeline` scaffolding for vllm-omni
   wiring (registry, weight load, encoder/connector compat), but
   override `forward(req)` to delegate to `diffusers.LTX2Pipeline.
   __call__()` with our Neuron-resident DiT plugged in via
   `pipe.transformer = self._NeuronTransformerWrapper`. This bypasses
   ALL of vllm-omni's `pipeline_ltx2.forward()` + `ltx2_transformer.
   forward()` — diffusers' reference forward goes through
   diffusers-native `LTX2VideoTransformer3DModel` (without vLLM-Omni's
   parallel-layer port) but uses our compiled inner.

3. **Wait for vllm-omni's next release** which presumably fixes their
   LTX-2 path (we are on Beta 1; the team is iterating).

Option 2 is the most promising — it sidesteps vllm-omni's broken LTX
internals entirely while keeping the working scaffold. But it requires
also wrapping the diffusers-native DiT to do TP weight slicing
(diffusers' DiT is not vLLM parallel-layer-aware), or running the
DiT un-sharded on a single 96GB Neuron core (LTX-2 19B might
just fit).

### Files touched this round
- `customers/fal/neuron_ltx2_pipeline.py` — added `_NeuronTransformer
  Wrapper`, removed runtime sdpa monkey-patch (in-container patch
  already covers it), updated `load_weights` to handle the new
  `transformer.inner.*` namespace.



---

## Path 2 design sketch — diffusers LTX2Pipeline.__call__ swap-in (next session)

If we pivot to Path 2, the implementation is straightforward. Here's the
design so we can pick up cleanly.

### The structure

`NeuronLTX2Pipeline` (subclass of vllm-omni `LTX2Pipeline`) keeps its:
- registry entry (auto-loads as the `LTX2Pipeline` model_arch)
- `weights_sources` (transformer subfolder only)
- `to(...)` (selective device move)
- `compile(...)` (compile transformer only; encoder/VAE eager on CPU)

But adds:
- A NEW `__init__` path that builds a `diffusers.LTX2Pipeline` instance
  alongside (or instead of) the components we currently load piecemeal.
- A `_swap_transformer_to_neuron()` method (mirroring Jim Burtoft's
  `NeuronLTX2Pipeline._swap_transformer_to_neuron`) that:
    1. Takes the diffusers pipe's `pipe.transformer` (full diffusers DiT)
    2. Wraps it in `_NeuronTransformerWrapper` (already written)
    3. Stores the wrapper as `self.transformer` AND assigns it to
       `pipe.transformer` (so diffusers' forward() sees our wrapper)
    4. Frees the original transformer blocks

- A `forward(req)` override that drives diffusers' pipeline directly:
    ```python
    def forward(self, req):
        prompt = req.prompts[0]
        params = req.sampling_params
        result = self._diffusers_pipe(
            prompt=prompt,
            negative_prompt=params.negative_prompt,
            height=params.height, width=params.width,
            num_frames=params.num_frames,
            num_inference_steps=params.num_inference_steps,
            guidance_scale=params.guidance_scale,
            generator=torch.Generator("cpu").manual_seed(params.seed),
            output_type="pt",
        )
        return DiffusionOutput(output=result.frames)
    ```

### Trade-offs

- **NO tensor parallelism** — diffusers' DiT isn't vLLM-parallel-aware.
  LTX-2 19B at bf16 = ~38 GB weights, fits in a single 96 GB Neuron
  core. Inference is slower than TP=4 but it WORKS.
- Stage YAML: `tensor_parallel_size: 1` (or just don't set it).
- Loses vllm-omni's CFG-parallel + sequence-parallel features. For
  fal.ai's per-request video gen workload, that's probably fine —
  customers care about end-to-end latency per video, not max
  throughput.
- Skip-warmup still works the same way.

### Anti-patterns to avoid
- Don't try to load the diffusers DiT with vLLM parallel layers — it
  won't work. The whole point of Path 2 is to USE diffusers' native
  forward as-is.
- Don't try to also keep using vllm-omni's `pipeline_ltx2.forward`.
  That's the broken thing we're walking around.

### Ports that already use this pattern (proof points)
- Jim Burtoft's NxDI port: `neuron/external/pr-117-nxdi-diffusion-models/contrib/models/ltx2-video-audio/src/pipeline.py`
- Yifan's LTX-2.3 inference scaffold: `neuron/examples/LTX/ltx23_pipeline_v3.py`
- Both use `LTX2Pipeline.__call__` with a Neuron-resident transformer
  swap-in.

