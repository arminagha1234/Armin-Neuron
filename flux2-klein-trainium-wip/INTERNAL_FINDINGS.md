# Internal Amazon code search — FLUX.2 on Neuron is already shipping via NxDI

After hitting the vllm-omni Dynamo-vs-monkey-patch wall, I searched internal
code for prior FLUX.2 + Neuron work. Big finding: a different team has FLUX.2
**GREEN on Neuron** via a totally different stack.

## Summary

- **`NeuronAutoFixerAIM`** repo, `oncall_agent/generated_models/dit/flux2-dit-on-nxdi-neuron/`
  has a complete FLUX.2 NxDI port:
  - `neuron_flux2_dit.py` — full custom DiT (replaces diffusers' transformer)
  - `flux2_pipeline.py` — FlowMatchEuler scheduler
  - `flux2_tp_shard.py` — Megatron TP shard for fused-SwiGLU + fused-qkv-mlp
  - `flux2_dev_tp_run.py` — production-shape compile + run script
- **Status:** GREEN on FLUX.2-dev (32B sibling). Real on-Neuron run
  `T_65f22500-48c5-4963-b297-5c7f43ecec51`, generated 512×512 PNG,
  CLIP-score 0.4246. Run on trn2.48xlarge TP=8 LNC=2.
- **Klein-specific note** in their script header:
  > "the FLUX.2 DiT denoiser bf16 (~64 GB) cannot be traced on a single
  > NeuronCore (**klein-4B barely cleared @512^2**; dev OOMs / EBVF030)"

So FLUX.2-klein-4B at 512² works on a SINGLE core via their NxDI path.
At 1024² (production) it likely needs TP=2.

## Why it works (and our vllm-omni path doesn't)

The NxDI `parallel_model_trace` path uses the upstream diffusers
`apply_rotary_emb` + diffusers' `Flux2PosEmbed.forward` AS-IS. The trace
backend is **NEFF compilation via the static-shape NxDI path**, NOT
`torch.compile(backend="neuron")` via Dynamo.

In our vllm-omni stack:
- vllm-omni runs the model under `torch.compile(backend="vllm_neuron")`
- Dynamo introspects every nn.Module forward and inlines the bytecode
- `torch.polar` (used by diffusers' `get_1d_rotary_pos_embed(use_real=False)`)
  goes into the FX graph, then segfaults at execute (Neuron has no complex64)
- Monkey-patching the forward (instance-level OR class-level) doesn't change
  what Dynamo emits — it's keyed on bytecode

In the NxDI path:
- `parallel_model_trace` runs the forward with sample inputs
- HLO-level lowering of `torch.polar` is broken — same root cause
- BUT NxDI evidently has a workaround at the lowering level (or the larger
  `inline_weights_to_neff` + per-rank build avoids the bad lowering)
- Result: it compiles. Real PNG out the other end.

## Proven recipe (from their script)

```python
# 1. Build the diffusers Flux2Pipeline on CPU
pipe = Flux2Pipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", torch_dtype=bf16)
transformer = pipe.transformer

# 2. Capture one real forward's inputs (probe one pipeline step)
captured = {}
def _cap(**kw): captured.update(kw); raise StopIteration
transformer.forward = _cap
try: pipe(prompt=..., ...)
except StopIteration: pass
example = (captured["hidden_states"], captured["encoder_hidden_states"], ...)

# 3. parallel_model_trace with the gated-SwiGLU + fused-qkv-mlp shard
traced = parallel_model_trace(
    _build_sharded_transformer,   # rebuilds + applies flux2_tp_shard per rank
    example,
    tp_degree=TP,                  # 1 for klein@512², 2 for klein@1024²
    compiler_args="--model-type=transformer -O1 --lnc=2 ...",
    inline_weights_to_neff=True,
)
parallel_model_save(traced, "/path/to/sharded_neff/")

# 4. Wrap the traced NEFF as a drop-in transformer replacement
class NeuronDenoiser(nn.Module):
    def __init__(self, traced, ref): ...
    def __call__(self, hidden_states=..., return_dict=True, ...):
        out = self.traced(...)
        return Transformer2DModelOutput(sample=out)

pipe.transformer = NeuronDenoiser(parallel_model_load(SDIR), transformer)

# 5. Generate
img = pipe(prompt=..., num_inference_steps=8, height=512, width=512, ...).images[0]
```

## Gap between their work and our customer ask

| | Their FLUX.2-dev port | Our FLUX.2-klein-4B + zoom-LoRA |
|---|---|---|
| Model | 32B `FLUX.2-dev` (gated, single-files) | 4B `FLUX.2-klein-4B` (open) |
| TP | 8 (required for 32B) | likely 1 @ 512², 2 @ 1024² |
| LoRA | none | `fal/flux-2-klein-4B-zoom-lora` (need to merge offline) |
| Pipeline | text-to-image (`Flux2Pipeline`) | image-to-image (`Flux2KleinPipeline`) |
| Status | GREEN, compile + image | not started on this path |

## Action plan (replaces the vllm-omni path)

1. **Pull the AutoFixer code into our workspace** as a starting point.
2. **Adapt for klein**: swap `Flux2Pipeline` → `Flux2KleinPipeline` (image-
   to-image), add the `image=` input handling.
3. **Adapt for the LoRA**: merge `fal/flux-2-klein-4B-zoom-lora` into base
   weights using our existing `merge_lora.py`, point at merged dir.
4. **Try TP=1 first** at 512² (matching their klein note), then TP=2 at
   1024² for the production shape.
5. **Wire into a customer-grade serving wrapper** (FastAPI like Qwen-Image-
   Edit Path C) once the model itself works on-Neuron.

This bypasses vllm-omni entirely — same as the LTX-2 native PyTorch path
that shipped successfully (PR #7). vllm-omni-omni serving is a Phase 2
once the model is proven on-Neuron via NxDI.

## Repo references

- `code.amazon.com/packages/NeuronAutoFixerAIM/blobs/mainline/--/oncall_agent/generated_models/dit/flux2-dit-on-nxdi-neuron/`
  - `neuron_flux2_dit.py` (570 lines, complete denoiser)
  - `flux2_tp_shard.py` (483 lines, Megatron shard + CPU rank-3 proof)
  - `flux2_pipeline.py` (412 lines, FlowMatchEuler scheduler)
  - `flux2_dev_tp_run.py` (412 lines, runnable recipe)
  - `test_flux2_dit_port.py` (CPU rank-3 unit tests, 15 cases)
  - `test_flux2_tp_shard.py` (CPU rank-3 TP shard correctness)
- `code.amazon.com/packages/NeuronAutoFixerAIM/blobs/mainline/--/oncall_agent/enabled_models/dit/flux2_dev_tp8_512_dattn/MODEL.md`
  (the GREEN report for the 32B dev, real PNG + CLIP score)

## Note for the next session

The vllm-omni infrastructure I built (PR #9) is NOT wasted — it stays valid
for production-shape multi-tenant serving once the NxDI path proves the
model works on Neuron. The two paths are complementary:

1. **NxDI** = "does the model work on Neuron at all? what's the latency?"
   → benchmark answer
2. **vllm-omni** = "drop-in replacement for the GPU vLLM stack at scale"
   → production deployment shape

Same pattern as LTX-2: native PyTorch (NxDI-equivalent) shipped first, then
vllm-omni was attempted in parallel.
