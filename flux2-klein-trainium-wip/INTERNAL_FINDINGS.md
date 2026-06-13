# Internal Amazon code search — real-arithmetic RoPE math we can lift

After hitting the segfault on `torch.polar` in our vllm-omni FLUX.2-klein
pipeline, I searched internal code for prior FLUX.2 + Neuron work to see
how anyone handled the same complex64 issue.

## What I found

- **`NeuronAutoFixerAIM`** has FLUX.2 working on Neuron — they hit the
  exact same complex64/`torch.polar` problem (their docstring even calls
  out the same SIGKILL: *"torch-xla's complex lowering is incomplete →
  tracing the graph on Neuron HARD-CRASHES the process"*) and solved it
  by replacing the math with a real-valued rotation that produces the
  same outputs.

## The math we should lift (no `torch.polar`)

From their `neuron_flux2_dit.py`:

```python
def flux2_get_1d_rope(dim: int, pos: torch.Tensor, theta: float):
    """Real-valued (cos, sin) — equivalent to diffusers'
    get_1d_rotary_pos_embed(use_real=True, repeat_interleave_real=True)
    BUT without ever building a complex tensor.

    Returns: cos, sin — both shape (S, dim), all real fp32.
    """
    assert dim % 2 == 0
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2,
                                          dtype=torch.float64,
                                          device=pos.device) / dim))
    freqs = torch.outer(pos.double(), freqs)        # [S, dim/2]
    cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, dim]
    sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, dim]
    return cos, sin


def flux2_apply_rope(x, cos, sin):
    """Apply RoPE via the textbook real-arithmetic formula:
        out_real = x_real*cos - x_imag*sin
        out_imag = x_real*sin + x_imag*cos
    Equivalent to view_as_complex(x) * (cos+i*sin) -> view_as_real,
    but never builds a complex tensor.

    x:   (B, S, H, Dh)
    cos: (S, Dh)    sin: (S, Dh)
    """
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
    x_rot = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
    return (x.float() * cos + x_rot.float() * sin).to(x.dtype)
```

Their docstring confirms this is bit-equivalent to the complex version.
The whole file is at:
`code.amazon.com/packages/NeuronAutoFixerAIM/blobs/mainline/--/oncall_agent/generated_models/dit/flux2-dit-on-nxdi-neuron/neuron_flux2_dit.py#L195-L256`

(There's also a Lumina2 patch in the same repo at
`oncall_agent/generated_models/dit/sd35-large-mmdit-on-neuron/sd3_image_run.py:195`
named `_real_apply_rotary_emb` — same idea.)

## Why our last `_NeuronFluxPosEmbed` swap didn't take

Our v1 attempt called `get_1d_rotary_pos_embed(..., use_real=False)` then
did `.real`/`.imag` — which means a complex tensor still gets built
(just on CPU). When Dynamo traces `Flux2PosEmbed.forward` it inlines the
ORIGINAL bytecode (which still has `torch.polar` inside
`get_1d_rotary_pos_embed(use_real=False)` regardless of our swap), so
the FX graph still emits `torch.polar`.

The fix: replace the math AT THE BYTECODE LEVEL by patching
`diffusers.models.embeddings.get_1d_rotary_pos_embed` with a real-only
implementation that NEVER calls `torch.polar`. Even if Dynamo inlines
the original `Flux2PosEmbed.forward`, when it follows the inner call
to `get_1d_rotary_pos_embed`, it'll see our patched version's bytecode.

## Plan for next session (stays on vllm-omni)

1. **Add a `_real_get_1d_rotary_pos_embed` function** to
   `neuron_flux2_klein_pipeline.py` using the formula above.
2. **Patch `diffusers.models.embeddings.get_1d_rotary_pos_embed` at
   import time** in our pipeline's `__init__` (module-level patch — same
   trick we used for `get_timestep_embedding`).
3. **Also patch `apply_rotary_emb`** in
   `diffusers.models.embeddings` if Dynamo inlines that too (it's
   called from `Flux2Attention.forward` / `Flux2ParallelSelfAttention.forward`).
4. **Re-run** with the smoke test. The polar count in the FX graph
   should go from 16 → 0; if the segfault root cause was complex64,
   the segfault should go too.

## What this means for fal

- Same vllm-omni production-shape PR (#9), one more module-level patch,
  retry.
- We are staying on vllm-omni — no NxDI, no native-PyTorch fork — per
  the team's direction.
- If this works the customer story is unchanged: vllm-omni is the
  production-deploy shape; the fix removes the last surfacing complex64
  op in the FLUX.2 transformer trace.

## Honest expectation

The `torch.polar` bytecode-level patch is the most likely cleanly-stuck
fix we haven't tried yet. The Lumina2 precedent says it works on Neuron.
But there's a real chance the segfault has a second cause behind it
(activation memory, an unsupported XLA lowering of a different op, etc.)
that only becomes visible once polar is gone.
