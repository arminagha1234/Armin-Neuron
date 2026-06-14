"""Z-Image-Turbo on Neuron — real-arithmetic RoPE patch.

Patches:
1. RopeEmbedder.precompute_freqs_cis → returns (cos, sin) pairs instead of complex
2. RopeEmbedder.__call__ → returns (cos_cat, sin_cat) instead of complex cat
3. ZImageTransformer2DModel._prepare_sequence → handles tuple freqs
4. ZSingleStreamAttnProcessor.__call__ → uses cos/sin rotation instead of complex mul
"""
import torch
import time

# ── Patch RopeEmbedder ───────────────────────────────────────────────────────
import diffusers.models.transformers.transformer_z_image as z_mod
from torch.nn.utils.rnn import pad_sequence

_OrigRopeEmbedder = z_mod.RopeEmbedder


class RealRopeEmbedder:
    """Drop-in replacement for RopeEmbedder using real arithmetic only."""

    def __init__(self, theta=256.0, axes_dims=(16, 56, 56), axes_lens=(64, 128, 128)):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.freqs_cos = None
        self.freqs_sin = None

    def _precompute(self):
        cos_list, sin_list = [], []
        for d, e in zip(self.axes_dims, self.axes_lens):
            freqs = 1.0 / (self.theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
            t = torch.arange(e, dtype=torch.float32)
            angles = torch.outer(t, freqs)
            cos_list.append(angles.cos())
            sin_list.append(angles.sin())
        self.freqs_cos = cos_list
        self.freqs_sin = sin_list

    def __call__(self, ids):
        device = ids.device
        if self.freqs_cos is None:
            self._precompute()
            self.freqs_cos = [c.to(device) for c in self.freqs_cos]
            self.freqs_sin = [s.to(device) for s in self.freqs_sin]
        elif self.freqs_cos[0].device != device:
            self.freqs_cos = [c.to(device) for c in self.freqs_cos]
            self.freqs_sin = [s.to(device) for s in self.freqs_sin]

        cos_parts, sin_parts = [], []
        for i in range(len(self.axes_dims)):
            index = ids[:, i].long()
            cos_parts.append(self.freqs_cos[i][index])
            sin_parts.append(self.freqs_sin[i][index])

        # Return (cos_cat, sin_cat) as a tuple
        return (torch.cat(cos_parts, dim=-1), torch.cat(sin_parts, dim=-1))


z_mod.RopeEmbedder = RealRopeEmbedder


# ── Patch _prepare_sequence to handle tuple freqs ────────────────────────────

_orig_prepare_sequence = z_mod.ZImageTransformer2DModel._prepare_sequence


def _patched_prepare_sequence(self, *args, **kwargs):
    """Wraps _prepare_sequence: handles (cos, sin) tuple from RealRopeEmbedder."""
    # Call original but intercept the rope_embedder call
    # The original does: freqs_cis = list(self.rope_embedder(cat).split(lens))
    # We need to do it ourselves since it's inline

    # Temporarily make rope_embedder return a stacked tensor that can be split
    # Strategy: override _prepare_sequence entirely with the patched logic
    import inspect
    # Get the original source and replicate with our tuple handling
    # Actually simpler: just monkey-patch to handle tuples
    result = _orig_prepare_sequence(self, *args, **kwargs)
    return result


# Better approach: patch rope_embedder to return a SINGLE tensor that encodes
# cos/sin side-by-side (double the freq dim), then split in the attn processor.
# This way _prepare_sequence works unchanged (it just sees a bigger tensor).

class RealRopeEmbedderCompat:
    """Returns a single tensor [seq, 2*freq_dim] with cos||sin concatenated.
    
    _prepare_sequence can .split() and pad_sequence this normally.
    The attention processor separates cos/sin by splitting the last dim in half.
    """

    def __init__(self, theta=256.0, axes_dims=(16, 56, 56), axes_lens=(64, 128, 128)):
        self.theta = theta
        self.axes_dims = axes_dims
        self.axes_lens = axes_lens
        self.freqs_cos = None
        self.freqs_sin = None

    def _precompute(self):
        cos_list, sin_list = [], []
        for d, e in zip(self.axes_dims, self.axes_lens):
            freqs = 1.0 / (self.theta ** (torch.arange(0, d, 2, dtype=torch.float32) / d))
            t = torch.arange(e, dtype=torch.float32)
            angles = torch.outer(t, freqs)
            cos_list.append(angles.cos())
            sin_list.append(angles.sin())
        self.freqs_cos = cos_list
        self.freqs_sin = sin_list

    def __call__(self, ids):
        device = ids.device
        if self.freqs_cos is None:
            self._precompute()
            self.freqs_cos = [c.to(device) for c in self.freqs_cos]
            self.freqs_sin = [s.to(device) for s in self.freqs_sin]
        elif self.freqs_cos[0].device != device:
            self.freqs_cos = [c.to(device) for c in self.freqs_cos]
            self.freqs_sin = [s.to(device) for s in self.freqs_sin]

        cos_parts, sin_parts = [], []
        for i in range(len(self.axes_dims)):
            index = ids[:, i].long()
            cos_parts.append(self.freqs_cos[i][index])
            sin_parts.append(self.freqs_sin[i][index])

        cos_cat = torch.cat(cos_parts, dim=-1)  # (total_seq, freq_dim)
        sin_cat = torch.cat(sin_parts, dim=-1)  # (total_seq, freq_dim)
        # Return cos||sin concatenated: (total_seq, 2*freq_dim)
        # This is a single tensor that .split() and pad_sequence handle normally
        return torch.cat([cos_cat, sin_cat], dim=-1)


# Use the compat version
z_mod.RopeEmbedder = RealRopeEmbedderCompat


# ── Patch ZSingleStreamAttnProcessor ─────────────────────────────────────────

_OrigAttnProcessor = z_mod.ZSingleStreamAttnProcessor


class RealRopeAttnProcessor:
    """ZSingleStreamAttnProcessor with real-arithmetic RoPE."""

    _attention_backend = None
    _parallel_config = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, freqs_cis=None):
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Real-arithmetic RoPE (replaces view_as_complex path)
        if freqs_cis is not None:
            query = self._apply_rotary(query, freqs_cis)
            key = self._apply_rotary(key, freqs_cis)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        from diffusers.models.transformers.transformer_z_image import dispatch_attention_fn
        hidden_states = dispatch_attention_fn(
            query, key, value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)

        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)
        return output

    @staticmethod
    def _apply_rotary(x, freqs_cis):
        """Real-arithmetic rotary embedding.

        x: (batch, seq, heads, head_dim) — UNFLATTEN format
        freqs_cis: (batch, seq, 2*freq_dim) — cos||sin concatenated
        """
        # Split cos and sin from the double-width tensor
        freq_dim = freqs_cis.shape[-1] // 2
        cos = freqs_cis[..., :freq_dim]  # (batch, seq, freq_dim)
        sin = freqs_cis[..., freq_dim:]  # (batch, seq, freq_dim)

        x_float = x.float()
        d = x_float.shape[-1]
        x_reshape = x_float.reshape(*x_float.shape[:-1], d // 2, 2)
        x_even = x_reshape[..., 0]  # (batch, seq, heads, d//2)
        x_odd = x_reshape[..., 1]

        # cos/sin: (batch, seq, freq_dim) → need (batch, seq, 1, freq_dim) for heads broadcast
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)

        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        out = torch.stack([out_even, out_odd], dim=-1).flatten(-2)
        return out.to(x.dtype)


# ── Patch the transformer to use our RopeEmbedder + processor ────────────────

_orig_transformer_init = z_mod.ZImageTransformer2DModel.__init__


def _patched_init(self, *args, **kwargs):
    _orig_transformer_init(self, *args, **kwargs)
    # Replace rope_embedder with our real-arithmetic version
    self.rope_embedder = RealRopeEmbedder(
        theta=self.rope_theta,
        axes_dims=self.axes_dims,
        axes_lens=self.axes_lens,
    )
    # Replace attention processors
    count = 0
    for module in self.modules():
        if hasattr(module, 'processor') and isinstance(module.processor, _OrigAttnProcessor):
            module.processor = RealRopeAttnProcessor()
            count += 1
    print(f"[z-image-patch] Replaced rope_embedder + {count} attn processors", flush=True)


z_mod.ZImageTransformer2DModel.__init__ = _patched_init

print("[z-image-patch] Real-arithmetic RoPE patches installed")

# ── Load and run on Neuron ───────────────────────────────────────────────────

import torch_xla.core.xla_model as xm

print("Loading Z-Image-Turbo...")
from diffusers import ZImagePipeline

pipe = ZImagePipeline.from_pretrained(
    "Tongyi-MAI/Z-Image-Turbo", torch_dtype=torch.bfloat16
)

# ── Replace rope_embedder INSTANCE after loading ─────────────────────────────
transformer = pipe.transformer
transformer.rope_embedder = RealRopeEmbedderCompat(
    theta=transformer.rope_theta,
    axes_dims=transformer.axes_dims,
    axes_lens=transformer.axes_lens,
)
# Replace attention processors
count = 0
for module in transformer.modules():
    if hasattr(module, 'processor') and isinstance(module.processor, _OrigAttnProcessor):
        module.processor = RealRopeAttnProcessor()
        count += 1
print(f"[z-image-patch] Replaced rope_embedder + {count} attn processors on loaded model")

dev = xm.xla_device()
print(f"Moving to Neuron device: {dev}")
pipe = pipe.to(dev)
print("On Neuron. Generating...")

gen = torch.Generator(device="cpu").manual_seed(42)
t0 = time.time()
out = pipe(
    prompt="A photorealistic golden retriever puppy sitting in a sunny garden with colorful flowers",
    height=512,
    width=512,
    num_inference_steps=8,
    guidance_scale=3.5,
    generator=gen,
    output_type="pil",
)
elapsed = time.time() - t0
print(f"Generated in {elapsed:.1f}s")
out.images[0].save("/mnt/data/ltx2_work/z_image_neuron_v3.png")
print("Saved /mnt/data/ltx2_work/z_image_neuron_v3.png")
