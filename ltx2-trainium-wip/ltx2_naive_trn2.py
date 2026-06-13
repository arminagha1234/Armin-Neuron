"""Simplest possible LTX-2 on Trainium2 — single core, no TP, no
distributed. Pure native PyTorch + torch_neuronx using
`pipe.to(privateuseone:0)`.

If LTX-2's transformer fits in 24 GB user budget on a single core,
this is the lowest-effort native-pytorch path. If not, we know we
need TP=4 (which requires the Beta 2 DLC for the 'neuron' PG).
"""
import os
import sys
import time

os.environ.setdefault("TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch_neuronx  # noqa: F401

print(f"[smoke] torch={torch.__version__}", flush=True)
print(f"[smoke] torch_neuronx imported", flush=True)

# Pick a Neuron device
device = torch.device("privateuseone:0")
print(f"[smoke] device={device}", flush=True)

# Probe the device by creating a tiny tensor
try:
    x = torch.randn(4, 4, device=device)
    y = x @ x.T
    print(f"[smoke] device probe OK; result on {y.device}, shape={tuple(y.shape)}",
          flush=True)
except Exception as e:
    print(f"[smoke] device probe FAILED: {type(e).__name__}: {e}", flush=True)
    sys.exit(2)

print(f"\n[smoke] loading Lightricks/LTX-2 on CPU (bf16)...", flush=True)
t0 = time.time()
from diffusers import LTX2Pipeline
pipe = LTX2Pipeline.from_pretrained(
    "Lightricks/LTX-2",
    torch_dtype=torch.bfloat16,
)
print(f"[smoke] loaded in {time.time() - t0:.1f}s", flush=True)
print(f"[smoke] components: text_encoder={type(pipe.text_encoder).__name__}, "
      f"transformer={type(pipe.transformer).__name__}, "
      f"vae={type(pipe.vae).__name__}", flush=True)

# Param count quick check
def n_params(m):
    return sum(p.numel() for p in m.parameters())

print(f"[smoke] text_encoder params: {n_params(pipe.text_encoder)/1e9:.2f}B", flush=True)
print(f"[smoke] transformer params: {n_params(pipe.transformer)/1e9:.2f}B", flush=True)
print(f"[smoke] vae params: {n_params(pipe.vae)/1e9:.2f}B", flush=True)

# Move just transformer to Neuron first — see if it fits one core
print(f"\n[smoke] moving transformer to {device}...", flush=True)
t0 = time.time()
try:
    pipe.transformer = pipe.transformer.to(device)
    print(f"[smoke] ✓ transformer moved in {time.time() - t0:.1f}s", flush=True)
except Exception as e:
    import traceback
    print(f"[smoke] ✗ transformer.to failed: {type(e).__name__}: {e}",
          flush=True)
    traceback.print_exc()
    sys.exit(3)

# Try VAE too
print(f"\n[smoke] moving VAE to {device}...", flush=True)
try:
    pipe.vae = pipe.vae.to(device)
    print(f"[smoke] ✓ VAE moved", flush=True)
except Exception as e:
    print(f"[smoke] ✗ VAE.to failed: {type(e).__name__}: {e}", flush=True)

# Don't bother with text encoder — keep on CPU
print(f"\n[smoke] keeping text_encoder on CPU (Gemma3-12B too big)", flush=True)

# Force the pipeline's device routing
class _device_property:
    def __get__(self, obj, objtype=None):
        return device
LTX2Pipeline._execution_device = property(lambda self: device)

# Patch text encoder calls to keep on CPU
cpu = torch.device("cpu")
_orig_get = pipe.encode_prompt
import types

def _patched_encode_prompt(self, *a, **kw):
    # Force CPU for text encoder, then move embeds to Neuron
    kw["device"] = cpu
    out = _orig_get.__wrapped__(self, *a, **kw) if hasattr(_orig_get, "__wrapped__") else _orig_get(*a, **kw)
    # encode_prompt returns (prompt_embeds, prompt_attention_mask, neg, neg_mask)
    return tuple(t.to(device) if torch.is_tensor(t) else t for t in out)

# Try one tiny generation: 4 steps, smallest config
print(f"\n[smoke] attempting 4-step generation, 25 frames, 384×512...",
      flush=True)
t0 = time.time()
try:
    with torch.no_grad():
        out = pipe(
            prompt="a golden retriever runs across a meadow",
            height=384, width=512,
            num_frames=25,
            num_inference_steps=4,
            guidance_scale=4.0,
            max_sequence_length=1024,
            generator=torch.Generator(device="cpu").manual_seed(42),
            output_type="pil",
        )
    elapsed = time.time() - t0
    print(f"[smoke] ✓ generation complete in {elapsed:.1f}s, "
          f"frames={len(out.frames[0])}", flush=True)
except Exception as e:
    elapsed = time.time() - t0
    import traceback
    print(f"[smoke] ✗ generation failed after {elapsed:.1f}s: "
          f"{type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    sys.exit(4)
