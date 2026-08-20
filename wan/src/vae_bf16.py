"""VAE decode in bf16 (a bf16 decoder is used) — steady-state vs the fp32 41.7s.

Two timed decodes: first = compile, second = steady. Prints both.
  NEURON_CC_FLAGS="--model-type=unet-inference -O1 --auto-cast=none" python -u vae_bf16.py
"""
import time, torch
from diffusers import AutoencoderKLWan

m = AutoencoderKLWan.from_pretrained("/home/ubuntu/wan22", subfolder="vae",
                                     torch_dtype=torch.bfloat16).to("neuron").eval()
lat = torch.randn(1, 48, 13, 30, 52, dtype=torch.bfloat16, device="neuron")

def decode():
    t0 = time.time()
    with torch.no_grad():
        out = m.decode(lat)
    o = out.sample if hasattr(out, "sample") else out
    float(o.detach().float().flatten()[:1].cpu())
    return time.time() - t0

print(f"VAE_BF16_FIRST_SECONDS {decode():.1f}", flush=True)
print(f"VAE_BF16_STEADY_SECONDS {decode():.1f}", flush=True)
import os; os._exit(0)
