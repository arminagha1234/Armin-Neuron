"""Patch upstream Flux2PosEmbed.forward to be real-valued + CPU+fp32.

Edits the source file in-place to replace the forward() body with a
version that:
  1. Runs on CPU (use_real=True paths) to avoid torch.polar in graph
  2. Uses float32 freqs (Neuron has no float64)
  3. Returns real (cos, sin) tensors moved back to ids.device

Used as a one-shot patch for the vllm_omni container source. Run as:
    docker exec vllm_omni python3 upstream_patch.py
"""
import re
from pathlib import Path

SRC = Path("/opt/conda/lib/python3.12/site-packages/vllm_omni/diffusion/models/flux2_klein/flux2_klein_transformer.py")

NEW_FORWARD = '''    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # NEURON PATCH: Run on CPU + use_real=True to avoid torch.polar
        # (complex64) in the FX graph. Neuron Beta 3 has no complex
        # dtype support so torch.polar segfaults the runtime.
        # Move ids to CPU for freq compute, return cos/sin as real fp32
        # tensors on the original device.
        orig_device = ids.device
        ids_cpu = ids.to(device="cpu") if str(orig_device) != "cpu" else ids
        cos_out = []
        sin_out = []
        pos = ids_cpu.float()
        for i in range(len(self.axes_dim)):
            cos_i, sin_i = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[..., i],
                theta=self.theta,
                use_real=True,            # NEURON: real (cos, sin) instead of complex
                freqs_dtype=torch.float32,  # NEURON: fp32 instead of fp64
                # use_real=True returns each axis as (seq, dim/2) (NOT
                # (seq, dim)), so total cat over 4 axes of dim=32 each
                # = 64 — matching the original .real/.imag split shape.
                # We pass repeat_interleave=False to keep the fp32 split.
                repeat_interleave_real=False,
            )
            cos_out.append(cos_i.contiguous())
            sin_out.append(sin_i.contiguous())
        freqs_cos = torch.cat(cos_out, dim=-1).to(device=orig_device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(device=orig_device)
        return freqs_cos, freqs_sin
'''

src = SRC.read_text()

# Match the original forward (multiline, from def forward through return freqs_sin)
pattern = re.compile(
    r"    def forward\(self, ids: torch\.Tensor\) -> tuple\[torch\.Tensor, torch\.Tensor\]:\n"
    r"(?:.*\n)+?"
    r"        return freqs_cos, freqs_sin\n",
    re.MULTILINE,
)

# Only replace the FIRST match (Flux2PosEmbed.forward, not Flux2RopePrepare which has different signature)
m = pattern.search(src)
if not m:
    print("ERROR: pattern not found; aborting")
    raise SystemExit(1)

new_src = src[:m.start()] + NEW_FORWARD + src[m.end():]

SRC.write_text(new_src)
print(f"Patched {SRC} (replaced {m.end() - m.start()} bytes with {len(NEW_FORWARD)} bytes)")
