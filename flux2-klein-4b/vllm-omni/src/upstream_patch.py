# SPDX-License-Identifier: Apache-2.0
"""Patch upstream Flux2PosEmbed.forward to use real-arithmetic RoPE.

Replaces the class.forward in
`vllm_omni/diffusion/models/flux2_klein/flux2_klein_transformer.py`
with a real-only implementation that:
  - Computes cos/sin on CPU in fp32
  - Returns (freqs_cos, freqs_sin) of shape (S, sum(axes_dim)) — same
    shape contract as the original via .real/.imag concat
  - NEVER calls `torch.polar` or builds a complex tensor

This is the only level of patch that survives Dynamo's bytecode inlining
in the vllm-omni compile path: when Dynamo inlines `Flux2PosEmbed.forward`,
it inlines THIS new bytecode, so the FX graph never emits torch.polar.

Why monkey-patches don't work:
  - Instance-level `pe.forward = ...` — Dynamo unwraps to the class.
  - Class-level `Flux2PosEmbed.forward = ...` — Dynamo's nn.Module
    introspection caches the original bytecode by class identity at
    module load time; reassignment after that point is ignored on the
    first trace.
  - Module-level `diffusers.get_1d_rotary_pos_embed = ...` — only
    helps for callers that look up the function by name at call time.
    `Flux2PosEmbed.forward` already has the function imported into
    its closure, and Dynamo tracks the closure value.

Editing the source IS the canonical fix. Mirror the same approach used
elsewhere (`/opt/conda/lib/python3.12/site-packages/torch/_subclasses/.bak`).

Run inside the vllm_omni container:
    python3 /workspace/upstream_patch.py
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

SRC = Path(
    "/opt/conda/lib/python3.12/site-packages/vllm_omni/"
    "diffusion/models/flux2_klein/flux2_klein_transformer.py"
)
BAK = SRC.with_suffix(".py.bak")

# Real-arithmetic Flux2PosEmbed.forward. Drop-in replacement that
# returns the SAME shape contract as the original (freqs_cos, freqs_sin
# both shape (S, sum(axes_dim))) but never uses torch.polar.
NEW_FORWARD = '''    def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # NEURON PATCH: real-arithmetic RoPE in fp32. The original calls
        # `get_1d_rotary_pos_embed(..., use_real=False)` which returns
        # a complex tensor built via `torch.polar(ones_like, freqs)`.
        # Neuron Beta 3 has no complex64 OR float64 support — both
        # trip the XLA runtime. We compute cos/sin fully in fp32 on CPU
        # then move to ids.device. Bit-near-equivalent to the original
        # (numerical loss at fp32 is negligible for RoPE freqs at
        # dim<=128).
        orig_device = ids.device
        ids_cpu = ids.to(device="cpu") if str(orig_device) != "cpu" else ids
        cos_out = []
        sin_out = []
        pos = ids_cpu.float()
        for i in range(len(self.axes_dim)):
            dim = self.axes_dim[i]
            # freqs[k] = 1 / (theta^(2k/dim)) for k in [0, dim/2)
            inv_freq = 1.0 / (
                self.theta
                ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
            )
            # outer over pos[..., i] and inv_freq → (..., dim/2)
            angles = torch.outer(pos[..., i].float(), inv_freq)
            cos_out.append(angles.cos().to(torch.float32))
            sin_out.append(angles.sin().to(torch.float32))
        freqs_cos = torch.cat(cos_out, dim=-1).to(device=orig_device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(device=orig_device)
        return freqs_cos, freqs_sin
'''


def main():
    if not SRC.exists():
        raise SystemExit(f"target source not found: {SRC}")

    if not BAK.exists():
        shutil.copy(SRC, BAK)
        print(f"backed up original to {BAK}")
    else:
        # Restore from backup before patching so we always patch the
        # pristine upstream (idempotent).
        shutil.copy(BAK, SRC)
        print(f"restored from {BAK}")

    src = SRC.read_text()

    # Match Flux2PosEmbed.forward only (not Flux2RopePrepare.forward,
    # which has a different signature). The class is defined immediately
    # before this method, and the forward we want is the FIRST one with
    # signature `def forward(self, ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:`
    # ending in `return freqs_cos, freqs_sin`.
    pattern = re.compile(
        r"    def forward\(self, ids: torch\.Tensor\) -> tuple\[torch\.Tensor, torch\.Tensor\]:\n"
        r"(?:.*\n)+?"
        r"        return freqs_cos, freqs_sin\n",
        re.MULTILINE,
    )

    m = pattern.search(src)
    if not m:
        raise SystemExit("ERROR: pattern not found; aborting")

    new_src = src[: m.start()] + NEW_FORWARD + src[m.end():]
    SRC.write_text(new_src)
    print(
        f"Patched {SRC.name} (replaced {m.end() - m.start()} bytes "
        f"with {len(NEW_FORWARD)} bytes — real-arithmetic RoPE)"
    )


if __name__ == "__main__":
    main()
