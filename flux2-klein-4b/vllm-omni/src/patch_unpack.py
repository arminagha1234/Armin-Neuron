# SPDX-License-Identifier: Apache-2.0
"""Patch the call site of _unpack_latents_with_ids to run on CPU.

The unpack method does `out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)`
which trips Neuron with "Expected self.is_contiguous() to be true, but got
false". Easiest fix: move latents/latent_ids to CPU before the call. The
subsequent VAE decode is already on CPU, so the latents stay there.

Also patches the _unpatchify_latents call site to ensure consistent CPU
device through to the VAE decode step.

Idempotent: restores from .bak first, then re-applies patches.

Run inside the vllm_omni container:
    python3 /workspace/patch_unpack.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

SRC = Path(
    "/opt/conda/lib/python3.12/site-packages/vllm_omni/"
    "diffusion/models/flux2_klein/pipeline_flux2_klein.py"
)
BAK = SRC.with_suffix(".py.bak")


def main():
    if not SRC.exists():
        raise SystemExit(f"target source not found: {SRC}")

    if not BAK.exists():
        # No backup yet — make one from current state. The current state
        # is already partially patched (VAE decode CPU + scheduler bf16
        # fixes), so we capture that as the "base" we re-apply on top of.
        shutil.copy(SRC, BAK)
        print(f"backed up current state to {BAK}")
    else:
        # Restore from backup so we always patch deterministically.
        shutil.copy(BAK, SRC)
        print(f"restored from {BAK}")

    src = SRC.read_text()

    # ------------------------------------------------------------------
    # Patch 1: call site — move latents+ids to CPU before unpack.
    # ------------------------------------------------------------------
    OLD_CALL = "        latents = self._unpack_latents_with_ids(latents, latent_ids)"
    NEW_CALL = (
        "        # NEURON PATCH: scatter_ + expand on non-contiguous slice trips\n"
        "        # the lazy backend. Run unpack on CPU; VAE decode is also on CPU\n"
        "        # so we leave latents there.\n"
        "        _orig_lat_dev = latents.device\n"
        "        _lat_cpu = latents.to(device=\"cpu\").contiguous()\n"
        "        _ids_cpu = latent_ids.to(device=\"cpu\").contiguous() \\\n"
        "            if torch.is_tensor(latent_ids) else latent_ids\n"
        "        latents = self._unpack_latents_with_ids(_lat_cpu, _ids_cpu)"
    )
    if OLD_CALL not in src:
        # Maybe already patched; check for marker.
        if "NEURON PATCH: scatter_ + expand on non-contiguous" in src:
            print("call site already patched — skipping")
        else:
            raise SystemExit(
                "ERROR: could not locate _unpack_latents_with_ids call site"
            )
    else:
        src = src.replace(OLD_CALL, NEW_CALL, 1)
        print("patched _unpack_latents_with_ids call site (-> CPU)")

    # ------------------------------------------------------------------
    # Patch 2: latents_bn_mean/std — these use latents.dtype/device. Once
    # latents is on CPU, vae.bn.running_mean is already CPU (VAE pinned),
    # so the existing .to(device=latents.device) calls work fine without
    # changes. No-op for clarity.
    # ------------------------------------------------------------------

    SRC.write_text(src)
    print(f"wrote {SRC}")


if __name__ == "__main__":
    main()
