#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Deploy the patched ``attention_segmented_cte.py`` over the container's
installed ``vllm_neuron`` copy.

Why this is needed (and why PYTHONPATH alone is not enough):
  Gemma4's head_dim is 256 (sliding-window layers) / 512 (global layers). The
  stock vllm_neuron segmented-attention kernel RAISES for head_dim > 128, and
  its torch fallback gathers the full padded KV span. The bundled patched copy:
    * edit A       -> routes head_dim > 128 to a trace-safe torch fallback,
                      which is what enables chunked / segmented prefill above
                      16K for Gemma4; and
    * SWA windowed -> for the 49/60 sliding-window layers, gathers a STATIC
      gather        number of KV blocks at a DYNAMIC offset instead of the full
                      padded span (the ~1.9x TTFT win, pure PyTorch).
  vllm_neuron imports this module by its installed path, so the file on disk
  must be replaced -- a PYTHONPATH shadow does not take effect.

The installed file is located by scanning ``sys.path`` WITHOUT importing
vllm_neuron: importing it would cache the stale module in ``sys.modules`` and
make the overwrite a no-op for the current process. Idempotent (byte compare)
and backs up the original once to ``<file>.bak_armin``.
"""
import os
import shutil
import sys

_REL = os.path.join(
    "vllm_neuron", "functional", "attention", "attention_segmented_cte.py"
)
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_HERE, "attention_segmented_cte.py")


def _find_installed():
    """Return the path to the installed vllm_neuron segmented CTE file, or None.

    Scans sys.path + site-packages roots WITHOUT importing vllm_neuron.
    """
    bases = list(sys.path)
    try:
        import site

        bases += list(site.getsitepackages())
        user = site.getusersitepackages()
        if user:
            bases.append(user)
    except Exception:
        pass
    seen = set()
    for base in bases:
        if not base or base in seen:
            continue
        seen.add(base)
        cand = os.path.join(base, _REL)
        if os.path.isfile(cand):
            return cand
    return None


def deploy():
    """Overwrite the installed segmented CTE with the bundled patched copy.

    Returns True on success (or if already deployed), False otherwise.
    """
    if not os.path.isfile(_SRC):
        print(f"[deploy_segmented_cte] ERROR: bundled source missing: {_SRC}")
        return False
    dst = _find_installed()
    if dst is None:
        print(
            "[deploy_segmented_cte] WARNING: installed vllm_neuron "
            "attention_segmented_cte.py not found on sys.path -- chunked prefill "
            "for head_dim>128 (Gemma4) will NOT work. Is vllm_neuron installed?"
        )
        return False
    with open(_SRC, "rb") as f:
        src_bytes = f.read()
    with open(dst, "rb") as f:
        dst_bytes = f.read()
    if src_bytes == dst_bytes:
        print(f"[deploy_segmented_cte] already deployed: {dst}")
        return True
    bak = dst + ".bak_armin"
    if not os.path.exists(bak):
        shutil.copy2(dst, bak)
        print(f"[deploy_segmented_cte] backed up original -> {bak}")
    with open(dst, "wb") as f:
        f.write(src_bytes)
    print(f"[deploy_segmented_cte] deployed patched segmented CTE -> {dst}")
    return True


if __name__ == "__main__":
    sys.exit(0 if deploy() else 1)
