#!/bin/bash
# Install the Gemma4-31B model into the PUBLIC vLLM-Neuron v0.21 plugin (inside the DLC).
# Copies serving_pkg/gemma4 into vllm_neuron/model/gemma4 and registers the architecture
# in vllm_neuron/model/registry.py. Idempotent; backs up registry.py once.
#
# Run INSIDE the public DLC container:
#   bash install_public.sh
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"

# Locate the installed vllm_neuron/model package without importing vllm_neuron.
MODEL_DIR="$(python3 - <<'PY'
import os, sys
for base in sys.path:
    cand = os.path.join(base, "vllm_neuron", "model")
    if os.path.isdir(cand):
        print(cand); break
PY
)"
if [ -z "${MODEL_DIR:-}" ] || [ ! -d "$MODEL_DIR" ]; then
  echo "ERROR: could not find installed vllm_neuron/model on sys.path. Is the public DLC active?" >&2
  exit 1
fi
echo "[install] vllm_neuron model dir: $MODEL_DIR"

# 1. Deploy the gemma4 package.
DST="$MODEL_DIR/gemma4"
mkdir -p "$DST"
cp -f "$HERE/serving_pkg/gemma4/"*.py "$DST/"
echo "[install] deployed gemma4 package -> $DST"

# 1b. Apply the performance patches to the deployed model. WITHOUT these, the
#     GEMMA4_CTE_PREFILL / GEMMA4_BF16_FALLBACK env vars (set by run_benchmark.sh
#     and LAUNCH.md) are no-ops and <=16k TTFT regresses to the torch baseline
#     (measured up to ~2x slower). Both patches are idempotent, gated by their env
#     var (CTE default OFF, bf16 default ON once applied), and back up model.py.
echo "[install] applying performance patches (CTE prefill kernel + bf16 fallback)"
GEMMA4_MODEL_PY="$DST/model.py" python3 "$HERE/patches/patch_manual_sdpa_cte.py"
GEMMA4_MODEL_PY="$DST/model.py" python3 "$HERE/patch_bf16_fallback.py"

# 2. Register the architecture in registry.py (idempotent).
REG="$MODEL_DIR/registry.py"
python3 - "$REG" <<'PY'
import sys, io
reg = sys.argv[1]
src = open(reg).read()
changed = False
if "from .gemma4 import Gemma4ForConditionalGeneration" not in src:
    # add import after the last existing "from .<x> import" line
    lines = src.splitlines()
    idx = max(i for i, l in enumerate(lines) if l.startswith("from ."))
    lines.insert(idx + 1, "from .gemma4 import Gemma4ForConditionalGeneration")
    src = "\n".join(lines) + ("\n" if not src.endswith("\n") else "")
    changed = True
if '("Gemma4ForConditionalGeneration"' not in src:
    # add two entries after the "models = [" line
    anchor = "    models = ["
    entry = ('        ("Gemma4ForConditionalGeneration", Gemma4ForConditionalGeneration),\n'
             '        ("Gemma4ForCausalLM", Gemma4ForConditionalGeneration),\n')
    src = src.replace(anchor, anchor + "\n" + entry, 1)
    changed = True
if changed:
    import shutil, os
    bak = reg + ".bak_gemma4"
    if not os.path.exists(bak):
        shutil.copy2(reg, bak)
    open(reg, "w").write(src)
    print(f"[install] patched registry -> {reg} (backup {bak})")
else:
    print(f"[install] registry already has Gemma4 entries: {reg}")
PY

# 3. Sanity: confirm the class imports and is registered.
python3 - <<'PY'
from vllm_neuron.model.registry import get_models
names = [n for n, _ in get_models()]
assert "Gemma4ForConditionalGeneration" in names, names
assert "Gemma4ForCausalLM" in names, names
print("[install] OK — registered:", [n for n in names if n.startswith("Gemma4")])
PY
echo "[install] done."
