#!/usr/bin/env bash
# Install the qwen3_5 package INTO vllm_neuron/model/ and register it in
# registry.py ON DISK -- mirroring gemma4's install_public.sh.
#
# WHY: register.py monkey-patches registry.get_models() in-process. vLLM-Neuron
# spawns worker subprocesses (Worker_TP0..N via multiproc_executor) which
# re-import vllm_neuron.model.registry fresh, WITHOUT the patch. The workers then
# resolve vLLM's built-in Qwen3_5 stub, which has no from_configs ->
#   AttributeError: type object 'Qwen3_5ForConditionalGeneration' has no attribute 'from_configs'
# A source-level registration is inherited by every subprocess.
set -eu
SRC_PKG="${1:?usage: install_qwen35.sh /path/to/src/qwen3_5}"

MODEL_DIR="$(python3 - <<'PY'
import os, sys
for base in sys.path:
    cand = os.path.join(base, "vllm_neuron", "model")
    if os.path.isdir(cand):
        print(cand); break
PY
)"
[ -n "${MODEL_DIR:-}" ] && [ -d "$MODEL_DIR" ] || { echo "ERROR: vllm_neuron/model not found" >&2; exit 1; }
echo "[install] vllm_neuron model dir: $MODEL_DIR"

DST="$MODEL_DIR/qwen3_5"
mkdir -p "$DST/nki_kernels"
cp -f "$SRC_PKG"/*.py "$DST/"
[ -d "$SRC_PKG/nki_kernels" ] && cp -f "$SRC_PKG/nki_kernels"/*.py "$DST/nki_kernels/"
echo "[install] deployed qwen3_5 -> $DST"

# Neutralize the auto-register in the INSTALLED copy. registry.py will import
# this package, and __init__ calling register() (which imports
# vllm_neuron.model.registry) mid-import would be circular.
python3 - "$DST/__init__.py" <<'PY'
import sys, re
p = sys.argv[1]
s = open(p).read()
if "_register()" in s and "AUTO-REGISTER DISABLED" not in s:
    s = re.sub(
        r"try:\s*\n\s*from \.register import register as _register.*?\n(?=__all__)",
        "# AUTO-REGISTER DISABLED BY INSTALLER: registry.py imports this package\n"
        "# directly, so calling register() here would be a circular import.\n\n",
        s, flags=re.S)
    open(p, "w").write(s)
    print("[install] disabled auto-register in installed __init__.py")
else:
    print("[install] auto-register already absent/disabled")
PY

# Register in registry.py on disk (idempotent).
REG="$MODEL_DIR/registry.py"
python3 - "$REG" <<'PY'
import sys, os, shutil
reg = sys.argv[1]
src = open(reg).read()
changed = False
if "from .qwen3_5 import Qwen3_5ForConditionalGeneration" not in src:
    lines = src.splitlines()
    idx = max(i for i, l in enumerate(lines) if l.startswith("from ."))
    lines.insert(idx + 1, "from .qwen3_5 import Qwen3_5ForConditionalGeneration")
    src = "\n".join(lines) + ("\n" if not src.endswith("\n") else "")
    changed = True
if '("Qwen3_5ForConditionalGeneration"' not in src:
    anchor = "    models = ["
    entry = ('        ("Qwen3_5ForConditionalGeneration", Qwen3_5ForConditionalGeneration),\n'
             '        ("Qwen3_5ForCausalLM", Qwen3_5ForConditionalGeneration),\n')
    src = src.replace(anchor, anchor + "\n" + entry, 1)
    changed = True
if changed:
    bak = reg + ".bak_qwen35"
    if not os.path.exists(bak): shutil.copy2(reg, bak)
    open(reg, "w").write(src)
    print(f"[install] patched registry -> {reg}")
else:
    print("[install] registry already has Qwen3_5 entries")
PY

python3 - <<'PY'
from vllm_neuron.model.registry import get_models
names = [n for n, _ in get_models()]
hits = [n for n in names if n.startswith("Qwen3_5")]
assert hits, f"Qwen3_5 not registered; have {len(names)} models"
from vllm_neuron.model.qwen3_5 import Qwen3_5ForConditionalGeneration as C
assert hasattr(C, "from_configs"), "installed class lacks from_configs"
print("[install] OK — registered:", hits, "| from_configs present")
PY
echo "[install] done."
