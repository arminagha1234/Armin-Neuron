#!/usr/bin/env python3
"""Wire decode_hd256_v2 into the qwen3_5 package behind QWEN35_DECODE_V2=1.

Only touches nki_kernels/__init__.py -- model_bf16.py is untouched, because
call_decode_hd256 already invokes the kernel with keyword args and v2 has
identical parameter names (q, k_full, v_full, mask_bias, scale). So swapping
the imported kernel object is the whole change.

Idempotent; backs up once; py_compiles and auto-reverts on syntax error.
"""
import os
import py_compile
import shutil
import sys

INIT = ("/opt/conda/lib/python3.13/site-packages/vllm_neuron/model/"
        "qwen3_5/nki_kernels/__init__.py")

OLD = "from .decode_hd256 import decode_hd256_kernel as _decode_hd256_kernel"

NEW = '''# QWEN35_DECODE_V2=1 selects the v2 single-pass kernel. v1 measured 0.80x
# eager; v2 removes v1's three-loops-over-context structure, its per-chunk
# mask transpose, and its partition-dim softmax reduction. Default off so
# existing benchmarks stay valid.
if os.environ.get("QWEN35_DECODE_V2", "0") == "1":
    from .decode_hd256_v2 import decode_hd256_v2_kernel as _decode_hd256_kernel
else:
    from .decode_hd256 import decode_hd256_kernel as _decode_hd256_kernel'''

src = open(INIT).read()
if "QWEN35_DECODE_V2" in src:
    print("  V2_ALREADY_WIRED")
    sys.exit(0)
if OLD not in src:
    print("  ANCHOR_NOT_FOUND")
    sys.exit(2)
if src.count(OLD) != 1:
    print(f"  ANCHOR_AMBIGUOUS ({src.count(OLD)})")
    sys.exit(2)

out = src.replace(OLD, NEW, 1)
if "\nimport os" not in out and not out.startswith("import os"):
    lines = out.splitlines()
    idx = max((i for i, l in enumerate(lines)
               if l.startswith("import ") or l.startswith("from ")), default=0)
    # put `import os` before the first relative import so the gate can read env
    first_rel = min((i for i, l in enumerate(lines) if l.startswith("from .")),
                    default=idx)
    lines.insert(first_rel, "import os")
    out = "\n".join(lines)

bak = INIT + ".pre_v2"
if not os.path.exists(bak):
    shutil.copy2(INIT, bak)
open(INIT, "w").write(out)
try:
    py_compile.compile(INIT, doraise=True)
    print(f"  V2_WIRED_OK backup={bak}")
except py_compile.PyCompileError as e:
    shutil.copy2(bak, INIT)
    print(f"  REVERTED: {e}")
    sys.exit(3)
