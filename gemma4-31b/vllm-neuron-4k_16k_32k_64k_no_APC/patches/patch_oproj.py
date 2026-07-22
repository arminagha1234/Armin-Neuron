#!/usr/bin/env python3
"""
patch_oproj.py -- Lever 8 (RANK3): route the PREFILL output projection through
the NKI output_projection_cte kernel (NF.o_proj) instead of torch.matmul.

Gemma4 head_dim (256 SWA / 512 global) exceeds the o_proj wrapper's D<=128 gate
in the 4D path, BUT the wrapper's 3D [B,S,N*D] path folds head_dim into N with
D=min(128,ND) -> D=128 -> passes the gate. So passing attn_output as 3D
[1, T, Nh_per_rank*hd] routes it onto the CTE kernel (folds N; kernel does the
[N*D, H] matmul). The wrapper AUTO-falls-back to torch.matmul if _can_use_kernel
is False (e.g. off-device or bad shape), so this is a safe drop-in.

  ONLY the PREFILL site (forward_prefill, model.py:604) is patched -- that is the
  TTFT-relevant o_proj. Decode sites (:772/:919/:1084) are left on torch (TPOT,
  not TTFT; sdpa decode path).

Idempotent-safe: asserts the original torch.matmul o_proj line is present, backs
up to model.py.pre_oproj, ast.parse() syntax-checks.
"""
import ast, os, shutil, sys

P = "/opt/conda/lib/python3.13/site-packages/vllm_neuron/model/gemma4/model.py"
if len(sys.argv) > 1:
    P = sys.argv[1]
BACKUP = P + ".pre_oproj"

src = open(P).read()

# Unique to the prefill site (:604): assigns back to `attn_output` (decode sites
# assign to `output`). The 8-space indent matches forward_prefill's body.
old = "        attn_output = torch.matmul(attn_output, self.o_proj_weight)"
new = ("        # === LEVER8 o_proj NKI kernel (RANK3) ===\n"
       "        # 3D [1,T,Nh*hd] input -> wrapper folds head_dim into N (D=128) ->\n"
       "        # passes the D<=128 gate -> output_projection_cte kernel; auto-falls\n"
       "        # back to torch.matmul if kernel constraints fail. TTFT prefill site only.\n"
       "        attn_output = NF.o_proj(attn_output.unsqueeze(0), self.o_proj_weight).squeeze(0)")

assert old in src, (
    "prefill o_proj torch.matmul line NOT found (already patched or drifted). "
    "Expected exactly: " + repr(old)
)
assert "LEVER8 o_proj NKI kernel" not in src, "already patched -- aborting."
# Safety: the anchor must be unique so we don't touch decode sites.
assert src.count(old) == 1, (
    "anchor not unique (count=%d) -- refusing to patch to avoid hitting decode "
    "sites." % src.count(old)
)

new_src = src.replace(old, new, 1)
if not os.path.exists(BACKUP):
    shutil.copy2(P, BACKUP)
    print("BACKUP created:", BACKUP)
else:
    print("BACKUP exists (not overwriting):", BACKUP)
ast.parse(new_src)
print("ast.parse OK")
open(P, "w").write(new_src)
print("PATCH_OK bytes_before=%d bytes_after=%d" % (len(src), len(new_src)))
print("Patched forward_prefill o_proj -> NF.o_proj (prefill site only).")
