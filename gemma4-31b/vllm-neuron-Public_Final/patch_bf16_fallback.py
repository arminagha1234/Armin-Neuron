#!/usr/bin/env python3
"""BF16-FALLBACK A/B: change the _manual_sdpa fp32 fallback to bf16 matmul + fp32 softmax only.

The original fallback casts EVERYTHING to fp32 (.float()) — a correctness crutch copied from inf2.
On trn2 bf16 matmul is ~2-4x faster. This keeps the QK and PV matmuls in bf16 (the model dtype)
and only uses fp32 for the softmax reduction (the numerically sensitive part) — the standard
flash-attention precision policy.

Applies to the fp32 fallback block (works whether or not the fast-prefill patch is present, since
that patch keeps the same fp32 block as its exception fallback). Idempotent, backs up, ast-checks.
Gated by GEMMA4_BF16_FALLBACK (default 1 when applied). Reversible.
"""
import ast, os, sys

# Resolve the gemma4 model.py to patch. Overridable via GEMMA4_MODEL_PY so
# install_public.sh can target the freshly deployed package dir (and so it works
# on non-3.13 site-packages layouts).
CANDIDATES = [
    os.environ.get("GEMMA4_MODEL_PY", ""),
    "/opt/conda/lib/python3.13/site-packages/vllm_neuron/model/gemma4/model.py",
    "/opt/conda/lib/python3.11/site-packages/vllm_neuron/model/gemma4/model.py",
    os.path.join(os.path.dirname(__file__), "serving_pkg", "gemma4", "model.py"),
]
M = next((p for p in CANDIDATES if p and os.path.exists(p)), "")
if not M:
    print("ERROR: could not locate gemma4 model.py; set GEMMA4_MODEL_PY"); sys.exit(2)
M = os.path.abspath(M)
src = open(M).read()
if "BF16_FALLBACK_AB" in src:
    print("ALREADY PATCHED (bf16 fallback)"); sys.exit(0)

OLD = '''        # f32 matmul fallback
        scores = torch.bmm(q.float(), k.float().transpose(1, 2))
        scores = scores * self.scaling
        scores = scores + attn_mask.float()
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        out = torch.bmm(attn_weights, v.float())
        return out.to(q.dtype)'''

# If the fast-prefill patch isn't present, the block still says the original comment:
OLD2 = '''        # f32 matmul fallback (NF.flash_attention falls back to torch anyway)
        scores = torch.bmm(q.float(), k.float().transpose(1, 2))
        scores = scores * self.scaling
        scores = scores + attn_mask.float()
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        out = torch.bmm(attn_weights, v.float())
        return out.to(q.dtype)'''

NEW = '''        # BF16_FALLBACK_AB: bf16 matmul + fp32 softmax only (was all-fp32).
        import os as _os
        if _os.environ.get("GEMMA4_BF16_FALLBACK", "1") == "1":
            scores = torch.bmm(q, k.transpose(1, 2)).to(torch.float32)   # bf16 matmul -> fp32 accum
            scores = scores * self.scaling + attn_mask.float()
            attn_weights = torch.nn.functional.softmax(scores, dim=-1).to(q.dtype)  # back to bf16
            out = torch.bmm(attn_weights, v)                             # bf16 matmul
            return out.to(q.dtype)
        # original all-fp32 fallback
        scores = torch.bmm(q.float(), k.float().transpose(1, 2))
        scores = scores * self.scaling
        scores = scores + attn_mask.float()
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        out = torch.bmm(attn_weights, v.float())
        return out.to(q.dtype)'''

if OLD in src:
    src2 = src.replace(OLD, NEW, 1)
elif OLD2 in src:
    src2 = src.replace(OLD2, NEW, 1)
else:
    print("ERROR: fp32 fallback block not found"); sys.exit(2)

ast.parse(src2)
open(M + ".pre_bf16fb", "w").write(src)
open(M, "w").write(src2)
print("BF16_PATCH_OK backup=.pre_bf16fb bytes", len(src), "->", len(src2))
