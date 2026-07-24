#!/usr/bin/env python3
"""CTE-PREFILL LEVER — route Gemma4 single-shot prefill attention through the REAL
nkilib `attention_cte` NKI kernel (the d-tiled, hd512-capable flash kernel with
`_MAX_HEAD_DIM = 512`) instead of the fp32 `torch.bmm` fallback in `_manual_sdpa`.

THE PROBLEM THIS FIXES
----------------------
`gemma4/model.py::_manual_sdpa` does its OWN fp32 score-materializing attention:

    scores = torch.bmm(q.float(), k.float().transpose(1, 2))
    scores = scores * self.scaling
    scores = scores + attn_mask.float()
    attn_weights = softmax(scores, -1)
    out = torch.bmm(attn_weights, v.float())

It never calls a kernel. head_dim is 256 (SWA) or 512 (global) — both exceed the
stock `NF.flash_attention` wrapper cap of 128, so nothing routed there anyway. But
the *underlying* nkilib kernel `attention_cte` DOES support hd256/512 (d-tiled,
`num_d_tiles = ceil(d/128)`, DMA-transpose loads, single-PSUM-bank d-accumulation).

VALIDATED ON MEL (trn2.3xl, cc-2.26, 2026-07-23): attention_cte compiles clean
(no coloring stall — unlike gemma4_flash_prefill_v2) and matches an fp32 oracle
at cos >= 0.99997 for hd256 SWA + hd512 global, causal + sliding-window, T=512/2048,
both unexpanded [Nkv,T,D] and expanded [Nh,T,D] k/v. See CTE_PREFILL_WIRING.md.

THE CONTRACT (source-verified in NKILIB_CTE_INTERNALS.md, exercised in validation)
---------------------------------------------------------------------------------
  from nkilib.core.attention.attention_cte import attention_cte  # already @nki.jit
  # torch/xla serve path: mirror vllm_neuron/functional/attention/attention_cte.py
  jitted  = nki.jit()(attention_cte)
  wrapped = wrap_nki(jitted)
  out = wrapped[2](q=q_scaled, k=k, v=v, scale=1.0, causal_mask=True,
                   sliding_window=W_or_0, tp_q=True, tp_k=True, tp_out=False,
                   softmax_dtype=torch.float32)

Critical contract points enforced by this patch:
  * ALWAYS pre-scale q by `self.scaling` and pass `scale=1.0`. This satisfies the
    kernel's SWA/CP/prefix-caching `scale==1.0` assert (attention_cte.py:1563) for
    the sliding layers AND is exactly equivalent for the global layers. (This is
    what the shipped functional wrapper does — always pre-scale, always scale=1.0.)
  * `causal_mask=True` is a BOOL flag; the kernel builds its own causal (+window)
    mask internally, so the materialized `attn_mask` is DROPPED on the kernel path.
  * `tp_k=True` — k is [*, T, D]; the kernel dma-transposes at load. (Do NOT pass
    the default tp_k=False, which expects [*, D, T].)
  * `_manual_sdpa` receives k/v ALREADY GQA-expanded to [Nh, T, D] from its callers,
    so bs == bs_kv (plain MHA to the kernel). Validation confirms the expanded form
    is correct (cos 0.999992). Passing expanded keeps this patch local to
    `_manual_sdpa` (no caller changes). Native-GQA (unexpanded, bs_kv<bs) is a
    memory optimization that would require changing the callers — see the doc.
  * SEGMENTED-PREFILL GUARD: `_manual_sdpa` is also called from
    `_segmented_prefill_attention` where k/v carry a *prior* cache prepended, so
    seqlen_k > seqlen_q and the kernel's internal 0-based causal mask would be
    WRONG. We route to the kernel ONLY when q.shape[1] == k.shape[1] (plain aligned
    causal/SWA self-attention); otherwise fall through to torch.bmm.

SAFETY
------
  * Gated behind env GEMMA4_CTE_PREFILL=1 (default OFF -> unchanged torch.bmm path).
  * The kernel is imported+wrapped ONCE at module import, inside try/except; if
    anything is unavailable the flag self-disables and the model runs torch.bmm.
  * The per-call kernel invocation is wrapped in try/except: any RUNTIME error
    degrades to the correct-but-slow torch.bmm path (never wrong output). NOTE: an
    in-trace try/except cannot catch a *trace-time* lowering crash — but on-mel
    validation shows attention_cte compiles clean at hd256/512, so that risk is
    already retired; the try/except covers runtime surprises only.

Idempotent, backs up model.py to `.pre_cte_prefill`, ast-checks, reversible.
"""
import ast
import os
import sys

# Resolve the gemma4 model.py to patch (site-packages on the serve box), overridable.
CANDIDATES = [
    os.environ.get("GEMMA4_MODEL_PY", ""),
    "/opt/conda/lib/python3.13/site-packages/vllm_neuron/model/gemma4/model.py",
    "/opt/conda/lib/python3.11/site-packages/vllm_neuron/model/gemma4/model.py",
    # local repo copy (this working tree)
    os.path.join(os.path.dirname(__file__), "..", "serving_pkg", "gemma4", "model.py"),
]
M = next((p for p in CANDIDATES if p and os.path.exists(p)), "")
if not M:
    print("ERROR: could not locate gemma4 model.py; set GEMMA4_MODEL_PY")
    sys.exit(2)
M = os.path.abspath(M)

src = open(M).read()
if "CTE_PREFILL_LEVER" in src:
    print("ALREADY PATCHED (CTE_PREFILL_LEVER) target=%s" % M)
    sys.exit(0)

# --- 1. Module-level: import + wrap the nkilib attention_cte kernel once, gated by
#        env. Mirrors vllm_neuron/functional/attention/attention_cte.py exactly
#        (nki.jit() then wrap_nki), which is the proven torch/xla serve invocation.
IMPORT_ANCHOR = "import vllm_neuron.functional as NF\n"
if IMPORT_ANCHOR not in src:
    print("ERROR: NF import anchor not found (model.py changed?)")
    sys.exit(2)

IMPORT_BLOCK = IMPORT_ANCHOR + '''
# CTE_PREFILL_LEVER: route single-shot prefill attention through the real nkilib
# attention_cte kernel (hd512-capable, _MAX_HEAD_DIM=512) instead of the fp32
# torch.bmm in _manual_sdpa. Gated by GEMMA4_CTE_PREFILL=1 (default OFF). The
# kernel is imported+wrapped ONCE here (before any traced region); if unavailable
# the lever self-disables and _manual_sdpa uses its torch.bmm path unchanged.
import os as _cte_os
import logging as _cte_logging

_CTE_LOG = _cte_logging.getLogger(__name__)
_CTE_PREFILL_ENABLED = _cte_os.environ.get("GEMMA4_CTE_PREFILL", "0") == "1"
_CTE_KERNEL = None          # wrap_nki(nki.jit()(attention_cte)) NKIHOPCaller, or None
_CTE_CAN_RUN = None         # can_run_kernel(tensor) -> bool (device gate), or None
if _CTE_PREFILL_ENABLED:
    try:
        import nki as _cte_nki
        from nkilib.core.attention.attention_cte import attention_cte as _cte_attn
        from vllm_neuron.nki.nki_hop import wrap_nki as _cte_wrap
        from vllm_neuron.nki.nki_hop import can_run_kernel as _CTE_CAN_RUN
        # Mirror vllm_neuron/functional/attention/attention_cte.py: jit then wrap.
        _CTE_KERNEL = _cte_wrap(_cte_nki.jit()(_cte_attn))
        _CTE_LOG.warning("CTE_PREFILL_LEVER: attention_cte wired for prefill (hd256/512).")
    except Exception as _cte_e:  # pragma: no cover
        _CTE_LOG.warning(
            "CTE_PREFILL_LEVER: kernel unavailable (%r); using torch.bmm fallback",
            _cte_e,
        )
        _CTE_KERNEL = None
'''
src2 = src.replace(IMPORT_ANCHOR, IMPORT_BLOCK, 1)

# --- 2. Insert the CTE fast path at the TOP of _manual_sdpa's body, right before
#        the existing torch.bmm block (which stays as the fallback). ---
BMM_ANCHOR = (
    "        # f32 matmul fallback (NF.flash_attention falls back to torch anyway)\n"
    "        scores = torch.bmm(q.float(), k.float().transpose(1, 2))\n"
)
# Robustness: if the exact comment+line pair changed, fall back to just the bmm line.
BMM_ANCHOR_ALT = "        scores = torch.bmm(q.float(), k.float().transpose(1, 2))\n"

FAST_PATH = '''        # CTE_PREFILL_LEVER: try the real nkilib attention_cte kernel (hd256/512
        # capable) before the fp32 torch.bmm fallback below. Contract (validated on
        # mel, cc-2.26, cos>=0.99997): pre-scale q by self.scaling and pass scale=1.0
        # (satisfies the kernel's SWA scale==1.0 assert and is equivalent for global);
        # causal_mask=True lets the kernel build its own causal(+window) mask, so the
        # materialized attn_mask is dropped; tp_k=True (k is [*, T, D]). k/v arrive
        # GQA-expanded ([Nh, T, D]) => plain MHA to the kernel. GUARD: only the plain
        # aligned self-attention case (seqlen_q == seqlen_k); segmented prefill
        # (k carries prepended prior cache, seqlen_k > seqlen_q) MUST stay on torch.
        if (
            _CTE_PREFILL_ENABLED
            and _CTE_KERNEL is not None
            and q.shape[1] == k.shape[1]
            and (_CTE_CAN_RUN is None or _CTE_CAN_RUN(q))
        ):
            try:
                _cte_sw = (
                    int(self.sliding_window)
                    if getattr(self, "sliding_window", None) is not None
                    else 0
                )
                _cte_q = (q * self.scaling).contiguous()  # fold scale into q; scale=1.0
                _cte_out = _CTE_KERNEL[2](
                    q=_cte_q,
                    k=k.contiguous(),
                    v=v.contiguous(),
                    scale=1.0,
                    causal_mask=True,
                    sliding_window=_cte_sw,
                    tp_q=True,
                    tp_k=True,
                    tp_out=False,
                    softmax_dtype=torch.float32,
                )
                return _cte_out.to(q.dtype)
            except Exception as _cte_err:  # degrade to correct-but-slow torch path
                _CTE_LOG.warning(
                    "CTE_PREFILL_LEVER: attention_cte call failed (%r); "
                    "falling back to torch.bmm",
                    _cte_err,
                )
'''

if BMM_ANCHOR in src2:
    src2 = src2.replace(BMM_ANCHOR, FAST_PATH + BMM_ANCHOR, 1)
    mode = "inserted before commented torch.bmm block"
elif BMM_ANCHOR_ALT in src2:
    src2 = src2.replace(BMM_ANCHOR_ALT, FAST_PATH + BMM_ANCHOR_ALT, 1)
    mode = "inserted before torch.bmm line (alt anchor)"
else:
    print(
        "ERROR: could not find the _manual_sdpa torch.bmm anchor. The score line\n"
        "  'scores = torch.bmm(q.float(), k.float().transpose(1, 2))'\n"
        "was not found verbatim — inspect model.py::_manual_sdpa formatting."
    )
    sys.exit(2)

# --- Validate syntax + write with backup ---
try:
    ast.parse(src2)
except SyntaxError as e:
    print("ERROR: patched source failed to parse: %r" % e)
    sys.exit(3)

open(M + ".pre_cte_prefill", "w").write(src)
open(M, "w").write(src2)
print("PATCH_OK (CTE_PREFILL_LEVER) target=%s" % M)
print("  backup=%s.pre_cte_prefill  bytes %d -> %d" % (M, len(src), len(src2)))
print("  mode: %s" % mode)
print("  markers: CTE_PREFILL_LEVER=%s  _CTE_KERNEL=%s"
      % ("CTE_PREFILL_LEVER" in src2, "_CTE_KERNEL" in src2))
print("  enable with: GEMMA4_CTE_PREFILL=1  (default OFF -> unchanged torch.bmm)")
print("  revert with: mv %s.pre_cte_prefill %s" % (M, M))
