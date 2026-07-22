#!/usr/bin/env python3
"""
patch_swa_window_prior.py -- long-context (segmented prefill) TTFT fix.

!!! WARNING — DO NOT SHIP AS-IS. FAST BUT NUMERICALLY INCORRECT. !!!
Measured on-device (trn2, TP32, clean box): 32k 3.06->2.06s, 64k 6.10->4.11s (-33%).
BUT it FAILS token parity vs the full-span path (degenerate output on multi-chunk
prompts). Root cause: NF.flash_attention masks prior keys by their ABSOLUTE tensor
index, not relative to prior_used_len — so compacting the SWA prior to a window
slice makes the current chunk see those keys as thousands of tokens away -> all
masked -> SWA loses cross-chunk context. A correct version needs a KERNEL change
(a prior-absolute-offset arg on NF.flash_attention). Kept here only to document the
lever + its ~33% potential. See SWA_WINDOW_FINDING.md.

ROOT CAUSE (model.py:_segmented_prefill_attention): the segmented path gathers the
FULL prior-KV span (padded_kv_len = max_blocks*block_size, sized for max_model_len)
and passes it as k_prior for EVERY layer, masking only via the kernel's
`sliding_window` flag. The 50 SWA layers therefore gather + carry the entire prior
history when they only ever attend to the last `sliding_window` (1024) keys.

FIX (SWA layers only): gather just the trailing window of prior blocks -- a STATIC
number of blocks (w_blocks = ceil(window/bs)+1) at a DYNAMIC offset (index tensor of
static length -> trace-safe). prior_used_len = valid window length. Because SWA
attention depends only on RELATIVE distance and the windowed prior sits immediately
before the current chunk, prior_used_len + the kernel's sliding_window mask reproduce
the full-span result EXACTLY. Global (full-causal) layers keep the full span.

Robust to comment/whitespace drift: replaces the inclusive LINE RANGE from the
`padded_kv_len = max_blocks * block_size` line through the
`prior_used_len = cached_seq_len.reshape(1)...` line. Backs up to model.py.pre_swawin.
"""
import ast, os, shutil, sys

P = "/opt/conda/lib/python3.13/site-packages/vllm_neuron/model/gemma4/model.py"
if len(sys.argv) > 1:
    P = sys.argv[1]
BACKUP = P + ".pre_swawin"

src = open(P).read()
assert "SWA WINDOWED PRIOR" not in src, "already patched -- aborting."

lines = src.splitlines(keepends=True)
START = "padded_kv_len = max_blocks * block_size"
END = "prior_used_len = cached_seq_len.reshape(1).to(torch.int32)"
start_idx = [i for i, l in enumerate(lines) if START in l]
end_idx = [i for i, l in enumerate(lines) if END in l]
assert len(start_idx) == 1, f"expected 1 start anchor, found {len(start_idx)}"
assert len(end_idx) == 1, f"expected 1 end anchor, found {len(end_idx)}"
s, e = start_idx[0], end_idx[0]
assert s < e, "anchors out of order"

# Indent from the start line (should be 8 spaces inside the method).
indent = lines[s][: len(lines[s]) - len(lines[s].lstrip())]

new_body = '''# === SWA WINDOWED PRIOR (long-context TTFT fix) ===
# SWA layers only attend to the last `sliding_window` keys, so gather ONLY the
# trailing window of prior blocks (static count, dynamic offset -> trace-safe)
# instead of the full span. Global layers keep the full span.
_bs = block_size
_max_blocks = block_table.shape[1]
_bt_full = block_table[0].clamp_min(0).to(torch.int64)         # [max_blocks]
_csl = cached_seq_len.reshape(()).to(torch.int64)              # scalar prior len
if self.sliding_window is not None:
    w_blocks = (self.sliding_window + _bs - 1) // _bs + 1      # window + 1 block slack
    if w_blocks > _max_blocks:
        w_blocks = _max_blocks
    padded_kv_len = w_blocks * _bs                             # STATIC, small (~window)
    _n_valid = (_csl + _bs - 1) // _bs                         # ceil(cached_seq_len/bs)
    _start_blk = torch.clamp(_n_valid - w_blocks, min=0)       # dynamic scalar offset
    _pos = _start_blk + torch.arange(w_blocks, device=device, dtype=torch.int64)
    _pos = torch.clamp(_pos, max=_max_blocks - 1)              # [w_blocks] STATIC shape
    bt = torch.index_select(_bt_full, 0, _pos)                # [w_blocks] block ids
    _prior_valid = (_csl - _start_blk * _bs).clamp(min=0)     # valid prior in the slice
else:
    padded_kv_len = _max_blocks * _bs                          # full span (global layers)
    bt = _bt_full
    _prior_valid = _csl
k_blocks = torch.index_select(self.k_cache, 0, bt)
v_blocks = torch.index_select(self.v_cache, 0, bt)
k_prior = k_blocks.permute(1, 0, 2, 3).reshape(nkh, padded_kv_len, self.head_dim)
v_prior = v_blocks.permute(1, 0, 2, 3).reshape(nkh, padded_kv_len, self.head_dim)
# Dequantize FP8 cache values back to compute dtype.
if self.k_cache.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
    k_prior = k_prior.to(self.dtype) / self.k_scale_float
    v_prior = v_prior.to(self.dtype) / self.v_scale_float
# SWA windowed prior sits immediately before the current chunk, so prior_used_len +
# the kernel's sliding_window mask reproduce the full-span result (relative distance).
prior_used_len = _prior_valid.reshape(1).to(torch.int32)
'''
new_lines = [indent + l if l.strip() else l for l in new_body.splitlines(keepends=True)]

lines[s : e + 1] = new_lines
new_src = "".join(lines)
assert new_src != src, "no change"

if not os.path.exists(BACKUP):
    shutil.copy2(P, BACKUP)
    print("BACKUP created:", BACKUP)
else:
    print("BACKUP exists (not overwriting):", BACKUP)
ast.parse(new_src)
print("ast.parse OK")
open(P, "w").write(new_src)
print("PATCH_OK bytes_before=%d bytes_after=%d (replaced lines %d..%d)" % (len(src), len(new_src), s, e))
