#!/usr/bin/env python3
"""
patch_swa_window_prior_v2.py -- long-context (segmented prefill) TTFT fix, CORRECTED.

Supersedes patch_swa_window_prior.py (v1). v1 produced degenerate output and was
documented as "fast but numerically incorrect; kernel masks prior by ABSOLUTE
index." THAT DIAGNOSIS WAS WRONG.

CPU validation (swa_window_validate.py) against the kernel's own PyTorch fallback
proved the windowed-prior masking is CORRECT to 2.4e-7: attention_cte's causal +
sliding_window masks depend only on (q_pos - k_pos) DIFFERENCES:
    q_pos = arange(S_q) + prior_len + cp_offset
    k_pos = arange(prior_len + S_k)
    mask  = (q_pos < k_pos) | (q_pos >= k_pos + sliding_window)
so they are SHIFT-INVARIANT. When the SWA prior is windowed, the concatenated
[windowed_prior | current] maps to absolute positions CONTIGUOUSLY with a uniform
shift, which cancels out of every difference. With prior_used_len = the valid
windowed length and cp_offset = 0, the result equals the full-span result exactly.

v2's gather logic is essentially identical to that v1 attempt; on clean re-validation
(recompile-from-clean + CPU proofs + multi-chunk token-parity gate) it PASSES and is
-33% faster. So v1's reported "degenerate output" was a CONFOUNDED TEST (most likely a
stale NEFF / not-fully-recompiled serve or a combined patch state), not a windowing
flaw, and the "absolute-index" root cause was a misdiagnosis. On-device parity
(byte-identical 40 greedy tokens at 18k) is the gate that confirms correctness.

WHAT IT DOES (SWA layers only; global/full-causal layers keep the full span):
  gather ONLY the trailing w_blocks = ceil(window/bs)+1 cache blocks (STATIC count,
  DYNAMIC start offset -> static-shape index tensor, trace-safe), and set
  prior_used_len = the number of VALID prior tokens inside that slice
  (= cached_seq_len - start_blk*bs). The kernel slices k_prior[:, :prior_used_len]
  (valid data sits at the FRONT of the slice since it starts at start_blk*bs) and
  masks by relative distance -> reproduces the full-span SWA result.

Replaces the inclusive LINE RANGE from `padded_kv_len = max_blocks * block_size`
through `prior_used_len = cached_seq_len.reshape(1)...`. Backs up model.py.pre_swawin_v2.
"""
import ast, os, shutil, sys

P = "/opt/conda/lib/python3.13/site-packages/vllm_neuron/model/gemma4/model.py"
if len(sys.argv) > 1:
    P = sys.argv[1]
BACKUP = P + ".pre_swawin_v2"

src = open(P).read()
assert "SWA WINDOWED PRIOR V2" not in src, "already patched with v2 -- aborting."

lines = src.splitlines(keepends=True)
START = "padded_kv_len = max_blocks * block_size"
END = "prior_used_len = cached_seq_len.reshape(1).to(torch.int32)"
start_idx = [i for i, l in enumerate(lines) if START in l]
end_idx = [i for i, l in enumerate(lines) if END in l]
assert len(start_idx) == 1, f"expected 1 start anchor, found {len(start_idx)}"
assert len(end_idx) == 1, f"expected 1 end anchor, found {len(end_idx)}"
s, e = start_idx[0], end_idx[0]
assert s < e, "anchors out of order"

indent = lines[s][: len(lines[s]) - len(lines[s].lstrip())]

new_body = '''# === SWA WINDOWED PRIOR V2 (long-context TTFT fix, CPU-validated) ===
# SWA layers attend only to the last `sliding_window` keys. Gather ONLY the
# trailing w_blocks of prior cache (static count, dynamic start -> trace-safe)
# instead of the full span. Masking is shift-invariant (validated 2.4e-7), so
# prior_used_len = valid window length reproduces the full-span result exactly.
# Global (full-causal) layers keep the full span.
_bs = block_size
_max_blocks = block_table.shape[1]
_bt_full = block_table[0].clamp_min(0).to(torch.int64)        # [max_blocks]
_csl = cached_seq_len.reshape(()).to(torch.int64)             # scalar prior len
if self.sliding_window is not None:
    _w_blocks = (self.sliding_window + _bs - 1) // _bs + 1     # window + 1 block slack
    if _w_blocks > _max_blocks:
        _w_blocks = _max_blocks
    padded_kv_len = _w_blocks * _bs                            # STATIC, small (~window)
    _n_valid = (_csl + _bs - 1) // _bs                         # ceil(cached_seq_len/bs)
    _start_blk = torch.clamp(_n_valid - _w_blocks, min=0)      # dynamic scalar offset
    _pos = _start_blk + torch.arange(_w_blocks, device=device, dtype=torch.int64)
    _pos = torch.clamp(_pos, max=_max_blocks - 1)             # [w_blocks] STATIC shape
    bt = torch.index_select(_bt_full, 0, _pos)               # [w_blocks] block ids
    _prior_valid = (_csl - _start_blk * _bs).clamp(min=0)    # valid prior in the slice
else:
    padded_kv_len = _max_blocks * _bs                         # full span (global layers)
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
# Valid window prior sits at the FRONT of the slice (slice starts at start_blk*bs),
# so k_prior[:, :prior_used_len] is exactly [start_blk*bs, cached_seq_len).
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
