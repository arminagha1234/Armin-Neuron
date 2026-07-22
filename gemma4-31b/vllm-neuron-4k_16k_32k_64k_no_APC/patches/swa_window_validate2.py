"""
Validate the EXACT v2 gather plumbing (paged block cache + windowed dynamic-offset
gather + prior_used_len) against the full-span gather, using the kernel's CPU
fallback. This tests the part v1 may have gotten wrong (the gather/indexing), not
just the abstract masking.

Simulates a paged KV cache with a NON-identity block_table (permutation) so any
indexing error shows up. k_cache/v_cache content encodes absolute position, so a
wrong gather -> wrong keys -> output mismatch.
"""
import os
os.environ.setdefault("PJRT_DEVICE", "CPU")
import torch
import vllm_neuron.functional as NF

torch.manual_seed(0)

# --- small paged-cache config ---
bs = 4            # block_size
window = 8        # sliding window
C = 40            # cached prior length (abs [0, C))
T = 8             # current chunk length (abs [C, C+T))
hd = 16
nkh = 1
max_blocks = 16   # block_table width (max_model_len/bs = 64/4)
num_phys = 16     # physical cache blocks
device = torch.device("cpu")

# deterministic per-absolute-position embedding
_embed = torch.randn(C + T + 8, hd)
def pos_embed(abs_idx):  # [hd]
    return _embed[abs_idx]

# NON-identity logical->physical block map (permutation) to catch index bugs.
perm = torch.randperm(max_blocks)
block_table = perm.view(1, max_blocks).clone()          # [1, max_blocks]

# Fill physical k/v cache: logical block b (physical perm[b]) holds abs [b*bs,(b+1)*bs)
k_cache = torch.zeros(num_phys, nkh, bs, hd)
v_cache = torch.zeros(num_phys, nkh, bs, hd)
for b in range(max_blocks):
    phys = perm[b].item()
    for j in range(bs):
        a = b * bs + j
        if a < C:                      # only prior positions are "written"
            k_cache[phys, 0, j] = pos_embed(a)
            v_cache[phys, 0, j] = pos_embed(a) * 0.5 + 0.1

# current chunk q/k/v (abs [C, C+T))
q = torch.randn(nkh, T, hd)
k = torch.stack([pos_embed(C + m) for m in range(T)]).view(nkh, T, hd)
v = (k * 0.5 + 0.1)

cached_seq_len = torch.tensor([[C]], dtype=torch.int64)
scale = 1.0

def flash(k_prior, v_prior, prior_used_len):
    return NF.flash_attention(
        q=q, k=k.transpose(1, 2), v=v, scale=scale,
        causal_mask=True, sliding_window=window,
        k_prior=k_prior.transpose(1, 2), v_prior=v_prior,
        prior_used_len=torch.tensor([int(prior_used_len)], dtype=torch.int32),
        tp_q=True, tp_k=False, tp_out=False,
    )

# ---- FULL-SPAN gather (original path) ----
_bt_full = block_table[0].clamp_min(0).to(torch.int64)
kb = torch.index_select(k_cache, 0, _bt_full)          # [max_blocks, nkh, bs, hd]
vb = torch.index_select(v_cache, 0, _bt_full)
k_prior_full = kb.permute(1, 0, 2, 3).reshape(nkh, max_blocks * bs, hd)
v_prior_full = vb.permute(1, 0, 2, 3).reshape(nkh, max_blocks * bs, hd)
out_full = flash(k_prior_full, v_prior_full, C)

# ---- WINDOWED gather (v2 path) ----
_csl = cached_seq_len.reshape(()).to(torch.int64)
_w_blocks = (window + bs - 1) // bs + 1
_w_blocks = min(_w_blocks, max_blocks)
padded_kv_len = _w_blocks * bs
_n_valid = (_csl + bs - 1) // bs
_start_blk = torch.clamp(_n_valid - _w_blocks, min=0)
_pos = _start_blk + torch.arange(_w_blocks, dtype=torch.int64)
_pos = torch.clamp(_pos, max=max_blocks - 1)
bt = torch.index_select(_bt_full, 0, _pos)
kbw = torch.index_select(k_cache, 0, bt)
vbw = torch.index_select(v_cache, 0, bt)
k_prior_win = kbw.permute(1, 0, 2, 3).reshape(nkh, padded_kv_len, hd)
v_prior_win = vbw.permute(1, 0, 2, 3).reshape(nkh, padded_kv_len, hd)
_prior_valid = int((_csl - _start_blk * bs).clamp(min=0).item())
print(f"w_blocks={_w_blocks} start_blk={int(_start_blk)} padded_kv_len={padded_kv_len} prior_valid={_prior_valid}")
out_win = flash(k_prior_win, v_prior_win, _prior_valid)

diff = (out_full - out_win).abs().max().item()
print(f"MAX ABS DIFF windowed-vs-full (full gather plumbing) = {diff:.3e}")
print("RESULT:", "MATCH  (v2 gather CORRECT)" if diff < 1e-4 else "MISMATCH (gather bug -> debug here)")
