"""Static-shape batched DECODE for the V4-Flash reference model, as a monkey-patch.

PROBLEM
-------
`the vLLM-Neuron model adapter` calls `self.transformer(ids, 0)` with start_pos pinned to the
Python int 0, because a real `start_pos` int reaches shape-affecting expressions:

  * `freqs_cis[start_pos:start_pos+seqlen]`, `kv_cache[:, start_pos % win]`, `ape[start_pos%ratio]`
    -> a different graph per decode step (recompile per token).
  * `get_compress_topk_idxs` -> `arange(0, (start_pos+1)//ratio)`; LENGTH grows with position.
  * `Indexer`: `kv_cache[:bsz, :end_pos//ratio]` (growing slice) and
    `topk(min(index_topk, end_pos//ratio))` (position-dependent k).

The last three are genuinely dynamic shapes, which Neuron rejects. Result today: decode compiles
as a degenerate seqlen=1 PREFILL that is numerically wrong and cannot batch.

APPROACH
--------
Pass `start_pos` as a 0-d TENSOR for decode. Then position is DATA, not shape:
  * integer indexing/slicing -> `index_select` / `index_copy_`
  * growing lengths -> FIXED-width rows padded with -1, which is ALREADY the reference's
    "this slot is masked" sentinel (see the original prefill branches) and which `sparse_attn`
    honours (the plugin runs it DENSE-MASKED).
  * `should_compress` control flow -> unconditional compute + masked commit via `torch.where`.

DISCRIMINATOR: `torch.is_tensor(start_pos)`.
  int    -> PREFILL  -> delegate to the ORIGINAL unmodified function (zero regression risk)
  tensor -> DECODE   -> the static implementations below
This is a branch on TYPE, not on a traced value, so it does not cause retracing.

SCOPE / HONESTY
---------------
The reference takes ONE scalar `start_pos` for the whole batch, so this is correct only when the
batch advances in LOCKSTEP (all sequences at the same position). `vllm_v4_decode.py` does exactly
that. General continuous batching with ragged positions needs a per-sequence position vector and
is NOT addressed here -- do not present a lockstep throughput number as general serving support.

Usage:  import reference_model as rm; import decode_static_patch; decode_static_patch.apply()
"""

import os

import torch
import torch.nn.functional as F

_APPLIED = False
_orig = {}


# ============================================================================================
# Static index builders (bit-identical to the originals; see decode_static_idx.py for the
# 358-case parity test covering ring-fill, the pos==W-1 boundary, and multiple wraps).
# ============================================================================================
def window_topk_idxs_decode_static(window_size: int, bsz: int, pos: torch.Tensor) -> torch.Tensor:
    """Decode window indices. Shape is ALWAYS [bsz, 1, window_size]."""
    W = window_size
    pos = pos.reshape(())
    ar = torch.arange(W, device=pos.device)
    p = pos % W
    split = W - 1 - p
    ring = torch.where(ar < split, ar + p + 1, ar - split)          # ring full
    early = torch.where(ar <= pos, ar, torch.full_like(ar, -1))     # ring still filling
    matrix = torch.where(pos >= (W - 1), ring, early)
    return matrix.reshape(1, 1, W).expand(bsz, 1, W)


def compress_topk_idxs_decode_static(ratio: int, bsz: int, pos: torch.Tensor, offset: int,
                                     n_compress_max: int) -> torch.Tensor:
    """Decode compressed-KV indices. Shape is ALWAYS [bsz, 1, n_compress_max] (-1 padded)."""
    pos = pos.reshape(())
    ar = torch.arange(n_compress_max, device=pos.device)
    n_valid = (pos + 1) // ratio                                    # tensor, NOT a shape
    matrix = torch.where(ar < n_valid, ar + offset, torch.full_like(ar, -1))
    return matrix.reshape(1, 1, n_compress_max).expand(bsz, 1, n_compress_max)


def _idx1(t: torch.Tensor) -> torch.Tensor:
    """0-d -> 1-element long index tensor, for index_select."""
    return t.reshape(1).long()


# ---------------------------------------------------------------------------------------------
# Buffer read/write helpers.
#
# TWO THINGS torch_xla / Dynamo REJECT, both learned the hard way here:
#  1. Mutating a slice VIEW in place -- `buf[:bsz].index_copy_(...)` ->
#     "aten::as_strided ... no implementation for backend xla:0 ... View operators don't support
#      since the tensor's storage cannot be shared across devices."
#  2. Advanced indexing that mixes a slice with a 0-d TENSOR index --
#     `buf[:bsz, pos % win] = val` -> fails during the parallel decode trace.
#
# So: write with a one-hot blend + `copy_` on the FULL tensor (no view, no advanced indexing,
# static shapes), and read with `index_select` (functional, not a view).
# ---------------------------------------------------------------------------------------------
def _scatter_slot(buf: torch.Tensor, slot: torch.Tensor, val: torch.Tensor) -> None:
    """buf[:, slot] = val, where buf is [B, N, D], slot is a 0-d tensor, val is [B, D].

    Writes the FULL batch dimension, so `val` must cover every row of `buf`. In decode the vLLM
    batch equals the reference's max_batch_size, so this holds; assert rather than let it broadcast
    into something silently wrong.
    """
    assert val.shape[0] == buf.shape[0], (
        f"_scatter_slot needs val to cover the full batch: val={tuple(val.shape)} "
        f"buf={tuple(buf.shape)} (decode bsz must equal max_batch_size)")
    B = buf.shape[0]
    b_idx = torch.arange(B, device=buf.device)
    # same slot for every batch row; build it by broadcast-add rather than .expand() so the index
    # is a real tensor, not a view (mirrors the team's `torch.zeros_like(block_indices)` trick).
    s_idx = torch.zeros(B, dtype=torch.long, device=buf.device) + slot.reshape(()).long()
    buf.index_put_((b_idx, s_idx), val.to(buf.dtype))


def _gather_slot(buf: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
    """buf[:, slot] as [B, 1, D], without creating a view."""
    return buf.index_select(1, _idx1(slot))


# ============================================================================================
# Compressor.forward -- decode only
# ============================================================================================
def _compressor_forward(self, x, start_pos):
    rm = _orig["rm"]
    if not torch.is_tensor(start_pos):
        return _orig["compressor"](self, x, start_pos)

    assert self.kv_cache is not None
    bsz, seqlen, _ = x.size()
    assert seqlen == 1, f"decode compressor expects seqlen==1, got {seqlen}"
    ratio, overlap, d, rd = self.compress_ratio, self.overlap, self.head_dim, self.rope_head_dim
    dtype = x.dtype
    pos = start_pos.reshape(())

    x = x.float()
    kv = self.wkv(x)                       # [b,1,coff*d]
    score = self.wgate(x)

    # (start_pos + 1) % ratio == 0, as DATA rather than control flow.
    commit = (((pos + 1) % ratio) == 0)
    slot = pos % ratio

    # score += self.ape[start_pos % ratio]
    score = score + self.ape.index_select(0, _idx1(slot)).unsqueeze(0)

    # NOTE ON XLA: do NOT take a slice VIEW and then mutate it in place (e.g.
    # `self.kv_state[:bsz].index_copy_(...)`). torch_xla rejects that with
    #   "aten::as_strided ... no implementation for backend xla:0 ... View operators don't
    #    support since the tensor's storage cannot be shared across devices."
    # Instead keep the ORIGINAL indexed-assignment form and only swap the int index for a TENSOR
    # index; that lowers to a functional scatter, which XLA handles.
    if overlap:
        wslot = ratio + slot
        _scatter_slot(self.kv_state, wslot, kv.squeeze(1))
        _scatter_slot(self.score_state, wslot, score.squeeze(1))
        kv_state = torch.cat([self.kv_state[:bsz, :ratio, :d], self.kv_state[:bsz, ratio:, d:]], dim=1)
        score_state = torch.cat([self.score_state[:bsz, :ratio, :d], self.score_state[:bsz, ratio:, d:]], dim=1)
        kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
        # Roll the window ONLY on commit -> blend instead of branching. Rebuild the whole buffer
        # with cat + copy_ rather than assigning into a slice view (the original does
        # `kv_state[:bsz, :ratio] = kv_state[:bsz, ratio:]`, but that decode branch has never
        # actually been traced on device, so it is not a proven-safe pattern here).
        self.kv_state.copy_(torch.cat(
            [torch.where(commit, self.kv_state[:, ratio:], self.kv_state[:, :ratio]),
             self.kv_state[:, ratio:]], dim=1))
        self.score_state.copy_(torch.cat(
            [torch.where(commit, self.score_state[:, ratio:], self.score_state[:, :ratio]),
             self.score_state[:, ratio:]], dim=1))
    else:
        _scatter_slot(self.kv_state, slot, kv.squeeze(1))
        _scatter_slot(self.score_state, slot, score.squeeze(1))
        kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)

    # The original returns early when not should_compress. We must stay branch-free, so we always
    # compute and commit under a mask. Costs `ratio`x redundant compressor math on a small tensor.
    kv = self.norm(kv.to(dtype))
    # freqs_cis[start_pos + 1 - ratio]; clamped because when it would go negative `commit` is
    # False and the value is discarded anyway (an OOB index would still be a hard error).
    fpos = (pos + 1 - ratio).clamp_min(0)
    freqs_cis = self.freqs_cis.index_select(0, _idx1(fpos))
    kv = rm._V4_ROPE_APPLY(kv, rd, freqs_cis, False)
    kv = kv.float()
    if self.rotate:
        kv = rm.rotate_activation(kv)
        rm.fp4_act_quant(kv, rm.fp4_block_size, True)
    else:
        rm.act_quant(kv[..., :-rd], 64, rm.scale_fmt, rm.scale_dtype, True)

    # self.kv_cache[:bsz, start_pos // ratio] = kv  -- masked so a non-commit step is a no-op.
    # Read-modify-write via indexed assignment (NOT a view + index_copy_; see XLA note above).
    cslot = pos // ratio
    cur = _gather_slot(self.kv_cache, cslot)                       # [B,1,D]
    newv = torch.where(commit, kv.to(cur.dtype), cur)
    _scatter_slot(self.kv_cache, cslot, newv.squeeze(1))
    return kv


# ============================================================================================
# Indexer.forward -- decode only
# ============================================================================================
def _indexer_forward(self, x, qr, start_pos, offset):
    rm = _orig["rm"]
    if not torch.is_tensor(start_pos):
        return _orig["indexer"](self, x, qr, start_pos, offset)

    bsz, seqlen, _ = x.size()
    assert seqlen == 1
    ratio, rd = self.compress_ratio, self.rope_head_dim
    pos = start_pos.reshape(())

    if self.compressor.kv_cache is None:
        self.compressor.kv_cache = self.kv_cache
        self.compressor.freqs_cis = self.freqs_cis

    freqs_cis = self.freqs_cis.index_select(0, _idx1(pos))
    q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
    q = rm._V4_ROPE_APPLY(q, rd, freqs_cis, False)
    q = rm.rotate_activation(q)
    rm.fp4_act_quant(q, rm.fp4_block_size, True)
    self.compressor(x, start_pos)
    weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)

    # Original scores only kv_cache[:, :end_pos//ratio] (a GROWING slice) and then takes
    # topk(min(index_topk, end_pos//ratio)) (a position-dependent k). Both are dynamic shapes.
    # Static form: score the FULL buffer, -inf the not-yet-written tail, keep k FIXED, and map
    # any overflow pick to the -1 masked sentinel.
    n_max = self.kv_cache.size(1)
    n_valid = (pos + 1) // ratio
    index_score = torch.einsum("bshd,btd->bsht", q.float(), self.kv_cache[:bsz].float())
    index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)     # [b, s, n_max]
    if rm.world_size > 1:
        rm.dist.all_reduce(index_score)
    ar = torch.arange(n_max, device=index_score.device)
    index_score = index_score + torch.where(ar < n_valid, 0.0, float("-inf")).to(index_score.dtype)
    k = int(min(self.index_topk, n_max))
    topk_idxs = index_score.topk(k, dim=-1)[1]
    topk_idxs = torch.where(topk_idxs < n_valid, topk_idxs + offset, torch.full_like(topk_idxs, -1))
    return topk_idxs


# ============================================================================================
# Attention (MLA).forward -- decode only
# ============================================================================================
def _attention_forward(self, x, start_pos):
    rm = _orig["rm"]
    if not torch.is_tensor(start_pos):
        return _orig["attention"](self, x, start_pos)

    bsz, seqlen, _ = x.size()
    assert seqlen == 1
    pos = start_pos.reshape(())
    win, ratio, rd = self.window_size, self.compress_ratio, self.rope_head_dim

    if self.compress_ratio and self.compressor.kv_cache is None:
        self.compressor.kv_cache = self.kv_cache[:, win:]
        self.compressor.freqs_cis = self.freqs_cis
        if self.indexer is not None:
            self.indexer.freqs_cis = self.freqs_cis

    freqs_cis = self.freqs_cis.index_select(0, _idx1(pos))

    qr = q = self.q_norm(self.wq_a(x))
    q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
    q = rm._V4_ROPE_APPLY(q, rd, freqs_cis, False)

    kv = self.wkv(x)
    kv = self.kv_norm(kv)
    kv = rm._V4_ROPE_APPLY(kv, rd, freqs_cis, False)
    kv = kv.float()
    rm.act_quant(kv[..., :-rd], 64, rm.scale_fmt, rm.scale_dtype, True)

    topk_idxs = window_topk_idxs_decode_static(win, bsz, pos)
    if self.compress_ratio:
        offset = win
        n_comp_max = self.kv_cache.size(1) - win
        # NO INDEXER IN DECODE -- always use the static causal compressed indices.
        #
        # Three reasons, in order of force:
        #  1. It is what the shipped team port does: their DeepSeek-V3.2 `forward_decode` never calls
        #     the Indexer at all (it is prefill-only, and even there skipped when T <= index_topk).
        #  2. The plugin ALREADY reroutes `Indexer.forward` to `get_compress_topk_idxs` and argues in
        #     its own comment that this is EXACT, not an approximation, for seq <= ~2048: with
        #     index_topk=512 and #compressed = S/ratio <= 512, a top-512-of-<=512 selects EVERY
        #     available causal compressed position.
        #  3. That rerouted builder (`_cmp_native`) ignores start_pos and always uses the prefill
        #     formula, so at decode seqlen=1 it computes arange(1 // ratio) = arange(0) -> a 0-WIDTH
        #     tensor, and `unsqueeze(0).expand(...)` on that degenerate view is exactly the
        #     `aten::as_strided ... backend "xla:0"` failure that blocked the decode graph.
        # Set V4_DECODE_NO_INDEXER=0 to restore the learned selection (only meaningful once a real
        # sparse indexer kernel exists, and it will reintroduce the 0-width trace failure).
        if os.environ.get("V4_DECODE_NO_INDEXER", "1") == "1" or self.indexer is None:
            compress_topk_idxs = compress_topk_idxs_decode_static(ratio, bsz, pos, offset, n_comp_max)
        else:
            compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
        topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
    topk_idxs = topk_idxs.int()

    # Ring-buffer write at (start_pos % win). Identical to the original statement except the index
    # is a TENSOR instead of a Python int (see the XLA view note in _compressor_forward).
    _scatter_slot(self.kv_cache, pos % win, kv.squeeze(1))
    if self.compress_ratio:
        self.compressor(x, start_pos)

    o = rm.sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
    o = rm._V4_ROPE_APPLY(o, rd, freqs_cis, True)
    o = o.view(bsz, seqlen, self.n_local_groups, -1)
    wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
    o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
    return self.wo_b(o.flatten(2))



# =============================================================================================
# DEVICE-AWARE index builders.
#
# The shipped `get_window_topk_idxs` / `get_compress_topk_idxs` call `torch.arange(...)` with NO
# device, so their whole subgraph is built on CPU. Under torch_xla the trailing
# `matrix.unsqueeze(0).expand(bsz, -1, -1)` then fails to lower:
#
#   m_1 = torch.where(mask_3, full_like_3, add_263)
#   unsqueeze_30 = m_1.unsqueeze(0)          <-- RuntimeError: aten::as_strided ... backend "xla:0"
#   expand_5 = unsqueeze_30.expand(1, -1, -1)
#
# (Observed in the PREFILL branch, bsz=1.) These replacements are semantically identical -- the
# ONLY change is `device=`, taken from the plugin's `rm._v4_dev_ref` device carrier -- so nothing
# lands on CPU inside the traced graph. Verified against the originals in
# test_decode_static_idx_devparity (see decode_static_idx.py for the same-value guarantee).
# =============================================================================================
def _dev():
    t = getattr(_orig.get("rm"), "_v4_dev_ref", None)
    return t.device if torch.is_tensor(t) else torch.device("cpu")


def _window_topk_idxs(window_size, bsz, seqlen, start_pos):
    if torch.is_tensor(start_pos):
        return window_topk_idxs_decode_static(window_size, bsz, start_pos)
    d = _dev()
    if start_pos >= window_size - 1:
        sp = start_pos % window_size
        matrix = torch.cat([torch.arange(sp + 1, window_size, device=d),
                            torch.arange(0, sp + 1, device=d)], dim=0)
    elif start_pos > 0:
        matrix = F.pad(torch.arange(start_pos + 1, device=d),
                       (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seqlen, device=d).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size), device=d)
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


def _compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset):
    # Tensor start_pos never reaches here: the patched Attention calls
    # compress_topk_idxs_decode_static directly (it needs a static width budget).
    assert not torch.is_tensor(start_pos), "tensor start_pos must use compress_topk_idxs_decode_static"
    d = _dev()
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio, device=d) + offset
    else:
        matrix = torch.arange(seqlen // ratio, device=d).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1, device=d).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)

# ============================================================================================
def apply(rm=None):
    """Install the decode-static overrides on the reference_model module."""
    global _APPLIED
    if _APPLIED:
        return rm or _orig.get("rm")
    if rm is None:
        import reference_model as rm  # noqa: PLC0415
    _orig["rm"] = rm
    _orig["compressor"] = rm.Compressor.forward
    _orig["indexer"] = rm.Indexer.forward
    _orig["attention"] = rm.Attention.forward
    _orig["gw"] = rm.get_window_topk_idxs
    _orig["gc"] = rm.get_compress_topk_idxs
    rm.Compressor.forward = _compressor_forward
    rm.Indexer.forward = _indexer_forward
    rm.Attention.forward = _attention_forward
    # Device-aware builders (kills the CPU subgraph that cannot lower under XLA).
    rm.get_window_topk_idxs = _window_topk_idxs
    rm.get_compress_topk_idxs = _compress_topk_idxs
    _APPLIED = True
    print("[decode-static] patched Compressor/Indexer/Attention forward "
          "(int start_pos -> original prefill; tensor start_pos -> static decode)")
    return rm


def revert():
    global _APPLIED
    if not _APPLIED:
        return
    rm = _orig["rm"]
    rm.Compressor.forward = _orig["compressor"]
    rm.Indexer.forward = _orig["indexer"]
    rm.Attention.forward = _orig["attention"]
    if "gw" in _orig:
        rm.get_window_topk_idxs = _orig["gw"]
        rm.get_compress_topk_idxs = _orig["gc"]
    _APPLIED = False
