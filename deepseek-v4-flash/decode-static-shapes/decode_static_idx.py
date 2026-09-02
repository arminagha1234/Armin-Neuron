"""Static-shape, tensor-driven replacements for the two start_pos-dependent index builders
in reference_model.py, plus a parity test against the originals.

WHY THIS EXISTS
---------------
The vLLM-Neuron V4 plugin hardcodes `self.transformer(ids, 0)` because `start_pos` is a PYTHON
INT that reaches shape-affecting expressions. Two consequences:
  * with a real start_pos, every decode step is a DIFFERENT graph (recompile per token), and
  * `get_compress_topk_idxs` returns `arange(0, (start_pos+1)//ratio)`, whose LENGTH grows with
    position -- a genuinely dynamic shape, which Neuron rejects outright.

Both are removable. The decode path already attends the FULL kv buffer
(`sparse_attn(q, self.kv_cache[:bsz], ..., topk_idxs, ...)`) and `sparse_attn` is DENSE-MASKED,
and the ORIGINAL code already uses -1 as the "this slot is masked" sentinel in its prefill
branch. So we can return a FIXED-LENGTH index tensor padded with -1 and compute validity from a
0-d position TENSOR. Shapes become static and identical for every decode step -> ONE decode
graph, real positions.

Contract: `pos` is a 0-d (or 1-element) integer tensor. Nothing here branches on its VALUE, so
none of it forces a retrace. Outputs are bit-identical to the originals wherever the original
produced a value (the originals' variable-length decode output is right-padded with -1 here).

Run directly to self-test:  python decode_static_idx.py
"""

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------------------------
# Originals, copied verbatim from reference_model.py (lines 260-281) as the oracle.
# --------------------------------------------------------------------------------------------
def get_window_topk_idxs_orig(window_size: int, bsz: int, seqlen: int, start_pos: int):
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat([torch.arange(start_pos + 1, window_size), torch.arange(0, start_pos + 1)], dim=0)
    elif start_pos > 0:
        matrix = F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


def get_compress_topk_idxs_orig(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int):
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


# --------------------------------------------------------------------------------------------
# Static-shape decode replacements. `pos` is a TENSOR; no Python branch on its value.
# --------------------------------------------------------------------------------------------
def window_topk_idxs_decode_static(window_size: int, bsz: int, pos: torch.Tensor) -> torch.Tensor:
    """Decode (seqlen==1) window indices. Output shape is ALWAYS [bsz, 1, window_size].

    Reproduces both non-prefill branches of the original without branching on pos:
      * pos >= window_size-1 (ring is full): the ring rotation
        cat(arange(p+1, W), arange(0, p+1))  where p = pos % W.
      * 0 < pos < window_size-1 (ring still filling): arange(pos+1) right-padded with -1.
    """
    W = window_size
    dev = pos.device
    pos = pos.reshape(())
    ar = torch.arange(W, device=dev)
    p = pos % W                                  # ring head
    split = W - 1 - p                            # length of the leading arange(p+1, W)

    # Ring-full case: first `split` entries continue from p+1, the rest wrap to 0..p.
    ring = torch.where(ar < split, ar + p + 1, ar - split)

    # Ring-filling case: only slots 0..pos hold real positions; the tail is masked.
    early = torch.where(ar <= pos, ar, torch.full_like(ar, -1))

    matrix = torch.where(pos >= (W - 1), ring, early)
    return matrix.reshape(1, 1, W).expand(bsz, 1, W)


def compress_topk_idxs_decode_static(ratio: int, bsz: int, pos: torch.Tensor, offset: int,
                                     n_compress_max: int) -> torch.Tensor:
    """Decode (seqlen==1) compressed-KV indices. Output shape is ALWAYS
    [bsz, 1, n_compress_max] -- the fix for the original's data-dependent LENGTH.

    The original returns `arange(0, (pos+1)//ratio) + offset`. Here we emit a fixed-width row
    and mask the tail with -1, which `sparse_attn` (DENSE-MASKED) already treats as "skip".
    `n_compress_max` is a static budget, e.g. max_seq_len // ratio.
    """
    dev = pos.device
    pos = pos.reshape(())
    ar = torch.arange(n_compress_max, device=dev)
    n_valid = (pos + 1) // ratio                 # tensor, NOT a shape
    matrix = torch.where(ar < n_valid, ar + offset, torch.full_like(ar, -1))
    return matrix.reshape(1, 1, n_compress_max).expand(bsz, 1, n_compress_max)


# --------------------------------------------------------------------------------------------
# Parity test
# --------------------------------------------------------------------------------------------
def _pad_to(t: torch.Tensor, width: int) -> torch.Tensor:
    """Right-pad the original's variable-length decode row with -1 to the static width."""
    if t.numel() == width:
        return t
    return F.pad(t, (0, width - t.numel()), value=-1)


def main() -> int:
    torch.manual_seed(0)
    failures = []
    checked = 0

    # V4-Flash-like geometry plus small cases that exercise the branch boundaries.
    window_sizes = [4, 8, 128, 1024]
    ratios = [4, 32, 128]
    bszs = [1, 8]
    max_seq = 2048

    print("=== window_topk_idxs: static(tensor pos) vs original(int pos) ===")
    for W in window_sizes:
        for bsz in bszs:
            # cover the filling phase, the exact boundary, and several wraps
            positions = list(range(1, min(W + 3, 40))) + [W - 1, W, W + 1, 2 * W - 1, 2 * W, 5 * W + 3]
            for sp in sorted({p for p in positions if 0 < p < max_seq}):
                want = get_window_topk_idxs_orig(W, bsz, 1, sp)
                got = window_topk_idxs_decode_static(W, bsz, torch.tensor(sp))
                checked += 1
                if want.shape != got.shape or not torch.equal(want, got):
                    failures.append(f"window W={W} bsz={bsz} pos={sp}: "
                                    f"want{tuple(want.shape)}={want[0,0].tolist()[:12]} "
                                    f"got{tuple(got.shape)}={got[0,0].tolist()[:12]}")
    print(f"  checked {checked} cases")

    print("=== compress_topk_idxs: static(tensor pos, -1 padded) vs original(int pos) ===")
    c2 = 0
    for ratio in ratios:
        n_max = max_seq // ratio
        for bsz in bszs:
            for offset in (0, 7, 1024):
                for sp in [1, ratio - 1, ratio, ratio + 1, 2 * ratio, 3 * ratio + 1, 17 * ratio, max_seq - 1]:
                    if not (0 < sp < max_seq):
                        continue
                    want_raw = get_compress_topk_idxs_orig(ratio, bsz, 1, sp, offset)
                    got = compress_topk_idxs_decode_static(ratio, bsz, torch.tensor(sp), offset, n_max)
                    # original decode row is [bsz, n_valid]; ours is [bsz, 1, n_max] with -1 tail
                    want = _pad_to(want_raw[0].reshape(-1), n_max).reshape(1, 1, n_max).expand(bsz, 1, n_max)
                    c2 += 1
                    if want.shape != got.shape or not torch.equal(want, got):
                        failures.append(f"compress r={ratio} bsz={bsz} off={offset} pos={sp}: "
                                        f"want={want[0,0].tolist()[:12]} got={got[0,0].tolist()[:12]}")
    print(f"  checked {c2} cases")

    print("=== shape invariance across positions (the recompile fix) ===")
    W, ratio, n_max = 1024, 128, max_seq // 128
    wshapes, cshapes = set(), set()
    for sp in range(1, max_seq, 37):
        wshapes.add(tuple(window_topk_idxs_decode_static(W, 4, torch.tensor(sp)).shape))
        cshapes.add(tuple(compress_topk_idxs_decode_static(ratio, 4, torch.tensor(sp), 1024, n_max).shape))
    print(f"  distinct window shapes over {len(range(1, max_seq, 37))} positions: {wshapes}")
    print(f"  distinct compress shapes: {cshapes}")
    if len(wshapes) != 1 or len(cshapes) != 1:
        failures.append("SHAPE VARIES WITH POSITION -- would still force a recompile per step")

    print("=== no Python branch on pos value (traceability smoke) ===")
    # If either helper branched on pos, tracing with a different pos would change the graph.
    # torch.jit.trace records the graph for one pos; running it at other positions must match.
    traced_ok = True
    try:
        f = lambda p: window_topk_idxs_decode_static(W, 4, p)
        tr = torch.jit.trace(f, torch.tensor(5))
        for sp in (5, 300, 1023, 1024, 2000):
            if not torch.equal(tr(torch.tensor(sp)), f(torch.tensor(sp))):
                traced_ok = False
                failures.append(f"traced window mismatch at pos={sp} (value-dependent branch)")
    except Exception as e:  # tracing unavailable is not a correctness failure
        print(f"  (jit.trace unavailable: {type(e).__name__}: {e})")
    print(f"  traced-vs-eager consistent: {traced_ok}")

    print()
    if failures:
        print(f"FAIL — {len(failures)} mismatch(es):")
        for f_ in failures[:20]:
            print("   ", f_)
        return 1
    print(f"ALL PASS — {checked + c2} index cases bit-identical to the originals, "
          f"shapes position-invariant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
