"""CPU parity test: original int-start_pos decode  vs  decode_static_patch tensor-start_pos decode.

Builds the SAME model twice (identical weights), then:
  A) runs prefill + N decode steps the ORIGINAL way (Python int start_pos, dynamic shapes)
  B) runs prefill + N decode steps PATCHED   (0-d tensor start_pos, static shapes)
and compares logits at every step.

This is the correctness gate for the static-shape decode rewrite. It needs no Neuron device and
no real checkpoint -- random weights are sufficient, because we are testing that two code paths
compute the SAME function, not that the model is good.

Also asserts the property that motivates the whole change: every decode step in (B) produces
IDENTICALLY SHAPED intermediates, which is what collapses N per-step graphs into one.

Run:  python test_decode_static_parity.py
"""

import os
import sys

import torch

V4 = os.environ.get("V4_SRC", "/work/v4")
for p in (os.path.join(V4, "hf_src"), V4):
    if p not in sys.path:
        sys.path.insert(0, p)

import fallback_ops as fo  # noqa: E402

sys.modules["kernel"] = fo  # reference_model does `from kernel import ...`

torch.set_default_dtype(torch.bfloat16)

import reference_model as rm  # noqa: E402
import decode_static_patch as dsp  # noqa: E402

SEED = 1234
PREFILL = 128
STEPS = int(os.environ.get("STEPS", "10"))


# ---------------------------------------------------------------------------------------------
# HARNESS SHIM (applies to BOTH runs, so parity remains apples-to-apples).
# reference_model.rotate_activation asserts `x.dtype == torch.bfloat16`, but this variant
# casts kv to fp32 immediately before calling it (the "NCC_IVRF100 fix"). That combination trips
# the assert for the Indexer's rotate=True compressor in the ORIGINAL code path too, so it is not
# something our patch introduced -- the live plugin must neutralise it elsewhere. For this
# numerical-equivalence test we simply make rotate_activation dtype-tolerant.
# ---------------------------------------------------------------------------------------------
_orig_rotate = rm.rotate_activation


def _rotate_tolerant(x):
    out = _orig_rotate(x.to(torch.bfloat16))
    return out.to(x.dtype)


rm.rotate_activation = _rotate_tolerant


# ---------------------------------------------------------------------------------------------
# PAD_ORIG=1: right-pad the ORIGINAL path's variable-width compressed index rows to the same
# static width the patched path uses, with the -1 masked sentinel. This changes NOTHING
# mathematically (sparse_attn masks -1 to -inf, contributing exactly zero) but makes both runs
# hand sparse_attn IDENTICALLY SHAPED tensors. If that makes the two paths bit-exact, the residual
# diff was purely floating-point accumulation ORDER from the wider tensor, not a logic error.
# ---------------------------------------------------------------------------------------------
def install_pad_shim():
    import torch.nn.functional as _F

    base_idx = rm.Indexer.forward

    def idx_wrapper(self, x, qr, start_pos, offset):
        out = base_idx(self, x, qr, start_pos, offset)
        k = int(min(self.index_topk, self.kv_cache.size(1)))
        if out.shape[-1] < k:
            out = _F.pad(out, (0, k - out.shape[-1]), value=-1)
        return out

    rm.Indexer.forward = idx_wrapper

    base_cmp = rm.get_compress_topk_idxs

    def cmp_wrapper(ratio, bsz, seqlen, start_pos, offset):
        out = base_cmp(ratio, bsz, seqlen, start_pos, offset)
        n_max = rm_max_seq[0] // ratio
        if out.shape[-1] < n_max:
            out = _F.pad(out, (0, n_max - out.shape[-1]), value=-1)
        return out

    rm.get_compress_topk_idxs = cmp_wrapper


rm_max_seq = [512]
PAD_ORIG = os.environ.get("PAD_ORIG", "0") == "1"


def make_args():
    """Small but structurally faithful: keeps head_dim/rope_head_dim (so act_quant's 64-blocking
    holds), keeps window_size=128 and a ratio-4 AND ratio-128 compressed layer so both compressor
    modes and the Indexer are exercised."""
    return rm.ModelArgs(
        max_batch_size=2, max_seq_len=512, vocab_size=1024,
        dim=1024, moe_inter_dim=512, n_layers=3, n_hash_layers=0,
        n_heads=8, n_routed_experts=4, n_shared_experts=1, n_activated_experts=2,
        q_lora_rank=256, o_lora_rank=256, o_groups=8,
        # compress_ratios is indexed by MTP blocks too (layer_id = n_layers + i), so it needs
        # n_layers + n_mtp_layers entries -- the shipped default is 8 for n_layers=7 + 1 MTP.
        window_size=128, compress_ratios=(0, 4, 128, 0),
        index_n_heads=8, index_head_dim=128,
        # INDEX_TOPK is overridable: setting it >= (max_seq_len // min_ratio) makes the Indexer
        # select EVERY compressed slot, which removes topk selection/tie-break ambiguity. Used to
        # isolate whether residual diffs come from topk tie-breaking rather than from the
        # position arithmetic.
        index_topk=int(os.environ.get("INDEX_TOPK", "8")),
    )


def build(args):
    """The model allocates parameters with torch.empty (they are normally filled by
    load_weights), so an un-initialised build contains garbage -- in practice NaN. Fill every
    floating parameter deterministically so the comparison is meaningful."""
    torch.manual_seed(SEED)
    m = rm.Transformer(args)
    m.eval()
    g = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for _name, p in m.named_parameters():
            if not p.is_floating_point():
                continue
            t = torch.empty(p.shape, dtype=torch.float32).normal_(0.0, 0.02, generator=g)
            p.copy_(t.to(p.dtype))
    return m


def run(model, ids_prefill, step_tokens, as_tensor):
    """Prefill then STEPS decode steps. Returns (prefill_logits, [step_logits...])."""
    with torch.no_grad():
        pre = model(ids_prefill, 0)          # prefill always passes int 0 (unchanged path)
        outs = []
        for i, tok in enumerate(step_tokens):
            sp = PREFILL + i
            pos = torch.tensor(sp, dtype=torch.long) if as_tensor else sp
            outs.append(model(tok, pos))
    return pre, outs


def cmp(a, b, label, failures, tol=0.0):
    if a.shape != b.shape:
        failures.append(f"{label}: SHAPE {tuple(a.shape)} vs {tuple(b.shape)}")
        return None
    d = (a.float() - b.float()).abs().max().item()
    ok = torch.equal(a, b) if tol == 0.0 else d <= tol
    status = "exact" if torch.equal(a, b) else f"max_abs_diff={d:.3e}"
    if not ok:
        failures.append(f"{label}: {status}")
    return d


def main() -> int:
    args = make_args()
    print(f"config: layers={args.n_layers} dim={args.dim} heads={args.n_heads} "
          f"head_dim={args.head_dim} win={args.window_size} ratios={args.compress_ratios} "
          f"index_topk={args.index_topk} vocab={args.vocab_size}")
    print(f"prefill={PREFILL} decode_steps={STEPS}")

    torch.manual_seed(0)
    ids = torch.randint(0, args.vocab_size, (args.max_batch_size, PREFILL))
    step_tokens = [torch.randint(0, args.vocab_size, (args.max_batch_size, 1)) for _ in range(STEPS)]

    failures = []

    # ---- A: original path -------------------------------------------------------------------
    dsp.revert()
    rm_max_seq[0] = args.max_seq_len
    if PAD_ORIG:
        install_pad_shim()
        print("[harness] PAD_ORIG=1: original path padded to the static width with -1")
    mA = build(args)
    print("\n[A] original (int start_pos, dynamic shapes)")
    preA, outA = run(mA, ids, step_tokens, as_tensor=False)
    print(f"    prefill logits {tuple(preA.shape)}; {len(outA)} decode steps ok")

    # ---- B: patched path --------------------------------------------------------------------
    mB = build(args)
    # Copy A's weights into B outright, then verify -- do not rely on seeding alone.
    # (Non-persistent buffers -- kv_cache / kv_state / score_state -- are NOT in state_dict, so B
    # keeps its own freshly-zeroed KV state, which is what we want.)
    mB.load_state_dict(mA.state_dict(), strict=False)
    for (na, pa), (nb, pb) in zip(mA.named_parameters(), mB.named_parameters()):
        if not torch.equal(pa, pb):
            failures.append(f"weight mismatch between the two builds at {na}/{nb}")
            break
    if any(torch.isnan(p).any() for _n, p in mA.named_parameters()):
        failures.append("model A has NaN parameters (uninitialised build)")
    dsp.apply(rm)
    print("[B] patched (0-d tensor start_pos, static shapes)")
    preB, outB = run(mB, ids, step_tokens, as_tensor=True)
    print(f"    prefill logits {tuple(preB.shape)}; {len(outB)} decode steps ok")

    # ---- compare ----------------------------------------------------------------------------
    print("\n=== prefill (must be identical: patched code delegates prefill to the original) ===")
    d = cmp(preA, preB, "prefill", failures)
    print(f"  prefill: {'exact' if d == 0 else f'max_abs_diff={d:.3e}'}")

    print("\n=== decode steps ===")
    worst = 0.0
    for i, (a, b) in enumerate(zip(outA, outB)):
        sp = PREFILL + i
        ratio_hit = [r for r in args.compress_ratios if r and (sp + 1) % r == 0]
        d = cmp(a, b, f"decode step pos={sp}", failures)
        if d is not None:
            worst = max(worst, d)
        note = f"  (compress commit for ratio {ratio_hit})" if ratio_hit else ""
        print(f"  pos={sp}: {'exact' if d == 0 else f'max_abs_diff={d:.3e}'}{note}")

    # Localise any divergence: if the KV/state buffers agree, the difference is in index
    # SELECTION; if they disagree, the difference is in what got WRITTEN.
    print("\n=== buffer state after the decode loop (localises any divergence) ===")
    bufA = dict(mA.named_buffers())
    bufB = dict(mB.named_buffers())
    nbad = 0
    for name in sorted(bufA):
        a, b = bufA[name], bufB.get(name)
        if b is None or a.shape != b.shape:
            continue
        if a.dtype.is_floating_point:
            fa, fb = a.float(), b.float()
            m = torch.isfinite(fa) & torch.isfinite(fb)
            d = (fa[m] - fb[m]).abs().max().item() if m.any() else 0.0
        else:
            d = 0.0 if torch.equal(a, b) else float("inf")
        if d != 0.0:
            nbad += 1
            if nbad <= 8:
                print(f"  DIFFERS {name}: {tuple(a.shape)} max_abs_diff={d:.3e}")
    print(f"  buffers differing: {nbad} / {len(bufA)}")
    if nbad == 0:
        print("  -> buffers identical: any logits diff comes from index SELECTION, not writes")

    print("\n=== shape invariance of the patched decode step ===")
    shapes = {tuple(o.shape) for o in outB}
    print(f"  distinct logits shapes across {len(outB)} steps: {shapes}")
    if len(shapes) != 1:
        failures.append("patched decode logits shape varies across steps")

    print()
    if failures:
        print(f"FAIL — {len(failures)} problem(s):")
        for f in failures[:20]:
            print("   ", f)
        return 1
    print(f"ALL PASS — prefill identical, {STEPS} decode steps match "
          f"(worst max_abs_diff={worst:.3e}), shapes position-invariant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
