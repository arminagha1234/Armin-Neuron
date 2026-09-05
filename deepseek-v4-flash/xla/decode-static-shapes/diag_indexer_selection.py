"""Diagnostic: why do a few decode steps differ between the original and static-shape paths?

The parity test established that all 13 KV/state buffers are bit-identical after 40 decode steps,
so every WRITE is correct and the residual logits diff must come from the Indexer's returned
top-k index selection. This script records the Indexer output from BOTH paths and answers:

  Q1. Do the two paths select the same SET of valid compressed indices?
  Q2. If not, are the swapped indices TIED in index_score (i.e. an arbitrary tie-break that the
      original also resolves arbitrarily), or genuinely different scores (a real bug)?

Run:  python diag_indexer_selection.py
"""

import os
import sys

import torch

V4 = os.environ.get("V4_SRC", "/work/v4")
for p in (os.path.join(V4, "hf_src"), V4):
    if p not in sys.path:
        sys.path.insert(0, p)

import fallback_ops as fo  # noqa: E402

sys.modules["kernel"] = fo

torch.set_default_dtype(torch.bfloat16)

import reference_model as rm  # noqa: E402
import decode_static_patch as dsp  # noqa: E402

_orig_rotate = rm.rotate_activation
rm.rotate_activation = lambda x: _orig_rotate(x.to(torch.bfloat16)).to(x.dtype)

SEED = 1234
PREFILL = 128
STEPS = int(os.environ.get("STEPS", "30"))
INDEX_TOPK = int(os.environ.get("INDEX_TOPK", "8"))


def make_args():
    return rm.ModelArgs(
        max_batch_size=2, max_seq_len=512, vocab_size=1024,
        dim=1024, moe_inter_dim=512, n_layers=3, n_hash_layers=0,
        n_heads=8, n_routed_experts=4, n_shared_experts=1, n_activated_experts=2,
        q_lora_rank=256, o_lora_rank=256, o_groups=8,
        window_size=128, compress_ratios=(0, 4, 128, 0),
        index_n_heads=8, index_head_dim=128, index_topk=INDEX_TOPK,
    )


def build(args):
    torch.manual_seed(SEED)
    m = rm.Transformer(args)
    m.eval()
    g = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for _n, p in m.named_parameters():
            if p.is_floating_point():
                p.copy_(torch.empty(p.shape, dtype=torch.float32).normal_(0.0, 0.02, generator=g).to(p.dtype))
    return m


def install_recorder(store):
    """Wrap whatever Indexer.forward is currently installed and record its inputs/outputs."""
    cur = rm.Indexer.forward

    def wrapper(self, x, qr, start_pos, offset):
        out = cur(self, x, qr, start_pos, offset)
        pos = int(start_pos.item()) if torch.is_tensor(start_pos) else int(start_pos)
        store.append({
            "pos": pos,
            "ratio": self.compress_ratio,
            "idx": out.detach().clone(),
            "x": x.detach().clone(),
            "qr": qr.detach().clone(),
            "kv": self.kv_cache.detach().clone(),
            "mod": self,
        })
        return out

    rm.Indexer.forward = wrapper
    return cur


def recompute_index_score(mod, x, qr, kv, pos):
    """Replicate the Indexer's score math (pre-topk) so we can inspect ties."""
    with torch.no_grad():
        rd = mod.rope_head_dim
        freqs_cis = mod.freqs_cis.index_select(0, torch.tensor([pos]))
        q = mod.wq_b(qr).unflatten(-1, (mod.n_local_heads, mod.head_dim))
        q = rm._V4_ROPE_APPLY(q, rd, freqs_cis, False)
        q = rm.rotate_activation(q)
        rm.fp4_act_quant(q, rm.fp4_block_size, True)
        weights = mod.weights_proj(x) * (mod.softmax_scale * mod.n_heads ** -0.5)
        s = torch.einsum("bshd,btd->bsht", q.float(), kv.float())
        s = (s.relu_() * weights.unsqueeze(-1)).sum(dim=2)      # [b, s, n_slots]
    return s


def run(model, ids, toks, as_tensor, store):
    with torch.no_grad():
        model(ids, 0)
        for i, t in enumerate(toks):
            sp = PREFILL + i
            model(t, torch.tensor(sp, dtype=torch.long) if as_tensor else sp)
    return store


def main() -> int:
    args = make_args()
    print(f"index_topk={args.index_topk} steps={STEPS} ratios={args.compress_ratios}")
    torch.manual_seed(0)
    ids = torch.randint(0, args.vocab_size, (args.max_batch_size, PREFILL))
    toks = [torch.randint(0, args.vocab_size, (args.max_batch_size, 1)) for _ in range(STEPS)]

    # ---- A: original ----
    dsp.revert()
    recA = []
    baseA = install_recorder(recA)
    mA = build(args)
    run(mA, ids, toks, False, recA)
    rm.Indexer.forward = baseA

    # ---- B: patched ----
    mB = build(args)
    mB.load_state_dict(mA.state_dict(), strict=False)
    dsp.apply(rm)
    recB = []
    baseB = install_recorder(recB)
    run(mB, ids, toks, True, recB)
    rm.Indexer.forward = baseB

    # decode-only calls (prefill uses int start_pos and is recorded too; drop pos<PREFILL)
    decA = [r for r in recA if r["pos"] >= PREFILL]
    decB = [r for r in recB if r["pos"] >= PREFILL]
    print(f"indexer decode calls: A={len(decA)} B={len(decB)}")
    if len(decA) != len(decB):
        print("MISMATCHED CALL COUNTS -> the two paths call the Indexer a different number of times")
        return 1

    n_setdiff = 0
    first = None
    for k, (a, b) in enumerate(zip(decA, decB)):
        ia, ib = a["idx"], b["idx"]
        # compare the SET of valid (>=0) indices per (batch, query)
        sa = [sorted(v[v >= 0].tolist()) for v in ia.reshape(-1, ia.shape[-1])]
        sb = [sorted(v[v >= 0].tolist()) for v in ib.reshape(-1, ib.shape[-1])]
        if sa != sb:
            n_setdiff += 1
            if first is None:
                first = (k, a, b, sa, sb)

    print(f"calls whose selected VALID INDEX SET differs: {n_setdiff} / {len(decA)}")

    if first is None:
        print("\n=> Q1: selection sets are IDENTICAL on every call.")
        print("   So the logits diff is NOT selection. Remaining suspects: the width of")
        print("   topk_idxs passed to sparse_attn changing its numerics, or ORDER-dependent")
        print("   accumulation in the einsum over the gathered rows.")
        return 0

    k, a, b, sa, sb = first
    print(f"\n=> Q1: sets DIFFER. First at indexer decode call #{k}, pos={a['pos']}, "
          f"ratio={a['ratio']}")
    for row in range(min(len(sa), 2)):
        onlyA = sorted(set(sa[row]) - set(sb[row]))
        onlyB = sorted(set(sb[row]) - set(sa[row]))
        print(f"   row {row}: |A|={len(sa[row])} |B|={len(sb[row])} onlyA={onlyA[:8]} onlyB={onlyB[:8]}")

    # ---- Q2: are the swapped indices tied in score? ----
    print("\n=> Q2: score inspection at that call")
    s = recompute_index_score(a["mod"], a["x"], a["qr"], a["kv"], a["pos"])
    n_valid = (a["pos"] + 1) // a["ratio"]
    offset = a.get("offset", None)
    flat = s.reshape(-1, s.shape[-1])
    for row in range(min(flat.shape[0], 2)):
        sc = flat[row]
        valid_sc = sc[:n_valid]
        n_zero = int((valid_sc == 0).sum())
        uniq = int(torch.unique(valid_sc).numel())
        print(f"   row {row}: n_valid={n_valid} exact_zeros={n_zero} distinct_values={uniq}")
        onlyA = sorted(set(sa[row]) - set(sb[row]))
        onlyB = sorted(set(sb[row]) - set(sa[row]))
        # indices in the recorded output carry +offset; recover slot by subtracting the min offset
        # actually compare scores directly at the raw slot positions we can infer
        if onlyA and onlyB:
            print(f"      swapped-in scores  A-only: "
                  f"{[round(float(sc[i - (min(sa[row]) - 0)]), 6) if 0 <= i - 0 < sc.numel() else None for i in onlyA[:4]]}")
        if n_zero > 0 and n_zero >= n_valid - args.index_topk:
            print("      -> many EXACT ZEROS among valid scores: top-k has tied candidates, so a "
                  "variable-length topk and a fixed-width masked topk can legitimately pick "
                  "different members. Both are arbitrary; neither is 'more correct'.")
    print("\nCONCLUSION: selection-set difference confirmed. If distinct_values << n_valid the")
    print("difference is tie-break-only. To make the two paths agree bit-exactly, add the SAME")
    print("deterministic tiebreaker to index_score in both (e.g. -eps*slot_index).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
