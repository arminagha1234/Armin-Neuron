"""A1 — Multi-turn prefix-caching probe for CSM-1B TTFT.

Question this answers (honestly): for a CONVERSATIONAL model like CSM, does
per-turn TTFA grow with accumulated conversation context (so prefix-caching the
history is a big win, like it was for gemma4: 0.83 -> 0.42s), or is TTFA dominated
by the first-frame floor (backbone-step + 31-step depth decoder) regardless of how
much context precedes it?

If TTFA is ~flat vs context length  -> prefix caching does NOT help CSM (the floor
   is per-frame compute, not prefill). Pivot effort to the depth-loop (Tier B).
If TTFA grows with context length -> prefix caching is the #1 CSM win (Tier A).

Method:
  1. Build a synthetic multi-turn conversation of increasing length.
  2. For each context length L (tokens), measure:
       - prefill_ms:  time for the backbone PREFILL forward alone (isolates the
                      cost prefix-caching would eliminate)
       - ttfa_ms:     streaming time-to-first-audio for the next turn at that context
  3. Report prefill_ms and ttfa_ms vs L. The slope of prefill_ms vs L is the
     per-turn prize prefix-caching captures; compare it to the ~295ms/frame floor.

Reuses the validated offload setup (bf16 backbone/depth, fp32 codec, StaticCache).
Does not modify ../vllm_v1 (frozen reference).
"""
import os, sys, time, argparse, torch
import torch_xla.core.xla_model as xm
from transformers import AutoProcessor, CsmForConditionalGeneration

MODEL = os.environ.get("CSM_MODEL", os.path.expanduser("~/csm/csm_1b"))


def _to(obj, dev):
    if torch.is_tensor(obj):
        return obj.to(dev)
    try:
        from transformers.utils import ModelOutput
        if isinstance(obj, ModelOutput):
            for k in list(obj.keys()):
                obj[k] = _to(obj[k], dev)
            return obj
    except Exception:
        pass
    if obj.__class__.__name__.endswith("Cache"):
        return obj
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to(x, dev) for x in obj)
    if isinstance(obj, dict):
        return {k: _to(v, dev) for k, v in obj.items()}
    return obj


def _offload(module, dev, method="forward"):
    module.to(dev)
    for m in module.modules():
        for k, v in list(vars(m).items()):
            if torch.is_tensor(v) and v.device.type != "xla":
                setattr(m, k, v.to(dev))
    real = getattr(module, method)

    def wrapped(*a, **k):
        out = real(*_to(a, dev), **_to(k, dev))
        xm.mark_step()
        return _to(out, "cpu")
    setattr(module, method, wrapped)


class _FirstFrameStreamer:
class _FirstFrameStreamer:
    """(unused; kept for reference) Streamer-based TTFA probe. We switched to timing the
    full generate(max_new_tokens=1) call instead, because HF streamers receive the prompt
    on the first put() and fire ~immediately, corrupting a streamer-based first-frame time."""
    def __init__(self, model, eos_id):
        self.model = model
        self.eos_id = eos_id
        self.t0 = None
        self.ttfa = None
        self.n = 0

    def put(self, value):
        codes = value if value.dim() == 2 else value[None, :]
        if (codes[0, :-1] == self.eos_id).all():
            return
        self.n += 1
        if self.ttfa is None:
            self.ttfa = (time.perf_counter() - self.t0) * 1000.0

    def end(self):
        pass


def build_conversation(n_turns, words_per_turn):
    """A growing single-speaker conversation. Each turn is a chunk of text; the
    'context' for turn k is the concatenation of turns 0..k-1, prompting turn k.
    We approximate context length by repeating a sentence `words_per_turn` long."""
    base = ("the quick brown fox jumps over the lazy dog and then keeps "
            "running across the wide green field under a bright morning sky ")
    words = base.split()
    turns = []
    for _ in range(n_turns):
        txt = " ".join((words * ((words_per_turn // len(words)) + 1))[:words_per_turn])
        turns.append(txt)
    return turns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--words-per-turn", type=int, default=20)
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    dev = xm.xla_device()
    print(f"[a1] loading {args.model} ...")
    proc = AutoProcessor.from_pretrained(args.model)
    model = CsmForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16).eval()
    model.codec_model = model.codec_model.float()
    _offload(model.backbone_model, dev)
    _offload(model.codec_model, dev, method="decode")

    turns = build_conversation(args.turns, args.words_per_turn)

    # First-frame latency at growing context lengths. We time the full
    # generate(max_new_tokens=1) call: prefill(context) + ONE frame (backbone-step +
    # 31-step depth loop), output_audio=False so no codec. The per-frame floor is
    # constant, so the SLOPE of this latency vs context tokens isolates the
    # prefill-growth cost that prefix-caching would eliminate.
    def time_ttfa(text):
        inputs = proc(text, add_special_tokens=True, return_tensors="pt")
        L = inputs["input_ids"].shape[-1]
        gkw = dict(output_audio=False, do_sample=False, max_new_tokens=1,
                   cache_implementation="static")
        # warm (compile this prefill shape)
        with torch.no_grad():
            model.generate(**inputs, **gkw)
        # measure (best of 3)
        best = 1e9
        for _ in range(3):
            t = time.perf_counter()
            with torch.no_grad():
                model.generate(**inputs, **gkw)
            best = min(best, (time.perf_counter() - t) * 1000.0)
        return L, best

    print(f"\n[a1] {'ctx_turns':>9} {'ctx_tokens':>10} {'frame1_ms':>9}")
    rows = []
    for k in range(1, args.turns + 1):
        # context = turns 0..k-1 concatenated, single speaker [0]
        ctx_text = "[0]" + " ".join(turns[:k])
        L, ttfa_ms = time_ttfa(ctx_text)
        rows.append((k, L, ttfa_ms))
        print(f"[a1] {k:>9} {L:>10} {ttfa_ms:>9.1f}")

    # Slope analysis: how much does TTFA grow per added context token?
    if len(rows) >= 2:
        dL = rows[-1][1] - rows[0][1]
        d_ttfa = rows[-1][2] - rows[0][2]
        print(f"\n[a1] context grew {dL} tokens across {len(rows)} turns")
        print(f"[a1] TTFA    grew {d_ttfa:+.1f}ms  ({d_ttfa/max(dL,1):+.3f} ms/token)")
        print("\n[a1] VERDICT:")
        if d_ttfa > 50:
            print("  TTFA grows materially with context -> PREFIX CACHING IS A WIN (Tier A).")
            print(f"  A cached history would save ~{d_ttfa:.0f}ms/turn at this depth.")
        else:
            print("  TTFA ~flat vs context -> prefix caching does NOT move CSM TTFA.")
            print("  Floor is per-frame compute (backbone-step + 31-step depth), not prefill.")
            print("  => Redirect effort to the depth-loop (Tier B), not prefix caching.")


if __name__ == "__main__":
    sys.exit(main())
