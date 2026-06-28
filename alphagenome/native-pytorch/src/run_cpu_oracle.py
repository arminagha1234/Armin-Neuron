"""Stage 0/1: run AlphaGenome on CPU to produce the correctness oracle.

Saves summarized output stats to oracle_cpu.pt for later comparison against the
Neuron run.
"""
import sys, time, torch
from alphagenome_pytorch import AlphaGenome
from common import WEIGHTS, ORGANISM, SEQ_LEN, make_input, summarize, oracle_path, heads_to_run

OUT = oracle_path()


def main():
    torch.manual_seed(0)
    print(f"[cpu] loading {WEIGHTS}")
    model = AlphaGenome.from_pretrained(WEIGHTS, device="cpu")
    model.eval()

    x = make_input()
    heads = heads_to_run()
    print(f"[cpu] input {tuple(x.shape)} organism={ORGANISM} heads={heads}")

    t0 = time.time()
    with torch.no_grad():
        out = model.predict(x, organism_index=ORGANISM, heads=heads)
    dt = time.time() - t0
    print(f"[cpu] forward done in {dt:.1f}s")

    stats = summarize(out)
    torch.save({"stats": stats, "seq_len": x.shape[1], "organism": ORGANISM}, OUT)
    print(f"[cpu] saved oracle -> {OUT} ({len(stats)} keys)")
    for k in sorted(stats):
        s = stats[k]
        print(f"    {k:32s} shape={s['shape']} mean={s['mean']:+.4e} std={s['std']:.4e}")


if __name__ == "__main__":
    sys.exit(main())
