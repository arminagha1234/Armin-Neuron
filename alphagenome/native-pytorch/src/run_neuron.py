"""Stage 1: run AlphaGenome on a Trainium NeuronCore (native PyTorch / torch_xla)
and compare against the CPU oracle.

Usage:
    AG_SEQLEN=16384 python run_neuron.py     # short first, then scale to 131072
"""
import os, sys, time, torch

# Keep the first compile observable and bounded.
os.environ.setdefault("NEURON_CC_FLAGS", "--model-type=transformer")

import torch_xla
import torch_xla.core.xla_model as xm

from alphagenome_pytorch import AlphaGenome
from common import WEIGHTS, ORGANISM, SEQ_LEN, make_input, summarize, compare, oracle_path, heads_to_run


def main():
    torch.manual_seed(0)
    dev = xm.xla_device()
    print(f"[neuron] device={dev} seq_len={SEQ_LEN} organism={ORGANISM}")

    print(f"[neuron] loading {WEIGHTS}")
    model = AlphaGenome.from_pretrained(WEIGHTS, device="cpu")
    model.eval()
    model = model.to(dev)

    x = make_input().to(dev)
    heads = heads_to_run()
    print(f"[neuron] heads={heads}")

    print("[neuron] first forward (traces + compiles — may take minutes)...")
    t0 = time.time()
    with torch.no_grad():
        out = model.predict(x, organism_index=ORGANISM, heads=heads)
    xm.mark_step()
    # Pull everything to CPU (forces execution).
    stats = summarize(out)
    dt = time.time() - t0
    print(f"[neuron] first forward (compile+run) {dt:.1f}s, {len(stats)} keys")

    # Warm timing
    t0 = time.time()
    with torch.no_grad():
        out = model.predict(x, organism_index=ORGANISM, heads=heads)
    xm.mark_step()
    _ = summarize(out)
    print(f"[neuron] warm forward {time.time()-t0:.2f}s")

    # Compare to oracle if present
    op = oracle_path()
    if os.path.exists(op):
        ref = torch.load(op)["stats"]
        ok, rows = compare(ref, stats)
        print(f"\n[neuron] vs CPU oracle ({op}): {'ALL OK' if ok else 'MISMATCHES'}")
        for key, status, c, mr, dm in rows:
            print(f"    {key:32s} {status:7s} {c:18s} {mr:16s} {dm}")
        return 0 if ok else 1
    else:
        print(f"[neuron] no oracle at {op}; run run_cpu_oracle.py at this AG_SEQLEN first.")
        for k in sorted(stats):
            s = stats[k]
            print(f"    {k:32s} shape={s['shape']} mean={s['mean']:+.4e} std={s['std']:.4e}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
