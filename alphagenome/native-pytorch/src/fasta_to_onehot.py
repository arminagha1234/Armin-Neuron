"""Turn a DNA FASTA into an AlphaGenome one-hot input array.

Output: a .npy of shape (1, length, 4), channels A=0, C=1, G=2, T=3.
The sequence is cropped (centered) or zero-padded (centered) to `--length`,
which must be a multiple of 128.

Usage:
    python fasta_to_onehot.py --fasta region.fa --length 131072 --out my_seq.npy

No external deps beyond numpy. Reads the first record in the FASTA (or use
--record to pick one by name).
"""
import argparse
import numpy as np

BASE = {"A": 0, "C": 1, "G": 2, "T": 3,
        "a": 0, "c": 1, "g": 2, "t": 3}


def read_fasta(path, record=None):
    seqs, name, buf = {}, None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf)
                name = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if name is not None:
            seqs[name] = "".join(buf)
    if not seqs:
        raise SystemExit("No sequences found in FASTA.")
    if record:
        if record not in seqs:
            raise SystemExit(f"Record {record!r} not found. Have: {list(seqs)[:5]}...")
        return record, seqs[record]
    first = next(iter(seqs))
    return first, seqs[first]


def to_onehot(seq, length):
    if length % 128 != 0:
        raise SystemExit(f"--length must be a multiple of 128 (got {length}).")
    # Center-crop or center-pad to `length`.
    s = len(seq)
    if s > length:
        start = (s - length) // 2
        seq = seq[start:start + length]
        left = 0
    else:
        left = (length - s) // 2
    oh = np.zeros((length, 4), dtype=np.float32)
    for i, ch in enumerate(seq):
        j = BASE.get(ch)
        if j is not None:                    # unknown bases (N) stay all-zero
            oh[left + i, j] = 1.0
    return oh[None, :, :]                     # (1, length, 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--length", type=int, default=131072, help="multiple of 128")
    ap.add_argument("--record", default=None, help="FASTA record name (default: first)")
    ap.add_argument("--out", default="my_seq.npy")
    args = ap.parse_args()

    name, seq = read_fasta(args.fasta, args.record)
    oh = to_onehot(seq, args.length)
    np.save(args.out, oh)
    print(f"record={name} input_len={len(seq)} -> {args.out} shape={oh.shape}")


if __name__ == "__main__":
    main()
