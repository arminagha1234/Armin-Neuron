"""Shared helpers for AlphaGenome Trainium bring-up.

Keeps the input generation identical between the CPU oracle and the Neuron run so
outputs are comparable.
"""
import os
import numpy as np
import torch

WEIGHTS = "/scratch/alphagenome/hf/model_fold_0.safetensors"
# Sequence length must be a multiple of 128 (the encoder downsamples 1bp -> 128bp).
# Override with AG_SEQLEN to start short for the first Neuron compile, then scale.
SEQ_LEN = int(os.environ.get("AG_SEQLEN", "131072"))
ORGANISM = 0              # 0 = human, 1 = mouse
SEED = 1234


def oracle_path(seq_len: int = SEQ_LEN) -> str:
    return f"/scratch/alphagenome/oracle_cpu_{seq_len}.pt"


# All prediction heads. `splice_junctions` requires a top-k/sort over the sequence
# which lowers to an HLO `sort` that neuronx-cc does not support on trn2
# (NCC_EVRF029). We skip it on Neuron by default; everything else runs on-device.
ALL_HEADS = (
    "atac", "dnase", "procap", "cage", "rna_seq", "chip_tf", "chip_histone",
    "contact_maps", "splice_sites", "splice_site_usage", "splice_junctions",
)
SKIP_ON_NEURON = ("splice_junctions",)


def heads_to_run():
    """Head tuple to compute. Override with AG_HEADS='atac,dnase' to narrow."""
    env = os.environ.get("AG_HEADS")
    if env:
        return tuple(h.strip() for h in env.split(",") if h.strip())
    return tuple(h for h in ALL_HEADS if h not in SKIP_ON_NEURON)


def make_input(seq_len: int = SEQ_LEN, seed: int = SEED) -> torch.Tensor:
    """Deterministic one-hot DNA in NLC format: (1, seq_len, 4). A=0,C=1,G=2,T=3."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, 4, size=(1, seq_len))
    onehot = np.eye(4, dtype=np.float32)[idx]
    return torch.from_numpy(onehot)


def summarize(outputs: dict) -> dict:
    """Reduce the nested prediction dict to comparable stats per (head, resolution).

    Returns {key: {'shape', 'mean', 'std', 'absmax', 'slice'}} where slice is a
    small flattened sample for direct numerical comparison.
    """
    stats = {}

    def add(key, t):
        t = t.detach().to(torch.float32).cpu()
        flat = t.flatten()
        stats[key] = {
            "shape": tuple(t.shape),
            "mean": flat.mean().item(),
            "std": flat.std().item(),
            "absmax": flat.abs().max().item(),
            "slice": flat[:64].clone(),
        }

    for head, val in outputs.items():
        if isinstance(val, dict):
            for res, t in val.items():
                if torch.is_tensor(t):
                    add(f"{head}@{res}", t)
                elif isinstance(t, dict):
                    for sub, st in t.items():
                        if torch.is_tensor(st):
                            add(f"{head}@{res}.{sub}", st)
        elif torch.is_tensor(val):
            add(head, val)
    return stats


def compare(ref: dict, test: dict, atol: float = 1e-2, rtol: float = 1e-2):
    """Compare two summarize() dicts. Returns (all_ok, list_of_rows)."""
    rows = []
    all_ok = True
    for key in sorted(ref):
        if key not in test:
            rows.append((key, "MISSING", "", "", ""))
            all_ok = False
            continue
        r, t = ref[key], test[key]
        if r["shape"] != t["shape"]:
            rows.append((key, "SHAPE", str(r["shape"]), str(t["shape"]), ""))
            all_ok = False
            continue
        a = r["slice"]
        b = t["slice"]
        denom = a.abs().clamp_min(1e-6)
        max_rel = ((a - b).abs() / denom).max().item()
        # cosine over the sampled slice
        cos = torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
        ok = max_rel <= rtol or (a - b).abs().max().item() <= atol
        all_ok = all_ok and ok
        rows.append((key, "OK" if ok else "DIFF", f"cos={cos:.6f}",
                     f"maxrel={max_rel:.4f}", f"Δmean={t['mean']-r['mean']:+.4e}"))
    return all_ok, rows
