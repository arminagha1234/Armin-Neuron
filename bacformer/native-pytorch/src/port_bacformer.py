#!/usr/bin/env python
"""Port Bacformer (bacterial genome model) to AWS Trainium via torch-neuronx.

Bacformer is a two-stage model:
  Stage 1: a protein LM (ESM-2) embeds each protein in a genome -> one vector/protein.
  Stage 2: a transformer (the "Bacformer" encoder) contextualises the sequence of
           per-protein embeddings ordered along the chromosome.

Porting notes
-------------
* The upstream `protein_seqs_to_bacformer_inputs` helper pulls protein embeddings
  through `faesm`, which needs CUDA `flash-attn`. We bypass it and compute Stage-1
  embeddings with the plain HuggingFace `EsmModel` (mean-pooled per protein) — the
  same ESM-2 model ported in ../esm2.
* The Bacformer encoder itself (its `trust_remote_code` modeling file) already uses
  `torch.nn.functional.scaled_dot_product_attention`, NOT flash-attn, so it traces
  to a NeuronCore directly. This script compiles Stage 2.

Special-token scheme (from configuration_bacformer.py):
  PAD=0 MASK=1 CLS=2 SEP=3 PROT_EMB=4 END=5 ; base hidden_size = 480 (esm2_t12_35M).

Run on a Neuron box:
    source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate
    pip install -r requirements.txt
    python port_bacformer.py --n-proteins 16
"""
import argparse
import torch
import torch_neuronx
from transformers import AutoModel, AutoTokenizer, EsmModel

CLS, SEP, PROT_EMB, END = 2, 3, 4, 5
ESM_MODEL = "facebook/esm2_t12_35M_UR50D"  # 480-d, matches Bacformer base hidden_size
BACFORMER_MODEL = "macwiatrak/bacformer-masked-MAG"  # 26M base encoder


def cosine(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


class BacformerEncoderWrapper(torch.nn.Module):
    """Stage 2 only: contextualise per-protein embeddings -> last_hidden_state."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, protein_embeddings, special_tokens_mask, token_type_ids):
        out = self.model(
            protein_embeddings=protein_embeddings,
            special_tokens_mask=special_tokens_mask,
            token_type_ids=token_type_ids,
            attention_mask=None,        # single genome, no padding -> full attention
            return_attn_weights=False,
            return_dict=False,
        )
        return out[0]  # last_hidden_state


def esm_protein_embeddings(seqs, seqlen=64):
    """Stage 1: mean-pooled ESM-2 embedding per protein -> (N, 480)."""
    tok = AutoTokenizer.from_pretrained(ESM_MODEL)
    esm = EsmModel.from_pretrained(ESM_MODEL).eval()
    enc = tok(seqs, return_tensors="pt", padding="max_length",
              truncation=True, max_length=seqlen)
    with torch.no_grad():
        hidden = esm(**enc).last_hidden_state          # (N, L, 480)
    mask = enc["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(1) / mask.sum(1)      # mean over residues
    return pooled                                      # (N, 480)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-proteins", type=int, default=16)
    p.add_argument("--out", default="bacformer_neuron.pt")
    args = p.parse_args()

    # A tiny toy "genome": N protein sequences.
    toy = ["MGYDLVAGFQKNVRTI", "MKAILVVLLG", "MQLIESRFYKDPWGNVHATC",
           "MSTNPKPQRFAWL", "MADEQLTKQAILNAWK", "MTKQELIDRVAKS"]
    seqs = [toy[i % len(toy)] for i in range(args.n_proteins)]

    print(f"[bacformer] Stage 1: ESM-2 embedding {len(seqs)} proteins")
    prot = esm_protein_embeddings(seqs)                # (N, 480)
    n, dim = prot.shape

    # Build Stage-2 input sequence: [CLS] p0 p1 ... p(N-1) [END]
    seqlen = n + 2
    embeds = torch.zeros(1, seqlen, dim)
    embeds[0, 1:1 + n, :] = prot
    stm = torch.full((1, seqlen), PROT_EMB, dtype=torch.long)
    stm[0, 0] = CLS
    stm[0, -1] = END
    tti = torch.zeros(1, seqlen, dtype=torch.long)     # single contig
    example = (embeds, stm, tti)

    print(f"[bacformer] loading encoder {BACFORMER_MODEL}")
    model = AutoModel.from_pretrained(BACFORMER_MODEL, trust_remote_code=True).eval()
    wrapper = BacformerEncoderWrapper(model).eval()

    print("[bacformer] CPU reference forward")
    with torch.no_grad():
        cpu_repr = wrapper(*example)

    print("[bacformer] tracing Stage-2 encoder with torch_neuronx (compiling)...")
    neuron_model = torch_neuronx.trace(wrapper, example)

    print("[bacformer] Neuron forward")
    neu_repr = neuron_model(*example)

    print(f"[bacformer] last_hidden_state shape={tuple(cpu_repr.shape)}")
    print(f"[bacformer] max_abs_diff={float((cpu_repr-neu_repr).abs().max()):.3e} "
          f"cosine={cosine(cpu_repr, neu_repr):.6f}")

    torch.jit.save(neuron_model, args.out)
    print(f"[bacformer] saved compiled Stage-2 model -> {args.out}")
    assert cosine(cpu_repr, neu_repr) > 0.99, "cosine too low — port failed"
    print("[bacformer] PASS")


if __name__ == "__main__":
    main()
