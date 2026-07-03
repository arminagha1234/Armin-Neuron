"""
Run chai-1 end-to-end on Trainium via Native PyTorch (Beta 3).

Strategy: chai already threads a `device` arg through run_inference ->
load_exported -> ModuleWrapper, so we just pass device=torch.device("neuron").
All 6 TorchScript components (feature_embedding, bond_loss_input_proj,
token_embedder, trunk, diffusion_module, confidence_head) load on CPU then
.to("neuron") and execute eagerly on the NeuronCore.

We disable ESM embeddings (separate HF model, hardcoded cuda:0) and templates
(rigid.py hardcodes .cuda()) to keep the core path clean for the first pass.

Env: /home/ubuntu/workspace/native_venv  (torch 2.11 + native torch_neuronx)
"""
import logging, shutil, sys, time
from pathlib import Path

logging.basicConfig(level=logging.INFO)
import torch
import torch_neuronx  # registers the "neuron" PrivateUse1 backend

NEURON = torch.device("neuron")

from chai_lab.chai1 import run_inference

fasta = Path("/home/ubuntu/mini.fasta")
fasta.write_text(">protein|name=pep\nGAAL\n")
out = Path("/home/ubuntu/chai_out_neuron")
if out.exists():
    shutil.rmtree(out)
out.mkdir()

t0 = time.time()
candidates = run_inference(
    fasta_file=fasta,
    output_dir=out,
    num_trunk_recycles=1,
    num_diffn_timesteps=2,
    seed=42,
    device=NEURON,             # <-- the whole point: native neuron device
    use_esm_embeddings=False,  # ESM hardcodes cuda:0; skip for first pass
)
dt = time.time() - t0
print(f"NEURON end-to-end run OK in {dt:.1f}s")
print("cif_paths:", [str(p) for p in candidates.cif_paths])
scores = [rd.aggregate_score.item() for rd in candidates.ranking_data]
print("aggregate_scores:", scores)
print("NATIVE_E2E_DONE")
