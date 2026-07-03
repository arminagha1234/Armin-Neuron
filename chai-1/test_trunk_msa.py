"""Test the trunk on Neuron with a TRUNCATED MSA depth.

Hypothesis: the trunk's tensor_set_slice assertion is a 32-bit size overflow
driven by the giant (1, 16384, 256, 64) MSA tensor (mostly padding for a
no-MSA peptide). Truncating MSA depth should shrink tensors below the overflow
threshold. If it runs, 6/6 on Neuron is reachable via recycle_msa_subsample.

Usage: python test_trunk_msa.py <msa_depth>   e.g. 512
"""
import sys, time, torch
import torch_neuronx

DEPTH = int(sys.argv[1]) if len(sys.argv) > 1 else 512
MODELS = "/home/ubuntu/workspace/native_venv/lib/python3.12/site-packages/downloads/models_v2/"
d = torch.load("/home/ubuntu/cap_trunk.pt.inputs.pt", weights_only=False)
crop, kw = d["crop_size"], d["kw"]

# truncate MSA depth (dim 1 of msa_input_feats / msa_mask)
orig = tuple(kw["msa_input_feats"].shape)
kw["msa_input_feats"] = kw["msa_input_feats"][:, :DEPTH].contiguous()
kw["msa_mask"] = kw["msa_mask"][:, :DEPTH].contiguous()
print(f"MSA depth {orig[1]} -> {DEPTH}; msa_input_feats {tuple(kw['msa_input_feats'].shape)}", flush=True)

device = torch.device("neuron")
m = torch.jit.load(MODELS + "trunk.pt", map_location="cpu").to(device)
kw_dev = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in kw.items()}
fwd = getattr(m, f"forward_{crop}")
print(f"running trunk forward_{crop} on neuron (msa depth {DEPTH}) ...", flush=True)
t0 = time.time()
with torch.no_grad():
    out = fwd(**kw_dev)
outs = [o.detach().to("cpu").float() for o in (out if isinstance(out, (list, tuple)) else [out])]
print(f"TRUNK NEURON OK in {time.time()-t0:.1f}s; outputs={[tuple(o.shape) for o in outs]}", flush=True)
print("TRUNK_MSA_DONE", flush=True)
