"""Hybrid chai-1 run: 5 components on Neuron (native eager), trunk on CPU.

The trunk hits a Beta 3 runtime assertion (tensor_set_slice). Since it's the
only failing component, we run it on CPU and keep everything else -- including
the compute-heavy diffusion module (called ~200x) -- on the NeuronCore. chai's
per-component device plumbing + ModuleWrapper.forward(move_to_device=...) makes
the device boundary automatic: trunk consumes/produces CPU tensors, and the
next component's wrapper moves them back onto Neuron.
"""
import logging, shutil, time
from pathlib import Path
logging.basicConfig(level=logging.INFO)
import torch
import torch_neuronx  # registers "neuron" backend
import chai_lab.chai1 as C

NEURON = torch.device("neuron")
CPU = torch.device("cpu")
FORCE_CPU = {"trunk.pt"}  # components to keep on CPU

# 1) Don't move trunk onto Neuron; keep it on CPU inside the context manager.
_orig_ctx = C._component_moved_to
from contextlib import contextmanager

@contextmanager
def hybrid_ctx(comp_key, device):
    tgt = CPU if comp_key in FORCE_CPU else device
    if comp_key not in C._component_cache:
        C._component_cache[comp_key] = C.load_exported(comp_key, tgt)
    component = C._component_cache[comp_key]
    component.jit_module.to(tgt)
    try:
        yield component
    finally:
        component.jit_module.to(CPU)
C._component_moved_to = hybrid_ctx

# 2) When a component's module is on CPU, force its inputs to CPU too
#    (overriding the neuron move_to_device the caller passed).
_orig_fwd = C.ModuleWrapper.forward
def _keyfor(self):
    for k, v in C._component_cache.items():
        if v is self:
            return k
    return None
def hybrid_fwd(self, crop_size, *, return_on_cpu=False, move_to_device=None, **kw):
    try:
        mdev = next(self.jit_module.parameters()).device
    except StopIteration:
        mdev = None
    if mdev is not None and mdev.type == "cpu":
        move_to_device = CPU
    # confidence_head outputs feed CPU-side scoring (boolean-mask indexing,
    # softmax_einsum_and_cpu) -> return on CPU to avoid mixed-device ops.
    if _keyfor(self) == "confidence_head.pt":
        return_on_cpu = True
    return _orig_fwd(self, crop_size, return_on_cpu=return_on_cpu,
                     move_to_device=move_to_device, **kw)
C.ModuleWrapper.forward = hybrid_fwd

fasta = Path("/home/ubuntu/mini.fasta")
fasta.write_text(">protein|name=pep\nGAAL\n")
out = Path("/home/ubuntu/chai_out_hybrid")
if out.exists():
    shutil.rmtree(out)
out.mkdir()

t0 = time.time()
candidates = C.run_inference(
    fasta_file=fasta, output_dir=out,
    num_trunk_recycles=1, num_diffn_timesteps=2, seed=42,
    device=NEURON, use_esm_embeddings=False,
)
dt = time.time() - t0
print(f"HYBRID end-to-end OK in {dt:.1f}s (trunk=CPU, rest=Neuron)")
print("cif_paths:", [str(p) for p in candidates.cif_paths])
print("aggregate_scores:", [rd.aggregate_score.item() for rd in candidates.ranking_data])
print("HYBRID_DONE")
