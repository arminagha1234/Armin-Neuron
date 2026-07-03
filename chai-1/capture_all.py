"""Capture inputs + CPU reference outputs for every chai TorchScript component.
Run under the DLAMI venv (CPU). Saves cap_<comp>.inputs.pt and cap_<comp>.ref.pt
for each of the 6 components so we can test each on Neuron in isolation.
"""
import logging, shutil, torch
from pathlib import Path
logging.basicConfig(level=logging.WARNING)
import chai_lab.chai1 as C

_orig = C.ModuleWrapper.forward
SAVED = {}

def keyfor(self):
    for k, v in C._component_cache.items():
        if v is self:
            return k
    return None

def patched(self, crop_size, *, return_on_cpu=False, move_to_device=None, **kw):
    k = keyfor(self)
    if k and k not in SAVED:
        cpu_kw = {kk: (vv.detach().cpu() if torch.is_tensor(vv) else vv)
                  for kk, vv in kw.items()}
        torch.save({"crop_size": crop_size, "kw": cpu_kw},
                   f"/home/ubuntu/cap_{k}.inputs.pt")
        SAVED[k] = True
        print("CAPTURED", k, "crop", crop_size, "keys", list(cpu_kw.keys()), flush=True)
    res = _orig(self, crop_size, return_on_cpu=return_on_cpu,
                move_to_device=move_to_device, **kw)
    if k and f"out_{k}" not in SAVED:
        def tocpu(x):
            return x.detach().cpu() if torch.is_tensor(x) else x
        try:
            ros = [tocpu(x) for x in res] if isinstance(res, (list, tuple)) else tocpu(res)
            torch.save(ros, f"/home/ubuntu/cap_{k}.ref.pt")
        except Exception as e:
            print("refsave fail", k, e, flush=True)
        SAVED[f"out_{k}"] = True
    return res

C.ModuleWrapper.forward = patched

fasta = Path("/home/ubuntu/mini.fasta")
fasta.write_text(">protein|name=pep\nGAAL\n")
out = Path("/home/ubuntu/cap_out")
if out.exists():
    shutil.rmtree(out)
out.mkdir()
C.run_inference(fasta_file=fasta, output_dir=out, num_trunk_recycles=1,
                num_diffn_timesteps=2, seed=42, device="cpu",
                use_esm_embeddings=False)
print("CAPTURE_ALL_DONE", flush=True)
