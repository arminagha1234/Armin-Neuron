"""Measure the usable HBM on a single NeuronCore.

Allocates 1 GB bf16 tensors until it OOMs, so you know how big a model can fit
before you try to load one. Useful when you hit
"NRT EXECUTION FAILED: Failed to allocate resource".

    python3 hbm_probe.py

On a Trn1 NeuronCore (16 GB HBM / total_hbm=17179869184) this reports OOM at
~14 GB — the runtime reserves the rest. That's why a 7B (~13.5 GB bf16) just
fits on one core and an 8B (~16 GB) does not.

The OOM is expected and is the point of the probe; the error text it prints at
the end is the same signature you'd see from a too-large model.
"""
import torch, torch_neuronx

d = torch.device("neuron")
held = []
gb = 0.0
step = 1.0

print("probing single-core HBM...", flush=True)
try:
    while gb < 64:
        n = int(step * (1024 ** 3) // 2)   # bf16 elements per 'step' GB
        t = torch.ones(n, dtype=torch.bfloat16, device=d)
        _ = t.sum().item()                 # force the allocation to materialize
        held.append(t)
        gb += step
        print(f"held ~{gb:.0f} GB", flush=True)
except Exception as e:
    print(f"\nOOM after ~{gb:.0f} GB  ({type(e).__name__})")
    print("=> that is your usable per-core budget for weights + activations")
print("PROBE_DONE")
