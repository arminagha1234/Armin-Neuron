"""Cold-start + NEFF cache persistence (pre-PR). Run twice in SEPARATE processes with a
persistent NEURON_COMPILE_CACHE_URL: run 1 = cold (pays compile), run 2 = warm (should hit
the on-disk NEFF cache -> fast). Confirms the warmup/cache story for a service."""
import os
import sys
import time
import torch
import torch_neuronx  # noqa
from transformers import CsmForConditionalGeneration

M = "/host/csm_1b"
DEV = torch.device("neuron")
torch.set_num_threads(24)
cache_url = os.environ.get("NEURON_COMPILE_CACHE_URL", "default")

model = CsmForConditionalGeneration.from_pretrained(M, dtype=torch.bfloat16).eval()
bb = model.backbone_model.to(DEV)
for m in bb.modules():
    for k, v in list(vars(m).items()):
        if torch.is_tensor(v) and not getattr(v, "is_meta", False) and v.device.type != "neuron":
            setattr(m, k, v.to(DEV))
H = bb.config.hidden_size
N = 1024
emb = torch.randn(1, N, H, dtype=torch.bfloat16, device=DEV)
pos = torch.arange(N, device=DEV).unsqueeze(0)
cbb = torch.compile(bb, backend="neuron", dynamic=False)

t = time.perf_counter()
with torch.no_grad():
    cbb(inputs_embeds=emb, position_ids=pos, use_cache=False).last_hidden_state.cpu()
dt = time.perf_counter() - t
print("[cold] backbone compile+first-run: %.1fs  cache_url=%s" % (dt, cache_url), flush=True)
