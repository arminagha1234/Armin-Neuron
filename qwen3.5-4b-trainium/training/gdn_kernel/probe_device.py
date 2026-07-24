"""Probe: does @nki.jit gdn_chunk_fwd run on torch.device('neuron') directly?"""
import sys, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gdn_nki_fwd import gdn_chunk_fwd

dev = torch.device("neuron")
BT, D, T = 128, 128, 256
torch.manual_seed(0)
q = F.normalize(torch.randn(T, D), dim=-1)
k = F.normalize(torch.randn(T, D), dim=-1)
v = torch.randn(T, D)
g = (-F.softplus(torch.randn(T, 1)) * 0.3)
beta = torch.rand(T, 1).sigmoid()
eye = torch.eye(BT); neg = -torch.tril(torch.ones(BT, BT), -1)
Ld = torch.tril(torch.ones(BT, BT)); ones_row = torch.ones(1, max(BT, D))
args = [x.float().to(dev) for x in (q, k, v, g, beta, eye, neg, Ld, ones_row)]
try:
    o, fs = gdn_chunk_fwd(*args)
    print("CALL_OK", "o.device=", o.device, "o.shape=", tuple(o.shape))
    print("o sum:", float(o.float().sum().cpu()))
    print("DEVICE_PROBE: SUCCESS")
except Exception as e:
    import traceback; traceback.print_exc()
    print("DEVICE_PROBE: FAIL", repr(e))
