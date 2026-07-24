"""Validate a native-PyTorch Llama port on Trainium against a CPU reference.

Teacher-forced check: feed one prompt, compare the top-1 (argmax) next-token
prediction of the Neuron bf16 model against a CPU fp32 reference at every
position. >=95% agreement means the port is faithful.

    python3 validate.py huggyllama/llama-7b

LLaMA-1 7B scores 100% (39/39 positions) on a single Trn1 NeuronCore.
"""
import sys, time, torch, torch_neuronx
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = sys.argv[1] if len(sys.argv) > 1 else "huggyllama/llama-7b"
name = sys.argv[2] if len(sys.argv) > 2 else model_path
prompt = ("Artificial intelligence is transforming the world. In this paper we "
          "describe how large language models can be trained efficiently on custom "
          "accelerators such as AWS Trainium. The key idea is to")

tok = AutoTokenizer.from_pretrained(model_path)
ids = tok(prompt, return_tensors="pt").input_ids
print(f"[{name}] prompt tokens: {ids.shape[1]}", flush=True)


def per_pos_argmax(model, ids, device):
    model.eval()
    with torch.no_grad():
        out = model(ids.to(device), use_cache=False).logits
    return out.float().argmax(-1)[0].cpu()


print("loading CPU fp32 reference...", flush=True)
cpu_m = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32, attn_implementation="eager")
t0 = time.time(); cpu_pred = per_pos_argmax(cpu_m, ids, "cpu")
print("cpu forward %.1fs" % (time.time() - t0), flush=True)
del cpu_m

print("loading neuron bf16...", flush=True)
dev = torch.device("neuron")
nrn_m = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation="eager").to(dev)
t0 = time.time(); nrn_pred = per_pos_argmax(nrn_m, ids, dev)
print("neuron forward %.1fs (first run includes NEFF compile)" % (time.time() - t0), flush=True)

n = cpu_pred.shape[0]
match = (cpu_pred == nrn_pred).sum().item() / n * 100.0
print(f"[{name}] per-position top-1 agreement (neuron bf16 vs cpu fp32): {match:.1f}%  ({n} positions)")
print(f"[{name}] cpu    last-pos next-token: {tok.decode(cpu_pred[-1:])!r}")
print(f"[{name}] neuron last-pos next-token: {tok.decode(nrn_pred[-1:])!r}")
print("PORT_OK" if match >= 95.0 else "PORT_MISMATCH")
