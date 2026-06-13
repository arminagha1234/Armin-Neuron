"""Test E4B on Neuron — single core, TP=1 first, then TP=2.

Run: source /opt/torch-neuronx/.venv/bin/activate && python3 test_neuron_e4b.py
"""
import torch
import torch_neuronx
import time
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

# Neuron-safe RMSNorm (no .float() casts)
def _bf16_safe_forward(self, hidden_states):
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    normed = hidden_states * torch.pow(variance + self.eps, -0.5)
    if self.with_scale:
        normed = normed * self.weight
    return normed

Gemma4RMSNorm.forward = _bf16_safe_forward
print("Patched RMSNorm for Neuron compatibility")

MODEL_PATH = "google/gemma-4-E4B-it"

print("Loading model...")
proc = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    attn_implementation="eager",
)
model.eval()
print(f"Loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

# CPU reference first
messages = [{"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]}]
text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = proc(text=text, return_tensors="pt")

t0 = time.time()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=10, do_sample=False)
cpu_time = time.time() - t0
response = proc.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"\nCPU reference: {response} ({cpu_time:.1f}s)")

# Now try Neuron: move ENTIRE language model + lm_head to Neuron
# On trn2.48xl each core has plenty of memory (>>16 GB budget)
print("\nMoving ENTIRE model to Neuron...")
device = torch.device("neuron")
model = model.to(device)
print(f"Model on: {next(model.parameters()).device}")

# Move inputs to Neuron too
inputs_n = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

print("Generating on Neuron...")
t1 = time.time()
with torch.no_grad():
    out2 = model.generate(**inputs_n, max_new_tokens=10, do_sample=False)
neuron_time = time.time() - t1
response2 = proc.decode(out2[0].cpu()[inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(f"Neuron output: {response2} ({neuron_time:.1f}s)")

if response == response2:
    print("\n✅ CPU and Neuron outputs MATCH!")
else:
    print(f"\n⚠️ Outputs differ: CPU='{response}' vs Neuron='{response2}'")
