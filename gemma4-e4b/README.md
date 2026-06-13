# Gemma 4 E4B-it on Trainium/Inferentia

Google's [gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) (7.94B params) running on Neuron hardware with **correct text output**.

## ✅ Status: WORKING on Neuron

```
Q: What is the capital of France?   → A: The capital of France is **Paris**.
Q: What is 2+2?                     → A: 2 + 2 = **4**
```

| Instance | Status | Throughput | Cost | Notes |
|---|---|---|---|---|
| trn2.48xlarge | **✅ Working** | 3.3 tok/s (compiled) | ~$21.50/hr | Single core, full model |
| inf2.xlarge | **WIP** | — | $0.76/hr | Needs TP=2 (14.93 GB > 16 GB/core) |

## The Key Insight

E4B is a **multimodal model** (`Gemma4ForConditionalGeneration`). It requires `mm_token_type_ids` from `AutoProcessor`. Without this input → garbage output. With it → perfect text.

## Quick Start (trn2)

```python
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm
import torch

# Patch: Neuron doesn't support mixed bf16/f32 in RMSNorm
def _bf16_norm(self, hidden_states):
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    normed = hidden_states * torch.pow(variance + self.eps, -0.5)
    if self.with_scale:
        normed = normed * self.weight
    return normed
Gemma4RMSNorm.forward = _bf16_norm

proc = AutoProcessor.from_pretrained("google/gemma-4-E4B-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-4-E4B-it",
    torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager")
model = model.to("neuron")
model = torch.compile(model, backend="neuron")  # 84× speedup

messages = [{"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]}]
text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = proc(text=text, return_tensors="pt")
inputs = {k: v.to("neuron") for k, v in inputs.items()}
out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
# → "2 + 2 = **4**"
```

## Performance (trn2.48xlarge, single core)

| Mode | Time (9 tokens) | tok/s | Speedup |
|---|---|---|---|
| CPU (eager) | 6.0s | 1.5 | baseline |
| Neuron (eager) | 228s | 0.04 | — |
| **Neuron (compiled)** | **2.7s** | **3.3** | **84×** |

## Layout

```
gemma4-e4b/
├── README.md                              ← this file
├── native-pytorch/
│   ├── README.md
│   ├── src/
│   │   ├── run_e4b_neuron.py              ← working Neuron script
│   │   ├── run_e4b_native.py              ← CPU reference runner
│   │   ├── run_e4b.py                     ← TTFT benchmark
│   │   └── tp_plan.py                     ← TP sharding plan
│   └── results/
│       ├── neuron_compiled.md             ← 3.3 tok/s results
│       ├── neuron_working.md              ← first correct output
│       └── cpu_reference.md               ← Paris/4/Bonjour
└── vllm-neuron/                           ← earlier vLLM attempt (archived)
    └── ...
```

## Requirements

- Beta 3 DLC or PyTorch 2.11+ with `torch_neuronx`
- `transformers >= 5.12.0` (for Gemma4 model class)
- `torchvision` (for Gemma4Processor)
- Patch: `Gemma4RMSNorm.forward` → bf16-only (no `.float()` casts)

## Validation

- Date: 2026-06-13
- Instances: trn2.48xlarge (compiled, working) + inf2.xlarge (OOM on 1 core, needs TP=2)
- Transformers: 5.12.0, torch 2.11, torch_neuronx 2.11.3

## License

Model: [Gemma Terms of Use](https://ai.google.dev/gemma/terms)
