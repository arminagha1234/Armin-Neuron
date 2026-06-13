# ✅ Neuron Inference WORKING — 2026-06-13

## Result

```
CPU reference: 2 + 2 = **4** (4.4s)
Neuron output: 2 + 2 =  **4** (228.8s, eager, no compile)
```

**Correct output on Neuron!** Minor whitespace difference from bf16 precision.

## Configuration

- Instance: trn2.48xlarge
- Container: Beta 3 DLC (torch 2.11, torch_neuronx 2.11.3)
- Model: google/gemma-4-E4B-it (7.94B params)
- Device: `torch.device("neuron")` (single core, full model)
- Mode: eager (no torch.compile yet)
- Dtype: bfloat16
- Patch: Gemma4RMSNorm.forward → bf16-only (no .float() casts)

## Key Requirements

1. `AutoProcessor.apply_chat_template()` → provides `mm_token_type_ids`
2. `Gemma4RMSNorm.forward` patched to avoid `.float()` (Neuron mixed-dtype limitation)
3. Full model on device (14.93 GB fits on trn2 core, needs TP=2 for inf2.xlarge)

## Next Steps

- [ ] `torch.compile(backend="neuron")` for compiled inference (10-50× speedup expected)
- [ ] TP=2 for inf2.xlarge deployment (split across 2 cores)
- [ ] TTFT + throughput benchmarks
