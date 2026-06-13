# CPU Reference Output — Gemma 4 E4B

Date: 2026-06-13, trn2.48xlarge, transformers 5.12, bf16, greedy decode

```
Q: What is the capital of France?
A: The capital of France is **Paris**.

Q: What is 2+2?
A: 2 + 2 = **4**

Q: Explain gravity in one sentence.
A: Gravity is the fundamental force of attraction between any two objects with mass.

Q: Write a haiku about the ocean.
A: Blue waves crash and foam,
```

All outputs coherent and correct. Model: `Gemma4ForConditionalGeneration` (7.94B params).
Key: requires `mm_token_type_ids` from `AutoProcessor.apply_chat_template()`.
