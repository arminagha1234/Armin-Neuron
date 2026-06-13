# CPU Reference Output — Gemma 4 E4B

Date: 2026-06-13
Hardware: trn2.48xlarge (CPU only) + inf2.24xlarge (CPU only)
Transformers: 5.12.0, bf16, greedy decode, `Gemma4ForConditionalGeneration`

## Results (identical on both boxes)

```
Q: What is the capital of France?
A: The capital of France is **Paris**.

Q: What is 2+2?
A: 2 + 2 = **4**

Q: Explain gravity in one sentence.
A: Gravity is the fundamental force of attraction between any two objects with mass.

Q: Write a haiku about the ocean.
A: Blue waves crash and foam,

Q: Say hello in French.
A: The most common way to say "hello" in French is **Bonjour** (pronounced: bohn-zhoor).
```

## Key Discovery

E4B is a **multimodal model** (`Gemma4ForConditionalGeneration`). It requires
`mm_token_type_ids` from `AutoProcessor` to function. Without this tensor,
the model degenerates into repetition/garbage regardless of hardware.

Correct usage:
```python
proc = AutoProcessor.from_pretrained("google/gemma-4-E4B-it")
messages = [{"role": "user", "content": [{"type": "text", "text": "..."}]}]
text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
inputs = proc(text=text, return_tensors="pt")  # ← includes mm_token_type_ids
out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
```
