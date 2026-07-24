# Training smoke test (start here)

A tiny GPT trained from scratch on random data for 10 steps. Its only job: prove
your Trainium instance can run a **full native-PyTorch training step** (forward +
backward + `AdamW`) before you spend time on real Llama models.

## Run

```bash
python3 train_smoke.py
```

## Expected output

```
step  0  loss 7.0979  (47.8s)   <- first step compiles a NEFF
step  1  loss 6.9971  (0.1s)     <- cached, fast
 ...
step  9  loss 6.2086  (0.0s)
TRAIN_SMOKE_OK
```

Loss should decrease monotonically. The first step takes ~30-60s (NEFF
compilation); every step after that is ~0.05s because the compiled graph is
cached.

## What it exercises

- `torch.device("neuron")` — model + tensors on Trainium, no XLA, no `mark_step`
- `F.scaled_dot_product_attention` (causal) — the supported attention path
- `nn.LayerNorm`, `nn.GELU`, `nn.Embedding`
- Autograd (`loss.backward()`) and `torch.optim.AdamW` on device
- `bfloat16` compute

If this passes, your box is ready for the Llama examples. If it fails, fix the
environment first (see [`../SETUP.md`](../SETUP.md)) — don't debug Llama on a
broken setup.
