# OPD SFT Data Curation — Pure Native PyTorch on Trainium

Native-PyTorch (Beta-3) replacement for the Lightning-OPD **data-curation** stage:
sample prompts from OpenThoughts3-1.2M, generate teacher answers with **Qwen3-8B**,
save the prompt–answer dataset.

The upstream repo
([oldpilluwu/Lightning-OPD @ trainium](https://github.com/oldpilluwu/Lightning-OPD/tree/trainium))
does this with **vLLM + NxD-Inference**. This example does it with **only HuggingFace
`transformers` on `torch.device("neuron")`** — no vLLM, no NxDI — plus a hand-rolled
static-batched, KV-cached, top-p sampling decode loop.

## When to use this (and when not to)

| | vLLM + NxDI (upstream) | This (pure native PyTorch) |
|---|---|---|
| Throughput on 300k prompts | High (continuous batching, paged KV cache) | Lower (static batching) |
| Setup | vLLM venv + `vllm_neuron` plugin | just `transformers` in a native-PyTorch venv |
| Tensor parallel | yes | no (single NeuronCore — TP needs NxD, not "pure native") |
| Best for | the real 300k run | a native reference, a smoke test, or avoiding the vLLM-Neuron plugin |

**Honest guidance:** for the full 300k-prompt / 16k-token run, vLLM/NxDI is the right
tool — static batching here stalls each batch on its longest sequence and has no paged
KV cache. Use this when you specifically want a pure-native-PyTorch path (single-stack,
showcase, or to dodge the vLLM-Neuron plugin's rough edges).

## Scope

- **Single NeuronCore.** Qwen3-8B bf16 (~16 GB) fits in one trn2 logical core's HBM
  share. Multi-core tensor parallelism on Neuron requires NxD sharding, which is no
  longer "pure native PyTorch," so it's intentionally out of scope here.
- **Static batching + KV cache**, correct greedy/top-p sampling, EOS-aware early stop.

## Files

| File | What |
|---|---|
| `prepare_prompts.py` | stream OpenThoughts3-1.2M → `{"prompt": ...}` JSONL (300k or 64-smoke) |
| `generate_sft_data_native.py` | naive decode (reference): dynamic KV cache, recompiles per token |
| **`generate_sft_data_bucketed.py`** | **Route B (use this): StaticCache = compile-once decode + `messages`/`tokens` schema** |
| `requirements.txt` | `transformers`, `datasets`, `pandas`, `pyarrow` |

### Which script?
- **`generate_sft_data_bucketed.py`** is the one to run — StaticCache makes the decode
  compile once (not per token), and `--schema messages` emits parquet that's drop-in for
  the downstream SFT step. Validated on-device with Qwen3-8B (see RESULTS.md).
- `generate_sft_data_native.py` is the simpler reference that exposed the two on-device
  bugs (SDPA mask drop, on-device sampling).

## Environment (Beta-3 native PyTorch)

Run inside the Beta-3 native-PyTorch DLC / venv (`torch.device("neuron")`, no
`torch_xla`) — the same stack as the `clay/` example. `torch-neuronx` must be importable.
Then: `pip install transformers datasets pandas pyarrow`.

## Quick start

### 1. CPU smoke (no Neuron) — proves the decode loop is correct
```bash
python prepare_prompts.py --num-samples 64 --output data/prompts_smoke.jsonl
python generate_sft_data_native.py --device cpu --model sshleifer/tiny-gpt2 \
    --prompts data/prompts_smoke.jsonl --output data/out_smoke.parquet \
    --max-new-tokens 32 --batch-size 4 --no-chat-template
```

### 2. Real run on a trn2.3xlarge (single core), Qwen3-8B teacher
```bash
python prepare_prompts.py --num-samples 300000 --seed 42 \
    --output data/openthoughts3_300000.jsonl

NEURON_RT_NUM_CORES=1 python generate_sft_data_native.py --device neuron \
    --model Qwen/Qwen3-8B \
    --prompts data/openthoughts3_300000.jsonl \
    --output data/openthoughts3_300000_qwen3-8b.parquet \
    --max-new-tokens 16384 --batch-size 8 --temperature 0.7 --top-p 0.9
```

Start with `--limit 64` on the neuron run first to confirm the model loads, decodes,
and the answers look sane before committing to the full 300k.

## Output

Parquet with two columns: `prompt`, `answer` (falls back to JSONL if pyarrow is
unavailable). Wire this into the pipeline's SFT-training stage in place of the
vLLM-generated parquet.

## Validated on Trainium (see RESULTS.md)

Confirmed on a trn2.3xlarge, Beta-3 native DLC, `device="neuron"`, eager:
- **Qwen3-8B (real teacher), bf16, batched — coherent output.** Pure native PyTorch.
- bf16 works; batched left-padded decode works.

### The one gotcha we hit and fixed
On Neuron, the **SDPA attention path drops the 2D key-padding mask**, so left-padded
batches produce garbage on the shorter sequences. **Fix: `--attn-impl eager`** (now the
default), which applies the mask correctly. bf16 itself is fine — the earlier "bf16
garbage" was this same mask bug in disguise (it happened to be a batched run).

Sampling (`top_p_sample`) runs on **CPU in fp32**: `torch.multinomial`/sort aren't
reliable on-device and bf16 distributions can go degenerate. The model forward stays on
Neuron; only the tiny sampling step is on CPU.

## Known caveats / next steps

- **Throughput**: naive eager decode recompiles per sequence length (~seconds/token).
  Correct but slow — fine for a smoke/reference, too slow for the full 300k run.
  Next step (still pure native PyTorch): static-shape, length-bucketed decode with a
  preallocated KV cache so the graph compiles once per bucket, not once per step.
- **`torch.compile(backend="neuron")`** is intentionally NOT used — eager is the safe
  default (see the Lumos issue #546 NaN work on the beta compile path).
- **Driver**: the beta native userspace needs `aws-neuronx-dkms` **2.28** (public 2.29
  fails NRT init). Swap on the host before running in the DLC.
