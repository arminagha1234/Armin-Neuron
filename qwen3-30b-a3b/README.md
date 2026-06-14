# Qwen3-30B-A3B on AWS Trainium2

Google/Alibaba's [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)
(30B total params, ~3B active per token, MoE) served on Trainium2 via
vLLM-Neuron.

| Path | Status | TP | Output Quality | Notes |
|---|---|---|---|---|
| vLLM-Neuron | **Working** | 4 | ✅ Correct text | Chain-of-thought + final answer |

## Quick Start

```bash
# In the vLLM-Neuron container (v5 image):
vllm serve Qwen/Qwen3-30B-A3B \
    --tensor-parallel-size 4 \
    --max-model-len 2048 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 2048
```

Server comes up at `http://0.0.0.0:8000` after ~15 min cold compile.

## Validated output

```
User: What is the capital of France? Answer in one sentence.

Model: <think>
Okay, the user is asking for the capital of France... I'm pretty sure
Paris is the correct answer... Yes, Paris is the capital of France.
</think>

The capital of France is Paris.
```

- 132 completion tokens, 20 prompt tokens
- Chain-of-thought reasoning works correctly
- Temperature=0 (greedy)

## Validation

- Date: 2026-06-14
- Instance: trn2.48xlarge (`i-0c2806a95b490e26e`, us-east-2)
- Container: vllm-neuron-private-beta-trn10-v5
- TP=4, max_model_len=2048

## License

Model: [Apache-2.0](https://huggingface.co/Qwen/Qwen3-30B-A3B)
