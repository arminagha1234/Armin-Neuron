# References

Curated list of what to consult for kernel patterns, optimization
techniques, and prior art. Extends the original `websites_check` note.

## Neuron-specific (start here)

### Official docs
- [AWS Neuron SDK docs](https://awsdocs-neuron.readthedocs-hosted.com/) — canonical reference
- [NKI programming guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/) — kernel authoring
- [Neuron device security disclosures](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/security.html) — memory-access caveats
- Neuron ParallelCluster samples — https://github.com/aws-neuron/aws-neuron-parallelcluster-samples

### Existing internal tooling (already in this workspace)
- `neuron/neuron-agentic-development-merged/` — autoport + NKI writer/debugger/optimizer agents
- `neuron/vllm-neuron/` — vLLM-Neuron backend, kernel adapters
- `neuron/examples/` — reference implementations we can pattern-match against

### Existing skills (in `.kiro/skills/`, if applicable)
- `neuron-framework-autoport` — HF → NxDI porting workflow
- `neuron-framework-autoport-vllm-neuron` — HF → vLLM-Neuron
- `neuron-framework-equivalence` — the 8-stage source↔target verification pipeline
- `neuron-nki-writing`, `neuron-nki-debugging`, `neuron-nki-optimizing` — kernel dev
- `neuron-nki-profile-querying`, `neuron-nki-profile-analysis` — profile-guided optimization
- `neuron-framework-nativept` — native-PyTorch (TorchNeuron) path

## Reference implementations to steal patterns from

Semantic borrowing only (see `open-questions.md` Q4). Cite in-lesson.

### vLLM (Apache 2.0)
- Repo: https://github.com/vllm-project/vllm
- What to learn: continuous batching, paged attention, KV cache management,
  prefix caching, RoPE fusion, sampling, quantization pipelines
- Neuron-relevant: https://github.com/aws-neuron/upstreaming-to-vllm

### SGLang (Apache 2.0)
- Repo: https://github.com/sgl-project/sglang
- What to learn: radix cache, structured decoding, batching heuristics

### TensorRT-LLM (Apache 2.0, Nvidia terms)
- Repo: https://github.com/NVIDIA/TensorRT-LLM
- What to learn: fused kernel patterns (RMSNorm+RoPE+attention), custom
  attention variants, quantization-aware fusion

### FlashAttention family
- Repos: https://github.com/Dao-AILab/flash-attention (v1/v2/v3)
- What to learn: block-wise attention, softmax numerics, backward-pass
  optimization patterns (mostly training, but forward patterns transfer)

### Hugging Face
- transformers repo: reference implementations to diff against for
  equivalence
- optimum-neuron: existing HF↔Neuron bindings
- text-generation-inference (TGI): production inference patterns

### Other worth watching
- MLC-LLM (Apache 2.0): universal deployment, TVM-based
- Together AI's open recipes (when they publish them)
- OpenPipe / TogetherCompute performance write-ups

## Compiler autotuning / AutoML prior art (for phase 4)

- TVM AutoTVM: https://tvm.apache.org/docs/reference/api/python/auto_scheduler.html
- Ansor (TVM's cost-model-guided search)
- Google's XLA autotuning
- AlphaTensor / AlphaChip (RL over discrete search spaces)
- Auto-sklearn meta-learning warmstart

Not new terrain — pull from these, don't reinvent.

## Model catalogs (for the top-100 list)

- HF Trending: https://huggingface.co/models?sort=trending
- HF Downloads-last-30d (via API)
- LMSYS Chatbot Arena leaderboard (quality signal — orthogonal to perf but
  useful for "which models does the community care about")
- Open LLM Leaderboard: https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard

## Comparison baselines (for the leaderboard)

To keep our numbers honest:
- vLLM benchmark suite: https://github.com/vllm-project/vllm/tree/main/benchmarks
- MLPerf Inference (limited model coverage, but standardized)
- Nvidia's TensorRT-LLM published numbers
- Together AI / Fireworks / DeepInfra pricing pages (for perf-per-dollar
  external comparison)

## Reading list (background, once)

- "The Deep Learning Compiler: A Comprehensive Survey" (2020) — mental model
  for compiler autotuning
- vLLM's PagedAttention paper: https://arxiv.org/abs/2309.06180
- FlashAttention series papers (v1, v2, v3)
- Any recent MoE inference paper (DeepSeek-MoE's paper is good)

## What we're NOT consulting

Being explicit so we don't waste time:

- Nvidia CUDA-specific low-level docs (interesting but not portable)
- ONNX Runtime EPs (too generic, not perf-competitive for LLMs)
- Closed-source competitor benchmarks (can't verify, can't cite)
- Training-time optimization papers unless there's a clear inference transfer
