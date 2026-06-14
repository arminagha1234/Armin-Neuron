# GLM 5.1 — Native PyTorch (not the right shape)

**Status:** WIP / not applicable. See `../vllm-neuron/` for the production
serving path.

## Why no native PyTorch standalone path

GLM 5.1 is a 754B-parameter MoE model with 40B active params per token. A
pure native-PyTorch path (i.e. `torch.device("neuron")` + `torch.compile`)
makes sense for dense models that fit on a single core or a small TP group.
For a 754B MoE you need:

- **Tensor parallelism across many cores** (TP=32 minimum)
- **Expert parallelism** (256 experts spread across ranks)
- **KV cache management + paged attention** (continuous batching)
- **MoE token routing across ranks** (all-to-all dispatch)

vLLM-Neuron already provides all of this. Reimplementing it natively
would mean rebuilding most of vLLM's serving stack with no benefit to
the customer. The right path for GLM 5.1 on Trainium2 is `vllm-neuron/`.

## When you'd want a native path

If a customer wants to run GLM 5.1 as a research artifact (single-call
inference, no batching, no continuous serving), we'd reach for
`torch_neuronx.trace()` or NxDI. That's not what production users
typically need.

## Cross-reference

- Production serving: `../vllm-neuron/README.md`
- Engineering details: `../RESULTS.md`
