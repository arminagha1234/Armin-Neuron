# Single-Stream Block Sharding — the path to 3MP/4MP fp32

## Why this is THE next step

The v2 TP plan shards only the 5 double-stream blocks' attention. The
**20 single-stream blocks (the bulk of the 4B params) and ALL FFNs run
fully replicated** on every core. In fp32 that replicated activation is
the OOM driver at ≥3MP (the failed `NRT alloc 843MB` is a replicated
fp32 FFN/MLP intermediate).

If the single-stream blocks + FFNs shard correctly, the per-core fp32
activation drops ~world_size×, which is what lets fp32 (the only correct
recipe) fit at 3MP and 4MP. fp32 already gives std 16.99 at 1.6MP — the
ONLY thing missing for higher res is fitting it.

## The two fused projections to split

### 1. Double-stream FFN — `Flux2FeedForward` (5 blocks × 2 = ff + ff_context)
```
linear_in : Linear(dim=3072 -> inner*2 = 2*9216 = 18432)   # [gate ; value]
act_fn    : Flux2SwiGLU  ->  x1,x2 = x.chunk(2); silu(x1)*x2
linear_out: Linear(inner=9216 -> dim=3072)
```
**SwiGLU scramble bug**: ColwiseParallel on the fused `linear_in` splits
[gate;value] uniformly, so each rank's `chunk(2)` pairs wrong halves
(this is the proven std=45 bug).

**Fix (Llama-style, known-correct):** replace the fused `linear_in` with
TWO separate Linears:
```
gate_proj  = Linear(dim -> inner)   # weight = linear_in.weight[:inner]
value_proj = Linear(dim -> inner)   # weight = linear_in.weight[inner:]
```
Rewire `Flux2FeedForward.forward` to:
```
g = self.gate_proj(x); v = self.value_proj(x)
x = self.act_fn_silu(g) * v          # no chunk — already separated
x = self.linear_out(x)
```
TP plan: `gate_proj` + `value_proj` = ColwiseParallel,
`linear_out` = RowwiseParallel. Now each rank holds inner/N of gate AND
value (same column range) → correct, and the inner activation shards N×.

### 2. Single-stream — `Flux2ParallelSelfAttention` (20 blocks)
```
to_qkv_mlp_proj: Linear(dim -> 3*inner + mlp_hidden*mlp_mult)
                 split -> qkv (3*inner=9216), mlp (mlp_hidden*2 SwiGLU)
mlp_act_fn     : Flux2SwiGLU on the mlp half
to_out         : Linear(inner + mlp_hidden -> dim)
```
Split `to_qkv_mlp_proj` into FIVE shardable Colwise linears:
```
to_q, to_k, to_v   : Linear(dim -> inner)      from weight rows [0:inner],[inner:2inner],[2inner:3inner]
mlp_gate, mlp_value: Linear(dim -> mlp_hidden)  from the two SwiGLU halves of the mlp slice
```
Split `to_out` (RowwiseParallel) input dim accordingly: the concatenated
[attn_out(inner) ; mlp(mlp_hidden)] — `to_out` is Rowwise on its input
dim, so it shards naturally once the upstream is sharded, BUT the concat
of two independently-sharded tensors needs care. Follow the NxDI pattern:
**separate proj_out for attn and mlp**, each RowwiseParallel(reduce=False),
summed, then ONE all-reduce:
```
attn_out_proj = Linear(inner -> dim)     # Rowwise, reduce_output=False
mlp_out_proj  = Linear(mlp_hidden -> dim)# Rowwise, reduce_output=False
out = reduce_from_tp(attn_out_proj(attn) + mlp_out_proj(mlp))
```
Rewire `Flux2ParallelSelfAttnProcessor.__call__` to use the split
linears + separate out-projs. Patch `attn.heads = heads//N` on these
blocks too (apply_tp_fixes must cover single-stream now).

## Reference implementation (copy from)
`neuron/external/pr-117-nxdi-diffusion-models/src/neuronx_distributed_inference/models/diffusers/flux/modeling_flux.py`
- separate `proj_out_attn` + `proj_out_mlp`, both RowParallel(reduce_output=False), summed → one `reduce_from_tensor_model_parallel_region`
- `attention_wrapper_sharded_without_swap()` (line ~82) for the Q/K/V layout

## Weight-splitting helper (load-time surgery)
```python
def split_fused_linear(fused: nn.Linear, sizes: list[int], bias=False):
    """Split a fused Linear's out_features into separate Linears."""
    outs, off = [], 0
    for s in sizes:
        lin = nn.Linear(fused.in_features, s, bias=bias)
        lin.weight.data = fused.weight.data[off:off+s].clone()
        if bias and fused.bias is not None:
            lin.bias.data = fused.bias.data[off:off+s].clone()
        outs.append(lin); off += s
    return outs
```
Do this on CPU (meta-safe) BEFORE `.to(device)` and before
`parallelize_module`. Register the new submodules on the block, rewire
`forward`, then the TP plan keys reference the new submodule paths.

## Validation ladder (each ~6-10 min compile)
1. **1280² TP=2 fp32 with v3 plan** — MUST still give std ~17 (correctness
   regression check). Previously TP=2 fp32 @1280² OOM'd; if v3 sharding
   now makes it FIT at TP=2, that alone proves the memory win.
2. **1792² (3MP) TP=4 fp32 v3** — the real target. Should now fit (FFN +
   single-stream sharded) and give std ~17.
3. **2048² (4MP) TP=8 fp32 v3** — the goal. If it fits + std ~17, done.

## Risk / gotchas
- The SwiGLU split must preserve exact weight rows (gate = first inner,
  value = second inner). Verify with a CPU forward-equivalence check
  (split-FFN output == original FFN output, atol 1e-4) BEFORE sharding.
- The single-stream `to_out` separate-proj sum must reduce ONCE (not
  per-proj) to avoid double all-reduce.
- apply_tp_fixes must now patch `attn.heads` on single-stream blocks too.
- Keep fp32-residual + fp32-norms installed (they're orthogonal and
  needed for correctness; v3 only changes WHERE compute shards).

## Expected outcome
fp32 (correct) + full sharding (fits) = correct output at 3MP and 4MP.
This is the combination that resolves the memory×precision tension the
mixed-precision ladder could not.
