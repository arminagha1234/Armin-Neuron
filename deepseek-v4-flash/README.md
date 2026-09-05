# DeepSeek-V4-Flash on Trainium2

Getting **DeepSeek-V4-Flash** (284B MoE, 43 layers, 256 routed experts, top-6, MLA with
compressed sparse attention) to decode on a single `trn2.48xlarge`, down two different
compilation paths, and measuring both honestly.

Everything here is self-measured. Where a number is not comparable to another number,
that is stated rather than glossed.

---

## The two paths

|  | [`xla/`](xla/) | [`native-pytorch/`](native-pytorch/) |
|---|---|---|
| how the graph reaches the hardware | `torch.compile` -> torch-xla -> HLO -> `neuronx-cc` | `torch.compile` -> torch-mlir -> StableHLO -> `neuronx-cc`, **no XLA** |
| status | working, tuned | **working**, un-tuned |
| best measured decode | **22.25 tok/s** @ batch 8 | **~8 tok/s +/- 1** @ batch 1 |
| golden argmax | matches | matches |

**The two throughput numbers are not comparable.** One is batch 8, the other batch 1, and
decode on this model is weight-DMA-bound: each step streams expert weights out of HBM and
does very little math per token, so batching amortises a fixed cost and moves throughput by
more than an order of magnitude. On the XLA path the same model measures **1.35 tok/s at
batch 1** and 37.81 at batch 128. A matched-batch native measurement is in progress; until
it exists, no claim is made about which path is faster.

## Why bother with the native path at all

The XLA path works and is faster today. The native path matters for what it makes possible
later, not for what it measures now:

- custom NKI kernels can be called directly instead of through an XLA lowering table
- the debug loop is minutes rather than hours -- no HLO dump, no rebuild
- data-dependent control flow traces under Dynamo, where XLA specialises per shape

None of that shows up in a tok/s number. It shows up in how quickly the next optimisation
can be attempted.

## What each folder contains

**[`xla/`](xla/)** -- the working baseline. Full 43-layer batched decode at 22.25 tok/s,
TP=32, batch 8, golden argmax matched. Four ceilings had to be cleared to get there, each
written up with the error it produces, because all four look like bugs in your own code.
Plus two supporting investigations: [`decode-static-shapes/`](xla/decode-static-shapes/)
(making decode shapes static, and two XLA traps) and
[`fp4-expert-gemm/`](xla/fp4-expert-gemm/) (why FP4 expert weights must be dequantised on
this hardware generation).

**[`native-pytorch/`](native-pytorch/)** -- the native path, and the 15 blockers between a
model that imports and a model that decodes. Includes
[`path-analysis/`](native-pytorch/path-analysis/) -- the full blocker writeup, including the
one that took days and turned out to be a single missing configuration flag -- and
[`hyper-connection-fusion/`](native-pytorch/hyper-connection-fusion/), seven NKI fusion
increments for the hyper-connection boundary, validated against float64 references.

## The model, briefly

```
43 decoder layers, hidden 4096, 64 attention heads, head_dim 512 (MLA, no absorption)
256 routed experts, top-6, moe_intermediate_size 2048, 1 shared expert
hyper-connections: hc_mult=4, 20 Sinkhorn iterations per boundary
compressed sparse attention: per-layer compress_ratios [0, 0, 4, 128, 4, 128, ...]
first 3 layers route by a token-id -> expert table instead of top-k
sliding window 128
```

Two properties dominate every engineering decision here. Decode is **weight-DMA-bound**, so
throughput is a memory-bandwidth story rather than a FLOPs story. And the per-layer
structure is **heterogeneous** -- a 4-layer slice covering `compress_ratios` `[0, 0, 4, 128]`
exercises every distinct layer type in the network, which makes a fast compile probe possible.
