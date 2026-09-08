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
| best measured decode | **22.25 tok/s** @ batch 8 | **32.3 tok/s** @ batch 8 (fused HC-Sinkhorn NKI kernel: +9% over the same-build un-fused baseline; 5.0x over batch 1) |
| golden argmax | matches | matches |

**These two figures use different measurement conventions, so read them carefully.** The XLA
number is steady-state decode (prefill-excluded); the native number is aggregate over a
128-token generation. Decode on this model is weight-DMA-bound -- each step streams expert
weights out of HBM and does little math per token -- so batch size dominates throughput. The
one clean apples-to-apples is batch 1 vs batch 1: XLA measures **1.35 tok/s**, native **6.2**,
so the native path is well ahead at batch 1. The fuller picture: native decode now scales to a
**5.0x aggregate at batch 8** (30.84 tok/s un-fused; a fused HC-Sinkhorn NKI kernel lifts this ~9% to 32.3, golden matched), then hits an HBM ceiling --
batch 16 is *lower* (22.89 tok/s) and batch 32 does not fit (OOM) at 43 layers. The XLA path
keeps gaining out to **37.81 tok/s at batch 128** because its expert kernel already batches;
getting the native path past the batch-8 wall needs a batch-aware MoE kernel and/or weight
quantisation -- not more kernel fusion (see *Pushing decode throughput* below).

## Why bother with the native path at all

The XLA path scales further today (it keeps gaining out to batch 128, where the native path
is still HBM-capped at batch 8). The native path matters for what it makes possible
later, not for what it measures now:

- custom NKI kernels can be called directly instead of through an XLA lowering table
- the debug loop is minutes rather than hours -- no HLO dump, no rebuild
- data-dependent control flow traces under Dynamo, where XLA specialises per shape

None of that shows up in a tok/s number. It shows up in how quickly the next optimisation
can be attempted.

## Pushing decode throughput: what moved it, what didn't

Once the native path decoded correctly, the question was how much faster it could go at the
batch-8 ceiling. The answer, after a set of controlled A/B runs -- each golden-argmax
verified, each compared against a *same-build* baseline so stack drift can't masquerade as a
win:

**A fused HC-Sinkhorn NKI kernel: +9%** (29.59 -> 32.3 tok/s at batch 8, now on by default).
The hyper-connection boundary runs a Sinkhorn normalisation (two sigmoids, a row softmax,
~20 iteration steps) twice per layer -- 86 tiny-op sequences per decode step. Collapsing each
into a single kernel removes the launch/scheduling overhead that dominates at low batch.

To spend the rest of the effort well, a skip-block probe measured where the decode step goes:
**~74% attention, ~26% MoE.** Two more kernels were then written, simulator-validated, and run
on-device. Both were numerically exact; neither was faster:

| kernel | correct | throughput | why |
|---|---|---|---|
| fused HC-Sinkhorn | yes | **+9%** | many tiny ops, launch-overhead-bound -> fusion wins |
| dual-source attention (both score matmuls + shared softmax + both output matmuls, fused) | yes | -1% | those are batched GEMMs the compiler already lowers well; a hand kernel can't beat them |
| compressed-KV scatter write | yes | -2% | the existing full-buffer update is already memory-optimal |

The pattern is consistent: **NKI fusion pays off on op-overhead-bound work (many small ops),
not on compute-bound matmuls the compiler already schedules well.** A skip-probe on the
compressed-sparse-attention path confirmed it is not a bottleneck either -- removing it does
not speed decode up. So the fusion lever is essentially spent at +9%; the rest of the
attention is dense projection matmuls that are already efficient.

Reshaping parallelism didn't help either. Raising expert-parallel degree to 16 leaves zero
KV-cache budget (per-rank expert weight is EP-invariant, while the intermediate shard
doubles), and dropping it to 4 fails to build its collectives. Expert-parallel is not a free
throughput lever for this model on this hardware.

**Where the next real gain is:** *reducing* the matmul work rather than fusing it -- running
the attention and expert GEMMs in FP8/low precision. The model side of that is implemented; it
is currently gated by compiler support for the FP8 format on this hardware generation, and
should unlock on a newer compiler.

One native-path payoff showed up throughout: every kernel above was validated on a CPU
simulator in minutes before it ever compiled for the device, which is what made trying (and
rejecting) three kernels in a day practical.

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
