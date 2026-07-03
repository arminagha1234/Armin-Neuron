# Fine-grained op fallback — findings (pushing toward 6/6)

_Run 2026-07-02, trn2.48xlarge, Native PyTorch Beta 3 native_venv._

Goal: keep as much of the failing **trunk** on the NeuronCore as possible by
forcing only the offending op(s) to CPU, instead of the whole trunk.

## Mechanism discovered

The native `torch_neuronx` backend exposes a CPU-fallback path
(`cpu_fallback_registrations.py`). A user can override a specific aten op on the
neuron backend from Python:

```python
lib = torch.library.Library("aten", "IMPL")
lib.impl("<op>", cpu_fallback_fn, "PrivateUse1")   # routes <op> to CPU
```

**Confirmed this actually intercepts ops inside a black-box ScriptModule** —
e.g. `[FALLBACK INVOKED] aten::linear` fired during `trunk.forward_256`. So
fine-grained CPU fallback is a real, usable lever (script: `test_trunk_fallback.py`,
`OPS=comma,separated,ops`).

## Results

| Fallback set | Trunk result |
|---|---|
| `cat` | assert (culprit not cat) |
| `einsum` | assert |
| slice family (`chunk,split,slice_scatter,select_scatter,index_put,index_copy,narrow,as_strided_scatter`) | assert |
| `linear` (confirmed intercepted) | assert |
| `copy_` (confirmed intercepted) | assert |
| **broad set** (compute + cheap ops) | **PAST THE ABORT** — ran, no assertion |
| **cheap ops only** (layer_norm, reshape, sigmoid, clone, contiguous, permute, silu, copy_, chunk, cat, mul, add, relu, split... — **einsum/linear/matmul/bmm kept on Neuron**) | **PAST THE ABORT** — ran, no assertion |

### Two decisive conclusions
1. **The culprit is a *cheap* op (not einsum/linear/matmul/bmm).** When only the
   cheap shape/norm ops are offloaded and all the heavy compute stays on the
   NeuronCore, the trunk runs past the point it always aborted. So the FLOPs
   *can* stay on Neuron.
2. **No single cheap op we tried in isolation unblocks it** (cat, chunk-family,
   copy_ each still assert alone). The minimal unblocking set spans several
   cheap ops — consistent with the failure living in a *fused NEFF* that groups
   several cheap ops around the bad slice, rather than one standalone op.

## Practical caveat

The "cheap-ops-on-CPU, compute-on-Neuron" config is **functionally past the
bug** but **impractically slow**: every offloaded cheap op bounces the large
(1,256,256,256) pair tensor CPU↔Neuron over PCIe, and the trunk does this many
times per block × recycle. The run cleared the abort and kept executing but did
not finish quickly enough to be useful as-is.

Reducing the transfer thrash requires bisecting to the **single** cheap culprit
op (so only that one bounces), which is several more multi-minute runs. That's
the natural next step, and it also sharpens the bug report (names the exact op).

## Where this leaves 6/6

- We moved the boundary meaningfully: from "**entire trunk on CPU**" (previous
  hybrid) to "**all trunk matmul/einsum FLOPs on Neuron**, only a cheap op class
  on CPU" — strictly more compute on the accelerator.
- **True 6/6 (unmodified trunk fully on Neuron) still needs the runtime fix** —
  the fused-NEFF slice overflow (`tensor_set_slice`, 2^32 signature) is a Beta 3
  runtime bug. Fine-grained fallback is a *mitigation*, not a fix, and its
  transfer overhead means it's only worth productionizing once bisected to the
  single culprit op.

## Final confirmation: it's a fused NEFF, not a single op

Additional runs nailed this down:
- The abort fires **very early** (first substantial NEFF, within seconds of
  execution start — not deep in the block stack). `NEURON_HIERARCHY_DEBUG=1` +
  `TORCH_NEURONX_ENABLE_STACK_TRACE=1` produced no per-op hierarchy line before
  the abort, confirming it's the first compiled subgraph's execution.
- **Every single-op offload aborts** — tested individually: `cat`, `einsum`,
  `chunk`(+slice family), `linear`, `copy_`, `layer_norm`. Each one is confirmed
  intercepted (fallback fires) yet the trunk still asserts.
- **Only offloading a GROUP of ops clears it.**

Conclusion: the eager backend **fuses a group of ops into one NEFF**, and that
fused NEFF contains the `tensor_set_slice` overflow. Offloading a single op
doesn't move the fusion boundary enough to avoid it; you must offload enough of
the group that the bad fused kernel never forms. There is therefore **no clean
single-op fallback** — the minimal mitigation is offloading the cheap-op group,
which works functionally but thrashes on PCIe transfers (impractical as-is).

This is strong, specific evidence for the bug report: **a fused subgraph
(cheap shape/norm ops around the 256^3 pair tensor) overflows a 32-bit size
field in the runtime's tensor_set_slice path.**

## Recommendation

1. File the trunk repro (`trunk_repro/`) with the Neuron team — the fused-NEFF
   `tensor_set_slice` overflow is the real fix.
2. Ship the working **hybrid** (trunk on CPU, 5/6 incl. diffusion on Neuron) as
   today's deliverable — it matches CPU scores and is fast.
3. Optional follow-up: bisect the cheap-op set to the single culprit to (a)
   minimize transfer overhead in a fine-grained config and (b) name the exact op
   in the bug report.
