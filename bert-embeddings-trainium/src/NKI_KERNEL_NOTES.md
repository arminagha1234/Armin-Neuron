# NKI fused-attention kernel — bring-up notes

**Status:** Compiles up to softmax broadcast. ~6 NKI ISA gotchas hit, partially fixed.
**Decision:** Stop here. Native+compile already delivers 12× over Path A on throughput and 3× on latency without NKI; expected NKI marginal gain (3–5% end-to-end) doesn't justify continued kernel iteration. Capturing all the gotchas for the next attempt.

## ISA gotchas hit in order (use these as a checklist for any future BERT-on-NKI attempt)

| # | Gotcha | Fix |
|---|---|---|
| 1 | `nl.softmax` is NOT a valid `nisa.activation` op | Implement softmax manually: max → subtract → exp → sum → reciprocal → multiply |
| 2 | `nisa.tensor_tensor` doesn't accept `op=nl.subtract` or `op=nl.divide` | Use `tensor_scalar(multiply -1)` + `add` for subtract; `activation(reciprocal)` + `tensor_tensor(multiply)` for divide |
| 3 | `nc_matmul` layout: `dst partition product (128) != stationary free product (32)` | Q@K^T: stationary needs `[K=D, M=S]`, moving needs `[K=D, N=S]`. Must transpose Q and K first to put D on partition dim. |
| 4 | `nc_transpose` signature: kwarg is `data`, NOT `src` | Use `nisa.nc_transpose(dst=..., data=...)` |
| 5 | Vector engine transpose limited to ≤[32, 32] | Use Tensor Engine transpose, which writes to PSUM, then `tensor_copy` to SBUF |
| 6 | Tensor Engine transpose dst dtype must equal input dtype on gen3+ | If input is bf16, allocate PSUM dst as bf16 (not float32) |
| 7 | `tensor_tensor` doesn't broadcast partition dim (`[1, S]` mask vs `[S, S]` scores) | Pre-broadcast on host: `mask.expand(S, S).contiguous()` before passing to kernel |
| 8 | `tensor_tensor` doesn't broadcast free dim either (`[S, 1]` row_max vs `[S, S]` scores) | Need to pre-broadcast row_max similarly, OR use `tensor_scalar` per row, OR find a fused softmax primitive |

## Where the kernel is right now

`native_run/native_nki_attention.py` — compiles through:
- ✅ DMA loads (Q, K, V, mask)
- ✅ Tensor Engine transpose Q, K → PSUM → SBUF
- ✅ nc_matmul Q @ K^T → scores in PSUM → SBUF
- ✅ `tensor_scalar(multiply scale)` works
- ✅ Mask add (with pre-broadcast `[S, S]`) works
- ❌ Softmax row-max subtract — needs the same broadcast trick we did for mask, or a different primitive

## What's likely needed to finish (for a future session, ~30–60 min)

1. Pre-broadcast `row_max` from `[S, 1]` to `[S, S]` (either in NKI via Tensor Engine matmul against an all-ones, or via separate per-row ops)
2. Same for `row_sum` / `inv_sum`
3. Validate shapes through to final `out_psum`
4. Correctness check vs HF reference (cosine ≥ 0.99)
5. Benchmark in the same harness (`bench_native.py` with `USE_NKI_ATTN=1 USE_COMPILE=0` first, then with `USE_COMPILE=1`)

## What we proved without finishing

- The NKI bridge (`torch_neuronx.nki_hop`) **does work** in the native+compile path. The kernel was successfully dispatched into the FX graph; the failures are inside the kernel body, not at the framework boundary.
- The vllm-neuron container is missing this bridge, so customers on that container can't fuse kernels at all today (already documented in `RESULTS.md`).
- For BERT inference workloads, native+compile is so fast (12× throughput, 3× latency over Path A) that NKI has limited remaining headroom.

## Scenarios where finishing this kernel would matter

- Long sequences (seq=512+) where the score matrix dominates — NKI's fused approach scales better than separate matmuls.
- Custom mask patterns the compiler can't auto-fuse.
- A model variant where attention is the actual bottleneck (we have no profiling data showing it is, for our 590→7,150→14,400 seq/s journey).
