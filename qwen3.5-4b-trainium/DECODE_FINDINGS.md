# Qwen3.5 decode: profiled. The cost is transposes, and the fix is compiler-blocked.

Follow-up to #98 (blocked triangular inverse, 4.77x prefill). After that landed,
decode became 77% of end-to-end, so it was profiled on device. Two candidate
fixes were implemented and measured; **neither should be shipped**. This document
exists so the next attempt starts from the evidence rather than repeating it.

## Device profile (neuron-profile, blocked-inverse build)

| metric | prefill | decode |
|---|---|---|
| total_time | 0.891 s | 0.0622 s |
| MFU | 13.06% | **1.32%** |
| MFU **max achievable** | 64.80% | **4.36%** |
| dynamic DMA active | 95.0% | **100.4%** |
| matmul arithmetic intensity | 142.4 | **9.6** |
| **transpose share of flops** | 14% | **86%** |
| DMA packets | 30.9 M | 10.0 M |
| avg DMA transfer | 297 KB | 51.8 KB |
| HBM read | 39.4 GB | 12.0 GB |
| matmul instructions | 2.35 M | 500 K |

The profile independently validates both host measurements: prefill device time
0.891 s against 0.906 s measured, decode 62.15 ms against 61.82 ms/token.

Three things stand out:

1. The compiler's own `mfu_inst_max_achievable` for the decode graph is **4.36%**.
   At 1.32% today, a *perfect* schedule of this graph is only ~3.3x. The graph
   structure is the limit, not the scheduler. Since decode is 77% of e2e, the
   entire decode lever is worth at most ~2.4x end to end.
2. **86% of decode FLOPs are transposes** (796 of 924 GFLOP).
3. 12.0 GB HBM read for a single token against ~2.2 GB of weights — 5.5x
   amplification, with arithmetic intensity 15x below prefill.

## Where the transposes come from

```python
kv_mem  = (new_state * k_t.unsqueeze(-1)).sum(dim=-2)
out_one = (new_state * q_t.unsqueeze(-1)).sum(dim=-2)
```

`new_state` is `[B, 32, head_k_dim=128, head_v_dim=128]` fp32 = 2 MiB with
`head_v_dim` on the free axis, so `dim=-2` is the **partition** axis. Reducing
along the partition axis requires a transpose on Neuron. This happens twice per
DeltaNet layer across 24 layers, each first materializing a 2 MiB intermediate.

## Fix B — drop the no-op transposes: NULL RESULT, do not ship

Decode has `seq_len == 1`, so the five `.transpose(1, 2).contiguous().float()`
calls followed by a later `[:, :, 0]` are a no-op permutation plus a forced copy
(`x.transpose(1,2)[:, :, 0] == x[:, 0]` when S == 1). Bitwise identical, so it
looked free.

Measured: **61.89 -> 61.94 ms/token**, i.e. noise. It compiled and produced a
different decode NEFF (`35af5851` vs `bbee3a43`), yet the profile is unchanged on
every metric — transpose share still 86%, MFU still 1.32%, HBM read still 12.0 GB.

The reason is simply size: those five tensors are `[B,32,128]` fp32 ≈ 16 KB each.
Removing five 16 KB copies cannot move a 62 ms graph. **Size the tensors before
assuming a copy matters** — in decode only the 2 MiB state is big.

## Fix A — reduce becomes a matmul: BLOCKED BY A COMPILER BUG

`sum_k A[k,v] * x[k]` is the matvec `x^T @ A`, so contracting k with a real
matmul removes both the 2 MiB intermediate and the partition-axis reduction.
Validated exact in numpy: cos 0.9999999, max diff 1.5e-05 (fp32 summation order).

It does not compile:

```
[INTERNAL_ERROR] [NCC_ITRF901] TritiumFusion assertion error: Unexpected remat axes
neuronx-cc compilation failed with 70
```

Tried in two forms, both failing at the same stage (hlo8):
- 4D batched `matmul`: `[B,H,1,128] @ [B,H,128,128]`
- 3D `bmm`: `[B*H,1,128] @ [B*H,128,128]`

So it is **not** the degenerate `M=1` axis, which was the initial hypothesis. The
compiler rejects feeding `new_state` into a matmul regardless of rank.
`new_state` is 2 MiB, is also consumed by the outer-product update, and is
written back to the KV cache; the rematerialization pass appears unable to handle
it across a matmul. neuronx-cc 2.31, nki 0.5.0. The error text asks for a support
ticket, which seems appropriate.

## The unblocked path

Store the recurrent state **transposed** as `[B, H, head_v_dim, head_k_dim]`.
Then `kv_mem[v] = sum_k state[v,k] * k_t[k]` reduces over the **last** (free)
axis — no transpose and no matmul, sidestepping NCC_ITRF901 entirely. The
outer-product update becomes `state[v,k] += delta[v] * k_t[k]`.

Cost: touches `_read_recurrent_state_from_cache`,
`_write_recurrent_state_to_cache`, and the prefill path that produces the state —
i.e. the prefill/decode state contract. It needs its own parity campaign.

Bear the 4.36% ceiling in mind before starting: the whole decode lever is ~2.4x
on e2e even if this works perfectly.

## Reproducing

Under `.tmp/indeed-bench/q35dec/` in the working environment:
`patch_fast_decode.py` (both fixes), `split_gates.py` (splits them for
bisecting), `add_bmm_gate.py` (the bmm variant), `test_decode_equiv.py` (numpy
equivalence), `prof_fixb.sh` (capture + view), `DECODE_CONCLUSION.md`.

`neuron-profile` usage that works, since the defaults do not:
```
neuron-profile capture -n <NEFF> -s out.ntff --profile-nth-exec=2
neuron-profile view -n <NEFF> -s <NTFF> --output-format summary-text
```
The default `--output-format db` needs InfluxDB, which is absent. Do not pass
`--output-file` with `summary-*`. Views take ~10 min for a 1.2 GB NTFF, exceed a
280 s exec window, and are CPU-bound so several can run in parallel.

Identifying graphs: `total_time` disambiguates them. This build compiles five
NEFFs — the four large ~45 MB ones are all prefill (0.891-0.893 s) and the single
small 7 MB one is decode (0.0622 s). Size is misleading.
