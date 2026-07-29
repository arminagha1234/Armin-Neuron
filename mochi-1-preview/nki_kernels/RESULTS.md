# Mochi-1 on Trainium2 — NKI kernel results

On-device results from the trn2.48xlarge (`i-0d040fd97f694dd63`), Beta-3 DLC
(torch 2.11, torch_neuronx 2.11.3, neuronx-cc 2.26, **nki 0.5.0**), TP=4 eager.

## Headline: flash-attention NKI kernel speeds up the real forward, more at scale

End-to-end denoise s/step (warm, warmup steps excluded), measured by
`ab_bench.py` driving the actual transformer forward:

| Config | eager baseline | + flash attention | speedup |
|---|---:|---:|---:|
| 19f, no-CFG (S=6616) | 2.611 s/step | 2.253 s/step | **1.16×** |
| 31f, CFG (S=9796, batch 2) | 7.183 s/step | **5.375 s/step** | **1.34×** |
| 61f, no-CFG (S=17,746) | 9.868 s/step | 11.273 s/step | **0.88× (slower)** |

**The speedup is NOT monotonic — it peaks around 31f and reverses by 61f.**
This contradicts the earlier crossover prediction (that longer sequences would
widen the win), and the on-device data overrules it. Two reasons the prediction
was wrong:

1. The port's `_attention_bmm` **already tiles the query axis** (auto q_chunk,
   256 MiB budget), so bmm never actually pays the O(S²)-HBM-spill penalty the
   crossover argument assumed. bmm degrades gracefully at long S.
2. The flash kernel's 31f-winning choices turn against it at 61f: per-plane K/V
   residency (~4.5 MB each per plane at 17.7k keys) strains the SBUF working
   set, and the single-pass full-row softmax over 17,746 keys grows costly
   relative to bmm's tiled softmax.

**Practical guidance: enable `MOCHI_NKI_ATTN=1` for ≤~31-frame clips (up to
1.34× faster); leave it off (bmm) for 61f+.** The kernel's sweet spot is
moderate joint-sequence lengths. A long-sequence-tuned variant (larger k-tiling,
2-pass blocked softmax, K/V streaming instead of full residency) would be needed
to win at 61f+ — logged as future work, not claimed.

### Microbenchmark (attention op alone, `bench_flash_vs_bmm.py`)

| Shape | PyTorch `_attention_bmm` | flash NKI (optimized) | speedup |
|---|---:|---:|---:|
| S=6616, 6 planes | 18.7 ms | 11.5 ms | 1.63× |
| S=9796, 6 planes | 39.9 ms | 23.7 ms | 1.68× |

The op-level 1.63–1.68× becomes 1.16–1.34× end-to-end (Amdahl: attention is
only part of each 48-block step; QKV/FFN/norm/RoPE are not yet accelerated).

## Biggest end-to-end win: rank-0-only VAE decode (2.1× total wall-clock)

Not a kernel — the largest single speedup of the whole effort. The CPU VAE
decode is ~40% of wall clock AND was being run redundantly by all 4 TP ranks,
which fight over the same 192 vCPUs. Gating it to rank 0 (non-zero ranks stop
at `output_type="latent"`; they still run the full collective denoise loop, then
wait at the barrier) frees the machine for the one decode that matters.

| 31f, 4 steps, TP=4 | Total wall-clock |
|---|---:|
| all ranks decode (old) | 321.5 s |
| **rank-0-only decode (fix)** | **151.6 s** |

**2.1× faster end-to-end**, output still correct (rank 0 writes all 31 frames).
This is `IMPROVEMENTS.md` §1.3, now verified on-device. Toggle
`MOCHI_FORCE_ALL_DECODE=1` restores the old behavior for A/B.

## How the flash kernel got fast

The first correct version was **2.3× slower** than bmm (43.9 ms). Four coupled
structural fixes turned it into a 1.63× win — all about instruction mix, not
FLOPs (the shape is compute-bound but tiny in absolute FLOPs, so it lives in an
overhead regime):

1. **K/V loaded + transposed once per plane** and kept resident in SBUF,
   instead of reloading + re-`nc_transpose`ing for every one of ~52 q-tiles.
2. **Per-key bias broadcast once per plane** across the 128 query partitions,
   removing ~10.8k redundant `nc_stream_shuffle` broadcasts.
3. **MM1 (QKᵀ) moving operand widened 128 → 512**, running the tensor engine
   near its 128×512 limit instead of wasting ~75% of it.
4. **Single-pass full-row softmax** (the whole Sk row fits in SBUF at every
   Mochi frame count) replacing the online-softmax rescale loop — numerically
   identical to the materialised reference, no per-tile correction traffic.

## Kernel validation status (on-device, vs CPU reference)

| Kernel | On-device | Notes |
|---|---|---|
| **flash attention** | ✅ PASS, wired, A/B'd | 5/5 cases, cosine ≥0.99999; 1.16–1.34× e2e |
| **rmsnorm** | ✅ PASS | rel ~2e-3; validated standalone, not yet wired |
| **swiglu** | ✅ PASS | rel 3.7e-3; validated standalone, not yet wired |
| **fused_qkv** | ✅ PASS | rel 3.6e-3; validated standalone, not yet wired |
| rope | ❌ compile error | `.ap()` strided-view lowering fails; lowest value (memory-bound; torch.compile likely subsumes) — deferred |

**Why only attention is wired:** attention is the one op `torch.compile`
provably cannot produce (stock SDPA miscomputes on the Neuron bf16 backend —
the reason the port hand-rolled the BMM shim). The compile research showed the
compiler subsumes QKV/norm/RoPE fusion, and those ops are small/memory-bound, so
their standalone speedup is unlikely to beat the compiler while wiring them
risks the working forward. They are validated and staged behind
`install_nki_kernels.py` flags for a future A/B, but not yet in the hot path.

## torch.compile (characterized, not a throughput lever)

Per `research/TORCH_COMPILE_ON_NEURON.md` and prior run reports
(`results/report_*_compiled.json`): compile's win on Mochi is **memory**
(whole-graph buffer assignment took the frame envelope 31→61), not speed —
warm per-step is ~8 s/step ≈ near-parity with eager. Use it to fit bigger
clips, not to go faster.

## How to reproduce

```bash
# inside the persistent `mochi` container, one device process at a time
# eager baseline
bash /host/run_ab31.sh --tag eager31
# flash attention
MOCHI_NKI_ATTN=1 bash /host/run_ab31.sh --tag attn31
# results land in results/ab_bench.jsonl
```

The flash kernel is wired via `nki_kernels/install_nki_kernels.py`
(`MOCHI_NKI_ATTN=1`), which swaps `neuron_compat._attention_bmm` for the kernel
**only for the D≤128 joint attention** — the `MochiAttentionPool` (head_dim 512)
correctly falls back to bmm.
