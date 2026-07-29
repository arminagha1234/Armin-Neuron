# Mochi-1 Trainium port — improvement ideas

Observations from reading the port source (`src/`) against the README/NOTES
claims. Grouped by category and ranked within each by expected payoff. Nothing
here is a correctness bug in the validated eager path — these are performance,
robustness, and code-clarity opportunities. Items the README already
acknowledges are marked *(known)*.

Status legend: **[perf]** speed/memory, **[correct]** latent numeric risk,
**[robust]** failure-mode/UX, **[clarity]** readability/maintenance.

### Implemented so far (offline suite: 73 passed, 0 failed)

| Item | Change | Verified |
|---|---|---|
| §1.1 | fp32 softmax in `_attention_bmm` (both branches) | new test `[10]`: bf16 path tracks fp32 oracle to 1.6e-3, masked keys don't leak |
| §1.3 | rank-0-only CPU VAE decode (non-zero ranks stop at `output_type="latent"`) | runner compiles + `--help` off-device |
| §2.3 | `strict=True` now errors on unexpected checkpoint keys | new test `[11]`: raises on ghost key, `strict=False` still warns+loads |
| §3.1 | dropped the dead `batch*heads != batch` clause in `_collapse_mask` | existing mask tests still pass |

Remaining items below are not yet applied (need a device measurement or are
lower priority).

---

## 1. Performance

### 1.1 Softmax runs in bf16 in the tiled attention path — **[correct/perf]**
`neuron_compat._attention_bmm`:
```python
scores = torch.bmm(query, key_t) * scale     # bf16 in, bf16 out
scores = scores + attn_mask                    # bf16
return torch.bmm(scores.softmax(dim=-1), value)
```
`scores` is bf16 (product of bf16 Q·Kᵀ), so `scores.softmax(dim=-1)` reduces
over up to ~9,796 keys in bf16. The `-10000.0` masked bias plus bf16's 8-bit
mantissa means the exp/normalize accumulates error across thousands of terms.
The NKI flash reference (`nki_kernels/attention/flash_attn_ref.py`) deliberately
does the online softmax in **fp32** — so the reference and the current
production path may not agree to the tolerances the tests assume.

*Fix:* upcast scores to fp32 before softmax and cast the probabilities back to
bf16 before the P·V matmul: `scores.float().softmax(-1).to(value.dtype)`. Costs
one fp32 tile per q-chunk (already the norm-tiling trade-off), and makes the
eager path match the fp32-softmax flash kernel. This is the single most likely
source of a future "kernel doesn't match eager" surprise.

### 1.2 Fuse the six separate QKV projections — **[perf]**
`mochi_neuron_attention.__call__` issues six independent GEMMs per block:
```python
query = attn.to_q(hidden_states)...      # 3072->3072
key   = attn.to_k(hidden_states)...      # 3072->3072
value = attn.to_v(hidden_states)...      # 3072->3072
enc_query = attn.add_q_proj(enc)...      # 1536->3072
enc_key   = attn.add_k_proj(enc)...      # 1536->3072
enc_value = attn.add_v_proj(enc)...      # 1536->3072
```
The three visual projections share one input (`hidden_states`) and the three
text projections share another (`encoder_hidden_states`). They can be fused into
**two** GEMMs (a `[3072→9216]` visual QKV and a `[1536→9216]` text QKV) by
concatenating weights at load time. Six small dispatches → two larger ones cuts
kernel-launch overhead and improves tensor-engine utilization. This is a
load-time weight-packing change in `mochi_meta_loader.py` (concat the column
shards) — the column-shard rule already applies to all six identically, so the
fused weight shards the same way. Worth it once `torch.compile` is measured;
compile may already do this.

### 1.3 Gate the CPU VAE decode to rank 0 — **[perf]** *(known)*
README §"Problems actually hit on device" already flags this: every one of the 4
TP ranks redundantly runs the full CPU VAE decode (~40% of wall clock). Only
rank 0 writes output. Gating decode to rank 0 (and broadcasting nothing, since
other ranks discard it) frees 3/4 of the CPU for rank 0's decode. Combined with
the `OMP_NUM_THREADS` fix already documented, this is the largest end-to-end
win that is *not* a device kernel. The transformer is the only collective stage,
so the pipeline can legitimately run decode on rank 0 alone.

### 1.4 `torch.compile` baseline is untried — **[perf]** *(known)*
The design doc (`nki_kernels/design/KERNEL_PLAN.md`) makes this the #0 action:
every latency number in the README is eager, so 6.3 s/step is an upper bound.
Compile likely absorbs the RMSNorm/RoPE/QKV-fusion wins (#1.2, and the norm and
rope kernels) outright. Measure before building kernels #2–5.

### 1.5 RoPE cos/sin tables rebuilt as fp32, crossed into a bf16 graph — **[perf]**
`patch_rope_cpu_precompute` caches fp32 tables and (unless `--rope-bf16`) hands
fp32 into the compiled transformer. `apply_rotary_emb` then upcasts Q/K to fp32
per block anyway (`x[...].float()`). Since RoPE precision was verified fine and
`--rope-bf16` was *not* needed (README), emitting bf16 tables by default would
shrink the constant tensors crossing the compile boundary and the per-block
upcast. Low priority (RoPE is cheap), but free once compile is on.

---

## 2. Robustness / failure modes

### 2.1 `--q-chunk` and `--norm-tile` silently invalidate the NEFF cache — **[robust]**
Both `set_attention_chunking` and the norm tile change the compiled graph shape.
The code comments say so, but there is no runtime warning when a user changes
them between `--compile` runs — they'll just eat a silent 10–30 min recompile
and may not know why. A one-line log at startup ("q_chunk=X, norm_tile=Y →
these define the NEFF cache key") would save real time on device.

### 2.2 Fully-masked-tile guard depends on the `-10000.0` sentinel, not `-inf` — **[robust]**
`build_joint_attention_bias` masks padded text keys with `-10000.0`; visual keys
are never masked, so no softmax row is ever fully masked and there's no NaN — the
NOTES reasoning is sound. But this invariant ("visual keys always present")
is load-bearing and implicit. If a future change ever masks visual keys (e.g. a
windowed/sparse attention experiment), a fully-masked row would silently produce
a uniform distribution over `-10000` biases rather than a NaN you'd notice.
Worth an assertion or comment at the `build_joint_attention_bias` call site
naming the invariant.

### 2.3 `strict=True` meta-check runs but `unexpected` keys only warn — **[robust]**
`load_weights_sharded` raises on leftover meta params (good) but merely prints a
warning for checkpoint keys absent from the model (`unexpected`). For a
first-time real-weight load (NOTES: "the loader has never read real Mochi
weights"), a silent mismatch here is exactly the kind of thing that produces
structurally-correct-but-wrong video. Consider promoting `unexpected` to an
error under `strict=True`, or at least asserting the expected loaded-tensor
count (the README knows it should be 1071).

---

## 3. Code clarity / minor

### 3.1 Dead sub-condition in `_collapse_mask` — **[clarity]**
```python
if attn_mask.shape[0] == batch and heads > 1 and batch * heads != batch:
```
`batch * heads != batch` is equivalent to `heads != 1`, which is already implied
by `heads > 1`. The third clause is always true when the second is, so it's
redundant. Harmless, but confusing to read. Drop it.

### 3.2 `_attention_bmm` recomputes `key_t` outside the tile loop but keeps full K resident — **[clarity/perf]**
`key_t = key.transpose(-1, -2)` is a view, so it's cheap, but the tiled path
still holds the full K/V for every q-chunk. This is inherent to the "each tile
attends to all keys" design (correct and intentional per NOTES), so it's not a
bug — but it's precisely the O(q_chunk·Sk) that the flash NKI kernel is meant to
replace. Worth a comment pointing at `nki_kernels/attention/` as the successor,
so the two don't drift.

### 3.3 `zero_padded_context` multiply is always on — **[clarity]**
`MochiNeuronAttnProcessor(zero_padded_context=True)` is the only call site
(`install_neuron_attn_processor`). NOTES explains it's for CPU-reference
comparability during debugging and is numerically inert to the video. Since no
accuracy-vs-CPU comparison has actually been run yet (README "Honest gaps"),
this multiply currently costs one elementwise per block for a debugging feature
nobody is using. Consider defaulting it off until the CPU-reference work
(§ Honest gaps) actually happens.

---

## 4. Cross-cutting: reconcile the NKI API version before the device session

Not a `src/` issue, but it blocks the kernel work: the three kernels the builder
agents produced split across two NKI APIs — `flash_attn_nki.py` and
`swiglu_nki.py` target the newer `import nki` 0.3.0 style (no `nl.mgrid`), while
`rmsnorm_nki.py` uses the classic `neuronxcc.nki` + `nl.load/store/mgrid` style
that matches the repo's `NKI_template.py`. Confirm which the Beta-3 DLC ships
and unify before compiling, or one of them will fail to build. See the per-agent
notes; this is the first thing to settle on-device.

---

## Suggested order

1. **§1.1 fp32 softmax** — cheap, removes a latent correctness/kernel-mismatch trap. Do first.
2. **§1.4 torch.compile baseline** — resets every number; likely absorbs several other items.
3. **§1.3 rank-0 VAE decode** — biggest non-kernel wall-clock win.
4. **§2.3 strict loader** — before the first real-weight run.
5. **§1.2 fused QKV** — only if compile didn't already do it.
6. Clarity items (§3) whenever the files are next touched.
