# Static-shape decode — one compiled graph for every token

How to make a decode step whose **graph shape does not depend on sequence position**,
so a single compiled graph serves every token instead of recompiling per step. Plus a
parity harness that proves the rewrite bit-exact against the reference.

The technique is general: it applies to any model whose reference implementation uses
`start_pos` in slice bounds.

---

## The problem

A typical PyTorch reference decode threads an integer `start_pos` and uses it to index:

```python
freqs_cis[start_pos : start_pos + seqlen]        # slice bound
kv_cache[:, start_pos % window] = kv            # integer index
should_compress = (start_pos + 1) % ratio == 0   # control flow
compressed_idx  = arange((start_pos + 1) // ratio)   # LENGTH grows with position
```

Two distinct problems fall out:

1. **Every decode step is a different graph.** Because position appears in shapes and
   control flow, step 129 traces differently from step 128 — so you recompile per token.
   The usual workaround is to pin `start_pos = 0`, which makes decode compile as a
   degenerate one-token *prefill*: it produces one graph, and the wrong answer.
2. **One shape is genuinely dynamic.** `arange((start_pos + 1) // ratio)` has a *length*
   that grows with position. Neuron rejects dynamic shapes outright.

## The fix

Pass `start_pos` as a **0-d tensor** so position is *data*, not *shape*:

| reference (dynamic) | static replacement |
|---|---|
| `freqs_cis[start_pos : start_pos+1]` | `index_select(freqs_cis, 0, pos)` |
| `kv_cache[:, start_pos % win] = kv` | `index_put_` with a group of index tensors |
| `(start_pos + 1) % ratio == 0` (**control flow**) | 0-d bool tensor; always compute, commit via `where(commit, new, current)` |
| `arange((pos+1) // ratio)` (**growing length**) | fixed-width row, tail filled with `-1` |
| growing cache slice + position-dependent `topk(k)` | full buffer, `-inf` on the unwritten tail, **fixed** `k`, overflow picks mapped to `-1` |

The `-1` trick is the key one and it is usually already available: masked-attention
kernels commonly treat a negative index as "skip", and reference implementations often
already use `-1` as a masked sentinel in their prefill branch. So a fixed-width row
padded with `-1` is *mathematically neutral* — you are not approximating anything.

Dispatch on **type**, not value:

```python
if not torch.is_tensor(start_pos):
    return original_forward(self, x, start_pos)   # int -> prefill, untouched
...                                               # tensor -> static decode
```

Branching on `torch.is_tensor` is a Python-level decision, so it does not cause
retracing, and it means the prefill path keeps running the original code verbatim — zero
regression risk on the path that already worked.

---

## Two XLA traps

Both cost real time, and both produce errors that point somewhere unhelpful.

**Do not mutate a slice view in place.**

```python
buf[:bsz].index_copy_(1, idx, val)
# RuntimeError: aten::as_strided ... no implementation for backend "xla:0".
#   View operators don't support since the tensor's storage cannot be shared across devices.
```

**Do not mix a slice with a 0-d tensor index in an assignment.** Taking the original
statement and swapping only the integer index for a tensor looks harmless and fails
during graph capture.

What works: build the write with a **group of index tensors** covering the leading dims
and `index_put_` on the **full** tensor; read with `index_select`. No views, no advanced
indexing, static shapes throughout.

**Bonus trap, unrelated to views.** A helper that ignores `start_pos` and always uses the
prefill formula will compute `arange(1 // ratio)` = `arange(0)` at decode — a **0-width**
tensor — and `unsqueeze(0).expand(...)` on a degenerate view is *also* an `as_strided`
lowering failure. If you see `as_strided` and your writes are already clean, look for a
zero-sized intermediate.

---

## Validation

Two harnesses, because "it compiles" is not "it is correct".

**Index builders** (`decode_static_idx.py`) — the static builders versus the verbatim
originals as oracle:

```
window indices   : 220 cases  bit-identical
compress indices : 138 cases  bit-identical (original's variable-length row -1 padded)
shape invariance : one distinct shape across 56 positions, for both builders
jit.trace vs eager: consistent at positions 5 / 300 / 1023 / 1024 / 2000
ALL PASS - 358 cases
```

Coverage includes the ring-fill phase, the exact `pos == W-1` boundary, and multiple ring
wraps. Shape invariance is the whole point — it is what collapses N graphs into one.

**Full model** (`test_decode_static_parity.py`) — build the model twice with identical
weights, run prefill plus N decode steps the original way (int `start_pos`) and the
patched way (0-d tensor), compare logits at every step:

```
prefill : exact
130 decode steps : bit-exact, worst max_abs_diff 0.000e+00
  including the step where two compression ratios commit simultaneously,
  and the step where the ring buffer wraps
buffer state after 40 steps : 0 of 13 buffers differ
```

The buffer check is what makes this convincing: if every KV/state buffer is
bit-identical, the position arithmetic and all writes are right.

### The part that nearly fooled me

A first run showed 128/130 steps exact with two steps differing by ~1e-4. The obvious
story was top-k tie-breaking. `diag_indexer_selection.py` **disproved** that — the two
paths select the identical index set on all 30 calls.

The real cause was **floating-point accumulation order**: the wider `-1`-padded index
tensor changes the summation order in the masked attention's softmax denominator and
output contraction. Proof: pad the *original* path to the same width (mathematically
neutral) and the difference vanishes entirely, in both a restrictive and a permissive
top-k configuration:

```
widths equalised, 130 steps -> ALL PASS, worst max_abs_diff 0.000e+00
```

Which also explains why only a *minority* of steps differed: the perturbation is far
below bf16 resolution (~0.008 near 1.0), so it only shows when a value sits on a rounding
boundary. Worth writing down as a general lesson — **if a rewrite changes tensor widths,
compare at equal widths before blaming your math.**

---

## Files

| file | what it does |
|---|---|
| `decode_static_idx.py` | static index builders + 358-case parity test. Run it directly. |
| `decode_static_patch.py` | the decode rewrite, applied as a monkey-patch (prefill delegates to the original) |
| `test_decode_static_parity.py` | full-model parity: prefill + N decode steps, original vs patched |
| `diag_indexer_selection.py` | localises any divergence to *selection* vs *writes* |

```bash
python decode_static_idx.py                            # index parity
STEPS=130 python test_decode_static_parity.py           # full-model parity
PAD_ORIG=1 STEPS=130 python test_decode_static_parity.py  # equal widths -> bit-exact
```

The tests need no accelerator and no real checkpoint — random weights are enough, because
what is being tested is that two code paths compute the same function.

Three harness details that will bite you if you rebuild this:

- The reference allocates parameters with `torch.empty` (normally filled by a weight
  loader), so an un-initialised build is garbage — in practice NaN. Initialise explicitly.
- A per-layer ratio list may be indexed by extra prediction blocks, so it needs
  `num_layers + num_extra_blocks` entries, not `num_layers`.
- Some reference helpers assert bf16 while a sibling code path casts to fp32 just before
  calling them. That trips in the *original* code too, so neutralise it with a shim applied
  to **both** runs, keeping the comparison apples-to-apples.

---

## Scope

This rewrite passes **one scalar position for the whole batch**, which is correct when
every sequence in the batch is at the same position (as in a fixed-length throughput
benchmark). General continuous batching with ragged positions needs a per-sequence
position vector — the same techniques apply, but the mask becomes
`slot_index >= position[b]` per sequence rather than a shared scalar. Do not read a
lockstep number as general serving support.
