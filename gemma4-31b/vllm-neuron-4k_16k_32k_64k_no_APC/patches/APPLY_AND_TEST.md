# Apply & Test — Gemma4-31B GA v0.21 optimization patches

Two verified optimization patches for `vllm_neuron/model/gemma4/model.py` plus a
correctness harness. **Nothing here launches on-device by itself** — you (or a
later on-device run) apply the patch, relaunch the serve, and validate.

| File | Design | Effect | Gate |
|------|--------|--------|------|
| `patch_segmented_nki.py` | 08 | `_segmented_prefill_attention` torch-SDPA → `NF.flash_attention` w/ native prefix cache (`k_prior`/`v_prior`/`prior_used_len`) | none (kernel has torch fallback) |
| `patch_qkv_proj.py` | 02 | `forward_prefill` QKV matmul + QK-norm + RoPE → single `NF.qkv_proj` (BSD, pre-RoPE RMS QK-norm fused); V-norm stays torch | TP `world_size >= 8` else original torch path |
| `test_correctness.py` | — | torch reference vs patched-contract (+ optional on-device kernel check) | — |

Target file inside the container:
`/opt/conda/lib/python3.13/site-packages/vllm_neuron/model/gemma4/model.py`
Container: `vllm_ga` on `ec2-3-19-59-18.us-east-2.compute.amazonaws.com`.

All three scripts were **dry-run verified** against a copy of the live GA v0.21
`model.py` (2026-07-21): both patches apply cleanly, pass `ast.parse`, are
idempotent-safe (re-run aborts), and the harness reports `ALL CONTRACT CHECKS
PASS` on CPU.

---

## 0. Copy the patch files into the container

From your laptop (the `.pem` may need `xattr -c`; copy via `cat` if the keypair
dir is permission-blocked):

```bash
cp /Users/aghaebra/Downloads/test_kiro/keypair/arminagha.pem /tmp/arminagha.pem 2>/dev/null || \
  cat /Users/aghaebra/Downloads/test_kiro/keypair/arminagha.pem > /tmp/arminagha.pem
xattr -c /tmp/arminagha.pem 2>/dev/null; chmod 600 /tmp/arminagha.pem

HOST=ubuntu@ec2-3-19-59-18.us-east-2.compute.amazonaws.com
PDIR=/Users/aghaebra/Downloads/test_kiro/customers/Hippocratic/gemma4_31_example/optimizations/patches

for f in patch_segmented_nki.py patch_qkv_proj.py test_correctness.py; do
  cat "$PDIR/$f" | ssh -o StrictHostKeyChecking=no -i /tmp/arminagha.pem $HOST "cat > /tmp/$f"
  ssh -o StrictHostKeyChecking=no -i /tmp/arminagha.pem $HOST "sudo docker cp /tmp/$f vllm_ga:/tmp/$f"
done
```

Then `ssh -i /tmp/arminagha.pem $HOST` and work from the host, invoking the
container with `sudo docker exec vllm_ga ...`.

---

## 1. Apply the patches (inside the container)

Each script creates its own backup and refuses to double-apply.

```bash
# Design 08 — segmented prefill via NKI flash. Backup: model.py.pre_segnki
sudo docker exec vllm_ga python3 /tmp/patch_segmented_nki.py

# Design 02 — fused qkv_proj (prefill site only). Backup: model.py.pre_qkv
sudo docker exec vllm_ga python3 /tmp/patch_qkv_proj.py
```

Expected tail from each: `ast.parse OK` then `PATCH_OK bytes_before=… bytes_after=…`.

> Apply order does not matter — they edit disjoint regions
> (`_segmented_prefill_attention` vs `forward_prefill` Steps 1–4). Each keeps its
> own backup, so you can revert them independently.

**Recommended rollout: apply and validate `patch_segmented_nki.py` FIRST**
(it's the 32k/64k TTFT fix and has an automatic kernel→torch fallback, so it's
the lower-risk, higher-payoff change). Add `patch_qkv_proj.py` second.

Sanity-check the edits landed:

```bash
sudo docker exec vllm_ga bash -lc "grep -n 'NKI FLASH-ATTENTION SEGMENTED\|FUSED NKI QKV_PROJ\|world_size >= 8' /opt/conda/lib/python3.13/site-packages/vllm_neuron/model/gemma4/model.py"
```

---

## 2. Run the correctness harness

CPU-only contract check (no Neuron device touched — safe anytime):

```bash
sudo docker exec -e GEMMA4_TEST_USE_KERNEL=0 vllm_ga python3 /tmp/test_correctness.py
```

Expect per-head `cos_min=1.000000` and `allclose=True` for the SWA (hd256) and
global (hd512) cases, ending in `RESULT: ALL CONTRACT CHECKS PASS`.

**On-device kernel numerics** (bf16, exercises the real `NF.flash_attention` and
`NF.qkv_proj`). Only run when you own the device — it allocates a NeuronCore:

```bash
sudo docker exec -e GEMMA4_TEST_USE_KERNEL=1 -e NEURON_RT_VISIBLE_CORES=0 \
  vllm_ga python3 /tmp/test_correctness.py
```

Pass criteria for the KERNEL lines: per-head `cos_min >= 0.99` and
`allclose(atol=1e-2, rtol=1e-2) = True`. Validate SWA (hd256) first, then the
global (hd512) partial-RoPE case (highest silent-bug risk per design docs).

---

## 3. Relaunch the serve

The container runs `sleep infinity`; serves are launched manually. Use the same
TP32 segmented-prefill launch that the current baseline uses. **`patch_qkv_proj`
requires TP >= 8** — at TP32 the fused branch is active; below TP8 it silently
uses the original torch path.

```bash
# kill any running serve first
sudo docker exec vllm_ga bash -lc "pkill -f 'vllm serve' || true"; sleep 5

sudo docker exec -d vllm_ga bash -lc '
  export GEMMA4_DECODE_BACKEND=sdpa
  export GEMMA4_SWA_DECODE_BACKEND=sdpa
  cd /
  vllm serve /models/gemma-4-31b-it-ga \
    --served-model-name gemma4 \
    --tensor-parallel-size 32 \
    --max-model-len 65536 \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 4 \
    --no-enable-prefix-caching \
    --additional-config '"'"'{"neuron_config":{"num_batched_tokens_buckets":[4096],"num_seqs_buckets":[4],"on_device_sampling_config":{"all_greedy":true}}}'"'"' \
    > /tmp/serve_patched.log 2>&1
'

# watch compile + startup (Neuron compile can take many minutes on first launch)
sudo docker exec vllm_ga bash -lc "tail -f /tmp/serve_patched.log" | grep -m1 "Application startup complete"
```

For a 64k-context TTFT test, relaunch with `--max-num-batched-tokens 65536`
(and the matching `num_batched_tokens_buckets`) — that is the case design 08
targets (24s→~1-2s at 32k, 47s→~2-4s at 64k, hypothesized).

---

## 4. Benchmark

`~/ga_bench.py` on the host measures TTFT via the chat template at 4k input,
concurrency 1/2/4 (baseline: 4k conc1 = 0.723s beta / compare to GA).

```bash
# host venv
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate 2>/dev/null || true
python3 ~/ga_bench.py
```

For the 32k / 64k win, edit `ga_bench.py`'s `for ctx in (4096,):` to
`for ctx in (32768, 65536):` (and relaunch the serve with a matching
`--max-num-batched-tokens`). Compare TTFT before vs after design 08.

---

## 5. REVERT

Each patch left a timestamped backup. Restore, then relaunch the serve (§3).

```bash
# revert design 02 (qkv_proj)
sudo docker exec vllm_ga bash -lc '
  P=/opt/conda/lib/python3.13/site-packages/vllm_neuron/model/gemma4/model.py
  [ -f "$P.pre_qkv" ] && cp "$P.pre_qkv" "$P" && echo "reverted qkv_proj" || echo "no .pre_qkv backup"
'

# revert design 08 (segmented nki)
sudo docker exec vllm_ga bash -lc '
  P=/opt/conda/lib/python3.13/site-packages/vllm_neuron/model/gemma4/model.py
  [ -f "$P.pre_segnki" ] && cp "$P.pre_segnki" "$P" && echo "reverted seg_nki" || echo "no .pre_segnki backup"
'
```

> If BOTH patches were applied, revert order matters only in that each backup is
> a snapshot from *before that specific patch ran*. Since they were applied
> seg-first-then-qkv (or independently), the safest full revert is:
> restore `.pre_segnki` if it exists **and** it predates the qkv edit — otherwise
> restore `.pre_qkv` first (undoes qkv), then re-derive. In practice the clean
> path is: **to fully revert, restore whichever backup was created FIRST**
> (`stat` the two `.pre_*` files; the older mtime is the pristine baseline). Then
> re-apply only the patches you still want.

Recompile is automatic on next serve launch (Neuron traces the patched graph).

---

## Notes / risks (carried from the design docs)

- **Design 08**: at 64k the torch-SDPA fp32 score tensor (~67GB) dominates, so
  the NKI flash path should win materially. `NF.flash_attention` auto-falls-back
  to torch if kernel constraints fail, so correctness is preserved. Confirm on
  device that `prior_used_len` (dynamic) is honored and not collapsed to static
  under `wrap_nki` — check attention output vs the FP32 oracle (§2 kernel check).
- **Design 02**: only the **prefill** QKV site is patched. Three more sites
  exist (`forward_decode` ~model.py:642, and two more at ~:791 / ~:939); see the
  `OTHER SITES` block at the bottom of `patch_qkv_proj.py`. Extend them one at a
  time after prefill is validated. Highest silent-bug risk is the global-layer
  partial-RoPE `rotate_half` convention — the harness validates it, but re-check
  the KERNEL cosine on the hd512 case specifically.
- **TP gate**: `patch_qkv_proj` needs TP>=8. At TP4 global `fused_qkv_dim=5120`
  trips the kernel's >4096 gate, so the patch keeps the torch fallback there.
- These are runtime-file edits inside the container, not a package rebuild. They
  are lost if the container is recreated — reapply from `/tmp` (§0–1).
```
