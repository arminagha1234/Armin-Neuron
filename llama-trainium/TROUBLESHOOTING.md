# Troubleshooting — Llama on Trainium (native PyTorch)

Every error in this list was actually hit while building these examples, with the
fix that resolved it. Symptoms are quoted from real logs.

---

## 1. Tensor-parallel run hangs forever on the collective

```
TDRV:exec_request_check_proxy_done_state  [nec_dev 0 gid 0]
  Timeout exceeded: Waiting on barrier proxy task: 120 sec
```
often preceded by:
```
CCOM WARN NET/OFI aws-ofi-nccl initialization failed
CCOM WARN OFI plugin initNet() failed is EFA enabled?
```

**Fix: set `TORCH_NEURONX_ENABLE_HOST_CC=1`.**

```bash
NEURON_RT_NUM_CORES=2 TORCH_NEURONX_ENABLE_HOST_CC=1 TORCH_NEURONX_ENABLE_ASYNC_NRT=1 \
  torchrun --nnodes 1 --nproc_per_node=2 --rdzv_backend c10d --rdzv_endpoint localhost:29500 \
  tp_validate.py ...
```

Without host collective communication the all-reduce tries the OFI/EFA **device**
path, which can't initialize inside the container, and blocks on the barrier
forever. With host CC the same forward completes in under a second.

**The OFI/EFA warning itself is a red herring** — it also appears in runs that
work fine. Don't chase it.

---

## 2. `NRT EXECUTION FAILED: Failed to allocate resource` / OOM

```
Neuron OOM: requested_size=... total_hbm=17179869184
NRT EXECUTION FAILED: lazy::AllocBind: NRT_RESOURCE
RuntimeError: NRT Execution error ... operation=aten::cat
```

The model's weights filled the core's HBM and there was nothing left for
activations. Each Trn1 NeuronCore has **16 GB HBM (~14 GB usable)**.

| Model (bf16) | Fits 1 Trn1 core? | Do this |
|---|---|---|
| 7B (~13.5 GB) | ✅ tight | single core |
| 8B (~16 GB) | ❌ | TP=2 |
| 13B (~26 GB) | ❌ | TP≥4 (needs Trn2, see #4) |

Note the failure often surfaces on an innocent-looking op (`aten::cat`) — that's
just the first allocation *after* the weights, not the real culprit.

**Quantization is not an option here:** int8/fp8 aren't supported in this beta and
fp16 is the same size as bf16. Use more cores, not smaller weights.

Measure your own usable HBM:
```bash
python3 hbm_probe.py
```

---

## 3. `Failed to execute the device barrier 1` at startup

```
File "/opt/torch-neuronx/torch_neuronx/distributed/backend.py", line 172, in _neuron_runtime_setup
RuntimeError: Failed to execute the device barrier 1
```

Two different causes:

**(a) Stale runtime state from a previous crashed/exited TP run.**
→ **Restart the container between TP runs.** A run that crashed or called
`os._exit()` leaves the Neuron runtime in a state that breaks the next
`init_process_group`.

**(b) Cross-chip TP on Trn1.** TP=2 (both cores on one chip) initializes fine;
TP≥4 spans multiple chips and fails here on the current beta — including with
`NEURON_RT_ROOT_COMM_ID` set. Use a **Trn2** for models needing TP≥4.

---

## 4. Ranks die with SIGABRT mid-run

```
rank : 1  exitcode : -6   Signal 6 (SIGABRT)
```

Caused by **rank desync**: rank 0 went off and did slow solo work (loading a
32 GB fp32 CPU reference) while the other ranks sat waiting in a collective, and
the watchdog aborted.

**Fix: never do long single-rank work inside a distributed run.** Precompute the
CPU reference in a separate, non-distributed process first:
```bash
python3 cpu_ref.py <model> ref.pt      # single process
# then the TP run just loads ref.pt
```

---

## 5. SIGSEGV *after* the result prints

```
[model] per-position top-1 agreement ...: 100.0%
PORT_OK
...
rank : 0  exitcode : -11  Signal 11 (SIGSEGV)
```

The beta backend can segfault inside `destroy_process_group` **teardown**, after
all real work succeeded. It's cosmetic, but `torchrun` reports it as a failure and
it will mask your passing result.

**Fix: exit cleanly once you have the answer.**
```python
sys.stdout.flush()
os._exit(0)
```
(And per #3a, restart the container before the next TP run.)

---

## 6. `huggingface-cli` does nothing

```
Warning: `huggingface-cli` is deprecated and no longer works. Use `hf` instead.
```

**Fix:** use the new CLI — `hf download <repo> --local-dir <dir>`, `hf auth login`.

---

## 7. HTTP 403 downloading a Llama model

Gated repo whose license you haven't accepted. Check quickly:
```bash
curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $HF_TOKEN" \
  https://huggingface.co/meta-llama/Llama-3.1-8B/resolve/main/config.json
```
- `200` — good to go
- `403` — token valid, **license not accepted** → accept it on the model's HF page
- `401` — token missing/invalid → `hf auth login`

For LLaMA-1 there is no gated official repo; `huggyllama/llama-7b` is a public
re-upload of the original weights.

---

## 8. `nrta_tensor_read/write ... requires Trn2 ... falling back to synchronous IO`

Benign on Trn1. Asynchronous IO is a Trn2 feature; the op still runs and results
are correct. Ignore it.

---

## 9. First run is very slow, later runs are fast

Expected. The first forward/step compiles a NEFF (tens of seconds for small
models, ~200 s for a 7B/8B forward, ~9-16 min for big training graphs). A
persistent cache makes subsequent runs fast. Keep TP degree and shapes **fixed**
run-to-run or you invalidate the cache and recompile.
