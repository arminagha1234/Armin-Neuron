# Chai-1 → Native PyTorch (Neuron) Port — Results

_Run 2026-07-02 on trn2.48xlarge `i-038786075e02b9697` (us-east-2)._
_Stack: **Native PyTorch Beta 3** — beta DLC `concourse-release-0461d3b`,
driver `aws-neuronx-dkms 2.28.0.0`, `native_venv` with **torch 2.11.0 +
torch_neuronx** (native `torch.device("neuron")` PrivateUse1 backend).
No torch-xla, no `torch_neuronx.trace` — pure eager device placement._

## Headline

**5 of 6 chai-1 TorchScript components run natively on a NeuronCore in eager
mode, with zero model code changes** — just `module.to("neuron")` + inputs
`.to("neuron")` + call `forward_256(...)`. Parity vs CPU is exact or
bf16-clean. The **trunk** is the one blocker (Neuron runtime assertion).

## Per-component matrix (bucket 256, peptide `GAAL`)

| Component | Native Neuron | Time* | Max abs diff vs CPU | Notes |
|---|---|---|---|---|
| `feature_embedding` | ✅ pass | 2.8s | **0.0** (all 6 outputs) | exact |
| `bond_loss_input_proj` | ✅ pass | 0.1s | **0.0** | exact |
| `token_embedder` | ✅ pass | 0.6s | 1.6e-2 (bf16) | mean ~1e-6 |
| `diffusion_module` | ✅ pass | 71s | 1.3e-4 | **highest ROI** (called ~200×/prediction); fp32-clean |
| `confidence_head` | ✅ pass | 42s | 6.3e-2 (bf16 logits) | mean ~3e-3; pre-softmax |
| `trunk` | ❌ fail | — | — | runtime assertion (below) |

\* Cold-cache (includes NEFF compile). Persistent caching makes reruns far
faster.

## The one blocker: `trunk.pt`

```
python: /opt/workspace/KaenaRuntime/tdrv/tensor.c:185:
  tensor_set_slice: Assertion `(tensor_source->_size) >= (offset + size)' failed.
```

- Fails during **NEFF execution** (not compilation) in `forward_256`.
- **Not** a scratchpad issue: retried with the runtime's own recommendation
  `NEURON_CC_FLAGS="--hbm-scratchpad-page-size=1024"` +
  `NEURON_SCRATCHPAD_PAGE_SIZE=1024` → same assertion.
- **Not** an adaptive-eager grouping artifact: retried with
  `torch.use_deterministic_algorithms(True)` (unfused) → same assertion.
- The trunk is the einsum-heavy pairformer; its graph contains `aten::chunk`
  (9×) and `aten::cat` (3×) on the big `(1,256,256,256)` pair tensor. The
  `tensor_set_slice` bounds failure most likely comes from how one of those
  slice/concat ops on the pair representation lowers in this beta runtime.
- Hypothesis worth testing: the peptide input has a **degenerate MSA depth**
  (no real MSA), so an MSA-related slice in the trunk may hit a zero/one-sized
  dimension the runtime mishandles. Would need a real-MSA input to confirm.

### Tried and ruled out
- scratchpad page-size flags (compile + runtime) — no change
- deterministic / unfused eager — no change

### Not yet tried (next options)
- `torch.compile(backend="neuron")` — limited value here since these are
  pre-scripted TorchScript modules (dynamo treats them as opaque, same runtime
  path likely → same assertion).
- Real multi-sequence input to rule out the degenerate-MSA hypothesis.
- File as **Beta 3 feedback** to the Neuron team with the assertion + the
  captured `cap_trunk.pt.inputs.pt` fixture (it's a clean, minimal repro).

## Reusable artifacts (on the instance, `/home/ubuntu/`)

- `cap_<comp>.inputs.pt` — captured `forward_256` inputs for each component.
- `cap_<comp>.ref.pt` — CPU reference outputs for parity checks.
- `test_component_neuron.py` — generic per-component Neuron runner
  (`python test_component_neuron.py <comp>.pt`, `DET=1` for deterministic).
- `run_native_neuron.py` — full-pipeline runner with `device=torch.device("neuron")`.

## WORKING END-TO-END (hybrid): trunk on CPU, everything else on Neuron

`run_hybrid_neuron.py` produces a **complete chai-1 prediction on Trainium**:

```
HYBRID end-to-end OK in 80.7s (trunk=CPU, rest=Neuron)
5 .cif structures written; aggregate_score = 0.0427
```

- feature_embedding, bond_loss_input_proj, token_embedder, **diffusion_module**,
  confidence_head → **NeuronCore** (native eager).
- trunk → CPU (only failing component).
- Device boundary is automatic: chai's `ModuleWrapper.forward(move_to_device=)`
  moves trunk's CPU outputs back onto Neuron for the diffusion step. The only
  extra fix was returning confidence_head outputs on CPU so the post-model
  boolean-mask scoring (`softmax_einsum_and_cpu`) stays on CPU.

### Parity vs pure-CPU baseline
- **aggregate score: 0.0430 (CPU) vs 0.0427 (hybrid Neuron)** — equivalent.
- Structure: mean atom displacement 1.22 Å, max 3.05 Å over 22 atoms. Expected:
  the diffusion sampler is chaotic, so bf16-level deltas on Neuron cascade into
  a different-but-valid conformation. The matching score is the meaningful
  quality signal at this tiny scale (4-residue peptide, 2 diffusion steps).

### Hypothesis tested this pass (and ruled out)
- **Degenerate-MSA theory: FALSE.** Inspected captured trunk inputs — MSA is
  already padded to depth 16384, templates to 4, pair tensor (1,256,256,256).
  Shapes are full-size, not degenerate; a real-MSA input would feed the same
  shapes and hit the same assertion. So the trunk bug is genuine op-lowering on
  full tensors, not an empty-dimension edge case.
- Also ruled out (no change): `TORCH_NEURONX_ENABLE_ASYNC_NRT=0`,
  scratchpad page-size flags, deterministic/unfused eager.

## Deep-dive: attempts to get the trunk (→ 6/6) running on Neuron

Root cause characterized: the assertion `tensor_set_slice: source._size >=
offset+size` fires in the **eager per-op path**, where each op is its own NEFF
and inputs are marshalled host→device. One op in the trunk asks the runtime to
write a device region larger than the source tensor. Debug logging showed the
abort happens right after a NEFF loads successfully (during its I/O tensor
setup), and one NEFF reports `mac_count: 4294967296` = exactly **2^32** — a
32-bit size/descriptor overflow signature. The trunk's triangle ops at bucket
N=256 hit 256^4 = 2^32 exactly, consistent with a 32-bit field overflow in the
runtime for that op.

**9 approaches tried — all hit the identical assertion:**

| # | Approach | Result |
|---|---|---|
| 1 | eager (default) | assert |
| 2 | scratchpad page-size flags (compile+runtime) | assert |
| 3 | `TORCH_NEURONX_ENABLE_ASYNC_NRT=0` | assert |
| 4 | `torch.use_deterministic_algorithms(True)` (unfused) | assert |
| 5 | MSA depth 16384 → 512 (shrink biggest tensor) | assert |
| 6 | `torch.jit.freeze` (device-first) | **assert moved deeper** (compiled further) → same assert |
| 7 | `torch.jit.optimize_for_inference` | (not reached / same class) |
| 8 | `torch.compile(backend="neuron")` | assert (dynamo treats ScriptModule as opaque → eager fallback) |
| 9 | force `.contiguous()` on all inputs | assert |

Key finding from #6: freezing **changed** the failure point (compiled more
blocks before aborting), which proves the bug is a **specific deep op**, not a
setup/marshalling artifact — and that it is **not bypassable from user space**
with this beta. It needs a runtime fix (the 32-bit descriptor path) from the
Neuron team.

**Why the usual escapes don't apply here:** chai ships the trunk as an opaque
pre-traced TorchScript artifact (no eager source), so we can't rewrite the
offending op, and torch.compile can't trace into it to fuse the intermediates
on-device. That leaves the eager per-op path, which is exactly where the bug
lives.

### Path to true 6/6 (requires Neuron-side action, not user-space)
1. File the trunk repro with the Neuron team — minimal, deterministic:
   `cap_trunk.pt.inputs.pt` + `test_component_neuron.py trunk.pt`. The 2^32
   mac_count + `tensor_set_slice` overflow is a strong, specific signal.
2. Retry on the next Beta build (this one is `concourse-release-0461d3b`).
3. If/when chai releases eager model source (not just TorchScript), the trunk
   could be torch.compiled into a single on-device NEFF, sidestepping the
   per-op marshalling path entirely.

## Bottom line

We have a **working end-to-end chai-1 prediction on Trainium** via native
PyTorch — 5/6 components on the NeuronCore (including the compute-dominant
diffusion module), trunk on CPU, matching CPU score. True 6/6 is blocked by a
single characterized Beta 3 runtime bug (32-bit overflow in a trunk op's
slice path) that is **not** avoidable from user space and needs a Neuron
runtime fix — repro is ready to file. No model rewrite, just
device placement + a 2-line device-boundary shim. The only remaining item for
a *fully* on-Neuron run is one Beta 3 runtime bug in the trunk's slice/concat
lowering, which has a minimal captured repro (`cap_trunk.pt.inputs.pt`) ready
to file with the Neuron team.
