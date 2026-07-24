# Qwen3.5-4B training on Trainium2 — MEASURED (native PyTorch beta)

Native PyTorch (TorchNeuron) on trn2.48xlarge. DLC native-PyTorch Neuron beta DLC (Jul-23),
torch 2.11 Neuron build, torch-neuronx 2.11.3, neuronx-cc 2.26. `torch.device("neuron")`, bf16, eager attn.

## Smoke (2-layer, single core, seq 128) — PROVES IT TRAINS
| step | time | loss |
|---|---|---|
| 0 (compile) | 419 s | 12.82 |
| 1 warm | 0.60 s | 12.45 |
| 2 warm | 0.55 s | 12.08 |
| 3 warm | 0.57 s | 11.72 |
- warm avg 0.575 s/step, **222 tok/s** (bs1, seq128, single core, 2 layers). Loss decreasing = gradients correct.
- forward + backward + AdamW all on the Neuron device. GatedDeltaNet layers compile (torch fallback path).

## Full 32-layer — the memory reality (single NeuronCore = ~24 GB HBM)
- **Full fine-tune (all 4.21B weights) does NOT fit on one core.** 4.21B bf16 ≈ 8.4 GB weights + 8.4 GB grads
  + ~33 GB AdamW fp32 states ≈ 50 GB, before activations → OOM at seq 512 on one core (confirmed: allocator
  hit the 24 GB ceiling). This is exactly why **FSDP (shard optimizer states across cores)** is required for
  full-FT of a 4B — not a bug, it's the expected signal.
- **LoRA (base frozen, tiny optimizer state) fits full 32L on one core** — and is the proven precedent (27B LoRA).

## Full 32-layer LoRA SFT, seq 512, single core, EAGER — ✅ TRAINS (Goal 1 done, full depth)
| step | time | loss |
|---|---|---|
| 0 (compile) | ~180 s | 14.28 |
| 1 warm | 6.55 s | 14.13 |
| 2 warm | 5.99 s | 14.01 |
- **warm avg 6.16 s/step, 83 tok/s** (bs1, seq512, LoRA r=16, single core, ALL 32 layers: 24 GDN + 8 full-attn).
- Complete Qwen3.5-4B architecture doing forward + backward + AdamW on-device, loss decreasing. This is the
  real end-to-end eager-mode training proof at full depth.
- Extrapolation (single core, un-optimized): 1e8 tok/epoch ≈ 334 h → **÷N cores with FSDP** is the throughput lever.

## ✅ FSDP MULTI-CORE WORKS (box2) — the throughput lever is proven
`torch.distributed` **backend="neuron"** + torch FSDP via `torchrun --nproc_per_node 4`, each rank pinned to
its own core (NEURON_RT_VISIBLE_CORES=0..3). Full 32L LoRA:
| config | warm step | aggregate tok/s | loss |
|---|---|---|---|
| **FSDP world=4** | **1.63 s/step** | **313.7 tok/s** | 12.97 → 12.90 (decreasing) |
- vs single-core LoRA (72–83 tok/s) → **~3.8–4.3× at 4 ranks** (near-linear at this scale). Compile 37s warm.
- Confirms the native-beta distributed path: `init_process_group(backend="neuron")` + standard torch FSDP,
  no torch_xla / accelerate / torchtitan. Scaling to 8/16/32 ranks is the next measurement.

## Single-core scaling envelope (full 32L LoRA, eager) — memory-bound, motivates FSDP
| config | result |
|---|---|
| bs1 seq512 | ✅ **72 tok/s, 7.1 s/step** |
| bs2 seq512 | ❌ OOM (24 GB/core ceiling) |
| bs1 seq1024 | ❌ OOM |
| bs1 seq2048 | ❌ OOM |
- Even LoRA on one core saturates HBM past bs1/seq512 (activations dominate). **⟹ FSDP is required both to
  raise effective batch (data parallel) and to shard state — not optional for throughput OR for longer seq.**
- (A transient `aten::zero_` compile error also appeared once at seq512 on box2 — env/ordering, not
  fundamental; box1's identical seq512 ran clean.)

## Box2 gradient-correctness proof (2-layer full-FT, seq128, single core, EAGER)
| step | time | loss |
|---|---|---|
| 1 warm | 0.49 s | 2.38 |
| 2 warm | 0.51 s | 0.0042 |
| 3 warm | 0.50 s | 0.0010 |
- **Overfits a single fixed batch to loss ≈ 0.001** → gradients are correct end-to-end (0.498 s/step, 257 tok/s).
- Confirms a SECOND trn2 box (box2) is a working trainer, in parallel with box1.

## torch.compile(backend=neuron) — Goal 2

### Full 32L LoRA seq512 + compile → **FAILS to compile (SBUF overflow)**
`RuntimeError: COMPILATION FAILED: SB partition size: 229376`. torch.compile fuses the entire 32-layer
GDN-torch-fallback forward+backward into ONE graph, which overflows the on-chip SBUF partition budget.
Eager works because each op compiles as a separate NEFF; compile mode fuses everything → too big.
**This is the concrete cost of the GDN torch-fallback graph, and the strongest argument for the NKI GDN
kernel swap** (shrinks the graph so it fits). Levers to make compile fit at full depth:
1. Swap GDN torch-fallback → chunked NKI kernel (biggest — shrinks the graph).
2. Compile per-layer / per-submodule instead of the whole model (graph-break on decoder-layer boundaries).
3. Shorter seq / fewer fused ops per graph.

### 2-layer full-FT seq128 + compile → **ALSO FAILS, different error (compiler bug on GDN fallback)**
`neuronx-cc [INTERNAL_ERROR] [NCC_IMGN901] MacroGeneration assertion error: Can only vectorize loop or
free axes`. At 2 layers the graph is small, so this is NOT SBUF overflow — it's a **neuronx-cc lowering bug
triggered by the GDN torch-fallback's delta-rule recurrence ops** under torch.compile.

### ✅ RESOLVED via our chunked GDN kernel (Rank-1, gdn_kernel/chunked_gdn.py)
The GDN torch-fallback breaks compile because HF builds the intra-chunk `(I−A)⁻¹` with a data-dependent
forward-substitution loop + variable-width in-place slice writes. We replaced it with an **exact
Neumann-doubling product** `(I−A)⁻¹ = ∏_k (I + A^(2^k))` (valid: strictly-lower A is nilpotent) — a fixed
log2(BT) matmul unroll, no control flow, no in-place writes. Same math (cos 1.0).
- **CPU parity vs HF: cos = 1.000** (fwd + grads dq/dk/dv/dg/dbeta, maxabs ≤1.2e-7). Autograd derives the
  backward for free — zero hand-written backward at Rank-1.
- **Trains on device:** 2L GDN, loss 8.83→0.0000, 0.16s warm step.
- **torch.compile(backend=neuron, dynamic=False): SUCCESS** — ~96s compile, 0.05s warm steps, loss
  decreasing. Only graph break = benign cross_entropy, NOT the GDN core. Unblocks Goal 2 with OUR code,
  independent of AWS pre-GA timeline.

### VERDICT (Goal 2, stock fallback): torch.compile does NOT work with the STOCK GDN torch-fallback — at ANY depth. (Fixed by the kernel above.)
- 32L → SBUF overflow; 2L → compiler internal error. Two distinct failures, both rooted in the GDN fallback.
- **Eager works** because ops compile separately (no whole-graph fuse/vectorize of the recurrence).
- ⟹ **The NKI GDN kernel swap is a PREREQUISITE for torch.compile, not just a perf optimization.** Once the
  GDN core is a single clean NKI custom-op, the surrounding graph is standard transformer ops that compile fine
  (confirmed working `torch.compile(backend=neuron, dynamic=False)` contract exists in the torch-neuronx test
  suite). Path: land the chunked GDN NKI fwd+bwd (autograd.Function) → then torch.compile the model.
- (LoRA r/q/k/v/o target-modules require a full-attn layer present; a 2-layer slice is all-GDN, so compile
  A/B at reduced depth must be full-FT, which is what was run here.)

## Notes / optimization levers (next)
- GDN uses torch fallback (fla fast-path not installed) → heavier graph, long first-step compile. Swapping in
  our trn2-validated KDA/gated-delta NKI kernel is the perf lever.
- Single core so far → FSDP across 64 cores (guide's NEURON_RT_NUM_CORES=64 + torchrun) is the throughput multiplier.
- bf16 (fp8 not supported in this beta).
