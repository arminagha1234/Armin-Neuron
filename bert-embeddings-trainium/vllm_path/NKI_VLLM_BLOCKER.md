# Why customer NKI kernels can't run inside vLLM-Neuron today

A focused analysis of the version-skew that blocks customer-written NKI kernels from being dispatched inside a vLLM-Neuron-served model. Includes a concrete reproduction.

## The dispatch chain

For a `@nki.jit` kernel called from a `torch.nn.Module.forward()` to run on Neuron hardware, three things must coexist in one Python environment:

1. **`nki`** — the kernel language (writes the kernel)
2. **`neuronx-cc`** — the compiler (compiles to NEFF)
3. **`torch_neuronx.nki_hop`** — the bridge that wraps `@nki.jit` into a torch custom op so PyTorch dispatch reaches the kernel

If any one is missing or ABI-mismatched, dispatch fails at import time before the kernel runs.

## Container inventory (verified 2026-06-08)

| Container | nki | neuronx-cc | torch | torch_neuronx | nki_hop bridge |
|---|---|---|---|---|---|
| `vllm-neuron` private-beta DLC | 0.4.0 ✅ | 2.25 ✅ | 2.10 | **MISSING** ❌ | ❌ |
| Beta 3 native DLC | 0.4.0 ✅ | 2.25 ✅ | **2.11** | 2.11.3 ✅ | ✅ |

So today: NKI dispatch works on the native DLC, **fails on the vllm-neuron DLC** at:
```
ImportError: cannot import name 'NKIHOPCaller' from 'torch_neuronx.nki_hop'
File "/opt/conda/.../nki/framework/kernel.py", line 46
```

## Can we just `pip install torch_neuronx` into the vllm-neuron container?

Tried. Doesn't work. Two distinct blockers stack:

### Blocker 1 — public Neuron pip index has no `nki_hop`-bearing build

```
$ pip index versions torch-neuronx --extra-index-url=https://pip.repos.neuron.amazonaws.com
torch-neuronx (2.9.0.2.14.27725+e2ff0410)
Available versions: 2.9.0.2.14.x, 2.9.0.2.13.x, ..., 2.5.1.x, 2.1.2.x, 1.13.x, 1.0
```

The public channel tops out at the `torch_neuronx 2.9.x` line (built for torch 2.9). **`nki_hop` is a `torch_neuronx 2.11.x` feature** (a Beta 3 / native-DLC release). It's not in the public channel.

Confirmed empirically:
```
$ ls /opt/conda/lib/python3.12/site-packages/torch_neuronx/nki_hop*
ls: cannot access ...: No such file or directory
```

### Blocker 2 — installing the public 2.9 build crashes anyway (torch C++ ABI mismatch)

```
$ pip install torch-neuronx --extra-index-url=https://pip.repos.neuron.amazonaws.com --no-deps
Successfully installed torch-neuronx-2.9.0.2.14.27725+e2ff0410

$ python3 -c "import torch_neuronx"
OSError: /opt/conda/lib/python3.12/site-packages/torch_neuronx/lib/libtorchneuron.so:
undefined symbol: _ZN3c104impl12PyObjectSlotD1Ev
```

The `libtorchneuron.so` was built against torch 2.9; the vllm-neuron container's torch is 2.10. The C++ ABI symbol `c10::impl::PyObjectSlot::~PyObjectSlot()` exists in 2.9 but not 2.10. `--no-deps` got past Python-level pip resolution but the underlying C library can't link.

Used `pip uninstall torch-neuronx` to revert; vllm-neuron itself is unaffected.

### Blocker 3 (would matter if 1 and 2 were solved) — vllm-neuron's wheel is pinned to torch 2.10

`vllm-neuron 0.19.0.0` is built against torch 2.10. The only `torch_neuronx` with `nki_hop` (2.11.3) requires torch 2.11. They cannot coexist without the Neuron team rebuilding one of them.

## The actual root cause

This is **not a customer-fixable problem.** It's a release-coordination matter inside the Neuron team. To enable customer NKI kernels in vLLM-Neuron, AWS needs to either:

(a) Ship a `torch_neuronx 2.10.x` build that includes `nki_hop` to the public pip channel, OR
(b) Rebase `vllm-neuron` onto torch 2.11 so it can use the Beta 3 `torch_neuronx 2.11.x` that already has `nki_hop`.

Option (b) is presumably the natural path — it lines up with vllm-neuron's GA preparation since torch 2.11 is the current Trainium platform target.

## What this means for adopters today

| Goal | Available today |
|---|---|
| Customer-written NKI kernel in a custom model, served via Triton + native+compile | ✅ Yes (Beta 3 native DLC) |
| Customer-written NKI kernel inside a vLLM-served model on Trainium | ❌ Blocked. Wait for vllm-neuron rebase to torch 2.11. |
| Stock vLLM-Neuron serving of supported models | ✅ Yes (uses kernels baked in by AWS) |

This contrib's `vllm_path/` includes a working `BertModel` + runner patch that exercises the model-extension surface vLLM-Neuron does have. It just can't fuse NKI kernels into that model until the version-alignment lands.

## Reproduction (10 minutes)

```bash
# Inside vllm_neuron container:

# 1. Snapshot env
pip freeze > /tmp/pip_before.txt

# 2. Confirm bridge is missing
python3 -c "from torch_neuronx.nki_hop import NKIHOPCaller" 2>&1 | tail -2
# Expected: ModuleNotFoundError: No module named 'torch_neuronx'

# 3. Try to install from Neuron public index
pip install torch-neuronx --extra-index-url=https://pip.repos.neuron.amazonaws.com --no-deps
# Installs torch_neuronx-2.9.0.2.14.x

# 4. Confirm install fails to load due to torch ABI
python3 -c "import torch_neuronx" 2>&1 | tail -2
# Expected: OSError: ... undefined symbol: _ZN3c104impl12PyObjectSlotD1Ev

# 5. Confirm nki_hop is absent from this version anyway
ls /opt/conda/lib/python3.12/site-packages/torch_neuronx/nki_hop* 2>&1
# Expected: No such file or directory

# 6. Revert
pip uninstall torch-neuronx -y
```

## Filing this back to the Neuron team

This document is the bug-report-quality artifact. Pair with:
- The working custom `BertModel` + runner patches in `vllm_path/` (proves the model-extension surface works)
- The partial NKI fused attention kernel in `src/` (proves NKI dispatch works on the native DLC)
- The 8-gotcha catalog in `src/NKI_KERNEL_NOTES.md` (separate ISA usability feedback)

Together: a tight ask for "next vllm-neuron release should ship `torch_neuronx 2.11.x` with `nki_hop`" with a reproduction the team can run in 10 minutes.
