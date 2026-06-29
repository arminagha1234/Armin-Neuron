# Running Sesame CSM-1B (Text-to-Speech) on AWS Trainium

Complete, copy-paste guide to run **Sesame CSM-1B** — a conversational text-to-speech
model — on **AWS Trainium2**. CSM turns text into 24 kHz speech (Llama-3.2-1B backbone
+ depth decoder + Mimi audio codec).

**You do not need to know anything about Trainium or Neuron.** Every command is copy-paste.

> **Status:** ✅ Working on Trainium. Heavy compute (16-layer backbone + Mimi codec)
> on a NeuronCore, validated **cosine 1.000000** vs CPU. **Streaming TTFA (time to
> first audio) = 241 ms warm — under the 500 ms target** (bf16 + streaming +
> StaticCache; see `results/PERF_PROGRESS.md`). Also shipped as a registered
> `CsmPipeline` in the vLLM-Omni Neuron plugin (see [vLLM-Omni](#vllm-omni-csmpipeline)).

---

## ⚠️ Two important prerequisites (read first)

### 1. Instance: launch a `trn2.3xlarge`
CSM-1B runs on a **single Trainium2 chip**, so a **`trn2.3xlarge`** (smallest trn2
slice) is enough — you do **not** need a `trn2.48xlarge`.

### 2. Software: you need the **native-PyTorch Neuron beta** — NOT the public beta
Validated on the **native-PyTorch Neuron beta** (torch_xla 2.9). The public Neuron
beta ships an older torch_xla whose int64-cast lowering breaks CSM's RoPE/mask casts.
Make sure the box has the **native-PyTorch beta** stack. If unsure, check with whoever
provisioned it — the public release will not work for this model.

---

## Step 0 — Launch
EC2 **`trn2.3xlarge`**, AMI with the **native-PyTorch Neuron beta**, ≥150 GB disk.

## Step 1 — Connect
```bash
ssh -i /path/to/mykey.pem ubuntu@<your-instance-public-dns>
```

## Step 2 — Working folder + environment
```bash
mkdir -p ~/csm && cd ~/csm
source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate     # native-PyTorch beta venv
neuron-ls                                                 # should list a NeuronCore
```

## Step 3 — Install transformers (CSM support)
```bash
pip install "transformers==4.56.2" soundfile
```
(Pin 4.56.2 — it's the version validated with this Neuron runtime.)

## Step 4 — Download the model
`sesame/csm-1b` is gated; use the ungated canonical HF conversion `eustlb/csm-1b`:
```bash
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("eustlb/csm-1b", local_dir="./csm_1b")
PY
export CSM_MODEL=$HOME/csm/csm_1b
```

## Step 5 — Get the script + generate speech
Copy `src/generate_speech.py` from this repo into `~/csm`, then:
```bash
python generate_speech.py --text "[0]Hello from Trainium." --out hello.wav
# "[0]" / "[1]" selects the speaker id
python generate_speech.py --text "[0]How are you today?" --max-new-tokens 256 --out q.wav
```
The **first run compiles for the chip (a few minutes)**; later runs are faster. Output
is a 24 kHz `.wav`.

---

## Why this just works (what the script handles)
HF `generate` can't be lowered to Neuron (its loop/cache bookkeeping uses int64
dynamic control flow). So `generate_speech.py` keeps the **generate loop on CPU and
offloads the heavy compute to the NeuronCore**:
- `backbone_model` (16-layer transformer) → Neuron
- `codec_model` (Mimi decode, codes → waveform) → Neuron
- depth decoder (tiny, 4 layers) + sampling + loop → CPU

It also moves Mimi's RVQ codebook tensors (`self.embed`) that `.to(device)` misses,
preserves `ModelOutput`/`Cache` across the device boundary, and runs fp32.

## A note on output determinism
CSM is autoregressive, so the exact waveform differs run-to-run / CPU-vs-Neuron — a
single argmax flip from sub-ULP fp32 differences cascades into a different (but
equally valid) speech realization. The model is proven correct by **teacher-forced
logit match (cosine 1.0, argmax 100%)** and **Mimi decode cosine 1.0** — see
`results/RESULTS.md`.

## vLLM-Omni CsmPipeline
A `CsmPipeline` (`src/csm_pipeline.py`) is implemented for the **vLLM-Omni Neuron
plugin** (`vllm_omni_neuron`), registered alongside `Wan22Pipeline`/`HelloWorldPipeline`
(`model_arch: CsmForConditionalGeneration`). It wraps the same offload logic behind the
omni pipeline interface (`forward(request) -> DiffusionOutput(output=<waveform>)`) for
serving via the omni `/v1/audio/speech` path. It registers and constructs in the omni
beta container; full in-container execution needs the native-PyTorch-beta torch_xla
(the container's bundled torch_xla has the int64-cast quirk noted above).

## Troubleshooting
| Symptom | Fix |
|---|---|
| `Expected self.dtype() == dst.dtype()` in RoPE/mask | Old torch_xla. Use the **native-PyTorch beta** (torch_xla 2.9). |
| `Expected XLA tensor. Got: torch.FloatTensor` | Mimi RVQ stray tensors not moved — use `generate_speech.py` (it moves them). |
| `Complex/NCC` or int64 `dot` errors | You ran the full model/generate on-device. Use the offload script. |
| First run very slow | One-time per-shape compile; it caches. |

## Files (src/)
- `generate_speech.py` — the one-command TTS tool (the deliverable).
- `csm_pipeline.py` — the vLLM-Omni `CsmPipeline` (Path A serving artifact).
- `run_csm_offload.py` — the offload run + CPU-compare harness.
- `run_csm_cpu.py` — CPU reference (oracle).

## Credits & license
- Model: `eustlb/csm-1b` (canonical HF conversion). Original: Sesame CSM
  ([sesame/csm-1b](https://huggingface.co/sesame/csm-1b)). Apache-2.0.
