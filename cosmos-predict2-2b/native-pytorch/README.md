# Cosmos-Predict2-2B — Native PyTorch on Trainium2

## Status

**Working.** Both Text2Image (`Cosmos2TextToImagePipeline`) and
Image-to-Video (`Cosmos2VideoToWorldPipeline`) generate end-to-end with
numerically correct output. DiT runs on `torch.device("neuron")`; T5 and
the WAN VAE run on CPU.

## Architecture

```
prompt ──► T5  (CPU) ──► prompt_embeds
                                │
image ──► VAE encode (CPU) ──► image_latents
                                │
                                ▼
                ┌──────────────────────────────────┐
       latents▶│  CosmosTransformer3DModel  (DiT)  │ ◀── on Neuron
                │  - self-attention + cross-attn   │
                │  - FlowMatchEuler stepping       │
                └────────────────┬─────────────────┘
                                 │ denoised latents
                                 ▼
                         VAE decode (CPU)
                                 │
                                 ▼
                          image / video frames
```

## Neuron porting fixes (3 small shims, no model surgery)

1. **`DummySafetyChecker`** — the Cosmos pipeline hard-requires the
   heavy `cosmos_guardrail` model chain at `__init__`. We pass a
   parameter-less `nn.Module` shim with `.device` / `.dtype` /
   `check_text_safety()` / `check_video_safety()` to bypass it.
   *(Use the real `cosmos_guardrail` for production deployments.)*
2. **`torchvision.transforms.functional.resize` shim** — Cosmos's DiT
   resizes the padding mask with torchvision; there's no ABI-matching
   torchvision build for Neuron's torch 2.11. We inject a tiny shim
   backed by `F.interpolate` (NEAREST for masks).
3. **DiT-on-Neuron forward wrapper** — moves the transformer to
   `torch.device("neuron")`, wraps `forward()` to shuttle tensors
   to/from the device, and calls `torch.neuron.synchronize()` per call.
   T5 + WAN-VAE stay on CPU unchanged.

No RoPE patch, no SDPA patch, no precision patches were needed. Cosmos's
DiT uses real-arithmetic RoPE in stock diffusers, and at the resolutions
tested (≤1024² image, 480×832 video) bf16 is precision-safe.

## Files

| File | Role |
|---|---|
| `src/cosmos_cpu_smoke.py` | CPU reference run + the 3 porting shims |
| `src/cosmos_neuron.py` | Text2Image runner (DiT on `neuron`) |
| `src/cosmos_video_neuron.py` | Video2World runner (DiT on `neuron`) + per-call DiT timing |
| `results/cosmos_t2i_1024.png` | 1024² Text2Image output (Neuron) |
| `results/cosmos_t2i_512_cpu_reference.png` | 512² CPU reference for std comparison |
| `results/cosmos_video_480x832_25f.mp4` | 480×832 × 25-frame Video2World clip (Neuron) |

## Reproduction (trn2.48xl, Beta 3 DLC)

```bash
# Setup (once per container)
pip install -U diffusers transformers accelerate safetensors opencv-python-headless

export NEURON_RT_VIRTUAL_CORE_SIZE=2 NEURON_LOGICAL_NC_CONFIG=2 \
       NEURON_SKIP_EFA_AFFINITY=1
export HF_HOME=/mnt/data/hf_cache
export HF_TOKEN=<your_huggingface_token>   # license must be accepted

# Text2Image, 1024² × 20 steps
H=1024 W=1024 STEPS=20 python3 src/cosmos_neuron.py

# Video2World, 480×832 × 25 frames × 12 steps
H=480 W=832 FRAMES=25 STEPS=12 python3 src/cosmos_video_neuron.py
```

Cold compile takes longer; persistent NEFF cache means subsequent
runs are warm.

## Known issues / next work

- **CPU-side bottleneck at video resolution.** 480×832 × 25f spends
  ~102 s on the CPU side (T5 + WAN-VAE decode) versus ~142 s on the
  Neuron DiT. Moving the WAN VAE onto Neuron (we have WAN VAE
  experience from the LTX-2 / WAN training work) should crush this.
- **Larger video shapes need TP.** Higher frame counts and resolution
  push activation memory past the single-chip envelope. The FLUX v3
  full-shard plan (split fused projections + sharded SwiGLU) transfers
  if/when the Cosmos DiT needs the same treatment.
- **vLLM-Omni serving path** is still the WIP stub (see `../vllm-omni/`).

## License

Apache-2.0 for this contrib code. NVIDIA Cosmos weights are subject to
NVIDIA's model license — accept on the Hugging Face model page.
