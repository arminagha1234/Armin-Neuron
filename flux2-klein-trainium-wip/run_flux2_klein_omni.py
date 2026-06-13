# SPDX-License-Identifier: Apache-2.0
"""FLUX.2-klein 4B image-to-image on Neuron via the vllm-omni Omni entrypoint.

Mirrors the LTX-2 / Wan 2.2 omni runner shape. Loads the model under our
`NeuronFlux2KleinPipeline` (registered for `model_arch="Flux2KleinPipeline"`),
optionally fuses an image-to-image zoom LoRA, and runs a few canonical
inference calls.

Usage:
    python run_flux2_klein_omni.py                          # T2I, 1024x1024, 28 steps
    python run_flux2_klein_omni.py --image input.png        # I2I (zoom LoRA path)
    python run_flux2_klein_omni.py --tensor-parallel-size 2 # TP=2 (default 1)
"""
import argparse
import os
import time

import torch
from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

# torch_neuronx patches F.gelu with a wrapper around the C builtin;
# Dynamo cannot trace through the C function in fullgraph mode.
torch.nn.functional.gelu = torch.ops.aten.gelu.default

parser = argparse.ArgumentParser(description="FLUX.2-klein 4B on Neuron via vLLM-Omni")
parser.add_argument("--dev", action="store_true",
                    help="Dev mode: smaller shape for quick smoke")
parser.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="TP size (4B fits on 1 core; default 1)")
parser.add_argument("--height", type=int, default=None)
parser.add_argument("--width", type=int, default=None)
parser.add_argument("--num-steps", type=int, default=None)
parser.add_argument("--guidance-scale", type=float, default=4.0,
                    help="CFG scale; LoRA card recommends 1.1 for the LoRA")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--model-path", type=str,
                    default="black-forest-labs/FLUX.2-klein-4B",
                    help="Local path or HF id for FLUX.2-klein checkpoint.")
parser.add_argument("--lora-path", type=str,
                    default=None,
                    help=(
                        "Local path or HF id for the LoRA. "
                        "Example image-to-image zoom LoRA file: "
                        "flux-red-zoom-lora.safetensors"
                    ))
parser.add_argument("--lora-scale", type=float, default=1.1,
                    help="LoRA scale; the zoom LoRA card recommends 1.1")
parser.add_argument("--image", type=str, default=None,
                    help=(
                        "Path to a reference image for image-to-image. "
                        "Required for the zoom LoRA: an image with a red "
                        "highlight box marking the zoom region."
                    ))
parser.add_argument("--prompt", type=str,
                    default="Zoom into the red highlighted area")
parser.add_argument("--output", type=str,
                    default="/work/flux2_klein_output.png",
                    help="Output PNG path.")
parser.add_argument("--stage-config", type=str, default=None,
                    help="Path to stage YAML. Defaults to flux2_klein_stage.yaml next to this script.")
parser.add_argument("--bench-runs", type=int, default=1,
                    help="How many timed runs after the warm-up call.")
args = parser.parse_args()

# Same env-var preamble as the LTX-2 runner.
os.environ["NEURON_USE_VANILLA_TORCH_XLA"] = "1"
os.environ["TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION"] = "1"
os.environ["VLLM_SLEEP_WHEN_IDLE"] = "1"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ["NEURON_RT_LOG_LEVEL_TDRV"] = "info"
os.environ["VLLM_NEURON_COMPILATION_TIMEOUT"] = "1800"
os.environ["NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE"] = "167772160"
os.environ["NEURON_SCRATCHPAD_PAGE_SIZE"] = "2048"
os.environ.setdefault("NEURON_CC_FLAGS",
                      "--model-type=transformer --optlevel 1")
os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_tp = args.tensor_parallel_size
_avail = len(os.sched_getaffinity(0))
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, _avail // _tp)))
os.environ.setdefault("MKL_NUM_THREADS", str(max(1, _avail // _tp)))


def _load_image(path):
    if path is None:
        return None
    from PIL import Image
    img = Image.open(path).convert("RGB")
    return img


def main():
    # Stage config selection
    if args.stage_config is None:
        here = os.path.dirname(os.path.abspath(__file__))
        stage_config = os.path.join(here, "flux2_klein_stage.yaml")
    else:
        stage_config = args.stage_config

    # Default canonical shape — FLUX.2-klein recommended size for the LoRA
    # is 1024x1024; in dev mode we drop to 512x512 for fast iteration.
    height = args.height or (512 if args.dev else 1024)
    width = args.width or (512 if args.dev else 1024)
    steps = args.num_steps or (8 if args.dev else 28)

    # Build the Omni engine.
    print(f"[run] starting Omni: model={args.model_path} tp={_tp} stage={stage_config}",
          flush=True)
    print(f"[run] target shape: {width}x{height}, {steps} steps", flush=True)

    engine = Omni(
        model=args.model_path,
        stage_configs_path=stage_config,
        stage_init_timeout=3600,
        init_timeout=3600,
    )

    # Optionally fuse the LoRA on the underlying transformer module.
    # vllm-omni doesn't expose load_lora_weights() directly, so we reach
    # into the pipeline state, fuse the LoRA via diffusers' loader API
    # at the inner DiT, and unload it back. The transformer was wrapped
    # in `_NeuronTransformerWrapper`; we patch on `inner`.
    if args.lora_path:
        print(f"[run] fusing LoRA: {args.lora_path} (scale={args.lora_scale})", flush=True)
        # NOTE: this runs on the engine driver process. The engine workers
        # may already have started loading; the LoRA weight load needs to
        # happen at pipeline construction time. For now we surface this
        # as a TODO; a v1 work-around is to merge LoRA into base weights
        # offline and pass the merged checkpoint to --model-path.
        print("[run] TODO: LoRA fusion via Omni driver path is not wired yet. "
              "v1 work-around: merge LoRA offline and pass --model-path "
              "<merged-dir>", flush=True)

    image_pil = _load_image(args.image) if args.image else None

    # Build the request: positional dict + sampling params (Omni convention).
    sp = OmniDiffusionSamplingParams(
        height=height,
        width=width,
        num_inference_steps=steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    request = {"prompt": args.prompt}
    if image_pil is not None:
        request["image"] = image_pil

    # Warm-up call (compile happens here for the first call).
    print(f"[run] warm-up call (compile + first generation)...", flush=True)
    t0 = time.time()
    out = engine.generate(request, sp)
    elapsed = time.time() - t0
    print(f"[run] warm-up: {elapsed:.1f}s", flush=True)

    if out and getattr(out[0], "request_output", None) is not None:
        ro = out[0].request_output
        # FLUX.2-klein returns `.images` on the request_output (PIL list).
        images = getattr(ro, "images", None) or getattr(ro, "image", None)
        from PIL import Image
        if images:
            img = images[0] if isinstance(images, list) else images
            if isinstance(img, Image.Image):
                os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
                img.save(args.output)
                print(f"[run] WROTE {args.output}", flush=True)
            else:
                print(f"[run] note: output is not a PIL image (type={type(img).__name__}); "
                      f"check post_process_func wiring", flush=True)
        else:
            print(f"[run] no images in request_output (fields: {dir(ro)[:10]})",
                  flush=True)
    else:
        print(f"[run] no output produced from warm-up call", flush=True)

    # Bench runs
    times = []
    for i in range(args.bench_runs):
        t0 = time.time()
        out = engine.generate(request, sp)
        t = time.time() - t0
        times.append(t)
        print(f"[run] bench[{i}]: {t:.2f}s", flush=True)

    if times:
        import statistics
        mean = statistics.mean(times)
        med = statistics.median(times)
        std = statistics.stdev(times) if len(times) > 1 else 0.0
        print(f"\n[run] === SUMMARY ===", flush=True)
        print(f"  mean   {mean:.2f}s", flush=True)
        print(f"  median {med:.2f}s", flush=True)
        print(f"  stdev  {std:.2f}s", flush=True)
        if steps:
            print(f"  per-step (mean) {mean/steps:.2f}s/step", flush=True)


if __name__ == "__main__":
    main()
