# SPDX-License-Identifier: Apache-2.0
"""LTX-2 T2V on Neuron via the Omni entrypoint.

Mirrors examples/wan22/run.py but points at LTX-2 + the NeuronLTX2Pipeline
that is registered for `model_arch="LTX2Pipeline"` in this container.

Usage:
    python run_ltx2_omni.py                           # Full: 25 frames, 384x512, 8 steps
    python run_ltx2_omni.py --dev                     # Dev: 5 frames, 192x256, 3 steps
    python run_ltx2_omni.py --tensor-parallel-size 4  # 4-way TP
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

parser = argparse.ArgumentParser(description="LTX-2 T2V on Neuron via vLLM-Omni")
parser.add_argument("--dev", action="store_true",
                    help="Dev mode: smaller shape for quick smoke")
parser.add_argument("--tensor-parallel-size", type=int, default=8)
parser.add_argument("--height", type=int, default=None)
parser.add_argument("--width", type=int, default=None)
parser.add_argument("--num-frames", type=int, default=None)
parser.add_argument("--num-steps", type=int, default=None)
parser.add_argument(
    "--model-path",
    type=str,
    default="Lightricks/LTX-2",
    help="Local path or HF id for LTX-2 checkpoint.",
)
parser.add_argument(
    "--stage-config",
    type=str,
    default=None,
    help="Path to stage YAML. Defaults to ltx2_stage.yaml next to this script.",
)
parser.add_argument("--output", type=str, default="/work/ltx2_output.mp4",
                    help="Output mp4 path.")
parser.add_argument(
    "--prompt",
    type=str,
    default=(
        "A fluffy orange cat walking gracefully across a sunny garden path, "
        "vibrant flowers in the background, smooth motion, high quality"
    ),
)
parser.add_argument("--negative-prompt", type=str,
                    default="low quality, blurry, distorted")
parser.add_argument("--guidance-scale", type=float, default=3.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--bench-runs", type=int, default=1,
                    help="How many timed runs after the warm-up call.")
args = parser.parse_args()

# Same env-var preamble as the Wan runner — these are required for
# torch_neuronx + the Omni multiproc executor to behave.
os.environ["NEURON_USE_VANILLA_TORCH_XLA"] = "1"
os.environ["TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION"] = "1"
os.environ["VLLM_SLEEP_WHEN_IDLE"] = "1"
os.environ["NEURON_LOGICAL_NC_CONFIG"] = "2"
os.environ["NEURON_RT_LOG_LEVEL_TDRV"] = "info"
os.environ["VLLM_NEURON_COMPILATION_TIMEOUT"] = "3600"  # LTX is bigger
os.environ["NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE"] = "167772160"
os.environ["NEURON_SCRATCHPAD_PAGE_SIZE"] = "2048"
os.environ.setdefault("NEURON_CC_FLAGS", "--model-type=transformer --optlevel 1")
os.environ.setdefault("NEURON_SKIP_EFA_AFFINITY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_tp = args.tensor_parallel_size
_avail = len(os.sched_getaffinity(0))
os.environ.setdefault("OMP_NUM_THREADS", str(max(1, _avail // _tp)))
os.environ.setdefault("MKL_NUM_THREADS", str(max(1, _avail // _tp)))


def _save_frames(frames, out_path: str):
    """Write the diffusion-pipeline output frames as an mp4."""
    from diffusers.utils import export_to_video

    for out_idx, output in enumerate(frames):
        video_tensor = output.detach().cpu()
        # (batch, C, T, H, W) → (T, H, W, C)
        if video_tensor.dim() == 5 and video_tensor.shape[1] in (3, 4):
            video_tensor = video_tensor[0].permute(1, 2, 3, 0)
        elif video_tensor.dim() == 4 and video_tensor.shape[0] in (3, 4):
            video_tensor = video_tensor.permute(1, 2, 3, 0)
        if video_tensor.is_floating_point():
            video_tensor = video_tensor.clamp(-1, 1) * 0.5 + 0.5
        video_array = video_tensor.float().numpy()
        path = out_path if out_idx == 0 else f"{out_path}_{out_idx}.mp4"
        export_to_video(list(video_array), path, fps=24)
        print(f"Saved {video_array.shape[0]} frames to {path}")


def main():
    stage_cfg = args.stage_config or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "ltx2_stage.yaml",
    )

    print(f"[run_ltx2_omni] stage_config: {stage_cfg}")
    print(f"[run_ltx2_omni] model: {args.model_path}")
    print(f"[run_ltx2_omni] TP: {args.tensor_parallel_size}")

    t_omni_start = time.perf_counter()
    omni = Omni(
        model=args.model_path,
        stage_configs_path=stage_cfg,
        stage_init_timeout=3600,
        init_timeout=3600,
    )
    t_omni_init = time.perf_counter() - t_omni_start
    print(f"[run_ltx2_omni] Omni init: {t_omni_init:.1f}s")

    # Conservative LTX-2 dev / full shapes (matching Jim Burtoft's PR #57
    # validated 384×512 / 25-frame / 8-step config for the full path)
    if args.dev:
        height, width, num_frames, num_steps = 192, 256, 5, 3
    else:
        height, width, num_frames, num_steps = 384, 512, 25, 8

    height = args.height or height
    width = args.width or width
    num_frames = args.num_frames or num_frames
    num_steps = args.num_steps or num_steps

    params = OmniDiffusionSamplingParams(
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )

    request = {"prompt": args.prompt, "negative_prompt": args.negative_prompt}

    print(f"[run_ltx2_omni] Generating: "
          f"{num_frames} frames, {height}×{width}, {num_steps} steps")

    # First call — kicks off the compile if NEFFs aren't cached.
    t_first = time.perf_counter()
    result = omni.generate(request, params)
    t_first_total = time.perf_counter() - t_first
    print(f"[run_ltx2_omni] First call total: {t_first_total:.1f}s "
          f"(includes compile if cold)")

    frames = result[0].request_output.images
    print(f"[run_ltx2_omni] frames len: {len(frames) if frames else 0}")
    if frames:
        _save_frames(frames, args.output)

    # Optional warm bench runs
    if args.bench_runs > 0:
        print(f"\n[run_ltx2_omni] Warm bench: {args.bench_runs} run(s)")
        warm_times = []
        for i in range(args.bench_runs):
            t = time.perf_counter()
            warm_result = omni.generate(request, params)
            elapsed = time.perf_counter() - t
            warm_times.append(elapsed)
            print(f"  run {i+1}: {elapsed:.2f}s")
        warm_times.sort()
        med = warm_times[len(warm_times) // 2]
        print(f"\n[run_ltx2_omni] WARM MEDIAN: {med:.2f}s "
              f"({num_frames} frames, {height}×{width}, {num_steps} steps)")
        if warm_result[0].request_output.images:
            _save_frames(
                warm_result[0].request_output.images,
                args.output.replace(".mp4", "_warm.mp4"),
            )


if __name__ == "__main__":
    main()
