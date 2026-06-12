"""LTX-2 19B native PyTorch on Trainium2 — Beta 3 stack.

Uses the Beta 3 device API (torch.device("neuron")) and the simpler
pattern from the Beta 3 user guide. Single-process to start (validates
the model + pipeline path); we'll add torchrun + sharding in a follow-up
once we know what fits.

Run inside the `beta3` container:
    sudo docker exec -it beta3 bash
    source /opt/torch-neuronx/.venv/bin/activate
    cd /workspace/path_c
    python ltx2_beta3.py [--num-steps 4] [--num-frames 25]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# Beta 3: device string is "neuron"
import torch
import torch_neuronx  # noqa: F401

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--num-frames", type=int, default=25)
    p.add_argument("--num-steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default=(
        "A golden retriever puppy runs across a sunny green meadow, "
        "its ears flapping in the wind. The camera follows from a low angle. "
        "Birds chirp in the background."
    ))
    p.add_argument("--output", default="results/ltx2_beta3_first.png")
    p.add_argument("--mode", choices=["transformer-only", "full"],
                   default="transformer-only",
                   help="transformer-only: just put DiT on Neuron, encoder+VAE on CPU. "
                        "full: try to put everything on Neuron (may OOM).")
    p.add_argument("--compile", action="store_true",
                   help="Wrap transformer with torch.compile(backend='neuron')")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"[ltx2-beta3] torch={torch.__version__}", flush=True)
    print(f"[ltx2-beta3] torch_neuronx={torch_neuronx.__version__}", flush=True)

    device = torch.device("neuron")
    cpu = torch.device("cpu")

    # Quick device sanity check
    x = torch.randn(4, 4, device=device)
    y = x @ x.T
    print(f"[ltx2-beta3] device probe ok: {y.shape} on {y.device}", flush=True)

    # Load LTX-2 pipeline on CPU first
    print(f"\n[ltx2-beta3] loading Lightricks/LTX-2 (CPU, bf16)...", flush=True)
    t0 = time.time()
    from diffusers import LTX2Pipeline
    pipe = LTX2Pipeline.from_pretrained(
        "Lightricks/LTX-2",
        torch_dtype=torch.bfloat16,
    )
    print(f"[ltx2-beta3] loaded in {time.time() - t0:.1f}s", flush=True)
    print(f"[ltx2-beta3] components: text_encoder={type(pipe.text_encoder).__name__}, "
          f"transformer={type(pipe.transformer).__name__}, "
          f"vae={type(pipe.vae).__name__}", flush=True)

    def n_params(m):
        return sum(p.numel() for p in m.parameters())

    print(f"[ltx2-beta3] text_encoder: {n_params(pipe.text_encoder)/1e9:.2f}B params",
          flush=True)
    print(f"[ltx2-beta3] transformer:  {n_params(pipe.transformer)/1e9:.2f}B params",
          flush=True)
    print(f"[ltx2-beta3] vae:          {n_params(pipe.vae)/1e9:.2f}B params",
          flush=True)

    # Move transformer to Neuron
    print(f"\n[ltx2-beta3] moving transformer to {device}...", flush=True)
    t0 = time.time()
    pipe.transformer = pipe.transformer.to(device)
    print(f"[ltx2-beta3] ✓ transformer moved in {time.time() - t0:.1f}s", flush=True)

    if args.mode == "full":
        # Try moving encoder + VAE too
        print(f"[ltx2-beta3] moving VAE to {device}...", flush=True)
        try:
            pipe.vae = pipe.vae.to(device)
            print(f"[ltx2-beta3] ✓ VAE moved", flush=True)
        except Exception as e:
            print(f"[ltx2-beta3] ✗ VAE.to failed: {type(e).__name__}: {e}", flush=True)
            print(f"[ltx2-beta3] keeping VAE on CPU", flush=True)

        print(f"[ltx2-beta3] moving text_encoder to {device}...", flush=True)
        try:
            pipe.text_encoder = pipe.text_encoder.to(device)
            print(f"[ltx2-beta3] ✓ text_encoder moved", flush=True)
        except Exception as e:
            print(f"[ltx2-beta3] ✗ text_encoder.to failed: {type(e).__name__}: {e}",
                  flush=True)
            print(f"[ltx2-beta3] keeping text_encoder on CPU", flush=True)

    # Optional: torch.compile the transformer
    if args.compile:
        print(f"\n[ltx2-beta3] wrapping transformer with torch.compile(backend='neuron')",
              flush=True)
        pipe.transformer = torch.compile(
            pipe.transformer, backend="neuron",
            dynamic=False, fullgraph=False,
        )

    # Generate
    print(f"\n[ltx2-beta3] generating: {args.width}×{args.height}, "
          f"{args.num_frames} frames, {args.num_steps} steps...", flush=True)
    print(f"[ltx2-beta3] (cold first call includes NEFF compilation; expect 5-15 min)",
          flush=True)

    t0 = time.time()
    try:
        with torch.no_grad():
            result = pipe(
                prompt=args.prompt,
                height=args.height, width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_steps,
                guidance_scale=4.0,
                max_sequence_length=1024,
                generator=torch.Generator(device="cpu").manual_seed(args.seed),
                output_type="pil",
            )
    except Exception as e:
        elapsed = time.time() - t0
        import traceback
        print(f"[ltx2-beta3] ✗ generation FAILED after {elapsed:.1f}s", flush=True)
        traceback.print_exc()
        return 1

    elapsed = time.time() - t0
    print(f"[ltx2-beta3] ✓ generated in {elapsed:.1f}s "
          f"({elapsed/args.num_steps:.2f}s/step avg)", flush=True)
    frames = result.frames[0]
    print(f"[ltx2-beta3] frames: {len(frames)}", flush=True)

    # Save first frame as PNG (full video export needs imageio)
    from pathlib import Path
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if frames:
        frames[0].save(out)
        print(f"[ltx2-beta3] WROTE {out}", flush=True)

        # Try MP4 export
        try:
            from diffusers.utils import export_to_video
            mp4_path = out.with_suffix(".mp4")
            export_to_video(frames, str(mp4_path), fps=24)
            print(f"[ltx2-beta3] WROTE {mp4_path}", flush=True)
        except Exception as e:
            print(f"[ltx2-beta3] mp4 export failed: {e}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
