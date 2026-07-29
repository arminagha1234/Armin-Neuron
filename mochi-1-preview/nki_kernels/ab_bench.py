"""A/B benchmark harness: measure Mochi denoise s/step across configurations.

Runs the *real* transformer forward (not a microbenchmark) so the numbers are
what actually matters end to end. Compares, on-device:

    eager                 -- the validated baseline (~6.8 s/step at 19f TP=4)
    +compile              -- torch.compile(backend="neuron")
    +nki:attn / :swiglu / :rmsnorm / :qkv / :rope  -- one kernel at a time
    +nki:all              -- every validated kernel
    +compile+nki:all      -- both together

Each configuration is selected purely by env vars (MOCHI_NKI_*, plus a
--compile flag), so this harness never edits the port -- it drives it.

Usage (inside the DLC container, ONE at a time to avoid core contention):

    # baseline
    torchrun --nproc_per_node 4 nki_kernels/ab_bench.py --frames 19 --steps 6 --tag eager

    # a single kernel
    MOCHI_NKI_ATTN=1 torchrun --nproc_per_node 4 nki_kernels/ab_bench.py \
        --frames 19 --steps 6 --tag attn

    # compile
    torchrun --nproc_per_node 4 nki_kernels/ab_bench.py \
        --frames 19 --steps 6 --tag compile --compile

Warmup steps are excluded from the timed average so NEFF-compile / first-step
cost does not pollute the per-step number. Results append to a JSONL so a
sweep script can collect them.

This measures the transformer only where possible: it times the full pipe()
call and divides by steps, but also installs a per-step timer on the
transformer wrapper so we get a clean denoise-only s/step independent of the
one-time weight load and the CPU VAE decode.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_MOCHI = _HERE.parent
_SRC = _MOCHI / "src"
for p in (str(_SRC), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=19)
    ap.add_argument("--steps", type=int, default=6, help="total denoise steps")
    ap.add_argument("--warmup", type=int, default=2,
                    help="leading steps excluded from the timed average")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=848)
    ap.add_argument("--guidance-scale", type=float, default=1.0)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--tag", default="eager", help="label for this config")
    ap.add_argument("--out", default="results/ab_bench.jsonl")
    ap.add_argument("--prompt", default="A chameleon changing colors, 4k.")
    return ap.parse_args()


class StepTimer:
    """Wraps the transformer to record per-forward wall time on rank 0."""

    def __init__(self, inner):
        self.inner = inner
        self.times: list[float] = []

    def __getattr__(self, name):
        # Delegate config/cache_context/etc. to the wrapped module.
        return getattr(self.inner, name)

    def __call__(self, *args, **kwargs):
        import torch
        t0 = time.time()
        out = self.inner(*args, **kwargs)
        # Force materialization so timing captures device work, not lazy queue.
        if torch.is_tensor(out):
            out.cpu()
        elif isinstance(out, (tuple, list)):
            for o in out:
                if torch.is_tensor(o):
                    o.cpu()
        self.times.append(time.time() - t0)
        return out


def main():
    args = parse_args()

    # Install the requested NKI kernels via env flags BEFORE building anything.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from neuron_compat import install_bmm_sdpa
    install_bmm_sdpa()

    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    def log(m):
        if rank == 0:
            print(f"[ab_bench:{args.tag}] {m}", flush=True)

    # Reuse the runner's own build path so the model is identical to production.
    import run_mochi_native as R

    # Monkeypatch: after the runner builds the transformer, install NKI kernels
    # and wrap it in a StepTimer. We do this by calling the runner's pieces
    # directly rather than main(), so we control timing.
    if args.compile:
        os.environ["_AB_COMPILE"] = "1"

    log(f"config: frames={args.frames} steps={args.steps} warmup={args.warmup} "
        f"compile={args.compile} flags="
        f"{[k for k in os.environ if k.startswith('MOCHI_NKI')]}")

    # Build via the runner's argv path so behavior matches production exactly.
    argv = [
        "run_mochi_native.py",
        "--num-frames", str(args.frames),
        "--num-steps", str(args.steps),
        "--guidance-scale", str(args.guidance_scale),
        "--height", str(args.height),
        "--width", str(args.width),
        "--prompt", args.prompt,
        "--output", f"results/ab_{args.tag}.mp4",
        "--report", f"results/ab_report_{args.tag}.json",
    ]
    if args.compile:
        argv.append("--compile")
    sys.argv = argv

    # Hook: install NKI kernels + StepTimer right after the transformer is built.
    _orig_build = R.build_transformer
    timer_holder = {}

    def _build_and_instrument(*a, **k):
        model = _orig_build(*a, **k)
        try:
            from install_nki_kernels import install_nki_kernels
            install_nki_kernels(model, rank=rank)
        except Exception as exc:
            log(f"NKI install skipped: {exc}")
        return model

    R.build_transformer = _build_and_instrument

    # Wrap NeuronTransformerWrapper so we time each denoise forward.
    _orig_wrapper = R.NeuronTransformerWrapper

    class _TimedWrapper(_orig_wrapper):
        def forward(self, *a, **k):
            t0 = time.time()
            out = super().forward(*a, **k)
            timer_holder.setdefault("times", []).append(time.time() - t0)
            return out

    R.NeuronTransformerWrapper = _TimedWrapper

    t0 = time.time()
    rc = R.main()
    total = time.time() - t0

    times = timer_holder.get("times", [])
    if rank == 0:
        timed = times[args.warmup:] if len(times) > args.warmup else times
        avg = sum(timed) / len(timed) if timed else float("nan")
        result = {
            "tag": args.tag,
            "frames": args.frames,
            "steps": args.steps,
            "warmup": args.warmup,
            "compile": args.compile,
            "nki_flags": [k for k in os.environ if k.startswith("MOCHI_NKI")],
            "denoise_s_per_step": round(avg, 4),
            "all_step_times": [round(t, 4) for t in times],
            "total_s": round(total, 1),
            "world_size": world_size,
            "rc": rc,
        }
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("a") as f:
            f.write(json.dumps(result) + "\n")
        log(f"RESULT denoise={avg:.3f} s/step  (steps={times})")
        log(f"appended to {out}")
    if dist.is_initialized():
        dist.barrier()
    return rc


if __name__ == "__main__":
    sys.exit(main())
