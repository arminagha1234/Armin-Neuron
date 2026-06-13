#!/usr/bin/env python3
"""Run Gemma 4 E4B-it on Trainium2 with native PyTorch + TP=2.

Native PyTorch path. NO vLLM, NO NxDI. Beta 3 stack:
  * torch 2.11.0
  * torch_neuronx 2.11.3 with `torch.device("neuron")`
  * neuronxcc 2.25 (lazy-imported by torch_neuronx)
  * transformers 5.12 (HF Gemma4 implementation handles E4B's PLE +
    KV-sharing in pure PyTorch, no fork required)
  * `parallelize_module(model, mesh, plan)` with backend="neuron" PG

Modes:
  * Default: prefill TTFT sweep across `--seq-lens`, optional
    `--compile` for `torch.compile(backend="neuron")`.
  * `--decode`: KV-cache decode TPOT (works but NOTE: each step
    recompiles a new graph because Beta 3 doesn't support dynamic
    shapes — TPOT is dominated by recompile time, not execution).
    Production decode needs a static-shape KV cache (pre-allocated
    slots + fixed-size attention mask).

Launch (inside a Beta 3 DLC container):

    NEURON_RT_VIRTUAL_CORE_SIZE=2 NEURON_RT_NUM_CORES=2 \\
    /opt/torch-neuronx/.venv/bin/torchrun \\
        --nproc_per_node=2 --rdzv_backend=c10d \\
        --rdzv_endpoint=localhost:29500 \\
        run_e4b.py \\
        --model /root/models/gemma-4-E4B-it \\
        --seq-lens 64,128,256,512,1024,2048 \\
        --warmup 1 --runs 3 \\
        [--compile] \\
        --out results.json
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module

# Local module — works whether run from src/ or from the package root.
try:
    from .tp_plan import build_e4b_tp_plan  # type: ignore
except ImportError:  # pragma: no cover
    from tp_plan import build_e4b_tp_plan  # type: ignore


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def log(msg: str, rank: int | None = None) -> None:
    r = rank if rank is not None else (dist.get_rank() if dist.is_initialized() else "?")
    print(f"[rank {r}] {msg}", flush=True)


def neuron_sync() -> None:
    if hasattr(torch, "neuron") and hasattr(torch.neuron, "synchronize"):
        torch.neuron.synchronize()


def time_call(model, input_ids, attention_mask, label, rank, *, use_cache=False, past_kv=None):
    """Time one forward pass with cross-rank barriers so all ranks measure the same window."""
    dist.barrier()
    t0 = time.time()
    # Use no_grad for KV-cache path: HF's KV-cache mutates tensors that
    # `inference_mode()` forbids ("Inference tensors do not track version
    # counter").
    ctx = torch.no_grad() if use_cache else torch.inference_mode()
    with ctx:
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask, use_cache=use_cache)
        if past_kv is not None:
            kwargs["past_key_values"] = past_kv
        out = model(**kwargs)
        next_tok = out.logits[:, -1, :].argmax(dim=-1)
        next_tok_cpu = next_tok.cpu()
    neuron_sync()
    dist.barrier()
    dt_ms = (time.time() - t0) * 1000.0
    if label:
        log(f"{label}: {dt_ms:.1f} ms", rank=rank)
    new_kv = getattr(out, "past_key_values", None) if use_cache else None
    return dt_ms, int(next_tok_cpu), new_kv


# -----------------------------------------------------------------------------
# Benchmarks
# -----------------------------------------------------------------------------

def benchmark_prefill_sweep(model, tokenizer, device, seq_lens, prompt, warmup, runs, rank):
    """Measure full-seq prefill TTFT across multiple seq_len buckets."""
    results = {}
    for seq_len in seq_lens:
        log(f"=== prefill seq_len = {seq_len} ===", rank=rank)
        tok = tokenizer(prompt, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=seq_len)
        input_ids = tok["input_ids"].to(device)
        attention_mask = tok["attention_mask"].to(device)

        compile_ms = None
        for w in range(warmup):
            dt, _, _ = time_call(model, input_ids, attention_mask,
                                  f"compile/warmup{w} (seq_len={seq_len})", rank)
            if w == 0:
                compile_ms = dt

        latencies = []
        last_tok = None
        for r in range(runs):
            dt, last_tok, _ = time_call(model, input_ids, attention_mask,
                                         f"run{r} (seq_len={seq_len})", rank)
            latencies.append(dt)

        if rank == 0:
            decoded = tokenizer.decode([last_tok])
            mean_ms = sum(latencies) / len(latencies)
            log(f"prefill seq_len={seq_len}: mean={mean_ms:.1f} ms, "
                f"min={min(latencies):.1f}, max={max(latencies):.1f}, "
                f"compile={compile_ms:.1f} ms, next_token={decoded!r}", rank=0)
            results[seq_len] = {
                "mean_ms": mean_ms,
                "min_ms": min(latencies),
                "max_ms": max(latencies),
                "samples_ms": latencies,
                "compile_ms": compile_ms,
                "next_token": decoded,
            }
    return results


def benchmark_decode_tpot(model, tokenizer, device, prompt_len, decode_tokens, prompt, rank):
    """Measure autoregressive decode TPOT with HF KV cache.

    NOTE: Beta 3 ``torch.compile(backend="neuron")`` does not support
    dynamic shapes. The HF KV-cache decode loop changes the
    attention_mask length by 1 every step, so each step recompiles a
    fresh graph. The numbers below reflect that recompile cost. For
    production decode, use a static-shape KV cache (fixed
    ``max_kv_len``, full-size mask with valid-position flags).
    """
    log(f"=== decode TPOT (prompt={prompt_len}, decode={decode_tokens}) ===", rank=rank)
    tok = tokenizer(prompt, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=prompt_len)
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)

    # Step 1: Prefill with use_cache=True (TTFT)
    log("prefill with KV cache (TTFT)", rank=rank)
    dist.barrier()
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    neuron_sync()
    dist.barrier()
    ttft_ms = (time.time() - t0) * 1000.0
    log(f"TTFT (prefill + first token sample) = {ttft_ms:.1f} ms", rank=rank)

    past_kv = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(dim=-1)
    generated = [int(next_tok.cpu())]

    # Step 2: decode loop, one token at a time
    log(f"decode loop ({decode_tokens} tokens)", rank=rank)
    decode_latencies = []
    cur_input = next_tok.unsqueeze(-1)
    cur_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=device)],
        dim=1,
    )
    for _ in range(decode_tokens - 1):
        dist.barrier()
        t0 = time.time()
        with torch.no_grad():
            out = model(input_ids=cur_input, attention_mask=cur_mask, use_cache=True,
                        past_key_values=past_kv)
            next_tok = out.logits[:, -1, :].argmax(dim=-1)
            past_kv = out.past_key_values
        neuron_sync()
        dist.barrier()
        decode_latencies.append((time.time() - t0) * 1000.0)
        generated.append(int(next_tok.cpu()))
        cur_input = next_tok.unsqueeze(-1)
        cur_mask = torch.cat(
            [cur_mask, torch.ones((1, 1), dtype=cur_mask.dtype, device=device)],
            dim=1,
        )

    if rank == 0:
        steady = decode_latencies[1:] if len(decode_latencies) > 1 else decode_latencies
        mean_decode_ms = sum(steady) / max(len(steady), 1)
        sorted_d = sorted(steady)
        median_decode_ms = sorted_d[len(sorted_d) // 2] if sorted_d else 0
        decoded_text = tokenizer.decode(generated)
        log(f"decode first step: {decode_latencies[0] if decode_latencies else 0:.1f} ms", rank=0)
        log(f"decode steady-state: mean={mean_decode_ms:.1f} ms, "
            f"median={median_decode_ms:.1f} ms over {len(steady)} steps", rank=0)
        log(f"generated text: {decoded_text!r}", rank=0)
        return {
            "prompt_len": prompt_len,
            "decode_tokens": decode_tokens,
            "ttft_ms": ttft_ms,
            "decode_first_ms": decode_latencies[0] if decode_latencies else None,
            "decode_steady_mean_ms": mean_decode_ms,
            "decode_steady_median_ms": median_decode_ms,
            "decode_samples_ms": decode_latencies,
            "generated_text": decoded_text,
            "tpot_throughput_tok_per_s": (
                1000.0 / mean_decode_ms if mean_decode_ms > 0 else None
            ),
        }
    return None


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/root/models/gemma-4-E4B-it",
                    help="local path to a Gemma4 E4B-it checkpoint dir "
                         "(use scripts/build_local_model.py to materialize one "
                         "with a patched tokenizer_config.json)")
    ap.add_argument("--seq-lens", default="64,128,256,512,1024,2048",
                    help="comma-separated prefill bucket sizes")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--compile", action="store_true",
                    help="apply torch.compile(backend='neuron')")
    ap.add_argument("--decode", action="store_true",
                    help="also run a KV-cache decode TPOT benchmark")
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--decode-prompt-len", type=int, default=128)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--out", default="results.json")
    ap.add_argument("--skip-prefill-sweep", action="store_true")
    args = ap.parse_args()
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world = dist.get_world_size()
    log(f"PG ready: world_size={world}", rank=rank)

    mesh = init_device_mesh("neuron", (world,))
    log(f"mesh ready: {mesh}", rank=rank)

    device = torch.device("neuron")

    log(f"loading model from {args.model}", rank=rank)
    from transformers import AutoTokenizer, AutoModelForImageTextToText

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    log(f"loaded on CPU in {time.time()-t0:.1f}s", rank=rank)

    log("building TP plan + parallelize_module", rank=rank)
    plan, _, _, owners, shareds = build_e4b_tp_plan(model)
    log(f"plan: {len(plan)} modules ({owners} owner attn / {shareds} shared attn)", rank=rank)
    t0 = time.time()
    parallelize_module(model, mesh, plan)
    log(f"parallelize_module returned in {time.time()-t0:.1f}s", rank=rank)

    log("moving model to neuron device (sharded)", rank=rank)
    t0 = time.time()
    model = model.to(device)
    neuron_sync()
    log(f"model on neuron in {time.time()-t0:.1f}s", rank=rank)
    model.eval()

    if args.compile:
        log("applying torch.compile(backend='neuron')", rank=rank)
        model = torch.compile(model, backend="neuron", dynamic=False)
        log("compile decorator applied (graph builds on first call)", rank=rank)

    results = {
        "world": world,
        "owner_layers": owners,
        "shared_layers": shareds,
        "compile_enabled": args.compile,
        "prefill": {},
        "decode": None,
    }

    if not args.skip_prefill_sweep:
        results["prefill"] = benchmark_prefill_sweep(
            model, tokenizer, device, seq_lens, args.prompt,
            args.warmup, args.runs, rank,
        )

    if args.decode:
        try:
            decode_results = benchmark_decode_tpot(
                model, tokenizer, device, args.decode_prompt_len,
                args.decode_tokens, args.prompt, rank,
            )
            if rank == 0:
                results["decode"] = decode_results
        except Exception as e:  # noqa: BLE001
            log(f"decode benchmark FAILED: {e!r}", rank=rank)
            if rank == 0:
                results["decode"] = {"error": repr(e)}

    if rank == 0:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        log(f"wrote {args.out}", rank=0)
        print("\n=== FINAL SUMMARY ===")
        print(f"world_size={world}, owner_layers={owners}, shared_layers={shareds}, "
              f"compile={args.compile}")
        if results["prefill"]:
            print("\nPREFILL (full-seq forward, no KV cache):")
            print(f"{'seq_len':>10} {'mean ms':>10} {'min ms':>10} {'max ms':>10} {'compile ms':>14}")
            for sl, r in results["prefill"].items():
                print(f"{sl:>10} {r['mean_ms']:>10.1f} {r['min_ms']:>10.1f} "
                      f"{r['max_ms']:>10.1f} {r.get('compile_ms', 0) or 0:>14.1f}")
        if results["decode"] and "error" not in results["decode"]:
            d = results["decode"]
            print(f"\nDECODE (KV-cache, prompt={d['prompt_len']}, gen={d['decode_tokens']}):")
            print(f"  TTFT (full prefill):   {d['ttft_ms']:.1f} ms")
            print(f"  Decode first token:    {d['decode_first_ms']:.1f} ms")
            print(f"  Decode steady (mean):  {d['decode_steady_mean_ms']:.1f} ms")
            print(f"  Decode steady (med):   {d['decode_steady_median_ms']:.1f} ms")
            tput = d.get("tpot_throughput_tok_per_s")
            if tput:
                print(f"  TPOT throughput:       {tput:.2f} tok/s")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
