"""
ModernBERT on Inferentia2 (inf2) — SageMaker Deployment

Customer requirements:
- Model: ModernBERT (answerdotai/ModernBERT-base, 149M params)
- Sequence lengths: 128 and 300 tokens
- Latency target: < 80ms (current baseline: 85.6ms)
- Throughput: 27K+ requests
- Batching: Dynamic batching via torch_neuronx
- Framework: Neuron SDK 2.23+
- Instance: inf2.xlarge (2 NeuronCores, 32GB HBM)

Usage:
  # Compile and test locally:
  python deploy_modernbert_inf2.py --compile --seq-len 128 --batch-sizes 1,4,8,16

  # Benchmark:
  python deploy_modernbert_inf2.py --benchmark --seq-len 128 --batch-sizes 1,4,8,16
"""

import argparse
import time
import os
import torch
import torch_neuronx
from transformers import AutoTokenizer, AutoModel


def compile_model(model_id, seq_len, batch_size, save_dir="compiled_models"):
    """Compile ModernBERT for a specific batch size and sequence length."""
    print(f"Compiling {model_id} (batch={batch_size}, seq_len={seq_len})...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torchscript=True)
    model.eval()

    # Create example inputs
    example_input = tokenizer(
        "This is an example sentence for compilation.",
        padding="max_length",
        max_length=seq_len,
        truncation=True,
        return_tensors="pt",
    )

    # Expand to batch size
    input_ids = example_input["input_ids"].expand(batch_size, -1)
    attention_mask = example_input["attention_mask"].expand(batch_size, -1)

    # Trace for Neuron
    traced = torch_neuronx.trace(
        model,
        (input_ids, attention_mask),
        compiler_args="--auto-cast=none --model-type=transformer",
    )

    # Save compiled model
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/modernbert_bs{batch_size}_seq{seq_len}.pt"
    torch.jit.save(traced, save_path)
    print(f"  Saved: {save_path}")
    return traced


def compile_dynamic_batching(model_id, seq_len, batch_sizes):
    """Compile for multiple batch sizes (dynamic batching support)."""
    print(f"\n{'='*60}")
    print(f"  Compiling ModernBERT for dynamic batching")
    print(f"  Model: {model_id}")
    print(f"  Seq len: {seq_len}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"{'='*60}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, attn_implementation="eager")
    model.eval()

    # Disable torch.compile in ModernBERT (conflicts with torch_neuronx.trace)
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.reset()

    compiled_models = {}
    for bs in batch_sizes:
        example_input = tokenizer(
            "Example sentence for compilation.",
            padding="max_length",
            max_length=seq_len,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = example_input["input_ids"].expand(bs, -1)
        attention_mask = example_input["attention_mask"].expand(bs, -1)

        t0 = time.time()
        traced = torch_neuronx.trace(
            model,
            (input_ids, attention_mask),
            compiler_args="--auto-cast=none --model-type=transformer",
        )
        compile_time = time.time() - t0
        print(f"  BS={bs}: compiled in {compile_time:.1f}s")

        save_path = f"compiled_models/modernbert_bs{bs}_seq{seq_len}.pt"
        os.makedirs("compiled_models", exist_ok=True)
        torch.jit.save(traced, save_path)
        compiled_models[bs] = traced

    return compiled_models


def benchmark(model_id, seq_len, batch_sizes, num_iterations=100):
    """Benchmark latency and throughput."""
    print(f"\n{'='*60}")
    print(f"  Benchmarking ModernBERT on Neuron")
    print(f"  Seq len: {seq_len}, Iterations: {num_iterations}")
    print(f"{'='*60}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    results = []
    for bs in batch_sizes:
        # Load compiled model
        save_path = f"compiled_models/modernbert_bs{bs}_seq{seq_len}.pt"
        if not os.path.exists(save_path):
            print(f"  BS={bs}: not compiled yet, compiling...")
            compile_model(model_id, seq_len, bs)

        model = torch.jit.load(save_path)

        # Create input
        example = tokenizer(
            ["Benchmark input sentence."] * bs,
            padding="max_length",
            max_length=seq_len,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = example["input_ids"]
        attention_mask = example["attention_mask"]

        # Warmup
        for _ in range(10):
            _ = model(input_ids, attention_mask)

        # Benchmark
        latencies = []
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            _ = model(input_ids, attention_mask)
            latencies.append((time.perf_counter() - t0) * 1000)  # ms

        import statistics
        p50 = statistics.median(latencies)
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        mean = statistics.mean(latencies)
        throughput = bs / (mean / 1000)  # requests/sec

        results.append({
            "batch_size": bs,
            "seq_len": seq_len,
            "p50_ms": round(p50, 2),
            "p99_ms": round(p99, 2),
            "mean_ms": round(mean, 2),
            "throughput_rps": round(throughput, 1),
        })
        print(f"  BS={bs}: P50={p50:.1f}ms, P99={p99:.1f}ms, Mean={mean:.1f}ms, {throughput:.0f} req/s")

    # Summary
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  {'BS':<6} {'P50':<10} {'P99':<10} {'Mean':<10} {'Req/s':<10}")
    print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for r in results:
        print(f"  {r['batch_size']:<6} {r['p50_ms']:<10} {r['p99_ms']:<10} {r['mean_ms']:<10} {r['throughput_rps']:<10}")

    print(f"\n  Customer target: < 80ms latency")
    best = min(results, key=lambda x: x["p50_ms"])
    print(f"  Best result: BS={best['batch_size']}, P50={best['p50_ms']}ms")
    if best["p50_ms"] < 80:
        print(f"  ✅ MEETS TARGET")
    else:
        print(f"  ❌ Does not meet target (try smaller batch or shorter seq)")

    return results


def main():
    parser = argparse.ArgumentParser(description="ModernBERT on Inferentia2")
    parser.add_argument("--model", default="answerdotai/ModernBERT-base")
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-sizes", type=str, default="1,4,8,16")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    if args.compile:
        compile_dynamic_batching(args.model, args.seq_len, batch_sizes)

    if args.benchmark:
        benchmark(args.model, args.seq_len, batch_sizes, args.iterations)

    if not args.compile and not args.benchmark:
        print("Use --compile to compile, --benchmark to benchmark, or both.")


if __name__ == "__main__":
    main()
