"""Run Gemma 4 E4B via native PyTorch on Neuron (no vLLM).

Uses HF transformers Gemma4ForConditionalGeneration directly.
HF's implementation handles PLE + KV-sharing + heterogeneous head_dim correctly.

Usage on trn2:
    python3 run_e4b_native.py --prompt "The capital of France is"
"""
import argparse
import time
import torch
import sys
sys.path.insert(0, "/work")

# Register gemma4 model type in transformers
import gemma4_transformers_stub
gemma4_transformers_stub.install()

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/models/gemma-4-E4B-it")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=30)
    parser.add_argument("--device", default="cpu", help="cpu or neuron")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model)

    print(f"Loading model on {args.device}...")
    t0 = time.time()

    if args.device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            attn_implementation="eager",
        )
    else:
        # For Neuron: load on CPU then move
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            attn_implementation="eager",
        )
        model = model.to(torch.device(args.device))

    model.eval()
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s")
    print(f"Model class: {type(model).__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # Generate
    inputs = tok(args.prompt, return_tensors="pt")
    if args.device != "cpu":
        inputs = {k: v.to(args.device) for k, v in inputs.items()}

    print(f"\nPrompt: {args.prompt}")
    print(f"Token IDs: {inputs['input_ids'].tolist()}")

    t1 = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=False,
        )
    gen_time = time.time() - t1

    response = tok.decode(outputs[0], skip_special_tokens=True)
    new_tokens = outputs.shape[1] - inputs["input_ids"].shape[1]
    print(f"\nGenerated ({new_tokens} tokens in {gen_time:.2f}s):")
    print(f"  {response}")
    print(f"\nTTFT: {gen_time:.2f}s (includes all {new_tokens} tokens)")


if __name__ == "__main__":
    main()
