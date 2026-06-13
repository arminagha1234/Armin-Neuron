"""Gemma 4 E4B on Trainium — Native PyTorch.

Uses HF Gemma4ForConditionalGeneration with proper multimodal processor.
Key insight: E4B requires mm_token_type_ids from the processor (it's a
multimodal model backbone, not a standalone text decoder).

Usage:
    # CPU reference (works today):
    python3 run_e4b_native.py --device cpu --prompt "What is 2+2?"

    # Neuron (compile the text decoder):
    python3 run_e4b_native.py --device neuron --prompt "What is 2+2?"
"""
import argparse
import time
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--prompt", default="What is the capital of France?")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--device", default="cpu", choices=["cpu", "neuron"])
    parser.add_argument("--compile", action="store_true", help="Apply torch.compile(backend='neuron')")
    args = parser.parse_args()

    from transformers import AutoProcessor, AutoModelForCausalLM

    print(f"Loading processor from {args.model}...")
    proc = AutoProcessor.from_pretrained(args.model)

    print(f"Loading model on {args.device}...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="eager",
    )

    if args.device == "neuron":
        print("Moving model to Neuron...")
        model = model.to(torch.device("neuron"))

    if args.compile:
        print("Compiling with torch.compile(backend='neuron')...")
        model = torch.compile(model, backend="neuron")

    model.eval()
    load_time = time.time() - t0
    print(f"Model loaded in {load_time:.1f}s ({type(model).__name__}, "
          f"{sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params)")

    # Build inputs with proper multimodal processor
    messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
    text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = proc(text=text, return_tensors="pt")

    if args.device == "neuron":
        inputs = {k: v.to("neuron") if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

    print(f"\nPrompt: {args.prompt}")
    print(f"Input keys: {list(inputs.keys())}")
    print(f"input_ids shape: {inputs['input_ids'].shape}")

    # Generate
    t1 = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
    gen_time = time.time() - t1

    # Decode only the new tokens
    new_token_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = proc.decode(new_token_ids, skip_special_tokens=True)
    n_tokens = len(new_token_ids)

    print(f"\nGenerated ({n_tokens} tokens in {gen_time:.2f}s, "
          f"{n_tokens/gen_time:.1f} tok/s):")
    print(f"  {response}")


if __name__ == "__main__":
    main()
