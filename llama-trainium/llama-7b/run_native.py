"""Run LLaMA-1 7B inference on a single Trainium NeuronCore (native PyTorch).

Loads the model as a normal HF model, moves it to device="neuron" in bf16, and
greedy-decodes a few tokens. This is the whole point of TorchNeuron: your model
stays plain PyTorch, you just change the device.

    python3 run_native.py --model huggyllama/llama-7b \
        --prompt "The capital of France is" --max-new-tokens 30

Notes:
  * A 7B model fits on ONE Trn1 NeuronCore (16 GB HBM). An 8B does not — see
    ../llama-3.1-8b/ for the tensor-parallel version.
  * Greedy decoding here re-runs the full sequence each step (no KV cache) to
    keep the example simple. Each new sequence length compiles a NEFF the first
    time, so the first ~max_new_tokens run is slow; re-runs hit the cache.
"""
import argparse, time, torch, torch_neuronx
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="huggyllama/llama-7b")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=30)
    args = ap.parse_args()

    device = torch.device("neuron")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager").to(device)
    model.eval()

    ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)
    print(f"prompt: {args.prompt!r}", flush=True)

    t0 = time.time()
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            logits = model(ids, use_cache=False).logits[:, -1, :]
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
    text = tok.decode(ids[0], skip_special_tokens=True)
    print(f"\n=== generation ({time.time()-t0:.1f}s incl. first-run compile) ===\n{text}")


if __name__ == "__main__":
    main()
