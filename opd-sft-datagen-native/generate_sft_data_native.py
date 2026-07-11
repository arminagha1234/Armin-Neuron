"""
Lightning-OPD SFT data curation — PURE NATIVE PyTorch on AWS Trainium.

Generates teacher answers for a set of prompts with Qwen3-8B, with NO vLLM and
NO NxD-Inference. Just HuggingFace `transformers` on `torch.device("neuron")`
(the Beta-3 native stack) + a hand-rolled static-batched, KV-cached, top-p
sampling decode loop.

This is the native-PyTorch replacement for the repo's
`trainium/sft_data_generation/generate_sft_data.sh` (which uses vLLM+NxDI).

Scope / honesty:
  * "Pure native" here means a single NeuronCore (no tensor parallelism — TP on
    Neuron needs NxD, which is not "pure native"). Qwen3-8B in bf16 (~16 GB)
    fits in one trn2 logical core's HBM share.
  * Static batching (no continuous batching / paged KV cache) — so a batch runs
    until its LONGEST sequence finishes. This is correct but slower than vLLM.
    For 300k prompts prefer vLLM/NxDI; use this for a native reference / smoke /
    when you want to avoid the vLLM-Neuron plugin.

Usage:
  # CPU smoke with a tiny model (no Neuron needed) — proves the loop is correct:
  python generate_sft_data_native.py --device cpu --model sshleifer/tiny-gpt2 \
      --prompts data/prompts_smoke.jsonl --output data/out_smoke.parquet \
      --max-new-tokens 32 --batch-size 4

  # Real run on a trn2.3xlarge (single core), Qwen3-8B teacher:
  NEURON_RT_NUM_CORES=1 python generate_sft_data_native.py --device neuron \
      --model Qwen/Qwen3-8B --prompts data/openthoughts3_300000.jsonl \
      --output data/openthoughts3_300000_qwen3-8b.parquet \
      --max-new-tokens 16384 --batch-size 8 --temperature 0.7 --top-p 0.9
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-8B", help="HF teacher model id")
    p.add_argument("--prompts", required=True, help="JSONL with a 'prompt' field per line")
    p.add_argument("--output", required=True, help="Output parquet path (prompt, answer)")
    p.add_argument("--device", default="neuron", choices=["neuron", "cpu"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=16384)
    p.add_argument("--max-prompt-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="cap #prompts (0 = all); for smoke")
    p.add_argument("--no-chat-template", action="store_true",
                   help="feed the raw prompt instead of applying the chat template")
    p.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa"],
                   help="attention impl; 'eager' correctly applies the 2D padding mask "
                        "for batched left-padded decode on Neuron (sdpa can drop it)")
    return p.parse_args()


def maybe_import_torch_neuronx(device: str):
    """Register the 'neuron' device. No-op on CPU."""
    if device != "neuron":
        return
    try:
        import torch_neuronx  # noqa: F401
        print("[env] torch_neuronx imported (neuron device registered)")
    except Exception as exc:  # noqa: BLE001
        print(f"[env] WARNING: torch_neuronx import failed: {exc}")


def load_prompts(path: str, limit: int) -> list[str]:
    prompts: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            # OpenThoughts3 stores prompt as a chat list [{role, content}, ...];
            # accept a plain string too.
            if isinstance(prompt, list):
                # take the last user turn's content as the question text
                text = None
                for turn in prompt:
                    if isinstance(turn, dict) and turn.get("role") == "user":
                        text = turn.get("content")
                prompt = text if text is not None else json.dumps(prompt)
            if not isinstance(prompt, str) or not prompt:
                continue
            prompts.append(prompt)
            if limit and len(prompts) >= limit:
                break
    return prompts


def build_inputs(tokenizer, batch_prompts, use_chat_template, max_prompt_tokens, device):
    if use_chat_template and tokenizer.chat_template:
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in batch_prompts
        ]
    else:
        texts = batch_prompts
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_tokens,
    )
    return {k: v.to(device) for k, v in enc.items()}


def top_p_sample(logits, temperature, top_p):
    """Nucleus sampling on [B, vocab] logits. Returns [B] token ids.

    Sampling is done on CPU in fp32: the ops (sort/cumsum/multinomial) are tiny
    and not all reliably supported on the Neuron device, and fp32 avoids the
    degenerate bf16 distributions that make torch.multinomial raise on inf/nan.
    The heavy model forward stays on the accelerator; only this step is on CPU.
    """
    logits = logits.detach().float().cpu()
    # scrub any non-finite logits coming back from the device
    logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # keep tokens until cumulative prob exceeds top_p (always keep the first)
    keep = (cumulative - sorted_probs) <= top_p
    keep[..., 0] = True
    sorted_probs = sorted_probs * keep
    denom = sorted_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    sorted_probs = sorted_probs / denom
    choice = torch.multinomial(sorted_probs, num_samples=1)  # [B,1] into sorted space
    return sorted_idx.gather(-1, choice).squeeze(-1)


@torch.no_grad()
def generate_batch(model, tokenizer, inputs, args, eos_id):
    """Static-batched, KV-cached decode loop. Returns list[str] of new text."""
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    bsz = input_ids.shape[0]
    device = input_ids.device

    # Prefill.
    position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
    )
    past = out.past_key_values
    next_logits = out.logits[:, -1, :]

    generated = [[] for _ in range(bsz)]
    finished = torch.zeros(bsz, dtype=torch.bool, device=device)
    cur_len = attention_mask.long().sum(-1)  # per-sequence real length

    for _ in range(args.max_new_tokens):
        next_tok = top_p_sample(next_logits, args.temperature, args.top_p).to(device)  # [B]
        next_tok = torch.where(finished, torch.full_like(next_tok, eos_id), next_tok)
        for b in range(bsz):
            if not finished[b]:
                generated[b].append(int(next_tok[b]))
        finished = finished | (next_tok == eos_id)
        if bool(finished.all()):
            break
        # advance one step with the KV cache
        attention_mask = torch.cat(
            [attention_mask, torch.ones(bsz, 1, device=device, dtype=attention_mask.dtype)],
            dim=-1,
        )
        cur_len = cur_len + (~finished).long()
        step_pos = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)[:, -1:]
        out = model(
            input_ids=next_tok.unsqueeze(-1),
            attention_mask=attention_mask,
            position_ids=step_pos,
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        next_logits = out.logits[:, -1, :]

    return [tokenizer.decode(g, skip_special_tokens=True) for g in generated]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    maybe_import_torch_neuronx(args.device)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    device = torch.device(args.device)

    print(f"[cfg] model={args.model} device={args.device} dtype={args.dtype} "
          f"batch={args.batch_size} max_new={args.max_new_tokens} "
          f"temp={args.temperature} top_p={args.top_p}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"  # required for correct batched decode
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id

    print("[load] loading model weights...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation=args.attn_impl
    )
    model = model.to(device=device)
    model.eval()
    print(f"[load] model on {args.device} in {time.time()-t0:.1f}s "
          f"({sum(p.numel() for p in model.parameters())/1e9:.2f}B params)")

    prompts = load_prompts(args.prompts, args.limit)
    print(f"[data] {len(prompts)} prompts from {args.prompts}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    records = []
    n_batches = (len(prompts) + args.batch_size - 1) // args.batch_size
    for bi in range(n_batches):
        batch = prompts[bi * args.batch_size:(bi + 1) * args.batch_size]
        inputs = build_inputs(tokenizer, batch, not args.no_chat_template,
                              args.max_prompt_tokens, device)
        t0 = time.time()
        answers = generate_batch(model, tokenizer, inputs, args, eos_id)
        dt = time.time() - t0
        for prompt, answer in zip(batch, answers):
            records.append({"prompt": prompt, "answer": answer})
        done = min((bi + 1) * args.batch_size, len(prompts))
        print(f"[gen] batch {bi+1}/{n_batches}  ({done}/{len(prompts)})  {dt:.1f}s", flush=True)

    # Write parquet (fallback to jsonl if pyarrow/pandas unavailable).
    try:
        import pandas as pd
        pd.DataFrame(records).to_parquet(args.output, index=False)
        print(f"[out] wrote {len(records)} rows -> {args.output}")
    except Exception as exc:  # noqa: BLE001
        fallback = str(Path(args.output).with_suffix(".jsonl"))
        with open(fallback, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        print(f"[out] parquet write failed ({exc}); wrote jsonl -> {fallback}")


if __name__ == "__main__":
    main()
