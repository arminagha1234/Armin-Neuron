"""
Lightning-OPD SFT data curation — PURE NATIVE PyTorch, STATIC-SHAPE decode.

Route B: fixes the throughput problem of the naive decode. Uses a preallocated
HuggingFace `StaticCache` so every decode step has identical tensor shapes, which
means the Neuron graph compiles ONCE per (batch, length) bucket instead of once
per generated token. Also carries the two correctness fixes proven in RESULTS.md:

  * attn_implementation="eager"  -> applies the 2D key-padding mask on Neuron
                                    (SDPA drops it -> garbage on padded batches)
  * sampling on CPU in fp32      -> torch.multinomial is unreliable on-device

Output schema (--schema):
  * simple   : columns [prompt, answer]
  * messages : columns [messages, tokens]  -- drop-in for the upstream SFT step
               (run_pipeline.sh validate_parquet requires {messages, tokens}).

Usage (inside the Beta-3 DLC container):
  python generate_sft_data_bucketed.py --device neuron --model Qwen/Qwen3-8B \
    --prompts data/openthoughts3_300000.jsonl \
    --output data/openthoughts3_300000_qwen3-8b.parquet \
    --schema messages --prefill-bucket 512 --max-new-tokens 1024 \
    --batch-size 8 --dtype bfloat16 --temperature 0.7 --top-p 0.9
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
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--prompts", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="neuron", choices=["neuron", "cpu"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--prefill-bucket", type=int, default=512,
                   help="prompts are left-padded to this fixed length (static prefill shape)")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa"])
    p.add_argument("--schema", default="messages", choices=["simple", "messages"])
    p.add_argument("--no-chat-template", action="store_true")
    return p.parse_args()


def maybe_import_torch_neuronx(device):
    if device != "neuron":
        return
    try:
        import torch_neuronx  # noqa: F401
        print("[env] torch_neuronx imported")
    except Exception as exc:  # noqa: BLE001
        print(f"[env] WARNING torch_neuronx import failed: {exc}")


def load_prompts(path, limit):
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pr = row.get("prompt")
            if isinstance(pr, list):
                text = None
                for t in pr:
                    if isinstance(t, dict) and t.get("role") == "user":
                        text = t.get("content")
                pr = text if text is not None else json.dumps(pr)
            if isinstance(pr, str) and pr:
                out.append(pr)
            if limit and len(out) >= limit:
                break
    return out


def top_p_sample_cpu(logits, temperature, top_p):
    """Nucleus sampling on CPU fp32. logits: [B, vocab] (any device/dtype)."""
    logits = logits.detach().float().cpu()
    logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    if temperature <= 0:
        return logits.argmax(dim=-1)
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sp, si = torch.sort(probs, descending=True, dim=-1)
    cum = torch.cumsum(sp, dim=-1)
    keep = (cum - sp) <= top_p
    keep[..., 0] = True
    sp = sp * keep
    sp = sp / sp.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    choice = torch.multinomial(sp, num_samples=1)
    return si.gather(-1, choice).squeeze(-1)


@torch.no_grad()
def generate_static(model, tokenizer, prompt_ids, prompt_mask, args, eos_id, device, dtype):
    """
    Static-shape decode with a preallocated StaticCache.

    prompt_ids/prompt_mask: [B, prefill_bucket] (left-padded).
    Returns list[list[int]] of generated token ids per sequence.
    """
    from transformers import StaticCache

    bsz, prefill_len = prompt_ids.shape
    total_len = prefill_len + args.max_new_tokens

    cache = StaticCache(
        config=model.config, max_batch_size=bsz, max_cache_len=total_len,
        device=device, dtype=dtype,
    )

    # Full-length attention mask: valid = real prompt tokens; future/pad = 0.
    attn_full = torch.zeros(bsz, total_len, dtype=torch.long, device=device)
    attn_full[:, :prefill_len] = prompt_mask.to(torch.long)

    # --- prefill (compiles once for this prefill bucket) ---
    prefill_pos = (prompt_mask.long().cumsum(-1) - 1).clamp_min(0)
    cache_position = torch.arange(prefill_len, device=device)
    out = model(
        input_ids=prompt_ids,
        attention_mask=attn_full[:, :prefill_len],
        position_ids=prefill_pos,
        past_key_values=cache,
        cache_position=cache_position,
        use_cache=True,
    )
    next_logits = out.logits[:, -1, :]

    # position of the first generated token per sequence = number of real prompt tokens
    seq_len = prompt_mask.long().sum(-1)  # [B]
    generated = [[] for _ in range(bsz)]
    finished = torch.zeros(bsz, dtype=torch.bool, device=device)

    # --- decode (identical shapes every step -> compiles ONCE) ---
    for step in range(args.max_new_tokens):
        next_tok = top_p_sample_cpu(next_logits, args.temperature, args.top_p).to(device)
        next_tok = torch.where(finished, torch.full_like(next_tok, eos_id), next_tok)
        for b in range(bsz):
            if not finished[b]:
                generated[b].append(int(next_tok[b]))
        finished = finished | (next_tok == eos_id)
        if bool(finished.all()):
            break

        pos = prefill_len + step                      # absolute slot in the cache
        attn_full[:, pos] = 1                         # newly written token is valid
        cache_position = torch.tensor([pos], device=device)
        step_pos = (seq_len + step).unsqueeze(-1)     # [B,1] rope position
        # Pass the FULL fixed-length mask every step so the decode graph has
        # constant shapes -> compiles ONCE (a growing slice reintroduces per-step
        # recompilation). StaticCache + cache_position handle causality.
        out = model(
            input_ids=next_tok.unsqueeze(-1),
            attention_mask=attn_full,
            position_ids=step_pos,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
        )
        next_logits = out.logits[:, -1, :]

    return generated


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    maybe_import_torch_neuronx(args.device)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    device = torch.device(args.device)
    print(f"[cfg] model={args.model} device={args.device} dtype={args.dtype} "
          f"batch={args.batch_size} prefill_bucket={args.prefill_bucket} "
          f"max_new={args.max_new_tokens} attn={args.attn_impl} schema={args.schema}")

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    eos_id = tok.eos_token_id

    print("[load] loading model...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation=args.attn_impl)
    model = model.to(device=device).eval()
    print(f"[load] on {args.device} in {time.time()-t0:.1f}s "
          f"({sum(p.numel() for p in model.parameters())/1e9:.2f}B)")

    prompts = load_prompts(args.prompts, args.limit)
    print(f"[data] {len(prompts)} prompts")

    def render(p):
        if not args.no_chat_template and tok.chat_template:
            return tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        return p

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    records = []
    nb = (len(prompts) + args.batch_size - 1) // args.batch_size
    for bi in range(nb):
        batch = prompts[bi * args.batch_size:(bi + 1) * args.batch_size]
        enc = tok([render(p) for p in batch], return_tensors="pt", padding="max_length",
                  truncation=True, max_length=args.prefill_bucket)
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)
        t0 = time.time()
        gen = generate_static(model, tok, ids, mask, args, eos_id, device, dtype)
        dt = time.time() - t0
        for prompt, g in zip(batch, gen):
            answer = tok.decode(g, skip_special_tokens=True)
            if args.schema == "messages":
                messages = [{"role": "user", "content": prompt},
                            {"role": "assistant", "content": answer}]
                records.append({"messages": messages, "tokens": g})
            else:
                records.append({"prompt": prompt, "answer": answer})
        done = min((bi + 1) * args.batch_size, len(prompts))
        print(f"[gen] batch {bi+1}/{nb} ({done}/{len(prompts)}) {dt:.1f}s", flush=True)

    try:
        import pandas as pd
        pd.DataFrame(records).to_parquet(args.output, index=False)
        print(f"[out] wrote {len(records)} rows -> {args.output}")
    except Exception as exc:  # noqa: BLE001
        fb = str(Path(args.output).with_suffix(".jsonl"))
        with open(fb, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        print(f"[out] parquet failed ({exc}); wrote jsonl -> {fb}")


if __name__ == "__main__":
    main()
