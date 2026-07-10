"""GRPO training phase (one subprocess): load model + rollouts, fixed-shape PG
update, save new policy, EXIT (releasing the Neuron device). Run by the coordinator.

Design note (why ONE batched graph, not a per-sample loop):
  An earlier version looped over samples with mark_step() after each backward().
  On torch-xla that produced a *fresh* compiled graph per sample (~4.5 min each
  on trn2), so a single training step never finished. This version runs the whole
  fixed-shape batch [N, L] through ONE forward + ONE backward, so exactly one
  train graph is compiled and then reused across every step. bf16 keeps the vocab
  logits tensor small ([N, L-1, V] in bf16) and the reduction is done in fp32 for
  numerical stability.
"""
import argparse, json
import torch
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--rollouts", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--train-steps", type=int, default=4)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--maxlen", type=int, default=96)
args = ap.parse_args()

dev = xm.xla_device()
tok = AutoTokenizer.from_pretrained(args.model)
tok.pad_token = tok.pad_token or tok.eos_token
rollouts = json.load(open(args.rollouts))  # [{prompt, comps:[...], rewards:[...]}]

L = args.maxlen
seqs, adv_list, masks = [], [], []
for r in rollouts:
    rs = torch.tensor(r["rewards"], dtype=torch.float32)
    adv = rs - rs.mean()
    plen = len(tok(r["prompt"]).input_ids)
    for comp, a in zip(r["comps"], adv.tolist()):
        enc = tok(r["prompt"] + comp, truncation=True, max_length=L,
                  padding="max_length", return_tensors="pt")
        ids = enc.input_ids[0]; real = int(enc.attention_mask[0].sum())
        m = torch.zeros(L - 1); m[max(plen - 1, 0):max(real - 1, 0)] = 1.0
        seqs.append(ids); adv_list.append(a); masks.append(m)

# Fixed-shape batch: [N, L]. One graph, reused every step.
batch = torch.stack(seqs).to(dev)                                   # [N, L] int
advb = torch.tensor(adv_list, dtype=torch.float32).to(dev)          # [N]
maskb = torch.stack(masks).to(dev)                                  # [N, L-1] f32
N = batch.shape[0]
print(f"[train] batch={tuple(batch.shape)} N={N} L={L}", flush=True)

model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to(dev)
model.train()
opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

targets = batch[:, 1:]                                              # [N, L-1]
for step in range(args.train_steps):
    opt.zero_grad(set_to_none=True)
    logits = model(input_ids=batch).logits[:, :-1, :]              # [N, L-1, V] bf16
    lf = logits.float()                                            # fp32 for stable reduce
    gathered = lf.gather(-1, targets.unsqueeze(-1)).squeeze(-1)    # [N, L-1]
    tok_lp = gathered - torch.logsumexp(lf, dim=-1)                # [N, L-1]
    comp_lp = (tok_lp * maskb).sum(-1) / maskb.sum(-1).clamp(min=1)  # [N]
    loss = -(advb * comp_lp).mean()                                # scalar
    loss.backward()
    opt.step()
    xm.mark_step()                                                 # one materialize per step
    print(f"[train] step {step} obj={float(loss.detach().to('cpu')):.4f}", flush=True)

model.to(torch.float32)  # save in fp32 so the serving stack reloads cleanly
model.save_pretrained(args.out); tok.save_pretrained(args.out)
print(f"[train] saved -> {args.out}", flush=True)
