"""Native-PyTorch training smoke test on AWS Trainium (TorchNeuron).

A tiny GPT trained from scratch for a few steps on random data. Run this FIRST to
confirm your Trainium box can do a full training step (forward + backward +
optimizer) in native PyTorch before you move on to real Llama models.

    python3 train_smoke.py

Expected: loss decreases over 10 steps and it prints TRAIN_SMOKE_OK. The first
step is slow (NEFF compile); the rest are fast (cached).
"""
import time, torch, torch_neuronx
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
device = torch.device("neuron")


class Attn(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h = h
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, D)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attn(d, h)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab=1024, d=256, h=8, L=4, T=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(T, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(L)])
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.emb(idx) + self.pos(pos)
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))


def main():
    vocab, T, B = 1024, 64, 8
    model = MiniGPT(vocab=vocab, T=T).to(device).to(torch.bfloat16)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    lossfn = nn.CrossEntropyLoss()

    torch.manual_seed(1)
    idx = torch.randint(0, vocab, (B, T), device=device)
    tgt = torch.randint(0, vocab, (B, T), device=device)

    losses = []
    for step in range(10):
        t0 = time.time()
        opt.zero_grad()
        logits = model(idx)
        loss = lossfn(logits.view(-1, vocab).float(), tgt.view(-1))
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))
        print(f"step {step:2d}  loss {losses[-1]:.4f}  ({time.time()-t0:.1f}s)", flush=True)

    print("first_loss %.4f  last_loss %.4f" % (losses[0], losses[-1]))
    print("TRAIN_SMOKE_OK" if losses[-1] < losses[0] else "TRAIN_SMOKE_LOSS_NOT_DECREASING")


if __name__ == "__main__":
    main()
