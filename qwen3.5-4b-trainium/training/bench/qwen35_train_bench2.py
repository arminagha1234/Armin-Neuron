"""Qwen3.5-4B native-PyTorch (TorchNeuron) training step benchmark on Trainium2.
Measures warm step time + tokens/sec on torch.device('neuron'), bf16, eager attn.

Modes:
  --lora    : LoRA SFT (base frozen, tiny optimizer state) -> fits full 32L on ONE core.
  (default) : full fine-tune (all weights + AdamW fp32 states) -> needs FSDP/multi-core for 4B.
  --compile : wrap model in torch.compile(backend='neuron', dynamic=False)  [Goal 2]
  --layers N: truncate to N layers for a fast smoke (0 = full 32L).
"""
import time, argparse, torch
from transformers import AutoModelForCausalLM, AutoConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/work/Qwen3.5-4B")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--lora", action="store_true", help="LoRA SFT (base frozen) - fits one core")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--compile", action="store_true", help="torch.compile(backend=neuron, dynamic=False)")
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()

    dev = torch.device("neuron")
    print(f"[cfg] model={a.model} seq={a.seq} bs={a.bs} steps={a.steps} "
          f"layers={a.layers or 'full'} mode={'LoRA' if a.lora else 'full-FT'} compile={a.compile}", flush=True)

    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True)
    if a.layers:
        if hasattr(cfg, "num_hidden_layers"): cfg.num_hidden_layers = a.layers
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "num_hidden_layers"):
            cfg.text_config.num_hidden_layers = a.layers

    print("[load] building model (bf16, eager attn)...", flush=True)
    t0 = time.time()
    if a.layers:
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True,
                    dtype=torch.bfloat16, attn_implementation="eager")
    else:
        model = AutoModelForCausalLM.from_pretrained(a.model, trust_remote_code=True,
                    dtype=torch.bfloat16, attn_implementation="eager")

    if a.lora:
        from peft import LoraConfig, get_peft_model
        lc = LoraConfig(r=a.lora_r, lora_alpha=a.lora_r*2, lora_dropout=0.0, bias="none",
                        target_modules=["q_proj","k_proj","v_proj","o_proj"], task_type="CAUSAL_LM")
        model = get_peft_model(model, lc)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[lora] trainable params = {trainable/1e6:.2f}M", flush=True)

    model = model.to(dev); model.train()
    tot = sum(p.numel() for p in model.parameters())
    print(f"[load] to-device done in {time.time()-t0:.1f}s; params={tot/1e9:.2f}B", flush=True)

    if a.compile:
        print("[compile] torch.compile(backend='neuron', dynamic=False)...", flush=True)
        model = torch.compile(model, backend="neuron", dynamic=False)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)
    V = getattr(cfg, "vocab_size", None) or cfg.text_config.vocab_size
    torch.manual_seed(0)
    ids = torch.randint(0, V, (a.bs, a.seq), device=dev)
    labels = ids.clone()

    times = []
    for step in range(a.steps):
        t = time.time()
        opt.zero_grad()
        out = model(input_ids=ids, labels=labels)
        loss = out.loss
        loss.backward()
        opt.step()
        dt = time.time() - t
        tag = "WARMUP/compile" if step == 0 else "warm"
        times.append((step, dt, float(loss.detach().float().cpu())))
        print(f"[step {step}] {tag} time={dt:.2f}s loss={times[-1][2]:.4f}", flush=True)

    warm = [t for s,t,_ in times if s > 0]
    if warm:
        avg = sum(warm)/len(warm); toks = a.bs * a.seq
        print(f"\n=== RESULT ===")
        print(f"warm_step_avg={avg:.3f}s  tokens/step={toks}  tokens/sec={toks/avg:.1f}  "
              f"(bs={a.bs} seq={a.seq}, {'LoRA' if a.lora else 'full-FT'}, "
              f"{'compiled' if a.compile else 'eager'}, single core)")
        for dtok in [1e8, 1e9]:
            hrs = (dtok / (toks/avg)) / 3600
            print(f"  extrapolate: {dtok:.0e} tokens/epoch @ 1 core = {hrs:.1f} h  (÷N with FSDP)")
    print("BENCH_DONE", flush=True)

if __name__ == "__main__":
    main()
