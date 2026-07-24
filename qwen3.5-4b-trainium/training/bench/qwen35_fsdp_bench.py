"""Qwen3.5-4B multi-core FSDP training benchmark on Trainium2 (NATIVE torch.distributed).

Launched via `torchrun --nproc_per_node N`. Uses the NATIVE Neuron PyTorch beta
(torch.device("neuron"), NOT torch_xla). The native stack registers a "neuron"
c10d ProcessGroup backend (torch_neuronx.distributed.backend) which, on
init_process_group, auto-pins NEURON_RT_VISIBLE_CORES per LOCAL_RANK, sets up the
NRT root comm, and pins NUMA.

Shards the model across N NeuronCores with upstream torch FSDP
(torch.distributed.fsdp.FullyShardedDataParallel), measures aggregate tokens/sec.

No accelerate / trl / torchtitan / torch_xla dependency (DLC lacks them + _lzma broken).
"""
import os
import time
import functools

import torch
import torch.distributed as dist

# Register the native "neuron" c10d backend. This MUST be imported before
# init_process_group(backend="neuron"). It also wires up NEURON_RT_VISIBLE_CORES
# resolution per LOCAL_RANK inside init_process_group.
import torch_neuronx
import torch_neuronx.distributed  # noqa: F401  (registers backend + collectives)

from transformers import AutoModelForCausalLM, AutoConfig

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


def _flush():
    """Flush/complete all outstanding Neuron kernels (native-stack equivalent of
    xm.mark_step()). The native beta executes eagerly, but synchronize() guarantees
    kernels finished before we time the step."""
    torch_neuronx.synchronize()


def _find_decoder_layer_classes(model):
    """Discover the repeated transformer decoder layer class(es) from the (possibly
    PEFT-wrapped) model so the FSDP auto_wrap_policy can shard per-layer.

    Qwen3.5 is a HYBRID model: its decoder ModuleList mixes linear_attention and
    full_attention layer types, which may be distinct nn.Module subclasses. We wrap
    every distinct class found in the deepest/largest 'layers' ModuleList."""
    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and len(mod) > 0 and name.endswith("layers"):
            if best is None or len(mod) > len(best):
                best = mod
    if best is None:
        raise RuntimeError("could not locate decoder ModuleList to derive layer class")
    return {type(m) for m in best}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/work/Qwen3.5-4B")
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()

    # ── Native Neuron distributed init ──────────────────────────────────────
    # torchrun sets RANK / WORLD_SIZE / LOCAL_RANK / LOCAL_WORLD_SIZE / MASTER_*.
    # The neuron backend maps LOCAL_RANK -> NEURON_RT_VISIBLE_CORES automatically
    # (one logical NeuronCore per rank). Do NOT touch the neuron device before this.
    dist.init_process_group(backend="neuron")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    dev = torch.device("neuron")

    if rank == 0:
        print(f"[cfg] world={world} model={a.model} seq={a.seq} bs={a.bs} "
              f"layers={a.layers or 'full'} mode={'LoRA' if a.lora else 'full-FT'}",
              flush=True)
    print(f"[rank {rank}] local_rank={local_rank} "
          f"NEURON_RT_VISIBLE_CORES={os.environ.get('NEURON_RT_VISIBLE_CORES')}", flush=True)

    # ── Build model on CPU (bf16, eager attn) ───────────────────────────────
    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True)
    if a.layers:
        if hasattr(cfg, "num_hidden_layers"):
            cfg.num_hidden_layers = a.layers
        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "num_hidden_layers"):
            cfg.text_config.num_hidden_layers = a.layers

    if a.layers:
        model = AutoModelForCausalLM.from_config(
            cfg, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="eager")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            a.model, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="eager")
    model.config.use_cache = False

    if a.lora:
        from peft import LoraConfig, get_peft_model
        # Qwen3.5 is HYBRID: full_attention layers expose q/k/v/o_proj; linear_attention
        # layers expose in_proj_*/out_proj. Only target modules that actually exist in
        # the (possibly truncated) model, else PEFT raises "Target modules not found".
        import torch.nn as _nn
        present = {n.split(".")[-1] for n, m in model.named_modules() if isinstance(m, _nn.Linear)}
        candidates = ["q_proj", "k_proj", "v_proj", "o_proj",          # full_attention
                      "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]  # linear_attention
        targets = [t for t in candidates if t in present]
        if rank == 0:
            print(f"[lora] target_modules={targets}", flush=True)
        lc = LoraConfig(r=a.lora_r, lora_alpha=a.lora_r * 2, lora_dropout=0.0, bias="none",
                        target_modules=targets, task_type="CAUSAL_LM")
        model = get_peft_model(model, lc)
        # PEFT creates LoRA adapters in fp32 while the base is bf16. FSDP's flat param
        # requires a UNIFORM dtype, so force the whole (base + adapter) model to bf16.
        model = model.to(torch.bfloat16)
        if rank == 0:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"[lora] trainable params = {trainable/1e6:.2f}M", flush=True)

    layer_classes = _find_decoder_layer_classes(model)
    if rank == 0:
        print(f"[fsdp] auto-wrap on {sorted(c.__name__ for c in layer_classes)}", flush=True)

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy, transformer_layer_cls=layer_classes)

    # Move to Neuron and shard with FULL_SHARD FSDP.
    # use_orig_params=True is required so LoRA's mixed frozen/trainable params
    # (different requires_grad in one module) can coexist in a flat param.
    model = model.to(dev)
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
        device_id=dev,
    )
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)

    V = getattr(cfg, "vocab_size", None) or cfg.text_config.vocab_size
    torch.manual_seed(1234 + rank)  # different data per rank (data parallel)
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
        _flush()
        loss_val = float(loss.detach().float().cpu())
        dt = time.time() - t
        if rank == 0:
            tag = "WARMUP/compile" if step == 0 else "warm"
            times.append((step, dt, loss_val))
            print(f"[step {step}] {tag} time={dt:.2f}s loss={loss_val:.4f}", flush=True)

    if rank == 0:
        warm = [t for s, t, _ in times if s > 0]
        if warm:
            avg = sum(warm) / len(warm)
            toks = a.bs * a.seq * world  # aggregate across ranks
            print(f"\n=== RESULT (FSDP world={world}) ===")
            print(f"warm_step_avg={avg:.3f}s  aggregate_tokens/step={toks}  "
                  f"aggregate_tokens/sec={toks/avg:.1f}  "
                  f"({'LoRA' if a.lora else 'full-FT'}, eager)", flush=True)
        print("BENCH_DONE", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
