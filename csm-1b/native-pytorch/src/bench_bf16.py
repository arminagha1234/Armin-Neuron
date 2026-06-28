"""CSM bf16 vs fp32 on the backbone (Neuron): correctness (teacher-forced) + speed.

Tests whether bf16 (with CsmRMSNorm kept fp32) preserves codebook-0 logits and how
much faster the backbone step is. The model's massive activations (~1e16) collapsed
under full bf16 before; this checks if fp32 norms are enough.
"""
import time, torch
import torch_xla.core.xla_model as xm
from transformers import AutoProcessor, CsmForConditionalGeneration

MODEL = "/scratch/csm/csm_1b"


def keep_norms_fp32(model):
    for m in model.modules():
        if "norm" in type(m).__name__.lower():
            m.float()


def time_fwd(model, seq, dev, n=5):
    model = model.to(dev)
    # warm (compile)
    with torch.no_grad():
        _ = model(input_ids=seq.to(dev)); xm.mark_step()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(input_ids=seq.to(dev))
        lg = out.logits; xm.mark_step(); lg = lg.float().cpu()
        ts.append((time.perf_counter() - t0) * 1000)
    import statistics
    return lg, statistics.median(ts)


def main():
    proc = AutoProcessor.from_pretrained(MODEL)
    # fixed teacher-forced sequence (text prompt; backbone cb0 path)
    inputs = proc("[0]Hello from Trainium latency test.", add_special_tokens=True, return_tensors="pt")
    seq = inputs["input_ids"]
    dev = xm.xla_device()

    print("[fp32] backbone on Neuron...")
    m32 = CsmForConditionalGeneration.from_pretrained(MODEL, dtype=torch.float32).eval()
    lg32, t32 = time_fwd(m32, seq, dev)
    del m32

    print("[bf16+fp32norms] backbone on Neuron...")
    m16 = CsmForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16).eval()
    keep_norms_fp32(m16)
    lg16, t16 = time_fwd(m16, seq, dev)

    import torch.nn.functional as F
    cos = F.cosine_similarity(lg32.flatten(), lg16.flatten(), dim=0).item()
    am = (lg32.argmax(-1) == lg16.argmax(-1)).float().mean().item()
    print("\n========== CSM backbone: bf16 vs fp32 (Neuron) ==========")
    print(f"  fp32 forward (median)        : {t32:7.1f} ms")
    print(f"  bf16+fp32norms forward (med) : {t16:7.1f} ms   ({t32/t16:.2f}x)")
    print(f"  logits cosine fp32-vs-bf16   : {cos:.6f}")
    print(f"  argmax agreement             : {am*100:.1f}%")
    print(f"  bf16 logits std (collapse?)  : {lg16.std().item():.4e} (fp32 {lg32.std().item():.4e})")
    print("=========================================================")


if __name__ == "__main__":
    main()
