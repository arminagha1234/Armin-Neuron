#!/usr/bin/env python3
"""Port autoresearch to AWS Trainium2 (Neuron Beta 3).

Patches train.py and prepare.py to replace CUDA-specific code with
Neuron equivalents. Run this once after cloning the repo.

Changes:
  train.py:
  - Replace Flash Attention 3 with F.scaled_dot_product_attention
  - Replace device="cuda" with device="neuron"
  - Remove autocast (Neuron does bf16 natively via model dtype)
  - Replace torch.cuda.synchronize with torch.neuron.synchronize
  - Replace torch.compile() with torch.compile(backend="neuron")
  - Replace H100 peak flops with Trn2 estimate
  - Remove torch.cuda.max_memory_allocated
  - Remove torch.cuda.get_device_capability

  prepare.py:
  - Replace device="cuda" with device="neuron"
  - Remove pin_memory (not supported on Neuron)
"""
import re
import sys
from pathlib import Path

def patch_train(path: Path):
    src = path.read_text()

    # 1. Remove the FA3 kernel loading block (lines 20-24)
    src = re.sub(
        r"from kernels import get_kernel\n"
        r"cap = torch\.cuda\.get_device_capability\(\)\n"
        r".*?fa3 = get_kernel\(repo\)\.flash_attn_interface\n",
        "# NEURON PORT: using F.scaled_dot_product_attention instead of FA3\n"
        "import contextlib\n",
        src,
        flags=re.DOTALL,
    )

    # 2. Replace fa3.flash_attn_func call with SDPA
    # FA3 signature: fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)
    # where q,k,v are (B, T, H, D) — need to transpose for SDPA which expects (B, H, T, D)
    old_fa3 = "        y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)"
    new_sdpa = """        # NEURON PORT: use SDPA (no Flash Attention on Trainium)
        q_t = q.transpose(1, 2)  # (B, H, T, D)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=True)
        y = y.transpose(1, 2)  # back to (B, T, H, D)"""
    src = src.replace(old_fa3, new_sdpa)

    # 3. Replace device
    src = src.replace('device = torch.device("cuda")', 'device = torch.device("neuron")')

    # 4. Replace autocast
    src = src.replace(
        'autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)',
        '# NEURON PORT: no autocast needed, model runs in bf16 natively\n'
        'autocast_ctx = contextlib.nullcontext()'
    )

    # 5. Replace peak flops (Trn2 single core ~190 TFLOPS bf16)
    src = src.replace(
        "H100_BF16_PEAK_FLOPS = 989.5e12",
        "# NEURON PORT: Trn2 single core ~190 TFLOPS bf16\n"
        "H100_BF16_PEAK_FLOPS = 190e12  # Trn2 estimate"
    )

    # 6. Replace torch.cuda.synchronize
    src = src.replace("torch.cuda.synchronize()", "torch.neuron.synchronize()")

    # 7. Replace torch.compile
    src = src.replace(
        "model = torch.compile(model, dynamic=False)",
        '# NEURON PORT: compile with neuron backend\n'
        'model = torch.compile(model, backend="neuron", dynamic=False)'
    )

    # 8. Replace peak VRAM
    src = src.replace(
        "peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024",
        "peak_vram_mb = 0  # NEURON PORT: no CUDA memory tracking"
    )

    # 9. Replace MFU reference name in prints (cosmetic)
    src = src.replace("H100_BF16_PEAK_FLOPS", "TRN2_BF16_PEAK_FLOPS")
    # Fix the variable name we already replaced
    src = src.replace(
        "TRN2_BF16_PEAK_FLOPS = 190e12  # Trn2 estimate",
        "TRN2_BF16_PEAK_FLOPS = 190e12  # Trn2 single-core estimate"
    )

    # 10. float32 matmul precision — keep it, works on Neuron
    # torch.set_float32_matmul_precision("high") is fine

    path.write_text(src)
    print(f"  patched {path}")


def patch_prepare(path: Path):
    src = path.read_text()

    # Replace device="cuda" with device="neuron"
    src = src.replace('device="cuda"', 'device="neuron"')

    # Remove pin_memory=True (Neuron doesn't support pinned memory)
    src = src.replace(", pin_memory=True", "")

    path.write_text(src)
    print(f"  patched {path}")


def main():
    root = Path(__file__).parent
    print("Porting autoresearch to Trainium2 (Neuron Beta 3)...")
    patch_train(root / "train.py")
    patch_prepare(root / "prepare.py")
    print("Done. Run: uv run prepare.py && uv run train.py")


if __name__ == "__main__":
    main()
