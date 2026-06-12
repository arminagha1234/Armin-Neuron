"""One-time LoRA merge tool.

Loads the base diffusers pipeline + the fal Multiple-Angles LoRA, fuses
the LoRA into the base transformer weights, saves a merged snapshot we
can point the meta-init loader at.

Trade-off: a merged snapshot is locked to one LoRA. fal hot-swap is
out of scope for Phase 1; if it's a hard requirement later, runtime
LoRA application becomes a Phase 4 task.

Usage:
    python merge_lora.py \
        --base-model-path "$HF_CACHE/.../snapshots/<sha>" \
        --lora-repo "fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA" \
        --output-dir /opt/dlami/nvme/fal/merged_2511_plus_fal_lora
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


def assert_lora_targets_transformer_only(state_dict: dict) -> None:
    """Sanity check before fuse.

    Raises ValueError listing offending keys if the LoRA touches
    anything outside `transformer.*`.
    """
    bad = [k for k in state_dict.keys() if not k.startswith("transformer.")]
    if bad:
        raise ValueError(
            f"LoRA touches non-transformer modules ({len(bad)} keys). "
            f"Examples: {bad[:5]}\n"
            "Phase 1 merge path assumes transformer-only LoRA. "
            "Switch to runtime LoRA application instead."
        )


def merge_lora_into_base(
    base_model_path: str,
    lora_repo: str,
    output_dir: str,
    *,
    weight_name: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> str:
    """Fuse the LoRA into the base transformer and save merged weights.

    Loads ONLY the transformer (not the whole pipeline) to keep host
    RAM usage small. peft attaches the LoRA, fuse_lora collapses it,
    and we save the merged safetensors snapshot.

    Pre:
        - `base_model_path` is a valid HF snapshot of `Qwen-Image-Edit-2511`
        - LoRA is transformer-only (or we abort)
    Post:
        - `output_dir` contains a transformer/ subfolder with merged
          .safetensors and a config.json identical to base
        - returns the path to the merged transformer (drop-in for
          `qwen_edit_meta_loader`).
    """
    from diffusers import QwenImageTransformer2DModel
    from diffusers.loaders.lora_pipeline import QwenImageLoraLoaderMixin
    from huggingface_hub import snapshot_download
    import safetensors.torch as st

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    transformer_dir = out / "transformer"
    transformer_dir.mkdir(parents=True, exist_ok=True)

    print(f"[merge_lora] loading transformer-only from {base_model_path}")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        os.path.join(base_model_path, "transformer"),
        torch_dtype=dtype,
    )

    print(f"[merge_lora] downloading LoRA {lora_repo}")
    lora_dir = snapshot_download(repo_id=lora_repo)
    # Find the .safetensors file
    candidates = list(Path(lora_dir).glob("*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"No .safetensors in {lora_dir}")
    lora_file = candidates[0] if not weight_name else (Path(lora_dir) / weight_name)
    print(f"[merge_lora] LoRA file: {lora_file}")

    # Audit BEFORE applying — if any key isn't transformer.*, abort fast
    state = st.load_file(str(lora_file), device="cpu")
    print(f"[merge_lora] {len(state)} LoRA keys")
    assert_lora_targets_transformer_only(state)

    # Apply via diffusers' loader mixin
    print("[merge_lora] applying LoRA via diffusers mixin")
    QwenImageLoraLoaderMixin.load_lora_into_transformer(
        state_dict=state,
        transformer=transformer,
        adapter_name="default",
    )

    print("[merge_lora] fusing LoRA into base weights")
    transformer.fuse_lora()
    transformer.unload_lora()

    # Save merged
    print(f"[merge_lora] saving merged transformer to {transformer_dir}")
    transformer.save_pretrained(transformer_dir)
    print("[merge_lora] done")
    return str(transformer_dir)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model-path", required=True)
    p.add_argument("--lora-repo", default="fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--weight-name", default=None)
    args = p.parse_args()

    out = merge_lora_into_base(
        base_model_path=args.base_model_path,
        lora_repo=args.lora_repo,
        output_dir=args.output_dir,
        weight_name=args.weight_name,
    )
    print(f"\nMERGED_TRANSFORMER={out}")


if __name__ == "__main__":
    main()
