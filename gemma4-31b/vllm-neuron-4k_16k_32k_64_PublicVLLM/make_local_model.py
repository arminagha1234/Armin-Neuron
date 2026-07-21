#!/usr/bin/env python3
"""Build a host-side Gemma4-31B model dir for the public vLLM-Neuron plugin.

Symlinks the HF-cache snapshot files into /scratch/models/gemma-4-31b-it and writes a
PATCHED tokenizer_config.json (strips `extra_special_tokens`, which is a list in Gemma 4
and crashes transformers 4.57's dict-expecting tokenizer init). Idempotent.
"""
import glob
import json
import os

HUB = "/scratch/hf_cache/hub/models--google--gemma-4-31b-it/snapshots"
DST = os.path.expanduser("~/models/gemma-4-31b-it")


def main():
    snaps = [d for d in glob.glob(os.path.join(HUB, "*")) if os.path.isdir(d)]
    assert snaps, f"no snapshot under {HUB}"
    snap = snaps[0]
    print("snapshot:", snap)
    os.makedirs(DST, exist_ok=True)

    for f in os.listdir(snap):
        src = os.path.join(snap, f)
        dst = os.path.join(DST, f)
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        if f in ("tokenizer_config.json", "config.json"):
            continue  # handled below (patched real copies, not symlinks)
        os.symlink(os.path.realpath(src), dst)

    # Patched tokenizer_config.json (real file, not a symlink)
    tsrc = os.path.join(snap, "tokenizer_config.json")
    c = json.load(open(tsrc))
    removed = c.pop("extra_special_tokens", "absent")
    with open(os.path.join(DST, "tokenizer_config.json"), "w") as fh:
        json.dump(c, fh, indent=2, ensure_ascii=False)
    print("tokenizer_config: removed extra_special_tokens =", removed)

    # Text-only config.json (real file, not a symlink): drop vision/audio so the
    # vLLM-Neuron plugin loads Gemma4 as a TEXT model (we only implement the text
    # decoder). Otherwise the plugin auto-builds a vision_neuron_config and routes
    # to the multimodal from_configs(text_neuron_config=..., vision_neuron_config=...)
    # path, which our text-only factory doesn't accept.
    cfg_src = os.path.join(snap, "config.json")
    cfg = json.load(open(cfg_src))
    drop = [
        "vision_config", "audio_config",
        "image_token_id", "video_token_id", "audio_token_id",
        "boi_token_id", "eoi_token_id", "boa_token_id", "eoa_token_id",
        "eoa_token_index", "vision_soft_tokens_per_image",
    ]
    dropped = [k for k in drop if cfg.pop(k, None) is not None]
    # Promote text_config to top-level AND switch model_type to the text variant
    # ("gemma4_text") so transformers loads Gemma4TextConfig (flat fields), not the
    # multimodal Gemma4Config (which expects a nested text_config and otherwise
    # falls back to default head counts).
    tc = cfg.get("text_config", {})
    if isinstance(tc, dict):
        for k, v in tc.items():
            cfg.setdefault(k, v)
        cfg["model_type"] = tc.get("model_type", cfg.get("model_type"))
    cfg.pop("text_config", None)
    # Use the TEXT causal-LM architecture so vLLM pairs it with Gemma4TextConfig
    # (the multimodal Gemma4ForConditionalGeneration arch requires the multimodal
    # Gemma4Config). Our model class is registered under this name too.
    cfg["architectures"] = ["Gemma4ForCausalLM"]
    with open(os.path.join(DST, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2, ensure_ascii=False)
    print("config.json: dropped", dropped, "; promoted text_config; model_type=", cfg.get("model_type"))

    n_st = len(glob.glob(os.path.join(DST, "*.safetensors")))
    print(f"built {DST}  ({n_st} safetensors + configs symlinked, tokenizer patched)")


if __name__ == "__main__":
    main()
