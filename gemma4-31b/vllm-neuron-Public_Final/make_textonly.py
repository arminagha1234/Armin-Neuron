#!/usr/bin/env python3
"""Build a TEXT-ONLY Gemma4-31B model dir for the public vLLM-Neuron plugin.
Symlinks weights from the flat source dir, writes a text-only config.json
(drops vision/audio, promotes text_config, sets Gemma4ForCausalLM arch) and a
patched tokenizer_config.json. Text-only config recipe for the public vLLM-Neuron plugin."""
import glob, json, os
SRC = "/root/models/gemma-4-31b"
DST = "/root/models/gemma-4-31b-text"
os.makedirs(DST, exist_ok=True)
for f in os.listdir(SRC):
    if f in ("config.json", "tokenizer_config.json", "tokenizer_config.json.bak"):
        continue
    s = os.path.join(SRC, f); d = os.path.join(DST, f)
    if os.path.islink(d) or os.path.exists(d): os.remove(d)
    os.symlink(os.path.realpath(s), d)
# patched tokenizer_config.json (strip extra_special_tokens list)
tc = json.load(open(os.path.join(SRC, "tokenizer_config.json")))
removed = tc.pop("extra_special_tokens", "absent")
json.dump(tc, open(os.path.join(DST, "tokenizer_config.json"), "w"), indent=2, ensure_ascii=False)
# text-only config.json
cfg = json.load(open(os.path.join(SRC, "config.json")))
drop = ["vision_config","audio_config","image_token_id","video_token_id","audio_token_id",
        "boi_token_id","eoi_token_id","boa_token_id","eoa_token_id","eoa_token_index",
        "vision_soft_tokens_per_image"]
dropped = [k for k in drop if cfg.pop(k, None) is not None]
txt = cfg.get("text_config", {})
if isinstance(txt, dict):
    for k,v in txt.items(): cfg.setdefault(k,v)
    cfg["model_type"] = txt.get("model_type", cfg.get("model_type"))
cfg.pop("text_config", None)
cfg["architectures"] = ["Gemma4ForCausalLM"]
json.dump(cfg, open(os.path.join(DST, "config.json"), "w"), indent=2, ensure_ascii=False)
print("removed extra_special_tokens:", removed)
print("dropped:", dropped)
print("model_type:", cfg.get("model_type"), "arch:", cfg["architectures"])
print("safetensors:", len(glob.glob(os.path.join(DST, "*.safetensors"))))
print("DST:", DST)
