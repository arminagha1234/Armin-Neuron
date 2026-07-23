"""Download OpenGVLab/VideoMAEv2-Base and inspect the checkpoint tensor keys/shapes.
This tells us exactly what parameter names the vendored VisionTransformer must match.
"""
import math
import os

from huggingface_hub import snapshot_download
from safetensors import safe_open

REPO = "OpenGVLab/VideoMAEv2-Base"

local = snapshot_download(
    REPO,
    allow_patterns=["*.safetensors", "config.json", "preprocessor_config.json", "*.py"],
)
print("downloaded to:", local)
print("files:", sorted(os.listdir(local)))

st = os.path.join(local, "model.safetensors")
keys = []
with safe_open(st, framework="pt") as f:
    for k in f.keys():
        keys.append((k, tuple(f.get_slice(k).get_shape())))

total = sum(int(math.prod(shp)) for _, shp in keys)
print(f"\nnum tensors: {len(keys)}   total params: {total/1e6:.2f} M")

print("\n--- first 12 keys ---")
for k, shp in keys[:12]:
    print(f"  {k}   {shp}")

print("\n--- keys for block 0 (attention/mlp structure) ---")
for k, shp in keys:
    if k.startswith("model.blocks.0.") or (".blocks.0." in k):
        print(f"  {k}   {shp}")

print("\n--- non-block keys (embeds / norms / head) ---")
for k, shp in keys:
    if ".blocks." not in k:
        print(f"  {k}   {shp}")

prefixes = sorted(set(k.split(".")[0] for k, _ in keys))
print("\ntop-level prefixes:", prefixes)
print("SAVE_LOCAL_PATH:", local)
