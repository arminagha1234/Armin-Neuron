# Tools

## calibrate_kv_scales.py

Per-layer FP8 KV scale calibration for Qwen3.5-4B.

Runs the HF transformers reference impl on CPU, hooks each GQA layer's
`k_proj` and `v_proj`, captures per-layer max-abs K and V over a small
calibration sample, and writes `kv_scales.json` that the model loads at
init when `KV_CACHE_DTYPE=fp8_e4m3` is enabled.

### Why per-layer calibration

Qwen3.5-4B's K/V values vary up to ~12× across the 8 GQA layers
(layer 19's V tops at 52 vs layer 27's K topping at 9.7).

Static FP8 KV scales (e.g. `scale=8` everywhere) under-utilize FP8's
dynamic range for layers with small K/V outliers and saturate the
layers with large outliers. Either failure mode produces drift and
incoherent decode.

Per-layer scaling sets `k_scale[L] = 240 * 0.9 / max_abs_K[L]` so each
layer uses ~90% of FP8's range — saturation-resistant and
precision-tight.

### Usage

```bash
# 1. Run on a CPU box that has the model and HF transformers 5.10.2:
sudo /data/cpu_venv/bin/python calibrate_kv_scales.py
# Writes /data/kv_scales.json (or wherever OUT points)

# 2. Make it visible to the serving container at:
#    /work/qwen35/kv_scales.json   (default search path)
#    or set KV_SCALES_PATH=/path/to/your/kv_scales.json

# 3. Launch with FP8 KV:
KV_CACHE_DTYPE=fp8_e4m3 ./serve.sh
```

### Output

`kv_scales.json` like:

```json
{
  "layers": {
    "3":  { "max_abs_k": 10.0, "max_abs_v":  7.5,  "k_scale": 21.5,  "v_scale": 28.7 },
    "7":  { "max_abs_k": 21.6, "max_abs_v": 17.6,  "k_scale": 10.0,  "v_scale": 12.3 },
    ...
  },
  "metadata": { ... }
}
```

`kv_scales.example.json` in this folder is the calibrated output for
Qwen3.5-4B from a 6-prompt calibration sample (2026-06-11). It works as
a starting point, but each customer should re-run calibration on a
sample matching their actual workload distribution.

### Status as of 2026-06-11

- Calibration script: working
- Model code path that loads per-layer scales: working (no-op for BF16
  KV, used when FP8 KV is enabled)
- FP8 KV serve path itself: graph-shape bug (`reshape [8, 32, 128, 128]`
  vs `(1, 32, 128, 128)`) blocks compile — to be tracked separately

For now, BF16 KV is the recommended config (working today, coherent
output, $/M numbers in HOTFIX_DELTANET_KVCACHE_2026-06-11.md).
