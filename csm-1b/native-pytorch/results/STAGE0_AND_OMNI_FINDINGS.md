# CSM-1B — Stage-0 Latency + vLLM-Omni Container Findings (2026-06-28)

## Part 1 — Stage-0 warm latency (single NeuronCore, fp32, current offload)
Measured with `src/bench_ttft.py` (warm pass + measure pass), 8 frames.

| Component | Warm latency |
|---|---|
| T_prefill (backbone over prompt) | **546 ms** |
| T_backbone_step (per frame, median) | **48 ms** |
| T_depth_total (per frame, 31 steps) | **291 ms** (~9.4 ms/step, CPU) |
| T_codec_1frame (Mimi decode) | **273 ms** |
| **TTFA_est (streaming first audio)** | **~1158 ms** |
| full 8-frame wall (non-stream) | 3476 ms |
| real-time budget @12.5 fps | 80 ms/frame |

### Reading
- ~1.16 s TTFA today — far from 100 ms / 500 ms. But the shape of the numbers is the
  point: **T_prefill (546 ms) and T_codec_1frame (273 ms) are implausibly large for the
  actual compute** (a short-prompt backbone forward and a 1-frame conv decode). They're
  dominated by **per-call host↔device sync + fp32 + first-shape recompile**, not math.
- **T_depth_total (291 ms) = 31 serial CPU steps** — the structural inner-loop cost.
- This validates the optimization plan's ranking: the wins are (1) streaming so TTFA is
  one frame not the whole clip, (2) bf16, (3) fixed-shape compiled graphs (kill
  recompiles), (4) fewer host syncs / on-device depth loop. The current offload path is
  sync/precision/recompile-bound, exactly what those levers remove.

### Implied target after the ladder (single core)
If prefill/codec drop to true compute (~tens of ms each) + bf16 halves backbone/depth +
streaming emits frame 0: TTFA into the low-hundreds ms single-core; <100 ms needs the
on-device depth loop and likely TP=2–4. <500 ms is clearly reachable.

## Part 2 — vLLM-Omni CsmPipeline in-container: root-caused blocker
The `CsmPipeline` is built + registered in `vllm_omni_neuron` (verified). Running its
`forward` **inside the omni beta container fails** — root cause now precisely
identified: **the container ships torch_xla 2.10.0 (newer than the validated 2.9.0),
which has multiple strictness regressions** that CSM's code trips. Hit, in order, each
patched and still cascading:

1. mask `attention_mask.to(device=xla, dtype=bool)` — fixed by pre-casting to bool.
2. RoPE `position_ids[:, None, :].float()` — int64→float `.float()` raises
   `Expected self.dtype()==dst.dtype()` (`_to_copy` regression). Mul-cast workaround
   got past it.
3. `torch.autocast(device_type="xla", enabled=False)` — xla AMP backend missing
   `get_amp_supported_dtype`. Pointed at "cpu" (no-op since disabled).
4. `rotate_half` `torch.cat((-x2, x1))` — `cat` requires contiguous on 2.10.
5. **`x[..., :h].contiguous()` itself raises `Expected self.is_contiguous()`** — i.e.
   on torch_xla 2.10 the standard contiguity workaround is *also* broken.

### Verdict
This is a **torch_xla 2.10 runtime regression**, not a pipeline-logic problem. Per-line
patching is futile when `.contiguous()` itself fails. The **same CsmPipeline compute
runs end-to-end and produces audio on torch_xla 2.9** (the native-PyTorch beta /
DLAMI), validated cosine 1.0.

### Fix (for the omni serving path)
Run the omni plugin on a **torch_xla 2.9 runtime** — i.e. an omni beta image built on
2.9, or pin torch/torch_xla to 2.9 in the container (needs vllm-omni 0.19 compat
check). This is an environment/runtime task for the omni image, not a model change.
The CsmPipeline source is correct and registered; it just needs the 2.9 runtime the
rest of the CSM work used.
