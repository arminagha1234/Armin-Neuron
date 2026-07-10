# GRPO generation (rollout serving) on Trainium — VALIDATED

Proves the missing piece for a **fully-Trainium GRPO** loop: the policy model's
rollout **generation** runs on Trainium via vLLM-Neuron (no GPU).

**Validated:** 2026-07-10, trn2.3xlarge (Melbourne), 4 NeuronCores / 96 GB, LNC2.

## Result
`Qwen/Qwen3-0.6B` served on vLLM-Neuron, TP=2, and answered correctly:
```
prompt:  "Solve: what is 12 times 8? The answer is"
output:  " 96. ..."   (temperature 0, 24 tokens)   ← 12×8=96 ✓
health=200, OpenAI-compatible /v1/completions on :8001
```

## Environment (host venv, NOT the beta native-PyTorch DLC)
- vLLM-Neuron venv: `/opt/aws_neuronx_venv_pytorch_inference_vllm_0_16` (vLLM **0.16.0**,
  neuron platform plugin `vllm_neuron`, NxD Inference backend)
- **Driver must be the PUBLIC `aws-neuronx-dkms 2.29.0.0`** + matching public
  `aws-neuronx-runtime-lib` / `aws-neuronx-collectives`.

## Two gotchas that cost us (write them down)
1. **Activate the venv — don't call the binary by absolute path.** Running
   `/opt/.../bin/vllm serve` fails with `FileNotFoundError: 'libneuronpjrt-path'`
   because torch_xla shells out to the `libneuronpjrt-path` helper, which must be on
   `PATH`. Fix: `source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate`
   first.
2. **Driver/NEFF version match.** With the **Beta-3 driver (2.28.0.0)** installed (from
   the Clay native-PyTorch work), the model compiled but failed to load at warmup:
   `NRT_UNSUPPORTED_NEFF_VERSION` / "Unsupported NEFF Version". The public vLLM-0.16
   compiler emits a NEFF the 2.28 beta driver can't load. Fix: restore the public
   driver:
   ```
   sudo apt-get install -y --allow-downgrades aws-neuronx-dkms=2.29.0.0 \
       aws-neuronx-runtime-lib aws-neuronx-collectives aws-neuronx-tools
   sudo rmmod neuron && sudo modprobe neuron    # no reboot needed
   ```
   > NOTE: this box now runs the PUBLIC 2.29 driver. To go back to the Clay/native
   > beta stack, reinstall the cached beta debs in `~/workspace/runtime_artifacts/`
   > (dkms 2.28 + runtime-lib 2.32.19 + collectives 2.32.16) and reload the module.

## Working serve command
```bash
source /opt/aws_neuronx_venv_pytorch_inference_vllm_0_16/bin/activate
vllm serve Qwen/Qwen3-0.6B \
  --tensor-parallel-size 2 --max-num-seqs 4 --max-model-len 2048 \
  --block-size 128 --no-enable-prefix-caching --port 8001
# first run compiles per bucket (Compiler status PASS x N); then Application startup complete.
curl http://localhost:8001/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"...","max_tokens":24,"temperature":0}'
```

## What this unlocks & what's still open
- **Unlocked:** the GRPO *policy generation* can run on Trainium — so a GRPO example
  need not ship any GPU code for inference.
- **Still open (integration):** TRL GRPO server-mode talks to a `trl vllm-serve`
  endpoint with a **weight-sync protocol** (init_communicator / update_named_param over
  the `vllm_group_port`), not plain OpenAI `/v1/completions`. To close the loop we need
  either (a) `trl vllm-serve` to drive the neuron backend, or (b) a small adapter that
  bridges TRL's weight-sync + generate calls to this vLLM-Neuron server. The hard
  capability question ("can the policy model generate on Trainium?") is now **yes**;
  wiring TRL's weight sync to it is the remaining work.


---

# Path A tested — disk/checkpoint weight-reload (2026-07-10, trn2.3xl)

Goal: can we update the vLLM-Neuron server's policy weights *without a full recompile*,
fast enough for a GRPO loop? Measured it directly.

## What the loader does (from `vllm_neuron/.../neuronx_distributed_model_loader.py`)
- Weights are loaded via a **path-based** `load_weights(model_name_or_path, ...)` →
  `model.load(compiled_path)`. Weights are part of the precompiled artifact bundle;
  **there is no per-tensor `update_named_param` / live hot-swap** on the neuron worker.
- The vLLM artifacts dir is wiped per start, BUT the underlying **neuronx-cc NEFF
  cache** (`/var/tmp/neuron-compile-cache`) persists → a warm start skips the expensive
  compile.

## Measured (Qwen3-0.6B, TP=2, LNC2)
| Scenario | Time to healthy | Notes |
|---|---|---|
| Cold serve (first compile) | ~4–5 min | many `Compiler status PASS` |
| **Warm restart (NEFF cache hit)** | **~2 min** | **1** compile pass; dominated by imports + HLO trace + checkpoint **shard + DMA to device** ("SKIPPING pre-sharding … sharded during load time") |
| Generation after warm reload | correct (`2+2=`→`4`) | model works post-reload |

A dedicated `/reload_weights` (re-DMA on an already-live worker, skipping imports/trace)
would be **faster than the 2-min full restart** — but still a **full checkpoint
re-shard + DMA (tens of seconds)**, because there is no per-tensor update.

## Verdict
- **Per-step GRPO (standard):** ❌ impractical. A training step is ~seconds; a
  full-checkpoint reload is tens of seconds → the loop would be dominated by weight
  reload. No per-tensor hot-swap exists to make per-step sync cheap.
- **Periodic-sync GRPO (every N steps):** ✅ feasible as a variant — reload the policy
  into the server every N steps, accept more off-policy rollouts. Uses only supported
  path-based `load_weights`. This is a different (valid) algorithm knob, not standard
  per-step GRPO.
- **The real per-step unlock** = a native **per-tensor weight-update** path in
  vLLM-Neuron (accept a broadcast tensor and DMA it into the compiled model's weight
  slot without re-shard/recompile) + a **non-NCCL transport** for the trainer→server
  broadcast (TRL's is NCCL/XCCL only). Both are **framework work → Neuron-team ask**,
  not a wire-up.

## Bottom line for the customer
- ✅ Proven: policy generation runs on Trainium (vLLM-Neuron).
- ✅ Proven: GRPO training runs on Trainium (deepmath, FSDP eager).
- ⚠️ The *coupling* (live weight sync) is the gap: standard per-step server-mode GRPO
  needs (a) NCCL-free weight sync and (b) live per-tensor weight update on the neuron
  worker — neither exists today. Zero-GPU GRPO is achievable now only in a
  **periodic-sync** form; per-step needs the two framework features above.


---

# END-TO-END zero-GPU GRPO loop — WORKING (2026-07-10, trn2.3xl)

The full periodic-sync loop now runs to completion on a **single 4-core trn2.3xlarge,
zero GPU**, with a real learning signal. Both phases (vLLM-Neuron generation +
torch-xla policy update) time-share the one Neuron device.

## Result (2 rounds, 4 prompts × group 4 = 16 rollouts/round, 3 train-steps/round)
| Round | Policy served | MEAN_REWARD | Train obj (step0→2) |
|---|---|---|---|
| 0 | `Qwen/Qwen3-0.6B` (base) | **0.500** | −0.030 → −0.113 → −0.191 |
| 1 | `round0` ckpt (reloaded into vLLM) | **0.750** | −0.053 → −0.141 → −0.240 |

`[DONE] periodic-sync GRPO completed on Trainium (zero GPU).`

Reward improved **0.50 → 0.75** across one sync — reloading the trained checkpoint into
the server produced measurably better rollouts. (Toy arithmetic task, tiny run — the
point is the *loop closes and moves the policy*, on-device, no GPU anywhere.)

## Architecture that made it work
Two scripts, coordinated so **only one process holds the Neuron device at a time**:

- **`grpo_orchestrate.py`** — device-FREE coordinator. Never imports torch_xla. Per
  round: `free_device()` → start vLLM-Neuron server (subprocess) → generate+score over
  HTTP `/v1/completions` → kill server + `free_device()` → run training subprocess →
  next round uses the saved checkpoint (periodic weight sync).
- **`grpo_train_phase.py`** — training subprocess. Grabs the device, does the GRPO
  update, saves the new policy, EXITS (guaranteeing release).

`free_device()` kills any `/dev/neuron0` holder and waits until the device is free, so
the serving and training phases never contend.

## Two failure modes we hit and fixed (write them down)
1. **Leaked vLLM `EngineCore` / orphaned `neuronx-cc` compilers held the device** across
   phases → next phase couldn't acquire it. Fix: `free_device()` (`fuser -k /dev/neuron0`
   + wait) between every phase, and kill stray `neuronx-cc`/`walrus_driver` before a run.
2. **Per-sample training loop caused per-sample recompiles.** The first version looped
   over the 16 samples calling `backward()` + `mark_step()` each iteration. On torch-xla
   that emitted a *fresh compiled graph per sample* (~4.5 min each on trn2) — a single
   train step never finished (9+ distinct graphs and counting). **Fix: one fixed-shape
   batched graph** `[N, L]=[16,96]`, one forward + one backward per step, in **bf16**
   with the vocab reduction (`logsumexp`) done in fp32. Now the train graph compiles
   **once** (~1–2 s per small graph, ~90 s total incl. model load) and is reused every
   step. See `grpo_train_phase.py` header comment.

## Run it
```bash
# on the trn2.3xl, public 2.29 driver, vLLM-0.16 venv on PATH
python grpo_orchestrate.py --rounds 2 --prompts 4 --group 4 \
    --train-steps 3 --ckpt-root /home/ubuntu/grpo_run
```

## Honest scope
- This is the **periodic-sync** GRPO variant (reload the whole checkpoint into the
  server between rounds), NOT standard per-step server-mode GRPO. Per-step still needs
  the two framework features noted above (NCCL-free weight sync + live per-tensor update
  on the neuron worker).
- Tiny toy task, fp32 save / bf16 train, 0.6B model — this is a *capability* proof
  (the whole RL loop runs on Trainium with no GPU), not a tuned training recipe.
