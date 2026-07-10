# Single-box zero-GPU GRPO on Trainium (periodic-sync)

A minimal, **fully-Trainium** GRPO loop that runs on **one trn2.3xlarge (4 NeuronCores),
with no GPU anywhere** — both the rollout **generation** (vLLM-Neuron) and the policy
**training** (torch-xla) time-share the single Neuron device.

This complements the main `grpo-deepmath/` example (TRL server-mode GRPO, two hosts).
Here the whole loop is squeezed onto one small box to prove the capability end-to-end.

## What it does
Per round: serve the current policy on vLLM-Neuron → generate + score completions over
HTTP → tear the server down → run a GRPO update as a subprocess → next round reloads the
new checkpoint into the server (**periodic weight sync**).

## Files
- `grpo_orchestrate.py` — device-free coordinator (never touches the Neuron device).
  Runs each phase as a subprocess that grabs and releases the device, so serving and
  training never contend for it.
- `grpo_train_phase.py` — the GRPO update as a one-shot subprocess. Uses a **single
  fixed-shape batched graph** (compiled once, reused every step) — see the header
  comment for why a per-sample loop does NOT work on torch-xla.
- `GRPO_SERVING_ON_TRAINIUM.md` — full write-up: validated results, the driver/venv
  gotchas, why per-step server-mode GRPO needs framework work, and the measured
  end-to-end run (reward 0.50 → 0.75 across one sync).

## Run
```bash
# on a trn2.3xl: public 2.29 driver, vLLM-0.16 neuron venv activated (on PATH)
python grpo_orchestrate.py --rounds 2 --prompts 4 --group 4 \
    --train-steps 3 --ckpt-root ./grpo_run
```

## Scope (honest)
Toy arithmetic task, `Qwen/Qwen3-0.6B`, tiny run. This is a **capability proof** — the
entire RL loop closes on Trainium with zero GPU and the policy measurably improves — not
a tuned training recipe. It is the **periodic-sync** GRPO variant; standard per-step
server-mode GRPO still needs two framework features (NCCL-free weight sync + live
per-tensor weight update on the neuron worker), documented in the write-up.
