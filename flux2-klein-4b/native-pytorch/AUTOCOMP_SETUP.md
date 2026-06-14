# Autocomp (UC Berkeley) — running on Trainium2 Beta 3

**Date:** 2026-06-14
**Box:** `3.15.152.199` (trn2.48xl, Beta 3 container `beta3`)
**Autocomp:** UC Berkeley's LLM-driven NKI kernel optimizer
(https://github.com/ucb-bar/autocomp, arXiv 2505.18574)

## Status: WORKING end-to-end on our box

Got Autocomp's full optimization loop running on Trainium2 Beta 3:
LLM planning (Claude via Bedrock) → code generation → NKI compile →
on-device benchmark → beam search.

First proof run (MHA reference kernel, trn-advanced-nki1 prob 8):
```
2 iterations in 4.4 minutes
Baseline: 0.845 ms
Best:     0.844 ms
Speedup:  1.00x (no real gain at this tiny search budget on an
                 already-optimized flash kernel)
```

The 1.00× is expected: 2 iters / 2 plans / 2 candidates is a minimal
budget, and the MHA reference is already a tuned flash kernel. Most
generated candidates failed to compile (normal for early iterations).
The point of this run was to prove the infra works — it does.

## Two fixes required to make it run on this box

### 1. Use the NKI v1 agent/problem, not v2
The Beta 3 container has NO `torch_xla` (by design — it uses
`torch.device("neuron")`, not XLA). Autocomp's NKI v2 backend harness
imports `from torch_xla.core import xla_model`, so the v2 path fails
with `ModuleNotFoundError: torch_xla`.

NKI v1 (`built:trn2-nki1` + `trn-advanced-nki1`) runs baremetal via
`neuronxcc.nki` + numpy — no torch_xla. That's the path that works.

### 2. Pass the AWS session token to the Bedrock client (CODE FIX)
`autocomp/common/llm_utils.py` built the `AnthropicBedrock` (and boto
`bedrock-runtime`) clients with only `aws_access_key` + `aws_secret_key`,
NOT the session token. With temporary STS creds (Isengard
assumed-role), the session token is mandatory — without it Bedrock
returns 403 "security token is invalid."

Fix (3 edits in llm_utils.py):
- read `aws_session_token = _get_key("AWS_SESSION_TOKEN")` at module level
- pass `aws_session_token` to `AnthropicBedrock`/`AsyncAnthropicBedrock`
  when present
- pass `aws_session_token` to `boto3.client("bedrock-runtime", ...)`

This is a general bug in autocomp for anyone using temporary AWS creds.
Worth upstreaming to ucb-bar/autocomp.

## How to run (reproducible)

On the box, in the beta3 container, with fresh AWS creds (from
creds.json, ~hourly refresh — Isengard STS):

```bash
docker exec \
  -e WANDB_MODE=disabled \
  -e AWS_REGION=us-east-1 \
  -e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=... -e AWS_SESSION_TOKEN=... \
  -e NEURON_RT_VIRTUAL_CORE_SIZE=2 \
  beta3 bash -lc "cd /mnt/data/work/autocomp && python -m autocomp.search.run_search"
```

Config in `autocomp/search/run_search.py`:
```python
backend_name = "trn"
agent_name   = "built:trn2-nki1"      # NKI v1 (no torch_xla needed)
hw_config    = TrnHardwareConfig("trn2.48xlarge")
prob_type    = "trn-advanced-nki1"
prob_id      = 8                       # MHA
models       = ["aws::us.anthropic.claude-opus-4-5-20251101-v1:0"]
iterations   = 2                       # bump to 8-10 for real gains
```

## To point it at the FLUX attention kernel (next step)

Autocomp optimizes a registered "problem" = a reference kernel +
test harness. To target our FLUX attention:
1. Add `sols/trn-advanced-nki1/9_flux_attn_ref.py` — the FLUX attention
   as a standalone NKI v1 kernel (adapt from our flux2_attention_cte.py,
   but the v1 baremetal numpy I/O contract).
2. Add `harnesses/trn-advanced-nki1/9_flux_attn_test.py` with a
   `// SUBSTITUTE HERE` marker and a numpy correctness check.
3. Set `prob_id = 9`, bump iterations to ~8.

Caveat: our earlier finding stands — FLUX attention at single-rank is
not the bottleneck (the DiT loop is saturated; host-CPU is the wall).
So even a faster attention kernel won't move the 6.86s end-to-end much.
Autocomp is more valuable pointed at a kernel that IS the bottleneck —
e.g. if a future workload is attention-bound, or for the VAE conv
kernels (where per-block compile was slower than CPU — autocomp might
find a better tiling).

## Artifacts
- `/mnt/data/work/autocomp/` on the box — full install, configured
- `results/autocomp_mha_run.log` — the 2-iteration proof run
- llm_utils.py session-token fix (local copy in .tmp/autocomp)

Box clean, no instance stopped.
