# Phase 3 — HTTP serving wrap

Thin FastAPI front-end over the production compile+real-RoPE pipeline
from `run_compiled.py`. Reuses the same TP plan, meta-init loader, and
real-valued RoPE — no duplication of model construction logic
(per Requirement 8 AC#3).

## Architecture

```
                  ┌──────────────────────────────────────┐
                  │  uvicorn server.py (single process)  │
client ──HTTP──▶  │  /edit /health                       │
                  │  asyncio.Lock around socket          │
                  └─────────────┬────────────────────────┘
                                │ Unix socket
                                │ /tmp/fal_pipeline.sock
                                ▼
                  ┌──────────────────────────────────────┐
                  │  torchrun --nproc_per_node=4         │
                  │  worker.py                           │
                  │  ├─ rank 0: socket reader            │
                  │  └─ rank 1-3: lockstep with rank 0   │
                  │     via barrier + shared file        │
                  └──────────────────────────────────────┘
```

The worker is the persistent torchrun process. It loads the transformer
+ pipeline ONCE (~30s for the streamed weights + first cold compile)
and then loops on requests forever. The server is a separate uvicorn
process that handles HTTP and forwards each request as one line of
newline-terminated JSON over a Unix socket.

Single in-flight request is enforced by an `asyncio.Lock` in the
server. The worker's all-ranks-same-pipeline invariant
(see `.kiro/steering/neuron-tp-on-beta2.md`) is preserved via
`dist.barrier()` + shared `/tmp/fal_request.json` handoff to fan rank
0's request out to ranks 1-3.

## Files

| File | Purpose |
|---|---|
| `worker.py` | torchrun-launched persistent pipeline process |
| `server.py` | FastAPI app — `/edit` and `/health` |
| `launch_worker.sh` | torchrun launcher, mirrors `run_compiled_28step.sh` env |
| `launch_server.sh` | uvicorn launcher (port 8000 by default) |
| `test_client.py` | Smoke-test client; sends the same fixed-seed input as `run_compiled_28step.sh` |

## Smoke-test workflow

```bash
# Inside the fal_beta2 container, two terminals.

# Terminal 1: start the worker (foreground; takes ~5 min cold to compile)
bash /work/path_c/serve/launch_worker.sh
# Wait for: "[worker r0] pipeline ready, entering serve loop"

# Terminal 2: start the server
bash /work/path_c/serve/launch_server.sh

# Terminal 3 (or local): smoke-test
curl http://localhost:8000/health
# {"status":"ok","socket":"/tmp/fal_pipeline.sock"}

cd /work/path_c
python serve/test_client.py --output results/serve_test_output.png
# Should produce a 512×512 PNG; latency ~170s if worker is warm.
```

## Validation gate (Phase 3.0 acceptance)

Per Requirement 8 AC#1:
- `serve_test_output.png` should have cosine ≥ 0.9999 vs
  `results/run_compiled_28step/output.png` (same seed, same input,
  same steps).
- p99 latency ≤ 90s at 28 steps 512×512.

The 90s p99 target is ambitious — current warm baseline is ~169s.
Phase 2.5's deferred VAE-on-Neuron two-phase loading is the most
likely path to closing that gap. For Phase 3.0 ship we document the
actual measured p99 and flag the gap.

## Anti-patterns (do not change)

- ❌ Don't add multiple uvicorn workers — the worker process serves
  one request at a time. Multiple HTTP workers hitting one Unix
  socket would deadlock on the FastAPI lock.
- ❌ Don't try to skip rank 1-3 by gating on `if rank == 0:` — every
  rank must run the full pipeline call (the all-ranks-same-pipeline
  invariant). The barrier + shared-file handoff is the working pattern.
- ❌ Don't reach for `dist.broadcast_object_list` — the Beta 2
  `'neuron'` PG doesn't support object collectives. The shared-file
  handoff is correct.
