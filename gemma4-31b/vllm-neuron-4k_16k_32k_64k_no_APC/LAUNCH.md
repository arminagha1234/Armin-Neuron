# Run the Gemma4-31B cold-TTFT benchmark on Trainium2 — 4 steps, public image

Runs the **`google/gemma-4-31b-it`** cold-TTFT benchmark on a **trn2.48xlarge** using the **public**
AWS Neuron vLLM image from Neuron's public ECR gallery — **no private registry / beta access required**.

The flow is four commands: **(1)** check you're on the latest public image, **(2)** pull it if not,
**(3)** download this benchmark, **(4)** run it. Copy-paste top to bottom, all on the trn2 box.

---

## Before you start (one-time)

1. **A trn2.48xlarge you can SSH into**, launched from a recent **[Neuron Base DLAMI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/)**
   (Docker + the Neuron driver come preinstalled and version-matched). SSH in, then run everything below
   **on the box**. Confirm the driver is live:
   ```bash
   neuron-ls        # expect 16 devices / 64 cores. If this errors, your DLAMI/driver is too old — update it.
   ```

2. **The Gemma4-31B weights at `~/models/gemma-4-31b-it`.** Pick ONE:
   - **From Hugging Face** (accept the license first, then use a token):
     ```bash
     export HF_TOKEN=<your_hf_token>
     pip install -q "huggingface_hub[cli]"
     huggingface-cli download google/gemma-4-31b-it --local-dir ~/models/gemma-4-31b-it
     ```
   - **Provided out-of-band** (a tarball from your AWS/model contact): extract it so
     `~/models/gemma-4-31b-it/` contains the `*.safetensors` + tokenizer files.

3. **One-line tokenizer fix.** Gemma4 ships `extra_special_tokens` as a list; transformers wants a dict,
   and the mismatch crashes with `'list' object has no attribute 'keys'`. Strip it once:
   ```bash
   python3 - <<'EOF'
   import json, os, shutil
   p = os.path.expanduser("~/models/gemma-4-31b-it/tokenizer_config.json")
   shutil.copy(p, p + ".bak")
   c = json.load(open(p)); c.pop("extra_special_tokens", None)
   json.dump(c, open(p, "w"), indent=2, ensure_ascii=False)
   print("tokenizer patched:", p)
   EOF
   ```

---

## Step 1 — Check you're on the latest public vLLM-Neuron image

The image lives in Neuron's **public** ECR gallery, so **no `docker login` is needed**. List the published
tags and take the newest (highest vLLM version + highest `sdkX.YZ`):
```bash
TOKEN=$(curl -s "https://public.ecr.aws/token/?scope=repository:neuron/pytorch-inference-vllm-neuronx:pull" \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")
curl -s -H "Authorization: Bearer $TOKEN" \
  "https://public.ecr.aws/v2/neuron/pytorch-inference-vllm-neuronx/tags/list" \
  | python3 -c "import sys,json;print('\n'.join(sorted(json.load(sys.stdin)['tags'])))" | tail -10
```
As of this writing the newest is the one below — pin it to a variable you'll reuse:
```bash
export NEURON_VLLM_IMAGE="public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.21.0.1.0.0-neuronx-py313-sdk2.31.0-ubuntu24.04"
```
Now see what you already have locally:
```bash
docker images public.ecr.aws/neuron/pytorch-inference-vllm-neuronx --format '{{.Repository}}:{{.Tag}}'
```
If that already lists `$NEURON_VLLM_IMAGE`, **skip Step 2**.

## Step 2 — If you don't have it, pull it (~25 GB)

```bash
docker pull "$NEURON_VLLM_IMAGE"
```

## Step 3 — Download the benchmark

```bash
git clone https://github.com/arminagha1234/Armin-Neuron.git
cd Armin-Neuron/gemma4-31b/vllm-neuron-4k_16k_32k_64k_no_APC
```

## Step 4 — Run it

One command. It launches the container on all 16 Neuron devices, mounts your weights + this repo, and runs
the full cold sweep (4k/8k/16k/32k/64k, concurrency 1–32). The repo's `serving_pkg/` registers Gemma4 with
vLLM automatically (nothing to install). **First launch compiles the model — this takes a while; NEFFs are
cached under `~/neuron_cache` so re-runs are fast.** Results are written to `results_<timestamp>/` in this
folder on the host.
```bash
mkdir -p ~/neuron_cache
sudo docker run --rm \
  $(for i in $(seq 0 15); do echo --device /dev/neuron$i; done) \
  -v "$PWD:/bench" \
  -v "$HOME/models:/root/models" \
  -v "$HOME/neuron_cache:/root/.cache" \
  -e NEURON_SKIP_EFA_AFFINITY=1 \
  -p 8000:8000 --ipc=host \
  --entrypoint bash "$NEURON_VLLM_IMAGE" \
  -c "cd /bench && MODEL=/root/models/gemma-4-31b-it bash run_benchmark.sh"
```
> Runs for a while (compile + full sweep). Do it under `tmux`/`screen`, or append ` > run.log 2>&1 &` and
> `tail -f run.log`, so an SSH drop doesn't kill it.

**Fast smoke test first** (single size, one server compile) — same command with `ONLY=4k`:
```bash
sudo docker run --rm \
  $(for i in $(seq 0 15); do echo --device /dev/neuron$i; done) \
  -v "$PWD:/bench" -v "$HOME/models:/root/models" -v "$HOME/neuron_cache:/root/.cache" \
  -e NEURON_SKIP_EFA_AFFINITY=1 -p 8000:8000 --ipc=host \
  --entrypoint bash "$NEURON_VLLM_IMAGE" \
  -c "cd /bench && MODEL=/root/models/gemma-4-31b-it ONLY=4k bash run_benchmark.sh"
```

## Read the results

```bash
cat results_*/summary.txt          # TTFT / TPOT / E2E / throughput per input size
ls  results_*/                     # per-size JSON + CSV + serve logs
```
Full measured tables are in **[RESULTS.md](./RESULTS.md)**.

## Hit the API from your laptop (optional)

While the server is up (port 8000), from your laptop:
```bash
ssh -i <your-key>.pem -L 8000:127.0.0.1:8000 -N ubuntu@<YOUR_INSTANCE_IP>
# then, on your laptop:
curl http://localhost:8000/v1/models
```

---

## Troubleshooting (the things people actually hit)

- **`neuron-ls` fails / `NRT ... could not be initialized`** — the host Neuron driver is missing or too old
  for the image's runtime. Launch from a **recent Neuron Base DLAMI** so the driver matches the image's SDK.
- **`'list' object has no attribute 'keys'`** — you skipped the tokenizer fix in "Before you start".
- **Device busy / only some cores initialize** — a stale process holds Neuron cores. Nothing else should be
  using Neuron while this runs:
  ```bash
  neuron-ls                                  # shows which PID owns each core
  pkill -9 -f "vllm serve"; pkill -9 -f EngineCore; pkill -9 -f multiproc_executor
  ```
- **"Hangs" for many minutes on first launch** — that's the cold NEFF compile. Watch it:
  ```bash
  tail -f results_*/serve_len*.log
  ```
- **Out of disk during pull/compile** — the image is ~25 GB and the model ~60 GB. Make sure `/` (or wherever
  Docker's data-root lives) has room; on instances with a large instance-store NVMe, point Docker's
  `data-root` at it.
