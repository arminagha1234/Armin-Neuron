# SETUP — Step 1: vLLM-Neuron Beta container on Trainium2

This is the one-time environment setup. After this you'll have a running container with the Neuron
runtime + vLLM-Neuron, ready to serve Gemma4-31B. Then continue with Step 2/3 in the main
[README](./README.md).

> **Note:** the base vLLM-Neuron Beta image officially supports **Llama3** and **GPT-OSS**. **Gemma4-31B
> support is added by the [`serving_pkg/`](./serving_pkg/) in this repo** — no changes to the beta image
> are needed; `launch_serve.sh` puts `serving_pkg/` on `PYTHONPATH` automatically.

## 1a. Prerequisites
- A **Trainium2** instance — `trn2.48xlarge` (16 Neuron devices / 64 cores). Results in the README are
  from this instance type, TP=32.
- Docker installed, and AWS CLI configured.
- **Access to the vLLM-Neuron Beta image.** It lives in an AWS ECR registry — **ask your AWS account
  team** for the image URI and for ECR pull access on your account. Set it as `$BETA_IMAGE`:
  ```bash
  export BETA_IMAGE="<beta image URI from your AWS account team>"
  ```
  (We intentionally don't hardcode the registry here — your account team provides the correct URI.)

## 1b. Log in to ECR and pull the beta image
```bash
# Registry host is the part of $BETA_IMAGE before the first '/'
REGISTRY="${BETA_IMAGE%%/*}"
REGION="$(echo "$REGISTRY" | sed -E 's/.*\.dkr\.ecr\.([^.]+)\.amazonaws\.com/\1/')"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"

docker pull "$BETA_IMAGE"
```

## 1c. Install the Neuron driver (extract it from the image)
The Neuron driver is a host kernel module; the matching `.deb` ships inside the image. Extract and install
it on the host so the container can see the devices:
```bash
docker create --name neuron_tmp "$BETA_IMAGE"
docker cp neuron_tmp:/opt/aws/neuron/driver ./neuron_driver
docker rm neuron_tmp
sudo dpkg -i ./neuron_driver/*.deb        # installs aws-neuronx-dkms (builds the kernel module)
# verify the devices exist:
ls /dev/neuron*                            # expect /dev/neuron0 .. /dev/neuron15
```

## 1d. Run the container
Mount all 16 Neuron devices, persist the HF + NEFF (compiled-kernel) caches so re-runs skip recompiles,
and expose the OpenAI API port:
```bash
mkdir -p "$HOME/hf_cache" "$HOME/vllm_neff_cache"

docker run -d --name vllm_neuron \
  --device /dev/neuron0  --device /dev/neuron1  --device /dev/neuron2  --device /dev/neuron3 \
  --device /dev/neuron4  --device /dev/neuron5  --device /dev/neuron6  --device /dev/neuron7 \
  --device /dev/neuron8  --device /dev/neuron9  --device /dev/neuron10 --device /dev/neuron11 \
  --device /dev/neuron12 --device /dev/neuron13 --device /dev/neuron14 --device /dev/neuron15 \
  -v "$HOME/hf_cache:/root/.cache/huggingface" \
  -v "$HOME/vllm_neff_cache:/root/.cache/neuron" \
  -e NEURON_SKIP_EFA_AFFINITY=1 \
  -p 8000:8000 \
  --ipc=host \
  "$BETA_IMAGE" \
  sleep infinity
```
- `--device /dev/neuron0..15` — the 16 Trainium2 devices.
- `-v hf_cache` / `-v vllm_neff_cache` — persist model weights + compiled NEFFs across runs.
- `NEURON_SKIP_EFA_AFFINITY=1` — required on single-node TP.
- `-p 8000:8000` — the vLLM OpenAI-compatible API.
- `--ipc=host` — shared memory for the multi-worker TP server.

## 1e. Exec into the container
```bash
docker exec -it vllm_neuron bash
```
You're now inside the environment. Continue with **Step 2** in the [README](./README.md)
(`git clone` this repo inside the container, then `bash run_benchmark.sh`).

## Model access
`google/gemma-4-31B-it` is gated on Hugging Face — make sure you've accepted the license and have a token
available (`huggingface-cli login`, or set `HF_TOKEN`) so the weights can download into `hf_cache`.
