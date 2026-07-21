# Trainium 2 + Gemma 4-31B — line-by-line launch runbook

Copy-paste, top to bottom. Each block is labeled:
- **[LAPTOP]** = run on your Mac/laptop
- **[INSTANCE]** = run on the trn2 box after you SSH in (on the host, not in a container)
- **[CONTAINER]** = run inside the vLLM-Neuron container

Assumes you already have a **trn2.48xlarge** you can SSH to.
If you need to *create* one: trn2 is Capacity-Blocks-only — see "Appendix: launch an instance".

End state: Gemma 4-31B served on Trainium via vLLM-Neuron, TTFT/TPOT/E2E numbers for
4k/16k/32k/64k × concurrency 1–32.

---

## 0. [LAPTOP] Set your connection variables
Fill in your instance IP and key path, then paste. Everything below reuses these.
```bash
export TRN_HOST=ubuntu@<YOUR_INSTANCE_IP>
export TRN_KEY=~/.ssh/<your-key>.pem
```

Test the connection:
```bash
ssh -i "$TRN_KEY" -o StrictHostKeyChecking=accept-new "$TRN_HOST" 'echo connected; uname -a'
```

## 1. [LAPTOP] SSH into the instance
```bash
ssh -i "$TRN_KEY" "$TRN_HOST"
```
Everything from here until "PORT FORWARD" is run **on the instance**.

---

## 2. [INSTANCE] Set up storage (only if the root disk is < ~200 GB)
The beta image (~25 GB) + the model (~62 GB) + compiled kernels won't fit on a small root
disk. Put Docker's storage on the instance-store NVMe. (Skip if your root disk is large.)
```bash
lsblk                                   # find the big unmounted NVMe (often /dev/nvme1n1)
sudo mkfs.ext4 -F /dev/nvme1n1
sudo mkdir -p /scratch && sudo mount /dev/nvme1n1 /scratch
sudo systemctl stop docker docker.socket containerd 2>/dev/null || true
sudo rm -rf /var/lib/containerd && sudo mkdir -p /scratch/containerd
sudo ln -s /scratch/containerd /var/lib/containerd
echo '{ "data-root": "/scratch/docker" }' | sudo tee /etc/docker/daemon.json
sudo systemctl start containerd docker
```

## 3. [INSTANCE] Set the beta image URI (persist it so new shells keep it)
```bash
echo 'export BETA_IMAGE="421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-28ce3c3:pytorch-2.10-inference-neuron-py312-sdk2.x.x-ubuntu24.04-neuron-ops-release-2.30-vllm-neuron-private-beta-trn10-v5"' >> ~/.bashrc
source ~/.bashrc
echo "$BETA_IMAGE"
```

## 4. [INSTANCE] Log in to ECR and pull the image
> **One-time access:** the image is in AWS account `421672808698`. Your instance's IAM role
> needs `AmazonEC2ContainerRegistryReadOnly`, AND your AWS account must be added to that repo's
> policy by the AWS Neuron/account team (send them your account ID). Otherwise `docker pull`
> returns `pull access denied`.
```bash
aws sts get-caller-identity        # sanity: prints your account + instance role
aws ecr get-login-password --region us-east-1 \
  | sudo docker login --username AWS --password-stdin 421672808698.dkr.ecr.us-east-1.amazonaws.com
sudo docker pull "$BETA_IMAGE"
sudo docker images | grep concourse-release-28ce3c3     # confirm it's local
```

## 5. [INSTANCE] Install the matching Neuron driver (idempotent)
```bash
if neuron-ls >/dev/null 2>&1; then
  echo "Neuron driver already OK — skipping"
else
  sudo docker create --name neuron_tmp "$BETA_IMAGE"
  sudo docker cp neuron_tmp:/opt/aws/neuron/driver ./neuron_driver
  sudo docker rm neuron_tmp
  sudo apt-get update -qq && sudo apt-get install -y -qq dkms build-essential
  sudo dpkg -i ./neuron_driver/*.deb
  sudo modprobe neuron
fi
neuron-ls                          # expect 16 devices / 64 cores on trn2.48xlarge
```

## 6. [INSTANCE] Get the Gemma 4-31B weights
The model must live at `~/models/gemma-4-31b-it`. Pick ONE:

**Option A — download from Hugging Face** (needs a token + accepted license):
```bash
export HF_TOKEN=<your_hf_token>
pip install -q "huggingface_hub[cli]"
huggingface-cli download google/gemma-4-31B-it \
  --local-dir ~/models/gemma-4-31b-it --local-dir-use-symlinks False
```

**Option B — weights provided out-of-band** (your AWS/Hippocratic contact gives you a tarball):
```bash
mkdir -p ~/models
# copy/extract the provided weights so this path exists:
ls ~/models/gemma-4-31b-it        # must contain *.safetensors + tokenizer files
```

## 7. [INSTANCE] Patch the tokenizer (ONE line — prevents a known crash)
Gemma 4 ships `extra_special_tokens` as a list; the container's transformers wants a dict and
crashes `'list' object has no attribute 'keys'`. Strip it once:
```bash
python3 - <<'EOF'
import json, os, shutil
p = os.path.expanduser("~/models/gemma-4-31b-it/tokenizer_config.json")
shutil.copy(p, p + ".bak")
c = json.load(open(p))
print("removed:", c.pop("extra_special_tokens", None))
json.dump(c, open(p, "w"), indent=2, ensure_ascii=False)
print("patched", p)
EOF
```

## 8. [INSTANCE] Start the container (all 16 devices + caches + model + API port)
```bash
mkdir -p "$HOME/hf_cache" "$HOME/vllm_neff_cache"
sudo docker run -d --name vllm_neuron \
  $(for i in $(seq 0 15); do echo --device /dev/neuron$i; done) \
  -v "$HOME/models:/root/models" \
  -v "$HOME/hf_cache:/root/.cache/huggingface" \
  -v "$HOME/vllm_neff_cache:/root/.cache/neuron" \
  -e NEURON_SKIP_EFA_AFFINITY=1 \
  -p 8000:8000 --ipc=host \
  "$BETA_IMAGE" sleep infinity
sudo docker ps        # confirm vllm_neuron is Up
```

## 9. [INSTANCE] Exec into the container
```bash
sudo docker exec -it vllm_neuron bash
```
Everything below runs **inside the container**.

---

## 10. [CONTAINER] Clone the benchmark repo
```bash
cd /root
git clone https://github.com/arminagha1234/Armin-Neuron.git
cd Armin-Neuron/gemma4-31b/vllm-neuron-4k_16k_32k_64k
```

## 11. [CONTAINER] Run the benchmark
Points at the local weights you prepared. It launches the Gemma 4 server per input size
(first launch compiles — 5–15 min each, cached after), runs concurrency 1–32, writes results.
```bash
MODEL=/root/models/gemma-4-31b-it bash run_benchmark.sh
```
Fast smoke test (one size first, ~10 min):
```bash
MODEL=/root/models/gemma-4-31b-it ONLY=4k bash run_benchmark.sh
```

## 12. [CONTAINER] Read the results
```bash
cat results_*/summary.txt          # TTFT / TPOT / E2E / throughput per input size
ls  results_*/                     # per-size JSON + CSV + serve logs
```
Ballpark TTFT (trn2.48xlarge, TP=32, concurrency 1): ~0.41 s @4k, ~0.46 s @16k, ~0.50 s @32k,
~0.68 s @64k. Full tables are in the repo's `RESULTS.md`.

---

## PORT FORWARD — hit the API from your laptop (optional)
The server speaks the OpenAI API on port 8000. From your **[LAPTOP]** (new terminal):
```bash
ssh -i "$TRN_KEY" -L 8000:127.0.0.1:8000 -N "$TRN_HOST"
```
Then, on your laptop:
```bash
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/completions -H 'Content-Type: application/json' \
  -d '{"model":"gemma4","prompt":"The capital of France is","max_tokens":16}'
```

---

## Troubleshooting (the things people actually hit)
- **`pull access denied` (step 4)** — your AWS account isn't on the beta ECR repo policy yet, or
  the instance role lacks ECR read. Not a typo. Get added by the AWS Neuron/account team.
- **`'list' object has no attribute 'keys'`** — you skipped step 7 (tokenizer patch).
- **`NRT ... could not be initialized` / device busy** — a stale process holds a Neuron core.
  ```bash
  neuron-ls                    # shows which PID owns each chip
  sudo docker exec vllm_neuron bash -lc 'pkill -9 -f "vllm serve"; pkill -9 -f EngineCore; pkill -9 -f multiproc_executor'
  ```
  Don't run a notebook/other Neuron process while the server is up.
- **Server "hangs" for 10-15 min on first launch** — that's the cold NEFF compile; it's cached
  afterward. Watch it: `sudo docker exec vllm_neuron tail -f /root/Armin-Neuron/gemma4-31b/vllm-neuron-4k_16k_32k_64k/results_*/serve_len*.log`
- **`BETA_IMAGE: unbound variable`** — you opened a new shell; `source ~/.bashrc` (step 3).
- **Out of disk during pull/compile** — do step 2 (put Docker on the big NVMe).

---

## Appendix: launch a fresh trn2 instance (if you don't have one)
trn2 is **Capacity-Blocks-only** (not on-demand). Reserve one, then launch with an IAM instance
profile that has ECR read:
```bash
# from [LAPTOP], AWS CLI configured
aws ec2 describe-capacity-block-offerings --instance-type trn2.48xlarge \
  --capacity-duration-hours 24 --region us-east-2
# after purchasing a Capacity Block, run-instances with:
#   --iam-instance-profile Name=<role-with-AmazonEC2ContainerRegistryReadOnly>
#   --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":1000,"VolumeType":"gp3"}}]'
```
Then start at step 1.
