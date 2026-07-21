# LAUNCH — Gemma4-31B on PUBLIC GA vLLM-Neuron, step by step

Copy-paste, top to bottom. Do the blocks **in order**. Each block is labeled:
- **[LAPTOP]** = run on your Mac/laptop
- **[INSTANCE]** = run on the trn2 box after you SSH in (on the host, NOT in a container)
- **[CONTAINER]** = run inside the vLLM-Neuron container

**What you get at the end:** Gemma4-31B answering prompts on Trainium, plus TTFT/TPOT/E2E
numbers for 4k/16k/32k/64k input × concurrency 1–32.

**Unlike the private-beta example, this uses a PUBLIC image — there is no ECR login, no
account allowlisting, no special access. Anyone can pull it.**

You need: a **trn2.48xlarge** running the **Neuron 2.31 DLAMI** (Deep Learning AMI — comes with
the Neuron driver preinstalled) that you can SSH into. If you don't have one, see
"Appendix A: get an instance" at the bottom, then come back to Step 0.

---

## 0. [LAPTOP] Set two variables (your instance address + SSH key)
Fill in the two values, then paste. Everything below reuses them.
```bash
export TRN_HOST=ubuntu@<YOUR_INSTANCE_PUBLIC_DNS_OR_IP>
export TRN_KEY=~/.ssh/<your-key>.pem
```
Test it connects:
```bash
ssh -i "$TRN_KEY" -o StrictHostKeyChecking=accept-new "$TRN_HOST" 'echo CONNECTED; uname -a'
```
You should see `CONNECTED`. If not, fix `TRN_HOST` / `TRN_KEY` before going on.

## 1. [LAPTOP] SSH into the instance
```bash
ssh -i "$TRN_KEY" "$TRN_HOST"
```
Everything from here until "PORT FORWARD" runs **on the instance**.

## 2. [INSTANCE] Check the Neuron devices are visible
```bash
neuron-ls
```
Expect **16 devices / 32 cores** (trn2.48xlarge). If `neuron-ls` is "command not found", you
are not on a Neuron DLAMI — use the Neuron 2.31 DLAMI (Appendix A).

## 3. [INSTANCE] Make a big scratch disk for the model + compiled kernels
The model (~62 GB) + compiled graphs won't fit on a small root disk. Put them on the instance
NVMe. (If your root disk is already ≥ 400 GB you can `sudo mkdir -p /scratch` and skip the rest.)
```bash
lsblk                                    # find the big unmounted NVMe (often /dev/nvme1n1)
sudo mkfs.ext4 -F /dev/nvme1n1           # ONLY if it's empty/unused
sudo mkdir -p /scratch && sudo mount /dev/nvme1n1 /scratch
sudo chown "$USER" /scratch
df -h /scratch                           # confirm hundreds of GB free
```

## 4. [INSTANCE] Pull the PUBLIC vLLM-Neuron image (no login needed)
```bash
export IMG="public.ecr.aws/neuron/pytorch-inference-vllm-neuronx:0.21.0.1.0.0-neuronx-py313-sdk2.31.0-ubuntu24.04"
sudo docker pull "$IMG"
sudo docker images | grep vllm-neuronx    # confirm it's local
```
This is a public image — if the pull says "no basic auth credentials", just retry; no
`docker login` is required for `public.ecr.aws`.

## 5. [INSTANCE] Start the container (all 16 devices + scratch + API port)
```bash
sudo docker run -d --name vllm_public \
  $(for i in $(seq 0 15); do echo --device /dev/neuron$i; done) \
  --cap-add SYS_ADMIN --cap-add IPC_LOCK --ipc=host \
  -v /scratch:/scratch \
  -e NEURON_SKIP_EFA_AFFINITY=1 \
  -e HF_HOME=/scratch/hf_cache \
  -e VLLM_CACHE_ROOT=/scratch/neff_public \
  -p 8000:8000 \
  "$IMG" sleep infinity
sudo docker ps        # confirm vllm_public is Up
```

## 6. [INSTANCE] Download the Gemma4-31B weights (into scratch)
You need a Hugging Face token and to have accepted the Gemma license on the model page.
```bash
sudo docker exec -it vllm_public bash -lc '
  export HF_HOME=/scratch/hf_cache
  pip install -q "huggingface_hub[cli]"
  huggingface-cli login    # paste your HF token when prompted
  huggingface-cli download google/gemma-4-31b-it
'
```
This downloads ~62 GB into `/scratch/hf_cache`. It only happens once (cached on /scratch).

## 7. [CONTAINER] Get this example into the container
```bash
sudo docker exec -it vllm_public bash
```
Now you are **inside the container**. Get the code:
```bash
cd /root
git clone https://github.com/arminagha1234/Armin-Neuron.git
cd Armin-Neuron/gemma4-31b/vllm-neuron-4k_16k_32k_64_PublicVLLM
ls    # you should see install_public.sh, make_local_model.py, run_benchmark_public.sh, ...
```

## 8. [CONTAINER] Install the Gemma4 model into the plugin (one command)
```bash
bash install_public.sh
```
Expect it to end with:
```
[install] OK — registered: ['Gemma4ForConditionalGeneration', 'Gemma4ForCausalLM']
[install] done.
```

## 9. [CONTAINER] Build the local model directory (one command)
```bash
python3 make_local_model.py
```
Expect it to end with something like `built /root/models/gemma-4-31b-it (... safetensors ...)`.

## 10. [CONTAINER] Smoke test — serve 4k and ask it a question (~5 min first time)
Start the server (first launch compiles the model, ~5–15 min; cached afterward):
```bash
MODEL=/root/models/gemma-4-31b-it \
LEN=5120 SEG=512 BUCKETS=512 MNS=32 KV_CACHE_DTYPE=auto APC=1 \
  bash launch_serve_public.sh
```
Wait until it prints `server READY`. Then, in the **same** container shell, ask it something:
```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"gemma4","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":16,"temperature":0}'
```
You should see **Paris** in the response. 🎉 If you do, the model works.

> IMPORTANT: this is an **instruction-tuned** model. Always use `/v1/chat/completions`
> (the chat endpoint). A bare `/v1/completions` prompt with no chat template will ramble/repeat
> — that's normal for an -IT model, not a bug.

## 11. [CONTAINER] Run the full benchmark (all sizes, all concurrencies)
```bash
MODEL=/root/models/gemma-4-31b-it bash run_benchmark_public.sh
```
This launches a server for each input size (4k → 16k → 32k → 64k), sweeps concurrency
1,2,4,8,16,32, and writes results. First compile of each size takes 10–20 min; total run is a
few hours. To try just one size first:
```bash
ONLY=4k MODEL=/root/models/gemma-4-31b-it bash run_benchmark_public.sh
```

## 12. [CONTAINER] Read the results
```bash
cat results_*/summary.txt          # TTFT / TPOT / E2E / throughput per input size
ls  results_*/                     # per-size JSON + CSV + serve logs
```
For reference numbers, see [`RESULTS.md`](./RESULTS.md).

---

## PORT FORWARD — hit the API from your laptop (optional)
From your **[LAPTOP]** in a new terminal:
```bash
ssh -i "$TRN_KEY" -L 8000:127.0.0.1:8000 -N "$TRN_HOST"
```
Then on your laptop:
```bash
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"gemma4","messages":[{"role":"user","content":"Say hi"}],"max_tokens":16}'
```

---

## Troubleshooting (the things people actually hit)
- **`neuron-ls: command not found`** — you're not on a Neuron DLAMI. Use the Neuron 2.31 DLAMI
  (Appendix A).
- **Out of disk during download/compile** — you skipped Step 3. Put `/scratch` on the big NVMe.
- **`huggingface-cli download` 401/403** — accept the Gemma license on the HF model page and
  make sure you ran `huggingface-cli login` with a valid token.
- **Server "hangs" for 10–15 min on first launch** — that's the one-time compile; it's cached
  afterward. Watch it: `tail -f /root/serve_len*_seg512_tp32.log`.
- **Reply is gibberish / repeats** — you used `/v1/completions` with a bare prompt. Use
  `/v1/chat/completions` (Step 10). Instruction-tuned models need the chat template.
- **`NRT ... could not be initialized` / device busy** — a stale process holds a Neuron core:
  ```bash
  sudo docker exec vllm_public bash -lc 'pkill -9 -f "vllm serve"; pkill -9 -f EngineCore'
  ```
- **Start over cleanly** — `sudo docker rm -f vllm_public` then redo Step 5 (weights on
  `/scratch` are kept, so you don't re-download).

---

## Appendix A: get a trn2.48xlarge
trn2 is **Capacity-Blocks-only** (not plain on-demand). From your **[LAPTOP]** with AWS CLI:
```bash
# find an available Capacity Block
aws ec2 describe-capacity-block-offerings --instance-type trn2.48xlarge \
  --capacity-duration-hours 24 --region us-east-2
# after purchasing one, launch with the Neuron 2.31 DLAMI and a big root/scratch disk:
#   --image-id <Neuron 2.31 DLAMI for your region>
#   --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":1000,"VolumeType":"gp3"}}]'
```
Find the Neuron 2.31 DLAMI AMI ID in the AWS console (search "Deep Learning AMI Neuron") or the
Neuron docs. Then start at Step 0.
