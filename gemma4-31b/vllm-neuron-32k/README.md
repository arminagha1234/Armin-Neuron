# Gemma4-31B vLLM-Neuron — WORKING 32K-in / 500-out serve (v2, place of truth)

Updated, fresh-compiling snapshot of the files + commands that serve Gemma4-31B
at **32K input + 500 output** on vLLM-Neuron (Trainium2 / trn2.48xlarge), with
the SWA windowed-gather TTFT optimization applied.

**Validated on hardware 2026-06-26** on a clean box (fresh compile, no cached
NEFFs, zero I-485 / SBUF overflow). This is the canonical copy — `vllm_32k_working/`
is the older V7 reference; this v2 is the current truth.

## Headline numbers (TP=32, fresh compile)

TTFT (median, streaming first token), 500-token output serve, chunked prefill
seg=4096, `max-num-seqs=1`:

| Input | Prompt tokens | TTFT |
|---:|---:|---:|
| 1K | 931 | 0.832 s |
| 2K | 1,849 | 0.834 s |
| 4K | 3,694 | 0.840 s |
| 8K | 7,384 | 0.850 s |
| 16K | 14,755 | 0.873 s |
| **32K** | **28,813** | **0.918 s** |

- **TTFT is essentially flat (~0.83 s → ~0.92 s) from 1K to 32K input.**
- Decode throughput: **~2.9 tok/s** (batch-1; bound by the head_dim>128 decode path).
- Correctness: **7/7** — needle "BANANA-7731" retrieved @ 5/50/95% depth across ~27K ctx.

Before the SWA windowed gather, TTFT was ~1.64 s (1K) → 1.74 s (32K); windowing
cut it ~1.9×, with no kernel-compile risk (pure PyTorch path).

## What makes this work (3 changes on top of the canonical gemma4 model)

1. **Edit B — `gemma4/model.py` `forward_prefill`**: gate the chunked path on the
   INT `kv_segment_size` only (compile-time constant), calling
   `NF.segmented_attention`. The stock model gated on the TENSOR `cached_seq_len > 0`,
   which Dynamo can't trace → blocked chunked prefill above 16K. (Single-shot
   prefill is capped at 16384; >16K MUST be chunked.)
2. **Edit A — `attention_segmented_cte.py` `segmented_attention()`**: route
   head_dim > 128 (gemma4 is 256 SWA / 512 global) to the trace-safe PyTorch
   fallback instead of raising. The NKI segmented kernel caps head_dim at 128.
3. **SWA windowed gather (1b) — `attention_segmented_cte.py`
   `_torch_segmented_attention_impl`**: for the 49/60 sliding-window layers
   (window 1024), gather a STATIC number of KV blocks at a DYNAMIC offset
   (`index_select`) instead of the full padded span, with absolute-position
   masks. Global layers (11/60) keep the full causal gather. This is the ~1.9×
   TTFT win and is pure PyTorch (cannot cause I-485).

## Container image (vLLM-Neuron private beta, v5 / sdk2.30 / vLLM 0.19.0)
```
421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-28ce3c3:pytorch-2.10-inference-neuron-py312-sdk2.x.x-ubuntu24.04-neuron-ops-release-2.30-vllm-neuron-private-beta-trn10-v5
```
Install the **matched 2.30 driver** from the image (host driver mismatch was a
real failure mode — a 2.28 host driver against the 2.30 runtime is part of what
produced the earlier I-485 confusion).

## Files
| file | what it is |
|---|---|
| `gemma4/` | model package — `model.py` (canonical + edit B), `attention_decode_kernel.py`, `config.py`, `factory.py`, `__init__.py` |
| `attention_segmented_cte.py` | patched segmented wrapper (edit A + SWA windowed `_torch_segmented_attention_impl`) — deploy over the container's site-packages copy |
| `patch_segmented_cte.py` | minimal in-container patcher for edit A only (alternative to deploying the whole file) |
| `gemma4_register.py`, `gemma4_transformers_stub.py`, `sitecustomize.py` | model registration via PYTHONPATH (no vLLM fork) |
| `make_local_model.py` | builds `/root/models/gemma-4-31b-it` w/ patched tokenizer (dynamic snapshot resolve) |
| `launch.sh` | serve launcher: `bash launch.sh LEN BUCKETS SEG TP MNS` |
| `bench_ttft.py` | TTFT sweep (the table above) |
| `bench_32k.py` | full 32K-in/500-out TTFT + decode-throughput bench |
| `test_serving.py` | 7/7 correctness incl. needle@5/50/95% across ~27K |

## Reproduce (from cold box)

### 0. Box + storage
trn2.48xlarge (16 Neuron devices). Mount instance-store for space and point
Docker's containerd store there (the image + 62GB model won't fit on a 96G root):
```bash
sudo mkfs.ext4 -F /dev/nvme1n1 && sudo mkdir -p /scratch && sudo mount /dev/nvme1n1 /scratch
sudo systemctl stop docker docker.socket containerd
sudo rm -rf /var/lib/containerd && sudo mkdir -p /scratch/containerd
sudo ln -s /scratch/containerd /var/lib/containerd
echo '{ "data-root": "/scratch/docker" }' | sudo tee /etc/docker/daemon.json
sudo systemctl start containerd docker
```

### 1. Pull image + install matched driver
```bash
IMG=421672808698.dkr.ecr.us-east-1.amazonaws.com/concourse-release-28ce3c3:pytorch-2.10-inference-neuron-py312-sdk2.x.x-ubuntu24.04-neuron-ops-release-2.30-vllm-neuron-private-beta-trn10-v5
aws ecr get-login-password --region us-east-1 | sudo docker login --username AWS --password-stdin 421672808698.dkr.ecr.us-east-1.amazonaws.com
sudo docker pull "$IMG"
sudo docker create --name d "$IMG" && sudo docker cp d:/opt/aws/neuron/driver/. /scratch/driver/ && sudo docker rm d
sudo dpkg -i /scratch/driver/aws-neuronx-dkms_*.deb
sudo rmmod neuron; sudo modprobe neuron        # confirm: modinfo neuron | grep ^version
```

### 2. Start container (16 devices, /scratch-backed caches)
```bash
sudo docker run -d --privileged --name vllm_neuron \
  -v /scratch/hf_cache:/root/.cache/huggingface -v /scratch/neff_cache:/root/.cache/vllm \
  -v /scratch/work:/work --env HF_TOKEN=$HF_TOKEN --env NEURON_SKIP_EFA_AFFINITY=1 \
  -p 8000:8000 --ipc=host \
  $(for i in $(seq 0 15); do echo --device /dev/neuron$i; done) "$IMG" sleep infinity
```

### 3. Deploy this package + model
```bash
sudo docker cp gemma4/ vllm_neuron:/work/pkg/gemma4/
sudo docker cp gemma4_register.py vllm_neuron:/work/pkg/
sudo docker cp gemma4_transformers_stub.py vllm_neuron:/work/pkg/
sudo docker cp sitecustomize.py vllm_neuron:/work/pkg/
sudo docker cp make_local_model.py vllm_neuron:/work/pkg/
# edit A — patched segmented wrapper (windowed):
sudo docker cp attention_segmented_cte.py \
  vllm_neuron:/opt/conda/lib/python3.12/site-packages/vllm_neuron/functional/attention/attention_segmented_cte.py
sudo docker cp launch.sh bench_ttft.py vllm_neuron:/work/
# model:
sudo docker exec vllm_neuron python3 -c "from huggingface_hub import snapshot_download,os; snapshot_download('google/gemma-4-31b-it', token=os.environ['HF_TOKEN'])"
sudo docker exec vllm_neuron python3 /work/pkg/make_local_model.py
```

### 4. Serve (32K in + 500 out → max-model-len 36864) and test
```bash
sudo docker exec vllm_neuron bash -lc 'cd /work && bash launch.sh 36864 4096 4096 32 1'
# wait for "Application startup complete." then:
sudo docker exec vllm_neuron python3 /work/test_serving.py   # 7/7
sudo docker exec vllm_neuron python3 -u /work/bench_ttft.py  # the TTFT table
```

## Gotchas (hard-won)
- `max-model-len` caps **input + output combined** → 32K in + 500 out needs ≥ 33,268 (we use 36,864).
- Chunked prefill (SEG < LEN) is REQUIRED above 16K; single-shot is capped at 16384.
- `num_batched_tokens_buckets` last value must equal `--max-num-batched-tokens`.
- **Tar the NEFF cache the moment it works** (`/scratch/neff_cache`). Never clear
  `/tmp/nki_cache` without a backup — that's what exposed a latent overflow before.
- The I-485 / `NCC_IGCA037` (274016 B) overflow seen previously was an old box's
  corrupted runtime + driver mismatch, NOT this recipe. A clean box + matched
  2.30 driver compiles the stock NKI model fresh with zero overflow.

## Next levers (not yet applied — see ../vllm_32l_explore/results/TTFT_OPTIMIZATION.md)
- 1a bf16 matmuls in the fallback (~2× on the bmms, still pure torch).
- `_V2_PREFILL` d-tiled flash NKI prefill kernel (already in `model.py` for the
  non-chunked branch; compiled clean at 4K) for ≤16K single-shot sub-500ms.
- d-tile the segmented NKI kernel (highest ceiling, real kernel work, I-485 surface).
