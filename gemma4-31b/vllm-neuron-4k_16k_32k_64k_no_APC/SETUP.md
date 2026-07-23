# SETUP — get a Trainium2 box ready (public image, no private access)

One-time environment prep before you run the benchmark. When you're done you'll have a **trn2.48xlarge**
with the Neuron driver + Docker, ready for the 4-step run in **[LAUNCH.md](./LAUNCH.md)**.

> **No private / beta access needed.** This benchmark runs on the **public** AWS Neuron vLLM image from
> Neuron's public ECR gallery (`public.ecr.aws/neuron/pytorch-inference-vllm-neuronx`). Gemma4-31B support
> is added by the [`serving_pkg/`](./serving_pkg/) in this repo — `launch_serve.sh` puts it on `PYTHONPATH`
> automatically, so there's nothing to install into the image.

## 1. Get a Trainium2 instance
- **`trn2.48xlarge`** — 16 Neuron devices / 64 cores. The README numbers are from this instance type, TP=32.
- trn2 is offered via **EC2 Capacity Blocks** (reserve capacity in advance).
- Launch it from a recent **[Neuron Base DLAMI](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/)**
  so **Docker and the Neuron driver come preinstalled and version-matched**. This avoids the #1 setup
  failure: a host driver too old for the image's runtime, which makes Neuron device init fail with a
  cryptic `NRT ... could not be initialized`.

## 2. Verify the driver is live
SSH into the box and confirm the devices show up:
```bash
neuron-ls        # expect 16 devices / 64 cores. If this errors, your DLAMI/driver is too old — update it.
```

## 3. Get the Gemma4-31B weights
Put them at `~/models/gemma-4-31b-it` (a directory containing the `*.safetensors` + tokenizer files). Two ways:
- **Hugging Face** — `google/gemma-4-31b-it` is gated; accept the license, then download with a token.
- **Provided out-of-band** — extract a tarball from your AWS/model contact into that path.

The exact download command **and the one-line tokenizer fix** are in the "Before you start" section of
**[LAUNCH.md](./LAUNCH.md)** (don't skip the tokenizer fix — it prevents a known startup crash).

## Next → run it
Everything else — checking/pulling the public image, cloning this repo, and running the cold sweep — is the
copy-paste **4-step flow in [LAUNCH.md](./LAUNCH.md)**.
