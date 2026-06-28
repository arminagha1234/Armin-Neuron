# Sesame CSM-1B (Text-to-Speech) on Trainium

[Sesame CSM-1B](https://huggingface.co/sesame/csm-1b) — a conversational text-to-speech
model (Llama-3.2-1B backbone + depth decoder + Mimi audio codec) — running on AWS
Trainium2.

- **[native-pytorch/](native-pytorch/)** — step-by-step guide + scripts to run CSM-1B
  TTS on a Trainium instance. Heavy compute (backbone + Mimi codec) on a NeuronCore,
  validated **cosine 1.000000** vs CPU. Includes a one-command `generate_speech.py`
  and a `CsmPipeline` for the vLLM-Omni Neuron plugin.

Start here: **[native-pytorch/README.md](native-pytorch/README.md)**.

> Runs on a single Trainium chip — a **trn2.3xlarge** is enough. Requires the
> **native-PyTorch Neuron beta**.
