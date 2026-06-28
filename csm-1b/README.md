# Sesame CSM-1B (Text-to-Speech) on Trainium

[Sesame CSM-1B](https://huggingface.co/sesame/csm-1b) — a conversational text-to-speech
model (Llama-3.2-1B backbone + depth decoder + Mimi audio codec) — running on AWS
Trainium2. Generates 24 kHz speech from text.

Two paths, by intent:

- **[vllm-omni/](vllm-omni/)** — the **vLLM-Omni serving path**: a `CsmPipeline`
  registered in the `vllm_omni_neuron` plugin (alongside `Wan22Pipeline`) for the
  `/v1/audio/speech` endpoint. Built + registered; in-container `forward` is pending a
  torch_xla version match (details inside).
- **[native-pytorch/](native-pytorch/)** — the **validated, audio-producing path**:
  a one-command `generate_speech.py` that runs CSM end-to-end on a NeuronCore (heavy
  compute offloaded), with a generated `.wav` and the latency harness. The omni
  pipeline wraps exactly this offload logic.

Core compute validated on a NeuronCore vs CPU: backbone **cosine 1.000000**
(teacher-forced, argmax 100%), Mimi codec **cosine 1.000000**.

> Runs on a single Trainium chip — a **trn2.3xlarge** is enough. Requires the
> **native-PyTorch Neuron beta** (the public beta's older torch_xla breaks CSM's
> int64 casts).

Start with **[native-pytorch/README.md](native-pytorch/README.md)** to generate speech,
or **[vllm-omni/README.md](vllm-omni/README.md)** for the serving pipeline.
