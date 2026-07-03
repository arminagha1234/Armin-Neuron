<h1 align="center">🧠⚡ Armin-Neuron</h1>

<p align="center">
  <b>Big models. Weird shapes. One pile of Trainium.</b><br>
  A field journal of frontier models taught to run on AWS Trainium2 —
  in native PyTorch, the hard way, on purpose.
</p>

<p align="center">
  <em>📷 [ Trainium2 photo — placeholder ]</em><br>
  <sub>add <code>assets/trainium.jpg</code> then replace this line with an &lt;img&gt; tag</sub>
</p>

---

## What is this?

This repo is a working zoo of large models — LLMs, MoEs, diffusion image
and video — ported to **Trainium2** with **native PyTorch + `torch_neuronx`**
(Beta 3 stack). No opinionated framework hiding the model code. Just
`torch.device("neuron")`, tensor parallelism when it gets big, and a lot
of "why is it gray" debugging until it isn't gray anymore.

Each folder is a model. Most follow the same shape: a **`native-pytorch/`**
lowest-latency path and a **`vllm-*/`** serving path, each with its own
README, benchmark, source, and results.

> Built by the **Build on Trainium** crew. If a researcher can read the
> code and change it, we did our job.

<!--
  PERSONAL PHOTO SLOT — add assets/armin-chip.jpg, then replace the
  placeholder below with:
  <p align="center"><img src="assets/armin-chip.jpg" alt="Armin Aghebrahim holding a Trainium2 chip" width="55%"></p>
-->

<p align="center">
  <em>🧑‍🔧 [ Armin Aghebrahim holding a Trainium2 chip — placeholder ]</em><br>
  <sub>add <code>assets/armin-chip.jpg</code> and swap this line for an &lt;img&gt; tag</sub>
</p>

---

## ⭐ Featured: FLUX.2-klein-4B goes high-res

The headline run. FLUX.2-klein-4B shipped at 1 MP — we pushed it up the
resolution ladder on a single trn2.48xlarge.

| Resolution | Status | How |
|---|---|---|
| 1024² (1 MP) | ✅ shipped | single-core bf16, ~4.2 s |
| 1280² (1.6 MP) | ✅ correct | fp32 + **v3 full-shard**, TP=2 |
| 1792² (3.2 MP) | ✅ **correct — new!** | fp32 + v3 full-shard, TP=4 |
| 2048² (4 MP) | 🟡 fits + runs end-to-end | fp32 + v3 full-shard, TP=8 |

The trick: split the fused SwiGLU FFN and the single-stream QKV+MLP
projections into separately-shardable linears so the **whole** model
shards — not just attention heads. That let fp32 (the precision the model
actually needs at high token counts) finally fit. 3 MP went from a blank
gray field to a real image. → [`flux2-klein-4b/`](flux2-klein-4b/)

<p align="center">
  <em>📊 [ resolution / precision / TP progress chart — placeholder ]</em>
</p>

---

## 🗂️ The model zoo

| Model | Type | Folder |
|---|---|---|
| FLUX.2-klein-4B | Image diffusion (DiT) | [`flux2-klein-4b/`](flux2-klein-4b/) |
| FLUX.2-klein-9B | Image diffusion (DiT) | [`flux2-klein-9b/`](flux2-klein-9b/) |
| LTX-2 (18.88B) | Video + audio diffusion | [`ltx2/`](ltx2/) |
| Z-Image | Text-to-image | [`z-image/`](z-image/) |
| Qwen-Image-Edit | Image editing MMDiT | [`qwen-image-edit-trainium/`](qwen-image-edit-trainium/) |
| GLM 5.1 | 754B MoE + MLA + DSA | [`glm5.1/`](glm5.1/) |
| Qwen3-30B-A3B | MoE LLM | [`qwen3-30b-a3b/`](qwen3-30b-a3b/) |
| Qwen3.5-4B | Dense LLM | [`qwen3.5-4b-trainium/`](qwen3.5-4b-trainium/) |
| Qwen3.6-27B | Dense LLM | [`qwen3.6-27b-trainium/`](qwen3.6-27b-trainium/) |
| Gemma4-31B / e4b | LLM | [`gemma4-31b/`](gemma4-31b/) · [`gemma4-e4b/`](gemma4-e4b/) |
| ModernBERT+ | Encoder | [`Modernbert+/`](Modernbert+/) |
| BERT embeddings | Encoder | [`bert-embeddings-trainium/`](bert-embeddings-trainium/) |
| ESM-2 | Protein LM (encoder) | [`esm2/`](esm2/) |
| SigLIP / SigLIP2 | Vision-language (ViT) | [`siglip/`](siglip/) |
| CLIP | Vision-language (ViT) | [`clip/`](clip/) |
| Bacformer | Bacterial-genome model (2-stage) | [`bacformer/`](bacformer/) |
| RiNALMo | RNA LM (encoder, 650M) | [`rinalmo/`](rinalmo/) |
| NKI kernels | Hand-written Trainium kernels | [`nki-kernels/`](nki-kernels/) |

*(Folders evolve — browse the repo for the current full list.)*

---

## 🧪 House rules (a.k.a. hard-won lessons)

- **Native PyTorch first.** `torch.device("neuron")`,
  `torch.compile(backend="neuron")`, `torch_neuronx.trace()`. Frameworks
  that hide the model are a last resort.
- **bf16 is great until it isn't.** Past ~1 MP the residual/softmax
  reductions need fp32 or the image collapses to gray. Ask me how I know.
- **Tensor parallelism shards weights — not the sequence.** If it still
  OOMs at more cores, you're not sharding what's actually big.
- **Verify on CPU before you burn a 15-minute compile.** Weight-split
  equivalence checks have saved more hours than they cost.

---

## 🚀 Quickstart shape

```bash
# Beta 3 DLC on a trn2.48xlarge, then per-model:
cd <model>/native-pytorch
cat README.md          # exact env + repro command lives here
```

Every model folder carries its own README with the exact env vars,
launch command, and benchmark numbers.

---

## 🖼️ Want your own photos in here?

The header currently uses a real FLUX.2 image we generated on Trainium.
If you'd rather add personal/hardware photos:

1. Drop the files into `assets/` (e.g. `armin-chip.jpg`, `trainium.jpg`).
2. Uncomment the photo block near the top of this README.

Use your own photos, or official AWS press images **with attribution**
(AWS Newsroom / re:Invent media kit). Don't commit images you don't have
rights to.

---

<p align="center"><i>Trainium2 · Native PyTorch · Build on Trainium</i></p>
