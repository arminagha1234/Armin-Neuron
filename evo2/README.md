# Evo 2 on Trainium

[Evo 2](https://arcinstitute.org/tools/evo) (Arc Institute) — a StripedHyena2
autoregressive DNA foundation model — running on AWS Trainium2.

- **[native-pytorch/](native-pytorch/)** — step-by-step guide + scripts to run
  **Evo2-1B** on a Trainium instance. Working with **cosine 1.000000, top-1 100%**
  vs a CPU reference. Generates embeddings / next-token logits for DNA sequences.

Start here if you've never used Trainium: **[native-pytorch/README.md](native-pytorch/README.md)**.

> **Scope:** the 1B model is validated end-to-end. The 7B and 40B checkpoints use
> the same recipe plus multi-core sharding — see `native-pytorch/results/RESULTS.md`.
