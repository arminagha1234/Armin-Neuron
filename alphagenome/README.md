# AlphaGenome on Trainium

[AlphaGenome](https://www.nature.com/articles/s41586-025-10014-0) (Google DeepMind's
DNA sequence model) running on AWS Trainium2 via the community PyTorch port.

Predicts hundreds of genomic tracks (ATAC, DNase, CAGE, RNA-seq, ChIP, contact maps,
splice sites) at single-base resolution from DNA sequences up to 131,072 bp.

- **[native-pytorch/](native-pytorch/)** — step-by-step guide + scripts to run it on
  a Trainium instance. Working at the full 131,072-bp window; all track heads match a
  CPU reference to ~6 decimals.

Start here if you've never used Trainium: **[native-pytorch/README.md](native-pytorch/README.md)**.
