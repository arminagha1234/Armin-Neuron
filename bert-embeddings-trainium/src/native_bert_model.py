# SPDX-License-Identifier: Apache-2.0
"""
Native BERT encoder for the NKI-on-Trainium experiment.

Same forward shape as our Path A vllm-side BertEncoderModel — mean-pool over
real tokens, attention mask from input_ids != PAD, positions passed in — so a
direct A/B with vLLM is meaningful.

Toggle USE_NKI_ATTN=1 to swap in the fused NKI attention kernel.
"""
import math
import os

import torch
import torch.nn as nn


def _maybe_nki_attention():
    if os.environ.get("USE_NKI_ATTN") == "1":
        from native_nki_attention import fused_attention
        return fused_attention
    return None


class BertEmbeddings(nn.Module):
    def __init__(self, hidden, vocab, max_pos, type_vocab, eps, pad_id):
        super().__init__()
        self.word = nn.Embedding(vocab, hidden, padding_idx=pad_id)
        self.pos = nn.Embedding(max_pos, hidden)
        self.tok_type = nn.Embedding(type_vocab, hidden)
        self.ln = nn.LayerNorm(hidden, eps=eps)

    def forward(self, ids, type_ids, pos_ids):
        e = self.word(ids) + self.pos(pos_ids) + self.tok_type(type_ids)
        return self.ln(e)


class BertSelfAttention(nn.Module):
    def __init__(self, hidden, n_heads, head_dim):
        super().__init__()
        self.n = n_heads
        self.d = head_dim
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)

    def _shape(self, x, b, s):
        return x.view(b, s, self.n, self.d).transpose(1, 2)

    def forward(self, x, mask):
        b, s, _ = x.shape
        q = self._shape(self.q(x), b, s)
        k = self._shape(self.k(x), b, s)
        v = self._shape(self.v(x), b, s)

        nki = _maybe_nki_attention()
        if nki is not None:
            out = nki(q, k, v, mask)
            return out.transpose(1, 2).reshape(b, s, self.n * self.d)

        # Native attention (baseline)
        scale = 1.0 / math.sqrt(self.d)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        if mask is not None:
            scores = scores + mask
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)
        return out.transpose(1, 2).reshape(b, s, self.n * self.d)


class BertLayer(nn.Module):
    def __init__(self, hidden, n_heads, head_dim, inter, eps):
        super().__init__()
        self.attention = BertSelfAttention(hidden, n_heads, head_dim)
        self.attn_out = nn.Linear(hidden, hidden)
        self.attn_ln = nn.LayerNorm(hidden, eps=eps)
        self.intermediate = nn.Linear(hidden, inter)
        self.output = nn.Linear(inter, hidden)
        self.out_ln = nn.LayerNorm(hidden, eps=eps)

    def forward(self, x, mask):
        a = self.attn_out(self.attention(x, mask))
        x = self.attn_ln(x + a)
        g = self.intermediate(x)
        gelu = 0.5 * g * (1.0 + torch.tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)))
        h = self.output(gelu)
        return self.out_ln(x + h)


class BertEncoder(nn.Module):
    """Match the math of our Path A vllm-side BertEncoderModel.

    forward(ids, positions, attention_mask) where:
      - ids: [B, S]
      - positions: [B, S] long
      - attention_mask_real: [B, S] (1.0 for real, 0.0 for pad)  — built outside
    Returns mean-pooled embedding [B, hidden].
    """
    def __init__(self, hf_config, dtype):
        super().__init__()
        self.dtype = dtype
        self.hidden = hf_config.hidden_size
        self.n_heads = hf_config.num_attention_heads
        self.head_dim = self.hidden // self.n_heads
        self.embeddings = BertEmbeddings(
            self.hidden, hf_config.vocab_size, hf_config.max_position_embeddings,
            hf_config.type_vocab_size, hf_config.layer_norm_eps, hf_config.pad_token_id,
        )
        self.layers = nn.ModuleList([
            BertLayer(self.hidden, self.n_heads, self.head_dim,
                      hf_config.intermediate_size, hf_config.layer_norm_eps)
            for _ in range(hf_config.num_hidden_layers)
        ])

    def forward(self, ids, positions, real_mask_2d):
        b, s = ids.shape
        type_ids = torch.zeros_like(ids)
        # additive mask [B, 1, 1, S]
        attn_mask = ((1.0 - real_mask_2d).to(self.dtype) * -1e4).view(b, 1, 1, s)
        x = self.embeddings(ids, type_ids, positions).to(self.dtype)
        for layer in self.layers:
            x = layer(x, attn_mask)
        # masked mean over real tokens
        m = real_mask_2d.to(x.dtype).unsqueeze(-1)
        emb = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        return emb


def load_from_hf(hf_model, ours):
    """Copy stock HF BERT weights into our module."""
    sd_hf = hf_model.state_dict()
    sd = {}

    def g(key):
        if key in sd_hf:
            return sd_hf[key]
        if f"bert.{key}" in sd_hf:
            return sd_hf[f"bert.{key}"]
        raise KeyError(key)

    sd["embeddings.word.weight"] = g("embeddings.word_embeddings.weight")
    sd["embeddings.pos.weight"] = g("embeddings.position_embeddings.weight")
    sd["embeddings.tok_type.weight"] = g("embeddings.token_type_embeddings.weight")
    sd["embeddings.ln.weight"] = g("embeddings.LayerNorm.weight")
    sd["embeddings.ln.bias"] = g("embeddings.LayerNorm.bias")

    for i in range(len(ours.layers)):
        p = f"encoder.layer.{i}"
        sd[f"layers.{i}.attention.q.weight"] = g(f"{p}.attention.self.query.weight")
        sd[f"layers.{i}.attention.q.bias"] = g(f"{p}.attention.self.query.bias")
        sd[f"layers.{i}.attention.k.weight"] = g(f"{p}.attention.self.key.weight")
        sd[f"layers.{i}.attention.k.bias"] = g(f"{p}.attention.self.key.bias")
        sd[f"layers.{i}.attention.v.weight"] = g(f"{p}.attention.self.value.weight")
        sd[f"layers.{i}.attention.v.bias"] = g(f"{p}.attention.self.value.bias")
        sd[f"layers.{i}.attn_out.weight"] = g(f"{p}.attention.output.dense.weight")
        sd[f"layers.{i}.attn_out.bias"] = g(f"{p}.attention.output.dense.bias")
        sd[f"layers.{i}.attn_ln.weight"] = g(f"{p}.attention.output.LayerNorm.weight")
        sd[f"layers.{i}.attn_ln.bias"] = g(f"{p}.attention.output.LayerNorm.bias")
        sd[f"layers.{i}.intermediate.weight"] = g(f"{p}.intermediate.dense.weight")
        sd[f"layers.{i}.intermediate.bias"] = g(f"{p}.intermediate.dense.bias")
        sd[f"layers.{i}.output.weight"] = g(f"{p}.output.dense.weight")
        sd[f"layers.{i}.output.bias"] = g(f"{p}.output.dense.bias")
        sd[f"layers.{i}.out_ln.weight"] = g(f"{p}.output.LayerNorm.weight")
        sd[f"layers.{i}.out_ln.bias"] = g(f"{p}.output.LayerNorm.bias")

    ours.load_state_dict(sd, strict=False)
