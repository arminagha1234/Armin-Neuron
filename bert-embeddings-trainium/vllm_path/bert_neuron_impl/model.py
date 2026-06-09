# SPDX-License-Identifier: Apache-2.0
"""
BERT encoder implementation for the vllm-neuron backend.

This is a bidirectional encoder: no KV cache, no causal mask, learned
position + token-type embeddings, standard post-LN transformer blocks, and a
mean-pooled sentence embedding output.

IMPORTANT CONTRACT NOTES (discovered by reading neuron_model_runner.py):
- The runner ALWAYS calls the model with decode-style kwargs:
  input_ids, positions, attn_metadata, sampling_params, sampling_positions,
  spec_decode_metadata, rank, logit_mask. An encoder ignores most of these.
- The runner expects get_kv_spec()/bind_kv_cache() — we return an empty KV
  spec so no KV cache blocks are allocated.
- The runner feeds the model output into a SAMPLING pipeline and the engine
  is configured is_pooling_model=False. This is the known wall: there is no
  embedding/pooling output path on the Neuron runner. We still produce a
  pooled embedding so we can localize exactly where the runner rejects it.
"""
import torch
import torch.nn as nn
from transformers import PretrainedConfig

from vllm_neuron.model.neuron_config import NeuronConfig
from vllm_neuron.model.kv_cache import KVSpec

from .config import BertNeuronConfig


class BertEmbeddings(nn.Module):
    def __init__(self, c: BertNeuronConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(c.vocab_size, c.hidden_size, c.pad_token_id)
        self.position_embeddings = nn.Embedding(c.max_position_embeddings, c.hidden_size)
        self.token_type_embeddings = nn.Embedding(c.type_vocab_size, c.hidden_size)
        self.LayerNorm = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)

    def forward(self, input_ids, token_type_ids, position_ids):
        e = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        return self.LayerNorm(e)


class BertSelfAttention(nn.Module):
    def __init__(self, c: BertNeuronConfig):
        super().__init__()
        self.n = c.num_attention_heads
        self.d = c.head_dim
        self.q = nn.Linear(c.hidden_size, c.hidden_size)
        self.k = nn.Linear(c.hidden_size, c.hidden_size)
        self.v = nn.Linear(c.hidden_size, c.hidden_size)

    def _shape(self, x, b, s):
        return x.view(b, s, self.n, self.d).transpose(1, 2)

    def forward(self, x, mask):
        b, s, _ = x.shape
        q = self._shape(self.q(x), b, s)
        k = self._shape(self.k(x), b, s)
        v = self._shape(self.v(x), b, s)
        # Explicit bidirectional attention with plain ops (traceable under the
        # vllm-neuron compiler; raw F.scaled_dot_product_attention is marked
        # "skipped" by dynamo because attention must route through the
        # registered unified_attention custom op).
        scale = 1.0 / (self.d ** 0.5)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        if mask is not None:
            # mask is additive [B, 1, 1, S] with 0 for real, large-negative for pad
            scores = scores + mask
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)
        return out.transpose(1, 2).reshape(b, s, self.n * self.d)


class BertLayer(nn.Module):
    def __init__(self, c: BertNeuronConfig):
        super().__init__()
        self.attention = BertSelfAttention(c)
        self.attn_out = nn.Linear(c.hidden_size, c.hidden_size)
        self.attn_ln = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)
        self.intermediate = nn.Linear(c.hidden_size, c.intermediate_size)
        self.output = nn.Linear(c.intermediate_size, c.hidden_size)
        self.out_ln = nn.LayerNorm(c.hidden_size, eps=c.layer_norm_eps)

    def forward(self, x, mask):
        a = self.attn_out(self.attention(x, mask))
        x = self.attn_ln(x + a)
        # gelu via pure tensor ops — both nn.GELU and F.gelu lower to a C
        # builtin (torch._C._nn.gelu) that the vllm-neuron dynamo tracer marks
        # "skipped". The tanh approximation uses only traceable ops.
        g = self.intermediate(x)
        gelu = 0.5 * g * (1.0 + torch.tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)))
        h = self.output(gelu)
        return self.out_ln(x + h)


class BertEncoderModel(nn.Module):
    """Top-level encoder registered with vLLM as the model class."""

    def __init__(self, config: BertNeuronConfig):
        super().__init__()
        self.config = config
        self.dtype = config.torch_dtype
        self.rank = 0
        self.world_size = 1
        self.embeddings = BertEmbeddings(config)
        self.layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        # vLLM pooling interface: the runner calls model.pooler.get_pooling_updates(task)
        # and the engine queries pooler.get_supported_tasks(). Reuse vLLM's
        # built-in embedding pooler so the pooling pipeline is satisfied.
        self.pooler = self._build_embedding_pooler(config)

    @staticmethod
    def _build_embedding_pooler(config):
        try:
            from vllm.model_executor.layers.pooler import DispatchPooler
            from vllm.config.pooler import PoolerConfig
            pc = PoolerConfig(pooling_type="MEAN")
            return DispatchPooler.for_embedding(pc)
        except Exception:
            # Fallback: a tiny shim implementing the methods the runner touches.
            import torch.nn as _nn

            class _ShimPooler(_nn.Module):
                def get_supported_tasks(self):
                    return ("embed", "encode")

                def get_pooling_updates(self, task):
                    from vllm.pooling_params import PoolingParams  # noqa
                    class _U:
                        def apply(self, params):
                            return None
                    return _U()

                def forward(self, hidden_states, pooling_metadata):
                    return hidden_states

            return _ShimPooler()

    @classmethod
    def from_configs(cls, hf_config: PretrainedConfig, neuron_config: NeuronConfig):
        config = BertNeuronConfig.from_configs(hf_config, neuron_config)
        return cls(config)

    # ── KV cache: encoders have none ─────────────────────────────────────
    def get_kv_spec(self):
        return KVSpec(layers=[])

    def bind_kv_cache(self, kv_caches):
        return  # no-op

    # ── forward: tolerate the runner's decode-style kwargs ───────────────
    def forward(self, input_ids, positions=None, attn_metadata=None, **kwargs):
        # Runner passes flat input_ids [num_tokens] and positions [num_tokens].
        # IMPORTANT: vLLM-Neuron pads positions by REPEATING the last real
        # position (e.g. real tokens at pos 0..12, then 12,12,12,...) — it
        # does NOT pass arange. Using torch.arange here gives wrong position
        # embeddings for padded slots, which corrupts the mean-pool.
        # Use vLLM's actual positions, and mean-pool over REAL tokens only
        # (input_ids != PAD=0) for a correct sentence embedding.
        input_ids = input_ids.reshape(1, -1)
        s = input_ids.shape[1]
        device = input_ids.device
        if positions is not None:
            position_ids = positions.reshape(1, s).to(torch.long)
        else:
            position_ids = torch.arange(s, device=device).reshape(1, s)
        token_type_ids = torch.zeros(1, s, dtype=torch.long, device=device)

        # Attention mask from real-token signal: real tokens have input_ids != 0.
        # Build additive mask [1, 1, 1, S]: 0 for real, large-negative for pad.
        real = (input_ids != 0).to(self.dtype)                  # [1, S]
        attn_mask = (1.0 - real) * -1e4                         # [1, S]
        attn_mask = attn_mask.view(1, 1, 1, s)                  # [1, 1, 1, S]

        x = self.embeddings(input_ids, token_type_ids, position_ids).to(self.dtype)
        for layer in self.layers:
            x = layer(x, attn_mask)

        # Masked mean-pool: average only over real tokens (input_ids != 0).
        mask = real.unsqueeze(-1)                               # [1, S, 1]
        summed = (x * mask).sum(dim=1)                          # [1, H]
        counts = mask.sum(dim=1).clamp(min=1.0)                 # [1, 1]
        emb = summed / counts
        return emb

    # ── weight loading: map HF bert.* keys → our module names ────────────
    def load_weights(self, checkpoint_path: str, device: torch.device, cache_dir):
        from safetensors import safe_open
        import os, glob

        # checkpoint_path may be a local dir OR an HF repo id. Resolve to a
        # local dir of safetensors either way.
        local_dir = checkpoint_path
        if not os.path.isdir(checkpoint_path):
            from huggingface_hub import snapshot_download
            local_dir = snapshot_download(
                checkpoint_path,
                cache_dir=cache_dir,
                allow_patterns=["*.safetensors", "*.bin", "*.json"],
            )

        files = glob.glob(os.path.join(local_dir, "*.safetensors"))
        if not files:
            files = glob.glob(os.path.join(local_dir, "**", "*.safetensors"), recursive=True)

        hf = {}
        if files:
            for f in files:
                with safe_open(f, framework="pt") as sf:
                    for k in sf.keys():
                        hf[k] = sf.get_tensor(k)
        else:
            # fall back to pytorch_model.bin
            import torch as _t
            bins = glob.glob(os.path.join(local_dir, "*.bin"))
            for b in bins:
                hf.update(_t.load(b, map_location="cpu"))

        # HF BERT checkpoints prefix keys with "bert." (e.g.
        # bert.embeddings.word_embeddings.weight). Some sentence-transformers
        # exports drop it. Normalize by trying both.
        def g(key):
            if key in hf:
                return hf[key]
            if f"bert.{key}" in hf:
                return hf[f"bert.{key}"]
            raise KeyError(f"{key} (also tried bert.{key}); "
                           f"available sample: {list(hf)[:6]}")

        sd = {}
        sd["embeddings.word_embeddings.weight"] = g("embeddings.word_embeddings.weight")
        sd["embeddings.position_embeddings.weight"] = g("embeddings.position_embeddings.weight")
        sd["embeddings.token_type_embeddings.weight"] = g("embeddings.token_type_embeddings.weight")
        sd["embeddings.LayerNorm.weight"] = g("embeddings.LayerNorm.weight")
        sd["embeddings.LayerNorm.bias"] = g("embeddings.LayerNorm.bias")

        for i in range(self.config.num_hidden_layers):
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

        sd = {k: v.to(self.config.torch_dtype) for k, v in sd.items()}
        self.load_state_dict(sd, strict=False, assign=True)

    def load_weights_lite(self, checkpoint_path: str, device: torch.device, cache_dir):
        return  # encoder has no KV-cache scales to preload
