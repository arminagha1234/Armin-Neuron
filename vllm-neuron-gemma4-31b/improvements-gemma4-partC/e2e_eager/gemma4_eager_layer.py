"""Faithful eager Gemma4 decoder layer (decode path) — baseline vs NKI.

Builds ONE real Gemma4 decoder layer with all the real sublayers:
  input_layernorm -> attention(q/k/v proj, qk-norm, v-norm, RoPE, GQA, SDPA, o_proj)
  -> post_attention_layernorm -> residual
  -> pre_feedforward_layernorm -> GeGLU MLP -> post_feedforward_layernorm -> residual

Two execution modes, SAME weights:
  - use_nki=False : pure PyTorch-on-Neuron eager (the serving fallback path)
  - use_nki=True  : the 4 NKI kernels swapped in
      * input / pre_ff / post_attn / post_ff norms  -> nki_qk_rmsnorm / nki_fused_rmsnorm_residual
      * q_norm / k_norm / v_norm                     -> nki_qk_rmsnorm
      * attention score+softmax+AV core              -> nki_decode_attention_hd256/512
      * gelu_tanh(gate)*up                           -> nki_geglu_mlp

RoPE stays torch in both paths (no RoPE kernel) so the comparison isolates the
norm / attention / GeGLU wins exactly.

Gemma RMSNorm semantics are (1 + weight); we pass (1 + w) to the kernels.
"""
import sys, os
sys.path.insert(0, "/work")
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# NKI kernels (validated)
from nki_qk_rmsnorm import nki_qk_rmsnorm
from nki_fused_rmsnorm_residual import nki_fused_rmsnorm_residual
from nki_geglu_mlp import nki_geglu_mlp
from nki_decode_attention_hd256 import nki_decode_attention_hd256
from nki_decode_attention_hd512 import nki_decode_attention_hd512

try:
    from torch_neuronx import wrap_nki
    W_QK = wrap_nki(nki_qk_rmsnorm)
    W_RMSRES = wrap_nki(nki_fused_rmsnorm_residual)
    W_GEGLU = wrap_nki(nki_geglu_mlp)
    W_A256 = wrap_nki(nki_decode_attention_hd256)
    W_A512 = wrap_nki(nki_decode_attention_hd512)
except Exception:
    W_QK = W_RMSRES = W_GEGLU = W_A256 = W_A512 = None


# ---------------- torch reference sublayers ----------------

def torch_rmsnorm(x, weight, eps=1e-6):
    """Gemma RMSNorm: x * rsqrt(mean(x^2)+eps) * (1 + weight)."""
    var = x.pow(2).mean(-1, keepdim=True)
    xn = x * torch.rsqrt(var + eps)
    return xn * (1.0 + weight)


def torch_rmsnorm_noscale(x, eps=1e-6):
    """V-norm: RMSNorm with no learnable scale."""
    var = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin, rotary_dim):
    """Apply RoPE to the first rotary_dim dims of q,k (partial RoPE supported).
    q,k: [H, D] for a single decode token. cos/sin: [rotary_dim]."""
    if rotary_dim == q.shape[-1]:
        return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_rot = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return torch.cat([q_rot, q_pass], -1), torch.cat([k_rot, k_pass], -1)


# ---------------- the decoder layer ----------------

class Gemma4DecoderLayer:
    """Holds weights for one decoder layer; runs decode for one token.

    is_global selects head_dim=512 / 4 KV heads / partial RoPE, else SWA
    head_dim=256 / 16 KV heads / full RoPE.
    """

    def __init__(self, dev, is_global=False, hidden=5376, inter=21504,
                 n_heads=32, dtype=torch.float32):
        self.dev = dev
        self.dtype = dtype
        self.is_global = is_global
        self.H = hidden
        self.I = inter
        self.n_heads = n_heads
        self.hd = 512 if is_global else 256
        self.n_kv = 4 if is_global else 16
        self.rotary_dim = 128 if is_global else 256  # 0.25*512 vs full
        self.eps = 1e-6
        self.cap = 50.0  # attn logit softcap (Gemma4)

        q_out = n_heads * self.hd
        kv_out = self.n_kv * self.hd

        def rw(*shape):
            return (torch.randn(*shape, dtype=dtype) * 0.02).to(dev)

        # projections (no bias, Gemma style)
        self.w_q = rw(q_out, hidden)
        self.w_k = rw(kv_out, hidden)
        self.w_v = rw(kv_out, hidden)
        self.w_o = rw(hidden, q_out)
        self.w_gate = rw(inter, hidden)
        self.w_up = rw(inter, hidden)
        self.w_down = rw(hidden, inter)

        # norm weights (the learned delta; effective scale is 1+w)
        self.n_input = rw(1, hidden).squeeze(0)
        self.n_post_attn = rw(1, hidden).squeeze(0)
        self.n_pre_ff = rw(1, hidden).squeeze(0)
        self.n_post_ff = rw(1, hidden).squeeze(0)
        self.n_q = rw(1, self.hd).squeeze(0)
        self.n_k = rw(1, self.hd).squeeze(0)

        # RoPE tables for the single decode position (pos = S, the new token)
        self._rope_cache = {}

    def _rope(self, pos):
        if pos in self._rope_cache:
            return self._rope_cache[pos]
        theta = 1000000.0 if self.is_global else 10000.0
        half = self.rotary_dim // 2
        inv = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / half))
        ang = pos * inv  # [half]
        ang = torch.cat([ang, ang], -1).to(self.dev)  # [rotary_dim]
        c, s = ang.cos(), ang.sin()
        self._rope_cache[pos] = (c, s)
        return c, s

    # ---- baseline (torch) ----
    def forward_torch(self, hidden, K_cache, V_cache):
        S = K_cache.shape[1]
        residual = hidden
        x = torch_rmsnorm(hidden, self.n_input, self.eps)            # input_layernorm

        q = (x @ self.w_q.T).view(self.n_heads, self.hd)
        k = (x @ self.w_k.T).view(self.n_kv, self.hd)
        v = (x @ self.w_v.T).view(self.n_kv, self.hd)
        q = torch_rmsnorm(q, self.n_q, self.eps)
        k = torch_rmsnorm(k, self.n_k, self.eps)
        v = torch_rmsnorm_noscale(v, self.eps)

        c, s = self._rope(S)
        q, k = apply_rope(q, k, c, s, self.rotary_dim)

        # write new k,v to cache slot (we just append conceptually; here cache already sized S+1)
        Kc = torch.cat([K_cache, k.unsqueeze(1)], dim=1)  # [n_kv, S+1, hd]
        Vc = torch.cat([V_cache, v.unsqueeze(1)], dim=1)
        rep = self.n_heads // self.n_kv
        Kf = Kc.repeat_interleave(rep, dim=0)  # [n_heads, S+1, hd]
        Vf = Vc.repeat_interleave(rep, dim=0)

        qh = q.unsqueeze(1)  # [n_heads,1,hd]
        scores = torch.bmm(qh, Kf.transpose(1, 2))  # [n_heads,1,S+1]
        # NOTE: attn logit softcap omitted here to match the NKI attention core
        # (softcap is a separate kernel, benchmarked independently). Keeping both
        # paths identical isolates the norm/attention/GeGLU comparison.
        probs = torch.softmax(scores, dim=-1)
        attn = torch.bmm(probs, Vf).reshape(1, self.n_heads * self.hd)
        o = attn @ self.w_o.T  # [1,H]

        x = torch_rmsnorm(o, self.n_post_attn, self.eps)
        hidden = residual + x

        residual = hidden
        x = torch_rmsnorm(hidden, self.n_pre_ff, self.eps)
        gate = x @ self.w_gate.T
        up = x @ self.w_up.T
        act = F.gelu(gate, approximate="tanh") * up
        down = act @ self.w_down.T
        x = torch_rmsnorm(down, self.n_post_ff, self.eps)
        hidden = residual + x
        return hidden

    # ---- NKI ----
    def forward_nki(self, hidden, K_cache, V_cache):
        S = K_cache.shape[1]
        residual = hidden
        # input_layernorm via nki_qk_rmsnorm, weight = 1 + n_input
        x = W_QK(hidden, (1.0 + self.n_input).unsqueeze(0), self.eps)

        q = (x @ self.w_q.T).view(self.n_heads, self.hd)
        k = (x @ self.w_k.T).view(self.n_kv, self.hd)
        v = (x @ self.w_v.T).view(self.n_kv, self.hd)
        q = W_QK(q, (1.0 + self.n_q).unsqueeze(0), self.eps)
        k = W_QK(k, (1.0 + self.n_k).unsqueeze(0), self.eps)
        # v-norm: no scale -> weight = ones
        v = W_QK(v, torch.ones(1, self.hd, dtype=self.dtype, device=self.dev), self.eps)

        c, s = self._rope(S)
        q, k = apply_rope(q, k, c, s, self.rotary_dim)

        Kc = torch.cat([K_cache, k.unsqueeze(1)], dim=1)
        Vc = torch.cat([V_cache, v.unsqueeze(1)], dim=1)
        rep = self.n_heads // self.n_kv
        Kf = Kc.repeat_interleave(rep, dim=0)
        Vf = Vc.repeat_interleave(rep, dim=0)

        attn_kernel = W_A512 if self.is_global else W_A256
        # NOTE: the NKI attention kernel does score+softmax+AV (no softcap inside).
        # To match baseline we'd need softcap; we keep it out of both for the
        # attention-core comparison and verify the small resulting diff.
        outs = []
        for h in range(self.n_heads):
            q_t = q[h].reshape(self.hd, 1).contiguous()
            k_t = Kf[h].transpose(0, 1).contiguous()  # [hd, S+1]
            v_d = Vf[h].contiguous()                   # [S+1, hd]
            outs.append(attn_kernel(q_t, k_t, v_d, 1.0))  # [1, hd]
        attn = torch.cat(outs, dim=1)  # [1, n_heads*hd]
        o = attn @ self.w_o.T

        # post_attention_layernorm + residual fused
        hidden = W_RMSRES(residual, o, (1.0 + self.n_post_attn).unsqueeze(0), self.eps)

        residual = hidden
        x = W_QK(hidden, (1.0 + self.n_pre_ff).unsqueeze(0), self.eps)
        gate = x @ self.w_gate.T
        up = x @ self.w_up.T
        act = W_GEGLU(gate, up)
        down = act @ self.w_down.T
        hidden = W_RMSRES(residual, down, (1.0 + self.n_post_ff).unsqueeze(0), self.eps)
        return hidden

    def new_token(self):
        return (torch.randn(1, self.H, dtype=self.dtype) * 0.1).to(self.dev)

    def new_cache(self, S):
        K = (torch.randn(self.n_kv, S, self.hd, dtype=self.dtype) * 0.1).to(self.dev)
        V = (torch.randn(self.n_kv, S, self.hd, dtype=self.dtype) * 0.1).to(self.dev)
        return K, V
