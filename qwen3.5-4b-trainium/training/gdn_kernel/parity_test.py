"""
CPU parity test for chunked_gdn.chunked_gdn_forward.

Compares our compile-friendly chunked forward against:
  (a) the EXACT HF `torch_chunk_gated_delta_rule` (transformers 5.14.1),
      imported from transformers if available, else an embedded verbatim copy;
  (b) a sequential recurrent oracle (the ground-truth gated-delta recurrence).

Runs in fp32 on CPU at the real GDN dims (H=32 v-heads, K=V=128, seq multiple
of chunk). Reports cosine similarity + max abs err. Target: cos > 0.999.

Also runs torch.autograd.gradcheck-style finite check by comparing autograd
grads of our forward vs the HF reference forward (both differentiable), to
confirm the backward autograd builds matches numerically.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from chunked_gdn import chunked_gdn_forward, l2norm


# ---- embedded verbatim copy of HF torch_chunk_gated_delta_rule (oracle) ---- #
def hf_torch_chunk_gated_delta_rule(
    query, key, value, g, beta, chunk_size=64,
    initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=False, **kwargs,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


@torch.no_grad()
def recurrent_oracle(query, key, value, g, beta, use_qk_l2norm_in_kernel=True):
    """Sequential per-token gated-delta recurrence (ground truth)."""
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)
    q, k, v, beta_, g_ = [x.transpose(1, 2).float() for x in (query, key, value, beta, g)]
    B, H, T, K = k.shape
    V = v.shape[-1]
    scale = 1 / (K ** 0.5)
    q = q * scale
    S = q.new_zeros(B, H, K, V)
    out = torch.zeros(B, H, T, V)
    for t in range(T):
        g_t = g_[:, :, t].exp()[..., None, None]          # [B,H,1,1]
        S = S * g_t
        kv = (S * k[:, :, t][..., None]).sum(-2)          # [B,H,V]
        delta = (v[:, :, t] - kv) * beta_[:, :, t][..., None]
        S = S + k[:, :, t][..., None] * delta[..., None, :]
        out[:, :, t] = (S * q[:, :, t][..., None]).sum(-2)
    return out.transpose(1, 2).contiguous()


def cos(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def make_inputs(B=1, T=256, H=32, K=128, V=128, seed=0):
    torch.manual_seed(seed)
    query = torch.randn(B, T, H, K, dtype=torch.float32)
    key = torch.randn(B, T, H, K, dtype=torch.float32)
    value = torch.randn(B, T, H, V, dtype=torch.float32)
    # gate raw a -> g = -exp(A_log)*softplus(a+dt_bias), scalar per head [B,T,H]
    a = torch.randn(B, T, H, dtype=torch.float32)
    A_log = torch.log(torch.empty(H).uniform_(0, 16))
    dt_bias = torch.ones(H)
    g = -A_log.exp() * F.softplus(a + dt_bias)   # [B,T,H]
    b = torch.randn(B, T, H, dtype=torch.float32)
    beta = b.sigmoid()                            # [B,T,H]
    return query, key, value, g, beta


def run(B=1, T=256, H=32, K=128, V=128, chunk_size=128, seed=0):
    query, key, value, g, beta = make_inputs(B, T, H, K, V, seed)

    ours, _ = chunked_gdn_forward(query, key, value, g, beta,
                                  chunk_size=chunk_size, use_qk_l2norm_in_kernel=True)
    # HF reference at its native chunk_size=64 (result is chunk-invariant)
    hf64, _ = hf_torch_chunk_gated_delta_rule(query, key, value, g, beta,
                                              chunk_size=64, use_qk_l2norm_in_kernel=True)
    hf128, _ = hf_torch_chunk_gated_delta_rule(query, key, value, g, beta,
                                               chunk_size=chunk_size, use_qk_l2norm_in_kernel=True)
    rec = recurrent_oracle(query, key, value, g, beta, use_qk_l2norm_in_kernel=True)

    print(f"--- dims B={B} T={T} H={H} K={K} V={V} chunk={chunk_size} seed={seed} ---")
    print(f"ours vs HF(chunk={chunk_size}): cos={cos(ours, hf128):.6f}  maxabs={(ours-hf128).abs().max():.3e}")
    print(f"ours vs HF(chunk=64):          cos={cos(ours, hf64):.6f}  maxabs={(ours-hf64).abs().max():.3e}")
    print(f"ours vs recurrent oracle:      cos={cos(ours, rec):.6f}  maxabs={(ours-rec).abs().max():.3e}")
    print(f"HF(128) vs recurrent oracle:   cos={cos(hf128, rec):.6f}  (reference sanity)")
    return cos(ours, hf128), cos(ours, rec)


def gradcheck_vs_hf(B=1, T=128, H=4, K=32, V=32, chunk_size=128, seed=1):
    """Compare autograd grads of our forward vs HF forward on a small shape."""
    query, key, value, g, beta = make_inputs(B, T, H, K, V, seed)

    def grads(fn, cs):
        q = query.clone().requires_grad_(True)
        k = key.clone().requires_grad_(True)
        v = value.clone().requires_grad_(True)
        gg = g.clone().requires_grad_(True)
        bb = beta.clone().requires_grad_(True)
        out, _ = fn(q, k, v, gg, bb, chunk_size=cs, use_qk_l2norm_in_kernel=True)
        loss = (out * out).sum()
        loss.backward()
        return [t.grad.clone() for t in (q, k, v, gg, bb)]

    go = grads(chunked_gdn_forward, chunk_size)
    gh = grads(hf_torch_chunk_gated_delta_rule, 64)
    names = ["dq", "dk", "dv", "dg", "dbeta"]
    print(f"--- gradcheck (autograd) B={B} T={T} H={H} K={K} V={V} ---")
    ok = True
    for n, a, b in zip(names, go, gh):
        c = cos(a, b)
        ok = ok and (c > 0.999)
        print(f"  {n}: cos(ours, HF)={c:.6f}  maxabs={(a-b).abs().max():.3e}")
    return ok


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)
    c1, _ = run(B=1, T=256, H=32, K=128, V=128, chunk_size=128, seed=0)
    c2, _ = run(B=2, T=384, H=32, K=128, V=128, chunk_size=128, seed=3)
    print()
    ok_grad = gradcheck_vs_hf()
    print()
    passed = (c1 > 0.999) and (c2 > 0.999) and ok_grad
    print(f"PARITY_RESULT: forward_cos_min={min(c1, c2):.6f}  grad_ok={ok_grad}  "
          f"{'PASS' if passed else 'FAIL'}")
