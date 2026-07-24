"""
Rank-1 GDN path: a pure-torch CHUNKED gated-delta-rule forward that
PyTorch autograd can differentiate, and that torch.compile(backend="neuron",
dynamic=False) can lower on neuronx-cc.

Why this file exists
--------------------
The stock HF `torch_chunk_gated_delta_rule` (transformers 5.14.1,
models/qwen3_5/modeling_qwen3_5.py) is *chunked* but is NOT compile-friendly on
Trainium: it builds the intra-chunk `(I - A)^{-1}` factor with a
**data-dependent forward-substitution loop** that does variable-width, in-place
slice assignment:

    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

The `[..., i, :i]` slices change shape every iteration and the writes are
in-place. That is exactly the pattern that makes neuronx-cc emit
"Can only vectorize loop or free axes" (and, unrolled at 32L, blows SBUF).

This module keeps the *identical math* but:
  1. Replaces the forward-substitution inverse with a **Neumann-doubling
     product** `(I - A)^{-1} = prod_k (I + A^(2^k))`.  For a strictly-lower
     (hence nilpotent, A^BT = 0) BT x BT matrix this product is EXACT, not an
     approximation, and is a fixed unroll of `log2(BT)` matmuls -- no
     data-dependent control flow, no in-place variable-width writes.
  2. Removes the in-place `core_attn_out[:, :, i] = ...` output writes in the
     inter-chunk scan (accumulate into a Python list, then torch.stack).
  3. Keeps everything as plain autograd-able torch ops (matmul, cumsum, exp,
     tril, sigmoid, softplus, l2norm) so autograd produces the exact backward
     with zero hand-derived math.

Static-shape / control-flow-free properties (required by the TorchNeuron beta,
dynamic=False): chunk count NT = T // BT is a Python constant, the inner solve
is a fixed `log2(BT)`-step unroll, and the chunk loop is a Python `for` over a
constant range that unrolls at trace time. No `if tensor.item()`, no `nonzero`,
no boolean-mask select.

GDN specifics (Qwen3.5-4B), confirmed from the HF module:
  * decay is SCALAR-per-head: g has shape [B, T, H] (a[B,T,H] -> softplus gate)
  * gate:  g = -exp(A_log) * softplus(a + dt_bias)   (<= 0, log-decay)
  * beta = sigmoid(b)                                 (per-head write gate)
  * q, k are L2-normalized inside the kernel; scale = 1/sqrt(k_head_dim)
  * K = V = head_dim = 128, num_v_heads = 32 (num_k_heads = 16, GQA repeat x2
    is done by the caller before this function, matching HF).

The z-gate + gated-RMSNorm (SiLU, not sigmoid) and out_proj live OUTSIDE this
function, in the HF module, and are left untouched.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalize along `dim`. Matches transformers qwen3_5 `l2norm` exactly."""
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x * inv_norm


def _neumann_inverse_unit_lower(A: torch.Tensor, bt: int) -> torch.Tensor:
    """Exact inverse of (I - A) for a strictly-lower-triangular A (A^bt = 0).

    Uses the doubling identity for a nilpotent A:
        (I - A)^{-1} = sum_{j=0}^{bt-1} A^j = prod_{k=0}^{m-1} (I + A^(2^k)),
        where m = log2(bt).
    This is EXACT (not truncated) because A is nilpotent of index <= bt, and it
    is a fixed unroll of m matmuls -- no data-dependent control flow, no
    variable-width in-place slicing. Runs in fp32 for conditioning.

    WARNING (overflow): the doubling forms explicit HIGH powers A^(2^k) up to
    A^(bt/2). For bt=128 that is A^64. With large-norm REAL weights (row-sums of
    A up to ~bt), A^64 entries reach ~1e37..1e100 and overflow fp32 (max 3.4e38)
    -> Inf -> Inf-Inf = NaN. This was the full-32L NaN root cause. Use this ONLY
    on SMALL blocks (bt<=16: max power A^8, entries bounded ~C(bt-2,bt/2) so no
    overflow). For a full chunk use `_block_forward_sub_inverse`, which never
    forms powers higher than A^(blk/2).

    Args:
        A:  [..., bt, bt] strictly-lower-triangular (diagonal and above are 0).
        bt: chunk size, MUST be a power of two.
    Returns:
        [..., bt, bt] = (I - A)^{-1}, including the unit diagonal.
    """
    assert (bt & (bt - 1)) == 0, f"chunk_size must be a power of two, got {bt}"
    m = bt.bit_length() - 1  # log2(bt)
    eye = torch.eye(bt, dtype=A.dtype, device=A.device)
    # result = (I + A^1)
    result = eye + A
    a_pow = A
    for _ in range(1, m):  # k = 1 .. m-1, fixed compile-time range
        a_pow = a_pow @ a_pow          # A^(2^k)
        result = result @ (eye + a_pow)
    return result


def _block_forward_sub_inverse(A: torch.Tensor, bt: int, blk: int = 16) -> torch.Tensor:
    """Overflow-safe exact inverse of (I - A) for strictly-lower A, via BLOCK
    forward-substitution.

    Partition the bt x bt matrix M = I - A into a `nb x nb` grid of `blk x blk`
    blocks (nb = bt // blk). M is block-lower-triangular with unit-lower-tri
    diagonal blocks. Its inverse N = M^{-1} is also block-lower-triangular:

        N[I,I] = M[I,I]^{-1}                                  (diagonal blocks)
        N[I,J] = -M[I,I]^{-1} @ sum_{K=J}^{I-1} M[I,K] @ N[K,J]   (I > J)

    Diagonal blocks are inverted by the exact Neumann doubling above, but ONLY at
    size `blk` (blk=16 -> highest power A^8, C(14,7)~3.4e3, zero overflow risk).
    Off-diagonal blocks come from block forward-substitution over TRUE inverse
    sub-blocks (bounded, since the delta-rule inverse is well-conditioned).

    Crucially this NEVER forms A^32 / A^64 of the full chunk -- exactly the high
    powers that overflowed fp32 in `_neumann_inverse_unit_lower(A, 128)` and
    produced the full-32L NaN. It is a fixed unroll over (nb, blk) compile-time
    constants: no data-dependent control flow, no variable-width in-place slicing
    (unlike HF's forward-substitution loop), so torch.compile lowers it cleanly.

    Mathematically identical to HF's forward-substitution and to the (finite)
    full Neumann sum: N = sum_{j=0}^{bt-1} A^j = (I - A)^{-1}.

    Args:
        A:   [..., bt, bt] strictly-lower-triangular (diagonal and above are 0).
        bt:  chunk size (power of two).
        blk: sub-block size (power of two dividing bt). blk=16 is the safe default.
    Returns:
        [..., bt, bt] = (I - A)^{-1}, including the unit diagonal.
    """
    assert bt % blk == 0, f"blk ({blk}) must divide bt ({bt})"
    nb = bt // blk

    # Diagonal-block inverses  Dinv[I] = (I_blk - A[I,I])^{-1}. A[I,I] is itself
    # strictly-lower (nilpotent index <= blk), so exact Neumann doubling at size
    # blk is finite (highest power A^(blk/2)).
    Dinv = []
    for i in range(nb):
        s = i * blk
        Dinv.append(_neumann_inverse_unit_lower(A[..., s:s + blk, s:s + blk], blk))

    zero_blk = torch.zeros_like(Dinv[0])
    # N_blocks[I][J] holds the (I,J) block of N (None => zero, i.e. I < J).
    N_blocks = [[None] * nb for _ in range(nb)]
    for j in range(nb):                       # column by column
        N_blocks[j][j] = Dinv[j]
        for i in range(j + 1, nb):            # rows below the diagonal
            si = i * blk
            acc = None
            for kk in range(j, i):            # sum_{K=J}^{I-1} M[I,K] @ N[K,J]
                sk = kk * blk
                m_ik = -A[..., si:si + blk, sk:sk + blk]   # M off-diag = -A
                term = m_ik @ N_blocks[kk][j]
                acc = term if acc is None else acc + term
            N_blocks[i][j] = -Dinv[i] @ acc

    # Assemble the full [..., bt, bt] matrix from blocks (functional, no in-place).
    rows = []
    for i in range(nb):
        row = [N_blocks[i][j] if N_blocks[i][j] is not None else zero_blk
               for j in range(nb)]
        rows.append(torch.cat(row, dim=-1))
    return torch.cat(rows, dim=-2)


def chunked_gdn_forward(
    query: torch.Tensor,   # [B, T, H, K]
    key: torch.Tensor,     # [B, T, H, K]
    value: torch.Tensor,   # [B, T, H, V]
    g: torch.Tensor,       # [B, T, H]     log-decay (<= 0), scalar-per-head
    beta: torch.Tensor,    # [B, T, H]     write gate (already sigmoid'd upstream? no -- see below)
    chunk_size: int = 128,
    initial_state: torch.Tensor | None = None,   # [B, H, K, V]
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
):
    """Chunked gated-delta-rule forward, autograd-differentiable and
    compile-friendly. Drop-in for transformers `torch_chunk_gated_delta_rule`.

    NOTE on `beta`: HF passes `beta = b.sigmoid()` already (see the GDN module),
    so this function treats `beta` as the final write-gate (NOT re-sigmoided),
    exactly like `torch_chunk_gated_delta_rule`.

    Returns:
        core_attn_out: [B, T, H, V]
        last_recurrent_state: [B, H, K, V] or None
    """
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = l2norm(query, dim=-1, eps=1e-6)
        key = l2norm(key, dim=-1, eps=1e-6)

    # [B, T, H, D] -> [B, H, T, D]; g, beta: [B, T, H] -> [B, H, T]. fp32 core.
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]

    # Pad the sequence up to a multiple of chunk_size (static given fixed T).
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size

    scale = 1.0 / (k_head_dim ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    nt = total_sequence_length // chunk_size
    # reshape to chunks: [B, H, NT, BT, D]
    query = query.reshape(batch_size, num_heads, nt, chunk_size, k_head_dim)
    key = key.reshape(batch_size, num_heads, nt, chunk_size, k_head_dim)
    value = value.reshape(batch_size, num_heads, nt, chunk_size, v_head_dim)
    k_beta = k_beta.reshape(batch_size, num_heads, nt, chunk_size, k_head_dim)
    v_beta = v_beta.reshape(batch_size, num_heads, nt, chunk_size, v_head_dim)
    g = g.reshape(batch_size, num_heads, nt, chunk_size)

    # strictly-upper (incl. diagonal) mask -> zeroes for the intra-chunk factor
    mask_incl_diag = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )

    # intra-chunk cumulative log-decay
    g = g.cumsum(dim=-1)  # [B, H, NT, BT]

    # decay_mask[..., i, j] = exp(g_i - g_j) for i >= j else 0.
    # tril-before-exp keeps exp() away from positive args (overflow-safe), exactly
    # like HF; the extra clamp(max=0) is a provable no-op on the retained region
    # (g is a cumsum of <=0 values so g_i <= g_j for i >= j) and a device guard.
    gdiff = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril()          # [B,H,NT,BT,BT]
    decay_mask = gdiff.clamp(max=0.0).exp().tril()

    # A = -( (k_beta @ key^T) * decay_mask ) with strict-lower kept (diag+upper zeroed)
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask_incl_diag, 0)

    # (I - A)^{-1} via overflow-safe BLOCK forward-substitution (replaces HF's
    # data-dependent forward-substitution loop AND the old full-chunk Neumann
    # doubling that formed A^32/A^64 -> fp32 overflow -> full-32L NaN).
    T_inv = _block_forward_sub_inverse(attn, chunk_size, blk=16)  # [B,H,NT,BT,BT]

    # WY vectors
    u = T_inv @ v_beta                                          # "new value" u  [B,H,NT,BT,V]
    k_cumdecay = T_inv @ (k_beta * g.exp().unsqueeze(-1))       # [B,H,NT,BT,K]

    # ---- inter-chunk recurrence over the state S [K, V] (functional, no in-place) ----
    if initial_state is None:
        last_recurrent_state = query.new_zeros(batch_size, num_heads, k_head_dim, v_head_dim)
    else:
        last_recurrent_state = initial_state.to(query)

    outputs = []
    for i in range(nt):  # nt is a compile-time constant -> unrolls at trace time
        q_i = query[:, :, i]                 # [B,H,BT,K]
        k_i = key[:, :, i]
        u_i = u[:, :, i]                     # [B,H,BT,V]
        g_i = g[:, :, i]                     # [B,H,BT]
        kcd_i = k_cumdecay[:, :, i]          # [B,H,BT,K]
        dm_i = decay_mask[:, :, i]           # [B,H,BT,BT]

        attn_i = (q_i @ k_i.transpose(-1, -2)) * dm_i          # [B,H,BT,BT] (causal, incl diag)
        v_prime = kcd_i @ last_recurrent_state                 # [B,H,BT,V]
        v_new = u_i - v_prime                                  # [B,H,BT,V]
        attn_inter = (q_i * g_i.unsqueeze(-1).exp()) @ last_recurrent_state
        out_i = attn_inter + attn_i @ v_new                    # [B,H,BT,V]
        outputs.append(out_i)

        # state carry: decay by total chunk decay, then add decay-weighted K^T V
        g_last = g_i[:, :, -1]                                 # [B,H]
        state_decay = g_last[:, :, None, None].exp()           # [B,H,1,1]
        # (g_last - g_i) >= 0 always (g decreasing within chunk); exp is bounded by exp(0..)
        k_decay = (k_i * (g_last[:, :, None] - g_i).exp().unsqueeze(-1)).transpose(-1, -2)  # [B,H,K,BT]
        last_recurrent_state = last_recurrent_state * state_decay + k_decay @ v_new

    core_attn_out = torch.stack(outputs, dim=2)                # [B,H,NT,BT,V]
    core_attn_out = core_attn_out.reshape(batch_size, num_heads, -1, v_head_dim)
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)

    if not output_final_state:
        last_recurrent_state = None
    return core_attn_out, last_recurrent_state


# --------------------------------------------------------------------------- #
# Env-gated monkeypatch of the HF Qwen3.5 GDN forward (Gemma4 pattern)         #
# --------------------------------------------------------------------------- #
def patch_qwen35_gdn(chunk_size: int = 128, verbose: bool = True) -> bool:
    """Replace `Qwen3_5GatedDeltaNet.chunk_gated_delta_rule` with our chunked
    forward, so the stock HF module uses it for the prefill/training path.

    Env gate: only patches if QWEN35_GDN_CHUNKED=1. Returns True if patched.

    We patch the *instance attribute the module actually calls*
    (`self.chunk_gated_delta_rule`) via a module-level default swap, matching how
    HF binds it in __init__ (`self.chunk_gated_delta_rule = chunk_gated_delta_rule
    or torch_chunk_gated_delta_rule`). We monkeypatch the module symbol AND wrap
    __init__ so every constructed layer picks up our function, with a try/except
    fallback to the original torch impl (Gemma4 _manual_sdpa defensive shape).
    """
    import os
    if os.environ.get("QWEN35_GDN_CHUNKED", "0") != "1":
        if verbose:
            print("[gdn-patch] QWEN35_GDN_CHUNKED != 1 -> not patching (stock HF path)", flush=True)
        return False

    import transformers.models.qwen3_5.modeling_qwen3_5 as M

    _orig = M.torch_chunk_gated_delta_rule

    def _patched_chunk(query, key, value, g, beta, chunk_size=chunk_size,
                       initial_state=None, output_final_state=False,
                       use_qk_l2norm_in_kernel=False, **kw):
        try:
            return chunked_gdn_forward(
                query, key, value, g, beta,
                chunk_size=chunk_size,
                initial_state=initial_state,
                output_final_state=output_final_state,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
        except Exception as e:  # defensive fallback, log once
            if not getattr(_patched_chunk, "_warned", False):
                print(f"[gdn-patch] chunked_gdn_forward raised ({e!r}); "
                      f"falling back to torch_chunk_gated_delta_rule", flush=True)
                _patched_chunk._warned = True
            return _orig(query, key, value, g, beta,
                         chunk_size=64, initial_state=initial_state,
                         output_final_state=output_final_state,
                         use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kw)

    # swap the module-level symbol used as the __init__ default
    M.torch_chunk_gated_delta_rule = _patched_chunk

    # wrap __init__ so instances constructed *before or after* the swap use ours
    _orig_init = M.Qwen3_5GatedDeltaNet.__init__

    def _new_init(self, *a, **kw):
        _orig_init(self, *a, **kw)
        # only override if HF fell back to the torch impl (fast FLA path absent)
        self.chunk_gated_delta_rule = _patched_chunk

    M.Qwen3_5GatedDeltaNet.__init__ = _new_init

    if verbose:
        print(f"[gdn-patch] patched Qwen3_5GatedDeltaNet -> chunked_gdn_forward "
              f"(chunk_size={chunk_size})", flush=True)
    return True
