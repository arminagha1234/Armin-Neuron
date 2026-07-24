#!/usr/bin/env python3
"""VALIDATION: nkilib attention_cte for Gemma4 prefill on cc-2.26 / trn2.

This container has NO torch_neuronx, so we drive the already-@nki.jit kernel with
NUMPY inputs -> nki's StandaloneKernel path: compile via ncc + run on the real
NeuronCore (executor=None => hardware). This is exactly what proves compile+cosine.

Confirms the EXACT call contract that will be wired into model.py::_manual_sdpa:
  from nkilib.core.attention.attention_cte import attention_cte   # already @nki.jit
  attention_cte(q, k, v, scale=..., causal_mask=True, sliding_window=W_or_0,
                tp_q=True, tp_k=True, tp_out=False)

Tests hd256 (SWA) + hd512 (global), causal + sliding-window, T in {512, 2048},
BOTH unexpanded k/v ([Nkv,T,D], GQA native) and expanded ([Nh,T,D]).

SWA constraint: kernel asserts scale==1.0 for sliding-window; we pre-scale q and
pass scale=1.0. Global (no SWA): pass scale directly.

Run on mel (1 core):  NEURON_RT_VISIBLE_CORES=0 python3 validate_cte_prefill.py
Optionally add:        NEURON_CC_FLAGS="-O1"  to test the compile-stall fix.
"""
import os, time, sys, traceback
import numpy as np
import ml_dtypes

BF16 = ml_dtypes.bfloat16


def torch_oracle(q, k, v, scale, sw):
    """FP32 numpy reference. q=[Nh,T,D], k/v=[Nkv,T,D] (unexpanded). Returns fp32 [Nh,T,D]."""
    Nh, T, D = q.shape
    Nkv = k.shape[0]
    g = Nh // Nkv
    kk = np.repeat(k.astype(np.float32), g, axis=0)   # [Nh,T,D]
    vv = np.repeat(v.astype(np.float32), g, axis=0)
    s = np.matmul(q.astype(np.float32), kk.transpose(0, 2, 1)) * scale  # [Nh,T,T]
    r = np.arange(T)
    m = (r[None, :] <= r[:, None])                     # causal j<=i
    if sw:
        m = m & ((r[:, None] - r[None, :]) < sw)       # window (i-j)<sw
    s = np.where(m[None], s, -np.inf)
    s = s - s.max(axis=-1, keepdims=True)
    e = np.exp(s)
    p = e / e.sum(axis=-1, keepdims=True)
    p = np.nan_to_num(p)
    return np.matmul(p, vv)                             # [Nh,T,D] fp32


def cos(a, b):
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


try:
    from nkilib.core.attention.attention_cte import attention_cte  # already @nki.jit
    print("imported nkilib attention_cte (no re-jit):", type(attention_cte), flush=True)
except Exception as e:
    print("IMPORT FAIL:", repr(e)[:300], flush=True)
    sys.exit(1)

print(f"NEURON_CC_FLAGS={os.environ.get('NEURON_CC_FLAGS', '<default>')}", flush=True)


def run_case(name, Nh, Nkv, T, D, sw, expand_kv):
    tag = f"{name} [Nh={Nh} Nkv={Nkv} T={T} D={D} sw={sw} expand_kv={expand_kv}]"
    print(f"\n=== {tag} ===", flush=True)
    scale = 1.0 / (D ** 0.5)
    rng = np.random.default_rng(0)
    q = rng.standard_normal((Nh, T, D)).astype(np.float32)
    k = rng.standard_normal((Nkv, T, D)).astype(np.float32)
    v = rng.standard_normal((Nkv, T, D)).astype(np.float32)
    ref = torch_oracle(q, k, v, scale, sw)  # oracle uses unexpanded k/v

    k_in, v_in = k, v
    if expand_kv:
        g = Nh // Nkv
        k_in = np.repeat(k, g, axis=0)
        v_in = np.repeat(v, g, axis=0)

    # SWA => scale must be 1.0; fold scale into q. Global => pass scale directly.
    if sw > 0:
        q_bf = (q * scale).astype(BF16)
        call_scale = 1.0
    else:
        q_bf = q.astype(BF16)
        call_scale = scale
    k_bf = k_in.astype(BF16)
    v_bf = v_in.astype(BF16)

    try:
        t0 = time.time()
        out = attention_cte(
            q_bf, k_bf, v_bf,
            scale=call_scale,
            causal_mask=True,
            sliding_window=sw,          # int, 0 == none
            tp_q=True,
            tp_k=True,                  # k is [*, T, D]; dma_transpose at load
            tp_out=False,
        )
        el = time.time() - t0
        out = np.asarray(out)
        c = cos(out, ref)
        status = "PASS" if c >= 0.999 else "LOW_COS"
        print(f"  {status}  compiled+ran {el:.1f}s  cos={c:.6f}  out_shape={tuple(out.shape)}",
              flush=True)
        return (tag, status, el, c)
    except Exception as e:
        print(f"  FAIL: {repr(e)[:400]}", flush=True)
        traceback.print_exc()
        return (tag, "FAIL", None, None)


results = []
cases = [
    # name,           Nh, Nkv, T,   D,   sw,   expand_kv
    ("hd512_global",   4,  2,  512,  512, 0,    False),
    ("hd512_global",   4,  2,  512,  512, 0,    True),
    ("hd256_swa",      4,  2,  512,  256, 1024, False),
    ("hd256_swa",      4,  2,  512,  256, 1024, True),
    ("hd512_global",   4,  2,  2048, 512, 0,    False),
    ("hd256_swa",      4,  2,  2048, 256, 1024, False),
]
for c in cases:
    results.append(run_case(*c))

print("\n===== SUMMARY =====", flush=True)
for tag, st, el, c in results:
    els = f"{el:.1f}s" if el is not None else "   -  "
    cs = f"cos={c:.6f}" if c is not None else "cos=  -   "
    print(f"  {st:7s} {els:>7s} {cs}  {tag}", flush=True)
print("\nVALIDATE_CTE_PREFILL_DONE", flush=True)
