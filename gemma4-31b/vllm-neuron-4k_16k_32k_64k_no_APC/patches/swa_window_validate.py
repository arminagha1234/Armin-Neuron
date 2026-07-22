"""
Validate that WINDOWING the SWA prior (gather only the last `w` prior tokens,
prior_used_len=w) produces the SAME attention output as the FULL-span prior,
using the kernel's own PyTorch fallback (attention_cte.flash_attention).

If windowed == full-span within tol, the segmented-prefill SWA-window fix is
CORRECT (shift-invariant masking) and worth validating on-device. If not, the
mismatch localizes the real bug my earlier patch hit.

Mirrors the model.py segmented call:
  q=[Nh,T,D] tp_q=True; k=[Nkv,D,T] tp_k=False; v=[Nkv,T,D];
  k_prior=[Nkv,D,P] ; v_prior=[Nkv,P,D]; prior_used_len; sliding_window.
"""
import os
os.environ.setdefault("PJRT_DEVICE", "CPU")
import torch
import vllm_neuron.functional as NF

torch.manual_seed(0)

Nh = 1          # heads (no GQA for the test)
D  = 16         # head dim (small; torch ref has no 128 cap)
P  = 20         # prior length (already-cached tokens, abs positions [0,P))
T  = 8          # current chunk length (abs positions [P, P+T))
WINDOW = 4      # sliding window
w = WINDOW + 2  # windowed-prior length we keep (>= WINDOW so early queries covered)
scale = 1.0     # Gemma4 uses 1.0

# Random prior + current-chunk projected tensors.
k_prior_full = torch.randn(Nh, P, D)   # [Nkv, P, D]
v_prior_full = torch.randn(Nh, P, D)
q = torch.randn(Nh, T, D)              # [Nh, T, D]
k = torch.randn(Nh, T, D)              # [Nkv, T, D]
v = torch.randn(Nh, T, D)

def run(k_prior, v_prior, prior_used_len):
    return NF.flash_attention(
        q=q,
        k=k.transpose(1, 2),                 # [Nkv, D, T]
        v=v,
        scale=scale,
        causal_mask=True,
        sliding_window=WINDOW,
        k_prior=k_prior.transpose(1, 2),     # [Nkv, D, P_slice]
        v_prior=v_prior,
        prior_used_len=torch.tensor([prior_used_len], dtype=torch.int32),
        tp_q=True, tp_k=False, tp_out=False,
    )

# (1) FULL-SPAN prior: prior_used_len = P.
out_full = run(k_prior_full, v_prior_full, P)

# (2) WINDOWED prior: keep only the last w prior tokens (abs [P-w, P)),
#     prior_used_len = w. Shift-invariance => should match full-span.
k_prior_win = k_prior_full[:, P - w:, :].contiguous()
v_prior_win = v_prior_full[:, P - w:, :].contiguous()
out_win = run(k_prior_win, v_prior_win, w)

# (3) BROKEN control: window the tensor but LIE about prior_used_len = P
#     (what a naive/incorrect patch might do) -> expect mismatch.
try:
    out_bad = run(k_prior_win, v_prior_win, w if False else w)  # placeholder
except Exception as e:
    out_bad = None

diff = (out_full - out_win).abs().max().item()
print(f"out_full shape={tuple(out_full.shape)}  out_win shape={tuple(out_win.shape)}")
print(f"MAX ABS DIFF windowed-vs-full = {diff:.3e}")
print("RESULT:", "MATCH  (windowed prior is CORRECT)" if diff < 1e-4 else "MISMATCH (fix invalid as-is)")
