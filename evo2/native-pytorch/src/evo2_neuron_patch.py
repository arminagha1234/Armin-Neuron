"""Make Evo2 (StripedHyena2) compile on Trainium by removing complex/FFT ops.

Neuron's compiler (neuronx-cc) does not support complex dtypes, so the Hyena
FFT-convolutions fail with NCC_EVRF004. Both Hyena FFT paths convolve a signal
with a *real, finite* filter, so each is mathematically identical to a real
depthwise causal `F.conv1d` (the FFT is just an O(L log L) implementation of the
same linear convolution). This module swaps them for conv1d:

  1. parallel_fir, fir_length>=128  -> patched `fftconv_func` uses conv1d.
  2. parallel_iir (hcl blocks)       -> set config.long_fir_threshold so the
     engine's existing conv1d branch runs instead of rfft/irfft.

Import `apply_evo2_neuron_patches()` BEFORE constructing the model, and set
`config.long_fir_threshold` (helper `patch_config` does it).
"""
import importlib
import torch
import torch.nn.functional as F


def _conv1d_fftconv(u, k, D, dropout_mask, gelu=True, k_rev=None,
                    bidirectional=False, **kwargs):
    """Real depthwise causal convolution, drop-in for engine.fftconv_func.

    Computes the same linear convolution of `u` (B, C, L) with per-channel
    filter `k`, then adds the residual gain `u * D`. No complex/FFT ops.
    Evo2 prefill only uses the non-bidirectional, k_rev=None path.
    """
    assert not bidirectional and k_rev is None, "Neuron patch covers Evo2 prefill path only"
    seqlen = u.shape[-1]
    C = u.shape[-2]

    k = k.squeeze()
    if k.dim() == 1:                      # shared filter -> per-channel
        k = k.unsqueeze(0).expand(C, -1)
    Lk = k.shape[-1]

    # Original fftconv computes in fp32; match it, then cast back.
    # conv1d cross-correlates, so flip the (causal) filter; depthwise via groups=C.
    weight = k.flip(-1).reshape(C, 1, Lk).to(torch.float32)
    u3 = u.reshape(-1, C, seqlen).to(torch.float32)
    y = F.conv1d(u3, weight, padding=Lk - 1, groups=C)[..., :seqlen]
    y = y.reshape(u.shape)

    out = y + u.to(torch.float32) * D.to(torch.float32).unsqueeze(-1)
    return out.to(dtype=u.dtype)


def _conv1d_parallel_iir(self, z_pre, h, D, L, poles, residues, t, dims, layer_idx,
                         inference_params=None, **kwargs):
    """Real causal-conv replacement for HyenaInferenceEngine.parallel_iir (prefill).

    The hcl long convolution y = conv(x1v, h) is computed via FFT in the original
    (complex -> unsupported on Neuron). The modal impulse response `h` is real and
    length L, so this is an exact depthwise *causal convolution*. NOTE: the port's
    own `long_fir_threshold` conv1d branch is wrong (it correlates instead of
    convolving — no filter flip); this implementation flips the filter and matches
    the FFT path to fp32 round-off.
    """
    hidden_size = dims[0]
    x2, x1, v = z_pre.split([hidden_size, hidden_size, hidden_size], dim=1)
    if self.hyena_flip_x1x2:
        x1, x2 = x2, x1
    x1v = x1 * v                                  # (B, H, L)

    hh = h
    if hh.dim() == 3:                             # (1, H, L) -> (H, L)
        hh = hh[0]
    if hh.dim() == 1:
        hh = hh.unsqueeze(0).expand(x1v.shape[1], -1)
    Lk = hh.shape[-1]
    # Original FFT path computes in fp32; match it, then cast back.
    weight = hh.flip(-1).reshape(x1v.shape[1], 1, Lk).to(torch.float32)   # causal: flip
    x1v32 = x1v.to(torch.float32)
    y = F.conv1d(x1v32, weight, groups=x1v.shape[1], padding=Lk - 1)[..., :L]

    y = (y + x1v32 * D.to(torch.float32).unsqueeze(-1)) * x2.to(torch.float32)
    return y.to(x1v.dtype).permute(0, 2, 1)


def apply_evo2_neuron_patches(model_dirname: str | None = None):
    """Monkeypatch the model's remote engine: FFT FIR + FFT IIR -> real conv1d.

    Replaces `engine.fftconv_func` (FIR path) and
    `HyenaInferenceEngine.parallel_iir` (IIR/hcl path) so no complex/FFT ops reach
    neuronx-cc. Both replacements are validated against the FFT originals on CPU.
    """
    import sys
    n = 0
    for name, m in list(sys.modules.items()):
        if name.endswith(".engine") and hasattr(m, "fftconv_func"):
            m.fftconv_func = _conv1d_fftconv
            if hasattr(m, "HyenaInferenceEngine"):
                m.HyenaInferenceEngine.parallel_iir = _conv1d_parallel_iir
            n += 1
    if n == 0:
        raise RuntimeError("evo2 engine module not imported yet; import the model first")
    return n


def patch_config(cfg, seqlen):
    """parallel_iir is replaced wholesale (real conv1d), so disable the buggy
    built-in long_fir_threshold branch by leaving it None."""
    cfg.long_fir_threshold = None
    return cfg
