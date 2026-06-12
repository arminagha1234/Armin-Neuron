"""Real-valued RoPE replacement for QwenImage's complex64 QwenEmbedRope.

The stock `QwenEmbedRope` in diffusers uses `torch.polar(...)` to build a
complex64 freqs tensor, then `apply_rotary_emb_qwen` uses complex multiply.
On Neuron Beta 2:
  - Eager mode: complex64 ops fall back to CPU when
    `TORCH_NEURONX_FALLBACK_ONLY_FOR_UNIMPLEMENTED_OPS=0` (slow but works).
  - torch.compile mode: lowering crashes with
    `LLVM ERROR: unhandled type for getConstantWithGivenDtypeAndValue` /
    `DecomposeComplexOps pass crashed unexpectedly`.

This module provides a drop-in real-valued replacement. Same outputs as
stock (verified by `test_rope_equivalence.py`), but no complex
intermediates — works in both eager and compile mode.

The implementation is lifted from the proven contrib version
(`neuron/external/pr-117-nxdi-diffusion-models/contrib/models/Qwen-Image-Edit/src/neuron_rope.py`)
which has been validated end-to-end on Neuron. We keep the contrib math
(interleaved `use_real_unbind_dim=-1`, matching diffusers' stock layout)
and adapt the public API to match `customers/fal/path_c/`'s scaffold.

Math equivalence:
    Stock (complex):
        freqs_complex = torch.polar(ones, freqs_real)   # = exp(i*theta)
        x_complex     = view_as_complex(x_pairs)
        out_complex   = x_complex * freqs_complex
        out_real      = view_as_real(out_complex).flatten(-2)

    Real-valued (mathematically identical, no complex):
        cos = freqs_real.cos()
        sin = freqs_real.sin()
        x_a = x[..., 0::2]                              # real part
        x_b = x[..., 1::2]                              # imag part
        out_a = x_a * cos - x_b * sin                   # = real(x*exp(i*theta))
        out_b = x_a * sin + x_b * cos                   # = imag(x*exp(i*theta))
        out   = interleave(out_a, out_b)                # back to [a0,b0,a1,b1,...]

Integration order (CRITICAL — see design.md §"Phase 2.1"):
    1. dist.init_process_group(backend='neuron')
    2. with torch.device('meta'): model = ...
    3. mesh = init_device_mesh('neuron', (world_size,))
    4. parallelize_module(model, mesh, plan)
    5. apply_tp_fixes(model, world_size, rank)
    6. load_weights_sharded(...)
    7. (optional) rebuild pos_freqs/neg_freqs (no-op if step 8 runs)
    8. install_real_rope(model)                          ← THIS module
    9. (optional) torch.compile(model, backend='neuron', ...)
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
from torch import nn


class QwenEmbedRopeReal(nn.Module):
    """Real-valued drop-in for diffusers' `QwenEmbedRope`.

    Same constructor signature (`theta`, `axes_dim`, `scale_rope`).
    forward() returns `((vid_cos, vid_sin), (txt_cos, txt_sin))` instead
    of one complex tensor; pair this with `apply_rotary_emb_real` (which
    `install_real_rope` patches into the diffusers module).
    """

    def __init__(self, theta: int, axes_dim: List[int], scale_rope: bool = False):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.scale_rope = scale_rope

        # Same position indices as stock
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1

        # Stock: torch.polar(ones, freqs) -> exp(i*freqs) (complex64)
        # Real:  cos(freqs), sin(freqs) (two fp32 tensors)
        self.pos_freqs_cos, self.pos_freqs_sin = self._compute_all_freqs(pos_index)
        self.neg_freqs_cos, self.neg_freqs_sin = self._compute_all_freqs(neg_index)

    @staticmethod
    def _rope_params_real(
        index: torch.Tensor, dim: int, theta: int = 10000
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-axis freqs as (cos, sin) instead of complex polar.

        Equivalent to stock's `rope_params` followed by `torch.polar(ones, ·)`,
        but split into real-valued cos/sin.

        Returns:
            (cos, sin), each shape [len(index), dim//2]
        """
        assert dim % 2 == 0, "RoPE axes_dim entries must be even"
        freqs = torch.outer(
            index.float(),
            1.0 / torch.pow(theta, torch.arange(0, dim, 2).float() / dim),
        )
        return torch.cos(freqs), torch.sin(freqs)

    def _compute_all_freqs(
        self, index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-axis freqs concatenated along the last dim.

        Stock's `pos_freqs` is the concatenation of three per-axis complex
        tensors. We mirror that for both cos and sin.
        """
        cos_parts: List[torch.Tensor] = []
        sin_parts: List[torch.Tensor] = []
        for dim in self.axes_dim:
            cos_f, sin_f = self._rope_params_real(index, dim, self.theta)
            cos_parts.append(cos_f)
            sin_parts.append(sin_f)
        return torch.cat(cos_parts, dim=1), torch.cat(sin_parts, dim=1)

    def forward(
        self,
        video_fhw: Union[Tuple[int, int, int], List],
        txt_seq_lens: Optional[List[int]] = None,
        device: Optional[torch.device] = None,
        max_txt_seq_len: Optional[Union[int, torch.Tensor]] = None,
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]
    ]:
        """Mirror of stock `QwenEmbedRope.forward`, returning real freqs.

        Mirrors the *exact* algorithm from diffusers 0.38's QwenEmbedRope:
            - Outer list = batch wrapping (warns on mixed sizes, takes [0])
            - Inner list = per-image (canvas, input1, input2, ...) with
              potentially different (H, W). Frame axis uses an idx offset
              per image so the canvas and each input get distinct frame
              positions.
            - max_vid_index = max(H, W) across ALL inner-list images
              (or max(H/2, W/2) under scale_rope).
            - txt_freqs is `pos_freqs[max_vid_index : max_vid_index + max_txt_seq_len]`.

        Returns:
            ((vid_cos, vid_sin), (txt_cos, txt_sin))
            vid_*: [sum_i(frame_i * H_i * W_i), sum(axes_dim)//2]
            txt_*: [max_txt_seq_len, sum(axes_dim)//2]
        """
        # Compatibility: stock signature accepts either txt_seq_lens
        # (deprecated) or max_txt_seq_len.
        if txt_seq_lens is not None and max_txt_seq_len is None:
            max_txt_seq_len = (
                max(txt_seq_lens) if isinstance(txt_seq_lens, list) else txt_seq_lens
            )
        if max_txt_seq_len is None:
            raise ValueError(
                "Either max_txt_seq_len or txt_seq_lens must be provided."
            )

        # If outer list (batch wrapping), take first batch entry like stock.
        if isinstance(video_fhw, list) and len(video_fhw) > 0 and isinstance(
            video_fhw[0], list
        ):
            video_fhw = video_fhw[0]
        # If a single tuple, wrap into a list for uniform iteration.
        if not isinstance(video_fhw, list):
            video_fhw = [video_fhw]

        # Per-image freqs, with monotonically-increasing frame `idx`.
        vid_cos_parts: List[torch.Tensor] = []
        vid_sin_parts: List[torch.Tensor] = []
        max_vid_index = 0
        for idx, fhw in enumerate(video_fhw):
            frame, height, width = fhw
            cos_i, sin_i = self._compute_video_freqs(
                frame, height, width, idx, device
            )
            vid_cos_parts.append(cos_i)
            vid_sin_parts.append(sin_i)
            if self.scale_rope:
                max_vid_index = max(height // 2, width // 2, max_vid_index)
            else:
                max_vid_index = max(height, width, max_vid_index)

        vid_cos = torch.cat(vid_cos_parts, dim=0)
        vid_sin = torch.cat(vid_sin_parts, dim=0)

        max_txt = int(max_txt_seq_len)
        pos_cos = (
            self.pos_freqs_cos.to(device) if device is not None else self.pos_freqs_cos
        )
        pos_sin = (
            self.pos_freqs_sin.to(device) if device is not None else self.pos_freqs_sin
        )
        txt_cos = pos_cos[max_vid_index : max_vid_index + max_txt]
        txt_sin = pos_sin[max_vid_index : max_vid_index + max_txt]

        return (vid_cos, vid_sin), (txt_cos, txt_sin)

    def _compute_video_freqs(
        self,
        frame: int,
        height: int,
        width: int,
        idx: int = 0,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build per-token (cos, sin) for one [F, H, W] image at frame
        offset `idx`.

        Mirrors stock's `_compute_video_freqs` exactly: the frame axis
        slice is `freqs_pos[0][idx : idx + frame]`, NOT `[:frame]`. This
        is essential for the Plus pipeline where canvas (idx=0) and
        each input (idx=1, 2, ...) need distinct frame positions.
        """
        seq_lens = frame * height * width

        pos_cos = (
            self.pos_freqs_cos.to(device) if device is not None else self.pos_freqs_cos
        )
        pos_sin = (
            self.pos_freqs_sin.to(device) if device is not None else self.pos_freqs_sin
        )
        neg_cos = (
            self.neg_freqs_cos.to(device) if device is not None else self.neg_freqs_cos
        )
        neg_sin = (
            self.neg_freqs_sin.to(device) if device is not None else self.neg_freqs_sin
        )

        split_dims = [d // 2 for d in self.axes_dim]
        pos_cos_split = pos_cos.split(split_dims, dim=1)
        pos_sin_split = pos_sin.split(split_dims, dim=1)
        neg_cos_split = neg_cos.split(split_dims, dim=1)
        neg_sin_split = neg_sin.split(split_dims, dim=1)

        # Frame axis with per-image offset (THE KEY DETAIL).
        f_cos = (
            pos_cos_split[0][idx : idx + frame]
            .view(frame, 1, 1, -1)
            .expand(frame, height, width, -1)
        )
        f_sin = (
            pos_sin_split[0][idx : idx + frame]
            .view(frame, 1, 1, -1)
            .expand(frame, height, width, -1)
        )

        if self.scale_rope:
            h_neg_len = height - height // 2
            h_cos = torch.cat(
                [neg_cos_split[1][-h_neg_len:], pos_cos_split[1][: height // 2]], dim=0
            )
            h_sin = torch.cat(
                [neg_sin_split[1][-h_neg_len:], pos_sin_split[1][: height // 2]], dim=0
            )
            h_cos = h_cos.view(1, height, 1, -1).expand(frame, height, width, -1)
            h_sin = h_sin.view(1, height, 1, -1).expand(frame, height, width, -1)

            w_neg_len = width - width // 2
            w_cos = torch.cat(
                [neg_cos_split[2][-w_neg_len:], pos_cos_split[2][: width // 2]], dim=0
            )
            w_sin = torch.cat(
                [neg_sin_split[2][-w_neg_len:], pos_sin_split[2][: width // 2]], dim=0
            )
            w_cos = w_cos.view(1, 1, width, -1).expand(frame, height, width, -1)
            w_sin = w_sin.view(1, 1, width, -1).expand(frame, height, width, -1)
        else:
            h_cos = (
                pos_cos_split[1][:height]
                .view(1, height, 1, -1)
                .expand(frame, height, width, -1)
            )
            h_sin = (
                pos_sin_split[1][:height]
                .view(1, height, 1, -1)
                .expand(frame, height, width, -1)
            )
            w_cos = (
                pos_cos_split[2][:width]
                .view(1, 1, width, -1)
                .expand(frame, height, width, -1)
            )
            w_sin = (
                pos_sin_split[2][:width]
                .view(1, 1, width, -1)
                .expand(frame, height, width, -1)
            )

        cos = torch.cat([f_cos, h_cos, w_cos], dim=-1).reshape(seq_lens, -1)
        sin = torch.cat([f_sin, h_sin, w_sin], dim=-1).reshape(seq_lens, -1)
        return cos.clone().contiguous(), sin.clone().contiguous()


def apply_rotary_emb_real(
    x: torch.Tensor,
    freqs_cis,
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> torch.Tensor:
    """Drop-in replacement for diffusers' `apply_rotary_emb_qwen`.

    Accepts a (cos, sin) tuple in `freqs_cis` instead of a complex tensor.
    Falls through to a pass-through if `freqs_cis` is somehow still
    complex (defensive — should never happen post-`install_real_rope`).

    Math (interleaved layout, matches `use_real_unbind_dim=-1`):
        x_pairs[..., 0] is real part, x_pairs[..., 1] is imag part
        out_real[..., 0] = real*cos - imag*sin
        out_real[..., 1] = real*sin + imag*cos

    Args:
        x: shape [B, S, H, D] (B=batch, S=seq, H=heads, D=head_dim)
        freqs_cis: (cos, sin) tuple, each [S, D//2]
    """
    # Defensive fallback — if a complex tensor sneaks through, do nothing
    # special (let the caller's complex path handle it).
    if isinstance(freqs_cis, torch.Tensor) and torch.is_complex(freqs_cis):
        # Stock behavior: complex multiply then view_as_real then flatten.
        # We don't try to emulate that here; if this branch ever fires, the
        # patch ordering is wrong and the caller should be fixed.
        raise RuntimeError(
            "apply_rotary_emb_real received a complex tensor — "
            "install_real_rope() didn't replace pos_embed properly."
        )

    cos, sin = freqs_cis
    # cos/sin: [S, D//2]; expand to [S, D] by per-pair repeat (so element k
    # of the pair-axis broadcasts to elements 2k AND 2k+1 of the head dim).
    cos = cos.repeat_interleave(2, dim=-1)
    sin = sin.repeat_interleave(2, dim=-1)
    # Broadcast to x: [1, S, 1, D]
    cos = cos.unsqueeze(0).unsqueeze(2).to(x.device)
    sin = sin.unsqueeze(0).unsqueeze(2).to(x.device)

    if use_real_unbind_dim == -1:
        # Stock layout: x = [a0, b0, a1, b1, ...] in head_dim
        orig_shape = x.shape
        x_pairs = x.view(*orig_shape[:-1], -1, 2)               # [..., D//2, 2]
        x_rotated = torch.cat(
            [-x_pairs[..., 1:2], x_pairs[..., 0:1]], dim=-1
        )                                                        # [..., D//2, 2]
        x_rotated = x_rotated.view(orig_shape)
    elif use_real_unbind_dim == -2:
        # Half-split layout: x = [a0, a1, ..., b0, b1, ...]
        half_d = x.shape[-1] // 2
        x_real = x[..., :half_d]
        x_imag = x[..., half_d:]
        x_rotated = torch.cat([-x_imag, x_real], dim=-1)
    else:
        raise ValueError(
            f"use_real_unbind_dim={use_real_unbind_dim} but should be -1 or -2."
        )

    # out = x * cos + x_rotated * sin (fp32 then back to input dtype)
    return (x.float() * cos + x_rotated.float() * sin).to(x.dtype)


def install_real_rope(transformer: nn.Module) -> nn.Module:
    """Replace `transformer.pos_embed` and patch the module-level
    `apply_rotary_emb_qwen` in diffusers with the real-valued versions.

    Call AFTER `apply_tp_fixes` and `load_weights_sharded`. See the
    integration order in this module's docstring and design.md §"Phase 2.1".

    Returns the transformer (in-place modification, return is for chaining).
    """
    old = transformer.pos_embed
    new = QwenEmbedRopeReal(
        theta=old.theta, axes_dim=old.axes_dim, scale_rope=old.scale_rope
    )
    transformer.pos_embed = new

    # Patch the diffusers module-level rotary apply.
    import diffusers.models.transformers.transformer_qwenimage as qwen_mod

    if not hasattr(qwen_mod, "_orig_apply_rotary_emb_qwen"):
        qwen_mod._orig_apply_rotary_emb_qwen = qwen_mod.apply_rotary_emb_qwen
    qwen_mod.apply_rotary_emb_qwen = apply_rotary_emb_real

    return transformer
