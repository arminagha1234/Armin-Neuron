"""
Self-contained VideoMAE v2 *pretraining* model for native PyTorch on AWS Trainium.

Faithful port of OpenGVLab/VideoMAEv2 `models/modeling_pretrain.py`
(PretrainVisionTransformer = encoder-on-visible + decoder + mask-token + pixel head),
with ONE deliberate change for Neuron compatibility:

    the reference selects visible/masked tokens with BOOLEAN mask indexing
    (`x[~mask].reshape(B, -1, C)`), which is a data-dependent (dynamic-shape)
    `masked_select`. Compiled accelerators (Neuron, and XLA/TPU) need static shapes,
    so we instead pass integer index tensors (ids_keep / ids_mask) and use
    `torch.gather`. Tube masking keeps the visible/masked counts identical across
    samples, so this is exact and fully static. This is the standard "MAE-on-XLA"
    trick and is the only pretraining-specific adaptation required.

Reuses the already-vendored Block / PatchEmbed / get_sinusoid_encoding_table /
trunc_normal_ from modeling_videomaev2_native.py (identical to the repo's
modeling_finetune building blocks). No timm / DeepSpeed / flash-attn / custom CUDA.

Base config (pretrain_videomae_base_patch16_224):
  encoder: embed 768, depth 12, heads 12
  decoder: embed 384, depth 8,  heads 6,  out = 3*tubelet*patch^2 = 1536
  img 224, patch 16, tubelet 2, frames 16  ->  T'=8, H'=W'=14, N=1568 tokens
"""
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from modeling_videomaev2_native import (
    Block,
    PatchEmbed,
    get_sinusoid_encoding_table,
    trunc_normal_,
)


class PretrainEncoder(nn.Module):
    """ViT encoder that runs only on the *visible* tokens (selected via gather)."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 drop_path_rate=0.0, norm_layer=nn.LayerNorm, init_values=0.0,
                 tubelet_size=2, num_frames=16):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim,
                                      num_frames=num_frames, tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches
        # fixed sin-cos positional embedding [1, N, C]
        self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop_path=dpr[i], norm_layer=norm_layer,
                  init_values=init_values) for i in range(depth)])
        self.norm = norm_layer(embed_dim)

    def forward(self, x, ids_keep):
        # x: (B, C, T, H, W) -> patch tokens (B, N, C)
        x = self.patch_embed(x)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        B, N, C = x.shape
        # STATIC gather of visible tokens instead of boolean x[~mask]
        idx = ids_keep.unsqueeze(-1).expand(-1, -1, C)          # (B, N_vis, C)
        x_vis = torch.gather(x, 1, idx)                          # (B, N_vis, C)
        for blk in self.blocks:
            x_vis = blk(x_vis)
        return self.norm(x_vis)


class PretrainDecoder(nn.Module):
    """Lightweight ViT decoder that predicts per-patch pixels for masked tokens."""

    def __init__(self, num_classes=1536, embed_dim=384, depth=8, num_heads=6,
                 mlp_ratio=4.0, qkv_bias=True, drop_path_rate=0.0,
                 norm_layer=nn.LayerNorm, init_values=0.0):
        super().__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop_path=dpr[i], norm_layer=norm_layer,
                  init_values=init_values) for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, return_token_num):
        for blk in self.blocks:
            x = blk(x)
        # only decode/return the trailing mask-token predictions
        x = self.head(self.norm(x[:, -return_token_num:]))
        return x


class PretrainVideoMAE(nn.Module):
    """Full VideoMAE(v2) pretraining model (dual-masking off = standard full decode)."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
                 decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=6,
                 decoder_num_classes=1536, mlp_ratio=4.0, qkv_bias=True,
                 drop_path_rate=0.0, tubelet_size=2, num_frames=16, init_values=0.0):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        assert decoder_num_classes == 3 * tubelet_size * patch_size ** 2
        self.encoder = PretrainEncoder(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=encoder_embed_dim, depth=encoder_depth, num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_path_rate=drop_path_rate,
            norm_layer=norm_layer, init_values=init_values, tubelet_size=tubelet_size,
            num_frames=num_frames)
        self.decoder = PretrainDecoder(
            num_classes=decoder_num_classes, embed_dim=decoder_embed_dim,
            depth=decoder_depth, num_heads=decoder_num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, drop_path_rate=drop_path_rate, norm_layer=norm_layer,
            init_values=init_values)
        self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        num_patches = self.encoder.patch_embed.num_patches
        self.pos_embed = get_sinusoid_encoding_table(num_patches, decoder_embed_dim)
        trunc_normal_(self.mask_token, std=.02)

    def forward(self, x, ids_keep, ids_mask):
        x_vis = self.encoder(x, ids_keep)                    # (B, N_vis, C_e)
        x_vis = self.encoder_to_decoder(x_vis)               # (B, N_vis, C_d)
        B, N_vis, C = x_vis.shape
        expand_pos = self.pos_embed.expand(B, -1, -1).type_as(x).to(x.device).clone().detach()
        pos_vis = torch.gather(expand_pos, 1, ids_keep.unsqueeze(-1).expand(-1, -1, C))
        pos_mask = torch.gather(expand_pos, 1, ids_mask.unsqueeze(-1).expand(-1, -1, C))
        N_mask = ids_mask.shape[1]
        mask_tokens = self.mask_token.expand(B, N_mask, -1)
        x_full = torch.cat([x_vis + pos_vis, mask_tokens + pos_mask], dim=1)
        return self.decoder(x_full, N_mask)                  # (B, N_mask, decoder_num_classes)


def build_pretrain_videomae_base(**overrides):
    cfg = dict(img_size=224, patch_size=16, in_chans=3,
               encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
               decoder_embed_dim=384, decoder_depth=8, decoder_num_heads=6,
               decoder_num_classes=1536, mlp_ratio=4.0, qkv_bias=True,
               tubelet_size=2, num_frames=16)
    cfg.update(overrides)
    return PretrainVideoMAE(**cfg)


# ----------------------------------------------------------------------------------
# Tube masking -> integer keep/mask indices (host-side, numpy). Same spatial pattern
# tiled across all T' temporal positions, matching TubeMaskingGenerator. Counts are
# identical per sample so the batched index tensors are static-shape.
# ----------------------------------------------------------------------------------
def tube_mask_indices(batch, Tp=8, Hp=14, Wp=14, mask_ratio=0.9, rng=None):
    rng = rng or np.random
    n_spatial = Hp * Wp                       # 196
    n_mask_sp = int(mask_ratio * n_spatial)   # 176 at 0.9
    n_vis_sp = n_spatial - n_mask_sp          # 20
    keep_list, mask_list = [], []
    for _ in range(batch):
        perm = rng.permutation(n_spatial)
        vis_sp = np.sort(perm[:n_vis_sp])
        mask_sp = np.sort(perm[n_vis_sp:])
        # tile the same spatial choice across all Tp temporal groups
        ids_keep = np.concatenate([t * n_spatial + vis_sp for t in range(Tp)])
        ids_mask = np.concatenate([t * n_spatial + mask_sp for t in range(Tp)])
        keep_list.append(ids_keep)
        mask_list.append(ids_mask)
    ids_keep = torch.as_tensor(np.stack(keep_list), dtype=torch.long)   # (B, Tp*n_vis_sp)
    ids_mask = torch.as_tensor(np.stack(mask_list), dtype=torch.long)   # (B, Tp*n_mask_sp)
    return ids_keep, ids_mask
