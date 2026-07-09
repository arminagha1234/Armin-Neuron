"""Clay Encoder + Decoder (verbatim from Clay-foundation/model claymodel/model.py).

Only the frozen DINOv2 teacher (timm) and the LightningModule wrapper are omitted,
since neither is part of the encoder/decoder/dynamic-embedding training path.
"""

import math

import torch
from einops import rearrange, repeat
from torch import nn

from claymodel.backbone import Transformer
from claymodel.factory import DynamicEmbedding
from claymodel.utils import posemb_sincos_2d_with_gsd

torch.set_float32_matmul_precision("medium")


class Encoder(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        mask_ratio,
        patch_size,
        shuffle,
        dim,
        depth,
        heads,
        dim_head,
        mlp_ratio,
        fused_attn=True,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.shuffle = shuffle
        self.dim = dim
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        self.patch_embedding = DynamicEmbedding(
            wave_dim=128,
            num_latent_tokens=128,
            patch_size=patch_size,
            embed_dim=dim,
            is_decoder=False,
        )

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=int(dim * mlp_ratio),
            fused_attn=fused_attn,
        )

    def to_patch_embed(self, cube, waves):
        patches, waves_encoded = self.patch_embedding(cube, waves)  # [B L D]
        return patches, waves_encoded  # ([B L D], [N D])

    def add_encodings(self, patches, time, latlon, gsd):
        B, L, D = patches.shape

        grid_size = int(math.sqrt(L))
        self.num_patches = grid_size**2

        pos_encoding = (
            posemb_sincos_2d_with_gsd(
                h=grid_size,
                w=grid_size,
                dim=(self.dim - 8),
                gsd=gsd,
            )
            .to(patches.device)
            .detach()
        )  # [L (D - 8)]

        time_latlon = torch.hstack((time, latlon)).to(patches.device).detach()  # [B 8]

        pos_encoding = repeat(pos_encoding, "L D -> B L D", B=B)  # [B L (D - 8)]
        time_latlon = repeat(time_latlon, "B D -> B L D", L=L)  # [B L 8]
        pos_metadata_encoding = torch.cat(
            (pos_encoding, time_latlon), dim=-1
        )  # [B L D]

        patches = patches + pos_metadata_encoding.to(patches.dtype)  # bf16-safe
        return patches  # [B L D]

    @torch._dynamo.disable  # argsort->AwsNeuronTopK + RNG don't lower under compile;
    def mask_out(self, patches):  # run masking as an eager island, compile the rest
        B, L, D = patches.shape

        if self.shuffle:  # Shuffle the patches
            noise = torch.randn((B, L), device=patches.device)  # [B L]
        else:
            noise = rearrange(
                torch.arange(B * L, device=patches.device), "(B L) -> B L", B=B, L=L
            )

        random_indices = torch.argsort(noise, dim=-1)  # [B L]
        reverse_indices = torch.argsort(random_indices, dim=-1)  # [B L]

        num_masked_patches = int(self.mask_ratio * self.num_patches)
        masked_indices, unmasked_indices = (
            random_indices[:, :num_masked_patches],  # [B mask_ratio * L]
            random_indices[:, num_masked_patches:],  # [B (1 - mask_ratio) * L]
        )

        masked_matrix = torch.zeros((B, L), device=patches.device)  # [B L] = 0
        masked_matrix[:, :num_masked_patches] = 1  # [B mask_ratio * L] = 1
        masked_matrix = torch.gather(
            masked_matrix, dim=1, index=reverse_indices
        )  # [B L] -> [B L] - reorder the patches

        batch_indices = rearrange(
            torch.arange(B, device=patches.device), "B -> B 1"
        )  # [B 1]
        unmasked_patches = patches[
            batch_indices, unmasked_indices, :
        ]  # [B L:(1 - mask_ratio) D]
        _ = patches[batch_indices, masked_indices, :]  # [B L:mask_ratio D]

        return (
            unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        )

    def forward(self, datacube):
        cube, time, latlon, gsd, waves = (
            datacube["pixels"],  # [B C H W]
            datacube["time"],  # [B 2]
            datacube["latlon"],  # [B 2]
            datacube["gsd"],  # 1
            datacube["waves"],  # [N]
        )  # [B C H W]

        B, C, H, W = cube.shape

        patches, waves_encoded = self.to_patch_embed(cube, waves)  # [B L D]
        patches = self.add_encodings(patches, time, latlon, gsd)  # [B L D]

        (
            unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        ) = self.mask_out(patches)

        cls_tokens = repeat(self.cls_token, "1 1 D -> B 1 D", B=B)  # [B 1 D]
        unmasked_patches = torch.cat(
            (cls_tokens, unmasked_patches), dim=1
        )  # [B (1 + L) D]

        encoded_unmasked_patches = self.transformer(unmasked_patches)

        return (
            encoded_unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        )


class Decoder(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        mask_ratio,
        patch_size,
        encoder_dim,
        dim,
        depth,
        heads,
        dim_head,
        mlp_ratio,
        fused_attn=True,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.encoder_dim = encoder_dim
        self.dim = dim

        self.enc_to_dec = (
            nn.Linear(encoder_dim, dim) if encoder_dim != dim else nn.Identity()
        )
        self.mask_patch = nn.Parameter(torch.randn(dim))
        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=int(dim * mlp_ratio),
            fused_attn=fused_attn,
        )
        self.embed_to_pixels = DynamicEmbedding(
            wave_dim=128,
            num_latent_tokens=128,
            patch_size=patch_size,
            embed_dim=dim,
            is_decoder=True,
        )

    def reconstruct_and_add_encoding(  # noqa: PLR0913
        self,
        unmasked_patches,
        unmasked_indices,
        masked_indices,
        masked_matrix,
        time,
        latlon,
        gsd,
    ):
        B, L = masked_matrix.shape
        grid_size = int(math.sqrt(L))
        self.num_patches = grid_size**2
        cls_tokens, unmasked_patches = (
            unmasked_patches[:, :1, :],
            unmasked_patches[:, 1:, :],
        )

        pos_encoding = (
            posemb_sincos_2d_with_gsd(
                h=grid_size, w=grid_size, dim=(self.dim - 8), gsd=gsd
            )
            .to(unmasked_patches.device)
            .detach()
        )
        time_latlon = (
            torch.hstack((time, latlon)).to(unmasked_patches.device).detach()
        )

        pos_encoding = repeat(pos_encoding, "L D -> B L D", B=B)
        time_latlon = repeat(time_latlon, "B D -> B L D", L=L)
        pos_metadata_encoding = torch.cat(
            (pos_encoding, time_latlon), dim=-1
        ).to(unmasked_patches.dtype)  # bf16-safe

        batch_indices = rearrange(
            torch.arange(B, device=unmasked_patches.device), "B -> B 1"
        )

        num_masked_patches = int(self.mask_ratio * self.num_patches)
        masked_patches = repeat(
            self.mask_patch, "D -> B L D", B=B, L=num_masked_patches
        )

        masked_patches = (
            masked_patches + pos_metadata_encoding[batch_indices, masked_indices, :]
        )
        unmasked_patches = (
            unmasked_patches + pos_metadata_encoding[batch_indices, unmasked_indices, :]
        )

        decoder_patches = torch.zeros(
            (B, self.num_patches, self.dim), device=unmasked_patches.device,
            dtype=unmasked_patches.dtype,
        )
        decoder_patches[batch_indices, unmasked_indices, :] = unmasked_patches
        decoder_patches[batch_indices, masked_indices, :] = masked_patches

        decoder_patches = torch.cat((cls_tokens, decoder_patches), dim=1)

        return decoder_patches

    def forward(  # noqa: PLR0913
        self,
        encoded_unmasked_patches,
        unmasked_indices,
        masked_indices,
        masked_matrix,
        time,
        latlon,
        gsd,
        waves,
    ):
        encoded_unmasked_patches = self.enc_to_dec(encoded_unmasked_patches)

        decoder_patches = self.reconstruct_and_add_encoding(
            encoded_unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
            time,
            latlon,
            gsd,
        )

        decoded_patches = self.transformer(decoder_patches)

        pixels, waves = self.embed_to_pixels(decoded_patches, waves)
        pixels = pixels[:, 1:, :]
        return pixels, waves


# ---------------------------------------------------------------------------
# Full ClayMAE (with frozen DINOv2 teacher + representation loss), verbatim from
# Clay-foundation/model claymodel/model.py, with two minimal edits for Beta-3:
#   1) torchvision v2.Resize -> F.interpolate (drops the torchvision dependency)
#   2) fused_attn threaded through so we can use manual attention (SDPA-bwd blocker)
# The MRL head is left commented out exactly as in upstream (they use proj).
# ---------------------------------------------------------------------------
import random

import torch.nn.functional as F
from einops import reduce


class _HFDinoTeacher(nn.Module):
    """Frozen DINOv2 teacher via HuggingFace transformers (no torchvision dep).

    Same DINOv2 weights lineage as timm's *_dinov2.lvd142m; returns pooled
    [B, hidden] features, matching timm create_model(num_classes=0) semantics.
    """

    def __init__(self, name="facebook/dinov2-large"):
        super().__init__()
        from transformers import Dinov2Model
        self.model = Dinov2Model.from_pretrained(name)
        self.num_features = self.model.config.hidden_size

    def forward(self, rgb):
        out = self.model(pixel_values=rgb, interpolate_pos_encoding=True)
        return out.pooler_output


def _build_teacher(teacher, impl):
    if impl == "transformers":
        return _HFDinoTeacher(teacher)
    import timm  # lazy: only needed for the timm path (requires torchvision)
    return timm.create_model(teacher, pretrained=True, num_classes=0)


class ClayMAE(nn.Module):
    def __init__(  # noqa: PLR0913
        self,
        mask_ratio,
        patch_size,
        norm_pix_loss,
        shuffle,
        metadata,
        teacher,
        dolls,
        doll_weights,
        dim,
        depth,
        heads,
        dim_head,
        mlp_ratio,
        decoder_dim,
        decoder_depth,
        decoder_heads,
        decoder_dim_head,
        decoder_mlp_ratio,
        fused_attn=True,
        teacher_impl="transformers",
        **kwargs,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        self.shuffle = shuffle
        self.metadata = metadata
        self.teacher = _build_teacher(teacher, teacher_impl)
        self.teacher_chip_size = 518
        self.proj = nn.Linear(dim, self.teacher.num_features)

        self.encoder = Encoder(
            mask_ratio=mask_ratio, patch_size=patch_size, shuffle=shuffle, dim=dim,
            depth=depth, heads=heads, dim_head=dim_head, mlp_ratio=mlp_ratio,
            fused_attn=fused_attn,
        )
        self.decoder = Decoder(
            mask_ratio=mask_ratio, patch_size=patch_size, encoder_dim=dim,
            dim=decoder_dim, depth=decoder_depth, heads=decoder_heads,
            dim_head=decoder_dim_head, mlp_ratio=decoder_mlp_ratio,
            fused_attn=fused_attn,
        )
        self.freeze_teacher()

    def teacher_resize(self, x):
        return F.interpolate(
            x, size=(self.teacher_chip_size, self.teacher_chip_size),
            mode="bilinear", align_corners=False,
        )

    def freeze_teacher(self):
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()

    def per_pixel_loss(self, cube, pixels, masked_matrix):
        patches = rearrange(
            cube, "B C (h p1) (w p2) -> B (h w) (C p1 p2)",
            p1=self.patch_size, p2=self.patch_size,
        )
        if self.norm_pix_loss:
            mean = patches.mean(dim=-1, keepdim=True)
            var = patches.var(dim=-1, keepdim=True)
            patches = (patches - mean) / (var + 1e-6) ** 0.5
        loss = F.l1_loss(patches, pixels, reduction="none")
        loss = reduce(loss, "B L D -> B L", reduction="mean")
        loss = (loss * masked_matrix).sum() / masked_matrix.sum()
        return loss

    def forward(self, datacube):
        platform = datacube["platform"][0]
        waves = torch.tensor(list(self.metadata[platform].bands.wavelength.values()))
        gsd = torch.tensor(self.metadata[platform].gsd)

        _pixels = datacube["pixels"].clone()
        batch_size, channels, _, _ = _pixels.size()
        prob_drop_all = 0.10
        prob_drop_half = 0.20
        for i in range(batch_size):
            if torch.any(datacube["latlon"][i] != 0):
                rand_val = random.random()
                if rand_val < prob_drop_all:
                    _pixels[i, :, :, :] = 0
                elif rand_val < prob_drop_all + prob_drop_half:
                    channel_indices = torch.randperm(channels)[: channels // 2]
                    _pixels[i, channel_indices, :, :] = 0

        (
            encoded_unmasked_patches,
            unmasked_indices,
            masked_indices,
            masked_matrix,
        ) = self.encoder({
            "pixels": _pixels,
            "time": datacube["time"],
            "latlon": datacube["latlon"],
            "gsd": gsd,
            "waves": waves,
        })

        pixels, waves = self.decoder(
            encoded_unmasked_patches, unmasked_indices, masked_indices,
            masked_matrix, datacube["time"], datacube["latlon"], gsd, waves,
        )

        reconstruction_loss = self.per_pixel_loss(
            datacube["pixels"], pixels, masked_matrix
        )
        if platform == "modis":
            reconstruction_loss /= 10

        representations = self.proj(encoded_unmasked_patches[:, 0, :])

        with torch.no_grad():
            if platform == "sentinel-1-rtc":
                r = datacube["pixels"][:, 0, :, :]
                g = datacube["pixels"][:, 1, :, :]
                b = (r + g) / 2
                rgb = torch.stack((r, g, b), dim=1)
            else:
                indices = self.metadata[platform].rgb_indices
                rgb = datacube["pixels"][:, indices, :, :]
            rgb = self.teacher_resize(rgb)
            target = self.teacher(rgb)

        representation_loss = 1.0 - F.cosine_similarity(representations, target).mean()

        loss = 0.9 * reconstruction_loss + 0.1 * representation_loss
        return (loss, reconstruction_loss, representation_loss)


def clay_mae_config(size):
    cfgs = {
        "tiny": dict(dim=192, depth=6, heads=4, dim_head=48, mlp_ratio=2,
                     decoder_dim=96, decoder_depth=3, decoder_heads=2,
                     decoder_dim_head=48, decoder_mlp_ratio=2),
        "small": dict(dim=384, depth=6, heads=6, dim_head=64, mlp_ratio=2,
                      decoder_dim=192, decoder_depth=4, decoder_heads=4,
                      decoder_dim_head=64, decoder_mlp_ratio=2),
        "base": dict(dim=768, depth=12, heads=12, dim_head=64, mlp_ratio=4,
                     decoder_dim=512, decoder_depth=4, decoder_heads=4,
                     decoder_dim_head=64, decoder_mlp_ratio=4),
        "large": dict(dim=1024, depth=24, heads=16, dim_head=64, mlp_ratio=4,
                      decoder_dim=512, decoder_depth=4, decoder_heads=4,
                      decoder_dim_head=64, decoder_mlp_ratio=4),
    }
    return cfgs[size]
