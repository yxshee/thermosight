"""Vision Transformer model used for ThermoSight."""

IMG_SIZE = 460
PATCH_SIZE = 8
EMBED_DIM = 768
ENC_LAYERS = 12
HEADS = 12
NUM_CLASSES = 4


import math
import torch
from torch import nn
import torch.nn.functional as F


class PatchEmbedding(nn.Module):
    """Converts images to patch embeddings."""

    def __init__(self, embed_dim: int, patch_size: int, num_patches: int, dropout: float, in_channels: int) -> None:
        super().__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.positional_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = self.patcher(x).permute(0, 2, 1)
        x = torch.cat([cls_token, x], dim=1)

        n_tokens = x.size(1)
        n_orig = self.positional_embeddings.size(1)
        if n_tokens != n_orig:
            cls_pos = self.positional_embeddings[:, :1, :]
            spatial_pos = self.positional_embeddings[:, 1:, :]
            gs_old = int(math.sqrt(spatial_pos.size(1)))
            gs_new = int(math.sqrt(n_tokens - 1))
            spatial_pos = spatial_pos.transpose(1, 2).view(1, -1, gs_old, gs_old)
            new_sp = F.interpolate(spatial_pos, size=(gs_new, gs_new), mode="bilinear", align_corners=False)
            new_sp = new_sp.flatten(2).transpose(1, 2)
            pos_emb = torch.cat((cls_pos, new_sp), dim=1)
        else:
            pos_emb = self.positional_embeddings

        x = x + pos_emb
        x = self.dropout(x)
        return x


class ViT(nn.Module):
    """Vision Transformer for image classification."""

    def __init__(
        self,
        num_patches: int | None = None,
        img_size: int = IMG_SIZE,
        num_classes: int = NUM_CLASSES,
        patch_size: int = PATCH_SIZE,
        embed_dim: int = EMBED_DIM,
        num_encoders: int = ENC_LAYERS,
        num_heads: int = HEADS,
        dropout: float = 0.0,
        activation: str = "gelu",
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        if num_patches is None:
            num_patches = (img_size // patch_size) ** 2
        self.embeddings_block = PatchEmbedding(embed_dim, patch_size, num_patches, dropout, in_channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=4 * embed_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x)
        x = self.mlp_head(x[:, 0, :])
        return x

