"""Sight: turning images into tokens the language model can read.

MotherBrain's transformer consumes a sequence of vectors. Text becomes vectors
by looking words up in an embedding table; an image becomes vectors by cutting
it into square patches and projecting each one. After that the model cannot
tell them apart, which is the whole trick behind multimodal models of this
shape: one sequence, two sources.

The encoder here is a small vision transformer. It is deliberately its own
tower rather than a few extra layers of the language model, so that an image
is understood before it is handed over, and so that a text-only model stays
byte-for-byte unchanged when no image is present.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from motherbrain.config import ModelConfig
from motherbrain.model import RMSNorm


class PatchEmbed(nn.Module):
    """Cut the image into patches and project each into a vector.

    A convolution whose kernel and stride both equal the patch size is exactly
    "chop into tiles, then apply a linear layer to each tile" - the standard
    way to write it, and faster than doing it literally.
    """

    def __init__(self, image_size: int, patch_size: int, width: int,
                 channels: int = 3) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError(
                f"image_size {image_size} is not divisible by patch_size {patch_size}")
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(channels, width, kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        b, c, h, w = images.shape
        if (h, w) != (self.image_size, self.image_size):
            images = F.interpolate(images, size=(self.image_size, self.image_size),
                                   mode="bilinear", align_corners=False)
        x = self.proj(images)                      # (B, width, gh, gw)
        return x.flatten(2).transpose(1, 2)        # (B, n_patches, width)


class VisionBlock(nn.Module):
    """One transformer block over image patches.

    Bidirectional rather than causal: a patch in the top-left may depend on one
    in the bottom-right, and there is no reading order to respect.
    """

    def __init__(self, width: int, heads: int, mlp_ratio: float = 4.0,
                 dropout: float = 0.0) -> None:
        super().__init__()
        if width % heads:
            raise ValueError(f"vision width {width} is not divisible by {heads} heads")
        self.heads = heads
        self.head_dim = width // heads
        self.norm1 = RMSNorm(width)
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.norm2 = RMSNorm(width)
        hidden = int(width * mlp_ratio)
        self.fc1 = nn.Linear(width, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, width, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, w = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).view(b, n, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.proj(attended.transpose(1, 2).reshape(b, n, w))
        h = self.norm2(x)
        return x + self.fc2(F.gelu(self.fc1(h)))


class VisionTower(nn.Module):
    """Image in, language-model tokens out.

    The output is `n_patches` vectors of `d_model`, which the language model
    consumes exactly as it consumes word embeddings.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch = PatchEmbed(cfg.image_size, cfg.patch_size, cfg.vision_width)
        self.pos = nn.Parameter(
            torch.zeros(1, self.patch.n_patches, cfg.vision_width))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([
            VisionBlock(cfg.vision_width, cfg.vision_heads, dropout=cfg.dropout)
            for _ in range(cfg.vision_layers)
        ])
        self.norm = RMSNorm(cfg.vision_width)
        # The bridge: whatever the vision tower learned, expressed in the
        # language model's own dimensions.
        self.to_text = nn.Linear(cfg.vision_width, cfg.d_model, bias=False)

    @property
    def n_tokens(self) -> int:
        """How many positions an image occupies in the sequence."""
        return self.patch.n_patches

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch(images) + self.pos
        for block in self.blocks:
            x = block(x)
        return self.to_text(self.norm(x))


def load_image(path: str, size: int) -> torch.Tensor:
    """Read an image file into the (1, 3, size, size) tensor the tower wants."""
    from PIL import Image

    with Image.open(path) as img:
        img = img.convert("RGB").resize((size, size), Image.BILINEAR)
        # bytearray rather than bytes: torch will not take a read-only buffer
        # without warning about writability it cannot guarantee.
        data = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    x = data.view(size, size, 3).permute(2, 0, 1).float() / 255.0
    # Centre on zero, the range these blocks are initialised for.
    return ((x - 0.5) / 0.5).unsqueeze(0)
