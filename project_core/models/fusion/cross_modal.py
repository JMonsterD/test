"""
Cross-modal encoders and attention for HSI/MSI interaction.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvEncoder(nn.Module):
    def __init__(self, in_ch: int, d_model: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, d_model, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(d_model, d_model, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncY(_ConvEncoder):
    pass


class EncZ(_ConvEncoder):
    pass


class EncX(_ConvEncoder):
    pass


class CrossAttn(nn.Module):
    def __init__(self, d_model: int = 64, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        h, _ = self.attn(q, k, v, need_weights=False)
        return self.norm(q + h)


__all__ = ["EncY", "EncZ", "EncX", "CrossAttn"]
