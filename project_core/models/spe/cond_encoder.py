"""
CondMLP：逐像素光谱条件编码（Bm→64），禁止任何 2D 空间算子。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CondMLP(nn.Module):
    def __init__(self, bm: int, hidden: int = 128, out_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(bm, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, z_lr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_lr: [B,Bm,h,w]
        Returns:
            cond: [B,64,h,w]
        """
        b, bm, h, w = z_lr.shape
        z_flat = z_lr.permute(0, 2, 3, 1).reshape(-1, bm)
        h1 = F.silu(self.fc1(z_flat))
        out = self.fc2(h1)
        out = out.view(b, h, w, -1).permute(0, 3, 1, 2)
        return out
