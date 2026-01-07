"""
Residual-guided alignment block.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualGuidedAlign(nn.Module):
    def __init__(self, bh: int, bm: int, cond_dim: int = 64, hidden: int = 32, clip: float = 0.5):
        super().__init__()
        self.clip = float(clip)
        in_ch = bh + bm + 1
        self.cond = nn.Linear(cond_dim, hidden)
        self.conv1 = nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden, bh, kernel_size=3, padding=1)

    def forward(
        self,
        ry_hr: torch.Tensor,
        rz_hr: torch.Tensor,
        edge_z: torch.Tensor,
        theta_feat: torch.Tensor,
        mem_feat: torch.Tensor,
    ) -> torch.Tensor:
        cond = torch.cat([theta_feat, mem_feat], dim=1)
        feat = torch.cat([ry_hr, rz_hr, edge_z], dim=1)
        h = self.conv1(feat)
        h = h + self.cond(cond).view(cond.shape[0], -1, 1, 1)
        h = F.silu(h)
        h = F.silu(self.conv2(h))
        out = self.conv3(h)
        if self.clip > 0:
            out = self.clip * torch.tanh(out / self.clip)
        return out


__all__ = ["ResidualGuidedAlign"]
