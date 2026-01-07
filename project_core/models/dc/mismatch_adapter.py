"""
Measurement mismatch adapter: small correction heads for y/z domains.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeasMismatchAdapter(nn.Module):
    def __init__(self, bh: int, bm: int, cond_dim: int = 64, hidden: int = 32, clip: float = 0.02):
        super().__init__()
        self.clip = float(clip)
        self.cond_y = nn.Linear(cond_dim, hidden)
        self.cond_z = nn.Linear(cond_dim, hidden)
        in_y = bh * 3
        in_z = bm * 3
        self.y_conv1 = nn.Conv2d(in_y, hidden, kernel_size=3, padding=1)
        self.y_conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.y_conv3 = nn.Conv2d(hidden, bh, kernel_size=3, padding=1)
        self.z_conv1 = nn.Conv2d(in_z, hidden, kernel_size=3, padding=1)
        self.z_conv2 = nn.Conv2d(hidden, hidden, kernel_size=3, padding=1)
        self.z_conv3 = nn.Conv2d(hidden, bm, kernel_size=3, padding=1)

    def _apply_head(
        self,
        y_hat: torch.Tensor,
        y_obs: torch.Tensor,
        cond: torch.Tensor,
        conv1: nn.Conv2d,
        conv2: nn.Conv2d,
        conv3: nn.Conv2d,
        cond_proj: nn.Linear,
    ) -> torch.Tensor:
        res = (y_obs - y_hat).abs()
        feat = torch.cat([y_hat, y_obs, res], dim=1)
        h = conv1(feat)
        cond_bias = cond_proj(cond).view(cond.shape[0], -1, 1, 1)
        h = F.silu(h + cond_bias)
        h = F.silu(conv2(h))
        out = conv3(h)
        if self.clip > 0:
            out = self.clip * torch.tanh(out / self.clip)
        return out

    def apply(
        self,
        y_hat: torch.Tensor,
        z_hat: torch.Tensor,
        y_lr: torch.Tensor,
        z_hr: torch.Tensor,
        theta_feat: torch.Tensor,
        mem_feat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cond = torch.cat([theta_feat, mem_feat], dim=1)
        dy = self._apply_head(y_hat, y_lr, cond, self.y_conv1, self.y_conv2, self.y_conv3, self.cond_y)
        dz = self._apply_head(z_hat, z_hr, cond, self.z_conv1, self.z_conv2, self.z_conv3, self.cond_z)
        return dy, dz


__all__ = ["MeasMismatchAdapter"]
