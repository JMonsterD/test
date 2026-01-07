"""
MambaIR-style sequence block: local enhance + Mamba2 + channel attention + LayerScale.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from mamba_ssm.modules.mamba2 import Mamba2


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


class ChannelAttention(nn.Module):
    def __init__(self, d_model: int, reduction: int = 4):
        super().__init__()
        hidden = max(1, d_model // reduction)
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)
        gate = self.fc2(torch.relu(self.fc1(pooled))).sigmoid()
        return x * gate.view(gate.shape[0], 1, gate.shape[1])


class MambaIRSeqBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        mlp_ratio: int = 4,
        grid_hw: int | None = None,
    ):
        super().__init__()
        self.grid_hw = grid_hw
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.mamba = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.SiLU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )
        self.local_dw = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.ca = ChannelAttention(d_model)
        self.gamma = nn.Parameter(1e-3 * torch.ones(d_model))

    def _local_enhance(self, x: torch.Tensor) -> torch.Tensor:
        b, l, c = x.shape
        if self.grid_hw is None:
            hw = int(math.sqrt(l))
        else:
            hw = self.grid_hw
        if hw * hw != l:
            return x
        grid = x.view(b, hw, hw, c).permute(0, 3, 1, 2)
        grid = self.local_dw(grid)
        return grid.permute(0, 2, 3, 1).reshape(b, l, c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self._local_enhance(x)
        x = x + local * self.gamma.view(1, 1, -1)
        h = self.norm1(x)
        h = self.mamba(h)
        x = x + h
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        x = self.ca(x)
        return x


__all__ = ["MambaIRSeqBlock"]
