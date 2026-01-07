"""
轻量 Mamba 封装：序列 Mamba block。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mamba_ssm.modules.mamba2 import Mamba2
from models.utils.mamba_ir_block import MambaIRSeqBlock


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(norm + self.eps)
        return x * self.weight


class MambaSeqBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        mlp_ratio: int = 4,
        use_rmsnorm: bool = True,
    ):
        super().__init__()
        Norm = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.norm1 = Norm(d_model)
        self.mamba = Mamba2(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm2 = Norm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.SiLU(),
            nn.Linear(d_model * mlp_ratio, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self.mamba(h)
        x = x + h
        h = self.norm2(x)
        h = self.mlp(h)
        return x + h


__all__ = ["RMSNorm", "MambaSeqBlock", "MambaIRSeqBlock"]
