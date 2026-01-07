"""
SPE Light-Mamba blocks：谱压缩、token 嵌入、Mamba 栈、读出。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.mamba_wrappers import MambaSeqBlock


class SpectralCompressor(nn.Module):
    def __init__(self, k: int = 32, mode: str = "segment_mean", bh: int | None = None):
        super().__init__()
        self.k = k
        self.mode = mode
        self.bh = bh
        self.proj = None
        if mode == "learned_proj":
            if bh is None:
                raise ValueError("SpectralCompressor(mode=learned_proj) requires bh")
            self.proj = nn.Linear(bh, k)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: [BN, Bh]
        Returns:
            q: [BN, K]
        """
        bn, bh = y.shape
        if self.mode not in ("segment_mean", "learned_proj", "none"):
            raise ValueError(f"SpectralCompressor mode must be segment_mean/learned_proj/none, got {self.mode}")
        if self.mode == "none":
            return y
        k = min(self.k, bh) if self.k > 0 else bh
        if k <= 0:
            return y
        if self.mode == "segment_mean":
            if bh % k == 0:
                return y.view(bn, k, bh // k).mean(dim=-1)
            pad = (k - (bh % k)) % k
            if pad > 0:
                y = F.pad(y, (0, pad), mode="constant", value=0.0)
                bh = y.shape[1]
            return y.view(bn, k, bh // k).mean(dim=-1)
        if self.mode == "learned_proj":
            if self.proj is None:
                raise RuntimeError("learned_proj requires initialized proj")
            return self.proj(y)
        return y


class SpeTokenEmbed(nn.Module):
    def __init__(self, k: int, d_s: int):
        super().__init__()
        self.k = k
        self.proj = nn.Linear(1, d_s)
        self.pos_emb = nn.Parameter(torch.zeros(1, k, d_s))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """
        q: [BN, K]
        """
        tokens = self.proj(q.unsqueeze(-1))
        return tokens + self.pos_emb


class SpeMambaStack(nn.Module):
    def __init__(self, d_s: int, layers: int = 2, d_state: int = 16, d_conv: int = 3, expand: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList(
            [MambaSeqBlock(d_s, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class SpeReadout(nn.Module):
    def __init__(self, d_s: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.score = nn.Linear(d_s, heads)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        scores = self.score(tokens)
        w = torch.softmax(scores, dim=1)
        return (w.unsqueeze(-1) * tokens.unsqueeze(2)).sum(dim=1)
