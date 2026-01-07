"""
SPA Light-Mamba blocks：LocalStem / GridMamba / FuseHead。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.resample import anti_alias_downsample, upsample_to
from models.utils.mamba_wrappers import MambaSeqBlock


class SpaLocalStem(nn.Module):
    def __init__(self, in_ch: int, c_spa: int, blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList()
        ch = in_ch
        for _ in range(max(1, blocks)):
            self.blocks.append(nn.Sequential(nn.Conv2d(ch, c_spa, kernel_size=3, padding=1), nn.SiLU()))
            ch = c_spa

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx, blk in enumerate(self.blocks):
            out = blk(x)
            x = out if idx == 0 else x + out
        return x


class SpaGridMamba(nn.Module):
    def __init__(
        self,
        c_spa: int,
        grid_down: int = 1,
        grid_d: int = 64,
        mamba_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        grid_L_max: int = 64,
        grid_L_min: int = 4,
        grid_min_hw: int = 2,
    ):
        super().__init__()
        self.grid_down = max(1, grid_down)
        self.grid_L_max = grid_L_max
        self.grid_L_min = grid_L_min
        self.grid_min_hw = grid_min_hw
        self.proj_in = nn.Conv2d(c_spa, grid_d, kernel_size=1)
        self.proj_out = nn.Conv2d(grid_d, c_spa, kernel_size=1)
        self.blocks = nn.ModuleList(
            [MambaSeqBlock(grid_d, d_state=d_state, d_conv=d_conv, expand=expand) for _ in range(mamba_layers)]
        )

    def _choose_gd(self, h: int, w: int) -> int:
        safe_gd = min(self.grid_down, max(1, min(h, w) // self.grid_min_hw))
        if self.grid_L_max <= 0:
            return safe_gd
        if h * w <= self.grid_L_max:
            return 1
        max_gd = max(1, safe_gd)
        for gd in range(1, max_gd + 1):
            hg = (h + gd - 1) // gd
            wg = (w + gd - 1) // gd
            if hg * wg <= self.grid_L_max and hg >= self.grid_min_hw and wg >= self.grid_min_hw:
                return gd
        return 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        gd = self._choose_gd(h, w)
        hg = (h + gd - 1) // gd
        wg = (w + gd - 1) // gd
        L = hg * wg
        if self.grid_L_min > 0 and L < self.grid_L_min:
            gd = 1
            hg, wg = h, w
            L = h * w
        x_g = anti_alias_downsample(x, gd)
        x_g = self.proj_in(x_g)
        hg, wg = x_g.shape[-2:]
        self.last_grid_hw = (int(hg), int(wg))
        self.last_grid_down = int(gd)
        self.last_grid_L = int(hg * wg)
        seq = x_g.permute(0, 2, 3, 1).reshape(b, hg * wg, -1)
        for blk in self.blocks:
            seq = blk(seq)
        x_g2 = seq.view(b, hg, wg, -1).permute(0, 3, 1, 2).contiguous()
        x_g2 = self.proj_out(x_g2)
        x_up = upsample_to(x_g2, (h, w), mode="bilinear")
        return x_up


class SpaFuseHead(nn.Module):
    def __init__(self, c_spa: int, alpha_init: float = 0.1, post: bool = False):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.post = post
        if post:
            self.dw = nn.Conv2d(c_spa, c_spa, kernel_size=3, padding=1, groups=c_spa)
            self.pw = nn.Conv2d(c_spa, c_spa, kernel_size=1)

    def forward(self, x: torch.Tensor, x_up: torch.Tensor) -> torch.Tensor:
        out = x + self.alpha * x_up
        if self.post:
            out = self.pw(F.silu(self.dw(out)))
        return out
