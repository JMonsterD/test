"""
SPA 分支核心：接收 LR-HSI 与 HR-MSI，输出 LR 网格空间特征 F_spa。

按 Light-Mamba 规范实现：Local Stem + Grid-Mamba + 融合。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .msi_adapter import edge_extract
from .spa_blocks_lightmamba import SpaLocalStem, SpaGridMamba, SpaFuseHead
from models.utils.freq import proj_P


class SpaCore(nn.Module):
    def __init__(
        self,
        bh: int,
        bm: int,
        c_spa: int = 64,
        scale: int = 4,
        spa_local_blocks: int = 1,
        spa_grid_down: int = 1,
        spa_grid_d: int = 64,
        spa_mamba_layers: int = 1,
        spa_mamba_d_state: int = 16,
        spa_mamba_d_conv: int = 3,
        spa_grid_L_max: int = 64,
        spa_grid_L_min: int = 4,
        spa_grid_min_hw: int = 2,
        spa_fuse_alpha_init: float = 0.1,
        spa_film: bool = True,
    ):
        super().__init__()
        self.scale = scale
        self.spa_film = spa_film
        self.conv_y = nn.Conv2d(bh, 32, kernel_size=1)
        self.conv_z_pu = nn.Conv2d(bm * scale * scale, 64, kernel_size=1)
        self.conv_edge_pu = nn.Conv2d(scale * scale, 16, kernel_size=1)
        self.conv_ry_pu = nn.Conv2d(scale * scale, 16, kernel_size=1)

        cat_channels = 32 + 64 + 16 + 16
        self.stem = SpaLocalStem(cat_channels, c_spa, blocks=spa_local_blocks)
        self.film = nn.Linear(64, 2 * c_spa)
        self.grid = SpaGridMamba(
            c_spa,
            grid_down=spa_grid_down,
            grid_d=spa_grid_d,
            mamba_layers=spa_mamba_layers,
            d_state=spa_mamba_d_state,
            d_conv=spa_mamba_d_conv,
            grid_L_max=spa_grid_L_max,
            grid_L_min=spa_grid_L_min,
            grid_min_hw=spa_grid_min_hw,
        )
        self.fuse = SpaFuseHead(c_spa, alpha_init=spa_fuse_alpha_init, post=False)

    def forward(self, y: torch.Tensor, z: torch.Tensor, ry: torch.Tensor, rz: torch.Tensor, u_spa: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: [B,Bh,h,w]
            z: [B,Bm,H,W]
            ry: [B,Bh,h,w]
            rz: [B,Bm,H,W]
            u_spa: [B,64]
        Returns:
            f_spa: [B,C_spa,h,w]
        """
        b, _, h, w = y.shape
        _, _, H, W = z.shape
        s = H // h

        if s != self.scale:
            raise ValueError(f"spa scale mismatch: expected {self.scale}, got {s}")

        z_pu = F.pixel_unshuffle(z, s)
        e_hr = edge_extract(proj_P(z))
        e_pu = F.pixel_unshuffle(e_hr, s)
        ry_up = F.interpolate(ry, size=(H, W), mode="bilinear", align_corners=False)
        ry_pu = F.pixel_unshuffle(proj_P(ry_up), s)

        cat = torch.cat(
            [
                self.conv_y(y),
                self.conv_z_pu(z_pu),
                self.conv_edge_pu(e_pu),
                self.conv_ry_pu(ry_pu),
            ],
            dim=1,
        )
        x = self.stem(cat)
        if self.spa_film:
            gamma_beta = self.film(u_spa).view(b, 2, -1, 1, 1)
            gamma = gamma_beta[:, 0]
            beta = gamma_beta[:, 1]
            x = (1 + gamma) * x + beta
        x_up = self.grid(x)
        self.last_grid_hw = getattr(self.grid, "last_grid_hw", None)
        self.last_grid_down = getattr(self.grid, "last_grid_down", None)
        self.last_grid_L = getattr(self.grid, "last_grid_L", None)
        x = self.fuse(x, x_up)
        return x
