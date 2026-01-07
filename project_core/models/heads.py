"""
Heads：Spa/Spe/DC 三个更新头。

- HeadSpa：Conv1x1 -> PixelShuffle 上采样
- HeadSpe：Conv1x1 -> 双线性上采样
- HeadDC ：固定算子回注（Az^T + Ay^T），无可学习参数
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.operators import az_transpose, apply_psf_blur_transpose, build_psf_kernels, warp_inverse_apply


class HeadSpa(nn.Module):
    def __init__(self, c_in: int, bh: int, scale: int = 4, zero_init: bool = False, clip_scale: float = 0.0):
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv2d(c_in, bh * scale * scale, kernel_size=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        if zero_init:
            nn.init.zeros_(self.conv.weight)
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

    def forward(self, f_spa: torch.Tensor, u_spa: torch.Tensor, clip_scale: float | None = None) -> torch.Tensor:
        x = self.conv(f_spa)
        x = self.pixel_shuffle(x)
        return x


class HeadSpe(nn.Module):
    def __init__(self, c_in: int, bh: int, scale: int = 4, zero_init: bool = False, clip_scale: float = 0.0):
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv2d(c_in, bh, kernel_size=1)
        if zero_init:
            nn.init.zeros_(self.conv.weight)
            if self.conv.bias is not None:
                nn.init.zeros_(self.conv.bias)

    def forward(self, f_spe: torch.Tensor, u_spe: torch.Tensor, clip_scale: float | None = None) -> torch.Tensor:
        x = self.conv(f_spe)
        x = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        return x


class HeadDC(nn.Module):
    def __init__(self, scale: int = 4, w_y: float = 1.0, w_z: float = 1.0, srf_smile=None, clip_value: float = 2.0):
        super().__init__()
        self.scale = scale
        self.w_y = float(w_y)
        self.w_z = float(w_z)
        self.srf_smile = srf_smile
        self.clip_value = float(clip_value)

    def forward(
        self,
        ry: torch.Tensor,
        rz: torch.Tensor,
        theta: dict,
        *,
        w_y: float | None = None,
        w_z: float | None = None,
        srf_smile=None,
    ) -> torch.Tensor:
        w_y_eff = self.w_y if w_y is None else float(w_y)
        w_z_eff = self.w_z if w_z is None else float(w_z)
        srf_smile_eff = self.srf_smile if srf_smile is None else srf_smile
        ry_up = F.interpolate(ry, scale_factor=self.scale, mode="nearest") / (self.scale * self.scale)
        k = build_psf_kernels(theta)
        ry_bp = warp_inverse_apply(apply_psf_blur_transpose(ry_up, k=k), theta, self.scale)
        rz_bp = warp_inverse_apply(apply_psf_blur_transpose(az_transpose(rz, theta, srf_smile_eff), k=k), theta, self.scale)
        d_x_dc = w_y_eff * ry_bp + w_z_eff * rz_bp
        if self.clip_value > 0:
            d_x_dc = torch.clamp(d_x_dc, -self.clip_value, self.clip_value)
        return d_x_dc
