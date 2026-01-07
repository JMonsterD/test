"""
MSI Adapter：无参数预处理（anti-alias 下采样与边缘提取）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.utils.resample import anti_alias_downsample


def _sobel_kernels(device, dtype):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=dtype)
    ky = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=device, dtype=dtype)
    return kx.view(1, 1, 3, 3), ky.view(1, 1, 3, 3)


def edge_extract(z: torch.Tensor) -> torch.Tensor:
    """
    对 MSI 求平均后提取梯度幅值。
    """
    z_mean = z.mean(dim=1, keepdim=True)
    kx, ky = _sobel_kernels(z.device, z.dtype)
    gx = F.conv2d(z_mean, kx, padding=1)
    gy = F.conv2d(z_mean, ky, padding=1)
    grad = torch.sqrt(gx.pow(2) + gy.pow(2) + 1e-12)
    return grad


def to_lr(z: torch.Tensor, scale: int) -> torch.Tensor:
    return anti_alias_downsample(z, scale)
