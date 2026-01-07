"""
评测指标实现：PSNR/SSIM/SAM/ERGAS 以及 QNR（Dλ/Ds）。
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def psnr(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> torch.Tensor:
    mse = F.mse_loss(x, y)
    return 10 * torch.log10(max_val * max_val / (mse + 1e-8))


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, device=None, dtype=None):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel = g[:, None] * g[None, :]
    return kernel


def ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5, max_val: float = 1.0) -> torch.Tensor:
    """
    计算 Bh 均值的 SSIM。
    """
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    kernel_2d = _gaussian_kernel(window_size, sigma, device=x.device, dtype=x.dtype)
    kernel = kernel_2d.view(1, 1, window_size, window_size).repeat(x.shape[1], 1, 1, 1)
    pad = window_size // 2
    mu_x = F.conv2d(x, kernel, padding=pad, groups=x.shape[1])
    mu_y = F.conv2d(y, kernel, padding=pad, groups=y.shape[1])
    sigma_x = F.conv2d(x * x, kernel, padding=pad, groups=x.shape[1]) - mu_x ** 2
    sigma_y = F.conv2d(y * y, kernel, padding=pad, groups=y.shape[1]) - mu_y ** 2
    sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=x.shape[1]) - mu_x * mu_y

    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
    return ssim_map.mean()


def sam(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    光谱角（弧度）。
    """
    x_flat = x.flatten(start_dim=2)
    y_flat = y.flatten(start_dim=2)
    dot = (x_flat * y_flat).sum(dim=1)
    norm = torch.norm(x_flat, dim=1) * torch.norm(y_flat, dim=1)
    cos = torch.clamp(dot / (norm + 1e-8), -1.0, 1.0)
    angle = torch.acos(cos)
    return angle.mean()


def ergas(x: torch.Tensor, y: torch.Tensor, scale: int = 4) -> torch.Tensor:
    """
    误差全局相对均方根。
    """
    bh = x.shape[1]
    rmse = torch.sqrt((x - y).pow(2).mean(dim=[2, 3]))
    mean_ref = y.mean(dim=[2, 3])
    ergas_val = 100 / scale * torch.sqrt(((rmse / (mean_ref + 1e-8)) ** 2).mean(dim=1))
    return ergas_val.mean()


def uiqi_window(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    mean_x = x.mean()
    mean_y = y.mean()
    var_x = x.var()
    var_y = y.var()
    cov = ((x - mean_x) * (y - mean_y)).mean()
    return (4 * mean_x * mean_y * cov) / ((mean_x ** 2 + mean_y ** 2) * (var_x + var_y) + 1e-8)


def qnr(y_hat: torch.Tensor, y_lr: torch.Tensor, z_hat: torch.Tensor, z_hr: torch.Tensor, window: int = 32, stride: int = 32) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    按 vFinal 口径计算 QNR、Dλ、Ds。
    """
    b, bh, h, w = y_hat.shape
    _, bm, H, W = z_hat.shape
    def window_mean_uiqi(a, b):
        qs = []
        _, Hh, Ww = a.shape
        h_step = max(1, Hh - window + 1)
        w_step = max(1, Ww - window + 1)
        for i in range(0, h_step, stride):
            for j in range(0, w_step, stride):
                i_end = min(i + window, Hh)
                j_end = min(j + window, Ww)
                qs.append(uiqi_window(a[:, i:i_end, j:j_end], b[:, i:i_end, j:j_end]))
        return torch.stack(qs).mean() if qs else torch.tensor(0.0, device=a.device)

    q_list_lambda = []
    for bi in range(b):
        q_bands = []
        for band in range(bh):
            q_bands.append(window_mean_uiqi(y_hat[bi, band : band + 1], y_lr[bi, band : band + 1]))
        q_list_lambda.append(torch.stack(q_bands).mean())
    d_lambda = 1 - torch.stack(q_list_lambda).mean()

    q_list_s = []
    for bi in range(b):
        q_ms = []
        for band in range(bm):
            q_ms.append(window_mean_uiqi(z_hat[bi, band : band + 1], z_hr[bi, band : band + 1]))
        q_list_s.append(torch.stack(q_ms).mean())
    d_s = 1 - torch.stack(q_list_s).mean()
    qnr_val = (1 - d_lambda) * (1 - d_s)
    return qnr_val, d_lambda, d_s
