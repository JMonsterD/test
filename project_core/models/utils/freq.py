"""
Frequency helpers: P/LP/HP/TV utilities for the model.
"""

from __future__ import annotations

import torch

from models.utils.resample import anti_alias_downsample, upsample_to


def proj_P(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Channel-mean projection with per-sample normalization.
    """
    p = x.mean(dim=1, keepdim=True)
    mean = p.mean(dim=(2, 3), keepdim=True)
    std = p.std(dim=(2, 3), keepdim=True, unbiased=False)
    return (p - mean) / (std + eps)


def lowpass(x: torch.Tensor, scale: int) -> torch.Tensor:
    """
    Low-pass: anti-aliased downsample and upsample back.
    """
    if scale <= 1:
        return x
    h, w = x.shape[-2:]
    x_lr = anti_alias_downsample(x, scale)
    return upsample_to(x_lr, (h, w), mode="bilinear")


def highpass(x: torch.Tensor, scale: int) -> torch.Tensor:
    """
    High-pass: X - LP(X).
    """
    return x - lowpass(x, scale)


def tv_spatial(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] < 2 or x.shape[-2] < 2:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    tv_x = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    tv_y = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    return tv_x + tv_y


def tv_band(x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] < 2:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    return (x[:, 1:, :, :] - x[:, :-1, :, :]).abs().mean()
