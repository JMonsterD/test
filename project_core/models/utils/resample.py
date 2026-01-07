"""
固定重采样工具：抗混叠下采样与上采样。
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def anti_alias_downsample(x: torch.Tensor, factor: int, mode: str = "mean_reshape") -> torch.Tensor:
    if factor <= 1:
        return x
    b, c, h, w = x.shape
    if mode == "mean_reshape" and h % factor == 0 and w % factor == 0:
        h2 = h // factor
        w2 = w // factor
        x_view = x.reshape(b, c, h2, factor, w2, factor)
        return x_view.mean(dim=(3, 5))
    h2 = max(1, (h + factor - 1) // factor)
    w2 = max(1, (w + factor - 1) // factor)
    return F.interpolate(x, size=(h2, w2), mode="area")


def upsample_to(x: torch.Tensor, size_hw: Tuple[int, int], mode: str = "bilinear") -> torch.Tensor:
    if x.shape[-2:] == size_hw:
        return x
    align = None
    if mode in ("bilinear", "bicubic"):
        align = False
    return F.interpolate(x, size=size_hw, mode=mode, align_corners=align)
