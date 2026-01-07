"""
RealDataset：真实数据占位实现，遵循统一返回 schema。
"""

from __future__ import annotations

from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset


class RealDataset(Dataset):
    def __init__(self, samples: Optional[List[Dict]] = None, bh: int = 0, bm: int = 0, h: int = 0, w: int = 0, scale: int = 4):
        super().__init__()
        self.samples = samples
        self.bh = bh
        self.bm = bm
        self.h = h
        self.w = w
        self.scale = scale

    def __len__(self):
        if self.samples is not None:
            return len(self.samples)
        return 1

    def __getitem__(self, idx):
        if self.samples is not None:
            sample = self.samples[idx]
            sample.setdefault("scale", self.scale)
            sample.setdefault("sample_id", f"real_{idx}")
            return sample
        # 默认占位返回全零张量
        H = self.h * self.scale
        W = self.w * self.scale
        y_lr = torch.zeros(self.bh, self.h, self.w)
        z_hr = torch.zeros(self.bm, H, W)
        return {
            "Y_lr": y_lr,
            "Z_hr": z_hr,
            "scale": self.scale,
            "sample_id": f"real_{idx}",
        }
