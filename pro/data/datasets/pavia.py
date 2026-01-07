"""
Pavia 数据集读取与随机 patch 采样（ratio=8）。

从 .mat 文件加载 I_HS/I_MS/I_REF，返回 NCHW patch 与全图。
"""

from __future__ import annotations

import os
import random
from typing import Dict

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset


class PaviaDataset(Dataset):
    def __init__(
        self,
        mat_path: str,
        patch_size: int = 64,
        patches_per_epoch: int = 200,
        scale: int | None = None,
        seed: int = 2025,
        return_gt: bool = True,
    ):
        super().__init__()
        if not os.path.exists(mat_path):
            raise FileNotFoundError(f"mat file not found: {mat_path}")
        mat = scipy.io.loadmat(mat_path)
        for key in ("I_HS", "I_MS", "I_REF"):
            if key not in mat:
                raise KeyError(f"missing key {key} in {mat_path}")
        ratio_val = mat.get("ratio", None)
        if ratio_val is not None:
            ratio_val = int(np.array(ratio_val).squeeze())
        if scale is None:
            scale = ratio_val if ratio_val is not None else 8
        self.scale = int(scale)

        i_hs = np.array(mat["I_HS"], dtype=np.float32)
        i_ms = np.array(mat["I_MS"], dtype=np.float32)
        i_ref = np.array(mat["I_REF"], dtype=np.float32)

        # HWC -> CHW
        self.y_lr = torch.from_numpy(i_hs).permute(2, 0, 1).contiguous()
        self.z_hr = torch.from_numpy(i_ms).permute(2, 0, 1).contiguous()
        self.x_gt = torch.from_numpy(i_ref).permute(2, 0, 1).contiguous()

        self.bh = int(self.y_lr.shape[0])
        self.bm = int(self.z_hr.shape[0])
        h_lr, w_lr = self.y_lr.shape[1:]
        h_hr, w_hr = self.z_hr.shape[1:]
        if h_hr // h_lr != self.scale or w_hr // w_lr != self.scale:
            raise ValueError(f"scale mismatch: H/W ratio {h_hr//h_lr}x{w_hr//w_lr} vs scale={self.scale}")
        if patch_size % self.scale != 0:
            raise ValueError(f"patch_size must be divisible by scale, got patch_size={patch_size}, scale={self.scale}")
        self.patch_size = int(patch_size)
        self.lr_patch = self.patch_size // self.scale
        if self.lr_patch <= 0:
            raise ValueError("lr_patch must be > 0")
        if h_lr < self.lr_patch or w_lr < self.lr_patch:
            raise ValueError("patch_size too large for current dataset")
        self.patches_per_epoch = int(patches_per_epoch)
        self.rng = random.Random(seed)
        self.return_gt = bool(return_gt)

    def __len__(self) -> int:
        return self.patches_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        h_lr, w_lr = self.y_lr.shape[1:]
        i = self.rng.randint(0, h_lr - self.lr_patch)
        j = self.rng.randint(0, w_lr - self.lr_patch)
        y_patch = self.y_lr[:, i : i + self.lr_patch, j : j + self.lr_patch]
        i_hr = i * self.scale
        j_hr = j * self.scale
        z_patch = self.z_hr[:, i_hr : i_hr + self.patch_size, j_hr : j_hr + self.patch_size]
        x_patch = self.x_gt[:, i_hr : i_hr + self.patch_size, j_hr : j_hr + self.patch_size]
        sample = {
            "Y_lr": y_patch,
            "Z_hr": z_patch,
            "scale": self.scale,
            "sample_id": f"pavia_{idx}",
        }
        if self.return_gt:
            sample["X_gt"] = x_patch
        return sample

    def get_full(self) -> Dict[str, torch.Tensor]:
        sample = {
            "Y_lr": self.y_lr,
            "Z_hr": self.z_hr,
            "scale": self.scale,
            "sample_id": "pavia_full",
        }
        if self.return_gt:
            sample["X_gt"] = self.x_gt
        return sample
