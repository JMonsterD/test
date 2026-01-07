"""
SyntheticDataset: generate synthetic degradation data for vNext.
"""

from __future__ import annotations

import random
from typing import Dict

import torch
from torch.utils.data import Dataset

from models.operators import ay_forward, az_forward, omega_theta, SRFSmile


def sample_theta(
    b: int,
    bh: int,
    bm: int,
    device,
    *,
    psf_k: int = 8,
    warp_ctrl_hw: int = 8,
    smile_ctrl_w: int = 16,
    theta_feat_dim: int = 32,
) -> Dict[str, torch.Tensor]:
    warp_ctrl = (torch.rand(b, 2, 3, warp_ctrl_hw, warp_ctrl_hw, device=device) - 0.5) * 0.5
    delta_lr = (torch.rand(b, 2, device=device) - 0.5) * 0.5
    psf_coef = torch.randn(b, bh, psf_k, device=device) * 0.1
    smile_shift_col = (torch.rand(b, bm, smile_ctrl_w, device=device) - 0.5) * 0.5
    smile_temp_col = torch.randn(b, bm, smile_ctrl_w, device=device) * 0.1
    theta_feat = torch.zeros(b, theta_feat_dim, device=device)
    return {
        "warp_ctrl": warp_ctrl,
        "delta_lr": delta_lr,
        "psf_coef": psf_coef,
        "smile_shift_col": smile_shift_col,
        "smile_temp_col": smile_temp_col,
        "theta_feat": theta_feat,
    }


class SyntheticDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        bh: int,
        bm: int,
        h: int,
        w: int,
        scale: int = 4,
        seed: int = 2025,
        enforce_patch: bool = True,
        return_gt: bool = True,
        psf_k: int = 8,
        warp_ctrl_hw: int = 8,
        smile_ctrl_w: int = 16,
        theta_feat_dim: int = 32,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.bh = bh
        self.bm = bm
        if enforce_patch:
            self.h = 32
            self.w = 32
        else:
            self.h = h
            self.w = w
        self.scale = scale
        self.rng = random.Random(seed)
        self.return_gt = bool(return_gt)
        self.psf_k = psf_k
        self.warp_ctrl_hw = warp_ctrl_hw
        self.smile_ctrl_w = smile_ctrl_w
        self.theta_feat_dim = theta_feat_dim
        self.srf_smile = SRFSmile(bm=bm, bh=bh)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        torch.manual_seed(self.rng.randint(0, 10_000_000))
        H = self.h * self.scale
        W = self.w * self.scale
        x_gt = torch.rand(self.bh, H, W)
        theta_gt = sample_theta(
            1,
            self.bh,
            self.bm,
            device=x_gt.device,
            psf_k=self.psf_k,
            warp_ctrl_hw=self.warp_ctrl_hw,
            smile_ctrl_w=self.smile_ctrl_w,
            theta_feat_dim=self.theta_feat_dim,
        )
        x_gt_b = x_gt.unsqueeze(0)
        theta_proj = omega_theta(theta_gt)
        z_hr = az_forward(x_gt_b, theta_proj, scale=self.scale, srf_smile=self.srf_smile)[0]
        y_lr = ay_forward(x_gt_b, theta_proj, self.scale)[0]
        noise_z = torch.randn_like(z_hr) * self.rng.uniform(0, 0.01)
        noise_y = torch.randn_like(y_lr) * self.rng.uniform(0, 0.01)
        z_hr = torch.clamp(z_hr + noise_z, 0.0, 1.0)
        y_lr = torch.clamp(y_lr + noise_y, 0.0, 1.0)
        sample = {
            "Y_lr": y_lr,
            "Z_hr": z_hr,
            "scale": self.scale,
            "sample_id": f"synth_{idx}",
        }
        if self.return_gt:
            sample["X_gt"] = x_gt
            sample["theta_gt"] = {k: v[0] for k, v in theta_proj.items()}
        return sample
