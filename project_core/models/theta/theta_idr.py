"""
Theta IDR utilities: view augmentation, VICReg loss, feature queue, EMA helper.
"""

from __future__ import annotations

import copy
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class IDRViewsAugment(nn.Module):
    def __init__(self, noise_std: float = 0.01, brightness: float = 0.02):
        super().__init__()
        self.noise_std = float(noise_std)
        self.brightness = float(brightness)

    def forward(self, y_lr: torch.Tensor, z_hr: torch.Tensor) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        if self.noise_std <= 0 and self.brightness <= 0:
            return (y_lr, z_hr), (y_lr, z_hr)
        noise_y = torch.randn_like(y_lr) * self.noise_std
        noise_z = torch.randn_like(z_hr) * self.noise_std
        if self.brightness > 0:
            b_y = (torch.rand(y_lr.shape[0], 1, 1, 1, device=y_lr.device, dtype=y_lr.dtype) * 2 - 1) * self.brightness
            b_z = (torch.rand(z_hr.shape[0], 1, 1, 1, device=z_hr.device, dtype=z_hr.dtype) * 2 - 1) * self.brightness
        else:
            b_y = torch.zeros(y_lr.shape[0], 1, 1, 1, device=y_lr.device, dtype=y_lr.dtype)
            b_z = torch.zeros(z_hr.shape[0], 1, 1, 1, device=z_hr.device, dtype=z_hr.dtype)
        y2 = (y_lr + noise_y + b_y).clamp(0.0, 1.0)
        z2 = (z_hr + noise_z + b_z).clamp(0.0, 1.0)
        return (y_lr, z_hr), (y2, z2)


class FeatureQueue(nn.Module):
    def __init__(self, q_size: int = 64):
        super().__init__()
        self.q_size = int(q_size)
        self.register_buffer("_queue", torch.zeros(0))
        self.register_buffer("_ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_count", torch.zeros(1, dtype=torch.long))

    def _init_queue(self, feat_dim: int, device, dtype) -> None:
        self._queue = torch.zeros(self.q_size, feat_dim, device=device, dtype=dtype)
        self._ptr = torch.zeros(1, dtype=torch.long, device=device)
        self._count = torch.zeros(1, dtype=torch.long, device=device)

    @torch.no_grad()
    def update(self, feats: torch.Tensor) -> None:
        if feats is None or feats.numel() == 0:
            return
        feats = feats.detach()
        if self._queue.numel() == 0 or self._queue.shape[1] != feats.shape[1] or self._queue.device != feats.device:
            self._init_queue(feats.shape[1], feats.device, feats.dtype)
        b = feats.shape[0]
        for i in range(b):
            idx = int(self._ptr.item())
            self._queue[idx] = feats[i]
            self._ptr[0] = (self._ptr[0] + 1) % self.q_size
            self._count[0] = min(self._count[0] + 1, self.q_size)

    def get(self) -> torch.Tensor | None:
        if self._queue.numel() == 0 or int(self._count.item()) <= 0:
            return None
        return self._queue[: int(self._count.item())]


class VICRegLoss(nn.Module):
    def __init__(self, var_min: float = 1.0, cov_coeff: float = 0.1):
        super().__init__()
        self.var_min = float(var_min)
        self.cov_coeff = float(cov_coeff)

    def forward(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor,
        feature_queue: FeatureQueue | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        L_inv = (feat1 - feat2).pow(2).mean()
        stats_feat = feat1
        if feature_queue is not None:
            queued = feature_queue.get()
            if queued is not None and queued.numel() > 0:
                queued = queued.to(device=feat1.device, dtype=feat1.dtype)
                stats_feat = torch.cat([stats_feat, queued], dim=0)
        b_eff = stats_feat.shape[0]
        if b_eff < 2:
            L_var = feat1.new_tensor(0.0)
            L_cov = feat1.new_tensor(0.0)
        else:
            std = stats_feat.std(dim=0, unbiased=False)
            L_var = torch.relu(self.var_min - std).mean()
            feat_centered = stats_feat - stats_feat.mean(dim=0, keepdim=True)
            cov = (feat_centered.T @ feat_centered) / float(b_eff - 1)
            off_diag = cov - torch.diag(torch.diag(cov))
            L_cov = (off_diag.pow(2).sum()) / float(cov.numel())
        L_idr = L_inv + L_var + self.cov_coeff * L_cov
        return L_idr, {"L_inv": L_inv, "L_var": L_var, "L_cov": L_cov}


class ThetaEMA(nn.Module):
    def __init__(self, model: nn.Module, ema: float = 0.999):
        super().__init__()
        self.ema = float(ema)
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for p_ema, p in zip(self.model.parameters(), model.parameters()):
            p_ema.mul_(self.ema).add_(p, alpha=1.0 - self.ema)
        for b_ema, b in zip(self.model.buffers(), model.buffers()):
            b_ema.copy_(b)

    def blackbox(self, *args, **kwargs):
        return self.model.blackbox(*args, **kwargs)


__all__ = ["IDRViewsAugment", "FeatureQueue", "VICRegLoss", "ThetaEMA"]
