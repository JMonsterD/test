"""
Fusion core with Memory-Mamba and NSS scanning.
"""

from __future__ import annotations

from typing import Dict, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.mamba_wrappers import MambaIRSeqBlock


_POS_CACHE: dict[tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}
_NSS_CACHE: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


def _pos_enc(h: int, w: int, device, dtype) -> torch.Tensor:
    key = (h, w, device, dtype)
    cached = _POS_CACHE.get(key)
    if cached is not None:
        return cached
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype),
        indexing="ij",
    )
    pos = torch.stack([xx, yy], dim=-1).view(1, h * w, 2)
    _POS_CACHE[key] = pos
    return pos


def nss_ids(
    h: int,
    w: int,
    scan_len: int = 4,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (h, w, scan_len)
    cached = _NSS_CACHE.get(key)
    if cached is not None and device is None:
        return cached

    idx = []
    if scan_len <= 0:
        scan_len = w
    num_stripes = math.ceil(w / scan_len)
    for s in range(num_stripes):
        col_start = s * scan_len
        col_end = min((s + 1) * scan_len, w)
        cols = list(range(col_start, col_end))
        if s % 2 == 1:
            cols = cols[::-1]
        for r in range(h):
            row_cols = cols if (r % 2 == 0) else cols[::-1]
            for c in row_cols:
                idx.append(r * w + c)
    perm = torch.tensor(idx, dtype=torch.long)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel())
    if device is not None:
        perm = perm.to(device)
        inv = inv.to(device)
    if device is None:
        _NSS_CACHE[key] = (perm, inv)
    return perm, inv


class DEQMemory(nn.Module):
    def __init__(self, mem_dim: int = 32, map_channels: int = 2):
        super().__init__()
        self.mem_dim = mem_dim
        self.map_channels = map_channels
        self.mlp = nn.Sequential(
            nn.Linear(4, mem_dim),
            nn.SiLU(),
            nn.Linear(mem_dim, mem_dim),
        )
        if map_channels > 0:
            self.map_proj = nn.Conv2d(1, map_channels, kernel_size=1)
        else:
            self.map_proj = None

    def forward(self, deq_stats: Dict) -> Tuple[torch.Tensor, torch.Tensor | None]:
        if not isinstance(deq_stats, dict):
            raise ValueError("deq_stats must be a dict")
        b = int(deq_stats.get("batch", 1) or 1)
        device = deq_stats.get("device", None)
        dtype = deq_stats.get("dtype", None)

        def _as_tensor(val, default=0.0):
            if torch.is_tensor(val):
                return val
            return torch.tensor(val if val is not None else default)

        k_star = _as_tensor(deq_stats.get("k_star", 0.0))
        r_tail = deq_stats.get("r_tail", 0.0)
        if isinstance(r_tail, (list, tuple)) and r_tail:
            r_tail = r_tail[-1]
        r_tail = _as_tensor(r_tail)
        dx_last = _as_tensor(deq_stats.get("dx_last", 0.0))
        fail_reason = str(deq_stats.get("fail_reason", "ok"))
        fail_flag = 1.0 if fail_reason != "ok" else 0.0
        fail_flag = _as_tensor(fail_flag)

        stats = torch.stack([k_star, r_tail, dx_last, fail_flag], dim=0).view(1, -1)
        if device is not None:
            stats = stats.to(device=device)
        if dtype is not None:
            stats = stats.to(dtype=dtype)
        if b > 1:
            stats = stats.expand(b, -1)
        mem_vec = self.mlp(stats)

        mem_map = None
        res_map = deq_stats.get("res_map", None)
        if self.map_proj is not None and res_map is not None:
            if res_map.dim() == 3:
                res_map = res_map.unsqueeze(1)
            mem_map = self.map_proj(res_map)
        return mem_vec, mem_map


class FusionCoreNSSMamba(nn.Module):
    def __init__(
        self,
        bh: int,
        d_model: int = 64,
        mamba_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 3,
        expand: int = 2,
        theta_dim: int = 32,
        mem_dim: int = 32,
        grid_hw: int = 8,
        scan_len: int = 4,
        map_channels: int = 2,
        out_clip: float = 0.1,
    ):
        super().__init__()
        self.bh = bh
        self.d_model = d_model
        self.theta_dim = theta_dim
        self.mem_dim = mem_dim
        self.grid_hw = grid_hw
        self.scan_len = scan_len
        self.map_channels = map_channels
        self.out_clip = out_clip
        self.theta_proj = nn.LazyLinear(theta_dim) if theta_dim > 0 else None
        self.mem_proj = nn.LazyLinear(mem_dim) if mem_dim > 0 else None
        input_dim = 4 + map_channels + 3 * d_model + 2 + theta_dim + mem_dim
        self.token_proj = nn.Linear(input_dim, d_model)
        self.mamba = nn.ModuleList(
            [MambaIRSeqBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand, grid_hw=grid_hw) for _ in range(mamba_layers)]
        )
        self.out_proj = nn.Conv2d(d_model, bh, kernel_size=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        dX_spa: torch.Tensor,
        dX_spe: torch.Tensor,
        theta_feat: torch.Tensor | None,
        mem_feat: torch.Tensor | None,
        fx_hr: torch.Tensor,
        fy_lr: torch.Tensor,
        fz_hr: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        b, _, h, w = dX_spa.shape
        hg = self.grid_hw
        wg = self.grid_hw
        spa_mean = F.adaptive_avg_pool2d(dX_spa.mean(dim=1, keepdim=True), (hg, wg))
        spa_abs = F.adaptive_avg_pool2d(dX_spa.abs().mean(dim=1, keepdim=True), (hg, wg))
        spe_mean = F.adaptive_avg_pool2d(dX_spe.mean(dim=1, keepdim=True), (hg, wg))
        spe_abs = F.adaptive_avg_pool2d(dX_spe.abs().mean(dim=1, keepdim=True), (hg, wg))
        if self.map_channels > 0:
            deq_pool = torch.zeros(b, self.map_channels, hg, wg, device=dX_spa.device, dtype=dX_spa.dtype)
        else:
            deq_pool = torch.zeros(b, 0, hg, wg, device=dX_spa.device, dtype=dX_spa.dtype)

        fx_pool = F.adaptive_avg_pool2d(fx_hr, (hg, wg))
        fy_hr = F.interpolate(fy_lr, size=(h, w), mode="bilinear", align_corners=False)
        fy_pool = F.adaptive_avg_pool2d(fy_hr, (hg, wg))
        fz_pool = F.adaptive_avg_pool2d(fz_hr, (hg, wg))
        feat = torch.cat([spa_mean, spa_abs, spe_mean, spe_abs, deq_pool, fx_pool, fy_pool, fz_pool], dim=1)
        feat = feat.permute(0, 2, 3, 1).reshape(b, hg * wg, -1)

        pos = _pos_enc(hg, wg, dX_spa.device, dX_spa.dtype)
        pos = pos.expand(b, -1, -1)

        if theta_feat is None:
            theta_feat = torch.zeros(b, self.theta_dim, device=dX_spa.device, dtype=dX_spa.dtype)
        elif self.theta_dim > 0 and theta_feat.shape[1] != self.theta_dim and self.theta_proj is not None:
            theta_feat = self.theta_proj(theta_feat)
        if mem_feat is None:
            mem_feat = torch.zeros(b, self.mem_dim, device=dX_spa.device, dtype=dX_spa.dtype)
        elif self.mem_dim > 0 and mem_feat.shape[1] != self.mem_dim and self.mem_proj is not None:
            mem_feat = self.mem_proj(mem_feat)

        theta_feat = theta_feat.view(b, 1, -1).expand(b, hg * wg, -1)
        mem_feat = mem_feat.view(b, 1, -1).expand(b, hg * wg, -1)

        tokens = torch.cat([feat, pos, theta_feat, mem_feat], dim=-1)
        tokens = self.token_proj(tokens)

        perm, inv = nss_ids(hg, wg, self.scan_len, device=dX_spa.device)
        tokens_scan = tokens[:, perm, :]
        for blk in self.mamba:
            tokens_scan = blk(tokens_scan)
        tokens_out = tokens_scan[:, inv, :]

        grid = tokens_out.view(b, hg, wg, self.d_model).permute(0, 3, 1, 2)
        out = self.out_proj(grid)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        if self.out_clip and self.out_clip > 0:
            out = self.out_clip * torch.tanh(out / self.out_clip)
        return out, {}


class FusionCore(FusionCoreNSSMamba):
    pass


__all__ = ["FusionCore", "FusionCoreNSSMamba", "DEQMemory", "nss_ids"]
