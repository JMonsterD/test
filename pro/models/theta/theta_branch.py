"""
Theta branch (vNext): predicts black-box degradation parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.operators import ay_forward, az_forward, omega_theta


def _pos_enc(height: int, width: int, device, dtype) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=dtype),
        torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xx, yy], dim=0).unsqueeze(0)  # [1,2,H,W]


def _get_field(m_in, name: str):
    if isinstance(m_in, dict):
        return m_in.get(name, None)
    return getattr(m_in, name, None)


class ThetaCore(nn.Module):
    def __init__(
        self,
        bh: int,
        bm: int,
        hidden: int = 128,
        feat_dim: int = 32,
        map_dim: int = 64,
        warp_ctrl_hw: int = 8,
        warp_poly_order: int = 2,
        psf_k: int = 8,
        smile_ctrl_w: int = 16,
    ):
        super().__init__()
        self.bh = bh
        self.bm = bm
        self.feat_dim = feat_dim
        self.map_dim = map_dim
        self.warp_ctrl_hw = warp_ctrl_hw
        self.warp_poly_order = warp_poly_order
        self.psf_k = psf_k
        self.smile_ctrl_w = smile_ctrl_w

        self.enc_m = nn.Linear(32, hidden)
        self.enc_u = nn.Linear(64, hidden)
        self.enc_misc = nn.Linear(3, hidden)
        self.enc_tok = nn.Linear(128, hidden)
        self.enc_map = nn.Linear(map_dim, hidden)

        self.map_stem = nn.Sequential(
            nn.Conv2d(10, map_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(map_dim, map_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.head_warp = nn.Conv2d(map_dim, 2 * (warp_poly_order + 1), kernel_size=1)
        self.head_smile_shift = nn.Conv1d(map_dim, bm, kernel_size=1)
        self.head_smile_temp = nn.Conv1d(map_dim, bm, kernel_size=1)

        self.mlp = nn.Sequential(
            nn.Linear(hidden * 5, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.head_delta = nn.Linear(hidden, 2)
        self.head_psf = nn.Linear(hidden, bh * psf_k)
        self.head_feat = nn.Linear(hidden, feat_dim)

    def forward(self, x_t: torch.Tensor, ry: torch.Tensor, rz: torch.Tensor, m_in: dict, u_theta: torch.Tensor) -> dict:
        """
        Args:
            x_t: [B,Bh,H,W]
            ry: [B,Bh,h,w]
            rz: [B,Bm,H,W]
            m_in: message with stats/map/tokens_in
            u_theta: [B,64]
        Returns:
            delta_theta_raw: dict of warp/psf/smile parameters
        """
        b = x_t.shape[0]
        misc = torch.stack(
            [
                ry.abs().mean(dim=[1, 2, 3]),
                rz.abs().mean(dim=[1, 2, 3]),
                x_t.abs().mean(dim=[1, 2, 3]),
            ],
            dim=1,
        )
        stats = _get_field(m_in, "stats")
        if stats is None:
            stats = torch.zeros(b, 32, device=x_t.device, dtype=x_t.dtype)
        tokens_in = _get_field(m_in, "tokens_in")
        if tokens_in is None:
            tokens_in = torch.zeros(b, 1, 128, device=x_t.device, dtype=x_t.dtype)
        map_in = _get_field(m_in, "map")
        if map_in is None:
            map_in = torch.zeros(b, 8, 64, 64, device=x_t.device, dtype=x_t.dtype)
        pos = _pos_enc(map_in.shape[2], map_in.shape[3], map_in.device, map_in.dtype)
        map_feat = self.map_stem(torch.cat([map_in, pos.expand(b, -1, -1, -1)], dim=1))
        map_vec = self.enc_map(map_feat.mean(dim=(2, 3)))

        feat_m = self.enc_m(stats)
        feat_u = self.enc_u(u_theta)
        feat_misc = self.enc_misc(misc)
        feat_tok = self.enc_tok(tokens_in.mean(dim=1))
        h = torch.cat([feat_m, feat_u, feat_misc, feat_tok, map_vec], dim=1)
        h = self.mlp(h)

        delta_lr = self.head_delta(h)
        psf_coef = self.head_psf(h).view(b, self.bh, self.psf_k)
        theta_feat = self.head_feat(h)

        warp_map = F.adaptive_avg_pool2d(map_feat, (self.warp_ctrl_hw, self.warp_ctrl_hw))
        warp_ctrl = self.head_warp(warp_map)
        warp_ctrl = warp_ctrl.view(b, 2, self.warp_poly_order + 1, self.warp_ctrl_hw, self.warp_ctrl_hw)

        col_feat = map_feat.mean(dim=2)
        col_feat = F.adaptive_avg_pool1d(col_feat, self.smile_ctrl_w)
        smile_shift_col = self.head_smile_shift(col_feat)
        smile_temp_col = self.head_smile_temp(col_feat)

        return {
            "warp_ctrl": warp_ctrl,
            "delta_lr": delta_lr,
            "psf_coef": psf_coef,
            "smile_shift_col": smile_shift_col,
            "smile_temp_col": smile_temp_col,
            "theta_feat": theta_feat,
        }

    def blackbox(
        self,
        y_lr: torch.Tensor,
        z_hr: torch.Tensor,
        *,
        scale: int,
        srf_smile=None,
    ) -> tuple[dict, torch.Tensor]:
        b, bh, h, w = y_lr.shape
        _, bm, H, W = z_hr.shape
        x0 = F.interpolate(y_lr, scale_factor=scale, mode="bilinear", align_corners=False)
        theta0 = {
            "warp_ctrl": torch.zeros(b, 2, self.warp_poly_order + 1, self.warp_ctrl_hw, self.warp_ctrl_hw, device=y_lr.device, dtype=y_lr.dtype),
            "delta_lr": torch.zeros(b, 2, device=y_lr.device, dtype=y_lr.dtype),
            "psf_coef": torch.zeros(b, self.bh, self.psf_k, device=y_lr.device, dtype=y_lr.dtype),
            "smile_shift_col": torch.zeros(b, self.bm, self.smile_ctrl_w, device=y_lr.device, dtype=y_lr.dtype),
            "smile_temp_col": torch.zeros(b, self.bm, self.smile_ctrl_w, device=y_lr.device, dtype=y_lr.dtype),
            "theta_feat": torch.zeros(b, self.feat_dim, device=y_lr.device, dtype=y_lr.dtype),
        }
        theta0 = omega_theta(theta0)
        y_hat0 = ay_forward(x0, theta0, scale=scale)
        if srf_smile is None:
            z_hat0 = torch.zeros_like(z_hr)
        else:
            z_hat0 = az_forward(x0, theta0, scale=scale, srf_smile=srf_smile)
        ry0 = y_lr - y_hat0
        rz0 = z_hr - z_hat0
        ry_mag = ry0.abs().mean(dim=1, keepdim=True)
        rz_mag = rz0.abs().mean(dim=1, keepdim=True)
        ry_mag_up = F.interpolate(ry_mag, size=(H, W), mode="bilinear", align_corners=False)
        c0 = F.adaptive_avg_pool2d(rz_mag, (64, 64))
        c1 = F.adaptive_avg_pool2d(ry_mag_up, (64, 64))
        pad = torch.zeros(b, 6, 64, 64, device=y_lr.device, dtype=y_lr.dtype)
        map_in = torch.cat([c0, c1, pad], dim=1)
        stats = []
        for t in (ry0, rz0):
            t_abs = t.abs()
            stats.append(t_abs.mean(dim=(1, 2, 3)))
            stats.append(t_abs.amax(dim=(1, 2, 3)))
            stats.append(t.mean(dim=(1, 2, 3)))
            stats.append(t.flatten(start_dim=1).std(dim=1, unbiased=False))
        stats = torch.stack(stats, dim=1)
        if stats.shape[1] < 32:
            pad_stats = torch.zeros(b, 32 - stats.shape[1], device=y_lr.device, dtype=y_lr.dtype)
            stats = torch.cat([stats, pad_stats], dim=1)
        tokens_in = torch.zeros(b, 1, 128, device=y_lr.device, dtype=y_lr.dtype)
        m_in = {"stats": stats, "map": map_in, "tokens_in": tokens_in}
        u_theta = torch.zeros(b, 64, device=y_lr.device, dtype=y_lr.dtype)
        theta_raw = self.forward(x0, ry0, rz0, m_in, u_theta)
        theta_hat = omega_theta(theta_raw)
        return theta_hat, theta_hat["theta_feat"]


def OmegaTheta(theta: dict) -> dict:
    return omega_theta(theta)
