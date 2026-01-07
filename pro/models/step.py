"""
Step operator: X-only DEQ mapping.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from models.operators import ay_forward, az_forward
from models.spa.msi_adapter import edge_extract


def init_state(y_lr: torch.Tensor, scale: int) -> torch.Tensor:
    x0 = F.interpolate(y_lr, scale_factor=scale, mode="bilinear", align_corners=False)
    return x0


def _safe_ratio(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    return num / (den + 1e-6)


def _pool_tokens(x: torch.Tensor, g: int) -> torch.Tensor:
    pooled = F.adaptive_avg_pool2d(x, (g, g))
    return pooled.flatten(2).transpose(1, 2)


def _unpool_tokens(tokens: torch.Tensor, g: int, size_hw: Tuple[int, int]) -> torch.Tensor:
    b, l, c = tokens.shape
    grid = tokens.view(b, g, g, c).permute(0, 3, 1, 2)
    return F.interpolate(grid, size=size_hw, mode="bilinear", align_corners=False)


def step_fn(x_t: torch.Tensor, inputs: Dict[str, torch.Tensor], modules: Dict, params: Dict) -> Tuple[torch.Tensor, Dict]:
    y_lr = inputs["Y_lr"]
    z_hr = inputs["Z_hr"]
    theta_hat = inputs["theta_hat"]
    theta_feat = inputs.get("theta_feat", None)
    scale = int(params.get("scale", 4))
    srf_smile = modules.get("srf_smile", None)

    # (1) forward projection (raw residuals for Context)
    y_hat0 = ay_forward(x_t, theta_hat, scale)
    z_hat0 = az_forward(x_t, theta_hat, scale, srf_smile)
    ry0 = y_lr - y_hat0
    rz0 = z_hr - z_hat0

    # (2) context
    ctx = modules["context"](ry0, rz0, theta_feat)
    u_spa = ctx.u_spa
    u_spe = ctx.u_spe
    alpha_x = ctx.alpha_x
    beta_dc = ctx.beta_dc
    gamma_rg = ctx.gamma_rg
    w_spa = ctx.w_spa
    w_spe = ctx.w_spe
    mem_feat = ctx.mem_feat

    # (3) mismatch adapter
    dy, dz = modules["mismatch_adapter"].apply(y_hat0, z_hat0, y_lr, z_hr, theta_feat, mem_feat)
    y_hat = y_hat0 + dy
    z_hat = z_hat0 + dz
    ry = y_lr - y_hat
    rz = z_hr - z_hat

    # (4) branches + heads
    f_spa = modules["spa"](y_lr, z_hr, ry, rz, u_spa)
    f_spe = modules["spe"](y_lr, z_hr, ry, rz, u_spe)
    dX_spa = modules["head_spa"](f_spa, u_spa)
    dX_spe = modules["head_spe"](f_spe, u_spe)

    # (5) cross-modal features
    fx_hr = modules["enc_x"](x_t)
    fy_lr = modules["enc_y"](y_lr)
    fz_hr = modules["enc_z"](z_hr)
    g = int(params.get("cross_grid", 16))
    fy_hr = F.interpolate(fy_lr, size=fx_hr.shape[-2:], mode="bilinear", align_corners=False)
    tx = _pool_tokens(fx_hr, g)
    ty = _pool_tokens(fy_hr, g)
    tz = _pool_tokens(fz_hr, g)
    kv = torch.cat([ty, tz], dim=1)
    txa = modules["cross_attn"](tx, kv, kv)
    fx_hr = fx_hr + _unpool_tokens(txa, g, fx_hr.shape[-2:])

    # (6) fusion + RGAlign
    w_spa_view = w_spa.view(-1, 1, 1, 1)
    w_spe_view = w_spe.view(-1, 1, 1, 1)
    dX_spa_w = dX_spa * w_spa_view
    dX_spe_w = dX_spe * w_spe_view
    dX_fuse, _ = modules["fusion"](dX_spa_w, dX_spe_w, theta_feat, mem_feat, fx_hr, fy_lr, fz_hr)

    ry_hr = F.interpolate(ry.abs(), scale_factor=scale, mode="bilinear", align_corners=False)
    edge_z = edge_extract(z_hr)
    dX_rg = modules["rg_align"](ry_hr, rz.abs(), edge_z, theta_feat, mem_feat)
    dX_prior = dX_fuse + gamma_rg.view(-1, 1, 1, 1) * dX_rg

    # (7) DC update
    dX_dc = modules["head_dc"](ry, rz, theta_hat, w_y=1.0, w_z=1.0, srf_smile=srf_smile)

    # (8) update
    x_tp = x_t + alpha_x.view(-1, 1, 1, 1) * (dX_prior + beta_dc.view(-1, 1, 1, 1) * dX_dc)

    aux = {
        "alpha_x_mean": float(alpha_x.mean().item()),
        "alpha_x_max": float(alpha_x.max().item()),
        "beta_dc_mean": float(beta_dc.mean().item()),
        "beta_dc_max": float(beta_dc.max().item()),
        "w_spa_mean": float(w_spa.mean().item()),
        "w_spe_mean": float(w_spe.mean().item()),
        "router_entropy_mean": float(ctx.router_entropy.mean().item()),
        "dx_spa_over_dc": _safe_ratio(dX_spa.abs().mean(dim=(1, 2, 3)), dX_dc.abs().mean(dim=(1, 2, 3))),
        "dx_spe_over_dc": _safe_ratio(dX_spe.abs().mean(dim=(1, 2, 3)), dX_dc.abs().mean(dim=(1, 2, 3))),
        "router_w_spa": w_spa,
        "router_w_spe": w_spe,
        "mem_feat": mem_feat,
        "gamma_rg": gamma_rg,
        "gamma_rg_mean": float(gamma_rg.mean().item()),
        "gamma_rg_max": float(gamma_rg.max().item()),
        "dy": dy,
        "dz": dz,
        "dX_rg": dX_rg,
        "y_hat": y_hat,
        "z_hat": z_hat,
        "ry_mean": float(ry.mean().item()),
        "ry_abs_mean": float(ry.abs().mean().item()),
        "rz_mean": float(rz.mean().item()),
        "rz_abs_mean": float(rz.abs().mean().item()),
    }
    return x_tp, aux


__all__ = ["init_state", "step_fn"]
