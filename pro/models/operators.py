"""Operators: Ay/Az/PSF/SRF/Warp/OmegaTheta."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ThetaDict = Dict[str, torch.Tensor]
# ===== PSF =====
PSF_KERNEL_SIZE = 9
PSF_BASIS_COUNT = 8
_PSF_BASIS_CACHE: dict[tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}


def _psf_basis(kernel_size: int, kpsf: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (kernel_size, kpsf, device, dtype)
    cached = _PSF_BASIS_CACHE.get(key)
    if cached is not None:
        return cached
    radius = kernel_size // 2
    grid = torch.arange(kernel_size, device=device, dtype=dtype) - radius
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    sigmas = torch.linspace(0.6, 2.4, steps=kpsf, device=device, dtype=dtype)
    basis = []
    for sigma in sigmas:
        denom = 2.0 * sigma.pow(2) + 1e-6
        kernel = torch.exp(-(xx.pow(2) + yy.pow(2)) / denom)
        kernel = kernel / (kernel.sum() + 1e-6)
        basis.append(kernel)
    out = torch.stack(basis, dim=0)
    _PSF_BASIS_CACHE[key] = out
    return out


def build_psf_kernels(theta: ThetaDict, kernel_size: int = PSF_KERNEL_SIZE) -> torch.Tensor | None:
    psf_coef = theta.get("psf_coef", None)
    if psf_coef is None:
        return None
    psf_coef = psf_coef.to(device=next(iter(theta.values())).device, dtype=next(iter(theta.values())).dtype)
    if psf_coef.dim() != 3:
        raise ValueError("psf_coef must be [B,Bh,Kpsf]")
    b, bh, kpsf = psf_coef.shape
    basis = _psf_basis(kernel_size, kpsf, psf_coef.device, psf_coef.dtype)  # [K,ks,ks]
    kernel_raw = torch.einsum("bck,kxy->bcxy", psf_coef, basis)
    kernel = F.softplus(kernel_raw)
    kernel = kernel / (kernel.sum(dim=(2, 3), keepdim=True) + 1e-6)
    return kernel.view(b, bh, 1, kernel_size, kernel_size)


def apply_psf_blur(x: torch.Tensor, theta: ThetaDict | None = None, k: torch.Tensor | None = None) -> torch.Tensor:
    if k is None:
        if theta is None:
            return x
        k = build_psf_kernels(theta)
    if k is None:
        return x
    b, bh, h, w = x.shape
    kernel_size = k.shape[-1]
    x_reshaped = x.reshape(1, b * bh, h, w)
    weight = k.reshape(b * bh, 1, kernel_size, kernel_size)
    out = F.conv2d(x_reshaped, weight, padding=kernel_size // 2, groups=b * bh)
    return out.view(b, bh, h, w)


def apply_psf_blur_transpose(x: torch.Tensor, theta: ThetaDict | None = None, k: torch.Tensor | None = None) -> torch.Tensor:
    if k is None:
        if theta is None:
            return x
        k = build_psf_kernels(theta)
    if k is None:
        return x
    k_flip = torch.flip(k, dims=[-2, -1])
    b, bh, h, w = x.shape
    kernel_size = k_flip.shape[-1]
    x_reshaped = x.reshape(1, b * bh, h, w)
    weight = k_flip.reshape(b * bh, 1, kernel_size, kernel_size)
    out = F.conv2d(x_reshaped, weight, padding=kernel_size // 2, groups=b * bh)
    return out.view(b, bh, h, w)

# ===== Warp =====
_WARP_GRID_CACHE: dict[tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}
_LAMBDA_CACHE: dict[tuple[int, int, torch.device, torch.dtype], torch.Tensor] = {}


def _base_grid(height: int, width: int, device, dtype) -> torch.Tensor:
    key = (height, width, device, dtype)
    cached = _WARP_GRID_CACHE.get(key)
    if cached is not None:
        return cached
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    x_norm = 2.0 * (xx + 0.5) / float(width) - 1.0
    y_norm = 2.0 * (yy + 0.5) / float(height) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0)  # [1,H,W,2]
    _WARP_GRID_CACHE[key] = grid
    return grid


def _lambda_pows(bh: int, p: int, device, dtype) -> torch.Tensor:
    key = (bh, p, device, dtype)
    cached = _LAMBDA_CACHE.get(key)
    if cached is not None:
        return cached
    lam = torch.linspace(-1.0, 1.0, steps=bh, device=device, dtype=dtype)
    pows = [torch.ones_like(lam)]
    for i in range(1, p + 1):
        pows.append(lam.pow(i))
    out = torch.stack(pows, dim=1)  # [Bh, P+1]
    _LAMBDA_CACHE[key] = out
    return out


def _flow_to_norm(flow: torch.Tensor, height: int, width: int) -> torch.Tensor:
    scale = torch.tensor([2.0 / float(width), 2.0 / float(height)], device=flow.device, dtype=flow.dtype)
    return flow * scale.view(1, 1, 2, 1, 1)


def _upsample_warp_ctrl(warp_ctrl: torch.Tensor, height: int, width: int) -> torch.Tensor:
    b, c, p1, hc, wc = warp_ctrl.shape
    warp_ctrl_flat = warp_ctrl.view(b, c * p1, hc, wc)
    warp_ctrl_hr = F.interpolate(warp_ctrl_flat, size=(height, width), mode="bilinear", align_corners=False)
    return warp_ctrl_hr.view(b, c, p1, height, width)


def _warp_apply(x: torch.Tensor, theta: ThetaDict, scale: int, sign: float, chunk_bands: int = 8) -> torch.Tensor:
    warp_ctrl = theta.get("warp_ctrl", None)
    if warp_ctrl is None:
        return x
    b, bh, h, w = x.shape
    warp_ctrl = warp_ctrl.to(device=x.device, dtype=x.dtype)
    if warp_ctrl.dim() != 5 or warp_ctrl.shape[1] != 2:
        raise ValueError("warp_ctrl must be [B,2,P+1,Hc,Wc]")
    p1 = warp_ctrl.shape[2]
    warp_ctrl_hr = _upsample_warp_ctrl(warp_ctrl, h, w)
    lam_pows = _lambda_pows(bh, p1 - 1, x.device, x.dtype)

    delta_lr = theta.get("delta_lr", None)
    if delta_lr is None:
        delta_lr = torch.zeros(b, 2, device=x.device, dtype=x.dtype)
    delta_lr = delta_lr.to(device=x.device, dtype=x.dtype)
    delta_hr = delta_lr * float(scale)

    base_grid = _base_grid(h, w, x.device, x.dtype)
    if chunk_bands <= 0:
        chunk_bands = bh
    out_chunks = []
    for start in range(0, bh, chunk_bands):
        end = min(start + chunk_bands, bh)
        lam_chunk = lam_pows[start:end]  # [K,P+1]
        flow = torch.einsum("b c p h w, k p -> b k c h w", warp_ctrl_hr, lam_chunk)
        flow = flow + delta_hr.view(b, 1, 2, 1, 1)
        flow = flow * float(sign)
        flow_norm = _flow_to_norm(flow, h, w)
        grid = base_grid + flow_norm.permute(0, 1, 3, 4, 2).reshape(b * (end - start), h, w, 2)
        x_chunk = x[:, start:end, :, :].reshape(b * (end - start), 1, h, w)
        out = F.grid_sample(x_chunk, grid, mode="bilinear", padding_mode="border", align_corners=False)
        out_chunks.append(out.view(b, end - start, h, w))
    return torch.cat(out_chunks, dim=1)


def build_keystone_flow(
    theta: ThetaDict,
    *,
    scale: int,
    out_hw: tuple[int, int],
    chunk_bands: int = 8,
) -> torch.Tensor | None:
    warp_ctrl = theta.get("warp_ctrl", None)
    if warp_ctrl is None:
        return None
    psf_coef = theta.get("psf_coef", None)
    if psf_coef is None or psf_coef.dim() != 3:
        raise ValueError("build_keystone_flow requires psf_coef with shape [B,Bh,Kpsf]")
    b, bh, _ = psf_coef.shape
    warp_ctrl = warp_ctrl.to(device=psf_coef.device, dtype=psf_coef.dtype)
    if warp_ctrl.dim() != 5 or warp_ctrl.shape[1] != 2:
        raise ValueError("warp_ctrl must be [B,2,P+1,Hc,Wc]")
    height, width = out_hw
    p1 = warp_ctrl.shape[2]
    warp_ctrl_hr = _upsample_warp_ctrl(warp_ctrl, height, width)
    lam_pows = _lambda_pows(bh, p1 - 1, psf_coef.device, psf_coef.dtype)
    delta_lr = theta.get("delta_lr", None)
    if delta_lr is None:
        delta_lr = torch.zeros(b, 2, device=psf_coef.device, dtype=psf_coef.dtype)
    delta_hr = delta_lr.to(device=psf_coef.device, dtype=psf_coef.dtype) * float(scale)
    if chunk_bands <= 0:
        chunk_bands = bh
    out_chunks = []
    for start in range(0, bh, chunk_bands):
        end = min(start + chunk_bands, bh)
        lam_chunk = lam_pows[start:end]
        flow = torch.einsum("b c p h w, k p -> b k c h w", warp_ctrl_hr, lam_chunk)
        flow = flow + delta_hr.view(b, 1, 2, 1, 1)
        out_chunks.append(flow)
    return torch.cat(out_chunks, dim=1)


def warp_apply(x_hr: torch.Tensor, theta: ThetaDict, scale: int, chunk_bands: int = 8) -> torch.Tensor:
    """Apply Keystone warp using per-band polynomial flow + global shift."""
    return _warp_apply(x_hr, theta, scale, sign=1.0, chunk_bands=chunk_bands)


def warp_inverse_apply(x_hr: torch.Tensor, theta: ThetaDict, scale: int, chunk_bands: int = 8) -> torch.Tensor:
    """Approximate inverse warp using negative flow (small deformation assumption)."""
    return _warp_apply(x_hr, theta, scale, sign=-1.0, chunk_bands=chunk_bands)

# ===== SRF + Smile =====

class SRFSmile(nn.Module):
    def __init__(self, bm: int, bh: int):
        super().__init__()
        self.bm = bm
        self.bh = bh
        self.srf_base_logits = nn.Parameter(torch.zeros(bm, bh))

    def srf_base(self) -> torch.Tensor:
        return torch.softmax(self.srf_base_logits, dim=-1)

    @staticmethod
    def _upsample_col_param(param: torch.Tensor | None, width: int, bm: int, device, dtype) -> torch.Tensor:
        if param is None:
            return torch.zeros(1, bm, width, device=device, dtype=dtype)
        b, bm_in, wcs = param.shape
        if bm_in != bm:
            raise ValueError("smile param bm mismatch")
        param = param.to(device=device, dtype=dtype)
        param_1d = param.view(b * bm, 1, wcs)
        out = F.interpolate(param_1d, size=width, mode="linear", align_corners=False)
        return out.view(b, bm, width)

    def build_weights(self, theta: ThetaDict, width: int) -> torch.Tensor:
        b = next(iter(theta.values())).shape[0]
        device = next(iter(theta.values())).device
        dtype = next(iter(theta.values())).dtype
        shift_raw = theta.get("smile_shift_col", None)
        temp_raw = theta.get("smile_temp_col", None)
        shift = self._upsample_col_param(shift_raw, width, self.bm, device, dtype)
        temp = self._upsample_col_param(temp_raw, width, self.bm, device, dtype)
        if shift.shape[0] == 1 and b > 1:
            shift = shift.expand(b, -1, -1)
        if temp.shape[0] == 1 and b > 1:
            temp = temp.expand(b, -1, -1)

        srf_base = self.srf_base().to(device=device, dtype=dtype)
        base = srf_base.view(1, self.bm, self.bh, 1).expand(b, -1, -1, -1)
        base = base.reshape(b * self.bm, 1, self.bh, 1)

        grid_y = torch.linspace(-1.0, 1.0, steps=self.bh, device=device, dtype=dtype).view(1, self.bh, 1)
        shift_norm = shift * (2.0 / float(max(self.bh - 1, 1)))
        grid_y = grid_y + shift_norm.unsqueeze(2)
        grid_y = grid_y.view(b * self.bm, self.bh, width)
        grid_x = torch.zeros_like(grid_y)
        grid = torch.stack([grid_x, grid_y], dim=-1)

        shifted = F.grid_sample(base, grid, mode="bilinear", padding_mode="border", align_corners=True)
        shifted = shifted.view(b, self.bm, self.bh, width)
        logits = torch.log(shifted + 1e-6) / (temp.unsqueeze(2) + 1e-6)
        weights = torch.softmax(logits, dim=2)
        return weights

    def az_mix(self, x_blur: torch.Tensor, theta: ThetaDict, chunk_cols: int = 32) -> torch.Tensor:
        b, bh, h, w = x_blur.shape
        weights = self.build_weights(theta, w)
        out_chunks = []
        if chunk_cols <= 0:
            chunk_cols = w
        for start in range(0, w, chunk_cols):
            end = min(start + chunk_cols, w)
            x_chunk = x_blur[:, :, :, start:end]
            w_chunk = weights[:, :, :, start:end]
            z_chunk = torch.einsum("b k h c, b m k c -> b m h c", x_chunk, w_chunk)
            out_chunks.append(z_chunk)
        return torch.cat(out_chunks, dim=3)

    def az_transpose(self, rz: torch.Tensor, theta: ThetaDict, chunk_cols: int = 32) -> torch.Tensor:
        b, bm, h, w = rz.shape
        weights = self.build_weights(theta, w)
        out_chunks = []
        if chunk_cols <= 0:
            chunk_cols = w
        for start in range(0, w, chunk_cols):
            end = min(start + chunk_cols, w)
            r_chunk = rz[:, :, :, start:end]
            w_chunk = weights[:, :, :, start:end]
            x_chunk = torch.einsum("b m h c, b m k c -> b k h c", r_chunk, w_chunk)
            out_chunks.append(x_chunk)
        return torch.cat(out_chunks, dim=3)

    def base_entropy(self) -> torch.Tensor:
        srf = self.srf_base()
        log_srf = torch.log(srf + 1e-8)
        entropy = -(srf * log_srf).sum(dim=1)
        return entropy


def upsample_smile_params(theta: ThetaDict, width: int, bm: int) -> tuple[torch.Tensor, torch.Tensor]:
    b = next(iter(theta.values())).shape[0]
    device = next(iter(theta.values())).device
    dtype = next(iter(theta.values())).dtype
    shift_raw = theta.get("smile_shift_col", None)
    temp_raw = theta.get("smile_temp_col", None)
    shift = SRFSmile._upsample_col_param(shift_raw, width, bm, device, dtype)
    temp = SRFSmile._upsample_col_param(temp_raw, width, bm, device, dtype)
    if shift.shape[0] == 1 and b > 1:
        shift = shift.expand(b, -1, -1)
    if temp.shape[0] == 1 and b > 1:
        temp = temp.expand(b, -1, -1)
    return shift, temp

# ===== Ay / Az =====

def ay_forward(x: torch.Tensor, theta: ThetaDict, scale: int) -> torch.Tensor:
    """Ay: HR-HSI -> LR-HSI using warp, PSF blur, and average pooling."""
    x_w = warp_apply(x, theta, scale)
    k = build_psf_kernels(theta)
    x_b = apply_psf_blur(x_w, k=k)
    return F.avg_pool2d(x_b, kernel_size=scale, stride=scale)


def az_forward(x: torch.Tensor, theta: ThetaDict, scale: int, srf_smile: SRFSmile | None) -> torch.Tensor:
    """Az: HR-HSI -> HR-MSI via warp, PSF blur, and SRF+Smile mixing."""
    if srf_smile is None:
        raise ValueError("srf_smile module is required for az_forward")
    x_w = warp_apply(x, theta, scale)
    k = build_psf_kernels(theta)
    x_b = apply_psf_blur(x_w, k=k)
    return srf_smile.az_mix(x_b, theta)


def az_transpose(rz: torch.Tensor, theta: ThetaDict, srf_smile: SRFSmile | None) -> torch.Tensor:
    """S^T: back-project MSI residual to HSI (no PSF/warp)."""
    if srf_smile is None:
        raise ValueError("srf_smile module is required for az_transpose")
    return srf_smile.az_transpose(rz, theta)

# ===== OmegaTheta =====

def omega_theta(
    theta: ThetaDict,
    *,
    warp_max: float = 1.0,
    delta_max_lr: float = 2.0,
    shift_max: float = 2.0,
    temp_min: float = 0.3,
    temp_max: float = 3.0,
    return_stats: bool = False,
) -> ThetaDict | Tuple[ThetaDict, Dict[str, torch.Tensor]]:
    """Project theta to a feasible domain for stable operators."""
    out: ThetaDict = {}
    stats: Dict[str, torch.Tensor] = {}

    warp_ctrl = theta.get("warp_ctrl", None)
    if warp_ctrl is not None:
        warp_ctrl = warp_max * torch.tanh(warp_ctrl)
        warp_ctrl = warp_ctrl - warp_ctrl.mean(dim=(-1, -2), keepdim=True)
        out["warp_ctrl"] = warp_ctrl
        stats["warp_rms"] = torch.sqrt(warp_ctrl.pow(2).mean(dim=(1, 2, 3, 4)))
        stats["warp_max"] = warp_ctrl.abs().amax(dim=(1, 2, 3, 4))

    delta_lr = theta.get("delta_lr", None)
    if delta_lr is not None:
        delta_lr = delta_max_lr * torch.tanh(delta_lr)
        out["delta_lr"] = delta_lr
        stats["delta_norm"] = torch.sqrt(delta_lr.pow(2).sum(dim=1))

    psf_coef = theta.get("psf_coef", None)
    if psf_coef is not None:
        out["psf_coef"] = psf_coef

    smile_shift_col = theta.get("smile_shift_col", None)
    if smile_shift_col is not None:
        smile_shift_col = shift_max * torch.tanh(smile_shift_col)
        out["smile_shift_col"] = smile_shift_col
        stats["smile_shift_rms"] = torch.sqrt(smile_shift_col.pow(2).mean(dim=(1, 2)))
        stats["smile_shift_max"] = smile_shift_col.abs().amax(dim=(1, 2))

    smile_temp_col = theta.get("smile_temp_col", None)
    if smile_temp_col is not None:
        smile_temp_col = F.softplus(smile_temp_col) + temp_min
        smile_temp_col = torch.clamp(smile_temp_col, max=temp_max)
        out["smile_temp_col"] = smile_temp_col
        stats["smile_temp_mean"] = smile_temp_col.mean(dim=(1, 2))
        stats["smile_temp_std"] = smile_temp_col.std(dim=(1, 2), unbiased=False)

    theta_feat = theta.get("theta_feat", None)
    if theta_feat is not None:
        out["theta_feat"] = theta_feat

    if return_stats:
        return out, stats
    return out


__all__ = [
    "ay_forward",
    "az_forward",
    "az_transpose",
    "omega_theta",
    "build_psf_kernels",
    "build_keystone_flow",
    "upsample_smile_params",
    "apply_psf_blur",
    "apply_psf_blur_transpose",
    "SRFSmile",
    "warp_apply",
    "warp_inverse_apply",
]
