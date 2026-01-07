"""
BlindHSIModel (A-adaptive Architecture)
按照修正.md实现X与Theta的纠缠更新
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from models.context import ContextEncoderLite
from models.branches import SpaCore, SpeCore, HeadSpa, HeadSpe
from models.fusion_core import FusionCoreNSSMamba
from models.fusion.cross_modal import EncX, EncY, EncZ, CrossAttn
from models.dc.mismatch_adapter import MeasMismatchAdapter
from models.align.residual_guided import ResidualGuidedAlign
from models.operators_adaptive import KeystoneResampler, SmileResampler
from models.theta.theta_adaptive import ThetaAdaptiveCore, init_theta_state
from models.step_adaptive import (
    AdaptiveState,
    AdaptiveStepFunction,
    init_adaptive_state,
    pack_state,
    unpack_state,
)
from models.config import SpaConfig, SpeConfig, FusionConfig
from models.deq_implicit import DEQImplicit, fixed_point_iteration


class HeadDCAdaptive(nn.Module):
    """
    DC Head for A-adaptive architecture
    基于物理残差和Theta状态的数据一致性更新
    """

    def __init__(
        self,
        bh: int,
        bm: int,
        scale: int = 4,
        hidden: int = 64,
    ):
        super().__init__()
        self.scale = scale

        # 将残差映射到HR空间
        self.proj_ry = nn.Sequential(
            nn.Conv2d(bh, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, bh, kernel_size=1),
        )

        self.proj_rz = nn.Sequential(
            nn.Conv2d(bm, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, bh, kernel_size=1),
        )

    def forward(
        self,
        ry: torch.Tensor,
        rz: torch.Tensor,
        theta_state,
    ) -> torch.Tensor:
        """
        Args:
            ry: [B, Bh, H_lr, W_lr] Y域残差
            rz: [B, Bm, H, W] Z域残差
            theta_state: ThetaState

        Returns:
            dX_dc: [B, Bh, H, W] DC更新量
        """
        import torch.nn.functional as F

        # Upsample ry to HR
        ry_hr = F.interpolate(ry, scale_factor=self.scale, mode="bilinear", align_corners=False)

        # Project residuals
        dry = self.proj_ry(ry_hr)
        drz = self.proj_rz(rz)

        # Simple weighted sum (可以扩展为更复杂的物理反投影)
        dX_dc = dry + drz

        return dX_dc


class MainNetAdaptive(nn.Module):
    """
    A-adaptive MainNet with X-Theta entangled updates

    核心改动:
    1. DEQ状态从 X 扩展为 (X, Theta)
    2. Theta在循环内部更新(自适应)
    3. 使用物理重采样算子而非矩阵乘法
    """

    def __init__(
        self,
        bh: int,
        bm: int,
        scale: int = 4,
        spa_cfg: SpaConfig | None = None,
        spe_cfg: SpeConfig | None = None,
        fusion_cfg: FusionConfig | None = None,
        head_last_zero_init: bool = True,
        prior_dx_clip_scale: float = 0.0,
        theta_damping: float = 0.5,  # Theta阻尼系数
        theta_hidden: int = 128,
        theta_feat_dim: int = 32,
        theta_map_dim: int = 64,
        max_keystone_shift: float = 2.0,
        max_smile_shift: float = 2.0,
    ) -> None:
        super().__init__()
        spa_cfg = spa_cfg or SpaConfig()
        spe_cfg = spe_cfg or SpeConfig()
        fusion_cfg = fusion_cfg or FusionConfig()

        self.bh = bh
        self.bm = bm
        self.scale = scale
        self.theta_damping = theta_damping

        # ===== Core Modules =====
        self.context = ContextEncoderLite(theta_dim=theta_feat_dim)

        self.spa = SpaCore(
            bh=bh,
            bm=bm,
            c_spa=spa_cfg.c_spa,
            scale=scale,
            spa_local_blocks=spa_cfg.local_blocks,
            spa_grid_down=spa_cfg.grid_down,
            spa_grid_d=spa_cfg.grid_d,
            spa_mamba_layers=spa_cfg.mamba_layers,
            spa_mamba_d_state=spa_cfg.mamba_d_state,
            spa_mamba_d_conv=spa_cfg.mamba_d_conv,
            spa_grid_L_max=spa_cfg.grid_L_max,
            spa_grid_L_min=spa_cfg.grid_L_min,
            spa_grid_min_hw=spa_cfg.grid_min_hw,
            spa_fuse_alpha_init=spa_cfg.fuse_alpha_init,
            spa_film=spa_cfg.film,
        )

        self.spe = SpeCore(
            bh=bh,
            bm=bm,
            c_spe=spe_cfg.c_spe,
            d=spe_cfg.d,
            scale=scale,
            spe_k_mode=spe_cfg.k_mode,
            spe_k_max=spe_cfg.k_max,
            spe_mamba_layers=spe_cfg.mamba_layers,
            spe_mamba_d_state=spe_cfg.mamba_d_state,
            spe_mamba_d_conv=spe_cfg.mamba_d_conv,
            spe_compress_mode=spe_cfg.compress_mode,
            spe_tokenize_mode=spe_cfg.tokenize_mode,
            spe_patch_stride=spe_cfg.patch_stride,
            spe_patch_kernel=spe_cfg.patch_kernel,
            spe_patch_padding=spe_cfg.patch_padding,
            spe_chunk_bn=spe_cfg.chunk_bn,
            spe_readout_heads=spe_cfg.readout_heads,
        )

        self.fusion = FusionCoreNSSMamba(
            bh=bh,
            d_model=fusion_cfg.d_model,
            mamba_layers=fusion_cfg.mamba_layers,
            d_state=fusion_cfg.mamba_d_state,
            d_conv=fusion_cfg.mamba_d_conv,
            expand=fusion_cfg.mamba_expand,
            theta_dim=theta_feat_dim,
            mem_dim=fusion_cfg.mem_dim,
            grid_hw=fusion_cfg.grid_hw,
            scan_len=fusion_cfg.scan_len,
            map_channels=fusion_cfg.map_channels,
            out_clip=fusion_cfg.out_clip,
        )

        self.enc_x = EncX(bh, d_model=fusion_cfg.d_model)
        self.enc_y = EncY(bh, d_model=fusion_cfg.d_model)
        self.enc_z = EncZ(bm, d_model=fusion_cfg.d_model)
        self.cross_attn = CrossAttn(d_model=fusion_cfg.d_model, heads=4)

        cond_dim = theta_feat_dim + fusion_cfg.mem_dim
        self.mismatch_adapter = MeasMismatchAdapter(bh=bh, bm=bm, cond_dim=cond_dim)
        self.rg_align = ResidualGuidedAlign(bh=bh, bm=bm, cond_dim=cond_dim)

        self.head_spa = HeadSpa(
            c_in=spa_cfg.c_spa,
            bh=bh,
            scale=scale,
            zero_init=head_last_zero_init,
            clip_scale=prior_dx_clip_scale,
        )

        self.head_spe = HeadSpe(
            c_in=spe_cfg.c_spe,
            bh=bh,
            scale=scale,
            zero_init=head_last_zero_init,
            clip_scale=prior_dx_clip_scale,
        )

        # ===== A-adaptive Components =====
        self.theta_adaptive = ThetaAdaptiveCore(
            bh=bh,
            bm=bm,
            hidden=theta_hidden,
            feat_dim=theta_feat_dim,
            map_dim=theta_map_dim,
            max_keystone_shift=max_keystone_shift,
            max_smile_shift=max_smile_shift,
        )

        self.keystone_resampler = KeystoneResampler(max_shift_pixels=max_keystone_shift)
        self.smile_resampler = SmileResampler(max_shift_bands=max_smile_shift)

        self.head_dc_adaptive = HeadDCAdaptive(bh=bh, bm=bm, scale=scale)

        self._last_deq: DEQImplicit | None = None

    def forward(
        self,
        y_lr: torch.Tensor,
        z_hr: torch.Tensor,
        *,
        ctx: Dict | None = None,
        ablations: Dict | None = None,
    ) -> Dict:
        """
        Forward pass with A-adaptive architecture

        Args:
            y_lr: [B, Bh, H_lr, W_lr] 低分辨率观测
            z_hr: [B, Bm, H, W] 高分辨率观测
            ctx: 配置上下文
            ablations: 消融开关

        Returns:
            包含 X, Theta, 辅助信息的字典
        """
        # 默认参数 (按修正.md调整)
        defaults = {
            "steps": 35,  # 增加迭代数(双变量收敛慢)
            "scale": self.scale,
            "deq_tol": 1e-4,
            "fwd_check_every": 1,
            "fwd_min_iter": 10,  # 防止Theta未更新就退出
            "stagnation_check_start": 25,
            "fwd_soft_accept_res": 1e-3,
            "gmres_max_iter": 12,
            "gmres_tol": 1e-4,
            "gmres_early_stop": True,
            "gmres_early_stop_min_iter": 4,
            "cross_grid": 16,
            "bwd_mode": "implicit",
        }
        params = dict(defaults)
        if isinstance(ctx, dict):
            ctx_payload = ctx.get("params") if isinstance(ctx.get("params"), dict) else ctx
            params.update(ctx_payload)
        if ablations is not None:
            params["ablations"] = ablations

        steps = int(params.get("steps", 35))
        deq_tol = float(params.get("deq_tol", 1e-4))

        B, Bh, H_lr, W_lr = y_lr.shape
        _, Bm, H, W = z_hr.shape

        # 初始化 AdaptiveState (按修正.md: X从上采样, Theta从零畸变)
        state0 = init_adaptive_state(y_lr, scale=self.scale, bh=self.bh)

        # 打包为单个tensor (DEQ需要)
        z0_packed = pack_state(state0)

        # 构造模块字典
        modules = {
            "context": self.context,
            "mismatch_adapter": self.mismatch_adapter,
            "spa": self.spa,
            "spe": self.spe,
            "fusion": self.fusion,
            "head_spa": self.head_spa,
            "head_spe": self.head_spe,
            "head_dc_adaptive": self.head_dc_adaptive,
            "rg_align": self.rg_align,
            "enc_x": self.enc_x,
            "enc_y": self.enc_y,
            "enc_z": self.enc_z,
            "cross_attn": self.cross_attn,
            "theta_adaptive": self.theta_adaptive,
            "keystone_resampler": self.keystone_resampler,
            "smile_resampler": self.smile_resampler,
        }

        params_deq = dict(params)
        params_deq["scale"] = self.scale
        params_deq["theta_damping"] = self.theta_damping
        params_deq["collect_stats"] = False
        params_deq["fwd_check_every"] = int(params_deq.get("fwd_check_every", 1))
        params_deq["fwd_min_iter"] = int(params_deq.get("fwd_min_iter", 10))

        deq_stats = {
            "batch": B,
            "device": y_lr.device,
            "dtype": y_lr.dtype,
        }
        params_deq["deq_stats"] = deq_stats

        # 构造bundle
        step_bundle = {
            "inputs": {"Y_lr": y_lr, "Z_hr": z_hr},
            "modules": modules,
            "params": params_deq,
            "deq_stats": deq_stats,
            "last_aux": None,
            "bh": self.bh,
            "height": H,
            "width": W,
        }

        # 定义闭包 (解包->调用->打包)
        def _adaptive_step_fn(z_packed: torch.Tensor, bundle: Dict) -> torch.Tensor:
            # 解包状态
            state_in = unpack_state(
                z_packed,
                bh=bundle["bh"],
                height=bundle["height"],
                width=bundle["width"],
            )

            # 创建 AdaptiveStepFunction
            step_fn = AdaptiveStepFunction(
                modules=bundle["modules"],
                params=bundle["params"],
                inputs=bundle["inputs"],
                theta_damping=bundle["params"]["theta_damping"],
            )

            # 执行一步纠缠更新
            state_out, aux = step_fn(state_in)

            # 保存辅助信息
            bundle["last_aux"] = aux

            # 打包返回
            z_out_packed = pack_state(state_out)
            return z_out_packed

        # DEQ求解
        deq = DEQImplicit(_adaptive_step_fn, solver=fixed_point_iteration, max_iter=steps, tol=deq_tol)
        z_star_packed = deq(step_bundle, z0_packed)
        self._last_deq = deq

        # 解包最终状态
        state_star = unpack_state(
            z_star_packed,
            bh=self.bh,
            height=H,
            width=W,
        )

        last_aux = step_bundle.get("last_aux") or {}

        def _scalar(val, default: float = 0.0) -> float:
            if val is None:
                return default
            if torch.is_tensor(val):
                return float(val.mean().item())
            return float(val)

        solver = {
            "pcdeq_iter_fwd_last": int(deq.iter_forward),
            "pcdeq_res_fwd_last": float(deq.res_forward[-1]) if deq.res_forward else 0.0,
            "pcdeq_stop_reason": deq.stop_reason_forward,
            "pcdeq_fail_reason": deq.fail_reason_forward,
            "theta_gmres_iters": int(deq.iter_backward),
            "theta_gmres_relres": float(deq.res_backward[-1]) if deq.res_backward else 0.0,
            "theta_gmres_fail_reason": deq.fail_reason_backward,
            # 从 aux 提取统计
            "y_hat_ry_mean": _scalar(last_aux.get("ry_mean", 0.0)),
            "z_hat_rz_mean": _scalar(last_aux.get("rz_mean", 0.0)),
            "x_min": _scalar(last_aux.get("x_min", 0.0)),
            "x_max": _scalar(last_aux.get("x_max", 1.0)),
            "keystone_flow_mean": _scalar(last_aux.get("keystone_flow_mean", 0.0)),
            "smile_shift_mean": _scalar(last_aux.get("smile_shift_mean", 0.0)),
        }

        return {
            "X": state_star.x,
            "theta": state_star.theta,
            "aux": last_aux,
            "solver": solver,
        }


BlindHSIModelAdaptive = MainNetAdaptive

__all__ = ["MainNetAdaptive", "BlindHSIModelAdaptive"]
