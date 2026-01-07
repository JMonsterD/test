"""
Step operator for A-adaptive DEQ
按照修正.md实现X与Theta的纠缠更新
"""

from __future__ import annotations

from typing import Dict, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.theta.theta_adaptive import ThetaState, ThetaAdaptiveCore, init_theta_state
from models.operators_adaptive import KeystoneResampler, SmileResampler


@dataclass
class AdaptiveState:
    """扩展的DEQ状态: (X, Theta)"""
    x: torch.Tensor          # [B, Bh, H, W]
    theta: ThetaState        # Theta状态


def init_adaptive_state(
    y_lr: torch.Tensor,
    scale: int,
    bh: int,
) -> AdaptiveState:
    """
    初始化扩展状态

    按照修正.md:
    - X: 初始化为上采样的 Y_lr
    - Theta: 初始化为零畸变（Identity）
    """
    B, _, H_lr, W_lr = y_lr.shape
    H, W = H_lr * scale, W_lr * scale

    # 初始化 X
    x0 = F.interpolate(y_lr, scale_factor=scale, mode="bilinear", align_corners=False)

    # 初始化 Theta (零畸变)
    theta0 = init_theta_state(
        batch_size=B,
        bh=bh,
        height=H,
        width=W,
        device=y_lr.device,
        dtype=y_lr.dtype,
    )

    return AdaptiveState(x=x0, theta=theta0)


class AdaptiveStepFunction:
    """
    A-adaptive Step Function (纠缠更新)

    核心逻辑 (修正.md 第3节):
    1. 物理正向 (Entanglement): 利用当前 X 和 Theta 计算 Y_hat, Z_hat
    2. Theta更新 (A-adaptive): Theta_new = Theta + ΔTheta(residual)
    3. X更新 (Reconstruction): X_new = X + ΔX(residual)
    """

    def __init__(
        self,
        modules: Dict,
        params: Dict,
        inputs: Dict,
        theta_damping: float = 0.5,  # Theta阻尼系数
    ):
        """
        Args:
            modules: 包含所有网络模块的字典
            params: 配置参数
            inputs: 输入数据 (Y_lr, Z_hr等)
            theta_damping: Theta更新的阻尼系数 (0.5 = 50%阻尼)
        """
        self.modules = modules
        self.params = params
        self.inputs = inputs
        self.theta_damping = theta_damping

        # 提取模块
        self.theta_net: ThetaAdaptiveCore = modules["theta_adaptive"]
        self.keystone_resampler: KeystoneResampler = modules["keystone_resampler"]
        self.smile_resampler: SmileResampler = modules["smile_resampler"]

        # 提取输入
        self.y_lr = inputs["Y_lr"]
        self.z_hr = inputs["Z_hr"]
        self.scale = int(params.get("scale", 4))

    def _physical_forward(
        self,
        x: torch.Tensor,
        theta: ThetaState,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        物理正向模型: A(X, Theta) -> (Y_hat, Z_hat)

        Step 1: 纠缠正向投影
        """
        # 1. Apply Keystone distortion
        x_warped = self.keystone_resampler(x, theta.keystone_flow)

        # 2. Apply PSF blur (optional)
        # TODO: 如果需要PSF，在这里添加

        # 3. Downsample to Y_lr
        y_hat = F.avg_pool2d(x_warped, kernel_size=self.scale)

        # 4. Apply Smile distortion to X
        x_smile = self.smile_resampler(x, theta.smile_shift)

        # 5. Spectral integration (simulate Z_hr)
        # 使用SRF进行光谱降维 (如果有srf_smile)
        srf_smile = self.modules.get("srf_smile")
        if srf_smile is not None:
            # 使用SRF模块进行光谱降维
            # SRF: [Bm, Bh] 或可学习模块
            from models.operators import az_forward_simple
            # 简化版本：直接用matmul
            Bm = self.z_hr.shape[1]
            Bh = x_smile.shape[1]
            # 生成简单的平均SRF矩阵
            srf_matrix = torch.ones(Bm, Bh, device=x.device, dtype=x.dtype) / (Bh / Bm)
            # [Bm, Bh] @ [B, Bh, H, W] -> [B, Bm, H, W]
            x_smile_flat = x_smile.permute(0, 2, 3, 1).reshape(-1, Bh)  # [B*H*W, Bh]
            z_hat_flat = torch.matmul(x_smile_flat, srf_matrix.T)  # [B*H*W, Bm]
            z_hat = z_hat_flat.reshape(x.shape[0], x.shape[2], x.shape[3], self.z_hr.shape[1])  # [B, H, W, Bm]
            z_hat = z_hat.permute(0, 3, 1, 2)  # [B, Bm, H, W]
        else:
            # Fallback: 简单平均降维
            Bm = self.z_hr.shape[1]
            Bh = x_smile.shape[1]
            # 将Bh个波段分组平均到Bm个波段
            if Bh % Bm == 0:
                bands_per_group = Bh // Bm
                z_hat = x_smile.reshape(x.shape[0], Bm, bands_per_group, x.shape[2], x.shape[3]).mean(dim=2)
            else:
                # 不能整除时使用matmul
                srf_matrix = torch.ones(Bm, Bh, device=x.device, dtype=x.dtype) / (Bh / Bm)
                x_smile_flat = x_smile.permute(0, 2, 3, 1).reshape(-1, Bh)
                z_hat_flat = torch.matmul(x_smile_flat, srf_matrix.T)
                z_hat = z_hat_flat.reshape(x.shape[0], x.shape[2], x.shape[3], Bm)
                z_hat = z_hat.permute(0, 3, 1, 2)

        return y_hat, z_hat

    def _theta_update(
        self,
        theta_old: ThetaState,
        ry: torch.Tensor,
        rz: torch.Tensor,
    ) -> Tuple[ThetaState, torch.Tensor]:
        """
        Theta自适应更新

        Step 2: Theta_new = Theta + damping * ΔTheta(residual)
        """
        # Theta网络根据残差预测增量
        theta_new_raw, theta_feat = self.theta_net(
            y_lr=self.y_lr,
            z_hr=self.z_hr,
            ry=ry,
            rz=rz,
            prev_theta_state=theta_old,
        )

        # 应用阻尼
        if self.theta_damping < 1.0:
            theta_new = ThetaState(
                keystone_flow=theta_old.keystone_flow + self.theta_damping * (theta_new_raw.keystone_flow - theta_old.keystone_flow),
                smile_shift=theta_old.smile_shift + self.theta_damping * (theta_new_raw.smile_shift - theta_old.smile_shift),
                psf_coef=theta_new_raw.psf_coef,
            )
        else:
            theta_new = theta_new_raw

        return theta_new, theta_feat

    def _x_update(
        self,
        x_old: torch.Tensor,
        ry: torch.Tensor,
        rz: torch.Tensor,
        theta: ThetaState,
    ) -> torch.Tensor:
        """
        X重建更新

        Step 3: X_new = X + ΔX(residual, theta)

        这里调用原有的分支网络 (spa, spe, fusion, DC等)
        """
        # 使用原有的context, spa, spe, fusion, DC逻辑
        # 这部分保持与原 step.py 相似

        # Context
        ctx = self.modules["context"](ry, rz, theta_feat=None)

        # Branches
        f_spa = self.modules["spa"](self.y_lr, self.z_hr, ry, rz, ctx.u_spa)
        f_spe = self.modules["spe"](self.y_lr, self.z_hr, ry, rz, ctx.u_spe)

        # Heads
        dX_spa = self.modules["head_spa"](f_spa, ctx.u_spa)
        dX_spe = self.modules["head_spe"](f_spe, ctx.u_spe)

        # Weighted fusion
        w_spa_view = ctx.w_spa.view(-1, 1, 1, 1)
        w_spe_view = ctx.w_spe.view(-1, 1, 1, 1)
        dX_prior = dX_spa * w_spa_view + dX_spe * w_spe_view

        # DC term (基于残差)
        dX_dc = self.modules["head_dc_adaptive"](ry, rz, theta)

        # Combined update
        alpha_x = ctx.alpha_x.view(-1, 1, 1, 1)
        beta_dc = ctx.beta_dc.view(-1, 1, 1, 1)
        x_new = x_old + alpha_x * (dX_prior + beta_dc * dX_dc)

        return x_new

    def __call__(self, state: AdaptiveState) -> Tuple[AdaptiveState, Dict]:
        """
        执行一步纠缠更新

        Args:
            state: 当前状态 (X, Theta)

        Returns:
            new_state: 新状态
            aux: 辅助信息
        """
        x_old = state.x
        theta_old = state.theta

        # Step 1: 物理正向 (Entanglement)
        y_hat, z_hat = self._physical_forward(x_old, theta_old)

        # 计算残差
        ry = self.y_lr - y_hat
        rz = self.z_hr - z_hat

        # Step 2: Theta更新 (A-adaptive)
        theta_new, theta_feat = self._theta_update(theta_old, ry, rz)

        # Step 3: X更新 (Reconstruction)
        x_new_raw = self._x_update(x_old, ry, rz, theta_new)

        # 稳定性约束: Clamp X to [0, 1]
        x_new = torch.clamp(x_new_raw, min=0.0, max=1.0)

        # 构造新状态
        new_state = AdaptiveState(x=x_new, theta=theta_new)

        # 辅助信息
        aux = {
            "y_hat": y_hat,
            "z_hat": z_hat,
            "ry_mean": ry.mean().item(),
            "rz_mean": rz.mean().item(),
            "x_min": x_new.min().item(),
            "x_max": x_new.max().item(),
            "keystone_flow_mean": theta_new.keystone_flow.abs().mean().item(),
            "smile_shift_mean": theta_new.smile_shift.abs().mean().item(),
        }

        return new_state, aux


def pack_state(state: AdaptiveState) -> torch.Tensor:
    """
    将AdaptiveState打包为单个tensor（用于DEQ）

    这是一个trick：DEQ需要单个tensor作为状态
    我们将X和Theta flatten并concat
    """
    # 将Theta状态flatten
    theta_flat = torch.cat([
        state.theta.keystone_flow.flatten(1),
        state.theta.smile_shift.flatten(1),
    ], dim=1)  # [B, ...]

    # 将X flatten
    x_flat = state.x.flatten(1)  # [B, Bh*H*W]

    # Concat
    packed = torch.cat([x_flat, theta_flat], dim=1)  # [B, total_dim]

    return packed


def unpack_state(
    packed: torch.Tensor,
    bh: int,
    height: int,
    width: int,
) -> AdaptiveState:
    """
    从packed tensor恢复AdaptiveState
    """
    B = packed.shape[0]

    # 计算各部分维度
    x_dim = bh * height * width
    keystone_dim = bh * 2 * height * width
    smile_dim = bh * width

    # 分离
    x_flat = packed[:, :x_dim]
    keystone_flat = packed[:, x_dim:x_dim + keystone_dim]
    smile_flat = packed[:, x_dim + keystone_dim:x_dim + keystone_dim + smile_dim]

    # Reshape
    x = x_flat.reshape(B, bh, height, width)
    keystone_flow = keystone_flat.reshape(B, bh, 2, height, width)
    smile_shift = smile_flat.reshape(B, bh, width)

    theta = ThetaState(
        keystone_flow=keystone_flow,
        smile_shift=smile_shift,
        psf_coef=None,
    )

    return AdaptiveState(x=x, theta=theta)


__all__ = [
    "AdaptiveState",
    "AdaptiveStepFunction",
    "init_adaptive_state",
    "pack_state",
    "unpack_state",
]
