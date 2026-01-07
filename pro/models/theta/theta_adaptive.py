"""
Theta Branch for A-adaptive Architecture
按照修正.md改造，支持增量学习和硬约束
"""

from __future__ import annotations

from typing import Dict, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ThetaState:
    """Theta状态（用于DEQ纠缠更新）"""
    keystone_flow: torch.Tensor  # [B, Bh, 2, H, W] 几何流场
    smile_shift: torch.Tensor    # [B, Bh, W] 光谱偏移
    psf_coef: torch.Tensor | None = None  # [B, Bh, Kpsf] PSF系数（可选）


class ThetaAdaptiveCore(nn.Module):
    """
    A-adaptive Theta分支

    关键改动：
    1. 输出控制图（flow, shift）而非大矩阵
    2. 增加Tanh硬约束层
    3. 改为增量学习：输出 ΔTheta

    输出：
        - keystone_flow: [B, Bh, 2, H, W] 几何畸变流场
        - smile_shift: [B, Bh, W] 光谱偏移
        - theta_feat: [B, feat_dim] 特征向量（用于条件）
    """

    def __init__(
        self,
        bh: int,
        bm: int,
        hidden: int = 128,
        feat_dim: int = 32,
        map_dim: int = 64,
        max_keystone_shift: float = 2.0,  # 最大几何偏移（像素）
        max_smile_shift: float = 2.0,     # 最大光谱偏移（波段）
        psf_k: int = 8,
    ):
        super().__init__()
        self.bh = bh
        self.bm = bm
        self.feat_dim = feat_dim
        self.map_dim = map_dim
        self.max_keystone_shift = max_keystone_shift
        self.max_smile_shift = max_smile_shift
        self.psf_k = psf_k

        # ===== Encoders =====
        # 编码残差和观测
        self.enc_y = nn.Sequential(
            nn.Conv2d(bh, map_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.enc_z = nn.Sequential(
            nn.Conv2d(bm, map_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.enc_ry = nn.Sequential(
            nn.Conv2d(bh, map_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        self.enc_rz = nn.Sequential(
            nn.Conv2d(bm, map_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(map_dim * 4, map_dim * 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(map_dim * 2, map_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )

        # ===== Output Heads =====
        # Head_Keystone: 输出 [B, Bh, 2, H, W] 的流场
        self.head_keystone = nn.Sequential(
            nn.Conv2d(map_dim, map_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(map_dim, bh * 2, kernel_size=1),  # Output: [B, Bh*2, H, W]
        )

        # Head_Smile: 输出 [B, Bh, W] 的波长偏移
        # 先global pooling到 [B, C, W]，然后输出
        self.head_smile_pool = nn.AdaptiveAvgPool2d((1, None))  # Pool height -> 1
        self.head_smile = nn.Sequential(
            nn.Conv1d(map_dim, map_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(map_dim, bh, kernel_size=1),  # Output: [B, Bh, W]
        )

        # Global feature head (for conditioning)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head_feat = nn.Sequential(
            nn.Linear(map_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, feat_dim),
        )

        # Optional: PSF coefficient head
        self.head_psf = nn.Sequential(
            nn.Linear(map_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, bh * psf_k),
        )

    def _apply_keystone_constraint(self, flow_raw: torch.Tensor) -> torch.Tensor:
        """
        对Keystone流场施加Tanh硬约束

        Args:
            flow_raw: [B, Bh, 2, H, W]

        Returns:
            flow_constrained: [B, Bh, 2, H, W]
        """
        return self.max_keystone_shift * torch.tanh(flow_raw / self.max_keystone_shift)

    def _apply_smile_constraint(self, shift_raw: torch.Tensor) -> torch.Tensor:
        """
        对Smile偏移施加Tanh硬约束

        Args:
            shift_raw: [B, Bh, W]

        Returns:
            shift_constrained: [B, Bh, W]
        """
        return self.max_smile_shift * torch.tanh(shift_raw / self.max_smile_shift)

    def forward(
        self,
        y_lr: torch.Tensor,
        z_hr: torch.Tensor,
        ry: torch.Tensor,
        rz: torch.Tensor,
        prev_theta_state: ThetaState | None = None,
    ) -> Tuple[ThetaState, torch.Tensor]:
        """
        Forward pass (增量学习)

        Args:
            y_lr: [B, Bh, H_lr, W_lr] 低分辨率观测
            z_hr: [B, Bm, H, W] 高分辨率观测
            ry: [B, Bh, H_lr, W_lr] Y域残差
            rz: [B, Bm, H, W] Z域残差
            prev_theta_state: 前一步的Theta状态（首次为None）

        Returns:
            new_theta_state: 新的Theta状态
            theta_feat: [B, feat_dim] 特征向量
        """
        B, Bh, H_lr, W_lr = y_lr.shape
        _, Bm, H, W = z_hr.shape

        # Upsample Y to match Z resolution
        y_lr_up = F.interpolate(y_lr, size=(H, W), mode='bilinear', align_corners=False)
        ry_up = F.interpolate(ry, size=(H, W), mode='bilinear', align_corners=False)

        # Encode inputs
        feat_y = self.enc_y(y_lr_up)     # [B, map_dim, H, W]
        feat_z = self.enc_z(z_hr)        # [B, map_dim, H, W]
        feat_ry = self.enc_ry(ry_up)    # [B, map_dim, H, W]
        feat_rz = self.enc_rz(rz)       # [B, map_dim, H, W]

        # Fuse
        feat_cat = torch.cat([feat_y, feat_z, feat_ry, feat_rz], dim=1)  # [B, map_dim*4, H, W]
        feat_fused = self.fusion(feat_cat)  # [B, map_dim, H, W]

        # ===== Head: Keystone Flow =====
        keystone_raw = self.head_keystone(feat_fused)  # [B, Bh*2, H, W]
        keystone_raw = keystone_raw.view(B, Bh, 2, H, W)  # [B, Bh, 2, H, W]

        # Apply Tanh constraint
        delta_keystone = self._apply_keystone_constraint(keystone_raw)

        # Incremental update
        if prev_theta_state is not None:
            keystone_flow = prev_theta_state.keystone_flow + delta_keystone
            # Re-constrain after update
            keystone_flow = self._apply_keystone_constraint(keystone_flow)
        else:
            keystone_flow = delta_keystone

        # ===== Head: Smile Shift =====
        # Pool height dimension: [B, map_dim, H, W] -> [B, map_dim, W]
        feat_1d = self.head_smile_pool(feat_fused).squeeze(2)  # [B, map_dim, W]

        smile_raw = self.head_smile(feat_1d)  # [B, Bh, W]

        # Apply Tanh constraint
        delta_smile = self._apply_smile_constraint(smile_raw)

        # Incremental update
        if prev_theta_state is not None:
            smile_shift = prev_theta_state.smile_shift + delta_smile
            # Re-constrain
            smile_shift = self._apply_smile_constraint(smile_shift)
        else:
            smile_shift = delta_smile

        # ===== Head: Global Feature =====
        feat_global = self.global_pool(feat_fused).flatten(1)  # [B, map_dim]
        theta_feat = self.head_feat(feat_global)  # [B, feat_dim]

        # ===== Optional: PSF Coefficients =====
        psf_coef = self.head_psf(feat_global)  # [B, Bh*psf_k]
        psf_coef = psf_coef.view(B, Bh, self.psf_k)  # [B, Bh, psf_k]
        psf_coef = F.softplus(psf_coef)  # Ensure positive

        # Construct new state
        new_theta_state = ThetaState(
            keystone_flow=keystone_flow,
            smile_shift=smile_shift,
            psf_coef=psf_coef,
        )

        return new_theta_state, theta_feat


def init_theta_state(
    batch_size: int,
    bh: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> ThetaState:
    """
    初始化Theta状态为零畸变（Identity）

    按照修正.md：初始化为零畸变而非随机值
    """
    keystone_flow = torch.zeros(batch_size, bh, 2, height, width, device=device, dtype=dtype)
    smile_shift = torch.zeros(batch_size, bh, width, device=device, dtype=dtype)

    return ThetaState(
        keystone_flow=keystone_flow,
        smile_shift=smile_shift,
        psf_coef=None,
    )


__all__ = [
    "ThetaAdaptiveCore",
    "ThetaState",
    "init_theta_state",
]
