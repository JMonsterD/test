"""
Context encoder (Lite): controller + router without stateful memory.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ContextOut:
    u_spa: torch.Tensor
    u_spe: torch.Tensor
    alpha_x: torch.Tensor
    beta_dc: torch.Tensor
    gamma_rg: torch.Tensor
    w_spa: torch.Tensor
    w_spe: torch.Tensor
    gate_temp: torch.Tensor
    mem_feat: torch.Tensor
    router_entropy: torch.Tensor
    router_balance: torch.Tensor


class ContextEncoderLite(nn.Module):
    def __init__(self, theta_dim: int = 32):
        super().__init__()
        self.theta_dim = theta_dim
        self.stat_proj = nn.Linear(32 + theta_dim, 128)
        self.gate_proj = nn.Linear(128, 128)
        self.u_spa = nn.Linear(128, 64)
        self.u_spe = nn.Linear(128, 64)
        self.alpha_x_head = nn.Linear(128, 1)
        self.beta_dc_head = nn.Linear(128, 1)
        self.gamma_rg_head = nn.Linear(128, 1)
        self.mem_head = nn.Linear(128, 32)
        self.gate_temp_head = nn.Linear(128, 1)
        self.router_head = nn.Linear(128, 2)
        nn.init.zeros_(self.router_head.weight)
        nn.init.zeros_(self.router_head.bias)

    @staticmethod
    def _safe_std(t: torch.Tensor, dim, keepdim: bool = True) -> torch.Tensor:
        var = t.var(dim=dim, keepdim=keepdim, unbiased=False)
        return torch.sqrt(var + 1e-6)

    def _build_stats(self, ry: torch.Tensor, rz: torch.Tensor) -> torch.Tensor:
        def _stats(t: torch.Tensor) -> torch.Tensor:
            t_abs = t.abs()
            mean_abs = t_abs.mean(dim=(1, 2, 3))
            max_abs = t_abs.amax(dim=(1, 2, 3))
            mean_val = t.mean(dim=(1, 2, 3))
            std_val = self._safe_std(t.flatten(start_dim=1), dim=1, keepdim=False)
            return torch.stack([mean_abs, max_abs, mean_val, std_val], dim=1)

        s = torch.cat([_stats(ry), _stats(rz)], dim=1)  # [B,8]
        if s.shape[1] < 32:
            pad = torch.zeros(s.shape[0], 32 - s.shape[1], device=s.device, dtype=s.dtype)
            s = torch.cat([s, pad], dim=1)
        return s

    @staticmethod
    def _map_range(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        return lo + (hi - lo) * torch.sigmoid(x)

    def forward(self, ry: torch.Tensor, rz: torch.Tensor, theta_feat: torch.Tensor | None) -> ContextOut:
        stats = self._build_stats(ry, rz)
        if theta_feat is None:
            theta_feat = torch.zeros(stats.shape[0], self.theta_dim, device=stats.device, dtype=stats.dtype)
        feat = torch.cat([stats, theta_feat], dim=1)
        feat = self.stat_proj(feat)
        gate_feat = self.gate_proj(feat)
        u_spa = self.u_spa(gate_feat)
        u_spe = self.u_spe(gate_feat)
        alpha_x = self._map_range(self.alpha_x_head(gate_feat), 0.02, 0.15)
        beta_dc = self._map_range(self.beta_dc_head(gate_feat), 0.0, 0.70)
        gamma_rg = self._map_range(self.gamma_rg_head(gate_feat), 0.0, 0.5)
        mem_feat = self.mem_head(gate_feat)
        gate_temp = self._map_range(self.gate_temp_head(gate_feat), 0.5, 2.0)

        logits = self.router_head(gate_feat)
        temp = gate_temp.view(-1, 1).clamp(min=0.1)
        router_w = torch.softmax(logits / temp, dim=-1)
        w_spa = router_w[:, 0]
        w_spe = router_w[:, 1]
        router_entropy = -(router_w * torch.log(router_w + 1e-6)).sum(dim=1)
        router_balance = w_spa - 0.5

        return ContextOut(
            u_spa=u_spa,
            u_spe=u_spe,
            alpha_x=alpha_x,
            beta_dc=beta_dc,
            gamma_rg=gamma_rg,
            w_spa=w_spa,
            w_spe=w_spe,
            gate_temp=gate_temp,
            mem_feat=mem_feat,
            router_entropy=router_entropy,
            router_balance=router_balance,
        )


__all__ = ["ContextEncoderLite", "ContextOut"]
