"""
Spe 分支核心：逐像素谱建模，输出 LR 网格谱特征 F_spe。

实现要点：
- 禁用 Conv2d/空间注意力/grid_sample 等空间算子
- Bh 先压到 K 的短序列，再跑轻量 Mamba
- CondMLP 提供条件，FiLM 调制来自 u_spe 与 cond
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .spectral_blocks_lightmamba import SpectralCompressor, SpeTokenEmbed, SpeMambaStack, SpeReadout
from .cond_encoder import CondMLP
from models.utils.resample import anti_alias_downsample


class SpeCore(nn.Module):
    def __init__(
        self,
        bh: int,
        bm: int,
        c_spe: int = 64,
        d: int = 64,
        scale: int = 4,
        spe_k_mode: str = "cap",
        spe_k_max: int = 32,
        spe_mamba_layers: int = 1,
        spe_mamba_d_state: int = 16,
        spe_mamba_d_conv: int = 3,
        spe_compress_mode: str = "segment_mean",
        spe_tokenize_mode: str = "segment_mean",
        spe_patch_stride: int = 2,
        spe_patch_kernel: int = 4,
        spe_patch_padding: int = 1,
        spe_chunk_bn: int = 0,
        spe_readout_heads: int = 4,
    ):
        super().__init__()
        self.scale = scale
        if spe_tokenize_mode in ("patch_conv1d", "conv1"):
            self.tokenize_mode = "conv1"
        else:
            self.tokenize_mode = spe_tokenize_mode
        if self.tokenize_mode not in ("segment_mean", "conv1"):
            raise ValueError(f"spe_tokenize_mode must be 'segment_mean' or 'conv1', got {self.tokenize_mode}")
        self.spe_chunk_bn = spe_chunk_bn
        self.spe_k_mode = spe_k_mode
        if spe_compress_mode == "none":
            k_eff = bh
        elif spe_k_mode == "full":
            k_eff = bh
        else:
            k_cap = bh if spe_k_max <= 0 else spe_k_max
            k_eff = min(bh, max(1, k_cap))
        self.k_eff = k_eff
        self.readout_heads = max(1, spe_readout_heads)
        self.compressor = SpectralCompressor(k=k_eff, mode=spe_compress_mode, bh=bh)
        self.token_embed = SpeTokenEmbed(k=k_eff, d_s=d)
        self.patch_conv = None
        if self.tokenize_mode == "conv1":
            self.patch_conv = nn.Conv1d(
                1,
                d,
                kernel_size=spe_patch_kernel,
                stride=spe_patch_stride,
                padding=spe_patch_padding,
            )
        self.token_norm = nn.LayerNorm(d)
        self.film = nn.Linear(130, 2 * d)
        self.mamba = SpeMambaStack(d_s=d, layers=spe_mamba_layers, d_state=spe_mamba_d_state, d_conv=spe_mamba_d_conv)
        self.readout = SpeReadout(d_s=d, heads=self.readout_heads)
        self.out_proj = nn.Linear(self.readout_heads * d, c_spe)
        self.cond_mlp = CondMLP(bm=bm, out_dim=64)
        self._assert_no_spatial_ops()

    def __setattr__(self, name, value):
        banned = (nn.Conv2d, nn.ConvTranspose2d, nn.AvgPool2d, nn.MaxPool2d, nn.MultiheadAttention)
        if isinstance(value, banned):
            raise RuntimeError(f"SpeCore contains banned spatial op: {value.__class__.__name__} at attribute {name}")
        super().__setattr__(name, value)

    def _assert_no_spatial_ops(self):
        banned = (nn.Conv2d, nn.ConvTranspose2d, nn.AvgPool2d, nn.MaxPool2d, nn.MultiheadAttention, nn.Upsample, nn.Unfold)
        for name, module in self.named_modules():
            if isinstance(module, banned):
                raise RuntimeError(f"SpeCore contains banned spatial op: {module.__class__.__name__} at {name}")

    def forward(self, y: torch.Tensor, z: torch.Tensor, ry: torch.Tensor, rz: torch.Tensor, u_spe: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y: [B,Bh,h,w]
            z: [B,Bm,H,W]
            ry, rz: 残差（未直接使用，可用于扩展）
            u_spe: [B,64]
        Returns:
            f_spe: [B,C_spe,h,w]
        """
        b, bh, h, w = y.shape
        _, _, H, W = z.shape
        s = H // h
        z_lr = anti_alias_downsample(z, s)
        cond = self.cond_mlp(z_lr)  # [B,64,h,w]

        # tokenization
        y_flat = y.permute(0, 2, 3, 1).reshape(-1, bh)
        if self.tokenize_mode == "conv1":
            if self.patch_conv is None:
                raise RuntimeError("SpeCore conv1 tokenization requires patch_conv to be initialized.")
            tokens = self.patch_conv(y_flat.unsqueeze(1)).transpose(1, 2)
        else:
            q = self.compressor(y_flat)
            tokens = self.token_embed(q)
        self.k_eff = tokens.shape[1]

        # FiLM 调制（含残差引导）
        cond_flat = cond.permute(0, 2, 3, 1).reshape(-1, cond.shape[1])
        ry_mag = ry.abs().mean(dim=1)
        rz_lr = anti_alias_downsample(rz, s)
        rz_mag = rz_lr.abs().mean(dim=1)
        res_flat = torch.stack([ry_mag, rz_mag], dim=-1).reshape(-1, 2)
        u_rep = u_spe[:, None, :].repeat(1, h * w, 1).reshape(-1, u_spe.shape[1])
        mod = torch.cat([u_rep, cond_flat, res_flat], dim=-1)
        gamma, beta = self.film(mod).chunk(2, dim=-1)
        tokens = self.token_norm(tokens)
        tokens = (1 + gamma.unsqueeze(1)) * tokens + beta.unsqueeze(1)

        if self.spe_chunk_bn and tokens.shape[0] > self.spe_chunk_bn:
            chunks = torch.split(tokens, self.spe_chunk_bn, dim=0)
            out_chunks = []
            for chunk in chunks:
                out_chunks.append(self.mamba(chunk))
            tokens = torch.cat(out_chunks, dim=0)
        else:
            tokens = self.mamba(tokens)

        feat = self.readout(tokens)
        out = self.out_proj(feat.reshape(feat.shape[0], -1))
        out = out.view(b, h, w, -1).permute(0, 3, 1, 2)
        return out
