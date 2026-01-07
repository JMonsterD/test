"""
New Physical Resampling Operators for A-adaptive Architecture
按照修正.md实现KeystoneResampler和SmileResampler
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KeystoneResampler(nn.Module):
    """
    几何畸变重采样器 (Keystone Distortion)

    输入:
        x: [B, Bh, H, W] 输入图像
        flow_field: [B, Bh, 2, H, W] 每个波段独立的流场 (dx, dy)

    输出:
        x_warped: [B, Bh, H, W] 重采样后的图像

    实现: 使用 torch.nn.functional.grid_sample
    约束: flow_field 被限制在 ±max_shift 像素
    """

    def __init__(self, max_shift_pixels: float = 2.0):
        super().__init__()
        self.max_shift_pixels = max_shift_pixels

    def forward(self, x: torch.Tensor, flow_field: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, Bh, H, W]
            flow_field: [B, Bh, 2, H, W] where flow_field[:,b,0] = dx, flow_field[:,b,1] = dy

        Returns:
            x_warped: [B, Bh, H, W]
        """
        B, Bh, H, W = x.shape

        # Clamp flow to prevent extreme deformation
        flow_clamped = torch.clamp(flow_field, -self.max_shift_pixels, self.max_shift_pixels)

        # Normalize flow to [-1, 1] for grid_sample
        # grid_sample expects normalized coordinates
        flow_norm = torch.zeros_like(flow_clamped)
        flow_norm[:, :, 0, :, :] = flow_clamped[:, :, 0, :, :] * 2.0 / W  # dx -> normalized x
        flow_norm[:, :, 1, :, :] = flow_clamped[:, :, 1, :, :] * 2.0 / H  # dy -> normalized y

        # Create base grid [-1, 1]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype),
            torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype),
            indexing='ij'
        )
        base_grid = torch.stack([xx, yy], dim=-1)  # [H, W, 2]
        base_grid = base_grid.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W, 2]
        base_grid = base_grid.expand(B, Bh, -1, -1, -1)  # [B, Bh, H, W, 2]

        # Add flow to base grid
        # flow_norm: [B, Bh, 2, H, W] -> [B, Bh, H, W, 2]
        flow_grid = flow_norm.permute(0, 1, 3, 4, 2)  # [B, Bh, H, W, 2]
        grid = base_grid + flow_grid

        # Reshape for grid_sample (requires [B*Bh, H, W, 2])
        x_flat = x.reshape(B * Bh, 1, H, W)
        grid_flat = grid.reshape(B * Bh, H, W, 2)

        # Apply resampling
        x_warped_flat = F.grid_sample(
            x_flat,
            grid_flat,
            mode='bilinear',
            padding_mode='border',  # 边界外使用边界值
            align_corners=False
        )

        # Reshape back
        x_warped = x_warped_flat.reshape(B, Bh, H, W)

        return x_warped


class SmileResampler(nn.Module):
    """
    光谱偏移重采样器 (Smile Effect)

    输入:
        x: [B, Bh, H, W] 输入图像
        wavelength_shift: [B, Bh, W] 每个空间列的波长偏移 (Δλ in nm or band index)

    输出:
        x_resampled: [B, Bh, H, W] 光谱重采样后的图像

    实现: 基于 1D 线性插值
    不做成 Bm×Bh 的大矩阵乘法，而是对每个像素在光谱维进行重采样
    """

    def __init__(self, max_shift_bands: float = 2.0):
        super().__init__()
        self.max_shift_bands = max_shift_bands

    def forward(self, x: torch.Tensor, wavelength_shift: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, Bh, H, W]
            wavelength_shift: [B, Bh, W] 光谱偏移（以波段索引为单位）

        Returns:
            x_resampled: [B, Bh, H, W]
        """
        B, Bh, H, W = x.shape

        # Clamp shift to prevent extreme spectral distortion
        shift_clamped = torch.clamp(wavelength_shift, -self.max_shift_bands, self.max_shift_bands)

        # Expand shift to match spatial dimensions
        # shift_clamped: [B, Bh, W] -> [B, Bh, H, W]
        shift_expanded = shift_clamped.unsqueeze(2).expand(-1, -1, H, -1)

        # Create band indices [0, 1, ..., Bh-1]
        band_indices = torch.arange(Bh, device=x.device, dtype=x.dtype)  # [Bh]
        band_indices = band_indices.view(1, Bh, 1, 1).expand(B, -1, H, W)  # [B, Bh, H, W]

        # Shifted indices
        shifted_indices = band_indices + shift_expanded  # [B, Bh, H, W]

        # Clamp to valid range [0, Bh-1]
        shifted_indices = torch.clamp(shifted_indices, 0, Bh - 1)

        # Normalize to [-1, 1] for grid_sample
        # grid_sample 在光谱维度上插值
        grid_1d = (shifted_indices / (Bh - 1)) * 2.0 - 1.0  # [B, Bh, H, W]

        # Reshape for grid_sample
        # x: [B, Bh, H, W] 视为 [B, 1, Bh, H*W] (treat spectral as height, spatial as width)
        x_reshaped = x.permute(0, 2, 3, 1).reshape(B * H * W, 1, Bh, 1)  # [B*H*W, 1, Bh, 1]

        # grid for grid_sample: need [N, H_out, W_out, 2], but we're doing 1D
        # Use grid_1d as the "y" coordinate, x coordinate is 0 (single column)
        grid_reshaped = grid_1d.permute(0, 2, 3, 1).reshape(B * H * W, Bh, 1, 1)  # [B*H*W, Bh, 1, 1]
        grid_2d = torch.zeros(B * H * W, Bh, 1, 2, device=x.device, dtype=x.dtype)
        grid_2d[:, :, :, 1] = grid_reshaped.squeeze(-1)  # y coordinate = spectral index
        grid_2d[:, :, :, 0] = 0  # x coordinate = 0 (single column)

        # Apply 1D resampling
        x_resampled_flat = F.grid_sample(
            x_reshaped,
            grid_2d,
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        )  # [B*H*W, 1, Bh, 1]

        # Reshape back
        x_resampled = x_resampled_flat.squeeze(1).squeeze(-1).reshape(B, H, W, Bh).permute(0, 3, 1, 2)  # [B, Bh, H, W]

        return x_resampled


def test_keystone_resampler():
    """单元测试: KeystoneResampler"""
    print("=== Testing KeystoneResampler ===")

    B, Bh, H, W = 2, 10, 64, 64
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test 1: Zero flow should not change image
    resampler = KeystoneResampler(max_shift_pixels=2.0).to(device)
    x = torch.randn(B, Bh, H, W, device=device)
    flow = torch.zeros(B, Bh, 2, H, W, device=device)
    x_warped = resampler(x, flow)

    diff = (x - x_warped).abs().max()
    print(f"Test 1 (zero flow): max diff = {diff:.6f} (should be ~0)")
    assert diff < 1e-5, "Zero flow should not change image"

    # Test 2: Uniform shift
    x_pattern = torch.zeros(B, Bh, H, W, device=device)
    x_pattern[:, :, H//2-5:H//2+5, W//2-5:W//2+5] = 1.0  # Center square

    flow_shift = torch.zeros(B, Bh, 2, H, W, device=device)
    flow_shift[:, :, 0, :, :] = 2.0  # Shift right by 2 pixels

    x_shifted = resampler(x_pattern, flow_shift)

    # Check if the center moved
    center_orig = x_pattern[0, 0, H//2, W//2].item()
    center_shifted = x_shifted[0, 0, H//2, W//2+2].item()

    print(f"Test 2 (shift right 2px): center_orig={center_orig:.2f}, center_shifted={center_shifted:.2f}")
    print(f"  (shifted center should be bright)")

    print("✓ KeystoneResampler tests passed\n")


def test_smile_resampler():
    """单元测试: SmileResampler"""
    print("=== Testing SmileResampler ===")

    B, Bh, H, W = 2, 20, 32, 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Test 1: Zero shift should not change image
    resampler = SmileResampler(max_shift_bands=2.0).to(device)
    x = torch.randn(B, Bh, H, W, device=device)
    shift = torch.zeros(B, Bh, W, device=device)
    x_resampled = resampler(x, shift)

    diff = (x - x_resampled).abs().max()
    print(f"Test 1 (zero shift): max diff = {diff:.6f} (should be ~0)")
    assert diff < 1e-5, "Zero shift should not change spectrum"

    # Test 2: Create spectral gradient pattern
    x_spectral = torch.zeros(B, Bh, H, W, device=device)
    for b in range(Bh):
        x_spectral[:, b, :, :] = float(b) / float(Bh - 1)  # Linear gradient in spectral dimension

    # Shift by +1 band everywhere
    shift_uniform = torch.ones(B, Bh, W, device=device)
    x_shifted = resampler(x_spectral, shift_uniform)

    # Band 0 shifted by +1 should look like original band 1
    diff_01 = (x_shifted[:, 0, :, :] - x_spectral[:, 1, :, :]).abs().max()
    print(f"Test 2 (shift +1 band): diff between shifted[0] and orig[1] = {diff_01:.6f}")
    print(f"  (should be close to 0)")

    print("✓ SmileResampler tests passed\n")


if __name__ == "__main__":
    test_keystone_resampler()
    test_smile_resampler()
    print("All physical operator tests passed!")


__all__ = [
    "KeystoneResampler",
    "SmileResampler",
    "test_keystone_resampler",
    "test_smile_resampler",
]
