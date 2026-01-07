"""
Config dataclasses for module and training schedules.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SpaConfig:
    c_spa: int = 64
    local_blocks: int = 1
    grid_down: int = 1
    grid_d: int = 64
    mamba_layers: int = 1
    mamba_d_state: int = 16
    mamba_d_conv: int = 3
    grid_L_max: int = 64
    grid_L_min: int = 4
    grid_min_hw: int = 2
    fuse_alpha_init: float = 0.1
    film: bool = True


@dataclass
class SpeConfig:
    c_spe: int = 64
    d: int = 64
    k_mode: str = "cap"
    k_max: int = 32
    readout_heads: int = 4
    mamba_layers: int = 1
    mamba_d_state: int = 16
    mamba_d_conv: int = 3
    compress_mode: str = "segment_mean"
    tokenize_mode: str = "segment_mean"
    patch_stride: int = 2
    patch_kernel: int = 4
    patch_padding: int = 1
    chunk_bn: int = 0


@dataclass
class ThetaConfig:
    hidden: int = 128
    feat_dim: int = 32
    map_dim: int = 64
    warp_ctrl_hw: int = 8
    warp_poly_order: int = 2
    psf_k: int = 8
    smile_ctrl_w: int = 16


@dataclass
class FusionConfig:
    d_model: int = 64
    mamba_layers: int = 2
    mamba_d_state: int = 16
    mamba_d_conv: int = 3
    mamba_expand: int = 2
    theta_dim: int = 32
    mem_dim: int = 32
    grid_hw: int = 8
    scan_len: int = 4
    map_channels: int = 2
    out_clip: float = 0.1


@dataclass
class DEQConfig:
    max_iter: int = 40
    tol: float = 1e-4
    fwd_min_iter: int = 5
    stagnation_check_start: int = 20
    fwd_soft_accept_res: float = 1e-3
    gmres_max_iter: int = 12
    gmres_tol: float = 1e-4
    gmres_early_stop: bool = True
    gmres_early_stop_min_iter: int = 4


@dataclass
class LossConfig:
    w_y: float = 1.0
    w_z: float = 1.0
    w_tv_spa: float = 0.02
    w_tv_spe: float = 0.01
    w_ent: float = 0.01
    w_bal: float = 0.01
    w_theta_idr: float = 0.05
    w_theta_phys: float = 0.10
    w_subspace: float = 0.02
    w_mismatch_reg: float = 0.02
    w_rg: float = 0.01
    w_edge: float = 0.0
    w_hp: float = 0.0
    w_edge_z: float = 0.0
    w_hp_z: float = 0.0
    w_cross: float = 0.0
    w_band_grad: float = 0.0
    w_sam_lr: float = 0.0
    w_tv_s: float = 0.0
    w_tv_b: float = 0.0
    w_theta: float = 0.0
    w_warp_smooth: float = 0.05
    w_warp_mag: float = 0.01
    w_smile_smooth: float = 0.02
    w_srf_ent: float = 0.001
    w_srf_smooth: float = 0.01
    w_psf_smooth: float = 0.01
    w_psf_center: float = 0.01


@dataclass
class StageSchedule:
    stage1_ratio: float = 0.40
    stage2_ratio: float = 0.60
