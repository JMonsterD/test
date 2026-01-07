"""BlindHSIModel (MainNet): assemble modules and run DEQ implicit training."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from models.context import ContextEncoderLite
from models.branches import SpaCore, SpeCore, ThetaCore, HeadSpa, HeadSpe, HeadDC
from models.fusion_core import FusionCoreNSSMamba
from models.fusion.cross_modal import EncX, EncY, EncZ, CrossAttn
from models.dc.mismatch_adapter import MeasMismatchAdapter
from models.align.residual_guided import ResidualGuidedAlign
from models.theta.theta_idr import FeatureQueue, ThetaEMA
from models.operators import SRFSmile
from models.config import SpaConfig, SpeConfig, ThetaConfig, FusionConfig
from models.step import init_state, step_fn
from models.deq_implicit import DEQImplicit, fixed_point_iteration


class MainNet(nn.Module):
    def __init__(
        self,
        bh: int,
        bm: int,
        scale: int = 4,
        spa_cfg: SpaConfig | None = None,
        spe_cfg: SpeConfig | None = None,
        theta_cfg: ThetaConfig | None = None,
        fusion_cfg: FusionConfig | None = None,
        head_last_zero_init: bool = True,
        prior_dx_clip_scale: float = 0.0,
    ) -> None:
        super().__init__()
        spa_cfg = spa_cfg or SpaConfig()
        spe_cfg = spe_cfg or SpeConfig()
        theta_cfg = theta_cfg or ThetaConfig()
        fusion_cfg = fusion_cfg or FusionConfig()

        self.scale = scale
        self.context = ContextEncoderLite(theta_dim=theta_cfg.feat_dim)
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
        self.srf_smile = SRFSmile(bm=bm, bh=bh)
        self.theta = ThetaCore(
            bh=bh,
            bm=bm,
            hidden=theta_cfg.hidden,
            feat_dim=theta_cfg.feat_dim,
            map_dim=theta_cfg.map_dim,
            warp_ctrl_hw=theta_cfg.warp_ctrl_hw,
            warp_poly_order=theta_cfg.warp_poly_order,
            psf_k=theta_cfg.psf_k,
            smile_ctrl_w=theta_cfg.smile_ctrl_w,
        )
        self.theta_ema = ThetaEMA(self.theta)
        self.theta_queue = FeatureQueue(q_size=64)
        self.fusion = FusionCoreNSSMamba(
            bh=bh,
            d_model=fusion_cfg.d_model,
            mamba_layers=fusion_cfg.mamba_layers,
            d_state=fusion_cfg.mamba_d_state,
            d_conv=fusion_cfg.mamba_d_conv,
            expand=fusion_cfg.mamba_expand,
            theta_dim=theta_cfg.feat_dim,
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
        cond_dim = theta_cfg.feat_dim + fusion_cfg.mem_dim
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
        self.head_dc = HeadDC(scale=scale, srf_smile=self.srf_smile)
        self._last_deq: DEQImplicit | None = None

    def forward(
        self,
        y_lr: torch.Tensor,
        z_hr: torch.Tensor,
        *,
        ctx: Dict | None = None,
        ablations: Dict | None = None,
    ) -> Dict:
        defaults = {
            "steps": 25,
            "scale": self.scale,
            "deq_tol": 1e-4,
            "fwd_check_every": 1,
            "fwd_min_iter": 5,
            "stagnation_check_start": 20,
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
        if str(params.get("bwd_mode", "implicit")) != "implicit":
            raise RuntimeError("MainNet only supports implicit backward")

        steps = int(params.get("steps", 25))
        deq_tol = float(params.get("deq_tol", 1e-4))

        theta_hat, theta_feat = self.theta.blackbox(
            y_lr,
            z_hr,
            scale=self.scale,
            srf_smile=self.srf_smile,
        )
        x0 = init_state(y_lr, scale=self.scale)

        modules = {
            "context": self.context,
            "mismatch_adapter": self.mismatch_adapter,
            "spa": self.spa,
            "spe": self.spe,
            "fusion": self.fusion,
            "srf_smile": self.srf_smile,
            "head_spa": self.head_spa,
            "head_spe": self.head_spe,
            "head_dc": self.head_dc,
            "rg_align": self.rg_align,
            "enc_x": self.enc_x,
            "enc_y": self.enc_y,
            "enc_z": self.enc_z,
            "cross_attn": self.cross_attn,
        }
        params_deq = dict(params)
        params_deq["scale"] = self.scale
        params_deq["collect_stats"] = False
        params_deq["fwd_check_every"] = int(params_deq.get("fwd_check_every", 1))
        params_deq["fwd_min_iter"] = int(params_deq.get("fwd_min_iter", 5))
        deq_stats = {
            "batch": y_lr.shape[0],
            "device": y_lr.device,
            "dtype": y_lr.dtype,
        }
        params_deq["deq_stats"] = deq_stats

        x_bundle = {
            "inputs": {"Y_lr": y_lr, "Z_hr": z_hr, "theta_hat": theta_hat, "theta_feat": theta_feat},
            "modules": modules,
            "params": params_deq,
            "deq_stats": deq_stats,
            "last_aux": None,
        }

        def _pcdeq_f_x(z_in: torch.Tensor, bundle: Dict) -> torch.Tensor:
            x_next, aux = step_fn(z_in, bundle["inputs"], bundle["modules"], bundle["params"])
            bundle["last_aux"] = aux
            return x_next

        deq = DEQImplicit(_pcdeq_f_x, solver=fixed_point_iteration, max_iter=steps, tol=deq_tol)
        x_star = deq(x_bundle, x0)
        self._last_deq = deq
        last_aux = x_bundle.get("last_aux") or {}

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
            "alpha_x_mean": _scalar(last_aux.get("alpha_x_mean", 0.0)),
            "beta_dc_mean": _scalar(last_aux.get("beta_dc_mean", 0.0)),
            "w_spa_mean": _scalar(last_aux.get("w_spa_mean", 0.0)),
            "w_spe_mean": _scalar(last_aux.get("w_spe_mean", 0.0)),
            "router_entropy_mean": _scalar(last_aux.get("router_entropy_mean", 0.0)),
            "dx_spa_over_dc": _scalar(last_aux.get("dx_spa_over_dc", 0.0)),
            "dx_spe_over_dc": _scalar(last_aux.get("dx_spe_over_dc", 0.0)),
        }

        return {
            "X": x_star,
            "theta": theta_hat,
            "theta_feat": theta_feat,
            "aux": last_aux,
            "solver": solver,
        }


BlindHSIModel = MainNet

__all__ = ["MainNet", "BlindHSIModel"]
