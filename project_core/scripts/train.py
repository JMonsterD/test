"""
Training entry: implicit-only DEQ, logging via LogBus.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from data.datasets.synthetic import SyntheticDataset
from data.datasets.pavia import PaviaDataset
from data.dataloader import build_dataloader
from models.config import SpaConfig, SpeConfig, ThetaConfig, FusionConfig, DEQConfig, LossConfig
from models.model import MainNet
from models.operators import (
    ay_forward,
    az_forward,
    build_psf_kernels,
    build_keystone_flow,
    upsample_smile_params,
)
from models.theta.theta_idr import IDRViewsAugment, VICRegLoss
from models.utils.freq import tv_spatial, tv_band
from eval.metrics import psnr, ssim, sam, ergas, qnr
from utils.logbus import LogBus


def _add_args(parser: argparse.ArgumentParser, specs: Dict[str, Dict]) -> None:
    for name, spec in specs.items():
        parser.add_argument(f"--{name}", **spec)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    _add_args(
        parser,
        {
            "dataset": {"type": str, "default": "pavia", "choices": ["synthetic", "pavia"]},
            "data_path": {"type": str, "default": "datasets/pavia_data_r8_30_40.mat"},
            "scale": {"type": int, "default": 8},
            "patch_size": {"type": int, "default": 96},
            "patches_per_epoch": {"type": int, "default": 100},
            "epochs": {"type": int, "default": 50},
            "train_batch_size": {"type": int, "default": 8},
            "eval_interval": {"type": int, "default": 10},
            "eval_steps": {"type": int, "default": -1},
            "debug_deq_interval": {"type": int, "default": 0},
            "outdir": {"type": str, "default": "outputs"},
            "log_file": {"type": str, "default": "train.log"},
            "bh": {"type": int, "default": 103},
            "bm": {"type": int, "default": 4},
        },
    )
    _add_args(
        parser,
        {
            "spa_c": {"type": int, "default": 64},
            "spe_c": {"type": int, "default": 64},
            "spe_d": {"type": int, "default": 64},
            "spe_k_max": {"type": int, "default": 32},
            "spe_tokenize_mode": {"type": str, "default": "segment_mean"},
            "spe_patch_stride": {"type": int, "default": 2},
            "spe_patch_kernel": {"type": int, "default": 4},
            "spe_patch_padding": {"type": int, "default": 1},
            "theta_hidden": {"type": int, "default": 128},
            "theta_feat_dim": {"type": int, "default": 32},
            "fusion_d_model": {"type": int, "default": 64},
            "cross_grid": {"type": int, "default": 16},
        },
    )
    _add_args(
        parser,
        {
            "deq_max_iter": {"type": int, "default": 40},
            "deq_tol": {"type": float, "default": 1e-4},
            "fwd_min_iter": {"type": int, "default": 5},
            "fwd_check_every": {"type": int, "default": 1},
            "stagnation_check_start": {"type": int, "default": 20},
            "fwd_soft_accept_res": {"type": float, "default": 1e-3},
            "gmres_max_iter": {"type": int, "default": 12},
            "gmres_tol": {"type": float, "default": 1e-4},
            "gmres_early_stop": {"type": int, "default": 1},
            "gmres_early_stop_min_iter": {"type": int, "default": 4},
        },
    )
    _add_args(
        parser,
        {
            "w_y": {"type": float, "default": 1.0},
            "w_z": {"type": float, "default": 1.0},
            "w_tv_spa": {"type": float, "default": 0.02},
            "w_tv_spe": {"type": float, "default": 0.01},
            "w_ent": {"type": float, "default": 0.01},
            "w_bal": {"type": float, "default": 0.01},
            "w_theta_idr": {"type": float, "default": 0.05},
            "w_theta_phys": {"type": float, "default": 0.10},
            "w_subspace": {"type": float, "default": 0.02},
            "w_mismatch_reg": {"type": float, "default": 0.02},
            "w_rg": {"type": float, "default": 0.01},
        },
    )
    return parser.parse_args()


def _tv2d(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] < 2 or x.shape[-2] < 2:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    dh = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
    dw = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
    return dh + dw


def _tv1d(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] < 2:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    return (x[..., 1:] - x[..., :-1]).abs().mean()


def _segment_mean(x: torch.Tensor, k: int = 32) -> torch.Tensor:
    b, bh, h, w = x.shape
    k = min(int(k), bh) if k > 0 else bh
    if k <= 0:
        return x
    if bh % k == 0:
        return x.view(b, k, bh // k, h, w).mean(dim=2)
    pad = (k - (bh % k)) % k
    if pad > 0:
        x = F.pad(x, (0, 0, 0, 0, 0, pad), mode="constant", value=0.0)
    bh_pad = x.shape[1]
    return x.view(b, k, bh_pad // k, h, w).mean(dim=2)


def _subspace_recon(x: torch.Tensor, k: int = 32) -> torch.Tensor:
    b, bh, h, w = x.shape
    k = min(int(k), bh) if k > 0 else bh
    if k <= 0:
        return x
    seg = _segment_mean(x, k)
    seg_len = int((bh + k - 1) // k)
    recon = seg.unsqueeze(2).repeat(1, 1, seg_len, 1, 1).reshape(b, k * seg_len, h, w)
    return recon[:, :bh]


def compute_loss(
    x_hat: torch.Tensor,
    theta_hat: Dict[str, torch.Tensor],
    y_lr: torch.Tensor,
    z_hr: torch.Tensor,
    *,
    loss_cfg: LossConfig,
    scale: int,
    srf_smile,
    aux: Dict | None = None,
    theta_idr: torch.Tensor | None = None,
    idr_parts: Dict[str, torch.Tensor] | None = None,
    theta_kd: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if aux is not None and aux.get("y_hat") is not None and aux.get("z_hat") is not None:
        y_hat = aux["y_hat"]
        z_hat = aux["z_hat"]
    else:
        y_hat = ay_forward(x_hat, theta_hat, scale=scale)
        z_hat = az_forward(x_hat, theta_hat, scale=scale, srf_smile=srf_smile)

    L_y = (y_hat - y_lr).abs().mean()
    L_z = (z_hat - z_hr).abs().mean()
    loss = loss_cfg.w_y * L_y + loss_cfg.w_z * L_z
    comps: Dict[str, float] = {"Ly": float(L_y.item()), "Lz": float(L_z.item())}

    if loss_cfg.w_tv_spa > 0:
        L_tv_spa = tv_spatial(x_hat)
        loss = loss + loss_cfg.w_tv_spa * L_tv_spa
        comps["L_tv_spa"] = float(L_tv_spa.item())

    if loss_cfg.w_tv_spe > 0:
        L_tv_spe = tv_band(x_hat)
        loss = loss + loss_cfg.w_tv_spe * L_tv_spe
        comps["L_tv_spe"] = float(L_tv_spe.item())

    if aux is not None:
        w_spa = aux.get("router_w_spa")
        w_spe = aux.get("router_w_spe")
        if w_spa is not None and w_spe is not None and loss_cfg.w_ent > 0:
            w_stack = torch.stack([w_spa, w_spe], dim=-1)
            L_ent = (w_stack * torch.log(w_stack + 1e-6)).sum(dim=-1).mean()
            loss = loss + loss_cfg.w_ent * L_ent
            comps["L_ent"] = float(L_ent.item())
        if w_spa is not None and loss_cfg.w_bal > 0:
            L_bal = (w_spa.mean() - 0.5).pow(2)
            loss = loss + loss_cfg.w_bal * L_bal
            comps["L_bal"] = float(L_bal.item())

        dy = aux.get("dy")
        dz = aux.get("dz")
        if dy is not None and dz is not None and loss_cfg.w_mismatch_reg > 0:
            L_mismatch = dy.pow(2).mean() + dz.pow(2).mean()
            loss = loss + loss_cfg.w_mismatch_reg * L_mismatch
            comps["L_mismatch_reg"] = float(L_mismatch.item())

        dX_rg = aux.get("dX_rg")
        if dX_rg is not None and loss_cfg.w_rg > 0:
            L_rg = dX_rg.pow(2).mean()
            loss = loss + loss_cfg.w_rg * L_rg
            comps["L_rg"] = float(L_rg.item())

    if loss_cfg.w_subspace > 0:
        x_rec = _subspace_recon(x_hat, k=32)
        L_subspace = (x_hat - x_rec).abs().mean()
        loss = loss + loss_cfg.w_subspace * L_subspace
        comps["L_subspace"] = float(L_subspace.item())

    if loss_cfg.w_theta_phys > 0:
        psf_k = build_psf_kernels(theta_hat)
        if psf_k is None:
            L_psf_sum = x_hat.new_tensor(0.0)
            L_psf_center = x_hat.new_tensor(0.0)
            L_psf_smooth = x_hat.new_tensor(0.0)
        else:
            k = psf_k.squeeze(2)
            k_sum = k.sum(dim=(-1, -2))
            L_psf_sum = (k_sum - 1.0).pow(2).mean()
            ks = k.shape[-1]
            coord = torch.arange(ks, device=k.device, dtype=k.dtype)
            yy, xx = torch.meshgrid(coord, coord, indexing="ij")
            mass = k_sum + 1e-6
            com_x = (k * xx).sum(dim=(-1, -2)) / mass
            com_y = (k * yy).sum(dim=(-1, -2)) / mass
            center = (ks - 1) / 2.0
            L_psf_center = ((com_x - center).pow(2) + (com_y - center).pow(2)).mean()
            L_psf_smooth = _tv2d(k)

        flow = None
        if theta_hat.get("warp_ctrl") is not None and theta_hat.get("psf_coef") is not None:
            flow = build_keystone_flow(theta_hat, scale=scale, out_hw=(x_hat.shape[2], x_hat.shape[3]))
        if flow is None:
            L_warp_smooth = x_hat.new_tensor(0.0)
            L_warp_mag = x_hat.new_tensor(0.0)
        else:
            L_warp_smooth = _tv2d(flow)
            L_warp_mag = flow.pow(2).mean()
        delta_lr = theta_hat.get("delta_lr")
        L_delta_mag = delta_lr.pow(2).mean() if delta_lr is not None else x_hat.new_tensor(0.0)

        smile_shift, smile_temp = upsample_smile_params(theta_hat, width=z_hr.shape[-1], bm=z_hr.shape[1])
        L_smile_smooth = _tv1d(smile_shift) + _tv1d(smile_temp)
        L_smile_mag = smile_shift.pow(2).mean()

        L_theta_phys = (
            L_psf_sum
            + L_psf_center
            + L_psf_smooth
            + L_warp_smooth
            + L_warp_mag
            + L_delta_mag
            + L_smile_smooth
            + L_smile_mag
        )
        loss = loss + loss_cfg.w_theta_phys * L_theta_phys
        comps.update(
            {
                "L_theta_phys": float(L_theta_phys.item()),
                "L_psf_sum": float(L_psf_sum.item()),
                "L_psf_center": float(L_psf_center.item()),
                "L_psf_smooth": float(L_psf_smooth.item()),
                "L_warp_smooth": float(L_warp_smooth.item()),
                "L_warp_mag": float(L_warp_mag.item()),
                "L_delta_mag": float(L_delta_mag.item()),
                "L_smile_smooth": float(L_smile_smooth.item()),
                "L_smile_mag": float(L_smile_mag.item()),
            }
        )

    if theta_idr is not None and loss_cfg.w_theta_idr > 0:
        loss = loss + loss_cfg.w_theta_idr * theta_idr
        comps["L_theta_idr"] = float(theta_idr.item())
    if idr_parts is not None:
        for k, v in idr_parts.items():
            comps[k] = float(v.item())
    if theta_kd is not None:
        comps["L_theta_kd"] = float(theta_kd.item())

    return loss, comps


def _build_configs(args: argparse.Namespace) -> Tuple[SpaConfig, SpeConfig, ThetaConfig, FusionConfig, DEQConfig, LossConfig]:
    spa_cfg = SpaConfig(c_spa=args.spa_c)
    spe_cfg = SpeConfig(
        c_spe=args.spe_c,
        d=args.spe_d,
        k_max=args.spe_k_max,
        tokenize_mode=args.spe_tokenize_mode,
        patch_stride=args.spe_patch_stride,
        patch_kernel=args.spe_patch_kernel,
        patch_padding=args.spe_patch_padding,
    )
    theta_cfg = ThetaConfig(hidden=args.theta_hidden, feat_dim=args.theta_feat_dim)
    fusion_cfg = FusionConfig(d_model=args.fusion_d_model)
    deq_cfg = DEQConfig(
        max_iter=args.deq_max_iter,
        tol=args.deq_tol,
        fwd_min_iter=args.fwd_min_iter,
        stagnation_check_start=args.stagnation_check_start,
        fwd_soft_accept_res=args.fwd_soft_accept_res,
        gmres_max_iter=args.gmres_max_iter,
        gmres_tol=args.gmres_tol,
        gmres_early_stop=bool(args.gmres_early_stop),
        gmres_early_stop_min_iter=args.gmres_early_stop_min_iter,
    )
    loss_cfg = LossConfig(
        w_y=args.w_y,
        w_z=args.w_z,
        w_tv_spa=args.w_tv_spa,
        w_tv_spe=args.w_tv_spe,
        w_ent=args.w_ent,
        w_bal=args.w_bal,
        w_theta_idr=args.w_theta_idr,
        w_theta_phys=args.w_theta_phys,
        w_subspace=args.w_subspace,
        w_mismatch_reg=args.w_mismatch_reg,
        w_rg=args.w_rg,
    )
    return spa_cfg, spe_cfg, theta_cfg, fusion_cfg, deq_cfg, loss_cfg


def _build_ctx_params(args: argparse.Namespace, deq_cfg: DEQConfig) -> Dict[str, object]:
    return {
        "steps": int(deq_cfg.max_iter),
        "scale": int(args.scale),
        "deq_tol": float(deq_cfg.tol),
        "fwd_check_every": int(args.fwd_check_every),
        "fwd_min_iter": int(deq_cfg.fwd_min_iter),
        "stagnation_check_start": int(deq_cfg.stagnation_check_start),
        "fwd_soft_accept_res": float(deq_cfg.fwd_soft_accept_res),
        "gmres_max_iter": int(deq_cfg.gmres_max_iter),
        "gmres_tol": float(deq_cfg.gmres_tol),
        "gmres_early_stop": bool(deq_cfg.gmres_early_stop),
        "gmres_early_stop_min_iter": int(deq_cfg.gmres_early_stop_min_iter),
        "cross_grid": int(args.cross_grid),
        "bwd_mode": "implicit",
    }


def _mean_update(acc: Dict[str, float], vals: Dict[str, object]) -> None:
    for k, v in vals.items():
        try:
            acc[k] = acc.get(k, 0.0) + float(v)
        except (TypeError, ValueError):
            continue


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spa_cfg, spe_cfg, theta_cfg, fusion_cfg, deq_cfg, loss_cfg = _build_configs(args)

    if args.dataset == "synthetic":
        dataset = SyntheticDataset(
            num_samples=args.patches_per_epoch,
            bh=args.bh,
            bm=args.bm,
            h=args.patch_size // args.scale,
            w=args.patch_size // args.scale,
            scale=args.scale,
        )
        eval_dataset = dataset
    else:
        dataset = PaviaDataset(
            mat_path=args.data_path,
            patch_size=args.patch_size,
            patches_per_epoch=args.patches_per_epoch,
            scale=args.scale,
            return_gt=True,
        )
        eval_dataset = dataset

    loader = build_dataloader(dataset, batch_size=args.train_batch_size, shuffle=True)

    model = MainNet(
        bh=args.bh,
        bm=args.bm,
        scale=args.scale,
        spa_cfg=spa_cfg,
        spe_cfg=spe_cfg,
        theta_cfg=theta_cfg,
        fusion_cfg=fusion_cfg,
    ).to(device)
    opt = AdamW(model.parameters(), lr=2e-4)

    idr_aug = IDRViewsAugment()
    idr_loss = VICRegLoss()

    logbus = LogBus(args.outdir, args.log_file)
    ctx_params = _build_ctx_params(args, deq_cfg)
    global_step = 0
    diag_interval = 20

    for epoch in range(args.epochs):
        model.train()
        start = time.time()
        loss_sum = 0.0
        n = 0
        solver_sum: Dict[str, float] = {}
        loss_sum_parts: Dict[str, float] = {}
        solver_last: Dict[str, object] = {}
        for batch in loader:
            y_lr = batch["Y_lr"].to(device)
            z_hr = batch["Z_hr"].to(device)
            out = model(y_lr, z_hr, ctx=ctx_params)
            aux = out.get("aux") or {}

            (y1, z1), (y2, z2) = idr_aug(y_lr, z_hr)
            theta_feat1 = out["theta_feat"]
            _, theta_feat2 = model.theta.blackbox(y2, z2, scale=args.scale, srf_smile=model.srf_smile)
            L_idr, idr_parts = idr_loss(theta_feat1, theta_feat2, feature_queue=model.theta_queue)
            with torch.no_grad():
                _, theta_feat_ema = model.theta_ema.blackbox(y1, z1, scale=args.scale, srf_smile=model.srf_smile)
            L_kd = (theta_feat2 - theta_feat_ema).pow(2).mean()
            L_theta_idr = L_idr + 0.5 * L_kd
            theta_feat_std_mean = theta_feat1.std(dim=0, unbiased=False).mean()

            loss, comps = compute_loss(
                out["X"],
                out["theta"],
                y_lr,
                z_hr,
                loss_cfg=loss_cfg,
                scale=args.scale,
                srf_smile=model.srf_smile,
                aux=aux,
                theta_idr=L_theta_idr,
                idr_parts=idr_parts,
                theta_kd=L_kd,
            )
            comps["theta_feat_std_mean"] = float(theta_feat_std_mean.item())

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            model.theta_queue.update(theta_feat_ema)
            model.theta_ema.update(model.theta)

            loss_sum += float(loss.item())
            n += 1
            _mean_update(loss_sum_parts, comps)

            solver = dict(out.get("solver", {}) or {})
            last_deq = getattr(model, "_last_deq", None)
            if last_deq is not None:
                solver["theta_gmres_iters"] = int(last_deq.iter_backward)
                solver["theta_gmres_relres"] = float(last_deq.res_backward[-1]) if last_deq.res_backward else 0.0
                solver["theta_gmres_fail_reason"] = last_deq.fail_reason_backward
            solver_last = solver
            _mean_update(solver_sum, solver)

            if diag_interval > 0 and global_step % diag_interval == 0:
                y_hat = aux.get("y_hat")
                z_hat = aux.get("z_hat")
                if y_hat is None or z_hat is None:
                    y_hat = ay_forward(out["X"], out["theta"], scale=args.scale)
                    z_hat = az_forward(out["X"], out["theta"], scale=args.scale, srf_smile=model.srf_smile)
                x_hat = out["X"]
                x_gt = batch.get("X_gt")
                mse_x = None
                if x_gt is not None:
                    x_gt = x_gt.to(device)
                    mse_x = (x_hat - x_gt).pow(2).mean()
                record = {
                    "type": "diag",
                    "epoch": epoch,
                    "step": global_step,
                    "x_hat_min": float(x_hat.min().item()),
                    "x_hat_max": float(x_hat.max().item()),
                    "x_hat_mean": float(x_hat.mean().item()),
                    "y_hat_min": float(y_hat.min().item()),
                    "y_hat_max": float(y_hat.max().item()),
                    "y_hat_mean": float(y_hat.mean().item()),
                    "y_lr_min": float(y_lr.min().item()),
                    "y_lr_max": float(y_lr.max().item()),
                    "y_lr_mean": float(y_lr.mean().item()),
                    "z_hat_min": float(z_hat.min().item()),
                    "z_hat_max": float(z_hat.max().item()),
                    "z_hat_mean": float(z_hat.mean().item()),
                    "z_hr_min": float(z_hr.min().item()),
                    "z_hr_max": float(z_hr.max().item()),
                    "z_hr_mean": float(z_hr.mean().item()),
                    "pcdeq_stop_reason": solver.get("pcdeq_stop_reason"),
                    "pcdeq_iter_fwd_last": solver.get("pcdeq_iter_fwd_last"),
                    "pcdeq_res_fwd_last": solver.get("pcdeq_res_fwd_last"),
                    "ry_mean": aux.get("ry_mean"),
                    "ry_abs_mean": aux.get("ry_abs_mean"),
                    "rz_mean": aux.get("rz_mean"),
                    "rz_abs_mean": aux.get("rz_abs_mean"),
                }
                if mse_x is not None:
                    record["mse_x"] = float(mse_x.item())
                line = (
                    f"[diag] step {global_step} x_hat[{record['x_hat_min']:.3f},{record['x_hat_max']:.3f}] "
                    f"y_hat[{record['y_hat_min']:.3f},{record['y_hat_max']:.3f}] "
                    f"z_hat[{record['z_hat_min']:.3f},{record['z_hat_max']:.3f}]"
                )
                logbus.deq(record, line=line)

            if args.debug_deq_interval > 0 and global_step % args.debug_deq_interval == 0:
                record = {"type": "deq_iter", "epoch": epoch, "step": global_step}
                record.update(solver)
                logbus.deq(record)
            global_step += 1

        loss_epoch = loss_sum / max(1, n)
        loss_avg = {k: v / max(1, n) for k, v in loss_sum_parts.items()}
        solver_avg = {k: v / max(1, n) for k, v in solver_sum.items()}

        epoch_time = time.time() - start
        eta = epoch_time * (args.epochs - epoch - 1)
        line = (
            f"epoch {epoch} loss {loss_epoch:.4f} time {epoch_time:.1f}s eta {_fmt_eta(eta)} "
            f"Ly {loss_avg.get('Ly', 0.0):.4f} Lz {loss_avg.get('Lz', 0.0):.4f}"
        )
        record = {"epoch": epoch, "loss": loss_epoch, "type": "train"}
        record.update(loss_avg)
        record.update(solver_avg)
        record["pcdeq_stop_reason"] = solver_last.get("pcdeq_stop_reason")
        record["theta_gmres_fail_reason"] = solver_last.get("theta_gmres_fail_reason")
        logbus.step(record, line=line)

        if args.eval_interval > 0 and (epoch + 1) % args.eval_interval == 0:
            run_full_eval(model, eval_dataset, args, ctx_params, logbus, epoch + 1)
            model.train()


def run_full_eval(
    model: MainNet,
    eval_dataset,
    args: argparse.Namespace,
    ctx_params: Dict[str, object],
    logbus: LogBus,
    epoch_idx: int,
) -> None:
    model.eval()
    with torch.no_grad():
        full = eval_dataset.get_full()
        y_full = full["Y_lr"].unsqueeze(0).to(next(model.parameters()).device)
        z_full = full["Z_hr"].unsqueeze(0).to(next(model.parameters()).device)
        x_full = full.get("X_gt")
        if x_full is not None:
            x_full = x_full.unsqueeze(0).to(next(model.parameters()).device)
        eval_scale = int(full.get("scale", args.scale))
        steps_override = args.eval_steps if args.eval_steps > 0 else None
        eval_ctx = dict(ctx_params)
        if steps_override is not None:
            eval_ctx["steps"] = int(steps_override)
        out = model(y_full, z_full, ctx=eval_ctx)
        x_hat = out["X"]
        y_hat = ay_forward(x_hat, out["theta"], scale=eval_scale)
        z_hat = az_forward(x_hat, out["theta"], scale=eval_scale, srf_smile=model.srf_smile)
        qnr_val, d_lambda, d_s = qnr(y_hat, y_full, z_hat, z_full)
        record = {
            "epoch": epoch_idx,
            "type": "eval",
            "qnr": float(qnr_val.item()),
            "d_lambda": float(d_lambda.item()),
            "d_s": float(d_s.item()),
        }
        if x_full is not None:
            record.update(
                {
                    "psnr": float(psnr(x_hat, x_full).item()),
                    "psnr_az": float(psnr(z_hat, z_full).item()),
                    "ssim": float(ssim(x_hat, x_full).item()),
                    "sam": float(sam(x_hat, x_full).item()),
                    "ergas": float(ergas(x_hat, x_full, scale=eval_scale).item()),
                }
            )
            line = (
                f"[eval] epoch {epoch_idx} psnr {record['psnr']:.3f} psnr_az {record['psnr_az']:.3f} "
                f"ssim {record['ssim']:.4f} sam {record['sam']:.4f} ergas {record['ergas']:.4f} qnr {record['qnr']:.4f}"
            )
        else:
            line = (
                f"[eval] epoch {epoch_idx} qnr {record['qnr']:.4f} "
                f"d_lambda {record['d_lambda']:.4f} d_s {record['d_s']:.4f}"
            )
        logbus.eval(record, line=line)


def _fmt_eta(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds + 0.5), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


if __name__ == "__main__":
    main()
