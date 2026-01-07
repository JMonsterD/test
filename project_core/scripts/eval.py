"""
Minimal eval entry for implicit-only model.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from data.datasets.synthetic import SyntheticDataset
from data.datasets.pavia import PaviaDataset
from models.model import MainNet
from models.operators import ay_forward, az_forward
from eval.metrics import psnr, ssim, sam, ergas, qnr
from utils.logbus import LogBus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="pavia", choices=["synthetic", "pavia"])
    parser.add_argument("--data_path", type=str, default="datasets/pavia_data_r8_30_40.mat")
    parser.add_argument("--scale", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--bh", type=int, default=103)
    parser.add_argument("--bm", type=int, default=4)
    parser.add_argument("--deq_max_iter", type=int, default=40)
    parser.add_argument("--deq_tol", type=float, default=1e-4)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--outdir", type=str, default="outputs")
    parser.add_argument("--log_file", type=str, default="eval.log")
    parser.add_argument("--weights", type=str, default="")
    return parser.parse_args()


def _load_weights(model: MainNet, ckpt_path: str) -> None:
    if not ckpt_path:
        return
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"weights not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)


def _build_eval_batch(args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    if args.dataset == "synthetic":
        dataset = SyntheticDataset(
            num_samples=1,
            bh=args.bh,
            bm=args.bm,
            h=args.patch_size // args.scale,
            w=args.patch_size // args.scale,
            scale=args.scale,
        )
        sample = dataset[0]
        return {
            "Y_lr": sample["Y_lr"],
            "Z_hr": sample["Z_hr"],
            "X_gt": sample.get("X_gt"),
            "scale": sample.get("scale", args.scale),
        }
    dataset = PaviaDataset(
        mat_path=args.data_path,
        patch_size=args.patch_size,
        patches_per_epoch=1,
        scale=args.scale,
        return_gt=True,
    )
    sample = dataset.get_full()
    return {
        "Y_lr": sample["Y_lr"],
        "Z_hr": sample["Z_hr"],
        "X_gt": sample.get("X_gt"),
        "scale": sample.get("scale", args.scale),
    }


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MainNet(bh=args.bh, bm=args.bm, scale=args.scale).to(device)
    _load_weights(model, args.weights)
    model.eval()

    batch = _build_eval_batch(args)
    y_lr = batch["Y_lr"].unsqueeze(0).to(device)
    z_hr = batch["Z_hr"].unsqueeze(0).to(device)
    x_gt = batch.get("X_gt")
    if x_gt is not None:
        x_gt = x_gt.unsqueeze(0).to(device)
    eval_scale = int(batch.get("scale", args.scale))
    steps = int(args.eval_steps) if args.eval_steps > 0 else int(args.deq_max_iter)
    ctx = {"steps": steps, "deq_tol": float(args.deq_tol), "scale": int(args.scale)}

    with torch.no_grad():
        out = model(y_lr, z_hr, ctx=ctx)
        x_hat = out["X"]
        y_hat = ay_forward(x_hat, out["theta"], scale=eval_scale)
        z_hat = az_forward(x_hat, out["theta"], scale=eval_scale, srf_smile=model.srf_smile)
        qnr_val, d_lambda, d_s = qnr(y_hat, y_lr, z_hat, z_hr)
        record = {
            "type": "eval",
            "qnr": float(qnr_val.item()),
            "d_lambda": float(d_lambda.item()),
            "d_s": float(d_s.item()),
        }
        if x_gt is not None:
            record.update(
                {
                    "psnr": float(psnr(x_hat, x_gt).item()),
                    "psnr_az": float(psnr(z_hat, z_hr).item()),
                    "ssim": float(ssim(x_hat, x_gt).item()),
                    "sam": float(sam(x_hat, x_gt).item()),
                    "ergas": float(ergas(x_hat, x_gt, scale=eval_scale).item()),
                }
            )
            line = (
                f"[eval] psnr {record['psnr']:.3f} psnr_az {record['psnr_az']:.3f} "
                f"ssim {record['ssim']:.4f} sam {record['sam']:.4f} ergas {record['ergas']:.4f} qnr {record['qnr']:.4f}"
            )
        else:
            line = (
                f"[eval] qnr {record['qnr']:.4f} d_lambda {record['d_lambda']:.4f} d_s {record['d_s']:.4f}"
            )

    logbus = LogBus(args.outdir, args.log_file)
    logbus.eval(record, line=line)


if __name__ == "__main__":
    main()
