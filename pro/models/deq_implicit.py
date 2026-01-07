"""
Implicit-only DEQ container: fixed-point forward + GMRES implicit backward.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as P
from torch import autograd

def fixed_point_iteration(
    f: Callable[[torch.Tensor], torch.Tensor],
    z0: torch.Tensor,
    max_iter: int = 200,
    tol: float = 1e-4,
    check_every: int = 1,
    min_iter: int = 3,
    stagnation_check_start: int = 20,
    stats: dict | None = None,
    record_trace: bool = False,
    trace_k: int = 3,
    backoff_enable: bool = False,
    backoff_factor: float = 0.5,
    backoff_steps: int = 3,
    backoff_improve_min: float = 0.01,
    eff_step_init: float | None = None,
    eff_step_min: float = 1e-4,
    **_kwargs,
) -> Tuple[torch.Tensor, List[float], int, str]:
    """
    固定点迭代求解 z = f(z)。
    返回: (z, res_list, iters, stop_reason)。
    残差定义：||z_next_raw - z|| / (||z_next_raw|| + 1e-5)。
    """
    if int(max_iter) <= 0:
        return z0, [], 0, "maxiter"
    z = z0
    res: List[float] = []
    res_after: List[float] = []
    stop_reason = "maxiter"
    check_every = max(1, int(check_every))
    min_iter = max(1, int(min_iter))
    stagnation_check_start = max(1, int(stagnation_check_start))
    backoff_factor = float(backoff_factor)
    backoff_steps = max(0, int(backoff_steps))
    backoff_improve_min = float(backoff_improve_min)
    eff_step = float(eff_step_init) if eff_step_init is not None else 1.0
    eff_step_min = max(float(eff_step_min), 1e-6)
    backoff_triggered = 0
    backoff_total = 0
    eff_step_min_used = eff_step
    stagnated_iter = 0
    err_prev_after: float | None = None
    with P.cached():
        for i in range(1, max_iter + 1):
            z_next_raw = f(z)
            if i % check_every == 0:
                denom = torch.norm(z_next_raw).item()
                err = torch.norm(z_next_raw - z).item() / (denom + 1e-5)
                if not math.isfinite(err):
                    stop_reason = "naninf"
                    z = z_next_raw
                    break
                res.append(err)
                if i >= stagnation_check_start and len(res) >= 5:
                    window = res[-5:]
                    if res[-1] > 0 and min(window) / res[-1] > 0.98:
                        z = z_next_raw
                        stop_reason = "stagnated"
                        stagnated_iter = i
                        break
                if err < tol and i >= min_iter:
                    z = z_next_raw
                    stop_reason = "ok"
                    break
            z_next = z + eff_step * (z_next_raw - z)
            err_after = torch.norm(z_next - z).item() / (torch.norm(z_next).item() + 1e-5)
            if i % check_every == 0:
                res_after.append(err_after)
            if backoff_enable and err_prev_after is not None and backoff_steps > 0:
                backoff_used = 0
                while (
                    err_after > err_prev_after * (1.0 - backoff_improve_min)
                    and backoff_used < backoff_steps
                ):
                    eff_step = max(eff_step * backoff_factor, eff_step_min)
                    backoff_used += 1
                    z_next = z + eff_step * (z_next_raw - z)
                    err_after = torch.norm(z_next - z).item() / (torch.norm(z_next).item() + 1e-5)
                if backoff_used > 0:
                    backoff_triggered += 1
                    backoff_total += backoff_used
            eff_step_min_used = min(eff_step_min_used, eff_step)
            z = z_next
            err_prev_after = err_after
    if stats is not None:
        stats["stagnated_iter"] = stagnated_iter
        stats["backoff_triggered"] = backoff_triggered
        stats["backoff_total"] = backoff_total
        stats["eff_step_min_used"] = eff_step_min_used
        stats["res_after"] = res_after
        if record_trace and res:
            k = max(1, int(trace_k))
            stats["res_trace_head"] = res[:k]
            stats["res_trace_tail"] = res[-k:]
            if res_after:
                stats["res_after_trace_head"] = res_after[:k]
                stats["res_after_trace_tail"] = res_after[-k:]
    return z, res, i, stop_reason


def gmres_solve(
    linear_op: Callable[[torch.Tensor], torch.Tensor],
    u: torch.Tensor,
    max_iter: int = 10,
    tol: float = 1e-4,
    M_inv: Callable[[torch.Tensor], torch.Tensor] | None = None,
    early_stop: bool = False,
    early_stop_min_iter: int = 3,
) -> Tuple[torch.Tensor, dict]:
    """
    GMRES：对 (I - J^T)v = u 形式的线性系统做 Krylov 迭代，可选左预条件 M_inv。
    linear_op 应返回 (I - J^T)@v 结果。
    """
    device = u.device
    x = torch.zeros_like(u)
    x_approx = x

    def apply_pre(v):
        return M_inv(v) if M_inv is not None else v

    r0 = apply_pre(u - linear_op(x))
    beta = r0.norm()
    info = {
        "converged": False,
        "iters": 0,
        "relres": float("inf"),
        "fail_reason": "ok",
        "neumann_used": False,
        "trivial": False,
    }
    if beta < 1e-12:
        info["fail_reason"] = "trivial"
        info["trivial"] = True
        info["relres"] = 0.0
        return u, info
    V = [r0 / beta]
    H = torch.zeros((max_iter + 1, max_iter), device=device, dtype=u.dtype)
    g = torch.zeros((max_iter + 1,), device=device, dtype=u.dtype)
    g[0] = beta
    relres_prev = None
    non_decrease = 0
    for k in range(max_iter):
        w = apply_pre(linear_op(V[k]))
        for j in range(k + 1):
            H[j, k] = torch.dot(w.flatten(), V[j].flatten())
            w = w - H[j, k] * V[j]
        H[k + 1, k] = w.norm()
        if H[k + 1, k] > 1e-12:
            V.append(w / H[k + 1, k])
        Hk = H[: k + 2, : k + 1]
        gk = g[: k + 2]
        y, *_ = torch.linalg.lstsq(Hk, gk)
        x_approx = sum(y_i * v_i for y_i, v_i in zip(y, V[: k + 1]))
        res = (gk - Hk @ y).norm()
        info["iters"] = k + 1
        info["relres"] = res.item() / (beta.item() + 1e-12)
        if not torch.isfinite(res):
            info["fail_reason"] = "naninf"
            return x_approx, info
        if info["relres"] < tol:
            info["converged"] = True
            return x_approx, info
        if relres_prev is not None:
            if info["relres"] > relres_prev + 1e-12:
                non_decrease += 1
            else:
                non_decrease = 0
        relres_prev = info["relres"]
        if early_stop and (k + 1) >= early_stop_min_iter and non_decrease >= 2:
            info["fail_reason"] = "early_stop"
            return x_approx, info
    info["fail_reason"] = "maxiter"
    return x_approx, info


class DEQImplicit(nn.Module):
    """
    Adapter DEQ: supports z_init + x_bundle(dict), FPI forward, GMRES implicit backward.
    """

    def __init__(
        self,
        f: Callable[[torch.Tensor, Dict], torch.Tensor],
        solver: Callable | None = None,
        max_iter: int = 200,
        tol: float = 1e-4,
    ):
        super().__init__()
        self.f = f
        self.solver = solver or fixed_point_iteration
        self.max_iter = max_iter
        self.tol = tol
        self.iter_forward = 0
        self.iter_backward = 0
        self.res_forward: List[float] = []
        self.res_backward: List[float] = []
        self.fail_reason_forward = "ok"
        self.fail_reason_backward = "ok"
        self.stop_reason_forward = "ok"
        self.stagnated_forward = False

    def _solve_forward(self, x_bundle: Dict, z_init: torch.Tensor):
        if z_init is None:
            raise ValueError("DEQ requires z_init")
        if self.f is None:
            raise RuntimeError("DEQ missing mapping f(z, x_bundle)")
        params = x_bundle.get("params") or {}
        deq_stats = x_bundle.get("deq_stats")
        bwd_mode = str(params.get("bwd_mode", "implicit"))
        if bwd_mode != "implicit":
            raise RuntimeError(f"DEQ only supports implicit backward, got bwd_mode={bwd_mode}")
        fwd_check_every = int(params.get("fwd_check_every", 1) or 1)
        fwd_min_iter = int(params.get("fwd_min_iter", 3) or 3)
        stagnation_check_start = int(params.get("stagnation_check_start", 20) or 20)
        fwd_soft_accept_res = float(params.get("fwd_soft_accept_res", 1e-3))
        fwd_backoff_enable = bool(params.get("fwd_backoff_enable", True))
        fwd_backoff_factor = float(params.get("fwd_backoff_factor", 0.5))
        fwd_backoff_steps = int(params.get("fwd_backoff_steps", 3) or 0)
        fwd_backoff_improve_min = float(params.get("fwd_backoff_improve_min", 0.01))
        eff_step_init = params.get("eff_step_init", None)
        if eff_step_init is None:
            eff_step_init = params.get("eff_step_max", 1.0)
        eff_step_min = float(params.get("fwd_eff_step_min", 1e-4))
        record_trace = bool(params.get("record_trace", False))
        trace_k = int(params.get("trace_k", 3))

        def f_apply(z: torch.Tensor) -> torch.Tensor:
            return self.f(z, x_bundle)

        with torch.no_grad():
            z_star, res_fwd, iters_fwd, stop_reason = self.solver(
                lambda z: f_apply(z),
                z_init,
                max_iter=self.max_iter,
                tol=self.tol,
                check_every=fwd_check_every,
                min_iter=fwd_min_iter,
                record_trace=record_trace,
                trace_k=trace_k,
                stats=deq_stats,
                backoff_enable=fwd_backoff_enable,
                backoff_factor=fwd_backoff_factor,
                backoff_steps=fwd_backoff_steps,
                backoff_improve_min=fwd_backoff_improve_min,
                eff_step_init=eff_step_init,
                eff_step_min=eff_step_min,
                stagnation_check_start=stagnation_check_start,
            )
        self.iter_forward = iters_fwd
        self.res_forward = res_fwd
        self.stop_reason_forward = stop_reason
        res_last = res_fwd[-1] if res_fwd else float("inf")
        soft_accept = stop_reason in ("stagnated", "maxiter")
        soft_accept_res_ok = soft_accept and res_last <= fwd_soft_accept_res
        if stop_reason == "naninf":
            self.fail_reason_forward = stop_reason
            raise RuntimeError(
                f"DEQ forward failed: reason={stop_reason} iters={iters_fwd} res_last={res_last:.3e}"
            )
        if stop_reason == "ok":
            self.fail_reason_forward = "ok"
        elif soft_accept_res_ok:
            self.fail_reason_forward = "ok"
        else:
            self.fail_reason_forward = stop_reason
        self.stagnated_forward = False
        if res_fwd and len(res_fwd) >= 10:
            window = res_fwd[-5:]
            last_err = res_fwd[-1]
            if last_err > 0 and min(window) / last_err > 0.98:
                self.stagnated_forward = True
        if isinstance(deq_stats, dict):
            deq_stats["k_star"] = iters_fwd
            deq_stats["r_tail"] = res_fwd
            deq_stats["fail_reason"] = self.fail_reason_forward
            deq_stats["stop_reason"] = stop_reason
            deq_stats["stagnated_forward"] = self.stagnated_forward
            deq_stats["res_last_forward"] = float(res_last) if math.isfinite(res_last) else res_last
            deq_stats["soft_accept_forward"] = soft_accept
            deq_stats["soft_accept_res_ok"] = soft_accept_res_ok
            deq_stats["stop_reason_forward"] = stop_reason
            deq_stats["fail_reason_forward"] = self.fail_reason_forward
            deq_stats["iter_forward"] = iters_fwd
        return z_star, f_apply, params, deq_stats

    def _gmres_backward(
        self,
        grad: torch.Tensor,
        z0: torch.Tensor,
        f0: torch.Tensor,
        params: Dict,
        deq_stats: Dict | None,
        gmres_cfg: Dict | None = None,
    ) -> Tuple[torch.Tensor, Dict]:
        gmres_cfg = gmres_cfg or params.get("bwd_gmres_cfg") or {}
        max_iter = int(gmres_cfg.get("max_iter", params.get("gmres_max_iter", self.max_iter)))
        tol = float(gmres_cfg.get("tol", params.get("gmres_tol", self.tol)))
        early_stop = bool(gmres_cfg.get("early_stop", params.get("gmres_early_stop", True)))
        early_stop_min_iter = int(
            gmres_cfg.get("early_stop_min_iter", params.get("gmres_early_stop_min_iter", 3))
        )
        pc_enable = bool(gmres_cfg.get("pc_enable", False)) or bool(params.get("bwd_gmres_pc_enable", False))
        pc_diag_min = float(params.get("bwd_gmres_pc_diag_min", 1e-2))
        pc_diag_max = float(params.get("bwd_gmres_pc_diag_max", 1e2))
        pc_diag = None
        if pc_enable:
            r = torch.empty_like(z0).bernoulli_(0.5).mul_(2).add_(-1)
            jtr = autograd.grad(f0, z0, r, retain_graph=True)[0]
            diag_est = (1.0 - (jtr * r)).detach()
            if torch.isfinite(diag_est).all():
                abs_diag = diag_est.abs().clamp(min=pc_diag_min, max=pc_diag_max)
                sign = torch.where(diag_est >= 0, torch.ones_like(diag_est), -torch.ones_like(diag_est))
                pc_diag = sign * abs_diag

        def linop(v):
            jtv = autograd.grad(f0, z0, v, retain_graph=True)[0]
            return v - jtv

        M_inv = None
        if pc_diag is not None:

            def M_inv(v):
                return v / pc_diag

        g, info = gmres_solve(
            linop,
            grad,
            max_iter=max_iter,
            tol=tol,
            M_inv=M_inv,
            early_stop=early_stop,
            early_stop_min_iter=early_stop_min_iter,
        )
        self.iter_backward = int(info.get("iters", 0))
        relres = float(info.get("relres", 0.0))
        self.res_backward = [relres]
        fail_reason = str(info.get("fail_reason", "ok"))
        if info.get("converged", False):
            self.fail_reason_backward = "ok"
        else:
            self.fail_reason_backward = fail_reason
        if self.fail_reason_backward == "naninf":
            g = grad
        if not torch.isfinite(g).all():
            g = grad
            self.fail_reason_backward = "naninf"
            info["fail_reason"] = "naninf"
        if isinstance(deq_stats, dict):
            deq_stats["iter_backward"] = self.iter_backward
            deq_stats["res_backward"] = self.res_backward
            deq_stats["res_backward_last"] = self.res_backward[-1] if self.res_backward else 0.0
            deq_stats["fail_reason_backward"] = self.fail_reason_backward
            deq_stats["gmres_relres"] = relres
            deq_stats["gmres_fail_reason"] = self.fail_reason_backward
            deq_stats["theta_gmres_iters"] = self.iter_backward
            deq_stats["theta_gmres_relres"] = relres
            deq_stats["theta_gmres_fail_reason"] = self.fail_reason_backward
        return g, info

    def solve(self, f, z0, *, batch_ctx: Dict | None = None):
        x_bundle = batch_ctx or {}
        old_f = self.f
        if f is not None:
            self.f = f
        try:
            z_star, _, _, deq_stats = self._solve_forward(x_bundle, z0)
        finally:
            self.f = old_f
        return z_star, deq_stats

    def implicit_backward(
        self,
        f,
        x_star: torch.Tensor,
        dl_dxstar: torch.Tensor,
        *,
        batch_ctx: Dict | None = None,
        gmres_cfg: Dict | None = None,
    ):
        x_bundle = batch_ctx or {}
        params = x_bundle.get("params") or {}
        deq_stats = x_bundle.get("deq_stats")
        with torch.enable_grad():
            x0 = x_star.detach().requires_grad_(True)
            f0 = f(x0) if batch_ctx is None else f(x0, x_bundle)
            g, info = self._gmres_backward(dl_dxstar, x0, f0, params, deq_stats, gmres_cfg=gmres_cfg)
        return g, info

    def forward(self, x_bundle: Dict, z_init: torch.Tensor) -> torch.Tensor:
        self.iter_backward = 0
        self.res_backward = []
        self.fail_reason_backward = "ok"
        z_star, f_apply, params, deq_stats = self._solve_forward(x_bundle, z_init)
        z = f_apply(z_star)
        if not torch.is_grad_enabled():
            return z
        if not z.requires_grad:
            self.fail_reason_backward = "z_no_grad"
            if isinstance(deq_stats, dict):
                deq_stats["fail_reason_backward"] = self.fail_reason_backward
            raise RuntimeError("DEQ backward requires grad, but z has no grad_fn.")
        z_star_det = z_star.detach()

        def backward_hook(grad: torch.Tensor) -> torch.Tensor:
            with torch.enable_grad():
                z0 = z_star_det.clone().detach().requires_grad_(True)
                f0 = f_apply(z0)
                g, _ = self._gmres_backward(grad, z0, f0, params, deq_stats)
            return g

        z.register_hook(backward_hook)
        return z


__all__ = ["DEQImplicit", "fixed_point_iteration"]
