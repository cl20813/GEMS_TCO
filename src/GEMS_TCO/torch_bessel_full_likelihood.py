"""Torch full-likelihood diagnostic for anisotropic Bessel-Matern smooth fits.

This module is intentionally narrow: it is a validation bridge between the
existing SciPy direct-Bessel likelihood and a torch autograd full GP likelihood.
The Bessel K value is evaluated with SciPy for numerical stability, while a
custom torch autograd function supplies finite-difference derivatives with
respect to scaled distance and smoothness.

It is meant for small tiles / simulation diagnostics, not for production-scale
GPU Vecchia fitting.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch
from scipy.special import gammaln, kv


TWO_PI_LOG = float(np.log(2.0 * np.pi))


@dataclass
class TorchNaturalParams:
    sigmasq: torch.Tensor
    range_lat: torch.Tensor
    range_lon: torch.Tensor
    smooth: torch.Tensor
    nugget: torch.Tensor
    phi1: torch.Tensor
    phi2: torch.Tensor
    phi3: torch.Tensor

    def scalar_record(self) -> dict[str, float]:
        out = {}
        for key, value in asdict(self).items():
            out[key] = float(value.detach().cpu())
        out["sigma"] = math.sqrt(max(out["sigmasq"], 0.0))
        return out


def raw_to_smooth_torch(raw: torch.Tensor, smooth_bounds: tuple[float, float]) -> torch.Tensor:
    lo, hi = map(float, smooth_bounds)
    return raw.new_tensor(lo) + raw.new_tensor(hi - lo) * torch.sigmoid(raw)


def smooth_to_raw_np(smooth: float, smooth_bounds: tuple[float, float]) -> float:
    lo, hi = map(float, smooth_bounds)
    p = min(max((float(smooth) - lo) / (hi - lo), 1e-12), 1.0 - 1e-12)
    return float(math.log(p / (1.0 - p)))


def raw_from_natural_np(
    sigmasq: float,
    range_lat: float,
    range_lon: float,
    smooth: float,
    nugget: float,
    nugget_mode: str = "free",
    smooth_bounds: tuple[float, float] = (0.05, 2.5),
) -> np.ndarray:
    sigmasq = max(float(sigmasq), 1e-12)
    range_lat = max(float(range_lat), 1e-12)
    range_lon = max(float(range_lon), 1e-12)
    phi2 = 1.0 / range_lon
    phi3 = (range_lon / range_lat) ** 2
    phi1 = sigmasq * phi2
    vals = [
        math.log(max(phi1, 1e-300)),
        math.log(max(phi2, 1e-300)),
        math.log(max(phi3, 1e-300)),
        smooth_to_raw_np(float(smooth), smooth_bounds),
    ]
    if str(nugget_mode) == "free":
        vals.append(math.log(max(float(nugget), 1e-300)))
    return np.asarray(vals, dtype=np.float64)


def natural_from_raw_torch(
    raw: torch.Tensor,
    nugget_mode: str = "free",
    fixed_nugget: float = 0.0,
    smooth_bounds: tuple[float, float] = (0.05, 2.5),
) -> TorchNaturalParams:
    phi1 = torch.exp(raw[0])
    phi2 = torch.exp(raw[1])
    phi3 = torch.exp(raw[2])
    sigmasq = phi1 / phi2
    range_lon = 1.0 / phi2
    range_lat = 1.0 / (phi2 * torch.sqrt(phi3).clamp_min(1e-12))
    smooth = raw_to_smooth_torch(raw[3], smooth_bounds)
    if str(nugget_mode) == "free":
        nugget = torch.exp(raw[4])
    elif str(nugget_mode) == "fixed0":
        nugget = raw.new_tensor(0.0)
    else:
        nugget = raw.new_tensor(float(fixed_nugget))
    return TorchNaturalParams(
        sigmasq=sigmasq,
        range_lat=range_lat,
        range_lon=range_lon,
        smooth=smooth,
        nugget=nugget,
        phi1=phi1,
        phi2=phi2,
        phi3=phi3,
    )


def make_mean_design_torch(coords: torch.Tensor, mean_design: str = "lat") -> torch.Tensor:
    ones = torch.ones((coords.shape[0], 1), dtype=coords.dtype, device=coords.device)
    lat = coords[:, 0:1] - torch.mean(coords[:, 0:1])
    lon = coords[:, 1:2] - torch.mean(coords[:, 1:2])
    if mean_design in {"constant", "intercept"}:
        return ones
    if mean_design in {"lat", "base", "hour_spatial"}:
        return torch.cat([ones, lat], dim=1)
    if mean_design == "latlon":
        return torch.cat([ones, lat, lon], dim=1)
    raise ValueError(f"Unsupported mean_design={mean_design!r}")


def matern_corr_bessel_np(scaled_distance: np.ndarray, smooth: float) -> np.ndarray:
    r = np.asarray(scaled_distance, dtype=np.float64)
    out = np.empty_like(r)
    zero = r <= 0.0
    out[zero] = 1.0
    z = r[~zero]
    if z.size:
        nu = float(smooth)
        if nu <= 0.0:
            return np.full_like(r, np.nan)
        arg = np.sqrt(2.0 * nu) * z
        log_prefactor = (1.0 - nu) * math.log(2.0) - float(gammaln(nu))
        vals = np.exp(log_prefactor + nu * np.log(arg)) * kv(nu, arg)
        out[~zero] = np.nan_to_num(vals, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(out, 0.0, 1.0)


class _MaternBesselCorr(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scaled_distance: torch.Tensor, smooth: torch.Tensor):
        r_np = scaled_distance.detach().cpu().numpy().astype(np.float64, copy=False)
        nu = float(smooth.detach().cpu())
        corr_np = matern_corr_bessel_np(r_np, nu)
        ctx.save_for_backward(scaled_distance, smooth)
        return torch.as_tensor(corr_np, dtype=scaled_distance.dtype, device=scaled_distance.device)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        scaled_distance, smooth = ctx.saved_tensors
        r_np = scaled_distance.detach().cpu().numpy().astype(np.float64, copy=False)
        nu = float(smooth.detach().cpu())

        eps_r = max(1e-5, 1e-4 * max(float(np.nanmedian(np.abs(r_np))), 1.0))
        rp = np.maximum(r_np + eps_r, 0.0)
        rm = np.maximum(r_np - eps_r, 0.0)
        dc_dr_np = (matern_corr_bessel_np(rp, nu) - matern_corr_bessel_np(rm, nu)) / (rp - rm + 1e-300)
        dc_dr_np[r_np <= 0.0] = 0.0

        eps_nu = max(1e-4, 1e-4 * max(abs(nu), 1.0))
        nu_p = nu + eps_nu
        nu_m = max(nu - eps_nu, 1e-6)
        dc_dnu_np = (matern_corr_bessel_np(r_np, nu_p) - matern_corr_bessel_np(r_np, nu_m)) / (nu_p - nu_m)

        dc_dr = torch.as_tensor(dc_dr_np, dtype=grad_output.dtype, device=grad_output.device)
        dc_dnu = torch.as_tensor(dc_dnu_np, dtype=grad_output.dtype, device=grad_output.device)
        grad_r = grad_output * dc_dr
        grad_nu = torch.sum(grad_output * dc_dnu).to(dtype=smooth.dtype, device=smooth.device)
        return grad_r, grad_nu


def matern_corr_bessel_torch(scaled_distance: torch.Tensor, smooth: torch.Tensor) -> torch.Tensor:
    return _MaternBesselCorr.apply(scaled_distance, smooth)


def covariance_from_raw_torch(
    raw: torch.Tensor,
    coords: torch.Tensor,
    nugget_mode: str = "free",
    fixed_nugget: float = 0.0,
    smooth_bounds: tuple[float, float] = (0.05, 2.5),
    jitter: float = 1e-6,
) -> torch.Tensor:
    params = natural_from_raw_torch(raw, nugget_mode=nugget_mode, fixed_nugget=fixed_nugget, smooth_bounds=smooth_bounds)
    d_lat = coords[:, None, 0] - coords[None, :, 0]
    d_lon = coords[:, None, 1] - coords[None, :, 1]
    dist2 = (
        (d_lat / params.range_lat.clamp_min(1e-12)).pow(2)
        + (d_lon / params.range_lon.clamp_min(1e-12)).pow(2)
    )
    # Clamp only for the derivative at exact zero-distance diagonal entries.
    # Without this, sqrt'(0)=inf can produce 0*inf NaNs in range gradients.
    scaled = torch.sqrt(dist2.clamp_min(1e-24))
    corr = matern_corr_bessel_torch(scaled, params.smooth)
    cov = params.sigmasq * corr
    idx = torch.arange(coords.shape[0], device=coords.device)
    cov = cov.clone()
    cov[idx, idx] = cov[idx, idx] + params.nugget + cov.new_tensor(float(jitter))
    return cov


def profiled_full_nll_torch(
    raw: torch.Tensor,
    y: torch.Tensor,
    coords: torch.Tensor,
    nugget_mode: str = "free",
    fixed_nugget: float = 0.0,
    smooth_bounds: tuple[float, float] = (0.05, 2.5),
    mean_design: str = "lat",
    jitter: float = 1e-6,
    scale_by_n: bool = True,
    ridge: float = 1e-8,
) -> torch.Tensor:
    n = int(y.shape[0])
    K = covariance_from_raw_torch(
        raw,
        coords,
        nugget_mode=nugget_mode,
        fixed_nugget=fixed_nugget,
        smooth_bounds=smooth_bounds,
        jitter=jitter,
    )
    X = make_mean_design_torch(coords, mean_design)
    L = torch.linalg.cholesky(K)
    kinv_y = torch.cholesky_solve(y.reshape(-1, 1), L)
    kinv_X = torch.cholesky_solve(X, L)
    xt_k_x = X.T @ kinv_X
    xt_k_y = X.T @ kinv_y
    eye = torch.eye(xt_k_x.shape[0], dtype=xt_k_x.dtype, device=xt_k_x.device)
    beta = torch.linalg.solve(xt_k_x + eye * float(ridge), xt_k_y)
    resid = y.reshape(-1, 1) - X @ beta
    kinv_resid = torch.cholesky_solve(resid, L)
    quad = (resid.T @ kinv_resid).squeeze()
    logdet = 2.0 * torch.sum(torch.log(torch.diagonal(L)))
    nll = 0.5 * (raw.new_tensor(float(n)) * raw.new_tensor(TWO_PI_LOG) + logdet + quad)
    return nll / float(n) if scale_by_n else nll


def finite_difference_grad_np(
    objective,
    raw: Sequence[float],
    eps: float = 1e-4,
) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    grad = np.full_like(raw, np.nan)
    for i in range(raw.size):
        step = eps * max(abs(float(raw[i])), 1.0)
        plus = raw.copy()
        minus = raw.copy()
        plus[i] += step
        minus[i] -= step
        grad[i] = (float(objective(plus)) - float(objective(minus))) / (2.0 * step)
    return grad


def torch_value_and_grad(
    raw_np: Sequence[float],
    y_np: np.ndarray,
    coords_np: np.ndarray,
    nugget_mode: str = "free",
    fixed_nugget: float = 0.0,
    smooth_bounds: tuple[float, float] = (0.05, 2.5),
    mean_design: str = "lat",
    jitter: float = 1e-6,
) -> tuple[float, np.ndarray, dict[str, float]]:
    raw = torch.tensor(np.asarray(raw_np, dtype=np.float64), dtype=torch.float64, requires_grad=True)
    y = torch.tensor(np.asarray(y_np, dtype=np.float64), dtype=torch.float64)
    coords = torch.tensor(np.asarray(coords_np, dtype=np.float64), dtype=torch.float64)
    loss = profiled_full_nll_torch(
        raw,
        y=y,
        coords=coords,
        nugget_mode=nugget_mode,
        fixed_nugget=fixed_nugget,
        smooth_bounds=smooth_bounds,
        mean_design=mean_design,
        jitter=jitter,
        scale_by_n=True,
    )
    loss.backward()
    return float(loss.detach().cpu()), raw.grad.detach().cpu().numpy().copy(), natural_from_raw_torch(
        raw.detach(), nugget_mode=nugget_mode, fixed_nugget=fixed_nugget, smooth_bounds=smooth_bounds
    ).scalar_record()


def fit_full_matern_torch(
    y_np: np.ndarray,
    coords_np: np.ndarray,
    start_raw: Sequence[float],
    nugget_mode: str = "free",
    fixed_nugget: float = 0.0,
    smooth_bounds: tuple[float, float] = (0.05, 2.5),
    mean_design: str = "lat",
    jitter: float = 1e-6,
    max_iter: int = 80,
    max_eval: int | None = None,
    history_size: int = 20,
    tolerance_grad: float = 1e-7,
    tolerance_change: float = 1e-9,
    bounds: dict[str, tuple[float, float]] | None = None,
    device: str | torch.device | None = None,
) -> dict:
    """Optimize the smooth-free full likelihood with torch LBFGS.

    Bounds are enforced by clamping the raw vector before every closure
    evaluation.  This keeps the diagnostic comparable to SciPy L-BFGS-B without
    introducing another transform layer.
    """
    if device is None:
        requested = os.environ.get("TORCH_FULL_DEVICE", "cpu").strip().lower()
        if requested == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(requested or "cpu")
    else:
        device = torch.device(device)

    y = torch.tensor(np.asarray(y_np, dtype=np.float64), dtype=torch.float64, device=device)
    coords = torch.tensor(np.asarray(coords_np, dtype=np.float64), dtype=torch.float64, device=device)
    raw = torch.tensor(np.asarray(start_raw, dtype=np.float64), dtype=torch.float64, device=device, requires_grad=True)

    if bounds is None:
        bounds = {}
    raw_los = np.full(len(start_raw), -np.inf, dtype=np.float64)
    raw_his = np.full(len(start_raw), np.inf, dtype=np.float64)
    if "log_phi1" in bounds:
        raw_los[0], raw_his[0] = bounds["log_phi1"]
    if "log_phi2" in bounds:
        raw_los[1], raw_his[1] = bounds["log_phi2"]
    if "log_phi3" in bounds:
        raw_los[2], raw_his[2] = bounds["log_phi3"]
    if "smooth_raw" in bounds:
        raw_los[3], raw_his[3] = bounds["smooth_raw"]
    if len(start_raw) > 4 and "log_nugget" in bounds:
        raw_los[4], raw_his[4] = bounds["log_nugget"]
    lo_t = torch.tensor(raw_los, dtype=torch.float64, device=device)
    hi_t = torch.tensor(raw_his, dtype=torch.float64, device=device)

    def clamp_raw_() -> None:
        with torch.no_grad():
            raw.copy_(torch.minimum(torch.maximum(raw, lo_t), hi_t))

    opt = torch.optim.LBFGS(
        [raw],
        lr=1.0,
        max_iter=int(max_iter),
        max_eval=int(max_eval if max_eval is not None else max_iter),
        history_size=int(history_size),
        line_search_fn="strong_wolfe",
        tolerance_grad=float(tolerance_grad),
        tolerance_change=float(tolerance_change),
    )
    calls = 0
    loss_value = float("nan")

    def closure():
        nonlocal calls, loss_value
        clamp_raw_()
        opt.zero_grad(set_to_none=True)
        loss = profiled_full_nll_torch(
            raw,
            y=y,
            coords=coords,
            nugget_mode=nugget_mode,
            fixed_nugget=fixed_nugget,
            smooth_bounds=smooth_bounds,
            mean_design=mean_design,
            jitter=jitter,
            scale_by_n=True,
        )
        loss.backward()
        loss_value = float(loss.detach().cpu())
        calls += 1
        return loss

    try:
        opt.step(closure)
        clamp_raw_()
        with torch.no_grad():
            final_loss = profiled_full_nll_torch(
                raw,
                y=y,
                coords=coords,
                nugget_mode=nugget_mode,
                fixed_nugget=fixed_nugget,
                smooth_bounds=smooth_bounds,
                mean_design=mean_design,
                jitter=jitter,
                scale_by_n=True,
            )
        success = bool(torch.isfinite(final_loss).item())
        message = "finite_loss"
        loss_value = float(final_loss.detach().cpu())
    except Exception as exc:
        success = False
        message = repr(exc)

    raw_np = raw.detach().cpu().numpy().copy()
    params = natural_from_raw_torch(
        raw.detach(),
        nugget_mode=nugget_mode,
        fixed_nugget=fixed_nugget,
        smooth_bounds=smooth_bounds,
    ).scalar_record()
    params.update({
        "success": success,
        "loss": float(loss_value),
        "message": message,
        "n_eval": int(calls),
        "raw_params": raw_np.tolist(),
        "torch_device": str(device),
    })
    return params
