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
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch

from GEMS_TCO.spatial.matern_bessel import (
    _validate_nugget_mode,
    _validate_observations,
    _validate_smooth_bounds,
    matern_corr_bessel as matern_corr_bessel_np,
    raw_from_natural as raw_from_natural_numpy,
    smooth_to_raw as smooth_to_raw_numpy,
)

TWO_PI_LOG = float(np.log(2.0 * np.pi))


@dataclass
class TorchMaternParameters:
    signal_variance: torch.Tensor
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
        out["signal_standard_deviation"] = math.sqrt(max(out["signal_variance"], 0.0))
        return out


def raw_to_smooth_torch(raw: torch.Tensor, smooth_bounds: tuple[float, float]) -> torch.Tensor:
    lo, hi = _validate_smooth_bounds(smooth_bounds)
    if raw.numel() != 1 or not torch.isfinite(raw).all():
        raise ValueError("raw smoothness must be one finite scalar")
    return raw.new_tensor(lo) + raw.new_tensor(hi - lo) * torch.sigmoid(raw)


def smooth_to_raw_np(smooth: float, smooth_bounds: tuple[float, float]) -> float:
    return smooth_to_raw_numpy(smooth, smooth_bounds)


def raw_from_natural_np(
    signal_variance: float,
    range_lat: float,
    range_lon: float,
    smooth: float,
    nugget: float,
    nugget_mode: str = "free",
    smooth_bounds: tuple[float, float] = (0.05, 2.5),
) -> np.ndarray:
    return raw_from_natural_numpy(
        signal_variance=signal_variance,
        range_lat=range_lat,
        range_lon=range_lon,
        smooth=smooth,
        nugget=nugget,
        nugget_mode=nugget_mode,
        smooth_bounds=smooth_bounds,
    )


def natural_from_raw_torch(
    raw: torch.Tensor,
    nugget_mode: str = "free",
    fixed_nugget: float = 0.0,
    smooth_bounds: tuple[float, float] = (0.05, 2.5),
) -> TorchMaternParameters:
    mode = _validate_nugget_mode(nugget_mode, fixed_nugget)
    expected = 5 if mode == "free" else 4
    if raw.ndim != 1 or raw.numel() != expected:
        raise ValueError(f"raw must have shape ({expected},) in {mode!r} mode, got {tuple(raw.shape)}")
    if not raw.is_floating_point() or not torch.isfinite(raw).all():
        raise ValueError("raw parameters must be finite floating-point values")
    phi1 = torch.exp(raw[0])
    phi2 = torch.exp(raw[1])
    phi3 = torch.exp(raw[2])
    transformed = torch.stack([phi1, phi2, phi3])
    if not torch.isfinite(transformed).all() or torch.any(transformed <= 0):
        raise ValueError("raw parameters underflow or overflow the natural parameterization")
    signal_variance = phi1 / phi2
    range_lon = 1.0 / phi2
    range_lat = 1.0 / (phi2 * torch.sqrt(phi3).clamp_min(1e-12))
    smooth = raw_to_smooth_torch(raw[3], smooth_bounds)
    if mode == "free":
        nugget = torch.exp(raw[4])
        if not torch.isfinite(nugget) or bool((nugget <= 0).item()):
            raise ValueError("raw nugget underflows or overflows the natural parameterization")
    elif mode == "fixed0":
        nugget = raw.new_tensor(0.0)
    else:
        nugget = raw.new_tensor(float(fixed_nugget))
    return TorchMaternParameters(
        signal_variance=signal_variance,
        range_lat=range_lat,
        range_lon=range_lon,
        smooth=smooth,
        nugget=nugget,
        phi1=phi1,
        phi2=phi2,
        phi3=phi3,
    )


def make_mean_design_torch(coords: torch.Tensor, mean_design: str = "lat") -> torch.Tensor:
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (n, 2), got {tuple(coords.shape)}")
    if not coords.is_floating_point() or not torch.isfinite(coords).all():
        raise ValueError("coords must contain only finite floating-point values")
    ones = torch.ones((coords.shape[0], 1), dtype=coords.dtype, device=coords.device)
    lat = coords[:, 0:1] - torch.mean(coords[:, 0:1])
    lon = coords[:, 1:2] - torch.mean(coords[:, 1:2])
    if mean_design in {"constant", "intercept"}:
        return ones
    if mean_design == "lat":
        return torch.cat([ones, lat], dim=1)
    if mean_design == "latlon":
        return torch.cat([ones, lat, lon], dim=1)
    raise ValueError(f"Unsupported mean_design={mean_design!r}")


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
        dc_dr_np = (matern_corr_bessel_np(rp, nu) - matern_corr_bessel_np(rm, nu)) / (
            rp - rm + 1e-300
        )
        dc_dr_np[r_np <= 0.0] = 0.0

        eps_nu = max(1e-4, 1e-4 * max(abs(nu), 1.0))
        nu_p = nu + eps_nu
        nu_m = max(nu - eps_nu, 1e-6)
        dc_dnu_np = (matern_corr_bessel_np(r_np, nu_p) - matern_corr_bessel_np(r_np, nu_m)) / (
            nu_p - nu_m
        )

        dc_dr = torch.as_tensor(dc_dr_np, dtype=grad_output.dtype, device=grad_output.device)
        dc_dnu = torch.as_tensor(dc_dnu_np, dtype=grad_output.dtype, device=grad_output.device)
        grad_r = grad_output * dc_dr
        grad_nu = torch.sum(grad_output * dc_dnu).to(dtype=smooth.dtype, device=smooth.device)
        return grad_r, grad_nu


def matern_corr_bessel_torch(scaled_distance: torch.Tensor, smooth: torch.Tensor) -> torch.Tensor:
    if not scaled_distance.is_floating_point() or not smooth.is_floating_point():
        raise TypeError("scaled_distance and smooth must be floating-point tensors")
    if smooth.numel() != 1 or not torch.isfinite(smooth).all() or bool((smooth <= 0).item()):
        raise ValueError("smooth must be one finite positive scalar")
    if not torch.isfinite(scaled_distance).all() or torch.any(scaled_distance < 0):
        raise ValueError("scaled_distance must contain only finite, non-negative values")
    return _MaternBesselCorr.apply(scaled_distance, smooth)


def covariance_from_raw_torch(
    raw: torch.Tensor,
    coords: torch.Tensor,
    nugget_mode: str = "free",
    fixed_nugget: float = 0.0,
    smooth_bounds: tuple[float, float] = (0.05, 2.5),
    jitter: float = 1e-6,
) -> torch.Tensor:
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (n, 2), got {tuple(coords.shape)}")
    if not coords.is_floating_point() or not torch.isfinite(coords).all():
        raise ValueError("coords must contain only finite floating-point values")
    if raw.device != coords.device:
        raise ValueError("raw and coords must be on the same device")
    if raw.dtype != coords.dtype:
        raise ValueError("raw and coords must have the same floating-point dtype")
    jitter = float(jitter)
    if not math.isfinite(jitter) or jitter < 0.0:
        raise ValueError("jitter must be finite and non-negative")
    params = natural_from_raw_torch(
        raw, nugget_mode=nugget_mode, fixed_nugget=fixed_nugget, smooth_bounds=smooth_bounds
    )
    d_lat = coords[:, None, 0] - coords[None, :, 0]
    d_lon = coords[:, None, 1] - coords[None, :, 1]
    dist2 = (d_lat / params.range_lat.clamp_min(1e-12)).pow(2) + (
        d_lon / params.range_lon.clamp_min(1e-12)
    ).pow(2)
    positive = dist2 > 0
    safe = torch.sqrt(dist2.clamp_min(torch.finfo(dist2.dtype).eps))
    scaled = torch.where(positive, safe, torch.zeros_like(dist2))
    corr = matern_corr_bessel_torch(scaled, params.smooth)
    cov = params.signal_variance * corr
    idx = torch.arange(coords.shape[0], device=coords.device)
    cov = cov.clone()
    cov[idx, idx] = cov[idx, idx] + params.nugget + cov.new_tensor(jitter)
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
    ridge: float = 0.0,
) -> torch.Tensor:
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1:
        raise ValueError(f"y must have shape (n,) or (n, 1), got {tuple(y.shape)}")
    if y.shape[0] == 0 or coords.shape != (y.shape[0], 2):
        raise ValueError("coords must have shape (len(y), 2) and y must be non-empty")
    if not y.is_floating_point() or not torch.isfinite(y).all():
        raise ValueError("y must contain only finite floating-point values")
    if raw.device != y.device or raw.device != coords.device:
        raise ValueError("raw, y, and coords must be on the same device")
    if raw.dtype != y.dtype or raw.dtype != coords.dtype:
        raise ValueError("raw, y, and coords must have the same floating-point dtype")
    ridge = float(ridge)
    if not math.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and non-negative")
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
    if n < X.shape[1] or int(torch.linalg.matrix_rank(X).item()) < X.shape[1]:
        raise ValueError("mean design is rank-deficient for the supplied coordinates")
    L = torch.linalg.cholesky(K)
    kinv_y = torch.cholesky_solve(y.reshape(-1, 1), L)
    kinv_X = torch.cholesky_solve(X, L)
    xt_k_x = X.T @ kinv_X
    xt_k_y = X.T @ kinv_y
    eye = torch.eye(xt_k_x.shape[0], dtype=xt_k_x.dtype, device=xt_k_x.device)
    beta = torch.linalg.solve(xt_k_x + eye * ridge, xt_k_y)
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
    raw = torch.tensor(
        np.asarray(raw_np, dtype=np.float64), dtype=torch.float64, requires_grad=True
    )
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
    return (
        float(loss.detach().cpu()),
        raw.grad.detach().cpu().numpy().copy(),
        natural_from_raw_torch(
            raw.detach(),
            nugget_mode=nugget_mode,
            fixed_nugget=fixed_nugget,
            smooth_bounds=smooth_bounds,
        ).scalar_record(),
    )


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
    """Optimize the full anisotropic Matérn likelihood with torch LBFGS.

    Bounds are enforced by clamping the raw vector before every closure
    evaluation.  The returned ``success`` flag requires a finite final loss and
    projected gradient no larger than ``tolerance_grad``; a merely finite iterate
    is reported as non-converged.
    """
    y_array, coords_array = _validate_observations(y_np, coords_np)
    mode = _validate_nugget_mode(nugget_mode, fixed_nugget)
    _validate_smooth_bounds(smooth_bounds)
    if int(max_iter) <= 0 or int(history_size) <= 0:
        raise ValueError("max_iter and history_size must be positive")
    if max_eval is not None and int(max_eval) <= 0:
        raise ValueError("max_eval must be positive when provided")
    if not math.isfinite(float(tolerance_grad)) or float(tolerance_grad) < 0.0:
        raise ValueError("tolerance_grad must be finite and non-negative")
    if not math.isfinite(float(tolerance_change)) or float(tolerance_change) < 0.0:
        raise ValueError("tolerance_change must be finite and non-negative")
    jitter = float(jitter)
    if not math.isfinite(jitter) or jitter < 0.0:
        raise ValueError("jitter must be finite and non-negative")
    device = torch.device("cpu" if device is None else device)

    y = torch.tensor(y_array, dtype=torch.float64, device=device)
    coords = torch.tensor(coords_array, dtype=torch.float64, device=device)
    raw = torch.tensor(
        np.asarray(start_raw, dtype=np.float64),
        dtype=torch.float64,
        device=device,
        requires_grad=True,
    )
    natural_from_raw_torch(
        raw,
        nugget_mode=mode,
        fixed_nugget=fixed_nugget,
        smooth_bounds=smooth_bounds,
    )

    if bounds is None:
        bounds = {}
    allowed_bounds = {"log_phi1", "log_phi2", "log_phi3", "smooth_raw", "log_nugget"}
    unknown_bounds = set(bounds) - allowed_bounds
    if unknown_bounds:
        raise ValueError(f"unknown raw bounds: {sorted(unknown_bounds)}")
    if mode != "free" and "log_nugget" in bounds:
        raise ValueError("log_nugget bounds are only valid when nugget_mode='free'")
    for name, pair in bounds.items():
        if len(pair) != 2:
            raise ValueError(f"bounds[{name!r}] must contain (lower, upper)")
        lower, upper = map(float, pair)
        if not np.isfinite([lower, upper]).all() or lower > upper:
            raise ValueError(f"bounds[{name!r}] must be finite and increasing")
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
    nll_value = float("nan")
    gradient_max = float("nan")

    def closure():
        nonlocal calls, loss_value
        clamp_raw_()
        opt.zero_grad(set_to_none=True)
        loss = profiled_full_nll_torch(
            raw,
            y=y,
            coords=coords,
            nugget_mode=mode,
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
        opt.zero_grad(set_to_none=True)
        final_loss = profiled_full_nll_torch(
            raw,
            y=y,
            coords=coords,
            nugget_mode=mode,
            fixed_nugget=fixed_nugget,
            smooth_bounds=smooth_bounds,
            mean_design=mean_design,
            jitter=jitter,
            scale_by_n=True,
        )
        final_loss.backward()
        grad = raw.grad.detach().clone()
        at_lower = raw.detach() <= lo_t
        at_upper = raw.detach() >= hi_t
        projected_grad = torch.where(
            (at_lower & (grad > 0)) | (at_upper & (grad < 0)),
            torch.zeros_like(grad),
            grad,
        )
        gradient_max = float(projected_grad.abs().max().cpu())
        finite = bool(torch.isfinite(final_loss).item() and torch.isfinite(raw).all().item())
        success = bool(finite and gradient_max <= float(tolerance_grad))
        message = "converged" if success else "finite_loss_without_gradient_convergence"
        loss_value = float(final_loss.detach().cpu())
        nll_value = loss_value * int(y.shape[0])
    except Exception as exc:
        success = False
        message = repr(exc)

    raw_np = raw.detach().cpu().numpy().copy()
    params = natural_from_raw_torch(
        raw.detach(),
        nugget_mode=mode,
        fixed_nugget=fixed_nugget,
        smooth_bounds=smooth_bounds,
    ).scalar_record()
    params.update(
        {
            "success": success,
            "loss": float(loss_value),
            "nll": float(nll_value),
            "message": message,
            "n_eval": int(calls),
            "gradient_max": float(gradient_max),
            "raw_params": raw_np.tolist(),
            "torch_device": str(device),
        }
    )
    return params
