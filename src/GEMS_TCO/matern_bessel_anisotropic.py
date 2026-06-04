"""
Direct-Bessel anisotropic Matern likelihood helpers for pure-space tests.

Parameterization follows the existing space-time code:

    phi2 = 1 / range_lon
    phi3 = (range_lon / range_lat)^2
    phi1 = sigmasq * phi2

so

    sigmasq   = phi1 / phi2
    range_lon = 1 / phi2
    range_lat = 1 / (phi2 * sqrt(phi3))

The smoothness parameter is estimated through a bounded logit transform.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.special import gammaln, kv


TWO_PI_LOG = float(np.log(2.0 * np.pi))
PENALTY = 1.0e30


@dataclass
class NaturalParams:
    sigmasq: float
    range_lat: float
    range_lon: float
    smooth: float
    nugget: float
    phi1: float
    phi2: float
    phi3: float

    def to_record(self) -> dict[str, float]:
        out = asdict(self)
        out["sigma"] = math.sqrt(max(float(self.sigmasq), 0.0))
        return out


def _clip_prob(x: float) -> float:
    return min(max(float(x), 1e-12), 1.0 - 1e-12)


def smooth_to_raw(smooth: float, smooth_bounds: tuple[float, float]) -> float:
    lo, hi = map(float, smooth_bounds)
    if not lo < hi:
        raise ValueError(f"smooth_bounds must be increasing, got {smooth_bounds}")
    p = _clip_prob((float(smooth) - lo) / (hi - lo))
    return float(math.log(p / (1.0 - p)))


def raw_to_smooth(raw: float, smooth_bounds: tuple[float, float]) -> float:
    lo, hi = map(float, smooth_bounds)
    z = float(raw)
    if z >= 0:
        e = math.exp(-z)
        p = 1.0 / (1.0 + e)
    else:
        e = math.exp(z)
        p = e / (1.0 + e)
    return float(lo + (hi - lo) * p)


def raw_from_natural(
    sigmasq: float,
    range_lat: float,
    range_lon: float,
    smooth: float,
    nugget: float,
    nugget_mode: str,
    smooth_bounds: tuple[float, float],
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
        smooth_to_raw(float(smooth), smooth_bounds),
    ]
    if str(nugget_mode) == "free":
        vals.append(math.log(max(float(nugget), 1e-300)))
    return np.asarray(vals, dtype=np.float64)


def natural_from_raw(
    raw: Sequence[float],
    nugget_mode: str,
    fixed_nugget: float,
    smooth_bounds: tuple[float, float],
) -> NaturalParams:
    raw = np.asarray(raw, dtype=np.float64)
    phi1 = float(np.exp(raw[0]))
    phi2 = float(np.exp(raw[1]))
    phi3 = float(np.exp(raw[2]))
    sigmasq = phi1 / phi2
    range_lon = 1.0 / phi2
    range_lat = 1.0 / (phi2 * math.sqrt(phi3))
    smooth = raw_to_smooth(float(raw[3]), smooth_bounds)
    if str(nugget_mode) == "free":
        nugget = float(np.exp(raw[4]))
    elif str(nugget_mode) == "fixed0":
        nugget = 0.0
    else:
        nugget = float(fixed_nugget)
    return NaturalParams(
        sigmasq=float(sigmasq),
        range_lat=float(range_lat),
        range_lon=float(range_lon),
        smooth=float(smooth),
        nugget=float(nugget),
        phi1=float(phi1),
        phi2=float(phi2),
        phi3=float(phi3),
    )


def make_mean_design(coords: np.ndarray, mean_design: str = "lat") -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    ones = np.ones((coords.shape[0], 1), dtype=np.float64)
    lat = coords[:, 0:1] - float(np.mean(coords[:, 0]))
    lon = coords[:, 1:2] - float(np.mean(coords[:, 1]))
    design = str(mean_design)
    if design == "constant":
        return ones
    if design in {"lat", "base", "hour_spatial"}:
        return np.hstack([ones, lat])
    if design == "latlon":
        return np.hstack([ones, lat, lon])
    raise ValueError(f"Unsupported mean_design={mean_design!r}")


def pairwise_deltas(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(coords, dtype=np.float64)
    d_lat = coords[:, None, 0] - coords[None, :, 0]
    d_lon = coords[:, None, 1] - coords[None, :, 1]
    return d_lat, d_lon


def matern_corr_bessel(scaled_distance: np.ndarray, smooth: float) -> np.ndarray:
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


def covariance_from_deltas(
    d_lat: np.ndarray,
    d_lon: np.ndarray,
    params: NaturalParams,
    jitter: float = 1e-6,
) -> np.ndarray:
    if (
        params.sigmasq <= 0.0
        or params.range_lat <= 0.0
        or params.range_lon <= 0.0
        or params.smooth <= 0.0
        or params.nugget < 0.0
    ):
        raise ValueError("Invalid covariance parameters")
    scaled = np.sqrt((d_lat / params.range_lat) ** 2 + (d_lon / params.range_lon) ** 2)
    corr = matern_corr_bessel(scaled, params.smooth)
    cov = float(params.sigmasq) * corr
    diag = np.diag_indices_from(cov)
    cov[diag] += float(params.nugget) + float(jitter)
    return cov


def _bounds_ok(params: NaturalParams, bounds: dict[str, tuple[float, float]]) -> bool:
    for name in ("sigmasq", "range_lat", "range_lon", "smooth", "nugget"):
        if name not in bounds:
            continue
        lo, hi = bounds[name]
        val = getattr(params, name)
        if val < float(lo) or val > float(hi):
            return False
    return True


def profiled_full_nll(
    raw: Sequence[float],
    y: np.ndarray,
    coords: np.ndarray,
    nugget_mode: str,
    fixed_nugget: float,
    smooth_bounds: tuple[float, float],
    param_bounds: dict[str, tuple[float, float]],
    mean_design: str = "lat",
    jitter: float = 1e-6,
    d_lat: np.ndarray | None = None,
    d_lon: np.ndarray | None = None,
    scale_by_n: bool = True,
) -> float:
    y = np.asarray(y, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    n = int(y.shape[0])
    if n <= 0:
        return PENALTY
    try:
        params = natural_from_raw(raw, nugget_mode, fixed_nugget, smooth_bounds)
        if not _bounds_ok(params, param_bounds):
            return PENALTY
        if d_lat is None or d_lon is None:
            d_lat, d_lon = pairwise_deltas(coords)
        cov = covariance_from_deltas(d_lat, d_lon, params, jitter=jitter)
        X = make_mean_design(coords, mean_design)
        c, lower = cho_factor(cov, lower=True, check_finite=False)
        kinv_y = cho_solve((c, lower), y, check_finite=False)
        kinv_X = cho_solve((c, lower), X, check_finite=False)
        xt_k_x = X.T @ kinv_X
        xt_k_y = X.T @ kinv_y
        beta = np.linalg.solve(xt_k_x + np.eye(X.shape[1]) * 1e-8, xt_k_y)
        resid = y - X @ beta
        kinv_resid = cho_solve((c, lower), resid, check_finite=False)
        quad = float(resid.T @ kinv_resid)
        logdet = 2.0 * float(np.sum(np.log(np.diag(c))))
        nll = 0.5 * (n * TWO_PI_LOG + logdet + quad)
        if not np.isfinite(nll):
            return PENALTY
        return float(nll / n) if scale_by_n else float(nll)
    except (LinAlgError, np.linalg.LinAlgError, ValueError, FloatingPointError, OverflowError):
        return PENALTY


def _default_start_grid(
    var_y: float,
    range_lat_init: float,
    range_lon_init: float,
    smooth_init: float,
    nugget_init: float,
    nugget_mode: str,
    smooth_bounds: tuple[float, float],
    n_restarts: int,
) -> list[np.ndarray]:
    var_y = max(float(var_y), 1e-8)
    starts = [
        (0.75 * var_y, range_lat_init, range_lon_init, smooth_init, nugget_init),
        (0.90 * var_y, 0.60, 0.60, 0.30, 0.10 * var_y),
        (0.60 * var_y, 0.25, 0.50, 0.80, 0.20 * var_y),
        (0.60 * var_y, 0.50, 0.25, 0.80, 0.20 * var_y),
        (1.00 * var_y, 1.00, 1.00, 1.20, 0.05 * var_y),
    ]
    out: list[np.ndarray] = []
    for sigmasq, rlat, rlon, smooth, nugget in starts[: max(1, int(n_restarts))]:
        out.append(
            raw_from_natural(
                sigmasq=sigmasq,
                range_lat=rlat,
                range_lon=rlon,
                smooth=smooth,
                nugget=max(float(nugget), 1e-8),
                nugget_mode=nugget_mode,
                smooth_bounds=smooth_bounds,
            )
        )
    return out


def _raw_bounds(
    nugget_mode: str,
    var_y: float,
    range_bounds: tuple[float, float],
    nugget_bounds: tuple[float, float] | None,
) -> list[tuple[float, float]]:
    min_range, max_range = map(float, range_bounds)
    if min_range <= 0.0 or max_range <= min_range:
        raise ValueError(f"range_bounds must be positive/increasing, got {range_bounds}")
    log_phi2_bounds = (math.log(1.0 / max_range), math.log(1.0 / min_range))
    bounds = [
        (-40.0, 40.0),          # log phi1
        log_phi2_bounds,        # log phi2
        (-8.0, 8.0),            # log phi3
        (-8.0, 8.0),            # smooth logit within smooth_bounds
    ]
    if str(nugget_mode) == "free":
        if nugget_bounds is None:
            nugget_bounds = (max(float(var_y) * 1e-8, 1e-10), max(float(var_y) * 1e3, 1e-8))
        bounds.append((math.log(float(nugget_bounds[0])), math.log(float(nugget_bounds[1]))))
    return bounds


def _select_best(results: Iterable[dict]) -> dict:
    finite = [r for r in results if np.isfinite(r.get("loss", np.inf))]
    if not finite:
        return {
            "success": False,
            "loss": np.inf,
            "nll": np.inf,
            "message": "all_restarts_failed",
            "n_eval": 0,
        }
    return min(finite, key=lambda r: float(r["loss"]))


def fit_full_matern(
    y: np.ndarray,
    coords: np.ndarray,
    nugget_mode: str = "free",
    fixed_nugget: float = 0.0,
    mean_design: str = "lat",
    smooth_bounds: tuple[float, float] = (0.05, 2.5),
    range_bounds: tuple[float, float] = (0.03, 5.0),
    range_lat_init: float = 0.35,
    range_lon_init: float = 0.35,
    smooth_init: float = 0.5,
    nugget_init: float | None = None,
    jitter: float = 1e-6,
    n_restarts: int = 3,
    maxiter: int = 60,
    maxfun: int = 240,
    maxls: int = 20,
    maxcor: int = 20,
    method: str = "L-BFGS-B",
) -> dict:
    y = np.asarray(y, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    var_y = max(float(np.nanvar(y, ddof=1)), 1e-8)
    if nugget_init is None:
        nugget_init = 0.20 * var_y
    starts = _default_start_grid(
        var_y=var_y,
        range_lat_init=range_lat_init,
        range_lon_init=range_lon_init,
        smooth_init=smooth_init,
        nugget_init=float(nugget_init),
        nugget_mode=nugget_mode,
        smooth_bounds=smooth_bounds,
        n_restarts=n_restarts,
    )
    d_lat, d_lon = pairwise_deltas(coords)
    param_bounds = {
        "sigmasq": (1e-12, max(var_y * 1e5, 1e-6)),
        "range_lat": range_bounds,
        "range_lon": range_bounds,
        "smooth": smooth_bounds,
        "nugget": (0.0, max(var_y * 1e4, 1e-6)),
    }
    bounds = _raw_bounds(nugget_mode, var_y, range_bounds, None)

    def objective(raw: np.ndarray) -> float:
        return profiled_full_nll(
            raw,
            y=y,
            coords=coords,
            nugget_mode=nugget_mode,
            fixed_nugget=fixed_nugget,
            smooth_bounds=smooth_bounds,
            param_bounds=param_bounds,
            mean_design=mean_design,
            jitter=jitter,
            d_lat=d_lat,
            d_lon=d_lon,
            scale_by_n=True,
        )

    results = []
    options = {"maxiter": int(maxiter), "ftol": 1e-7}
    if str(method).upper() == "L-BFGS-B":
        options["maxls"] = int(maxls)
        options["maxcor"] = int(maxcor)
    if int(maxfun) > 0:
        options["maxfun"] = int(maxfun)
    for start in starts:
        res = minimize(
            objective,
            start,
            method=method,
            bounds=bounds if method.upper() == "L-BFGS-B" else None,
            options=options,
        )
        loss = float(res.fun) if np.isfinite(res.fun) else np.inf
        nll = profiled_full_nll(
            res.x,
            y=y,
            coords=coords,
            nugget_mode=nugget_mode,
            fixed_nugget=fixed_nugget,
            smooth_bounds=smooth_bounds,
            param_bounds=param_bounds,
            mean_design=mean_design,
            jitter=jitter,
            d_lat=d_lat,
            d_lon=d_lon,
            scale_by_n=False,
        )
        params = natural_from_raw(res.x, nugget_mode, fixed_nugget, smooth_bounds)
        rec = params.to_record()
        rec.update(
            {
                "success": bool(res.success and np.isfinite(loss)),
                "loss": loss,
                "nll": float(nll),
                "message": str(res.message),
                "n_eval": int(getattr(res, "nfev", 0)),
                "raw_params": np.asarray(res.x, dtype=float).tolist(),
            }
        )
        results.append(rec)
    best = _select_best(results)
    best["n_restarts"] = int(len(starts))
    return best


def vecchia_batches_to_numpy(model) -> list[dict]:
    batches = []
    for batch in getattr(model, "_cluster_batches", []):
        coords = batch.coords.detach().cpu().numpy().astype(np.float64, copy=False)
        d_lat = coords[:, :, None, 0] - coords[:, None, :, 0]
        d_lon = coords[:, :, None, 1] - coords[:, None, :, 1]
        batches.append(
            {
                "max_cond_points": int(batch.max_cond_points),
                "target_size": int(batch.target_size),
                "coords": coords,
                "d_lat": d_lat,
                "d_lon": d_lon,
                "X": batch.X.detach().cpu().numpy().astype(np.float64, copy=False),
                "y": batch.y.detach().cpu().numpy().astype(np.float64, copy=False),
            }
        )
    return batches


def profiled_vecchia_cluster_nll(
    raw: Sequence[float],
    batches: Sequence[dict],
    n_features: int,
    nugget_mode: str,
    fixed_nugget: float,
    smooth_bounds: tuple[float, float],
    param_bounds: dict[str, tuple[float, float]],
    jitter: float = 1e-6,
) -> float:
    try:
        params = natural_from_raw(raw, nugget_mode, fixed_nugget, smooth_bounds)
        if not _bounds_ok(params, param_bounds):
            return PENALTY
        xt_sinv_x = np.zeros((int(n_features), int(n_features)), dtype=np.float64)
        xt_sinv_y = np.zeros((int(n_features), 1), dtype=np.float64)
        yt_sinv_y = 0.0
        logdet = 0.0
        total_n = 0

        for batch in batches:
            coords_all = batch["coords"]
            X_all = batch["X"]
            y_all = batch["y"]
            d_lat_all = batch.get("d_lat")
            d_lon_all = batch.get("d_lon")
            m = int(batch["max_cond_points"])
            t = int(batch["target_size"])
            target = slice(m, m + t)
            for i in range(coords_all.shape[0]):
                X = X_all[i]
                y = y_all[i]
                if d_lat_all is None or d_lon_all is None:
                    d_lat, d_lon = pairwise_deltas(coords_all[i])
                else:
                    d_lat, d_lon = d_lat_all[i], d_lon_all[i]
                K = covariance_from_deltas(d_lat, d_lon, params, jitter=jitter)
                c, lower = cho_factor(K, lower=True, check_finite=False)
                z_X = solve_triangular(c, X, lower=lower, check_finite=False)
                z_y = solve_triangular(c, y, lower=lower, check_finite=False)
                u_X = z_X[target, :]
                u_y = z_y[target, :]
                diag = np.diag(c)[target]
                if np.any(diag <= 1e-12) or not np.all(np.isfinite(diag)):
                    return PENALTY
                xt_sinv_x += u_X.T @ u_X
                xt_sinv_y += u_X.T @ u_y
                yt_sinv_y += float((u_y.T @ u_y).squeeze())
                logdet += 2.0 * float(np.sum(np.log(diag)))
                total_n += int(t)

        if total_n <= 0:
            return PENALTY
        beta = np.linalg.solve(xt_sinv_x + np.eye(int(n_features)) * 1e-8, xt_sinv_y)
        quad = yt_sinv_y - 2.0 * float((beta.T @ xt_sinv_y).squeeze()) + float((beta.T @ xt_sinv_x @ beta).squeeze())
        loss = 0.5 * (logdet + quad) / total_n
        return float(loss) if np.isfinite(loss) else PENALTY
    except (LinAlgError, np.linalg.LinAlgError, ValueError, FloatingPointError, OverflowError):
        return PENALTY


def fit_vecchia_matern_from_batches(
    batches: Sequence[dict],
    n_features: int,
    y_var: float,
    nugget_mode: str = "free",
    fixed_nugget: float = 0.0,
    smooth_bounds: tuple[float, float] = (0.05, 2.5),
    range_bounds: tuple[float, float] = (0.03, 5.0),
    range_lat_init: float = 0.35,
    range_lon_init: float = 0.35,
    smooth_init: float = 0.5,
    nugget_init: float | None = None,
    jitter: float = 1e-6,
    n_restarts: int = 3,
    maxiter: int = 60,
    maxfun: int = 240,
    maxls: int = 20,
    maxcor: int = 20,
    method: str = "L-BFGS-B",
) -> dict:
    var_y = max(float(y_var), 1e-8)
    if nugget_init is None:
        nugget_init = 0.20 * var_y
    starts = _default_start_grid(
        var_y=var_y,
        range_lat_init=range_lat_init,
        range_lon_init=range_lon_init,
        smooth_init=smooth_init,
        nugget_init=float(nugget_init),
        nugget_mode=nugget_mode,
        smooth_bounds=smooth_bounds,
        n_restarts=n_restarts,
    )
    param_bounds = {
        "sigmasq": (1e-12, max(var_y * 1e5, 1e-6)),
        "range_lat": range_bounds,
        "range_lon": range_bounds,
        "smooth": smooth_bounds,
        "nugget": (0.0, max(var_y * 1e4, 1e-6)),
    }
    bounds = _raw_bounds(nugget_mode, var_y, range_bounds, None)

    def objective(raw: np.ndarray) -> float:
        return profiled_vecchia_cluster_nll(
            raw,
            batches=batches,
            n_features=n_features,
            nugget_mode=nugget_mode,
            fixed_nugget=fixed_nugget,
            smooth_bounds=smooth_bounds,
            param_bounds=param_bounds,
            jitter=jitter,
        )

    results = []
    options = {"maxiter": int(maxiter), "ftol": 1e-7}
    if str(method).upper() == "L-BFGS-B":
        options["maxls"] = int(maxls)
        options["maxcor"] = int(maxcor)
    if int(maxfun) > 0:
        options["maxfun"] = int(maxfun)
    for start in starts:
        res = minimize(
            objective,
            start,
            method=method,
            bounds=bounds if method.upper() == "L-BFGS-B" else None,
            options=options,
        )
        loss = float(res.fun) if np.isfinite(res.fun) else np.inf
        params = natural_from_raw(res.x, nugget_mode, fixed_nugget, smooth_bounds)
        rec = params.to_record()
        rec.update(
            {
                "success": bool(res.success and np.isfinite(loss)),
                "loss": loss,
                "nll": float(loss),
                "message": str(res.message),
                "n_eval": int(getattr(res, "nfev", 0)),
                "raw_params": np.asarray(res.x, dtype=float).tolist(),
            }
        )
        results.append(rec)
    best = _select_best(results)
    best["n_restarts"] = int(len(starts))
    return best
