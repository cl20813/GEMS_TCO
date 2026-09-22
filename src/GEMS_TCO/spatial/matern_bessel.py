"""Direct-Bessel anisotropic Matérn likelihood for pure-spatial inference.

Parameterization follows the existing space-time code:

    phi2 = 1 / range_lon
    phi3 = (range_lon / range_lat)^2
    phi1 = signal_variance * phi2

so

    signal_variance = phi1 / phi2
    range_lon = 1 / phi2
    range_lat = 1 / (phi2 * sqrt(phi3))

The smoothness parameter is estimated through a bounded logit transform, and
the same covariance implementation is used to validate block-Vecchia fits.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
from scipy.linalg import LinAlgError, cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize
from scipy.special import gammaln, kv

TWO_PI_LOG = float(np.log(2.0 * np.pi))
PENALTY = 1.0e30
_NUGGET_MODES = frozenset({"free", "fixed", "fixed0"})

__all__ = [
    "MaternParameters",
    "fit_full_matern",
    "fit_vecchia_matern_from_batches",
    "matern_corr_bessel",
    "vecchia_batches_to_numpy",
]


def _validate_nugget_mode(nugget_mode: str, fixed_nugget: float = 0.0) -> str:
    mode = str(nugget_mode)
    if mode not in _NUGGET_MODES:
        raise ValueError(f"nugget_mode must be one of {sorted(_NUGGET_MODES)}, got {mode!r}")
    fixed_nugget = float(fixed_nugget)
    if mode == "fixed" and (not np.isfinite(fixed_nugget) or fixed_nugget < 0.0):
        raise ValueError("fixed_nugget must be finite and non-negative in fixed mode")
    return mode


def _validate_smooth_bounds(smooth_bounds: tuple[float, float]) -> tuple[float, float]:
    lo, hi = map(float, smooth_bounds)
    if not np.isfinite([lo, hi]).all() or lo <= 0.0 or not lo < hi:
        raise ValueError(
            f"smooth_bounds must be finite, positive, and increasing, got {smooth_bounds}"
        )
    return lo, hi


def _validate_observations(y: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=np.float64)
    coords = np.asarray(coords, dtype=np.float64)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1:
        raise ValueError(f"y must have shape (n,) or (n, 1), got {y.shape}")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"coords must have shape (n, 2), got {coords.shape}")
    if coords.shape[0] != y.shape[0]:
        raise ValueError(f"y and coords lengths differ: {y.shape[0]} != {coords.shape[0]}")
    if y.size == 0:
        raise ValueError("at least one observation is required")
    if not np.isfinite(y).all() or not np.isfinite(coords).all():
        raise ValueError("y and coords must contain only finite values")
    return y, coords


@dataclass
class MaternParameters:
    """Natural and internal parameters for the anisotropic Matérn model.

    ``range_lat`` and ``range_lon`` use the standard Matérn convention in
    which the Bessel-function argument is ``sqrt(2 * smooth) * distance``.
    The ``phi`` fields retain the optimization parameterization used by the
    original space-time implementation.
    """

    signal_variance: float
    range_lat: float
    range_lon: float
    smooth: float
    nugget: float
    phi1: float
    phi2: float
    phi3: float

    def to_record(self) -> dict[str, float]:
        """Return serializable natural parameters and signal standard deviation."""

        out = asdict(self)
        out["signal_standard_deviation"] = math.sqrt(max(float(self.signal_variance), 0.0))
        return out


def _clip_prob(x: float) -> float:
    return min(max(float(x), 1e-12), 1.0 - 1e-12)


def smooth_to_raw(smooth: float, smooth_bounds: tuple[float, float]) -> float:
    """Map bounded Matérn smoothness to an unconstrained logit parameter."""

    lo, hi = _validate_smooth_bounds(smooth_bounds)
    smooth = float(smooth)
    if not np.isfinite(smooth) or smooth < lo or smooth > hi:
        raise ValueError(f"smooth must lie within smooth_bounds={smooth_bounds}, got {smooth}")
    p = _clip_prob((smooth - lo) / (hi - lo))
    return float(math.log(p / (1.0 - p)))


def raw_to_smooth(raw: float, smooth_bounds: tuple[float, float]) -> float:
    """Map an unconstrained logit parameter to bounded Matérn smoothness."""

    lo, hi = _validate_smooth_bounds(smooth_bounds)
    z = float(raw)
    if not np.isfinite(z):
        raise ValueError(f"raw smoothness must be finite, got {raw}")
    if z >= 0:
        e = math.exp(-z)
        p = 1.0 / (1.0 + e)
    else:
        e = math.exp(z)
        p = e / (1.0 + e)
    return float(lo + (hi - lo) * p)


def raw_from_natural(
    signal_variance: float,
    range_lat: float,
    range_lon: float,
    smooth: float,
    nugget: float,
    nugget_mode: str,
    smooth_bounds: tuple[float, float],
) -> np.ndarray:
    """Convert natural covariance parameters to the optimizer parameterization."""

    mode = _validate_nugget_mode(nugget_mode, fixed_nugget=nugget)
    values = {
        "signal_variance": float(signal_variance),
        "range_lat": float(range_lat),
        "range_lon": float(range_lon),
    }
    invalid = [name for name, value in values.items() if not np.isfinite(value) or value <= 0]
    if invalid:
        raise ValueError(f"{', '.join(invalid)} must be finite and positive")
    signal_variance = values["signal_variance"]
    range_lat = values["range_lat"]
    range_lon = values["range_lon"]
    phi2 = 1.0 / range_lon
    phi3 = (range_lon / range_lat) ** 2
    phi1 = signal_variance * phi2
    vals = [
        math.log(max(phi1, 1e-300)),
        math.log(max(phi2, 1e-300)),
        math.log(max(phi3, 1e-300)),
        smooth_to_raw(float(smooth), smooth_bounds),
    ]
    if mode == "free":
        nugget = float(nugget)
        if not np.isfinite(nugget) or nugget <= 0.0:
            raise ValueError("nugget must be finite and positive in free mode")
        vals.append(math.log(nugget))
    return np.asarray(vals, dtype=np.float64)


def natural_from_raw(
    raw: Sequence[float],
    nugget_mode: str,
    fixed_nugget: float,
    smooth_bounds: tuple[float, float],
) -> MaternParameters:
    """Convert optimizer parameters to validated natural Matérn parameters."""

    raw = np.asarray(raw, dtype=np.float64)
    mode = _validate_nugget_mode(nugget_mode, fixed_nugget)
    expected = 5 if mode == "free" else 4
    if raw.ndim != 1 or raw.size != expected:
        raise ValueError(f"raw must have shape ({expected},) in {mode!r} mode, got {raw.shape}")
    if not np.isfinite(raw).all():
        raise ValueError("raw parameters must be finite")
    try:
        phi1 = math.exp(float(raw[0]))
        phi2 = math.exp(float(raw[1]))
        phi3 = math.exp(float(raw[2]))
    except OverflowError as exc:
        raise ValueError("raw parameters overflow the natural parameterization") from exc
    if min(phi1, phi2, phi3) <= 0.0 or not np.isfinite([phi1, phi2, phi3]).all():
        raise ValueError("raw parameters underflow or overflow the natural parameterization")
    signal_variance = phi1 / phi2
    range_lon = 1.0 / phi2
    range_lat = 1.0 / (phi2 * math.sqrt(phi3))
    smooth = raw_to_smooth(float(raw[3]), smooth_bounds)
    if mode == "free":
        try:
            nugget = math.exp(float(raw[4]))
        except OverflowError as exc:
            raise ValueError("raw nugget overflows the natural parameterization") from exc
        if nugget <= 0.0 or not math.isfinite(nugget):
            raise ValueError("raw nugget underflows or overflows the natural parameterization")
    elif mode == "fixed0":
        nugget = 0.0
    else:
        nugget = float(fixed_nugget)
    return MaternParameters(
        signal_variance=float(signal_variance),
        range_lat=float(range_lat),
        range_lon=float(range_lon),
        smooth=float(smooth),
        nugget=float(nugget),
        phi1=float(phi1),
        phi2=float(phi2),
        phi3=float(phi3),
    )


def make_mean_design(coords: np.ndarray, mean_design: str = "lat") -> np.ndarray:
    """Construct a centered spatial mean-design matrix.

    Supported designs are ``"constant"`` (or ``"intercept"``), ``"lat"``,
    and ``"latlon"``.  Coordinate columns are centered before inclusion.
    """

    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError(f"coords must be a finite array with shape (n, 2), got {coords.shape}")
    ones = np.ones((coords.shape[0], 1), dtype=np.float64)
    lat = coords[:, 0:1] - float(np.mean(coords[:, 0]))
    lon = coords[:, 1:2] - float(np.mean(coords[:, 1]))
    design = str(mean_design)
    if design in {"constant", "intercept"}:
        return ones
    if design == "lat":
        return np.hstack([ones, lat])
    if design == "latlon":
        return np.hstack([ones, lat, lon])
    raise ValueError(f"Unsupported mean_design={mean_design!r}")


def pairwise_deltas(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return signed pairwise latitude and longitude differences."""

    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2 or not np.isfinite(coords).all():
        raise ValueError(f"coords must be a finite array with shape (n, 2), got {coords.shape}")
    d_lat = coords[:, None, 0] - coords[None, :, 0]
    d_lon = coords[:, None, 1] - coords[None, :, 1]
    return d_lat, d_lon


def matern_corr_bessel(scaled_distance: np.ndarray, smooth: float) -> np.ndarray:
    """Evaluate the standard Matérn correlation at nonnegative distances."""

    r = np.asarray(scaled_distance, dtype=np.float64)
    nu = float(smooth)
    if not np.isfinite(nu) or nu <= 0.0:
        raise ValueError(f"smooth must be finite and positive, got {smooth}")
    if not np.isfinite(r).all() or np.any(r < 0.0):
        raise ValueError("scaled_distance must contain only finite, non-negative values")
    out = np.empty_like(r)
    zero = r == 0.0
    out[zero] = 1.0
    z = r[~zero]
    if z.size:
        arg = np.sqrt(2.0 * nu) * z
        vals = np.empty_like(arg)
        near_zero = arg <= np.sqrt(np.finfo(np.float64).eps)
        vals[near_zero] = 1.0
        regular = ~near_zero
        if np.any(regular):
            arg_regular = arg[regular]
            log_prefactor = (1.0 - nu) * math.log(2.0) - float(gammaln(nu))
            with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
                log_vals = log_prefactor + nu * np.log(arg_regular) + np.log(kv(nu, arg_regular))
                vals[regular] = np.exp(log_vals)
        out[~zero] = np.nan_to_num(vals, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(out, 0.0, 1.0)


def covariance_from_deltas(
    d_lat: np.ndarray,
    d_lon: np.ndarray,
    params: MaternParameters,
    jitter: float = 1e-6,
) -> np.ndarray:
    """Build an anisotropic Matérn covariance matrix from coordinate deltas.

    ``jitter`` is numerical diagonal stabilization and is distinct from the
    statistical nugget stored in ``params``.
    """

    param_values = np.asarray(
        [
            params.signal_variance,
            params.range_lat,
            params.range_lon,
            params.smooth,
            params.nugget,
        ],
        dtype=np.float64,
    )
    if (
        not np.isfinite(param_values).all()
        or np.any(param_values[:4] <= 0.0)
        or params.nugget < 0.0
    ):
        raise ValueError("Invalid covariance parameters")
    d_lat = np.asarray(d_lat, dtype=np.float64)
    d_lon = np.asarray(d_lon, dtype=np.float64)
    if d_lat.shape != d_lon.shape or d_lat.ndim != 2 or d_lat.shape[0] != d_lat.shape[1]:
        raise ValueError("d_lat and d_lon must be square arrays with identical shapes")
    if not np.isfinite(d_lat).all() or not np.isfinite(d_lon).all():
        raise ValueError("d_lat and d_lon must contain only finite values")
    jitter = float(jitter)
    if not np.isfinite(jitter) or jitter < 0.0:
        raise ValueError("jitter must be finite and non-negative")
    scaled = np.sqrt((d_lat / params.range_lat) ** 2 + (d_lon / params.range_lon) ** 2)
    corr = matern_corr_bessel(scaled, params.smooth)
    cov = float(params.signal_variance) * corr
    diag = np.diag_indices_from(cov)
    cov[diag] += float(params.nugget) + float(jitter)
    return cov


def _bounds_ok(params: MaternParameters, bounds: dict[str, tuple[float, float]]) -> bool:
    for name in ("signal_variance", "range_lat", "range_lon", "smooth", "nugget"):
        if name not in bounds:
            continue
        lo, hi = bounds[name]
        val = getattr(params, name)
        if not np.isfinite([lo, hi, val]).all() or val < float(lo) or val > float(hi):
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
    """Evaluate the profiled full Gaussian negative log-likelihood.

    The mean coefficients are profiled by generalized least squares.  Invalid
    covariance parameters or factorizations return the module-level
    ``PENALTY`` so numerical optimizers can reject the candidate.
    """

    y, coords = _validate_observations(y, coords)
    _validate_nugget_mode(nugget_mode, fixed_nugget)
    jitter = float(jitter)
    if not math.isfinite(jitter) or jitter < 0.0:
        raise ValueError("jitter must be finite and non-negative")
    n = int(y.shape[0])
    X = make_mean_design(coords, mean_design)
    if n < X.shape[1] or np.linalg.matrix_rank(X) < X.shape[1]:
        raise ValueError("mean design is rank-deficient for the supplied coordinates")
    if (d_lat is None) != (d_lon is None):
        raise ValueError("d_lat and d_lon must either both be provided or both be omitted")
    if d_lat is None:
        d_lat, d_lon = pairwise_deltas(coords)
    try:
        params = natural_from_raw(raw, nugget_mode, fixed_nugget, smooth_bounds)
        if not _bounds_ok(params, param_bounds):
            return PENALTY
        cov = covariance_from_deltas(d_lat, d_lon, params, jitter=jitter)
        c, lower = cho_factor(cov, lower=True, check_finite=False)
        kinv_y = cho_solve((c, lower), y, check_finite=False)
        kinv_X = cho_solve((c, lower), X, check_finite=False)
        xt_k_x = X.T @ kinv_X
        xt_k_y = X.T @ kinv_y
        beta = np.linalg.solve(xt_k_x, xt_k_y)
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
    var_y = float(var_y)
    if not math.isfinite(var_y) or var_y <= 0.0:
        raise ValueError("var_y must be finite and positive")
    n_restarts = int(n_restarts)
    if not 1 <= n_restarts <= 5:
        raise ValueError("n_restarts must be between 1 and 5")
    lo_smooth, hi_smooth = _validate_smooth_bounds(smooth_bounds)

    def bounded_smooth(value: float) -> float:
        return float(np.clip(float(value), lo_smooth, hi_smooth))

    starts = [
        (0.75 * var_y, range_lat_init, range_lon_init, smooth_init, nugget_init),
        (0.90 * var_y, 0.60, 0.60, 0.30, 0.10 * var_y),
        (0.60 * var_y, 0.25, 0.50, 0.80, 0.20 * var_y),
        (0.60 * var_y, 0.50, 0.25, 0.80, 0.20 * var_y),
        (1.00 * var_y, 1.00, 1.00, 1.20, 0.05 * var_y),
    ]
    out: list[np.ndarray] = []
    for signal_variance, rlat, rlon, smooth, nugget in starts[:n_restarts]:
        out.append(
            raw_from_natural(
                signal_variance=signal_variance,
                range_lat=rlat,
                range_lon=rlon,
                smooth=bounded_smooth(smooth),
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
    mode = _validate_nugget_mode(nugget_mode)
    min_range, max_range = map(float, range_bounds)
    if not np.isfinite([min_range, max_range]).all() or min_range <= 0.0 or max_range <= min_range:
        raise ValueError(f"range_bounds must be positive/increasing, got {range_bounds}")
    log_phi2_bounds = (math.log(1.0 / max_range), math.log(1.0 / min_range))
    bounds = [
        (-40.0, 40.0),  # log phi1
        log_phi2_bounds,  # log phi2
        (-8.0, 8.0),  # log phi3
        (-8.0, 8.0),  # smooth logit within smooth_bounds
    ]
    if mode == "free":
        if nugget_bounds is None:
            nugget_bounds = (max(float(var_y) * 1e-8, 1e-10), max(float(var_y) * 1e3, 1e-8))
        if (
            not np.isfinite(nugget_bounds).all()
            or float(nugget_bounds[0]) <= 0.0
            or float(nugget_bounds[1]) <= float(nugget_bounds[0])
        ):
            raise ValueError("nugget_bounds must be finite, positive, and increasing")
        bounds.append((math.log(float(nugget_bounds[0])), math.log(float(nugget_bounds[1]))))
    return bounds


_RESTART_EVALUATION_ERRORS = (
    LinAlgError,
    np.linalg.LinAlgError,
    FloatingPointError,
    OverflowError,
    TypeError,
    ValueError,
)
_RESTART_OPTIMIZER_ERRORS = (
    LinAlgError,
    np.linalg.LinAlgError,
    FloatingPointError,
    OverflowError,
    RuntimeError,
)


def _restart_record(
    raw: Sequence[float],
    *,
    converged: bool,
    optimizer_message: str,
    optimizer_status: int | None,
    n_eval: int,
    loss_fn: Callable[[np.ndarray], float],
    nll_fn: Callable[[np.ndarray], float],
    parameter_fn: Callable[[np.ndarray], MaternParameters],
) -> dict:
    """Evaluate one optimizer candidate without confusing validity and convergence."""

    try:
        raw_array = np.asarray(raw, dtype=np.float64)
        raw_record = raw_array.tolist()
    except _RESTART_EVALUATION_ERRORS as exc:
        return {
            "success": False,
            "valid": False,
            "converged": bool(converged),
            "loss": np.inf,
            "nll": np.inf,
            "message": str(optimizer_message),
            "evaluation_message": (f"raw_parameter_conversion_failed: {type(exc).__name__}: {exc}"),
            "optimizer_status": optimizer_status,
            "n_eval": int(n_eval),
            "raw_params": None,
        }

    loss = np.inf
    nll = np.inf
    parameters: dict[str, float] = {}
    evaluation_message = "valid"
    conversion_succeeded = False
    try:
        loss = float(loss_fn(raw_array))
        nll = float(nll_fn(raw_array))
        parameters = parameter_fn(raw_array).to_record()
        conversion_succeeded = True
    except _RESTART_EVALUATION_ERRORS as exc:
        evaluation_message = f"evaluation_failed: {type(exc).__name__}: {exc}"

    valid = bool(
        conversion_succeeded
        and np.isfinite(loss)
        and np.isfinite(nll)
        and loss < PENALTY
        and nll < PENALTY
    )
    if conversion_succeeded and not valid:
        evaluation_message = "nonfinite_or_penalized_evaluation"

    record: dict = dict(parameters)
    record.update(
        {
            # ``success`` retains its historical meaning: a usable converged fit.
            # ``valid`` and ``converged`` expose the two independent facts.
            "success": bool(valid and converged),
            "valid": valid,
            "converged": bool(converged),
            "loss": float(loss),
            "nll": float(nll),
            "message": str(optimizer_message),
            "evaluation_message": evaluation_message,
            "optimizer_status": optimizer_status,
            "n_eval": int(n_eval),
            "raw_params": raw_record,
        }
    )
    return record


def _run_scipy_restarts(
    starts: Sequence[np.ndarray],
    *,
    objective: Callable[[np.ndarray], float],
    total_nll: Callable[[np.ndarray], float],
    parameter_fn: Callable[[np.ndarray], MaternParameters],
    method: str,
    bounds: Sequence[tuple[float, float]] | None,
    options: dict,
) -> list[dict]:
    """Run and record independent SciPy restarts, including numerical failures."""

    records: list[dict] = []
    for start in starts:
        try:
            result = minimize(
                objective,
                start,
                method=method,
                bounds=bounds,
                options=options,
            )
            raw = result.x
            converged = bool(result.success)
            message = str(result.message)
            status_value = getattr(result, "status", None)
            status = None if status_value is None else int(status_value)
            n_eval = int(getattr(result, "nfev", 0))
        except _RESTART_OPTIMIZER_ERRORS as exc:
            # The initial state remains a legitimate finite candidate even when
            # SciPy aborts before returning an OptimizeResult.
            raw = start
            converged = False
            message = f"optimizer_failed: {type(exc).__name__}: {exc}"
            status = None
            n_eval = 0

        records.append(
            _restart_record(
                raw,
                converged=converged,
                optimizer_message=message,
                optimizer_status=status,
                n_eval=n_eval,
                loss_fn=objective,
                nll_fn=total_nll,
                parameter_fn=parameter_fn,
            )
        )
    return records


def _select_best(results: Iterable[dict]) -> dict:
    """Select the lowest-loss valid state, independently of optimizer status."""

    records = list(results)
    valid = [
        record
        for record in records
        if bool(record.get("valid", False))
        and np.isfinite(record.get("loss", np.inf))
        and np.isfinite(record.get("nll", np.inf))
        and float(record["loss"]) < PENALTY
        and float(record["nll"]) < PENALTY
    ]
    if valid:
        best = dict(min(valid, key=lambda record: float(record["loss"])))
    else:
        best = {
            "success": False,
            "valid": False,
            "converged": False,
            "loss": np.inf,
            "nll": np.inf,
            "message": "all_restarts_invalid",
            "evaluation_message": "no_finite_evaluable_state",
            "optimizer_status": None,
            "n_eval": int(sum(int(record.get("n_eval", 0)) for record in records)),
            "raw_params": None,
        }
    best["n_restarts"] = len(records)
    best["n_valid_restarts"] = len(valid)
    best["n_converged_restarts"] = sum(bool(record.get("converged", False)) for record in records)
    best["restart_records"] = records
    return best


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
    """Fit an anisotropic Matérn model by profiled full Gaussian likelihood.

    ``nugget_mode`` may be ``"free"``, ``"fixed"``, or ``"fixed0"``.  The
    returned dictionary records the lowest-loss valid restart, natural
    parameters, optimizer diagnostics, and both normalized loss and total NLL.
    ``valid`` denotes a finite evaluable state, while ``converged`` reports the
    SciPy optimizer status; these are intentionally independent.  Per-restart
    diagnostics are retained in ``restart_records``.
    """

    y, coords = _validate_observations(y, coords)
    mode = _validate_nugget_mode(nugget_mode, fixed_nugget)
    _validate_smooth_bounds(smooth_bounds)
    if y.size < 2:
        raise ValueError("at least two observations are required to estimate variance")
    if int(maxiter) <= 0 or int(maxls) <= 0 or int(maxcor) <= 0:
        raise ValueError("maxiter, maxls, and maxcor must be positive")
    if int(maxfun) == 0 or int(maxfun) < -1:
        raise ValueError("maxfun must be positive or -1 to use the optimizer default")
    jitter = float(jitter)
    if not math.isfinite(jitter) or jitter < 0.0:
        raise ValueError("jitter must be finite and non-negative")
    var_y = max(float(np.var(y, ddof=1)), 1e-8)
    if nugget_init is None:
        nugget_init = 0.20 * var_y
    starts = _default_start_grid(
        var_y=var_y,
        range_lat_init=range_lat_init,
        range_lon_init=range_lon_init,
        smooth_init=smooth_init,
        nugget_init=float(nugget_init),
        nugget_mode=mode,
        smooth_bounds=smooth_bounds,
        n_restarts=n_restarts,
    )
    d_lat, d_lon = pairwise_deltas(coords)
    nugget_upper = max(var_y * 1e4, float(fixed_nugget) if mode == "fixed" else 0.0, 1e-6)
    param_bounds = {
        "signal_variance": (1e-12, max(var_y * 1e5, 1e-6)),
        "range_lat": range_bounds,
        "range_lon": range_bounds,
        "smooth": smooth_bounds,
        "nugget": (0.0, nugget_upper),
    }
    bounds = _raw_bounds(mode, var_y, range_bounds, None)

    def objective(raw: np.ndarray) -> float:
        return profiled_full_nll(
            raw,
            y=y,
            coords=coords,
            nugget_mode=mode,
            fixed_nugget=fixed_nugget,
            smooth_bounds=smooth_bounds,
            param_bounds=param_bounds,
            mean_design=mean_design,
            jitter=jitter,
            d_lat=d_lat,
            d_lon=d_lon,
            scale_by_n=True,
        )

    def total_nll(raw: np.ndarray) -> float:
        return profiled_full_nll(
            raw,
            y=y,
            coords=coords,
            nugget_mode=mode,
            fixed_nugget=fixed_nugget,
            smooth_bounds=smooth_bounds,
            param_bounds=param_bounds,
            mean_design=mean_design,
            jitter=jitter,
            d_lat=d_lat,
            d_lon=d_lon,
            scale_by_n=False,
        )

    def convert_parameters(raw: np.ndarray) -> MaternParameters:
        return natural_from_raw(raw, mode, fixed_nugget, smooth_bounds)

    options = {"maxiter": int(maxiter), "ftol": 1e-7}
    normalized_method = str(method).upper()
    if normalized_method == "L-BFGS-B":
        options["maxls"] = int(maxls)
        options["maxcor"] = int(maxcor)
    if int(maxfun) > 0:
        options["maxfun"] = int(maxfun)
    results = _run_scipy_restarts(
        starts,
        objective=objective,
        total_nll=total_nll,
        parameter_fn=convert_parameters,
        method=str(method),
        bounds=bounds if normalized_method == "L-BFGS-B" else None,
        options=options,
    )
    best = _select_best(results)
    return best


def vecchia_batches_to_numpy(model) -> list[dict]:
    """Export a precomputed spatial Vecchia model for SciPy-based fitting.

    Call ``model.precompute_conditioning_sets()`` before this adapter, then
    pass the returned dictionaries, ``model.n_features``, and a positive
    response-variance estimate to :func:`fit_vecchia_matern_from_batches`.
    """

    if not getattr(model, "is_precomputed", False):
        raise ValueError("model must have precomputed conditioning sets")
    batches = []
    cluster_batches = getattr(model, "_cluster_batches", None)
    if not cluster_batches:
        raise ValueError("model contains no precomputed cluster batches")
    for batch in cluster_batches:
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
                "is_dummy": batch.is_dummy.detach().cpu().numpy().astype(bool, copy=False),
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
    scale_by_n: bool = True,
) -> float:
    """Evaluate the profiled Vecchia negative log-likelihood for exported batches."""

    mode = _validate_nugget_mode(nugget_mode, fixed_nugget)
    if int(n_features) <= 0:
        raise ValueError("n_features must be positive")
    jitter = float(jitter)
    if not math.isfinite(jitter) or jitter < 0.0:
        raise ValueError("jitter must be finite and non-negative")
    try:
        params = natural_from_raw(raw, mode, fixed_nugget, smooth_bounds)
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
            dummy_all = batch.get("is_dummy")
            m = int(batch["max_cond_points"])
            t = int(batch["target_size"])
            if m < 0 or t <= 0:
                raise ValueError(
                    "batch conditioning size must be non-negative and target size positive"
                )
            if coords_all.ndim != 3 or coords_all.shape[2] != 2:
                raise ValueError("batch coords must have shape (batch, points, 2)")
            if X_all.shape[:2] != coords_all.shape[:2] or X_all.shape[2] != int(n_features):
                raise ValueError("batch X shape is incompatible with coords or n_features")
            if y_all.shape != (*coords_all.shape[:2], 1):
                raise ValueError("batch y must have shape (batch, points, 1)")
            if m + t > coords_all.shape[1]:
                raise ValueError("batch target slice extends beyond the point dimension")
            target = slice(m, m + t)
            for i in range(coords_all.shape[0]):
                X = X_all[i]
                y = y_all[i]
                if d_lat_all is None or d_lon_all is None:
                    d_lat, d_lon = pairwise_deltas(coords_all[i])
                else:
                    d_lat, d_lon = d_lat_all[i], d_lon_all[i]
                K = covariance_from_deltas(d_lat, d_lon, params, jitter=jitter)
                if dummy_all is not None:
                    dummy = np.asarray(dummy_all[i], dtype=bool).reshape(-1)
                    if dummy.shape[0] != K.shape[0]:
                        raise ValueError("batch is_dummy shape is incompatible with coords")
                    if np.any(dummy):
                        real_pair = (~dummy)[:, None] & (~dummy)[None, :]
                        K = np.where(real_pair, K, 0.0)
                        K[np.diag_indices_from(K)] += dummy.astype(np.float64)
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
        beta = np.linalg.solve(xt_sinv_x, xt_sinv_y)
        quad = (
            yt_sinv_y
            - 2.0 * float((beta.T @ xt_sinv_y).squeeze())
            + float((beta.T @ xt_sinv_x @ beta).squeeze())
        )
        nll = 0.5 * (total_n * TWO_PI_LOG + logdet + quad)
        loss = nll / total_n if scale_by_n else nll
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
    """Fit the same anisotropic Matérn family to exported Vecchia batches.

    Use :func:`vecchia_batches_to_numpy` to produce the batch dictionaries from
    a maintained block-target spatial Vecchia model after it has precomputed
    its conditioning sets.  The returned ``valid`` and ``converged`` flags
    separately report evaluability and optimizer status; diagnostics for every
    restart are retained in ``restart_records``.
    """

    mode = _validate_nugget_mode(nugget_mode, fixed_nugget)
    _validate_smooth_bounds(smooth_bounds)
    var_y = float(y_var)
    if not math.isfinite(var_y) or var_y <= 0.0:
        raise ValueError("y_var must be finite and positive")
    if int(n_features) <= 0:
        raise ValueError("n_features must be positive")
    if not batches:
        raise ValueError("batches must contain at least one cluster batch")
    if int(maxiter) <= 0 or int(maxls) <= 0 or int(maxcor) <= 0:
        raise ValueError("maxiter, maxls, and maxcor must be positive")
    if int(maxfun) == 0 or int(maxfun) < -1:
        raise ValueError("maxfun must be positive or -1 to use the optimizer default")
    jitter = float(jitter)
    if not math.isfinite(jitter) or jitter < 0.0:
        raise ValueError("jitter must be finite and non-negative")
    if nugget_init is None:
        nugget_init = 0.20 * var_y
    starts = _default_start_grid(
        var_y=var_y,
        range_lat_init=range_lat_init,
        range_lon_init=range_lon_init,
        smooth_init=smooth_init,
        nugget_init=float(nugget_init),
        nugget_mode=mode,
        smooth_bounds=smooth_bounds,
        n_restarts=n_restarts,
    )
    nugget_upper = max(var_y * 1e4, float(fixed_nugget) if mode == "fixed" else 0.0, 1e-6)
    param_bounds = {
        "signal_variance": (1e-12, max(var_y * 1e5, 1e-6)),
        "range_lat": range_bounds,
        "range_lon": range_bounds,
        "smooth": smooth_bounds,
        "nugget": (0.0, nugget_upper),
    }
    bounds = _raw_bounds(mode, var_y, range_bounds, None)

    def objective(raw: np.ndarray) -> float:
        return profiled_vecchia_cluster_nll(
            raw,
            batches=batches,
            n_features=n_features,
            nugget_mode=mode,
            fixed_nugget=fixed_nugget,
            smooth_bounds=smooth_bounds,
            param_bounds=param_bounds,
            jitter=jitter,
            scale_by_n=True,
        )

    def total_nll(raw: np.ndarray) -> float:
        return profiled_vecchia_cluster_nll(
            raw,
            batches=batches,
            n_features=n_features,
            nugget_mode=mode,
            fixed_nugget=fixed_nugget,
            smooth_bounds=smooth_bounds,
            param_bounds=param_bounds,
            jitter=jitter,
            scale_by_n=False,
        )

    def convert_parameters(raw: np.ndarray) -> MaternParameters:
        return natural_from_raw(raw, mode, fixed_nugget, smooth_bounds)

    options = {"maxiter": int(maxiter), "ftol": 1e-7}
    normalized_method = str(method).upper()
    if normalized_method == "L-BFGS-B":
        options["maxls"] = int(maxls)
        options["maxcor"] = int(maxcor)
    if int(maxfun) > 0:
        options["maxfun"] = int(maxfun)
    results = _run_scipy_restarts(
        starts,
        objective=objective,
        total_nll=total_nll,
        parameter_fn=convert_parameters,
        method=str(method),
        bounds=bounds if normalized_method == "L-BFGS-B" else None,
        options=options,
    )
    best = _select_best(results)
    return best
