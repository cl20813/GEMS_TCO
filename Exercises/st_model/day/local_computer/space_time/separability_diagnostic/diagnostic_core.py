"""Numerical core for the advected-separable covariance diagnostic.

The module intentionally has no project-specific data-loading code.  It
implements the two covariance families, the KL projection of the joint
space-time Matern model onto the advected-separable family, and a stable
generalized eigensolver.  The first experiment fixes ``nu=1/2`` so both
families have exactly matching exponential spatial and temporal margins.

The statistical nugget is zero.  ``numerical_jitter_ratio`` is a separately
reported diagonal regularizer used only by dense linear algebra.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np
import scipy.linalg
import scipy.optimize
from scipy.stats import qmc


@dataclass(frozen=True)
class CovarianceParameters:
    """Physical parameters shared by the joint and separable families."""

    variance: float
    range_lat: float
    range_lon: float
    range_time: float
    advec_lat: float
    advec_lon: float
    nugget: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in asdict(self).items()}


@dataclass(frozen=True)
class LagGeometry:
    """Dense pairwise lags for one independent day/design."""

    delta_lat: np.ndarray
    delta_lon: np.ndarray
    delta_time: np.ndarray

    @property
    def size(self) -> int:
        return int(self.delta_lat.shape[0])


@dataclass(frozen=True)
class NullFitAttempt:
    start_index: int
    objective_per_observation: float
    variance: float
    range_lat: float
    range_lon: float
    range_time: float
    advec_lat: float
    advec_lon: float
    iterations: int
    evaluations: int
    converged: bool
    message: str


@dataclass(frozen=True)
class NullFitResult:
    parameters: CovarianceParameters
    objective_per_observation: float
    attempts: tuple[NullFitAttempt, ...]
    best_start_index: int
    bounds: dict[str, tuple[float, float]]
    boundary_parameters: tuple[str, ...]


@dataclass(frozen=True)
class GeneralizedEigenResult:
    """Eigenpairs ordered by decreasing KL contribution ``g(lambda)``."""

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    scores: np.ndarray
    matrix_kl: float
    spectral_kl: float
    max_relative_residual: float
    max_null_orthonormality_error: float


DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "range_lat": (0.03, 2.0),
    "range_lon": (0.03, 3.0),
    "range_time": (0.10, 10.0),
    "advec_lat": (-0.40, 0.40),
    "advec_lon": (-0.80, 0.40),
}


def _validate_coordinates(coordinates: np.ndarray) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 3:
        raise ValueError("coordinates must have shape (observations, 3)")
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates must be finite")
    return np.ascontiguousarray(coordinates)


def pairwise_lags(coordinates: np.ndarray) -> LagGeometry:
    """Return latitude, longitude, and local-time pairwise differences."""

    coordinates = _validate_coordinates(coordinates)
    return LagGeometry(
        delta_lat=coordinates[:, None, 0] - coordinates[None, :, 0],
        delta_lon=coordinates[:, None, 1] - coordinates[None, :, 1],
        delta_time=coordinates[:, None, 2] - coordinates[None, :, 2],
    )


def _validate_parameters(parameters: CovarianceParameters) -> None:
    positive = (
        parameters.variance,
        parameters.range_lat,
        parameters.range_lon,
        parameters.range_time,
    )
    if not np.isfinite(np.asarray(positive)).all() or min(positive) <= 0.0:
        raise ValueError("variance and ranges must be finite and positive")
    if not np.isfinite(parameters.advec_lat) or not np.isfinite(parameters.advec_lon):
        raise ValueError("advection parameters must be finite")
    if not np.isfinite(parameters.nugget) or parameters.nugget < 0.0:
        raise ValueError("nugget must be finite and non-negative")


def standardized_moving_lag_norms(
    geometry: LagGeometry,
    parameters: CovarianceParameters,
) -> tuple[np.ndarray, np.ndarray]:
    """Return standardized spatial and temporal lags in moving coordinates.

    The returned arrays are

    ``s = ||(h - v u) / ell_space||_2`` and
    ``t = |u| / ell_time``.

    Keeping these two non-negative components separate makes the structural
    comparison transparent: the joint exponential correlation is
    ``exp(-hypot(s, t))``, whereas its same-margin separable competitor is
    ``exp(-s - t)``.
    """

    _validate_parameters(parameters)
    shifted_lat = geometry.delta_lat - parameters.advec_lat * geometry.delta_time
    shifted_lon = geometry.delta_lon - parameters.advec_lon * geometry.delta_time
    spatial_norm = np.hypot(
        shifted_lat / parameters.range_lat,
        shifted_lon / parameters.range_lon,
    )
    temporal_norm = np.abs(geometry.delta_time) / parameters.range_time
    return spatial_norm, temporal_norm


def _validate_standardized_norms(
    spatial_norm: np.ndarray | float,
    temporal_norm: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    spatial_norm, temporal_norm = np.broadcast_arrays(
        np.asarray(spatial_norm, dtype=np.float64),
        np.asarray(temporal_norm, dtype=np.float64),
    )
    if (
        not np.isfinite(spatial_norm).all()
        or not np.isfinite(temporal_norm).all()
        or np.any(spatial_norm < 0.0)
        or np.any(temporal_norm < 0.0)
    ):
        raise ValueError("standardized lag norms must be finite and non-negative")
    return spatial_norm, temporal_norm


def same_margin_exponential_log_gap(
    spatial_norm: np.ndarray | float,
    temporal_norm: np.ndarray | float,
) -> np.ndarray:
    """Return ``log(R_joint) - log(R_separable)`` for matched margins.

    This is ``s + t - hypot(s, t)``.  It is zero on either axis, strictly
    positive for a genuinely mixed lag, and is maximized at ``s=t`` when the
    joint radius ``hypot(s, t)`` is held fixed.
    """

    spatial_norm, temporal_norm = _validate_standardized_norms(spatial_norm, temporal_norm)
    return spatial_norm + temporal_norm - np.hypot(spatial_norm, temporal_norm)


def same_margin_exponential_correlation_gap(
    spatial_norm: np.ndarray | float,
    temporal_norm: np.ndarray | float,
) -> np.ndarray:
    """Return ``R_joint - R_separable`` for matched exponential margins."""

    spatial_norm, temporal_norm = _validate_standardized_norms(spatial_norm, temporal_norm)
    return np.exp(-np.hypot(spatial_norm, temporal_norm)) - np.exp(-spatial_norm - temporal_norm)


def squared_exponential_correlation_from_norms(
    spatial_norm: np.ndarray | float,
    temporal_norm: np.ndarray | float,
) -> np.ndarray:
    """Correlation obtained by literally removing the outer square root.

    ``exp(-(s**2 + t**2))`` factorizes exactly as
    ``exp(-s**2) exp(-t**2)``.  It is therefore separable, but its axis
    margins are squared-exponential rather than exponential.  Consequently,
    this transformation does *not* preserve the covariance objective or its
    optimizer in general.
    """

    spatial_norm, temporal_norm = _validate_standardized_norms(spatial_norm, temporal_norm)
    return np.exp(-(spatial_norm**2 + temporal_norm**2))


def balanced_fixed_radius_gaps(
    radius: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """Maximum matched-margin gaps at a fixed joint standardized radius.

    For ``s**2 + t**2 = radius**2``, both maxima occur at
    ``s=t=radius/sqrt(2)``.  The returned values are respectively the maximum
    log-correlation gap and correlation gap.
    """

    radius = np.asarray(radius, dtype=np.float64)
    if not np.isfinite(radius).all() or np.any(radius < 0.0):
        raise ValueError("radius must be finite and non-negative")
    balanced_sum = np.sqrt(2.0) * radius
    return (
        balanced_sum - radius,
        np.exp(-radius) - np.exp(-balanced_sum),
    )


def joint_matern_half_correlation(
    geometry: LagGeometry,
    parameters: CovarianceParameters,
) -> np.ndarray:
    """Correlation of the intrinsically nonseparable joint Matern, nu=1/2."""

    spatial_norm, temporal_norm = standardized_moving_lag_norms(geometry, parameters)
    return np.exp(-np.hypot(spatial_norm, temporal_norm))


def advected_separable_correlation(
    geometry: LagGeometry,
    parameters: CovarianceParameters,
) -> np.ndarray:
    """Return ``R_S(h-vu) R_T(u)`` with exponential axis margins.

    This uses the same nu=1/2 spatial and temporal marginal families as the
    joint model.  The only structural difference is Euclidean combination in
    the joint model versus a product (additive exponent) in the null model.
    """

    spatial_norm, temporal_norm = standardized_moving_lag_norms(geometry, parameters)
    return np.exp(-spatial_norm - temporal_norm)


def covariance_from_correlation(
    correlation: np.ndarray,
    *,
    variance: float,
    nugget: float = 0.0,
    numerical_jitter_ratio: float = 0.0,
) -> np.ndarray:
    """Scale a correlation matrix and add distinct statistical/numerical diagonals."""

    correlation = np.asarray(correlation, dtype=np.float64)
    if correlation.ndim != 2 or correlation.shape[0] != correlation.shape[1]:
        raise ValueError("correlation must be square")
    if variance <= 0.0 or nugget < 0.0 or numerical_jitter_ratio < 0.0:
        raise ValueError("variance must be positive and diagonal terms non-negative")
    covariance = float(variance) * (correlation + correlation.T) * 0.5
    diagonal = np.diag_indices_from(covariance)
    covariance[diagonal] += float(nugget) + float(variance) * float(numerical_jitter_ratio)
    return covariance


def joint_matern_half_covariance(
    geometry: LagGeometry,
    parameters: CovarianceParameters,
    *,
    numerical_jitter_ratio: float = 0.0,
) -> np.ndarray:
    return covariance_from_correlation(
        joint_matern_half_correlation(geometry, parameters),
        variance=parameters.variance,
        nugget=parameters.nugget,
        numerical_jitter_ratio=numerical_jitter_ratio,
    )


def advected_separable_covariance(
    geometry: LagGeometry,
    parameters: CovarianceParameters,
    *,
    numerical_jitter_ratio: float = 0.0,
) -> np.ndarray:
    return covariance_from_correlation(
        advected_separable_correlation(geometry, parameters),
        variance=parameters.variance,
        nugget=parameters.nugget,
        numerical_jitter_ratio=numerical_jitter_ratio,
    )


def g_score(eigenvalues: np.ndarray) -> np.ndarray:
    """Per-direction KL contribution for ``Sigma1 w = lambda Sigma0 w``."""

    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    if np.any(eigenvalues <= 0.0) or not np.isfinite(eigenvalues).all():
        raise ValueError("generalized eigenvalues must be finite and positive")
    return 0.5 * (eigenvalues - 1.0 - np.log(eigenvalues))


def _shape_from_raw(raw: np.ndarray) -> tuple[float, float, float, float, float]:
    raw = np.asarray(raw, dtype=np.float64)
    if raw.shape != (5,):
        raise ValueError("raw shape parameter vector must have length five")
    return (
        float(np.exp(raw[0])),
        float(np.exp(raw[1])),
        float(np.exp(raw[2])),
        float(raw[3]),
        float(raw[4]),
    )


def _raw_from_parameters(parameters: CovarianceParameters) -> np.ndarray:
    _validate_parameters(parameters)
    return np.asarray(
        [
            np.log(parameters.range_lat),
            np.log(parameters.range_lon),
            np.log(parameters.range_time),
            parameters.advec_lat,
            parameters.advec_lon,
        ],
        dtype=np.float64,
    )


def _raw_bounds(
    bounds: dict[str, tuple[float, float]],
) -> list[tuple[float, float]]:
    required = tuple(DEFAULT_BOUNDS)
    if tuple(bounds) != required:
        missing = sorted(set(required) - set(bounds))
        extra = sorted(set(bounds) - set(required))
        raise ValueError(f"invalid bounds keys; missing={missing}, extra={extra}")
    raw: list[tuple[float, float]] = []
    for name in ("range_lat", "range_lon", "range_time"):
        lower, upper = bounds[name]
        if not (0.0 < lower < upper):
            raise ValueError(f"invalid positive bounds for {name}")
        raw.append((float(np.log(lower)), float(np.log(upper))))
    for name in ("advec_lat", "advec_lon"):
        lower, upper = bounds[name]
        if not lower < upper:
            raise ValueError(f"invalid bounds for {name}")
        raw.append((float(lower), float(upper)))
    return raw


def profiled_null_objective(
    raw_shape: np.ndarray,
    geometries: Sequence[LagGeometry],
    true_covariances: Sequence[np.ndarray],
    *,
    numerical_jitter_ratio: float,
) -> tuple[float, float]:
    """Return profiled expected Gaussian NLL per observation and variance.

    For ``Sigma0 = sigma0^2 R0`` the variance has the closed-form optimum

    ``sigma0^2 = sum trace(R0^-1 Sigma1) / sum n``.

    Constants independent of the null parameters are omitted from the first
    returned value.
    """

    if len(geometries) == 0 or len(geometries) != len(true_covariances):
        raise ValueError("geometries and true_covariances must be non-empty and aligned")
    range_lat, range_lon, range_time, advec_lat, advec_lon = _shape_from_raw(raw_shape)
    unit_parameters = CovarianceParameters(
        variance=1.0,
        range_lat=range_lat,
        range_lon=range_lon,
        range_time=range_time,
        advec_lat=advec_lat,
        advec_lon=advec_lon,
        nugget=0.0,
    )
    total_dimension = 0
    total_logdet_correlation = 0.0
    total_trace = 0.0
    for geometry, true_covariance in zip(geometries, true_covariances):
        correlation = advected_separable_covariance(
            geometry,
            unit_parameters,
            numerical_jitter_ratio=numerical_jitter_ratio,
        )
        if true_covariance.shape != correlation.shape:
            raise ValueError("true covariance shape does not match lag geometry")
        factor = scipy.linalg.cholesky(
            correlation,
            lower=True,
            check_finite=False,
            overwrite_a=False,
        )
        solved = scipy.linalg.cho_solve(
            (factor, True),
            true_covariance,
            check_finite=False,
            overwrite_b=False,
        )
        total_dimension += geometry.size
        total_logdet_correlation += float(2.0 * np.log(np.diag(factor)).sum())
        total_trace += float(np.trace(solved))
    variance = total_trace / float(total_dimension)
    if not np.isfinite(variance) or variance <= 0.0:
        raise scipy.linalg.LinAlgError("profiled variance is not positive")
    objective = (
        total_dimension * np.log(variance) + total_logdet_correlation + total_trace / variance
    ) / float(total_dimension)
    return float(objective), float(variance)


def _candidate_starts(
    initial: CovarianceParameters,
    raw_bounds: Sequence[tuple[float, float]],
    *,
    n_starts: int,
    random_seed: int,
) -> list[np.ndarray]:
    if n_starts < 1:
        raise ValueError("n_starts must be positive")
    starts = [_raw_from_parameters(initial)]
    if n_starts >= 2:
        starts.append(
            np.asarray(
                [starts[0][0], starts[0][1], starts[0][2], 0.0, 0.0],
                dtype=np.float64,
            )
        )
    if n_starts >= 3:
        starts.append(
            np.asarray(
                [
                    starts[0][0] + np.log(0.5),
                    starts[0][1] + np.log(0.5),
                    starts[0][2] + np.log(2.0),
                    0.0,
                    0.0,
                ],
                dtype=np.float64,
            )
        )
    if n_starts >= 4:
        starts.append(
            np.asarray(
                [
                    starts[0][0] + np.log(1.5),
                    starts[0][1] + np.log(1.5),
                    starts[0][2] + np.log(0.5),
                    1.5 * starts[0][3],
                    1.5 * starts[0][4],
                ],
                dtype=np.float64,
            )
        )
    remaining = n_starts - len(starts)
    if remaining > 0:
        sampler = qmc.LatinHypercube(d=5, seed=int(random_seed))
        unit = sampler.random(remaining)
        lower = np.asarray([item[0] for item in raw_bounds], dtype=np.float64)
        upper = np.asarray([item[1] for item in raw_bounds], dtype=np.float64)
        starts.extend(lower + unit * (upper - lower))
    lower = np.asarray([item[0] for item in raw_bounds], dtype=np.float64)
    upper = np.asarray([item[1] for item in raw_bounds], dtype=np.float64)
    return [np.clip(np.asarray(start, dtype=np.float64), lower, upper) for start in starts]


def fit_kl_optimal_null(
    geometries: Sequence[LagGeometry],
    true_covariances: Sequence[np.ndarray],
    initial: CovarianceParameters,
    *,
    numerical_jitter_ratio: float = 1.0e-10,
    bounds: dict[str, tuple[float, float]] | None = None,
    n_starts: int = 8,
    random_seed: int = 20260922,
    max_iterations: int = 120,
) -> NullFitResult:
    """Fit the strongest null in the specified bounded parameter family."""

    bounds = dict(DEFAULT_BOUNDS if bounds is None else bounds)
    raw_bounds = _raw_bounds(bounds)
    starts = _candidate_starts(
        initial,
        raw_bounds,
        n_starts=n_starts,
        random_seed=random_seed,
    )

    def objective(raw: np.ndarray) -> float:
        try:
            value, _ = profiled_null_objective(
                raw,
                geometries,
                true_covariances,
                numerical_jitter_ratio=numerical_jitter_ratio,
            )
            return value if np.isfinite(value) else 1.0e100
        except (ValueError, FloatingPointError, scipy.linalg.LinAlgError):
            return 1.0e100

    attempts: list[NullFitAttempt] = []
    raw_results: list[np.ndarray] = []
    for start_index, start in enumerate(starts):
        result = scipy.optimize.minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=raw_bounds,
            options={
                "maxiter": int(max_iterations),
                "maxls": 40,
                "ftol": 1.0e-11,
                "gtol": 1.0e-7,
            },
        )
        value, variance = profiled_null_objective(
            result.x,
            geometries,
            true_covariances,
            numerical_jitter_ratio=numerical_jitter_ratio,
        )
        range_lat, range_lon, range_time, advec_lat, advec_lon = _shape_from_raw(result.x)
        attempts.append(
            NullFitAttempt(
                start_index=start_index,
                objective_per_observation=value,
                variance=variance,
                range_lat=range_lat,
                range_lon=range_lon,
                range_time=range_time,
                advec_lat=advec_lat,
                advec_lon=advec_lon,
                iterations=int(result.nit),
                evaluations=int(result.nfev),
                converged=bool(result.success),
                message=str(result.message),
            )
        )
        raw_results.append(np.asarray(result.x, dtype=np.float64))
    finite = [
        index for index, item in enumerate(attempts) if np.isfinite(item.objective_per_observation)
    ]
    if not finite:
        raise RuntimeError("all KL-null optimization starts failed")
    best_index = min(finite, key=lambda index: attempts[index].objective_per_observation)
    best = attempts[best_index]
    best_raw = raw_results[best_index]

    raw_lower = np.asarray([item[0] for item in raw_bounds])
    raw_upper = np.asarray([item[1] for item in raw_bounds])
    widths = raw_upper - raw_lower
    boundary_names = (
        "range_lat",
        "range_lon",
        "range_time",
        "advec_lat",
        "advec_lon",
    )
    boundary = tuple(
        name
        for name, value, lower, upper, width in zip(
            boundary_names,
            best_raw,
            raw_lower,
            raw_upper,
            widths,
        )
        if min(value - lower, upper - value) <= 1.0e-4 * width
    )
    parameters = CovarianceParameters(
        variance=best.variance,
        range_lat=best.range_lat,
        range_lon=best.range_lon,
        range_time=best.range_time,
        advec_lat=best.advec_lat,
        advec_lon=best.advec_lon,
        nugget=0.0,
    )
    return NullFitResult(
        parameters=parameters,
        objective_per_observation=best.objective_per_observation,
        attempts=tuple(attempts),
        best_start_index=best.start_index,
        bounds=bounds,
        boundary_parameters=boundary,
    )


def gaussian_kl(true_covariance: np.ndarray, null_covariance: np.ndarray) -> float:
    """Return ``KL[N(0,Sigma1) || N(0,Sigma0)]`` without forming an inverse."""

    true_covariance = np.asarray(true_covariance, dtype=np.float64)
    null_covariance = np.asarray(null_covariance, dtype=np.float64)
    if true_covariance.shape != null_covariance.shape or true_covariance.ndim != 2:
        raise ValueError("covariances must be same-size square matrices")
    factor0 = scipy.linalg.cholesky(null_covariance, lower=True, check_finite=False)
    factor1 = scipy.linalg.cholesky(true_covariance, lower=True, check_finite=False)
    trace = float(
        np.trace(scipy.linalg.cho_solve((factor0, True), true_covariance, check_finite=False))
    )
    logdet0 = float(2.0 * np.log(np.diag(factor0)).sum())
    logdet1 = float(2.0 * np.log(np.diag(factor1)).sum())
    return 0.5 * (trace - true_covariance.shape[0] + logdet0 - logdet1)


def solve_generalized_eigenproblem(
    true_covariance: np.ndarray,
    null_covariance: np.ndarray,
) -> GeneralizedEigenResult:
    """Solve ``Sigma1 w = lambda Sigma0 w`` using a symmetric-definite driver."""

    true_covariance = np.asarray(true_covariance, dtype=np.float64)
    null_covariance = np.asarray(null_covariance, dtype=np.float64)
    if true_covariance.shape != null_covariance.shape or true_covariance.ndim != 2:
        raise ValueError("covariances must be same-size square matrices")
    true_covariance = (true_covariance + true_covariance.T) * 0.5
    null_covariance = (null_covariance + null_covariance.T) * 0.5
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        true_covariance,
        null_covariance,
        type=1,
        driver="gvd",
        check_finite=False,
    )
    scores = g_score(eigenvalues)
    order = np.argsort(scores, kind="stable")[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.float64)
    scores = np.asarray(scores[order], dtype=np.float64)

    # Fix the arbitrary sign so figures and saved arrays are reproducible.
    largest_rows = np.argmax(np.abs(eigenvectors), axis=0)
    signs = np.sign(eigenvectors[largest_rows, np.arange(eigenvectors.shape[1])])
    signs[signs == 0.0] = 1.0
    eigenvectors *= signs

    left = true_covariance @ eigenvectors
    null_times_vectors = null_covariance @ eigenvectors
    residual = left - null_times_vectors * eigenvalues[None, :]
    denominator = np.linalg.norm(left, axis=0) + np.abs(eigenvalues) * np.linalg.norm(
        null_times_vectors, axis=0
    )
    relative = np.linalg.norm(residual, axis=0) / np.maximum(denominator, 1.0e-300)
    gram = eigenvectors.T @ null_times_vectors
    orthonormality = float(np.max(np.abs(gram - np.eye(gram.shape[0]))))
    matrix_kl = gaussian_kl(true_covariance, null_covariance)
    return GeneralizedEigenResult(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        scores=scores,
        matrix_kl=matrix_kl,
        spectral_kl=float(scores.sum()),
        max_relative_residual=float(relative.max()),
        max_null_orthonormality_error=orthonormality,
    )


def standardize_directions_for_design(
    directions: np.ndarray,
    null_covariance: np.ndarray,
    true_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transfer fixed directions and normalize each to unit null variance."""

    directions = np.asarray(directions, dtype=np.float64)
    null_covariance = np.asarray(null_covariance, dtype=np.float64)
    true_covariance = np.asarray(true_covariance, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[0] != null_covariance.shape[0]:
        raise ValueError("directions and covariance dimensions do not align")
    null_variances = np.einsum(
        "ik,ij,jk->k", directions, null_covariance, directions, optimize=True
    )
    if np.any(null_variances <= 0.0):
        raise scipy.linalg.LinAlgError("a transferred direction has nonpositive null variance")
    standardized = directions / np.sqrt(null_variances)[None, :]
    projected_null = standardized.T @ null_covariance @ standardized
    projected_true = standardized.T @ true_covariance @ standardized
    return (
        standardized,
        (projected_null + projected_null.T) * 0.5,
        (projected_true + projected_true.T) * 0.5,
    )


def covariance_log_likelihood_ratio(
    values: np.ndarray,
    null_covariance: np.ndarray,
    true_covariance: np.ndarray,
) -> float:
    """Log density ratio ``log f1(values) - log f0(values)``."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    null_factor = scipy.linalg.cholesky(null_covariance, lower=True, check_finite=False)
    true_factor = scipy.linalg.cholesky(true_covariance, lower=True, check_finite=False)
    if len(values) != null_covariance.shape[0] or null_covariance.shape != true_covariance.shape:
        raise ValueError("values and covariance dimensions do not align")
    null_quad = float(
        values @ scipy.linalg.cho_solve((null_factor, True), values, check_finite=False)
    )
    true_quad = float(
        values @ scipy.linalg.cho_solve((true_factor, True), values, check_finite=False)
    )
    logdet_null = float(2.0 * np.log(np.diag(null_factor)).sum())
    logdet_true = float(2.0 * np.log(np.diag(true_factor)).sum())
    return 0.5 * (logdet_null - logdet_true + null_quad - true_quad)


def block_diagonal_covariance(covariances: Iterable[np.ndarray]) -> np.ndarray:
    matrices = [np.asarray(item, dtype=np.float64) for item in covariances]
    if not matrices:
        raise ValueError("at least one covariance is required")
    return scipy.linalg.block_diag(*matrices)


__all__ = [
    "CovarianceParameters",
    "DEFAULT_BOUNDS",
    "GeneralizedEigenResult",
    "LagGeometry",
    "NullFitAttempt",
    "NullFitResult",
    "advected_separable_correlation",
    "advected_separable_covariance",
    "balanced_fixed_radius_gaps",
    "block_diagonal_covariance",
    "covariance_from_correlation",
    "covariance_log_likelihood_ratio",
    "fit_kl_optimal_null",
    "g_score",
    "gaussian_kl",
    "joint_matern_half_correlation",
    "joint_matern_half_covariance",
    "pairwise_lags",
    "profiled_null_objective",
    "same_margin_exponential_correlation_gap",
    "same_margin_exponential_log_gap",
    "solve_generalized_eigenproblem",
    "squared_exponential_correlation_from_norms",
    "standardized_moving_lag_norms",
    "standardize_directions_for_design",
]
