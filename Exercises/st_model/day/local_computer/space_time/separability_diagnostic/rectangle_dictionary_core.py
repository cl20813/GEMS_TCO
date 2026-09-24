"""Linear algebra for rectangle-dictionary interaction contrasts.

The observation vector is time-major: ``index = time * anchors + anchor``.
For spatial endpoints ``p < q`` and temporal endpoints ``k < ell``, a column
of the dictionary is

``(e_ell - e_k) kron (e_q - e_p)``.

All optimizers in this module use the intrinsic numerator
``Delta_int = Sigma1 - SigmaM`` and the fitted-null normalization ``Sigma0``.
Consequently their eigenvalue is a Sigma0-scaled variance *difference*, not a
variance ratio and not a KL contribution.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal, Sequence

import numpy as np
import scipy.linalg


Branch = Literal["positive", "negative"]


@dataclass(frozen=True)
class DictionarySVD:
    basis: np.ndarray
    singular_values: np.ndarray
    full_singular_values: np.ndarray
    right_transpose: np.ndarray
    rank: int
    absolute_tolerance: float
    relative_tolerance: float
    reconstruction_relative_error: float


@dataclass(frozen=True)
class ContrastSolution:
    eigenvalue: float
    filter_weights: np.ndarray
    reduced_coefficients: np.ndarray
    null_variance: float
    intrinsic_difference: float
    constrained_relative_residual: float


@dataclass(frozen=True)
class GreedyStep:
    branch: Branch
    size: int
    added_index: int
    selected_indices: tuple[int, ...]
    eigenvalue: float
    standardized_coefficients: np.ndarray
    filter_weights: np.ndarray
    winner_runner_up_gap: float
    winner_tie_count: int
    skipped_dependent_candidates: int


def _symmetric_matrix(matrix: np.ndarray, *, name: str) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite")
    asymmetry = float(np.max(np.abs(matrix - matrix.T)))
    scale = max(float(np.max(np.abs(matrix))), 1.0)
    if asymmetry > 1.0e-10 * scale:
        raise ValueError(f"{name} is not symmetric: max asymmetry={asymmetry:.3e}")
    return (matrix + matrix.T) * 0.5


def _fix_filter_signs(
    coefficients: np.ndarray,
    filters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Make the largest-magnitude observation weight positive per filter."""

    coefficients = np.asarray(coefficients, dtype=np.float64)
    filters = np.asarray(filters, dtype=np.float64)
    vector_input = filters.ndim == 1
    if vector_input:
        filters = filters[:, None]
        coefficients = coefficients[:, None]
    largest_rows = np.argmax(np.abs(filters), axis=0)
    signs = np.sign(filters[largest_rows, np.arange(filters.shape[1])])
    signs[signs == 0.0] = 1.0
    filters = filters * signs[None, :]
    coefficients = coefficients * signs[None, :]
    if vector_input:
        return coefficients[:, 0], filters[:, 0]
    return coefficients, filters


def constrained_relative_residual(
    weights: np.ndarray,
    eigenvalue: float,
    basis: np.ndarray,
    intrinsic_difference: np.ndarray,
    null_covariance: np.ndarray,
) -> float:
    """Return a basis-invariant residual norm within a constrained span."""

    weights = np.asarray(weights, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    intrinsic_difference = _symmetric_matrix(
        intrinsic_difference,
        name="intrinsic_difference",
    )
    null_covariance = _symmetric_matrix(null_covariance, name="null_covariance")
    reduced_null = basis.T @ null_covariance @ basis
    factor = scipy.linalg.cholesky(
        (reduced_null + reduced_null.T) * 0.5,
        lower=True,
        check_finite=False,
    )

    def inverse_metric_norm(vector: np.ndarray) -> float:
        solved = scipy.linalg.cho_solve((factor, True), vector, check_finite=False)
        squared = float(vector @ solved)
        return float(np.sqrt(max(squared, 0.0)))

    left = basis.T @ (intrinsic_difference @ weights)
    right = basis.T @ (null_covariance @ weights)
    residual = left - float(eigenvalue) * right
    denominator = inverse_metric_norm(left) + abs(float(eigenvalue)) * inverse_metric_norm(right)
    return inverse_metric_norm(residual) / max(denominator, 1.0e-300)


def rectangle_endpoints(anchor_count: int, time_count: int) -> np.ndarray:
    """Return every ``(p, q, k, ell)`` with ``p < q`` and ``k < ell``."""

    if anchor_count < 2 or time_count < 2:
        raise ValueError("at least two anchors and two times are required")
    spatial_pairs = tuple(combinations(range(int(anchor_count)), 2))
    temporal_pairs = tuple(combinations(range(int(time_count)), 2))
    return np.asarray(
        [(p, q, k, ell) for p, q in spatial_pairs for k, ell in temporal_pairs],
        dtype=np.int64,
    )


def rectangle_matrix(
    anchor_count: int,
    time_count: int,
    endpoints: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the dense observation-space dictionary and endpoint table."""

    if endpoints is None:
        endpoints = rectangle_endpoints(anchor_count, time_count)
    endpoints = np.asarray(endpoints, dtype=np.int64)
    if endpoints.ndim != 2 or endpoints.shape[1] != 4:
        raise ValueError("endpoints must have shape (rectangles, 4)")
    dimension = int(anchor_count) * int(time_count)
    matrix = np.zeros((dimension, len(endpoints)), dtype=np.float64)
    columns = np.arange(len(endpoints))
    p, q, k, ell = endpoints.T
    if (
        np.any(p < 0)
        or np.any(q >= anchor_count)
        or np.any(k < 0)
        or np.any(ell >= time_count)
        or np.any(p >= q)
        or np.any(k >= ell)
    ):
        raise ValueError("rectangle endpoints are outside canonical p<q, k<ell bounds")
    matrix[k * anchor_count + p, columns] = 1.0
    matrix[k * anchor_count + q, columns] = -1.0
    matrix[ell * anchor_count + p, columns] = -1.0
    matrix[ell * anchor_count + q, columns] = 1.0
    return matrix, endpoints


def rectangle_structure_errors(
    matrix: np.ndarray,
    *,
    anchor_count: int,
    time_count: int,
) -> tuple[float, float]:
    """Return maximum per-anchor temporal and per-time spatial sum errors."""

    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.shape[0] != anchor_count * time_count:
        raise ValueError("matrix does not match the stated space-time dimensions")
    weights = matrix.reshape(time_count, anchor_count, matrix.shape[1])
    temporal_sum_error = float(np.max(np.abs(weights.sum(axis=0))))
    spatial_sum_error = float(np.max(np.abs(weights.sum(axis=1))))
    return temporal_sum_error, spatial_sum_error


def standardize_dictionary(
    matrix: np.ndarray,
    null_covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale each column to unit fitted-null variance without whitening it."""

    matrix = np.asarray(matrix, dtype=np.float64)
    null_covariance = _symmetric_matrix(null_covariance, name="null_covariance")
    if matrix.ndim != 2 or matrix.shape[0] != len(null_covariance):
        raise ValueError("dictionary and covariance dimensions do not align")
    null_times_matrix = null_covariance @ matrix
    variances = np.einsum("ik,ik->k", matrix, null_times_matrix, optimize=True)
    if not np.isfinite(variances).all() or np.any(variances <= 0.0):
        raise scipy.linalg.LinAlgError("a rectangle has nonpositive null variance")
    scales = np.sqrt(variances)
    standardized = matrix / scales[None, :]
    return standardized, scales, variances


def dictionary_svd(
    standardized_matrix: np.ndarray,
    *,
    relative_tolerance: float,
) -> DictionarySVD:
    """Return a rank-revealing compact SVD of the standardized dictionary."""

    standardized_matrix = np.asarray(standardized_matrix, dtype=np.float64)
    if standardized_matrix.ndim != 2 or not np.isfinite(standardized_matrix).all():
        raise ValueError("standardized_matrix must be a finite matrix")
    if not 0.0 < relative_tolerance < 1.0:
        raise ValueError("relative_tolerance must lie in (0, 1)")
    left, singular_values, right_transpose = scipy.linalg.svd(
        standardized_matrix,
        full_matrices=False,
        check_finite=False,
        lapack_driver="gesdd",
    )
    absolute_tolerance = float(relative_tolerance * singular_values[0])
    rank = int(np.sum(singular_values > absolute_tolerance))
    if rank == 0:
        raise scipy.linalg.LinAlgError("rectangle dictionary has numerical rank zero")
    basis = np.asarray(left[:, :rank], dtype=np.float64)
    retained_values = np.asarray(singular_values[:rank], dtype=np.float64)
    retained_right = np.asarray(right_transpose[:rank], dtype=np.float64)
    reconstruction = (basis * retained_values[None, :]) @ retained_right
    error = float(
        np.linalg.norm(standardized_matrix - reconstruction) / np.linalg.norm(standardized_matrix)
    )
    return DictionarySVD(
        basis=basis,
        singular_values=retained_values,
        full_singular_values=np.asarray(singular_values, dtype=np.float64),
        right_transpose=retained_right,
        rank=rank,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=float(relative_tolerance),
        reconstruction_relative_error=error,
    )


def solve_constrained_contrast(
    intrinsic_difference: np.ndarray,
    null_covariance: np.ndarray,
    basis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Solve all extrema of ``w' Delta_int w / w' Sigma0 w`` in a span."""

    intrinsic_difference = _symmetric_matrix(
        intrinsic_difference,
        name="intrinsic_difference",
    )
    null_covariance = _symmetric_matrix(null_covariance, name="null_covariance")
    if intrinsic_difference.shape != null_covariance.shape:
        raise ValueError("difference and null covariance shapes differ")
    if basis is None:
        basis = np.eye(len(null_covariance), dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[0] != len(null_covariance):
        raise ValueError("basis and covariance dimensions do not align")
    reduced_difference = basis.T @ intrinsic_difference @ basis
    reduced_null = basis.T @ null_covariance @ basis
    reduced_difference = (reduced_difference + reduced_difference.T) * 0.5
    reduced_null = (reduced_null + reduced_null.T) * 0.5
    eigenvalues, coefficients = scipy.linalg.eigh(
        reduced_difference,
        reduced_null,
        type=1,
        driver="gvd",
        check_finite=False,
    )
    filters = basis @ coefficients
    coefficients, filters = _fix_filter_signs(coefficients, filters)
    return (
        np.asarray(eigenvalues, dtype=np.float64),
        np.asarray(coefficients, dtype=np.float64),
        np.asarray(filters, dtype=np.float64),
    )


def summarize_contrast_solution(
    eigenvalue: float,
    reduced_coefficients: np.ndarray,
    filter_weights: np.ndarray,
    basis: np.ndarray,
    intrinsic_difference: np.ndarray,
    null_covariance: np.ndarray,
) -> ContrastSolution:
    """Compute normalization and constrained residual diagnostics."""

    filter_weights = np.asarray(filter_weights, dtype=np.float64)
    reduced_coefficients = np.asarray(reduced_coefficients, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    null_variance = float(filter_weights @ null_covariance @ filter_weights)
    intrinsic = float(filter_weights @ intrinsic_difference @ filter_weights)
    relative = constrained_relative_residual(
        filter_weights,
        eigenvalue,
        basis,
        intrinsic_difference,
        null_covariance,
    )
    return ContrastSolution(
        eigenvalue=float(eigenvalue),
        filter_weights=filter_weights,
        reduced_coefficients=reduced_coefficients,
        null_variance=null_variance,
        intrinsic_difference=intrinsic,
        constrained_relative_residual=relative,
    )


def recover_rectangle_coefficients(
    decomposition: DictionarySVD,
    reduced_coefficients: np.ndarray,
    rectangle_scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover minimum-norm standardized and corresponding raw coefficients."""

    reduced_coefficients = np.asarray(reduced_coefficients, dtype=np.float64)
    rectangle_scales = np.asarray(rectangle_scales, dtype=np.float64)
    if reduced_coefficients.shape != (decomposition.rank,):
        raise ValueError("reduced coefficient length does not match retained rank")
    if rectangle_scales.shape != (decomposition.right_transpose.shape[1],):
        raise ValueError("rectangle scales do not match dictionary width")
    standardized = decomposition.right_transpose.T @ (
        reduced_coefficients / decomposition.singular_values
    )
    raw = standardized / rectangle_scales
    return standardized, raw


def sigma0_metric_projection(
    basis: np.ndarray,
    null_covariance: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project directions onto ``span(basis)`` in the fitted-null metric."""

    basis = np.asarray(basis, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    if directions.ndim == 1:
        directions = directions[:, None]
    null_covariance = _symmetric_matrix(null_covariance, name="null_covariance")
    if basis.shape[0] != len(null_covariance) or directions.shape[0] != len(null_covariance):
        raise ValueError("basis/directions and covariance dimensions do not align")
    reduced_null = basis.T @ null_covariance @ basis
    factor = scipy.linalg.cholesky(reduced_null, lower=True, check_finite=False)
    right = basis.T @ null_covariance @ directions
    coefficients = scipy.linalg.cho_solve((factor, True), right, check_finite=False)
    projected = basis @ coefficients
    return projected, coefficients


def direction_retention(
    projected: np.ndarray,
    original: np.ndarray,
    null_covariance: np.ndarray,
) -> np.ndarray:
    """Return fitted-null energy retained by metric projections."""

    projected = np.asarray(projected, dtype=np.float64)
    original = np.asarray(original, dtype=np.float64)
    if projected.ndim == 1:
        projected = projected[:, None]
    if original.ndim == 1:
        original = original[:, None]
    numerator = np.einsum(
        "ik,ij,jk->k",
        projected,
        null_covariance,
        projected,
        optimize=True,
    )
    denominator = np.einsum(
        "ik,ij,jk->k",
        original,
        null_covariance,
        original,
        optimize=True,
    )
    retention = numerator / denominator
    if np.any(retention < -1.0e-10) or np.any(retention > 1.0 + 1.0e-10):
        raise ArithmeticError("metric projection retention lies outside [0, 1]")
    return retention


def cluster_overlap(
    dictionary_basis: np.ndarray,
    cluster_directions: np.ndarray,
    null_covariance: np.ndarray,
) -> dict[str, object]:
    """Return basis-invariant squared canonical correlations of two subspaces."""

    cluster_gram = cluster_directions.T @ null_covariance @ cluster_directions
    factor = scipy.linalg.cholesky(cluster_gram, lower=True, check_finite=False)
    normalized_cluster = scipy.linalg.solve_triangular(
        factor,
        cluster_directions.T,
        lower=True,
        check_finite=False,
    ).T
    projected, _ = sigma0_metric_projection(
        dictionary_basis,
        null_covariance,
        normalized_cluster,
    )
    overlap = normalized_cluster.T @ null_covariance @ projected
    overlap = (overlap + overlap.T) * 0.5
    squared_cosines = np.linalg.eigvalsh(overlap)
    squared_cosines = np.sort(squared_cosines)[::-1]
    if np.any(squared_cosines < -1.0e-9) or np.any(squared_cosines > 1.0 + 1.0e-9):
        raise ArithmeticError("cluster canonical correlation lies outside [0, 1]")
    squared_cosines = np.clip(squared_cosines, 0.0, 1.0)
    return {
        "squared_canonical_correlations": squared_cosines,
        "mean_retention": float(squared_cosines.mean()),
        "minimum_retention": float(squared_cosines.min()),
        "maximum_principal_angle_degrees": float(
            np.degrees(np.arccos(np.sqrt(squared_cosines.min())))
        ),
    }


def filter_variance_metrics(
    weights: np.ndarray,
    true_covariance: np.ndarray,
    matched_covariance: np.ndarray,
    null_covariance: np.ndarray,
) -> dict[str, float]:
    """Evaluate intrinsic, compensation, total, ratio, and one-direction KL."""

    weights = np.asarray(weights, dtype=np.float64)
    v0 = float(weights @ null_covariance @ weights)
    vm = float(weights @ matched_covariance @ weights)
    v1 = float(weights @ true_covariance @ weights)
    if min(v0, vm, v1) <= 0.0:
        raise scipy.linalg.LinAlgError("a projected covariance variance is nonpositive")
    intrinsic = v1 - vm
    compensation = vm - v0
    total = v1 - v0
    rho0 = v1 / v0
    rhom = v1 / vm

    def g(value: float) -> float:
        return 0.5 * (value - 1.0 - np.log(value))

    magnitude = abs(intrinsic) + abs(compensation)
    return {
        "v0": v0,
        "v_matched": vm,
        "v_true": v1,
        "delta_intrinsic": intrinsic,
        "delta_compensation": compensation,
        "delta_total": total,
        "rho_fitted": rho0,
        "rho_matched": rhom,
        "g_fitted": g(rho0),
        "g_matched": g(rhom),
        "interaction_absolute_share": (
            float(abs(intrinsic) / magnitude) if magnitude > 1.0e-14 else np.nan
        ),
    }


def projected_gaussian_kl(
    filters: np.ndarray,
    true_covariance: np.ndarray,
    null_covariance: np.ndarray,
) -> float:
    """KL of the joint projected Gaussian, using Cholesky solves."""

    filters = np.asarray(filters, dtype=np.float64)
    if filters.ndim == 1:
        filters = filters[:, None]
    projected_true = filters.T @ true_covariance @ filters
    projected_null = filters.T @ null_covariance @ filters
    projected_true = (projected_true + projected_true.T) * 0.5
    projected_null = (projected_null + projected_null.T) * 0.5
    factor0 = scipy.linalg.cholesky(projected_null, lower=True, check_finite=False)
    factor1 = scipy.linalg.cholesky(projected_true, lower=True, check_finite=False)
    trace = float(
        np.trace(
            scipy.linalg.cho_solve(
                (factor0, True),
                projected_true,
                check_finite=False,
            )
        )
    )
    dimension = filters.shape[1]
    logdet0 = float(2.0 * np.log(np.diag(factor0)).sum())
    logdet1 = float(2.0 * np.log(np.diag(factor1)).sum())
    return 0.5 * (trace - dimension + logdet0 - logdet1)


def _extreme_selected_solution(
    standardized_matrix: np.ndarray,
    intrinsic_times_matrix: np.ndarray,
    null_times_matrix: np.ndarray,
    selected: Sequence[int],
    branch: Branch,
) -> tuple[float, np.ndarray, np.ndarray]:
    columns = np.asarray(selected, dtype=np.int64)
    selected_matrix = standardized_matrix[:, columns]
    reduced_difference = selected_matrix.T @ intrinsic_times_matrix[:, columns]
    reduced_null = selected_matrix.T @ null_times_matrix[:, columns]
    reduced_difference = (reduced_difference + reduced_difference.T) * 0.5
    reduced_null = (reduced_null + reduced_null.T) * 0.5
    values, vectors = scipy.linalg.eigh(
        reduced_difference,
        reduced_null,
        type=1,
        check_finite=False,
    )
    index = -1 if branch == "positive" else 0
    coefficients = vectors[:, index]
    weights = selected_matrix @ coefficients
    coefficients, weights = _fix_filter_signs(coefficients, weights)
    return float(values[index]), coefficients, weights


def greedy_extreme_path(
    standardized_matrix: np.ndarray,
    intrinsic_difference: np.ndarray,
    null_covariance: np.ndarray,
    *,
    branch: Branch,
    maximum_size: int,
    dependence_tolerance: float = 1.0e-10,
    score_tie_tolerance: float = 1.0e-12,
) -> list[GreedyStep]:
    """Forward-select rectangles, reoptimizing the requested extreme each step."""

    if branch not in ("positive", "negative"):
        raise ValueError("branch must be positive or negative")
    standardized_matrix = np.asarray(standardized_matrix, dtype=np.float64)
    intrinsic_difference = _symmetric_matrix(
        intrinsic_difference,
        name="intrinsic_difference",
    )
    null_covariance = _symmetric_matrix(null_covariance, name="null_covariance")
    if standardized_matrix.shape[0] != len(null_covariance):
        raise ValueError("dictionary and covariance dimensions do not align")
    if maximum_size < 1 or maximum_size > standardized_matrix.shape[1]:
        raise ValueError("maximum_size is outside dictionary width")
    if dependence_tolerance <= 0.0 or score_tie_tolerance < 0.0:
        raise ValueError("greedy tolerances must be non-negative, with positive dependence")
    intrinsic_times_matrix = intrinsic_difference @ standardized_matrix
    null_times_matrix = null_covariance @ standardized_matrix
    intrinsic_diagonal = np.einsum(
        "ik,ik->k",
        standardized_matrix,
        intrinsic_times_matrix,
        optimize=True,
    )
    null_diagonal = np.einsum(
        "ik,ik->k",
        standardized_matrix,
        null_times_matrix,
        optimize=True,
    )
    single_scores = intrinsic_diagonal / null_diagonal

    def choose_extreme(
        scores: np.ndarray,
        indices: np.ndarray,
    ) -> tuple[int, float, int]:
        best = float(np.max(scores) if branch == "positive" else np.min(scores))
        tolerance = score_tie_tolerance * max(1.0, abs(best))
        tied = np.abs(scores - best) <= tolerance
        winner = int(np.min(indices[tied]))
        remaining_scores = scores[indices != winner]
        if len(remaining_scores):
            runner = float(
                np.max(remaining_scores) if branch == "positive" else np.min(remaining_scores)
            )
            gap = abs(best - runner)
        else:
            gap = 0.0
        return winner, float(gap), int(tied.sum())

    first, first_gap, first_tie_count = choose_extreme(
        single_scores,
        np.arange(len(single_scores), dtype=np.int64),
    )
    selected: list[int] = [first]
    first_score, first_coefficients, first_weights = _extreme_selected_solution(
        standardized_matrix,
        intrinsic_times_matrix,
        null_times_matrix,
        selected,
        branch,
    )
    steps = [
        GreedyStep(
            branch=branch,
            size=1,
            added_index=first,
            selected_indices=tuple(selected),
            eigenvalue=first_score,
            standardized_coefficients=first_coefficients,
            filter_weights=first_weights,
            winner_runner_up_gap=first_gap,
            winner_tie_count=first_tie_count,
            skipped_dependent_candidates=0,
        )
    ]
    all_indices = np.arange(standardized_matrix.shape[1])
    for size in range(2, maximum_size + 1):
        selected_matrix = standardized_matrix[:, selected]
        selected_null = selected_matrix.T @ null_times_matrix[:, selected]
        selected_factor = scipy.linalg.cholesky(
            (selected_null + selected_null.T) * 0.5,
            lower=True,
            check_finite=False,
        )
        cross_null = selected_matrix.T @ null_times_matrix
        selected_difference = selected_matrix.T @ intrinsic_times_matrix[:, selected]
        selected_difference = (selected_difference + selected_difference.T) * 0.5
        cross_difference = selected_matrix.T @ intrinsic_times_matrix
        projected_coefficients = scipy.linalg.cho_solve(
            (selected_factor, True),
            cross_null,
            check_finite=False,
        )
        residual_variance = null_diagonal - np.einsum(
            "ik,ik->k",
            cross_null,
            projected_coefficients,
            optimize=True,
        )
        independent = residual_variance > dependence_tolerance * null_diagonal
        independent[np.asarray(selected, dtype=np.int64)] = False
        candidate_indices = all_indices[independent]
        if not len(candidate_indices):
            raise RuntimeError("no independent rectangle remains for greedy selection")
        candidate_results: list[tuple[float, int]] = []
        for candidate in candidate_indices:
            trial_size = len(selected) + 1
            trial_difference = np.empty((trial_size, trial_size), dtype=np.float64)
            trial_null = np.empty((trial_size, trial_size), dtype=np.float64)
            trial_difference[:-1, :-1] = selected_difference
            trial_difference[:-1, -1] = cross_difference[:, candidate]
            trial_difference[-1, :-1] = cross_difference[:, candidate]
            trial_difference[-1, -1] = intrinsic_diagonal[candidate]
            trial_null[:-1, :-1] = selected_null
            trial_null[:-1, -1] = cross_null[:, candidate]
            trial_null[-1, :-1] = cross_null[:, candidate]
            trial_null[-1, -1] = null_diagonal[candidate]
            extreme_index = trial_size - 1 if branch == "positive" else 0
            try:
                score = float(
                    scipy.linalg.eigh(
                        trial_difference,
                        trial_null,
                        type=1,
                        driver="gvx",
                        subset_by_index=[extreme_index, extreme_index],
                        eigvals_only=True,
                        check_finite=False,
                    )[0]
                )
            except scipy.linalg.LinAlgError:
                continue
            candidate_results.append((score, int(candidate)))
        if not candidate_results:
            raise RuntimeError("all independent greedy candidates failed")
        result_scores = np.asarray([item[0] for item in candidate_results], dtype=np.float64)
        result_indices = np.asarray([item[1] for item in candidate_results], dtype=np.int64)
        winner_index, gap, tie_count = choose_extreme(result_scores, result_indices)
        selected.append(winner_index)
        score, coefficients, weights = _extreme_selected_solution(
            standardized_matrix,
            intrinsic_times_matrix,
            null_times_matrix,
            selected,
            branch,
        )
        previous = steps[-1].eigenvalue
        if branch == "positive" and score < previous - 1.0e-10:
            raise ArithmeticError("positive greedy objective decreased")
        if branch == "negative" and score > previous + 1.0e-10:
            raise ArithmeticError("negative greedy objective increased")
        steps.append(
            GreedyStep(
                branch=branch,
                size=size,
                added_index=winner_index,
                selected_indices=tuple(selected),
                eigenvalue=score,
                standardized_coefficients=coefficients,
                filter_weights=weights,
                winner_runner_up_gap=float(gap),
                winner_tie_count=tie_count,
                skipped_dependent_candidates=int((~independent).sum() - len(selected) + 1),
            )
        )
    return steps


__all__ = [
    "Branch",
    "ContrastSolution",
    "DictionarySVD",
    "GreedyStep",
    "cluster_overlap",
    "constrained_relative_residual",
    "dictionary_svd",
    "direction_retention",
    "filter_variance_metrics",
    "greedy_extreme_path",
    "projected_gaussian_kl",
    "recover_rectangle_coefficients",
    "rectangle_endpoints",
    "rectangle_matrix",
    "rectangle_structure_errors",
    "sigma0_metric_projection",
    "solve_constrained_contrast",
    "standardize_dictionary",
    "summarize_contrast_solution",
]
