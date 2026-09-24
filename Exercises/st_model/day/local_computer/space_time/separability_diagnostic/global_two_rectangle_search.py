"""Blockwise exhaustive search over every unordered pair of rectangles.

The dictionary columns supplied to this module are normalized to unit variance
under the fitted-null covariance.  For columns ``i`` and ``j`` the reduced
generalized eigenproblem is therefore

``D v = mu G v`` with ``G = [[1, g_ij], [g_ij, 1]]``.

The ordinary path uses the closed-form quadratic equation.  Pairs whose null
Gram matrix is nearly singular are evaluated separately after rotating to the
symmetric/antisymmetric basis and whitening by ``1 + g`` and ``1 - g``.  No
rectangle-by-rectangle Gram matrix and no vector of all pair scores is stored.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Callable

import numpy as np
import scipy.linalg


@dataclass(frozen=True)
class PairCandidate:
    first_index: int
    second_index: int
    analytic_eigenvalue: float


@dataclass(frozen=True)
class PairSearchResult:
    minimum_eigenvalue: float
    candidates: tuple[PairCandidate, ...]
    pair_count: int
    ordinary_pair_count: int
    near_singular_pair_count: int
    nonpositive_gram_pair_count: int
    clamped_discriminant_count: int
    near_singular_records: tuple[dict[str, float | int], ...]
    block_size: int
    tie_tolerance: float
    singular_threshold: float


def _tie_radius(value: float, relative_tolerance: float) -> float:
    return float(relative_tolerance * max(1.0, abs(float(value))))


def stable_quadratic_lower_root(
    leading: np.ndarray,
    linear_target: np.ndarray,
    constant: np.ndarray,
    *,
    discriminant_roundoff_multiplier: float = 128.0,
) -> tuple[np.ndarray, int]:
    """Return the smaller root of ``A*x^2 - B*x + C = 0``.

    A materially negative discriminant is an error.  A negative discriminant
    within a scale-aware floating-point roundoff bound is clamped to zero.
    The product-of-roots identity is used to avoid cancellation.
    """

    leading, linear_target, constant = np.broadcast_arrays(
        np.asarray(leading, dtype=np.float64),
        np.asarray(linear_target, dtype=np.float64),
        np.asarray(constant, dtype=np.float64),
    )
    if np.any(leading <= 0.0):
        raise ValueError("quadratic leading coefficients must be positive")
    raw_discriminant = linear_target * linear_target - 4.0 * leading * constant
    discriminant_scale = np.abs(linear_target * linear_target) + np.abs(4.0 * leading * constant)
    roundoff_bound = (
        float(discriminant_roundoff_multiplier)
        * np.finfo(np.float64).eps
        * np.maximum(discriminant_scale, np.finfo(np.float64).tiny)
    )
    materially_negative = raw_discriminant < -roundoff_bound
    if np.any(materially_negative):
        worst = float(np.min(raw_discriminant[materially_negative]))
        raise FloatingPointError(
            "the generalized-eigen discriminant is materially negative: " f"minimum={worst:.6e}"
        )
    clamped = (raw_discriminant < 0.0) & ~materially_negative
    discriminant = np.maximum(raw_discriminant, 0.0)
    square_root = np.sqrt(discriminant)
    signed = np.where(linear_target >= 0.0, 1.0, -1.0)
    stable_numerator = 0.5 * (linear_target + signed * square_root)

    first = stable_numerator / leading
    second = np.empty_like(first)
    nonzero = stable_numerator != 0.0
    second[nonzero] = constant[nonzero] / stable_numerator[nonzero]
    # The only exact zero-numerator case with a nonnegative discriminant is a
    # repeated zero root.  The direct expression is harmless there.
    second[~nonzero] = (linear_target[~nonzero] - square_root[~nonzero]) / (2.0 * leading[~nonzero])
    return np.minimum(first, second), int(np.count_nonzero(clamped))


def ordinary_pair_lower_eigenvalue(
    diagonal_first: float | np.ndarray,
    cross_difference: np.ndarray,
    diagonal_second: np.ndarray,
    null_correlation: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Evaluate the lower generalized eigenvalue for well-conditioned pairs."""

    a, b, c, g = np.broadcast_arrays(
        np.asarray(diagonal_first, dtype=np.float64),
        np.asarray(cross_difference, dtype=np.float64),
        np.asarray(diagonal_second, dtype=np.float64),
        np.asarray(null_correlation, dtype=np.float64),
    )
    leading = 1.0 - g * g
    linear_target = a + c - 2.0 * g * b
    constant = a * c - b * b
    return stable_quadratic_lower_root(leading, linear_target, constant)


def near_singular_pair_lower_eigenvalue(
    diagonal_first: float | np.ndarray,
    cross_difference: np.ndarray,
    diagonal_second: np.ndarray,
    null_correlation: np.ndarray,
) -> np.ndarray:
    """Evaluate near-singular positive-definite pairs in a whitened basis."""

    a, b, c, g = np.broadcast_arrays(
        np.asarray(diagonal_first, dtype=np.float64),
        np.asarray(cross_difference, dtype=np.float64),
        np.asarray(diagonal_second, dtype=np.float64),
        np.asarray(null_correlation, dtype=np.float64),
    )
    plus = 1.0 + g
    minus = 1.0 - g
    if np.any(plus <= 0.0) or np.any(minus <= 0.0):
        raise scipy.linalg.LinAlgError("near-singular null Gram matrix is not positive definite")

    # Q' D Q in the basis ((1,1)/sqrt(2), (1,-1)/sqrt(2)).
    symmetric = 0.5 * (a + c + 2.0 * b) / plus
    antisymmetric = 0.5 * (a + c - 2.0 * b) / minus
    coupling = 0.5 * (a - c) / np.sqrt(plus * minus)
    center = 0.5 * (symmetric + antisymmetric)
    radius = np.hypot(0.5 * (symmetric - antisymmetric), coupling)
    return center - radius


def _normalize_dictionary(
    dictionary: np.ndarray,
    null_covariance: np.ndarray,
) -> np.ndarray:
    dictionary = np.asarray(dictionary, dtype=np.float64)
    null_covariance = np.asarray(null_covariance, dtype=np.float64)
    null_times = null_covariance @ dictionary
    diagonal = np.einsum("ik,ik->k", dictionary, null_times, optimize=True)
    if np.any(~np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
        raise scipy.linalg.LinAlgError("dictionary contains a nonpositive null variance")
    return dictionary / np.sqrt(diagonal)[None, :]


def exhaustive_two_rectangle_search(
    standardized_dictionary: np.ndarray,
    intrinsic_difference: np.ndarray,
    null_covariance: np.ndarray,
    *,
    block_size: int = 256,
    tie_tolerance: float = 1.0e-12,
    singular_threshold: float = 1.0e-10,
    maximum_saved_near_singular: int = 10_000,
    progress: Callable[[int, int], None] | None = None,
) -> PairSearchResult:
    """Search all unordered column pairs without materializing all scores."""

    if block_size < 1:
        raise ValueError("block_size must be positive")
    if tie_tolerance < 0.0 or not 0.0 < singular_threshold < 1.0:
        raise ValueError("invalid pair-search tolerance")
    if maximum_saved_near_singular < 0:
        raise ValueError("maximum_saved_near_singular must be nonnegative")

    dictionary = _normalize_dictionary(standardized_dictionary, null_covariance)
    intrinsic_difference = np.asarray(intrinsic_difference, dtype=np.float64)
    null_covariance = np.asarray(null_covariance, dtype=np.float64)
    dimension, rectangle_count = dictionary.shape
    if intrinsic_difference.shape != (dimension, dimension):
        raise ValueError("intrinsic difference and dictionary dimensions do not align")
    if null_covariance.shape != (dimension, dimension):
        raise ValueError("null covariance and dictionary dimensions do not align")
    if rectangle_count < 2:
        raise ValueError("at least two rectangle columns are required")

    intrinsic_times = intrinsic_difference @ dictionary
    null_times = null_covariance @ dictionary
    intrinsic_diagonal = np.einsum("ik,ik->k", dictionary, intrinsic_times, optimize=True)

    minimum = np.inf
    candidates: list[PairCandidate] = []
    pair_count = 0
    ordinary_count = 0
    near_count = 0
    nonpositive_count = 0
    clamped_count = 0
    # Max-heap represented by negative A retains the smallest values of
    # A=1-g^2, i.e. the most singular saved diagnostics.
    near_heap: list[tuple[float, int, int, float, float]] = []

    def consider(first: int, second_indices: np.ndarray, values: np.ndarray) -> None:
        nonlocal minimum, candidates
        finite = np.isfinite(values)
        if not np.any(finite):
            return
        seconds = second_indices[finite]
        finite_values = values[finite]
        row_minimum = float(np.min(finite_values))
        if row_minimum < minimum:
            minimum = row_minimum
            radius = _tie_radius(minimum, tie_tolerance)
            candidates = [
                item for item in candidates if abs(item.analytic_eigenvalue - minimum) <= radius
            ]
        radius = _tie_radius(minimum, tie_tolerance)
        tied = np.abs(finite_values - minimum) <= radius
        candidates.extend(
            PairCandidate(first, int(second), float(value))
            for second, value in zip(seconds[tied], finite_values[tied])
        )

    total_blocks = (rectangle_count + block_size - 1) // block_size
    for block_number, first_start in enumerate(range(0, rectangle_count, block_size), start=1):
        first_stop = min(first_start + block_size, rectangle_count)
        block = dictionary[:, first_start:first_stop]
        cross_difference = block.T @ intrinsic_times
        cross_null = block.T @ null_times
        for local_first, first in enumerate(range(first_start, first_stop)):
            second_indices = np.arange(first + 1, rectangle_count, dtype=np.int64)
            if not len(second_indices):
                continue
            b = cross_difference[local_first, first + 1 :]
            g = cross_null[local_first, first + 1 :]
            a = float(intrinsic_diagonal[first])
            c = intrinsic_diagonal[first + 1 :]
            gram_determinant = 1.0 - g * g
            pair_count += len(second_indices)

            ordinary = gram_determinant > singular_threshold
            if np.any(ordinary):
                values, clamped = ordinary_pair_lower_eigenvalue(
                    a,
                    b[ordinary],
                    c[ordinary],
                    g[ordinary],
                )
                ordinary_count += int(np.count_nonzero(ordinary))
                clamped_count += clamped
                consider(first, second_indices[ordinary], values)

            near = ~ordinary
            if np.any(near):
                near_seconds = second_indices[near]
                near_g = g[near]
                near_a = gram_determinant[near]
                positive = (1.0 + near_g > 0.0) & (1.0 - near_g > 0.0)
                near_count += int(np.count_nonzero(positive))
                nonpositive_count += int(np.count_nonzero(~positive))
                near_values = np.full(len(near_seconds), np.nan, dtype=np.float64)
                if np.any(positive):
                    near_values[positive] = near_singular_pair_lower_eigenvalue(
                        a,
                        b[near][positive],
                        c[near][positive],
                        near_g[positive],
                    )
                    consider(first, near_seconds[positive], near_values[positive])

                if maximum_saved_near_singular:
                    for second, determinant, correlation, value in zip(
                        near_seconds,
                        near_a,
                        near_g,
                        near_values,
                    ):
                        entry = (
                            -float(determinant),
                            int(first),
                            int(second),
                            float(correlation),
                            float(value),
                        )
                        if len(near_heap) < maximum_saved_near_singular:
                            heapq.heappush(near_heap, entry)
                        elif float(determinant) < -near_heap[0][0]:
                            heapq.heapreplace(near_heap, entry)
        if progress is not None:
            progress(block_number, total_blocks)

    if not np.isfinite(minimum):
        raise scipy.linalg.LinAlgError("no positive-definite two-rectangle pair was found")
    radius = _tie_radius(minimum, tie_tolerance)
    candidates = sorted(
        (item for item in candidates if abs(item.analytic_eigenvalue - minimum) <= radius),
        key=lambda item: (item.first_index, item.second_index),
    )
    # Defensive uniqueness check: every unordered pair should be visited once.
    unique = {(item.first_index, item.second_index) for item in candidates}
    if len(unique) != len(candidates):
        raise AssertionError("the block search accumulated a duplicate pair")

    near_records = []
    for negative_determinant, first, second, correlation, value in sorted(
        near_heap,
        key=lambda item: (-item[0], item[1], item[2]),
    ):
        near_records.append(
            {
                "first_index": first,
                "second_index": second,
                "g_ij": correlation,
                "one_minus_g_squared": -negative_determinant,
                "lower_eigenvalue": value,
            }
        )
    expected = rectangle_count * (rectangle_count - 1) // 2
    if pair_count != expected:
        raise AssertionError(f"visited {pair_count} pairs, expected {expected}")
    if ordinary_count + near_count + nonpositive_count != expected:
        raise AssertionError("pair accounting does not sum to the expected total")
    return PairSearchResult(
        minimum_eigenvalue=float(minimum),
        candidates=tuple(candidates),
        pair_count=pair_count,
        ordinary_pair_count=ordinary_count,
        near_singular_pair_count=near_count,
        nonpositive_gram_pair_count=nonpositive_count,
        clamped_discriminant_count=clamped_count,
        near_singular_records=tuple(near_records),
        block_size=int(block_size),
        tie_tolerance=float(tie_tolerance),
        singular_threshold=float(singular_threshold),
    )


def validate_pair_candidate(
    candidate: PairCandidate,
    standardized_dictionary: np.ndarray,
    intrinsic_difference: np.ndarray,
    null_covariance: np.ndarray,
) -> dict[str, float | int | str]:
    """Recheck one candidate with SciPy and both reduced-space residuals."""

    columns = np.asarray([candidate.first_index, candidate.second_index], dtype=np.int64)
    basis = np.asarray(standardized_dictionary, dtype=np.float64)[:, columns].copy()
    null_covariance = np.asarray(null_covariance, dtype=np.float64)
    basis_null = null_covariance @ basis
    basis_variances = np.einsum("ik,ik->k", basis, basis_null, optimize=True)
    if np.any(basis_variances <= 0.0):
        raise scipy.linalg.LinAlgError("candidate has a nonpositive null variance")
    basis /= np.sqrt(basis_variances)[None, :]
    difference = basis.T @ np.asarray(intrinsic_difference, dtype=np.float64) @ basis
    gram = basis.T @ null_covariance @ basis
    difference = 0.5 * (difference + difference.T)
    gram = 0.5 * (gram + gram.T)
    values, vectors = scipy.linalg.eigh(
        difference,
        gram,
        type=1,
        check_finite=False,
    )
    eigenvalue = float(values[0])
    coefficients = np.asarray(vectors[:, 0], dtype=np.float64)
    filter_weights = basis @ coefficients
    largest = int(np.argmax(np.abs(filter_weights)))
    if filter_weights[largest] < 0.0:
        coefficients = -coefficients
        filter_weights = -filter_weights
    left = difference @ coefficients
    right = eigenvalue * (gram @ coefficients)
    residual = float(
        np.linalg.norm(left - right)
        / max(np.linalg.norm(left) + np.linalg.norm(right), np.finfo(np.float64).tiny)
    )
    ratio = (
        float(coefficients[1] / coefficients[0])
        if abs(coefficients[0]) > np.finfo(np.float64).eps
        else float(np.copysign(np.inf, coefficients[1]))
    )
    return {
        "first_index": candidate.first_index,
        "second_index": candidate.second_index,
        "analytic_eigenvalue": candidate.analytic_eigenvalue,
        "scipy_eigenvalue": eigenvalue,
        "analytic_scipy_absolute_difference": abs(candidate.analytic_eigenvalue - eigenvalue),
        "reduced_relative_residual": residual,
        "null_variance": float(filter_weights @ null_covariance @ filter_weights),
        "intrinsic_quadratic": float(filter_weights @ intrinsic_difference @ filter_weights),
        "coefficient_first": float(coefficients[0]),
        "coefficient_second": float(coefficients[1]),
        "eigenvector_ratio_second_over_first": ratio,
        # This is meaningful only under the dictionary's stored endpoint
        # orientation.  Reversing either rectangle atom reverses its
        # coefficient without changing the resulting observation filter.
        "relative_coefficient_sign_canonical": (
            "opposite" if coefficients[0] * coefficients[1] < 0.0 else "same"
        ),
        "d_ii": float(difference[0, 0]),
        "d_ij": float(difference[0, 1]),
        "d_jj": float(difference[1, 1]),
        "g_ii": float(gram[0, 0]),
        "g_ij": float(gram[0, 1]),
        "g_jj": float(gram[1, 1]),
        "one_minus_g_squared": float(1.0 - gram[0, 1] ** 2),
    }
