from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.linalg


HERE = Path(__file__).resolve().parent
MODULE_DIR = HERE.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from analyze_mode_count_path import (  # noqa: E402
    clipped_k_grid,
    llr_from_eigenvalues,
    monte_carlo_llr,
)


def direct_llr(values: np.ndarray, null: np.ndarray, alternative: np.ndarray) -> float:
    null_factor = scipy.linalg.cholesky(null, lower=True)
    alternative_factor = scipy.linalg.cholesky(alternative, lower=True)
    null_quad = values @ scipy.linalg.cho_solve((null_factor, True), values)
    alternative_quad = values @ scipy.linalg.cho_solve((alternative_factor, True), values)
    return float(
        0.5
        * (
            2.0 * np.log(np.diag(null_factor)).sum()
            - 2.0 * np.log(np.diag(alternative_factor)).sum()
            + null_quad
            - alternative_quad
        )
    )


def test_eigenvalue_llr_formula_matches_direct_diagonal_covariances() -> None:
    eigenvalues = np.asarray([0.35, 1.0, 2.6])
    squared = np.asarray([[0.2, 1.3, 2.1], [3.0, 0.5, 0.7]])
    null = np.eye(3)
    alternative = np.diag(eigenvalues)

    expected_null = np.asarray([direct_llr(np.sqrt(row), null, alternative) for row in squared])
    expected_alternative = np.asarray(
        [direct_llr(np.sqrt(eigenvalues * row), null, alternative) for row in squared]
    )
    np.testing.assert_allclose(
        llr_from_eigenvalues(squared, eigenvalues, hypothesis="null"),
        expected_null,
        rtol=1e-14,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        llr_from_eigenvalues(squared, eigenvalues, hypothesis="alternative"),
        expected_alternative,
        rtol=1e-14,
        atol=1e-14,
    )


def test_equal_covariances_give_identically_zero_llr() -> None:
    squared = np.square(np.random.default_rng(42).standard_normal((20, 4)))
    eigenvalues = np.ones(4)
    np.testing.assert_array_equal(
        llr_from_eigenvalues(squared, eigenvalues, hypothesis="null"),
        np.zeros(20),
    )
    np.testing.assert_array_equal(
        llr_from_eigenvalues(squared, eigenvalues, hypothesis="alternative"),
        np.zeros(20),
    )


def test_chunked_monte_carlo_is_reproducible_and_detects_strong_alternative() -> None:
    eigenvalues = np.asarray([0.1, 6.0, 10.0])
    first = monte_carlo_llr(
        eigenvalues,
        observed_llr=0.0,
        replicates=10_000,
        alpha=0.05,
        random_seed=12345,
        chunk_size=137,
    )
    second = monte_carlo_llr(
        eigenvalues,
        observed_llr=0.0,
        replicates=10_000,
        alpha=0.05,
        random_seed=12345,
        chunk_size=137,
    )
    assert first == second
    assert first.power > 0.65
    assert 0.0 < first.p_value <= 1.0


def test_k_grid_clips_and_deduplicates_without_reordering() -> None:
    assert clipped_k_grid([1, 4, 20, 800, 900], dimension=100) == [1, 4, 20, 100]
