from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
MODULE_DIR = HERE.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from rectangle_dictionary_core import (  # noqa: E402
    cluster_overlap,
    constrained_relative_residual,
    dictionary_svd,
    direction_retention,
    greedy_extreme_path,
    projected_gaussian_kl,
    recover_rectangle_coefficients,
    rectangle_matrix,
    rectangle_structure_errors,
    sigma0_metric_projection,
    solve_constrained_contrast,
    standardize_dictionary,
)


def random_spd(rng: np.random.Generator, dimension: int, ridge: float = 0.5) -> np.ndarray:
    factor = rng.normal(size=(dimension, dimension))
    return factor @ factor.T / dimension + ridge * np.eye(dimension)


def test_rectangle_vector_matches_direct_four_point_difference() -> None:
    anchors = 3
    times = 4
    matrix, endpoints = rectangle_matrix(
        anchors,
        times,
        np.asarray([[0, 2, 1, 3]], dtype=np.int64),
    )
    values = np.arange(anchors * times, dtype=np.float64).reshape(times, anchors)
    p, q, k, ell = endpoints[0]
    direct = values[ell, q] - values[ell, p] - values[k, q] + values[k, p]

    np.testing.assert_allclose(matrix[:, 0] @ values.reshape(-1), direct)
    assert matrix[k * anchors + p, 0] == 1.0
    assert matrix[k * anchors + q, 0] == -1.0
    assert matrix[ell * anchors + p, 0] == -1.0
    assert matrix[ell * anchors + q, 0] == 1.0
    np.testing.assert_allclose(
        rectangle_structure_errors(matrix, anchor_count=anchors, time_count=times),
        0.0,
    )


def test_cross_covariance_can_create_both_extremes() -> None:
    null = np.eye(2)
    intrinsic = np.asarray([[0.0, 0.3], [0.3, 0.0]])
    values, _, filters = solve_constrained_contrast(intrinsic, null)

    np.testing.assert_allclose(values, [-0.3, 0.3], atol=1.0e-14)
    for index, value in enumerate(values):
        np.testing.assert_allclose(filters[:, index] @ null @ filters[:, index], 1.0)
        np.testing.assert_allclose(
            filters[:, index] @ intrinsic @ filters[:, index],
            value,
        )
        largest = int(np.argmax(np.abs(filters[:, index])))
        assert filters[largest, index] >= 0.0


def test_rectangle_variance_matches_separable_four_point_formula() -> None:
    spatial_correlation = 0.37
    temporal_correlation = 0.61
    variance = 2.4
    spatial = np.asarray([[1.0, spatial_correlation], [spatial_correlation, 1.0]])
    temporal = np.asarray([[1.0, temporal_correlation], [temporal_correlation, 1.0]])
    covariance = variance * np.kron(temporal, spatial)
    rectangle, _ = rectangle_matrix(2, 2)

    observed = float(rectangle[:, 0] @ covariance @ rectangle[:, 0])
    expected = 4.0 * variance * (1.0 - spatial_correlation) * (1.0 - temporal_correlation)
    np.testing.assert_allclose(observed, expected, rtol=1.0e-14, atol=1.0e-14)


def test_rank_reduction_and_coefficient_recovery_ignore_duplicate_columns() -> None:
    rng = np.random.default_rng(11)
    raw = rng.normal(size=(8, 5))
    raw = np.column_stack([raw, raw[:, 2], -2.0 * raw[:, 4]])
    null = random_spd(rng, 8)
    standardized, scales, _ = standardize_dictionary(raw, null)
    decomposition = dictionary_svd(standardized, relative_tolerance=1.0e-11)
    intrinsic = random_spd(rng, 8) - random_spd(rng, 8)
    values, coefficients, filters = solve_constrained_contrast(
        intrinsic,
        null,
        decomposition.basis,
    )

    assert decomposition.rank == 5
    for index in (0, -1):
        alpha_standardized, alpha_raw = recover_rectangle_coefficients(
            decomposition,
            coefficients[:, index],
            scales,
        )
        np.testing.assert_allclose(
            standardized @ alpha_standardized,
            filters[:, index],
            rtol=1.0e-11,
            atol=1.0e-11,
        )
        np.testing.assert_allclose(
            raw @ alpha_raw,
            filters[:, index],
            rtol=1.0e-11,
            atol=1.0e-11,
        )
        np.testing.assert_allclose(filters[:, index] @ null @ filters[:, index], 1.0)
        assert np.isfinite(values[index])


def test_sigma0_metric_projection_has_valid_retention_and_cluster_angles() -> None:
    rng = np.random.default_rng(21)
    null = random_spd(rng, 10)
    basis, _ = np.linalg.qr(rng.normal(size=(10, 6)))
    directions = rng.normal(size=(10, 2))
    projected, _ = sigma0_metric_projection(basis, null, directions)
    retention = direction_retention(projected, directions, null)
    overlap = cluster_overlap(basis, directions, null)

    assert np.all((retention >= 0.0) & (retention <= 1.0))
    squared = overlap["squared_canonical_correlations"]
    assert np.all((squared >= 0.0) & (squared <= 1.0))
    np.testing.assert_allclose(
        sigma0_metric_projection(basis, null, projected)[0],
        projected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_dense_extrema_are_invariant_to_dictionary_rescaling_and_duplication() -> None:
    rng = np.random.default_rng(31)
    null = random_spd(rng, 9)
    intrinsic = random_spd(rng, 9) - random_spd(rng, 9)
    raw = rng.normal(size=(9, 6))
    expanded = np.column_stack([raw * np.arange(1.0, 7.0), raw[:, 1], -raw[:, 3]])

    extrema = []
    for dictionary in (raw, expanded):
        standardized, _, _ = standardize_dictionary(dictionary, null)
        decomposition = dictionary_svd(standardized, relative_tolerance=1.0e-11)
        values, _, _ = solve_constrained_contrast(
            intrinsic,
            null,
            decomposition.basis,
        )
        extrema.append(values[[0, -1]])
    np.testing.assert_allclose(extrema[0], extrema[1], rtol=1.0e-11, atol=1.0e-11)

    full_values, _, _ = solve_constrained_contrast(intrinsic, null)
    standardized, _, _ = standardize_dictionary(raw, null)
    single_scores = np.einsum(
        "ik,ij,jk->k",
        standardized,
        intrinsic,
        standardized,
        optimize=True,
    )
    assert full_values[-1] >= extrema[0][-1] >= single_scores.max() - 1.0e-11
    assert full_values[0] <= extrema[0][0] <= single_scores.min() + 1.0e-11


def test_greedy_path_reoptimizes_and_is_monotone() -> None:
    rng = np.random.default_rng(41)
    null = random_spd(rng, 12)
    intrinsic = random_spd(rng, 12) - random_spd(rng, 12)
    raw = rng.normal(size=(12, 18))
    raw[:, -1] = raw[:, 0]
    standardized, _, _ = standardize_dictionary(raw, null)

    positive = greedy_extreme_path(
        standardized,
        intrinsic,
        null,
        branch="positive",
        maximum_size=5,
    )
    negative = greedy_extreme_path(
        standardized,
        intrinsic,
        null,
        branch="negative",
        maximum_size=5,
    )

    assert np.all(np.diff([step.eigenvalue for step in positive]) >= -1.0e-10)
    assert np.all(np.diff([step.eigenvalue for step in negative]) <= 1.0e-10)
    for path in (positive, negative):
        for step in path:
            np.testing.assert_allclose(
                step.filter_weights @ null @ step.filter_weights,
                1.0,
                rtol=1.0e-10,
                atol=1.0e-10,
            )
            largest = int(np.argmax(np.abs(step.filter_weights)))
            assert step.filter_weights[largest] >= 0.0


def test_tolerance_tie_uses_smallest_dictionary_index() -> None:
    null = np.eye(3)
    intrinsic = np.diag([1.0, 1.0, -0.5])
    dictionary = np.eye(3)

    step = greedy_extreme_path(
        dictionary,
        intrinsic,
        null,
        branch="positive",
        maximum_size=1,
        score_tie_tolerance=1.0e-12,
    )[0]

    assert step.added_index == 0
    assert step.winner_tie_count == 2


def test_constrained_residual_is_invariant_to_basis_reparameterization() -> None:
    rng = np.random.default_rng(51)
    null = random_spd(rng, 9)
    intrinsic = random_spd(rng, 9) - random_spd(rng, 9)
    basis = rng.normal(size=(9, 4))
    values, _, filters = solve_constrained_contrast(intrinsic, null, basis)
    transform = rng.normal(size=(4, 4))
    while abs(np.linalg.det(transform)) < 0.1:
        transform = rng.normal(size=(4, 4))

    first = constrained_relative_residual(
        filters[:, -1],
        values[-1],
        basis,
        intrinsic,
        null,
    )
    second = constrained_relative_residual(
        filters[:, -1],
        values[-1],
        basis @ transform,
        intrinsic,
        null,
    )

    np.testing.assert_allclose(first, second, rtol=1.0e-8, atol=1.0e-13)


def test_projected_kl_matches_scalar_g_and_is_coordinate_invariant() -> None:
    rng = np.random.default_rng(61)
    null = random_spd(rng, 7)
    truth = random_spd(rng, 7)
    one_filter = rng.normal(size=7)
    v0 = float(one_filter @ null @ one_filter)
    v1 = float(one_filter @ truth @ one_filter)
    ratio = v1 / v0
    expected = 0.5 * (ratio - 1.0 - np.log(ratio))
    np.testing.assert_allclose(
        projected_gaussian_kl(one_filter, truth, null),
        expected,
        rtol=1.0e-12,
        atol=1.0e-12,
    )

    filters = rng.normal(size=(7, 2))
    transform = np.asarray([[1.3, -0.2], [0.4, 0.9]])
    np.testing.assert_allclose(
        projected_gaussian_kl(filters, truth, null),
        projected_gaussian_kl(filters @ transform, truth, null),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_exact_rectangle_dictionary_has_theoretical_rank() -> None:
    raw, endpoints = rectangle_matrix(3, 3)
    standardized, _, _ = standardize_dictionary(raw, np.eye(9))
    decomposition = dictionary_svd(standardized, relative_tolerance=1.0e-12)

    assert len(endpoints) == 9
    assert decomposition.rank == (3 - 1) * (3 - 1)
