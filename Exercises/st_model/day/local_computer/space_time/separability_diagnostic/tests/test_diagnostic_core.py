from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
MODULE_DIR = HERE.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from diagnostic_core import (  # noqa: E402
    CovarianceParameters,
    advected_separable_correlation,
    advected_separable_covariance,
    balanced_fixed_radius_gaps,
    g_score,
    joint_matern_half_correlation,
    pairwise_lags,
    profiled_null_objective,
    same_margin_exponential_correlation_gap,
    same_margin_exponential_log_gap,
    solve_generalized_eigenproblem,
    squared_exponential_correlation_from_norms,
    standardized_moving_lag_norms,
    standardize_directions_for_design,
)


PARAMETERS = CovarianceParameters(
    variance=10.0,
    range_lat=0.2,
    range_lon=0.3,
    range_time=2.0,
    advec_lat=0.08,
    advec_lon=-0.2,
    nugget=0.0,
)


def test_axis_margins_match_but_mixed_lags_do_not() -> None:
    spatial_coordinates = np.asarray([[0.0, 0.0, 0.0], [0.11, -0.17, 0.0]], dtype=np.float64)
    spatial = pairwise_lags(spatial_coordinates)
    np.testing.assert_allclose(
        joint_matern_half_correlation(spatial, PARAMETERS),
        advected_separable_correlation(spatial, PARAMETERS),
        rtol=1e-13,
        atol=1e-13,
    )

    time = 1.5
    flow_coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [PARAMETERS.advec_lat * time, PARAMETERS.advec_lon * time, time],
        ],
        dtype=np.float64,
    )
    flow = pairwise_lags(flow_coordinates)
    np.testing.assert_allclose(
        joint_matern_half_correlation(flow, PARAMETERS),
        advected_separable_correlation(flow, PARAMETERS),
        rtol=1e-13,
        atol=1e-13,
    )

    mixed_coordinates = flow_coordinates.copy()
    mixed_coordinates[1, 0] += PARAMETERS.range_lat
    mixed = pairwise_lags(mixed_coordinates)
    difference = np.max(
        np.abs(
            joint_matern_half_correlation(mixed, PARAMETERS)
            - advected_separable_correlation(mixed, PARAMETERS)
        )
    )
    assert difference > 1e-3


def test_standardized_moving_norms_and_same_margin_gap_geometry() -> None:
    time = 1.5
    coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [PARAMETERS.advec_lat * time + PARAMETERS.range_lat, PARAMETERS.advec_lon * time, time],
        ],
        dtype=np.float64,
    )
    spatial_norm, temporal_norm = standardized_moving_lag_norms(
        pairwise_lags(coordinates), PARAMETERS
    )
    np.testing.assert_allclose(spatial_norm[0, 1], 1.0, atol=1e-14)
    np.testing.assert_allclose(temporal_norm[0, 1], time / PARAMETERS.range_time, atol=1e-14)

    axis = np.asarray([0.0, 0.4, 2.0])
    np.testing.assert_allclose(same_margin_exponential_log_gap(axis, 0.0), 0.0)
    np.testing.assert_allclose(same_margin_exponential_log_gap(0.0, axis), 0.0)
    np.testing.assert_allclose(same_margin_exponential_correlation_gap(axis, 0.0), 0.0)
    np.testing.assert_allclose(same_margin_exponential_correlation_gap(0.0, axis), 0.0)
    assert float(same_margin_exponential_log_gap(0.6, 0.8)) > 0.0
    assert float(same_margin_exponential_correlation_gap(0.6, 0.8)) > 0.0


def test_removing_outer_sqrt_is_separable_but_changes_axis_margins() -> None:
    spatial = np.asarray([0.25, 0.8, 1.7])
    temporal = np.asarray([0.5, 1.2, 0.3])
    no_sqrt = squared_exponential_correlation_from_norms(spatial, temporal)
    np.testing.assert_allclose(
        no_sqrt,
        np.exp(-(spatial**2)) * np.exp(-(temporal**2)),
        rtol=1e-14,
        atol=1e-14,
    )
    # On an axis, the original nu=1/2 margin is exp(-s), not exp(-s**2).
    assert not np.allclose(
        squared_exponential_correlation_from_norms(spatial, 0.0),
        np.exp(-spatial),
    )


def test_balanced_lag_maximizes_gap_at_fixed_joint_radius() -> None:
    radius = 2.3
    theta = np.linspace(0.0, 0.5 * np.pi, 20_001)
    spatial = radius * np.cos(theta)
    temporal = radius * np.sin(theta)
    log_gap = same_margin_exponential_log_gap(spatial, temporal)
    correlation_gap = same_margin_exponential_correlation_gap(spatial, temporal)
    expected_log, expected_correlation = balanced_fixed_radius_gaps(radius)
    np.testing.assert_allclose(np.max(log_gap), expected_log, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(np.max(correlation_gap), expected_correlation, rtol=1e-9, atol=1e-12)
    maximizing_theta = theta[int(np.argmax(log_gap))]
    np.testing.assert_allclose(maximizing_theta, np.pi / 4.0, atol=1e-12)


def test_profiled_variance_recovers_a_separable_covariance_scale() -> None:
    coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.2, 0.0],
            [0.08, 0.0, 1.0],
            [0.18, 0.2, 1.0],
        ],
        dtype=np.float64,
    )
    geometry = pairwise_lags(coordinates)
    covariance = advected_separable_covariance(geometry, PARAMETERS, numerical_jitter_ratio=1e-10)
    raw = np.asarray(
        [
            np.log(PARAMETERS.range_lat),
            np.log(PARAMETERS.range_lon),
            np.log(PARAMETERS.range_time),
            PARAMETERS.advec_lat,
            PARAMETERS.advec_lon,
        ]
    )
    _, variance = profiled_null_objective(
        raw,
        [geometry],
        [covariance],
        numerical_jitter_ratio=1e-10,
    )
    np.testing.assert_allclose(variance, PARAMETERS.variance, rtol=1e-11, atol=1e-11)


def test_generalized_eigen_solver_and_kl_identity() -> None:
    null = np.diag([2.0, 3.0, 5.0])
    expected = np.asarray([0.25, 1.0, 4.0])
    true = np.diag(np.diag(null) * expected)
    result = solve_generalized_eigenproblem(true, null)
    order = np.argsort(g_score(expected))[::-1]
    np.testing.assert_allclose(result.eigenvalues, expected[order], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(result.matrix_kl, result.spectral_kl, rtol=1e-13, atol=1e-13)
    assert result.max_relative_residual < 1e-14
    assert result.max_null_orthonormality_error < 1e-14


def test_transferred_directions_have_unit_null_variance() -> None:
    null = np.asarray([[2.0, 0.4], [0.4, 1.0]])
    true = np.asarray([[2.3, 0.1], [0.1, 0.8]])
    directions = np.asarray([[1.0, 1.0], [0.2, -0.7]])
    standardized, projected_null, projected_true = standardize_directions_for_design(
        directions, null, true
    )
    np.testing.assert_allclose(np.diag(projected_null), 1.0, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(projected_null, projected_null.T, atol=1e-14)
    np.testing.assert_allclose(projected_true, projected_true.T, atol=1e-14)
    assert standardized.shape == directions.shape


def test_equal_models_give_lambda_one_and_zero_score() -> None:
    covariance = np.asarray([[1.0, 0.3], [0.3, 1.0]])
    result = solve_generalized_eigenproblem(covariance, covariance)
    np.testing.assert_allclose(result.eigenvalues, 1.0, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(result.scores, 0.0, atol=1e-14)
