from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
MODULE_DIR = HERE.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from explore_eigen_directions import (  # noqa: E402
    assign_near_degenerate_clusters,
    cluster_space_time_spectral_energy,
    cluster_subspace_heatmap_arrays,
    direction_covariance_metrics,
    fiedler_anchor_order,
    graph_fourier_basis,
    knn_edges,
    pareto_mask,
    pattern_metrics,
    quadratic_diagonal,
)


def test_interaction_and_compensation_decomposition_is_exact() -> None:
    fitted_null = np.asarray(
        [
            [1.5, 0.2, 0.1],
            [0.2, 1.2, 0.05],
            [0.1, 0.05, 0.9],
        ]
    )
    matched = np.asarray(
        [
            [1.4, 0.3, 0.05],
            [0.3, 1.1, 0.1],
            [0.05, 0.1, 1.0],
        ]
    )
    truth = np.asarray(
        [
            [1.4, 0.4, 0.08],
            [0.4, 1.1, 0.18],
            [0.08, 0.18, 1.0],
        ]
    )
    directions = np.asarray(
        [
            [1.0, 0.2, -0.4],
            [0.1, -1.0, 0.5],
            [0.3, 0.4, 1.0],
        ]
    )

    normalized, frame = direction_covariance_metrics(
        directions,
        truth,
        fitted_null,
        matched,
    )

    np.testing.assert_allclose(quadratic_diagonal(fitted_null, normalized), 1.0)
    np.testing.assert_allclose(
        frame["delta_total"],
        frame["delta_intrinsic_interaction"] + frame["delta_fitted_null_compensation"],
        rtol=1e-14,
        atol=1e-14,
    )
    assert frame["interaction_absolute_share"].between(0.0, 1.0).all()
    assert frame["cancellation_fraction"].between(0.0, 1.0).all()


def test_near_degenerate_clusters_are_branch_specific() -> None:
    eigenvalues = np.asarray([1.30, 1.299, 1.20, 0.80, 0.801, 0.60])
    labels, sizes = assign_near_degenerate_clusters(
        eigenvalues,
        relative_log_gap=0.02,
    )

    assert labels[0] == labels[1]
    assert labels[3] == labels[4]
    assert labels[0] != labels[3]
    assert sizes.tolist() == [2, 2, 1, 2, 2, 1]


def test_pareto_mask_handles_equal_first_objective_without_order_bias() -> None:
    first = np.asarray([1.0, 1.0, 0.8, 0.7])
    second = np.asarray([0.2, 0.4, 0.6, 0.3])

    np.testing.assert_array_equal(
        pareto_mask(first, second),
        np.asarray([False, True, True, False]),
    )


def test_rank_one_alternating_filter_has_rank_one_space_time_pattern() -> None:
    temporal = np.asarray([1.0, -1.0, 1.0, -1.0])
    spatial = np.asarray([1.0, 0.5, -0.25])
    direction = np.outer(temporal, spatial).reshape(-1, 1)
    coordinates = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    graph_edges = {2: knn_edges(coordinates, 2)}

    metrics, temporal_frame = pattern_metrics(
        direction,
        time_count=4,
        anchor_count=3,
        graph_edges=graph_edges,
    )

    np.testing.assert_allclose(metrics.loc[0, "rank1_energy_fraction"], 1.0, atol=1e-14)
    np.testing.assert_allclose(metrics.loc[0, "space_time_effective_rank"], 1.0, atol=1e-14)
    assert metrics.loc[0, "temporal_first_difference_roughness"] > 2.0
    np.testing.assert_allclose(temporal_frame["time_energy_fraction"].sum(), 1.0)


def test_cluster_heatmap_summaries_are_invariant_to_mode_rotation() -> None:
    rng = np.random.default_rng(42)
    directions = rng.normal(size=(12, 2))
    angle = 0.73
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )

    original = cluster_subspace_heatmap_arrays(
        directions,
        time_count=4,
        anchor_count=3,
    )
    rotated = cluster_subspace_heatmap_arrays(
        directions @ rotation,
        time_count=4,
        anchor_count=3,
    )

    for original_value, rotated_value in zip(original[1:], rotated[1:]):
        np.testing.assert_allclose(original_value, rotated_value, rtol=1e-13, atol=1e-13)

    _, graph_eigenvectors = np.linalg.eigh(
        np.asarray(
            [
                [2.0, -1.0, 0.0],
                [-1.0, 2.0, -1.0],
                [0.0, -1.0, 2.0],
            ]
        )
    )
    original_spectrum = cluster_space_time_spectral_energy(original[0], graph_eigenvectors)
    rotated_spectrum = cluster_space_time_spectral_energy(rotated[0], graph_eigenvectors)
    np.testing.assert_allclose(original_spectrum, rotated_spectrum, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(original_spectrum.sum(), 1.0, atol=1e-14)


def test_fiedler_anchor_order_is_a_deterministic_permutation() -> None:
    coordinates = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ]
    )

    first = fiedler_anchor_order(coordinates, neighbors=2)
    second = fiedler_anchor_order(coordinates, neighbors=2)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(np.sort(first), np.arange(len(coordinates)))

    frequencies, eigenvectors = graph_fourier_basis(coordinates, neighbors=2)
    assert frequencies[0] >= -1.0e-12
    assert frequencies[1] > 1.0e-10
    assert np.all(np.diff(frequencies) >= -1.0e-12)
    np.testing.assert_allclose(
        eigenvectors.T @ eigenvectors,
        np.eye(len(coordinates)),
        rtol=1.0e-13,
        atol=1.0e-13,
    )
