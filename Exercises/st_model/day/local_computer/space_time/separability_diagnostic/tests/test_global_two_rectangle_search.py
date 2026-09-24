from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.linalg


HERE = Path(__file__).resolve().parent
MODULE_DIR = HERE.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from global_two_rectangle_search import (  # noqa: E402
    exhaustive_two_rectangle_search,
    near_singular_pair_lower_eigenvalue,
    ordinary_pair_lower_eigenvalue,
    stable_quadratic_lower_root,
    validate_pair_candidate,
)
from run_global_two_rectangle_search import _canonical_equivalence_key  # noqa: E402


def random_spd(rng: np.random.Generator, dimension: int) -> np.ndarray:
    factor = rng.normal(size=(dimension, dimension))
    return factor @ factor.T / dimension + 0.5 * np.eye(dimension)


def test_closed_form_matches_scipy_for_random_two_by_two_pairs() -> None:
    rng = np.random.default_rng(20260922)
    count = 200
    a = rng.normal(size=count)
    b = rng.normal(size=count)
    c = rng.normal(size=count)
    g = rng.uniform(-0.95, 0.95, size=count)
    observed, clamped = ordinary_pair_lower_eigenvalue(a, b, c, g)
    expected = []
    for ai, bi, ci, gi in zip(a, b, c, g):
        values = scipy.linalg.eigh(
            np.asarray([[ai, bi], [bi, ci]]),
            np.asarray([[1.0, gi], [gi, 1.0]]),
            eigvals_only=True,
            check_finite=False,
        )
        expected.append(values[0])
    np.testing.assert_allclose(observed, expected, rtol=2.0e-13, atol=2.0e-13)
    assert clamped == 0


def test_near_singular_whitening_matches_scipy() -> None:
    a = np.asarray([0.2, -0.3, 1.1])
    b = np.asarray([0.05, 0.4, -0.7])
    c = np.asarray([0.8, 0.9, -0.2])
    g = np.asarray([1.0 - 1.0e-12, -1.0 + 2.0e-12, 1.0 - 5.0e-11])
    observed = near_singular_pair_lower_eigenvalue(a, b, c, g)
    expected = [
        scipy.linalg.eigh(
            np.asarray([[ai, bi], [bi, ci]]),
            np.asarray([[1.0, gi], [gi, 1.0]]),
            eigvals_only=True,
            check_finite=False,
        )[0]
        for ai, bi, ci, gi in zip(a, b, c, g)
    ]
    np.testing.assert_allclose(observed, expected, rtol=2.0e-5, atol=2.0e-5)


def test_roundoff_negative_discriminant_is_clamped() -> None:
    root, clamped = stable_quadratic_lower_root(
        np.asarray([1.0]),
        np.asarray([2.0]),
        np.asarray([1.0 + np.finfo(np.float64).eps]),
    )
    np.testing.assert_allclose(root, [1.0], rtol=0.0, atol=2.0e-8)
    assert clamped == 1


def test_block_search_matches_brute_force_and_retains_all_ties() -> None:
    rng = np.random.default_rng(13)
    dimension = 7
    columns = 9
    null = random_spd(rng, dimension)
    dictionary = rng.normal(size=(dimension, columns))
    intrinsic = np.zeros((dimension, dimension), dtype=np.float64)
    result = exhaustive_two_rectangle_search(
        dictionary,
        intrinsic,
        null,
        block_size=3,
        tie_tolerance=1.0e-12,
    )
    assert result.pair_count == columns * (columns - 1) // 2
    assert result.minimum_eigenvalue == 0.0
    assert len(result.candidates) == result.pair_count
    assert result.near_singular_pair_count == 0
    assert result.nonpositive_gram_pair_count == 0


def test_block_search_random_minimum_and_validation_match_brute_force() -> None:
    rng = np.random.default_rng(17)
    dimension = 8
    columns = 11
    null = random_spd(rng, dimension)
    raw_intrinsic = rng.normal(size=(dimension, dimension))
    intrinsic = 0.5 * (raw_intrinsic + raw_intrinsic.T)
    dictionary = rng.normal(size=(dimension, columns))
    result = exhaustive_two_rectangle_search(
        dictionary,
        intrinsic,
        null,
        block_size=4,
        tie_tolerance=1.0e-12,
    )

    null_times = null @ dictionary
    scales = np.sqrt(np.einsum("ik,ik->k", dictionary, null_times, optimize=True))
    standardized = dictionary / scales[None, :]
    brute = []
    for first in range(columns):
        for second in range(first + 1, columns):
            basis = standardized[:, [first, second]]
            values = scipy.linalg.eigh(
                basis.T @ intrinsic @ basis,
                basis.T @ null @ basis,
                eigvals_only=True,
                check_finite=False,
            )
            brute.append((float(values[0]), first, second))
    expected = min(brute)
    np.testing.assert_allclose(result.minimum_eigenvalue, expected[0], rtol=1.0e-12)
    assert (result.candidates[0].first_index, result.candidates[0].second_index) == expected[1:]

    audit = validate_pair_candidate(
        result.candidates[0],
        dictionary,
        intrinsic,
        null,
    )
    np.testing.assert_allclose(audit["scipy_eigenvalue"], expected[0], rtol=1.0e-12)
    assert audit["reduced_relative_residual"] < 1.0e-12
    np.testing.assert_allclose(audit["null_variance"], 1.0, atol=1.0e-12)


def test_block_partition_does_not_change_minimum_or_ties() -> None:
    rng = np.random.default_rng(29)
    dimension = 9
    columns = 17
    null = random_spd(rng, dimension)
    raw_intrinsic = rng.normal(size=(dimension, dimension))
    intrinsic = 0.5 * (raw_intrinsic + raw_intrinsic.T)
    dictionary = rng.normal(size=(dimension, columns))
    first = exhaustive_two_rectangle_search(
        dictionary,
        intrinsic,
        null,
        block_size=3,
        tie_tolerance=1.0e-12,
    )
    second = exhaustive_two_rectangle_search(
        dictionary,
        intrinsic,
        null,
        block_size=11,
        tie_tolerance=1.0e-12,
    )
    np.testing.assert_allclose(
        first.minimum_eigenvalue,
        second.minimum_eigenvalue,
        rtol=0.0,
        atol=1.0e-14,
    )
    assert {(candidate.first_index, candidate.second_index) for candidate in first.candidates} == {
        (candidate.first_index, candidate.second_index) for candidate in second.candidates
    }


def test_exhaustive_search_routes_nearly_singular_pair_separately() -> None:
    dictionary = np.asarray(
        [
            [1.0, 1.0, 0.0],
            [0.0, 1.0e-4, 1.0],
            [0.0, 0.0, 0.5],
        ]
    )
    null = np.eye(3)
    intrinsic = np.asarray(
        [
            [0.2, -0.1, 0.0],
            [-0.1, 0.5, 0.1],
            [0.0, 0.1, -0.3],
        ]
    )
    result = exhaustive_two_rectangle_search(
        dictionary,
        intrinsic,
        null,
        block_size=2,
        tie_tolerance=1.0e-12,
        singular_threshold=1.0e-6,
    )
    assert result.near_singular_pair_count == 1
    assert result.nonpositive_gram_pair_count == 0
    assert result.ordinary_pair_count == 2
    assert result.pair_count == 3


def test_geometry_key_is_invariant_to_d4_reflection_and_pair_exchange() -> None:
    first = {"delta_lon_grid": 4, "delta_lat_grid": 4, "temporal_lag": 1}
    second = {"delta_lon_grid": 4, "delta_lat_grid": 2, "temporal_lag": 1}
    reflected_first = {"delta_lon_grid": -4, "delta_lat_grid": 4, "temporal_lag": 1}
    reflected_second = {"delta_lon_grid": -4, "delta_lat_grid": 2, "temporal_lag": 1}
    direct = _canonical_equivalence_key(
        first,
        second,
        delta_lon_x2=2,
        delta_lat_x2=4,
        delta_time_x2=2,
        orientation_dot=24,
        orientation_cross_magnitude=8,
    )
    reflected = _canonical_equivalence_key(
        reflected_first,
        reflected_second,
        delta_lon_x2=-2,
        delta_lat_x2=4,
        delta_time_x2=2,
        orientation_dot=24,
        orientation_cross_magnitude=8,
    )
    exchanged = _canonical_equivalence_key(
        second,
        first,
        delta_lon_x2=-2,
        delta_lat_x2=-4,
        delta_time_x2=-2,
        orientation_dot=24,
        orientation_cross_magnitude=8,
    )
    assert direct == reflected == exchanged


def test_geometry_key_retains_segment_orientation_relative_to_displacement() -> None:
    first = {"delta_lon_grid": 4, "delta_lat_grid": 0, "temporal_lag": 1}
    second = {"delta_lon_grid": 0, "delta_lat_grid": 2, "temporal_lag": 1}
    along_first = _canonical_equivalence_key(
        first,
        second,
        delta_lon_x2=2,
        delta_lat_x2=0,
        delta_time_x2=0,
        orientation_dot=0,
        orientation_cross_magnitude=8,
    )
    diagonal = _canonical_equivalence_key(
        first,
        second,
        delta_lon_x2=2,
        delta_lat_x2=2,
        delta_time_x2=0,
        orientation_dot=0,
        orientation_cross_magnitude=8,
    )
    assert along_first != diagonal
