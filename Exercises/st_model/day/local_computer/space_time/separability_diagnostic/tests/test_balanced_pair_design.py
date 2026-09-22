from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
MODULE_DIR = HERE.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from balanced_pair_design import (  # noqa: E402
    build_moving_rectangle_contrast_matrix,
    select_balanced_anchor_pairs,
)


def regular_grid(rows: int = 14, columns: int = 16) -> tuple[np.ndarray, np.ndarray]:
    row, column = np.meshgrid(np.arange(rows), np.arange(columns), indexing="ij")
    return 0.05 * row, 0.08 * column


def all_candidates(shape: tuple[int, int]) -> np.ndarray:
    return np.asarray(list(np.ndindex(shape)), dtype=np.int64)


def test_selection_is_deterministic_under_candidate_permutation() -> None:
    latitude, longitude = regular_grid()
    candidates = all_candidates(latitude.shape)
    shuffled = candidates[np.random.default_rng(1907).permutation(len(candidates))]
    expected_anchors, expected_metadata = select_balanced_anchor_pairs(
        candidates,
        latitude,
        longitude,
        range_lat=0.2,
        range_lon=0.3,
    )
    actual_anchors, actual_metadata = select_balanced_anchor_pairs(
        shuffled,
        latitude,
        longitude,
        range_lat=0.2,
        range_lon=0.3,
    )
    assert expected_anchors.shape == (100, 2)
    assert expected_metadata.pair_count == 50
    np.testing.assert_array_equal(actual_anchors, expected_anchors)
    np.testing.assert_array_equal(actual_metadata.endpoints, expected_metadata.endpoints)
    np.testing.assert_allclose(
        actual_metadata.insertion_separation,
        expected_metadata.insertion_separation,
        equal_nan=True,
    )


def test_selected_pairs_are_valid_local_pairs_without_endpoint_reuse() -> None:
    latitude, longitude = regular_grid()
    candidates = all_candidates(latitude.shape)
    anchors, metadata = select_balanced_anchor_pairs(
        candidates,
        latitude,
        longitude,
        pair_count=25,
        offset=(2, 2),
        range_lat=0.2,
        range_lon=0.3,
    )
    candidate_set = {tuple(candidate) for candidate in candidates}
    assert anchors.shape == (50, 2)
    assert metadata.endpoints.shape == (25, 2, 2)
    assert all(tuple(anchor) in candidate_set for anchor in anchors)
    np.testing.assert_array_equal(
        metadata.endpoints[:, 1] - metadata.endpoints[:, 0],
        np.tile([2, 2], (metadata.pair_count, 1)),
    )
    assert len({tuple(anchor) for anchor in anchors}) == len(anchors)
    np.testing.assert_array_equal(metadata.pair_anchor_indices, np.arange(50).reshape(25, 2))

    expected_physical_centers = np.column_stack(
        [
            latitude[anchors[:, 0], anchors[:, 1]].reshape(-1, 2).mean(axis=1),
            longitude[anchors[:, 0], anchors[:, 1]].reshape(-1, 2).mean(axis=1),
        ]
    )
    np.testing.assert_allclose(metadata.physical_centers, expected_physical_centers)


def test_contrast_dimensions_and_full_row_rank() -> None:
    contrast = build_moving_rectangle_contrast_matrix(50)
    assert contrast.shape == (50 * 7, 100 * 8)
    assert contrast.nnz == 4 * 50 * 7
    assert np.linalg.matrix_rank(contrast.toarray()) == contrast.shape[0]


def test_contrast_has_exact_time_major_coefficient_locations() -> None:
    pair_count = 3
    contrast = build_moving_rectangle_contrast_matrix(pair_count, time_count=4)
    anchor_count = 2 * pair_count
    for time_index in range(3):
        for pair_index in range(pair_count):
            row_index = time_index * pair_count + pair_index
            row = contrast.getrow(row_index)
            expected_columns = np.asarray(
                [
                    time_index * anchor_count + 2 * pair_index,
                    time_index * anchor_count + 2 * pair_index + 1,
                    (time_index + 1) * anchor_count + 2 * pair_index,
                    (time_index + 1) * anchor_count + 2 * pair_index + 1,
                ]
            )
            np.testing.assert_array_equal(row.indices, expected_columns)
            np.testing.assert_array_equal(row.data, [1.0, -1.0, -1.0, 1.0])
