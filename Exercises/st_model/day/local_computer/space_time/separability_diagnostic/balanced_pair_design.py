"""Balanced local anchor pairs and moving-rectangle contrasts.

This module is intentionally independent of the pilot runner.  It provides a
deterministic spatial design for a follow-up diagnostic: local pairs retain a
fixed grid offset, while pair centres are spread over the common-valid domain
by an anisotropically scaled max-min rule.

The flattened anchor order is pair-major::

    pair 0 endpoint 0, pair 0 endpoint 1,
    pair 1 endpoint 0, pair 1 endpoint 1, ...

When these anchors are observed in time-major order, the corresponding moving
rectangle for pair ``p`` and adjacent times ``t`` and ``t + 1`` is

``Y[p, 0, t] - Y[p, 1, t] - Y[p, 0, t + 1] + Y[p, 1, t + 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse


@dataclass(frozen=True)
class AnchorPairMetadata:
    """Metadata for a pair-major flattened anchor array.

    Attributes
    ----------
    endpoints
        Integer grid coordinates with shape ``(pair_count, 2, 2)``.  The
        second axis identifies the two endpoints and the last axis is
        ``(row, column)``.
    grid_centers
        Pair centres in fractional grid coordinates.
    physical_centers
        Pair centres in the latitude/longitude coordinate system supplied to
        :func:`select_balanced_anchor_pairs`.
    insertion_separation
        Distance from each newly inserted centre to the previously selected
        centres, after latitude/longitude scaling by the supplied ranges.  The
        first entry is ``NaN`` because no preceding centre exists.
    offset
        Directed grid offset from endpoint zero to endpoint one.
    range_scale
        Latitude and longitude scales used by the max-min distance.
    """

    endpoints: np.ndarray
    grid_centers: np.ndarray
    physical_centers: np.ndarray
    insertion_separation: np.ndarray
    offset: tuple[int, int]
    range_scale: tuple[float, float]

    @property
    def pair_count(self) -> int:
        return int(self.endpoints.shape[0])

    @property
    def pair_anchor_indices(self) -> np.ndarray:
        """Indices of each pair in the returned flattened anchor array."""

        return np.arange(2 * self.pair_count, dtype=np.int64).reshape(self.pair_count, 2)


def _validate_grid(latitude: np.ndarray, longitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    latitude = np.asarray(latitude, dtype=np.float64)
    longitude = np.asarray(longitude, dtype=np.float64)
    if latitude.ndim != 2 or latitude.shape != longitude.shape:
        raise ValueError("latitude and longitude must be two-dimensional arrays of equal shape")
    if latitude.size == 0 or not np.isfinite(latitude).all() or not np.isfinite(longitude).all():
        raise ValueError("latitude and longitude must be nonempty and finite")
    return latitude, longitude


def _canonical_candidates(candidates: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    candidates = np.asarray(candidates)
    if candidates.ndim != 2 or candidates.shape[1] != 2:
        raise ValueError("candidates must have shape (candidate_count, 2)")
    if not np.issubdtype(candidates.dtype, np.integer):
        if not np.isfinite(candidates).all() or not np.equal(candidates, np.rint(candidates)).all():
            raise ValueError("candidate grid coordinates must be integers")
    candidates = np.asarray(candidates, dtype=np.int64)
    if len(candidates) == 0:
        raise ValueError("at least one common-valid candidate is required")
    n_rows, n_columns = grid_shape
    inside = (
        (candidates[:, 0] >= 0)
        & (candidates[:, 0] < n_rows)
        & (candidates[:, 1] >= 0)
        & (candidates[:, 1] < n_columns)
    )
    if not inside.all():
        raise ValueError("candidate grid coordinates must lie inside the supplied grid")
    order = np.lexsort((candidates[:, 1], candidates[:, 0]))
    candidates = np.ascontiguousarray(candidates[order])
    if len(candidates) > 1 and np.any(np.all(candidates[1:] == candidates[:-1], axis=1)):
        raise ValueError("candidate grid coordinates must be unique")
    return candidates


def select_balanced_anchor_pairs(
    candidates: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    *,
    pair_count: int = 50,
    offset: tuple[int, int] = (2, 2),
    range_lat: float,
    range_lon: float,
) -> tuple[np.ndarray, AnchorPairMetadata]:
    """Select non-overlapping local pairs with max-min distributed centres.

    Both endpoints must occur in ``candidates``; callers can therefore pass
    the common-valid anchor candidates from a moving flow tube directly.  A
    pair consists of ``anchor`` and ``anchor + offset``.  Feasible pairs are
    sorted lexicographically before selection, so the result does not depend
    on the input ordering.

    The first pair is the one whose scaled physical centre is closest to the
    centroid of all feasible pair centres.  Each later pair maximizes its
    minimum scaled distance to selected centres, subject to never reusing an
    endpoint.  Lexicographic candidate order breaks exact ties.
    """

    latitude, longitude = _validate_grid(latitude, longitude)
    candidates = _canonical_candidates(candidates, latitude.shape)
    if isinstance(pair_count, bool) or not isinstance(pair_count, (int, np.integer)):
        raise ValueError("pair_count must be an integer")
    pair_count = int(pair_count)
    if pair_count < 1:
        raise ValueError("pair_count must be positive")
    if len(offset) != 2 or any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in offset
    ):
        raise ValueError("offset must contain two integers")
    offset = (int(offset[0]), int(offset[1]))
    if offset == (0, 0):
        raise ValueError("offset must connect two distinct grid cells")
    ranges = np.asarray([range_lat, range_lon], dtype=np.float64)
    if not np.isfinite(ranges).all() or np.any(ranges <= 0.0):
        raise ValueError("range_lat and range_lon must be finite and positive")

    candidate_set = {tuple(cell) for cell in candidates.tolist()}
    directed_offset = np.asarray(offset, dtype=np.int64)
    pair_list = [
        (tuple(anchor), tuple(anchor + directed_offset))
        for anchor in candidates
        if tuple(anchor + directed_offset) in candidate_set
    ]
    if not pair_list:
        raise ValueError("no candidate pair has both endpoints at the requested offset")
    feasible = np.asarray(pair_list, dtype=np.int64)

    endpoint_zero = feasible[:, 0]
    endpoint_one = feasible[:, 1]
    physical_zero = np.column_stack(
        [
            latitude[endpoint_zero[:, 0], endpoint_zero[:, 1]],
            longitude[endpoint_zero[:, 0], endpoint_zero[:, 1]],
        ]
    )
    physical_one = np.column_stack(
        [
            latitude[endpoint_one[:, 0], endpoint_one[:, 1]],
            longitude[endpoint_one[:, 0], endpoint_one[:, 1]],
        ]
    )
    physical_centers = 0.5 * (physical_zero + physical_one)
    scaled_centers = physical_centers / ranges
    domain_center = scaled_centers.mean(axis=0)

    first = int(np.argmin(np.sum((scaled_centers - domain_center) ** 2, axis=1)))
    selected = [first]
    used_endpoints = {tuple(feasible[first, 0]), tuple(feasible[first, 1])}
    minimum_distance_sq = np.sum((scaled_centers - scaled_centers[first]) ** 2, axis=1)
    insertion_separation = [np.nan]

    while len(selected) < pair_count:
        available = np.asarray(
            [
                tuple(pair[0]) not in used_endpoints and tuple(pair[1]) not in used_endpoints
                for pair in feasible
            ],
            dtype=bool,
        )
        if not available.any():
            raise ValueError(
                f"only {len(selected)} non-overlapping pairs can be selected "
                f"at offset {offset}; requested {pair_count}"
            )
        available_distance = np.where(available, minimum_distance_sq, -np.inf)
        next_index = int(np.argmax(available_distance))
        selected.append(next_index)
        insertion_separation.append(float(np.sqrt(available_distance[next_index])))
        used_endpoints.update((tuple(feasible[next_index, 0]), tuple(feasible[next_index, 1])))
        distance_sq = np.sum((scaled_centers - scaled_centers[next_index]) ** 2, axis=1)
        minimum_distance_sq = np.minimum(minimum_distance_sq, distance_sq)

    chosen = np.asarray(selected, dtype=np.int64)
    endpoints = np.ascontiguousarray(feasible[chosen])
    metadata = AnchorPairMetadata(
        endpoints=endpoints,
        grid_centers=endpoints.mean(axis=1),
        physical_centers=np.ascontiguousarray(physical_centers[chosen]),
        insertion_separation=np.asarray(insertion_separation, dtype=np.float64),
        offset=offset,
        range_scale=(float(range_lat), float(range_lon)),
    )
    return endpoints.reshape(-1, 2), metadata


def build_moving_rectangle_contrast_matrix(
    pair_count: int,
    *,
    time_count: int = 8,
    dtype: np.dtype | type = np.float64,
) -> scipy.sparse.csr_matrix:
    """Build adjacent-time local-pair contrasts for time-major observations.

    Columns are ordered by time, with the pair-major flattened anchors within
    every time block.  Rows are ordered by adjacent time interval and then by
    pair.  Thus row ``t * pair_count + p`` contains ``[1, -1, -1, 1]`` at
    ``(p0,t), (p1,t), (p0,t+1), (p1,t+1)`` respectively.
    """

    for value, name in ((pair_count, "pair_count"), (time_count, "time_count")):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
    pair_count = int(pair_count)
    time_count = int(time_count)
    if pair_count < 1:
        raise ValueError("pair_count must be positive")
    if time_count < 2:
        raise ValueError("time_count must be at least two")

    anchor_count = 2 * pair_count
    interval = np.repeat(np.arange(time_count - 1, dtype=np.int64), pair_count)
    pair = np.tile(np.arange(pair_count, dtype=np.int64), time_count - 1)
    row = np.repeat(np.arange(pair_count * (time_count - 1), dtype=np.int64), 4)
    first = 2 * pair
    columns = np.column_stack(
        [
            interval * anchor_count + first,
            interval * anchor_count + first + 1,
            (interval + 1) * anchor_count + first,
            (interval + 1) * anchor_count + first + 1,
        ]
    ).reshape(-1)
    coefficients = np.tile(np.asarray([1.0, -1.0, -1.0, 1.0], dtype=dtype), len(interval))
    shape = (pair_count * (time_count - 1), anchor_count * time_count)
    return scipy.sparse.csr_matrix((coefficients, (row, columns)), shape=shape, dtype=dtype)
