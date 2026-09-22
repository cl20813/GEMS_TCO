"""Spatial ordering and predecessor-neighbor construction.

The public names describe the mathematical operation rather than the backend.
Max-min ordering uses the bundled pybind11 implementation. Predecessor
neighbors use a SciPy k-d tree and explicitly enforce that every returned
neighbor precedes its target. Equal-distance neighbors are ordered by index so
regular-grid results are reproducible across platforms.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
from scipy.spatial import cKDTree

from ._maxmin import maxmin_order as _native_maxmin_order

__all__ = ["maxmin_order", "predecessor_neighbors"]


def _locations_array(locations, *, allow_empty: bool) -> np.ndarray:
    """Return a finite, contiguous two-dimensional float64 location array."""

    try:
        array = np.asarray(locations, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("locations must be convertible to a numeric NumPy array") from exc

    if array.ndim != 2:
        raise ValueError(f"locations must be two-dimensional, got shape {array.shape}")
    if array.shape[1] == 0:
        raise ValueError("locations must have at least one coordinate column")
    if not allow_empty and array.shape[0] == 0:
        raise ValueError("locations must contain at least one row")
    if not np.isfinite(array).all():
        raise ValueError("locations must contain only finite values")
    return np.ascontiguousarray(array)


def maxmin_order(locations) -> np.ndarray:
    """Return a zero-based max-min permutation of ``locations``.

    The first location is the observation closest to the coordinate-wise mean;
    each subsequent location maximizes its distance to the selected set.
    """

    array = _locations_array(locations, allow_empty=False)
    normalized = _normalize_euclidean_coordinates(array)
    order = np.asarray(_native_maxmin_order(normalized), dtype=np.int64)
    expected = np.arange(array.shape[0], dtype=np.int64)
    if order.shape != expected.shape or not np.array_equal(np.sort(order), expected):
        raise RuntimeError("the native max-min backend returned an invalid permutation")
    return order


def predecessor_neighbors(locations, max_neighbors: int = 10) -> np.ndarray:
    """Return nearest neighbors that precede each ordered location.

    Parameters
    ----------
    locations:
        Ordered location matrix with shape ``(n_locations, n_dimensions)``.
    max_neighbors:
        Maximum number of predecessor neighbors per row. Missing entries are
        padded with ``-1``.

    Returns
    -------
    numpy.ndarray
        Integer array of shape ``(n_locations, max_neighbors)``. Every
        nonnegative entry in row ``i`` is strictly less than ``i``.
    """

    if not isinstance(max_neighbors, Integral) or isinstance(max_neighbors, (bool, np.bool_)):
        raise TypeError("max_neighbors must be an integer")
    if max_neighbors < 0:
        raise ValueError("max_neighbors must be nonnegative")

    array = _locations_array(locations, allow_empty=True)
    n_locations = array.shape[0]
    neighbors = np.full((n_locations, int(max_neighbors)), -1, dtype=np.int64)
    if n_locations == 0 or max_neighbors == 0:
        return neighbors

    # Translation and one global positive scale preserve Euclidean rankings
    # while improving numerical resolution for coordinates with a large
    # common offset.
    normalized = _normalize_euclidean_coordinates(array)
    tree = cKDTree(normalized)

    requested = np.minimum(np.arange(n_locations), int(max_neighbors))
    unresolved = np.flatnonzero(requested > 0)
    search_width = min(n_locations, max(2 * int(max_neighbors) + 1, 2))
    while unresolved.size:
        distances, candidates = tree.query(normalized[unresolved], k=search_width, workers=1)
        distances = np.asarray(distances, dtype=np.float64)
        candidates = np.asarray(candidates, dtype=np.int64)
        if candidates.ndim == 1:
            candidates = candidates[:, None]
            distances = distances[:, None]
        still_unresolved: list[int] = []
        for result_row, target in enumerate(unresolved):
            count = int(requested[target])
            candidate_row = candidates[result_row]
            distance_row = distances[result_row]
            predecessor_mask = (candidate_row >= 0) & (candidate_row < target)
            preceding = candidate_row[predecessor_mask]
            if preceding.size >= count:
                # The k-d tree does not specify an index tie-break and a
                # truncated query can split a distance tie. Query the entire
                # boundary ball, then sort exact float64 distances followed by
                # index for a deterministic result.
                cutoff = np.partition(distance_row[predecessor_mask], count - 1)[count - 1]
                radius = np.nextafter(float(cutoff), np.inf)
                tied_candidates = np.asarray(
                    tree.query_ball_point(normalized[target], r=radius, workers=1),
                    dtype=np.int64,
                )
                tied_candidates = tied_candidates[
                    (tied_candidates >= 0) & (tied_candidates < target)
                ]
                deltas = normalized[tied_candidates] - normalized[target]
                exact_squared_distances = np.einsum("ij,ij->i", deltas, deltas)
                stable_order = np.lexsort((tied_candidates, exact_squared_distances))
                neighbors[target, :count] = tied_candidates[stable_order[:count]]
            elif search_width == n_locations:
                raise RuntimeError("neighbor search failed to return the required predecessors")
            else:
                still_unresolved.append(int(target))
        unresolved = np.asarray(still_unresolved, dtype=np.int64)
        search_width = min(n_locations, 2 * search_width)

    return neighbors


def _normalize_euclidean_coordinates(array: np.ndarray) -> np.ndarray:
    """Translate and uniformly scale coordinates without changing rankings."""

    shifted = array - array[0]
    if not np.isfinite(shifted).all():
        raise ValueError("the coordinate span is too large for stable distance calculation")
    scale = float(np.max(np.abs(shifted)))
    normalized = shifted if scale == 0.0 else shifted / scale
    return np.ascontiguousarray(normalized, dtype=np.float64)
