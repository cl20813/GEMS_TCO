#!/usr/bin/env python3
"""Build an oracle eigen-direction atlas for the nugget-zero pilot.

The completed pilot supplies a joint space-time Matern covariance ``Sigma1``
and its KL-optimal advected-separable approximation ``Sigma0``.  This script
reconstructs every generalized eigen-direction without using a response and
then separates two questions that are otherwise easy to conflate:

1. Which directions best distinguish ``Sigma1`` from the fitted ``Sigma0``?
2. Which of those differences are intrinsic space-time interaction rather
   than compensation caused by refitting the separable ranges and advection?

For the second question, ``SigmaM`` is the advected-separable covariance with
the *truth* variance, ranges, and advection.  It has the same exponential
spatial and temporal margins as ``Sigma1``.  For every direction normalized
so that ``w' Sigma0 w = 1``, the variance discrepancy has the exact additive
decomposition

    lambda - 1
      = w' (Sigma1 - SigmaM) w + w' (SigmaM - Sigma0) w.

The first term is labelled intrinsic interaction and the second fitted-null
compensation.  This label is available because the simulation truth is known;
it is not directly available for real data.

The atlas is exploratory.  Candidate flags are covariance-only annotations,
not a finalized test or a data-dependent choice of a held-out p-value.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import scipy.fft
import scipy.linalg
from scipy.spatial import cKDTree

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

from analyze_mode_count_path import (
    atomic_csv,
    atomic_json,
    covariance_parameters,
    sha256,
)
from diagnostic_core import (
    CovarianceParameters,
    advected_separable_covariance,
    g_score,
    joint_matern_half_covariance,
    pairwise_lags,
    solve_generalized_eigenproblem,
    standardized_moving_lag_norms,
)


HERE = Path(__file__).resolve().parent
DEFAULT_PILOT = HERE / "outputs/nugget0_five_day_092226"
DEFAULT_SPATIAL_BINS = (0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, math.inf)


@dataclass(frozen=True)
class CoordinateDay:
    date: str
    coordinates: np.ndarray
    ordering: tuple[tuple[int, int], ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--cluster-relative-log-gap",
        type=float,
        default=0.01,
        help="relative |log(lambda)| cluster span used only to flag rotatable mode clusters",
    )
    parser.add_argument(
        "--candidate-relative-g-floor",
        type=float,
        default=0.5,
        help="minimum branch-relative total g for interaction-dominant exemplars",
    )
    parser.add_argument("--graph-neighbors", type=int, nargs="+", default=(4, 6, 8))
    return parser


def load_coordinate_days(point_path: Path, dates: Iterable[str]) -> list[CoordinateDay]:
    """Read coordinate columns only; response columns never enter this process."""

    columns = (
        "date",
        "time_index",
        "anchor_rank",
        "source_latitude",
        "source_longitude",
    )
    frame = pd.read_csv(point_path, usecols=list(columns))
    frame["date"] = frame["date"].astype(str)
    requested_dates = tuple(str(date) for date in dates)
    unexpected = sorted(set(frame["date"]).difference(requested_dates))
    if unexpected:
        raise ValueError(f"selected point table contains unexpected dates: {unexpected}")
    result: list[CoordinateDay] = []
    reference_ordering: tuple[tuple[int, int], ...] | None = None
    for date in requested_dates:
        day = frame.loc[frame["date"] == date].copy()
        if day.empty:
            raise ValueError(f"selected point table has no rows for {date}")
        if day.duplicated(["time_index", "anchor_rank"]).any():
            raise ValueError(f"duplicate time/anchor positions found for {date}")
        day.sort_values(["time_index", "anchor_rank"], inplace=True)
        values = day[["source_latitude", "source_longitude", "time_index"]].to_numpy(
            dtype=np.float64
        )
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite coordinate found for {date}")
        ordering = tuple(
            (int(time_index), int(anchor_rank))
            for time_index, anchor_rank in day[["time_index", "anchor_rank"]].itertuples(
                index=False, name=None
            )
        )
        if reference_ordering is None:
            reference_ordering = ordering
        elif ordering != reference_ordering:
            raise ValueError("time/anchor ordering differs across independent days")
        result.append(
            CoordinateDay(
                date=date,
                coordinates=np.ascontiguousarray(values),
                ordering=ordering,
            )
        )
    return result


def quadratic_diagonal(matrix: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Return ``diag(W' matrix W)`` without forming the full projection."""

    matrix = np.asarray(matrix, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if directions.ndim != 2 or directions.shape[0] != matrix.shape[0]:
        raise ValueError("directions and matrix dimensions do not align")
    return np.sum(directions * (matrix @ directions), axis=0)


def normalize_directions(
    directions: np.ndarray,
    reference_covariance: np.ndarray,
) -> np.ndarray:
    """Scale every direction to unit variance under ``reference_covariance``."""

    variances = quadratic_diagonal(reference_covariance, directions)
    if not np.isfinite(variances).all() or np.any(variances <= 0.0):
        raise scipy.linalg.LinAlgError("a direction has nonpositive reference variance")
    return np.asarray(directions, dtype=np.float64) / np.sqrt(variances)[None, :]


def whiten_covariance_difference(
    difference: np.ndarray,
    reference_covariance: np.ndarray,
) -> np.ndarray:
    """Return ``L^-1 difference L^-T`` for ``reference_covariance = L L'``."""

    factor = scipy.linalg.cholesky(
        np.asarray(reference_covariance, dtype=np.float64),
        lower=True,
        check_finite=False,
    )
    left = scipy.linalg.solve_triangular(
        factor,
        np.asarray(difference, dtype=np.float64),
        lower=True,
        check_finite=False,
    )
    whitened = scipy.linalg.solve_triangular(
        factor,
        left.T,
        lower=True,
        check_finite=False,
    ).T
    return (whitened + whitened.T) * 0.5


def effective_count(energy: np.ndarray) -> float:
    energy = np.asarray(energy, dtype=np.float64)
    total = float(energy.sum())
    if total <= 0.0:
        return 0.0
    probability = energy / total
    return float(1.0 / np.square(probability).sum())


def entropy_effective_rank(singular_values: np.ndarray) -> float:
    energy = np.square(np.asarray(singular_values, dtype=np.float64))
    total = float(energy.sum())
    if total <= 0.0:
        return 0.0
    probability = energy / total
    positive = probability > 0.0
    return float(np.exp(-np.sum(probability[positive] * np.log(probability[positive]))))


def knn_edges(coordinates: np.ndarray, neighbors: int) -> np.ndarray:
    """Return unique undirected k-nearest-neighbor graph edges."""

    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (anchors, 2)")
    if not 1 <= int(neighbors) < len(coordinates):
        raise ValueError("neighbors must lie between one and anchors minus one")
    indices = cKDTree(coordinates).query(coordinates, k=int(neighbors) + 1)[1][:, 1:]
    pairs = {
        (min(source, int(target)), max(source, int(target)))
        for source, targets in enumerate(indices)
        for target in targets
        if source != int(target)
    }
    return np.asarray(sorted(pairs), dtype=np.int64)


def graph_roughness(weights: np.ndarray, edges: np.ndarray) -> float:
    """Mean squared edge difference divided by mean squared weight."""

    weights = np.asarray(weights, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    denominator = float(np.mean(np.square(weights)))
    if denominator <= 0.0:
        return 0.0
    differences = weights[:, edges[:, 0]] - weights[:, edges[:, 1]]
    return float(np.mean(np.square(differences)) / denominator)


def weighted_knn_adjacency(coordinates: np.ndarray, *, neighbors: int) -> np.ndarray:
    """Construct the symmetric Gaussian-weighted kNN adjacency matrix."""

    coordinates = np.asarray(coordinates, dtype=np.float64)
    edges = knn_edges(coordinates, neighbors)
    differences = coordinates[edges[:, 0]] - coordinates[edges[:, 1]]
    distances = np.linalg.norm(differences, axis=1)
    positive = distances[distances > 0.0]
    if not len(positive):
        raise ValueError("anchor coordinates do not contain distinct locations")
    scale = float(np.median(positive))
    weights = np.exp(-0.5 * np.square(distances / scale))
    adjacency = np.zeros((len(coordinates), len(coordinates)), dtype=np.float64)
    adjacency[edges[:, 0], edges[:, 1]] = weights
    adjacency[edges[:, 1], edges[:, 0]] = weights
    return adjacency


def graph_fourier_basis(
    coordinates: np.ndarray,
    *,
    neighbors: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized-Laplacian graph frequencies and Euclidean basis."""

    adjacency = weighted_knn_adjacency(coordinates, neighbors=neighbors)
    degree = adjacency.sum(axis=1)
    if np.any(degree <= 0.0):
        raise ValueError("kNN graph contains an isolated anchor")
    inverse_sqrt_degree = 1.0 / np.sqrt(degree)
    normalized_laplacian = np.eye(len(coordinates)) - (
        inverse_sqrt_degree[:, None] * adjacency * inverse_sqrt_degree[None, :]
    )
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        normalized_laplacian,
        check_finite=False,
    )
    if len(eigenvalues) < 2 or eigenvalues[1] <= 1.0e-10:
        raise ValueError("kNN graph must be connected for Fiedler ordering")
    return eigenvalues, eigenvectors


def fiedler_anchor_order(coordinates: np.ndarray, *, neighbors: int = 6) -> np.ndarray:
    """Order irregular anchors along a weighted kNN-graph Fiedler coordinate."""

    coordinates = np.asarray(coordinates, dtype=np.float64)
    _, eigenvectors = graph_fourier_basis(coordinates, neighbors=neighbors)
    fiedler = eigenvectors[:, 1]
    longitude_correlation = float(
        np.dot(fiedler - fiedler.mean(), coordinates[:, 1] - coordinates[:, 1].mean())
    )
    if longitude_correlation < 0.0:
        fiedler = -fiedler
    elif longitude_correlation == 0.0:
        largest = int(np.argmax(np.abs(fiedler)))
        if fiedler[largest] < 0.0:
            fiedler = -fiedler
    return np.argsort(fiedler, kind="stable")


def normalized_gram(gram: np.ndarray) -> np.ndarray:
    """Convert a positive-semidefinite Gram matrix to a similarity matrix."""

    gram = np.asarray(gram, dtype=np.float64)
    scale = np.sqrt(np.maximum(np.diag(gram), 0.0))
    denominator = np.outer(scale, scale)
    return np.divide(
        gram,
        denominator,
        out=np.zeros_like(gram),
        where=denominator > 0.0,
    )


def cluster_subspace_heatmap_arrays(
    directions: np.ndarray,
    *,
    time_count: int,
    anchor_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return raw weights and rotation-invariant cluster heatmap summaries."""

    directions = np.asarray(directions, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[0] != time_count * anchor_count:
        raise ValueError("cluster directions do not match the space-time dimensions")
    weights = directions.reshape(time_count, anchor_count, directions.shape[1])
    pointwise_energy = np.sqrt(np.square(weights).sum(axis=2))
    temporal_gram = np.einsum("tam,uam->tu", weights, weights, optimize=True)
    spatial_gram = np.einsum("tam,tbm->ab", weights, weights, optimize=True)
    return (
        weights,
        pointwise_energy,
        normalized_gram(temporal_gram),
        normalized_gram(spatial_gram),
    )


def cluster_space_time_spectral_energy(
    weights: np.ndarray,
    graph_eigenvectors: np.ndarray,
) -> np.ndarray:
    """Return cluster-rotation-invariant Euclidean filter-weight energy."""

    weights = np.asarray(weights, dtype=np.float64)
    graph_eigenvectors = np.asarray(graph_eigenvectors, dtype=np.float64)
    if weights.ndim != 3:
        raise ValueError("weights must have shape (time, anchors, modes)")
    if graph_eigenvectors.shape != (weights.shape[1], weights.shape[1]):
        raise ValueError("graph eigenvectors do not match the anchor dimension")
    temporal_coefficients = scipy.fft.dct(weights, type=2, axis=0, norm="ortho")
    joint_coefficients = np.einsum(
        "tam,ak->tkm",
        temporal_coefficients,
        graph_eigenvectors,
        optimize=True,
    )
    energy = np.square(joint_coefficients).sum(axis=2)
    total = float(energy.sum())
    if total <= 0.0:
        raise ValueError("cluster weights have zero energy")
    return energy / total


def leading_temporal_factor(
    weight_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return sign-fixed leading factors, rank-one energy, and effective rank."""

    left, singular_values, right_transpose = np.linalg.svd(
        np.asarray(weight_matrix, dtype=np.float64),
        full_matrices=False,
    )
    temporal = left[:, 0]
    spatial = right_transpose[0]
    largest = int(np.argmax(np.abs(temporal)))
    if temporal[largest] < 0.0:
        temporal = -temporal
        spatial = -spatial
    energy = np.square(singular_values)
    return (
        temporal,
        spatial,
        float(energy[0] / energy.sum()),
        entropy_effective_rank(singular_values),
    )


def pattern_metrics(
    directions: np.ndarray,
    *,
    time_count: int,
    anchor_count: int,
    graph_edges: dict[int, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize spatial localization and temporal structure of every mode."""

    directions = np.asarray(directions, dtype=np.float64)
    if directions.shape[0] != time_count * anchor_count:
        raise ValueError("direction dimension does not equal time_count * anchor_count")
    rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    for index in range(directions.shape[1]):
        weights = directions[:, index].reshape(time_count, anchor_count)
        temporal, _, rank1, effective_rank = leading_temporal_factor(weights)
        total_energy = float(np.square(weights).sum())
        time_energy = np.square(weights).sum(axis=1)
        anchor_energy = np.square(weights).sum(axis=0)
        dct_energy = np.square(scipy.fft.dct(weights, type=2, axis=0, norm="ortho")).sum(axis=1)
        dct_energy /= dct_energy.sum()
        adjacent_cosines = []
        for time_index in range(time_count - 1):
            first = weights[time_index]
            second = weights[time_index + 1]
            denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
            if denominator > 0.0:
                adjacent_cosines.append(float(first @ second / denominator))
        row: dict[str, Any] = {
            "mode": index + 1,
            "rank1_energy_fraction": rank1,
            "space_time_effective_rank": effective_rank,
            "temporal_first_difference_roughness": float(
                np.square(np.diff(weights, axis=0)).sum() / total_energy
            ),
            "mean_adjacent_spatial_map_cosine": float(np.mean(adjacent_cosines)),
            "effective_time_points": effective_count(time_energy),
            "effective_spatial_anchors": effective_count(anchor_energy),
            "dct_dc_fraction": float(dct_energy[0]),
            "dct_low_frequency_fraction": float(dct_energy[1:3].sum()),
            "dct_mid_frequency_fraction": float(dct_energy[3:4].sum()),
            "dct_high_frequency_fraction": float(dct_energy[4:].sum()),
        }
        for neighbors, edges in graph_edges.items():
            row[f"spatial_graph_roughness_k{neighbors}"] = graph_roughness(weights, edges)
        rows.append(row)
        temporal_rows.extend(
            {
                "mode": index + 1,
                "time_index": time_index,
                "leading_temporal_factor": float(value),
                "time_energy_fraction": float(time_energy[time_index] / total_energy),
            }
            for time_index, value in enumerate(temporal)
        )
    return pd.DataFrame(rows), pd.DataFrame(temporal_rows)


def assign_near_degenerate_clusters(
    eigenvalues: np.ndarray,
    *,
    relative_log_gap: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster nearby eigenvalues separately above and below one.

    Every cluster spans at most ``relative_log_gap`` of its largest
    ``|log(lambda)|`` signal.  It is a reproducible warning that individual
    vectors may rotate; it is not a statistical multiplicity test.
    """

    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    if relative_log_gap <= 0.0:
        raise ValueError("relative_log_gap must be positive")
    log_magnitude = np.abs(np.log(eigenvalues))
    cluster_ids = np.empty(len(eigenvalues), dtype=object)
    cluster_sizes = np.zeros(len(eigenvalues), dtype=np.int64)
    for branch, mask in (("gt", eigenvalues >= 1.0), ("lt", eigenvalues < 1.0)):
        indices = np.flatnonzero(mask)
        indices = indices[np.argsort(log_magnitude[indices], kind="stable")[::-1]]
        groups: list[list[int]] = []
        for index in indices:
            if not groups:
                groups.append([int(index)])
                continue
            cluster_reference = groups[-1][0]
            scale = max(
                log_magnitude[cluster_reference],
                log_magnitude[index],
                1.0e-15,
            )
            gap = abs(log_magnitude[cluster_reference] - log_magnitude[index]) / scale
            if gap <= relative_log_gap:
                groups[-1].append(int(index))
            else:
                groups.append([int(index)])
        for number, group in enumerate(groups, start=1):
            label = f"{branch}_{number:03d}"
            cluster_ids[group] = label
            cluster_sizes[group] = len(group)
    return cluster_ids, cluster_sizes


def pareto_mask(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Flag points not strictly dominated in both non-negative objectives."""

    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("objectives must be aligned one-dimensional arrays")
    result = np.zeros(len(first), dtype=bool)
    for index in range(len(first)):
        weakly_better = (first >= first[index]) & (second >= second[index])
        strictly_better = (first > first[index]) | (second > second[index])
        result[index] = not bool(np.any(weakly_better & strictly_better))
    return result


def direction_covariance_metrics(
    directions: np.ndarray,
    true_covariance: np.ndarray,
    fitted_null_covariance: np.ndarray,
    matched_margin_covariance: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Normalize directions and compute the exact interaction decomposition."""

    directions = normalize_directions(directions, fitted_null_covariance)
    true_variance = quadratic_diagonal(true_covariance, directions)
    matched_variance = quadratic_diagonal(matched_margin_covariance, directions)
    fitted_variance = quadratic_diagonal(fitted_null_covariance, directions)
    total_ratio = true_variance / fitted_variance
    intrinsic_ratio = true_variance / matched_variance
    delta_intrinsic = true_variance - matched_variance
    delta_compensation = matched_variance - fitted_variance
    delta_total = true_variance - fitted_variance
    scale = np.abs(delta_intrinsic) + np.abs(delta_compensation)
    interaction_share = np.divide(
        np.abs(delta_intrinsic),
        scale,
        out=np.zeros_like(scale),
        where=scale > 0.0,
    )
    magnitude_to_total = np.divide(
        scale,
        np.abs(delta_total),
        out=np.full_like(scale, math.inf),
        where=np.abs(delta_total) > 1.0e-14,
    )
    cancellation_fraction = 1.0 - np.divide(
        np.abs(delta_total),
        scale,
        out=np.ones_like(scale),
        where=scale > 0.0,
    )
    frame = pd.DataFrame(
        {
            "mode": np.arange(1, directions.shape[1] + 1),
            "total_variance_ratio": total_ratio,
            "total_log_variance_ratio": np.log(total_ratio),
            "total_g_score": g_score(total_ratio),
            "matched_margin_variance_relative_to_fitted_null": matched_variance,
            "intrinsic_interaction_variance_ratio": intrinsic_ratio,
            "intrinsic_interaction_log_variance_ratio": np.log(intrinsic_ratio),
            "intrinsic_interaction_g_score": g_score(intrinsic_ratio),
            "delta_total": delta_total,
            "delta_intrinsic_interaction": delta_intrinsic,
            "delta_fitted_null_compensation": delta_compensation,
            "interaction_absolute_share": interaction_share,
            "component_magnitude_to_total_ratio": magnitude_to_total,
            "cancellation_fraction": cancellation_fraction,
            "decomposition_absolute_error": np.abs(
                delta_total - delta_intrinsic - delta_compensation
            ),
            "total_variance_branch": np.where(total_ratio >= 1.0, "true_gt_null", "true_lt_null"),
            "intrinsic_interaction_branch": np.where(
                intrinsic_ratio >= 1.0,
                "joint_gt_margin_matched",
                "joint_lt_margin_matched",
            ),
        }
    )
    return directions, frame


def reference_subspace_membership(
    directions: np.ndarray,
    day_true: np.ndarray,
    day_null: np.ndarray,
    *,
    cumulative_kl_fraction: float = 0.8,
) -> tuple[np.ndarray, int]:
    """Membership of fixed directions in a day's leading KL eigensubspace."""

    day_eigen = solve_generalized_eigenproblem(day_true, day_null)
    cutoff = int(
        np.searchsorted(
            np.cumsum(day_eigen.scores) / day_eigen.scores.sum(),
            cumulative_kl_fraction,
        )
        + 1
    )
    standardized = normalize_directions(directions, day_null)
    coordinates = day_eigen.eigenvectors[:, :cutoff].T @ day_null @ standardized
    return np.square(coordinates).sum(axis=0), cutoff


def day_stability_metrics(
    directions_by_basis: dict[str, np.ndarray],
    dates: Sequence[str],
    true_covariances: dict[str, np.ndarray],
    null_covariances: dict[str, np.ndarray],
    matched_covariances: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for date in dates:
        for basis, directions in directions_by_basis.items():
            null_variance = quadratic_diagonal(null_covariances[date], directions)
            true_variance = quadratic_diagonal(true_covariances[date], directions)
            matched_variance = quadratic_diagonal(matched_covariances[date], directions)
            total_ratio = true_variance / null_variance
            intrinsic_ratio = true_variance / matched_variance
            membership, cutoff = reference_subspace_membership(
                directions,
                true_covariances[date],
                null_covariances[date],
            )
            rows.extend(
                {
                    "basis": basis,
                    "mode": mode + 1,
                    "date": date,
                    "total_variance_ratio": float(total_ratio[mode]),
                    "intrinsic_interaction_variance_ratio": float(intrinsic_ratio[mode]),
                    "day_80pct_kl_subspace_dimension": cutoff,
                    "day_80pct_kl_subspace_membership": float(membership[mode]),
                }
                for mode in range(directions.shape[1])
            )
    long = pd.DataFrame(rows)
    summaries = []
    for (basis, mode), group in long.groupby(["basis", "mode"], sort=False):
        total_log = np.log(group["total_variance_ratio"].to_numpy())
        intrinsic_log = np.log(group["intrinsic_interaction_variance_ratio"].to_numpy())
        summaries.append(
            {
                "basis": basis,
                "mode": mode,
                "design_day_total_log_ratio_sd": float(np.std(total_log, ddof=0)),
                "design_day_intrinsic_log_ratio_sd": float(np.std(intrinsic_log, ddof=0)),
                "design_day_total_sign_consistency": float(
                    max(np.mean(total_log >= 0.0), np.mean(total_log < 0.0))
                ),
                "design_day_intrinsic_sign_consistency": float(
                    max(np.mean(intrinsic_log >= 0.0), np.mean(intrinsic_log < 0.0))
                ),
                "design_day_total_ratio_min": float(np.exp(total_log.min())),
                "design_day_total_ratio_max": float(np.exp(total_log.max())),
                "design_day_80pct_membership_min": float(
                    group["day_80pct_kl_subspace_membership"].min()
                ),
                "design_day_80pct_membership_mean": float(
                    group["day_80pct_kl_subspace_membership"].mean()
                ),
            }
        )
    return long, pd.DataFrame(summaries)


def select_exploration_candidates(
    metrics: pd.DataFrame,
    *,
    relative_g_floor: float,
) -> pd.DataFrame:
    """Choose transparent exemplars; these flags do not define a final test."""

    if not 0.0 < relative_g_floor <= 1.0:
        raise ValueError("relative_g_floor must lie in (0, 1]")
    roles: dict[tuple[str, int], list[str]] = defaultdict(list)
    fitted = metrics.loc[metrics["basis"] == "fitted_null"].copy()
    for branch, short in (("true_gt_null", "gt"), ("true_lt_null", "lt")):
        subset = fitted.loc[fitted["total_variance_branch"] == branch]
        top = subset.loc[subset["total_g_score"].idxmax()]
        roles[("fitted_null", int(top["mode"]))].append(f"top_total_{short}")
        floor = relative_g_floor * float(subset["total_g_score"].max())
        same_sign = np.sign(subset["delta_intrinsic_interaction"]) == np.sign(subset["delta_total"])
        eligible = subset.loc[
            (subset["total_g_score"] >= floor)
            & same_sign
            & (subset["design_day_total_sign_consistency"] == 1.0)
        ]
        dominant = eligible.sort_values(
            ["interaction_absolute_share", "total_g_score"],
            ascending=False,
            kind="stable",
        ).iloc[0]
        roles[("fitted_null", int(dominant["mode"]))].append(f"interaction_dominant_{short}")

    interaction = metrics.loc[metrics["basis"] == "margin_matched_interaction"].copy()
    for branch, short in (
        ("joint_gt_margin_matched", "gt"),
        ("joint_lt_margin_matched", "lt"),
    ):
        subset = interaction.loc[interaction["intrinsic_interaction_branch"] == branch]
        top = subset.loc[subset["intrinsic_interaction_g_score"].idxmax()]
        roles[("margin_matched_interaction", int(top["mode"]))].append(f"top_intrinsic_{short}")

    rows = []
    for (basis, mode), labels in roles.items():
        row = metrics.loc[(metrics["basis"] == basis) & (metrics["mode"] == mode)].iloc[0]
        values = row.to_dict()
        values["candidate_roles"] = ";".join(labels)
        rows.append(values)
    return pd.DataFrame(rows).sort_values(["basis", "mode"], kind="stable").reset_index(drop=True)


def moving_anchor_coordinates(
    days: Sequence[CoordinateDay],
    truth: CovarianceParameters,
    *,
    time_count: int,
    anchor_count: int,
) -> np.ndarray:
    coordinates = []
    for day in days:
        values = day.coordinates.reshape(time_count, anchor_count, 3)
        moving = np.empty((time_count, anchor_count, 2), dtype=np.float64)
        moving[:, :, 0] = values[:, :, 0] - truth.advec_lat * values[:, :, 2]
        moving[:, :, 1] = values[:, :, 1] - truth.advec_lon * values[:, :, 2]
        coordinates.append(moving)
    return np.mean(np.stack(coordinates), axis=(0, 1))


def lag_attribution(
    candidates: pd.DataFrame,
    directions_by_basis: dict[str, np.ndarray],
    design_dates: Sequence[str],
    geometries: dict[str, Any],
    true_covariances: dict[str, np.ndarray],
    null_covariances: dict[str, np.ndarray],
    matched_covariances: dict[str, np.ndarray],
    truth: CovarianceParameters,
    *,
    spatial_bins: Sequence[float],
) -> pd.DataFrame:
    """Attribute quadratic-form differences to spatial and temporal lag bins."""

    spatial_bins = np.asarray(spatial_bins, dtype=np.float64)
    if spatial_bins[0] != 0.0 or not np.isinf(spatial_bins[-1]):
        raise ValueError("spatial bins must start at zero and end at infinity")
    accumulator: dict[tuple[Any, ...], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    matrices = {
        "intrinsic_interaction": lambda date: (true_covariances[date] - matched_covariances[date]),
        "fitted_null_compensation": lambda date: (
            matched_covariances[date] - null_covariances[date]
        ),
        "total": lambda date: true_covariances[date] - null_covariances[date],
    }
    for candidate in candidates.itertuples(index=False):
        direction = directions_by_basis[candidate.basis][:, int(candidate.mode) - 1]
        outer = np.outer(direction, direction)
        for date in design_dates:
            spatial_norm, _ = standardized_moving_lag_norms(geometries[date], truth)
            time_lag = np.rint(np.abs(geometries[date].delta_time)).astype(np.int64)
            spatial_index = np.digitize(spatial_norm, spatial_bins[1:-1], right=False)
            for component, difference in matrices.items():
                contribution = outer * difference(date) / len(design_dates)
                for time_value in range(int(time_lag.max()) + 1):
                    time_mask = time_lag == time_value
                    for bin_index in range(len(spatial_bins) - 1):
                        mask = time_mask & (spatial_index == bin_index)
                        values = contribution[mask]
                        key = (
                            candidate.basis,
                            int(candidate.mode),
                            candidate.candidate_roles,
                            component,
                            time_value,
                            bin_index,
                        )
                        accumulator[key][0] += float(values.sum())
                        accumulator[key][1] += float(np.abs(values).sum())
                        accumulator[key][2] += float(mask.sum()) / len(design_dates)
    rows = []
    for key, (signed, absolute, pair_count) in accumulator.items():
        basis, mode, roles, component, time_value, bin_index = key
        rows.append(
            {
                "basis": basis,
                "mode": mode,
                "candidate_roles": roles,
                "component": component,
                "time_lag_hours": time_value,
                "spatial_bin_index": bin_index,
                "spatial_norm_lower": spatial_bins[bin_index],
                "spatial_norm_upper": spatial_bins[bin_index + 1],
                "signed_quadratic_contribution": signed,
                "absolute_pair_contribution": absolute,
                "mean_ordered_pair_count": pair_count,
            }
        )
    frame = pd.DataFrame(rows)
    totals = frame.groupby(["basis", "mode", "component"])["absolute_pair_contribution"].transform(
        "sum"
    )
    frame["fraction_of_component_absolute_pair_contribution"] = np.divide(
        frame["absolute_pair_contribution"],
        totals,
        out=np.zeros(len(frame), dtype=np.float64),
        where=totals.to_numpy() > 0.0,
    )
    return frame


def interaction_cluster_lag_attribution(
    clusters: pd.DataFrame,
    directions: np.ndarray,
    design_dates: Sequence[str],
    geometries: dict[str, Any],
    true_covariances: dict[str, np.ndarray],
    matched_covariances: dict[str, np.ndarray],
    truth: CovarianceParameters,
    *,
    spatial_bins: Sequence[float],
) -> pd.DataFrame:
    """Bin rotation-invariant intrinsic-interaction contributions by lag."""

    spatial_bins = np.asarray(spatial_bins, dtype=np.float64)
    rows = []
    for cluster in clusters.itertuples(index=False):
        mode_indices = np.asarray(cluster.mode_numbers, dtype=np.int64) - 1
        cluster_directions = directions[:, mode_indices]
        pair_kernel = cluster_directions @ cluster_directions.T
        accumulator: dict[tuple[int, int], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
        for date in design_dates:
            spatial_norm, _ = standardized_moving_lag_norms(geometries[date], truth)
            time_lag = np.rint(np.abs(geometries[date].delta_time)).astype(np.int64)
            spatial_index = np.digitize(spatial_norm, spatial_bins[1:-1], right=False)
            difference = true_covariances[date] - matched_covariances[date]
            contribution = pair_kernel * difference / len(design_dates)
            for time_value in range(int(time_lag.max()) + 1):
                time_mask = time_lag == time_value
                for bin_index in range(len(spatial_bins) - 1):
                    mask = time_mask & (spatial_index == bin_index)
                    values = contribution[mask]
                    accumulator[(time_value, bin_index)][0] += float(values.sum())
                    accumulator[(time_value, bin_index)][1] += float(np.abs(values).sum())
                    accumulator[(time_value, bin_index)][2] += float(mask.sum()) / len(design_dates)
        for (time_value, bin_index), (signed, absolute, pair_count) in accumulator.items():
            rows.append(
                {
                    "basis": cluster.basis,
                    "near_degenerate_cluster": cluster.near_degenerate_cluster,
                    "modes": cluster.modes,
                    "candidate_roles": cluster.candidate_roles,
                    "time_lag_hours": time_value,
                    "spatial_bin_index": bin_index,
                    "spatial_norm_lower": spatial_bins[bin_index],
                    "spatial_norm_upper": spatial_bins[bin_index + 1],
                    "signed_quadratic_contribution": signed,
                    "absolute_pair_contribution": absolute,
                    "mean_ordered_pair_count": pair_count,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    totals = frame.groupby(["basis", "near_degenerate_cluster"])[
        "absolute_pair_contribution"
    ].transform("sum")
    frame["fraction_of_cluster_absolute_pair_contribution"] = np.divide(
        frame["absolute_pair_contribution"],
        totals,
        out=np.zeros(len(frame), dtype=np.float64),
        where=totals.to_numpy() > 0.0,
    )
    return frame


def plot_overview(metrics: pd.DataFrame, candidates: pd.DataFrame, path: Path) -> None:
    fitted = metrics.loc[metrics["basis"] == "fitted_null"].copy()
    colors = np.where(fitted["total_variance_ratio"] >= 1.0, "#26734d", "#7a3e9d")
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    axes[0, 0].scatter(
        fitted["interaction_absolute_share"],
        fitted["total_g_score"],
        c=colors,
        s=14,
        alpha=0.65,
        linewidths=0.0,
    )
    axes[0, 0].set_xlabel("Interaction absolute share |delta_int| / (|delta_int| + |delta_comp|)")
    axes[0, 0].set_ylabel("Total one-direction g(lambda)")

    signal = fitted.nlargest(200, "total_g_score")
    signal_colors = np.where(signal["total_variance_ratio"] >= 1.0, "#26734d", "#7a3e9d")
    axes[0, 1].scatter(
        signal["rank1_energy_fraction"],
        signal["temporal_first_difference_roughness"],
        c=signal_colors,
        s=18,
        alpha=0.7,
        linewidths=0.0,
    )
    axes[0, 1].set_xlabel("Rank-one space x time energy fraction")
    axes[0, 1].set_ylabel("Temporal first-difference roughness")

    for branch, label, color in (
        ("true_gt_null", "lambda > 1", "#26734d"),
        ("true_lt_null", "lambda < 1", "#7a3e9d"),
    ):
        values = fitted.loc[fitted["total_variance_branch"] == branch, "total_g_score"].to_numpy()
        axes[1, 0].plot(
            np.arange(1, len(values) + 1),
            np.cumsum(np.sort(values)[::-1]),
            label=label,
            color=color,
        )
    axes[1, 0].set_xlabel("Modes within variance branch")
    axes[1, 0].set_ylabel("Cumulative KL contribution")
    axes[1, 0].legend(frameon=False)

    axes[1, 1].scatter(
        fitted["total_g_score"],
        fitted["intrinsic_interaction_g_score"],
        c=colors,
        s=14,
        alpha=0.65,
        linewidths=0.0,
    )
    axes[1, 1].set_xlabel("Total fitted-null g")
    axes[1, 1].set_ylabel("Same-margin intrinsic-interaction g")

    for candidate in candidates.loc[candidates["basis"] == "fitted_null"].itertuples(index=False):
        row = fitted.loc[fitted["mode"] == candidate.mode].iloc[0]
        axes[0, 0].annotate(
            f"m{candidate.mode}",
            (row["interaction_absolute_share"], row["total_g_score"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
        axes[1, 1].annotate(
            f"m{candidate.mode}",
            (row["total_g_score"], row["intrinsic_interaction_g_score"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    fig.suptitle("Generalized eigen-direction atlas: discrimination and interaction")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_candidate_maps(
    candidates: pd.DataFrame,
    directions_by_basis: dict[str, np.ndarray],
    scaled_anchor_coordinates: np.ndarray,
    output_dir: Path,
    *,
    time_count: int,
    anchor_count: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    latitude = scaled_anchor_coordinates[:, 0]
    longitude = scaled_anchor_coordinates[:, 1]
    for candidate in candidates.itertuples(index=False):
        weights = directions_by_basis[candidate.basis][:, int(candidate.mode) - 1].reshape(
            time_count,
            anchor_count,
        )
        limit = float(np.max(np.abs(weights)))
        fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), constrained_layout=True)
        for time_index, axis in enumerate(axes.ravel()):
            scatter = axis.scatter(
                longitude,
                latitude,
                c=weights[time_index],
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
                s=24,
                edgecolors="none",
            )
            axis.set_title(f"hour {time_index}")
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("moving lon / range_lon")
            axis.set_ylabel("moving lat / range_lat")
        fig.colorbar(scatter, ax=axes, shrink=0.75, label="linear-combination weight")
        fig.suptitle(
            f"{candidate.basis}, mode {candidate.mode}: {candidate.candidate_roles}\n"
            f"g={candidate.total_g_score:.4g}, interaction share="
            f"{candidate.interaction_absolute_share:.3f}, "
            f"rank-1={candidate.rank1_energy_fraction:.3f}"
        )
        name = f"{candidate.basis}_mode_{int(candidate.mode):03d}.png"
        fig.savefig(output_dir / name, dpi=180)
        plt.close(fig)


def plot_temporal_candidates(
    candidates: pd.DataFrame,
    temporal_factors: pd.DataFrame,
    path: Path,
) -> None:
    count = len(candidates)
    fig, axes = plt.subplots(count, 1, figsize=(9, max(3.0, 2.2 * count)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, candidate in zip(axes, candidates.itertuples(index=False)):
        subset = temporal_factors.loc[
            (temporal_factors["basis"] == candidate.basis)
            & (temporal_factors["mode"] == candidate.mode)
        ]
        axis.plot(
            subset["time_index"],
            subset["leading_temporal_factor"],
            marker="o",
            label="leading temporal factor",
        )
        axis.plot(
            subset["time_index"],
            subset["time_energy_fraction"],
            marker="s",
            linestyle="--",
            label="time energy fraction",
        )
        axis.axhline(0.0, color="0.7", linewidth=0.8)
        axis.set_ylabel(f"{candidate.basis}\nm{int(candidate.mode)}")
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("Hour within independent day")
    fig.suptitle("Temporal structure of covariance-only exploration candidates")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_lag_attribution(frame: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    components = ("intrinsic_interaction", "fitted_null_compensation", "total")
    for (basis, mode, roles), candidate in frame.groupby(
        ["basis", "mode", "candidate_roles"], sort=False
    ):
        limit = float(candidate["signed_quadratic_contribution"].abs().max())
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
        for axis, component in zip(axes, components):
            subset = candidate.loc[candidate["component"] == component]
            table = subset.pivot(
                index="spatial_bin_index",
                columns="time_lag_hours",
                values="signed_quadratic_contribution",
            ).sort_index(ascending=False)
            image = axis.imshow(
                table.to_numpy(),
                aspect="auto",
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
            )
            axis.set_xticks(np.arange(len(table.columns)), labels=table.columns)
            labels = []
            for bin_index in table.index:
                row = subset.loc[subset["spatial_bin_index"] == bin_index].iloc[0]
                upper = row["spatial_norm_upper"]
                labels.append(
                    f"{row['spatial_norm_lower']:g}-{upper:g}"
                    if np.isfinite(upper)
                    else f">={row['spatial_norm_lower']:g}"
                )
            axis.set_yticks(np.arange(len(table.index)), labels=labels)
            axis.set_xlabel("time lag (hours)")
            axis.set_ylabel("standardized moving spatial lag")
            axis.set_title(component.replace("_", " "))
        fig.colorbar(image, ax=axes, shrink=0.8, label="signed quadratic contribution")
        fig.suptitle(f"{basis}, mode {int(mode)}: {roles}")
        fig.savefig(output_dir / f"{basis}_mode_{int(mode):03d}.png", dpi=180)
        plt.close(fig)


def plot_interaction_cluster_heatmaps(
    clusters: pd.DataFrame,
    directions: np.ndarray,
    cluster_lag_frame: pd.DataFrame,
    anchor_order: np.ndarray,
    scaled_anchor_coordinates: np.ndarray,
    graph_eigenvectors: np.ndarray,
    output_dir: Path,
    *,
    time_count: int,
    anchor_count: int,
) -> None:
    """Plot raw modes and rotation-invariant summaries for two-mode clusters."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for cluster in clusters.itertuples(index=False):
        if len(cluster.mode_numbers) != 2:
            raise ValueError("space-time cluster heatmaps currently require exactly two modes")
        mode_indices = np.asarray(cluster.mode_numbers, dtype=np.int64) - 1
        cluster_directions = directions[:, mode_indices]
        weights, amplitude, temporal_similarity, spatial_similarity = (
            cluster_subspace_heatmap_arrays(
                cluster_directions,
                time_count=time_count,
                anchor_count=anchor_count,
            )
        )
        spectral_energy = cluster_space_time_spectral_energy(weights, graph_eigenvectors)
        ordered_weights = weights[:, anchor_order, :]
        ordered_amplitude = amplitude[:, anchor_order]
        ordered_spatial_similarity = spatial_similarity[np.ix_(anchor_order, anchor_order)]
        ordered_coordinates = scaled_anchor_coordinates[anchor_order]
        lag_subset = cluster_lag_frame.loc[
            cluster_lag_frame["near_degenerate_cluster"] == cluster.near_degenerate_cluster
        ]
        lag_table = lag_subset.pivot(
            index="spatial_bin_index",
            columns="time_lag_hours",
            values="signed_quadratic_contribution",
        ).sort_index(ascending=False)

        fig, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
        weight_limit = float(np.max(np.abs(ordered_weights)))
        raw_images = []
        for panel, mode_number in enumerate(cluster.mode_numbers):
            image = axes[0, panel].imshow(
                ordered_weights[:, :, panel],
                origin="lower",
                aspect="auto",
                cmap="coolwarm",
                vmin=-weight_limit,
                vmax=weight_limit,
            )
            raw_images.append(image)
            axes[0, panel].set_title(f"raw eigenvector mode {mode_number} (sign/basis arbitrary)")
            axes[0, panel].set_xlabel("anchor position in graph ordering")
            axes[0, panel].set_ylabel("hour")
            axes[0, panel].set_yticks(np.arange(time_count))
        fig.colorbar(
            raw_images[0],
            ax=[axes[0, panel] for panel in range(len(cluster.mode_numbers))],
            shrink=0.75,
            label="unit-null-variance weight",
        )

        amplitude_image = axes[0, 2].imshow(
            ordered_amplitude,
            origin="lower",
            aspect="auto",
            cmap="viridis",
        )
        axes[0, 2].set_title("rotation-invariant subspace amplitude")
        axes[0, 2].set_xlabel("anchor position in graph ordering")
        axes[0, 2].set_ylabel("hour")
        axes[0, 2].set_yticks(np.arange(time_count))
        fig.colorbar(
            amplitude_image,
            ax=axes[0, 2],
            shrink=0.75,
            label="sqrt(sum mode weight²)",
        )

        path_axis = axes[0, 3]
        path_scatter = path_axis.scatter(
            ordered_coordinates[:, 1],
            ordered_coordinates[:, 0],
            c=np.arange(anchor_count),
            cmap="viridis",
            s=18,
            edgecolors="none",
            zorder=2,
        )
        path_axis.scatter(
            ordered_coordinates[[0, -1], 1],
            ordered_coordinates[[0, -1], 0],
            c=[0, anchor_count - 1],
            cmap="viridis",
            vmin=0,
            vmax=anchor_count - 1,
            s=55,
            edgecolors="black",
            linewidths=0.7,
            zorder=3,
        )
        path_axis.set_title("2D anchors; color is display order only")
        path_axis.set_xlabel("moving lon / range_lon")
        path_axis.set_ylabel("moving lat / range_lat")
        path_axis.set_aspect("equal", adjustable="box")
        path_axis.text(
            0.02,
            0.02,
            "adjacent columns do not define a spatial region",
            transform=path_axis.transAxes,
            fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
        )
        fig.colorbar(
            path_scatter,
            ax=path_axis,
            shrink=0.75,
            label="Fiedler-order column",
        )

        temporal_image = axes[1, 0].imshow(
            temporal_similarity,
            origin="lower",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
        )
        axes[1, 0].set_title("rotation-invariant temporal similarity")
        axes[1, 0].set_xlabel("hour")
        axes[1, 0].set_ylabel("hour")
        axes[1, 0].set_xticks(np.arange(time_count))
        axes[1, 0].set_yticks(np.arange(time_count))
        fig.colorbar(temporal_image, ax=axes[1, 0], shrink=0.75, label="normalized Gram")

        spatial_image = axes[1, 1].imshow(
            ordered_spatial_similarity,
            origin="lower",
            aspect="equal",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
        )
        axes[1, 1].set_title("rotation-invariant spatial similarity")
        axes[1, 1].set_xlabel("graph-ordered anchor position")
        axes[1, 1].set_ylabel("graph-ordered anchor position")
        fig.colorbar(spatial_image, ax=axes[1, 1], shrink=0.75, label="normalized Gram")

        spectrum_image = axes[1, 2].imshow(
            100.0 * spectral_energy,
            origin="lower",
            aspect="auto",
            cmap="magma",
            norm=PowerNorm(gamma=0.5),
        )
        axes[1, 2].set_title("rotation-invariant space x time spectrum")
        axes[1, 2].set_xlabel("graph spatial frequency rank (smooth to rough)")
        axes[1, 2].set_ylabel("temporal DCT frequency")
        axes[1, 2].set_yticks(np.arange(time_count))
        fig.colorbar(
            spectrum_image,
            ax=axes[1, 2],
            shrink=0.75,
            label="percent of Euclidean filter-weight energy",
        )

        lag_limit = float(lag_table.abs().to_numpy().max())
        lag_image = axes[1, 3].imshow(
            lag_table.to_numpy(),
            aspect="auto",
            cmap="coolwarm",
            vmin=-lag_limit,
            vmax=lag_limit,
        )
        axes[1, 3].set_xticks(np.arange(len(lag_table.columns)), labels=lag_table.columns)
        spatial_labels = []
        for bin_index in lag_table.index:
            row = lag_subset.loc[lag_subset["spatial_bin_index"] == bin_index].iloc[0]
            upper = row["spatial_norm_upper"]
            spatial_labels.append(
                f"{row['spatial_norm_lower']:g}-{upper:g}"
                if np.isfinite(upper)
                else f">={row['spatial_norm_lower']:g}"
            )
        axes[1, 3].set_yticks(np.arange(len(lag_table.index)), labels=spatial_labels)
        axes[1, 3].set_title("cluster intrinsic-interaction contribution")
        axes[1, 3].set_xlabel("time lag (hours)")
        axes[1, 3].set_ylabel("standardized moving spatial lag")
        fig.colorbar(
            lag_image,
            ax=axes[1, 3],
            shrink=0.75,
            label="signed quadratic contribution",
        )
        fig.suptitle(
            f"{cluster.near_degenerate_cluster}: modes {cluster.modes} — "
            f"{cluster.candidate_roles}"
        )
        mode_slug = "_".join(f"{int(value):03d}" for value in cluster.mode_numbers)
        fig.savefig(
            output_dir / f"{cluster.basis}_{cluster.near_degenerate_cluster}_modes_{mode_slug}.png",
            dpi=180,
        )
        plt.close(fig)


def write_report(
    path: Path,
    metrics: pd.DataFrame,
    candidates: pd.DataFrame,
    lag_frame: pd.DataFrame,
    interaction_clusters: pd.DataFrame,
    *,
    interaction_compensation_frobenius_cosine: float,
) -> None:
    fitted = metrics.loc[metrics["basis"] == "fitted_null"].sort_values("mode")
    interaction = metrics.loc[metrics["basis"] == "margin_matched_interaction"]
    total_kl = float(fitted["total_g_score"].sum())
    intrinsic_kl = float(interaction["intrinsic_interaction_g_score"].sum())
    cumulative = fitted["total_g_score"].cumsum() / total_kl
    thresholds = {
        level: int(np.searchsorted(cumulative.to_numpy(), level) + 1) for level in (0.5, 0.8, 0.9)
    }
    branch_kl = fitted.groupby("total_variance_branch")["total_g_score"].sum().to_dict()
    branch_patterns: dict[str, pd.Series] = {}
    for branch, group in fitted.groupby("total_variance_branch", sort=False):
        branch_patterns[branch] = group.nlargest(20, "total_g_score").median(numeric_only=True)
    lines = [
        "# Eigen-direction interaction atlas",
        "",
        "No response value was used to reconstruct, rank, or annotate these directions.",
        "",
        "## Global structure",
        "",
        f"- Total reference-covariance KL: `{total_kl:.8g}`.",
        f"- Same-margin intrinsic-interaction KL before null refitting: `{intrinsic_kl:.8g}`.",
        f"- Fraction of that KL remaining after the separable null is refitted: "
        f"`{total_kl / intrinsic_kl:.3%}`.",
        "- Null-whitened Frobenius cosine between intrinsic interaction and "
        f"fitted-null compensation: `{interaction_compensation_frobenius_cosine:.4f}`.",
        f"- Modes needed for 50%, 80%, and 90% cumulative KL: "
        f"`{thresholds[0.5]}`, `{thresholds[0.8]}`, `{thresholds[0.9]}`.",
        f"- KL from lambda > 1 directions: `{branch_kl.get('true_gt_null', 0.0):.8g}`.",
        f"- KL from lambda < 1 directions: `{branch_kl.get('true_lt_null', 0.0):.8g}`.",
        "",
        "## Two leading pattern families",
        "",
        "Among the 20 strongest modes in each variance branch:",
        "",
        "| branch | median rank-1 | median effective rank | median temporal roughness | median high-frequency weight energy |",
        "|---|---:|---:|---:|---:|",
        (
            "| lambda > 1 | "
            f"{branch_patterns['true_gt_null']['rank1_energy_fraction']:.3f} | "
            f"{branch_patterns['true_gt_null']['space_time_effective_rank']:.3f} | "
            f"{branch_patterns['true_gt_null']['temporal_first_difference_roughness']:.3f} | "
            f"{branch_patterns['true_gt_null']['dct_high_frequency_fraction']:.3f} |"
        ),
        (
            "| lambda < 1 | "
            f"{branch_patterns['true_lt_null']['rank1_energy_fraction']:.3f} | "
            f"{branch_patterns['true_lt_null']['space_time_effective_rank']:.3f} | "
            f"{branch_patterns['true_lt_null']['temporal_first_difference_roughness']:.3f} | "
            f"{branch_patterns['true_lt_null']['dct_high_frequency_fraction']:.3f} |"
        ),
        "",
        "## Covariance-only exploration candidates",
        "",
        "These are transparent exemplars, not a finalized selected test.",
        "",
        "| basis | mode | role | total g | intrinsic g | interaction share | rank-1 | high-frequency weight energy |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in candidates.itertuples(index=False):
        lines.append(
            f"| {row.basis} | {int(row.mode)} | {row.candidate_roles} | "
            f"{row.total_g_score:.5g} | {row.intrinsic_interaction_g_score:.5g} | "
            f"{row.interaction_absolute_share:.3f} | {row.rank1_energy_fraction:.3f} | "
            f"{row.dct_high_frequency_fraction:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Space-time cluster heatmaps",
            "",
            "Raw eigenvector signs are arbitrary and close eigenvalues permit within-cluster rotation.  "
            "The pointwise amplitude, temporal/spatial Gram, joint graph-Fourier/DCT spectrum, "
            "and cluster lag-attribution panels are therefore the stable interpretation targets.",
            "",
            "Heatmap columns use one-dimensional Fiedler graph seriation only.  Adjacent columns "
            "are not a spatial rectangle and must not define a spatial contrast.",
            "",
        ]
    )
    for cluster in interaction_clusters.itertuples(index=False):
        lines.append(
            f"- `{cluster.near_degenerate_cluster}`: modes `{cluster.modes}` "
            f"({cluster.candidate_roles}), eigenvalue range "
            f"`[{cluster.cluster_eigenvalue_min:.4f}, "
            f"{cluster.cluster_eigenvalue_max:.4f}]`; high-frequency Euclidean "
            f"filter-weight energy "
            f"`{cluster.cluster_dct_high_frequency_fraction:.3f}`, mean adjacent-hour "
            f"similarity `{cluster.cluster_mean_adjacent_temporal_similarity:.3f}`, "
            f"peak energy hour `{int(cluster.cluster_peak_energy_hour)}`, dominant joint "
            f"frequency `(temporal={int(cluster.cluster_dominant_temporal_frequency)}, "
            f"graph-rank={int(cluster.cluster_dominant_graph_frequency_rank)})`."
        )
    lines.extend(
        [
            "",
            "## Dominant intrinsic-interaction lag cells",
            "",
        ]
    )
    intrinsic = lag_frame.loc[lag_frame["component"] == "intrinsic_interaction"]
    for (basis, mode, roles), group in intrinsic.groupby(
        ["basis", "mode", "candidate_roles"], sort=False
    ):
        dominant = group.loc[group["absolute_pair_contribution"].idxmax()]
        upper = dominant["spatial_norm_upper"]
        interval = (
            f"[{dominant['spatial_norm_lower']:g}, {upper:g})"
            if np.isfinite(upper)
            else f"[{dominant['spatial_norm_lower']:g}, infinity)"
        )
        lines.append(
            f"- `{basis}` mode `{int(mode)}` ({roles}): time lag "
            f"`{int(dominant['time_lag_hours'])}` hours, moving spatial norm `{interval}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- A non-rank-one weight matrix describes the filter; it does not by itself prove covariance nonseparability.",
            "- Near-degenerate eigenvectors may rotate.  Interpret their labelled cluster or subspace before interpreting one mode.",
            "- The Fiedler ordering is only a display device; the 2D anchor-order panel records the original spatial geometry.",
            "- The graph-Fourier/DCT panel decomposes Euclidean filter-weight energy, not covariance variance or KL.  It is conditional on the standardized coordinates and k-nearest-neighbor graph; individual graph-frequency cells can rotate inside a repeated graph-Laplacian eigenspace.",
            "- The intrinsic/compensation split uses the known simulation truth.  A real-data analogue must remove the fitted-null nuisance tangent space.",
            "- Candidate construction and all tuning must remain on the covariance-only design split.  Held-out responses belong only in the final test.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    pilot_dir = args.pilot_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else pilot_dir / "eigen_direction_atlas"
    )
    manifest_path = pilot_dir / "run_manifest.json"
    point_path = pilot_dir / "selected_flow_tube_points.csv"
    eigenvalue_path = pilot_dir / "generalized_eigenvalues.csv"
    if not manifest_path.is_file() or not point_path.is_file():
        raise FileNotFoundError("pilot-dir must contain run_manifest.json and selected points")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = manifest["split"]
    if bool(split.get("responses_used_for_null_fit_or_direction_selection", True)):
        raise ValueError("pilot manifest does not certify covariance-only direction selection")
    design_dates = tuple(str(value) for value in split["design_dates"])
    heldout_dates = tuple(str(value) for value in split["heldout_dates"])
    all_dates = design_dates + heldout_dates
    days = load_coordinate_days(point_path, all_dates)
    day_by_date = {day.date: day for day in days}
    design_days = [day_by_date[date] for date in design_dates]

    truth = covariance_parameters(manifest["truth"])
    fitted_null = covariance_parameters(manifest["null"]["fit"]["parameters"])
    if truth.nugget != 0.0 or fitted_null.nugget != 0.0:
        raise ValueError("the atlas currently expects statistical nugget zero")
    jitter_ratio = float(manifest["numerics"]["numerical_jitter_ratio"])
    time_count = int(manifest["subset"]["hours"])
    dimension = int(manifest["subset"]["dimension_per_day"])
    if dimension % time_count != 0:
        raise ValueError("dimension is not divisible by the number of hours")
    anchor_count = dimension // time_count

    geometries = {date: pairwise_lags(day_by_date[date].coordinates) for date in all_dates}
    true_covariances = {
        date: joint_matern_half_covariance(
            geometries[date],
            truth,
            numerical_jitter_ratio=jitter_ratio,
        )
        for date in all_dates
    }
    null_covariances = {
        date: advected_separable_covariance(
            geometries[date],
            fitted_null,
            numerical_jitter_ratio=jitter_ratio,
        )
        for date in all_dates
    }
    matched_covariances = {
        date: advected_separable_covariance(
            geometries[date],
            truth,
            numerical_jitter_ratio=jitter_ratio,
        )
        for date in all_dates
    }
    reference_true = np.mean(np.stack([true_covariances[date] for date in design_dates]), axis=0)
    reference_null = np.mean(np.stack([null_covariances[date] for date in design_dates]), axis=0)
    reference_matched = np.mean(
        np.stack([matched_covariances[date] for date in design_dates]),
        axis=0,
    )

    fitted_eigen = solve_generalized_eigenproblem(reference_true, reference_null)
    interaction_eigen = solve_generalized_eigenproblem(reference_true, reference_matched)
    fitted_directions, fitted_metrics = direction_covariance_metrics(
        fitted_eigen.eigenvectors,
        reference_true,
        reference_null,
        reference_matched,
    )
    interaction_directions, interaction_metrics = direction_covariance_metrics(
        interaction_eigen.eigenvectors,
        reference_true,
        reference_null,
        reference_matched,
    )
    directions_by_basis = {
        "fitted_null": fitted_directions,
        "margin_matched_interaction": interaction_directions,
    }

    moving_coordinates = moving_anchor_coordinates(
        design_days,
        truth,
        time_count=time_count,
        anchor_count=anchor_count,
    )
    scaled_coordinates = np.column_stack(
        (
            (moving_coordinates[:, 0] - moving_coordinates[:, 0].mean()) / truth.range_lat,
            (moving_coordinates[:, 1] - moving_coordinates[:, 1].mean()) / truth.range_lon,
        )
    )
    graph_neighbors = tuple(sorted(set(int(value) for value in args.graph_neighbors)))
    graph_edges = {value: knn_edges(scaled_coordinates, value) for value in graph_neighbors}
    heatmap_graph_neighbors = min(6, anchor_count - 1)
    graph_frequencies, graph_eigenvectors = graph_fourier_basis(
        scaled_coordinates,
        neighbors=heatmap_graph_neighbors,
    )

    metric_frames = []
    temporal_frames = []
    for basis, directions in directions_by_basis.items():
        covariance_frame = fitted_metrics if basis == "fitted_null" else interaction_metrics
        eigen = fitted_eigen if basis == "fitted_null" else interaction_eigen
        patterns, temporal = pattern_metrics(
            directions,
            time_count=time_count,
            anchor_count=anchor_count,
            graph_edges=graph_edges,
        )
        clusters, cluster_sizes = assign_near_degenerate_clusters(
            eigen.eigenvalues,
            relative_log_gap=float(args.cluster_relative_log_gap),
        )
        covariance_frame = covariance_frame.copy()
        covariance_frame.insert(0, "basis", basis)
        covariance_frame["basis_eigenvalue"] = eigen.eigenvalues
        covariance_frame["basis_g_score"] = eigen.scores
        covariance_frame["near_degenerate_cluster"] = clusters
        covariance_frame["near_degenerate_cluster_size"] = cluster_sizes
        covariance_frame = covariance_frame.merge(patterns, on="mode", validate="one_to_one")
        metric_frames.append(covariance_frame)
        temporal.insert(0, "basis", basis)
        temporal_frames.append(temporal)
    metrics = pd.concat(metric_frames, ignore_index=True)
    temporal_factors = pd.concat(temporal_frames, ignore_index=True)

    day_long, day_summary = day_stability_metrics(
        directions_by_basis,
        design_dates,
        true_covariances,
        null_covariances,
        matched_covariances,
    )
    metrics = metrics.merge(day_summary, on=["basis", "mode"], validate="one_to_one")
    metrics["pareto_total_and_intrinsic_g"] = False
    for basis in directions_by_basis:
        mask = metrics["basis"] == basis
        metrics.loc[mask, "pareto_total_and_intrinsic_g"] = pareto_mask(
            metrics.loc[mask, "total_g_score"].to_numpy(),
            metrics.loc[mask, "intrinsic_interaction_g_score"].to_numpy(),
        )

    alignment = np.abs(fitted_directions.T @ reference_null @ interaction_directions)
    fitted_match = np.argmax(alignment, axis=1)
    interaction_match = np.argmax(alignment, axis=0)
    fitted_mask = metrics["basis"] == "fitted_null"
    interaction_mask = metrics["basis"] == "margin_matched_interaction"
    metrics.loc[fitted_mask, "closest_other_basis_mode"] = fitted_match + 1
    metrics.loc[fitted_mask, "closest_other_basis_absolute_cosine"] = alignment[
        np.arange(dimension), fitted_match
    ]
    metrics.loc[interaction_mask, "closest_other_basis_mode"] = interaction_match + 1
    metrics.loc[interaction_mask, "closest_other_basis_absolute_cosine"] = alignment[
        interaction_match, np.arange(dimension)
    ]

    cluster_rows = []
    for (basis, cluster), group in metrics.groupby(
        ["basis", "near_degenerate_cluster"],
        sort=False,
    ):
        cluster_rows.append(
            {
                "basis": basis,
                "near_degenerate_cluster": cluster,
                "cluster_size": len(group),
                "modes": ";".join(str(int(value)) for value in group["mode"]),
                "basis_eigenvalue_min": float(group["basis_eigenvalue"].min()),
                "basis_eigenvalue_max": float(group["basis_eigenvalue"].max()),
                "basis_g_sum": float(group["basis_g_score"].sum()),
                "total_g_sum": float(group["total_g_score"].sum()),
                "intrinsic_interaction_g_sum": float(group["intrinsic_interaction_g_score"].sum()),
                "interaction_absolute_share_median": float(
                    group["interaction_absolute_share"].median()
                ),
                "minimum_design_day_80pct_membership": float(
                    group["design_day_80pct_membership_min"].min()
                ),
            }
        )
    cluster_summary = pd.DataFrame(cluster_rows)

    branch_rows = []
    fitted_for_summary = metrics.loc[metrics["basis"] == "fitted_null"]
    for branch, group in fitted_for_summary.groupby("total_variance_branch", sort=False):
        top = group.nlargest(20, "total_g_score")
        branch_rows.append(
            {
                "total_variance_branch": branch,
                "top_mode_count": len(top),
                "top_modes": ";".join(str(int(value)) for value in top["mode"]),
                "branch_total_kl": float(group["total_g_score"].sum()),
                "top20_total_kl": float(top["total_g_score"].sum()),
                "median_rank1_energy_fraction": float(top["rank1_energy_fraction"].median()),
                "median_space_time_effective_rank": float(
                    top["space_time_effective_rank"].median()
                ),
                "median_temporal_first_difference_roughness": float(
                    top["temporal_first_difference_roughness"].median()
                ),
                "median_dct_high_frequency_fraction": float(
                    top["dct_high_frequency_fraction"].median()
                ),
                "median_effective_spatial_anchors": float(
                    top["effective_spatial_anchors"].median()
                ),
                "median_interaction_absolute_share": float(
                    top["interaction_absolute_share"].median()
                ),
            }
        )
    branch_summary = pd.DataFrame(branch_rows)

    whitened_interaction = whiten_covariance_difference(
        reference_true - reference_matched,
        reference_null,
    )
    whitened_compensation = whiten_covariance_difference(
        reference_matched - reference_null,
        reference_null,
    )
    frobenius_cosine = float(
        np.sum(whitened_interaction * whitened_compensation)
        / (np.linalg.norm(whitened_interaction) * np.linalg.norm(whitened_compensation))
    )

    candidates = select_exploration_candidates(
        metrics,
        relative_g_floor=float(args.candidate_relative_g_floor),
    )
    lag_frame = lag_attribution(
        candidates,
        directions_by_basis,
        design_dates,
        geometries,
        true_covariances,
        null_covariances,
        matched_covariances,
        truth,
        spatial_bins=DEFAULT_SPATIAL_BINS,
    )

    interaction_cluster_rows = []
    dominant_candidates = candidates.loc[
        (candidates["basis"] == "fitted_null")
        & candidates["candidate_roles"].str.contains("interaction_dominant")
    ]
    for candidate in dominant_candidates.itertuples(index=False):
        members = metrics.loc[
            (metrics["basis"] == candidate.basis)
            & (metrics["near_degenerate_cluster"] == candidate.near_degenerate_cluster)
        ].sort_values("mode")
        mode_numbers = tuple(int(value) for value in members["mode"])
        if len(mode_numbers) != 2:
            continue
        cluster_directions = fitted_directions[:, np.asarray(mode_numbers) - 1]
        cluster_weights = cluster_directions.reshape(time_count, anchor_count, 2)
        _, _, temporal_similarity, _ = cluster_subspace_heatmap_arrays(
            cluster_directions,
            time_count=time_count,
            anchor_count=anchor_count,
        )
        dct_energy = np.square(scipy.fft.dct(cluster_weights, type=2, axis=0, norm="ortho")).sum(
            axis=(1, 2)
        )
        dct_energy /= dct_energy.sum()
        spectral_energy = cluster_space_time_spectral_energy(
            cluster_weights,
            graph_eigenvectors,
        )
        time_energy = np.square(cluster_weights).sum(axis=(1, 2))
        time_energy /= time_energy.sum()
        graph_energy = spectral_energy.sum(axis=0)
        temporal_spectral_energy = spectral_energy.sum(axis=1)
        dominant_frequency = np.unravel_index(
            int(np.argmax(spectral_energy)),
            spectral_energy.shape,
        )
        interaction_cluster_rows.append(
            {
                "basis": candidate.basis,
                "near_degenerate_cluster": candidate.near_degenerate_cluster,
                "modes": ";".join(str(value) for value in mode_numbers),
                "mode_numbers": mode_numbers,
                "candidate_roles": candidate.candidate_roles,
                "cluster_eigenvalue_min": float(members["basis_eigenvalue"].min()),
                "cluster_eigenvalue_max": float(members["basis_eigenvalue"].max()),
                "cluster_total_g": float(members["total_g_score"].sum()),
                "cluster_intrinsic_interaction_g": float(
                    members["intrinsic_interaction_g_score"].sum()
                ),
                "cluster_dct_high_frequency_fraction": float(dct_energy[4:].sum()),
                "cluster_graph_frequency_centroid": float(np.dot(graph_frequencies, graph_energy)),
                "cluster_temporal_frequency_centroid": float(
                    np.dot(np.arange(time_count), temporal_spectral_energy)
                ),
                "cluster_dominant_temporal_frequency": int(dominant_frequency[0]),
                "cluster_dominant_graph_frequency_rank": int(dominant_frequency[1]),
                "cluster_mean_adjacent_temporal_similarity": float(
                    np.diag(temporal_similarity, k=1).mean()
                ),
                "cluster_early_late_temporal_similarity": float(temporal_similarity[:3, 5:].mean()),
                "cluster_peak_energy_hour": int(np.argmax(time_energy)),
                "cluster_peak_hour_energy_fraction": float(time_energy.max()),
            }
        )
    interaction_clusters = pd.DataFrame(interaction_cluster_rows)
    cluster_lag_frame = interaction_cluster_lag_attribution(
        interaction_clusters,
        fitted_directions,
        design_dates,
        geometries,
        true_covariances,
        matched_covariances,
        truth,
        spatial_bins=DEFAULT_SPATIAL_BINS,
    )
    anchor_order = fiedler_anchor_order(
        scaled_coordinates,
        neighbors=heatmap_graph_neighbors,
    )

    candidate_weight_rows = []
    for candidate in candidates.itertuples(index=False):
        weights = directions_by_basis[candidate.basis][:, int(candidate.mode) - 1].reshape(
            time_count,
            anchor_count,
        )
        for time_index in range(time_count):
            for anchor_index in range(anchor_count):
                candidate_weight_rows.append(
                    {
                        "basis": candidate.basis,
                        "mode": int(candidate.mode),
                        "candidate_roles": candidate.candidate_roles,
                        "time_index": time_index,
                        "anchor_rank": anchor_index + 1,
                        "moving_latitude": moving_coordinates[anchor_index, 0],
                        "moving_longitude": moving_coordinates[anchor_index, 1],
                        "weight": weights[time_index, anchor_index],
                    }
                )
    candidate_weights = pd.DataFrame(candidate_weight_rows)
    cluster_weight_rows = []
    inverse_anchor_order = np.empty(anchor_count, dtype=np.int64)
    inverse_anchor_order[anchor_order] = np.arange(anchor_count)
    for cluster in interaction_clusters.itertuples(index=False):
        for mode_number in cluster.mode_numbers:
            weights = fitted_directions[:, mode_number - 1].reshape(
                time_count,
                anchor_count,
            )
            for time_index in range(time_count):
                for anchor_index in range(anchor_count):
                    cluster_weight_rows.append(
                        {
                            "basis": cluster.basis,
                            "near_degenerate_cluster": cluster.near_degenerate_cluster,
                            "modes": cluster.modes,
                            "candidate_roles": cluster.candidate_roles,
                            "mode": mode_number,
                            "time_index": time_index,
                            "anchor_rank": anchor_index + 1,
                            "graph_order_position": int(inverse_anchor_order[anchor_index]),
                            "moving_latitude": moving_coordinates[anchor_index, 0],
                            "moving_longitude": moving_coordinates[anchor_index, 1],
                            "weight": weights[time_index, anchor_index],
                        }
                    )
    cluster_weights = pd.DataFrame(cluster_weight_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(output_dir / "eigen_direction_metrics.csv", metrics)
    atomic_csv(output_dir / "near_degenerate_cluster_summary.csv", cluster_summary)
    atomic_csv(output_dir / "top_branch_pattern_summary.csv", branch_summary)
    atomic_csv(output_dir / "design_day_direction_stability.csv", day_long)
    atomic_csv(output_dir / "temporal_factors.csv", temporal_factors)
    atomic_csv(output_dir / "exploration_candidates.csv", candidates)
    atomic_csv(output_dir / "exploration_candidate_weights.csv", candidate_weights)
    atomic_csv(output_dir / "lag_attribution.csv", lag_frame)
    cluster_manifest = interaction_clusters.drop(columns="mode_numbers").copy()
    atomic_csv(output_dir / "interaction_cluster_summary.csv", cluster_manifest)
    atomic_csv(output_dir / "interaction_cluster_weights.csv", cluster_weights)
    atomic_csv(output_dir / "interaction_cluster_lag_attribution.csv", cluster_lag_frame)
    plot_overview(metrics, candidates, output_dir / "figures/eigen_direction_overview.png")
    plot_candidate_maps(
        candidates,
        directions_by_basis,
        scaled_coordinates,
        output_dir / "figures/direction_maps",
        time_count=time_count,
        anchor_count=anchor_count,
    )
    plot_temporal_candidates(
        candidates,
        temporal_factors,
        output_dir / "figures/candidate_temporal_profiles.png",
    )
    plot_lag_attribution(lag_frame, output_dir / "figures/lag_attribution")
    plot_interaction_cluster_heatmaps(
        interaction_clusters,
        fitted_directions,
        cluster_lag_frame,
        anchor_order,
        scaled_coordinates,
        graph_eigenvectors,
        output_dir / "figures/space_time_cluster_heatmaps",
        time_count=time_count,
        anchor_count=anchor_count,
    )
    write_report(
        output_dir / "REPORT.md",
        metrics,
        candidates,
        lag_frame,
        interaction_clusters,
        interaction_compensation_frobenius_cosine=frobenius_cosine,
    )

    saved_eigen_error = None
    if eigenvalue_path.is_file():
        saved = pd.read_csv(eigenvalue_path)["eigenvalue"].to_numpy(dtype=np.float64)
        if saved.shape == fitted_eigen.eigenvalues.shape:
            saved_eigen_error = float(np.max(np.abs(saved - fitted_eigen.eigenvalues)))
    summary = {
        "script": str(Path(__file__).resolve()),
        "pilot_dir": str(pilot_dir),
        "inputs": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "selected_points": str(point_path),
            "selected_points_sha256": sha256(point_path),
        },
        "response_usage": {
            "response_columns_read": False,
            "direction_reconstruction_and_annotations_are_covariance_only": True,
            "heldout_responses_used": False,
        },
        "design_dates": design_dates,
        "heldout_dates_not_used_for_candidate_flags": heldout_dates,
        "dimension": dimension,
        "time_count": time_count,
        "anchor_count": anchor_count,
        "truth": truth.to_dict(),
        "fitted_null": fitted_null.to_dict(),
        "interaction_comparator": "advected separable covariance at the truth parameters",
        "fitted_null_eigenproblem": {
            "matrix_kl": fitted_eigen.matrix_kl,
            "spectral_kl": fitted_eigen.spectral_kl,
            "max_relative_residual": fitted_eigen.max_relative_residual,
            "max_null_orthonormality_error": fitted_eigen.max_null_orthonormality_error,
            "max_absolute_difference_from_saved_eigenvalues": saved_eigen_error,
        },
        "interaction_eigenproblem": {
            "matrix_kl": interaction_eigen.matrix_kl,
            "spectral_kl": interaction_eigen.spectral_kl,
            "max_relative_residual": interaction_eigen.max_relative_residual,
            "max_null_orthonormality_error": interaction_eigen.max_null_orthonormality_error,
        },
        "maximum_decomposition_absolute_error": float(
            metrics["decomposition_absolute_error"].max()
        ),
        "whitened_interaction_compensation_frobenius_cosine": frobenius_cosine,
        "cluster_relative_log_gap": float(args.cluster_relative_log_gap),
        "candidate_relative_g_floor": float(args.candidate_relative_g_floor),
        "graph_neighbor_sensitivity": graph_neighbors,
        "space_time_heatmap_graph_neighbors": heatmap_graph_neighbors,
        "candidate_count": len(candidates),
        "candidate_roles": candidates[["basis", "mode", "candidate_roles"]].to_dict(
            orient="records"
        ),
        "interaction_clusters_with_space_time_heatmaps": cluster_manifest.to_dict(orient="records"),
        "space_time_heatmap_anchor_order": ("weighted k-nearest-neighbor graph Fiedler ordering"),
        "space_time_spectrum_semantics": (
            "Temporal-DCT by normalized-graph-Laplacian Fourier decomposition of "
            "Euclidean cluster filter-weight energy; not covariance variance or KL"
        ),
        "interpretation_boundary": (
            "Oracle, covariance-only exploration under known simulation truth.  Candidate "
            "flags organize subsequent model reduction and are not a finalized test."
        ),
    }
    atomic_json(output_dir / "atlas_summary.json", summary)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "candidate_count": len(candidates),
                "max_decomposition_error": summary["maximum_decomposition_absolute_error"],
            }
        )
    )


if __name__ == "__main__":
    main()
