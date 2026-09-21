#!/usr/bin/env python3
"""Track one fitted day's full-eigen diagnostic at 400, 600, and 900 sites.

The adapted and fixed lag-6/4/3 fitted parameters and full-data GLS beta are
held fixed.  Only the common-valid spatial sampling design changes.  Two
designs are evaluated at every requested size:

* ``maxmin``: nested prefixes of one common-valid max-min ordering;
* ``regular``: exact-size quasi-regular lattices snapped to common-valid sites.

The same sites are used at all eight hours, giving dense eigensystems of size
3,200, 4,800, and 7,200 by default.  Besides the cumulative whitened-residual
curve, this script computes spatial roughness of every space-time covariance
eigenvector with a distance-scaled k-nearest-neighbor graph.  This separates
"high eigen-index" from physically rapid spatial variation.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.sparse
import scipy.spatial
import torch


HERE = Path(__file__).resolve().parent
SUBMIT_DIR = Path(os.environ.get("SLURM_SUBMIT_DIR", str(HERE))).resolve()
AMAREL_ROOT = Path("/home/jl2815/tco")
LOCAL_SRC = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/src")
for candidate in (AMAREL_ROOT, SUBMIT_DIR, HERE, LOCAL_SRC):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from GEMS_TCO import orderings  # noqa: E402
import vecchia_adapted_fixed_lag643_core as core  # noqa: E402
import vecchia_real60_adapted_fixed_full_eigen_lag643_run10 as reference  # noqa: E402


METHODS = reference.METHODS
DESIGNS = ("maxmin", "regular")
COLORS = reference.COLORS
LINESTYLES = {400: ":", 600: "--", 900: "-"}
MARKERS = {"adapted": "o", "fixed": "s"}
CACHE_DIR_NAME = ".full_eigen_cache"


def parse_counts(text: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted(set(int(part.strip()) for part in str(text).split(","))))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("counts must be comma-separated integers") from exc
    if len(values) < 2 or any(value < 10 for value in values):
        raise argparse.ArgumentTypeError("supply at least two spatial counts >= 10")
    return values


def zero_based_order(values: np.ndarray, n: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.int64).reshape(-1)
    if result.size == n and result.min() == 1 and result.max() == n:
        result = result - 1
    if (
        result.size != n
        or np.unique(result).size != n
        or result.min() < 0
        or result.max() >= n
    ):
        raise RuntimeError("Invalid max-min ordering")
    return result


def day_spec(date: str) -> dict[str, Any]:
    matches = [spec for spec in reference.date_specs() if spec["date"] == date]
    if len(matches) != 1:
        raise ValueError(
            f"{date} is not one of the 59 usable July dates; "
            "2025-07-24 is intentionally unavailable"
        )
    return matches[0]


def seed_fits_from_checkpoint(
    source_path: Path,
    output_root: Path,
    spec: dict[str, Any],
) -> list[dict[str, Any]]:
    existing = reference.load_fit_results(output_root)
    if existing:
        return existing
    if not source_path.is_file():
        print(
            f"Source fitted checkpoint not found: {source_path}; "
            "adapted/fixed will be fitted once for this date",
            flush=True,
        )
        return []
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    selected = [
        row
        for row in payload
        if str(row.get("dataset_id")) == spec["dataset_id"]
        and str(row.get("method")) in METHODS
    ]
    if {str(row["method"]) for row in selected} != set(METHODS):
        print(
            f"Source checkpoint has no complete adapted/fixed pair for {spec['date']}; "
            "the missing fits will be computed",
            flush=True,
        )
    if selected:
        reference.persist_fit_results(selected, output_root)
    return selected


def common_valid_indices(asset: core.DayAsset) -> np.ndarray:
    arrays = [value.detach().cpu().numpy() for value in asset.source_map.values()]
    valid = np.ones(arrays[0].shape[0], dtype=bool)
    for array in arrays:
        valid &= np.all(np.isfinite(array), axis=1)
    indices = np.flatnonzero(valid).astype(np.int64)
    if not len(indices):
        raise RuntimeError("No spatial site is valid at all eight hours")
    return indices


def nested_maxmin_selections(
    asset: core.DayAsset,
    eligible: np.ndarray,
    counts: tuple[int, ...],
) -> dict[int, np.ndarray]:
    grid = np.asarray(asset.grid_coords, dtype=np.float64)
    lon_lat = np.column_stack([grid[eligible, 1], grid[eligible, 0]])
    order = zero_based_order(orderings.maxmin_cpp(np.ascontiguousarray(lon_lat)), len(eligible))
    if max(counts) > len(order):
        raise RuntimeError(f"Requested {max(counts)} sites but only {len(order)} are eligible")
    return {count: eligible[order[:count]].copy() for count in counts}


def quasi_regular_targets(coords: np.ndarray, count: int) -> np.ndarray:
    """Create exactly count nearly equally spaced targets with the domain aspect ratio."""
    lat_min, lon_min = np.min(coords, axis=0)
    lat_max, lon_max = np.max(coords, axis=0)
    lat_span = float(lat_max - lat_min)
    lon_span = float(lon_max - lon_min)
    if min(lat_span, lon_span) <= 0.0:
        raise RuntimeError("Degenerate coordinate range")
    n_lat = max(2, int(round(math.sqrt(count * lat_span / lon_span))))
    base_lon = count // n_lat
    extra = count % n_lat
    if base_lon < 2:
        raise RuntimeError("Too few longitude targets for a quasi-regular lattice")
    extra_rows = set(
        int(value)
        for value in np.rint(np.linspace(0, n_lat - 1, extra)).astype(int)
    ) if extra else set()
    latitudes = np.linspace(lat_min, lat_max, n_lat)
    rows = []
    for row_index, latitude in enumerate(latitudes):
        n_lon = base_lon + int(row_index in extra_rows)
        longitudes = np.linspace(lon_min, lon_max, n_lon)
        rows.append(
            np.column_stack(
                [np.full(n_lon, latitude, dtype=np.float64), longitudes]
            )
        )
    targets = np.vstack(rows)
    if len(targets) != count:
        raise RuntimeError(f"Internal regular-target count error: {len(targets)} != {count}")
    return targets


def snap_unique_targets(
    eligible: np.ndarray,
    eligible_coords: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tree = scipy.spatial.cKDTree(eligible_coords)
    k = min(64, len(eligible))
    distances, candidates = tree.query(targets, k=k)
    if candidates.ndim == 1:
        candidates = candidates[:, None]
        distances = distances[:, None]
    used: set[int] = set()
    selected_local = np.empty(len(targets), dtype=np.int64)
    selected_distance = np.empty(len(targets), dtype=np.float64)
    # Assign the most constrained targets first: a large nearest-site distance
    # is a boundary/missingness target with fewer attractive alternatives.
    target_order = np.argsort(distances[:, 0])[::-1]
    for target_index in target_order:
        choice = None
        choice_distance = None
        for candidate, distance in zip(candidates[target_index], distances[target_index]):
            candidate_int = int(candidate)
            if candidate_int not in used:
                choice = candidate_int
                choice_distance = float(distance)
                break
        if choice is None:
            unused = np.asarray(
                [index for index in range(len(eligible)) if index not in used],
                dtype=np.int64,
            )
            delta = eligible_coords[unused] - targets[target_index]
            local = int(np.argmin(np.sum(delta * delta, axis=1)))
            choice = int(unused[local])
            choice_distance = float(np.linalg.norm(delta[local]))
        selected_local[target_index] = choice
        selected_distance[target_index] = choice_distance
        used.add(choice)
    selected = eligible[selected_local]
    if len(np.unique(selected)) != len(targets):
        raise RuntimeError("Regular target snapping produced duplicate sites")
    return selected, selected_distance


def regular_selections(
    asset: core.DayAsset,
    eligible: np.ndarray,
    counts: tuple[int, ...],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], dict[int, np.ndarray]]:
    grid = np.asarray(asset.grid_coords, dtype=np.float64)
    eligible_coords = grid[eligible, :2]
    selections: dict[int, np.ndarray] = {}
    targets_by_count: dict[int, np.ndarray] = {}
    snap_distances: dict[int, np.ndarray] = {}
    for count in counts:
        targets = quasi_regular_targets(eligible_coords, count)
        selected, distances = snap_unique_targets(eligible, eligible_coords, targets)
        selections[count] = selected
        targets_by_count[count] = targets
        snap_distances[count] = distances
    return selections, targets_by_count, snap_distances


def full_latitude_mean(asset: core.DayAsset) -> float:
    values = [
        tensor.to(device="cpu", dtype=torch.float32)
        for tensor in asset.source_map.values()
    ]
    stacked = torch.cat(values, dim=0)
    valid = torch.isfinite(stacked[:, 2])
    return float(stacked[valid, 0].mean().item())


def selected_data(
    asset: core.DayAsset,
    selected_indices: np.ndarray,
) -> np.ndarray:
    chunks = [
        tensor.detach().cpu().numpy()[selected_indices]
        for tensor in asset.source_map.values()
    ]
    result = np.ascontiguousarray(np.concatenate(chunks, axis=0), dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise RuntimeError("Non-finite value survived common-valid selection")
    return result


def fitted_design_and_response(
    selected: np.ndarray,
    latitude_mean: float,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack(
        [
            np.ones(len(selected), dtype=np.float64),
            selected[:, 0] - float(latitude_mean),
            selected[:, 4:11],
        ]
    )
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise RuntimeError("Selected fitted-mean design is rank deficient")
    return np.ascontiguousarray(design), np.ascontiguousarray(selected[:, 2])


def spatial_laplacian(
    coordinates: np.ndarray,
    neighbors: int,
) -> scipy.sparse.csr_matrix:
    n = len(coordinates)
    k = min(int(neighbors), n - 1)
    if k < 1:
        raise ValueError("At least two spatial sites are needed")
    tree = scipy.spatial.cKDTree(coordinates)
    distances, indices = tree.query(coordinates, k=k + 1)
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    cols = indices[:, 1:].reshape(-1).astype(np.int64)
    distance = distances[:, 1:].reshape(-1)
    positive = distance[distance > 0.0]
    floor = max(float(np.min(positive)) * 1e-6, np.finfo(np.float64).eps)
    weights = 1.0 / (float(k) * np.maximum(distance, floor) ** 2)
    adjacency = scipy.sparse.coo_matrix(
        (weights, (rows, cols)), shape=(n, n), dtype=np.float64
    ).tocsr()
    adjacency = adjacency.maximum(adjacency.T)
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    return (scipy.sparse.diags(degree) - adjacency).tocsr()


def geometry_summary(
    all_eligible_coords: np.ndarray,
    selected_coords: np.ndarray,
    design: str,
    count: int,
    snap_distances: np.ndarray | None,
) -> dict[str, Any]:
    selected_tree = scipy.spatial.cKDTree(selected_coords)
    nearest = selected_tree.query(selected_coords, k=2)[0][:, 1]
    coverage = selected_tree.query(all_eligible_coords, k=1)[0]
    return {
        "design": design,
        "n_spatial": int(count),
        "nearest_neighbor_min_deg": float(np.min(nearest)),
        "nearest_neighbor_median_deg": float(np.median(nearest)),
        "nearest_neighbor_p90_deg": float(np.quantile(nearest, 0.90)),
        "nyquist_wavelength_proxy_deg": float(2.0 * np.median(nearest)),
        "eligible_coverage_mean_deg": float(np.mean(coverage)),
        "eligible_coverage_p95_deg": float(np.quantile(coverage, 0.95)),
        "eligible_coverage_max_deg": float(np.max(coverage)),
        "regular_snap_mean_deg": (
            np.nan if snap_distances is None else float(np.mean(snap_distances))
        ),
        "regular_snap_max_deg": (
            np.nan if snap_distances is None else float(np.max(snap_distances))
        ),
    }


def full_eigen_with_roughness(
    selected: np.ndarray,
    design_matrix: np.ndarray,
    response: np.ndarray,
    spatial_graph: scipy.sparse.csr_matrix,
    fitted: dict[str, float],
    beta: np.ndarray,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    n_spatial = int(len(selected) // 8)
    n_total = int(len(selected))
    started = time.perf_counter()
    covariance = reference.fitted_covariance(
        selected, fitted, float(args.smooth), float(args.cov_jitter)
    )
    covariance_seconds = time.perf_counter() - started
    started = time.perf_counter()
    covariance = np.asarray(0.5 * (covariance + covariance.T), order="F")
    eigenvalues, eigenvectors = scipy.linalg.eigh(
        covariance,
        lower=True,
        overwrite_a=True,
        check_finite=False,
        driver="evd",
    )
    eigen_seconds = time.perf_counter() - started
    del covariance
    if not np.all(np.isfinite(eigenvalues)) or float(eigenvalues[0]) <= 0.0:
        raise RuntimeError(f"Covariance eigendecomposition is not SPD: {eigenvalues[0]}")
    eigenvalues = np.ascontiguousarray(eigenvalues[::-1])
    eigenvectors = np.ascontiguousarray(eigenvectors[:, ::-1])
    residual = response - design_matrix @ np.asarray(beta, dtype=np.float64)
    projections = eigenvectors.T @ residual
    scores = projections / np.sqrt(eigenvalues)
    energy = np.square(scores)
    cumulative = np.cumsum(energy)

    spatial_roughness = np.zeros(n_total, dtype=np.float64)
    spatial_started = time.perf_counter()
    for hour_index in range(8):
        block = eigenvectors[
            hour_index * n_spatial : (hour_index + 1) * n_spatial, :
        ]
        spatial_roughness += np.sum(block * (spatial_graph @ block), axis=0)
    spatial_seconds = time.perf_counter() - spatial_started
    temporal_roughness = np.zeros(n_total, dtype=np.float64)
    for hour_index in range(7):
        left = eigenvectors[
            hour_index * n_spatial : (hour_index + 1) * n_spatial, :
        ]
        right = eigenvectors[
            (hour_index + 1) * n_spatial : (hour_index + 2) * n_spatial, :
        ]
        temporal_roughness += np.sum(np.square(right - left), axis=0)
    spatial_roughness = np.maximum(spatial_roughness, 0.0)
    spatial_wavenumber = np.sqrt(spatial_roughness)
    spatial_wavelength = np.divide(
        2.0 * np.pi,
        spatial_wavenumber,
        out=np.full_like(spatial_wavenumber, np.inf),
        where=spatial_wavenumber > 0.0,
    )
    fraction = np.arange(1, n_total + 1, dtype=np.float64) / n_total
    mode_frame = pd.DataFrame(
        {
            "mode_index": np.arange(1, n_total + 1, dtype=np.int64),
            "mode_fraction": fraction,
            "covariance_eigenvalue": eigenvalues,
            "whitened_residual_energy": energy,
            "cumulative_energy_per_observation": cumulative / n_total,
            "spatial_roughness_per_degree2": spatial_roughness,
            "spatial_wavenumber_proxy_per_degree": spatial_wavenumber,
            "spatial_wavelength_proxy_deg": spatial_wavelength,
            "temporal_roughness_per_hour2": temporal_roughness,
        }
    )
    d_statistic = float(
        np.max(np.abs(cumulative - np.arange(1, n_total + 1)))
        / math.sqrt(2.0 * n_total)
    )
    summary = {
        "n_spatial": n_spatial,
        "n_total": n_total,
        "mean_y2": float(cumulative[-1] / n_total),
        "D": d_statistic,
        "min_covariance_eigenvalue": float(eigenvalues[-1]),
        "max_covariance_eigenvalue": float(eigenvalues[0]),
        "covariance_seconds": covariance_seconds,
        "eigendecomposition_seconds": eigen_seconds,
        "spatial_roughness_seconds": spatial_seconds,
        "spatial_wavenumber_median": float(np.median(spatial_wavenumber)),
        "spatial_wavenumber_p95": float(np.quantile(spatial_wavenumber, 0.95)),
        "spatial_wavenumber_max": float(np.max(spatial_wavenumber)),
        "temporal_roughness_median": float(np.median(temporal_roughness)),
        "temporal_roughness_p95": float(np.quantile(temporal_roughness, 0.95)),
    }
    del eigenvalues, eigenvectors, residual, projections, scores
    gc.collect()
    return mode_frame, summary


def cache_paths(output_root: Path, design: str, count: int, method: str) -> dict[str, Path]:
    root = output_root / CACHE_DIR_NAME / f"{design}_{count}" / method
    return {
        "root": root,
        "modes": root / "mode_diagnostics.csv",
        "summary": root / "summary.json",
        "complete": root / "COMPLETE",
    }


def load_cached(
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, dict[str, Any]] | None:
    if not (
        paths["complete"].is_file()
        and paths["modes"].is_file()
        and paths["summary"].is_file()
    ):
        return None
    return (
        pd.read_csv(paths["modes"]),
        json.loads(paths["summary"].read_text(encoding="utf-8")),
    )


def run_one_eigensystem(
    spec: dict[str, Any],
    design_name: str,
    count: int,
    method: str,
    selected_rows: np.ndarray,
    design_matrix: np.ndarray,
    response: np.ndarray,
    spatial_graph: scipy.sparse.csr_matrix,
    fit_row: pd.Series,
    geometry: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = cache_paths(args.output_root, design_name, count, method)
    cached = load_cached(paths)
    if cached is not None:
        print(f"  reuse {design_name} n={count} {method}", flush=True)
        return cached
    print(
        f"  full eigen {design_name} n={count} ({count * 8:,}) {method}",
        flush=True,
    )
    modes, summary = full_eigen_with_roughness(
        selected_rows,
        design_matrix,
        response,
        spatial_graph,
        reference.fitted_parameters(fit_row),
        np.asarray(fit_row["gls_beta"], dtype=np.float64),
        args,
    )
    metadata = {
        "dataset_id": spec["dataset_id"],
        "date": spec["date"],
        "year": spec["year"],
        "design": design_name,
        "n_spatial": int(count),
        "method": method,
    }
    for column, value in reversed(tuple(metadata.items())):
        modes.insert(0, column, value)
    summary = {
        **metadata,
        "native_nll_per_observation": float(fit_row["native_nll_per_observation"]),
        **geometry,
        **summary,
    }
    paths["root"].mkdir(parents=True, exist_ok=True)
    temporary = paths["modes"].with_suffix(".csv.tmp")
    modes.to_csv(temporary, index=False)
    temporary.replace(paths["modes"])
    reference.write_json(paths["summary"], summary)
    paths["complete"].write_text("complete\n", encoding="utf-8")
    return modes, summary


def binned_frequency_curve(modes: pd.DataFrame, bins: int = 80) -> pd.DataFrame:
    work = modes[["mode_fraction", "spatial_wavenumber_proxy_per_degree"]].copy()
    work["bin"] = np.minimum(
        (work["mode_fraction"].to_numpy(dtype=float) * bins).astype(int), bins - 1
    )
    return work.groupby("bin", as_index=False).agg(
        mode_fraction=("mode_fraction", "mean"),
        median_spatial_wavenumber=("spatial_wavenumber_proxy_per_degree", "median"),
    )


def resampled_cumulative(modes: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    return np.interp(
        grid,
        modes["mode_fraction"].to_numpy(dtype=float),
        modes["cumulative_energy_per_observation"].to_numpy(dtype=float),
    )


def build_change_table(
    all_modes: pd.DataFrame,
    metrics: pd.DataFrame,
    counts: tuple[int, ...],
) -> pd.DataFrame:
    grid = np.linspace(0.001, 1.0, 1000)
    pairs = list(zip(counts[:-1], counts[1:]))
    if (counts[0], counts[-1]) not in pairs:
        pairs.append((counts[0], counts[-1]))
    rows = []
    for design in DESIGNS:
        for method in METHODS:
            for left_count, right_count in pairs:
                left = all_modes[
                    all_modes["design"].eq(design)
                    & all_modes["method"].eq(method)
                    & all_modes["n_spatial"].eq(left_count)
                ]
                right = all_modes[
                    all_modes["design"].eq(design)
                    & all_modes["method"].eq(method)
                    & all_modes["n_spatial"].eq(right_count)
                ]
                difference = resampled_cumulative(left, grid) - resampled_cumulative(right, grid)
                left_metric = metrics[
                    metrics["design"].eq(design)
                    & metrics["method"].eq(method)
                    & metrics["n_spatial"].eq(left_count)
                ].iloc[0]
                right_metric = metrics[
                    metrics["design"].eq(design)
                    & metrics["method"].eq(method)
                    & metrics["n_spatial"].eq(right_count)
                ].iloc[0]
                rows.append(
                    {
                        "design": design,
                        "method": method,
                        "from_n_spatial": left_count,
                        "to_n_spatial": right_count,
                        "cumulative_curve_rmse": float(np.sqrt(np.mean(difference**2))),
                        "cumulative_curve_max_abs_change": float(np.max(np.abs(difference))),
                        "D_change": float(right_metric["D"] - left_metric["D"]),
                        "mean_y2_change": float(
                            right_metric["mean_y2"] - left_metric["mean_y2"]
                        ),
                        "spatial_wavenumber_p95_ratio": float(
                            right_metric["spatial_wavenumber_p95"]
                            / left_metric["spatial_wavenumber_p95"]
                        ),
                        "nyquist_proxy_ratio": float(
                            right_metric["nyquist_wavelength_proxy_deg"]
                            / left_metric["nyquist_wavelength_proxy_deg"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def plot_convergence(
    modes: pd.DataFrame,
    metrics: pd.DataFrame,
    counts: tuple[int, ...],
    date: str,
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(23, 10), constrained_layout=True)
    count_alpha = {
        count: 0.45 + 0.5 * index / max(len(counts) - 1, 1)
        for index, count in enumerate(counts)
    }
    for row_index, design in enumerate(DESIGNS):
        design_metrics = metrics[metrics["design"].eq(design)]
        for method in METHODS:
            for count in counts:
                subset = modes[
                    modes["design"].eq(design)
                    & modes["method"].eq(method)
                    & modes["n_spatial"].eq(count)
                ]
                label = f"{reference.LABELS[method]}, n={count}"
                axes[row_index, 0].plot(
                    subset["mode_fraction"],
                    subset["cumulative_energy_per_observation"],
                    color=COLORS[method],
                    linestyle=LINESTYLES.get(count, "-"),
                    alpha=count_alpha[count],
                    linewidth=1.6,
                    label=label,
                )
                frequency = binned_frequency_curve(subset)
                axes[row_index, 3].plot(
                    frequency["mode_fraction"],
                    frequency["median_spatial_wavenumber"],
                    color=COLORS[method],
                    linestyle=LINESTYLES.get(count, "-"),
                    alpha=count_alpha[count],
                    linewidth=1.6,
                    label=label,
                )
            method_metrics = design_metrics[design_metrics["method"].eq(method)].sort_values(
                "n_spatial"
            )
            axes[row_index, 1].plot(
                method_metrics["n_spatial"],
                method_metrics["D"],
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=1.8,
                label=reference.LABELS[method],
            )
            axes[row_index, 2].plot(
                method_metrics["n_spatial"],
                method_metrics["mean_y2"],
                color=COLORS[method],
                marker=MARKERS[method],
                linewidth=1.8,
                label=reference.LABELS[method],
            )
        axes[row_index, 0].plot([0, 1], [0, 1], color="0.3", linestyle="--", linewidth=1.0)
        axes[row_index, 0].set(
            xlabel="covariance-mode fraction",
            ylabel="cumulative whitened energy / observation",
            title=f"{design}: cumulative diagnostic",
        )
        axes[row_index, 0].legend(fontsize=7, ncol=2)
        axes[row_index, 1].set(
            xticks=counts,
            xlabel="common spatial sites per hour",
            ylabel="D",
            title=f"{design}: D convergence",
        )
        axes[row_index, 2].axhline(1.0, color="0.3", linestyle="--", linewidth=1.0)
        axes[row_index, 2].set(
            xticks=counts,
            xlabel="common spatial sites per hour",
            ylabel="mean whitened residual energy",
            title=f"{design}: mean Y-squared",
        )
        axes[row_index, 3].set_yscale("log")
        axes[row_index, 3].set(
            xlabel="covariance-mode fraction",
            ylabel="median spatial wavenumber proxy (1/degree)",
            title=f"{design}: physical roughness of eigenvectors",
        )
        for column in range(4):
            axes[row_index, column].grid(alpha=0.20)
    fig.suptitle(
        f"Real {date}: fixed fitted parameters, 400 -> 600 -> 900 sampling convergence",
        fontsize=15,
    )
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_geometry(metrics: pd.DataFrame, counts: tuple[int, ...], date: str, path: Path) -> None:
    geometry = metrics.drop_duplicates(["design", "n_spatial"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    definitions = (
        ("nearest_neighbor_median_deg", "median nearest-neighbor distance (degree)"),
        ("nyquist_wavelength_proxy_deg", "minimum wavelength proxy, 2 x median NN (degree)"),
        ("eligible_coverage_p95_deg", "95% eligible-site coverage distance (degree)"),
    )
    design_colors = {"maxmin": "#2A9D8F", "regular": "#8E5EA2"}
    for axis, (column, ylabel) in zip(axes, definitions):
        for design in DESIGNS:
            subset = geometry[geometry["design"].eq(design)].sort_values("n_spatial")
            axis.plot(
                subset["n_spatial"],
                subset[column],
                marker="o",
                linewidth=1.8,
                color=design_colors[design],
                label=design,
            )
        axis.set(xticks=counts, xlabel="common spatial sites per hour", ylabel=ylabel)
        axis.grid(alpha=0.20)
        axis.legend(fontsize=8)
    fig.suptitle(f"Real {date}: sampling geometry and resolvable-scale proxies")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_sampling_layouts(
    manifest: pd.DataFrame,
    counts: tuple[int, ...],
    date: str,
    path: Path,
) -> None:
    fig, axes = plt.subplots(
        len(DESIGNS),
        len(counts),
        figsize=(4.7 * len(counts), 4.2 * len(DESIGNS)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    design_colors = {"maxmin": "#2A9D8F", "regular": "#8E5EA2"}
    for row_index, design in enumerate(DESIGNS):
        for column_index, count in enumerate(counts):
            axis = axes[row_index, column_index]
            subset = manifest[
                manifest["design"].eq(design)
                & manifest["n_spatial"].eq(count)
            ]
            axis.scatter(
                subset["longitude"],
                subset["latitude"],
                s=7,
                alpha=0.72,
                color=design_colors[design],
                linewidths=0,
            )
            axis.set_title(f"{design}, n={count}")
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.15)
            if row_index == len(DESIGNS) - 1:
                axis.set_xlabel("longitude")
            if column_index == 0:
                axis.set_ylabel("latitude")
    fig.suptitle(f"Real {date}: common-valid spatial sampling layouts", fontsize=14)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def write_config(
    args: argparse.Namespace,
    spec: dict[str, Any],
    fits: pd.DataFrame,
) -> None:
    path = args.output_root / "run_config.json"
    fit_signature = {}
    for method in METHODS:
        row = fits[fits["method"].eq(method)].iloc[0]
        fit_signature[method] = {
            **{
                parameter: float(row[parameter])
                for parameter in reference.PARAMETERS
            },
            "gls_beta": [float(value) for value in row["gls_beta"]],
            "native_nll_per_observation": float(row["native_nll_per_observation"]),
        }
    signature = {
        "date": spec["date"],
        "counts": list(args.counts),
        "designs": list(DESIGNS),
        "smooth": args.smooth,
        "cov_jitter": args.cov_jitter,
        "graph_neighbors": args.graph_neighbors,
        "fit_signature": fit_signature,
    }
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous.get("configuration_signature") != signature:
            raise ValueError(f"{path} has different settings; use a new --output-root")
        created = previous["created"]
    else:
        created = datetime.now().isoformat(timespec="seconds")
    reference.write_json(
        path,
        {
            "created": created,
            "last_started": datetime.now().isoformat(timespec="seconds"),
            "host": socket.gethostname(),
            "configuration_signature": signature,
            "interpretation": (
                "The fitted covariance parameters and full-data GLS beta are fixed. "
                "Changes therefore measure sampling/eigensystem sensitivity, not refitting."
            ),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2024-07-03")
    parser.add_argument("--counts", type=parse_counts, default=parse_counts("400,600,900"))
    parser.add_argument(
        "--real-data-root", type=Path, default=Path("/home/jl2815/tco/data")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/home/jl2815/tco/exercise_output/summer/"
            "vecchia_20240703_sampling_convergence_full_eigen_lag643"
        ),
    )
    parser.add_argument(
        "--fit-checkpoint",
        type=Path,
        default=Path(
            "/home/jl2815/tco/exercise_output/summer/"
            "vecchia_real59_adapted_fixed_full_eigen_lag643_clean_v2/"
            "fit_checkpoint_full_precision.json"
        ),
    )
    parser.add_argument("--lat-range", default="-3,2")
    parser.add_argument("--lon-range", default="121,131")
    parser.add_argument("--smooth", type=float, default=0.5, choices=(0.5,))
    parser.add_argument("--target-chunk-size", type=int, default=256)
    parser.add_argument("--points-per-hour", type=int, default=400)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-steps", type=int, default=5)
    parser.add_argument("--lbfgs-eval", type=int, default=20)
    parser.add_argument("--lbfgs-history", type=int, default=40)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    parser.add_argument("--cov-jitter", type=float, default=1e-8)
    parser.add_argument("--graph-neighbors", type=int, default=6)
    parser.add_argument("--suppress-fit-prints", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.graph_neighbors < 2:
        raise ValueError("--graph-neighbors must be at least 2")
    args.output_root.mkdir(parents=True, exist_ok=True)
    spec = day_spec(args.date)
    records = seed_fits_from_checkpoint(args.fit_checkpoint, args.output_root, spec)
    asset, fits = reference.load_or_fit_day(spec, records, args)
    reference.persist_fit_results(records, args.output_root)
    write_config(args, spec, fits)
    eligible = common_valid_indices(asset)
    if max(args.counts) > len(eligible):
        raise RuntimeError(
            f"Only {len(eligible)} sites are valid at all eight hours; "
            f"cannot select {max(args.counts)}"
        )
    grid = np.asarray(asset.grid_coords, dtype=np.float64)
    eligible_coords = grid[eligible, :2]
    maxmin = nested_maxmin_selections(asset, eligible, args.counts)
    regular, regular_targets, regular_snap = regular_selections(asset, eligible, args.counts)
    selections = {"maxmin": maxmin, "regular": regular}
    latitude_mean = full_latitude_mean(asset)
    all_modes: list[pd.DataFrame] = []
    all_summaries: list[dict[str, Any]] = []
    manifest_rows: list[pd.DataFrame] = []
    started = time.perf_counter()

    for design_name in DESIGNS:
        for count in args.counts:
            indices = selections[design_name][count]
            coordinates = grid[indices, :2]
            snap = regular_snap[count] if design_name == "regular" else None
            geometry = geometry_summary(
                eligible_coords, coordinates, design_name, count, snap
            )
            target = regular_targets[count] if design_name == "regular" else None
            manifest = pd.DataFrame(
                {
                    "design": design_name,
                    "n_spatial": count,
                    "selection_rank": np.arange(1, count + 1),
                    "original_grid_index": indices,
                    "latitude": coordinates[:, 0],
                    "longitude": coordinates[:, 1],
                    "target_latitude": np.nan if target is None else target[:, 0],
                    "target_longitude": np.nan if target is None else target[:, 1],
                    "snap_distance_deg": np.nan if snap is None else snap,
                }
            )
            manifest_rows.append(manifest)
            rows = selected_data(asset, indices)
            design_matrix, response = fitted_design_and_response(rows, latitude_mean)
            graph = spatial_laplacian(coordinates, args.graph_neighbors)
            for method in METHODS:
                fit_row = fits[fits["method"].eq(method)].iloc[0]
                modes, summary = run_one_eigensystem(
                    spec,
                    design_name,
                    count,
                    method,
                    rows,
                    design_matrix,
                    response,
                    graph,
                    fit_row,
                    geometry,
                    args,
                )
                all_modes.append(modes)
                all_summaries.append(summary)
            del rows, design_matrix, response, graph
            gc.collect()

    mode_frame = pd.concat(all_modes, ignore_index=True)
    metric_frame = pd.DataFrame(all_summaries).sort_values(
        ["design", "method", "n_spatial"]
    )
    manifest_frame = pd.concat(manifest_rows, ignore_index=True)
    change_frame = build_change_table(mode_frame, metric_frame, args.counts)
    mode_frame.to_csv(args.output_root / "all_mode_diagnostics.csv", index=False)
    metric_frame.to_csv(args.output_root / "sampling_convergence_metrics.csv", index=False)
    manifest_frame.to_csv(args.output_root / "sampling_selection_manifest.csv", index=False)
    change_frame.to_csv(args.output_root / "sampling_change_tracking.csv", index=False)
    plot_convergence(
        mode_frame,
        metric_frame,
        args.counts,
        args.date,
        args.output_root / "sampling_convergence_400_600_900.png",
    )
    plot_geometry(
        metric_frame,
        args.counts,
        args.date,
        args.output_root / "sampling_geometry_400_600_900.png",
    )
    plot_sampling_layouts(
        manifest_frame,
        args.counts,
        args.date,
        args.output_root / "sampling_layouts_400_600_900.png",
    )
    reference.write_json(
        args.output_root / "RUN_COMPLETE.json",
        {
            "completed": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": time.perf_counter() - started,
            "date": args.date,
            "counts": args.counts,
            "designs": DESIGNS,
            "methods": METHODS,
            "n_full_eigendecompositions": len(args.counts) * len(DESIGNS) * len(METHODS),
        },
    )
    print(metric_frame.to_string(index=False), flush=True)
    print("\nChange tracking", flush=True)
    print(change_frame.to_string(index=False), flush=True)
    print(f"\nComplete: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
