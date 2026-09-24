#!/usr/bin/env python3
"""Run an oracle rectangle-dictionary experiment on an exact comoving grid.

The existing five-day flow tube uses rounded grid shifts and therefore does
not contain identical physical moving-coordinate anchors at every hour.  This
script does not relabel those observations as exact rectangles.  Instead it
uses their immutable manifest for the known truth parameters and constructs a
separate, exact Cartesian product of moving anchors and times.

The target is the Sigma0-scaled intrinsic contrast

    w' (Sigma1 - SigmaM) w / (w' Sigma0 w),

where SigmaM has the truth's exact spatial and temporal margins and Sigma0 is
refit on this exact geometry with the known truth advection held fixed.  The
result is an oracle covariance experiment, not a real-data test or p-value.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy
import scipy.linalg
import scipy.optimize
from scipy.stats import qmc

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    try:
        import tomli as tomllib
    except ModuleNotFoundError as error:  # pragma: no cover
        raise ModuleNotFoundError(
            "Python 3.10 requires the optional 'tomli' package to read this experiment config"
        ) from error

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from analyze_mode_count_path import atomic_csv, atomic_json, covariance_parameters, sha256
from diagnostic_core import (
    CovarianceParameters,
    advected_separable_covariance,
    joint_matern_half_covariance,
    pairwise_lags,
    profiled_null_objective,
    solve_generalized_eigenproblem,
)
from explore_eigen_directions import load_coordinate_days
from rectangle_dictionary_core import (
    cluster_overlap,
    constrained_relative_residual,
    dictionary_svd,
    direction_retention,
    filter_variance_metrics,
    greedy_extreme_path,
    projected_gaussian_kl,
    recover_rectangle_coefficients,
    rectangle_matrix,
    rectangle_structure_errors,
    sigma0_metric_projection,
    solve_constrained_contrast,
    standardize_dictionary,
    summarize_contrast_solution,
)


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "rectangle_dictionary_exact_grid.toml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    return parser


def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    config["_path"] = path
    return config


def validate_config(config: dict[str, Any]) -> None:
    geometry = config["geometry"]
    dictionary = config["dictionary"]
    monte_carlo = config["monte_carlo"]
    latitude_count = int(geometry["latitude_count"])
    longitude_count = int(geometry["longitude_count"])
    time_count = int(geometry["hours"])
    if min(latitude_count, longitude_count, time_count) < 2:
        raise ValueError("geometry needs at least two points along each configured dimension")
    sizes = tuple(int(value) for value in dictionary["greedy_sizes"])
    if sizes != (1, 2, 4, 8):
        raise ValueError("this predeclared experiment requires greedy_sizes = [1, 2, 4, 8]")
    if int(monte_carlo["replicates"]) < 1000 or int(monte_carlo["batch_size"]) < 1:
        raise ValueError("Monte Carlo needs at least 1000 replicates and a positive batch size")
    anchor_count = latitude_count * longitude_count
    dimension = anchor_count * time_count
    rectangle_count = math_combination(anchor_count, 2) * math_combination(time_count, 2)
    matrix_megabytes = dimension * rectangle_count * 8 / 1024**2
    maximum = float(dictionary["max_single_dense_matrix_megabytes"])
    if matrix_megabytes > maximum:
        raise MemoryError(
            f"one dense dictionary matrix would require {matrix_megabytes:.1f} MiB, "
            f"above the configured {maximum:.1f} MiB pilot guard"
        )


def math_combination(n: int, k: int) -> int:
    if k != 2:
        raise ValueError("this helper is intentionally limited to pairs")
    return n * (n - 1) // 2


def git_state() -> dict[str, Any]:
    """Return the exact tracked Git state without making the run depend on Git."""

    def run(*arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *arguments],
            cwd=HERE,
            check=False,
            capture_output=True,
            text=True,
        )

    commit = run("rev-parse", "HEAD")
    status = run("status", "--porcelain")
    if commit.returncode != 0 or status.returncode != 0:
        return {
            "available": False,
            "error": (commit.stderr or status.stderr).strip(),
        }
    tracked_changes = [line for line in status.stdout.splitlines() if line]
    return {
        "available": True,
        "commit": commit.stdout.strip(),
        "tracked_worktree_dirty": bool(tracked_changes),
        "tracked_changes": tracked_changes,
    }


def runtime_environment() -> dict[str, Any]:
    """Capture package, BLAS, platform, and thread settings used by this run."""

    blas_stream = io.StringIO()
    with contextlib.redirect_stdout(blas_stream):
        np.__config__.show()
    thread_keys = (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "thread_environment": {key: os.environ.get(key) for key in thread_keys},
        "numpy_blas_configuration": blas_stream.getvalue().strip(),
    }


def output_inventory(output_dir: Path) -> list[dict[str, Any]]:
    """Hash every completed output except the self-referential manifest."""

    manifest_name = "experiment_manifest.json"
    rows = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == manifest_name:
            continue
        rows.append(
            {
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return rows


def resolve_from_here(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (HERE / path).resolve()


def exact_comoving_coordinates(
    truth: CovarianceParameters,
    geometry_config: dict[str, Any],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Create exact ``source = moving_anchor + v0 * time`` coordinates."""

    latitude_values = np.linspace(
        float(geometry_config["standardized_latitude_min"]),
        float(geometry_config["standardized_latitude_max"]),
        int(geometry_config["latitude_count"]),
    )
    longitude_values = np.linspace(
        float(geometry_config["standardized_longitude_min"]),
        float(geometry_config["standardized_longitude_max"]),
        int(geometry_config["longitude_count"]),
    )
    standardized_latitude, standardized_longitude = np.meshgrid(
        latitude_values,
        longitude_values,
        indexing="ij",
    )
    moving_latitude = standardized_latitude.reshape(-1) * truth.range_lat
    moving_longitude = standardized_longitude.reshape(-1) * truth.range_lon
    anchor_count = len(moving_latitude)
    rows: list[dict[str, Any]] = []
    coordinate_parts = []
    for time_index in range(int(geometry_config["hours"])):
        source_latitude = moving_latitude + truth.advec_lat * time_index
        source_longitude = moving_longitude + truth.advec_lon * time_index
        coordinate_parts.append(
            np.column_stack(
                (
                    source_latitude,
                    source_longitude,
                    np.full(anchor_count, time_index, dtype=np.float64),
                )
            )
        )
        for anchor_index in range(anchor_count):
            rows.append(
                {
                    "time_index": time_index,
                    "anchor_index": anchor_index,
                    "latitude_grid_index": anchor_index // len(longitude_values),
                    "longitude_grid_index": anchor_index % len(longitude_values),
                    "moving_latitude": moving_latitude[anchor_index],
                    "moving_longitude": moving_longitude[anchor_index],
                    "standardized_moving_latitude": (
                        moving_latitude[anchor_index] / truth.range_lat
                    ),
                    "standardized_moving_longitude": (
                        moving_longitude[anchor_index] / truth.range_lon
                    ),
                    "source_latitude": source_latitude[anchor_index],
                    "source_longitude": source_longitude[anchor_index],
                }
            )
    return np.vstack(coordinate_parts), pd.DataFrame(rows)


def fit_fixed_advection_null(
    geometry: Any,
    true_covariance: np.ndarray,
    truth: CovarianceParameters,
    initial: CovarianceParameters,
    config: dict[str, Any],
    *,
    numerical_jitter_ratio: float,
) -> tuple[CovarianceParameters, pd.DataFrame, dict[str, Any]]:
    """KL-fit variance/ranges while holding the known advection exactly fixed."""

    bounds = np.log(
        np.asarray(
            [
                config["range_lat_bounds"],
                config["range_lon_bounds"],
                config["range_time_bounds"],
            ],
            dtype=np.float64,
        )
    )
    start_count = int(config["starts"])
    first = np.log([initial.range_lat, initial.range_lon, initial.range_time])
    starts = [np.clip(first, bounds[:, 0], bounds[:, 1])]
    if start_count >= 2:
        starts.append(np.log([truth.range_lat, truth.range_lon, truth.range_time]))
    remaining = start_count - len(starts)
    if remaining > 0:
        sampler = qmc.LatinHypercube(d=3, seed=int(config["random_seed"]))
        unit = sampler.random(remaining)
        starts.extend(bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0]))

    def evaluate(raw_ranges: np.ndarray) -> tuple[float, float]:
        raw = np.concatenate(
            (
                np.asarray(raw_ranges, dtype=np.float64),
                [truth.advec_lat, truth.advec_lon],
            )
        )
        return profiled_null_objective(
            raw,
            [geometry],
            [true_covariance],
            numerical_jitter_ratio=numerical_jitter_ratio,
        )

    def objective(raw_ranges: np.ndarray) -> float:
        try:
            value, _ = evaluate(raw_ranges)
            return value if np.isfinite(value) else 1.0e100
        except (ValueError, FloatingPointError, scipy.linalg.LinAlgError):
            return 1.0e100

    rows = []
    raw_results = []
    for start_index, start in enumerate(starts):
        result = scipy.optimize.minimize(
            objective,
            np.asarray(start, dtype=np.float64),
            method="L-BFGS-B",
            bounds=[tuple(value) for value in bounds],
            options={
                "maxiter": int(config["max_iterations"]),
                "maxls": 40,
                "ftol": 1.0e-11,
                "gtol": 1.0e-7,
            },
        )
        value, variance = evaluate(result.x)
        ranges = np.exp(result.x)
        projected_gradient = np.asarray(result.jac, dtype=np.float64)
        rows.append(
            {
                "start_index": start_index,
                "objective_per_observation": value,
                "variance": variance,
                "range_lat": ranges[0],
                "range_lon": ranges[1],
                "range_time": ranges[2],
                "advec_lat_fixed": truth.advec_lat,
                "advec_lon_fixed": truth.advec_lon,
                "iterations": int(result.nit),
                "evaluations": int(result.nfev),
                "gradient_max_abs": float(np.max(np.abs(projected_gradient))),
                "converged": bool(result.success),
                "message": str(result.message),
            }
        )
        raw_results.append(np.asarray(result.x, dtype=np.float64))
    attempts = pd.DataFrame(rows)
    attempts["objective_delta_from_best"] = (
        attempts["objective_per_observation"] - attempts["objective_per_observation"].min()
    )
    best_position = int(attempts["objective_per_observation"].argmin())
    best = attempts.iloc[best_position]
    parameters = CovarianceParameters(
        variance=float(best["variance"]),
        range_lat=float(best["range_lat"]),
        range_lon=float(best["range_lon"]),
        range_time=float(best["range_time"]),
        advec_lat=truth.advec_lat,
        advec_lon=truth.advec_lon,
        nugget=0.0,
    )
    widths = bounds[:, 1] - bounds[:, 0]
    best_raw = raw_results[best_position]
    boundary_names = ("range_lat", "range_lon", "range_time")
    boundary = [
        name
        for name, value, interval, width in zip(boundary_names, best_raw, bounds, widths)
        if min(value - interval[0], interval[1] - value) <= 1.0e-4 * width
    ]
    summary = {
        "best_start_index": int(best["start_index"]),
        "objective_per_observation": float(best["objective_per_observation"]),
        "fixed_advection": [truth.advec_lat, truth.advec_lon],
        "boundary_parameters": boundary,
        "optimizer": "L-BFGS-B",
        "ftol": 1.0e-11,
        "gtol": 1.0e-7,
        "maxls": 40,
    }
    return parameters, attempts, summary


def covariance_margin_checks(
    true_covariance: np.ndarray,
    matched_covariance: np.ndarray,
    *,
    anchor_count: int,
    time_count: int,
) -> dict[str, float]:
    same_time_errors = []
    same_anchor_errors = []
    for first_time in range(time_count):
        first_slice = slice(first_time * anchor_count, (first_time + 1) * anchor_count)
        same_time_errors.append(
            np.max(
                np.abs(
                    true_covariance[first_slice, first_slice]
                    - matched_covariance[first_slice, first_slice]
                )
            )
        )
        for second_time in range(first_time + 1, time_count):
            first = first_time * anchor_count + np.arange(anchor_count)
            second = second_time * anchor_count + np.arange(anchor_count)
            same_anchor_errors.append(
                np.max(
                    np.abs(
                        true_covariance[np.ix_(first, second)]
                        - matched_covariance[np.ix_(first, second)]
                    ).diagonal()
                )
            )
    return {
        "maximum_pure_spatial_margin_error": float(max(same_time_errors)),
        "maximum_pure_temporal_margin_error": float(max(same_anchor_errors)),
    }


def build_rectangle_metadata(
    endpoints: np.ndarray,
    anchor_table: pd.DataFrame,
    truth: CovarianceParameters,
    null_variances: np.ndarray,
) -> pd.DataFrame:
    anchors = (
        anchor_table.loc[anchor_table["time_index"] == 0]
        .sort_values("anchor_index")
        .reset_index(drop=True)
    )
    rows = []
    for rectangle_index, ((p, q, k, ell), null_variance) in enumerate(
        zip(endpoints, null_variances)
    ):
        delta_lat = float(anchors.loc[q, "moving_latitude"] - anchors.loc[p, "moving_latitude"])
        delta_lon = float(anchors.loc[q, "moving_longitude"] - anchors.loc[p, "moving_longitude"])
        rows.append(
            {
                "rectangle_index": rectangle_index,
                "rectangle_id": f"s{p:03d}_{q:03d}_t{k:02d}_{ell:02d}",
                "spatial_endpoint_p": int(p),
                "spatial_endpoint_q": int(q),
                "time_endpoint_k": int(k),
                "time_endpoint_l": int(ell),
                "moving_displacement_lat": delta_lat,
                "moving_displacement_lon": delta_lon,
                "standardized_spatial_distance": float(
                    np.hypot(delta_lat / truth.range_lat, delta_lon / truth.range_lon)
                ),
                "temporal_lag": int(ell - k),
                "orientation_convention": "+p,k;-q,k;-p,l;+q,l",
                "null_variance_raw": float(null_variance),
            }
        )
    return pd.DataFrame(rows)


def current_atlas_span_audit(
    pilot_dir: Path,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Algebraic double-centering audit; not a physical rectangle claim."""

    point_path = pilot_dir / "selected_flow_tube_points.csv"
    design_dates = tuple(str(value) for value in manifest["split"]["design_dates"])
    heldout_dates = tuple(str(value) for value in manifest["split"]["heldout_dates"])
    all_days = load_coordinate_days(point_path, design_dates + heldout_dates)
    days = [day for day in all_days if day.date in design_dates]
    truth = covariance_parameters(manifest["truth"])
    fitted = covariance_parameters(manifest["null"]["fit"]["parameters"])
    jitter = float(manifest["numerics"]["numerical_jitter_ratio"])
    true_covariances = []
    null_covariances = []
    maximum_drift = 0.0
    exact_groups = 0
    exact_tolerance = 1.0e-12
    for day in days:
        geometry = pairwise_lags(day.coordinates)
        true_covariances.append(
            joint_matern_half_covariance(
                geometry,
                truth,
                numerical_jitter_ratio=jitter,
            )
        )
        null_covariances.append(
            advected_separable_covariance(
                geometry,
                fitted,
                numerical_jitter_ratio=jitter,
            )
        )
        values = day.coordinates.reshape(8, 100, 3)
        moving = np.empty((8, 100, 2), dtype=np.float64)
        moving[:, :, 0] = values[:, :, 0] - truth.advec_lat * values[:, :, 2]
        moving[:, :, 1] = values[:, :, 1] - truth.advec_lon * values[:, :, 2]
        deviations = moving - moving[0:1]
        norms = np.hypot(deviations[:, :, 0], deviations[:, :, 1])
        maximum_drift = max(maximum_drift, float(norms.max()))
        exact_groups += int(np.sum(np.max(np.abs(deviations), axis=(0, 2)) <= exact_tolerance))
    reference_true = np.mean(np.stack(true_covariances), axis=0)
    reference_null = np.mean(np.stack(null_covariances), axis=0)
    atlas = solve_generalized_eigenproblem(reference_true, reference_null)
    spatial_contrasts = scipy.linalg.null_space(np.ones((1, 100)))
    temporal_contrasts = scipy.linalg.null_space(np.ones((1, 8)))
    double_centered_basis = np.kron(temporal_contrasts, spatial_contrasts)
    selected_modes = (13, 14, 39, 40)
    selected = atlas.eigenvectors[:, np.asarray(selected_modes) - 1]
    projected, _ = sigma0_metric_projection(
        double_centered_basis,
        reference_null,
        selected,
    )
    retention = direction_retention(projected, selected, reference_null)
    mode_rows = []
    for position, mode in enumerate(selected_modes):
        weights = selected[:, position].reshape(8, 100)
        mode_rows.append(
            {
                "mode": mode,
                "eigenvalue": float(atlas.eigenvalues[mode - 1]),
                "sigma0_metric_retention": float(retention[position]),
                "maximum_absolute_per_anchor_temporal_sum": float(
                    np.max(np.abs(weights.sum(axis=0)))
                ),
                "maximum_absolute_per_time_spatial_sum": float(np.max(np.abs(weights.sum(axis=1)))),
                "physical_rectangle_interpretation_valid": False,
                "reason": "source-coordinate moving anchors drift within anchor_rank",
            }
        )
    cluster_rows = []
    for label, columns in (("modes_13_14", (0, 1)), ("modes_39_40", (2, 3))):
        overlap = cluster_overlap(
            double_centered_basis,
            selected[:, columns],
            reference_null,
        )
        squared = overlap.pop("squared_canonical_correlations")
        cluster_rows.append(
            {
                "cluster": label,
                **overlap,
                "squared_canonical_correlation_1": float(squared[0]),
                "squared_canonical_correlation_2": float(squared[1]),
                "physical_rectangle_interpretation_valid": False,
            }
        )
    audit = {
        "stacking_order": "time-major: index=time*100+anchor",
        "date_count": len(days),
        "anchor_time_groups": len(days) * 100,
        "exact_comoving_anchor_groups": exact_groups,
        "exact_comoving_tolerance": exact_tolerance,
        "maximum_unstandardized_moving_coordinate_drift": maximum_drift,
        "truth_advection": [truth.advec_lat, truth.advec_lon],
        "legacy_fitted_advection": [fitted.advec_lat, fitted.advec_lon],
        "current_fitted_null_uses_truth_advection": bool(
            fitted.advec_lat == truth.advec_lat and fitted.advec_lon == truth.advec_lon
        ),
        "interpretation": (
            "Algebraic double-centering span only; current source coordinates do not "
            "define fixed physical comoving rectangles."
        ),
    }
    return pd.DataFrame(mode_rows), pd.DataFrame(cluster_rows), audit


def filter_row(
    name: str,
    family: str,
    branch: str,
    weights: np.ndarray,
    true_covariance: np.ndarray,
    matched_covariance: np.ndarray,
    null_covariance: np.ndarray,
    *,
    constraint_basis: np.ndarray,
    anchor_count: int,
    time_count: int,
    objective_mu: float,
    rectangle_count: int | None,
) -> dict[str, Any]:
    temporal_error, spatial_error = rectangle_structure_errors(
        weights,
        anchor_count=anchor_count,
        time_count=time_count,
    )
    intrinsic_difference_global = true_covariance - matched_covariance
    return {
        "filter_name": name,
        "family": family,
        "branch": branch,
        "rectangle_count": rectangle_count,
        "objective_mu": float(objective_mu),
        "constrained_relative_residual": constrained_relative_residual(
            weights,
            objective_mu,
            constraint_basis,
            intrinsic_difference_global,
            null_covariance,
        ),
        **filter_variance_metrics(
            weights,
            true_covariance,
            matched_covariance,
            null_covariance,
        ),
        "maximum_per_anchor_temporal_sum_error": temporal_error,
        "maximum_per_time_spatial_sum_error": spatial_error,
    }


def simulation_moments(
    filters: dict[str, np.ndarray],
    covariances: dict[str, np.ndarray],
    *,
    replicates: int,
    batch_size: int,
    random_seed: int,
) -> pd.DataFrame:
    names = tuple(filters)
    matrix = np.column_stack([filters[name] for name in names])
    rows = []
    for covariance_index, (model, covariance) in enumerate(covariances.items()):
        factor = scipy.linalg.cholesky(covariance, lower=True, check_finite=False)
        rng = np.random.default_rng(int(random_seed) + covariance_index)
        sum_squared = np.zeros(len(names), dtype=np.float64)
        sum_fourth = np.zeros(len(names), dtype=np.float64)
        completed = 0
        while completed < replicates:
            count = min(batch_size, replicates - completed)
            draws = rng.standard_normal((count, len(covariance))) @ factor.T
            squared = np.square(draws @ matrix)
            sum_squared += squared.sum(axis=0)
            sum_fourth += np.square(squared).sum(axis=0)
            completed += count
        empirical_mean = sum_squared / replicates
        empirical_variance = sum_fourth / replicates - np.square(empirical_mean)
        theoretical_variance = np.einsum(
            "ik,ij,jk->k",
            matrix,
            covariance,
            matrix,
            optimize=True,
        )
        for index, name in enumerate(names):
            expected_mean = float(theoretical_variance[index])
            expected_variance = 2.0 * expected_mean**2
            rows.append(
                {
                    "model": model,
                    "filter_name": name,
                    "replicates": replicates,
                    "expected_squared_projection_mean": expected_mean,
                    "empirical_squared_projection_mean": float(empirical_mean[index]),
                    "relative_mean_error": float(
                        (empirical_mean[index] - expected_mean) / expected_mean
                    ),
                    "expected_squared_projection_variance": expected_variance,
                    "empirical_squared_projection_variance": float(empirical_variance[index]),
                    "relative_variance_error": float(
                        (empirical_variance[index] - expected_variance) / expected_variance
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_objective_comparison(frame: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = {"positive": "#b33b2e", "negative": "#3568a8"}
    for axis, branch in zip(axes, ("positive", "negative")):
        subset = frame.loc[frame["branch"] == branch]
        greedy = subset.loc[subset["family"] == "greedy"].sort_values("rectangle_count")
        axis.plot(
            greedy["rectangle_count"],
            greedy["objective_mu"],
            marker="o",
            color=colors[branch],
            label="greedy reoptimized",
        )
        for family, label, linestyle in (
            ("dense_dictionary", "dictionary dense oracle", "--"),
            ("full_space", "unrestricted intrinsic oracle", ":"),
        ):
            value = float(subset.loc[subset["family"] == family, "objective_mu"].iloc[0])
            axis.axhline(value, color="0.2", linestyle=linestyle, label=label)
        axis.set_xlabel("number of actual rectangles")
        axis.set_ylabel("Sigma0-scaled intrinsic difference mu")
        axis.set_title(f"{branch} branch")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_filter_maps(
    filters: dict[str, np.ndarray],
    geometry_config: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = int(geometry_config["latitude_count"])
    columns = int(geometry_config["longitude_count"])
    hours = int(geometry_config["hours"])
    for name, weights in filters.items():
        cube = weights.reshape(hours, rows, columns)
        limit = float(np.max(np.abs(cube)))
        fig, axes = plt.subplots(2, 4, figsize=(12, 6), constrained_layout=True)
        for time_index, axis in enumerate(axes.ravel()):
            image = axis.imshow(
                cube[time_index],
                origin="lower",
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
            )
            axis.set_title(f"hour {time_index}")
            axis.set_xlabel("longitude grid")
            axis.set_ylabel("latitude grid")
        fig.colorbar(image, ax=axes, shrink=0.78, label="unit-null-variance weight")
        fig.suptitle(name)
        fig.savefig(output_dir / f"{name}.png", dpi=180)
        plt.close(fig)


def plot_selected_rectangles(
    branch: str,
    selected: pd.DataFrame,
    anchor_table: pd.DataFrame,
    path: Path,
) -> None:
    anchors = (
        anchor_table.loc[anchor_table["time_index"] == 0]
        .sort_values("anchor_index")
        .reset_index(drop=True)
    )
    coefficients = selected["standardized_coefficient"].to_numpy(dtype=np.float64)
    limit = float(np.max(np.abs(coefficients)))
    norm = plt.Normalize(vmin=-limit, vmax=limit)
    cmap = plt.get_cmap("coolwarm")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    axes[0].scatter(
        anchors["standardized_moving_longitude"],
        anchors["standardized_moving_latitude"],
        c="0.8",
        s=20,
        zorder=1,
    )
    segments = []
    segment_colors = []
    for row in selected.itertuples(index=False):
        p = int(row.spatial_endpoint_p)
        q = int(row.spatial_endpoint_q)
        segments.append(
            [
                (
                    anchors.loc[p, "standardized_moving_longitude"],
                    anchors.loc[p, "standardized_moving_latitude"],
                ),
                (
                    anchors.loc[q, "standardized_moving_longitude"],
                    anchors.loc[q, "standardized_moving_latitude"],
                ),
            ]
        )
        segment_colors.append(cmap(norm(row.standardized_coefficient)))
    axes[0].add_collection(
        LineCollection(segments, colors=segment_colors, linewidths=2.5, alpha=0.85)
    )
    axes[0].set(
        xlabel="moving lon / range_lon",
        ylabel="moving lat / range_lat",
        title="actual spatial endpoint pairs",
    )
    axes[0].set_aspect("equal", adjustable="box")
    y = np.arange(len(selected))
    for position, row in enumerate(selected.itertuples(index=False)):
        axes[1].plot(
            [row.time_endpoint_k, row.time_endpoint_l],
            [position, position],
            color=cmap(norm(row.standardized_coefficient)),
            linewidth=4,
        )
        axes[1].scatter(
            [row.time_endpoint_k, row.time_endpoint_l],
            [position, position],
            color=cmap(norm(row.standardized_coefficient)),
            s=28,
        )
    axes[1].set_yticks(y, labels=selected["rectangle_id"])
    axes[1].set(
        xlabel="hour endpoint",
        ylabel="rectangle",
        title="actual temporal endpoint pairs",
    )
    axes[1].grid(axis="x", alpha=0.2)
    axes[2].barh(y, coefficients, color=[cmap(norm(value)) for value in coefficients])
    axes[2].axvline(0.0, color="black", linewidth=0.8)
    axes[2].set_yticks(y, labels=[])
    axes[2].set(
        xlabel="coefficient on unit-null-variance rectangle",
        title="reoptimized signed coefficients",
    )
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes,
        shrink=0.78,
        label="standardized rectangle coefficient",
    )
    fig.suptitle(f"{branch} greedy 8-rectangle combination")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_singular_values(values: np.ndarray, ranks: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    axis.semilogy(np.arange(1, len(values) + 1), values, marker=".", markersize=3)
    for row in ranks.itertuples(index=False):
        axis.axhline(
            row.absolute_tolerance,
            linestyle="--",
            linewidth=0.9,
            label=f"rtol={row.relative_tolerance:g}, rank={row.rank}",
        )
    axis.set(
        xlabel="singular-value index",
        ylabel="singular value",
        title="Rank separation of the standardized rectangle dictionary",
    )
    axis.grid(alpha=0.2)
    axis.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    comparison: pd.DataFrame,
    greedy_path: pd.DataFrame,
    selected_rectangles: pd.DataFrame,
    monte_carlo: pd.DataFrame,
    projected_kl: pd.DataFrame,
    fitted_null: CovarianceParameters,
    fit_attempts: pd.DataFrame,
    atlas_modes: pd.DataFrame,
    atlas_clusters: pd.DataFrame,
    geometry_audit: dict[str, Any],
    margin_checks: dict[str, float],
    rank_sensitivity: pd.DataFrame,
    *,
    dictionary_count: int,
    theoretical_rank: int,
    numerical_jitter_ratio: float,
    score_tie_tolerance: float,
) -> None:
    def value(family: str, branch: str, size: int | None = None) -> float:
        subset = comparison.loc[(comparison["family"] == family) & (comparison["branch"] == branch)]
        if size is not None:
            subset = subset.loc[subset["rectangle_count"] == size]
        return float(subset["objective_mu"].iloc[0])

    lines = [
        "# Exact-comoving rectangle-dictionary oracle",
        "",
        "This is a covariance-only oracle experiment.  It does not use response values and is not a calibrated real-data test.",
        "",
        "## Geometry decision",
        "",
        f"- Across the `{geometry_audit['date_count']}` design days, the existing flow tube has `{geometry_audit['exact_comoving_anchor_groups']}` comoving groups out of `{geometry_audit['anchor_time_groups']}` within tolerance `{geometry_audit['exact_comoving_tolerance']:.1e}`.",
        f"- Its maximum unstandardized moving-coordinate drift is `{geometry_audit['maximum_unstandardized_moving_coordinate_drift']:.6g}` degrees.",
        "- Therefore the existing modes are audited only against the algebraic double-centering span; they are not claimed to be physical rectangle combinations.",
        "- The main experiment uses a separate exact Cartesian product of moving anchors and hours.",
        "",
        "## Matched-margin check",
        "",
        f"- Maximum pure-spatial Sigma1/SigmaM discrepancy: `{margin_checks['maximum_pure_spatial_margin_error']:.3e}`.",
        f"- Maximum pure-temporal Sigma1/SigmaM discrepancy: `{margin_checks['maximum_pure_temporal_margin_error']:.3e}`.",
        "- All three models use the same known advection and zero statistical nugget in this experiment.",
        f"- A `{numerical_jitter_ratio:.1e}` diagonal jitter is used only for numerical linear algebra; it is not a fitted or statistical nugget.",
        "",
        "## Strong fitted null",
        "",
        f"- All `{len(fit_attempts)}` starts used the pilot optimizer tolerances; `{int(fit_attempts['converged'].sum())}` converged.",
        f"- Known advection was fixed at `({fitted_null.advec_lat:.6g}, {fitted_null.advec_lon:.6g})`.",
        f"- The KL-optimal fitted parameters are variance `{fitted_null.variance:.8g}`, latitude range `{fitted_null.range_lat:.8g}`, longitude range `{fitted_null.range_lon:.8g}`, and time range `{fitted_null.range_time:.8g}`.",
        f"- The largest absolute optimizer-reported Jacobian component across starts is `{fit_attempts['gradient_max_abs'].max():.3e}`; attempt-level objective deltas and convergence messages are retained in the fit CSV.",
        "",
        "## Dictionary and rank",
        "",
        f"- Actual rectangles: `{dictionary_count}`.",
        f"- Theoretical full rectangle-span rank `(m-1)(T-1)`: `{theoretical_rank}`.",
        f"- Numerical ranks over the requested tolerances: `{'; '.join(str(int(v)) for v in rank_sensitivity['rank'])}`.",
        "- Individual rectangles were scaled to unit null variance. All fitted-null cross-correlations were accounted for exactly, without forming a rectangle-by-rectangle Gram matrix.",
        "",
        "## Intrinsic objective comparison",
        "",
        "The reported `mu` is a Sigma0-scaled intrinsic variance difference, not a variance ratio and not KL.",
        "",
        "| branch | single | greedy-2 | greedy-4 | greedy-8 | dense dictionary | unrestricted full space |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for branch in ("positive", "negative"):
        lines.append(
            f"| {branch} | {value('greedy', branch, 1):.6g} | "
            f"{value('greedy', branch, 2):.6g} | {value('greedy', branch, 4):.6g} | "
            f"{value('greedy', branch, 8):.6g} | "
            f"{value('dense_dictionary', branch):.6g} | "
            f"{value('full_space', branch):.6g} |"
        )
    positive_eight = comparison.loc[(comparison["filter_name"] == "greedy_08_positive")].iloc[0]
    negative_one = comparison.loc[(comparison["filter_name"] == "greedy_01_negative")].iloc[0]
    negative_two = comparison.loc[(comparison["filter_name"] == "greedy_02_negative")].iloc[0]
    negative_eight = comparison.loc[(comparison["filter_name"] == "greedy_08_negative")].iloc[0]
    dense_positive = comparison.loc[comparison["filter_name"] == "dense_dictionary_positive"].iloc[
        0
    ]
    dense_negative = comparison.loc[comparison["filter_name"] == "dense_dictionary_negative"].iloc[
        0
    ]
    combination_summaries = {}
    for branch in ("positive", "negative"):
        subset = selected_rectangles.loc[
            (selected_rectangles["branch"] == branch)
            & (selected_rectangles["combination_size"] == 8)
        ]
        combination_summaries[branch] = {
            "distances": sorted(set(subset["standardized_spatial_distance"].round(6))),
            "lags": sorted(set(int(value) for value in subset["temporal_lag"])),
            "symmetry_ties": int(
                (
                    greedy_path.loc[
                        (greedy_path["branch"] == branch) & (greedy_path["size"] <= 8),
                        "winner_tie_count",
                    ]
                    > 1
                ).sum()
            ),
        }
    null_simulation = monte_carlo.loc[monte_carlo["model"] == "fitted_null"]
    maximum_mc_mean_error = float(monte_carlo["relative_mean_error"].abs().max())
    maximum_null_mean_error = float(null_simulation["relative_mean_error"].abs().max())
    maximum_null_variance_error = float(null_simulation["relative_variance_error"].abs().max())
    lines.extend(
        [
            "",
            "The greedy rows are nested forward selections with all coefficients reoptimized after every addition.  They are not globally optimal k-rectangle subsets.",
            "",
            f"- Eight rectangles retain `{positive_eight.fraction_of_dense_extreme_magnitude:.1%}` of the positive dense-oracle magnitude and `{negative_eight.fraction_of_dense_extreme_magnitude:.1%}` of the negative magnitude.",
            f"- The best single rectangle on the minimizing branch is still positive (`mu={negative_one.objective_mu:.6g}`); the first negative interaction appears only after two rectangles are combined (`mu={negative_two.objective_mu:.6g}`).  Within this finite oracle dictionary and geometry, off-diagonal cross-rectangle covariance terms are therefore necessary for the signed negative contrast.",
            f"- After fitted-null compensation, the dense positive filter has `rho0={dense_positive.rho_fitted:.4f}`, `g={dense_positive.g_fitted:.5f}` and the dense negative filter has `rho0={dense_negative.rho_fitted:.4f}`, `g={dense_negative.g_fitted:.5f}`.  These population-covariance contrasts remain nonzero after oracle KL refitting; this is not a significance or power result.",
            "",
            "## Intrinsic, compensation, and total effects",
            "",
            "| filter | intrinsic | compensation | total | rho0 | marginal g(rho0) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for filter_name in (
        "dense_dictionary_positive",
        "dense_dictionary_negative",
        "greedy_08_positive",
        "greedy_08_negative",
    ):
        row = comparison.loc[comparison["filter_name"] == filter_name].iloc[0]
        lines.append(
            f"| `{filter_name}` | {row.delta_intrinsic:.6g} | "
            f"{row.delta_compensation:.6g} | {row.delta_total:.6g} | "
            f"{row.rho_fitted:.6g} | {row.g_fitted:.6g} |"
        )
    lines.extend(
        [
            "",
            "The two signed filters in each family are also evaluated jointly; joint projected KL retains their covariance and is not the sum of independent rectangle scores.",
            "",
            "| filter pair | joint projected KL | sum of marginal g values |",
            "|---|---:|---:|",
        ]
    )
    for row in projected_kl.itertuples(index=False):
        lines.append(
            f"| `{row.filter_pair}` | {row.joint_projected_kl:.6g} | "
            f"{row.sum_of_marginal_g_fitted:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Eight-rectangle geometry",
            "",
            f"- Positive branch: standardized spatial distances `{combination_summaries['positive']['distances']}` and temporal lags `{combination_summaries['positive']['lags']}` hours.",
            f"- Negative branch: standardized spatial distances `{combination_summaries['negative']['distances']}` and temporal lags `{combination_summaries['negative']['lags']}` hours.",
            f"- More than one candidate lay within the predeclared score tolerance `{score_tie_tolerance:.1e}` at `{combination_summaries['positive']['symmetry_ties']}` positive and `{combination_summaries['negative']['symmetry_ties']}` negative greedy steps.",
            "- The saved path uses the smallest rectangle index within each tolerance tie. Later greedy selections and the reported k-rectangle objective are conditional on that deterministic branch; tied current-step scores do not imply identical future paths.",
            "- Endpoint diagrams and the coefficient table show actual rectangles.  Pairwise lag attribution from the earlier atlas was not reinterpreted as a rectangle coefficient.",
            "",
            "## Fixed-filter simulation check",
            "",
            f"- Under Sigma0, the maximum absolute relative error in the simulated squared-projection mean was `{maximum_null_mean_error:.2%}` and in its variance was `{maximum_null_variance_error:.2%}`.",
            f"- Across Sigma0, SigmaM, and Sigma1, the maximum absolute relative mean error was `{maximum_mc_mean_error:.2%}`.",
            "- Combining many rectangles still produces one Gaussian projection per filter; the simulation does not treat its atoms as independent replications.",
            "- This Monte Carlo checks fixed-filter Gaussian moments only; it does not calibrate a test, select filters anew, or estimate power.",
            "",
            "## Existing atlas span audit",
            "",
            "| mode | fitted-null eigenvalue | algebraic retention | max anchor temporal sum | max time spatial sum |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in atlas_modes.itertuples(index=False):
        lines.append(
            f"| {int(row.mode)} | {row.eigenvalue:.6g} | "
            f"{row.sigma0_metric_retention:.4f} | "
            f"{row.maximum_absolute_per_anchor_temporal_sum:.4g} | "
            f"{row.maximum_absolute_per_time_spatial_sum:.4g} |"
        )
    lines.extend(["", "Basis-invariant two-dimensional cluster retention:", ""])
    for row in atlas_clusters.itertuples(index=False):
        lines.append(
            f"- `{row.cluster}`: mean `{row.mean_retention:.4f}`, minimum "
            f"`{row.minimum_retention:.4f}`, maximum principal angle "
            f"`{row.maximum_principal_angle_degrees:.2f}` degrees."
        )
    lines.extend(
        [
            "",
            "These retentions answer only whether the numerical weight arrays satisfy the double-centering constraints.  Because the source-coordinate anchors drift, they do not establish a combination of fixed physical rectangles.",
            "The original 100-by-8 atlas and the exact 25-by-8 oracle use different observation geometries, so their objective values and mode numbers are not an apples-to-apples performance comparison.",
            f"The legacy audit reproduces its original fitted-advection covariance `({geometry_audit['legacy_fitted_advection'][0]:.6g}, {geometry_audit['legacy_fitted_advection'][1]:.6g})`, rather than the fixed truth advection `({geometry_audit['truth_advection'][0]:.6g}, {geometry_audit['truth_advection'][1]:.6g})`; it is not part of the fixed-v0 exact-grid comparison.",
            "For near-degenerate legacy mode pairs, the two-dimensional canonical-retention summaries are primary; individual mode retentions are basis-dependent descriptions.",
            "",
            "## Interpretation boundary",
            "",
            "- Intrinsic optimization uses `Sigma1-SigmaM`; fitted-null discrimination is evaluated afterward from `v1/v0` and `g(v1/v0)`.",
            "- Intrinsic `mu` must not be identified with the existing total generalized eigenvalue `lambda-1`.",
            "- Minimum-norm dense dictionary coefficients are not unique physical contributions when the dictionary is redundant.",
            "- The single, greedy, dense, and full inequalities are checked only for the same intrinsic numerator and Sigma0 normalization.",
            "- No nugget estimation, advection estimation, sparsity penalty, response-based selection, or p-value calibration was added.",
            "- Every selected filter and percentage is specific to this regular 5-by-5-by-8 synthetic geometry; transfer to the warped 100-by-8 GEMS design requires a separate geometry-aware study.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    validate_config(config)
    experiment_config = config["experiment"]
    pilot_dir = resolve_from_here(experiment_config["pilot_dir"])
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else resolve_from_here(experiment_config["output_dir"])
    )
    manifest_path = pilot_dir / "run_manifest.json"
    point_path = pilot_dir / "selected_flow_tube_points.csv"
    if not manifest_path.is_file() or not point_path.is_file():
        raise FileNotFoundError("pilot manifest and selected points are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bool(manifest["split"].get("responses_used_for_null_fit_or_direction_selection", True)):
        raise ValueError("pilot does not certify covariance-only selection")
    truth = covariance_parameters(manifest["truth"])
    fitted_initial = covariance_parameters(manifest["null"]["fit"]["parameters"])
    if truth.nugget != 0.0:
        raise ValueError("this experiment requires statistical nugget zero")
    jitter = float(manifest["numerics"]["numerical_jitter_ratio"])

    started = time.perf_counter()
    coordinates, point_table = exact_comoving_coordinates(truth, config["geometry"])
    geometry = pairwise_lags(coordinates)
    anchor_count = int(config["geometry"]["latitude_count"]) * int(
        config["geometry"]["longitude_count"]
    )
    time_count = int(config["geometry"]["hours"])
    dimension = anchor_count * time_count
    true_covariance = joint_matern_half_covariance(
        geometry,
        truth,
        numerical_jitter_ratio=jitter,
    )
    matched_covariance = advected_separable_covariance(
        geometry,
        truth,
        numerical_jitter_ratio=jitter,
    )
    fit_started = time.perf_counter()
    fitted_null, fit_attempts, fit_summary = fit_fixed_advection_null(
        geometry,
        true_covariance,
        truth,
        fitted_initial,
        config["null_fit"],
        numerical_jitter_ratio=jitter,
    )
    fit_summary["seconds"] = time.perf_counter() - fit_started
    null_covariance = advected_separable_covariance(
        geometry,
        fitted_null,
        numerical_jitter_ratio=jitter,
    )
    intrinsic_difference = true_covariance - matched_covariance
    margin_checks = covariance_margin_checks(
        true_covariance,
        matched_covariance,
        anchor_count=anchor_count,
        time_count=time_count,
    )

    raw_dictionary, endpoints = rectangle_matrix(anchor_count, time_count)
    standardized_dictionary, rectangle_scales, raw_null_variances = standardize_dictionary(
        raw_dictionary,
        null_covariance,
    )
    metadata = build_rectangle_metadata(
        endpoints,
        point_table,
        truth,
        raw_null_variances,
    )
    standardized_null_variance = np.einsum(
        "ik,ij,jk->k",
        standardized_dictionary,
        null_covariance,
        standardized_dictionary,
        optimize=True,
    )
    single_intrinsic_numerator = np.einsum(
        "ik,ij,jk->k",
        standardized_dictionary,
        intrinsic_difference,
        standardized_dictionary,
        optimize=True,
    )
    metadata["standardized_null_variance_check"] = standardized_null_variance
    metadata["single_intrinsic_objective"] = single_intrinsic_numerator / standardized_null_variance
    svd_tolerance = float(config["dictionary"]["relative_svd_tolerance"])
    decomposition = dictionary_svd(
        standardized_dictionary,
        relative_tolerance=svd_tolerance,
    )
    theoretical_rank = (anchor_count - 1) * (time_count - 1)
    if decomposition.rank != theoretical_rank:
        raise ArithmeticError(
            f"rectangle rank {decomposition.rank} differs from theory {theoretical_rank}"
        )
    rank_rows = []
    for relative in config["dictionary"]["relative_svd_tolerance_sensitivity"]:
        absolute = float(relative) * decomposition.full_singular_values[0]
        rank_rows.append(
            {
                "relative_tolerance": float(relative),
                "absolute_tolerance": absolute,
                "rank": int(np.sum(decomposition.full_singular_values > absolute)),
            }
        )
    rank_sensitivity = pd.DataFrame(rank_rows)

    full_values, full_coefficients, full_filters = solve_constrained_contrast(
        intrinsic_difference,
        null_covariance,
    )
    dense_values, dense_coefficients, dense_filters = solve_constrained_contrast(
        intrinsic_difference,
        null_covariance,
        decomposition.basis,
    )
    full_solutions = {
        "negative": summarize_contrast_solution(
            full_values[0],
            full_coefficients[:, 0],
            full_filters[:, 0],
            np.eye(dimension),
            intrinsic_difference,
            null_covariance,
        ),
        "positive": summarize_contrast_solution(
            full_values[-1],
            full_coefficients[:, -1],
            full_filters[:, -1],
            np.eye(dimension),
            intrinsic_difference,
            null_covariance,
        ),
    }
    dense_solutions = {
        "negative": summarize_contrast_solution(
            dense_values[0],
            dense_coefficients[:, 0],
            dense_filters[:, 0],
            decomposition.basis,
            intrinsic_difference,
            null_covariance,
        ),
        "positive": summarize_contrast_solution(
            dense_values[-1],
            dense_coefficients[:, -1],
            dense_filters[:, -1],
            decomposition.basis,
            intrinsic_difference,
            null_covariance,
        ),
    }

    maximum_greedy_size = max(int(value) for value in config["dictionary"]["greedy_sizes"])
    greedy_paths = {
        branch: greedy_extreme_path(
            standardized_dictionary,
            intrinsic_difference,
            null_covariance,
            branch=branch,
            maximum_size=maximum_greedy_size,
            dependence_tolerance=float(config["dictionary"]["dependence_tolerance"]),
            score_tie_tolerance=float(config["dictionary"]["score_tie_tolerance"]),
        )
        for branch in ("positive", "negative")
    }

    comparison_rows = []
    filter_registry: dict[str, np.ndarray] = {}
    for branch in ("positive", "negative"):
        full_name = f"full_intrinsic_{branch}"
        dense_name = f"dense_dictionary_{branch}"
        filter_registry[full_name] = full_solutions[branch].filter_weights
        filter_registry[dense_name] = dense_solutions[branch].filter_weights
        comparison_rows.append(
            filter_row(
                full_name,
                "full_space",
                branch,
                full_solutions[branch].filter_weights,
                true_covariance,
                matched_covariance,
                null_covariance,
                constraint_basis=np.eye(dimension),
                anchor_count=anchor_count,
                time_count=time_count,
                objective_mu=full_solutions[branch].eigenvalue,
                rectangle_count=None,
            )
        )
        comparison_rows.append(
            filter_row(
                dense_name,
                "dense_dictionary",
                branch,
                dense_solutions[branch].filter_weights,
                true_covariance,
                matched_covariance,
                null_covariance,
                constraint_basis=decomposition.basis,
                anchor_count=anchor_count,
                time_count=time_count,
                objective_mu=dense_solutions[branch].eigenvalue,
                rectangle_count=None,
            )
        )
        for requested_size in config["dictionary"]["greedy_sizes"]:
            step = greedy_paths[branch][int(requested_size) - 1]
            name = f"greedy_{int(requested_size):02d}_{branch}"
            filter_registry[name] = step.filter_weights
            comparison_rows.append(
                filter_row(
                    name,
                    "greedy",
                    branch,
                    step.filter_weights,
                    true_covariance,
                    matched_covariance,
                    null_covariance,
                    constraint_basis=standardized_dictionary[
                        :, np.asarray(step.selected_indices, dtype=np.int64)
                    ],
                    anchor_count=anchor_count,
                    time_count=time_count,
                    objective_mu=step.eigenvalue,
                    rectangle_count=int(requested_size),
                )
            )
    comparison = pd.DataFrame(comparison_rows)
    comparison["same_sign_as_dense_extreme"] = False
    comparison["fraction_of_dense_extreme_magnitude"] = np.nan
    comparison["fraction_of_full_extreme_magnitude"] = np.nan
    for branch in ("positive", "negative"):
        branch_mask = comparison["branch"] == branch
        dense_mu = float(
            comparison.loc[
                branch_mask & (comparison["family"] == "dense_dictionary"),
                "objective_mu",
            ].iloc[0]
        )
        full_mu = float(
            comparison.loc[
                branch_mask & (comparison["family"] == "full_space"),
                "objective_mu",
            ].iloc[0]
        )
        comparison.loc[branch_mask, "same_sign_as_dense_extreme"] = np.sign(
            comparison.loc[branch_mask, "objective_mu"]
        ) == np.sign(dense_mu)
        comparison.loc[branch_mask, "fraction_of_dense_extreme_magnitude"] = comparison.loc[
            branch_mask, "objective_mu"
        ].abs() / abs(dense_mu)
        comparison.loc[branch_mask, "fraction_of_full_extreme_magnitude"] = comparison.loc[
            branch_mask, "objective_mu"
        ].abs() / abs(full_mu)

    for branch in ("positive", "negative"):
        full_mu = full_solutions[branch].eigenvalue
        dense_mu = dense_solutions[branch].eigenvalue
        single_mu = greedy_paths[branch][0].eigenvalue
        if branch == "positive" and not full_mu + 1.0e-10 >= dense_mu >= single_mu - 1.0e-10:
            raise ArithmeticError("positive full/dictionary/single ordering failed")
        if branch == "negative" and not full_mu - 1.0e-10 <= dense_mu <= single_mu + 1.0e-10:
            raise ArithmeticError("negative full/dictionary/single ordering failed")

    greedy_rows = []
    selected_rows = []
    for branch, path in greedy_paths.items():
        for step in path:
            greedy_rows.append(
                {
                    "branch": branch,
                    "size": step.size,
                    "added_rectangle_index": step.added_index,
                    "added_rectangle_id": metadata.loc[step.added_index, "rectangle_id"],
                    "selected_rectangle_indices": ";".join(
                        str(value) for value in step.selected_indices
                    ),
                    "objective_mu": step.eigenvalue,
                    "winner_runner_up_gap": step.winner_runner_up_gap,
                    "winner_tie_count": step.winner_tie_count,
                    "skipped_dependent_candidates": step.skipped_dependent_candidates,
                    "null_variance": float(
                        step.filter_weights @ null_covariance @ step.filter_weights
                    ),
                    "intrinsic_quadratic": float(
                        step.filter_weights @ intrinsic_difference @ step.filter_weights
                    ),
                }
            )
        for requested_size in config["dictionary"]["greedy_sizes"]:
            step = path[int(requested_size) - 1]
            for selection_order, (rectangle_index, coefficient) in enumerate(
                zip(step.selected_indices, step.standardized_coefficients),
                start=1,
            ):
                values = metadata.loc[rectangle_index].to_dict()
                values.update(
                    {
                        "branch": branch,
                        "combination_size": int(requested_size),
                        "selection_order": selection_order,
                        "standardized_coefficient": float(coefficient),
                        "raw_rectangle_coefficient": float(
                            coefficient / rectangle_scales[rectangle_index]
                        ),
                    }
                )
                selected_rows.append(values)
    greedy_frame = pd.DataFrame(greedy_rows)
    selected_frame = pd.DataFrame(selected_rows)

    dense_coefficient_rows = []
    for branch, column in (("negative", 0), ("positive", -1)):
        standardized_coefficients, raw_coefficients = recover_rectangle_coefficients(
            decomposition,
            dense_coefficients[:, column],
            rectangle_scales,
        )
        reconstructed = raw_dictionary @ raw_coefficients
        reconstruction_error = float(
            np.linalg.norm(reconstructed - dense_solutions[branch].filter_weights)
            / np.linalg.norm(dense_solutions[branch].filter_weights)
        )
        for rectangle_index in range(len(metadata)):
            dense_coefficient_rows.append(
                {
                    "branch": branch,
                    "rectangle_index": rectangle_index,
                    "rectangle_id": metadata.loc[rectangle_index, "rectangle_id"],
                    "standardized_minimum_norm_coefficient": float(
                        standardized_coefficients[rectangle_index]
                    ),
                    "raw_rectangle_coefficient": float(raw_coefficients[rectangle_index]),
                    "filter_reconstruction_relative_error": reconstruction_error,
                }
            )
    dense_coefficient_frame = pd.DataFrame(dense_coefficient_rows)

    weight_rows = []
    for name, weights in filter_registry.items():
        for row, weight in zip(point_table.itertuples(index=False), weights):
            weight_rows.append(
                {
                    "filter_name": name,
                    "time_index": row.time_index,
                    "anchor_index": row.anchor_index,
                    "moving_latitude": row.moving_latitude,
                    "moving_longitude": row.moving_longitude,
                    "standardized_moving_latitude": row.standardized_moving_latitude,
                    "standardized_moving_longitude": row.standardized_moving_longitude,
                    "weight": float(weight),
                }
            )
    filter_weights = pd.DataFrame(weight_rows)

    atlas_modes, atlas_clusters, geometry_audit = current_atlas_span_audit(
        pilot_dir,
        manifest,
    )
    monte_carlo_filters = {
        name: weights
        for name, weights in filter_registry.items()
        if name.startswith("dense_dictionary")
        or name.startswith("greedy_01")
        or name.startswith("greedy_08")
    }
    monte_carlo = simulation_moments(
        monte_carlo_filters,
        {
            "fitted_null": null_covariance,
            "matched_margin": matched_covariance,
            "truth": true_covariance,
        },
        replicates=int(config["monte_carlo"]["replicates"]),
        batch_size=int(config["monte_carlo"]["batch_size"]),
        random_seed=int(config["monte_carlo"]["random_seed"]),
    )

    projected_kl_rows = []
    for family in ("dense_dictionary", "greedy_08"):
        names = (f"{family}_positive", f"{family}_negative")
        filters = np.column_stack([filter_registry[name] for name in names])
        projected_kl_rows.append(
            {
                "filter_pair": ";".join(names),
                "dimension": 2,
                "joint_projected_kl": projected_gaussian_kl(
                    filters,
                    true_covariance,
                    null_covariance,
                ),
                "sum_of_marginal_g_fitted": float(
                    comparison.set_index("filter_name").loc[list(names), "g_fitted"].sum()
                ),
            }
        )
    projected_kl = pd.DataFrame(projected_kl_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(output_dir / "exact_comoving_points.csv", point_table)
    atomic_csv(output_dir / "fixed_advection_null_fit_attempts.csv", fit_attempts)
    atomic_csv(output_dir / "rectangle_dictionary_metadata.csv", metadata)
    atomic_csv(output_dir / "dictionary_rank_sensitivity.csv", rank_sensitivity)
    atomic_csv(output_dir / "intrinsic_filter_comparison.csv", comparison)
    atomic_csv(output_dir / "greedy_path.csv", greedy_frame)
    atomic_csv(output_dir / "greedy_selected_rectangles.csv", selected_frame)
    atomic_csv(output_dir / "dense_oracle_rectangle_coefficients.csv", dense_coefficient_frame)
    atomic_csv(output_dir / "filter_weights.csv", filter_weights)
    atomic_csv(output_dir / "existing_atlas_mode_span_audit.csv", atlas_modes)
    atomic_csv(output_dir / "existing_atlas_cluster_span_audit.csv", atlas_clusters)
    atomic_csv(output_dir / "fixed_filter_simulation_moments.csv", monte_carlo)
    atomic_csv(output_dir / "joint_projected_kl.csv", projected_kl)
    singular_frame = pd.DataFrame(
        {
            "singular_value_index": np.arange(1, len(decomposition.full_singular_values) + 1),
            "singular_value": decomposition.full_singular_values,
        }
    )
    atomic_csv(output_dir / "dictionary_singular_values.csv", singular_frame)

    plot_objective_comparison(
        comparison,
        output_dir / "figures/intrinsic_objective_comparison.png",
    )
    plot_filter_maps(
        {
            name: weights
            for name, weights in filter_registry.items()
            if name.startswith("dense_dictionary") or name.startswith("greedy_08")
        },
        config["geometry"],
        output_dir / "figures/filter_maps",
    )
    for branch in ("positive", "negative"):
        subset = selected_frame.loc[
            (selected_frame["branch"] == branch) & (selected_frame["combination_size"] == 8)
        ].sort_values("selection_order")
        plot_selected_rectangles(
            branch,
            subset,
            point_table,
            output_dir / f"figures/greedy_08_{branch}_rectangles.png",
        )
    plot_singular_values(
        decomposition.full_singular_values,
        rank_sensitivity,
        output_dir / "figures/dictionary_singular_values.png",
    )
    write_report(
        output_dir / "REPORT.md",
        comparison,
        greedy_frame,
        selected_frame,
        monte_carlo,
        projected_kl,
        fitted_null,
        fit_attempts,
        atlas_modes,
        atlas_clusters,
        geometry_audit,
        margin_checks,
        rank_sensitivity,
        dictionary_count=len(metadata),
        theoretical_rank=theoretical_rank,
        numerical_jitter_ratio=jitter,
        score_tie_tolerance=float(config["dictionary"]["score_tie_tolerance"]),
    )

    script_path = Path(__file__).resolve()
    core_path = HERE / "rectangle_dictionary_core.py"
    local_source_paths = (
        script_path,
        core_path,
        HERE / "diagnostic_core.py",
        HERE / "analyze_mode_count_path.py",
        HERE / "explore_eigen_directions.py",
    )
    run_configuration = {
        section: config[section]
        for section in ("experiment", "geometry", "null_fit", "dictionary", "monte_carlo")
    }
    summary = {
        "experiment": experiment_config["name"],
        "script": str(script_path),
        "config": str(config["_path"]),
        "config_sha256": sha256(config["_path"]),
        "output_dir": str(output_dir),
        "command": [sys.executable, *sys.argv],
        "run_configuration": run_configuration,
        "source_files": [
            {"path": str(path), "sha256": sha256(path)} for path in local_source_paths
        ],
        "git": git_state(),
        "runtime_environment": runtime_environment(),
        "pilot_manifest": str(manifest_path),
        "pilot_manifest_sha256": sha256(manifest_path),
        "selected_points": str(point_path),
        "selected_points_sha256": sha256(point_path),
        "response_columns_read": False,
        "oracle_not_real_data_test": True,
        "numerics": {
            "dtype": "float64",
            "numerical_jitter_ratio": jitter,
            "statistical_nugget": 0.0,
        },
        "geometry": {
            "type": "exact Cartesian product of moving anchors and hours",
            "anchor_count": anchor_count,
            "time_count": time_count,
            "dimension": dimension,
            "stacking_order": "time-major: index=time*anchor_count+anchor",
        },
        "truth": truth.to_dict(),
        "matched_margin": truth.to_dict(),
        "fitted_null_fixed_advection": fitted_null.to_dict(),
        "null_fit": fit_summary,
        "margin_checks": margin_checks,
        "dictionary": {
            "rectangle_count": len(metadata),
            "theoretical_rank": theoretical_rank,
            "retained_rank": decomposition.rank,
            "relative_svd_tolerance": decomposition.relative_tolerance,
            "absolute_svd_tolerance": decomposition.absolute_tolerance,
            "svd_reconstruction_relative_error": decomposition.reconstruction_relative_error,
            "condition_number_retained": float(
                decomposition.singular_values[0] / decomposition.singular_values[-1]
            ),
            "score_tie_tolerance": float(config["dictionary"]["score_tie_tolerance"]),
            "dependence_tolerance": float(config["dictionary"]["dependence_tolerance"]),
            "greedy_sizes": [int(value) for value in config["dictionary"]["greedy_sizes"]],
        },
        "greedy_tie_counts_by_step": greedy_frame[
            ["branch", "size", "winner_tie_count", "winner_runner_up_gap"]
        ].to_dict(orient="records"),
        "joint_projected_kl": projected_kl.to_dict(orient="records"),
        "existing_atlas_geometry_audit": geometry_audit,
        "ordering_checks": {
            "positive_full_ge_dictionary_ge_single": True,
            "negative_full_le_dictionary_le_single": True,
            "positive_greedy_nondecreasing": True,
            "negative_greedy_nonincreasing": True,
        },
        "maximum_null_normalization_error": float(
            np.max(np.abs(comparison["v0"].to_numpy() - 1.0))
        ),
        "maximum_dense_constrained_relative_residual": float(
            max(
                dense_solutions["positive"].constrained_relative_residual,
                dense_solutions["negative"].constrained_relative_residual,
            )
        ),
        "maximum_saved_filter_constrained_relative_residual": float(
            comparison["constrained_relative_residual"].max()
        ),
        "maximum_monte_carlo_absolute_relative_mean_error": float(
            monte_carlo["relative_mean_error"].abs().max()
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation_boundary": (
            "Sigma0-scaled intrinsic oracle on exact synthetic geometry. Existing atlas "
            "retention is algebraic only because the observed flow tube is warped."
        ),
    }
    summary["output_files"] = output_inventory(output_dir)
    atomic_json(output_dir / "experiment_manifest.json", summary)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "dictionary_count": len(metadata),
                "rank": decomposition.rank,
                "elapsed_seconds": summary["elapsed_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
