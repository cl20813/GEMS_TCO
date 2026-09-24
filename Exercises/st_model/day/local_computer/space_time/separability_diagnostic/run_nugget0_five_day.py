#!/usr/bin/env python3
"""Five-day pilot of an advected-separable null against a joint ST Matern.

The response values are never used to fit the oracle KL null or select the
diagnostic directions.  Three independent simulated days define the null and
the directions; the remaining two days are used for the displayed projection
statistics.  This split makes the experiment reusable when model fitting is
later made data-dependent.

The default dense design is a flow-aligned tube: 100 spatial anchors are
chosen by a deterministic max-min rule from cells observed at the
corresponding advected locations in all five days.  Days remain independent;
they are never concatenated as a continuous 40-hour process.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.linalg

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diagnostic_core import (
    CovarianceParameters,
    LagGeometry,
    advected_separable_correlation,
    advected_separable_covariance,
    covariance_log_likelihood_ratio,
    fit_kl_optimal_null,
    gaussian_kl,
    joint_matern_half_correlation,
    joint_matern_half_covariance,
    pairwise_lags,
    solve_generalized_eigenproblem,
    standardize_directions_for_design,
)


HERE = Path(__file__).resolve().parent
REPO = next(parent for parent in HERE.parents if (parent / "src/GEMS_TCO").is_dir())
DEFAULT_DATA_ROOT = (
    REPO / "outputs/sim_data/" "july_st_circulant_realpattern_smooth0p5_nugget0_matched5_090226"
)
DEFAULT_OUTPUT = HERE / "outputs/nugget0_five_day_092226"
KEY_PATTERN = re.compile(
    r"y(?P<year>\d{2})m(?P<month>\d{2})day(?P<day>\d{2})_hm(?P<hour>\d{2}):(?P<minute>\d{2})"
)


@dataclass(frozen=True)
class DayAsset:
    date: str
    keys: tuple[str, ...]
    frames: tuple[pd.DataFrame, ...]
    truth: dict[str, Any]
    source_path: str


@dataclass(frozen=True)
class DayDesign:
    date: str
    coordinates: np.ndarray
    residual: np.ndarray
    point_table: pd.DataFrame


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "to_dict"):
        return json_ready(value.to_dict())
    if hasattr(value, "__dataclass_fields__"):
        return json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.12g")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--spatial-anchors", type=int, default=100)
    parser.add_argument("--hours", type=int, default=8, choices=[8])
    parser.add_argument("--train-days", type=int, default=3)
    parser.add_argument("--top-modes", type=int, default=12)
    parser.add_argument("--optimizer-starts", type=int, default=8)
    parser.add_argument("--optimizer-max-iterations", type=int, default=120)
    parser.add_argument("--bootstrap-replicates", type=int, default=50_000)
    parser.add_argument("--random-seed", type=int, default=20260922)
    parser.add_argument("--numerical-jitter-ratio", type=float, default=1.0e-10)
    return parser


def key_parts(key: str) -> tuple[str, int, int]:
    match = KEY_PATTERN.fullmatch(str(key))
    if match is None:
        raise ValueError(f"unrecognized simulation key: {key}")
    date = f"20{match['year']}-{match['month']}-{match['day']}"
    return date, int(match["hour"]), int(match["minute"])


def load_day_assets(data_root: Path, hours: int) -> list[DayAsset]:
    manifest_path = data_root / "DERIVATION_MANIFEST.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected_dates = tuple(str(item) for item in manifest["selected_dates"])
    if float(manifest["truth_nugget"]) != 0.0:
        raise ValueError("the diagnostic requires the nugget-zero asset")

    assets: list[DayAsset] = []
    for year_dir in sorted(data_root.glob("*_july_st_circulant")):
        year_match = re.match(r"(?P<year>\d{4})_", year_dir.name)
        if year_match is None:
            continue
        year = int(year_match["year"])
        prefix = f"sim_july{year}_st_circulant"
        pickle_path = year_dir / f"{prefix}_gridded.pkl"
        truth_path = year_dir / f"{prefix}_truth.json"
        if not pickle_path.is_file() or not truth_path.is_file():
            raise FileNotFoundError(f"incomplete simulation asset under {year_dir}")
        frames_by_key = pd.read_pickle(pickle_path)
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        if float(truth["nugget"]) != 0.0 or not math.isclose(float(truth["smooth"]), 0.5):
            raise ValueError(f"unexpected truth family in {truth_path}")
        grouped: dict[str, list[tuple[str, pd.DataFrame]]] = {}
        for key, frame in frames_by_key.items():
            date, hour, minute = key_parts(str(key))
            if date in selected_dates:
                grouped.setdefault(date, []).append((f"{hour:02d}:{minute:02d}|{key}", frame))
        for date, records in grouped.items():
            ordered = sorted(records, key=lambda item: item[0])
            if len(ordered) != hours:
                raise ValueError(f"expected {hours} frames for {date}, found {len(ordered)}")
            assets.append(
                DayAsset(
                    date=date,
                    keys=tuple(item[0].split("|", 1)[1] for item in ordered),
                    frames=tuple(item[1] for item in ordered),
                    truth=truth,
                    source_path=str(pickle_path.resolve()),
                )
            )
    assets.sort(key=lambda asset: asset.date)
    observed_dates = tuple(asset.date for asset in assets)
    if observed_dates != tuple(sorted(selected_dates)):
        raise RuntimeError(
            f"selected dates mismatch: expected={sorted(selected_dates)}, observed={observed_dates}"
        )
    reference_truth = {
        key: float(assets[0].truth[key])
        for key in (
            "smooth",
            "sigmasq",
            "range_lat",
            "range_lon",
            "range_time",
            "advec_lat",
            "advec_lon",
            "nugget",
            "mean_intercept",
            "mean_lat_slope",
            "mean_lat_center",
        )
    }
    for asset in assets[1:]:
        current = {key: float(asset.truth[key]) for key in reference_truth}
        if current != reference_truth:
            raise ValueError(f"truth parameters differ for {asset.date}")
    return assets


def regular_grid(asset: DayAsset) -> tuple[np.ndarray, np.ndarray]:
    first = asset.frames[0]
    n_lat = int(first["Latitude"].nunique())
    n_lon = int(first["Longitude"].nunique())
    if n_lat * n_lon != len(first):
        raise ValueError("regular grid is not a complete rectangle")
    latitude = first["Latitude"].to_numpy(dtype=np.float64).reshape(n_lat, n_lon)
    longitude = first["Longitude"].to_numpy(dtype=np.float64).reshape(n_lat, n_lon)
    if not np.allclose(latitude, latitude[:, :1]) or not np.allclose(longitude, longitude[:1, :]):
        raise ValueError("row ordering is not a latitude-longitude product grid")
    reference = np.column_stack([latitude.ravel(), longitude.ravel()])
    for day in (asset,):
        for key, frame in zip(day.keys, day.frames):
            observed = frame[["Latitude", "Longitude"]].to_numpy(dtype=np.float64)
            if observed.shape != reference.shape or not np.allclose(observed, reference):
                raise ValueError(f"grid ordering differs at {key}")
    return latitude, longitude


def all_valid_tensor(
    assets: list[DayAsset],
    grid_shape: tuple[int, int],
) -> np.ndarray:
    valid = np.empty((len(assets), len(assets[0].frames), *grid_shape), dtype=bool)
    required = ("ColumnAmountO3", "Source_Latitude", "Source_Longitude")
    reference_grid = assets[0].frames[0][["Latitude", "Longitude"]].to_numpy(dtype=np.float64)
    for day_index, asset in enumerate(assets):
        for time_index, (key, frame) in enumerate(zip(asset.keys, asset.frames)):
            grid = frame[["Latitude", "Longitude"]].to_numpy(dtype=np.float64)
            if grid.shape != reference_grid.shape or not np.allclose(grid, reference_grid):
                raise ValueError(f"grid ordering differs at {key}")
            finite = np.logical_and.reduce(
                [np.isfinite(frame[column].to_numpy(dtype=np.float64)) for column in required]
            )
            valid[day_index, time_index] = finite.reshape(grid_shape)
    return valid


def flow_shifts(
    latitude: np.ndarray,
    longitude: np.ndarray,
    *,
    hours: int,
    advec_lat: float,
    advec_lon: float,
) -> list[tuple[int, int]]:
    row_step = float(np.median(np.diff(latitude[:, 0])))
    column_step = float(np.median(np.diff(longitude[0, :])))
    if row_step == 0.0 or column_step == 0.0:
        raise ValueError("grid steps must be nonzero")
    return [
        (
            int(np.rint(advec_lat * time_index / row_step)),
            int(np.rint(advec_lon * time_index / column_step)),
        )
        for time_index in range(hours)
    ]


def common_flow_anchor_candidates(
    valid: np.ndarray,
    shifts: list[tuple[int, int]],
) -> np.ndarray:
    n_rows, n_columns = valid.shape[-2:]
    candidates: list[tuple[int, int]] = []
    for row in range(n_rows):
        for column in range(n_columns):
            usable = True
            for time_index, (row_shift, column_shift) in enumerate(shifts):
                shifted_row = row + row_shift
                shifted_column = column + column_shift
                if not (
                    0 <= shifted_row < n_rows
                    and 0 <= shifted_column < n_columns
                    and valid[:, time_index, shifted_row, shifted_column].all()
                ):
                    usable = False
                    break
            if usable:
                candidates.append((row, column))
    if not candidates:
        raise RuntimeError("no grid anchor remains valid along the flow tube")
    return np.asarray(candidates, dtype=np.int64)


def deterministic_maxmin(
    candidates: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    *,
    count: int,
    range_lat: float,
    range_lon: float,
) -> tuple[np.ndarray, np.ndarray]:
    if count < 1 or count > len(candidates):
        raise ValueError("spatial anchor count is outside the candidate range")
    physical = np.column_stack(
        [
            latitude[candidates[:, 0], candidates[:, 1]],
            longitude[candidates[:, 0], candidates[:, 1]],
        ]
    )
    scaled = physical / np.asarray([range_lat, range_lon])
    domain_center = np.asarray([latitude.mean(), longitude.mean()]) / np.asarray(
        [range_lat, range_lon]
    )
    first = int(np.argmin(np.sum((scaled - domain_center) ** 2, axis=1)))
    selected = [first]
    chosen = np.zeros(len(candidates), dtype=bool)
    chosen[first] = True
    minimum_distance_sq = np.sum((scaled - scaled[first]) ** 2, axis=1)
    insertion_separation = [np.nan]
    for _ in range(1, count):
        available_distance = np.where(chosen, -np.inf, minimum_distance_sq)
        next_index = int(np.argmax(available_distance))
        insertion_separation.append(float(np.sqrt(available_distance[next_index])))
        selected.append(next_index)
        chosen[next_index] = True
        distance_sq = np.sum((scaled - scaled[next_index]) ** 2, axis=1)
        minimum_distance_sq = np.minimum(minimum_distance_sq, distance_sq)
    return candidates[np.asarray(selected)], np.asarray(insertion_separation)


def build_day_design(
    asset: DayAsset,
    anchors: np.ndarray,
    shifts: list[tuple[int, int]],
    grid_shape: tuple[int, int],
) -> DayDesign:
    n_columns = grid_shape[1]
    coordinate_parts: list[np.ndarray] = []
    residual_parts: list[np.ndarray] = []
    rows: list[pd.DataFrame] = []
    for time_index, (frame, (row_shift, column_shift)) in enumerate(zip(asset.frames, shifts)):
        shifted_rows = anchors[:, 0] + row_shift
        shifted_columns = anchors[:, 1] + column_shift
        flat = shifted_rows * n_columns + shifted_columns
        source_lat = frame["Source_Latitude"].to_numpy(dtype=np.float64)[flat]
        source_lon = frame["Source_Longitude"].to_numpy(dtype=np.float64)[flat]
        response = frame["ColumnAmountO3"].to_numpy(dtype=np.float64)[flat]
        grid_lat = frame["Latitude"].to_numpy(dtype=np.float64)[flat]
        grid_lon = frame["Longitude"].to_numpy(dtype=np.float64)[flat]
        if not np.isfinite(np.column_stack([source_lat, source_lon, response])).all():
            raise RuntimeError(f"flow tube contains missing observations on {asset.date}")
        mean = float(asset.truth["mean_intercept"]) + float(asset.truth["mean_lat_slope"]) * (
            source_lat - float(asset.truth["mean_lat_center"])
        )
        residual = response - mean
        coordinate_parts.append(
            np.column_stack(
                [source_lat, source_lon, np.full(len(flat), time_index, dtype=np.float64)]
            )
        )
        residual_parts.append(residual)
        rows.append(
            pd.DataFrame(
                {
                    "date": asset.date,
                    "time_index": time_index,
                    "anchor_rank": np.arange(1, len(anchors) + 1),
                    "anchor_row": anchors[:, 0],
                    "anchor_column": anchors[:, 1],
                    "row_shift": row_shift,
                    "column_shift": column_shift,
                    "grid_row": shifted_rows,
                    "grid_column": shifted_columns,
                    "grid_latitude": grid_lat,
                    "grid_longitude": grid_lon,
                    "source_latitude": source_lat,
                    "source_longitude": source_lon,
                    "response": response,
                    "true_mean": mean,
                    "true_residual": residual,
                }
            )
        )
    return DayDesign(
        date=asset.date,
        coordinates=np.vstack(coordinate_parts),
        residual=np.concatenate(residual_parts),
        point_table=pd.concat(rows, ignore_index=True),
    )


def true_parameters(truth: dict[str, Any]) -> CovarianceParameters:
    return CovarianceParameters(
        variance=float(truth["sigmasq"]),
        range_lat=float(truth["range_lat"]),
        range_lon=float(truth["range_lon"]),
        range_time=float(truth["range_time"]),
        advec_lat=float(truth["advec_lat"]),
        advec_lon=float(truth["advec_lon"]),
        nugget=float(truth["nugget"]),
    )


def stable_factor(covariance: np.ndarray) -> np.ndarray:
    covariance = (np.asarray(covariance) + np.asarray(covariance).T) * 0.5
    try:
        return scipy.linalg.cholesky(covariance, lower=True, check_finite=False)
    except scipy.linalg.LinAlgError:
        minimum = float(np.linalg.eigvalsh(covariance).min())
        correction = max(1.0e-12, -minimum + 1.0e-12)
        return scipy.linalg.cholesky(
            covariance + np.eye(len(covariance)) * correction,
            lower=True,
            check_finite=False,
        )


def likelihood_ratio_many(
    values: np.ndarray,
    null_covariance: np.ndarray,
    true_covariance: np.ndarray,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    factor0 = stable_factor(null_covariance)
    factor1 = stable_factor(true_covariance)
    precision0 = scipy.linalg.cho_solve(
        (factor0, True), np.eye(null_covariance.shape[0]), check_finite=False
    )
    precision1 = scipy.linalg.cho_solve(
        (factor1, True), np.eye(true_covariance.shape[0]), check_finite=False
    )
    constant = 0.5 * (2.0 * np.log(np.diag(factor0)).sum() - 2.0 * np.log(np.diag(factor1)).sum())
    quadratic = np.einsum("bi,ij,bj->b", values, precision0 - precision1, values, optimize=True)
    return constant + 0.5 * quadratic


def bootstrap_projection_test(
    null_covariance: np.ndarray,
    true_covariance: np.ndarray,
    observed: np.ndarray,
    *,
    replicates: int,
    random_seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    rng = np.random.default_rng(int(random_seed))
    factor0 = stable_factor(null_covariance)
    factor1 = stable_factor(true_covariance)
    standard = rng.standard_normal((replicates, len(observed)))
    null_draws = standard @ factor0.T
    true_draws = rng.standard_normal((replicates, len(observed))) @ factor1.T
    null_max = np.max(null_draws**2, axis=1)
    true_max = np.max(true_draws**2, axis=1)
    observed_max = float(np.max(observed**2))
    null_llr = likelihood_ratio_many(null_draws, null_covariance, true_covariance)
    true_llr = likelihood_ratio_many(true_draws, null_covariance, true_covariance)
    observed_llr = float(
        covariance_log_likelihood_ratio(observed, null_covariance, true_covariance)
    )
    max_critical = float(np.quantile(null_max, 0.95, method="higher"))
    llr_critical = float(np.quantile(null_llr, 0.95, method="higher"))
    summary = {
        "replicates": int(replicates),
        "familywise_alpha": 0.05,
        "max_squared": {
            "observed": observed_max,
            "null_critical_95": max_critical,
            "bootstrap_p_value": float((1 + np.sum(null_max >= observed_max)) / (replicates + 1)),
            "oracle_power": float(np.mean(true_max > max_critical)),
        },
        "top_subspace_log_likelihood_ratio": {
            "observed": observed_llr,
            "null_critical_95": llr_critical,
            "bootstrap_p_value": float((1 + np.sum(null_llr >= observed_llr)) / (replicates + 1)),
            "oracle_power": float(np.mean(true_llr > llr_critical)),
        },
    }
    distributions = {
        "null_max_squared": null_max,
        "true_max_squared": true_max,
        "null_llr": null_llr,
        "true_llr": true_llr,
    }
    return summary, distributions


def plot_spectrum(frame: pd.DataFrame, output_dir: Path) -> Path:
    path = output_dir / "figures/generalized_eigen_spectrum.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    color = np.where(frame["eigenvalue"].to_numpy() >= 1.0, "#b33b2e", "#3568a8")
    axes[0].scatter(frame["rank_by_g"], frame["log2_eigenvalue"], c=color, s=12)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set(xlabel="rank by g(lambda)", ylabel="log2(lambda)")
    axes[0].set_title("Null-versus-true variance ratios")
    axes[1].plot(frame["rank_by_g"], frame["g_score"], color="#5b2c83")
    axes[1].set_yscale("log")
    axes[1].set(xlabel="rank by g(lambda)", ylabel="g(lambda)")
    axes[1].set_title("Per-direction KL contribution")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_directions(
    directions: np.ndarray,
    reference_points: pd.DataFrame,
    fitted_null: CovarianceParameters,
    output_dir: Path,
) -> Path:
    path = output_dir / "figures/top_direction_patterns.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    shown = min(3, directions.shape[1])
    fig, axes = plt.subplots(shown, 2, figsize=(12, 4.2 * shown), squeeze=False)
    time = reference_points["time_index"].to_numpy(dtype=np.float64)
    lat = reference_points["source_latitude"].to_numpy(dtype=np.float64)
    lon = reference_points["source_longitude"].to_numpy(dtype=np.float64)
    moving_lat = lat - fitted_null.advec_lat * time
    moving_lon = lon - fitted_null.advec_lon * time
    for mode in range(shown):
        weights = directions[:, mode]
        limit = float(np.max(np.abs(weights)))
        for column, (x, y, title) in enumerate(
            (
                (lon, lat, "original coordinates"),
                (moving_lon, moving_lat, "fitted-null moving coordinates"),
            )
        ):
            scatter = axes[mode, column].scatter(
                x,
                y,
                c=weights,
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
                s=13,
                alpha=0.82,
                linewidths=0.0,
            )
            axes[mode, column].set(
                xlabel="longitude",
                ylabel="latitude",
                title=f"mode {mode + 1}: {title}",
            )
            fig.colorbar(scatter, ax=axes[mode, column], shrink=0.82, label="weight")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_bootstrap(
    distributions: dict[str, np.ndarray],
    summary: dict[str, Any],
    output_dir: Path,
) -> Path:
    path = output_dir / "figures/heldout_bootstrap_calibration.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    pairs = (
        (
            "null_max_squared",
            "true_max_squared",
            "max_squared",
            "max squared projection",
        ),
        (
            "null_llr",
            "true_llr",
            "top_subspace_log_likelihood_ratio",
            "top-subspace log likelihood ratio",
        ),
    )
    for axis, (null_key, true_key, summary_key, label) in zip(axes, pairs):
        null_values = distributions[null_key]
        true_values = distributions[true_key]
        lower = float(min(np.quantile(null_values, 0.005), np.quantile(true_values, 0.005)))
        upper = float(max(np.quantile(null_values, 0.995), np.quantile(true_values, 0.995)))
        bins = np.linspace(lower, upper, 80)
        axis.hist(null_values, bins=bins, density=True, alpha=0.55, label="null")
        axis.hist(true_values, bins=bins, density=True, alpha=0.55, label="joint Matern")
        axis.axvline(
            summary[summary_key]["null_critical_95"],
            color="black",
            linestyle="--",
            label="5% critical value",
        )
        axis.axvline(
            summary[summary_key]["observed"],
            color="#b33b2e",
            linewidth=1.5,
            label="held-out observed",
        )
        axis.set(xlabel=label, ylabel="density")
        axis.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_covariance_slices(
    truth: CovarianceParameters,
    fitted_null: CovarianceParameters,
    output_dir: Path,
) -> Path:
    path = output_dir / "figures/correlation_slice_comparison.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    lat_lag = np.linspace(-0.8, 0.8, 321)
    lon_lag = np.linspace(-1.2, 1.2, 321)
    time_lag = np.linspace(0.0, 7.0, 281)
    cases = (
        (
            axes[0, 0],
            LagGeometry(lat_lag, np.zeros_like(lat_lag), np.zeros_like(lat_lag)),
            lat_lag,
            "latitude lag at u=0",
        ),
        (
            axes[0, 1],
            LagGeometry(np.zeros_like(lon_lag), lon_lag, np.zeros_like(lon_lag)),
            lon_lag,
            "longitude lag at u=0",
        ),
        (
            axes[1, 0],
            LagGeometry(truth.advec_lat * time_lag, truth.advec_lon * time_lag, time_lag),
            time_lag,
            "time lag along true flow",
        ),
        (
            axes[1, 1],
            LagGeometry(
                truth.advec_lat * time_lag + truth.range_lat,
                truth.advec_lon * time_lag,
                time_lag,
            ),
            time_lag,
            "mixed lag: one latitude range off flow",
        ),
    )
    for axis, geometry, x, title in cases:
        axis.plot(x, joint_matern_half_correlation(geometry, truth), label="joint Matern truth")
        axis.plot(
            x,
            advected_separable_correlation(geometry, fitted_null),
            linestyle="--",
            label="KL-fitted separable null",
        )
        axis.set(xlabel=title, ylabel="correlation", ylim=(-0.02, 1.02))
        axis.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def write_results_markdown(
    output_dir: Path,
    *,
    truth: CovarianceParameters,
    null_fit: Any,
    reference_eigen: Any,
    bootstrap: dict[str, Any],
    train_dates: list[str],
    evaluation_dates: list[str],
    n_observations: int,
    design_description: str,
) -> None:
    fitted = null_fit.parameters
    boundary = ", ".join(null_fit.boundary_parameters) or "none"
    text = f"""# Nugget-zero advected-separable diagnostic: pilot result

## Design

- Independent design days: {', '.join(train_dates)}
- Held-out response days: {', '.join(evaluation_dates)}
- Dense dimension per day: {n_observations}
- Spatial design: {design_description}
- Statistical nugget: exactly 0 in both models
- Direction selection: covariance-only on the design split; no held-out response used

## Truth and strongest fitted null

The true joint Matern parameters are `{truth.to_dict()}`.

The KL-projected advected-separable null is `{fitted.to_dict()}`.  Parameters
at an optimization bound: **{boundary}**.  The best normalized expected-NLL
objective is `{null_fit.objective_per_observation:.10g}`.

On the averaged design covariance, the generalized spectrum has total KL
`{reference_eigen.matrix_kl:.8g}` and KL per observation
`{reference_eigen.matrix_kl / n_observations:.8g}`.  The spectral identity
sum g(lambda)=KL differs by
`{abs(reference_eigen.matrix_kl-reference_eigen.spectral_kl):.3e}`.

## Held-out calibration

- Max-projection bootstrap p-value: `{bootstrap['max_squared']['bootstrap_p_value']:.6g}`
- Max-projection oracle power at alpha=0.05: `{bootstrap['max_squared']['oracle_power']:.6g}`
- Top-subspace LLR bootstrap p-value: `{bootstrap['top_subspace_log_likelihood_ratio']['bootstrap_p_value']:.6g}`
- Top-subspace LLR oracle power at alpha=0.05: `{bootstrap['top_subspace_log_likelihood_ratio']['oracle_power']:.6g}`

These are pilot, alternative-specific results for this fixed spatial design.
They do not establish final power for the full GEMS domain.  A
publication run should repeat the fixed pipeline over subset sizes and seeds,
then use at least 1,000 complete null/alternative simulations.  If null fitting
or direction selection is performed on the same responses being tested, the
entire fit-and-select operation must be repeated inside each bootstrap draw.
"""
    (output_dir / "RESULTS.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    if args.spatial_anchors < 2:
        raise ValueError("spatial-anchors must be at least two")
    if args.train_days < 1:
        raise ValueError("train-days must be positive")
    if args.top_modes < 1:
        raise ValueError("top-modes must be positive")
    if args.numerical_jitter_ratio < 0.0:
        raise ValueError("numerical-jitter-ratio must be non-negative")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    workflow_started = time.perf_counter()

    print("Loading five independent nugget-zero day blocks", flush=True)
    assets = load_day_assets(Path(args.data_root), int(args.hours))
    if args.train_days >= len(assets):
        raise ValueError("train-days must leave at least one held-out day")
    truth = true_parameters(assets[0].truth)
    latitude, longitude = regular_grid(assets[0])
    valid = all_valid_tensor(assets, latitude.shape)
    shifts = flow_shifts(
        latitude,
        longitude,
        hours=int(args.hours),
        advec_lat=truth.advec_lat,
        advec_lon=truth.advec_lon,
    )
    candidates = common_flow_anchor_candidates(valid, shifts)
    anchors, insertion_separation = deterministic_maxmin(
        candidates,
        latitude,
        longitude,
        count=int(args.spatial_anchors),
        range_lat=truth.range_lat,
        range_lon=truth.range_lon,
    )
    designs = [build_day_design(asset, anchors, shifts, latitude.shape) for asset in assets]
    atomic_csv(
        output_dir / "selected_flow_tube_points.csv",
        pd.concat([design.point_table for design in designs], ignore_index=True),
    )

    geometries = [pairwise_lags(design.coordinates) for design in designs]
    true_covariances = [
        joint_matern_half_covariance(
            geometry,
            truth,
            numerical_jitter_ratio=float(args.numerical_jitter_ratio),
        )
        for geometry in geometries
    ]
    train_indices = list(range(int(args.train_days)))
    evaluation_indices = list(range(int(args.train_days), len(designs)))

    print(
        f"Fitting KL-optimal null with {args.optimizer_starts} starts on "
        f"{len(train_indices)} design days",
        flush=True,
    )
    fit_started = time.perf_counter()
    null_fit = fit_kl_optimal_null(
        [geometries[index] for index in train_indices],
        [true_covariances[index] for index in train_indices],
        truth,
        numerical_jitter_ratio=float(args.numerical_jitter_ratio),
        n_starts=int(args.optimizer_starts),
        random_seed=int(args.random_seed),
        max_iterations=int(args.optimizer_max_iterations),
    )
    fit_seconds = time.perf_counter() - fit_started
    fitted_null = null_fit.parameters
    margin_null = CovarianceParameters(**truth.to_dict())
    fitted_null_covariances = [
        advected_separable_covariance(
            geometry,
            fitted_null,
            numerical_jitter_ratio=float(args.numerical_jitter_ratio),
        )
        for geometry in geometries
    ]
    margin_null_covariances = [
        advected_separable_covariance(
            geometry,
            margin_null,
            numerical_jitter_ratio=float(args.numerical_jitter_ratio),
        )
        for geometry in geometries
    ]
    attempt_frame = pd.DataFrame([asdict(item) for item in null_fit.attempts]).sort_values(
        "objective_per_observation"
    )
    atomic_csv(output_dir / "null_fit_attempts.csv", attempt_frame)
    kl_rows = []
    for index, design in enumerate(designs):
        dimension = len(design.residual)
        kl_rows.append(
            {
                "date": design.date,
                "split": "design" if index in train_indices else "heldout",
                "dimension": dimension,
                "kl_fitted_null": gaussian_kl(
                    true_covariances[index], fitted_null_covariances[index]
                ),
                "kl_margin_matched_null": gaussian_kl(
                    true_covariances[index], margin_null_covariances[index]
                ),
            }
        )
    kl_frame = pd.DataFrame(kl_rows)
    kl_frame["kl_fitted_null_per_observation"] = kl_frame["kl_fitted_null"] / kl_frame["dimension"]
    kl_frame["kl_margin_matched_null_per_observation"] = (
        kl_frame["kl_margin_matched_null"] / kl_frame["dimension"]
    )
    atomic_csv(output_dir / "day_kl_comparison.csv", kl_frame)

    print("Solving the generalized eigenproblem on averaged design covariances", flush=True)
    reference_true = np.mean(np.stack([true_covariances[index] for index in train_indices]), axis=0)
    reference_null = np.mean(
        np.stack([fitted_null_covariances[index] for index in train_indices]), axis=0
    )
    eigen_started = time.perf_counter()
    eigen = solve_generalized_eigenproblem(reference_true, reference_null)
    eigen_seconds = time.perf_counter() - eigen_started
    dimension = reference_true.shape[0]
    eigen_frame = pd.DataFrame(
        {
            "rank_by_g": np.arange(1, dimension + 1),
            "eigenvalue": eigen.eigenvalues,
            "log2_eigenvalue": np.log2(eigen.eigenvalues),
            "g_score": eigen.scores,
            "cumulative_g": np.cumsum(eigen.scores),
            "cumulative_g_fraction": np.cumsum(eigen.scores) / eigen.scores.sum(),
            "variance_direction": np.where(
                eigen.eigenvalues >= 1.0, "true_gt_null", "true_lt_null"
            ),
        }
    )
    atomic_csv(output_dir / "generalized_eigenvalues.csv", eigen_frame)

    top_modes = min(int(args.top_modes), dimension)
    selected_directions = eigen.eigenvectors[:, :top_modes]
    reference_points = designs[train_indices[0]].point_table.copy()
    direction_rows = []
    for mode_index in range(top_modes):
        part = reference_points[
            [
                "time_index",
                "anchor_rank",
                "grid_latitude",
                "grid_longitude",
                "source_latitude",
                "source_longitude",
            ]
        ].copy()
        part.insert(0, "mode", mode_index + 1)
        part["training_eigenvalue"] = eigen.eigenvalues[mode_index]
        part["training_g_score"] = eigen.scores[mode_index]
        part["weight"] = selected_directions[:, mode_index]
        direction_rows.append(part)
    atomic_csv(
        output_dir / "selected_directions_long.csv",
        pd.concat(direction_rows, ignore_index=True),
    )

    projection_rows = []
    projected_null_covariances = []
    projected_true_covariances = []
    observed_parts = []
    for index in evaluation_indices:
        standardized, projected_null, projected_true = standardize_directions_for_design(
            selected_directions,
            fitted_null_covariances[index],
            true_covariances[index],
        )
        projected = standardized.T @ designs[index].residual
        observed_parts.append(projected)
        projected_null_covariances.append(projected_null)
        projected_true_covariances.append(projected_true)
        for mode_index in range(top_modes):
            marginal_lambda = float(projected_true[mode_index, mode_index])
            projection_rows.append(
                {
                    "date": designs[index].date,
                    "mode": mode_index + 1,
                    "training_eigenvalue": eigen.eigenvalues[mode_index],
                    "training_g_score": eigen.scores[mode_index],
                    "heldout_null_variance": projected_null[mode_index, mode_index],
                    "heldout_true_variance": marginal_lambda,
                    "projection": projected[mode_index],
                    "squared_projection": projected[mode_index] ** 2,
                }
            )
    projection_frame = pd.DataFrame(projection_rows)
    atomic_csv(output_dir / "heldout_projection_scores.csv", projection_frame)
    projected_null_block = scipy.linalg.block_diag(*projected_null_covariances)
    projected_true_block = scipy.linalg.block_diag(*projected_true_covariances)
    observed = np.concatenate(observed_parts)
    print(
        f"Calibrating {len(observed)} held-out projections with "
        f"{args.bootstrap_replicates:,} simulations",
        flush=True,
    )
    bootstrap, distributions = bootstrap_projection_test(
        projected_null_block,
        projected_true_block,
        observed,
        replicates=int(args.bootstrap_replicates),
        random_seed=int(args.random_seed) + 1,
    )
    atomic_json(output_dir / "bootstrap_summary.json", bootstrap)

    figure_paths = [
        plot_spectrum(eigen_frame, output_dir),
        plot_directions(selected_directions, reference_points, fitted_null, output_dir),
        plot_bootstrap(distributions, bootstrap, output_dir),
        plot_covariance_slices(truth, fitted_null, output_dir),
    ]
    elapsed = time.perf_counter() - workflow_started
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "output_dir": str(output_dir.resolve()),
        "data_contract": {
            "dates": [asset.date for asset in assets],
            "independent_eight_hour_blocks": True,
            "continuous_40_hour_process": False,
            "statistical_nugget": 0.0,
            "mean_removal": "known simulation mean at Source_Latitude",
            "coordinate_columns": ["Source_Latitude", "Source_Longitude", "local_time_0_to_7"],
        },
        "split": {
            "design_dates": [designs[index].date for index in train_indices],
            "heldout_dates": [designs[index].date for index in evaluation_indices],
            "responses_used_for_null_fit_or_direction_selection": False,
        },
        "subset": {
            "method": "truth-flow-aligned common-valid tubes plus deterministic anisotropic max-min",
            "candidate_anchor_count": int(len(candidates)),
            "selected_spatial_anchors": int(len(anchors)),
            "hours": int(args.hours),
            "dimension_per_day": int(dimension),
            "grid_shifts_by_local_hour": shifts,
            "last_maxmin_insertion_separation_scaled": float(insertion_separation[-1]),
        },
        "truth": truth,
        "null": {
            "family": "sigma2 * exp(-||h-vu||_anisotropic) * exp(-|u|/range_time)",
            "same_exponential_axis_margins_as_truth": True,
            "fit": null_fit,
            "fit_seconds": fit_seconds,
        },
        "numerics": {
            "dtype": "float64",
            "statistical_nugget": 0.0,
            "numerical_jitter_ratio": float(args.numerical_jitter_ratio),
            "eigen_seconds": eigen_seconds,
            "generalized_eigen_max_relative_residual": eigen.max_relative_residual,
            "generalized_eigen_max_null_orthonormality_error": eigen.max_null_orthonormality_error,
            "matrix_kl": eigen.matrix_kl,
            "spectral_kl": eigen.spectral_kl,
            "kl_identity_absolute_error": abs(eigen.matrix_kl - eigen.spectral_kl),
        },
        "selected_modes": {
            "count": top_modes,
            "true_variance_greater_than_null": int(np.sum(eigen.eigenvalues[:top_modes] > 1.0)),
            "true_variance_less_than_null": int(np.sum(eigen.eigenvalues[:top_modes] < 1.0)),
        },
        "bootstrap": bootstrap,
        "figures": [str(path.resolve()) for path in figure_paths],
        "total_seconds": elapsed,
        "interpretation_boundary": (
            "Oracle alternative-specific pilot on one fixed flow tube; final paper claims require "
            "subset/seed convergence and a full repeated-simulation study."
        ),
    }
    atomic_json(output_dir / "run_manifest.json", manifest)
    write_results_markdown(
        output_dir,
        truth=truth,
        null_fit=null_fit,
        reference_eigen=eigen,
        bootstrap=bootstrap,
        train_dates=manifest["split"]["design_dates"],
        evaluation_dates=manifest["split"]["heldout_dates"],
        n_observations=dimension,
        design_description=manifest["subset"]["method"],
    )
    print(
        json.dumps(
            json_ready(
                {
                    "output_dir": output_dir,
                    "fitted_null": fitted_null,
                    "boundary_parameters": null_fit.boundary_parameters,
                    "matrix_kl_per_observation": eigen.matrix_kl / dimension,
                    "bootstrap": bootstrap,
                    "total_seconds": elapsed,
                }
            ),
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
