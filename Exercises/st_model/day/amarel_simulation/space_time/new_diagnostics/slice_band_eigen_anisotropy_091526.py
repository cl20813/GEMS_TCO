#!/usr/bin/env python3
"""Directional band-slice eigenanalysis for three simulated and real GEMS days.

The proposed latitude slices are five one-degree latitude bands spanning
[-3, 2].  Every grid row inside a band supplies an east-west longitude
profile.  The primary longitude slices are five alternating one-degree bands
([122,123], [124,125], ..., [130,131]); every grid column inside a band
supplies a north-south latitude profile.  The complementary alternating set is
also evaluated as a robustness check.

Raw eigenvalues from the full east-west and north-south profiles are not
directly comparable because the two domains have different lengths and grid
resolutions.  The primary diagnostic therefore:

1. splits each 10-degree east-west profile into two 5-degree windows;
2. uses the full 5-degree north-south profile;
3. interpolates both directions to the same 80-point normalized coordinate;
4. pools the sample correlation by spatial lag to obtain an 80 x 80 stationary
   Toeplitz correlation matrix for every day and band;
5. eigendecomposes that matrix.

Pooling by lag is important here.  The latitude bands yield more profiles than
the longitude bands, so an unconstrained high-dimensional sample correlation
matrix has direction-dependent finite-sample eigenvalue spreading even under
the same covariance.  The lag-pooled Toeplitz estimate removes that artifact
and directly targets directional correlation decay.

The simulation residual uses the known generating mean.  The real-data mean
is fitted over the three selected days using an intercept, centered source
latitude, day fixed effects, and hour-slot fixed effects.  Persistent mean
profiles are additionally removed by the column centering implicit in each
sample correlation matrix.

This is a same-time spatial anisotropy diagnostic.  It cannot diagnose the
sign of advection, because reversing a spatial profile preserves the
eigenvalues of its covariance/correlation matrix.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
REPO = next(parent for parent in HERE.parents if (parent / "src/GEMS_TCO").is_dir())
DEFAULT_DATA_ROOT = Path("/Users/joonwonlee/Documents/GEMS_DATA")
DEFAULT_SIM_PATH = (
    DEFAULT_DATA_ROOT
    / "simulation/july_st_circulant_realpattern_smooth0p5"
    / "2024_july_st_circulant/sim_july2024_st_circulant_gridded.pkl"
)
DEFAULT_TRUTH_PATH = DEFAULT_SIM_PATH.with_name("sim_july2024_st_circulant_truth.json")
DEFAULT_REAL_PATH = DEFAULT_DATA_ROOT / "pickle_2024/tco_grid_24_07.pkl"
DEFAULT_OUTPUT = REPO / "outputs/summer_26/slice_band_eigen_anisotropy_202407_091526"

DAY_RE = re.compile(r"day(?P<day>\d+)_hm")
LAT_BANDS = [(-3.0, -2.0), (-2.0, -1.0), (-1.0, 0.0), (0.0, 1.0), (1.0, 2.0)]
LON_BANDS_PRIMARY = [
    (122.0, 123.0),
    (124.0, 125.0),
    (126.0, 127.0),
    (128.0, 129.0),
    (130.0, 131.0),
]
LON_BANDS_ALTERNATE = [
    (121.0, 122.0),
    (123.0, 124.0),
    (125.0, 126.0),
    (127.0, 128.0),
    (129.0, 130.0),
]
EW_WINDOWS = [(121.0, 126.0), (126.0, 131.0)]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--simulation", type=Path, default=DEFAULT_SIM_PATH)
    p.add_argument("--truth", type=Path, default=DEFAULT_TRUTH_PATH)
    p.add_argument("--real", type=Path, default=DEFAULT_REAL_PATH)
    p.add_argument("--days", nargs="+", type=int, default=[13, 19, 25])
    p.add_argument("--positions", type=int, default=80)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return p


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def key_day(key: str) -> int | None:
    match = DAY_RE.search(str(key))
    return None if match is None else int(match.group("day"))


def selected_keys(month: dict[str, pd.DataFrame], days: list[int]) -> dict[int, list[str]]:
    out: dict[int, list[str]] = {}
    for day in days:
        keys = sorted(key for key in month if key_day(str(key)) == day)
        if len(keys) != 8:
            raise RuntimeError(f"Expected 8 hourly fields for day {day}, got {len(keys)}")
        out[day] = keys
    return out


def grid_geometry(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    lats = np.sort(pd.to_numeric(frame["Latitude"], errors="coerce").dropna().unique())
    lons = np.sort(pd.to_numeric(frame["Longitude"], errors="coerce").dropna().unique())
    if len(lats) * len(lons) != len(frame):
        raise RuntimeError("The selected frame is not a complete rectangular grid")
    return lats.astype(float), lons.astype(float)


def real_mean_coefficients(
    month: dict[str, pd.DataFrame], keys_by_day: dict[int, list[str]], days: list[int]
) -> tuple[np.ndarray, list[str], float]:
    names = ["intercept", "source_lat_centered"]
    names += [f"day_{day}" for day in days[1:]]
    names += [f"hour_{hour}" for hour in range(1, 8)]
    latitude_values: list[np.ndarray] = []
    for keys in keys_by_day.values():
        for key in keys:
            values = pd.to_numeric(month[key]["Source_Latitude"], errors="coerce").to_numpy(float)
            latitude_values.append(values[np.isfinite(values)])
    latitude_center = float(np.mean(np.concatenate(latitude_values)))

    p = len(names)
    xtx = np.zeros((p, p), dtype=np.float64)
    xty = np.zeros(p, dtype=np.float64)
    for day_index, day in enumerate(days):
        for hour, key in enumerate(keys_by_day[day]):
            frame = month[key]
            y = pd.to_numeric(frame["ColumnAmountO3"], errors="coerce").to_numpy(float)
            source_lat = pd.to_numeric(
                frame["Source_Latitude"], errors="coerce"
            ).to_numpy(float)
            valid = np.isfinite(y) & np.isfinite(source_lat)
            x = np.zeros((int(valid.sum()), p), dtype=np.float64)
            x[:, 0] = 1.0
            x[:, 1] = source_lat[valid] - latitude_center
            if day_index > 0:
                x[:, 1 + day_index] = 1.0
            if hour > 0:
                x[:, 1 + (len(days) - 1) + hour] = 1.0
            xtx += x.T @ x
            xty += x.T @ y[valid]
    beta = np.linalg.solve(xtx, xty)
    return beta, names, latitude_center


def reconstruct_residuals(
    path: Path,
    days: list[int],
    kind: str,
    truth: dict[str, float] | None,
) -> tuple[dict[int, np.ndarray], np.ndarray, np.ndarray, dict[str, Any]]:
    month = pd.read_pickle(path)
    if not isinstance(month, dict):
        raise TypeError(f"Expected a dict pickle at {path}, got {type(month)}")
    keys_by_day = selected_keys(month, days)
    first = month[keys_by_day[days[0]][0]]
    lats, lons = grid_geometry(first)

    if kind == "real":
        beta, mean_names, latitude_center = real_mean_coefficients(
            month, keys_by_day, days
        )
    else:
        if truth is None:
            raise ValueError("Simulation truth is required")
        beta, mean_names = None, ["known_intercept", "known_latitude_slope"]
        latitude_center = float(truth["mean_lat_center"])

    residuals: dict[int, np.ndarray] = {}
    valid_counts: dict[str, int] = {}
    for day_index, day in enumerate(days):
        cube = np.full((8, len(lats), len(lons)), np.nan, dtype=np.float64)
        for hour, key in enumerate(keys_by_day[day]):
            frame = month[key]
            grid_lat = pd.to_numeric(frame["Latitude"], errors="coerce").to_numpy(float)
            grid_lon = pd.to_numeric(frame["Longitude"], errors="coerce").to_numpy(float)
            source_lat = pd.to_numeric(
                frame["Source_Latitude"], errors="coerce"
            ).to_numpy(float)
            values = pd.to_numeric(frame["ColumnAmountO3"], errors="coerce").to_numpy(float)
            valid = np.isfinite(values) & np.isfinite(source_lat)
            ii = np.searchsorted(lats, grid_lat[valid])
            jj = np.searchsorted(lons, grid_lon[valid])
            if kind == "simulation":
                assert truth is not None
                mean = float(truth["mean_intercept"]) + float(
                    truth["mean_lat_slope"]
                ) * (source_lat[valid] - float(truth["mean_lat_center"]))
            else:
                assert beta is not None
                x = np.zeros((int(valid.sum()), len(beta)), dtype=np.float64)
                x[:, 0] = 1.0
                x[:, 1] = source_lat[valid] - latitude_center
                if day_index > 0:
                    x[:, 1 + day_index] = 1.0
                if hour > 0:
                    x[:, 1 + (len(days) - 1) + hour] = 1.0
                mean = x @ beta
            cube[hour, ii, jj] = values[valid] - mean
            valid_counts[f"day{day:02d}_hour{hour}"] = int(valid.sum())
        residuals[day] = cube

    summary: dict[str, Any] = {
        "source": str(path),
        "kind": kind,
        "days": days,
        "grid_shape": [len(lats), len(lons)],
        "grid_latitude_range": [float(lats.min()), float(lats.max())],
        "grid_longitude_range": [float(lons.min()), float(lons.max())],
        "mean_model_columns": mean_names,
        "mean_latitude_center": latitude_center,
        "mean_coefficients": None if beta is None else beta.tolist(),
        "valid_counts": valid_counts,
    }
    del month
    return residuals, lats, lons, summary


def band_mask(values: np.ndarray, low: float, high: float, final: bool = False) -> np.ndarray:
    if final:
        return (values >= low - 1e-10) & (values <= high + 1e-10)
    return (values >= low - 1e-10) & (values < high - 1e-10)


def resample_profile(values: np.ndarray, positions: int) -> np.ndarray | None:
    values = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(values)
    if int(valid.sum()) < max(8, int(math.ceil(0.7 * len(values)))):
        return None
    source = np.linspace(0.0, 1.0, len(values), dtype=np.float64)
    target = np.linspace(0.0, 1.0, positions, dtype=np.float64)
    return np.interp(target, source[valid], values[valid])


def ew_profiles_for_band(
    cube: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    band: tuple[float, float],
    positions: int,
) -> np.ndarray:
    row_indexes = np.flatnonzero(
        band_mask(lats, *band, final=math.isclose(band[1], LAT_BANDS[-1][1]))
    )
    rows: list[np.ndarray] = []
    for hour in range(cube.shape[0]):
        for row in row_indexes:
            for window_index, (low, high) in enumerate(EW_WINDOWS):
                columns = band_mask(lons, low, high, final=window_index == len(EW_WINDOWS) - 1)
                profile = resample_profile(cube[hour, row, columns], positions)
                if profile is not None:
                    rows.append(profile)
    if not rows:
        raise RuntimeError(f"No east-west profiles in latitude band {band}")
    return np.stack(rows)


def ns_profiles_for_band(
    cube: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    band: tuple[float, float],
    positions: int,
) -> np.ndarray:
    final = math.isclose(band[1], 131.0)
    column_indexes = np.flatnonzero(band_mask(lons, *band, final=final))
    rows: list[np.ndarray] = []
    for hour in range(cube.shape[0]):
        for column in column_indexes:
            profile = resample_profile(cube[hour, :, column], positions)
            if profile is not None:
                rows.append(profile)
    if not rows:
        raise RuntimeError(f"No north-south profiles in longitude band {band}")
    return np.stack(rows)


def eigenspectrum(profiles: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    centered = profiles - np.mean(profiles, axis=0, keepdims=True)
    n_profiles, n_positions = centered.shape
    # The fixed denominator n_profiles*n_positions is the classical biased
    # autocovariance estimate.  It supplies a positive-semidefinite tapered
    # covariance sequence and applies exactly the same finite-window taper in
    # both directions.  Using n_positions-lag here can make the empirical
    # Toeplitz matrix indefinite.
    lag_covariance = np.asarray(
        [
            np.sum(centered[:, : n_positions - lag] * centered[:, lag:])
            / (n_profiles * n_positions)
            for lag in range(n_positions)
        ],
        dtype=np.float64,
    )
    if not np.isfinite(lag_covariance[0]) or lag_covariance[0] <= 0.0:
        raise RuntimeError("Non-positive pooled lag-zero variance")
    lag_correlation = lag_covariance / lag_covariance[0]
    offsets = np.abs(
        np.arange(n_positions)[:, None] - np.arange(n_positions)[None, :]
    )
    correlation = lag_correlation[offsets]
    correlation = 0.5 * (correlation + correlation.T)
    eigenvalues = np.linalg.eigvalsh(correlation)[::-1]
    minimum_eigenvalue = float(eigenvalues[-1])
    if minimum_eigenvalue < -1e-8:
        raise RuntimeError(
            f"Lag-pooled correlation is not PSD: minimum eigenvalue={minimum_eigenvalue}"
        )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    normalized = eigenvalues / np.sum(eigenvalues)
    positive_p = normalized[normalized > 0.0]
    cumulative = np.cumsum(normalized)

    def modes_for(level: float) -> int:
        return int(np.searchsorted(cumulative, level, side="left") + 1)

    metrics = {
        "n_profiles": int(len(profiles)),
        "n_positions": int(profiles.shape[1]),
        "mean_point_variance": float(np.mean(np.var(profiles, axis=0, ddof=1))),
        "pooled_lag_zero_variance": float(lag_covariance[0]),
        "lag1_correlation": float(lag_correlation[1]),
        "lag2_correlation": float(lag_correlation[2]),
        "minimum_toeplitz_eigenvalue": minimum_eigenvalue,
        "lambda1_fraction": float(normalized[0]),
        "top5_fraction": float(np.sum(normalized[:5])),
        "effective_rank_entropy": float(np.exp(-np.sum(positive_p * np.log(positive_p)))),
        "effective_rank_participation": float(1.0 / np.sum(normalized**2)),
        "spectral_rank_centroid": float(
            np.sum(normalized * np.arange(1, len(normalized) + 1)) / len(normalized)
        ),
        "modes_80": modes_for(0.80),
        "modes_90": modes_for(0.90),
        "modes_95": modes_for(0.95),
    }
    return eigenvalues, metrics


def analyze_direction(
    dataset: str,
    residuals: dict[int, np.ndarray],
    lats: np.ndarray,
    lons: np.ndarray,
    days: list[int],
    positions: int,
    lon_scheme: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    spectrum_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    lon_bands = LON_BANDS_PRIMARY if lon_scheme == "primary" else LON_BANDS_ALTERNATE
    directions = [
        ("east_west", "latitude_band", LAT_BANDS),
        ("north_south", "longitude_band", lon_bands),
    ]
    for day in days:
        cube = residuals[day]
        for direction, band_type, bands in directions:
            for band_index, band in enumerate(bands, start=1):
                if direction == "east_west":
                    profiles = ew_profiles_for_band(cube, lats, lons, band, positions)
                else:
                    profiles = ns_profiles_for_band(cube, lats, lons, band, positions)
                eigenvalues, metrics = eigenspectrum(profiles)
                total = float(np.sum(eigenvalues))
                common = {
                    "dataset": dataset,
                    "day": int(day),
                    "direction": direction,
                    "band_type": band_type,
                    "band_index": int(band_index),
                    "band_low": float(band[0]),
                    "band_high": float(band[1]),
                    "lon_scheme": lon_scheme,
                }
                metric_rows.append({**common, **metrics})
                cumulative = np.cumsum(eigenvalues / total)
                for rank, (value, cum) in enumerate(zip(eigenvalues, cumulative), start=1):
                    spectrum_rows.append(
                        {
                            **common,
                            "rank": int(rank),
                            "rank_fraction": float(rank / len(eigenvalues)),
                            "eigenvalue": float(value),
                            "eigenvalue_fraction": float(value / total),
                            "cumulative_fraction": float(cum),
                        }
                    )
    return pd.DataFrame(spectrum_rows), pd.DataFrame(metric_rows)


def daily_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "lambda1_fraction",
        "top5_fraction",
        "effective_rank_entropy",
        "effective_rank_participation",
        "spectral_rank_centroid",
        "modes_80",
        "modes_90",
        "modes_95",
        "mean_point_variance",
    ]
    primary = metrics[metrics["lon_scheme"].eq("primary")]
    rows: list[dict[str, Any]] = []
    for (dataset, day, direction), group in primary.groupby(
        ["dataset", "day", "direction"], sort=False
    ):
        row: dict[str, Any] = {
            "dataset": dataset,
            "day": int(day),
            "direction": direction,
            "n_bands": int(len(group)),
        }
        for column in columns:
            values = group[column].to_numpy(float)
            row[f"{column}_mean"] = float(np.mean(values))
            row[f"{column}_sd"] = float(np.std(values, ddof=1))
            row[f"{column}_min"] = float(np.min(values))
            row[f"{column}_max"] = float(np.max(values))
        rows.append(row)
    return pd.DataFrame(rows)


def anisotropy_summary(daily: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, day), group in daily.groupby(["dataset", "day"], sort=False):
        indexed = group.set_index("direction")
        ew = indexed.loc["east_west"]
        ns = indexed.loc["north_south"]
        rows.append(
            {
                "dataset": dataset,
                "day": int(day),
                "effective_rank_ew": float(ew["effective_rank_entropy_mean"]),
                "effective_rank_ns": float(ns["effective_rank_entropy_mean"]),
                "effective_rank_ratio_ew_over_ns": float(
                    ew["effective_rank_entropy_mean"]
                    / ns["effective_rank_entropy_mean"]
                ),
                "lambda1_fraction_ew": float(ew["lambda1_fraction_mean"]),
                "lambda1_fraction_ns": float(ns["lambda1_fraction_mean"]),
                "lambda1_ratio_ew_over_ns": float(
                    ew["lambda1_fraction_mean"] / ns["lambda1_fraction_mean"]
                ),
                "modes90_ew": float(ew["modes_90_mean"]),
                "modes90_ns": float(ns["modes_90_mean"]),
                "modes90_ratio_ew_over_ns": float(
                    ew["modes_90_mean"] / ns["modes_90_mean"]
                ),
            }
        )
    return pd.DataFrame(rows)


def aggregate_spectra(spectra: pd.DataFrame) -> pd.DataFrame:
    primary = spectra[spectra["lon_scheme"].eq("primary")]
    return (
        primary.groupby(["dataset", "direction", "rank", "rank_fraction"], as_index=False)
        .agg(
            eigenvalue_fraction_mean=("eigenvalue_fraction", "mean"),
            eigenvalue_fraction_min=("eigenvalue_fraction", "min"),
            eigenvalue_fraction_max=("eigenvalue_fraction", "max"),
            cumulative_fraction_mean=("cumulative_fraction", "mean"),
            cumulative_fraction_min=("cumulative_fraction", "min"),
            cumulative_fraction_max=("cumulative_fraction", "max"),
        )
        .sort_values(["dataset", "direction", "rank"])
    )


COLORS = {"east_west": "#2878B5", "north_south": "#D05A3A"}
LABELS = {"east_west": "E–W (longitude profiles)", "north_south": "N–S (latitude profiles)"}


def plot_spectra(aggregate: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)
    for row, dataset in enumerate(("simulation", "real")):
        for direction in ("east_west", "north_south"):
            part = aggregate[
                aggregate["dataset"].eq(dataset) & aggregate["direction"].eq(direction)
            ].sort_values("rank")
            rank = part["rank"].to_numpy(float)
            color = COLORS[direction]
            axes[row, 0].fill_between(
                rank,
                part["eigenvalue_fraction_min"].to_numpy(float),
                part["eigenvalue_fraction_max"].to_numpy(float),
                color=color,
                alpha=0.13,
            )
            axes[row, 0].plot(
                rank,
                part["eigenvalue_fraction_mean"],
                color=color,
                linewidth=2.0,
                label=LABELS[direction],
            )
            axes[row, 1].fill_between(
                rank,
                part["cumulative_fraction_min"].to_numpy(float),
                part["cumulative_fraction_max"].to_numpy(float),
                color=color,
                alpha=0.13,
            )
            axes[row, 1].plot(
                rank,
                part["cumulative_fraction_mean"],
                color=color,
                linewidth=2.0,
                label=LABELS[direction],
            )
        axes[row, 0].set_yscale("log")
        axes[row, 0].set(
            xlabel="eigenvalue rank (1 = largest)",
            ylabel="fraction of correlation trace",
            title=f"{dataset.capitalize()}: normalized eigenspectrum",
        )
        axes[row, 1].set(
            xlabel="eigenvalue rank (1 = largest)",
            ylabel="cumulative trace fraction",
            ylim=(0.0, 1.02),
            title=f"{dataset.capitalize()}: cumulative spectrum",
        )
        for axis in axes[row]:
            axis.grid(alpha=0.2)
            axis.legend(fontsize=9)
    fig.suptitle(
        "Matched 5-degree directional eigenanalysis\n"
        "mean curve and full range across 3 days × 5 bands",
        fontsize=14,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(metrics: pd.DataFrame, output: Path) -> None:
    primary = metrics[metrics["lon_scheme"].eq("primary")].copy()
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.5), constrained_layout=True)
    metric_specs = [
        ("effective_rank_entropy", "effective rank", None),
        ("lambda1_fraction", "largest-eigenvalue trace fraction", None),
    ]
    rng = np.random.default_rng(20260915)
    for row, dataset in enumerate(("simulation", "real")):
        subset = primary[primary["dataset"].eq(dataset)]
        for column, (metric, ylabel, ylim) in enumerate(metric_specs):
            axis = axes[row, column]
            for direction_index, direction in enumerate(("east_west", "north_south")):
                part = subset[subset["direction"].eq(direction)]
                for day_index, day in enumerate(sorted(part["day"].unique())):
                    values = part[part["day"].eq(day)][metric].to_numpy(float)
                    center = day_index + (direction_index - 0.5) * 0.22
                    jitter = rng.uniform(-0.045, 0.045, size=len(values))
                    axis.scatter(
                        np.full(len(values), center) + jitter,
                        values,
                        s=28,
                        color=COLORS[direction],
                        alpha=0.62,
                    )
                    axis.plot(
                        [center - 0.07, center + 0.07],
                        [np.mean(values), np.mean(values)],
                        color=COLORS[direction],
                        linewidth=3.0,
                    )
            days = sorted(part["day"].unique())
            axis.set_xticks(range(len(days)), [f"Jul {day}" for day in days])
            axis.set_ylabel(ylabel)
            axis.set_title(f"{dataset.capitalize()}: {ylabel}")
            if ylim is not None:
                axis.set_ylim(*ylim)
            axis.grid(axis="y", alpha=0.2)
            handles = [
                plt.Line2D([], [], color=COLORS[d], marker="o", linestyle="", label=LABELS[d])
                for d in ("east_west", "north_south")
            ]
            axis.legend(handles=handles, fontsize=8)
    fig.suptitle("Directional eigen concentration by day and 1-degree band", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_band_heatmaps(metrics: pd.DataFrame, output: Path) -> None:
    primary = metrics[metrics["lon_scheme"].eq("primary")]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.5), constrained_layout=True)
    all_values = primary["effective_rank_entropy"].to_numpy(float)
    vmin, vmax = float(np.min(all_values)), float(np.max(all_values))
    image = None
    for row, dataset in enumerate(("simulation", "real")):
        for column, direction in enumerate(("east_west", "north_south")):
            part = primary[
                primary["dataset"].eq(dataset) & primary["direction"].eq(direction)
            ]
            days = sorted(part["day"].unique())
            matrix = np.full((len(days), 5), np.nan)
            labels: list[str] = []
            for day_index, day in enumerate(days):
                day_part = part[part["day"].eq(day)].sort_values("band_index")
                matrix[day_index] = day_part["effective_rank_entropy"].to_numpy(float)
                if not labels:
                    labels = [
                        f"{low:g}–{high:g}" for low, high in zip(day_part["band_low"], day_part["band_high"])
                    ]
            axis = axes[row, column]
            image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    axis.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color="white", fontsize=9)
            axis.set_xticks(range(5), labels, rotation=25, ha="right")
            axis.set_yticks(range(len(days)), [f"Jul {day}" for day in days])
            axis.set_xlabel("latitude band" if direction == "east_west" else "longitude band")
            axis.set_ylabel("day")
            axis.set_title(f"{dataset.capitalize()} · {LABELS[direction]}")
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, shrink=0.88)
    colorbar.set_label("effective rank")
    fig.suptitle("Spatial stability of directional effective rank", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def write_report(
    output: Path,
    anisotropy: pd.DataFrame,
    metrics: pd.DataFrame,
    truth: dict[str, float],
) -> None:
    lines = [
        "# Latitude/longitude band eigenanalysis",
        "",
        "## Design",
        "",
        "- Days: July 13, 19, and 25, 2024 (eight hourly fields per day).",
        "- Latitude bands: [-3,-2), [-2,-1), [-1,0), [0,1), [1,2].",
        "- Primary alternating longitude bands: [122,123), [124,125), [126,127), [128,129), [130,131].",
        "- East-west 10-degree profiles are split into two 5-degree windows; both directions are resampled to 80 positions.",
        "- Within each day and band, correlations are pooled by spatial lag to form an 80x80 stationary Toeplitz correlation matrix before eigendecomposition.",
        "- Lag pooling avoids the finite-sample eigenvalue-spreading artifact caused by the unequal numbers of E-W and N-S profiles.",
        "- Lower effective rank and fewer modes for 90% trace indicate stronger concentration in smooth leading modes.",
        "",
        "## Simulation truth",
        "",
        f"- range_lat={truth['range_lat']}, range_lon={truth['range_lon']}; expected E-W smoothness is greater because range_lon/range_lat={truth['range_lon']/truth['range_lat']:.3f}.",
        "- This same-time analysis does not test advection sign.",
        "",
        "## Daily direction averages",
        "",
        "| dataset | day | eff.rank E-W | eff.rank N-S | E-W/N-S | lambda1 E-W | lambda1 N-S | modes90 E-W | modes90 N-S |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in anisotropy.itertuples(index=False):
        lines.append(
            f"| {row.dataset} | {row.day} | {row.effective_rank_ew:.3f} | "
            f"{row.effective_rank_ns:.3f} | {row.effective_rank_ratio_ew_over_ns:.3f} | "
            f"{row.lambda1_fraction_ew:.4f} | {row.lambda1_fraction_ns:.4f} | "
            f"{row.modes90_ew:.2f} | {row.modes90_ns:.2f} |"
        )
    lines += ["", "## Overall three-day summary", ""]
    for dataset, group in anisotropy.groupby("dataset", sort=False):
        ratio = group["effective_rank_ratio_ew_over_ns"].to_numpy(float)
        lambda_ratio = group["lambda1_ratio_ew_over_ns"].to_numpy(float)
        lines.append(
            f"- {dataset}: mean E-W/N-S effective-rank ratio={np.mean(ratio):.3f} "
            f"(daily range {np.min(ratio):.3f}–{np.max(ratio):.3f}); "
            f"mean lambda1 ratio={np.mean(lambda_ratio):.3f}."
        )

    robustness = metrics[
        metrics["direction"].eq("north_south")
    ].groupby(["dataset", "lon_scheme"])["effective_rank_entropy"].mean()
    lines += ["", "## Longitude-band offset robustness", ""]
    for dataset in ("simulation", "real"):
        primary = float(robustness.loc[(dataset, "primary")])
        alternate = float(robustness.loc[(dataset, "alternate")])
        lines.append(
            f"- {dataset}: primary={primary:.3f}, alternate={alternate:.3f}, "
            f"relative difference={(alternate-primary)/primary:.2%}."
        )
    lines += [
        "",
        "## Interpretation guardrails",
        "",
        "- Raw eigenvalue magnitudes from the original 10-degree E-W and 5-degree N-S domains are not comparable.",
        "- These spectra are descriptive because neighboring profiles and hours are correlated.",
        "- Axis anisotropy is supported when the direction contrast is stable across days and bands and recovers the known simulation ordering.",
        "- Signed transport asymmetry still requires positive-time-lag h versus -h diagnostics or odd-odd space-time contrasts.",
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parser().parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    truth = json.loads(args.truth.read_text(encoding="utf-8"))

    all_spectra: list[pd.DataFrame] = []
    all_metrics: list[pd.DataFrame] = []
    source_summaries: dict[str, Any] = {}
    for dataset, path, kind in (
        ("simulation", args.simulation, "simulation"),
        ("real", args.real, "real"),
    ):
        residuals, lats, lons, source_summary = reconstruct_residuals(
            path,
            list(args.days),
            kind,
            truth if kind == "simulation" else None,
        )
        source_summaries[dataset] = source_summary
        for scheme in ("primary", "alternate"):
            spectra, metrics = analyze_direction(
                dataset,
                residuals,
                lats,
                lons,
                list(args.days),
                int(args.positions),
                scheme,
            )
            # East-west results do not depend on longitude-band scheme; retain
            # them only once so summaries contain five rather than ten copies.
            if scheme == "alternate":
                spectra = spectra[spectra["direction"].eq("north_south")]
                metrics = metrics[metrics["direction"].eq("north_south")]
            all_spectra.append(spectra)
            all_metrics.append(metrics)
        del residuals

    spectra = pd.concat(all_spectra, ignore_index=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    daily = daily_summary(metrics)
    anisotropy = anisotropy_summary(daily)
    aggregate = aggregate_spectra(spectra)

    atomic_csv(args.output / "slice_eigenspectra.csv", spectra)
    atomic_csv(args.output / "slice_eigen_metrics.csv", metrics)
    atomic_csv(args.output / "daily_direction_summary.csv", daily)
    atomic_csv(args.output / "daily_anisotropy_summary.csv", anisotropy)
    atomic_csv(args.output / "aggregate_direction_spectra.csv", aggregate)
    atomic_json(
        args.output / "analysis_manifest.json",
        {
            "analysis": "matched-length latitude/longitude band eigenanalysis",
            "days": list(args.days),
            "positions": int(args.positions),
            "latitude_bands": LAT_BANDS,
            "longitude_bands_primary": LON_BANDS_PRIMARY,
            "longitude_bands_alternate": LON_BANDS_ALTERNATE,
            "east_west_windows": EW_WINDOWS,
            "correlation_not_covariance": True,
            "correlation_estimator": "biased lag-pooled stationary Toeplitz correlation",
            "sources": source_summaries,
            "simulation_truth": truth,
        },
    )
    plot_spectra(aggregate, args.output / "directional_eigenspectra.png")
    plot_metrics(metrics, args.output / "directional_eigen_metrics_by_day.png")
    plot_band_heatmaps(metrics, args.output / "directional_effective_rank_heatmaps.png")
    write_report(args.output, anisotropy, metrics, truth)
    print(anisotropy.to_string(index=False))
    print(f"Wrote results to {args.output}")


if __name__ == "__main__":
    main()
