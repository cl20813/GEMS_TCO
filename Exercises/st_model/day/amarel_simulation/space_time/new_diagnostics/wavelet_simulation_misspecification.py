#!/usr/bin/env python3
"""Validate the wavelet residual-energy diagnostic on simulated GEMS data.

The stored 2024-07-13 smoothness-0.5 simulation supplies the regular GEMS
geometry, the realistic missing-cell mask, one representative realization,
and the data-generating parameters.  Six *fixed* covariance specifications
are compared without fitting:

1. the true model;
2. longitude range 0.5 times truth;
3. longitude range 2 times truth;
4. rough Matérn smoothness 0.3;
5. smooth Matérn smoothness 1.0;
6. nugget 0 instead of the true nugget 1.

The arbitrary-smoothness correlations use the cubic-spline Matérn table from
``GEMS_TCO.vecchia_st_spline``.  Every other parameter stays at truth.

For each assumed model, simulation estimates the model variance of every
masked wavelet coefficient.  The diagnostic energy is

    observed coefficient squared / assumed-model expected coefficient squared.

Its expectation is one under the assumed model.  Independent null simulations
give 95% envelopes.  A separate batch of controlled true-model realizations is
then reused for every candidate to estimate bandwise rejection rates.  Thus the
output distinguishes a visually interesting single realization from repeated-
simulation localization performance.

No covariance matrix, precision matrix, eigendecomposition, Lanczos iteration,
or SLQ approximation is used in this experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
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


HERE = Path(__file__).resolve().parent
REPO = next(parent for parent in HERE.parents if (parent / "src/GEMS_TCO").is_dir())
SRC = REPO / "src"
SPLINE_SRC = SRC / "GEMS_TCO"
if str(SPLINE_SRC) not in sys.path:
    sys.path.insert(0, str(SPLINE_SRC))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from matern_spline import _build_matern_spline_coeffs  # noqa: E402
import wavelet_residual_energy_one_day as wave  # noqa: E402


SCENARIOS = (
    "true",
    "lon_short",
    "lon_long",
    "smooth_rough",
    "smooth_smoother",
    "nugget_zero",
)
SCENARIO_LABELS = {
    "true": "true model",
    "lon_short": "longitude range 0.5x",
    "lon_long": "longitude range 2x",
    "smooth_rough": "smoothness 0.3",
    "smooth_smoother": "smoothness 1.0",
    "nugget_zero": "nugget 0 (truth 1)",
}
SCENARIO_COLORS = {
    "true": "#169873",
    "lon_short": "#2878B5",
    "lon_long": "#6F4BA3",
    "smooth_rough": "#E17C05",
    "smooth_smoother": "#B23A48",
    "nugget_zero": "#6B6B6B",
}
EXPECTED_PRIMARY_BAND = {
    "true": "none",
    "lon_short": "low",
    "lon_long": "low",
    "smooth_rough": "high",
    "smooth_smoother": "high",
    "nugget_zero": "high",
}
ORIENTATIONS = (*wave.ORIENTATIONS, "all")
EPS = 1e-12


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2024-07-13")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(
            "/Users/joonwonlee/Documents/GEMS_DATA/simulation/"
            "july_st_circulant_realpattern_smooth0p5"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            REPO
            / "outputs/summer_26/"
            "wavelet_sim_20240713_true_lon_smooth_nugget_misspecification"
        ),
    )
    parser.add_argument("--delta-lat", type=float, default=0.044)
    parser.add_argument("--delta-lon", type=float, default=0.063)
    parser.add_argument("--hours-per-day", type=int, default=8)
    parser.add_argument("--wavelet", default="db2")
    parser.add_argument("--wavelet-level", type=int, default=3)
    parser.add_argument("--wavelet-mode", default="periodization")
    parser.add_argument("--calibration-simulations", type=int, default=32)
    parser.add_argument("--envelope-simulations", type=int, default=64)
    parser.add_argument("--power-simulations", type=int, default=64)
    parser.add_argument("--simulation-batch-size", type=int, default=4)
    parser.add_argument("--curve-points", type=int, default=256)
    parser.add_argument("--random-seed", type=int, default=20260909)
    parser.add_argument("--spline-n-points", type=int, default=4000)
    parser.add_argument("--spline-r-max", type=float, default=30.0)
    parser.add_argument("--max-embedding-factor", type=int, default=3)
    parser.add_argument("--negative-mass-tolerance", type=float, default=2e-4)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.wavelet_level != 3:
        raise ValueError("Use wavelet level 3 so D3/D2/D1 are low/middle/high")
    if args.calibration_simulations <= 2:
        raise ValueError("At least three calibration simulations are required")
    if min(args.envelope_simulations, args.power_simulations) < 16:
        raise ValueError("Use at least 16 envelope and power simulations")
    if args.simulation_batch_size <= 0 or args.curve_points < 32:
        raise ValueError("Batch size must be positive and curve points at least 32")
    if args.max_embedding_factor < 2:
        raise ValueError("The maximum embedding factor must be at least 2")


def simulation_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    year = int(args.date[:4])
    directory = args.data_root / f"{year}_july_st_circulant"
    prefix = f"sim_july{year}_st_circulant"
    return directory / f"{prefix}_gridded.pkl", directory / f"{prefix}_truth.json"


def load_truth(path: Path) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    keys = (
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
    missing = [key for key in keys if key not in raw]
    if missing:
        raise RuntimeError(f"Simulation truth is missing {missing}: {path}")
    return {key: float(raw[key]) for key in keys}


def load_stored_simulated_day(
    data_path: Path,
    date: str,
    truth: dict[str, float],
    hours_per_day: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    month = pd.read_pickle(data_path)
    if not isinstance(month, dict):
        raise TypeError(f"Expected a dict pickle, got {type(month)}")
    day_keys = [key for key in sorted(month) if wave.date_from_key(str(key)) == date]
    if len(day_keys) != int(hours_per_day):
        raise RuntimeError(
            f"Expected {hours_per_day} hourly fields on {date}, got {len(day_keys)}"
        )
    frames = [month[key] for key in day_keys]
    del month

    lats = np.sort(
        np.unique(pd.to_numeric(frames[0]["Latitude"], errors="coerce").to_numpy(float))
    )
    lons = np.sort(
        np.unique(pd.to_numeric(frames[0]["Longitude"], errors="coerce").to_numpy(float))
    )
    shape = (len(day_keys), len(lats), len(lons))
    residual = np.zeros(shape, dtype=np.float64)
    masks = np.zeros(shape, dtype=bool)
    counts: list[int] = []
    for hour, (key, frame) in enumerate(zip(day_keys, frames)):
        grid_lat = pd.to_numeric(frame["Latitude"], errors="coerce").to_numpy(float)
        grid_lon = pd.to_numeric(frame["Longitude"], errors="coerce").to_numpy(float)
        source_lat = pd.to_numeric(
            frame["Source_Latitude"], errors="coerce"
        ).to_numpy(float)
        values = pd.to_numeric(frame["ColumnAmountO3"], errors="coerce").to_numpy(float)
        ii = np.searchsorted(lats, grid_lat)
        jj = np.searchsorted(lons, grid_lon)
        if not (np.allclose(lats[ii], grid_lat) and np.allclose(lons[jj], grid_lon)):
            raise RuntimeError(f"Grid-index reconstruction failed for {key}")
        valid = np.isfinite(values) & np.isfinite(source_lat)
        mean = truth["mean_intercept"] + truth["mean_lat_slope"] * (
            source_lat[valid] - truth["mean_lat_center"]
        )
        residual[hour, ii[valid], jj[valid]] = values[valid] - mean
        masks[hour, ii[valid], jj[valid]] = True
        counts.append(int(valid.sum()))
    summary = {
        "source": str(data_path),
        "date": date,
        "hour_keys": [str(key) for key in day_keys],
        "hourly_valid_counts": counts,
        "valid_count": int(sum(counts)),
        "grid_cells": int(np.prod(shape)),
        "valid_fraction": float(sum(counts) / np.prod(shape)),
        "residual_mean_removed": (
            "known generating mean at Source_Latitude: intercept + latitude slope"
        ),
    }
    return residual, masks, lats, lons, [str(key) for key in day_keys], summary


def candidate_parameters(truth: dict[str, float]) -> dict[str, dict[str, float]]:
    keys = (
        "smooth",
        "sigmasq",
        "range_lat",
        "range_lon",
        "range_time",
        "advec_lat",
        "advec_lon",
        "nugget",
    )
    base = {key: float(truth[key]) for key in keys}
    candidates = {"true": dict(base)}
    candidates["lon_short"] = dict(base, range_lon=0.5 * base["range_lon"])
    candidates["lon_long"] = dict(base, range_lon=2.0 * base["range_lon"])
    candidates["smooth_rough"] = dict(base, smooth=0.3)
    candidates["smooth_smoother"] = dict(base, smooth=1.0)
    candidates["nugget_zero"] = dict(base, nugget=0.0)
    return candidates


def spline_matern_correlation(
    distance: np.ndarray,
    smooth: float,
    n_points: int,
    r_max: float,
) -> np.ndarray:
    if np.isclose(smooth, 0.5):
        return np.exp(-distance)
    coeffs = _build_matern_spline_coeffs(
        float(smooth), n_points=int(n_points), r_max=float(r_max)
    )
    clipped = np.clip(distance, 0.0, float(coeffs["r_max"]))
    flat = clipped.reshape(-1)
    knots = coeffs["knots"]
    index = np.searchsorted(knots, flat, side="right") - 1
    index = np.clip(index, 0, len(knots) - 2)
    dx = flat - knots[index]
    values = coeffs["a"][index] + dx * (
        coeffs["b"][index]
        + dx * (coeffs["c"][index] + dx * coeffs["d"][index])
    )
    values = np.clip(values, 0.0, 1.0)
    values[flat >= float(coeffs["r_max"])] = 0.0
    return values.reshape(distance.shape)


def odd_embedding_size(target: int, factor: int) -> int:
    size = int(factor) * (int(target) - 1) + 1
    return size if size % 2 == 1 else size + 1


def circulant_spectrum_spline(
    params: dict[str, float],
    target_shape: tuple[int, int, int],
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_lat, n_lon, n_time = target_shape
    attempts: list[dict[str, Any]] = []
    for factor in range(2, int(args.max_embedding_factor) + 1):
        embed_shape = tuple(
            odd_embedding_size(size, factor) for size in (n_lat, n_lon, n_time)
        )
        axes = []
        for size, step in zip(
            embed_shape, (float(args.delta_lat), float(args.delta_lon), 1.0)
        ):
            index = np.arange(size, dtype=np.float64)
            index[index > size // 2] -= size
            axes.append(index * step)
        h_lat, h_lon, h_time = np.meshgrid(*axes, indexing="ij", sparse=True)
        distance = np.sqrt(
            ((h_lat - params["advec_lat"] * h_time) / params["range_lat"]) ** 2
            + ((h_lon - params["advec_lon"] * h_time) / params["range_lon"]) ** 2
            + (h_time / params["range_time"]) ** 2
        )
        covariance = params["sigmasq"] * spline_matern_correlation(
            distance,
            params["smooth"],
            int(args.spline_n_points),
            float(args.spline_r_max),
        )
        covariance[0, 0, 0] += params["nugget"] + 1e-6
        eigenvalues = np.fft.fftn(covariance).real
        positive_mass = float(np.maximum(eigenvalues, 0.0).sum())
        negative_mass = float(-np.minimum(eigenvalues, 0.0).sum())
        attempt = {
            "factor": factor,
            "embedding_shape": list(embed_shape),
            "minimum_eigenvalue_before_clipping": float(eigenvalues.min()),
            "negative_eigenvalue_fraction": float(np.mean(eigenvalues < 0.0)),
            "negative_mass_fraction": negative_mass / max(positive_mass, EPS),
        }
        attempts.append(attempt)
        if attempt["negative_mass_fraction"] <= float(args.negative_mass_tolerance):
            return np.sqrt(np.maximum(eigenvalues, 0.0)), {
                **attempt,
                "attempts": attempts,
                "model_marginal_variance": float(
                    params["sigmasq"] + params["nugget"] + 1e-6
                ),
                "matern_evaluator": (
                    "exp(-r) exact for smooth=0.5; otherwise cubic spline from "
                    "GEMS_TCO.vecchia_st_spline._build_matern_spline_coeffs"
                ),
            }
        del covariance, eigenvalues, distance
    raise RuntimeError(
        "Circulant embedding retained material negative spectral mass: "
        + json.dumps(attempts)
    )


def curve_indices(size: int, points: int) -> np.ndarray:
    return np.unique(np.linspace(0, size - 1, min(points, size)).astype(np.int64))


def energy_by_band(
    normalized: dict[str, np.ndarray], valid: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {band: normalized[band][valid[band]] for band in wave.BANDS}


def calibrate_candidate(
    scenario: str,
    params: dict[str, float],
    observed_details: dict[str, np.ndarray],
    masks: np.ndarray,
    target_shape: tuple[int, int, int],
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    spectral_sqrt, embedding = circulant_spectrum_spline(params, target_shape, args)
    print(
        f"  embedding x{embedding['factor']} {tuple(embedding['embedding_shape'])}; "
        f"negative mass={embedding['negative_mass_fraction']:.2e}",
        flush=True,
    )
    calibration_rng = np.random.default_rng(
        wave.stable_seed(args.random_seed, "common", "calibration")
    )
    expected_square = wave.calibration_expectations(
        spectral_sqrt,
        target_shape,
        masks,
        observed_details,
        args,
        calibration_rng,
    )
    valid = wave.valid_expected_masks(expected_square)
    _, observed_normalized = wave.detail_energy_vector(
        observed_details, expected_square, valid
    )
    observed_summary = wave.summarize_normalized(observed_normalized)
    observed_band = energy_by_band(observed_normalized, valid)
    observed_global = np.concatenate([observed_band[band] for band in wave.BANDS])
    global_idx = curve_indices(len(observed_global), int(args.curve_points))
    band_idx = {
        band: curve_indices(len(observed_band[band]), int(args.curve_points))
        for band in wave.BANDS
    }

    reference_ratios = {
        (band, orientation): []
        for band in wave.BANDS
        for orientation in ORIENTATIONS
    }
    reference_global_curves: list[np.ndarray] = []
    reference_band_curves = {band: [] for band in wave.BANDS}
    envelope_rng = np.random.default_rng(
        wave.stable_seed(args.random_seed, "common", "envelope")
    )
    completed = 0
    for batch in wave.simulation_batches(
        spectral_sqrt,
        target_shape,
        int(args.envelope_simulations),
        int(args.simulation_batch_size),
        envelope_rng,
    ):
        for cube in batch:
            details = wave.wavelet_details(
                cube, masks, args.wavelet, args.wavelet_level, args.wavelet_mode
            )
            global_energy, normalized = wave.detail_energy_vector(
                details, expected_square, valid
            )
            summary = wave.summarize_normalized(normalized)
            for key, value in summary.items():
                reference_ratios[key].append(value)
            cumulative = np.cumsum(global_energy) / len(global_energy)
            reference_global_curves.append(cumulative[global_idx])
            values_by_band = energy_by_band(normalized, valid)
            for band in wave.BANDS:
                values = values_by_band[band]
                band_cumulative = np.cumsum(values) / len(values)
                reference_band_curves[band].append(band_cumulative[band_idx[band]])
            completed += 1
        print(
            f"      null envelope {completed}/{args.envelope_simulations}", flush=True
        )

    reference_ratios_array = {
        key: np.asarray(values, dtype=np.float64)
        for key, values in reference_ratios.items()
    }
    summary_rows = []
    for band in wave.BANDS:
        for orientation in ORIENTATIONS:
            reference = reference_ratios_array[(band, orientation)]
            observed = observed_summary[(band, orientation)]
            summary_rows.append(
                {
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "expected_primary_band": EXPECTED_PRIMARY_BAND[scenario],
                    "band": band,
                    "orientation": orientation,
                    "orientation_label": wave.ORIENTATION_LABELS[orientation],
                    "n_coefficients": int(
                        valid[band].sum()
                        if orientation == "all"
                        else valid[band][:, wave.ORIENTATIONS.index(orientation)].sum()
                    ),
                    "stored_day_energy_ratio": observed,
                    "null_mean": float(np.mean(reference)),
                    "null_q025": float(np.quantile(reference, 0.025)),
                    "null_q975": float(np.quantile(reference, 0.975)),
                    "stored_day_two_sided_p": wave.two_sided_monte_carlo_p(
                        observed, reference
                    ),
                }
            )

    global_curves = np.asarray(reference_global_curves)
    global_x = (global_idx + 1) / len(observed_global)
    global_observed = np.cumsum(observed_global) / len(observed_global)
    global_curve = pd.DataFrame(
        {
            "scenario": scenario,
            "mode_fraction": global_x,
            "observed_cumulative_energy": global_observed[global_idx],
            "expected_cumulative_energy": global_x,
            "null_q025": np.quantile(global_curves, 0.025, axis=0),
            "null_q975": np.quantile(global_curves, 0.975, axis=0),
        }
    )
    local_curve_frames = []
    for band in wave.BANDS:
        index = band_idx[band]
        values = observed_band[band]
        x = (index + 1) / len(values)
        cumulative = np.cumsum(values) / len(values)
        references = np.asarray(reference_band_curves[band])
        local_curve_frames.append(
            pd.DataFrame(
                {
                    "scenario": scenario,
                    "band": band,
                    "coefficient_fraction": x,
                    "observed_cumulative_energy": cumulative[index],
                    "expected_cumulative_energy": x,
                    "null_q025": np.quantile(references, 0.025, axis=0),
                    "null_q975": np.quantile(references, 0.975, axis=0),
                }
            )
        )
    return {
        "spectral_sqrt": spectral_sqrt if scenario == "true" else None,
        "embedding": embedding,
        "expected_square": expected_square,
        "valid": valid,
        "reference_ratios": reference_ratios_array,
        "summary": pd.DataFrame(summary_rows),
        "global_curve": global_curve,
        "local_curves": pd.concat(local_curve_frames, ignore_index=True),
        "runtime_seconds": time.perf_counter() - started,
    }


def evaluate_repeated_power(
    calibrated: dict[str, dict[str, Any]],
    truth_spectral_sqrt: np.ndarray,
    masks: np.ndarray,
    target_shape: tuple[int, int, int],
    args: argparse.Namespace,
) -> pd.DataFrame:
    values = {
        (scenario, band, orientation): []
        for scenario in SCENARIOS
        for band in wave.BANDS
        for orientation in ORIENTATIONS
    }
    rng = np.random.default_rng(wave.stable_seed(args.random_seed, "true", "power"))
    completed = 0
    for batch in wave.simulation_batches(
        truth_spectral_sqrt,
        target_shape,
        int(args.power_simulations),
        int(args.simulation_batch_size),
        rng,
    ):
        for cube in batch:
            details = wave.wavelet_details(
                cube, masks, args.wavelet, args.wavelet_level, args.wavelet_mode
            )
            for scenario in SCENARIOS:
                candidate = calibrated[scenario]
                _, normalized = wave.detail_energy_vector(
                    details, candidate["expected_square"], candidate["valid"]
                )
                summary = wave.summarize_normalized(normalized)
                for key, value in summary.items():
                    values[(scenario, *key)].append(value)
            completed += 1
        print(f"  repeated true-model days {completed}/{args.power_simulations}", flush=True)

    rows = []
    for scenario in SCENARIOS:
        for band in wave.BANDS:
            for orientation in ORIENTATIONS:
                estimates = np.asarray(values[(scenario, band, orientation)])
                null = calibrated[scenario]["reference_ratios"][(band, orientation)]
                lower = float(np.quantile(null, 0.025))
                upper = float(np.quantile(null, 0.975))
                rows.append(
                    {
                        "scenario": scenario,
                        "scenario_label": SCENARIO_LABELS[scenario],
                        "expected_primary_band": EXPECTED_PRIMARY_BAND[scenario],
                        "band": band,
                        "orientation": orientation,
                        "power_simulations": len(estimates),
                        "mean_true_dgp_energy_ratio": float(np.mean(estimates)),
                        "median_true_dgp_energy_ratio": float(np.median(estimates)),
                        "q025_true_dgp_energy_ratio": float(np.quantile(estimates, 0.025)),
                        "q975_true_dgp_energy_ratio": float(np.quantile(estimates, 0.975)),
                        "candidate_null_q025": lower,
                        "candidate_null_q975": upper,
                        "two_sided_rejection_rate": float(
                            np.mean((estimates < lower) | (estimates > upper))
                        ),
                        "low_side_rejection_rate": float(np.mean(estimates < lower)),
                        "high_side_rejection_rate": float(np.mean(estimates > upper)),
                    }
                )
    return pd.DataFrame(rows)


def plot_global_curves(
    curves: pd.DataFrame,
    output: Path,
    date: str,
    band_boundaries: tuple[float, float],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0), constrained_layout=True)
    for axis, scenario in zip(axes.flat, SCENARIOS):
        part = curves[curves["scenario"].eq(scenario)].sort_values("mode_fraction")
        x = part["mode_fraction"].to_numpy(float)
        axis.fill_between(
            x,
            part["null_q025"].to_numpy(float),
            part["null_q975"].to_numpy(float),
            color=SCENARIO_COLORS[scenario],
            alpha=0.14,
            label="candidate-model 95% envelope",
        )
        axis.plot(
            x,
            part["observed_cumulative_energy"].to_numpy(float),
            color=SCENARIO_COLORS[scenario],
            linewidth=2.0,
            label="stored simulated day",
        )
        axis.plot([0, 1], [0, 1], "--", color="0.25", linewidth=1.0, label="expected y=x")
        for boundary in band_boundaries:
            axis.axvline(boundary, color="0.55", linewidth=0.8)
        positions = (
            band_boundaries[0] / 2,
            sum(band_boundaries) / 2,
            (band_boundaries[1] + 1.0) / 2,
        )
        for position, band in zip(positions, wave.BANDS):
            axis.text(
                position,
                0.98,
                band,
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                color="0.4",
                fontsize=8,
            )
        axis.set(
            xlim=(0, 1),
            xlabel="wavelet coefficient fraction (D3 -> D2 -> D1)",
            ylabel="cumulative standardized energy",
            title=SCENARIO_LABELS[scenario],
        )
        axis.grid(alpha=0.18)
    axes.flat[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(
        f"Simulated {date}: global wavelet residual-energy checks", fontsize=15
    )
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_local_curves(
    curves: pd.DataFrame,
    summary: pd.DataFrame,
    power: pd.DataFrame,
    output: Path,
    date: str,
) -> None:
    fig, axes = plt.subplots(
        len(SCENARIOS), 3, figsize=(15.5, 22.0), constrained_layout=True, sharex=True
    )
    background = {"low": "#E6F2FF", "middle": "#EEEAFE", "high": "#FFECE5"}
    for row, scenario in enumerate(SCENARIOS):
        for column, band in enumerate(wave.BANDS):
            axis = axes[row, column]
            axis.set_facecolor(background[band])
            part = curves[
                curves["scenario"].eq(scenario) & curves["band"].eq(band)
            ].sort_values("coefficient_fraction")
            x = part["coefficient_fraction"].to_numpy(float)
            axis.fill_between(
                x,
                part["null_q025"].to_numpy(float),
                part["null_q975"].to_numpy(float),
                color=SCENARIO_COLORS[scenario],
                alpha=0.13,
            )
            axis.plot(
                x,
                part["observed_cumulative_energy"].to_numpy(float),
                color=SCENARIO_COLORS[scenario],
                linewidth=1.8,
            )
            axis.plot([0, 1], [0, 1], "--", color="0.25", linewidth=0.9)
            observed = summary[
                summary["scenario"].eq(scenario)
                & summary["band"].eq(band)
                & summary["orientation"].eq("all")
            ]["stored_day_energy_ratio"].iloc[0]
            rejection = power[
                power["scenario"].eq(scenario)
                & power["band"].eq(band)
                & power["orientation"].eq("all")
            ]["two_sided_rejection_rate"].iloc[0]
            axis.set(
                xlim=(0, 1),
                xlabel="within-band coefficient fraction",
                ylabel="cumulative standardized energy",
                title=(
                    f"{wave.BAND_LABELS[band]} | stored R={observed:.2f}, "
                    f"power={100 * rejection:.0f}%"
                ),
            )
            if column == 0:
                axis.text(
                    -0.24,
                    0.5,
                    SCENARIO_LABELS[scenario],
                    rotation=90,
                    va="center",
                    ha="center",
                    transform=axis.transAxes,
                    fontsize=11,
                    fontweight="bold" if scenario == "true" else "normal",
                )
            axis.grid(alpha=0.16)
    fig.suptitle(
        f"Simulated {date}: localized wavelet diagnostic\n"
        "line = stored realization; shading = assumed-model 95% envelope; "
        "power = rejection over independent true-model days",
        fontsize=15,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def annotated_heatmap(
    axis: plt.Axes,
    values: np.ndarray,
    title: str,
    fmt: str,
    cmap: str,
    vmin: float,
    vmax: float,
    center_labels: bool = False,
) -> Any:
    image = axis.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            label = fmt.format(values[row, column])
            color = "white" if abs(values[row, column] - (0.5 if center_labels else 0.0)) > 0.35 else "black"
            axis.text(column, row, label, ha="center", va="center", color=color, fontsize=9)
    axis.set_xticks(range(3), ["low D3", "middle D2", "high D1"])
    axis.set_yticks(range(len(SCENARIOS)), [SCENARIO_LABELS[item] for item in SCENARIOS])
    axis.set_title(title)
    return image


def plot_performance_heatmaps(
    summary: pd.DataFrame,
    power: pd.DataFrame,
    output: Path,
    date: str,
) -> None:
    stored = np.empty((len(SCENARIOS), 3), dtype=float)
    rejection = np.empty_like(stored)
    for row, scenario in enumerate(SCENARIOS):
        for column, band in enumerate(wave.BANDS):
            stored[row, column] = summary[
                summary["scenario"].eq(scenario)
                & summary["band"].eq(band)
                & summary["orientation"].eq("all")
            ]["stored_day_energy_ratio"].iloc[0]
            rejection[row, column] = power[
                power["scenario"].eq(scenario)
                & power["band"].eq(band)
                & power["orientation"].eq("all")
            ]["two_sided_rejection_rate"].iloc[0]
    log_ratio = np.log2(np.maximum(stored, EPS))
    limit = max(0.5, float(np.ceil(10 * np.max(np.abs(log_ratio))) / 10))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.3), constrained_layout=True)
    image1 = annotated_heatmap(
        axes[0],
        log_ratio,
        "Stored simulated day: log2(empirical/model energy)",
        "{:+.2f}",
        "RdBu_r",
        -limit,
        limit,
    )
    bar1 = fig.colorbar(image1, ax=axes[0], shrink=0.82)
    bar1.set_label("0 = calibrated; positive = excess energy")
    image2 = annotated_heatmap(
        axes[1],
        rejection,
        f"Detection rate across {int(power['power_simulations'].max())} true-model days",
        "{:.0%}",
        "magma",
        0.0,
        1.0,
        center_labels=True,
    )
    bar2 = fig.colorbar(image2, ax=axes[1], shrink=0.82)
    bar2.set_label("two-sided rejection rate")
    fig.suptitle(
        f"Wavelet diagnostic validation on simulated GEMS geometry ({date})", fontsize=14
    )
    fig.savefig(output, dpi=190, bbox_inches="tight")
    plt.close(fig)


def localization_table(power: pd.DataFrame) -> pd.DataFrame:
    pooled = power[power["orientation"].eq("all")].copy()
    rows = []
    for scenario in SCENARIOS:
        part = pooled[pooled["scenario"].eq(scenario)].set_index("band").loc[list(wave.BANDS)]
        rates = part["two_sided_rejection_rate"].to_numpy(float)
        mean_ratios = part["mean_true_dgp_energy_ratio"].to_numpy(float)
        absolute_log2_effect = np.abs(np.log2(np.maximum(mean_ratios, EPS)))
        # Rejection rates often saturate at 100% for a clear misspecification.
        # The largest absolute log-energy ratio is therefore a more useful
        # localization ranking than argmax(power).
        dominant = wave.BANDS[int(np.argmax(absolute_log2_effect))]
        expected = EXPECTED_PRIMARY_BAND[scenario]
        rows.append(
            {
                "scenario": scenario,
                "scenario_label": SCENARIO_LABELS[scenario],
                "expected_primary_band": expected,
                "dominant_detected_band": dominant,
                "dominant_matches_expectation": (
                    True if expected == "none" else dominant == expected
                ),
                "low_rejection_rate": rates[0],
                "middle_rejection_rate": rates[1],
                "high_rejection_rate": rates[2],
                "low_mean_energy_ratio": mean_ratios[0],
                "middle_mean_energy_ratio": mean_ratios[1],
                "high_mean_energy_ratio": mean_ratios[2],
                "low_absolute_log2_effect": absolute_log2_effect[0],
                "middle_absolute_log2_effect": absolute_log2_effect[1],
                "high_absolute_log2_effect": absolute_log2_effect[2],
                "false_positive_rate_if_true": float(np.max(rates))
                if scenario == "true"
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    started = time.perf_counter()
    args.output_root.mkdir(parents=True, exist_ok=True)
    data_path, truth_path = simulation_paths(args)
    truth = load_truth(truth_path)
    print(f"Load stored simulated day {args.date}", flush=True)
    residual, masks, lats, lons, _, data_summary = load_stored_simulated_day(
        data_path, args.date, truth, int(args.hours_per_day)
    )
    actual_dlat = float(np.median(np.diff(lats)))
    actual_dlon = float(np.median(np.diff(lons)))
    if not math.isclose(actual_dlat, args.delta_lat, abs_tol=1e-8, rel_tol=0.0):
        raise RuntimeError(f"Latitude spacing is {actual_dlat}, expected {args.delta_lat}")
    if not math.isclose(actual_dlon, args.delta_lon, abs_tol=1e-8, rel_tol=0.0):
        raise RuntimeError(f"Longitude spacing is {actual_dlon}, expected {args.delta_lon}")
    print(
        f"  grid={len(lats)}x{len(lons)}x{len(masks)}; "
        f"valid={data_summary['valid_count']:,}/{data_summary['grid_cells']:,}",
        flush=True,
    )

    target_shape = (len(lats), len(lons), len(masks))
    observed_details = wave.wavelet_details(
        residual, masks, args.wavelet, args.wavelet_level, args.wavelet_mode
    )
    parameters = candidate_parameters(truth)
    calibrated: dict[str, dict[str, Any]] = {}
    for index, scenario in enumerate(SCENARIOS, start=1):
        print(
            f"\n[{index}/{len(SCENARIOS)}] {SCENARIO_LABELS[scenario]}: calibrate and envelope",
            flush=True,
        )
        calibrated[scenario] = calibrate_candidate(
            scenario,
            parameters[scenario],
            observed_details,
            masks,
            target_shape,
            args,
        )
        print(
            f"  completed in {calibrated[scenario]['runtime_seconds']:.1f}s", flush=True
        )

    truth_spectral_sqrt = calibrated["true"]["spectral_sqrt"]
    if truth_spectral_sqrt is None:
        raise RuntimeError("True-model spectrum was not retained")
    print("\nRepeated-simulation localization check", flush=True)
    power = evaluate_repeated_power(
        calibrated, truth_spectral_sqrt, masks, target_shape, args
    )
    summary = pd.concat(
        [calibrated[scenario]["summary"] for scenario in SCENARIOS], ignore_index=True
    )
    global_curves = pd.concat(
        [calibrated[scenario]["global_curve"] for scenario in SCENARIOS],
        ignore_index=True,
    )
    local_curves = pd.concat(
        [calibrated[scenario]["local_curves"] for scenario in SCENARIOS],
        ignore_index=True,
    )
    localization = localization_table(power)

    wave.atomic_csv(args.output_root / "stored_day_scale_orientation_energy.csv", summary)
    wave.atomic_csv(args.output_root / "repeated_simulation_detection_rates.csv", power)
    wave.atomic_csv(args.output_root / "global_cumulative_curves.csv", global_curves)
    wave.atomic_csv(args.output_root / "localized_cumulative_curves.csv", local_curves)
    wave.atomic_csv(args.output_root / "localization_summary.csv", localization)
    true_counts = [
        int(calibrated["true"]["valid"][band].sum()) for band in wave.BANDS
    ]
    true_cumulative_counts = np.cumsum(true_counts) / sum(true_counts)
    plot_global_curves(
        global_curves,
        args.output_root / "global_cumulative_diagnostic.png",
        args.date,
        (float(true_cumulative_counts[0]), float(true_cumulative_counts[1])),
    )
    plot_local_curves(
        local_curves,
        summary,
        power,
        args.output_root / "localized_6x3_diagnostic.png",
        args.date,
    )
    plot_performance_heatmaps(
        summary,
        power,
        args.output_root / "diagnostic_performance_heatmaps.png",
        args.date,
    )

    total = time.perf_counter() - started
    run_summary = {
        "date": args.date,
        "purpose": "controlled validation of wavelet residual-energy localization",
        "truth": truth,
        "candidate_parameters": parameters,
        "expected_primary_band": EXPECTED_PRIMARY_BAND,
        "stored_simulation": data_summary,
        "grid_shape_time_lat_lon": [len(masks), len(lats), len(lons)],
        "grid_spacing": {"latitude": actual_dlat, "longitude": actual_dlon},
        "wavelet": {
            "name": args.wavelet,
            "level": args.wavelet_level,
            "mode": args.wavelet_mode,
            "bands": wave.BAND_LABELS,
            "band_coefficient_counts": dict(zip(wave.BANDS, true_counts)),
            "global_curve_band_boundaries": true_cumulative_counts.tolist(),
            "approximation_coefficients_included": False,
        },
        "monte_carlo": {
            "calibration_simulations": args.calibration_simulations,
            "envelope_simulations": args.envelope_simulations,
            "power_simulations": args.power_simulations,
            "independent_streams": True,
            "common_random_numbers_across_candidates": True,
            "inverse_variance_bias_correction": "m/(m-2)",
        },
        "important_scope": {
            "stored_day": (
                "existing high-resolution simulation sampled at source locations, then "
                "assigned to the regular grid"
            ),
            "power_replicates": (
                "controlled true covariance simulated directly on the regular grid and "
                "passed through the stored day missing mask"
            ),
            "candidate_models": "full stationary covariance; no Vecchia approximation",
            "time_interaction_localized": False,
        },
        "embedding": {
            "acceptance_rule": (
                "adaptive odd circulant embedding; clip only when negative spectral "
                "mass is at most the configured tolerance"
            ),
            "negative_mass_tolerance": args.negative_mass_tolerance,
            "by_scenario": {
                scenario: calibrated[scenario]["embedding"] for scenario in SCENARIOS
            },
        },
        "scenario_runtime_seconds": {
            scenario: calibrated[scenario]["runtime_seconds"] for scenario in SCENARIOS
        },
        "localization_summary": localization.to_dict(orient="records"),
        "total_runtime_seconds": total,
    }
    wave.write_json(args.output_root / "run_summary.json", run_summary)
    (args.output_root / "RUN_COMPLETE").write_text("complete\n", encoding="utf-8")
    print("\nPooled band rejection rates:", flush=True)
    print(
        localization[
            [
                "scenario_label",
                "low_rejection_rate",
                "middle_rejection_rate",
                "high_rejection_rate",
                "dominant_detected_band",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print(f"\nComplete in {total:.1f}s: {args.output_root}", flush=True)


if __name__ == "__main__":
    main()
