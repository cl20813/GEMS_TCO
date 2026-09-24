#!/usr/bin/env python3
"""Audit FFT generator bias for the already-frozen GEMS contrast.

No simulation and no contrast search are performed.  For each requested
circulant embedding size, this script applies the exact saved A/B/L coefficient
vectors to (1) the analytic covariance at the actual GEMS source coordinates,
(2) the analytic covariance after nearest FFT-grid mapping, and (3) the
spectrally corrected covariance that the FFT generator actually produces.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from apply_fixed_canonical_contrast_real_gems import (
    DEFAULT_DATA_FILE,
    DEFAULT_FIT_CSV,
    DEFAULT_ORACLE_DIR,
    _atomic_csv,
    _atomic_text,
    _grid_cube,
    _load_filtered_frames,
    _model_moments,
    _nearest_axis_indices,
)
from run_fixed_contrast_fft_bootstrap import (
    BASE_LAT_STEP,
    BASE_LON_STEP,
    _circulant_plan,
)


DEFAULT_OUTPUT_DIR = DEFAULT_ORACLE_DIR / "fft_generator_frozen_contrast_audit_20240701"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--fit-csv", type=Path, default=DEFAULT_FIT_CSV)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--strategy", default="standard_432")
    parser.add_argument("--day-index", type=int, default=0)
    parser.add_argument("--embedding-factors", default="2,4,8")
    parser.add_argument("--lat-factor-hr", type=int, default=1)
    parser.add_argument("--lon-factor-hr", type=int, default=1)
    parser.add_argument("--mapping-factors", default="1x1,100x10")
    parser.add_argument("--hr-pad", type=float, default=0.1)
    parser.add_argument("--latitude-min", type=float, default=-3.0)
    parser.add_argument("--latitude-max", type=float, default=2.0)
    parser.add_argument("--longitude-min", type=float, default=121.0)
    parser.add_argument("--longitude-max", type=float, default=131.0)
    parser.add_argument("--spectral-zero-tolerance", type=float, default=1.0e-10)
    parser.add_argument("--max-negative-spectral-mass", type=float, default=0.10)
    return parser


def _parse_positive_integers(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    if not values or any(value < 2 for value in values):
        raise ValueError("embedding factors must be comma-separated integers >= 2")
    return values


def _parse_mapping_factors(text: str) -> tuple[tuple[int, int], ...]:
    result = []
    for item in text.split(","):
        left, right = item.lower().strip().split("x")
        pair = (int(left), int(right))
        if min(pair) < 1:
            raise ValueError("mapping factors must be positive")
        result.append(pair)
    return tuple(result)


def _axes(args: argparse.Namespace, lat_factor: int, lon_factor: int):
    dlat = BASE_LAT_STEP / lat_factor
    dlon = BASE_LON_STEP / lon_factor
    lats = np.arange(
        args.latitude_min - args.hr_pad,
        args.latitude_max + args.hr_pad + 0.5 * dlat,
        dlat,
        dtype=np.float64,
    )
    lons = np.arange(
        args.longitude_min - args.hr_pad,
        args.longitude_max + args.hr_pad + 0.5 * dlon,
        dlon,
        dtype=np.float64,
    )
    return lats, lons


def _nearest_indices(values: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    step = float(axis[1] - axis[0])
    indices = np.rint((values - axis[0]) / step).astype(np.int64)
    if np.any(indices < 0) or np.any(indices >= len(axis)):
        raise ValueError("source coordinate lies outside the FFT lattice")
    return indices, axis[indices] - values


def _mapping_diagnostics(
    cube: dict[str, np.ndarray],
    fit: pd.Series,
    args: argparse.Namespace,
    mapping_factors: tuple[tuple[int, int], ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for lat_factor, lon_factor in mapping_factors:
        lats, lons = _axes(args, lat_factor, lon_factor)
        for time_index in range(cube["source_latitude"].shape[0]):
            source_lat = cube["source_latitude"][time_index]
            source_lon = cube["source_longitude"][time_index]
            valid = np.isfinite(source_lat) & np.isfinite(source_lon)
            lat_index, lat_error = _nearest_indices(source_lat[valid], lats)
            lon_index, lon_error = _nearest_indices(source_lon[valid], lons)
            pairs = np.column_stack([lat_index, lon_index])
            unique_count = np.unique(pairs, axis=0).shape[0]
            distance = np.sqrt(np.square(lat_error) + np.square(lon_error))
            standardized = np.sqrt(
                np.square(lat_error / float(fit["est_range_lat"]))
                + np.square(lon_error / float(fit["est_range_lon"]))
            )
            rows.append(
                {
                    "lat_factor_hr": lat_factor,
                    "lon_factor_hr": lon_factor,
                    "time_index": time_index,
                    "observed_source_count": int(valid.sum()),
                    "unique_fft_cell_count": int(unique_count),
                    "duplicated_assignments": int(valid.sum() - unique_count),
                    "exact_coordinate_matches": int(
                        ((np.abs(lat_error) <= 1.0e-12) & (np.abs(lon_error) <= 1.0e-12)).sum()
                    ),
                    "max_abs_lat_error": float(np.max(np.abs(lat_error))),
                    "max_abs_lon_error": float(np.max(np.abs(lon_error))),
                    "max_euclidean_degree_error": float(distance.max()),
                    "mean_euclidean_degree_error": float(distance.mean()),
                    "max_standardized_error": float(standardized.max()),
                    "mean_standardized_error": float(standardized.mean()),
                }
            )
    return pd.DataFrame(rows)


def _contrast_coordinates(
    samples: pd.DataFrame,
    cube: dict[str, np.ndarray],
    fit: pd.Series,
    lats_fft: np.ndarray,
    lons_fft: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    patterns = {
        "clockwise": np.asarray([[-2.0, -2.0], [2.0, 2.0], [-1.0, -2.0], [1.0, 2.0]]),
        "counterclockwise": np.asarray([[-2.0, 2.0], [2.0, -2.0], [-1.0, 2.0], [1.0, -2.0]]),
    }
    n = len(samples)
    actual = np.empty((n, 8, 3), dtype=np.float64)
    fft_indices = np.empty((n, 8, 3), dtype=np.int64)
    fft_coordinates = np.empty((n, 8, 3), dtype=np.float64)
    latitudes = cube["latitudes"]
    longitudes = cube["longitudes"]
    latitude_step = float(cube["latitude_step"])
    longitude_step = float(cube["longitude_step"])
    centers_lat = samples["center_latitude"].to_numpy(dtype=np.float64)
    centers_lon = samples["center_longitude"].to_numpy(dtype=np.float64)
    handedness = samples["handedness"].to_numpy(dtype=str)
    time_k = samples["time_k"].to_numpy(dtype=np.int64)
    point_template = ((0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (1, 2), (1, 3))

    for orientation, offsets_standardized in patterns.items():
        row_mask = handedness == orientation
        if not np.any(row_mask):
            continue
        row_indices = np.flatnonzero(row_mask)
        offsets = offsets_standardized * np.asarray(
            [float(fit["est_range_lat"]), float(fit["est_range_lon"])]
        )
        for point_index, (time_add, offset_index) in enumerate(point_template):
            times = time_k[row_mask] + time_add
            desired_lat = (
                centers_lat[row_mask]
                + offsets[offset_index, 0]
                + float(fit["est_advec_lat"]) * times
            )
            desired_lon = (
                centers_lon[row_mask]
                + offsets[offset_index, 1]
                + float(fit["est_advec_lon"]) * times
            )
            native_i, _, valid_i = _nearest_axis_indices(desired_lat, latitudes, latitude_step)
            native_j, _, valid_j = _nearest_axis_indices(desired_lon, longitudes, longitude_step)
            if not np.all(valid_i & valid_j):
                raise AssertionError("saved complete contrast contains an invalid regular-grid target")
            source_lat = cube["source_latitude"][times, native_i, native_j]
            source_lon = cube["source_longitude"][times, native_i, native_j]
            if not np.all(np.isfinite(source_lat) & np.isfinite(source_lon)):
                raise AssertionError("saved complete contrast contains a missing source coordinate")
            actual[row_indices, point_index, 0] = source_lat
            actual[row_indices, point_index, 1] = source_lon
            actual[row_indices, point_index, 2] = times
            fft_i, _ = _nearest_indices(source_lat, lats_fft)
            fft_j, _ = _nearest_indices(source_lon, lons_fft)
            fft_indices[row_indices, point_index, 0] = fft_i
            fft_indices[row_indices, point_index, 1] = fft_j
            fft_indices[row_indices, point_index, 2] = times
            fft_coordinates[row_indices, point_index, 0] = lats_fft[fft_i]
            fft_coordinates[row_indices, point_index, 1] = lons_fft[fft_j]
            fft_coordinates[row_indices, point_index, 2] = times
    return actual, fft_coordinates, fft_indices, time_k


def _quadratic_moments(
    covariance: np.ndarray, coefficient_a: np.ndarray, coefficient_b: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "h_aa": np.einsum("i,nij,j->n", coefficient_a, covariance, coefficient_a, optimize=True),
        "h_bb": np.einsum("i,nij,j->n", coefficient_b, covariance, coefficient_b, optimize=True),
        "h_ab": np.einsum("i,nij,j->n", coefficient_a, covariance, coefficient_b, optimize=True),
    }


def _fft_moments(plan, fft_indices, coefficient_a, coefficient_b):
    p_lat, p_lon, p_time = plan.embedding_shape
    i = fft_indices[..., 0]
    j = fft_indices[..., 1]
    t = fft_indices[..., 2]
    di = (i[:, :, None] - i[:, None, :]) % p_lat
    dj = (j[:, :, None] - j[:, None, :]) % p_lon
    dt = (t[:, :, None] - t[:, None, :]) % p_time
    covariance = plan.corrected_covariance[di, dj, dt]
    return _quadratic_moments(covariance, coefficient_a, coefficient_b)


def _with_l_variance(moments, d1, d2):
    result = {name: np.asarray(value) for name, value in moments.items()}
    result["variance_l"] = (
        d1**2 * result["h_aa"]
        + d2**2 * result["h_bb"]
        + 2.0 * d1 * d2 * result["h_ab"]
    )
    return result


def _summary_rows(
    factor: int,
    dgp: str,
    target,
    mapped,
    generated,
    model_gap: dict[str, float],
    d1: float,
    d2: float,
) -> list[dict[str, Any]]:
    rows = []
    target = _with_l_variance(target, d1, d2)
    mapped = _with_l_variance(mapped, d1, d2)
    generated = _with_l_variance(generated, d1, d2)
    for component in ("h_aa", "h_bb", "h_ab", "variance_l"):
        target_mean = float(np.mean(target[component]))
        mapped_mean = float(np.mean(mapped[component]))
        generated_mean = float(np.mean(generated[component]))
        gap = float(model_gap[component])
        total_bias = generated_mean - target_mean
        rows.append(
            {
                "embedding_factor": factor,
                "dgp": dgp,
                "component": component,
                "target_actual_source": target_mean,
                "target_fft_mapped_source": mapped_mean,
                "fft_generated": generated_mean,
                "mapping_bias": mapped_mean - target_mean,
                "spectral_correction_bias": generated_mean - mapped_mean,
                "total_generator_bias": total_bias,
                "separable_minus_joint_target_gap": gap,
                "abs_total_bias_over_abs_model_gap": (
                    abs(total_bias) / abs(gap) if gap != 0.0 else np.nan
                ),
                "signed_total_bias_over_model_gap": (
                    total_bias / gap if gap != 0.0 else np.nan
                ),
            }
        )
    return rows


def _write_figure(summary: pd.DataFrame, output_dir: Path) -> None:
    selected = summary.loc[summary["component"] == "variance_l"].copy()
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.7), constrained_layout=True)
    for dgp, group in selected.groupby("dgp", sort=True):
        axes[0].plot(
            group["embedding_factor"],
            group["spectrum_negative_mass_fraction"],
            marker="o",
            label=dgp,
        )
        axes[1].plot(
            group["embedding_factor"],
            group["signed_total_bias_over_model_gap"],
            marker="o",
            label=dgp,
        )
    axes[0].set_ylabel("negative spectral mass fraction")
    axes[1].set_ylabel("Var(L) generator bias / joint-separable gap")
    for axis in axes:
        axis.set_xlabel("embedding factor (all three axes)")
        axis.set_xticks(sorted(selected["embedding_factor"].unique()))
        axis.axhline(0.0, color="0.3", linewidth=0.8)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)
    figure.savefig(output_dir / "embedding_convergence_and_diagnostic_bias.png", dpi=220)
    figure.savefig(output_dir / "embedding_convergence_and_diagnostic_bias.pdf")
    plt.close(figure)


def main() -> None:
    args = build_parser().parse_args()
    factors = _parse_positive_integers(args.embedding_factors)
    mapping_factors = _parse_mapping_factors(args.mapping_factors)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fit_table = pd.read_csv(args.fit_csv.expanduser())
    selected = fit_table.loc[
        (fit_table["strategy"] == args.strategy)
        & (fit_table["day_idx"].astype(int) == int(args.day_index))
    ]
    if len(selected) != 1:
        raise ValueError("saved fit selection must yield exactly one row")
    fit = selected.iloc[0]
    coefficient_path = args.oracle_dir.expanduser() / "global_two_rectangle_search/global_pair_ties.csv"
    coefficients = pd.read_csv(coefficient_path, float_precision="round_trip").iloc[0]
    d1 = float(coefficients["raw_coefficient_first"])
    d2 = float(coefficients["raw_coefficient_second"])

    frames, monthly_mean = _load_filtered_frames(
        args.data_file.expanduser(),
        (args.latitude_min, args.latitude_max),
        (args.longitude_min, args.longitude_max),
    )
    keys = sorted(frames)
    start = int(args.day_index) * 8
    day_frames = [frames[key] for key in keys[start : start + 8]]
    cube = _grid_cube(day_frames, monthly_mean, fit)
    samples_path = args.oracle_dir.expanduser() / "real_gems_pilot_20240701/translated_contrast_samples.csv"
    samples = pd.read_csv(samples_path)

    lats_fft, lons_fft = _axes(args, args.lat_factor_hr, args.lon_factor_hr)
    actual_coordinates, mapped_coordinates, fft_indices, _ = _contrast_coordinates(
        samples, cube, fit, lats_fft, lons_fft
    )
    coefficient_a = np.asarray([1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    coefficient_b = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 1.0])
    target_all = _model_moments(actual_coordinates, fit, coefficient_a, coefficient_b)
    mapped_all = _model_moments(mapped_coordinates, fit, coefficient_a, coefficient_b)
    target_by_dgp = {
        dgp: {
            "h_aa": target_all[f"{dgp}_h_aa"],
            "h_bb": target_all[f"{dgp}_h_bb"],
            "h_ab": target_all[f"{dgp}_h_ab"],
        }
        for dgp in ("joint", "separable")
    }
    mapped_by_dgp = {
        dgp: {
            "h_aa": mapped_all[f"{dgp}_h_aa"],
            "h_bb": mapped_all[f"{dgp}_h_bb"],
            "h_ab": mapped_all[f"{dgp}_h_ab"],
        }
        for dgp in ("joint", "separable")
    }
    target_augmented = {
        dgp: _with_l_variance(target_by_dgp[dgp], d1, d2)
        for dgp in ("joint", "separable")
    }
    model_gap = {
        component: float(
            np.mean(target_augmented["separable"][component])
            - np.mean(target_augmented["joint"][component])
        )
        for component in ("h_aa", "h_bb", "h_ab", "variance_l")
    }

    mapping = _mapping_diagnostics(cube, fit, args, mapping_factors)
    _atomic_csv(output_dir / "source_to_fft_grid_mapping.csv", mapping)
    contrast_mapping_rows: list[dict[str, Any]] = []
    for mapping_lat_factor, mapping_lon_factor in mapping_factors:
        mapping_lats, mapping_lons = _axes(
            args, mapping_lat_factor, mapping_lon_factor
        )
        _, mapping_coordinates, _, _ = _contrast_coordinates(
            samples, cube, fit, mapping_lats, mapping_lons
        )
        mapping_moments = _model_moments(
            mapping_coordinates, fit, coefficient_a, coefficient_b
        )
        for dgp in ("joint", "separable"):
            mapped_components = _with_l_variance(
                {
                    "h_aa": mapping_moments[f"{dgp}_h_aa"],
                    "h_bb": mapping_moments[f"{dgp}_h_bb"],
                    "h_ab": mapping_moments[f"{dgp}_h_ab"],
                },
                d1,
                d2,
            )
            for component in ("h_aa", "h_bb", "h_ab", "variance_l"):
                actual_mean = float(np.mean(target_augmented[dgp][component]))
                mapped_mean = float(np.mean(mapped_components[component]))
                contrast_mapping_rows.append(
                    {
                        "lat_factor_hr": mapping_lat_factor,
                        "lon_factor_hr": mapping_lon_factor,
                        "dgp": dgp,
                        "component": component,
                        "target_actual_source": actual_mean,
                        "target_mapped_source": mapped_mean,
                        "mapping_bias": mapped_mean - actual_mean,
                        "abs_mapping_bias_over_abs_model_gap": (
                            abs(mapped_mean - actual_mean)
                            / abs(model_gap[component])
                            if model_gap[component] != 0.0
                            else np.nan
                        ),
                    }
                )
    contrast_mapping = pd.DataFrame(contrast_mapping_rows)
    _atomic_csv(output_dir / "frozen_contrast_mapping_bias.csv", contrast_mapping)
    embedding_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for factor in factors:
        for dgp in ("joint", "separable"):
            plan_args = SimpleNamespace(
                embedding_spatial_factor=factor,
                embedding_temporal_factor=factor,
                lat_factor_hr=args.lat_factor_hr,
                lon_factor_hr=args.lon_factor_hr,
                spectral_zero_tolerance=args.spectral_zero_tolerance,
                max_negative_spectral_mass=args.max_negative_spectral_mass,
            )
            plan = _circulant_plan(fit, dgp, lats_fft, lons_fft, 8, plan_args)
            generated = _fft_moments(plan, fft_indices, coefficient_a, coefficient_b)
            summary_rows.extend(
                _summary_rows(
                    factor,
                    dgp,
                    target_by_dgp[dgp],
                    mapped_by_dgp[dgp],
                    generated,
                    model_gap,
                    d1,
                    d2,
                )
            )
            embedding_rows.append(plan.diagnostics)
            print(
                f"factor={factor} dgp={dgp} negative_mass="
                f"{plan.diagnostics['spectrum_negative_mass_fraction']:.8f}",
                flush=True,
            )
            del plan, generated
            gc.collect()

    embedding = pd.DataFrame(embedding_rows).rename(
        columns={"embedding_spatial_factor": "embedding_factor"}
    )
    # _circulant_plan records dimensions rather than the requested multiplier.
    embedding["embedding_factor"] = np.repeat(np.asarray(factors), 2)
    summary = pd.DataFrame(summary_rows).merge(
        embedding[
            [
                "embedding_factor",
                "dgp",
                "spectrum_negative_mass_fraction",
                "spectrum_negative_fraction",
                "spectrum_min_before_clip",
                "embedding_is_exact_nonnegative",
            ]
        ],
        on=["embedding_factor", "dgp"],
        how="left",
        validate="many_to_one",
    )
    _atomic_csv(output_dir / "embedding_diagnostics.csv", embedding)
    _atomic_csv(output_dir / "frozen_contrast_generator_bias.csv", summary)
    _write_figure(summary, output_dir)

    l_rows = summary.loc[summary["component"] == "variance_l"]
    mapping_grouped = mapping.groupby(["lat_factor_hr", "lon_factor_hr"], as_index=False).agg(
        observed_source_count=("observed_source_count", "sum"),
        duplicated_assignments=("duplicated_assignments", "sum"),
        exact_coordinate_matches=("exact_coordinate_matches", "sum"),
        max_euclidean_degree_error=("max_euclidean_degree_error", "max"),
        max_standardized_error=("max_standardized_error", "max"),
    )
    lines = [
        "# FFT generator audit for the frozen contrast",
        "",
        "No simulation, refitting, or contrast selection was performed. Expectations were evaluated directly under the analytic and generated covariance matrices for every complete frozen contrast used in the real-data pilot.",
        "",
        f"Frozen translated contrast count: `{len(samples)}`. Joint-to-separable target Var(L) gap: `{model_gap['variance_l']:.17g}`.",
        "",
        "## Diagnostic-specific result",
        "",
        "| factor | DGP | target Var(L) | FFT Var(L) | total bias | |bias| / |model gap| | negative spectral mass |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in l_rows.itertuples(index=False):
        lines.append(
            f"| {row.embedding_factor} | `{row.dgp}` | {row.target_actual_source:.9g} | "
            f"{row.fft_generated:.9g} | {row.total_generator_bias:.9g} | "
            f"{row.abs_total_bias_over_abs_model_gap:.6g} | "
            f"{row.spectrum_negative_mass_fraction:.6g} |"
        )
    lines.extend(
        [
            "",
            "`mapping_bias` compares analytic covariance at nearest FFT cells with analytic covariance at actual source coordinates. `spectral_correction_bias` then compares the clipped/rescaled embedding covariance with the analytic covariance at those FFT cells. Their sum is the total generator bias.",
            "",
            "## Source-coordinate mapping",
            "",
            "| HR factors | observations over 8 hours | duplicate assignments | exact matches | max distance (degrees) | max standardized distance |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in mapping_grouped.itertuples(index=False):
        lines.append(
            f"| {row.lat_factor_hr}x{row.lon_factor_hr} | {row.observed_source_count} | "
            f"{row.duplicated_assignments} | {row.exact_coordinate_matches} | "
            f"{row.max_euclidean_degree_error:.8g} | {row.max_standardized_error:.8g} |"
        )
    lines.extend(
        [
            "",
            "Frozen Var(L) mapping bias (without spectral clipping):",
            "",
            "| HR factors | DGP | target Var(L) | mapped Var(L) | mapping bias | |bias| / |model gap| |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in contrast_mapping.loc[
        contrast_mapping["component"] == "variance_l"
    ].itertuples(index=False):
        lines.append(
            f"| {row.lat_factor_hr}x{row.lon_factor_hr} | `{row.dgp}` | "
            f"{row.target_actual_source:.9g} | {row.target_mapped_source:.9g} | "
            f"{row.mapping_bias:.9g} | {row.abs_mapping_bias_over_abs_model_gap:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "The spectrally corrected FFT generator should be treated as a primary approximation only if its frozen-diagnostic bias is small relative to the joint-versus-separable target gap and is stable as the embedding expands. Nonnegative-eigenvalue status is reported rather than inferred from generic covariance error.",
            "",
        ]
    )
    _atomic_text(output_dir / "REPORT.md", "\n".join(lines))
    manifest = {
        "data_file": str(args.data_file.expanduser().resolve()),
        "fit_csv": str(args.fit_csv.expanduser().resolve()),
        "coefficient_source": str(coefficient_path.resolve()),
        "translated_samples_source": str(samples_path.resolve()),
        "embedding_factors_all_axes": list(factors),
        "fft_grid_factors_for_covariance_audit": [args.lat_factor_hr, args.lon_factor_hr],
        "mapping_factors": [list(pair) for pair in mapping_factors],
        "contrast_search": False,
        "simulation": False,
        "model_gap": model_gap,
    }
    _atomic_text(output_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(l_rows.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
