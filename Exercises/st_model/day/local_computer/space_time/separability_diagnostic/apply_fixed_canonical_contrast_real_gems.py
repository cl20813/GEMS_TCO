#!/usr/bin/env python3
"""Apply the pre-fixed canonical two-contrast diagnostic to one GEMS day.

This is a descriptive real-data pilot, not a test with a known truth.  The
contrast geometry and coefficient ratio come from the saved exact-comoving
simulation and are not re-selected here.  The geometry is transferred in
fitted spatial-range units and evaluated along the fitted advection path on
the nearest regular GEMS grid cells.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[5]
DEFAULT_DATA_FILE = Path(
    "/Users/joonwonlee/Documents/GEMS_DATA/pickle_2024/tco_grid_24_07.pkl"
)
DEFAULT_FIT_CSV = (
    PROJECT_ROOT
    / "Exercises/st_model/day/local_computer/space_time/spectrum_diagnostics/outputs"
    / "real_july2024_st_heads_vecchia_lag432_one_day_gc_a075_b1_nugget0_061626"
    / "heads_vecchia_lag432_one_day_fit_summary.csv"
)
DEFAULT_ORACLE_DIR = HERE / "outputs/exact_comoving_rectangle_dictionary_092226"
DEFAULT_OUTPUT_DIR = DEFAULT_ORACLE_DIR / "real_gems_pilot_20240701"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", type=Path, default=DEFAULT_DATA_FILE)
    parser.add_argument("--fit-csv", type=Path, default=DEFAULT_FIT_CSV)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--strategy", default="standard_432")
    parser.add_argument("--day-index", type=int, default=0)
    parser.add_argument("--latitude-min", type=float, default=-3.0)
    parser.add_argument("--latitude-max", type=float, default=2.0)
    parser.add_argument("--longitude-min", type=float, default=121.0)
    parser.add_argument("--longitude-max", type=float, default=131.0)
    return parser


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)


def _atomic_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def _load_filtered_frames(
    path: Path,
    latitude_bounds: tuple[float, float],
    longitude_bounds: tuple[float, float],
) -> tuple[dict[str, pd.DataFrame], float]:
    loaded = pd.read_pickle(path)
    if not isinstance(loaded, Mapping):
        raise TypeError(f"{path} must contain a mapping of hourly DataFrames")
    frames: dict[str, pd.DataFrame] = {}
    total = 0.0
    count = 0
    required = {
        "Latitude",
        "Longitude",
        "Source_Latitude",
        "Source_Longitude",
        "ColumnAmountO3",
    }
    for key in sorted(loaded):
        frame = loaded[key]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{key!r} is not a DataFrame")
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{key!r} is missing columns {sorted(missing)}")
        keep = frame["Latitude"].between(*latitude_bounds) & frame["Longitude"].between(
            *longitude_bounds
        )
        selected = frame.loc[keep].reset_index(drop=True)
        ozone = pd.to_numeric(selected["ColumnAmountO3"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        finite = np.isfinite(ozone)
        total += float(ozone[finite].sum())
        count += int(finite.sum())
        frames[str(key)] = selected
    if not frames or count == 0:
        raise ValueError("no finite observations remain after filtering")
    return frames, total / count


def _uniform_axis(values: np.ndarray, name: str) -> tuple[np.ndarray, float]:
    axis = np.sort(np.unique(np.asarray(values, dtype=np.float64)))
    if axis.size < 2:
        raise ValueError(f"{name} axis must have at least two values")
    differences = np.diff(axis)
    step = float(np.median(differences))
    if not np.allclose(differences, step, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"{name} axis is not uniformly spaced")
    return axis, step


def _grid_cube(
    frames: list[pd.DataFrame],
    monthly_mean: float,
    fit: pd.Series,
) -> dict[str, np.ndarray]:
    reference = frames[0]
    latitudes, latitude_step = _uniform_axis(reference["Latitude"], "latitude")
    longitudes, longitude_step = _uniform_axis(reference["Longitude"], "longitude")
    shape = (len(frames), latitudes.size, longitudes.size)
    residual = np.full(shape, np.nan, dtype=np.float64)
    source_latitude = np.full(shape, np.nan, dtype=np.float64)
    source_longitude = np.full(shape, np.nan, dtype=np.float64)
    beta = np.asarray([float(fit[f"beta_{index}"]) for index in range(9)])
    latitude_mean = float(fit["gls_lat_mean"])

    reference_coordinates = reference[["Latitude", "Longitude"]].to_numpy(
        dtype=np.float64
    )
    for time_index, frame in enumerate(frames):
        coordinates = frame[["Latitude", "Longitude"]].to_numpy(dtype=np.float64)
        if coordinates.shape != reference_coordinates.shape or not np.allclose(
            coordinates, reference_coordinates, rtol=0.0, atol=1.0e-10
        ):
            raise ValueError("regular-grid coordinates or row order changed across the day")
        lat_index = np.rint(
            (frame["Latitude"].to_numpy(dtype=np.float64) - latitudes[0])
            / latitude_step
        ).astype(np.int64)
        lon_index = np.rint(
            (frame["Longitude"].to_numpy(dtype=np.float64) - longitudes[0])
            / longitude_step
        ).astype(np.int64)
        if np.unique(np.column_stack([lat_index, lon_index]), axis=0).shape[0] != len(frame):
            raise ValueError("regular-grid rows are not one-to-one")
        ozone = pd.to_numeric(frame["ColumnAmountO3"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        src_lat = pd.to_numeric(frame["Source_Latitude"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        src_lon = pd.to_numeric(frame["Source_Longitude"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        mean = beta[0] + beta[1] * (src_lat - latitude_mean)
        if time_index > 0:
            mean = mean + beta[time_index + 1]
        values = ozone - float(monthly_mean) - mean
        observed = np.isfinite(ozone) & np.isfinite(src_lat) & np.isfinite(src_lon)
        residual[time_index, lat_index[observed], lon_index[observed]] = values[observed]
        source_latitude[time_index, lat_index[observed], lon_index[observed]] = src_lat[
            observed
        ]
        source_longitude[time_index, lat_index[observed], lon_index[observed]] = src_lon[
            observed
        ]
    return {
        "latitudes": latitudes,
        "longitudes": longitudes,
        "latitude_step": np.asarray(latitude_step),
        "longitude_step": np.asarray(longitude_step),
        "residual": residual,
        "source_latitude": source_latitude,
        "source_longitude": source_longitude,
    }


def _nearest_axis_indices(
    targets: np.ndarray,
    axis: np.ndarray,
    step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.rint((targets - axis[0]) / step).astype(np.int64)
    valid = (indices >= 0) & (indices < axis.size)
    clipped = np.clip(indices, 0, axis.size - 1)
    errors = np.abs(axis[clipped] - targets)
    valid &= errors <= 0.5 * step + 1.0e-10
    return clipped, errors, valid


def _gc_correlation(distance: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    distance = np.asarray(distance, dtype=np.float64)
    return np.power(1.0 + np.power(np.maximum(distance, 0.0), alpha), -beta / alpha)


def _model_moments(
    coordinates: np.ndarray,
    fit: pd.Series,
    coefficient_a: np.ndarray,
    coefficient_b: np.ndarray,
) -> dict[str, np.ndarray]:
    delta = coordinates[:, None, :, :] - coordinates[:, :, None, :]
    delta_latitude = delta[..., 0]
    delta_longitude = delta[..., 1]
    delta_time = delta[..., 2]
    shifted_latitude = delta_latitude - float(fit["est_advec_lat"]) * delta_time
    shifted_longitude = delta_longitude - float(fit["est_advec_lon"]) * delta_time
    spatial_distance = np.sqrt(
        np.square(shifted_latitude / float(fit["est_range_lat"]))
        + np.square(shifted_longitude / float(fit["est_range_lon"]))
    )
    temporal_distance = np.abs(delta_time) / float(fit["est_range_time"])
    alpha = float(fit["gc_alpha"])
    beta = float(fit["gc_beta"])
    variance = float(fit["est_sigmasq"])
    joint = variance * _gc_correlation(
        np.sqrt(np.square(spatial_distance) + np.square(temporal_distance)),
        alpha,
        beta,
    )
    separable = variance * _gc_correlation(spatial_distance, alpha, beta) * _gc_correlation(
        temporal_distance, alpha, beta
    )
    same_time = np.abs(delta_time) < 1.0e-12
    margin_error = float(np.max(np.abs(joint[same_time] - separable[same_time])))
    if margin_error > 1.0e-11 * max(variance, 1.0):
        raise ArithmeticError("joint and separable spatial margins do not match")

    def quadratic(left: np.ndarray, matrix: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.einsum("i,nij,j->n", left, matrix, right, optimize=True)

    result: dict[str, np.ndarray] = {}
    for model_name, covariance in (("joint", joint), ("separable", separable)):
        result[f"{model_name}_h_aa"] = quadratic(
            coefficient_a, covariance, coefficient_a
        )
        result[f"{model_name}_h_bb"] = quadratic(
            coefficient_b, covariance, coefficient_b
        )
        result[f"{model_name}_h_ab"] = quadratic(
            coefficient_a, covariance, coefficient_b
        )
    return result


def _canonical_samples(
    cube: dict[str, np.ndarray],
    geometry_fit: pd.Series,
    coefficient_first: float,
    coefficient_second: float,
    covariance_fit: pd.Series | None = None,
) -> pd.DataFrame:
    if covariance_fit is None:
        covariance_fit = geometry_fit
    latitudes = cube["latitudes"]
    longitudes = cube["longitudes"]
    latitude_step = float(cube["latitude_step"])
    longitude_step = float(cube["longitude_step"])
    center_latitude, center_longitude = np.meshgrid(
        latitudes, longitudes, indexing="ij"
    )
    center_latitude = center_latitude.ravel()
    center_longitude = center_longitude.ravel()
    candidate_count = center_latitude.size

    patterns = {
        "clockwise": np.asarray(
            [[-2.0, -2.0], [2.0, 2.0], [-1.0, -2.0], [1.0, 2.0]]
        ),
        "counterclockwise": np.asarray(
            [[-2.0, 2.0], [2.0, -2.0], [-1.0, 2.0], [1.0, -2.0]]
        ),
    }
    coefficient_a = np.asarray([1.0, -1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    coefficient_b = np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, -1.0, -1.0, 1.0])
    coefficient_l = coefficient_first * coefficient_a + coefficient_second * coefficient_b
    records: list[pd.DataFrame] = []

    for handedness, standardized_offsets in patterns.items():
        physical_offsets = standardized_offsets * np.asarray(
            [
                float(geometry_fit["est_range_lat"]),
                float(geometry_fit["est_range_lon"]),
            ]
        )
        for time_k in range(7):
            time_l = time_k + 1
            point_specs = [
                (time_k, 0),
                (time_k, 1),
                (time_l, 0),
                (time_l, 1),
                (time_k, 2),
                (time_k, 3),
                (time_l, 2),
                (time_l, 3),
            ]
            values = np.full((candidate_count, 8), np.nan, dtype=np.float64)
            coordinates = np.full((candidate_count, 8, 3), np.nan, dtype=np.float64)
            maximum_grid_error = np.zeros(candidate_count, dtype=np.float64)
            maximum_source_error = np.zeros(candidate_count, dtype=np.float64)
            valid = np.ones(candidate_count, dtype=bool)
            for point_index, (time_index, offset_index) in enumerate(point_specs):
                desired_latitude = (
                    center_latitude
                    + physical_offsets[offset_index, 0]
                    + float(geometry_fit["est_advec_lat"]) * time_index
                )
                desired_longitude = (
                    center_longitude
                    + physical_offsets[offset_index, 1]
                    + float(geometry_fit["est_advec_lon"]) * time_index
                )
                latitude_index, latitude_error, latitude_valid = _nearest_axis_indices(
                    desired_latitude, latitudes, latitude_step
                )
                longitude_index, longitude_error, longitude_valid = _nearest_axis_indices(
                    desired_longitude, longitudes, longitude_step
                )
                regular_valid = latitude_valid & longitude_valid
                point_values = cube["residual"][
                    time_index, latitude_index, longitude_index
                ]
                point_source_latitude = cube["source_latitude"][
                    time_index, latitude_index, longitude_index
                ]
                point_source_longitude = cube["source_longitude"][
                    time_index, latitude_index, longitude_index
                ]
                observed = (
                    regular_valid
                    & np.isfinite(point_values)
                    & np.isfinite(point_source_latitude)
                    & np.isfinite(point_source_longitude)
                )
                valid &= observed
                values[:, point_index] = point_values
                coordinates[:, point_index, 0] = point_source_latitude
                coordinates[:, point_index, 1] = point_source_longitude
                coordinates[:, point_index, 2] = float(time_index)
                standardized_grid_error = np.maximum(
                    latitude_error / float(geometry_fit["est_range_lat"]),
                    longitude_error / float(geometry_fit["est_range_lon"]),
                )
                standardized_source_error = np.maximum(
                    np.abs(point_source_latitude - desired_latitude)
                    / float(geometry_fit["est_range_lat"]),
                    np.abs(point_source_longitude - desired_longitude)
                    / float(geometry_fit["est_range_lon"]),
                )
                maximum_grid_error = np.maximum(maximum_grid_error, standardized_grid_error)
                maximum_source_error = np.maximum(
                    maximum_source_error, standardized_source_error
                )
            selected_values = values[valid]
            selected_coordinates = coordinates[valid]
            if selected_values.size == 0:
                continue
            q_a = selected_values @ coefficient_a
            q_b = selected_values @ coefficient_b
            contrast = selected_values @ coefficient_l
            if not np.allclose(
                contrast,
                coefficient_first * q_a + coefficient_second * q_b,
                rtol=1.0e-13,
                atol=1.0e-13,
            ):
                raise ArithmeticError("saved two-contrast coefficient reconstruction failed")
            moments = _model_moments(
                selected_coordinates, covariance_fit, coefficient_a, coefficient_b
            )
            joint_variance = (
                coefficient_first**2 * moments["joint_h_aa"]
                + coefficient_second**2 * moments["joint_h_bb"]
                + 2.0
                * coefficient_first
                * coefficient_second
                * moments["joint_h_ab"]
            )
            separable_variance = (
                coefficient_first**2 * moments["separable_h_aa"]
                + coefficient_second**2 * moments["separable_h_bb"]
                + 2.0
                * coefficient_first
                * coefficient_second
                * moments["separable_h_ab"]
            )
            if np.any(joint_variance <= 0.0) or np.any(separable_variance <= 0.0):
                raise ArithmeticError("model produced a nonpositive contrast variance")
            records.append(
                pd.DataFrame(
                    {
                        "handedness": handedness,
                        "time_k": time_k,
                        "time_l": time_l,
                        "candidate_center_count": candidate_count,
                        "center_latitude": center_latitude[valid],
                        "center_longitude": center_longitude[valid],
                        "q_a": q_a,
                        "q_b": q_b,
                        "contrast_l": contrast,
                        "joint_variance_l": joint_variance,
                        "separable_variance_l": separable_variance,
                        "joint_h_aa": moments["joint_h_aa"],
                        "joint_h_bb": moments["joint_h_bb"],
                        "joint_h_ab": moments["joint_h_ab"],
                        "separable_h_aa": moments["separable_h_aa"],
                        "separable_h_bb": moments["separable_h_bb"],
                        "separable_h_ab": moments["separable_h_ab"],
                        "maximum_grid_error_standardized": maximum_grid_error[valid],
                        "maximum_source_error_standardized": maximum_source_error[valid],
                    }
                )
            )
    if not records:
        raise ValueError("no complete canonical contrasts were found in the selected day")
    return pd.concat(records, ignore_index=True)


def _summarize_group(frame: pd.DataFrame, label: str) -> dict[str, Any]:
    empirical_l2 = float(np.mean(np.square(frame["contrast_l"])))
    joint_variance = float(frame["joint_variance_l"].mean())
    separable_variance = float(frame["separable_variance_l"].mean())
    joint_ratio = empirical_l2 / joint_variance
    separable_ratio = empirical_l2 / separable_variance
    return {
        "group": label,
        "sample_count": len(frame),
        "empirical_h_aa": float(np.mean(np.square(frame["q_a"]))),
        "empirical_h_bb": float(np.mean(np.square(frame["q_b"]))),
        "empirical_h_ab": float(np.mean(frame["q_a"] * frame["q_b"])),
        "joint_h_aa": float(frame["joint_h_aa"].mean()),
        "joint_h_bb": float(frame["joint_h_bb"].mean()),
        "joint_h_ab": float(frame["joint_h_ab"].mean()),
        "separable_h_aa": float(frame["separable_h_aa"].mean()),
        "separable_h_bb": float(frame["separable_h_bb"].mean()),
        "separable_h_ab": float(frame["separable_h_ab"].mean()),
        "empirical_mean_l_squared": empirical_l2,
        "joint_mean_variance_l": joint_variance,
        "separable_mean_variance_l": separable_variance,
        "empirical_over_joint_pooled": joint_ratio,
        "empirical_over_separable_pooled": separable_ratio,
        "mean_individual_standardized_joint": float(
            np.mean(np.square(frame["contrast_l"]) / frame["joint_variance_l"])
        ),
        "mean_individual_standardized_separable": float(
            np.mean(np.square(frame["contrast_l"]) / frame["separable_variance_l"])
        ),
        "closer_variance_model": (
            "joint"
            if abs(np.log(joint_ratio)) < abs(np.log(separable_ratio))
            else "separable"
        ),
        "maximum_grid_error_standardized": float(
            frame["maximum_grid_error_standardized"].max()
        ),
        "maximum_source_error_standardized": float(
            frame["maximum_source_error_standardized"].max()
        ),
    }


def _summaries(samples: pd.DataFrame) -> pd.DataFrame:
    rows = [_summarize_group(samples, "all")]
    for (time_k, time_l), group in samples.groupby(["time_k", "time_l"], sort=True):
        rows.append(_summarize_group(group, f"time_{time_k}_{time_l}"))
    for handedness, group in samples.groupby("handedness", sort=True):
        rows.append(_summarize_group(group, f"mirror_{handedness}"))
    return pd.DataFrame(rows)


def _write_figure(summary: pd.DataFrame, output_dir: Path) -> None:
    time_summary = summary[summary["group"].str.startswith("time_")].copy()
    x = np.arange(len(time_summary))
    labels = [value.removeprefix("time_").replace("_", "-") for value in time_summary["group"]]
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    width = 0.25
    axes[0].bar(
        x - width,
        time_summary["empirical_mean_l_squared"],
        width,
        label="empirical",
        color="0.25",
    )
    axes[0].bar(
        x,
        time_summary["joint_mean_variance_l"],
        width,
        label="fitted joint GC",
        color="tab:blue",
    )
    axes[0].bar(
        x + width,
        time_summary["separable_mean_variance_l"],
        width,
        label="matched separable",
        color="tab:orange",
    )
    axes[0].set_ylabel("mean L squared or model Var(L)")
    axes[0].set_title("Fixed canonical contrast by adjacent time pair")
    axes[0].legend(frameon=False, fontsize=8)

    total = summary.loc[summary["group"] == "all"].iloc[0]
    component_labels = ["Var(QA)", "Var(QB)", "Cov(QA,QB)"]
    empirical = [total["empirical_h_aa"], total["empirical_h_bb"], total["empirical_h_ab"]]
    joint = [total["joint_h_aa"], total["joint_h_bb"], total["joint_h_ab"]]
    separable = [
        total["separable_h_aa"],
        total["separable_h_bb"],
        total["separable_h_ab"],
    ]
    component_x = np.arange(3)
    axes[1].bar(component_x - width, empirical, width, label="empirical", color="0.25")
    axes[1].bar(component_x, joint, width, label="fitted joint GC", color="tab:blue")
    axes[1].bar(
        component_x + width,
        separable,
        width,
        label="matched separable",
        color="tab:orange",
    )
    axes[1].axhline(0.0, color="0.2", linewidth=0.8)
    axes[1].set_xticks(component_x, component_labels)
    axes[1].set_title("Second-moment decomposition")
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2, linewidth=0.6)
    axes[0].set_xticks(x, labels)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "real_gems_fixed_canonical_pilot.png", dpi=220)
    figure.savefig(output_dir / "real_gems_fixed_canonical_pilot.pdf")
    plt.close(figure)


def _report(
    data_file: Path,
    fit_csv: Path,
    fit: pd.Series,
    coefficient_source: Path,
    coefficient_first: float,
    coefficient_second: float,
    monthly_mean: float,
    summary: pd.DataFrame,
) -> str:
    total = summary.loc[summary["group"] == "all"].iloc[0]
    return "\n".join(
        [
            "# Fixed canonical contrast on GEMS: one-day pilot",
            "",
            "This is a descriptive application of the contrast fixed in the exact-comoving simulation. No contrast search, covariance refit, or coefficient re-optimization was performed.",
            "",
            "## Inputs",
            "",
            f"- Data: `{data_file}`",
            f"- Existing fit: `{fit_csv}`; strategy `{fit['strategy']}`, day `{fit['day']}`",
            f"- Fixed coefficients: `{coefficient_source}`; d1=`{coefficient_first:.17g}`, d2=`{coefficient_second:.17g}`",
            f"- Monthly centering mean reproduced from the selected July region: `{monthly_mean:.17g}`",
            "- Mean model removed: intercept, centered source latitude, and seven nominal time indicators using the saved GLS coefficients.",
            "",
            "## Transfer from the simulation",
            "",
            "The two mirror geometries use A endpoints (+/-2 fitted latitude ranges, +/-2 fitted longitude ranges) and B endpoints (+/-1 fitted latitude range, +/-2 fitted longitude ranges). Each endpoint follows the saved fitted advection over time. Continuous targets are mapped to nearest regular-grid cells; covariance expectations use the actual source coordinates of the retained observations.",
            "",
            "## Descriptive result",
            "",
            f"- Complete translated contrast evaluations: `{int(total['sample_count'])}`",
            f"- Empirical mean L^2: `{total['empirical_mean_l_squared']:.8g}`",
            f"- Fitted joint-GC mean Var(L): `{total['joint_mean_variance_l']:.8g}`; empirical/model ratio `{total['empirical_over_joint_pooled']:.6f}`",
            f"- Matched-separable mean Var(L): `{total['separable_mean_variance_l']:.8g}`; empirical/model ratio `{total['empirical_over_separable_pooled']:.6f}`",
            f"- By absolute log variance ratio, the descriptive closer model is `{total['closer_variance_model']}`.",
            f"- Cross covariance Cov(QA,QB): empirical `{total['empirical_h_ab']:.8g}`, fitted joint GC `{total['joint_h_ab']:.8g}`, matched separable `{total['separable_h_ab']:.8g}`.",
            "",
            "The individual QA and QB variances are comparatively similar under the two fitted constructions; most of their separation for this fixed diagnostic is in the cross covariance. The comparison is nevertheless about this one fixed second moment only. A ratio nearer one does not establish that the corresponding full covariance model is correct.",
            "",
            "## Interpretation boundary",
            "",
            "This is not a calibrated real-data hypothesis test. The translated contrasts overlap heavily, the covariance parameters and mean were fitted to the same day, nearest-cell mapping perturbs exact comoving geometry, and the real truth is unknown. The result is therefore a model-checking demonstration, not validation of the diagnostic or evidence that one model is true.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "python apply_fixed_canonical_contrast_real_gems.py",
            "```",
            "",
        ]
    )


def main() -> None:
    args = build_parser().parse_args()
    data_file = args.data_file.expanduser().resolve()
    fit_csv = args.fit_csv.expanduser().resolve()
    oracle_dir = args.oracle_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    coefficient_source = oracle_dir / "global_two_rectangle_search/global_pair_ties.csv"
    for path in (data_file, fit_csv, coefficient_source):
        if not path.is_file():
            raise FileNotFoundError(path)

    fit_table = pd.read_csv(fit_csv)
    selected_fit = fit_table.loc[
        (fit_table["strategy"] == args.strategy)
        & (fit_table["day_idx"].astype(int) == int(args.day_index))
    ]
    if len(selected_fit) != 1:
        raise ValueError(
            f"expected one fit for strategy={args.strategy!r}, day={args.day_index}; "
            f"found {len(selected_fit)}"
        )
    fit = selected_fit.iloc[0]
    if not np.isclose(float(fit["est_nugget"]), 0.0):
        raise ValueError("this pilot expects the saved zero-nugget fit")

    ties = pd.read_csv(coefficient_source, float_precision="round_trip")
    coefficient_first = float(ties.iloc[0]["raw_coefficient_first"])
    coefficient_second = float(ties.iloc[0]["raw_coefficient_second"])
    if not np.allclose(
        ties["raw_coefficient_first"], coefficient_first, rtol=0.0, atol=1.0e-14
    ) or not np.allclose(
        ties["raw_coefficient_second"], coefficient_second, rtol=0.0, atol=1.0e-14
    ):
        raise ValueError("strict-tie rows do not share the saved coefficient pair")

    frames, monthly_mean = _load_filtered_frames(
        data_file,
        (float(args.latitude_min), float(args.latitude_max)),
        (float(args.longitude_min), float(args.longitude_max)),
    )
    keys = sorted(frames)
    start = int(args.day_index) * 8
    day_keys = keys[start : start + 8]
    if len(day_keys) != 8:
        raise ValueError(f"day index {args.day_index} does not have eight frames")
    day_frames = [frames[key] for key in day_keys]
    if any(len(frame) != int(fit["n_grid_full"]) for frame in day_frames):
        raise ValueError("loaded grid size differs from the saved fit")

    cube = _grid_cube(day_frames, monthly_mean, fit)
    samples = _canonical_samples(
        cube, fit, coefficient_first, coefficient_second
    )
    summary = _summaries(samples)
    _atomic_csv(output_dir / "translated_contrast_samples.csv", samples)
    _atomic_csv(output_dir / "real_gems_fixed_canonical_summary.csv", summary)
    _write_figure(summary, output_dir)
    manifest = {
        "data_file": str(data_file),
        "fit_csv": str(fit_csv),
        "coefficient_source": str(coefficient_source),
        "strategy": str(fit["strategy"]),
        "day_index": int(args.day_index),
        "day": str(fit["day"]),
        "day_keys": day_keys,
        "monthly_mean": monthly_mean,
        "coefficient_first": coefficient_first,
        "coefficient_second": coefficient_second,
        "fit_parameters": {
            name: float(fit[name])
            for name in (
                "est_sigmasq",
                "est_range_lat",
                "est_range_lon",
                "est_range_time",
                "est_advec_lat",
                "est_advec_lon",
                "est_nugget",
                "gc_alpha",
                "gc_beta",
                "gls_lat_mean",
            )
        },
        "interpretation": "descriptive same-day fixed-contrast model check; not a calibrated test",
    }
    _atomic_text(output_dir / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    _atomic_text(
        output_dir / "REPORT.md",
        _report(
            data_file,
            fit_csv,
            fit,
            coefficient_source,
            coefficient_first,
            coefficient_second,
            monthly_mean,
            summary,
        ),
    )
    print(f"Wrote real GEMS pilot to {output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
