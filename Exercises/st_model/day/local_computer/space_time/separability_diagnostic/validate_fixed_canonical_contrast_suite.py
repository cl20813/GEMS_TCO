#!/usr/bin/env python3
"""Small validation suite for the already selected canonical contrast family.

The 14 strict-tie filters from the saved exact-comoving oracle are fixed before
this analysis.  No contrast search or covariance fitting is performed.  For
each scenario, the script computes the exact 14-dimensional covariance of the
filter outputs and then simulates only that reduced Gaussian vector.

This is a controlled known-null check.  It validates the fixed statistic under
specified population covariances; it does not yet validate uncertainty from
estimating the null covariance on the same data.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from diagnostic_core import (
    CovarianceParameters,
    advected_separable_correlation,
    covariance_from_correlation,
    joint_matern_half_covariance,
    pairwise_lags,
)
from rectangle_dictionary_core import rectangle_matrix


HERE = Path(__file__).resolve().parent
DEFAULT_ORACLE_DIR = HERE / "outputs/exact_comoving_rectangle_dictionary_092226"
DEFAULT_OUTPUT_DIR = DEFAULT_ORACLE_DIR / "fixed_canonical_validation_suite"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replicates", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260923)
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


def _parameters(values: dict[str, Any], **updates: float) -> CovarianceParameters:
    names = (
        "variance",
        "range_lat",
        "range_lon",
        "range_time",
        "advec_lat",
        "advec_lon",
        "nugget",
    )
    selected = {name: float(values[name]) for name in names}
    selected.update({name: float(value) for name, value in updates.items()})
    return CovarianceParameters(**selected)


def _ordered_points(
    points: pd.DataFrame, anchor_count: int, time_count: int
) -> pd.DataFrame:
    ordered = points.sort_values(["time_index", "anchor_index"]).reset_index(drop=True)
    expected = pd.MultiIndex.from_product(
        [range(time_count), range(anchor_count)],
        names=["time_index", "anchor_index"],
    ).to_frame(index=False)
    actual = ordered[["time_index", "anchor_index"]].astype(np.int64)
    if not actual.reset_index(drop=True).equals(expected):
        raise ValueError("saved points do not have the declared time-major order")
    return ordered


def _fixed_filters(
    metadata: pd.DataFrame,
    ties: pd.DataFrame,
    anchor_count: int,
    time_count: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    endpoint_columns = [
        "spatial_endpoint_p",
        "spatial_endpoint_q",
        "time_endpoint_k",
        "time_endpoint_l",
    ]
    ordered_metadata = metadata.sort_values("rectangle_index").reset_index(drop=True)
    expected = np.arange(len(ordered_metadata), dtype=np.int64)
    if not np.array_equal(
        ordered_metadata["rectangle_index"].to_numpy(dtype=np.int64), expected
    ):
        raise ValueError("rectangle indices are not contiguous dictionary columns")
    endpoints = ordered_metadata[endpoint_columns].to_numpy(dtype=np.int64)
    dictionary, rebuilt = rectangle_matrix(anchor_count, time_count, endpoints)
    if not np.array_equal(endpoints, rebuilt):
        raise AssertionError("rectangle endpoint reconstruction changed the order")

    columns = []
    descriptions = []
    for filter_index, row in enumerate(
        ties.sort_values(["first_rectangle_index", "second_rectangle_index"]).itertuples(
            index=False
        )
    ):
        first = int(row.first_rectangle_index)
        second = int(row.second_rectangle_index)
        coefficients = np.asarray(
            [row.raw_coefficient_first, row.raw_coefficient_second], dtype=np.float64
        )
        weights = dictionary[:, [first, second]] @ coefficients
        columns.append(weights)
        descriptions.append(
            {
                "filter_index": filter_index,
                "first_rectangle_id": row.first_rectangle_id,
                "second_rectangle_id": row.second_rectangle_id,
                "first_raw_coefficient": coefficients[0],
                "second_raw_coefficient": coefficients[1],
                "time_k": int(row.first_time_endpoint_k),
                "time_l": int(row.first_time_endpoint_l),
                "handedness": row.relative_orientation_handedness,
            }
        )
    filters = np.column_stack(columns)
    if filters.shape[1] != 14:
        raise AssertionError(f"expected 14 fixed filters, found {filters.shape[1]}")
    return filters, pd.DataFrame(descriptions)


def _mixture_covariances(
    geometry: Any,
    base: CovarianceParameters,
    numerical_jitter_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Return aligned mixture, crossed mixture, and their matched separator."""

    short_space = CovarianceParameters(
        variance=base.variance,
        range_lat=0.60 * base.range_lat,
        range_lon=0.60 * base.range_lon,
        range_time=1.0,
        advec_lat=base.advec_lat,
        advec_lon=base.advec_lon,
        nugget=0.0,
    )
    long_space = CovarianceParameters(
        variance=base.variance,
        range_lat=2.00 * base.range_lat,
        range_lon=2.00 * base.range_lon,
        range_time=4.0,
        advec_lat=base.advec_lat,
        advec_lon=base.advec_lon,
        nugget=0.0,
    )
    shifted_lat = geometry.delta_lat - base.advec_lat * geometry.delta_time
    shifted_lon = geometry.delta_lon - base.advec_lon * geometry.delta_time
    spatial_short = np.exp(
        -np.hypot(
            shifted_lat / short_space.range_lat,
            shifted_lon / short_space.range_lon,
        )
    )
    spatial_long = np.exp(
        -np.hypot(
            shifted_lat / long_space.range_lat,
            shifted_lon / long_space.range_lon,
        )
    )
    temporal_short = np.exp(-np.abs(geometry.delta_time) / short_space.range_time)
    temporal_long = np.exp(-np.abs(geometry.delta_time) / long_space.range_time)
    weight = 0.5
    aligned_correlation = (
        weight * spatial_short * temporal_short
        + (1.0 - weight) * spatial_long * temporal_long
    )
    crossed_correlation = (
        weight * spatial_short * temporal_long
        + (1.0 - weight) * spatial_long * temporal_short
    )
    matched_correlation = (
        weight * spatial_short + (1.0 - weight) * spatial_long
    ) * (weight * temporal_short + (1.0 - weight) * temporal_long)
    kwargs = {
        "variance": base.variance,
        "nugget": 0.0,
        "numerical_jitter_ratio": numerical_jitter_ratio,
    }
    return (
        covariance_from_correlation(aligned_correlation, **kwargs),
        covariance_from_correlation(crossed_correlation, **kwargs),
        covariance_from_correlation(matched_correlation, **kwargs),
        {
            "weight_short_component": weight,
            "short_space_multiplier": 0.60,
            "long_space_multiplier": 2.00,
            "short_temporal_range": short_space.range_time,
            "long_temporal_range": long_space.range_time,
        },
    )


def _sample_statistics(
    covariance: np.ndarray,
    null_variances: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> np.ndarray:
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    tolerance = 256.0 * np.finfo(np.float64).eps * max(float(eigenvalues.max()), 1.0)
    if float(eigenvalues.min()) < -tolerance:
        raise np.linalg.LinAlgError("filter covariance is not positive semidefinite")
    transform = eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))
    draws = rng.standard_normal((replicates, covariance.shape[0])) @ transform.T
    return np.mean(draws * draws / null_variances[None, :], axis=1)


def _evaluate_scenario(
    name: str,
    description: str,
    filters: np.ndarray,
    null_covariance: np.ndarray,
    alternative_covariance: np.ndarray,
    replicates: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    null_filter_covariance = filters.T @ null_covariance @ filters
    alternative_filter_covariance = filters.T @ alternative_covariance @ filters
    null_variances = np.diag(null_filter_covariance)
    alternative_variances = np.diag(alternative_filter_covariance)
    if np.any(null_variances <= 0.0):
        raise np.linalg.LinAlgError("a canonical filter has nonpositive null variance")

    rng = np.random.default_rng(seed)
    calibration = _sample_statistics(
        null_filter_covariance, null_variances, replicates, rng
    )
    null_evaluation = _sample_statistics(
        null_filter_covariance, null_variances, replicates, rng
    )
    alternative = _sample_statistics(
        alternative_filter_covariance, null_variances, replicates, rng
    )
    lower_05 = float(np.quantile(calibration, 0.05))
    lower_025 = float(np.quantile(calibration, 0.025))
    upper_975 = float(np.quantile(calibration, 0.975))
    population_ratios = alternative_variances / null_variances
    is_null = bool(np.allclose(alternative_covariance, null_covariance, rtol=0.0, atol=0.0))
    row = {
        "scenario": name,
        "description": description,
        "is_null": is_null,
        "replicates_per_distribution": replicates,
        "filter_count": filters.shape[1],
        "population_mean_standardized_energy": float(population_ratios.mean()),
        "minimum_filter_variance_ratio": float(population_ratios.min()),
        "maximum_filter_variance_ratio": float(population_ratios.max()),
        "calibration_lower_05": lower_05,
        "calibration_lower_025": lower_025,
        "calibration_upper_975": upper_975,
        "null_evaluation_mean": float(null_evaluation.mean()),
        "null_lower_tail_rejection_rate": float(np.mean(null_evaluation < lower_05)),
        "null_two_sided_rejection_rate": float(
            np.mean((null_evaluation < lower_025) | (null_evaluation > upper_975))
        ),
        "alternative_evaluation_mean": float(alternative.mean()),
        "alternative_lower_tail_rejection_rate": float(np.mean(alternative < lower_05)),
        "alternative_two_sided_rejection_rate": float(
            np.mean((alternative < lower_025) | (alternative > upper_975))
        ),
    }
    filter_rows = pd.DataFrame(
        {
            "scenario": name,
            "filter_index": np.arange(filters.shape[1]),
            "null_variance": null_variances,
            "alternative_variance": alternative_variances,
            "variance_ratio": population_ratios,
        }
    )
    return row, filter_rows


def _write_figure(summary: pd.DataFrame, output_dir: Path) -> None:
    labels = summary["scenario"].tolist()
    x = np.arange(len(labels))
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    axes[0].bar(x, summary["population_mean_standardized_energy"], color="tab:blue")
    axes[0].axhline(1.0, color="0.25", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("population E[standardized energy]")
    axes[0].set_title("Fixed canonical statistic")

    width = 0.36
    axes[1].bar(
        x - width / 2,
        summary["alternative_lower_tail_rejection_rate"],
        width=width,
        label="pre-specified lower tail",
        color="tab:blue",
    )
    axes[1].bar(
        x + width / 2,
        summary["alternative_two_sided_rejection_rate"],
        width=width,
        label="two-sided",
        color="tab:orange",
    )
    axes[1].axhline(0.05, color="0.25", linestyle="--", linewidth=1.0)
    axes[1].set_ylabel("Monte Carlo rejection rate")
    axes[1].set_title("Known-null calibrated pilot")
    axes[1].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.set_xticks(x)
        axis.set_xticklabels(labels, rotation=28, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.2, linewidth=0.6)
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "fixed_canonical_validation_suite.png", dpi=220)
    figure.savefig(output_dir / "fixed_canonical_validation_suite.pdf")
    plt.close(figure)


def _report(
    oracle_dir: Path,
    summary: pd.DataFrame,
    replicates: int,
    mixture_settings: dict[str, float],
) -> str:
    table_lines = []
    for row in summary.itertuples(index=False):
        table_lines.append(
            f"| `{row.scenario}` | {row.population_mean_standardized_energy:.6f} | "
            f"{row.null_lower_tail_rejection_rate:.4f} | "
            f"{row.alternative_lower_tail_rejection_rate:.4f} | "
            f"{row.alternative_two_sided_rejection_rate:.4f} |"
        )
    return "\n".join(
        [
            "# Fixed canonical contrast: small validation suite",
            "",
            "The 14 strict-tie filters from the original exact-comoving oracle were fixed before defining these scenarios. This run performs no filter search and no covariance fitting.",
            f"Input oracle: `{oracle_dir}`.",
            "",
            "## Statistic and calibration",
            "",
            "For each replicate, the statistic is the mean of the 14 squared canonical filter outputs divided by their scenario-specific separable-null variances. Its population expectation is one under the known null. Because the filters are correlated, null quantiles are calibrated from the exact 14-dimensional Gaussian filter covariance rather than from an independence approximation.",
            "",
            f"Calibration, independent null evaluation, and alternative evaluation each use `{replicates:,}` reduced Gaussian replicates per scenario. The pre-specified direction is the lower tail, matching the negative discrepancy found in the training oracle; a two-sided rate is also reported descriptively.",
            "",
            "| scenario | population mean energy | null lower-tail rate | alternative lower-tail rate | alternative two-sided rate |",
            "|---|---:|---:|---:|---:|",
            *table_lines,
            "",
            "## Scenarios",
            "",
            "- `separable_null`: an exactly separable process; its alternative covariance equals its null covariance.",
            "- `joint_matern_training`: the original joint Matérn-half population benchmark.",
            "- `joint_matern_time_range_1` and `joint_matern_time_range_4`: fixed-filter parameter perturbations with their own matched-margin separable comparators.",
            "- `mixture_aligned`: an equal mixture of short-space/short-time and long-space/long-time separable components.",
            "- `mixture_crossed`: an equal mixture of short-space/long-time and long-space/short-time components. The aligned and crossed mixtures have identical spatial and temporal margins and therefore share the same matched separable comparator.",
            "",
            f"Mixture settings: short spatial multiplier `{mixture_settings['short_space_multiplier']}`, long spatial multiplier `{mixture_settings['long_space_multiplier']}`, short temporal range `{mixture_settings['short_temporal_range']}`, long temporal range `{mixture_settings['long_temporal_range']}`, weight `{mixture_settings['weight_short_component']}`.",
            "",
            "## Interpretation boundary",
            "",
            "This is a controlled known-parameter validation of a pre-fixed statistic. A null rate near 0.05 checks implementation and calibration without selection bias. Alternative rejection rates show whether the original canonical contrast transfers to these specified populations. They do not establish a universally powerful interaction diagnostic, and they do not yet include same-data covariance fitting or real-data uncertainty.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "python validate_fixed_canonical_contrast_suite.py",
            "```",
            "",
        ]
    )


def main() -> None:
    args = build_parser().parse_args()
    oracle_dir = args.oracle_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if args.replicates < 1_000:
        raise ValueError("replicates must be at least 1000")
    manifest_path = oracle_dir / "experiment_manifest.json"
    points_path = oracle_dir / "exact_comoving_points.csv"
    metadata_path = oracle_dir / "rectangle_dictionary_metadata.csv"
    ties_path = oracle_dir / "global_two_rectangle_search/global_pair_ties.csv"
    for path in (manifest_path, points_path, metadata_path, ties_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not bool(manifest.get("oracle_not_real_data_test", False)):
        raise ValueError("the supplied directory is not the exact-comoving oracle")
    anchor_count = int(manifest["geometry"]["anchor_count"])
    time_count = int(manifest["geometry"]["time_count"])
    points = _ordered_points(pd.read_csv(points_path), anchor_count, time_count)
    metadata = pd.read_csv(metadata_path)
    ties = pd.read_csv(ties_path, float_precision="round_trip")
    filters, filter_descriptions = _fixed_filters(
        metadata, ties, anchor_count, time_count
    )
    coordinates = points[
        ["source_latitude", "source_longitude", "time_index"]
    ].to_numpy(dtype=np.float64)
    geometry = pairwise_lags(coordinates)
    jitter = float(manifest["numerics"]["numerical_jitter_ratio"])
    base = _parameters(manifest["truth"])

    base_null = covariance_from_correlation(
        advected_separable_correlation(geometry, base),
        variance=base.variance,
        nugget=base.nugget,
        numerical_jitter_ratio=jitter,
    )
    training_joint = joint_matern_half_covariance(
        geometry, base, numerical_jitter_ratio=jitter
    )
    short_time = _parameters(manifest["truth"], range_time=1.0)
    long_time = _parameters(manifest["truth"], range_time=4.0)
    short_time_null = covariance_from_correlation(
        advected_separable_correlation(geometry, short_time),
        variance=short_time.variance,
        nugget=short_time.nugget,
        numerical_jitter_ratio=jitter,
    )
    long_time_null = covariance_from_correlation(
        advected_separable_correlation(geometry, long_time),
        variance=long_time.variance,
        nugget=long_time.nugget,
        numerical_jitter_ratio=jitter,
    )
    short_time_joint = joint_matern_half_covariance(
        geometry, short_time, numerical_jitter_ratio=jitter
    )
    long_time_joint = joint_matern_half_covariance(
        geometry, long_time, numerical_jitter_ratio=jitter
    )
    mixture_aligned, mixture_crossed, mixture_null, mixture_settings = _mixture_covariances(
        geometry, base, jitter
    )

    scenarios = [
        (
            "separable_null",
            "exact separable null",
            base_null,
            base_null,
        ),
        (
            "joint_matern_training",
            "original joint Matern-half benchmark",
            base_null,
            training_joint,
        ),
        (
            "joint_matern_time_range_1",
            "joint Matern-half with temporal range 1",
            short_time_null,
            short_time_joint,
        ),
        (
            "joint_matern_time_range_4",
            "joint Matern-half with temporal range 4",
            long_time_null,
            long_time_joint,
        ),
        (
            "mixture_aligned",
            "short-space/short-time plus long-space/long-time mixture",
            mixture_null,
            mixture_aligned,
        ),
        (
            "mixture_crossed",
            "short-space/long-time plus long-space/short-time mixture",
            mixture_null,
            mixture_crossed,
        ),
    ]
    rows = []
    filter_rows = []
    for scenario_index, (name, description, null_covariance, alternative_covariance) in enumerate(
        scenarios
    ):
        row, detail = _evaluate_scenario(
            name,
            description,
            filters,
            null_covariance,
            alternative_covariance,
            args.replicates,
            args.seed + 10_000 * scenario_index,
        )
        rows.append(row)
        filter_rows.append(detail)
    summary = pd.DataFrame(rows)
    details = pd.concat(filter_rows, ignore_index=True)

    null_row = summary.loc[summary["scenario"] == "separable_null"].iloc[0]
    monte_carlo_se = math.sqrt(0.05 * 0.95 / args.replicates)
    if abs(float(null_row["null_lower_tail_rejection_rate"]) - 0.05) > 4.0 * monte_carlo_se:
        raise ArithmeticError("known-null lower-tail rate failed its Monte Carlo check")
    if abs(float(null_row["null_two_sided_rejection_rate"]) - 0.05) > 4.0 * monte_carlo_se:
        raise ArithmeticError("known-null two-sided rate failed its Monte Carlo check")

    _atomic_csv(output_dir / "scenario_summary.csv", summary)
    _atomic_csv(output_dir / "filter_variance_ratios.csv", details)
    _atomic_csv(output_dir / "fixed_filter_definitions.csv", filter_descriptions)
    _write_figure(summary, output_dir)
    _atomic_text(
        output_dir / "REPORT.md",
        _report(oracle_dir, summary, args.replicates, mixture_settings),
    )
    print(f"Wrote fixed canonical validation suite to {output_dir}")
    print(
        summary[
            [
                "scenario",
                "population_mean_standardized_energy",
                "alternative_lower_tail_rejection_rate",
                "alternative_two_sided_rejection_rate",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
