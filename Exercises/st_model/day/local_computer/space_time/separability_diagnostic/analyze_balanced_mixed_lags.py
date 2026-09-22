#!/usr/bin/env python3
"""Audit where the selected design samples the sqrt/nonseparable contrast.

This is a response-free, no-refit structural audit of the completed five-day
pilot.  It separates two quantities that should not be conflated:

1. the analytic, margin-matched gap between ``exp(-hypot(s, t))`` and
   ``exp(-s-t)`` at the truth geometry; and
2. the actual correlation difference between the true joint model and the
   already fitted null, whose ranges and advection were allowed to change.

The script reads only coordinates and design identifiers from the selected
point table.  In particular, it does not load the response or residual
columns and it does not optimize any parameter.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm

from diagnostic_core import (
    CovarianceParameters,
    advected_separable_correlation,
    balanced_fixed_radius_gaps,
    joint_matern_half_correlation,
    pairwise_lags,
    same_margin_exponential_correlation_gap,
    same_margin_exponential_log_gap,
    squared_exponential_correlation_from_norms,
    standardized_moving_lag_norms,
)


HERE = Path(__file__).resolve().parent
DEFAULT_PILOT = HERE / "outputs/nugget0_five_day_092226"
DEFAULT_OUTPUT = DEFAULT_PILOT / "balanced_mixed_lag_audit"
COORDINATE_COLUMNS = (
    "date",
    "time_index",
    "anchor_rank",
    "source_latitude",
    "source_longitude",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--spatial-plot-limit", type=float, default=6.0)
    parser.add_argument("--relevant-radius", type=float, default=4.0)
    parser.add_argument("--spatial-bins", type=int, default=48)
    parser.add_argument("--temporal-bins", type=int, default=28)
    return parser


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
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


def parameters_from_dict(values: dict[str, Any]) -> CovarianceParameters:
    return CovarianceParameters(
        variance=float(values["variance"]),
        range_lat=float(values["range_lat"]),
        range_lon=float(values["range_lon"]),
        range_time=float(values["range_time"]),
        advec_lat=float(values["advec_lat"]),
        advec_lon=float(values["advec_lon"]),
        nugget=float(values.get("nugget", 0.0)),
    )


def load_pair_geometry(
    point_path: Path,
    truth: CovarianceParameters,
    fitted_null: CovarianceParameters,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    # Deliberately restrict the parser: response and true_residual are never loaded.
    points = pd.read_csv(point_path, usecols=list(COORDINATE_COLUMNS))
    all_values: dict[str, list[np.ndarray]] = {
        "spatial_norm": [],
        "temporal_norm": [],
        "matched_log_gap": [],
        "matched_correlation_gap": [],
        "fitted_correlation_difference": [],
    }
    day_rows: list[dict[str, Any]] = []
    for date, frame in points.groupby("date", sort=True):
        frame = frame.sort_values(["time_index", "anchor_rank"])
        coordinates = frame[["source_latitude", "source_longitude", "time_index"]].to_numpy(
            dtype=np.float64
        )
        geometry = pairwise_lags(coordinates)
        spatial_norm, temporal_norm = standardized_moving_lag_norms(geometry, truth)
        truth_correlation = joint_matern_half_correlation(geometry, truth)
        fitted_correlation = advected_separable_correlation(geometry, fitted_null)
        upper = np.triu_indices(len(coordinates), k=1)
        spatial_upper = spatial_norm[upper]
        temporal_upper = temporal_norm[upper]
        matched_log_gap = same_margin_exponential_log_gap(spatial_upper, temporal_upper)
        matched_correlation_gap = same_margin_exponential_correlation_gap(
            spatial_upper, temporal_upper
        )
        fitted_difference = truth_correlation[upper] - fitted_correlation[upper]
        all_values["spatial_norm"].append(spatial_upper)
        all_values["temporal_norm"].append(temporal_upper)
        all_values["matched_log_gap"].append(matched_log_gap)
        all_values["matched_correlation_gap"].append(matched_correlation_gap)
        all_values["fitted_correlation_difference"].append(fitted_difference)
        day_rows.append(
            {
                "date": date,
                "observations": len(coordinates),
                "unordered_nondiagonal_pairs": len(spatial_upper),
                "mean_matched_correlation_gap": float(np.mean(matched_correlation_gap)),
                "mean_absolute_fitted_correlation_difference": float(
                    np.mean(np.abs(fitted_difference))
                ),
            }
        )
    return pd.DataFrame(day_rows), {
        name: np.concatenate(chunks) for name, chunks in all_values.items()
    }


def binned_pair_table(
    values: dict[str, np.ndarray],
    *,
    spatial_limit: float,
    temporal_limit: float,
    spatial_bins: int,
    temporal_bins: int,
) -> pd.DataFrame:
    spatial_edges = np.linspace(0.0, spatial_limit, spatial_bins + 1)
    temporal_edges = np.linspace(0.0, temporal_limit, temporal_bins + 1)
    sample = np.column_stack([values["spatial_norm"], values["temporal_norm"]])
    count, _, _ = np.histogram2d(sample[:, 0], sample[:, 1], bins=(spatial_edges, temporal_edges))

    weighted: dict[str, np.ndarray] = {}
    for name in (
        "matched_log_gap",
        "matched_correlation_gap",
        "fitted_correlation_difference",
    ):
        total, _, _ = np.histogram2d(
            sample[:, 0],
            sample[:, 1],
            bins=(spatial_edges, temporal_edges),
            weights=values[name],
        )
        weighted[name] = np.divide(
            total,
            count,
            out=np.full_like(total, np.nan),
            where=count > 0,
        )

    records: list[dict[str, Any]] = []
    visible_count = float(np.sum(count))
    for spatial_index in range(spatial_bins):
        for temporal_index in range(temporal_bins):
            records.append(
                {
                    "spatial_bin_lower": spatial_edges[spatial_index],
                    "spatial_bin_upper": spatial_edges[spatial_index + 1],
                    "temporal_bin_lower": temporal_edges[temporal_index],
                    "temporal_bin_upper": temporal_edges[temporal_index + 1],
                    "pair_count": int(count[spatial_index, temporal_index]),
                    "fraction_of_visible_pairs": (
                        count[spatial_index, temporal_index] / visible_count
                        if visible_count
                        else np.nan
                    ),
                    "mean_margin_matched_log_gap": weighted["matched_log_gap"][
                        spatial_index, temporal_index
                    ],
                    "mean_margin_matched_correlation_gap": weighted["matched_correlation_gap"][
                        spatial_index, temporal_index
                    ],
                    "mean_fitted_null_actual_correlation_difference": weighted[
                        "fitted_correlation_difference"
                    ][spatial_index, temporal_index],
                }
            )
    return pd.DataFrame(records)


def metric_row(section: str, metric: str, value: float, definition: str) -> dict[str, Any]:
    return {
        "section": section,
        "metric": metric,
        "value": float(value),
        "definition": definition,
    }


def build_summary(
    values: dict[str, np.ndarray],
    *,
    spatial_limit: float,
    temporal_limit: float,
    relevant_radius: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    spatial = values["spatial_norm"]
    temporal = values["temporal_norm"]
    radius = np.hypot(spatial, temporal)
    mixed = (spatial > 1.0e-12) & (temporal > 1.0e-12)
    temporal_axis = temporal <= 1.0e-12
    spatial_axis = spatial <= 1.0e-12
    relevant = radius <= relevant_radius
    balance_score = np.divide(
        2.0 * spatial * temporal,
        radius**2,
        out=np.zeros_like(radius),
        where=radius > 0.0,
    )
    near_balanced = balance_score >= 0.9
    visible = (spatial <= spatial_limit) & (temporal <= temporal_limit)
    matched_log = values["matched_log_gap"]
    matched_correlation = values["matched_correlation_gap"]
    fitted_difference = values["fitted_correlation_difference"]
    total = len(spatial)

    rows = [
        metric_row("occupancy", "unordered_nondiagonal_pairs", total, "all five days"),
        metric_row("occupancy", "mixed_pair_fraction", np.mean(mixed), "s>0 and t>0"),
        metric_row("occupancy", "temporal_axis_pair_fraction", np.mean(temporal_axis), "t=0"),
        metric_row("occupancy", "moving_spatial_axis_pair_fraction", np.mean(spatial_axis), "s=0"),
        metric_row(
            "occupancy",
            "correlation_relevant_pair_fraction",
            np.mean(relevant),
            f"sqrt(s^2+t^2)<={relevant_radius:g}",
        ),
        metric_row(
            "occupancy",
            "near_balanced_fraction_among_relevant",
            np.mean(near_balanced[relevant]) if np.any(relevant) else np.nan,
            "2*s*t/(s^2+t^2)>=0.9 among correlation-relevant pairs",
        ),
        metric_row(
            "occupancy",
            "near_balanced_relevant_fraction_of_all_pairs",
            np.mean(near_balanced & relevant),
            "correlation-relevant and 2*s*t/(s^2+t^2)>=0.9",
        ),
        metric_row(
            "occupancy",
            "figure_window_pair_fraction",
            np.mean(visible),
            "pairs represented in the occupancy panel",
        ),
        metric_row(
            "margin_matched_structural_gap",
            "mean_log_correlation_gap_relevant",
            np.mean(matched_log[relevant]),
            "mean of s+t-hypot(s,t) for relevant pairs",
        ),
        metric_row(
            "margin_matched_structural_gap",
            "mean_correlation_gap_relevant",
            np.mean(matched_correlation[relevant]),
            "same truth ranges/advection and therefore exactly matched axis margins",
        ),
        metric_row(
            "margin_matched_structural_gap",
            "maximum_correlation_gap_observed",
            np.max(matched_correlation),
            "max of R_joint-R_separable with truth parameters shared",
        ),
        metric_row(
            "fitted_null_actual_difference",
            "mean_signed_correlation_difference",
            np.mean(fitted_difference),
            "R_true_joint(truth)-R_separable(fitted null)",
        ),
        metric_row(
            "fitted_null_actual_difference",
            "mean_absolute_correlation_difference",
            np.mean(np.abs(fitted_difference)),
            "ranges and advection differ between the two models",
        ),
        metric_row(
            "fitted_null_actual_difference",
            "mean_absolute_correlation_difference_relevant",
            np.mean(np.abs(fitted_difference[relevant])),
            f"restricted to sqrt(s^2+t^2)<={relevant_radius:g}",
        ),
        metric_row(
            "fitted_null_actual_difference",
            "rmse_correlation_difference",
            np.sqrt(np.mean(fitted_difference**2)),
            "ranges and advection differ between the two models",
        ),
        metric_row(
            "fitted_null_actual_difference",
            "rmse_correlation_difference_relevant",
            np.sqrt(np.mean(fitted_difference[relevant] ** 2)),
            f"restricted to sqrt(s^2+t^2)<={relevant_radius:g}",
        ),
        metric_row(
            "fitted_null_actual_difference",
            "truth_correlation_greater_fraction",
            np.mean(fitted_difference > 0.0),
            "sign is not constrained after fitting",
        ),
    ]

    radii = np.asarray([0.5, 1.0, 2.0, 3.0, relevant_radius])
    maximum_log, maximum_correlation = balanced_fixed_radius_gaps(radii)
    balanced = [
        {
            "joint_radius": radius_value,
            "balanced_spatial_norm": radius_value / np.sqrt(2.0),
            "balanced_temporal_norm": radius_value / np.sqrt(2.0),
            "maximum_log_correlation_gap": log_value,
            "maximum_correlation_gap": correlation_value,
        }
        for radius_value, log_value, correlation_value in zip(
            radii, maximum_log, maximum_correlation
        )
    ]

    no_sqrt = squared_exponential_correlation_from_norms(spatial[:10_000], temporal[:10_000])
    factorized = np.exp(-(spatial[:10_000] ** 2)) * np.exp(-(temporal[:10_000] ** 2))
    report = {
        "definitions": {
            "standardized_spatial_norm": "s=||(h-vu)/ell_space||_2 using truth parameters",
            "standardized_temporal_norm": "t=|u|/ell_time using truth parameters",
            "margin_matched_structural_gap": (
                "exp(-hypot(s,t)) - exp(-s-t); both models use the truth "
                "ranges/advection and have identical exponential axis margins"
            ),
            "fitted_null_actual_correlation_difference": (
                "R_true_joint(truth parameters) - R_separable(fitted parameters); "
                "this includes compensation by fitted ranges and advection"
            ),
        },
        "sqrt_removal": {
            "literal_result": "exp(-(s^2+t^2)) = exp(-s^2) exp(-t^2)",
            "exactly_separable": True,
            "maximum_factorization_error_checked": float(np.max(np.abs(no_sqrt - factorized))),
            "preserves_original_exponential_axis_margins": False,
            "preserves_likelihood_or_optimizer_in_general": False,
            "reason": (
                "on an axis exp(-s^2) replaces exp(-s), so covariance values and "
                "the likelihood objective change even though distance ordering is monotone"
            ),
        },
        "pair_occupancy": {
            "total_unordered_nondiagonal_pairs": total,
            "mixed_fraction": float(np.mean(mixed)),
            "correlation_relevant_radius": relevant_radius,
            "correlation_relevant_fraction": float(np.mean(relevant)),
            "near_balanced_fraction_among_relevant": (
                float(np.mean(near_balanced[relevant])) if np.any(relevant) else None
            ),
            "near_balanced_relevant_fraction_of_all_pairs": float(
                np.mean(near_balanced & relevant)
            ),
            "figure_spatial_limit": spatial_limit,
            "figure_temporal_limit": temporal_limit,
            "figure_window_fraction": float(np.mean(visible)),
        },
        "balanced_fixed_radius_maxima": balanced,
        "metrics": rows,
        "interpretation": (
            "The analytic margin-matched gap isolates the square-root geometry. "
            "The fitted-null difference is the relevant post-optimization discrepancy "
            "and may have either sign. They are intentionally reported separately."
        ),
    }
    return pd.DataFrame(rows), report


def plot_audit(
    path: Path,
    values: dict[str, np.ndarray],
    binned: pd.DataFrame,
    *,
    spatial_limit: float,
    temporal_limit: float,
    spatial_bins: int,
    temporal_bins: int,
) -> None:
    spatial_grid = np.linspace(0.0, spatial_limit, 301)
    temporal_grid = np.linspace(0.0, temporal_limit, 241)
    spatial_mesh, temporal_mesh = np.meshgrid(spatial_grid, temporal_grid)
    analytic_gap = same_margin_exponential_correlation_gap(spatial_mesh, temporal_mesh)

    shape = (spatial_bins, temporal_bins)
    count = binned["pair_count"].to_numpy().reshape(shape).T
    fitted_mean = (
        binned["mean_fitted_null_actual_correlation_difference"].to_numpy().reshape(shape).T
    )
    spatial_edges = np.linspace(0.0, spatial_limit, spatial_bins + 1)
    temporal_edges = np.linspace(0.0, temporal_limit, temporal_bins + 1)

    figure, axes = plt.subplots(1, 3, figsize=(15.5, 4.7), constrained_layout=True)
    analytic_image = axes[0].pcolormesh(
        spatial_grid,
        temporal_grid,
        analytic_gap,
        shading="auto",
        cmap="magma",
    )
    figure.colorbar(analytic_image, ax=axes[0], label=r"$R_{joint}-R_{sep}$")
    axes[0].set_title("Margin-matched structural gap")

    positive_count = count[count > 0]
    occupancy_image = axes[1].pcolormesh(
        spatial_edges,
        temporal_edges,
        np.ma.masked_where(count <= 0, count),
        shading="auto",
        cmap="viridis",
        norm=LogNorm(vmin=max(1.0, float(np.min(positive_count))), vmax=float(np.max(count))),
    )
    figure.colorbar(occupancy_image, ax=axes[1], label="unordered pair count (log scale)")
    axes[1].set_title("Selected-design pair occupancy")

    finite = fitted_mean[np.isfinite(fitted_mean)]
    absolute_limit = float(np.max(np.abs(finite))) if finite.size else 1.0
    actual_image = axes[2].pcolormesh(
        spatial_edges,
        temporal_edges,
        np.ma.masked_invalid(fitted_mean),
        shading="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-absolute_limit, vcenter=0.0, vmax=absolute_limit),
    )
    figure.colorbar(
        actual_image,
        ax=axes[2],
        label=r"mean $R_{true}-R_{fitted\ null}$",
    )
    axes[2].set_title("Actual gap after null fitting")

    balanced_end = min(spatial_limit, temporal_limit)
    for axis in axes:
        axis.plot([0.0, balanced_end], [0.0, balanced_end], "w--", lw=1.2, alpha=0.9)
        axis.set_xlim(0.0, spatial_limit)
        axis.set_ylim(0.0, temporal_limit)
        axis.set_xlabel(r"moving spatial norm $s$")
        axis.set_ylabel(r"temporal norm $t$")
    axes[1].lines[-1].set_color("black")
    axes[2].lines[-1].set_color("black")
    figure.suptitle(
        "Square-root contrast: analytic structure, sampled lags, and fitted compensation",
        fontsize=13,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = args.pilot_dir / "run_manifest.json"
    point_path = args.pilot_dir / "selected_flow_tube_points.csv"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    truth = parameters_from_dict(manifest["truth"])
    fitted_null = parameters_from_dict(manifest["null"]["fit"]["parameters"])

    day_summary, values = load_pair_geometry(point_path, truth, fitted_null)
    temporal_limit = max(float(np.max(values["temporal_norm"])), 1.0e-12)
    binned = binned_pair_table(
        values,
        spatial_limit=args.spatial_plot_limit,
        temporal_limit=temporal_limit,
        spatial_bins=args.spatial_bins,
        temporal_bins=args.temporal_bins,
    )
    summary, report = build_summary(
        values,
        spatial_limit=args.spatial_plot_limit,
        temporal_limit=temporal_limit,
        relevant_radius=args.relevant_radius,
    )
    report.update(
        {
            "source_manifest": manifest_path,
            "source_point_table": point_path,
            "columns_loaded_from_point_table": list(COORDINATE_COLUMNS),
            "response_columns_loaded": False,
            "parameters": {
                "truth": truth.to_dict(),
                "fitted_null": fitted_null.to_dict(),
            },
            "per_day": day_summary.to_dict(orient="records"),
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(args.output_dir / "summary_metrics.csv", summary)
    atomic_csv(args.output_dir / "pair_occupancy_by_bin.csv", binned)
    atomic_csv(args.output_dir / "per_day_summary.csv", day_summary)
    atomic_json(args.output_dir / "structural_audit_summary.json", report)
    figure_path = args.output_dir / "analytic_gap_vs_pair_occupancy.png"
    plot_audit(
        figure_path,
        values,
        binned,
        spatial_limit=args.spatial_plot_limit,
        temporal_limit=temporal_limit,
        spatial_bins=args.spatial_bins,
        temporal_bins=args.temporal_bins,
    )
    print(f"wrote audit to {args.output_dir.resolve()}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
