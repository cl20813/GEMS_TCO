#!/usr/bin/env python3
"""Decompose the first negative two-rectangle intrinsic contrast.

This is a deterministic post-processing analysis of the exact-comoving
rectangle-dictionary oracle.  It reconstructs the two rectangles selected at
the minimizing greedy step ``k=2`` and reports exactly how their off-diagonal
covariance-difference term overcomes both positive single-rectangle terms.

No response values are read.  The analysis uses the population covariance
matrices and fitted-null parameters already recorded by the oracle run; it is
not a calibrated test or a power calculation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analyze_mode_count_path import atomic_csv, atomic_json, covariance_parameters, sha256
from diagnostic_core import (
    advected_separable_covariance,
    joint_matern_half_covariance,
    pairwise_lags,
)
from rectangle_dictionary_core import (
    constrained_relative_residual,
    filter_variance_metrics,
    rectangle_matrix,
    rectangle_structure_errors,
    solve_constrained_contrast,
    standardize_dictionary,
)
from run_rectangle_dictionary_oracle import git_state, output_inventory, runtime_environment


HERE = Path(__file__).resolve().parent
DEFAULT_ORACLE_DIR = HERE / "outputs/exact_comoving_rectangle_dictionary_092226"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE_DIR)
    parser.add_argument("--output-dir", type=Path)
    return parser


def labeled_matrix(matrix: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """Return a square matrix with explicit row and column rectangle IDs."""

    frame = pd.DataFrame(np.asarray(matrix, dtype=np.float64), columns=labels)
    frame.insert(0, "row_rectangle_id", labels)
    return frame


def quadratic_components(
    matrix: np.ndarray,
    coefficients: np.ndarray,
    *,
    matrix_name: str,
) -> pd.DataFrame:
    """Split a two-variable quadratic form into two diagonals and one cross term."""

    matrix = np.asarray(matrix, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if matrix.shape != (2, 2) or coefficients.shape != (2,):
        raise ValueError("quadratic_components requires a 2-by-2 matrix and two coefficients")
    a, b = coefficients
    rows = [
        {
            "matrix": matrix_name,
            "component": "alpha1_squared_times_m11",
            "formula": "alpha1^2 * M11",
            "value": float(a * a * matrix[0, 0]),
        },
        {
            "matrix": matrix_name,
            "component": "alpha2_squared_times_m22",
            "formula": "alpha2^2 * M22",
            "value": float(b * b * matrix[1, 1]),
        },
        {
            "matrix": matrix_name,
            "component": "two_alpha1_alpha2_times_m12",
            "formula": "2 * alpha1 * alpha2 * M12",
            "value": float(2.0 * a * b * matrix[0, 1]),
        },
    ]
    total = float(coefficients @ matrix @ coefficients)
    rows.append(
        {
            "matrix": matrix_name,
            "component": "total",
            "formula": "alpha' * M * alpha",
            "value": total,
        }
    )
    return pd.DataFrame(rows)


def pair_quadratic_attribution(
    weights: np.ndarray,
    difference_matrix: np.ndarray,
    point_table: pd.DataFrame,
    *,
    contribution_column: str,
    zero_tolerance: float = 1.0e-14,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Attribute a quadratic form to unordered pairs of nonzero observations."""

    weights = np.asarray(weights, dtype=np.float64)
    difference_matrix = np.asarray(difference_matrix, dtype=np.float64)
    nonzero = np.flatnonzero(np.abs(weights) > zero_tolerance)
    sub_weights = weights[nonzero]
    sub_difference = difference_matrix[np.ix_(nonzero, nonzero)]
    contribution_matrix = np.outer(sub_weights, sub_weights) * sub_difference
    points = point_table.iloc[nonzero].reset_index(drop=True)
    labels = [
        f"a{int(row.anchor_index):02d}@t{int(row.time_index)}"
        for row in points.itertuples(index=False)
    ]
    rows: list[dict[str, Any]] = []
    for first in range(len(nonzero)):
        for second in range(first, len(nonzero)):
            first_row = points.iloc[first]
            second_row = points.iloc[second]
            multiplicity = 1.0 if first == second else 2.0
            contribution = multiplicity * contribution_matrix[first, second]
            spatial_distance = float(
                np.hypot(
                    first_row["standardized_moving_latitude"]
                    - second_row["standardized_moving_latitude"],
                    first_row["standardized_moving_longitude"]
                    - second_row["standardized_moving_longitude"],
                )
            )
            rows.append(
                {
                    "first_observation_index": int(nonzero[first]),
                    "second_observation_index": int(nonzero[second]),
                    "first_label": labels[first],
                    "second_label": labels[second],
                    "first_anchor": int(first_row["anchor_index"]),
                    "second_anchor": int(second_row["anchor_index"]),
                    "first_time": int(first_row["time_index"]),
                    "second_time": int(second_row["time_index"]),
                    "temporal_lag": int(
                        abs(int(first_row["time_index"]) - int(second_row["time_index"]))
                    ),
                    "standardized_spatial_distance": spatial_distance,
                    "multiplicity": int(multiplicity),
                    "weight_product": float(sub_weights[first] * sub_weights[second]),
                    "covariance_difference": float(sub_difference[first, second]),
                    contribution_column: float(contribution),
                }
            )
    return pd.DataFrame(rows), contribution_matrix, labels


def plot_filter_heatmap(
    weights: np.ndarray,
    *,
    time_count: int,
    latitude_count: int,
    longitude_count: int,
    path: Path,
) -> None:
    cube = np.asarray(weights).reshape(time_count, latitude_count, longitude_count)
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
    fig.suptitle("Negative two-rectangle filter")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_two_matrices(
    intrinsic_matrix: np.ndarray,
    null_gram: np.ndarray,
    labels: list[str],
    path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True)
    for axis, matrix, title, cmap in (
        (axes[0], intrinsic_matrix, "Standardized intrinsic D", "coolwarm"),
        (axes[1], null_gram, "Standardized fitted-null G0", "viridis"),
    ):
        if cmap == "coolwarm":
            limit = float(np.max(np.abs(matrix)))
            image = axis.imshow(matrix, cmap=cmap, vmin=-limit, vmax=limit)
        else:
            image = axis.imshow(matrix, cmap=cmap)
        for row in range(2):
            for column in range(2):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.6f}",
                    ha="center",
                    va="center",
                    color="black",
                )
        axis.set_xticks([0, 1], labels=["Q1", "Q2"])
        axis.set_yticks([0, 1], labels=["Q1", "Q2"])
        axis.set_title(title)
        fig.colorbar(image, ax=axis, shrink=0.78)
    fig.suptitle(f"Q1={labels[0]}   Q2={labels[1]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_quadratic_terms(components: pd.DataFrame, path: Path) -> None:
    component_order = (
        "alpha1_squared_times_m11",
        "alpha2_squared_times_m22",
        "two_alpha1_alpha2_times_m12",
    )
    labels = ("alpha1^2 M11", "alpha2^2 M22", "2 alpha1 alpha2 M12")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for axis, matrix_name, title in (
        (axes[0], "intrinsic_difference", "Intrinsic variance-difference terms"),
        (axes[1], "fitted_null", "Fitted-null normalization terms"),
    ):
        subset = components.loc[
            (components["matrix"] == matrix_name) & (components["component"].isin(component_order))
        ].set_index("component")
        values = np.asarray([subset.loc[name, "value"] for name in component_order])
        colors = ["#b33b2e" if value >= 0.0 else "#3568a8" for value in values]
        axis.bar(np.arange(3), values, color=colors)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(np.arange(3), labels=labels, rotation=18, ha="right")
        axis.set_ylabel("quadratic contribution")
        axis.set_title(title)
        for position, value in enumerate(values):
            axis.text(
                position,
                value,
                f"{value:+.5f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
            )
        axis.grid(axis="y", alpha=0.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pair_contributions(
    contribution_matrix: np.ndarray,
    labels: list[str],
    path: Path,
) -> None:
    limit = float(np.max(np.abs(contribution_matrix)))
    fig, axis = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    image = axis.imshow(
        contribution_matrix,
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    axis.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(labels)), labels=labels)
    axis.set_title("Ordered-pair contributions to w' (Sigma1-SigmaM) w")
    fig.colorbar(image, ax=axis, shrink=0.82, label="w_i w_j Delta_ij")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_time_lag_attribution(
    intrinsic: pd.DataFrame,
    total: pd.DataFrame,
    path: Path,
) -> None:
    frame = intrinsic.merge(total, on="temporal_lag", how="outer").fillna(0.0)
    lags = frame["temporal_lag"].to_numpy(dtype=np.float64)
    intrinsic_values = frame["intrinsic_quadratic_contribution"].to_numpy(dtype=np.float64)
    total_values = frame["total_quadratic_contribution"].to_numpy(dtype=np.float64)
    fig, axis = plt.subplots(figsize=(8, 4.4), constrained_layout=True)
    axis.bar(
        lags - 0.18,
        intrinsic_values,
        width=0.36,
        color="#3568a8",
        label="intrinsic: Sigma1-SigmaM",
    )
    axis.bar(
        lags + 0.18,
        total_values,
        width=0.36,
        color="#b33b2e",
        label="total: Sigma1-Sigma0",
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(
        xlabel="absolute temporal lag (hours)",
        ylabel="summed quadratic contribution",
        title="Final two-rectangle filter: attribution by time lag",
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_report(
    path: Path,
    rectangle_summary: pd.DataFrame,
    intrinsic_raw: np.ndarray,
    null_raw: np.ndarray,
    truth_raw: np.ndarray,
    matched_raw: np.ndarray,
    intrinsic_standardized: np.ndarray,
    null_standardized: np.ndarray,
    components: pd.DataFrame,
    metrics: dict[str, float],
    intrinsic_time_attribution: pd.DataFrame,
    total_time_attribution: pd.DataFrame,
    *,
    minimum_eigenvalue: float,
    maximum_eigenvalue: float,
    filter_matrix_rank: int,
    filter_singular_values: np.ndarray,
    rank_one_energy_fraction: float,
    first_step_tie_count: int,
    null_variance_error: float,
    reconstruction_error: float,
    constrained_residual: float,
    reduced_residual: float,
    full_space_relative_residual: float,
) -> None:
    first, second = list(rectangle_summary.itertuples(index=False))
    intrinsic_terms = components.loc[components["matrix"] == "intrinsic_difference"].set_index(
        "component"
    )
    null_terms = components.loc[components["matrix"] == "fitted_null"].set_index("component")
    d11 = float(intrinsic_standardized[0, 0])
    d22 = float(intrinsic_standardized[1, 1])
    d12 = float(intrinsic_standardized[0, 1])
    determinant = float(np.linalg.det(intrinsic_standardized))
    lines = [
        "# First negative two-rectangle contrast",
        "",
        "This is a population-covariance decomposition of the exact-comoving oracle. It reads no responses and is not a calibrated test.",
        "",
        "## The two actual rectangles",
        "",
        f"- `Q1 = {first.rectangle_id}` compares anchors `{int(first.spatial_endpoint_p)}` and `{int(first.spatial_endpoint_q)}` between hours `{int(first.time_endpoint_k)}` and `{int(first.time_endpoint_l)}`.",
        f"- `Q2 = {second.rectangle_id}` compares anchors `{int(second.spatial_endpoint_p)}` and `{int(second.spatial_endpoint_q)}` between hours `{int(second.time_endpoint_k)}` and `{int(second.time_endpoint_l)}`.",
        "",
        "With the orientation `+p,k -q,k -p,l +q,l`, the final unit-null-variance projection is",
        "",
        f"`L = ({first.raw_rectangle_coefficient:+.12g}) Q1 + ({second.raw_rectangle_coefficient:+.12g}) Q2`.",
        "",
        "Equivalently,",
        "",
        f"`Q1 = Y[{int(first.spatial_endpoint_q)},{int(first.time_endpoint_l)}] - Y[{int(first.spatial_endpoint_p)},{int(first.time_endpoint_l)}] - Y[{int(first.spatial_endpoint_q)},{int(first.time_endpoint_k)}] + Y[{int(first.spatial_endpoint_p)},{int(first.time_endpoint_k)}]`,",
        "",
        f"`Q2 = Y[{int(second.spatial_endpoint_q)},{int(second.time_endpoint_l)}] - Y[{int(second.spatial_endpoint_p)},{int(second.time_endpoint_l)}] - Y[{int(second.spatial_endpoint_q)},{int(second.time_endpoint_k)}] + Y[{int(second.spatial_endpoint_p)},{int(second.time_endpoint_k)}]`.",
        "",
        f"Writing `A_t=Y[0,t]-Y[24,t]`, `B_t=Y[1,t]-Y[23,t]`, `a={abs(first.raw_rectangle_coefficient):.12g}`, and `b={abs(second.raw_rectangle_coefficient):.12g}`, the same filter is exactly `L=a(A_0-A_7)-b(B_1-B_6)`.",
        "",
        "## Two-by-two generalized problem",
        "",
        "In the original raw rectangle coordinates,",
        "",
        "```text",
        f"D_raw  = [[{intrinsic_raw[0, 0]: .9f}, {intrinsic_raw[0, 1]: .9f}],",
        f"          [{intrinsic_raw[1, 0]: .9f}, {intrinsic_raw[1, 1]: .9f}]]",
        f"G0_raw = [[{null_raw[0, 0]: .9f}, {null_raw[0, 1]: .9f}],",
        f"          [{null_raw[1, 0]: .9f}, {null_raw[1, 1]: .9f}]]",
        "```",
        "",
        "After scaling each rectangle to unit fitted-null variance,",
        "",
        "```text",
        f"D  = [[{intrinsic_standardized[0, 0]: .9f}, {intrinsic_standardized[0, 1]: .9f}],",
        f"      [{intrinsic_standardized[1, 0]: .9f}, {intrinsic_standardized[1, 1]: .9f}]]",
        f"G0 = [[{null_standardized[0, 0]: .9f}, {null_standardized[0, 1]: .9f}],",
        f"      [{null_standardized[1, 0]: .9f}, {null_standardized[1, 1]: .9f}]]",
        "```",
        "",
        f"Both single-rectangle intrinsic differences are positive (`d11={d11:.9g}`, `d22={d22:.9g}`), but `d12^2={d12**2:.9g}` exceeds `d11*d22={d11*d22:.9g}`. Thus `det(D)={determinant:.9g}<0`, and D has one negative direction.",
        f"The generalized eigenvalues are `{minimum_eigenvalue:.9g}` and `{maximum_eigenvalue:.9g}`.",
        "",
        "## Why the combined direction is negative",
        "",
        "| contribution | intrinsic value | fitted-null normalization value |",
        "|---|---:|---:|",
        f"| alpha1^2 M11 | {intrinsic_terms.loc['alpha1_squared_times_m11', 'value']:.9g} | {null_terms.loc['alpha1_squared_times_m11', 'value']:.9g} |",
        f"| alpha2^2 M22 | {intrinsic_terms.loc['alpha2_squared_times_m22', 'value']:.9g} | {null_terms.loc['alpha2_squared_times_m22', 'value']:.9g} |",
        f"| 2 alpha1 alpha2 M12 | {intrinsic_terms.loc['two_alpha1_alpha2_times_m12', 'value']:.9g} | {null_terms.loc['two_alpha1_alpha2_times_m12', 'value']:.9g} |",
        f"| total | {intrinsic_terms.loc['total', 'value']:.9g} | {null_terms.loc['total', 'value']:.9g} |",
        "",
        f"The two positive diagonal terms sum to `{intrinsic_terms.loc['alpha1_squared_times_m11', 'value'] + intrinsic_terms.loc['alpha2_squared_times_m22', 'value']:.9g}`, while the cross term is `{intrinsic_terms.loc['two_alpha1_alpha2_times_m12', 'value']:.9g}`: its magnitude is `{abs(intrinsic_terms.loc['two_alpha1_alpha2_times_m12', 'value']) / (intrinsic_terms.loc['alpha1_squared_times_m11', 'value'] + intrinsic_terms.loc['alpha2_squared_times_m22', 'value']):.3f}` times larger.",
        f"The raw cross-covariance is `{truth_raw[0, 1]:.9g}` under truth and `{matched_raw[0, 1]:.9g}` under the matched-margin model. Because the rectangles enter with opposite signs, the larger positive truth cross-covariance creates stronger cancellation.",
        "This concerns a covariance difference; no variance itself is negative, and it does not assert that either model's correlation is negative.",
        "",
        "## Fitted-null compensation",
        "",
        "| quantity | value |",
        "|---|---:|",
        f"| v0 | {metrics['v0']:.9g} |",
        f"| vM | {metrics['v_matched']:.9g} |",
        f"| v1 | {metrics['v_true']:.9g} |",
        f"| intrinsic v1-vM | {metrics['delta_intrinsic']:.9g} |",
        f"| compensation vM-v0 | {metrics['delta_compensation']:.9g} |",
        f"| total v1-v0 | {metrics['delta_total']:.9g} |",
        f"| rho0=v1/v0 | {metrics['rho_fitted']:.9g} |",
        f"| rhoM=v1/vM | {metrics['rho_matched']:.9g} |",
        f"| g(rho0) | {metrics['g_fitted']:.9g} |",
        "",
        "Here the fitted-null refitting effect has the same negative sign as the intrinsic difference, so it reinforces rather than offsets it. Truth variance is about 14.74% below fitted-null variance for this fixed filter.",
        "",
        "## What temporal pattern does the final filter represent?",
        "",
        f"The observation-weight matrix has rank `{filter_matrix_rank}` with nonzero singular values `{filter_singular_values[0]:.9g}` and `{filter_singular_values[1]:.9g}`. Its best rank-one approximation retains only `{rank_one_energy_fraction:.2%}` of Frobenius energy.",
        "The two rectangles use different spatial contrasts, so the filter cannot be written exactly as one spatial vector times a single temporal pattern such as `W0-W1+W6-W7`.",
        "It is exactly the sum of two separable four-point contrasts: an outer diagonal contrast over hours 0 and 7 and a different inner diagonal contrast over hours 1 and 6. Rectangle widths 7 and 5 hours therefore describe the atoms, not a unique temporal scale of the combined filter.",
        "",
        "Quadratic-form attribution for the final filter:",
        "",
        "| absolute time lag | intrinsic Sigma1-SigmaM | total Sigma1-Sigma0 |",
        "|---:|---:|---:|",
    ]
    combined_attribution = intrinsic_time_attribution.merge(
        total_time_attribution,
        on="temporal_lag",
        how="outer",
    ).fillna(0.0)
    for row in combined_attribution.itertuples(index=False):
        lines.append(
            f"| {int(row.temporal_lag)} | {row.intrinsic_quadratic_contribution:.9g} | {row.total_quadratic_contribution:.9g} |"
        )
    lines.extend(
        [
            "",
            "These lag sums are quadratic-form attributions of the final filter, not rectangle coefficients and not independent pieces of information.",
            "The atom endpoint spans are 5 and 7 hours, but the strongest negative intrinsic contribution occurs at lag 1; endpoint width, weight support, and quadratic-form lag attribution are distinct summaries.",
            "",
            "## Selection and interpretation boundary",
            "",
            f"- The first minimizing rectangle had `{first_step_tie_count}` candidates within the predeclared tie tolerance. `Q1` is the lexicographically selected representative.",
            "- `Q2` is the best second addition conditional on that Q1. This is a greedy path, not an exhaustive globally optimal search over every rectangle pair.",
            "- The overall sign of L is arbitrary; the relative minus sign between Q1 and Q2 is the identified feature.",
            "- This is an exact-grid oracle population result, not evidence of significance or performance on the warped GEMS observation geometry.",
            "",
            "## Numerical checks",
            "",
            f"- Null normalization error: `{null_variance_error:.3e}`.",
            f"- Reconstruction error from the two raw rectangles: `{reconstruction_error:.3e}`.",
            f"- Basis-invariant constrained residual: `{constrained_residual:.3e}`.",
            f"- Direct reduced 2-by-2 residual: `{reduced_residual:.3e}`.",
            f"- Full observation-space relative residual: `{full_space_relative_residual:.3e}`; it need not vanish because this is a two-rectangle constrained eigenproblem.",
            "- The saved oracle k=2 objective, recomputed generalized eigenvalue, intrinsic quadratic sum, and pair-attribution sum are required to agree before outputs are written.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    oracle_dir = args.oracle_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else oracle_dir / "negative_two_rectangle_decomposition"
    )
    manifest_path = oracle_dir / "experiment_manifest.json"
    greedy_path = oracle_dir / "greedy_path.csv"
    metadata_path = oracle_dir / "rectangle_dictionary_metadata.csv"
    points_path = oracle_dir / "exact_comoving_points.csv"
    for path in (manifest_path, greedy_path, metadata_path, points_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not bool(manifest.get("oracle_not_real_data_test", False)):
        raise ValueError("input manifest is not the certified covariance oracle")
    truth = covariance_parameters(manifest["truth"])
    fitted_null = covariance_parameters(manifest["fitted_null_fixed_advection"])
    jitter = float(manifest["numerics"]["numerical_jitter_ratio"])
    geometry_record = manifest["geometry"]
    anchor_count = int(geometry_record["anchor_count"])
    time_count = int(geometry_record["time_count"])
    geometry_config = manifest["run_configuration"]["geometry"]
    latitude_count = int(geometry_config["latitude_count"])
    longitude_count = int(geometry_config["longitude_count"])
    if anchor_count != latitude_count * longitude_count:
        raise ValueError("manifest geometry is inconsistent")

    point_table = pd.read_csv(points_path).sort_values(
        ["time_index", "anchor_index"], ignore_index=True
    )
    expected_order = [
        (time_index, anchor_index)
        for time_index in range(time_count)
        for anchor_index in range(anchor_count)
    ]
    observed_order = list(
        point_table[["time_index", "anchor_index"]].itertuples(index=False, name=None)
    )
    if observed_order != expected_order:
        raise ValueError("exact-comoving point table is not in certified time-major order")
    coordinates = point_table[["source_latitude", "source_longitude", "time_index"]].to_numpy(
        dtype=np.float64
    )
    lag_geometry = pairwise_lags(coordinates)
    true_covariance = joint_matern_half_covariance(
        lag_geometry,
        truth,
        numerical_jitter_ratio=jitter,
    )
    matched_covariance = advected_separable_covariance(
        lag_geometry,
        truth,
        numerical_jitter_ratio=jitter,
    )
    null_covariance = advected_separable_covariance(
        lag_geometry,
        fitted_null,
        numerical_jitter_ratio=jitter,
    )
    intrinsic_difference = true_covariance - matched_covariance

    greedy = pd.read_csv(greedy_path)
    selected_row = greedy.loc[(greedy["branch"] == "negative") & (greedy["size"] == 2)]
    if len(selected_row) != 1:
        raise ValueError("expected exactly one negative greedy k=2 row")
    selected_row = selected_row.iloc[0]
    selected_indices = [
        int(value) for value in str(selected_row["selected_rectangle_indices"]).split(";")
    ]
    if len(selected_indices) != 2:
        raise ValueError("negative greedy k=2 row does not contain two rectangles")

    metadata = pd.read_csv(metadata_path).set_index("rectangle_index", drop=False)
    selected_metadata = metadata.loc[selected_indices].reset_index(drop=True)
    endpoints = selected_metadata[
        [
            "spatial_endpoint_p",
            "spatial_endpoint_q",
            "time_endpoint_k",
            "time_endpoint_l",
        ]
    ].to_numpy(dtype=np.int64)
    raw_rectangles, returned_endpoints = rectangle_matrix(
        anchor_count,
        time_count,
        endpoints,
    )
    if not np.array_equal(endpoints, returned_endpoints):
        raise ArithmeticError("rectangle endpoints changed during reconstruction")
    standardized_rectangles, scales, raw_null_variances = standardize_dictionary(
        raw_rectangles,
        null_covariance,
    )
    values, coefficients, filters = solve_constrained_contrast(
        intrinsic_difference,
        null_covariance,
        standardized_rectangles,
    )
    minimum_eigenvalue = float(values[0])
    maximum_eigenvalue = float(values[-1])
    standardized_coefficients = coefficients[:, 0]
    raw_coefficients = standardized_coefficients / scales
    weights = filters[:, 0]
    reconstructed_weights = raw_rectangles @ raw_coefficients
    reconstruction_error = float(
        np.linalg.norm(reconstructed_weights - weights) / np.linalg.norm(weights)
    )
    if not np.isclose(
        minimum_eigenvalue,
        float(selected_row["objective_mu"]),
        rtol=1.0e-9,
        atol=1.0e-11,
    ):
        raise ArithmeticError("recomputed k=2 eigenvalue differs from the oracle greedy path")

    projected_raw = {
        "truth": raw_rectangles.T @ true_covariance @ raw_rectangles,
        "matched_margin": raw_rectangles.T @ matched_covariance @ raw_rectangles,
        "fitted_null": raw_rectangles.T @ null_covariance @ raw_rectangles,
    }
    projected_raw["intrinsic_difference"] = projected_raw["truth"] - projected_raw["matched_margin"]
    projected_standardized = {
        name: standardized_rectangles.T @ covariance @ standardized_rectangles
        for name, covariance in (
            ("truth", true_covariance),
            ("matched_margin", matched_covariance),
            ("fitted_null", null_covariance),
        )
    }
    projected_standardized["intrinsic_difference"] = (
        projected_standardized["truth"] - projected_standardized["matched_margin"]
    )
    labels = selected_metadata["rectangle_id"].astype(str).tolist()

    component_frames = [
        quadratic_components(
            matrix,
            standardized_coefficients,
            matrix_name=name,
        )
        for name, matrix in projected_standardized.items()
    ]
    components = pd.concat(component_frames, ignore_index=True)
    intrinsic_total = float(
        components.loc[
            (components["matrix"] == "intrinsic_difference") & (components["component"] == "total"),
            "value",
        ].iloc[0]
    )
    null_total = float(
        components.loc[
            (components["matrix"] == "fitted_null") & (components["component"] == "total"),
            "value",
        ].iloc[0]
    )
    if not np.isclose(intrinsic_total, minimum_eigenvalue, rtol=1.0e-11, atol=1.0e-12):
        raise ArithmeticError("quadratic components do not sum to the eigenvalue")
    if not np.isclose(null_total, 1.0, rtol=1.0e-11, atol=1.0e-12):
        raise ArithmeticError("fitted-null quadratic components do not normalize to one")

    metrics = filter_variance_metrics(
        weights,
        true_covariance,
        matched_covariance,
        null_covariance,
    )
    temporal_error, spatial_error = rectangle_structure_errors(
        weights,
        anchor_count=anchor_count,
        time_count=time_count,
    )
    if (
        projected_standardized["intrinsic_difference"][0, 0] <= 0.0
        or projected_standardized["intrinsic_difference"][1, 1] <= 0.0
    ):
        raise ArithmeticError("a selected single rectangle is not intrinsically positive")
    if minimum_eigenvalue >= 0.0:
        raise ArithmeticError("the selected two-rectangle direction is not negative")

    rectangle_rows = []
    observation_rows = []
    component_weights = raw_rectangles * raw_coefficients[None, :]
    for selection_order, row in enumerate(selected_metadata.itertuples(index=False), start=1):
        rectangle_rows.append(
            {
                **row._asdict(),
                "selection_order": selection_order,
                "null_standard_deviation_raw": float(scales[selection_order - 1]),
                "standardized_coefficient": float(standardized_coefficients[selection_order - 1]),
                "raw_rectangle_coefficient": float(raw_coefficients[selection_order - 1]),
                "single_intrinsic_objective_recomputed": float(
                    projected_standardized["intrinsic_difference"][
                        selection_order - 1, selection_order - 1
                    ]
                ),
            }
        )
        nonzero = np.flatnonzero(raw_rectangles[:, selection_order - 1])
        for observation_index in nonzero:
            point = point_table.iloc[observation_index]
            observation_rows.append(
                {
                    "selection_order": selection_order,
                    "rectangle_id": row.rectangle_id,
                    "observation_index": int(observation_index),
                    "anchor_index": int(point["anchor_index"]),
                    "time_index": int(point["time_index"]),
                    "rectangle_sign": int(raw_rectangles[observation_index, selection_order - 1]),
                    "raw_rectangle_coefficient": float(raw_coefficients[selection_order - 1]),
                    "contribution_to_final_weight": float(
                        component_weights[observation_index, selection_order - 1]
                    ),
                    "moving_latitude": float(point["moving_latitude"]),
                    "moving_longitude": float(point["moving_longitude"]),
                    "source_latitude": float(point["source_latitude"]),
                    "source_longitude": float(point["source_longitude"]),
                }
            )
    rectangle_summary = pd.DataFrame(rectangle_rows)
    rectangle_observations = pd.DataFrame(observation_rows)

    filter_weights = point_table.copy()
    for column in range(2):
        filter_weights[f"rectangle_{column + 1}_weight_contribution"] = component_weights[:, column]
    filter_weights["final_weight"] = weights
    filter_weights["is_nonzero"] = np.abs(weights) > 1.0e-14
    nonzero_filter_weights = filter_weights.loc[filter_weights["is_nonzero"]].copy()
    filter_matrix = weights.reshape(time_count, anchor_count)
    filter_matrix_rank = int(np.linalg.matrix_rank(filter_matrix, tol=1.0e-12))
    filter_singular_values = np.linalg.svd(filter_matrix, compute_uv=False)
    rank_one_energy_fraction = float(
        filter_singular_values[0] ** 2 / np.square(filter_singular_values).sum()
    )

    pair_attribution, contribution_matrix, pair_labels = pair_quadratic_attribution(
        weights,
        intrinsic_difference,
        point_table,
        contribution_column="intrinsic_quadratic_contribution",
    )
    if not np.isclose(
        pair_attribution["intrinsic_quadratic_contribution"].sum(),
        minimum_eigenvalue,
        rtol=1.0e-11,
        atol=1.0e-12,
    ):
        raise ArithmeticError("pair attribution does not sum to the intrinsic objective")
    total_difference = true_covariance - null_covariance
    total_pair_attribution, _, total_pair_labels = pair_quadratic_attribution(
        weights,
        total_difference,
        point_table,
        contribution_column="total_quadratic_contribution",
    )
    if total_pair_labels != pair_labels:
        raise ArithmeticError("intrinsic and total attributions use different observations")
    if not np.isclose(
        total_pair_attribution["total_quadratic_contribution"].sum(),
        metrics["delta_total"],
        rtol=1.0e-11,
        atol=1.0e-12,
    ):
        raise ArithmeticError("total pair attribution does not sum to truth-null difference")
    time_attribution = (
        pair_attribution.groupby("temporal_lag", as_index=False)["intrinsic_quadratic_contribution"]
        .sum()
        .sort_values("temporal_lag", ignore_index=True)
    )
    time_attribution["fraction_of_signed_total"] = (
        time_attribution["intrinsic_quadratic_contribution"] / minimum_eigenvalue
    )
    total_time_attribution = (
        total_pair_attribution.groupby("temporal_lag", as_index=False)[
            "total_quadratic_contribution"
        ]
        .sum()
        .sort_values("temporal_lag", ignore_index=True)
    )
    total_time_attribution["fraction_of_signed_total"] = (
        total_time_attribution["total_quadratic_contribution"] / metrics["delta_total"]
    )
    space_time_attribution = (
        pair_attribution.assign(
            standardized_spatial_distance_rounded=pair_attribution[
                "standardized_spatial_distance"
            ].round(8)
        )
        .groupby(
            ["temporal_lag", "standardized_spatial_distance_rounded"],
            as_index=False,
        )["intrinsic_quadratic_contribution"]
        .sum()
        .sort_values(
            ["temporal_lag", "standardized_spatial_distance_rounded"],
            ignore_index=True,
        )
    )
    total_space_time_attribution = (
        total_pair_attribution.assign(
            standardized_spatial_distance_rounded=total_pair_attribution[
                "standardized_spatial_distance"
            ].round(8)
        )
        .groupby(
            ["temporal_lag", "standardized_spatial_distance_rounded"],
            as_index=False,
        )["total_quadratic_contribution"]
        .sum()
        .sort_values(
            ["temporal_lag", "standardized_spatial_distance_rounded"],
            ignore_index=True,
        )
    )

    constrained_residual = constrained_relative_residual(
        weights,
        minimum_eigenvalue,
        standardized_rectangles,
        intrinsic_difference,
        null_covariance,
    )
    reduced_residual_vector = (
        projected_standardized["intrinsic_difference"] @ standardized_coefficients
        - minimum_eigenvalue * projected_standardized["fitted_null"] @ standardized_coefficients
    )
    reduced_residual = float(np.linalg.norm(reduced_residual_vector))
    full_residual_vector = intrinsic_difference @ weights - minimum_eigenvalue * (
        null_covariance @ weights
    )
    full_residual_denominator = np.linalg.norm(intrinsic_difference @ weights) + abs(
        minimum_eigenvalue
    ) * np.linalg.norm(null_covariance @ weights)
    full_space_relative_residual = float(
        np.linalg.norm(full_residual_vector) / full_residual_denominator
    )
    first_step_tie_count = int(
        greedy.loc[
            (greedy["branch"] == "negative") & (greedy["size"] == 1), "winner_tie_count"
        ].iloc[0]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_csv(output_dir / "rectangle_summary.csv", rectangle_summary)
    atomic_csv(output_dir / "rectangle_observation_terms.csv", rectangle_observations)
    atomic_csv(output_dir / "final_filter_weights.csv", filter_weights)
    atomic_csv(output_dir / "nonzero_final_filter_weights.csv", nonzero_filter_weights)
    atomic_csv(output_dir / "quadratic_term_decomposition.csv", components)
    atomic_csv(output_dir / "intrinsic_pair_contributions.csv", pair_attribution)
    atomic_csv(output_dir / "intrinsic_time_lag_attribution.csv", time_attribution)
    atomic_csv(output_dir / "intrinsic_space_time_lag_attribution.csv", space_time_attribution)
    atomic_csv(output_dir / "total_pair_contributions.csv", total_pair_attribution)
    atomic_csv(output_dir / "total_time_lag_attribution.csv", total_time_attribution)
    atomic_csv(output_dir / "total_space_time_lag_attribution.csv", total_space_time_attribution)
    for basis_name, matrices in (
        ("raw", projected_raw),
        ("standardized", projected_standardized),
    ):
        for matrix_name, matrix in matrices.items():
            atomic_csv(
                output_dir / f"{matrix_name}_matrix_{basis_name}.csv",
                labeled_matrix(matrix, labels),
            )

    plot_filter_heatmap(
        weights,
        time_count=time_count,
        latitude_count=latitude_count,
        longitude_count=longitude_count,
        path=output_dir / "figures/negative_two_filter_heatmap.png",
    )
    plot_two_matrices(
        projected_standardized["intrinsic_difference"],
        projected_standardized["fitted_null"],
        labels,
        output_dir / "figures/two_rectangle_matrices.png",
    )
    plot_quadratic_terms(
        components,
        output_dir / "figures/quadratic_term_decomposition.png",
    )
    plot_pair_contributions(
        contribution_matrix,
        pair_labels,
        output_dir / "figures/intrinsic_pair_contribution_heatmap.png",
    )
    plot_time_lag_attribution(
        time_attribution,
        total_time_attribution,
        output_dir / "figures/intrinsic_time_lag_attribution.png",
    )
    null_variance_error = abs(metrics["v0"] - 1.0)
    write_report(
        output_dir / "REPORT.md",
        rectangle_summary,
        projected_raw["intrinsic_difference"],
        projected_raw["fitted_null"],
        projected_raw["truth"],
        projected_raw["matched_margin"],
        projected_standardized["intrinsic_difference"],
        projected_standardized["fitted_null"],
        components,
        metrics,
        time_attribution,
        total_time_attribution,
        minimum_eigenvalue=minimum_eigenvalue,
        maximum_eigenvalue=maximum_eigenvalue,
        filter_matrix_rank=filter_matrix_rank,
        filter_singular_values=filter_singular_values,
        rank_one_energy_fraction=rank_one_energy_fraction,
        first_step_tie_count=first_step_tie_count,
        null_variance_error=null_variance_error,
        reconstruction_error=reconstruction_error,
        constrained_residual=constrained_residual,
        reduced_residual=reduced_residual,
        full_space_relative_residual=full_space_relative_residual,
    )

    script_path = Path(__file__).resolve()
    source_paths = (
        script_path,
        HERE / "rectangle_dictionary_core.py",
        HERE / "diagnostic_core.py",
        HERE / "run_rectangle_dictionary_oracle.py",
        HERE / "analyze_mode_count_path.py",
    )
    input_paths = (manifest_path, greedy_path, metadata_path, points_path)
    analysis_manifest = {
        "analysis": "negative_two_rectangle_decomposition",
        "oracle_directory": str(oracle_dir),
        "output_directory": str(output_dir),
        "command": [sys.executable, *sys.argv],
        "response_columns_read": False,
        "population_covariance_analysis_not_test": True,
        "selected_rectangle_indices": selected_indices,
        "selected_rectangle_ids": labels,
        "standardized_coefficients": standardized_coefficients,
        "raw_rectangle_coefficients": raw_coefficients,
        "generalized_eigenvalues": values,
        "variance_metrics": metrics,
        "intrinsic_matrix_determinant": float(
            np.linalg.det(projected_standardized["intrinsic_difference"])
        ),
        "d12_squared_exceeds_d11_d22": bool(
            projected_standardized["intrinsic_difference"][0, 1] ** 2
            > projected_standardized["intrinsic_difference"][0, 0]
            * projected_standardized["intrinsic_difference"][1, 1]
        ),
        "filter_matrix_rank": filter_matrix_rank,
        "filter_singular_values": filter_singular_values,
        "rank_one_frobenius_energy_fraction": rank_one_energy_fraction,
        "first_step_tie_count": first_step_tie_count,
        "selection_scope": (
            "second rectangle is greedy-optimal conditional on the lexicographically "
            "selected first-step tie representative; not an exhaustive pair optimum"
        ),
        "maximum_rectangle_structure_error": max(temporal_error, spatial_error),
        "null_variance_error": null_variance_error,
        "filter_reconstruction_relative_error": reconstruction_error,
        "basis_invariant_constrained_relative_residual": constrained_residual,
        "direct_reduced_residual_norm": reduced_residual,
        "full_space_relative_residual_not_constrained_target": full_space_relative_residual,
        "intrinsic_attribution_sum": float(
            pair_attribution["intrinsic_quadratic_contribution"].sum()
        ),
        "total_attribution_sum": float(
            total_pair_attribution["total_quadratic_contribution"].sum()
        ),
        "numerical_jitter_ratio": jitter,
        "git": git_state(),
        "runtime_environment": runtime_environment(),
        "inputs": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in input_paths
        ],
        "source_files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in source_paths
        ],
    }
    # ``output_inventory`` excludes the oracle's ``experiment_manifest.json``.
    # This post-processing run has a differently named manifest, so remove it
    # explicitly to avoid recording a stale, self-referential hash on reruns.
    analysis_manifest["output_files"] = [
        row for row in output_inventory(output_dir) if row["path"] != "analysis_manifest.json"
    ]
    atomic_json(output_dir / "analysis_manifest.json", analysis_manifest)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "rectangles": labels,
                "minimum_eigenvalue": minimum_eigenvalue,
                "rho_fitted": metrics["rho_fitted"],
            }
        )
    )


if __name__ == "__main__":
    main()
