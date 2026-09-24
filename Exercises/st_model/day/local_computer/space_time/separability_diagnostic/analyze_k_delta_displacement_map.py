#!/usr/bin/env python3
"""Map the time-difference covariance discrepancy over every spatial lag.

The analysis is restricted to the saved exact-comoving population setting.  It
does not read responses, simulate data, search for contrasts, optimize filter
coefficients, or refit covariance parameters.  For every spatial displacement
available on the saved 5 by 5 moving-coordinate grid, it computes

    K_delta,m(h) = Cov_m(Y(s,0)-Y(s,1), Y(s+h,0)-Y(s+h,1))

directly from four entries of the saved-design population covariance matrix for
the true joint model (m=1) and the matched-margin separable model (m=M).
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
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

from diagnostic_core import (
    CovarianceParameters,
    advected_separable_covariance,
    joint_matern_half_covariance,
    pairwise_lags,
)


HERE = Path(__file__).resolve().parent
DEFAULT_ORACLE_DIR = HERE / "outputs/exact_comoving_rectangle_dictionary_092226"
DEFAULT_OUTPUT_DIR = DEFAULT_ORACLE_DIR / "k_delta_displacement_map"
MODEL_LABELS = ("1", "M")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _parameters(values: dict[str, Any]) -> CovarianceParameters:
    names = (
        "variance",
        "range_lat",
        "range_lon",
        "range_time",
        "advec_lat",
        "advec_lon",
        "nugget",
    )
    return CovarianceParameters(**{name: float(values[name]) for name in names})


def _ordered_points(points: pd.DataFrame, anchor_count: int, time_count: int) -> pd.DataFrame:
    ordered = points.sort_values(["time_index", "anchor_index"]).reset_index(drop=True)
    expected = pd.MultiIndex.from_product(
        [range(time_count), range(anchor_count)], names=["time_index", "anchor_index"]
    ).to_frame(index=False)
    actual = ordered[["time_index", "anchor_index"]].astype(np.int64).reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError("saved points do not have the declared time-major Cartesian order")
    return ordered


def _anchor_maps(
    points: pd.DataFrame,
) -> tuple[dict[int, tuple[int, int]], dict[tuple[int, int], int]]:
    anchors = points.loc[points["time_index"] == 0].sort_values("anchor_index")
    forward: dict[int, tuple[int, int]] = {}
    for row in anchors.itertuples(index=False):
        latitude = float(row.standardized_moving_latitude)
        longitude = float(row.standardized_moving_longitude)
        rounded = (int(round(latitude)), int(round(longitude)))
        if not np.allclose((latitude, longitude), rounded, rtol=0.0, atol=1.0e-12):
            raise ValueError("the saved standardized moving grid is not integer-valued")
        forward[int(row.anchor_index)] = rounded
    reverse = {coordinates: anchor for anchor, coordinates in forward.items()}
    if len(reverse) != len(forward):
        raise ValueError("moving-coordinate anchors are not unique")
    return forward, reverse


def _available_displacements(forward: dict[int, tuple[int, int]]) -> list[tuple[int, int]]:
    coordinates = tuple(forward.values())
    return sorted(
        {
            (second[0] - first[0], second[1] - first[1])
            for first in coordinates
            for second in coordinates
        }
    )


def _k_delta_placements(
    covariance: np.ndarray,
    h: tuple[int, int],
    anchor_count: int,
    forward: dict[int, tuple[int, int]],
    reverse: dict[tuple[int, int], int],
) -> np.ndarray:
    """Evaluate K_delta(h) independently at every valid base anchor."""

    values = []
    for first_anchor, (latitude, longitude) in forward.items():
        second_anchor = reverse.get((latitude + h[0], longitude + h[1]))
        if second_anchor is None:
            continue
        first_time0 = first_anchor
        first_time1 = anchor_count + first_anchor
        second_time0 = second_anchor
        second_time1 = anchor_count + second_anchor
        values.append(
            covariance[first_time0, second_time0]
            - covariance[first_time0, second_time1]
            - covariance[first_time1, second_time0]
            + covariance[first_time1, second_time1]
        )
    if not values:
        raise ValueError(f"no valid placement for displacement {h}")
    return np.asarray(values, dtype=np.float64)


def _displacement_table(
    covariances: dict[str, np.ndarray],
    displacements: list[tuple[int, int]],
    anchor_count: int,
    forward: dict[int, tuple[int, int]],
    reverse: dict[tuple[int, int], int],
) -> pd.DataFrame:
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for h in displacements:
        record: dict[str, Any] = {
            "h": f"({h[0]},{h[1]})",
            "h_lat": h[0],
            "h_lon": h[1],
            "radius_squared": h[0] * h[0] + h[1] * h[1],
            "radius": math.hypot(*h),
        }
        placement_count = None
        for model in MODEL_LABELS:
            values = _k_delta_placements(
                covariances[model], h, anchor_count, forward, reverse
            )
            if placement_count is None:
                placement_count = len(values)
            elif placement_count != len(values):
                raise AssertionError("models used different spatial placements")
            record[f"K_delta_{model}"] = float(np.mean(values))
            record[f"K_delta_{model}_minimum"] = float(np.min(values))
            record[f"K_delta_{model}_maximum"] = float(np.max(values))
            record[f"K_delta_{model}_stationarity_span"] = float(np.ptp(values))
        record["placement_count"] = int(placement_count)
        record["delta_K_delta"] = record["K_delta_1"] - record["K_delta_M"]
        record["absolute_delta_K_delta"] = abs(record["delta_K_delta"])
        records[h] = record

    for h, record in records.items():
        opposite = records[(-h[0], -h[1])]
        record["opposite_lag_error_1"] = abs(record["K_delta_1"] - opposite["K_delta_1"])
        record["opposite_lag_error_M"] = abs(record["K_delta_M"] - opposite["K_delta_M"])
        record["opposite_lag_error_delta"] = abs(
            record["delta_K_delta"] - opposite["delta_K_delta"]
        )
    return pd.DataFrame(records.values()).sort_values(
        ["h_lat", "h_lon"], ignore_index=True
    )


def _radial_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for radius_squared, group in table.groupby("radius_squared", sort=True):
        rows.append(
            {
                "radius_squared": int(radius_squared),
                "radius": float(np.sqrt(radius_squared)),
                "displacement_count": len(group),
                "K_delta_1_mean": float(group["K_delta_1"].mean()),
                "K_delta_1_directional_spread": float(np.ptp(group["K_delta_1"])),
                "K_delta_M_mean": float(group["K_delta_M"].mean()),
                "K_delta_M_directional_spread": float(np.ptp(group["K_delta_M"])),
                "delta_K_delta_mean": float(group["delta_K_delta"].mean()),
                "delta_K_delta_minimum": float(group["delta_K_delta"].min()),
                "delta_K_delta_maximum": float(group["delta_K_delta"].max()),
                "delta_K_delta_directional_spread": float(
                    np.ptp(group["delta_K_delta"])
                ),
            }
        )
    return pd.DataFrame(rows)


def _heatmap_matrix(
    table: pd.DataFrame,
    value: str,
    latitude_values: list[int],
    longitude_values: list[int],
) -> np.ndarray:
    lookup = {
        (int(row.h_lat), int(row.h_lon)): float(getattr(row, value))
        for row in table.itertuples(index=False)
    }
    return np.asarray(
        [[lookup[(latitude, longitude)] for longitude in longitude_values] for latitude in latitude_values],
        dtype=np.float64,
    )


def _write_heatmaps(table: pd.DataFrame, output_dir: Path) -> None:
    latitude_values = sorted(table["h_lat"].astype(int).unique())
    longitude_values = sorted(table["h_lon"].astype(int).unique())
    truth = _heatmap_matrix(table, "K_delta_1", latitude_values, longitude_values)
    matched = _heatmap_matrix(table, "K_delta_M", latitude_values, longitude_values)
    difference = _heatmap_matrix(table, "delta_K_delta", latitude_values, longitude_values)

    figure, axes = plt.subplots(1, 3, figsize=(15.5, 5.1), constrained_layout=True)
    common_norm = Normalize(vmin=min(float(truth.min()), float(matched.min())), vmax=max(float(truth.max()), float(matched.max())))
    extent = (
        longitude_values[0] - 0.5,
        longitude_values[-1] + 0.5,
        latitude_values[0] - 0.5,
        latitude_values[-1] + 0.5,
    )
    first = axes[0].imshow(
        truth,
        origin="lower",
        extent=extent,
        cmap="viridis",
        norm=common_norm,
        interpolation="nearest",
    )
    axes[1].imshow(
        matched,
        origin="lower",
        extent=extent,
        cmap="viridis",
        norm=common_norm,
        interpolation="nearest",
    )
    maximum_absolute_difference = float(np.max(np.abs(difference)))
    difference_norm = TwoSlopeNorm(
        vmin=-maximum_absolute_difference,
        vcenter=0.0,
        vmax=maximum_absolute_difference,
    )
    third = axes[2].imshow(
        difference,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        norm=difference_norm,
        interpolation="nearest",
    )

    titles = (
        r"True joint $K_{\delta,1}(h)$",
        r"Matched separable $K_{\delta,M}(h)$",
        r"Difference $\delta K_\delta(h)$"
        "\nsolid: selected (1,0); dashed: tied shortest lags",
    )
    for axis, title in zip(axes, titles):
        axis.set_title(title)
        axis.set_xlabel(r"$h_{\mathrm{lon}}$ (standardized moving grid)")
        axis.set_ylabel(r"$h_{\mathrm{lat}}$ (standardized moving grid)")
        axis.set_xticks(longitude_values)
        axis.set_yticks(latitude_values)
        axis.axhline(0.0, color="white", linewidth=0.5, alpha=0.55)
        axis.axvline(0.0, color="white", linewidth=0.5, alpha=0.55)
        axis.set_aspect("equal")

    for row_index, latitude in enumerate(latitude_values):
        for column_index, longitude in enumerate(longitude_values):
            value = difference[row_index, column_index]
            text_color = "white" if abs(value) > 0.46 * maximum_absolute_difference else "black"
            axes[2].text(
                longitude,
                latitude,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=6.5,
                color=text_color,
            )

    # The original oracle search selected h=(1,0).  Mark it separately from
    # the three equally short lags so the figure does not imply that its
    # direction is intrinsically unique in this numerator map.
    for latitude, longitude in ((-1, 0), (0, -1), (0, 1)):
        axes[2].add_patch(
            Rectangle(
                (longitude - 0.5, latitude - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="black",
                linewidth=1.0,
                linestyle=(0, (3, 2)),
            )
        )
    axes[2].add_patch(
        Rectangle(
            (-0.5, 0.5),
            1.0,
            1.0,
            fill=False,
            edgecolor="black",
            linewidth=2.0,
        )
    )
    common_colorbar = figure.colorbar(first, ax=axes[:2], shrink=0.82, pad=0.02)
    common_colorbar.set_label(r"$K_\delta(h)$")
    difference_colorbar = figure.colorbar(third, ax=axes[2], shrink=0.82, pad=0.02)
    difference_colorbar.set_label(r"$K_{\delta,1}(h)-K_{\delta,M}(h)$")
    figure.suptitle(
        "Exact-comoving fixed $u=1$ audit after selection: all spatial displacements",
        fontsize=13,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "k_delta_displacement_heatmaps.png", dpi=220)
    figure.savefig(output_dir / "k_delta_displacement_heatmaps.pdf")
    plt.close(figure)


def _report(
    oracle_dir: Path,
    truth: CovarianceParameters,
    matched: CovarianceParameters,
    numerical_jitter_ratio: float,
    table: pd.DataFrame,
    radial: pd.DataFrame,
    maximum_stationarity_span: float,
    maximum_opposite_error: float,
    maximum_directional_spread: float,
) -> str:
    origin = table.loc[(table["h_lat"] == 0) & (table["h_lon"] == 0)].iloc[0]
    nonzero = table.loc[table["radius_squared"] > 0].copy()
    maximum = nonzero.loc[nonzero["absolute_delta_K_delta"].idxmax()]
    shortest = nonzero.loc[nonzero["radius_squared"] == nonzero["radius_squared"].min()]
    radial_lines = []
    for row in radial.itertuples(index=False):
        radial_lines.append(
            f"| {int(row.radius_squared)} | {row.radius:.6g} | {int(row.displacement_count)} | {row.K_delta_1_mean:.12g} | {row.K_delta_M_mean:.12g} | {row.delta_K_delta_mean:+.12g} |"
        )
    shortest_labels = ", ".join(shortest["h"].tolist())
    temporal_lag = 1.0 / truth.range_time
    covariance_factor = 2.0 * truth.variance
    return "\n".join(
        [
            "# Exact-comoving K-delta displacement map",
            "",
            "This is a population-covariance view of the saved exact-comoving experiment. It performs no simulation, contrast search, coefficient optimization, covariance refit, bootstrap, or power analysis.",
            "",
            "## Inputs and definition",
            "",
            f"- Oracle directory: `{oracle_dir}`",
            "- Coordinates: saved `exact_comoving_points.csv`, in standardized moving-coordinate `(latitude, longitude)` units.",
            "- Parameters: saved `experiment_manifest.json` entries `truth` and `matched_margin`.",
            f"- `Sigma_1`: joint Matern-half with variance/ranges/advection/nugget `({truth.variance:.17g}, {truth.range_lat:.17g}, {truth.range_lon:.17g}, {truth.range_time:.17g}, {truth.advec_lat:.17g}, {truth.advec_lon:.17g}, {truth.nugget:.17g})`.",
            f"- `Sigma_M`: advected-separable with matched parameters `({matched.variance:.17g}, {matched.range_lat:.17g}, {matched.range_lon:.17g}, {matched.range_time:.17g}, {matched.advec_lat:.17g}, {matched.advec_lon:.17g}, {matched.nugget:.17g})`.",
            f"- Numerical jitter ratio `{numerical_jitter_ratio:.17g}` is retained to reproduce the saved matrices; statistical nugget is zero.",
            "",
            "For each available signed displacement `h`, every valid base anchor was evaluated independently using the four original covariance entries",
            "",
            "`K_delta,m(h) = C_m((s,0),(s+h,0)) - C_m((s,0),(s+h,1)) - C_m((s,1),(s+h,0)) + C_m((s,1),(s+h,1))`.",
            "",
            "No contrast covariance was inverted to obtain these values.",
            "",
            "## Numerical checks",
            "",
            f"- Signed displacements evaluated: `{len(table)}`.",
            f"- Maximum spread across valid base anchors at a fixed displacement: `{maximum_stationarity_span:.3e}`.",
            f"- Maximum `h` versus `-h` covariance-symmetry error: `{maximum_opposite_error:.3e}`.",
            f"- `delta K_delta(0,0)={origin.delta_K_delta:+.17g}`; this was computed, not set to zero.",
            "",
            "## Result",
            "",
            f"The largest nonzero-lag absolute discrepancy is `{maximum.absolute_delta_K_delta:.12g}` at radius `{maximum.radius:.6g}`. All four shortest signed lags `{shortest_labels}` have the same value `delta K_delta={shortest.iloc[0]['delta_K_delta']:+.12g}`.",
            "",
            "Thus `(1,0)` is not directionally unique in the intrinsic truth-minus-matched numerator. It is one of the four shortest nonzero lags. The discrepancy is radially symmetric on the standardized moving grid to numerical precision and its magnitude decreases with radius over the available grid.",
            "",
            "In particular, the current map does not support an advection-parallel versus advection-perpendicular distinction after transforming to the exact moving coordinates. It supports the more general short-range increment-covariance interpretation.",
            "",
            "For these saved parameters, let `r=sqrt(h_lat^2+h_lon^2)` and `tau=1/range_time`. Apart from the common diagonal jitter at the origin, the two maps reduce to",
            "",
            f"- `K_delta,1(r) = {covariance_factor:.12g} [exp(-r) - exp(-sqrt(r^2 + tau^2))]`,",
            f"- `K_delta,M(r) = {covariance_factor:.12g} [exp(-r) - exp(-(r + tau))]`,",
            f"- with `tau={temporal_lag:.12g}` for the fixed one-step time lag.",
            "",
            "This directly explains the observed radial symmetry. It also shows why the interpretation is specific to the available grid: the discrepancy is zero at `r=0`, largest at the shortest available nonzero radius here, and then decreases across the radii represented by this grid.",
            "",
            "This map examines only the numerator discrepancy `Sigma_1 - Sigma_M`. The original generalized quotient also used the fitted-null `Sigma_0` in its denominator, so a strict orientation preference in that quotient must not be attributed to directional structure in this map.",
            "",
            "| radius squared | radius | signed lag count | K_delta,1 | K_delta,M | delta K_delta |",
            "|---:|---:|---:|---:|---:|---:|",
            *radial_lines,
            "",
            f"Maximum within-radius directional spread of `delta K_delta`: `{maximum_directional_spread:.3e}`.",
            "",
            "## Reproduction",
            "",
            "From the diagnostic directory:",
            "",
            "```bash",
            "python analyze_k_delta_displacement_map.py",
            "```",
            "",
        ]
    )


def main() -> None:
    args = build_parser().parse_args()
    oracle_dir = args.oracle_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest_path = oracle_dir / "experiment_manifest.json"
    points_path = oracle_dir / "exact_comoving_points.csv"
    for path in (manifest_path, points_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not bool(manifest.get("oracle_not_real_data_test", False)):
        raise ValueError("the supplied directory is not marked as the exact-comoving oracle")
    anchor_count = int(manifest["geometry"]["anchor_count"])
    time_count = int(manifest["geometry"]["time_count"])
    points = _ordered_points(pd.read_csv(points_path), anchor_count, time_count)
    forward, reverse = _anchor_maps(points)
    displacements = _available_displacements(forward)
    if len(displacements) != 81:
        raise ValueError(f"expected 81 displacements for the saved 5 by 5 grid, found {len(displacements)}")

    truth = _parameters(manifest["truth"])
    matched = _parameters(manifest["matched_margin"])
    jitter = float(manifest["numerics"]["numerical_jitter_ratio"])
    coordinates = points[["source_latitude", "source_longitude", "time_index"]].to_numpy(
        dtype=np.float64
    )
    geometry = pairwise_lags(coordinates)
    covariances = {
        "1": joint_matern_half_covariance(
            geometry, truth, numerical_jitter_ratio=jitter
        ),
        "M": advected_separable_covariance(
            geometry, matched, numerical_jitter_ratio=jitter
        ),
    }
    table = _displacement_table(
        covariances, displacements, anchor_count, forward, reverse
    )
    radial = _radial_summary(table)

    stationarity_columns = [
        "K_delta_1_stationarity_span",
        "K_delta_M_stationarity_span",
    ]
    opposite_columns = [
        "opposite_lag_error_1",
        "opposite_lag_error_M",
        "opposite_lag_error_delta",
    ]
    maximum_stationarity_span = float(table[stationarity_columns].to_numpy().max())
    maximum_opposite_error = float(table[opposite_columns].to_numpy().max())
    maximum_directional_spread = float(
        radial["delta_K_delta_directional_spread"].max()
    )
    origin_delta = float(
        table.loc[(table["h_lat"] == 0) & (table["h_lon"] == 0), "delta_K_delta"].iloc[0]
    )
    if max(
        maximum_stationarity_span,
        maximum_opposite_error,
        maximum_directional_spread,
        abs(origin_delta),
    ) > 1.0e-12:
        raise ArithmeticError("exact-comoving stationarity, symmetry, or matched-margin check failed")

    nonnegative = table.loc[(table["h_lat"] >= 0) & (table["h_lon"] >= 0)].copy()
    nonnegative = nonnegative.sort_values(
        ["radius_squared", "h_lat", "h_lon"], ignore_index=True
    )
    nonzero = table.loc[table["radius_squared"] > 0]
    largest = nonzero.loc[nonzero["absolute_delta_K_delta"].idxmax()]
    summary = pd.DataFrame(
        [
            {"quantity": "signed_displacement_count", "value": len(table)},
            {"quantity": "delta_K_delta_at_origin", "value": origin_delta},
            {
                "quantity": "largest_nonzero_absolute_delta_K_delta",
                "value": float(largest["absolute_delta_K_delta"]),
            },
            {"quantity": "largest_nonzero_radius", "value": float(largest["radius"])},
            {"quantity": "maximum_stationarity_span", "value": maximum_stationarity_span},
            {"quantity": "maximum_opposite_lag_error", "value": maximum_opposite_error},
            {
                "quantity": "maximum_within_radius_directional_spread",
                "value": maximum_directional_spread,
            },
        ]
    )
    settings = pd.DataFrame(
        [
            {
                "model": model,
                "covariance_family": family,
                **parameters.to_dict(),
                "numerical_jitter_ratio": jitter,
                "source": source,
            }
            for model, family, parameters, source in (
                ("1", "joint_matern_half", truth, "experiment_manifest.json: truth"),
                ("M", "advected_separable", matched, "experiment_manifest.json: matched_margin"),
            )
        ]
    )

    _atomic_csv(output_dir / "k_delta_all_displacements.csv", table)
    _atomic_csv(output_dir / "k_delta_nonnegative_quadrant.csv", nonnegative)
    _atomic_csv(output_dir / "k_delta_radial_summary.csv", radial)
    _atomic_csv(output_dir / "model_settings.csv", settings)
    _atomic_csv(output_dir / "analysis_summary.csv", summary)
    _write_heatmaps(table, output_dir)
    _atomic_text(
        output_dir / "REPORT.md",
        _report(
            oracle_dir=oracle_dir,
            truth=truth,
            matched=matched,
            numerical_jitter_ratio=jitter,
            table=table,
            radial=radial,
            maximum_stationarity_span=maximum_stationarity_span,
            maximum_opposite_error=maximum_opposite_error,
            maximum_directional_spread=maximum_directional_spread,
        ),
    )
    print(f"Wrote K_delta displacement map to {output_dir}")
    print(
        "Largest nonzero |delta K_delta| = "
        f"{float(largest['absolute_delta_K_delta']):.17g} at radius "
        f"{float(largest['radius']):.17g}"
    )


if __name__ == "__main__":
    main()
