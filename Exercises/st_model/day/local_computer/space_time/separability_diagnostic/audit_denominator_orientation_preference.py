#!/usr/bin/env python3
"""Audit whether H0 selects an orientation inside one fixed geometry class.

This script is deliberately not a contrast search.  It reconstructs the saved
exact-comoving population covariances and examines only rectangle pairs with
the already selected spatial signature:

* outer squared length 32;
* inner squared length 20;
* common spatial center;
* cross-endpoint distances 1 and 5; and
* the same one-step temporal interval for both rectangles.

For every rotation/reflection and time translation that fits the saved 5 by 5
dictionary, it compares the raw-basis numerator Delta H, fitted-null metric H0,
and the minimum generalized eigenvalue of (Delta H, H0).  No responses are
read, and no simulation, covariance fit, coefficient optimization, or global
pair search is performed.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.linalg

from diagnostic_core import (
    CovarianceParameters,
    advected_separable_covariance,
    joint_matern_half_covariance,
    pairwise_lags,
)
from rectangle_dictionary_core import rectangle_matrix


HERE = Path(__file__).resolve().parent
DEFAULT_ORACLE_DIR = HERE / "outputs/exact_comoving_rectangle_dictionary_092226"
DEFAULT_OUTPUT_DIR = DEFAULT_ORACLE_DIR / "denominator_orientation_audit"
OUTER_LENGTH_SQUARED = 32
INNER_LENGTH_SQUARED = 20
CROSS_DISTANCE_SQUARED_SIGNATURE = (1, 1, 25, 25)


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


def _atomic_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(contents, encoding="utf-8")
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


def _ordered_points(
    points: pd.DataFrame,
    anchor_count: int,
    time_count: int,
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


def _add_spatial_geometry(
    metadata: pd.DataFrame,
    anchor_rows: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for rectangle in metadata.itertuples(index=False):
        p = int(rectangle.spatial_endpoint_p)
        q = int(rectangle.spatial_endpoint_q)
        p_lat = int(anchor_rows.loc[p, "latitude_grid_index"])
        p_lon = int(anchor_rows.loc[p, "longitude_grid_index"])
        q_lat = int(anchor_rows.loc[q, "latitude_grid_index"])
        q_lon = int(anchor_rows.loc[q, "longitude_grid_index"])
        delta_lat = q_lat - p_lat
        delta_lon = q_lon - p_lon
        rows.append(
            {
                "rectangle_index": int(rectangle.rectangle_index),
                "p_lat": p_lat,
                "p_lon": p_lon,
                "q_lat": q_lat,
                "q_lon": q_lon,
                "delta_lat": delta_lat,
                "delta_lon": delta_lon,
                "length_squared": delta_lat * delta_lat + delta_lon * delta_lon,
                "center_lat_x2": p_lat + q_lat,
                "center_lon_x2": p_lon + q_lon,
            }
        )
    geometry = pd.DataFrame(rows).set_index("rectangle_index")
    enriched = metadata.set_index("rectangle_index", drop=False).join(
        geometry,
        how="left",
        rsuffix="_geometry",
    )
    if enriched[["delta_lat", "delta_lon", "length_squared"]].isna().any().any():
        raise ValueError("failed to attach spatial geometry to the rectangle metadata")
    return enriched


def _squared_distance(
    first: tuple[int, int],
    second: tuple[int, int],
) -> int:
    return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2


def _cross_distance_signature(outer: pd.Series, inner: pd.Series) -> tuple[int, ...]:
    outer_points = (
        (int(outer["p_lat"]), int(outer["p_lon"])),
        (int(outer["q_lat"]), int(outer["q_lon"])),
    )
    inner_points = (
        (int(inner["p_lat"]), int(inner["p_lon"])),
        (int(inner["q_lat"]), int(inner["q_lon"])),
    )
    return tuple(
        sorted(
            _squared_distance(outer_point, inner_point)
            for outer_point in outer_points
            for inner_point in inner_points
        )
    )


def _fixed_signature_pairs(
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    candidates = metadata.loc[
        (metadata["temporal_lag"] == 1)
        & (metadata["time_endpoint_l"] == metadata["time_endpoint_k"] + 1)
    ]
    rows = []
    screened_pair_count = 0
    for (time_k, time_l), interval in candidates.groupby(
        ["time_endpoint_k", "time_endpoint_l"], sort=True
    ):
        outers = interval.loc[interval["length_squared"] == OUTER_LENGTH_SQUARED]
        inners = interval.loc[interval["length_squared"] == INNER_LENGTH_SQUARED]
        screened_pair_count += len(outers) * len(inners)
        for _, outer in outers.iterrows():
            for _, inner in inners.iterrows():
                if (
                    int(outer["center_lat_x2"]) != int(inner["center_lat_x2"])
                    or int(outer["center_lon_x2"]) != int(inner["center_lon_x2"])
                ):
                    continue
                cross_signature = _cross_distance_signature(outer, inner)
                if cross_signature != CROSS_DISTANCE_SQUARED_SIGNATURE:
                    continue
                outer_lat = int(outer["delta_lat"])
                outer_lon = int(outer["delta_lon"])
                inner_lat = int(inner["delta_lat"])
                inner_lon = int(inner["delta_lon"])
                cross_product = outer_lon * inner_lat - outer_lat * inner_lon
                if abs(inner_lat) > abs(inner_lon):
                    axis_assignment = "inner_long_latitude"
                elif abs(inner_lon) > abs(inner_lat):
                    axis_assignment = "inner_long_longitude"
                else:
                    raise AssertionError("the selected inner segment must have a long axis")
                rows.append(
                    {
                        "time_k": int(time_k),
                        "time_l": int(time_l),
                        "outer_rectangle_index": int(outer["rectangle_index"]),
                        "inner_rectangle_index": int(inner["rectangle_index"]),
                        "outer_rectangle_id": outer["rectangle_id"],
                        "inner_rectangle_id": inner["rectangle_id"],
                        "outer_delta_lat": outer_lat,
                        "outer_delta_lon": outer_lon,
                        "inner_delta_lat": inner_lat,
                        "inner_delta_lon": inner_lon,
                        "center_lat_grid": 0.5 * float(outer["center_lat_x2"]),
                        "center_lon_grid": 0.5 * float(outer["center_lon_x2"]),
                        "outer_length_squared": OUTER_LENGTH_SQUARED,
                        "inner_length_squared": INNER_LENGTH_SQUARED,
                        "short_cross_distance": 1.0,
                        "long_cross_distance": 5.0,
                        "cross_distance_squared_multiset": ";".join(
                            str(value) for value in cross_signature
                        ),
                        "axis_assignment": axis_assignment,
                        "relative_handedness": (
                            "counterclockwise" if cross_product > 0 else "clockwise"
                        ),
                        "orientation_key": (
                            f"outer=({outer_lat},{outer_lon});"
                            f"inner=({inner_lat},{inner_lon})"
                        ),
                    }
                )
    frame = pd.DataFrame(rows).sort_values(
        ["time_k", "outer_rectangle_index", "inner_rectangle_index"],
        ignore_index=True,
    )
    return frame, screened_pair_count


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _minimum_generalized_eigenvalue(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> float:
    return float(
        scipy.linalg.eigh(
            _symmetrize(numerator),
            _symmetrize(denominator),
            eigvals_only=True,
            check_finite=False,
        )[0]
    )


def _matrix_elements(prefix: str, matrix: np.ndarray) -> dict[str, float]:
    return {
        f"{prefix}_AA": float(matrix[0, 0]),
        f"{prefix}_AB": float(matrix[0, 1]),
        f"{prefix}_BB": float(matrix[1, 1]),
    }


def _evaluate_pairs(
    pairs: pd.DataFrame,
    raw_dictionary: np.ndarray,
    intrinsic_difference: np.ndarray,
    null_covariance: np.ndarray,
    symmetrized_null_covariance: np.ndarray,
    strict_tie_ids: set[tuple[str, str]],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    calculations = []
    for pair in pairs.itertuples(index=False):
        indices = np.asarray(
            [pair.outer_rectangle_index, pair.inner_rectangle_index],
            dtype=np.int64,
        )
        basis = raw_dictionary[:, indices]
        delta_h = _symmetrize(basis.T @ intrinsic_difference @ basis)
        h0 = _symmetrize(basis.T @ null_covariance @ basis)
        symmetrized_h0 = _symmetrize(basis.T @ symmetrized_null_covariance @ basis)
        unordered_ids = tuple(sorted((pair.outer_rectangle_id, pair.inner_rectangle_id)))
        calculations.append(
            {
                **pair._asdict(),
                "reported_strict_global_tie": unordered_ids in strict_tie_ids,
                **_matrix_elements("deltaH", delta_h),
                **_matrix_elements("H0", h0),
                "lambda_min": _minimum_generalized_eigenvalue(delta_h, h0),
                **_matrix_elements("axis_symmetrized_H0", symmetrized_h0),
                "axis_symmetrized_lambda_min": _minimum_generalized_eigenvalue(
                    delta_h, symmetrized_h0
                ),
                "_delta_h": delta_h,
                "_h0": h0,
            }
        )

    selected = [
        row
        for row in calculations
        if row["reported_strict_global_tie"] and row["time_k"] == 0
    ]
    if not selected:
        raise ValueError("no time-(0,1) representative occurs in the strict tie table")
    reference = min(selected, key=lambda row: row["outer_rectangle_index"])
    reference_delta_h = reference["_delta_h"]
    reference_h0 = reference["_h0"]

    for row in calculations:
        delta_h = row.pop("_delta_h")
        h0 = row.pop("_h0")
        row["deltaH_max_abs_deviation_from_reference"] = float(
            np.max(np.abs(delta_h - reference_delta_h))
        )
        row["H0_max_abs_deviation_from_reference"] = float(
            np.max(np.abs(h0 - reference_h0))
        )
        row["lambda_reference_deltaH_actual_H0"] = _minimum_generalized_eigenvalue(
            reference_delta_h, h0
        )
        row["lambda_actual_deltaH_reference_H0"] = _minimum_generalized_eigenvalue(
            delta_h, reference_h0
        )

    frame = pd.DataFrame(calculations).sort_values(
        ["time_k", "orientation_key"], ignore_index=True
    )
    return frame, reference_delta_h, reference_h0


def _orientation_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for orientation_key, group in pairs.groupby("orientation_key", sort=True):
        first = group.sort_values("time_k").iloc[0]
        rows.append(
            {
                "orientation_key": orientation_key,
                "axis_assignment": first["axis_assignment"],
                "relative_handedness": first["relative_handedness"],
                "time_translation_count": len(group),
                "reported_strict_tie_count": int(
                    group["reported_strict_global_tie"].sum()
                ),
                "deltaH_AA": float(group["deltaH_AA"].mean()),
                "deltaH_AB": float(group["deltaH_AB"].mean()),
                "deltaH_BB": float(group["deltaH_BB"].mean()),
                "H0_AA": float(group["H0_AA"].mean()),
                "H0_AB": float(group["H0_AB"].mean()),
                "H0_BB": float(group["H0_BB"].mean()),
                "lambda_min": float(group["lambda_min"].mean()),
                "lambda_time_translation_span": float(np.ptp(group["lambda_min"])),
                "axis_symmetrized_lambda_min": float(
                    group["axis_symmetrized_lambda_min"].mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("lambda_min", ignore_index=True)


def _axis_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for assignment, group in pairs.groupby("axis_assignment", sort=True):
        rows.append(
            {
                "axis_assignment": assignment,
                "pair_count": len(group),
                "spatial_orientation_count": group["orientation_key"].nunique(),
                "reported_strict_tie_count": int(
                    group["reported_strict_global_tie"].sum()
                ),
                "deltaH_AA": float(group["deltaH_AA"].mean()),
                "deltaH_AB": float(group["deltaH_AB"].mean()),
                "deltaH_BB": float(group["deltaH_BB"].mean()),
                "H0_AA": float(group["H0_AA"].mean()),
                "H0_AB": float(group["H0_AB"].mean()),
                "H0_BB": float(group["H0_BB"].mean()),
                "lambda_min": float(group["lambda_min"].mean()),
                "lambda_within_assignment_span": float(np.ptp(group["lambda_min"])),
                "axis_symmetrized_lambda_min": float(
                    group["axis_symmetrized_lambda_min"].mean()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("lambda_min", ignore_index=True)


def _format_matrix(matrix: np.ndarray) -> str:
    return (
        "[["
        f"{matrix[0, 0]:.15g}, {matrix[0, 1]:.15g}"
        "], ["
        f"{matrix[1, 0]:.15g}, {matrix[1, 1]:.15g}"
        "]]"
    )


def _report(
    oracle_dir: Path,
    screened_pair_count: int,
    pairs: pd.DataFrame,
    orientations: pd.DataFrame,
    axes: pd.DataFrame,
    reference_delta_h: np.ndarray,
    reference_h0: np.ndarray,
    fitted_ratio: float,
    target_ratio: float,
    summary: dict[str, float],
) -> str:
    best = axes.iloc[0]
    alternate = axes.iloc[1]
    h0_difference = np.asarray(
        [
            best.H0_AA - alternate.H0_AA,
            best.H0_AB - alternate.H0_AB,
            best.H0_BB - alternate.H0_BB,
        ]
    )
    orientation_lines = []
    for row in orientations.itertuples(index=False):
        orientation_lines.append(
            "| "
            f"`{row.orientation_key}` | `{row.axis_assignment}` | "
            f"{int(row.time_translation_count)} | "
            f"{int(row.reported_strict_tie_count)} | "
            f"{row.H0_AB:.15g} | {row.H0_BB:.15g} | "
            f"{row.lambda_min:+.15g} |"
        )
    return "\n".join(
        [
            "# Denominator orientation audit",
            "",
            "This audit uses the saved exact-comoving population covariances. It performs no simulation, covariance refit, coefficient optimization, or unrestricted pair search.",
            f"Input oracle: `{oracle_dir}`.",
            "",
            "## Fixed candidate class",
            "",
            "Only pairs with signature `(outer norm squared, inner norm squared, short cross distance, long cross distance) = (32, 20, 1, 5)` were retained. Both rectangles also use the same one-step time interval and have the same spatial center.",
            "",
            f"- Length-filtered pairs screened: `{screened_pair_count}` (not 35,275,800).",
            f"- Exact signature matches: `{len(pairs)}` = `{pairs['orientation_key'].nunique()}` spatial orientations x `7` time translations.",
            f"- All centers are at grid location `({pairs['center_lat_grid'].iloc[0]:.6g}, {pairs['center_lon_grid'].iloc[0]:.6g})`.",
            f"- Actual strict global ties among these pairs: `{int(pairs['reported_strict_global_tie'].sum())}`.",
            "",
            "## Numerator",
            "",
            "The raw-atom numerator matrix is, using outer contrast first and inner contrast second,",
            "",
            f"`Delta H = {_format_matrix(reference_delta_h)}`.",
            "",
            f"The maximum entrywise deviation over all 28 rotation/reflection/time variants is `{summary['maximum_deltaH_deviation']:.3e}`. Thus the intrinsic numerator is the same to floating-point precision.",
            "",
            "## Fitted-null denominator",
            "",
            f"The selected reference has `H0 = {_format_matrix(reference_h0)}`.",
            f"The fitted longitude/latitude range ratio is `{fitted_ratio:.17g}`, whereas the exact standardized-grid target is `{target_ratio:.12g}`; the relative ratio error is `{fitted_ratio / target_ratio - 1.0:+.3e}`.",
            "",
            "| spatial orientation | axis assignment | time copies | strict ties | H0_AB | H0_BB | lambda_min |",
            "|---|---|---:|---:|---:|---:|---:|",
            *orientation_lines,
            "",
            f"Between the two axis assignments, `(H0_AA, H0_AB, H0_BB)` changes by `({h0_difference[0]:+.3e}, {h0_difference[1]:+.3e}, {h0_difference[2]:+.3e})`, while `Delta H` does not change materially.",
            f"The resulting objective gap is `{summary['lambda_gap']:.15g}`. The more negative class is `{best.axis_assignment}` and contains all `{int(best.reported_strict_tie_count)}` strict ties.",
            "",
            "Holding `Delta H` fixed at the reference while allowing only the actual `H0` to vary reproduces every direct generalized eigenvalue within "
            f"`{summary['maximum_H0_only_eigenvalue_error']:.3e}`. Conversely, holding `H0` fixed while allowing only the tiny computed `Delta H` roundoff to vary leaves a spread of only `{summary['fixed_H0_lambda_spread']:.3e}`.",
            "",
            "## Boundary and exact-tie checks",
            "",
            "The finite 5 by 5 boundary permits four spatial variants, all at the same center. Each axis assignment has two mirror variants and seven time translations, so the boundary does not favor one assignment by availability or placement. It limits the class, but it does not create the objective split.",
            "",
            f"Within each spatial orientation, the maximum time-translation spread is `{summary['maximum_time_translation_span']:.3e}`. Mirror variants within an axis assignment are also tied to numerical precision.",
            "",
            "As a denominator-only sensitivity check (not a refit), the fitted spatial-range product was preserved while its longitude/latitude ratio was set exactly to the target ratio. All 28 objectives then span only "
            f"`{summary['axis_symmetrized_lambda_spread']:.3e}`.",
            "",
            "## Conclusion",
            "",
            "The strict reported orientation was selected by `H0`, not by the isotropic numerator and not by unequal boundary availability. More precisely, the split is caused by the "
            f"`{summary['fitted_range_ratio_relative_error']:+.3e}` relative residual in the fitted-null spatial range ratio. It creates a `{summary['lambda_gap']:.3e}` objective gap: large enough to exceed the stored strict tie tolerance, but not evidence of a scientifically meaningful directional interaction. With exact axis symmetry, all 28 variants are a symmetry tie.",
            "",
            "## Reproduction",
            "",
            "From the diagnostic directory:",
            "",
            "```bash",
            "python audit_denominator_orientation_preference.py",
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
    expected_indices = np.arange(len(metadata), dtype=np.int64)
    ordered_metadata = metadata.sort_values("rectangle_index").reset_index(drop=True)
    if not np.array_equal(
        ordered_metadata["rectangle_index"].to_numpy(dtype=np.int64), expected_indices
    ):
        raise ValueError("rectangle indices are not contiguous dictionary columns")

    anchor_rows = points.loc[points["time_index"] == 0].set_index("anchor_index")
    enriched = _add_spatial_geometry(ordered_metadata, anchor_rows)
    pairs, screened_pair_count = _fixed_signature_pairs(enriched)
    if screened_pair_count != 168 or len(pairs) != 28:
        raise AssertionError(
            f"expected 168 screened and 28 matched pairs, found "
            f"{screened_pair_count} and {len(pairs)}"
        )

    endpoint_columns = [
        "spatial_endpoint_p",
        "spatial_endpoint_q",
        "time_endpoint_k",
        "time_endpoint_l",
    ]
    endpoints = ordered_metadata[endpoint_columns].to_numpy(dtype=np.int64)
    raw_dictionary, rebuilt_endpoints = rectangle_matrix(
        anchor_count, time_count, endpoints
    )
    if not np.array_equal(endpoints, rebuilt_endpoints):
        raise AssertionError("rectangle endpoint reconstruction changed the order")

    truth = _parameters(manifest["truth"])
    matched = _parameters(manifest["matched_margin"])
    fitted_null = _parameters(manifest["fitted_null_fixed_advection"])
    jitter = float(manifest["numerics"]["numerical_jitter_ratio"])
    coordinates = points[
        ["source_latitude", "source_longitude", "time_index"]
    ].to_numpy(dtype=np.float64)
    geometry = pairwise_lags(coordinates)
    sigma_1 = joint_matern_half_covariance(
        geometry, truth, numerical_jitter_ratio=jitter
    )
    sigma_m = advected_separable_covariance(
        geometry, matched, numerical_jitter_ratio=jitter
    )
    sigma_0 = advected_separable_covariance(
        geometry, fitted_null, numerical_jitter_ratio=jitter
    )
    intrinsic_difference = sigma_1 - sigma_m

    target_ratio = truth.range_lon / truth.range_lat
    fitted_ratio = fitted_null.range_lon / fitted_null.range_lat
    symmetric_latitude_range = math.sqrt(
        fitted_null.range_lat * fitted_null.range_lon / target_ratio
    )
    symmetric_longitude_range = target_ratio * symmetric_latitude_range
    axis_symmetric_parameters = replace(
        fitted_null,
        range_lat=symmetric_latitude_range,
        range_lon=symmetric_longitude_range,
    )
    axis_symmetric_sigma_0 = advected_separable_covariance(
        geometry,
        axis_symmetric_parameters,
        numerical_jitter_ratio=jitter,
    )

    strict_ties = pd.read_csv(ties_path, float_precision="round_trip")
    strict_tie_ids = {
        tuple(sorted((row.first_rectangle_id, row.second_rectangle_id)))
        for row in strict_ties.itertuples(index=False)
    }
    pair_results, reference_delta_h, reference_h0 = _evaluate_pairs(
        pairs,
        raw_dictionary,
        intrinsic_difference,
        sigma_0,
        axis_symmetric_sigma_0,
        strict_tie_ids,
    )
    orientations = _orientation_summary(pair_results)
    axes = _axis_summary(pair_results)

    maximum_delta_h_deviation = float(
        pair_results["deltaH_max_abs_deviation_from_reference"].max()
    )
    direct = pair_results["lambda_min"].to_numpy(dtype=np.float64)
    h0_only = pair_results["lambda_reference_deltaH_actual_H0"].to_numpy(
        dtype=np.float64
    )
    fixed_h0 = pair_results["lambda_actual_deltaH_reference_H0"].to_numpy(
        dtype=np.float64
    )
    summary = {
        "screened_pair_count": screened_pair_count,
        "matched_pair_count": len(pair_results),
        "spatial_orientation_count": pair_results["orientation_key"].nunique(),
        "strict_tie_count": int(pair_results["reported_strict_global_tie"].sum()),
        "maximum_deltaH_deviation": maximum_delta_h_deviation,
        "maximum_H0_deviation": float(
            pair_results["H0_max_abs_deviation_from_reference"].max()
        ),
        "lambda_minimum": float(direct.min()),
        "lambda_maximum": float(direct.max()),
        "lambda_gap": float(direct.max() - direct.min()),
        "maximum_H0_only_eigenvalue_error": float(np.max(np.abs(direct - h0_only))),
        "fixed_H0_lambda_spread": float(np.ptp(fixed_h0)),
        "maximum_time_translation_span": float(
            orientations["lambda_time_translation_span"].max()
        ),
        "axis_symmetrized_lambda_spread": float(
            np.ptp(pair_results["axis_symmetrized_lambda_min"])
        ),
        "truth_range_lon_over_lat": target_ratio,
        "fitted_range_lon_over_lat": fitted_ratio,
        "fitted_range_ratio_relative_error": fitted_ratio / target_ratio - 1.0,
        "axis_symmetrized_range_lat": symmetric_latitude_range,
        "axis_symmetrized_range_lon": symmetric_longitude_range,
    }

    if maximum_delta_h_deviation > 1.0e-12:
        raise ArithmeticError("rotation/reflection variants changed raw Delta H")
    if summary["maximum_H0_only_eigenvalue_error"] > 1.0e-12:
        raise ArithmeticError("H0-only reconstruction did not reproduce direct objectives")
    if summary["fixed_H0_lambda_spread"] > 1.0e-12:
        raise ArithmeticError("fixed-H0 objectives retained a material numerator spread")
    if summary["axis_symmetrized_lambda_spread"] > 1.0e-12:
        raise ArithmeticError("axis-symmetrized denominator did not restore the tie")
    if summary["strict_tie_count"] != len(strict_ties):
        raise ArithmeticError("fixed-signature membership disagrees with strict tie table")

    summary_frame = pd.DataFrame(
        [{"quantity": key, "value": value} for key, value in summary.items()]
    )
    settings = pd.DataFrame(
        [
            {
                "model": name,
                **parameters.to_dict(),
                "numerical_jitter_ratio": jitter,
                "source": source,
            }
            for name, parameters, source in (
                ("Sigma_1", truth, "experiment_manifest.json: truth"),
                ("Sigma_M", matched, "experiment_manifest.json: matched_margin"),
                (
                    "Sigma_0",
                    fitted_null,
                    "experiment_manifest.json: fitted_null_fixed_advection",
                ),
                (
                    "Sigma_0_axis_symmetrized_sensitivity",
                    axis_symmetric_parameters,
                    "deterministic sensitivity; fitted range product preserved",
                ),
            )
        ]
    )

    _atomic_csv(output_dir / "orientation_pair_audit.csv", pair_results)
    _atomic_csv(output_dir / "spatial_orientation_summary.csv", orientations)
    _atomic_csv(output_dir / "axis_assignment_summary.csv", axes)
    _atomic_csv(output_dir / "audit_summary.csv", summary_frame)
    _atomic_csv(output_dir / "model_settings.csv", settings)
    _atomic_text(
        output_dir / "REPORT.md",
        _report(
            oracle_dir,
            screened_pair_count,
            pair_results,
            orientations,
            axes,
            reference_delta_h,
            reference_h0,
            fitted_ratio,
            target_ratio,
            summary,
        ),
    )
    print(f"Wrote denominator orientation audit to {output_dir}")
    print(
        "lambda gap = "
        f"{summary['lambda_gap']:.17g}; "
        "axis-symmetrized spread = "
        f"{summary['axis_symmetrized_lambda_spread']:.17g}"
    )


if __name__ == "__main__":
    main()
