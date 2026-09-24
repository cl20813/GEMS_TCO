#!/usr/bin/env python3
"""Find and classify the globally minimizing pair of rectangle contrasts.

This post-processing script reconstructs the covariance-only exact-comoving
oracle from its immutable manifest and saved point table.  It exhaustively
searches all unordered rectangle pairs in float64 blocks, validates every
minimum-tolerance candidate with :func:`scipy.linalg.eigh`, and groups tied
pairs by translation-invariant relative geometry.

No response values are read.  The result is an oracle population statement,
not a calibrated test, p-value, or claim about the warped GEMS geometry.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
import scipy
import scipy.linalg

from analyze_mode_count_path import atomic_json, covariance_parameters, sha256
from diagnostic_core import (
    advected_separable_covariance,
    joint_matern_half_covariance,
    pairwise_lags,
)
from global_two_rectangle_search import (
    exhaustive_two_rectangle_search,
    validate_pair_candidate,
)
from rectangle_dictionary_core import (
    filter_variance_metrics,
    rectangle_matrix,
    standardize_dictionary,
)
from run_rectangle_dictionary_oracle import git_state, runtime_environment


HERE = Path(__file__).resolve().parent
DEFAULT_ORACLE_DIR = HERE / "outputs/exact_comoving_rectangle_dictionary_092226"


@dataclass(frozen=True)
class ReconstructedProblem:
    raw_dictionary: np.ndarray
    standardized_dictionary: np.ndarray
    null_standard_deviations: np.ndarray
    true_covariance: np.ndarray
    matched_covariance: np.ndarray
    null_covariance: np.ndarray
    intrinsic_difference: np.ndarray
    endpoints: np.ndarray
    lag_geometry: Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle-dir", type=Path, default=DEFAULT_ORACLE_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument(
        "--verification-block-size",
        type=int,
        default=511,
        help="independent block partition used for the numerical audit",
    )
    parser.add_argument("--tie-tolerance", type=float)
    parser.add_argument(
        "--geometry-sensitivity-tolerance",
        type=float,
        default=1.0e-8,
        help="wider screen used to expose symmetry-near minimizers",
    )
    parser.add_argument("--singular-threshold", type=float, default=1.0e-10)
    parser.add_argument("--maximum-saved-near-singular", type=int, default=10_000)
    return parser


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    """Write audit tables atomically without truncating float64 values."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(relative_to) if relative_to is not None else path),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def _output_inventory(output_dir: Path) -> list[dict[str, Any]]:
    """Hash completed outputs while excluding the self-referential manifest."""

    return [
        _file_record(path, relative_to=output_dir)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "search_manifest.json"
    ]


def _read_oracle(oracle_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    oracle_dir = oracle_dir.expanduser().resolve()
    manifest_path = oracle_dir / "experiment_manifest.json"
    point_path = oracle_dir / "exact_comoving_points.csv"
    metadata_path = oracle_dir / "rectangle_dictionary_metadata.csv"
    missing = [path for path in (manifest_path, point_path, metadata_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"oracle inputs are missing: {missing}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not bool(manifest.get("oracle_not_real_data_test", False)):
        raise ValueError("input manifest is not marked as the covariance-only oracle")
    points = pd.read_csv(point_path)
    metadata = pd.read_csv(metadata_path)
    return manifest, points, metadata


def _reconstruct_problem(
    manifest: dict[str, Any],
    points: pd.DataFrame,
    metadata: pd.DataFrame,
) -> ReconstructedProblem:
    required_points = {
        "time_index",
        "anchor_index",
        "source_latitude",
        "source_longitude",
    }
    if not required_points.issubset(points.columns):
        raise ValueError(f"point table is missing {sorted(required_points - set(points.columns))}")
    ordered = points.sort_values(["time_index", "anchor_index"]).reset_index(drop=True)
    coordinates = ordered[["source_latitude", "source_longitude", "time_index"]].to_numpy(
        dtype=np.float64
    )
    truth = covariance_parameters(manifest["truth"])
    fitted_null = covariance_parameters(manifest["fitted_null_fixed_advection"])
    jitter = float(manifest["numerics"]["numerical_jitter_ratio"])
    geometry = pairwise_lags(coordinates)
    true_covariance = joint_matern_half_covariance(
        geometry,
        truth,
        numerical_jitter_ratio=jitter,
    )
    matched_covariance = advected_separable_covariance(
        geometry,
        truth,
        numerical_jitter_ratio=jitter,
    )
    null_covariance = advected_separable_covariance(
        geometry,
        fitted_null,
        numerical_jitter_ratio=jitter,
    )
    intrinsic_difference = true_covariance - matched_covariance
    endpoint_columns = [
        "spatial_endpoint_p",
        "spatial_endpoint_q",
        "time_endpoint_k",
        "time_endpoint_l",
    ]
    endpoints = metadata.sort_values("rectangle_index")[endpoint_columns].to_numpy(dtype=np.int64)
    anchor_count = int(manifest["geometry"]["anchor_count"])
    time_count = int(manifest["geometry"]["time_count"])
    raw_dictionary, rebuilt_endpoints = rectangle_matrix(
        anchor_count,
        time_count,
        endpoints,
    )
    if not np.array_equal(endpoints, rebuilt_endpoints):
        raise AssertionError("saved and reconstructed rectangle endpoints differ")
    standardized, scales, _ = standardize_dictionary(raw_dictionary, null_covariance)
    if standardized.shape[1] != int(manifest["dictionary"]["rectangle_count"]):
        raise AssertionError("reconstructed dictionary width differs from the manifest")
    return ReconstructedProblem(
        raw_dictionary=raw_dictionary,
        standardized_dictionary=standardized,
        null_standard_deviations=scales,
        true_covariance=true_covariance,
        matched_covariance=matched_covariance,
        null_covariance=null_covariance,
        intrinsic_difference=intrinsic_difference,
        endpoints=endpoints,
        lag_geometry=geometry,
    )


def _interval_overlap(first: tuple[float, float], second: tuple[float, float]) -> float:
    return float(max(0.0, min(first[1], second[1]) - max(first[0], second[0])))


def _time_arrangement(k1: int, l1: int, k2: int, l2: int) -> str:
    if (k1, l1) == (k2, l2):
        return "identical"
    if l1 < k2:
        return "first_before_second_disjoint"
    if l2 < k1:
        return "second_before_first_disjoint"
    if l1 == k2 or l2 == k1:
        return "touching_at_endpoint"
    if k1 <= k2 and l2 <= l1:
        return "second_nested_in_first"
    if k2 <= k1 and l1 <= l2:
        return "first_nested_in_second"
    return "partially_overlapping"


def _shape_and_center(
    rectangle: pd.Series,
    anchor_rows: pd.DataFrame,
) -> dict[str, Any]:
    p = int(rectangle["spatial_endpoint_p"])
    q = int(rectangle["spatial_endpoint_q"])
    k = int(rectangle["time_endpoint_k"])
    ell = int(rectangle["time_endpoint_l"])
    p_row = anchor_rows.loc[p]
    q_row = anchor_rows.loc[q]
    delta_lat_grid = int(q_row["latitude_grid_index"] - p_row["latitude_grid_index"])
    delta_lon_grid = int(q_row["longitude_grid_index"] - p_row["longitude_grid_index"])
    norm_squared = delta_lat_grid**2 + delta_lon_grid**2
    temporal_lag = ell - k
    return {
        "p": p,
        "q": q,
        "k": k,
        "ell": ell,
        "delta_lat_grid": delta_lat_grid,
        "delta_lon_grid": delta_lon_grid,
        "spatial_norm_squared_grid": norm_squared,
        "spatial_length_grid": math.sqrt(norm_squared),
        "temporal_lag": temporal_lag,
        "shape_key": f"space_norm2={norm_squared};time_lag={temporal_lag}",
        "center_lat_grid": 0.5
        * (float(p_row["latitude_grid_index"]) + float(q_row["latitude_grid_index"])),
        "center_lon_grid": 0.5
        * (float(p_row["longitude_grid_index"]) + float(q_row["longitude_grid_index"])),
        "center_time": 0.5 * (k + ell),
        "lat_bounds": tuple(
            sorted(
                (
                    float(p_row["standardized_moving_latitude"]),
                    float(q_row["standardized_moving_latitude"]),
                )
            )
        ),
        "lon_bounds": tuple(
            sorted(
                (
                    float(p_row["standardized_moving_longitude"]),
                    float(q_row["standardized_moving_longitude"]),
                )
            )
        ),
    }


def _canonical_equivalence_key(
    first: dict[str, Any],
    second: dict[str, Any],
    *,
    delta_lon_x2: int,
    delta_lat_x2: int,
    delta_time_x2: int,
    orientation_dot: int,
    orientation_cross_magnitude: int,
) -> tuple[Any, ...]:
    """Canonicalize full relative geometry under square-grid symmetries.

    The key removes absolute space/time translation, rectangle-pair order,
    endpoint orientation, and the eight rotations/reflections of a square.
    Unlike a key based only on lengths and ``abs(cross)``, it retains the
    orientation of a nonzero center displacement relative to both segments.
    """

    first_vector = (
        int(first["delta_lon_grid"]),
        int(first["delta_lat_grid"]),
    )
    second_vector = (
        int(second["delta_lon_grid"]),
        int(second["delta_lat_grid"]),
    )
    center_vector = (int(delta_lon_x2), int(delta_lat_x2))
    # (a, b, c, d) represents (x, y) -> (a*x+b*y, c*x+d*y).
    dihedral_matrices = (
        (1, 0, 0, 1),
        (0, -1, 1, 0),
        (-1, 0, 0, -1),
        (0, 1, -1, 0),
        (-1, 0, 0, 1),
        (1, 0, 0, -1),
        (0, 1, 1, 0),
        (0, -1, -1, 0),
    )

    def transform(vector: tuple[int, int], matrix: tuple[int, int, int, int]):
        x, y = vector
        a, b, c, d = matrix
        return (a * x + b * y, c * x + d * y)

    candidates: list[tuple[Any, ...]] = []
    for matrix in dihedral_matrices:
        vector_one = transform(first_vector, matrix)
        vector_two = transform(second_vector, matrix)
        center = transform(center_vector, matrix)
        for sign_one in (-1, 1):
            for sign_two in (-1, 1):
                shape_one = (
                    sign_one * vector_one[0],
                    sign_one * vector_one[1],
                    int(first["temporal_lag"]),
                )
                shape_two = (
                    sign_two * vector_two[0],
                    sign_two * vector_two[1],
                    int(second["temporal_lag"]),
                )
                direct = (
                    shape_one,
                    shape_two,
                    center[0],
                    center[1],
                    int(delta_time_x2),
                    abs(int(orientation_dot)),
                    int(orientation_cross_magnitude),
                )
                swapped = (
                    shape_two,
                    shape_one,
                    -center[0],
                    -center[1],
                    -int(delta_time_x2),
                    abs(int(orientation_dot)),
                    int(orientation_cross_magnitude),
                )
                candidates.extend((direct, swapped))
    return min(candidates)


def _geometry_fields(
    first_row: pd.Series,
    second_row: pd.Series,
    anchor_rows: pd.DataFrame,
) -> dict[str, Any]:
    first = _shape_and_center(first_row, anchor_rows)
    second = _shape_and_center(second_row, anchor_rows)
    delta_lon = float(second["center_lon_grid"] - first["center_lon_grid"])
    delta_lat = float(second["center_lat_grid"] - first["center_lat_grid"])
    delta_time = float(second["center_time"] - first["center_time"])
    delta_lon_x2 = int(round(2.0 * delta_lon))
    delta_lat_x2 = int(round(2.0 * delta_lat))
    delta_time_x2 = int(round(2.0 * delta_time))
    dot = int(
        first["delta_lat_grid"] * second["delta_lat_grid"]
        + first["delta_lon_grid"] * second["delta_lon_grid"]
    )
    cross = int(
        first["delta_lon_grid"] * second["delta_lat_grid"]
        - first["delta_lat_grid"] * second["delta_lon_grid"]
    )
    signed_angle = float(math.degrees(math.atan2(cross, dot)))
    denominator = math.sqrt(
        float(first["spatial_norm_squared_grid"]) * float(second["spatial_norm_squared_grid"])
    )
    cosine = float(np.clip(dot / denominator, -1.0, 1.0))
    unsigned_angle = float(math.degrees(math.acos(cosine)))
    if cross == 0 and dot > 0:
        orientation = "parallel_same"
    elif cross == 0 and dot < 0:
        orientation = "parallel_opposite"
    elif dot == 0:
        orientation = "perpendicular"
    else:
        orientation = f"angle_{unsigned_angle:.6g}_degrees"
    handedness = "counterclockwise" if cross > 0 else "clockwise" if cross < 0 else "collinear"

    first_anchors = {first["p"], first["q"]}
    second_anchors = {second["p"], second["q"]}
    first_times = {first["k"], first["ell"]}
    second_times = {second["k"], second["ell"]}
    first_support = {(a, t) for a in first_anchors for t in first_times}
    second_support = {(a, t) for a in second_anchors for t in second_times}
    support_intersection = len(first_support & second_support)
    support_union = len(first_support | second_support)
    latitude_overlap = _interval_overlap(first["lat_bounds"], second["lat_bounds"])
    longitude_overlap = _interval_overlap(first["lon_bounds"], second["lon_bounds"])
    temporal_overlap = _interval_overlap(
        (first["k"], first["ell"]),
        (second["k"], second["ell"]),
    )
    class_key = _canonical_equivalence_key(
        first,
        second,
        delta_lon_x2=delta_lon_x2,
        delta_lat_x2=delta_lat_x2,
        delta_time_x2=delta_time_x2,
        orientation_dot=dot,
        orientation_cross_magnitude=abs(cross),
    )
    return {
        "shape_first": first["shape_key"],
        "shape_second": second["shape_key"],
        "first_spatial_delta_lat_grid": first["delta_lat_grid"],
        "first_spatial_delta_lon_grid": first["delta_lon_grid"],
        "second_spatial_delta_lat_grid": second["delta_lat_grid"],
        "second_spatial_delta_lon_grid": second["delta_lon_grid"],
        "first_spatial_length_grid": first["spatial_length_grid"],
        "second_spatial_length_grid": second["spatial_length_grid"],
        "first_temporal_lag": first["temporal_lag"],
        "second_temporal_lag": second["temporal_lag"],
        "first_center_lon_grid": first["center_lon_grid"],
        "first_center_lat_grid": first["center_lat_grid"],
        "first_center_time": first["center_time"],
        "second_center_lon_grid": second["center_lon_grid"],
        "second_center_lat_grid": second["center_lat_grid"],
        "second_center_time": second["center_time"],
        "center_displacement_lon_grid": delta_lon,
        "center_displacement_lat_grid": delta_lat,
        "center_displacement_time": delta_time,
        "time_arrangement": _time_arrangement(first["k"], first["ell"], second["k"], second["ell"]),
        "relative_orientation": orientation,
        "relative_orientation_handedness": handedness,
        "relative_orientation_dot_grid": dot,
        "relative_orientation_cross_grid": cross,
        "relative_orientation_unsigned_angle_degrees": unsigned_angle,
        "relative_orientation_signed_angle_degrees": signed_angle,
        "shared_anchor_count": len(first_anchors & second_anchors),
        "shared_time_endpoint_count": len(first_times & second_times),
        "shared_support_point_count": support_intersection,
        "support_jaccard": support_intersection / support_union,
        "standardized_latitude_bbox_overlap": latitude_overlap,
        "standardized_longitude_bbox_overlap": longitude_overlap,
        "spatial_bbox_overlap_area": latitude_overlap * longitude_overlap,
        "temporal_interval_overlap": temporal_overlap,
        "spacetime_bbox_overlap_volume": (latitude_overlap * longitude_overlap * temporal_overlap),
        "equivalence_key": json.dumps(class_key, separators=(",", ":")),
    }


def _equivalence_classes(ties: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for class_number, (key, group) in enumerate(
        ties.groupby("equivalence_key", sort=True), start=1
    ):
        representative = group.sort_values(
            ["first_rectangle_index", "second_rectangle_index"]
        ).iloc[0]
        rows.append(
            {
                "equivalence_class": f"class_{class_number:04d}",
                "equivalence_key": key,
                "pair_count": len(group),
                "minimum_scipy_eigenvalue": float(group["scipy_eigenvalue"].min()),
                "maximum_scipy_eigenvalue": float(group["scipy_eigenvalue"].max()),
                "representative_first_rectangle_id": representative["first_rectangle_id"],
                "representative_second_rectangle_id": representative["second_rectangle_id"],
                "shape_first": representative["shape_first"],
                "shape_second": representative["shape_second"],
                "center_displacement_lon_grid": representative["center_displacement_lon_grid"],
                "center_displacement_lat_grid": representative["center_displacement_lat_grid"],
                "center_displacement_time": representative["center_displacement_time"],
                "relative_orientation": representative["relative_orientation"],
                "relative_orientation_handedness_variants": ";".join(
                    sorted(set(group["relative_orientation_handedness"]))
                ),
                "relative_coefficient_signs_canonical": ";".join(
                    sorted(set(group["relative_coefficient_sign_canonical"]))
                ),
            }
        )
    classes = pd.DataFrame(rows)
    if len(classes):
        classes = classes.sort_values(
            ["pair_count", "equivalence_key"], ascending=[False, True]
        ).reset_index(drop=True)
        class_map = {
            key: f"class_{index + 1:04d}" for index, key in enumerate(classes["equivalence_key"])
        }
        classes["equivalence_class"] = classes["equivalence_key"].map(class_map)
    return classes


def _describe_candidates(
    validated: pd.DataFrame,
    metadata: pd.DataFrame,
    anchor_rows: pd.DataFrame,
    problem: ReconstructedProblem,
) -> pd.DataFrame:
    """Attach invariant variance metrics and exact relative geometry."""

    rows = []
    for row in validated.itertuples(index=False):
        first_index = int(row.first_index)
        second_index = int(row.second_index)
        first_meta = metadata.loc[first_index]
        second_meta = metadata.loc[second_index]
        coefficients = np.asarray([row.coefficient_first, row.coefficient_second], dtype=np.float64)
        indices = np.asarray([first_index, second_index], dtype=np.int64)
        weights = problem.standardized_dictionary[:, indices] @ coefficients
        metrics = filter_variance_metrics(
            weights,
            problem.true_covariance,
            problem.matched_covariance,
            problem.null_covariance,
        )
        raw_coefficients = coefficients / problem.null_standard_deviations[indices]
        values = {
            "first_rectangle_index": first_index,
            "second_rectangle_index": second_index,
            "first_rectangle_id": first_meta["rectangle_id"],
            "second_rectangle_id": second_meta["rectangle_id"],
            "first_spatial_endpoint_p": int(first_meta["spatial_endpoint_p"]),
            "first_spatial_endpoint_q": int(first_meta["spatial_endpoint_q"]),
            "first_time_endpoint_k": int(first_meta["time_endpoint_k"]),
            "first_time_endpoint_l": int(first_meta["time_endpoint_l"]),
            "second_spatial_endpoint_p": int(second_meta["spatial_endpoint_p"]),
            "second_spatial_endpoint_q": int(second_meta["spatial_endpoint_q"]),
            "second_time_endpoint_k": int(second_meta["time_endpoint_k"]),
            "second_time_endpoint_l": int(second_meta["time_endpoint_l"]),
            **row._asdict(),
            "raw_coefficient_first": float(raw_coefficients[0]),
            "raw_coefficient_second": float(raw_coefficients[1]),
            "raw_coefficient_ratio_second_over_first": float(
                raw_coefficients[1] / raw_coefficients[0]
            ),
            **metrics,
        }
        values.update(_geometry_fields(first_meta, second_meta, anchor_rows))
        rows.append(values)
    return pd.DataFrame(rows).sort_values(
        ["first_rectangle_index", "second_rectangle_index"], ignore_index=True
    )


def _assign_equivalence_classes(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    classes = _equivalence_classes(rows)
    class_map = dict(zip(classes["equivalence_key"], classes["equivalence_class"]))
    labeled = rows.copy()
    labeled.insert(0, "equivalence_class", labeled["equivalence_key"].map(class_map))
    return labeled, classes


def _axis_symmetry_sensitivity(
    candidates: pd.DataFrame,
    problem: ReconstructedProblem,
    manifest: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Re-evaluate screened pairs after removing tiny fitted-axis asymmetry.

    The fitted spatial-range product is preserved while its longitude/latitude
    ratio is set to the exact ratio used by the truth model.  This is a local
    sensitivity audit of the screened motif variants, not another global fit.
    """

    truth = covariance_parameters(manifest["truth"])
    fitted = covariance_parameters(manifest["fitted_null_fixed_advection"])
    target_ratio = float(truth.range_lon / truth.range_lat)
    fitted_ratio = float(fitted.range_lon / fitted.range_lat)
    symmetric_latitude_range = math.sqrt(fitted.range_lat * fitted.range_lon / target_ratio)
    symmetric_longitude_range = target_ratio * symmetric_latitude_range
    symmetric_parameters = replace(
        fitted,
        range_lat=symmetric_latitude_range,
        range_lon=symmetric_longitude_range,
    )
    symmetric_null = advected_separable_covariance(
        problem.lag_geometry,
        symmetric_parameters,
        numerical_jitter_ratio=float(manifest["numerics"]["numerical_jitter_ratio"]),
    )
    symmetric_dictionary, _, _ = standardize_dictionary(
        problem.raw_dictionary,
        symmetric_null,
    )
    rows = []
    for row in candidates.itertuples(index=False):
        indices = np.asarray(
            [row.first_rectangle_index, row.second_rectangle_index], dtype=np.int64
        )
        basis = symmetric_dictionary[:, indices]
        difference = basis.T @ problem.intrinsic_difference @ basis
        gram = basis.T @ symmetric_null @ basis
        eigenvalue = float(
            scipy.linalg.eigh(
                0.5 * (difference + difference.T),
                0.5 * (gram + gram.T),
                eigvals_only=True,
                check_finite=False,
            )[0]
        )
        rows.append(
            {
                "first_rectangle_index": int(row.first_rectangle_index),
                "second_rectangle_index": int(row.second_rectangle_index),
                "first_rectangle_id": row.first_rectangle_id,
                "second_rectangle_id": row.second_rectangle_id,
                "original_scipy_eigenvalue": float(row.scipy_eigenvalue),
                "axis_symmetrized_scipy_eigenvalue": eigenvalue,
            }
        )
    frame = pd.DataFrame(rows)
    summary = {
        "truth_range_lon_over_lat": target_ratio,
        "fitted_range_lon_over_lat": fitted_ratio,
        "relative_ratio_error": fitted_ratio / target_ratio - 1.0,
        "symmetrized_range_lat": symmetric_latitude_range,
        "symmetrized_range_lon": symmetric_longitude_range,
        "symmetrized_candidate_minimum": float(frame["axis_symmetrized_scipy_eigenvalue"].min()),
        "symmetrized_candidate_maximum": float(frame["axis_symmetrized_scipy_eigenvalue"].max()),
        "symmetrized_candidate_spread": float(
            frame["axis_symmetrized_scipy_eigenvalue"].max()
            - frame["axis_symmetrized_scipy_eigenvalue"].min()
        ),
    }
    return frame, summary


def _write_report(
    path: Path,
    ties: pd.DataFrame,
    classes: pd.DataFrame,
    sensitivity_pairs: pd.DataFrame,
    sensitivity_classes: pd.DataFrame,
    search: Any,
    verification_search: Any,
    greedy_pair: tuple[int, int],
    greedy_mu: float,
    *,
    tie_tolerance: float,
    sensitivity_tolerance: float,
    next_gap: float,
    symmetry_summary: dict[str, float],
) -> None:
    global_mu = float(ties["scipy_eigenvalue"].min())
    greedy_is_global = any(
        (int(row.first_rectangle_index), int(row.second_rectangle_index)) == greedy_pair
        for row in ties.itertuples(index=False)
    )
    largest_class = int(classes["pair_count"].max())
    all_one_class = len(classes) == 1
    sign_patterns = sorted(set(ties["relative_coefficient_sign_canonical"]))
    temporal_shifts = int(ties["first_center_time"].nunique())
    handedness_variants = int(ties["relative_orientation_handedness"].nunique())
    representative = ties.sort_values(["first_rectangle_index", "second_rectangle_index"]).iloc[0]
    diagonal_first = representative["coefficient_first"] ** 2 * representative["d_ii"]
    cross = (
        2.0
        * representative["coefficient_first"]
        * representative["coefficient_second"]
        * representative["d_ij"]
    )
    diagonal_second = representative["coefficient_second"] ** 2 * representative["d_jj"]
    diagonal_sum = diagonal_first + diagonal_second
    same_interval = bool(
        np.all(ties["time_arrangement"] == "identical")
        and np.all(ties["first_temporal_lag"] == 1)
        and np.all(ties["second_temporal_lag"] == 1)
    )
    co_centered = bool(
        np.all(ties["center_displacement_lon_grid"] == 0.0)
        and np.all(ties["center_displacement_lat_grid"] == 0.0)
        and np.all(ties["center_displacement_time"] == 0.0)
    )
    p1 = int(representative["first_spatial_endpoint_p"])
    q1 = int(representative["first_spatial_endpoint_q"])
    p2 = int(representative["second_spatial_endpoint_p"])
    q2 = int(representative["second_spatial_endpoint_q"])
    k = int(representative["first_time_endpoint_k"])
    ell = int(representative["first_time_endpoint_l"])
    raw_one = float(representative["raw_coefficient_first"])
    raw_two = float(representative["raw_coefficient_second"])
    lines = [
        "# Global two-rectangle exhaustive search",
        "",
        "This is a covariance-only exact-comoving oracle calculation. It reads no responses and is not a calibrated test.",
        "",
        "## Objective boundary",
        "",
        "The exhaustive target is `mu(w) = w' (Sigma1-SigmaM) w / (w' Sigma0 w)`, where SigmaM is the matched-margin advected-separable covariance and Sigma0 is the fitted null.",
        "It therefore isolates the intrinsic interaction discrepancy. It is **not** an exhaustive minimization of `Sigma1-Sigma0`, `rho0`, or `g(rho0)`.",
        "",
        "## Result",
        "",
        f"- Exhaustively evaluated `{search.pair_count:,}` unordered pairs of `{int((1 + math.sqrt(1 + 8 * search.pair_count)) / 2):,}` rectangles.",
        f"- Global minimum: `mu = {global_mu:.12g}`.",
        f"- Strict ties within the declared tolerance `{tie_tolerance:.3g}`: `{len(ties)}`.",
        f"- Strict-tie D4-canonical relative-geometry classes: `{len(classes)}`.",
        f"- Largest equivalence class: `{largest_class}` pairs.",
        f"- Temporal placements represented among the ties: `{temporal_shifts}`; mirror-handed variants: `{handedness_variants}`.",
        f"- Greedy pair `{greedy_pair[0]};{greedy_pair[1]}` has `mu = {greedy_mu:.12g}` and is {'a global-tolerance minimizer' if greedy_is_global else 'not globally minimizing'}.",
        f"- The next symmetry-near level is `{next_gap:.6g}` above the minimum.",
        f"- At sensitivity tolerance `{sensitivity_tolerance:.3g}`, `{len(sensitivity_pairs)}` pairs occupy `{len(sensitivity_classes)}` D4-canonical class(es).",
        "",
        "## Interpretation",
        "",
    ]
    if all_one_class:
        lines.append("Every strict tie belongs to one relative-geometry orbit.")
        lines.append("")
        if sign_patterns == ["opposite"]:
            lines.append(
                "Under the stored `p<q, k<l` endpoint orientation, the two rectangle coefficients have opposite signs."
            )
        else:
            lines.extend(
                [
                    "Under the stored `p<q, k<l` endpoint orientation, the two rectangle coefficients have the same relative sign. This sign label is convention-dependent; reversing either atom reverses its coefficient without changing the final filter.",
                    "",
                ]
            )
        if co_centered and same_interval:
            lines.extend(
                [
                    "> Under the canonical endpoint orientation, the intrinsic optimum is a same-signed, one-hour temporal difference of two co-centered, unequal-length, non-collinear spatial contrasts.",
                    "",
                    "`Co-centered` is deliberate: the two spatial segments share a center but are neither parallel nor geometrically concentric.",
                ]
            )
    else:
        lines.extend(
            [
                "The global ties occupy more than one relative-geometry class, so the mechanism is not uniquely described by a single translation class.",
                "The class table should be used before making the stronger single-motif statement.",
            ]
        )
    lines.extend(
        [
            "",
            "## Representative filter",
            "",
            f"The representative strict tie is `{representative['first_rectangle_id']}` with `{representative['second_rectangle_id']}`.",
            f"Its standardized coefficient ratio is `{representative['eigenvector_ratio_second_over_first']:.12g}` and its fitted-null rectangle correlation is `g_ij={representative['g_ij']:.12g}`.",
            f"The spatial squared lengths are `{representative['first_spatial_length_grid'] ** 2:.12g}` and `{representative['second_spatial_length_grid'] ** 2:.12g}`, their center displacement is `({representative['center_displacement_lon_grid']:.6g},{representative['center_displacement_lat_grid']:.6g},{representative['center_displacement_time']:.6g})`, and their unsigned angle is `{representative['relative_orientation_unsigned_angle_degrees']:.10g}` degrees. They share `{int(representative['shared_time_endpoint_count'])}` time endpoints, `{int(representative['shared_anchor_count'])}` spatial anchors, and `{int(representative['shared_support_point_count'])}` observation-support points.",
            f"In raw rectangle units, define `S_t = ({raw_one:.12g})(Y[{p1},t]-Y[{q1},t]) + ({raw_two:.12g})(Y[{p2},t]-Y[{q2},t])`. The filter is exactly `L=S_{k}-S_{ell}`.",
            "",
            "The intrinsic quadratic-form terms are",
            "",
            f"- first diagonal: `{diagonal_first:+.12g}`;",
            f"- cross term: `{cross:+.12g}`;",
            f"- second diagonal: `{diagonal_second:+.12g}`;",
            f"- total: `{diagonal_first + cross + diagonal_second:+.12g}`.",
            "",
            f"The negative cross term is `{abs(cross) / diagonal_sum:.3f}` times the two positive diagonal terms combined.",
            "",
            "For this fixed filter, the fitted-null comparison is",
            "",
            f"- intrinsic `v1-vM = {representative['delta_intrinsic']:+.12g}`;",
            f"- compensation `vM-v0 = {representative['delta_compensation']:+.12g}`;",
            f"- total `v1-v0 = {representative['delta_total']:+.12g}`;",
            f"- `rho0 = {representative['rho_fitted']:.12g}` and `g(rho0) = {representative['g_fitted']:.12g}`.",
            "",
            "## Geometry-equivalence rule",
            "",
            "The saved class key retains both segment vectors, their time widths, center displacement, and relative dot/cross geometry. It is canonical under pair exchange, space/time translation, endpoint reversal, and the eight rotations/reflections of the square grid.",
            "The strict ties have zero center displacement. The wider sensitivity set contains the 90-degree rotated counterparts in the same geometric class; their small objective gap is reported rather than silently declaring them exact ties.",
            f"The fitted-null longitude/latitude range ratio is `{symmetry_summary['fitted_range_lon_over_lat']:.12g}` versus the exact target ratio `{symmetry_summary['truth_range_lon_over_lat']:.12g}`. After preserving the fitted range product but imposing the exact ratio, the 28 screened motif variants span only `{symmetry_summary['symmetrized_candidate_spread']:.3e}` in objective value.",
            "That last calculation is a local axis-symmetry sensitivity audit of the 28 screened pairs, not a refit or a second global search.",
            "",
            "## Numerical audit",
            "",
            f"- Ordinary closed-form pairs: `{search.ordinary_pair_count:,}`.",
            f"- Nearly singular positive-definite pairs: `{search.near_singular_pair_count:,}`.",
            f"- Non-positive two-column null Gram matrices: `{search.nonpositive_gram_pair_count:,}`.",
            f"- Tiny negative discriminants clamped to zero in the primary block partition: `{search.clamped_discriminant_count:,}`.",
            f"- Tiny negative discriminants clamped in the verification partition: `{verification_search.clamped_discriminant_count:,}`.",
            f"- Maximum analytic/SciPy discrepancy among final ties: `{ties['analytic_scipy_absolute_difference'].max():.3e}`.",
            f"- Maximum reduced generalized-eigen residual among final ties: `{ties['reduced_relative_residual'].max():.3e}`.",
            f"- Maximum null-normalization error among final ties: `{np.max(np.abs(ties['null_variance'] - 1.0)):.3e}`.",
            "",
            "The clamp count is an execution-order diagnostic and may vary with BLAS reduction order; the minimum, validated tie set, and equivalence classes are the scientific reproducibility targets.",
            "The primary and verification block partitions are required to return the same strict candidate indices and minimum before any files are written.",
            "",
            "`global_pair_ties.csv` retains rectangle IDs, sizes, center displacement, temporal arrangement, overlap diagnostics, `g_ij`, coefficients, and fitted-null variance metrics for every strict tie.",
        ]
    )
    _atomic_text(path, "\n".join(lines) + "\n")


def main() -> None:
    args = build_parser().parse_args()
    started = time.perf_counter()
    oracle_dir = args.oracle_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else oracle_dir / "global_two_rectangle_search"
    )
    manifest, points, metadata = _read_oracle(oracle_dir)
    problem = _reconstruct_problem(manifest, points, metadata)
    tie_tolerance = (
        float(args.tie_tolerance)
        if args.tie_tolerance is not None
        else float(manifest["dictionary"]["score_tie_tolerance"])
    )
    sensitivity_tolerance = float(args.geometry_sensitivity_tolerance)
    if tie_tolerance < 0.0 or sensitivity_tolerance < tie_tolerance:
        raise ValueError("geometry-sensitivity-tolerance must be at least the strict tie tolerance")
    if int(args.verification_block_size) < 1:
        raise ValueError("verification-block-size must be positive")

    def primary_progress(completed: int, total: int) -> None:
        print(f"primary pair blocks: {completed}/{total}", flush=True)

    def verification_progress(completed: int, total: int) -> None:
        print(f"verification pair blocks: {completed}/{total}", flush=True)

    search = exhaustive_two_rectangle_search(
        problem.standardized_dictionary,
        problem.intrinsic_difference,
        problem.null_covariance,
        block_size=int(args.block_size),
        tie_tolerance=tie_tolerance,
        singular_threshold=float(args.singular_threshold),
        maximum_saved_near_singular=int(args.maximum_saved_near_singular),
        progress=primary_progress,
    )
    verification_search = exhaustive_two_rectangle_search(
        problem.standardized_dictionary,
        problem.intrinsic_difference,
        problem.null_covariance,
        block_size=int(args.verification_block_size),
        tie_tolerance=sensitivity_tolerance,
        singular_threshold=float(args.singular_threshold),
        maximum_saved_near_singular=0,
        progress=verification_progress,
    )
    if search.pair_count != verification_search.pair_count:
        raise AssertionError("block partitions visited different pair counts")
    if not np.isclose(
        search.minimum_eigenvalue,
        verification_search.minimum_eigenvalue,
        rtol=0.0,
        atol=1.0e-13,
    ):
        raise ArithmeticError("block partitions produced different global minima")
    if (
        search.near_singular_pair_count != verification_search.near_singular_pair_count
        or search.nonpositive_gram_pair_count != verification_search.nonpositive_gram_pair_count
    ):
        raise AssertionError("block partitions disagree on singular-pair accounting")

    validation_rows = [
        validate_pair_candidate(
            candidate,
            problem.standardized_dictionary,
            problem.intrinsic_difference,
            problem.null_covariance,
        )
        for candidate in verification_search.candidates
    ]
    screened_validation = pd.DataFrame(validation_rows)
    scipy_minimum = float(screened_validation["scipy_eigenvalue"].min())
    strict_radius = tie_tolerance * max(1.0, abs(scipy_minimum))
    sensitivity_radius = sensitivity_tolerance * max(1.0, abs(scipy_minimum))
    validated = screened_validation.loc[
        np.abs(screened_validation["scipy_eigenvalue"] - scipy_minimum) <= sensitivity_radius
    ].copy()
    strict_validated = validated.loc[
        np.abs(validated["scipy_eigenvalue"] - scipy_minimum) <= strict_radius
    ].copy()
    if strict_validated.empty:
        raise AssertionError("SciPy validation removed every analytic candidate")
    if float(validated["analytic_scipy_absolute_difference"].max()) > 1.0e-10:
        raise ArithmeticError("analytic and SciPy pair eigenvalues disagree")
    if float(validated["reduced_relative_residual"].max()) > 1.0e-10:
        raise ArithmeticError("validated pair has an excessive reduced residual")

    primary_indices = {
        (candidate.first_index, candidate.second_index) for candidate in search.candidates
    }
    validated_strict_indices = set(
        strict_validated[["first_index", "second_index"]].itertuples(index=False, name=None)
    )
    verification_analytic_strict_indices = {
        (candidate.first_index, candidate.second_index)
        for candidate in verification_search.candidates
        if abs(candidate.analytic_eigenvalue - verification_search.minimum_eigenvalue)
        <= strict_radius
    }
    if primary_indices != validated_strict_indices:
        raise ArithmeticError("strict analytic candidate set differs from the wider SciPy screen")
    if primary_indices != verification_analytic_strict_indices:
        raise ArithmeticError("strict candidate indices depend on the block partition")

    metadata = metadata.sort_values("rectangle_index").set_index("rectangle_index")
    anchor_rows = (
        points.loc[points["time_index"] == 0].sort_values("anchor_index").set_index("anchor_index")
    )
    described = _describe_candidates(
        validated,
        metadata,
        anchor_rows,
        problem,
    )
    strict_rows = described.loc[
        np.abs(described["scipy_eigenvalue"] - scipy_minimum) <= strict_radius
    ].copy()
    sensitivity_rows = described.loc[
        np.abs(described["scipy_eigenvalue"] - scipy_minimum) <= sensitivity_radius
    ].copy()
    ties, classes = _assign_equivalence_classes(strict_rows)
    sensitivity_pairs, sensitivity_classes = _assign_equivalence_classes(sensitivity_rows)
    outside_strict = sensitivity_pairs.loc[
        sensitivity_pairs["scipy_eigenvalue"] > scipy_minimum + strict_radius
    ]
    next_gap = (
        float(outside_strict["scipy_eigenvalue"].min() - scipy_minimum)
        if len(outside_strict)
        else float("nan")
    )
    axis_symmetry_pairs, axis_symmetry_summary = _axis_symmetry_sensitivity(
        sensitivity_pairs,
        problem,
        manifest,
    )

    greedy = pd.read_csv(oracle_dir / "greedy_path.csv")
    greedy_row = greedy.loc[(greedy["branch"] == "negative") & (greedy["size"] == 2)].iloc[0]
    greedy_pair = tuple(
        sorted(int(value) for value in str(greedy_row["selected_rectangle_indices"]).split(";"))
    )
    greedy_mu = float(greedy_row["objective_mu"])

    near = pd.DataFrame(search.near_singular_records)
    if len(near):
        near["first_rectangle_id"] = near["first_index"].map(metadata["rectangle_id"])
        near["second_rectangle_id"] = near["second_index"].map(metadata["rectangle_id"])
    else:
        near = pd.DataFrame(
            columns=[
                "first_index",
                "second_index",
                "first_rectangle_id",
                "second_rectangle_id",
                "g_ij",
                "one_minus_g_squared",
                "lower_eigenvalue",
            ]
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_csv(output_dir / "global_pair_ties.csv", ties)
    _atomic_csv(output_dir / "equivalence_classes.csv", classes)
    _atomic_csv(output_dir / "geometry_sensitivity_pairs.csv", sensitivity_pairs)
    _atomic_csv(
        output_dir / "geometry_sensitivity_equivalence_classes.csv",
        sensitivity_classes,
    )
    _atomic_csv(output_dir / "axis_symmetry_sensitivity.csv", axis_symmetry_pairs)
    _atomic_csv(output_dir / "near_singular_pairs.csv", near)
    _write_report(
        output_dir / "REPORT.md",
        ties,
        classes,
        sensitivity_pairs,
        sensitivity_classes,
        search,
        verification_search,
        greedy_pair,
        greedy_mu,
        tie_tolerance=tie_tolerance,
        sensitivity_tolerance=sensitivity_tolerance,
        next_gap=next_gap,
        symmetry_summary=axis_symmetry_summary,
    )
    elapsed = time.perf_counter() - started
    representative = ties.sort_values(["first_rectangle_index", "second_rectangle_index"]).iloc[0]
    input_paths = (
        oracle_dir / "experiment_manifest.json",
        oracle_dir / "exact_comoving_points.csv",
        oracle_dir / "rectangle_dictionary_metadata.csv",
        oracle_dir / "greedy_path.csv",
    )
    source_paths = (
        Path(__file__).resolve(),
        HERE / "global_two_rectangle_search.py",
        HERE / "rectangle_dictionary_core.py",
        HERE / "diagnostic_core.py",
        HERE / "analyze_mode_count_path.py",
        HERE / "run_rectangle_dictionary_oracle.py",
    )
    run_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "oracle_dir": str(oracle_dir),
        "oracle_manifest_sha256": sha256(oracle_dir / "experiment_manifest.json"),
        "population_covariance_only": True,
        "responses_read": False,
        "objective": "(Sigma1-SigmaM) normalized by Sigma0",
        "not_global_for": ["Sigma1-Sigma0", "rho0", "g(rho0)"],
        "dtype": "float64",
        "rectangle_count": int(problem.standardized_dictionary.shape[1]),
        "unordered_pair_count": search.pair_count,
        "primary_block_size": search.block_size,
        "verification_block_size": verification_search.block_size,
        "block_invariant_minimum": True,
        "block_invariant_strict_candidate_indices": True,
        "tie_tolerance": search.tie_tolerance,
        "geometry_sensitivity_tolerance": sensitivity_tolerance,
        "singular_threshold": search.singular_threshold,
        "ordinary_pair_count": search.ordinary_pair_count,
        "near_singular_pair_count": search.near_singular_pair_count,
        "nonpositive_gram_pair_count": search.nonpositive_gram_pair_count,
        "saved_near_singular_pair_count": len(near),
        "primary_clamped_discriminant_count_execution_dependent": (
            search.clamped_discriminant_count
        ),
        "verification_clamped_discriminant_count_execution_dependent": (
            verification_search.clamped_discriminant_count
        ),
        "analytic_global_minimum": search.minimum_eigenvalue,
        "validated_global_minimum": scipy_minimum,
        "strict_global_tie_count": len(ties),
        "strict_equivalence_class_count": len(classes),
        "geometry_sensitivity_pair_count": len(sensitivity_pairs),
        "geometry_sensitivity_equivalence_class_count": len(sensitivity_classes),
        "next_objective_level_gap": next_gap,
        "axis_symmetry_sensitivity": axis_symmetry_summary,
        "equivalence_group": (
            "space/time translation, unordered pair exchange, endpoint reversal, "
            "and square-grid D4 rotations/reflections"
        ),
        "greedy_pair": list(greedy_pair),
        "greedy_objective_mu": greedy_mu,
        "greedy_pair_is_global_tie": bool(
            np.any(
                (ties["first_rectangle_index"] == greedy_pair[0])
                & (ties["second_rectangle_index"] == greedy_pair[1])
            )
        ),
        "maximum_analytic_scipy_absolute_difference": float(
            ties["analytic_scipy_absolute_difference"].max()
        ),
        "maximum_reduced_relative_residual": float(ties["reduced_relative_residual"].max()),
        "representative_fitted_null_metrics": {
            key: float(representative[key])
            for key in (
                "v0",
                "v_matched",
                "v_true",
                "delta_intrinsic",
                "delta_compensation",
                "delta_total",
                "rho_fitted",
                "g_fitted",
            )
        },
        "elapsed_seconds": elapsed,
        "runtime": runtime_environment(),
        "runtime_summary": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "git": git_state(),
        "interpretation_boundary": (
            "Exact-comoving covariance oracle only; not a calibrated test and not a "
            "claim about the warped GEMS observation geometry. The exhaustive target "
            "is intrinsic Sigma1-SigmaM, not total Sigma1-Sigma0."
        ),
        "inputs": [_file_record(path) for path in input_paths],
        "source_files": [_file_record(path) for path in source_paths],
    }
    run_manifest["output_files"] = _output_inventory(output_dir)
    atomic_json(output_dir / "search_manifest.json", run_manifest)
    print(
        f"global minimum={scipy_minimum:.12g}; ties={len(ties)}; "
        f"sensitivity_pairs={len(sensitivity_pairs)}; classes={len(classes)}; "
        f"elapsed={elapsed:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()
