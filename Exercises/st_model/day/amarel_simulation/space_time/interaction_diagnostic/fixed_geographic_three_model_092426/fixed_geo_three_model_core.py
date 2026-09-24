"""Shared, deterministic core for the fixed-geographic three-model study."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


MODEL_ORDER = ("gc", "matern05", "separable")
DAY_PATTERN = re.compile(r"day(?P<day>\d{2})")


def task_source_files() -> tuple[Path, ...]:
    """Return every source file whose change invalidates fitted task caches."""

    study_directory = Path(__file__).resolve().parent
    project_root = study_directory.parents[6]
    relative_sources = (
        "src/GEMS_TCO/__init__.py",
        "src/GEMS_TCO/orderings.py",
        "src/GEMS_TCO/data/__init__.py",
        "src/GEMS_TCO/data/loading.py",
        "src/GEMS_TCO/data/preprocessing.py",
        "src/GEMS_TCO/spatial/__init__.py",
        "src/GEMS_TCO/spatial/matern_spline.py",
        "src/GEMS_TCO/vecchia/__init__.py",
        "src/GEMS_TCO/vecchia/_base.py",
        "src/GEMS_TCO/vecchia/_native_covariance.py",
        "src/GEMS_TCO/vecchia/grouped_batched.py",
        "src/GEMS_TCO/vecchia/corridor_neighbors/__init__.py",
        "src/GEMS_TCO/vecchia/corridor_neighbors/_geometry.py",
        "src/GEMS_TCO/vecchia/corridor_neighbors/corridor_lag432.py",
        "src/GEMS_TCO/vecchia/corridor_neighbors/corridor_lag643.py",
        "src/GEMS_TCO/vecchia/corridor_neighbors/directional_lag432.py",
        "src/GEMS_TCO/vecchia/corridor_neighbors/directional_lag643.py",
        "src/GEMS_TCO/vecchia/corridor_neighbors/generalized_cauchy.py",
        "src/GEMS_TCO/vecchia/corridor_neighbors/separable_exponential.py",
        "src/GEMS_TCO/vecchia/corridor_neighbors/spline.py",
        "cpp/maxmin_order.cpp",
    )
    return (
        study_directory / "fixed_geo_three_model_core.py",
        study_directory / "run_fixed_geo_three_model_day.py",
        *(project_root / relative for relative in relative_sources),
    )


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def study_signature(config_path: Path, design_path: Path, dates_path: Path) -> str:
    digest = hashlib.sha256()
    project_root = Path(__file__).resolve().parent.parents[6]
    signature_files = (config_path, design_path, dates_path, *task_source_files())
    for path in signature_files:
        if not path.is_file():
            raise FileNotFoundError(f"signature source is missing: {path}")
        try:
            stable_name = path.resolve().relative_to(project_root).as_posix()
        except ValueError:
            stable_name = path.name
        digest.update(stable_name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def atomic_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    compression = "gzip" if path.suffix == ".gz" else None
    frame.to_csv(
        temporary,
        index=False,
        float_format="%.17g",
        compression=compression,
    )
    temporary.replace(path)


def clean_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean_json(item) for item in value]
    return value


def select_day_frames(
    frames: Mapping[str, pd.DataFrame], day: int, expected_slots: int = 8
) -> dict[str, pd.DataFrame]:
    selected: dict[str, pd.DataFrame] = {}
    for key, frame in frames.items():
        match = DAY_PATTERN.search(str(key))
        if match and int(match.group("day")) == int(day):
            selected[str(key)] = frame.reset_index(drop=True)
    selected = dict(sorted(selected.items()))
    if len(selected) != int(expected_slots):
        raise ValueError(
            f"day {int(day):02d} has {len(selected)} time slots; expected {expected_slots}"
        )
    return selected


def _uniform_axis(values: np.ndarray, name: str) -> tuple[np.ndarray, float]:
    axis = np.sort(np.unique(np.asarray(values, dtype=np.float64)))
    if axis.size < 2:
        raise ValueError(f"{name} axis must contain at least two coordinates")
    increments = np.diff(axis)
    step = float(np.median(increments))
    if not np.allclose(increments, step, rtol=0.0, atol=1e-10):
        raise ValueError(f"{name} regular-grid axis is not uniform")
    return axis, step


def _nearest_axis_indices(
    targets: np.ndarray, axis: np.ndarray, step: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.rint((np.asarray(targets) - axis[0]) / step).astype(np.int64)
    inside = (raw >= 0) & (raw < axis.size)
    clipped = np.clip(raw, 0, axis.size - 1)
    error = np.abs(axis[clipped] - targets)
    inside &= error <= 0.5 * step + 1e-10
    return clipped, error, inside


def _frame_cubes(
    frames: Mapping[str, pd.DataFrame], monthly_mean: float, time_origin_hours: float
) -> dict[str, np.ndarray]:
    ordered = list(frames.values())
    reference = ordered[0]
    required = {
        "Latitude",
        "Longitude",
        "Source_Latitude",
        "Source_Longitude",
        "ColumnAmountO3",
        "Hours_elapsed",
    }
    for frame in ordered:
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"daily frame is missing columns {sorted(missing)}")

    latitudes, latitude_step = _uniform_axis(
        reference["Latitude"].to_numpy(dtype=np.float64), "latitude"
    )
    longitudes, longitude_step = _uniform_axis(
        reference["Longitude"].to_numpy(dtype=np.float64), "longitude"
    )
    shape = (len(ordered), latitudes.size, longitudes.size)
    response = np.full(shape, np.nan, dtype=np.float64)
    source_latitude = np.full(shape, np.nan, dtype=np.float64)
    source_longitude = np.full(shape, np.nan, dtype=np.float64)
    actual_time_hours = np.empty(len(ordered), dtype=np.float64)
    model_time_hours = np.empty(len(ordered), dtype=np.float64)
    regular_reference = reference[["Latitude", "Longitude"]].to_numpy(dtype=np.float64)

    for time_index, frame in enumerate(ordered):
        regular = frame[["Latitude", "Longitude"]].to_numpy(dtype=np.float64)
        if regular.shape != regular_reference.shape or not np.allclose(
            regular, regular_reference, rtol=0.0, atol=1e-10
        ):
            raise ValueError("regular-grid coordinates or row order changed within a day")
        latitude_index = np.rint((regular[:, 0] - latitudes[0]) / latitude_step).astype(np.int64)
        longitude_index = np.rint((regular[:, 1] - longitudes[0]) / longitude_step).astype(np.int64)
        if np.unique(np.column_stack([latitude_index, longitude_index]), axis=0).shape[0] != len(
            frame
        ):
            raise ValueError("regular-grid cells are not one-to-one")
        ozone = pd.to_numeric(frame["ColumnAmountO3"], errors="coerce").to_numpy(dtype=np.float64)
        source_lat = pd.to_numeric(frame["Source_Latitude"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        source_lon = pd.to_numeric(frame["Source_Longitude"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        hours = pd.to_numeric(frame["Hours_elapsed"], errors="coerce").to_numpy(dtype=np.float64)
        finite_hours = hours[np.isfinite(hours)]
        if finite_hours.size == 0:
            raise ValueError("an hourly frame contains no finite Hours_elapsed value")
        actual_time_hours[time_index] = float(np.median(finite_hours))
        if not np.allclose(finite_hours, actual_time_hours[time_index], rtol=0.0, atol=1e-10):
            raise ValueError("Hours_elapsed is not constant within an hourly frame")
        model_time_hours[time_index] = float(
            np.round(actual_time_hours[time_index] - float(time_origin_hours))
        )
        observed = np.isfinite(ozone) & np.isfinite(source_lat) & np.isfinite(source_lon)
        response[time_index, latitude_index[observed], longitude_index[observed]] = ozone[
            observed
        ] - float(monthly_mean)
        source_latitude[time_index, latitude_index[observed], longitude_index[observed]] = (
            source_lat[observed]
        )
        source_longitude[time_index, latitude_index[observed], longitude_index[observed]] = (
            source_lon[observed]
        )

    return {
        "latitudes": latitudes,
        "longitudes": longitudes,
        "latitude_step": np.asarray(latitude_step),
        "longitude_step": np.asarray(longitude_step),
        "response": response,
        "source_latitude": source_latitude,
        "source_longitude": source_longitude,
        "actual_time_hours": actual_time_hours,
        "model_time_hours": model_time_hours,
    }


def _mean_design(
    source_latitude: np.ndarray, time_index: int, latitude_center: float
) -> np.ndarray:
    design = np.zeros((source_latitude.size, 9), dtype=np.float64)
    design[:, 0] = 1.0
    design[:, 1] = source_latitude - latitude_center
    if time_index > 0:
        design[:, time_index + 1] = 1.0
    return design


def _common_ols(cube: dict[str, np.ndarray]) -> tuple[np.ndarray, float]:
    latitude_values = cube["source_latitude"][np.isfinite(cube["response"])]
    latitude_center = float(np.mean(latitude_values))
    rows: list[np.ndarray] = []
    values: list[np.ndarray] = []
    for time_index in range(cube["response"].shape[0]):
        response = cube["response"][time_index].ravel()
        source_latitude = cube["source_latitude"][time_index].ravel()
        observed = np.isfinite(response) & np.isfinite(source_latitude)
        rows.append(_mean_design(source_latitude[observed], time_index, latitude_center))
        values.append(response[observed])
    design = np.concatenate(rows, axis=0)
    response = np.concatenate(values)
    beta, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    if rank != 9:
        raise RuntimeError(f"common OLS mean design has rank {rank}, expected 9")
    return beta, latitude_center


def prepare_fixed_contrasts(
    frames: Mapping[str, pd.DataFrame],
    monthly_mean: float,
    frozen_design: Mapping[str, Any],
    wx_tolerance: float,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Build shared empirical contrasts and their exact source coordinates."""

    time_origin_hours = float(frozen_design["model_time_origin_hours"])
    cube = _frame_cubes(frames, monthly_mean, time_origin_hours)
    time_count = cube["response"].shape[0]
    lag = int(frozen_design["temporal_lag"])
    if lag != 1 or time_count != 8:
        raise ValueError("the frozen design requires eight slots and lag one")
    expected_model_time_increments = np.ones(time_count - 1, dtype=np.float64)
    model_time_increments = np.diff(cube["model_time_hours"])
    if not np.array_equal(model_time_increments, expected_model_time_increments):
        raise RuntimeError(
            "rounded model times no longer match the frozen nominal adjacent-slot convention: "
            f"observed increments {model_time_increments.tolist()}"
        )
    coefficient_a = np.asarray(
        frozen_design["contrast_coefficients_on_eight_points"]["Q_A"], dtype=np.float64
    )
    coefficient_b = np.asarray(
        frozen_design["contrast_coefficients_on_eight_points"]["Q_B"], dtype=np.float64
    )
    d1 = float(frozen_design["raw_coefficients"]["d1"])
    d2 = float(frozen_design["raw_coefficients"]["d2"])

    latitudes = cube["latitudes"]
    longitudes = cube["longitudes"]
    regular_grid_shape = [int(latitudes.size), int(longitudes.size)]
    expected_grid_shape = frozen_design.get("expected_regular_grid_shape")
    if expected_grid_shape is not None and regular_grid_shape != [
        int(value) for value in expected_grid_shape
    ]:
        raise RuntimeError(
            "regular-grid shape changed: "
            f"observed {regular_grid_shape}, expected {expected_grid_shape}"
        )
    grid_axis_digest = hashlib.sha256()
    grid_axis_digest.update(np.asarray(latitudes, dtype="<f8").tobytes())
    grid_axis_digest.update(np.asarray(longitudes, dtype="<f8").tobytes())
    grid_axis_sha256 = grid_axis_digest.hexdigest()
    expected_grid_axis_sha256 = frozen_design.get("expected_regular_grid_axes_sha256")
    if expected_grid_axis_sha256 is not None and grid_axis_sha256 != expected_grid_axis_sha256:
        raise RuntimeError(
            "regular-grid axes changed: "
            f"observed {grid_axis_sha256}, expected {expected_grid_axis_sha256}"
        )
    latitude_step = float(cube["latitude_step"])
    longitude_step = float(cube["longitude_step"])
    center_latitude, center_longitude = np.meshgrid(latitudes, longitudes, indexing="ij")
    center_latitude = center_latitude.ravel()
    center_longitude = center_longitude.ravel()
    center_latitude_index, center_longitude_index = np.meshgrid(
        np.arange(latitudes.size), np.arange(longitudes.size), indexing="ij"
    )
    center_latitude_index = center_latitude_index.ravel()
    center_longitude_index = center_longitude_index.ravel()

    observed_latitudes = cube["source_latitude"][np.isfinite(cube["response"])]
    latitude_center = float(np.mean(observed_latitudes))
    batches: list[dict[str, Any]] = []
    geometry_hash = hashlib.sha256()
    anchor_counts: dict[str, int] = {}
    coverage_by_mirror_and_time: dict[str, dict[str, float | int]] = {}
    frozen_latitude_range = float(frozen_design["frozen_ranges"]["latitude"])
    frozen_longitude_range = float(frozen_design["frozen_ranges"]["longitude"])

    for handedness in ("clockwise", "counterclockwise"):
        offsets = np.asarray(frozen_design["physical_offsets"][handedness], dtype=np.float64)
        desired_latitude = center_latitude[:, None] + offsets[None, :, 0]
        desired_longitude = center_longitude[:, None] + offsets[None, :, 1]
        latitude_index, latitude_error, latitude_valid = _nearest_axis_indices(
            desired_latitude, latitudes, latitude_step
        )
        longitude_index, longitude_error, longitude_valid = _nearest_axis_indices(
            desired_longitude, longitudes, longitude_step
        )
        geometry_valid = np.all(latitude_valid & longitude_valid, axis=1)
        selected_anchor = np.flatnonzero(geometry_valid)
        latitude_index = latitude_index[geometry_valid]
        longitude_index = longitude_index[geometry_valid]
        selected_desired_latitude = desired_latitude[geometry_valid]
        selected_desired_longitude = desired_longitude[geometry_valid]
        anchor_counts[handedness] = int(selected_anchor.size)
        geometry_hash.update(handedness.encode("utf-8"))
        geometry_hash.update(
            np.column_stack(
                [
                    center_latitude_index[geometry_valid],
                    center_longitude_index[geometry_valid],
                    latitude_index,
                    longitude_index,
                ]
            )
            .astype("<i8", copy=False)
            .tobytes()
        )
        maximum_grid_error_standardized = np.sqrt(
            np.square(latitude_error / frozen_latitude_range)
            + np.square(longitude_error / frozen_longitude_range)
        )[geometry_valid].max(axis=1)

        for time_index in range(time_count - lag):
            next_time = time_index + lag
            point_specs = (
                (time_index, 0),
                (time_index, 1),
                (next_time, 0),
                (next_time, 1),
                (time_index, 2),
                (time_index, 3),
                (next_time, 2),
                (next_time, 3),
            )
            count = selected_anchor.size
            values = np.full((count, 8), np.nan, dtype=np.float64)
            coordinates = np.full((count, 8, 3), np.nan, dtype=np.float64)
            desired_coordinates = np.empty((count, 8, 2), dtype=np.float64)
            point_design = np.zeros((count, 8, 9), dtype=np.float64)
            for point_index, (slot, endpoint) in enumerate(point_specs):
                lat_index = latitude_index[:, endpoint]
                lon_index = longitude_index[:, endpoint]
                values[:, point_index] = cube["response"][slot, lat_index, lon_index]
                coordinates[:, point_index, 0] = cube["source_latitude"][slot, lat_index, lon_index]
                coordinates[:, point_index, 1] = cube["source_longitude"][
                    slot, lat_index, lon_index
                ]
                coordinates[:, point_index, 2] = cube["model_time_hours"][slot]
                desired_coordinates[:, point_index, 0] = selected_desired_latitude[:, endpoint]
                desired_coordinates[:, point_index, 1] = selected_desired_longitude[:, endpoint]
                point_design[:, point_index, :] = _mean_design(
                    coordinates[:, point_index, 0], slot, latitude_center
                )
            valid = np.isfinite(values).all(axis=1) & np.isfinite(coordinates).all(axis=(1, 2))
            coverage_key = f"{handedness}_t{time_index:02d}_t{next_time:02d}"
            coverage_by_mirror_and_time[coverage_key] = {
                "candidate_count": int(count),
                "complete_count": int(np.count_nonzero(valid)),
                "complete_fraction": float(np.mean(valid)),
            }
            if not np.any(valid):
                continue
            source_snap_error_standardized = np.sqrt(
                np.square(
                    (coordinates[:, :, 0] - desired_coordinates[:, :, 0]) / frozen_latitude_range
                )
                + np.square(
                    (coordinates[:, :, 1] - desired_coordinates[:, :, 1]) / frozen_longitude_range
                )
            )
            batches.append(
                {
                    "handedness": handedness,
                    "time_index": time_index,
                    "anchor": selected_anchor[valid],
                    "center_latitude_index": center_latitude_index[geometry_valid][valid],
                    "center_longitude_index": center_longitude_index[geometry_valid][valid],
                    "center_latitude": center_latitude[geometry_valid][valid],
                    "center_longitude": center_longitude[geometry_valid][valid],
                    "values": values[valid],
                    "coordinates": coordinates[valid],
                    "design": point_design[valid],
                    "maximum_grid_error_standardized": maximum_grid_error_standardized[valid],
                    "source_snap_error_standardized": source_snap_error_standardized[valid],
                }
            )

    if not batches:
        raise RuntimeError("no complete frozen contrasts are available for this day")
    raw_values = np.concatenate([batch["values"] for batch in batches], axis=0)
    coordinates = np.concatenate([batch["coordinates"] for batch in batches], axis=0)
    point_design = np.concatenate([batch["design"] for batch in batches], axis=0)
    wx_a = np.einsum("i,nip->np", coefficient_a, point_design, optimize=True)
    wx_b = np.einsum("i,nip->np", coefficient_b, point_design, optimize=True)
    wx_max = float(max(np.max(np.abs(wx_a)), np.max(np.abs(wx_b))))

    common_beta: np.ndarray | None = None
    if wx_max <= float(wx_tolerance):
        analysis_values = raw_values
        mean_rule = "monthly_centered_raw_y_because_wx_is_zero"
    else:
        common_beta, common_latitude_center = _common_ols(cube)
        if not np.isclose(common_latitude_center, latitude_center, rtol=0.0, atol=1e-12):
            raise ArithmeticError("latitude centering changed during the common mean audit")
        analysis_values = raw_values - np.einsum(
            "nip,p->ni", point_design, common_beta, optimize=True
        )
        mean_rule = "one_common_daywise_ols_mean_for_all_models"

    raw_q_a = raw_values @ coefficient_a
    raw_q_b = raw_values @ coefficient_b
    q_a = analysis_values @ coefficient_a
    q_b = analysis_values @ coefficient_b
    mean_adjustment_q_a = raw_q_a - q_a
    mean_adjustment_q_b = raw_q_b - q_b
    contrast_l = d1 * q_a + d2 * q_b
    sample_count = len(q_a)
    source_snap_error_standardized = np.concatenate(
        [batch["source_snap_error_standardized"] for batch in batches], axis=0
    )
    regular_grid_snap_error_standardized = np.concatenate(
        [batch["maximum_grid_error_standardized"] for batch in batches]
    )
    per_sample_maximum_source_error = np.max(source_snap_error_standardized, axis=1)
    samples = pd.DataFrame(
        {
            "sample_id": np.arange(sample_count, dtype=np.int64),
            "handedness": np.concatenate(
                [np.repeat(batch["handedness"], len(batch["anchor"])) for batch in batches]
            ),
            "time_t": np.concatenate(
                [np.repeat(batch["time_index"], len(batch["anchor"])) for batch in batches]
            ),
            "time_t_plus_1": np.concatenate(
                [np.repeat(batch["time_index"] + 1, len(batch["anchor"])) for batch in batches]
            ),
            "anchor_flat_index": np.concatenate([batch["anchor"] for batch in batches]),
            "anchor_latitude_index": np.concatenate(
                [batch["center_latitude_index"] for batch in batches]
            ),
            "anchor_longitude_index": np.concatenate(
                [batch["center_longitude_index"] for batch in batches]
            ),
            "anchor_latitude": np.concatenate([batch["center_latitude"] for batch in batches]),
            "anchor_longitude": np.concatenate([batch["center_longitude"] for batch in batches]),
            "q_a": q_a,
            "q_b": q_b,
            "raw_q_a": raw_q_a,
            "raw_q_b": raw_q_b,
            "common_mean_adjustment_q_a": mean_adjustment_q_a,
            "common_mean_adjustment_q_b": mean_adjustment_q_b,
            "contrast_l": contrast_l,
            "maximum_regular_grid_snap_error_standardized": regular_grid_snap_error_standardized,
            "maximum_source_error_standardized": per_sample_maximum_source_error,
            **{
                f"source_error_standardized_{point_name}": source_snap_error_standardized[
                    :, point_index
                ]
                for point_index, point_name in enumerate(frozen_design["point_order"])
            },
        }
    )
    mapping_sha256 = geometry_hash.hexdigest()
    expected_counts = frozen_design.get("expected_anchor_count_by_mirror")
    if (
        expected_counts is not None
        and {str(key): int(value) for key, value in expected_counts.items()} != anchor_counts
    ):
        raise RuntimeError(
            f"fixed anchor counts changed: observed {anchor_counts}, expected {expected_counts}"
        )
    expected_mapping_sha256 = frozen_design.get("expected_anchor_mapping_sha256")
    if expected_mapping_sha256 is not None and mapping_sha256 != expected_mapping_sha256:
        raise RuntimeError(
            "fixed anchor mapping changed: "
            f"observed {mapping_sha256}, expected {expected_mapping_sha256}"
        )
    adjustment_rms_q_a = float(np.sqrt(np.mean(np.square(mean_adjustment_q_a))))
    adjustment_rms_q_b = float(np.sqrt(np.mean(np.square(mean_adjustment_q_b))))
    raw_sd_q_a = float(np.std(raw_q_a))
    raw_sd_q_b = float(np.std(raw_q_b))
    potential_sample_count = int(sum(anchor_counts.values()) * (time_count - lag))
    audit = {
        "sample_count": sample_count,
        "potential_sample_count": potential_sample_count,
        "complete_contrast_fraction": sample_count / potential_sample_count,
        "coverage_by_mirror_and_time": coverage_by_mirror_and_time,
        "candidate_anchor_count_by_mirror": anchor_counts,
        "frozen_anchor_mapping_sha256": mapping_sha256,
        "regular_grid_shape": regular_grid_shape,
        "regular_grid_axes_sha256": grid_axis_sha256,
        "regular_latitude_step": latitude_step,
        "regular_longitude_step": longitude_step,
        "raw_frame_time_hours": cube["actual_time_hours"].tolist(),
        "raw_adjacent_time_increments_hours": np.diff(cube["actual_time_hours"]).tolist(),
        "rounded_model_time_hours": cube["model_time_hours"].tolist(),
        "rounded_model_adjacent_time_increments_hours": model_time_increments.tolist(),
        "model_time_origin_hours": time_origin_hours,
        "model_time_convention": frozen_design["model_time_convention"],
        "maximum_regular_grid_snap_error_standardized_median": float(
            np.median(regular_grid_snap_error_standardized)
        ),
        "maximum_regular_grid_snap_error_standardized_p95": float(
            np.quantile(regular_grid_snap_error_standardized, 0.95)
        ),
        "maximum_regular_grid_snap_error_standardized_max": float(
            np.max(regular_grid_snap_error_standardized)
        ),
        "source_endpoint_error_standardized_median": float(
            np.median(source_snap_error_standardized)
        ),
        "source_endpoint_error_standardized_p95": float(
            np.quantile(source_snap_error_standardized, 0.95)
        ),
        "source_endpoint_error_standardized_max": float(np.max(source_snap_error_standardized)),
        "per_sample_maximum_source_error_standardized_median": float(
            np.median(per_sample_maximum_source_error)
        ),
        "per_sample_maximum_source_error_standardized_p95": float(
            np.quantile(per_sample_maximum_source_error, 0.95)
        ),
        "per_sample_maximum_source_error_standardized_max": float(
            np.max(per_sample_maximum_source_error)
        ),
        "wx_max_abs": wx_max,
        "wx_tolerance": float(wx_tolerance),
        "wx_numerically_zero": bool(wx_max <= float(wx_tolerance)),
        "empirical_mean_rule": mean_rule,
        "common_ols_beta": None if common_beta is None else common_beta.tolist(),
        "common_ols_latitude_center": latitude_center if common_beta is not None else None,
        "common_mean_adjustment_rms_q_a": adjustment_rms_q_a,
        "common_mean_adjustment_rms_q_b": adjustment_rms_q_b,
        "common_mean_adjustment_max_abs_q_a": float(np.max(np.abs(mean_adjustment_q_a))),
        "common_mean_adjustment_max_abs_q_b": float(np.max(np.abs(mean_adjustment_q_b))),
        "common_mean_adjustment_rms_over_raw_sd_q_a": (
            adjustment_rms_q_a / raw_sd_q_a if raw_sd_q_a > 0 else None
        ),
        "common_mean_adjustment_rms_over_raw_sd_q_b": (
            adjustment_rms_q_b / raw_sd_q_b if raw_sd_q_b > 0 else None
        ),
        "monthly_centering_mean": float(monthly_mean),
    }
    return samples, coordinates, audit


def physical_to_raw_parameters(physical: Mapping[str, float], include_nugget: bool) -> np.ndarray:
    variance = float(physical["signal_variance"])
    range_lat = float(physical["range_lat"])
    range_lon = float(physical["range_lon"])
    range_time = float(physical["range_time"])
    if min(variance, range_lat, range_lon, range_time) <= 0:
        raise ValueError("variance and ranges must be positive")
    phi2 = 1.0 / range_lon
    raw = [
        math.log(variance * phi2),
        math.log(phi2),
        math.log((range_lon / range_lat) ** 2),
        math.log((range_lon / range_time) ** 2),
        float(physical["advec_lat"]),
        float(physical["advec_lon"]),
    ]
    if include_nugget:
        nugget = float(physical["nugget"])
        if nugget <= 0:
            raise ValueError("a fitted nugget must be positive on the log scale")
        raw.append(math.log(nugget))
    return np.asarray(raw, dtype=np.float64)


def contrast_covariances(
    coordinates: np.ndarray,
    physical: Mapping[str, float],
    model_name: str,
    frozen_design: Mapping[str, Any],
    gc_alpha: float = 0.75,
    gc_beta: float = 1.0,
) -> np.ndarray:
    """Evaluate each sample's 2x2 covariance under a fitted model."""

    if model_name not in MODEL_ORDER:
        raise ValueError(f"unknown model {model_name!r}")
    coordinates = np.asarray(coordinates, dtype=np.float64)
    delta = coordinates[:, :, None, :] - coordinates[:, None, :, :]
    delta_time = delta[..., 2]
    shifted_latitude = delta[..., 0] - float(physical["advec_lat"]) * delta_time
    shifted_longitude = delta[..., 1] - float(physical["advec_lon"]) * delta_time
    spatial_distance = np.sqrt(
        np.square(shifted_latitude / float(physical["range_lat"]))
        + np.square(shifted_longitude / float(physical["range_lon"]))
    )
    temporal_distance = np.abs(delta_time) / float(physical["range_time"])
    if model_name == "separable":
        correlation = np.exp(-spatial_distance - temporal_distance)
    else:
        joint_distance = np.sqrt(np.square(spatial_distance) + np.square(temporal_distance))
        if model_name == "matern05":
            correlation = np.exp(-joint_distance)
        else:
            correlation = np.power(
                1.0 + np.power(joint_distance, float(gc_alpha)),
                -float(gc_beta) / float(gc_alpha),
            )
    covariance = float(physical["signal_variance"]) * correlation
    nugget = float(physical.get("nugget", 0.0))
    if nugget:
        exact_match = np.all(delta == 0.0, axis=-1)
        covariance = covariance + nugget * exact_match
    coefficient = np.stack(
        [
            np.asarray(
                frozen_design["contrast_coefficients_on_eight_points"]["Q_A"],
                dtype=np.float64,
            ),
            np.asarray(
                frozen_design["contrast_coefficients_on_eight_points"]["Q_B"],
                dtype=np.float64,
            ),
        ]
    )
    return np.einsum("ai,nij,bj->nab", coefficient, covariance, coefficient, optimize=True)


def score_model(
    samples: pd.DataFrame,
    contrast_covariance: np.ndarray,
    model_name: str,
    frozen_design: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    def empirical_correlation(covariance: float, variance_a: float, variance_b: float):
        denominator = math.sqrt(max(variance_a * variance_b, 0.0))
        return covariance / denominator if denominator > 0.0 else None

    q_a = samples["q_a"].to_numpy(dtype=np.float64)
    q_b = samples["q_b"].to_numpy(dtype=np.float64)
    h_aa = contrast_covariance[:, 0, 0]
    h_ab = contrast_covariance[:, 0, 1]
    h_bb = contrast_covariance[:, 1, 1]
    determinant = h_aa * h_bb - np.square(h_ab)
    if np.any(h_aa <= 0) or np.any(h_bb <= 0) or np.any(determinant <= 0):
        raise ArithmeticError(f"{model_name} produced a non-positive 2x2 contrast covariance")
    quadratic = (
        h_bb * np.square(q_a) - 2.0 * h_ab * q_a * q_b + h_aa * np.square(q_b)
    ) / determinant
    score = 0.5 * (np.log(determinant) + quadratic)
    d1 = float(frozen_design["raw_coefficients"]["d1"])
    d2 = float(frozen_design["raw_coefficients"]["d2"])
    var_l_diagonal_a = d1 * d1 * h_aa
    var_l_diagonal_b = d2 * d2 * h_bb
    var_l_cross = 2.0 * d1 * d2 * h_ab
    var_l = var_l_diagonal_a + var_l_diagonal_b + var_l_cross
    predicted = pd.DataFrame(
        {
            "sample_id": samples["sample_id"].to_numpy(),
            "model": model_name,
            "h_aa": h_aa,
            "h_bb": h_bb,
            "h_ab": h_ab,
            "rho_ab": h_ab / np.sqrt(h_aa * h_bb),
            "var_l_diagonal_a": var_l_diagonal_a,
            "var_l_diagonal_b": var_l_diagonal_b,
            "var_l_cross": var_l_cross,
            "var_l": var_l,
            "constant_free_gaussian_score": score,
        }
    )
    empirical_second_moment_a = float(np.mean(np.square(q_a)))
    empirical_second_moment_b = float(np.mean(np.square(q_b)))
    empirical_cross_moment_ab = float(np.mean(q_a * q_b))
    empirical_second_moment_rho = empirical_correlation(
        empirical_cross_moment_ab,
        empirical_second_moment_a,
        empirical_second_moment_b,
    )
    empirical_centered_variance_a = float(np.var(q_a))
    empirical_centered_variance_b = float(np.var(q_b))
    empirical_centered_covariance_ab = float(np.mean((q_a - np.mean(q_a)) * (q_b - np.mean(q_b))))
    empirical_centered_rho = empirical_correlation(
        empirical_centered_covariance_ab,
        empirical_centered_variance_a,
        empirical_centered_variance_b,
    )
    empirical_diag_a = d1 * d1 * empirical_second_moment_a
    empirical_diag_b = d2 * d2 * empirical_second_moment_b
    empirical_cross = 2.0 * d1 * d2 * empirical_cross_moment_ab
    model_v_a = float(np.mean(h_aa))
    model_v_b = float(np.mean(h_bb))
    model_c_ab = float(np.mean(h_ab))
    summary = {
        "model": model_name,
        "sample_count": int(len(samples)),
        "mean_contrast_score": float(np.mean(score)),
        "empirical_mean_q_a": float(np.mean(q_a)),
        "empirical_mean_q_b": float(np.mean(q_b)),
        "empirical_second_moment_q_a": empirical_second_moment_a,
        "empirical_second_moment_q_b": empirical_second_moment_b,
        "empirical_cross_moment_q_a_q_b": empirical_cross_moment_ab,
        "empirical_second_moment_rho_ab": empirical_second_moment_rho,
        "empirical_centered_variance_q_a": empirical_centered_variance_a,
        "empirical_centered_variance_q_b": empirical_centered_variance_b,
        "empirical_centered_covariance_q_a_q_b": empirical_centered_covariance_ab,
        "empirical_centered_rho_ab": empirical_centered_rho,
        "empirical_second_moment_l_diagonal_a": empirical_diag_a,
        "empirical_second_moment_l_diagonal_b": empirical_diag_b,
        "empirical_second_moment_l_cross": empirical_cross,
        "empirical_second_moment_l": empirical_diag_a + empirical_diag_b + empirical_cross,
        "model_v_a": model_v_a,
        "model_v_b": model_v_b,
        "model_c_ab": model_c_ab,
        "model_rho_ab_pooled": model_c_ab / math.sqrt(model_v_a * model_v_b),
        "model_mean_pointwise_rho_ab": float(np.mean(h_ab / np.sqrt(h_aa * h_bb))),
        "model_var_l_diagonal_a": float(np.mean(var_l_diagonal_a)),
        "model_var_l_diagonal_b": float(np.mean(var_l_diagonal_b)),
        "model_var_l_cross": float(np.mean(var_l_cross)),
        "model_var_l": float(np.mean(var_l)),
        "minimum_contrast_covariance_determinant": float(np.min(determinant)),
    }
    return predicted, summary


def expected_gaussian_score(truth: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Population version of the constant-free bivariate Gaussian score."""

    a = candidate[:, 0, 0]
    b = candidate[:, 0, 1]
    c = candidate[:, 1, 1]
    determinant = a * c - b * b
    inverse = np.empty_like(candidate)
    inverse[:, 0, 0] = c / determinant
    inverse[:, 1, 1] = a / determinant
    inverse[:, 0, 1] = inverse[:, 1, 0] = -b / determinant
    trace = np.einsum("nij,nji->n", inverse, truth, optimize=True)
    return 0.5 * (np.log(determinant) + trace)
