"""Core calculations for the six-scenario two-contrast simulation diagnostic.

The diagnostic is deliberately narrow.  It evaluates the already selected
pair of rectangle contrasts ``Q_A`` and ``Q_B`` and tracks the cross center
``E[Q_A Q_B]``.  Geometry is fixed in standardized Lagrangian coordinates;
only the simulation truth ranges convert it to physical latitude/longitude.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1
SUMMARY_CSV = "daily_two_contrast_centers.csv"
STRATUM_CSV = "daily_two_contrast_strata.csv"


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


def signature_for_files(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        digest.update(resolved.name.encode("utf-8"))
        digest.update(resolved.read_bytes())
    return digest.hexdigest()


def clean_json(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    return value


def atomic_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        clean_json(value), indent=2, sort_keys=True, allow_nan=False
    ) + "\n"
    atomic_text(path, payload)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)


def _uniform_axis(values: np.ndarray, name: str) -> tuple[np.ndarray, float]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    axis = np.sort(np.unique(finite))
    if axis.size < 2:
        raise ValueError(f"{name} axis must contain at least two finite values")
    increments = np.diff(axis)
    step = float(np.median(increments))
    if step <= 0.0 or not np.allclose(increments, step, rtol=0.0, atol=1e-8):
        raise ValueError(f"{name} regular-grid axis is not uniform")
    return axis, step


def _nearest_axis_indices(
    desired: np.ndarray, axis: np.ndarray, step: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.rint((np.asarray(desired, dtype=np.float64) - axis[0]) / step).astype(
        np.int64
    )
    inside = (raw >= 0) & (raw < axis.size)
    clipped = np.clip(raw, 0, axis.size - 1)
    error = np.abs(axis[clipped] - desired)
    inside &= error <= 0.5 * step + 1e-8
    return clipped, error, inside


def build_day_cube(
    frames: Mapping[str, pd.DataFrame],
    day_manifest: pd.DataFrame,
    truth: Mapping[str, Any],
) -> dict[str, Any]:
    """Map one independent eight-hour day to dense regular-grid arrays."""

    required_manifest = {
        "date",
        "block_index",
        "local_time",
        "hour_key",
        "analysis_hours_elapsed",
        "simulation_time_index",
    }
    missing_manifest = required_manifest.difference(day_manifest.columns)
    if missing_manifest:
        raise ValueError(f"day manifest is missing {sorted(missing_manifest)}")
    manifest = day_manifest.sort_values("local_time").reset_index(drop=True)
    expected_slots = int(truth["hours_per_day"])
    if len(manifest) != expected_slots:
        raise ValueError(
            f"day has {len(manifest)} manifest rows; expected {expected_slots}"
        )
    expected_local = np.arange(expected_slots, dtype=np.int64)
    if not np.array_equal(manifest["local_time"].to_numpy(np.int64), expected_local):
        raise ValueError("manifest local_time is not exactly 0,...,7")
    if not np.array_equal(
        manifest["simulation_time_index"].to_numpy(np.int64), expected_local
    ):
        raise ValueError("manifest simulation_time_index is not exactly 0,...,7")
    ordered_keys = manifest["hour_key"].astype(str).tolist()
    absent = [key for key in ordered_keys if key not in frames]
    if absent:
        raise KeyError(f"daily gridded pickle is missing hour keys {absent}")

    first = frames[ordered_keys[0]]
    required_columns = {
        "Latitude",
        "Longitude",
        "ColumnAmountO3",
        "Hours_elapsed",
        "Source_Latitude",
        "Source_Longitude",
        "Simulation_Block",
        "Simulation_Time_Index",
    }
    missing = required_columns.difference(first.columns)
    if missing:
        raise ValueError(f"gridded frame is missing columns {sorted(missing)}")
    latitudes, latitude_step = _uniform_axis(
        pd.to_numeric(first["Latitude"], errors="coerce").to_numpy(np.float64),
        "latitude",
    )
    longitudes, longitude_step = _uniform_axis(
        pd.to_numeric(first["Longitude"], errors="coerce").to_numpy(np.float64),
        "longitude",
    )
    shape = (expected_slots, latitudes.size, longitudes.size)
    residual = np.full(shape, np.nan, dtype=np.float64)
    source_latitude = np.full(shape, np.nan, dtype=np.float64)
    source_longitude = np.full(shape, np.nan, dtype=np.float64)
    # The simulator evaluates the latent field after snapping de-advected
    # source coordinates to this fine FFT lattice.  Reconstruct those exact
    # sampling coordinates so the analytic oracle is evaluated at the points
    # that generated the saved response, rather than at the nearby raw source
    # coordinates.
    simulation_latitude = np.full(shape, np.nan, dtype=np.float64)
    simulation_longitude = np.full(shape, np.nan, dtype=np.float64)
    simulation_mapping_error = np.full(shape, np.nan, dtype=np.float64)
    simulation_grid = truth["simulation_grid"]
    fft_latitude_first = float(simulation_grid["latitude_first"])
    fft_longitude_first = float(simulation_grid["longitude_first"])
    fft_latitude_count = int(simulation_grid["latitude_count"])
    fft_longitude_count = int(simulation_grid["longitude_count"])
    fft_latitude_step = float(truth["delta_latitude"])
    fft_longitude_step = float(truth["delta_longitude"])
    advec_lat = float(truth["advec_lat"])
    advec_lon = float(truth["advec_lon"])
    range_lat = float(truth["range_lat"])
    range_lon = float(truth["range_lon"])
    finite_counts: list[int] = []
    analysis_hours: list[float] = []
    block_index = int(manifest["block_index"].iloc[0])
    if manifest["block_index"].astype(int).nunique() != 1:
        raise ValueError("block_index varies within a day")

    for local_time, key in enumerate(ordered_keys):
        frame = frames[key]
        missing = required_columns.difference(frame.columns)
        if missing:
            raise ValueError(f"{key} is missing columns {sorted(missing)}")
        frame_block = pd.to_numeric(frame["Simulation_Block"], errors="raise")
        frame_time = pd.to_numeric(frame["Simulation_Time_Index"], errors="raise")
        if frame_block.nunique() != 1 or int(frame_block.iloc[0]) != block_index:
            raise ValueError(f"{key} has inconsistent Simulation_Block")
        if frame_time.nunique() != 1 or int(frame_time.iloc[0]) != local_time:
            raise ValueError(f"{key} has inconsistent Simulation_Time_Index")
        hours = pd.to_numeric(frame["Hours_elapsed"], errors="coerce").to_numpy(
            np.float64
        )
        finite_hours = hours[np.isfinite(hours)]
        if finite_hours.size == 0 or not np.allclose(
            finite_hours, finite_hours[0], rtol=0.0, atol=1e-10
        ):
            raise ValueError(f"{key} Hours_elapsed is missing or nonconstant")
        analysis_hour = float(finite_hours[0])
        expected_hour = float(manifest["analysis_hours_elapsed"].iloc[local_time])
        if not math.isclose(analysis_hour, expected_hour, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError(f"{key} Hours_elapsed differs from the manifest")
        analysis_hours.append(analysis_hour)

        regular_lat = pd.to_numeric(frame["Latitude"], errors="coerce").to_numpy(
            np.float64
        )
        regular_lon = pd.to_numeric(frame["Longitude"], errors="coerce").to_numpy(
            np.float64
        )
        row_index = np.rint((regular_lat - latitudes[0]) / latitude_step).astype(
            np.int64
        )
        column_index = np.rint(
            (regular_lon - longitudes[0]) / longitude_step
        ).astype(np.int64)
        grid_valid = (
            np.isfinite(regular_lat)
            & np.isfinite(regular_lon)
            & (row_index >= 0)
            & (row_index < latitudes.size)
            & (column_index >= 0)
            & (column_index < longitudes.size)
        )
        if np.any(grid_valid):
            latitude_axis_error = np.abs(latitudes[row_index[grid_valid]] - regular_lat[grid_valid])
            longitude_axis_error = np.abs(
                longitudes[column_index[grid_valid]] - regular_lon[grid_valid]
            )
            if np.any(latitude_axis_error > 1e-8) or np.any(
                longitude_axis_error > 1e-8
            ):
                raise ValueError(f"{key} regular-grid axes differ from the first hour")
        if np.unique(
            np.column_stack([row_index[grid_valid], column_index[grid_valid]]), axis=0
        ).shape[0] != int(np.count_nonzero(grid_valid)):
            raise ValueError(f"{key} regular-grid cells are not one-to-one")

        response = pd.to_numeric(frame["ColumnAmountO3"], errors="coerce").to_numpy(
            np.float64
        )
        source_lat = pd.to_numeric(
            frame["Source_Latitude"], errors="coerce"
        ).to_numpy(np.float64)
        source_lon = pd.to_numeric(
            frame["Source_Longitude"], errors="coerce"
        ).to_numpy(np.float64)
        observed = (
            grid_valid
            & np.isfinite(response)
            & np.isfinite(source_lat)
            & np.isfinite(source_lon)
        )
        known_mean = float(truth["mean_intercept"]) + float(
            truth["mean_lat_slope"]
        ) * (source_lat[observed] - float(truth["mean_lat_center"]))
        rr = row_index[observed]
        cc = column_index[observed]
        comoving_latitude = source_lat[observed] - advec_lat * local_time
        comoving_longitude = source_lon[observed] - advec_lon * local_time
        fft_latitude_index = np.rint(
            (comoving_latitude - fft_latitude_first) / fft_latitude_step
        ).astype(np.int64)
        fft_longitude_index = np.rint(
            (comoving_longitude - fft_longitude_first) / fft_longitude_step
        ).astype(np.int64)
        if (
            np.any(fft_latitude_index < 0)
            or np.any(fft_latitude_index >= fft_latitude_count)
            or np.any(fft_longitude_index < 0)
            or np.any(fft_longitude_index >= fft_longitude_count)
        ):
            raise ValueError(f"{key} source coordinate lies outside the FFT lattice")
        sampled_comoving_latitude = (
            fft_latitude_first + fft_latitude_index * fft_latitude_step
        )
        sampled_comoving_longitude = (
            fft_longitude_first + fft_longitude_index * fft_longitude_step
        )
        latitude_mapping_error = np.abs(
            sampled_comoving_latitude - comoving_latitude
        )
        longitude_mapping_error = np.abs(
            sampled_comoving_longitude - comoving_longitude
        )
        if np.any(latitude_mapping_error > 0.5 * fft_latitude_step + 1e-10) or np.any(
            longitude_mapping_error > 0.5 * fft_longitude_step + 1e-10
        ):
            raise ValueError(f"{key} nearest FFT mapping exceeds one half-cell")
        residual[local_time, rr, cc] = response[observed] - known_mean
        source_latitude[local_time, rr, cc] = source_lat[observed]
        source_longitude[local_time, rr, cc] = source_lon[observed]
        simulation_latitude[local_time, rr, cc] = (
            sampled_comoving_latitude + advec_lat * local_time
        )
        simulation_longitude[local_time, rr, cc] = (
            sampled_comoving_longitude + advec_lon * local_time
        )
        simulation_mapping_error[local_time, rr, cc] = np.sqrt(
            np.square(latitude_mapping_error / range_lat)
            + np.square(longitude_mapping_error / range_lon)
        )
        finite_counts.append(int(np.count_nonzero(observed)))

    if not np.allclose(np.diff(analysis_hours), 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("analysis Hours_elapsed does not increase by one within the day")
    return {
        "residual": residual,
        "source_latitude": source_latitude,
        "source_longitude": source_longitude,
        "simulation_latitude": simulation_latitude,
        "simulation_longitude": simulation_longitude,
        "simulation_mapping_error": simulation_mapping_error,
        "latitudes": latitudes,
        "longitudes": longitudes,
        "latitude_step": latitude_step,
        "longitude_step": longitude_step,
        "analysis_hours": analysis_hours,
        "local_times": expected_local.astype(np.float64),
        "block_index": block_index,
        "finite_count_min": int(min(finite_counts)),
        "finite_count_max": int(max(finite_counts)),
        "finite_count_mean": float(np.mean(finite_counts)),
    }


def contrast_coefficients(design: Mapping[str, Any]) -> np.ndarray:
    coefficient = np.stack(
        [
            np.asarray(
                design["geometry"]["contrast_coefficients_on_eight_points"][name],
                dtype=np.float64,
            )
            for name in ("Q_A", "Q_B")
        ]
    )
    if coefficient.shape != (2, 8):
        raise ValueError("the two contrast coefficient matrix must have shape 2 x 8")
    if not np.allclose(coefficient.sum(axis=1), 0.0, rtol=0.0, atol=1e-14):
        raise ValueError("both Q_A and Q_B must be zero-sum contrasts")
    return coefficient


def _physical_offsets(
    standardized: Sequence[Sequence[float]], truth: Mapping[str, Any]
) -> np.ndarray:
    values = np.asarray(standardized, dtype=np.float64)
    if values.shape != (4, 2):
        raise ValueError("each mirror geometry must contain four 2D endpoints")
    values = values.copy()
    values[:, 0] *= float(truth["range_lat"])
    values[:, 1] *= float(truth["range_lon"])
    return values


def prepare_flow_following_samples(
    cube: Mapping[str, Any], truth: Mapping[str, Any], design: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Construct all translated Q_A/Q_B samples for one independent day."""

    lag = int(design["geometry"]["temporal_lag"])
    if lag != 1:
        raise ValueError("the selected two-contrast geometry requires temporal lag one")
    coefficient = contrast_coefficients(design)
    latitudes = np.asarray(cube["latitudes"], dtype=np.float64)
    longitudes = np.asarray(cube["longitudes"], dtype=np.float64)
    center_latitude, center_longitude = np.meshgrid(
        latitudes, longitudes, indexing="ij"
    )
    center_latitude = center_latitude.ravel()
    center_longitude = center_longitude.ravel()
    center_row, center_column = np.meshgrid(
        np.arange(latitudes.size, dtype=np.int64),
        np.arange(longitudes.size, dtype=np.int64),
        indexing="ij",
    )
    center_row = center_row.ravel()
    center_column = center_column.ravel()

    batches: list[dict[str, Any]] = []
    coverage: dict[str, Any] = {}
    potential = 0
    advec_lat = float(truth["advec_lat"])
    advec_lon = float(truth["advec_lon"])
    range_lat = float(truth["range_lat"])
    range_lon = float(truth["range_lon"])
    residual = np.asarray(cube["residual"], dtype=np.float64)
    source_lat = np.asarray(cube["source_latitude"], dtype=np.float64)
    source_lon = np.asarray(cube["source_longitude"], dtype=np.float64)
    simulation_lat = np.asarray(cube["simulation_latitude"], dtype=np.float64)
    simulation_lon = np.asarray(cube["simulation_longitude"], dtype=np.float64)
    simulation_mapping_error = np.asarray(
        cube["simulation_mapping_error"], dtype=np.float64
    )
    time_count = residual.shape[0]

    point_specs = (
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
    )
    for handedness, standardized in design["geometry"][
        "standardized_geometry"
    ].items():
        offsets = _physical_offsets(standardized, truth)
        for time_t in range(time_count - lag):
            time_next = time_t + lag
            desired_latitude = np.empty((center_latitude.size, 8), dtype=np.float64)
            desired_longitude = np.empty_like(desired_latitude)
            latitude_index = np.empty(desired_latitude.shape, dtype=np.int64)
            longitude_index = np.empty(desired_latitude.shape, dtype=np.int64)
            endpoint_valid = np.ones(desired_latitude.shape, dtype=bool)
            grid_error = np.empty(desired_latitude.shape, dtype=np.float64)
            for point_index, (time_side, endpoint) in enumerate(point_specs):
                slot = time_t if time_side == 0 else time_next
                desired_latitude[:, point_index] = (
                    center_latitude + offsets[endpoint, 0] + advec_lat * slot
                )
                desired_longitude[:, point_index] = (
                    center_longitude + offsets[endpoint, 1] + advec_lon * slot
                )
                lat_index, lat_error, lat_valid = _nearest_axis_indices(
                    desired_latitude[:, point_index],
                    latitudes,
                    float(cube["latitude_step"]),
                )
                lon_index, lon_error, lon_valid = _nearest_axis_indices(
                    desired_longitude[:, point_index],
                    longitudes,
                    float(cube["longitude_step"]),
                )
                latitude_index[:, point_index] = lat_index
                longitude_index[:, point_index] = lon_index
                endpoint_valid[:, point_index] = lat_valid & lon_valid
                grid_error[:, point_index] = np.sqrt(
                    np.square(lat_error / range_lat)
                    + np.square(lon_error / range_lon)
                )
            geometry_valid = np.all(endpoint_valid, axis=1)
            selected = np.flatnonzero(geometry_valid)
            potential += int(selected.size)
            values = np.empty((selected.size, 8), dtype=np.float64)
            coordinates = np.empty((selected.size, 8, 3), dtype=np.float64)
            source_error = np.empty((selected.size, 8), dtype=np.float64)
            fft_mapping_error = np.empty((selected.size, 8), dtype=np.float64)
            for point_index, (time_side, _) in enumerate(point_specs):
                slot = time_t if time_side == 0 else time_next
                rr = latitude_index[selected, point_index]
                cc = longitude_index[selected, point_index]
                values[:, point_index] = residual[slot, rr, cc]
                coordinates[:, point_index, 0] = simulation_lat[slot, rr, cc]
                coordinates[:, point_index, 1] = simulation_lon[slot, rr, cc]
                coordinates[:, point_index, 2] = float(slot)
                fft_mapping_error[:, point_index] = simulation_mapping_error[
                    slot, rr, cc
                ]
                source_error[:, point_index] = np.sqrt(
                    np.square(
                        (
                            source_lat[slot, rr, cc]
                            - desired_latitude[selected, point_index]
                        )
                        / range_lat
                    )
                    + np.square(
                        (
                            source_lon[slot, rr, cc]
                            - desired_longitude[selected, point_index]
                        )
                        / range_lon
                    )
                )
            complete = np.isfinite(values).all(axis=1) & np.isfinite(coordinates).all(
                axis=(1, 2)
            )
            coverage_key = f"{handedness}_t{time_t:02d}_t{time_next:02d}"
            coverage[coverage_key] = {
                "geometry_candidate_count": int(selected.size),
                "complete_count": int(np.count_nonzero(complete)),
                "complete_fraction": (
                    float(np.mean(complete)) if selected.size else 0.0
                ),
            }
            if not np.any(complete):
                continue
            selected_complete = selected[complete]
            contrast = values[complete] @ coefficient.T
            batches.append(
                {
                    "q_a": contrast[:, 0],
                    "q_b": contrast[:, 1],
                    "coordinates": coordinates[complete],
                    "handedness": np.repeat(handedness, np.count_nonzero(complete)),
                    "time_t": np.repeat(time_t, np.count_nonzero(complete)),
                    "time_t_plus_1": np.repeat(
                        time_next, np.count_nonzero(complete)
                    ),
                    "anchor_row": center_row[selected_complete],
                    "anchor_column": center_column[selected_complete],
                    "grid_error": np.max(grid_error[selected_complete], axis=1),
                    "source_error": np.max(source_error[complete], axis=1),
                    "fft_mapping_error": np.max(
                        fft_mapping_error[complete], axis=1
                    ),
                }
            )

    if not batches:
        raise RuntimeError("no complete flow-following two-contrast samples were found")
    samples = {
        name: np.concatenate([batch[name] for batch in batches], axis=0)
        for name in (
            "q_a",
            "q_b",
            "coordinates",
            "handedness",
            "time_t",
            "time_t_plus_1",
            "anchor_row",
            "anchor_column",
            "grid_error",
            "source_error",
            "fft_mapping_error",
        )
    }
    sample_count = int(samples["q_a"].size)
    audit = {
        "coordinate_frame": design["geometry"]["coordinate_frame"],
        "oracle_coordinate_source": (
            "reconstructed nearest fine FFT comoving lattice plus advection"
        ),
        "sample_count": sample_count,
        "potential_geometry_sample_count": int(potential),
        "complete_fraction_of_geometry_candidates": sample_count / max(potential, 1),
        "coverage_by_mirror_and_time": coverage,
        "regular_grid_shape": [int(latitudes.size), int(longitudes.size)],
        "regular_latitude_step": float(cube["latitude_step"]),
        "regular_longitude_step": float(cube["longitude_step"]),
        "maximum_grid_snap_error_standardized_median": float(
            np.median(samples["grid_error"])
        ),
        "maximum_grid_snap_error_standardized_p95": float(
            np.quantile(samples["grid_error"], 0.95)
        ),
        "maximum_grid_snap_error_standardized_max": float(
            np.max(samples["grid_error"])
        ),
        "maximum_source_endpoint_error_standardized_median": float(
            np.median(samples["source_error"])
        ),
        "maximum_source_endpoint_error_standardized_p95": float(
            np.quantile(samples["source_error"], 0.95)
        ),
        "maximum_source_endpoint_error_standardized_max": float(
            np.max(samples["source_error"])
        ),
        "maximum_fft_lattice_mapping_error_standardized_median": float(
            np.median(samples["fft_mapping_error"])
        ),
        "maximum_fft_lattice_mapping_error_standardized_p95": float(
            np.quantile(samples["fft_mapping_error"], 0.95)
        ),
        "maximum_fft_lattice_mapping_error_standardized_max": float(
            np.max(samples["fft_mapping_error"])
        ),
        "finite_observations_per_hour_min": int(cube["finite_count_min"]),
        "finite_observations_per_hour_max": int(cube["finite_count_max"]),
        "finite_observations_per_hour_mean": float(cube["finite_count_mean"]),
    }
    return samples, audit


def radial_correlation(distance: np.ndarray, truth: Mapping[str, Any]) -> np.ndarray:
    distance = np.asarray(distance, dtype=np.float64)
    family = str(truth["family"])
    if family == "matern":
        if not math.isclose(float(truth["matern_nu"]), 0.5, abs_tol=1e-14):
            raise ValueError("the diagnostic currently supports Matérn nu=0.5")
        return np.exp(-distance)
    if family == "generalized_cauchy":
        alpha = float(truth["cauchy_a"])
        beta = float(truth["cauchy_b"])
        scale = float(
            truth.get(
                "cauchy_efold_scale",
                (math.exp(alpha / beta) - 1.0) ** (1.0 / alpha),
            )
        )
        return np.power(
            1.0 + np.power(scale * distance, alpha), -beta / alpha
        )
    raise ValueError(f"unsupported covariance family {family!r}")


def covariance_components(
    delta: np.ndarray, truth: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Return matched separable and joint endpoint covariance arrays."""

    delta = np.asarray(delta, dtype=np.float64)
    time = delta[..., 2]
    shifted_latitude = delta[..., 0] - float(truth["advec_lat"]) * time
    shifted_longitude = delta[..., 1] - float(truth["advec_lon"]) * time
    spatial_distance = np.sqrt(
        np.square(shifted_latitude / float(truth["range_lat"]))
        + np.square(shifted_longitude / float(truth["range_lon"]))
    )
    temporal_distance = np.abs(time) / float(truth["range_time"])
    spatial = radial_correlation(spatial_distance, truth)
    temporal = radial_correlation(temporal_distance, truth)
    joint = radial_correlation(
        np.sqrt(np.square(spatial_distance) + np.square(temporal_distance)), truth
    )
    variance = float(truth["sigmasq"])
    separable_covariance = variance * spatial * temporal
    joint_covariance = variance * joint
    nugget = float(truth.get("nugget", 0.0))
    if nugget:
        same_observation = np.all(delta == 0.0, axis=-1)
        separable_covariance = separable_covariance + nugget * same_observation
        joint_covariance = joint_covariance + nugget * same_observation
    return separable_covariance, joint_covariance


def pointwise_contrast_covariances(
    coordinates: np.ndarray,
    truth: Mapping[str, Any],
    design: Mapping[str, Any],
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    coefficient = contrast_coefficients(design)
    separable = np.empty((coordinates.shape[0], 2, 2), dtype=np.float64)
    joint = np.empty_like(separable)
    for start in range(0, coordinates.shape[0], int(chunk_size)):
        stop = min(start + int(chunk_size), coordinates.shape[0])
        points = coordinates[start:stop]
        delta = points[:, :, None, :] - points[:, None, :, :]
        covariance_separable, covariance_joint = covariance_components(delta, truth)
        separable[start:stop] = np.einsum(
            "ai,nij,bj->nab",
            coefficient,
            covariance_separable,
            coefficient,
            optimize=True,
        )
        joint[start:stop] = np.einsum(
            "ai,nij,bj->nab",
            coefficient,
            covariance_joint,
            coefficient,
            optimize=True,
        )
    return separable, joint


def _correlation(cross: float, variance_a: float, variance_b: float) -> float | None:
    denominator = math.sqrt(max(variance_a * variance_b, 0.0))
    return cross / denominator if denominator > 0.0 else None


def summarize_center(
    q_a: np.ndarray,
    q_b: np.ndarray,
    h_separable: np.ndarray,
    h_joint: np.ndarray,
    truth_eta: float,
    d1: float,
    d2: float,
) -> dict[str, Any]:
    q_a = np.asarray(q_a, dtype=np.float64)
    q_b = np.asarray(q_b, dtype=np.float64)
    h_truth = (1.0 - float(truth_eta)) * h_separable + float(truth_eta) * h_joint
    empirical_aa = float(np.mean(np.square(q_a)))
    empirical_bb = float(np.mean(np.square(q_b)))
    empirical_ab = float(np.mean(q_a * q_b))
    centered_aa = float(np.mean(np.square(q_a - np.mean(q_a))))
    centered_bb = float(np.mean(np.square(q_b - np.mean(q_b))))
    centered_ab = float(
        np.mean((q_a - np.mean(q_a)) * (q_b - np.mean(q_b)))
    )
    separable_mean = h_separable.mean(axis=0)
    joint_mean = h_joint.mean(axis=0)
    truth_mean = h_truth.mean(axis=0)
    cross_denominator = float(np.sum(h_joint[:, 0, 1] - h_separable[:, 0, 1]))
    raw_cross_numerator = float(np.sum(q_a * q_b - h_separable[:, 0, 1]))
    centered_cross_numerator = float(
        q_a.size * centered_ab - np.sum(h_separable[:, 0, 1])
    )
    eta_hat_raw = (
        raw_cross_numerator / cross_denominator
        if abs(cross_denominator) > 1e-14
        else None
    )
    eta_hat_centered = (
        centered_cross_numerator / cross_denominator
        if abs(cross_denominator) > 1e-14
        else None
    )
    return {
        "sample_count": int(q_a.size),
        "empirical_mean_q_a": float(np.mean(q_a)),
        "empirical_mean_q_b": float(np.mean(q_b)),
        "empirical_second_moment_q_a": empirical_aa,
        "empirical_second_moment_q_b": empirical_bb,
        "empirical_cross_moment_q_a_q_b": empirical_ab,
        "empirical_second_moment_rho_ab": _correlation(
            empirical_ab, empirical_aa, empirical_bb
        ),
        "empirical_centered_variance_q_a": centered_aa,
        "empirical_centered_variance_q_b": centered_bb,
        "empirical_centered_covariance_q_a_q_b": centered_ab,
        "empirical_centered_rho_ab": _correlation(
            centered_ab, centered_aa, centered_bb
        ),
        "separable_h_aa": float(separable_mean[0, 0]),
        "separable_h_ab": float(separable_mean[0, 1]),
        "separable_h_bb": float(separable_mean[1, 1]),
        "separable_rho_ab": _correlation(
            float(separable_mean[0, 1]),
            float(separable_mean[0, 0]),
            float(separable_mean[1, 1]),
        ),
        "joint_h_aa": float(joint_mean[0, 0]),
        "joint_h_ab": float(joint_mean[0, 1]),
        "joint_h_bb": float(joint_mean[1, 1]),
        "joint_rho_ab": _correlation(
            float(joint_mean[0, 1]),
            float(joint_mean[0, 0]),
            float(joint_mean[1, 1]),
        ),
        "truth_h_aa": float(truth_mean[0, 0]),
        "truth_h_ab": float(truth_mean[0, 1]),
        "truth_h_bb": float(truth_mean[1, 1]),
        "truth_rho_ab": _correlation(
            float(truth_mean[0, 1]),
            float(truth_mean[0, 0]),
            float(truth_mean[1, 1]),
        ),
        "cross_center_joint_minus_separable": float(
            joint_mean[0, 1] - separable_mean[0, 1]
        ),
        "empirical_minus_truth_h_ab": empirical_ab - float(truth_mean[0, 1]),
        "absolute_empirical_minus_truth_h_ab": abs(
            empirical_ab - float(truth_mean[0, 1])
        ),
        "centered_cross_sensitivity_minus_raw_truth_h_ab": centered_ab
        - float(truth_mean[0, 1]),
        "eta_hat_from_cross_center": eta_hat_raw,
        "centered_cross_eta_scaled_sensitivity": eta_hat_centered,
        "cross_eta_numerator_sum": raw_cross_numerator,
        "cross_eta_denominator_sum": cross_denominator,
        "empirical_minus_separable_h_ab": empirical_ab
        - float(separable_mean[0, 1]),
        "truth_minus_separable_h_ab": float(
            truth_mean[0, 1] - separable_mean[0, 1]
        ),
        "empirical_l_cross_term": 2.0 * d1 * d2 * empirical_ab,
        "truth_l_cross_term": 2.0 * d1 * d2 * float(truth_mean[0, 1]),
        "separable_l_cross_term": 2.0 * d1 * d2 * float(separable_mean[0, 1]),
        "joint_l_cross_term": 2.0 * d1 * d2 * float(joint_mean[0, 1]),
        "empirical_l_cross_excess_vs_separable": 2.0
        * d1
        * d2
        * (empirical_ab - float(separable_mean[0, 1])),
        "truth_l_cross_excess_vs_separable": 2.0
        * d1
        * d2
        * float(truth_mean[0, 1] - separable_mean[0, 1]),
        "empirical_minus_truth_l_cross_term": 2.0
        * d1
        * d2
        * (empirical_ab - float(truth_mean[0, 1])),
        "empirical_l_diagonal_a": d1 * d1 * empirical_aa,
        "empirical_l_diagonal_b": d2 * d2 * empirical_bb,
        "empirical_l_second_moment": d1 * d1 * empirical_aa
        + d2 * d2 * empirical_bb
        + 2.0 * d1 * d2 * empirical_ab,
        "truth_l_diagonal_a": d1 * d1 * float(truth_mean[0, 0]),
        "truth_l_diagonal_b": d2 * d2 * float(truth_mean[1, 1]),
        "truth_l_second_moment": d1 * d1 * float(truth_mean[0, 0])
        + d2 * d2 * float(truth_mean[1, 1])
        + 2.0 * d1 * d2 * float(truth_mean[0, 1]),
    }


def evaluate_day(
    cube: Mapping[str, Any],
    truth: Mapping[str, Any],
    design: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    samples, audit = prepare_flow_following_samples(cube, truth, design)
    minimum = int(design["diagnostic"]["minimum_complete_samples_per_day"])
    if samples["q_a"].size < minimum:
        raise RuntimeError(
            f"only {samples['q_a'].size} complete samples; minimum is {minimum}"
        )
    separable, joint = pointwise_contrast_covariances(
        samples["coordinates"],
        truth,
        design,
        int(design["diagnostic"]["oracle_covariance_chunk_size"]),
    )
    d1 = float(design["geometry"]["secondary_l_coefficients"]["d1"])
    d2 = float(design["geometry"]["secondary_l_coefficients"]["d2"])
    eta = float(truth["interaction_eta"])
    pooled = summarize_center(
        samples["q_a"], samples["q_b"], separable, joint, eta, d1, d2
    )
    pooled.update(
        {
            "maximum_grid_snap_error_standardized_median": audit[
                "maximum_grid_snap_error_standardized_median"
            ],
            "maximum_grid_snap_error_standardized_p95": audit[
                "maximum_grid_snap_error_standardized_p95"
            ],
            "maximum_grid_snap_error_standardized_max": audit[
                "maximum_grid_snap_error_standardized_max"
            ],
            "maximum_source_endpoint_error_standardized_median": audit[
                "maximum_source_endpoint_error_standardized_median"
            ],
            "maximum_source_endpoint_error_standardized_p95": audit[
                "maximum_source_endpoint_error_standardized_p95"
            ],
            "maximum_source_endpoint_error_standardized_max": audit[
                "maximum_source_endpoint_error_standardized_max"
            ],
            "maximum_fft_lattice_mapping_error_standardized_median": audit[
                "maximum_fft_lattice_mapping_error_standardized_median"
            ],
            "maximum_fft_lattice_mapping_error_standardized_p95": audit[
                "maximum_fft_lattice_mapping_error_standardized_p95"
            ],
            "maximum_fft_lattice_mapping_error_standardized_max": audit[
                "maximum_fft_lattice_mapping_error_standardized_max"
            ],
            "embedding_relative_covariance_distortion_bound": float(
                truth["embedding_diagnostics"][
                    "relative_covariance_distortion_bound"
                ]
            ),
            "potential_geometry_sample_count": audit[
                "potential_geometry_sample_count"
            ],
            "complete_fraction_of_geometry_candidates": audit[
                "complete_fraction_of_geometry_candidates"
            ],
        }
    )
    strata: list[dict[str, Any]] = []
    for handedness in design["geometry"]["standardized_geometry"]:
        for time_t in range(int(truth["hours_per_day"]) - 1):
            selected = (samples["handedness"] == handedness) & (
                samples["time_t"].astype(np.int64) == time_t
            )
            if not np.any(selected):
                continue
            row = summarize_center(
                samples["q_a"][selected],
                samples["q_b"][selected],
                separable[selected],
                joint[selected],
                eta,
                d1,
                d2,
            )
            row.update(
                {
                    "handedness": handedness,
                    "time_t": int(time_t),
                    "time_t_plus_1": int(time_t + 1),
                    "grid_snap_error_standardized_median": float(
                        np.median(samples["grid_error"][selected])
                    ),
                    "source_endpoint_error_standardized_median": float(
                        np.median(samples["source_error"][selected])
                    ),
                    "fft_lattice_mapping_error_standardized_median": float(
                        np.median(samples["fft_mapping_error"][selected])
                    ),
                }
            )
            strata.append(row)
    return pooled, strata, audit


def checkpoint_compatible(
    record: Mapping[str, Any], expected: Mapping[str, Any]
) -> bool:
    if record.get("status") != "complete" or int(
        record.get("schema_version", -1)
    ) != SCHEMA_VERSION:
        return False
    return all(record.get(name) == value for name, value in expected.items())


def rebuild_scenario_outputs(
    scenario_dir: Path,
    scenario_id: str,
    selected_dates: Sequence[str],
    diagnostic_signature: str,
    truth_sha256: str,
    source_run_request_sha256: str,
) -> dict[str, Any]:
    summary_rows: list[dict[str, Any]] = []
    stratum_rows: list[dict[str, Any]] = []
    day_root = scenario_dir / "day_json"
    for date in selected_dates:
        path = day_root / f"{date}.json"
        if not path.is_file():
            continue
        record = load_json(path)
        required = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "diagnostic_signature": diagnostic_signature,
            "truth_sha256": truth_sha256,
            "source_run_request_sha256": source_run_request_sha256,
            "scenario_id": scenario_id,
            "date": date,
        }
        mismatch = {
            name: (record.get(name), value)
            for name, value in required.items()
            if record.get(name) != value
        }
        if mismatch:
            raise RuntimeError(f"incompatible checkpoint {path}: {mismatch}")
        row_required = {
            name: value
            for name, value in required.items()
            if name not in {"schema_version", "status"}
        }
        summary_row = dict(record["summary"])
        summary_mismatch = {
            name: (summary_row.get(name), value)
            for name, value in row_required.items()
            if summary_row.get(name) != value
        }
        if summary_mismatch:
            raise RuntimeError(
                f"incompatible summary inside checkpoint {path}: {summary_mismatch}"
            )
        summary_rows.append(summary_row)
        for raw_row in record.get("strata", []):
            row = dict(raw_row)
            row_mismatch = {
                name: (row.get(name), value)
                for name, value in row_required.items()
                if row.get(name) != value
            }
            if row_mismatch:
                raise RuntimeError(
                    f"incompatible stratum inside checkpoint {path}: {row_mismatch}"
                )
            stratum_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary = summary.sort_values(["block_index", "date"]).reset_index(drop=True)
    strata = pd.DataFrame(stratum_rows)
    if not strata.empty:
        strata = strata.sort_values(
            ["block_index", "date", "handedness", "time_t"]
        ).reset_index(drop=True)
    atomic_csv(scenario_dir / SUMMARY_CSV, summary)
    atomic_csv(scenario_dir / STRATUM_CSV, strata)
    progress = {
        "schema_version": SCHEMA_VERSION,
        "diagnostic_signature": diagnostic_signature,
        "truth_sha256": truth_sha256,
        "source_run_request_sha256": source_run_request_sha256,
        "configured_days": int(len(selected_dates)),
        "completed_days": int(len(summary)),
        "remaining_days": int(len(selected_dates) - len(summary)),
        "completed_dates": summary["date"].astype(str).tolist()
        if not summary.empty
        else [],
        "incompatible_checkpoint_dates": [],
    }
    atomic_json(scenario_dir / "progress.json", progress)
    if int(progress["remaining_days"]) != 0:
        (scenario_dir / "COMPLETE.json").unlink(missing_ok=True)
    return progress


def _summary_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    metrics = (
        "empirical_cross_moment_q_a_q_b",
        "truth_h_ab",
        "empirical_minus_truth_h_ab",
        "empirical_l_cross_term",
        "truth_l_cross_term",
        "empirical_minus_truth_l_cross_term",
        "empirical_l_cross_excess_vs_separable",
        "truth_l_cross_excess_vs_separable",
        "eta_hat_from_cross_center",
        "empirical_second_moment_q_a",
        "truth_h_aa",
        "empirical_second_moment_q_b",
        "truth_h_bb",
    )
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(
        ["scenario_id", "family", "interaction_eta"], sort=False
    ):
        scenario_id, family, eta = keys
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(
                np.float64
            )
            if values.size == 0:
                continue
            sd = float(np.std(values, ddof=1)) if values.size > 1 else None
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "family": family,
                    "interaction_eta": float(eta),
                    "metric": metric,
                    "n_days": int(values.size),
                    "mean": float(np.mean(values)),
                    "sd_across_independent_days": sd,
                    "se_across_independent_days": (
                        sd / math.sqrt(values.size) if sd is not None else None
                    ),
                    "median": float(np.median(values)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )
        numerator = pd.to_numeric(
            group["cross_eta_numerator_sum"], errors="coerce"
        ).sum(min_count=1)
        denominator = pd.to_numeric(
            group["cross_eta_denominator_sum"], errors="coerce"
        ).sum(min_count=1)
        ratio = (
            float(numerator / denominator)
            if pd.notna(numerator)
            and pd.notna(denominator)
            and abs(float(denominator)) > 1e-14
            else None
        )
        if ratio is not None:
            rows.append(
                {
                    "scenario_id": scenario_id,
                    "family": family,
                    "interaction_eta": float(eta),
                    "metric": "eta_hat_ratio_of_cross_sums",
                    "n_days": int(len(group)),
                    "mean": ratio,
                    "sd_across_independent_days": None,
                    "se_across_independent_days": None,
                    "median": ratio,
                    "minimum": ratio,
                    "maximum": ratio,
                }
            )
    return pd.DataFrame(rows)


def _paired_eta_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    metrics = (
        "empirical_cross_moment_q_a_q_b",
        "truth_h_ab",
        "empirical_l_cross_term",
        "truth_l_cross_term",
        "empirical_l_cross_excess_vs_separable",
        "truth_l_cross_excess_vs_separable",
        "eta_hat_from_cross_center",
    )
    rows: list[dict[str, Any]] = []
    for (family, date), group in frame.groupby(["family", "date"], sort=False):
        by_eta = {float(row.interaction_eta): row for row in group.itertuples(index=False)}
        if set(by_eta) != {0.0, 0.5, 1.0}:
            continue
        if len(group) != 3:
            raise ValueError(f"{family} {date} does not contain exactly three eta rows")
        for column in ("block_index", "seed", "sample_count"):
            values = set(group[column].tolist())
            if len(values) != 1:
                raise ValueError(
                    f"paired eta rows disagree on {column} for {family} {date}: {values}"
                )
        base = {
            "family": family,
            "date": str(date),
            "block_index": int(by_eta[0.0].block_index),
            "seed": int(by_eta[0.0].seed),
            "sample_count": int(by_eta[0.0].sample_count),
        }
        for metric in metrics:
            value0 = float(getattr(by_eta[0.0], metric))
            value05 = float(getattr(by_eta[0.5], metric))
            value1 = float(getattr(by_eta[1.0], metric))
            base[f"{metric}_eta0"] = value0
            base[f"{metric}_eta0p5"] = value05
            base[f"{metric}_eta1"] = value1
            base[f"{metric}_eta0p5_minus_eta0"] = value05 - value0
            base[f"{metric}_eta1_minus_eta0"] = value1 - value0
            base[f"{metric}_linearity_curvature"] = value1 - 2.0 * value05 + value0
        rows.append(base)
    return pd.DataFrame(rows)


def rebuild_master_outputs(
    output_root: Path,
    scenario_ids: Sequence[str],
    selected_dates: Sequence[str],
    diagnostic_signature: str,
    input_identities: Mapping[str, Mapping[str, str]],
) -> dict[str, Any]:
    """Atomically rebuild shared CSVs; safe when six scenario jobs run together."""

    if set(input_identities) != set(scenario_ids):
        raise ValueError("input identities do not cover the configured scenarios")
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".master_csv.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        summary_frames: list[pd.DataFrame] = []
        stratum_frames: list[pd.DataFrame] = []
        for scenario_id in scenario_ids:
            scenario_dir = output_root / "scenarios" / scenario_id
            summary_path = scenario_dir / SUMMARY_CSV
            stratum_path = scenario_dir / STRATUM_CSV
            if summary_path.is_file():
                try:
                    frame = pd.read_csv(summary_path)
                except pd.errors.EmptyDataError:
                    frame = pd.DataFrame()
                if not frame.empty:
                    summary_frames.append(frame)
            if stratum_path.is_file():
                try:
                    frame = pd.read_csv(stratum_path)
                except pd.errors.EmptyDataError:
                    frame = pd.DataFrame()
                if not frame.empty:
                    stratum_frames.append(frame)
        all_summary = (
            pd.concat(summary_frames, ignore_index=True)
            if summary_frames
            else pd.DataFrame()
        )
        if not all_summary.empty:
            if "diagnostic_signature" not in all_summary:
                raise ValueError("master rows do not contain diagnostic_signature")
            signatures = set(all_summary["diagnostic_signature"].astype(str))
            if signatures != {diagnostic_signature}:
                raise ValueError(f"master rows mix diagnostic signatures: {signatures}")
            duplicates = all_summary.duplicated(["scenario_id", "date"], keep=False)
            if duplicates.any():
                raise ValueError("duplicate (scenario_id,date) rows in master inputs")
            unknown_scenarios = set(all_summary["scenario_id"].astype(str)).difference(
                scenario_ids
            )
            if unknown_scenarios:
                raise ValueError(f"unknown scenarios in master inputs: {unknown_scenarios}")
            unknown_dates = set(all_summary["date"].astype(str)).difference(selected_dates)
            if unknown_dates:
                raise ValueError(f"unknown dates in master inputs: {unknown_dates}")
            for scenario_id, expected_identity in input_identities.items():
                rows = all_summary[
                    all_summary["scenario_id"].astype(str) == scenario_id
                ]
                if rows.empty:
                    continue
                for column in ("truth_sha256", "source_run_request_sha256"):
                    if column not in rows:
                        raise ValueError(f"master rows do not contain {column}")
                    observed = set(rows[column].astype(str))
                    expected = {str(expected_identity[column])}
                    if observed != expected:
                        raise ValueError(
                            f"{scenario_id} master rows have stale {column}: "
                            f"observed={observed}, expected={expected}"
                        )
            order = {value: index for index, value in enumerate(scenario_ids)}
            all_summary["_scenario_order"] = all_summary["scenario_id"].map(order)
            all_summary = all_summary.sort_values(
                ["_scenario_order", "block_index", "date"]
            ).drop(columns="_scenario_order")
        all_strata = (
            pd.concat(stratum_frames, ignore_index=True)
            if stratum_frames
            else pd.DataFrame()
        )
        if not all_strata.empty:
            order = {value: index for index, value in enumerate(scenario_ids)}
            all_strata["_scenario_order"] = all_strata["scenario_id"].map(order)
            all_strata = all_strata.sort_values(
                ["_scenario_order", "block_index", "date", "handedness", "time_t"]
            ).drop(columns="_scenario_order")
        atomic_csv(output_root / "all_daily_two_contrast_centers.csv", all_summary)
        atomic_csv(output_root / "all_daily_two_contrast_strata.csv", all_strata)
        scenario_summary = _summary_statistics(all_summary)
        paired = _paired_eta_rows(all_summary)
        atomic_csv(output_root / "scenario_day_summary.csv", scenario_summary)
        atomic_csv(output_root / "paired_eta_effects_by_day.csv", paired)
        expected_pairs = {
            (scenario_id, str(date))
            for scenario_id in scenario_ids
            for date in selected_dates
        }
        observed_pairs = (
            set(
                zip(
                    all_summary["scenario_id"].astype(str),
                    all_summary["date"].astype(str),
                )
            )
            if not all_summary.empty
            else set()
        )
        complete_scenarios: list[str] = []
        for scenario_id in scenario_ids:
            scenario_dir = output_root / "scenarios" / scenario_id
            marker_path = scenario_dir / "COMPLETE.json"
            summary_path = scenario_dir / SUMMARY_CSV
            stratum_path = scenario_dir / STRATUM_CSV
            if (
                not marker_path.is_file()
                or not summary_path.is_file()
                or not stratum_path.is_file()
            ):
                continue
            marker = load_json(marker_path)
            expected_identity = input_identities[scenario_id]
            marker_ok = (
                marker.get("status") == "complete"
                and marker.get("diagnostic_signature") == diagnostic_signature
                and marker.get("scenario_id") == scenario_id
                and marker.get("truth_sha256")
                == expected_identity["truth_sha256"]
                and marker.get("source_run_request_sha256")
                == expected_identity["source_run_request_sha256"]
                and int(marker.get("completed_days", -1)) == len(selected_dates)
                and marker.get("daily_csv_sha256") == sha256_file(summary_path)
                and marker.get("stratum_csv_sha256") == sha256_file(stratum_path)
            )
            if marker_ok:
                complete_scenarios.append(scenario_id)
        complete = observed_pairs == expected_pairs and set(complete_scenarios) == set(
            scenario_ids
        )
        progress = {
            "schema_version": SCHEMA_VERSION,
            "diagnostic_signature": diagnostic_signature,
            "scenario_count": int(all_summary["scenario_id"].nunique())
            if not all_summary.empty
            else 0,
            "completed_scenario_days": int(len(all_summary)),
            "expected_scenario_days": int(len(selected_dates) * len(scenario_ids)),
            "complete_scenarios": complete_scenarios,
            "expected_complete_scenarios": int(len(scenario_ids)),
            "complete": bool(complete),
        }
        atomic_json(output_root / "master_progress.json", progress)
        final_path = output_root / "FINAL_COMPLETE.json"
        if complete:
            atomic_json(
                final_path,
                {
                    **progress,
                    "master_csv_sha256": sha256_file(
                        output_root / "all_daily_two_contrast_centers.csv"
                    ),
                },
            )
        else:
            final_path.unlink(missing_ok=True)
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    return progress
