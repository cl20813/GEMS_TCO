#!/usr/bin/env python3
"""Generate one local GEMS day from an advected generalized-Cauchy field.

The historical July generator evaluates the advected covariance directly on
an even BCCB lattice.  Its Nyquist faces are not centrosymmetric, and dropping
the resulting imaginary spectrum before clipping can materially change the
small cross term used by the interaction diagnostic.  This local generator
uses the equivalent comoving construction instead:

1. generate a zero-advection space-time GC field W(x, t) by FFT;
2. observe Z(s, t) = W(s - v * (t - t0), t);
3. retain the template's original source coordinates and Hours_elapsed.

The intended analytic GC covariance and the effective post-clipping FFT
covariance are both recoverable.  The latter is the exact population truth for
the generated lattice field and should be the primary target in fit audits.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent

import generate_july_st_circulant_real_locations_2022_2025_gencauchy_a0p75_b1p0_nugget0_060626 as _reference_generator  # noqa: E402


DELTA_LAT_BASE = _reference_generator.DELTA_LAT_BASE
DELTA_LON_BASE = _reference_generator.DELTA_LON_BASE
griddify_one_to_one = _reference_generator.griddify_one_to_one
parse_gems_hour_key = _reference_generator.parse_gems_hour_key


@dataclass(frozen=True)
class GeneratedPaths:
    real_locations: Path
    gridded: Path
    manifest: Path
    truth: Path


@dataclass
class EmbeddingPlan:
    latitude_axis: np.ndarray
    longitude_axis: np.ndarray
    delta_latitude: float
    delta_longitude: float
    spectrum: np.ndarray
    effective_covariance: np.ndarray
    diagnostics: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pd.to_pickle(value, temporary)
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)


def _input_path(input_root: Path, year: int) -> Path:
    return input_root / f"pickle_{year}" / f"tco_grid_{str(year)[2:]}_07.pkl"


def _output_paths(output_dir: Path, date: str) -> GeneratedPaths:
    prefix = f"gc_fft_{date}"
    return GeneratedPaths(
        real_locations=output_dir / f"{prefix}_real_locations.pkl",
        gridded=output_dir / f"{prefix}_gridded.pkl",
        manifest=output_dir / f"{prefix}_manifest.csv",
        truth=output_dir / f"{prefix}_truth.json",
    )


def _select_day(template: Mapping[str, pd.DataFrame], date: str) -> dict[str, pd.DataFrame]:
    target = pd.Timestamp(date).date()
    selected: list[tuple[pd.Timestamp, str, pd.DataFrame]] = []
    for key, frame in template.items():
        timestamp = parse_gems_hour_key(str(key))
        if timestamp is not None and timestamp.date() == target:
            selected.append((timestamp, str(key), frame.reset_index(drop=True)))
    selected.sort(key=lambda item: item[0])
    if len(selected) != 8:
        raise ValueError(f"{date} has {len(selected)} template hours; expected exactly 8")

    hours = np.asarray(
        [
            float(
                np.nanmedian(
                    pd.to_numeric(frame["Hours_elapsed"], errors="coerce").to_numpy(
                        dtype=np.float64
                    )
                )
            )
            for _, _, frame in selected
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(hours)) or not np.array_equal(
        np.diff(np.rint(hours)).astype(np.int64), np.ones(7, dtype=np.int64)
    ):
        raise ValueError(f"{date} template Hours_elapsed are not eight adjacent hours: {hours}")
    return {key: frame for _, key, frame in selected}


def _axis_from_values(values: np.ndarray, step: float, pad: float) -> np.ndarray:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("cannot build a simulation axis from no finite coordinates")
    start = math.floor((float(finite.min()) - float(pad)) / step) * step
    stop = math.ceil((float(finite.max()) + float(pad)) / step) * step
    count = int(round((stop - start) / step)) + 1
    return start + np.arange(count, dtype=np.float64) * step


def _signed_periodic_lags(size: int, step: float) -> np.ndarray:
    index = np.arange(int(size), dtype=np.float64)
    signed = np.where(index <= int(size) // 2, index, index - int(size))
    return signed * float(step)


def _build_embedding_from_axes(
    latitude_axis: np.ndarray,
    longitude_axis: np.ndarray,
    delta_latitude: float,
    delta_longitude: float,
    time_steps: int,
    physical: Mapping[str, float],
    gc_alpha: float,
    gc_beta: float,
    embedding_spatial_factor: int,
    embedding_temporal_factor: int,
    max_negative_spectral_mass: float,
) -> EmbeddingPlan:
    required_positive = (
        "signal_variance",
        "range_lat",
        "range_lon",
        "range_time",
    )
    for name in required_positive:
        value = float(physical[name])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value}")
    nugget = float(physical.get("nugget", 0.0))
    if not math.isfinite(nugget) or nugget < 0.0:
        raise ValueError(f"nugget must be finite and nonnegative, got {nugget}")
    if not math.isfinite(float(gc_alpha)) or not 0.0 < float(gc_alpha) <= 2.0:
        raise ValueError("gc_alpha must be finite and in (0, 2]")
    if not math.isfinite(float(gc_beta)) or float(gc_beta) <= 0.0:
        raise ValueError("gc_beta must be finite and positive")
    if int(time_steps) < 2:
        raise ValueError("time_steps must be at least 2")
    if not math.isfinite(float(max_negative_spectral_mass)) or not (
        0.0 <= float(max_negative_spectral_mass) < 1.0
    ):
        raise ValueError("max_negative_spectral_mass must be finite and in [0, 1)")
    if embedding_spatial_factor < 2 or embedding_temporal_factor < 2:
        raise ValueError("embedding factors must both be at least 2")
    # ``factor * (n - 1) + 1`` gives the standard odd 2n-1 embedding when
    # factor=2.  Odd sizes avoid self-reflecting Nyquist faces entirely.
    shape = (
        int(embedding_spatial_factor) * (int(latitude_axis.size) - 1) + 1,
        int(embedding_spatial_factor) * (int(longitude_axis.size) - 1) + 1,
        int(embedding_temporal_factor) * (int(time_steps) - 1) + 1,
    )
    latitude_lag = _signed_periodic_lags(shape[0], delta_latitude)[:, None, None]
    longitude_lag = _signed_periodic_lags(shape[1], delta_longitude)[None, :, None]
    time_lag = _signed_periodic_lags(shape[2], 1.0)[None, None, :]
    distance = np.sqrt(
        np.square(latitude_lag / float(physical["range_lat"]))
        + np.square(longitude_lag / float(physical["range_lon"]))
        + np.square(time_lag / float(physical["range_time"]))
    )
    intended_covariance = float(physical["signal_variance"]) * np.power(
        1.0 + np.power(distance, float(gc_alpha)),
        -float(gc_beta) / float(gc_alpha),
    )
    inverse_latitude = (-np.arange(shape[0], dtype=np.int64)) % shape[0]
    inverse_longitude = (-np.arange(shape[1], dtype=np.int64)) % shape[1]
    inverse_time = (-np.arange(shape[2], dtype=np.int64)) % shape[2]
    central_symmetry_error = 0.0
    for time_index, inverse_time_index in enumerate(inverse_time):
        reflected = intended_covariance[:, :, inverse_time_index][
            np.ix_(inverse_latitude, inverse_longitude)
        ]
        central_symmetry_error = max(
            central_symmetry_error,
            float(np.max(np.abs(intended_covariance[:, :, time_index] - reflected))),
        )
    if central_symmetry_error > 1.0e-10:
        raise ArithmeticError(
            "comoving covariance is not centrally symmetric: "
            f"max error {central_symmetry_error:.6g}"
        )
    raw_spectrum = np.fft.rfftn(intended_covariance)
    del distance, intended_covariance
    gc.collect()
    real_spectrum = raw_spectrum.real
    # This mask exactly matches the subsequent max(spectrum, 0) operation so
    # diagnostics and the generator label describe every clipped coefficient.
    negative = real_spectrum < 0.0
    frequency_weights = np.full(real_spectrum.shape[-1], 2.0, dtype=np.float64)
    frequency_weights[0] = 1.0
    if shape[-1] % 2 == 0:
        frequency_weights[-1] = 1.0
    weights = frequency_weights.reshape((1, 1, -1))
    absolute_mass = float(np.sum(np.abs(real_spectrum) * weights))
    negative_mass = float(np.sum(np.abs(real_spectrum) * negative * weights))
    negative_mass_fraction = negative_mass / absolute_mass if absolute_mass else 0.0
    max_abs_real = float(np.max(np.abs(real_spectrum)))
    spectrum_min_before_clipping = float(np.min(real_spectrum))
    max_abs_imaginary = float(np.max(np.abs(raw_spectrum.imag)))
    imaginary_to_real = max_abs_imaginary / max_abs_real if max_abs_real else 0.0
    if imaginary_to_real > 1.0e-10:
        raise ArithmeticError(
            "zero-advection embedding is not numerically real: "
            f"max |imag| / max |real| = {imaginary_to_real:.6g}"
        )
    if negative_mass_fraction > float(max_negative_spectral_mass):
        raise ArithmeticError(
            "FFT embedding requires excessive eigenvalue clipping: "
            f"negative mass fraction {negative_mass_fraction:.6g} > "
            f"{float(max_negative_spectral_mass):.6g}"
        )

    spectrum = np.maximum(real_spectrum, 0.0)
    effective_covariance = np.fft.irfftn(
        spectrum, s=shape, axes=(0, 1, 2)
    ).real
    variance_before = float(effective_covariance[0, 0, 0])
    if not variance_before > 0.0:
        raise ArithmeticError("clipped FFT spectrum has non-positive variance")
    variance_scale = float(physical["signal_variance"]) / variance_before
    spectrum *= variance_scale
    effective_covariance *= variance_scale
    del raw_spectrum, real_spectrum
    gc.collect()
    weighted_negative_count = float(np.sum(negative * weights))
    diagnostics = {
        "construction": "zero-advection comoving FFT field sampled at s-v*(t-t0)",
        "simulation_grid_shape": [
            int(latitude_axis.size),
            int(longitude_axis.size),
            int(time_steps),
        ],
        "embedding_shape": [int(value) for value in shape],
        "embedding_cells": int(np.prod(shape)),
        "spectrum_min_before_clipping": spectrum_min_before_clipping,
        "spectrum_negative_count": int(weighted_negative_count),
        "spectrum_negative_fraction": float(weighted_negative_count / np.prod(shape)),
        "spectrum_negative_mass_fraction": negative_mass_fraction,
        "spectrum_max_abs_imaginary": max_abs_imaginary,
        "spectrum_max_abs_imaginary_over_max_abs_real": imaginary_to_real,
        "central_symmetry_max_error": central_symmetry_error,
        "variance_before_renormalization": variance_before,
        "variance_renormalization_scale": variance_scale,
        "variance_after_renormalization": float(effective_covariance[0, 0, 0]),
        "generator_label": (
            "exact_circulant_embedding"
            if weighted_negative_count == 0.0
            else "spectrally_corrected_circulant_embedding"
        ),
    }
    del negative
    gc.collect()
    return EmbeddingPlan(
        latitude_axis=np.asarray(latitude_axis, dtype=np.float64),
        longitude_axis=np.asarray(longitude_axis, dtype=np.float64),
        delta_latitude=float(delta_latitude),
        delta_longitude=float(delta_longitude),
        spectrum=spectrum,
        effective_covariance=effective_covariance,
        diagnostics=diagnostics,
    )


def _build_plan_for_frames(
    frames: Mapping[str, pd.DataFrame],
    physical: Mapping[str, float],
    gc_alpha: float,
    gc_beta: float,
    lat_factor_hr: int,
    lon_factor_hr: int,
    pad: float,
    embedding_spatial_factor: int,
    embedding_temporal_factor: int,
    max_negative_spectral_mass: float,
) -> EmbeddingPlan:
    if lat_factor_hr < 1 or lon_factor_hr < 1:
        raise ValueError("high-resolution factors must be positive integers")
    if not math.isfinite(float(pad)) or float(pad) < 0.0:
        raise ValueError("pad must be finite and nonnegative")
    delta_latitude = DELTA_LAT_BASE / float(lat_factor_hr)
    delta_longitude = DELTA_LON_BASE / float(lon_factor_hr)
    deadvected_latitudes: list[np.ndarray] = []
    deadvected_longitudes: list[np.ndarray] = []
    for local_time, frame in enumerate(frames.values()):
        latitude = pd.to_numeric(frame["Source_Latitude"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        longitude = pd.to_numeric(frame["Source_Longitude"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        deadvected_latitudes.append(
            latitude - float(physical["advec_lat"]) * float(local_time)
        )
        deadvected_longitudes.append(
            longitude - float(physical["advec_lon"]) * float(local_time)
        )
    latitude_axis = _axis_from_values(
        np.concatenate(deadvected_latitudes), delta_latitude, pad
    )
    longitude_axis = _axis_from_values(
        np.concatenate(deadvected_longitudes), delta_longitude, pad
    )
    return _build_embedding_from_axes(
        latitude_axis,
        longitude_axis,
        delta_latitude,
        delta_longitude,
        len(frames),
        physical,
        gc_alpha,
        gc_beta,
        embedding_spatial_factor,
        embedding_temporal_factor,
        max_negative_spectral_mass,
    )


def _nearest_indices(
    values: np.ndarray, axis: np.ndarray, step: float
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.rint((np.asarray(values, dtype=np.float64) - float(axis[0])) / step).astype(
        np.int64
    )
    if np.any(raw < 0) or np.any(raw >= axis.size):
        raise ValueError("a de-advected source coordinate falls outside the FFT grid")
    error = np.abs(axis[raw] - values)
    if np.any(error > 0.5 * step + 1.0e-10):
        raise ValueError("nearest FFT grid mapping exceeds one half-cell")
    return raw, error


def truth_physical(truth: Mapping[str, Any]) -> dict[str, float]:
    if str(truth.get("correlation_model")) != "generalized_cauchy":
        raise ValueError("truth metadata is not generalized Cauchy")
    return {
        "signal_variance": float(truth["sigmasq"]),
        "range_lat": float(truth["range_lat"]),
        "range_lon": float(truth["range_lon"]),
        "range_time": float(truth["range_time"]),
        "advec_lat": float(truth["advec_lat"]),
        "advec_lon": float(truth["advec_lon"]),
        "nugget": float(truth["nugget"]),
    }


def rebuild_embedding(truth: Mapping[str, Any]) -> EmbeddingPlan:
    physical = truth_physical(truth)
    grid = truth["simulation_grid"]
    latitude_axis = float(grid["latitude_first"]) + np.arange(
        int(grid["latitude_count"]), dtype=np.float64
    ) * float(grid["delta_latitude"])
    longitude_axis = float(grid["longitude_first"]) + np.arange(
        int(grid["longitude_count"]), dtype=np.float64
    ) * float(grid["delta_longitude"])
    return _build_embedding_from_axes(
        latitude_axis,
        longitude_axis,
        float(grid["delta_latitude"]),
        float(grid["delta_longitude"]),
        int(truth["hours_per_day"]),
        physical,
        float(truth["cauchy_a"]),
        float(truth["cauchy_b"]),
        int(truth["embedding_spatial_factor"]),
        int(truth["embedding_temporal_factor"]),
        float(truth["max_negative_spectral_mass"]),
    )


def embedding_contrast_covariances(
    sample_coordinates: np.ndarray,
    truth: Mapping[str, Any],
    frozen_design: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    """Return exact post-clipping FFT truth for every frozen A/B sample."""

    plan = rebuild_embedding(truth)
    physical = truth_physical(truth)
    coordinates = np.asarray(sample_coordinates, dtype=np.float64)
    first_model_time = float(
        np.rint(
            float(truth["hours_elapsed"][0])
            - float(frozen_design["model_time_origin_hours"])
        )
    )
    time = np.rint(coordinates[..., 2] - first_model_time).astype(np.int64)
    if np.any(time < 0) or np.any(time >= int(truth["hours_per_day"])):
        raise ValueError("contrast coordinates do not map to the simulated local time axis")
    deadvected_latitude = coordinates[..., 0] - float(physical["advec_lat"]) * time
    deadvected_longitude = coordinates[..., 1] - float(physical["advec_lon"]) * time
    latitude_index, latitude_error = _nearest_indices(
        deadvected_latitude, plan.latitude_axis, plan.delta_latitude
    )
    longitude_index, longitude_error = _nearest_indices(
        deadvected_longitude, plan.longitude_axis, plan.delta_longitude
    )

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
    result = np.zeros((coordinates.shape[0], 2, 2), dtype=np.float64)
    shape = plan.effective_covariance.shape
    for left_contrast in range(2):
        for right_contrast in range(2):
            value = np.zeros(coordinates.shape[0], dtype=np.float64)
            for left_point in range(8):
                left_weight = coefficient[left_contrast, left_point]
                if left_weight == 0.0:
                    continue
                for right_point in range(8):
                    weight = left_weight * coefficient[right_contrast, right_point]
                    if weight == 0.0:
                        continue
                    value += weight * plan.effective_covariance[
                        (latitude_index[:, left_point] - latitude_index[:, right_point])
                        % shape[0],
                        (longitude_index[:, left_point] - longitude_index[:, right_point])
                        % shape[1],
                        (time[:, left_point] - time[:, right_point]) % shape[2],
                    ]
                    nugget = float(physical.get("nugget", 0.0))
                    if nugget > 0.0:
                        same_observation = np.all(
                            coordinates[:, left_point, :]
                            == coordinates[:, right_point, :],
                            axis=1,
                        )
                        value += weight * nugget * same_observation
            result[:, left_contrast, right_contrast] = value
    mapping = {
        "fft_mapping_max_abs_latitude_error": float(np.max(latitude_error)),
        "fft_mapping_max_abs_longitude_error": float(np.max(longitude_error)),
    }
    return result, mapping


def generate_day(
    *,
    date: str,
    input_root: Path,
    output_dir: Path,
    seed: int,
    physical: Mapping[str, float],
    gc_alpha: float,
    gc_beta: float,
    mean_intercept: float,
    mean_lat_slope: float,
    mean_lat_center: float,
    lat_factor_hr: int,
    lon_factor_hr: int,
    pad: float,
    embedding_spatial_factor: int,
    embedding_temporal_factor: int,
    max_negative_spectral_mass: float,
    overwrite: bool = False,
) -> GeneratedPaths:
    timestamp = pd.Timestamp(date)
    if timestamp.month != 7:
        raise ValueError("the current local GEMS template contract expects a July date")
    template_path = _input_path(Path(input_root), int(timestamp.year))
    if not template_path.is_file():
        raise FileNotFoundError(template_path)
    paths = _output_paths(Path(output_dir), date)
    reference_generator = Path(_reference_generator.__file__).resolve()
    request = {
        "schema_version": 1,
        "date": date,
        "template_sha256": sha256_file(template_path),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "reference_generator_sha256": sha256_file(reference_generator),
        "seed": int(seed),
        "physical": {name: float(value) for name, value in physical.items()},
        "cauchy_a": float(gc_alpha),
        "cauchy_b": float(gc_beta),
        "mean_intercept": float(mean_intercept),
        "mean_lat_slope": float(mean_lat_slope),
        "mean_lat_center": float(mean_lat_center),
        "lat_factor_hr": int(lat_factor_hr),
        "lon_factor_hr": int(lon_factor_hr),
        "pad": float(pad),
        "embedding_spatial_factor": int(embedding_spatial_factor),
        "embedding_temporal_factor": int(embedding_temporal_factor),
        "max_negative_spectral_mass": float(max_negative_spectral_mass),
    }
    request_sha256 = _json_hash(request)
    if all(path.is_file() for path in paths.__dict__.values()) and not overwrite:
        saved_truth = json.loads(paths.truth.read_text(encoding="utf-8"))
        if saved_truth.get("request_sha256") != request_sha256:
            raise RuntimeError(
                f"existing simulation in {output_dir} has a different request; "
                "rerun with overwrite=True"
            )
        return paths

    template = pd.read_pickle(template_path)
    if not isinstance(template, dict):
        raise TypeError(f"{template_path} is not a dict pickle")
    frames = _select_day(template, date)
    plan = _build_plan_for_frames(
        frames,
        physical,
        gc_alpha,
        gc_beta,
        lat_factor_hr,
        lon_factor_hr,
        pad,
        embedding_spatial_factor,
        embedding_temporal_factor,
        max_negative_spectral_mass,
    )

    rng = np.random.default_rng(int(seed))
    embedding_shape = plan.effective_covariance.shape
    white_noise = rng.standard_normal(embedding_shape)
    white_spectrum = np.fft.rfftn(white_noise, axes=(0, 1, 2))
    del white_noise
    gc.collect()
    np.sqrt(plan.spectrum, out=plan.spectrum)
    white_spectrum *= plan.spectrum
    field_full = np.fft.irfftn(
        white_spectrum,
        s=embedding_shape,
        axes=(0, 1, 2),
    ).real
    del white_spectrum
    gc.collect()
    field = field_full[
        : plan.latitude_axis.size,
        : plan.longitude_axis.size,
        : len(frames),
    ].copy()
    del field_full
    plan.spectrum = np.empty((0,), dtype=np.float64)
    plan.effective_covariance = np.empty((0,), dtype=np.float64)
    gc.collect()
    real_frames: dict[str, pd.DataFrame] = {}
    gridded_frames: dict[str, pd.DataFrame] = {}
    manifest_rows: list[dict[str, Any]] = []
    time_values: list[float] = []
    for local_time, (key, frame) in enumerate(frames.items()):
        source_latitude = pd.to_numeric(
            frame["Source_Latitude"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        source_longitude = pd.to_numeric(
            frame["Source_Longitude"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        valid = np.isfinite(source_latitude) & np.isfinite(source_longitude)
        response = np.full(len(frame), np.nan, dtype=np.float64)
        latitude_error = np.asarray([], dtype=np.float64)
        longitude_error = np.asarray([], dtype=np.float64)
        if np.any(valid):
            deadvected_latitude = source_latitude[valid] - float(
                physical["advec_lat"]
            ) * float(local_time)
            deadvected_longitude = source_longitude[valid] - float(
                physical["advec_lon"]
            ) * float(local_time)
            latitude_index, latitude_error = _nearest_indices(
                deadvected_latitude, plan.latitude_axis, plan.delta_latitude
            )
            longitude_index, longitude_error = _nearest_indices(
                deadvected_longitude, plan.longitude_axis, plan.delta_longitude
            )
            mean = float(mean_intercept) + float(mean_lat_slope) * (
                source_latitude[valid] - float(mean_lat_center)
            )
            nugget = float(physical.get("nugget", 0.0))
            noise = (
                rng.normal(0.0, math.sqrt(nugget), size=int(np.count_nonzero(valid)))
                if nugget > 0.0
                else 0.0
            )
            response[valid] = (
                mean + field[latitude_index, longitude_index, local_time] + noise
            )

        real_frame = frame.copy()
        real_frame["ColumnAmountO3"] = response
        gridded_frame, griddification = griddify_one_to_one(
            frame,
            response,
            source_latitude,
            source_longitude,
            valid,
        )
        # Preserve the original time convention used by the frozen design and
        # ProcessedDataLoader.  The historical generator replaced this by 0..7.
        if not np.array_equal(
            pd.to_numeric(gridded_frame["Hours_elapsed"], errors="coerce").to_numpy(),
            pd.to_numeric(frame["Hours_elapsed"], errors="coerce").to_numpy(),
            equal_nan=True,
        ):
            raise AssertionError("griddification changed Hours_elapsed")
        time_value = float(
            np.nanmedian(
                pd.to_numeric(frame["Hours_elapsed"], errors="coerce").to_numpy(
                    dtype=np.float64
                )
            )
        )
        time_values.append(time_value)
        real_frames[key] = real_frame
        gridded_frames[key] = gridded_frame
        manifest_rows.append(
            {
                "date": date,
                "local_time": int(local_time),
                "hour_key": key,
                "hours_elapsed": time_value,
                "n_source_coordinates": int(np.count_nonzero(valid)),
                "n_real_finite": int(np.count_nonzero(np.isfinite(response))),
                "n_gridded_finite": int(gridded_frame["ColumnAmountO3"].notna().sum()),
                "max_abs_latitude_mapping_error": (
                    float(np.max(latitude_error)) if latitude_error.size else None
                ),
                "max_abs_longitude_mapping_error": (
                    float(np.max(longitude_error)) if longitude_error.size else None
                ),
                **griddification,
            }
        )
    if not np.array_equal(
        np.diff(np.rint(time_values)).astype(np.int64),
        np.ones(len(time_values) - 1, dtype=np.int64),
    ):
        raise AssertionError("generated rounded Hours_elapsed do not have unit increments")

    truth = {
        "schema_version": 1,
        "request": request,
        "request_sha256": request_sha256,
        "date": date,
        "input_template": str(template_path),
        "input_template_sha256": sha256_file(template_path),
        "hours_per_day": int(len(frames)),
        "hours_elapsed": time_values,
        "hours_elapsed_policy": "preserved verbatim from the real GEMS template",
        "block_generation": "one 8-hour comoving 3D circulant embedding",
        "correlation_model": "generalized_cauchy",
        "correlation_formula": "(1 + r^a)^(-b/a)",
        "advection_construction": "Z(s,t)=W(s-v*(t-t0),t)",
        "cauchy_a": float(gc_alpha),
        "cauchy_b": float(gc_beta),
        "sigmasq": float(physical["signal_variance"]),
        "range_lat": float(physical["range_lat"]),
        "range_lon": float(physical["range_lon"]),
        "range_time": float(physical["range_time"]),
        "advec_lat": float(physical["advec_lat"]),
        "advec_lon": float(physical["advec_lon"]),
        "nugget": float(physical.get("nugget", 0.0)),
        "mean_intercept": float(mean_intercept),
        "mean_lat_slope": float(mean_lat_slope),
        "mean_lat_center": float(mean_lat_center),
        "lat_factor_hr": int(lat_factor_hr),
        "lon_factor_hr": int(lon_factor_hr),
        "embedding_spatial_factor": int(embedding_spatial_factor),
        "embedding_temporal_factor": int(embedding_temporal_factor),
        "max_negative_spectral_mass": float(max_negative_spectral_mass),
        "simulation_grid": {
            "latitude_first": float(plan.latitude_axis[0]),
            "latitude_count": int(plan.latitude_axis.size),
            "delta_latitude": float(plan.delta_latitude),
            "longitude_first": float(plan.longitude_axis[0]),
            "longitude_count": int(plan.longitude_axis.size),
            "delta_longitude": float(plan.delta_longitude),
            "pad": float(pad),
        },
        "embedding_diagnostics": plan.diagnostics,
        "seed": int(seed),
        "griddification_rule": (
            "nearest regular grid cell, one-to-one, then axis-wise half-cell threshold"
        ),
    }
    _atomic_pickle(paths.real_locations, real_frames)
    _atomic_pickle(paths.gridded, gridded_frames)
    _atomic_csv(paths.manifest, pd.DataFrame(manifest_rows))
    _atomic_json(paths.truth, truth)
    return paths


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--date", default="2024-07-13")
    result.add_argument(
        "--input-root", type=Path, default=Path("/Users/joonwonlee/Documents/GEMS_DATA")
    )
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--seed", type=int, default=20250926)
    result.add_argument("--sigmasq", type=float, default=10.0)
    result.add_argument("--range-lat", type=float, default=0.2)
    result.add_argument("--range-lon", type=float, default=0.3)
    result.add_argument("--range-time", type=float, default=2.0)
    result.add_argument("--advec-lat", type=float, default=0.08)
    result.add_argument("--advec-lon", type=float, default=-0.2)
    result.add_argument("--nugget", type=float, default=0.0)
    result.add_argument("--cauchy-a", type=float, default=0.75)
    result.add_argument("--cauchy-b", type=float, default=1.0)
    result.add_argument("--mean-intercept", type=float, default=260.0)
    result.add_argument("--mean-lat-slope", type=float, default=1.0)
    result.add_argument("--mean-lat-center", type=float, default=-0.5)
    result.add_argument("--lat-factor-hr", type=int, default=2)
    result.add_argument("--lon-factor-hr", type=int, default=2)
    result.add_argument("--pad", type=float, default=0.1)
    result.add_argument("--embedding-spatial-factor", type=int, default=2)
    result.add_argument("--embedding-temporal-factor", type=int, default=2)
    result.add_argument("--max-negative-spectral-mass", type=float, default=0.005)
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    physical = {
        "signal_variance": float(args.sigmasq),
        "range_lat": float(args.range_lat),
        "range_lon": float(args.range_lon),
        "range_time": float(args.range_time),
        "advec_lat": float(args.advec_lat),
        "advec_lon": float(args.advec_lon),
        "nugget": float(args.nugget),
    }
    paths = generate_day(
        date=str(args.date),
        input_root=args.input_root.expanduser().resolve(),
        output_dir=args.output_dir.expanduser().resolve(),
        seed=int(args.seed),
        physical=physical,
        gc_alpha=float(args.cauchy_a),
        gc_beta=float(args.cauchy_b),
        mean_intercept=float(args.mean_intercept),
        mean_lat_slope=float(args.mean_lat_slope),
        mean_lat_center=float(args.mean_lat_center),
        lat_factor_hr=int(args.lat_factor_hr),
        lon_factor_hr=int(args.lon_factor_hr),
        pad=float(args.pad),
        embedding_spatial_factor=int(args.embedding_spatial_factor),
        embedding_temporal_factor=int(args.embedding_temporal_factor),
        max_negative_spectral_mass=float(args.max_negative_spectral_mass),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps({name: str(path) for name, path in paths.__dict__.items()}, indent=2))


if __name__ == "__main__":
    main()
