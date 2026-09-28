#!/usr/bin/env python3
"""Generate controlled space-time interaction simulations on real GEMS locations.

The production design uses independent eight-hour July day blocks.  For each
covariance family it changes only the Lagrangian space-time coupling parameter
eta while holding the spatial margin, flow-following temporal margin, variance,
ranges, advection, mean, nugget, locations, and random seeds fixed.

Unlike the historical multi-day generators, this program never inserts
advection directly into an even BCCB covariance column.  It builds a centrally
symmetric zero-advection covariance on an odd 2n-1 embedding and samples

    Z(s, t) = W(s - v * local_time, local_time).

This avoids dropping a nonzero imaginary FFT component at Nyquist faces.  It
also audits negative spectral mass, restores the requested marginal variance
after any accepted roundoff-level clipping, preserves the original template
time in a separate column, and writes resumable atomic day checkpoints before
final consolidation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import resource
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import fft as scipy_fft
from scipy.spatial import cKDTree


DELTA_LAT_BASE = 0.044
DELTA_LON_BASE = 0.063
HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "st_interaction_scenarios_092726.json"
DEFAULT_LOCAL_INPUT = Path("/Users/joonwonlee/Documents/GEMS_DATA")
DEFAULT_LOCAL_OUTPUT = HERE / "st_interaction_july2024_nugget0_092726"


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    family: str
    interaction_eta: float
    matern_nu: float | None
    cauchy_a: float | None
    cauchy_b: float | None
    normalize_to_efold_range: bool
    role: str


@dataclass
class EmbeddingPlan:
    latitude_axis: np.ndarray
    longitude_axis: np.ndarray
    delta_latitude: float
    delta_longitude: float
    shape: tuple[int, int, int]
    sqrt_spectrum: np.ndarray
    fft_workers: int
    diagnostics: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pd.to_pickle(value, temporary)
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.17g")
    temporary.replace(path)


def parse_hour_key(key: str) -> pd.Timestamp | None:
    match = re.match(
        r"^y(?P<yy>\d{2})m(?P<mm>\d{2})day(?P<dd>\d{2})_hm"
        r"(?P<hh>\d{2}):(?P<minute>\d{2})$",
        str(key),
    )
    if match is None:
        return None
    part = {name: int(value) for name, value in match.groupdict().items()}
    return pd.Timestamp(
        year=2000 + part["yy"],
        month=part["mm"],
        day=part["dd"],
        hour=part["hh"],
        minute=part["minute"],
        tz="UTC",
    )


def template_path(input_root: Path, year: int) -> Path:
    return input_root / f"pickle_{year}" / f"tco_grid_{str(year)[2:]}_07.pkl"


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if int(config.get("schema_version", -1)) != 1:
        raise ValueError("scenario config must have schema_version=1")
    scenarios = config.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) != 6:
        raise ValueError("this production suite requires exactly six scenarios")
    identifiers = [str(value["scenario_id"]) for value in scenarios]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("scenario_id values must be unique")
    return config


def select_scenario(config: Mapping[str, Any], index: int | None, scenario_id: str | None) -> Scenario:
    raw_scenarios = list(config["scenarios"])
    if scenario_id is not None:
        matches = [value for value in raw_scenarios if value["scenario_id"] == scenario_id]
        if len(matches) != 1:
            raise ValueError(f"unknown scenario_id: {scenario_id}")
        raw = matches[0]
    else:
        if index is None:
            raise ValueError("provide --scenario-index or --scenario-id")
        if index < 0 or index >= len(raw_scenarios):
            raise ValueError(f"scenario index {index} is outside 0..{len(raw_scenarios)-1}")
        raw = raw_scenarios[index]
    family = str(raw["family"])
    if family not in {"matern", "generalized_cauchy"}:
        raise ValueError(f"unsupported family: {family}")
    eta = float(raw["interaction_eta"])
    if not 0.0 <= eta <= 1.0:
        raise ValueError("interaction_eta must be in [0,1]")
    result = Scenario(
        scenario_id=str(raw["scenario_id"]),
        family=family,
        interaction_eta=eta,
        matern_nu=float(raw["matern_nu"]) if family == "matern" else None,
        cauchy_a=float(raw["cauchy_a"]) if family == "generalized_cauchy" else None,
        cauchy_b=float(raw["cauchy_b"]) if family == "generalized_cauchy" else None,
        normalize_to_efold_range=bool(raw.get("normalize_to_efold_range", False)),
        role=str(raw.get("role", "")),
    )
    if result.family == "matern" and result.matern_nu != 0.5:
        raise ValueError("this production generator intentionally supports Matérn nu=0.5 only")
    if result.family == "generalized_cauchy":
        assert result.cauchy_a is not None and result.cauchy_b is not None
        if not 0.0 < result.cauchy_a <= 2.0 or result.cauchy_b <= 0.0:
            raise ValueError("generalized-Cauchy requires 0<a<=2 and b>0")
    return result


def group_complete_dates(
    template: Mapping[str, pd.DataFrame],
    year: int,
    month: int,
    start_day: int,
    n_days: int,
    hours_per_day: int,
) -> list[tuple[str, list[tuple[str, pd.DataFrame, pd.Timestamp]]]]:
    grouped: dict[str, list[tuple[pd.Timestamp, str, pd.DataFrame]]] = {}
    for key, frame in template.items():
        timestamp = parse_hour_key(str(key))
        if timestamp is None or timestamp.year != year or timestamp.month != month:
            continue
        date = timestamp.strftime("%Y-%m-%d")
        grouped.setdefault(date, []).append((timestamp, str(key), frame.reset_index(drop=True)))
    requested = [
        (pd.Timestamp(year=year, month=month, day=start_day, tz="UTC") + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(n_days)
    ]
    result: list[tuple[str, list[tuple[str, pd.DataFrame, pd.Timestamp]]]] = []
    for date in requested:
        entries = sorted(grouped.get(date, []), key=lambda item: item[0])
        if len(entries) != hours_per_day:
            raise ValueError(
                f"{date} contains {len(entries)} template hours; expected {hours_per_day}"
            )
        if len({item[1] for item in entries}) != hours_per_day:
            raise ValueError(f"{date} contains duplicate hour keys")
        result.append((date, [(key, frame, stamp) for stamp, key, frame in entries]))
    return result


def finite_source_coordinates(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    latitude = pd.to_numeric(frame["Source_Latitude"], errors="coerce").to_numpy(dtype=np.float64)
    longitude = pd.to_numeric(frame["Source_Longitude"], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(latitude) & np.isfinite(longitude)
    return latitude, longitude, valid


def axis_from_values(values: list[np.ndarray], step: float, pad: float) -> np.ndarray:
    lower = min(float(np.min(value)) for value in values if value.size)
    upper = max(float(np.max(value)) for value in values if value.size)
    start = math.floor((lower - pad) / step) * step
    stop = math.ceil((upper + pad) / step) * step
    count = int(round((stop - start) / step)) + 1
    return start + np.arange(count, dtype=np.float64) * step


def universal_comoving_axes(
    dates: list[tuple[str, list[tuple[str, pd.DataFrame, pd.Timestamp]]]],
    physical: Mapping[str, float],
    lat_factor: int,
    lon_factor: int,
    pad: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    delta_lat = DELTA_LAT_BASE / float(lat_factor)
    delta_lon = DELTA_LON_BASE / float(lon_factor)
    latitude_values: list[np.ndarray] = []
    longitude_values: list[np.ndarray] = []
    for _, entries in dates:
        for local_time, (_, frame, _) in enumerate(entries):
            latitude, longitude, valid = finite_source_coordinates(frame)
            if valid.any():
                latitude_values.append(
                    latitude[valid] - float(physical["advec_lat"]) * local_time
                )
                longitude_values.append(
                    longitude[valid] - float(physical["advec_lon"]) * local_time
                )
    if not latitude_values or not longitude_values:
        raise ValueError("selected templates contain no finite source coordinates")
    return (
        axis_from_values(latitude_values, delta_lat, pad),
        axis_from_values(longitude_values, delta_lon, pad),
        delta_lat,
        delta_lon,
    )


def signed_periodic_lags(size: int, step: float) -> np.ndarray:
    index = np.arange(size, dtype=np.float64)
    signed = np.where(index <= size // 2, index, index - size)
    return signed * float(step)


def radial_correlation_inplace(distance: np.ndarray, scenario: Scenario) -> np.ndarray:
    if scenario.family == "matern":
        # Matérn nu=0.5 under the repository range convention.
        np.negative(distance, out=distance)
        np.exp(distance, out=distance)
        return distance
    assert scenario.cauchy_a is not None and scenario.cauchy_b is not None
    scale = 1.0
    if scenario.normalize_to_efold_range:
        scale = (math.exp(scenario.cauchy_a / scenario.cauchy_b) - 1.0) ** (
            1.0 / scenario.cauchy_a
        )
    np.multiply(distance, scale, out=distance)
    np.power(distance, scenario.cauchy_a, out=distance)
    distance += 1.0
    np.power(distance, -scenario.cauchy_b / scenario.cauchy_a, out=distance)
    return distance


def theoretical_correlation(ds: float, dt: float, scenario: Scenario) -> float:
    spatial = radial_correlation_inplace(np.asarray(float(ds)), scenario).item()
    temporal = radial_correlation_inplace(np.asarray(float(dt)), scenario).item()
    joint = radial_correlation_inplace(np.asarray(math.hypot(ds, dt)), scenario).item()
    return float(
        (1.0 - scenario.interaction_eta) * spatial * temporal
        + scenario.interaction_eta * joint
    )


def symmetry_sample_error(covariance: np.ndarray) -> float:
    candidates = lambda n: sorted({0, 1, 2, n // 3, n // 2, n - 2, n - 1})
    error = 0.0
    for i in candidates(covariance.shape[0]):
        for j in candidates(covariance.shape[1]):
            for k in candidates(covariance.shape[2]):
                error = max(
                    error,
                    abs(
                        float(covariance[i, j, k])
                        - float(covariance[(-i) % covariance.shape[0], (-j) % covariance.shape[1], (-k) % covariance.shape[2]])
                    ),
                )
    return error


def estimate_memory(shape: tuple[int, int, int]) -> dict[str, float | int]:
    cells = int(np.prod(shape))
    rfft_cells = int(shape[0] * shape[1] * (shape[2] // 2 + 1))
    real_gib = cells * 8.0 / 2**30
    complex_gib = rfft_cells * 16.0 / 2**30
    spectrum_gib = rfft_cells * 8.0 / 2**30
    # Conservative simultaneous arrays during covariance/spectrum construction.
    conservative_peak = 3.0 * real_gib + 2.0 * complex_gib + spectrum_gib
    return {
        "embedding_cells": cells,
        "rfft_cells": rfft_cells,
        "one_float64_embedding_gib": real_gib,
        "one_complex128_rfft_gib": complex_gib,
        "one_float64_rfft_spectrum_gib": spectrum_gib,
        "conservative_peak_working_set_gib": conservative_peak,
    }


def build_embedding(
    latitude_axis: np.ndarray,
    longitude_axis: np.ndarray,
    delta_latitude: float,
    delta_longitude: float,
    hours_per_day: int,
    physical: Mapping[str, float],
    scenario: Scenario,
    spatial_factor: int,
    temporal_factor: int,
    max_negative_mass: float,
    max_relative_covariance_distortion: float,
    fft_workers: int,
) -> EmbeddingPlan:
    shape = (
        spatial_factor * (latitude_axis.size - 1) + 1,
        spatial_factor * (longitude_axis.size - 1) + 1,
        temporal_factor * (hours_per_day - 1) + 1,
    )
    if any(value % 2 == 0 for value in shape):
        raise AssertionError(f"odd embedding required, got {shape}")
    print(f"embedding shape={shape}, memory={estimate_memory(shape)}", flush=True)
    lat = signed_periodic_lags(shape[0], delta_latitude) / float(physical["range_lat"])
    lon = signed_periodic_lags(shape[1], delta_longitude) / float(physical["range_lon"])
    tim = signed_periodic_lags(shape[2], 1.0) / float(physical["range_time"])

    spatial_squared = np.square(lat)[:, None, None] + np.square(lon)[None, :, None]
    eta = scenario.interaction_eta
    if eta == 0.0:
        np.sqrt(spatial_squared, out=spatial_squared)
        spatial_corr = radial_correlation_inplace(spatial_squared, scenario)
        temporal_corr = radial_correlation_inplace(np.abs(tim.copy()), scenario)
        covariance = spatial_corr * temporal_corr[None, None, :]
    elif eta == 1.0:
        distance = spatial_squared + np.square(tim)[None, None, :]
        del spatial_squared
        np.sqrt(distance, out=distance)
        covariance = radial_correlation_inplace(distance, scenario)
    else:
        spatial_distance = np.sqrt(spatial_squared)
        spatial_corr = radial_correlation_inplace(spatial_distance, scenario)
        temporal_corr = radial_correlation_inplace(np.abs(tim.copy()), scenario)
        separable = spatial_corr * temporal_corr[None, None, :]
        del spatial_corr, spatial_distance, temporal_corr
        distance = spatial_squared + np.square(tim)[None, None, :]
        del spatial_squared
        np.sqrt(distance, out=distance)
        joint = radial_correlation_inplace(distance, scenario)
        separable *= 1.0 - eta
        joint *= eta
        separable += joint
        covariance = separable
        del separable, joint
    covariance *= float(physical["sigmasq"])
    covariance[0, 0, 0] = float(physical["sigmasq"])
    central_error = symmetry_sample_error(covariance)
    if central_error > 1.0e-10:
        raise ArithmeticError(f"covariance is not centrally symmetric: {central_error}")

    started = time.perf_counter()
    raw_spectrum = scipy_fft.rfftn(
        covariance,
        axes=(0, 1, 2),
        workers=fft_workers,
        overwrite_x=True,
    )
    fft_seconds = time.perf_counter() - started
    del covariance
    gc.collect()
    max_real = float(np.max(np.abs(raw_spectrum.real)))
    max_imaginary = float(np.max(np.abs(raw_spectrum.imag)))
    imaginary_ratio = max_imaginary / max_real if max_real else 0.0
    if imaginary_ratio > 1.0e-10:
        raise ArithmeticError(
            f"imaginary spectrum leakage {imaginary_ratio:.6g} exceeds 1e-10"
        )
    spectrum = raw_spectrum.real.copy()
    del raw_spectrum
    gc.collect()

    negative = spectrum < 0.0
    weights = np.full(spectrum.shape[-1], 2.0, dtype=np.float64)
    weights[0] = 1.0
    if shape[-1] % 2 == 0:
        weights[-1] = 1.0
    weight = weights.reshape((1, 1, -1))
    absolute_mass = float(np.sum(np.abs(spectrum) * weight))
    negative_mass = float(np.sum(np.abs(spectrum) * negative * weight))
    negative_fraction = negative_mass / absolute_mass if absolute_mass else 0.0
    minimum = float(np.min(spectrum))
    negative_count = int(np.count_nonzero(negative))
    if negative_fraction > max_negative_mass:
        raise ArithmeticError(
            "FFT embedding requires excessive eigenvalue clipping: "
            f"negative mass {negative_fraction:.8g} > {max_negative_mass:.8g}"
        )
    np.maximum(spectrum, 0.0, out=spectrum)
    variance_before = float(np.sum(spectrum * weight) / np.prod(shape))
    variance_scale = float(physical["sigmasq"]) / variance_before
    spectrum *= variance_scale
    variance_after = float(np.sum(spectrum * weight) / np.prod(shape))
    # The inverse DFT of a spectral perturbation is bounded pointwise by its
    # weighted spectral L1 norm divided by the number of embedding cells.
    # Clipping contributes negative_mass and the subsequent positive-spectrum
    # rescaling contributes the same amount (up to roundoff).
    covariance_distortion_bound = (
        abs(1.0 - variance_scale) * variance_before
        + negative_mass / float(np.prod(shape))
    )
    relative_covariance_distortion_bound = (
        covariance_distortion_bound / float(physical["sigmasq"])
    )
    if relative_covariance_distortion_bound > max_relative_covariance_distortion:
        raise ArithmeticError(
            "spectral correction can distort a covariance entry too much: "
            f"relative bound {relative_covariance_distortion_bound:.8g} > "
            f"{max_relative_covariance_distortion:.8g}"
        )
    del negative
    gc.collect()

    diagnostics = {
        "construction": "zero-advection comoving FFT, then Z(s,t)=W(s-v*local_time,t)",
        "embedding_shape": list(shape),
        "memory_estimate": estimate_memory(shape),
        "central_symmetry_sample_max_error": central_error,
        "spectrum_fft_seconds": fft_seconds,
        "fft_workers": fft_workers,
        "spectrum_min_before_clipping": minimum,
        "spectrum_negative_count_rfft": negative_count,
        "spectrum_negative_mass_fraction": negative_fraction,
        "spectrum_max_abs_imaginary": max_imaginary,
        "spectrum_max_abs_imaginary_over_max_abs_real": imaginary_ratio,
        "variance_before_renormalization": variance_before,
        "variance_renormalization_scale": variance_scale,
        "variance_after_renormalization": variance_after,
        "absolute_covariance_distortion_bound": covariance_distortion_bound,
        "relative_covariance_distortion_bound": relative_covariance_distortion_bound,
        "max_relative_covariance_distortion_allowed": max_relative_covariance_distortion,
        "generator_label": (
            "exact_circulant_embedding"
            if negative_count == 0
            else "spectrally_corrected_circulant_embedding"
        ),
    }
    np.sqrt(spectrum, out=spectrum)
    return EmbeddingPlan(
        latitude_axis=latitude_axis,
        longitude_axis=longitude_axis,
        delta_latitude=delta_latitude,
        delta_longitude=delta_longitude,
        shape=shape,
        sqrt_spectrum=spectrum,
        fft_workers=fft_workers,
        diagnostics=diagnostics,
    )


def nearest_indices(values: np.ndarray, axis: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    raw = np.rint((values - float(axis[0])) / step).astype(np.int64)
    if np.any(raw < 0) or np.any(raw >= axis.size):
        raise ValueError("a de-advected coordinate lies outside the universal FFT grid")
    error = np.abs(axis[raw] - values)
    if np.any(error > 0.5 * step + 1.0e-10):
        raise ValueError("nearest FFT mapping exceeds one half-cell")
    return raw, error


def griddify_one_to_one(
    frame: pd.DataFrame,
    response: np.ndarray,
    source_latitude: np.ndarray,
    source_longitude: np.ndarray,
    source_valid: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = frame.copy()
    output["ColumnAmountO3"] = np.nan
    grid_lat = pd.to_numeric(output["Latitude"], errors="coerce").to_numpy(dtype=float)
    grid_lon = pd.to_numeric(output["Longitude"], errors="coerce").to_numpy(dtype=float)
    grid_valid = np.isfinite(grid_lat) & np.isfinite(grid_lon)
    grid_rows = np.flatnonzero(grid_valid)
    source_rows = np.flatnonzero(source_valid & np.isfinite(response))
    diagnostics: dict[str, Any] = {
        "n_grid_rows": len(output),
        "n_grid_finite": int(grid_valid.sum()),
        "n_source_valid": len(source_rows),
        "n_collision_lost": 0,
        "n_assigned_after_threshold": 0,
        "n_rejected_threshold": 0,
        "threshold_lat": DELTA_LAT_BASE / 2.0,
        "threshold_lon": DELTA_LON_BASE / 2.0,
    }
    if not len(grid_rows) or not len(source_rows):
        return output, diagnostics
    tree = cKDTree(np.column_stack([grid_lat[grid_valid], grid_lon[grid_valid]]))
    distance, local_cell = tree.query(
        np.column_stack([source_latitude[source_rows], source_longitude[source_rows]]), k=1
    )
    best_source = np.full(len(grid_rows), -1, dtype=np.int64)
    best_distance = np.full(len(grid_rows), np.inf)
    for cell, source, value in zip(local_cell, source_rows, distance):
        if value < best_distance[cell]:
            diagnostics["n_collision_lost"] += int(best_source[cell] >= 0)
            best_source[cell] = source
            best_distance[cell] = value
        else:
            diagnostics["n_collision_lost"] += 1
    filled = np.flatnonzero(best_source >= 0)
    source_choice = best_source[filled]
    grid_choice = grid_rows[filled]
    keep = (
        np.abs(source_latitude[source_choice] - grid_lat[grid_choice]) <= DELTA_LAT_BASE / 2.0
    ) & (
        np.abs(source_longitude[source_choice] - grid_lon[grid_choice]) <= DELTA_LON_BASE / 2.0
    )
    diagnostics["n_rejected_threshold"] = int((~keep).sum())
    diagnostics["n_assigned_after_threshold"] = int(keep.sum())
    selected_source = source_choice[keep]
    selected_grid = grid_choice[keep]
    output.loc[output.index[selected_grid], "ColumnAmountO3"] = response[selected_source]
    output.loc[output.index[selected_grid], "Source_Latitude"] = source_latitude[selected_source]
    output.loc[output.index[selected_grid], "Source_Longitude"] = source_longitude[selected_source]
    output.loc[output.index[selected_grid], "Template_Hours_elapsed"] = (
        frame.iloc[selected_source]["Template_Hours_elapsed"].to_numpy()
    )
    return output, diagnostics


def generate_fft_field(plan: EmbeddingPlan, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white_noise = rng.standard_normal(plan.shape)
    white_spectrum = scipy_fft.rfftn(
        white_noise,
        axes=(0, 1, 2),
        workers=plan.fft_workers,
        overwrite_x=True,
    )
    del white_noise
    gc.collect()
    white_spectrum *= plan.sqrt_spectrum
    field = scipy_fft.irfftn(
        white_spectrum,
        s=plan.shape,
        axes=(0, 1, 2),
        workers=plan.fft_workers,
        overwrite_x=True,
    ).real
    del white_spectrum
    gc.collect()
    return field


def max_rss_gib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / 2**30
    return value * 1024.0 / 2**30


def prepare_frame_time(
    frame: pd.DataFrame,
    block_index: int,
    local_time: int,
    analysis_day_origin: float,
) -> tuple[pd.DataFrame, float, float]:
    output = frame.copy()
    template_time = pd.to_numeric(output["Hours_elapsed"], errors="coerce").to_numpy(dtype=np.float64)
    finite = template_time[np.isfinite(template_time)]
    if not finite.size:
        raise ValueError("template frame has no finite Hours_elapsed")
    template_value = float(np.median(finite))
    analysis_value = float(analysis_day_origin + local_time)
    output["Template_Hours_elapsed"] = output["Hours_elapsed"]
    output["Hours_elapsed"] = analysis_value
    output["Simulation_Block"] = int(block_index)
    output["Simulation_Time_Index"] = int(local_time)
    return output, template_value, analysis_value


def day_request_hash(
    run_request: Mapping[str, Any], date: str, seed: int, keys: list[str]
) -> str:
    return json_hash({"run_request": run_request, "date": date, "seed": seed, "keys": keys})


def generate_day_checkpoint(
    *,
    date: str,
    entries: list[tuple[str, pd.DataFrame, pd.Timestamp]],
    block_index: int,
    seed: int,
    plan: EmbeddingPlan,
    physical: Mapping[str, float],
    run_request: Mapping[str, Any],
    checkpoint_root: Path,
    overwrite: bool,
) -> dict[str, Any]:
    day_dir = checkpoint_root / date
    success_path = day_dir / "SUCCESS.json"
    expected_hash = day_request_hash(run_request, date, seed, [entry[0] for entry in entries])
    required = [
        day_dir / "real_locations.pkl",
        day_dir / "gridded.pkl",
        day_dir / "manifest.csv",
        success_path,
    ]
    if all(path.is_file() for path in required) and not overwrite:
        saved = json.loads(success_path.read_text(encoding="utf-8"))
        if saved.get("day_request_sha256") != expected_hash:
            raise RuntimeError(f"checkpoint request mismatch for {date}; use --overwrite")
        print(f"reuse checkpoint {date}", flush=True)
        return saved

    started = time.perf_counter()
    field = generate_fft_field(plan, seed)
    real_frames: dict[str, pd.DataFrame] = {}
    gridded_frames: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    first_template_time = pd.to_numeric(
        entries[0][1]["Hours_elapsed"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not np.isfinite(first_template_time).any():
        raise ValueError(f"{date} first frame has no finite Hours_elapsed")
    analysis_day_origin = float(np.nanmedian(first_template_time))
    for local_time, (key, raw_frame, stamp) in enumerate(entries):
        frame, template_hours, analysis_hours = prepare_frame_time(
            raw_frame, block_index, local_time, analysis_day_origin
        )
        source_lat, source_lon, valid = finite_source_coordinates(frame)
        response = np.full(len(frame), np.nan, dtype=np.float64)
        lat_error = np.asarray([], dtype=np.float64)
        lon_error = np.asarray([], dtype=np.float64)
        if valid.any():
            comoving_lat = source_lat[valid] - float(physical["advec_lat"]) * local_time
            comoving_lon = source_lon[valid] - float(physical["advec_lon"]) * local_time
            lat_index, lat_error = nearest_indices(
                comoving_lat, plan.latitude_axis, plan.delta_latitude
            )
            lon_index, lon_error = nearest_indices(
                comoving_lon, plan.longitude_axis, plan.delta_longitude
            )
            known_mean = float(physical["mean_intercept"]) + float(
                physical["mean_lat_slope"]
            ) * (source_lat[valid] - float(physical["mean_lat_center"]))
            response[valid] = known_mean + field[lat_index, lon_index, local_time]
        real = frame.copy()
        real["ColumnAmountO3"] = response
        gridded, grid_diag = griddify_one_to_one(
            frame, response, source_lat, source_lon, valid
        )
        real_frames[key] = real
        gridded_frames[key] = gridded
        rows.append(
            {
                "date": date,
                "block_index": block_index,
                "local_time": local_time,
                "hour_key": key,
                "template_timestamp": stamp.isoformat(),
                "template_hours_elapsed": template_hours,
                "analysis_hours_elapsed": analysis_hours,
                "analysis_minus_template_hours": analysis_hours - template_hours,
                "simulation_time_index": local_time,
                "seed": seed,
                "n_source_coordinates": int(valid.sum()),
                "n_real_finite": int(np.isfinite(response).sum()),
                "n_gridded_finite": int(gridded["ColumnAmountO3"].notna().sum()),
                "max_abs_latitude_mapping_error": float(np.max(lat_error)) if lat_error.size else None,
                "max_abs_longitude_mapping_error": float(np.max(lon_error)) if lon_error.size else None,
                **grid_diag,
            }
        )
    del field
    gc.collect()
    day_dir.mkdir(parents=True, exist_ok=True)
    atomic_pickle(day_dir / "real_locations.pkl", real_frames)
    atomic_pickle(day_dir / "gridded.pkl", gridded_frames)
    atomic_csv(day_dir / "manifest.csv", pd.DataFrame(rows))
    success = {
        "date": date,
        "block_index": block_index,
        "seed": seed,
        "day_request_sha256": expected_hash,
        "n_hours": len(entries),
        "elapsed_seconds": time.perf_counter() - started,
        "max_rss_gib": max_rss_gib(),
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(success_path, success)
    print(
        f"completed {date}: {success['elapsed_seconds']:.1f}s, "
        f"peak RSS {success['max_rss_gib']:.1f} GiB",
        flush=True,
    )
    return success


def consolidate(
    output_year: Path,
    checkpoint_root: Path,
    dates: list[str],
    truth: Mapping[str, Any],
    overwrite: bool,
) -> None:
    prefix = f"sim_july{int(truth['year'])}_st_circulant"
    final_paths = {
        "real": output_year / f"{prefix}_real_locations.pkl",
        "grid": output_year / f"{prefix}_gridded.pkl",
        "manifest": output_year / f"{prefix}_manifest.csv",
        "grid_diag": output_year / f"{prefix}_griddification_diag.csv",
        "embedding_csv": output_year / f"{prefix}_embedding_diag.csv",
        "embedding_json": output_year / f"{prefix}_embedding_diag.json",
        "truth": output_year / f"{prefix}_truth.json",
        "complete": output_year / "COMPLETE.json",
    }
    if all(path.is_file() for path in final_paths.values()) and not overwrite:
        complete = json.loads(final_paths["complete"].read_text(encoding="utf-8"))
        if complete.get("run_request_sha256") != truth["run_request_sha256"]:
            raise RuntimeError("existing final asset has a different request; use --overwrite")
        print(f"final asset already complete: {output_year}", flush=True)
        return
    real_all: dict[str, pd.DataFrame] = {}
    grid_all: dict[str, pd.DataFrame] = {}
    manifests = []
    for date in dates:
        day_dir = checkpoint_root / date
        real = pd.read_pickle(day_dir / "real_locations.pkl")
        grid = pd.read_pickle(day_dir / "gridded.pkl")
        overlap = set(real_all).intersection(real)
        if overlap:
            raise RuntimeError(f"duplicate real-location keys during consolidation: {overlap}")
        real_all.update(real)
        grid_all.update(grid)
        manifests.append(pd.read_csv(day_dir / "manifest.csv"))
    output_year.mkdir(parents=True, exist_ok=True)
    atomic_pickle(final_paths["real"], real_all)
    atomic_pickle(final_paths["grid"], grid_all)
    manifest = pd.concat(manifests, ignore_index=True)
    atomic_csv(final_paths["manifest"], manifest)
    grid_columns = [
        name
        for name in manifest.columns
        if name
        in {
            "date",
            "block_index",
            "local_time",
            "hour_key",
            "n_grid_rows",
            "n_grid_finite",
            "n_source_valid",
            "n_collision_lost",
            "n_assigned_after_threshold",
            "n_rejected_threshold",
            "threshold_lat",
            "threshold_lon",
        }
    ]
    atomic_csv(final_paths["grid_diag"], manifest[grid_columns])
    embedding_diagnostics = dict(truth["embedding_diagnostics"])
    atomic_csv(
        final_paths["embedding_csv"],
        pd.json_normalize([embedding_diagnostics], sep="."),
    )
    atomic_json(final_paths["embedding_json"], embedding_diagnostics)
    atomic_json(final_paths["truth"], truth)
    atomic_json(
        final_paths["complete"],
        {
            "run_request_sha256": truth["run_request_sha256"],
            "scenario_id": truth["scenario_id"],
            "n_days": len(dates),
            "n_hours": len(real_all),
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "files": {name: str(path.name) for name, path in final_paths.items() if name != "complete"},
        },
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    group = result.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario-index", type=int)
    group.add_argument("--scenario-id")
    result.add_argument("--input-root", type=Path, default=DEFAULT_LOCAL_INPUT)
    result.add_argument("--output-root", type=Path, default=DEFAULT_LOCAL_OUTPUT)
    result.add_argument("--start-day", type=int)
    result.add_argument("--n-days", type=int)
    result.add_argument(
        "--axis-n-days",
        type=int,
        help="number of dates used to fix universal FFT axes (must be >= n-days)",
    )
    result.add_argument("--lat-factor-hr", type=int)
    result.add_argument("--lon-factor-hr", type=int)
    result.add_argument("--max-negative-spectral-mass", type=float)
    result.add_argument("--max-relative-covariance-distortion", type=float)
    result.add_argument("--fft-workers", type=int)
    result.add_argument("--preflight-only", action="store_true")
    result.add_argument("--overwrite", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_config(config_path)
    scenario = select_scenario(config, args.scenario_index, args.scenario_id)
    calendar = dict(config["calendar"])
    grid = dict(config["grid"])
    physical = {name: float(value) for name, value in config["shared_parameters"].items()}
    year = int(calendar["year"])
    month = int(calendar["month"])
    start_day = int(args.start_day if args.start_day is not None else calendar["start_day"])
    n_days = int(args.n_days if args.n_days is not None else calendar["n_days"])
    axis_n_days = int(args.axis_n_days if args.axis_n_days is not None else n_days)
    hours_per_day = int(calendar["hours_per_day"])
    lat_factor = int(args.lat_factor_hr or grid["lat_factor_hr"])
    lon_factor = int(args.lon_factor_hr or grid["lon_factor_hr"])
    max_negative_mass = float(
        args.max_negative_spectral_mass
        if args.max_negative_spectral_mass is not None
        else grid["max_negative_spectral_mass"]
    )
    max_relative_covariance_distortion = float(
        args.max_relative_covariance_distortion
        if args.max_relative_covariance_distortion is not None
        else grid["max_relative_covariance_distortion_bound"]
    )
    fft_workers = int(
        args.fft_workers
        if args.fft_workers is not None
        else os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("OMP_NUM_THREADS", "1"))
    )
    common_random_numbers = bool(config["random_design"]["common_random_numbers"])
    if not common_random_numbers:
        raise ValueError(
            "this controlled paired suite requires common_random_numbers=true"
        )
    if physical["nugget"] != 0.0:
        raise ValueError("this controlled suite requires nugget exactly zero")
    if (
        n_days <= 0
        or axis_n_days < n_days
        or lat_factor <= 0
        or lon_factor <= 0
        or fft_workers <= 0
    ):
        raise ValueError(
            "n_days, high-resolution factors, and FFT workers must be positive, "
            "and axis_n_days must be at least n_days"
        )

    input_path = template_path(args.input_root.expanduser().resolve(), year)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    print(f"loading template {input_path}", flush=True)
    template = pd.read_pickle(input_path)
    if not isinstance(template, dict):
        raise TypeError(f"{input_path} is not a dict pickle")
    dates = group_complete_dates(
        template, year, month, start_day, n_days, hours_per_day
    )
    axis_dates = (
        dates
        if axis_n_days == n_days
        else group_complete_dates(
            template, year, month, start_day, axis_n_days, hours_per_day
        )
    )
    axes = universal_comoving_axes(
        axis_dates,
        physical,
        lat_factor,
        lon_factor,
        float(grid["pad"]),
    )
    embedding_shape = (
        int(grid["embedding_spatial_factor"]) * (axes[0].size - 1) + 1,
        int(grid["embedding_spatial_factor"]) * (axes[1].size - 1) + 1,
        int(grid["embedding_temporal_factor"]) * (hours_per_day - 1) + 1,
    )
    print(
        json.dumps(
            {
                "scenario": scenario.__dict__,
                "dates": [date for date, _ in dates],
                "grid_axis_dates": [date for date, _ in axis_dates],
                "simulation_grid": [axes[0].size, axes[1].size, hours_per_day],
                "embedding_shape": embedding_shape,
                "memory_estimate": estimate_memory(embedding_shape),
                "correlation_landmarks": {
                    "R_space_1": theoretical_correlation(1.0, 0.0, scenario),
                    "R_time_1": theoretical_correlation(0.0, 1.0, scenario),
                    "R_corner_1_1": theoretical_correlation(1.0, 1.0, scenario),
                },
            },
            indent=2,
        ),
        flush=True,
    )
    if args.preflight_only:
        print("preflight complete; no simulation files written", flush=True)
        return

    output_root = args.output_root.expanduser().resolve()
    output_year = output_root / scenario.scenario_id / f"{year}_july_st_circulant"
    checkpoint_root = output_year / "day_checkpoints"
    request = {
        "schema_version": 1,
        "suite_name": config["suite_name"],
        "scenario": scenario.__dict__,
        "physical": physical,
        "year": year,
        "month": month,
        "start_day": start_day,
        "n_days": n_days,
        "axis_n_days": axis_n_days,
        "hours_per_day": hours_per_day,
        "selected_dates": [date for date, _ in dates],
        "grid_axis_dates": [date for date, _ in axis_dates],
        "lat_factor_hr": lat_factor,
        "lon_factor_hr": lon_factor,
        "pad": float(grid["pad"]),
        "embedding_spatial_factor": int(grid["embedding_spatial_factor"]),
        "embedding_temporal_factor": int(grid["embedding_temporal_factor"]),
        "max_negative_spectral_mass": max_negative_mass,
        "max_relative_covariance_distortion": max_relative_covariance_distortion,
        "fft_workers": fft_workers,
        "base_seed": int(config["random_design"]["base_seed"]),
        "common_random_numbers": common_random_numbers,
        "input_template_sha256": sha256_file(input_path),
        "scenario_config_sha256": sha256_file(config_path),
        "generator_sha256": sha256_file(Path(__file__).resolve()),
    }
    request_sha = json_hash(request)
    complete_path = output_year / "COMPLETE.json"
    truth_path = output_year / f"sim_july{year}_st_circulant_truth.json"
    if complete_path.is_file() and truth_path.is_file() and not args.overwrite:
        complete = json.loads(complete_path.read_text(encoding="utf-8"))
        saved_truth = json.loads(truth_path.read_text(encoding="utf-8"))
        if (
            complete.get("run_request_sha256") != request_sha
            or saved_truth.get("run_request_sha256") != request_sha
        ):
            raise RuntimeError(
                f"existing final asset has a different request under {output_year}; "
                "use --overwrite only after checking the target"
            )
        print(f"final asset already complete; no FFT work needed: {output_year}", flush=True)
        return
    if args.overwrite and complete_path.is_file():
        # Invalidate success before replacing any constituent file, so an
        # interrupted overwrite cannot leave a stale COMPLETE marker.
        complete_path.unlink()
    plan = build_embedding(
        axes[0],
        axes[1],
        axes[2],
        axes[3],
        hours_per_day,
        physical,
        scenario,
        int(grid["embedding_spatial_factor"]),
        int(grid["embedding_temporal_factor"]),
        max_negative_mass,
        max_relative_covariance_distortion,
        fft_workers,
    )
    run_started = time.perf_counter()
    day_summaries = []
    for block_index, (date, entries) in enumerate(dates):
        day_of_month = int(pd.Timestamp(date).day)
        seed = int(config["random_design"]["base_seed"]) + 1000 * year + day_of_month
        day_summaries.append(
            generate_day_checkpoint(
                date=date,
                entries=entries,
                block_index=block_index,
                seed=seed,
                plan=plan,
                physical=physical,
                run_request=request,
                checkpoint_root=checkpoint_root,
                overwrite=bool(args.overwrite),
            )
        )
    cauchy_efold_scale = (
        (math.exp(scenario.cauchy_a / scenario.cauchy_b) - 1.0)
        ** (1.0 / scenario.cauchy_a)
        if scenario.family == "generalized_cauchy"
        and scenario.normalize_to_efold_range
        else 1.0
    )
    truth = {
        "schema_version": 1,
        "suite_name": config["suite_name"],
        "scenario_id": scenario.scenario_id,
        "family": scenario.family,
        "correlation_model": scenario.family,
        "interaction_eta": scenario.interaction_eta,
        "role": scenario.role,
        "matern_nu": scenario.matern_nu,
        "smooth": scenario.matern_nu,
        "cauchy_a": scenario.cauchy_a,
        "cauchy_b": scenario.cauchy_b,
        "cauchy_efold_scale": (
            cauchy_efold_scale
            if scenario.family == "generalized_cauchy"
            else None
        ),
        "range_parameterization": "effective e-fold distance: R(1)=exp(-1)",
        "kernel_range_lat": physical["range_lat"] / cauchy_efold_scale,
        "kernel_range_lon": physical["range_lon"] / cauchy_efold_scale,
        "kernel_range_time": physical["range_time"] / cauchy_efold_scale,
        "fitter_truth_parameters": {
            "parameterization": (
                "repository kernel scale; use these range values directly in the fitter"
            ),
            "family": scenario.family,
            "sigmasq": physical["sigmasq"],
            "range_lat": physical["range_lat"] / cauchy_efold_scale,
            "range_lon": physical["range_lon"] / cauchy_efold_scale,
            "range_time": physical["range_time"] / cauchy_efold_scale,
            "advec_lat": physical["advec_lat"],
            "advec_lon": physical["advec_lon"],
            "nugget": physical["nugget"],
            "matern_nu": scenario.matern_nu,
            "cauchy_a": scenario.cauchy_a,
            "cauchy_b": scenario.cauchy_b,
        },
        "correlation_formula": config["coupling_definition"],
        **physical,
        "year": year,
        "month": month,
        "selected_dates": [date for date, _ in dates],
        "grid_axis_dates": [date for date, _ in axis_dates],
        "n_independent_days": n_days,
        "n_hours": n_days * hours_per_day,
        "hours_per_day": hours_per_day,
        "block_generation": "independent daily eight-hour comoving circulant embeddings",
        "cross_day_policy": "block diagonal; downstream fitting and diagnostics must not form cross-day pairs",
        "hours_elapsed_policy": (
            "Hours_elapsed is the first template time plus integer local_time, "
            "exactly matching the FFT lattice; original values are retained in "
            "Template_Hours_elapsed."
        ),
        "simulation_time_policy": (
            "FFT local_time and Simulation_Time_Index are integer 0..7.  The "
            "04:48--07:48 template times are shifted by +5 minutes in analysis "
            "Hours_elapsed so generated and fitted covariance use the same time geometry."
        ),
        "known_mean_formula": "mean_intercept + mean_lat_slope*(Source_Latitude-mean_lat_center)",
        "oracle_residual_formula": "ColumnAmountO3 minus known_mean_formula",
        "common_random_numbers": common_random_numbers,
        "seed_rule": config["random_design"]["seed_rule"],
        "lat_factor_hr": lat_factor,
        "lon_factor_hr": lon_factor,
        "delta_lat_base": DELTA_LAT_BASE,
        "delta_lon_base": DELTA_LON_BASE,
        "delta_latitude": axes[2],
        "delta_longitude": axes[3],
        "sampling_coordinate_tolerance": {
            "max_abs_latitude": axes[2] / 2.0,
            "max_abs_longitude": axes[3] / 2.0,
            "max_standardized_latitude": axes[2] / (2.0 * physical["range_lat"]),
            "max_standardized_longitude": axes[3] / (2.0 * physical["range_lon"]),
            "interpretation": (
                "analytic covariance is exact on the FFT lattice; saved source "
                "coordinates are mapped to the nearest lattice point"
            ),
        },
        "embedding_spatial_factor": int(grid["embedding_spatial_factor"]),
        "embedding_temporal_factor": int(grid["embedding_temporal_factor"]),
        "max_negative_spectral_mass": max_negative_mass,
        "max_relative_covariance_distortion": max_relative_covariance_distortion,
        "fft_workers": fft_workers,
        "griddification_rule": (
            "nearest template grid cell, one-to-one, then axis-wise half-cell threshold"
        ),
        "simulation_grid": {
            "latitude_first": float(axes[0][0]),
            "latitude_count": int(axes[0].size),
            "longitude_first": float(axes[1][0]),
            "longitude_count": int(axes[1].size),
            "hours_per_day": hours_per_day,
        },
        "embedding_diagnostics": plan.diagnostics,
        "correlation_landmarks": {
            "R_space_1": theoretical_correlation(1.0, 0.0, scenario),
            "R_flow_time_1": theoretical_correlation(0.0, 1.0, scenario),
            "R_corner_1_1": theoretical_correlation(1.0, 1.0, scenario),
        },
        "input_template": str(input_path),
        "run_request": request,
        "run_request_sha256": request_sha,
        "day_summaries": day_summaries,
        "generator_elapsed_seconds_before_consolidation": time.perf_counter() - run_started,
        "max_rss_gib": max_rss_gib(),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy_fft_workers": fft_workers,
            "cpu_count": os.cpu_count(),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    consolidate(
        output_year,
        checkpoint_root,
        [date for date, _ in dates],
        truth,
        overwrite=bool(args.overwrite),
    )
    print(f"complete: {output_year}", flush=True)


if __name__ == "__main__":
    main()
