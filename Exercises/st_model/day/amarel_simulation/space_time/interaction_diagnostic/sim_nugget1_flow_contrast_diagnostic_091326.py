#!/usr/bin/env python3
"""Nugget-1 simulation test of local space-time contrasts in flow coordinates.

The experiment uses the local smooth=0.5, nugget=1 July simulation.  One
adapted lag-4/3/2 Vecchia model (batch size 64) is fitted on July 13.  The
known simulation mean is then removed from three independent eight-hour day
blocks, so contrast sensitivity can be evaluated with independent-day
standard errors rather than a spatial jackknife alone.

Diagnostics:
1. Oriented cross-variogram level and h-versus-minus-h asymmetry.
2. Seven-point odd/even contrast matrices in latitude/longitude coordinates.
3. The same seven-point matrices after replacing the two spatial axes by an
   integer-grid approximation to the fitted flow-parallel/perpendicular axes,
   including a perpendicular axis in the fitted covariance metric.
4. Mixed space-time rectangle variances, including their known inability to
   distinguish v from -v.

Advection comparators keep every non-advection parameter at its true value:
zero, half speed, double speed, reversed, +45 degree rotation, and +90 degree
rotation.  The fitted lag-4/3/2 model is included as a separate comparator.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.nn import Parameter


HERE = Path(__file__).resolve().parent
REPO = next(parent for parent in HERE.parents if (parent / "src/GEMS_TCO").is_dir())
SRC = REPO / "src"
VECCHIA_APPROX = HERE.parent / "vecchia_approximation"
for path in (HERE, SRC, VECCHIA_APPROX):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import sim_space_time_contrast_diagnostic_090326 as base  # noqa: E402
import vecchia_adapted_fixed_lag643_core as core  # noqa: E402
from GEMS_TCO.vecchia_realdata_corridor_width_4x4_lag432 import (  # noqa: E402
    DirectionalRealDataCorridorWidth4x4Lag432VecchiaFit,
)


DTYPE = torch.float64
PARAMETER_NAMES = base.PARAMETER_NAMES
CONTRAST_MATRIX = base.six_contrast_matrix()
DAY_RE = re.compile(r"day(?P<day>\d+)_hm(?P<hour>\d+):")

DEFAULT_ROOT = Path(
    "/Users/joonwonlee/Documents/GEMS_DATA/simulation/"
    "july_st_circulant_realpattern_smooth0p5"
)
DEFAULT_OUTPUT = HERE / "simulation_nugget1_flow_contrast_diagnostic_20240713_091326"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--synthetic-root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--fit-day", type=int, default=13)
    p.add_argument("--days", nargs="+", type=int, default=[13, 19, 25])
    p.add_argument("--smooth", type=float, default=0.5)
    p.add_argument("--target-chunk-size", type=int, default=64, choices=[64])
    p.add_argument("--lbfgs-lr", type=float, default=1.0)
    p.add_argument("--lbfgs-steps", type=int, default=5)
    p.add_argument("--lbfgs-eval", type=int, default=20)
    p.add_argument("--lbfgs-history", type=int, default=10)
    p.add_argument("--grad-tol", type=float, default=1e-5)
    p.add_argument("--space-scales", nargs="+", type=int, default=[1, 2])
    p.add_argument("--time-scales", nargs="+", type=int, default=[1, 2])
    p.add_argument("--surface-max-lag-cells", type=int, default=6)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--force-refit", action="store_true")
    return p


def loader_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        synthetic_data_root=Path(args.synthetic_root),
        smooth=float(args.smooth),
        truth_nugget=1.0,
        hours_per_day=8,
        lat_range="-3,2",
        lon_range="121,131",
        keep_exact_loc=True,
        empirical_max_lat_offset=20,
        empirical_max_lon_offset=20,
        empirical_min_pair_count=1000,
        empirical_smooth_bandwidth_deg=0.063,
        subgrid_max_condition_number=100.0,
    )


def rotate_vector(lat: float, lon: float, angle_degrees: float) -> tuple[float, float]:
    angle = math.radians(float(angle_degrees))
    cosine, sine = math.cos(angle), math.sin(angle)
    return cosine * lat - sine * lon, sine * lat + cosine * lon


def fit_or_load(
    asset: core.DayAsset,
    seed: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    path = Path(args.output_dir) / "fit_result.json"
    if path.is_file() and not args.force_refit:
        print(f"Reusing fit {path}", flush=True)
        return json.loads(path.read_text(encoding="utf-8"))

    mapped = {
        key: tensor.to(device="cpu", dtype=DTYPE).contiguous()
        for key, tensor in asset.source_map.items()
    }
    model = DirectionalRealDataCorridorWidth4x4Lag432VecchiaFit(
        smooth=float(args.smooth),
        input_map=mapped,
        grid_coords=asset.grid_coords,
        reference_advec_lat=float(seed["seed_lat"]),
        reference_advec_lon=float(seed["seed_lon"]),
        daily_stride=2,
        target_chunk_size=int(args.target_chunk_size),
        min_target_points=1,
    )
    started = time.perf_counter()
    model.precompute_conditioning_sets()
    precompute_s = time.perf_counter() - started

    initial = {
        **{name: float(asset.truth[name]) for name in PARAMETER_NAMES},
        "advec_lat": float(seed["seed_lat"]),
        "advec_lon": float(seed["seed_lon"]),
        "nugget": 1.0,
    }
    raw = core.physical_to_raw(initial)
    parameters = [
        Parameter(torch.tensor(value, dtype=DTYPE, device="cpu"), requires_grad=True)
        for value in raw
    ]
    optimizer = model.set_optimizer(
        parameters,
        lr=float(args.lbfgs_lr),
        max_iter=int(args.lbfgs_eval),
        max_eval=int(args.lbfgs_eval),
        history_size=int(args.lbfgs_history),
    )
    calls = 0
    original_likelihood = model.vecchia_batched_likelihood

    def counted(raw_parameters: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original_likelihood(raw_parameters)

    model.vecchia_batched_likelihood = counted
    started = time.perf_counter()
    returned, step_index = model.fit_vecc_lbfgs(
        parameters,
        optimizer,
        max_steps=int(args.lbfgs_steps),
        grad_tol=float(args.grad_tol),
    )
    fit_s = time.perf_counter() - started
    raw_final = [float(parameter.detach().item()) for parameter in parameters]
    raw_tensor = torch.as_tensor(raw_final, dtype=DTYPE)
    with torch.no_grad():
        final_nll = float(original_likelihood(raw_tensor).detach().item())
        beta = model.get_gls_beta(raw_tensor).detach().cpu().numpy()
    fitted = core.raw_to_physical(raw_final)
    result = {
        "status": "ok",
        "date": asset.date,
        "geometry": "adapted",
        "lag_pattern": "4/3/2",
        "target_chunk_size": int(args.target_chunk_size),
        "smooth": float(args.smooth),
        "nugget_mode": "estimated",
        "truth": {name: float(asset.truth[name]) for name in PARAMETER_NAMES},
        "seed": seed,
        "initial": initial,
        "fitted": {name: float(fitted[name]) for name in PARAMETER_NAMES},
        "errors": {
            f"error_{name}": float(fitted[name] - asset.truth[name])
            for name in PARAMETER_NAMES
        },
        "raw_final": raw_final,
        "gls_beta": beta.tolist(),
        "lat_mean_val": float(model.lat_mean_val),
        "precompute_s": float(precompute_s),
        "fit_s": float(fit_s),
        "precompute_plus_fit_s": float(precompute_s + fit_s),
        "optimizer_likelihood_calls": int(calls),
        "outer_steps": int(step_index) + 1,
        "fit_returned_nll": float(returned[-1]),
        "final_native_nll": float(final_nll),
        "model_summary": model.cluster_summary(),
    }
    base.atomic_json(path, result)
    del model, mapped, parameters, optimizer
    gc.collect()
    return result


def load_month_arrays(
    data_path: Path,
    truth: dict[str, Any],
    selected_days: list[int],
) -> dict[str, Any]:
    data = pd.read_pickle(data_path)
    first = next(iter(data.values()))
    lats = np.sort(pd.to_numeric(first["Latitude"], errors="coerce").unique())
    lons = np.sort(pd.to_numeric(first["Longitude"], errors="coerce").unique())
    lats = lats[np.isfinite(lats)]
    lons = lons[np.isfinite(lons)]
    lat_index = {round(float(value), 8): index for index, value in enumerate(lats)}
    lon_index = {round(float(value), 8): index for index, value in enumerate(lons)}

    parsed: dict[int, list[tuple[int, str]]] = {}
    for key in data:
        match = DAY_RE.search(key)
        if match is None:
            continue
        parsed.setdefault(int(match.group("day")), []).append(
            (int(match.group("hour")), key)
        )
    requested = list(dict.fromkeys(int(day) for day in selected_days))
    days = [day for day in requested if day in parsed and len(parsed[day]) == 8]
    if days != requested:
        missing = [day for day in requested if day not in days]
        raise ValueError(f"Requested days missing or incomplete: {missing}")
    residual = np.full((len(days), 8, len(lats), len(lons)), np.nan, dtype=np.float64)
    valid_counts: list[int] = []
    for day_index, day in enumerate(days):
        keys = sorted(parsed[day])
        for hour_index, (_, key) in enumerate(keys):
            frame = data[key]
            grid_lat = pd.to_numeric(frame["Latitude"], errors="coerce").to_numpy(float)
            grid_lon = pd.to_numeric(frame["Longitude"], errors="coerce").to_numpy(float)
            source_lat = pd.to_numeric(frame["Source_Latitude"], errors="coerce").to_numpy(float)
            y = pd.to_numeric(frame["ColumnAmountO3"], errors="coerce").to_numpy(float)
            valid = np.isfinite(grid_lat) & np.isfinite(grid_lon) & np.isfinite(source_lat) & np.isfinite(y)
            rows = np.asarray([lat_index[round(float(value), 8)] for value in grid_lat[valid]], dtype=np.int64)
            cols = np.asarray([lon_index[round(float(value), 8)] for value in grid_lon[valid]], dtype=np.int64)
            mean = float(truth["mean_intercept"]) + float(truth["mean_lat_slope"]) * (
                source_lat[valid] - float(truth["mean_lat_center"])
            )
            residual[day_index, hour_index, rows, cols] = y[valid] - mean
            valid_counts.append(int(valid.sum()))
    return {
        "days": days,
        "residual": residual,
        "lats": lats,
        "lons": lons,
        "lat_step": float(np.median(np.diff(lats))),
        "lon_step": float(np.median(np.diff(lons))),
        "valid_count_min": int(min(valid_counts)),
        "valid_count_max": int(max(valid_counts)),
        "valid_count_mean": float(np.mean(valid_counts)),
    }


def choose_integer_axis(
    target_lat: float,
    target_lon: float,
    lat_step: float,
    lon_step: float,
    max_cells: int = 8,
) -> tuple[int, int, dict[str, float]]:
    target = np.asarray([target_lat, target_lon], dtype=np.float64)
    target_length = float(np.linalg.norm(target))
    target_unit = target / target_length
    best: tuple[float, int, int, np.ndarray] | None = None
    for dr in range(-max_cells, max_cells + 1):
        for dc in range(-max_cells, max_cells + 1):
            if dr == 0 and dc == 0:
                continue
            physical = np.asarray([dr * lat_step, dc * lon_step], dtype=np.float64)
            length = float(np.linalg.norm(physical))
            dot = float(np.dot(physical / length, target_unit))
            if dot <= 0.0:
                continue
            angular = math.acos(min(max(dot, -1.0), 1.0))
            length_error = math.log(length / target_length)
            score = 4.0 * angular**2 + length_error**2
            candidate = (score, dr, dc, physical)
            if best is None or candidate[0] < best[0]:
                best = candidate
    if best is None:
        raise RuntimeError("No integer flow axis found")
    _, dr, dc, physical = best
    direction_error = math.degrees(
        math.acos(
            min(
                max(float(np.dot(physical / np.linalg.norm(physical), target_unit)), -1.0),
                1.0,
            )
        )
    )
    return dr, dc, {
        "physical_lat": float(physical[0]),
        "physical_lon": float(physical[1]),
        "physical_length": float(np.linalg.norm(physical)),
        "target_length": target_length,
        "direction_error_degrees": float(direction_error),
        "relative_length_error": float(np.linalg.norm(physical) / target_length - 1.0),
    }


def flow_axes(
    fitted: dict[str, float], lat_step: float, lon_step: float
) -> dict[str, Any]:
    v_lat = float(fitted["advec_lat"])
    v_lon = float(fitted["advec_lon"])
    parallel = choose_integer_axis(v_lat, v_lon, lat_step, lon_step)
    speed = math.hypot(v_lat, v_lon)
    perpendicular_target = (-v_lon / speed * speed, v_lat / speed * speed)
    perpendicular = choose_integer_axis(
        perpendicular_target[0], perpendicular_target[1], lat_step, lon_step
    )
    # Orthogonality in the fitted covariance metric.  If x*=(lat/r_lat,
    # lon/r_lon), a physical vector h is perpendicular to v when
    # h_lat*v_lat/r_lat^2 + h_lon*v_lon/r_lon^2 = 0.
    metric_target = np.asarray(
        [
            -v_lon * float(fitted["range_lat"]) ** 2,
            v_lat * float(fitted["range_lon"]) ** 2,
        ],
        dtype=np.float64,
    )
    metric_target *= speed / float(np.linalg.norm(metric_target))
    metric_perpendicular = choose_integer_axis(
        float(metric_target[0]),
        float(metric_target[1]),
        lat_step,
        lon_step,
    )
    dot = (
        parallel[2]["physical_lat"] * perpendicular[2]["physical_lat"]
        + parallel[2]["physical_lon"] * perpendicular[2]["physical_lon"]
    )
    cosine = dot / (
        parallel[2]["physical_length"] * perpendicular[2]["physical_length"]
    )
    return {
        "parallel": {"dr": parallel[0], "dc": parallel[1], **parallel[2]},
        "perpendicular": {
            "dr": perpendicular[0],
            "dc": perpendicular[1],
            **perpendicular[2],
        },
        "metric_perpendicular": {
            "dr": metric_perpendicular[0],
            "dc": metric_perpendicular[1],
            **metric_perpendicular[2],
        },
        "axis_angle_deviation_from_90_degrees": float(
            abs(90.0 - math.degrees(math.acos(min(max(cosine, -1.0), 1.0))))
        ),
    }


def model_candidates(
    truth: dict[str, Any], fit_result: dict[str, Any]
) -> dict[str, dict[str, float]]:
    truth_params = {name: float(truth[name]) for name in PARAMETER_NAMES}
    fitted = {name: float(fit_result["fitted"][name]) for name in PARAMETER_NAMES}
    v_lat = truth_params["advec_lat"]
    v_lon = truth_params["advec_lon"]
    rot45 = rotate_vector(v_lat, v_lon, 45.0)
    rot90 = rotate_vector(v_lat, v_lon, 90.0)
    return {
        "truth": truth_params,
        "fitted_432": fitted,
        "zero": {**truth_params, "advec_lat": 0.0, "advec_lon": 0.0},
        "half_speed": {
            **truth_params,
            "advec_lat": 0.5 * v_lat,
            "advec_lon": 0.5 * v_lon,
        },
        "double_speed": {
            **truth_params,
            "advec_lat": 2.0 * v_lat,
            "advec_lon": 2.0 * v_lon,
        },
        "reversed": {
            **truth_params,
            "advec_lat": -v_lat,
            "advec_lon": -v_lon,
        },
        "rotated_45": {
            **truth_params,
            "advec_lat": rot45[0],
            "advec_lon": rot45[1],
        },
        "rotated_90": {
            **truth_params,
            "advec_lat": rot90[0],
            "advec_lon": rot90[1],
        },
    }


def aligned_slices(length: int, shift: int) -> tuple[slice, slice]:
    if shift >= 0:
        return slice(0, length - shift), slice(shift, length)
    return slice(-shift, length), slice(0, length + shift)


def extract_stencil(
    day: np.ndarray,
    axis1: tuple[int, int],
    axis2: tuple[int, int],
    scale: int,
    tau: int,
) -> np.ndarray:
    shifts = [
        (0, 0, 0),
        (0, axis1[0] * scale, axis1[1] * scale),
        (0, -axis1[0] * scale, -axis1[1] * scale),
        (0, axis2[0] * scale, axis2[1] * scale),
        (0, -axis2[0] * scale, -axis2[1] * scale),
        (tau, 0, 0),
        (-tau, 0, 0),
    ]
    row_margin = max(abs(value[1]) for value in shifts)
    col_margin = max(abs(value[2]) for value in shifts)
    times = np.arange(tau, day.shape[0] - tau)
    rows = np.arange(row_margin, day.shape[1] - row_margin)
    cols = np.arange(col_margin, day.shape[2] - col_margin)
    points = []
    for dt, dr, dc in shifts:
        points.append(day[np.ix_(times + dt, rows + dr, cols + dc)])
    stacked = np.stack(points, axis=-1)
    valid = np.all(np.isfinite(stacked), axis=-1)
    return stacked[valid]


def model_stencil_covariance(
    axis1: tuple[int, int],
    axis2: tuple[int, int],
    scale: int,
    tau: int,
    lat_step: float,
    lon_step: float,
    params: dict[str, float],
    smooth: float,
) -> np.ndarray:
    a1 = np.asarray([axis1[0] * scale * lat_step, axis1[1] * scale * lon_step])
    a2 = np.asarray([axis2[0] * scale * lat_step, axis2[1] * scale * lon_step])
    coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [a1[0], a1[1], 0.0],
            [-a1[0], -a1[1], 0.0],
            [a2[0], a2[1], 0.0],
            [-a2[0], -a2[1], 0.0],
            [0.0, 0.0, float(tau)],
            [0.0, 0.0, -float(tau)],
        ],
        dtype=np.float64,
    )
    difference = coordinates[:, None, :] - coordinates[None, :, :]
    covariance = base.covariance_values(
        difference[..., 0],
        difference[..., 1],
        difference[..., 2],
        params,
        smooth,
        separable=False,
    )
    return CONTRAST_MATRIX @ covariance @ CONTRAST_MATRIX.T


def contrast_names(frame: str) -> tuple[str, ...]:
    if frame == "lat_lon":
        return ("D_lat", "D_lon", "D_time", "Q_lat", "Q_lon", "Q_time")
    return (
        "D_parallel",
        "D_perpendicular",
        "D_time",
        "Q_parallel",
        "Q_perpendicular",
        "Q_time",
    )


def contrast_sector(i: int, j: int) -> str | None:
    if i not in (0, 1, 3, 4) or j not in (2, 5):
        return None
    spatial = "odd" if i in (0, 1) else "even"
    temporal = "odd" if j == 2 else "even"
    return f"{spatial}_{temporal}"


def compute_stencil_diagnostics(
    month: dict[str, Any],
    axes: dict[str, tuple[tuple[int, int], tuple[int, int]]],
    candidates: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_rows: list[dict[str, Any]] = []
    day_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    residual = month["residual"]
    for frame, (axis1, axis2) in axes.items():
        names = contrast_names(frame)
        for scale in sorted(set(args.space_scales)):
            for tau in sorted(set(args.time_scales)):
                moments = []
                counts = []
                for day_index, day_number in enumerate(month["days"]):
                    values = extract_stencil(residual[day_index], axis1, axis2, scale, tau)
                    contrasts = values @ CONTRAST_MATRIX.T
                    moment = contrasts.T @ contrasts / len(contrasts)
                    moments.append(moment)
                    counts.append(len(contrasts))
                    for i in range(6):
                        for j in range(6):
                            day_rows.append(
                                {
                                    "day": day_number,
                                    "frame": frame,
                                    "space_scale": scale,
                                    "tau": tau,
                                    "contrast_i": names[i],
                                    "contrast_j": names[j],
                                    "moment": float(moment[i, j]),
                                    "n_stencils": int(len(contrasts)),
                                }
                            )
                moments_array = np.stack(moments)
                empirical = moments_array.mean(axis=0)
                se = moments_array.std(axis=0, ddof=1) / math.sqrt(len(moments_array))
                model_matrices = {
                    model: model_stencil_covariance(
                        axis1,
                        axis2,
                        scale,
                        tau,
                        month["lat_step"],
                        month["lon_step"],
                        params,
                        float(args.smooth),
                    )
                    for model, params in candidates.items()
                }
                for i in range(6):
                    for j in range(6):
                        row: dict[str, Any] = {
                            "frame": frame,
                            "space_scale": scale,
                            "tau": tau,
                            "contrast_i": names[i],
                            "contrast_j": names[j],
                            "sector": contrast_sector(i, j),
                            "empirical": float(empirical[i, j]),
                            "independent_day_se": float(se[i, j]),
                            "n_days": int(len(moments_array)),
                            "mean_n_stencils_per_day": float(np.mean(counts)),
                        }
                        for model, matrix in model_matrices.items():
                            row[model] = float(matrix[i, j])
                            row[f"z_{model}"] = (
                                float((empirical[i, j] - matrix[i, j]) / se[i, j])
                                if se[i, j] > 0.0
                                else np.nan
                            )
                        full_rows.append(row)
                components = {
                    "axis1_DxDt": [(0, 2)],
                    "axis2_DxDt": [(1, 2)],
                    "odd_odd": [(0, 2), (1, 2)],
                    "even_even": [(3, 5), (4, 5)],
                    "odd_even": [(0, 5), (1, 5)],
                    "even_odd": [(3, 2), (4, 2)],
                    "all_space_time": [
                        (0, 2),
                        (1, 2),
                        (3, 5),
                        (4, 5),
                        (0, 5),
                        (1, 5),
                        (3, 2),
                        (4, 2),
                    ],
                }
                for model, matrix in model_matrices.items():
                    z_matrix = np.divide(
                        empirical - matrix,
                        se,
                        out=np.full_like(empirical, np.nan),
                        where=se > 0,
                    )
                    for component, indexes in components.items():
                        z = np.asarray([z_matrix[i, j] for i, j in indexes])
                        raw = np.asarray([empirical[i, j] - matrix[i, j] for i, j in indexes])
                        summary_rows.append(
                            {
                                "frame": frame,
                                "space_scale": scale,
                                "tau": tau,
                                "model": model,
                                "component": component,
                                "rms_z": float(np.sqrt(np.nanmean(z**2))),
                                "max_abs_z": float(np.nanmax(np.abs(z))),
                                "rmse_raw": float(np.sqrt(np.nanmean(raw**2))),
                            }
                        )
    return pd.DataFrame(full_rows), pd.DataFrame(day_rows), pd.DataFrame(summary_rows)


def aggregate_stencil_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (frame, model, component), group in summary.groupby(
        ["frame", "model", "component"], sort=False
    ):
        values = group["rms_z"].to_numpy(float)
        rows.append(
            {
                "frame": frame,
                "model": model,
                "component": component,
                "rms_z_across_scales": float(np.sqrt(np.mean(values**2))),
                "median_rms_z": float(np.median(values)),
                "max_rms_z": float(np.max(values)),
                "n_scale_pairs": int(len(values)),
            }
        )
    return pd.DataFrame(rows)


def paired_arrays_4d(
    residual: np.ndarray, dr: int, dc: int, tau: int
) -> tuple[np.ndarray, np.ndarray]:
    row_a, row_b = aligned_slices(residual.shape[2], dr)
    col_a, col_b = aligned_slices(residual.shape[3], dc)
    a = residual[:, : residual.shape[1] - tau, row_a, col_a]
    b = residual[:, tau:, row_b, col_b]
    valid = np.isfinite(a) & np.isfinite(b)
    return a, np.where(valid, b, np.nan)


def compute_oriented_surface(
    month: dict[str, Any],
    candidates: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    residual = month["residual"]
    rows = []
    max_lag = int(args.surface_max_lag_cells)
    for tau in sorted(set(args.time_scales)):
        for dr in range(-max_lag, max_lag + 1):
            for dc in range(-max_lag, max_lag + 1):
                row_a, row_b = aligned_slices(residual.shape[2], dr)
                col_a, col_b = aligned_slices(residual.shape[3], dc)
                a = residual[:, : residual.shape[1] - tau, row_a, col_a]
                b = residual[:, tau:, row_b, col_b]
                valid = np.isfinite(a) & np.isfinite(b)
                empirical = float(0.5 * np.mean((b[valid] - a[valid]) ** 2))
                row: dict[str, Any] = {
                    "tau": tau,
                    "dr": dr,
                    "dc": dc,
                    "h_lat": dr * month["lat_step"],
                    "h_lon": dc * month["lon_step"],
                    "n_pairs": int(valid.sum()),
                    "empirical": empirical,
                }
                for model, params in candidates.items():
                    covariance = float(
                        base.covariance_values(
                            np.asarray(row["h_lat"]),
                            np.asarray(row["h_lon"]),
                            float(tau),
                            params,
                            float(args.smooth),
                            separable=False,
                        )
                    )
                    row[model] = float(params["sigmasq"] + params["nugget"] - covariance)
                rows.append(row)
    surface = pd.DataFrame(rows)
    index = surface.set_index(["tau", "dr", "dc"])
    for source in ["empirical", *candidates]:
        surface[f"asymmetry_{source}"] = [
            float(row[source] - index.loc[(row.tau, -row.dr, -row.dc), source])
            for _, row in surface.iterrows()
        ]
    summaries = []
    nonzero = surface[(surface["dr"] != 0) | (surface["dc"] != 0)]
    for model in candidates:
        summaries.append(
            {
                "model": model,
                "semivariogram_rmse": float(
                    np.sqrt(np.mean((nonzero["empirical"] - nonzero[model]) ** 2))
                ),
                "asymmetry_rmse": float(
                    np.sqrt(
                        np.mean(
                            (
                                nonzero["asymmetry_empirical"]
                                - nonzero[f"asymmetry_{model}"]
                            )
                            ** 2
                        )
                    )
                ),
            }
        )
    minima = []
    for tau, group in surface.groupby("tau"):
        for source in ["empirical", *candidates]:
            selected = group.loc[group[source].idxmin()]
            minima.append(
                {
                    "tau": int(tau),
                    "source": source,
                    "dr": int(selected["dr"]),
                    "dc": int(selected["dc"]),
                    "h_lat": float(selected["h_lat"]),
                    "h_lon": float(selected["h_lon"]),
                    "semivariogram_minimum": float(selected[source]),
                }
            )
    return surface, pd.DataFrame(summaries), pd.DataFrame(minima)


def extract_rectangle(
    day: np.ndarray, dr: int, dc: int, tau: int
) -> np.ndarray:
    row_a, row_b = aligned_slices(day.shape[1], dr)
    col_a, col_b = aligned_slices(day.shape[2], dc)
    z00 = day[: day.shape[0] - tau, row_a, col_a]
    zh0 = day[: day.shape[0] - tau, row_b, col_b]
    z0t = day[tau:, row_a, col_a]
    zht = day[tau:, row_b, col_b]
    values = np.stack([z00, zh0, z0t, zht], axis=-1)
    valid = np.all(np.isfinite(values), axis=-1)
    contrast = values[valid] @ np.asarray([0.5, -0.5, -0.5, 0.5])
    return contrast


def model_rectangle_variance(
    dr: int,
    dc: int,
    tau: int,
    lat_step: float,
    lon_step: float,
    params: dict[str, float],
    smooth: float,
) -> float:
    h_lat, h_lon = dr * lat_step, dc * lon_step
    coordinates = np.asarray(
        [[0, 0, 0], [h_lat, h_lon, 0], [0, 0, tau], [h_lat, h_lon, tau]],
        dtype=np.float64,
    )
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    covariance = base.covariance_values(
        differences[..., 0],
        differences[..., 1],
        differences[..., 2],
        params,
        smooth,
        separable=False,
    )
    contrast = np.asarray([0.5, -0.5, -0.5, 0.5])
    return float(contrast @ covariance @ contrast)


def compute_rectangle_diagnostics(
    month: dict[str, Any],
    directions: dict[str, tuple[int, int]],
    candidates: dict[str, dict[str, float]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for direction, (base_dr, base_dc) in directions.items():
        for scale in sorted(set(args.space_scales)):
            for tau in sorted(set(args.time_scales)):
                for sign in (-1, 1):
                    dr, dc = sign * scale * base_dr, sign * scale * base_dc
                    daily = []
                    counts = []
                    for day in month["residual"]:
                        contrast = extract_rectangle(day, dr, dc, tau)
                        daily.append(float(np.mean(contrast**2)))
                        counts.append(len(contrast))
                    empirical = float(np.mean(daily))
                    se = float(np.std(daily, ddof=1) / math.sqrt(len(daily)))
                    row: dict[str, Any] = {
                        "direction": direction,
                        "sign": sign,
                        "space_scale": scale,
                        "tau": tau,
                        "dr": dr,
                        "dc": dc,
                        "empirical_variance": empirical,
                        "independent_day_se": se,
                        "n_days": len(daily),
                        "mean_n_rectangles_per_day": float(np.mean(counts)),
                    }
                    for model, params in candidates.items():
                        value = model_rectangle_variance(
                            dr,
                            dc,
                            tau,
                            month["lat_step"],
                            month["lon_step"],
                            params,
                            float(args.smooth),
                        )
                        row[model] = value
                        row[f"z_{model}"] = (empirical - value) / se
                    rows.append(row)
    frame = pd.DataFrame(rows)
    summaries = []
    for model in candidates:
        for direction, group in frame.groupby("direction"):
            z = group[f"z_{model}"].to_numpy(float)
            summaries.append(
                {
                    "model": model,
                    "direction": direction,
                    "rms_z": float(np.sqrt(np.mean(z**2))),
                    "max_abs_z": float(np.max(np.abs(z))),
                    "sign_pair_model_max_abs_difference": float(
                        max(
                            abs(
                                pair[pair["sign"] == 1][model].iloc[0]
                                - pair[pair["sign"] == -1][model].iloc[0]
                            )
                            for _, pair in group.groupby(["space_scale", "tau"])
                        )
                    ),
                }
            )
    return frame, pd.DataFrame(summaries)


def plot_geometry(
    axes_meta: dict[str, Any],
    truth: dict[str, Any],
    fitted: dict[str, Any],
    output: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.axhline(0, color="0.8", lw=0.8)
    ax.axvline(0, color="0.8", lw=0.8)
    vectors = {
        "truth v": (truth["advec_lon"], truth["advec_lat"], "#111111"),
        "fitted v": (fitted["advec_lon"], fitted["advec_lat"], "#2ca02c"),
        "grid parallel": (
            axes_meta["parallel"]["physical_lon"],
            axes_meta["parallel"]["physical_lat"],
            "#1f77b4",
        ),
        "grid perpendicular": (
            axes_meta["perpendicular"]["physical_lon"],
            axes_meta["perpendicular"]["physical_lat"],
            "#d62728",
        ),
        "grid metric-perpendicular": (
            axes_meta["metric_perpendicular"]["physical_lon"],
            axes_meta["metric_perpendicular"]["physical_lat"],
            "#9467bd",
        ),
    }
    for label, (x, y, color) in vectors.items():
        ax.arrow(0, 0, x, y, width=0.002, length_includes_head=True, color=color)
        ax.plot([], [], color=color, lw=3, label=label)
    ax.set_aspect("equal")
    ax.set_xlabel("longitude displacement per hour")
    ax.set_ylabel("latitude displacement per hour")
    ax.set_title("Fitted-flow coordinate frame on the observation grid")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", frameon=True)
    path = output / "flow_coordinate_geometry.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_stencil_summary(aggregate: pd.DataFrame, output: Path) -> Path:
    models = ["truth", "fitted_432", "zero", "half_speed", "double_speed", "reversed", "rotated_45", "rotated_90"]
    components = ["axis1_DxDt", "axis2_DxDt", "odd_odd", "even_even"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for ax, component in zip(axes.flat, components):
        subset = aggregate[aggregate["component"].eq(component)]
        x = np.arange(len(models))
        width = 0.25
        for offset, frame, color in [
            (-width, "lat_lon", "#7f7f7f"),
            (0.0, "flow", "#1f77b4"),
            (width, "flow_metric", "#9467bd"),
        ]:
            values = [
                float(subset[(subset["frame"] == frame) & (subset["model"] == model)]["rms_z_across_scales"].iloc[0])
                for model in models
            ]
            ax.bar(x + offset, values, width, label=frame, color=color, alpha=0.88)
        ax.set_yscale("log")
        ax.axhline(1.0, color="black", lw=0.8, ls="--")
        ax.set_xticks(x, models, rotation=34, ha="right")
        ax.set_ylabel("RMS z across (space scale, time lag)")
        title = component
        if component == "axis1_DxDt":
            title = "axis 1 odd-odd: lat vs flow-parallel"
        elif component == "axis2_DxDt":
            title = "axis 2 odd-odd: lon vs flow-perpendicular"
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend()
    path = output / "latlon_vs_flow_contrast_mismatch.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_mismatch_panel(
    aggregate: pd.DataFrame,
    surface_summary: pd.DataFrame,
    rectangle_summary: pd.DataFrame,
    output: Path,
) -> Path:
    models = ["truth", "fitted_432", "zero", "half_speed", "double_speed", "reversed", "rotated_45", "rotated_90"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    flow = aggregate[(aggregate["frame"] == "flow_metric") & aggregate["component"].isin(["axis1_DxDt", "axis2_DxDt"])]
    width = 0.36
    x = np.arange(len(models))
    for offset, component, label, color in [
        (-width / 2, "axis1_DxDt", "parallel x time", "#1f77b4"),
        (width / 2, "axis2_DxDt", "perpendicular x time", "#d62728"),
    ]:
        values = [float(flow[(flow["model"] == m) & (flow["component"] == component)]["rms_z_across_scales"].iloc[0]) for m in models]
        axes[0].bar(x + offset, values, width, label=label, color=color)
    axes[0].set_yscale("log")
    axes[0].set_title("Covariance-metric flow odd-odd attribution")
    axes[0].set_ylabel("RMS z")
    axes[0].legend()

    surf = surface_summary.set_index("model")
    axes[1].bar(x - width / 2, [surf.loc[m, "semivariogram_rmse"] for m in models], width, label="level RMSE", color="#2ca02c")
    axes[1].bar(x + width / 2, [surf.loc[m, "asymmetry_rmse"] for m in models], width, label="asymmetry RMSE", color="#9467bd")
    axes[1].set_yscale("log")
    axes[1].set_title("Oriented cross-variogram")
    axes[1].set_ylabel("RMSE")
    axes[1].legend()

    rect = rectangle_summary.groupby("model", as_index=True)["rms_z"].apply(lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2))))
    axes[2].bar(x, [rect.loc[m] for m in models], color="#ff7f0e")
    axes[2].set_yscale("log")
    axes[2].set_title("Mixed-rectangle variance")
    axes[2].set_ylabel("RMS z")
    for ax in axes:
        ax.axhline(1.0, color="black", lw=0.8, ls="--")
        ax.set_xticks(x, models, rotation=38, ha="right")
        ax.grid(axis="y", alpha=0.2)
    path = output / "advection_mismatch_diagnostic_panel.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_surface(
    surface: pd.DataFrame,
    axes_meta: dict[str, Any],
    output: Path,
) -> Path:
    models = ["empirical", "truth", "zero", "reversed", "rotated_90"]
    tau = 1
    selected = surface[surface["tau"] == tau]
    drs = np.sort(selected["dr"].unique())
    dcs = np.sort(selected["dc"].unique())
    fig, axes = plt.subplots(2, len(models), figsize=(17, 7.2), constrained_layout=True)
    for column, source in enumerate(models):
        level = selected.pivot(index="dr", columns="dc", values=source).reindex(index=drs, columns=dcs)
        asym_col = f"asymmetry_{source}"
        asym = selected.pivot(index="dr", columns="dc", values=asym_col).reindex(index=drs, columns=dcs)
        im0 = axes[0, column].imshow(level, origin="lower", extent=[dcs.min() - 0.5, dcs.max() + 0.5, drs.min() - 0.5, drs.max() + 0.5], cmap="viridis")
        axes[0, column].set_title(source)
        fig.colorbar(im0, ax=axes[0, column], shrink=0.72)
        limit = float(np.nanmax(np.abs(asym.to_numpy())))
        im1 = axes[1, column].imshow(asym, origin="lower", extent=[dcs.min() - 0.5, dcs.max() + 0.5, drs.min() - 0.5, drs.max() + 0.5], cmap="coolwarm", vmin=-limit, vmax=limit)
        fig.colorbar(im1, ax=axes[1, column], shrink=0.72)
        for row in (0, 1):
            axes[row, column].set_xlabel("longitude lag (cells)")
            axes[row, column].set_ylabel("latitude lag (cells)")
    axes[0, 0].set_ylabel("cross-variogram level\nlatitude lag (cells)")
    axes[1, 0].set_ylabel("h vs -h asymmetry\nlatitude lag (cells)")
    fig.suptitle("Nugget-1 simulation: oriented cross-variogram, time lag 1")
    path = output / "oriented_cross_variogram_advection_mismatch.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def write_results(
    output: Path,
    truth: dict[str, Any],
    fit_result: dict[str, Any],
    axes_meta: dict[str, Any],
    aggregate: pd.DataFrame,
    surface_summary: pd.DataFrame,
    minima: pd.DataFrame,
    rectangle_summary: pd.DataFrame,
) -> None:
    def stencil_value(frame: str, model: str, component: str) -> float:
        return float(
            aggregate[
                aggregate["frame"].eq(frame)
                & aggregate["model"].eq(model)
                & aggregate["component"].eq(component)
            ]["rms_z_across_scales"].iloc[0]
        )

    def surface_value(model: str, column: str) -> float:
        return float(surface_summary[surface_summary["model"].eq(model)][column].iloc[0])

    rectangle_global = (
        rectangle_summary.groupby("model")["rms_z"]
        .apply(lambda values: float(np.sqrt(np.mean(np.asarray(values) ** 2))))
        .to_dict()
    )
    selected_models = ["truth", "fitted_432", "zero", "half_speed", "double_speed", "reversed", "rotated_45", "rotated_90"]
    table = [
        "| model | metric-flow parallel-time RMS z | metric-flow perpendicular-time RMS z | oriented asymmetry RMSE | rectangle RMS z |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in selected_models:
        table.append(
            f"| {model} | {stencil_value('flow_metric', model, 'axis1_DxDt'):.3f} | "
            f"{stencil_value('flow_metric', model, 'axis2_DxDt'):.3f} | "
            f"{surface_value(model, 'asymmetry_rmse'):.4f} | "
            f"{rectangle_global[model]:.3f} |"
        )
    empirical_minima = minima[minima["source"].eq("empirical")]
    minimum_lines = [
        f"- tau={int(row.tau)}: empirical minimum (dr,dc)=({int(row.dr)},{int(row.dc)}), "
        f"physical lag=({row.h_lat:.3f},{row.h_lon:.3f}); truth displacement="
        f"({truth['advec_lat'] * row.tau:.3f},{truth['advec_lon'] * row.tau:.3f})"
        for row in empirical_minima.itertuples()
    ]
    text = f"""# Nugget-1 flow-coordinate contrast diagnostic

## Data and fit

- Local asset: smooth=0.5 July 2024 simulation, three independent eight-hour days (July 13, 19, 25).
- Confirmed truth: sigma2={truth['sigmasq']}, ranges=({truth['range_lat']},
  {truth['range_lon']}, {truth['range_time']}), advection=({truth['advec_lat']},
  {truth['advec_lon']}), **nugget={truth['nugget']}**.
- One fit only: 2024-07-13, adapted lag 4/3/2, batch size 64, nugget estimated.
- Fitted advection=({fit_result['fitted']['advec_lat']:.6f},
  {fit_result['fitted']['advec_lon']:.6f}), fitted nugget={fit_result['fitted']['nugget']:.6f},
  mean NLL={fit_result['final_native_nll']:.9f}, optimization time={fit_result['fit_s']:.2f} s.
- Empirical contrast moments use the known simulation mean and independent-day
  standard errors across three days.  This isolates covariance/advection
  sensitivity from fitted-mean contamination.

## Fitted-flow grid axes

- Parallel integer offset: (dr,dc)=({axes_meta['parallel']['dr']},{axes_meta['parallel']['dc']}),
  physical=({axes_meta['parallel']['physical_lat']:.3f},{axes_meta['parallel']['physical_lon']:.3f}),
  direction error={axes_meta['parallel']['direction_error_degrees']:.2f} degrees.
- Perpendicular integer offset: (dr,dc)=({axes_meta['perpendicular']['dr']},{axes_meta['perpendicular']['dc']}),
  physical=({axes_meta['perpendicular']['physical_lat']:.3f},{axes_meta['perpendicular']['physical_lon']:.3f}),
  direction error={axes_meta['perpendicular']['direction_error_degrees']:.2f} degrees.
- Deviation between the two implemented grid axes and 90 degrees:
  {axes_meta['axis_angle_deviation_from_90_degrees']:.2f} degrees.
- Covariance-metric perpendicular offset: (dr,dc)=({axes_meta['metric_perpendicular']['dr']},{axes_meta['metric_perpendicular']['dc']}),
  physical=({axes_meta['metric_perpendicular']['physical_lat']:.3f},{axes_meta['metric_perpendicular']['physical_lon']:.3f}),
  direction error={axes_meta['metric_perpendicular']['direction_error_degrees']:.2f} degrees relative to the covariance-metric perpendicular target.

The metric-perpendicular direction is defined by
`h_lat*v_lat/range_lat^2 + h_lon*v_lon/range_lon^2 = 0`.  This is the natural
orthogonality condition after scaling space by the fitted anisotropic ranges;
ordinary Euclidean perpendicularity does not have this property when the two
ranges differ.

## Contrasts being tested

At a center `(s,t)`, the seven-point stencil contains the center, `+/-` two
spatial directions, and `+/-` time.  It is projected onto six zero-sum,
unit-norm contrasts:

- odd first differences: `D_a=[Z(s+h_a,t)-Z(s-h_a,t)]/sqrt(2)` and
  `D_t=[Z(s,t+tau)-Z(s,t-tau)]/sqrt(2)`;
- even second differences: `Q_a=[Z(s+h_a,t)+Z(s-h_a,t)-2Z(s,t)]/sqrt(6)`
  and the analogous `Q_t`.

The diagnostic compares the empirical and model-implied entries of the local
contrast covariance matrix.  In particular, `E[D_a D_t]` is the odd-odd
space-time component: it changes under flow reversal and is the most direct
local advection diagnostic.  `E[Q_a Q_t]` is even-even and measures symmetric
space-time curvature.  The four-point rectangle
`[Z(s,t)-Z(s+h,t)-Z(s,t+tau)+Z(s+h,t+tau)]/2` measures mixed roughness.

The nugget is not being ignored.  It raises the cross-variogram surface and
the rectangle variance.  It cancels from the odd-odd cross-moment because its
spatial and temporal contrasts use disjoint observations, while the shared
center makes it enter the even-even cross-moment.

## Main comparison

{chr(10).join(table)}

The flow-parallel odd-odd component targets speed/sign mismatch along the
transport path.  The flow-perpendicular odd-odd component localizes angular
misalignment.  Their separation is the main gain over reporting latitude and
longitude components only.

For the metric-flow frame, half/double-speed errors give parallel RMS z values
of {stencil_value('flow_metric', 'half_speed', 'axis1_DxDt'):.3f} and
{stencil_value('flow_metric', 'double_speed', 'axis1_DxDt'):.3f}, while their
perpendicular values remain {stencil_value('flow_metric', 'half_speed', 'axis2_DxDt'):.3f}
and {stencil_value('flow_metric', 'double_speed', 'axis2_DxDt'):.3f}.  In
contrast, 45/90-degree rotations increase the perpendicular values to
{stencil_value('flow_metric', 'rotated_45', 'axis2_DxDt'):.3f} and
{stencil_value('flow_metric', 'rotated_90', 'axis2_DxDt'):.3f}.  Thus this
coordinate change provides attribution, not merely a larger omnibus score.

## Cross-variogram minima

{chr(10).join(minimum_lines)}

## Interpretation guardrails

- Oriented cross-variogram asymmetry and odd-odd contrasts retain advection
  sign.  Reversing the velocity should be strongly visible.
- Mixed-rectangle variance measures local joint space-time roughness, but its
  theoretical value is invariant under v -> -v.  It can diagnose missing or
  wrong-speed interaction but cannot identify advection sign by itself.
- The independent-day z scores are sensitivity measures under this known-mean
  simulation, not formal p-values: with only three independent days, their
  standard errors have two degrees of freedom.  A real-data goodness-of-fit test still needs a parametric
  bootstrap that refits the mean and covariance parameters.
"""
    (output / "RESULTS.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parser().parse_args()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(value <= 0 for value in [*args.space_scales, *args.time_scales]):
        raise ValueError("space/time scales must be positive")

    year_dir = Path(args.synthetic_root) / f"{args.year}_july_st_circulant"
    truth_path = year_dir / f"sim_july{args.year}_st_circulant_truth.json"
    data_path = year_dir / f"sim_july{args.year}_st_circulant_gridded.pkl"
    truth_raw = json.loads(truth_path.read_text(encoding="utf-8"))
    if not math.isclose(float(truth_raw["nugget"]), 1.0, abs_tol=1e-12):
        raise ValueError(f"Expected nugget=1 local asset, got {truth_raw['nugget']}")
    print(f"Confirmed local truth nugget={truth_raw['nugget']}", flush=True)

    spec = {
        "dataset_id": f"synthetic_{args.year}07{args.fit_day:02d}",
        "year": int(args.year),
        "month": 7,
        "day": int(args.fit_day),
        "date": f"{args.year}-07-{args.fit_day:02d}",
    }
    fit_asset = core.load_synthetic_asset(spec, loader_args(args))
    seed = core.m3_q3_seed(fit_asset, loader_args(args))
    base.atomic_json(args.output_dir / "initializer.json", seed)
    print(
        f"M3/Q3 seed=({seed['seed_lat']:.6f},{seed['seed_lon']:.6f}); fitting adapted 4/3/2",
        flush=True,
    )
    fit_result = fit_or_load(fit_asset, seed, args)
    print(
        f"Fit advection=({fit_result['fitted']['advec_lat']:.6f},"
        f"{fit_result['fitted']['advec_lon']:.6f}), nugget={fit_result['fitted']['nugget']:.6f}",
        flush=True,
    )

    print(f"Loading selected independent days: {args.days}", flush=True)
    month = load_month_arrays(data_path, truth_raw, args.days)
    axes_meta = flow_axes(
        fit_result["fitted"], month["lat_step"], month["lon_step"]
    )
    base.atomic_json(args.output_dir / "flow_axes.json", axes_meta)
    axes = {
        "lat_lon": ((1, 0), (0, 1)),
        "flow": (
            (axes_meta["parallel"]["dr"], axes_meta["parallel"]["dc"]),
            (
                axes_meta["perpendicular"]["dr"],
                axes_meta["perpendicular"]["dc"],
            ),
        ),
        "flow_metric": (
            (axes_meta["parallel"]["dr"], axes_meta["parallel"]["dc"]),
            (
                axes_meta["metric_perpendicular"]["dr"],
                axes_meta["metric_perpendicular"]["dc"],
            ),
        ),
    }
    candidates = model_candidates(truth_raw, fit_result)
    base.atomic_json(args.output_dir / "model_candidates.json", candidates)

    print("Computing lat/lon and fitted-flow seven-point contrasts", flush=True)
    stencil, daily, stencil_summary = compute_stencil_diagnostics(
        month, axes, candidates, args
    )
    aggregate = aggregate_stencil_summary(stencil_summary)
    base.atomic_csv(args.output_dir / "contrast_matrix_entries.csv", stencil)
    base.atomic_csv(args.output_dir / "contrast_daily_moments.csv", daily)
    base.atomic_csv(args.output_dir / "contrast_scale_summary.csv", stencil_summary)
    base.atomic_csv(args.output_dir / "contrast_aggregate_summary.csv", aggregate)

    print("Computing pooled oriented cross-variogram surfaces", flush=True)
    surface, surface_summary, minima = compute_oriented_surface(
        month, candidates, args
    )
    base.atomic_csv(args.output_dir / "oriented_cross_variogram.csv", surface)
    base.atomic_csv(args.output_dir / "oriented_cross_variogram_summary.csv", surface_summary)
    base.atomic_csv(args.output_dir / "cross_variogram_minima.csv", minima)

    print("Computing mixed rectangle contrasts", flush=True)
    directions = {
        "lat": (1, 0),
        "lon": (0, 1),
        "parallel": axes["flow"][0],
        "perpendicular": axes["flow"][1],
        "metric_perpendicular": axes["flow_metric"][1],
    }
    rectangle, rectangle_summary = compute_rectangle_diagnostics(
        month, directions, candidates, args
    )
    base.atomic_csv(args.output_dir / "mixed_rectangle.csv", rectangle)
    base.atomic_csv(args.output_dir / "mixed_rectangle_summary.csv", rectangle_summary)

    figures = [
        plot_geometry(axes_meta, truth_raw, fit_result["fitted"], args.output_dir),
        plot_stencil_summary(aggregate, args.output_dir),
        plot_mismatch_panel(aggregate, surface_summary, rectangle_summary, args.output_dir),
        plot_surface(surface, axes_meta, args.output_dir),
    ]
    write_results(
        args.output_dir,
        truth_raw,
        fit_result,
        axes_meta,
        aggregate,
        surface_summary,
        minima,
        rectangle_summary,
    )
    config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "truth_path": str(truth_path),
        "data_path": str(data_path),
        "truth": {name: float(truth_raw[name]) for name in PARAMETER_NAMES},
        "fit": fit_result,
        "month": {
            "n_independent_days": len(month["days"]),
            "days": month["days"],
            "grid_shape": list(month["residual"].shape),
            "lat_step": month["lat_step"],
            "lon_step": month["lon_step"],
            "valid_per_hour_min": month["valid_count_min"],
            "valid_per_hour_max": month["valid_count_max"],
            "valid_per_hour_mean": month["valid_count_mean"],
        },
        "flow_axes": axes_meta,
        "settings": {
            "space_scales": sorted(set(args.space_scales)),
            "time_scales": sorted(set(args.time_scales)),
            "selected_days": list(dict.fromkeys(int(day) for day in args.days)),
            "surface_max_lag_cells": int(args.surface_max_lag_cells),
            "independent_day_standard_errors": True,
            "mean_mode": "known_simulation_mean",
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "torch_version": torch.__version__,
        },
        "total_s": time.perf_counter() - started,
        "figures": [str(path) for path in figures],
    }
    base.atomic_json(args.output_dir / "run_config.json", config)
    print(f"Completed in {config['total_s']:.2f} s: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
