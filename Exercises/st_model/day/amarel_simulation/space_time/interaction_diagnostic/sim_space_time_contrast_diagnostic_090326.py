#!/usr/bin/env python3
"""Fit one adapted lag-432 simulation and explore space-time contrasts.

This prototype separates four complementary diagnostics:

1. Oriented cross-variogram surfaces, including their h versus -h asymmetry.
2. A log-correlation interaction surface that is zero for a separable
   correlation function with the same spatial and temporal margins.
3. A six-direction local stencil transformed into odd (central-gradient) and
   even (central-curvature) contrasts.  Its space-time cross block separates
   signed/advection-sensitive odd-odd interaction from symmetric even-even
   interaction.
4. Four-corner mixed rectangle (plaquette) contrasts with coefficients
   (+1,-1,-1,+1).  They annihilate arbitrary additive space-only plus
   time-only means and every affine mean in the joint coordinates.

All contrast coefficients sum to zero.  The fit uses the adapted 4/3/2
corridor, 4x4 target blocks, target_chunk_size=64, float64, and the M3/Q3
advection initializer.  The default simulation has known nugget zero, which is
held fixed while the other six covariance parameters are optimized once.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import gamma, kv
from torch.nn import Parameter


HERE = Path(__file__).resolve().parent
REPO = next(parent for parent in HERE.parents if (parent / "src/GEMS_TCO").is_dir())
SRC = REPO / "src"
VECCHIA_APPROX = HERE.parent / "vecchia_approximation"
for path in (SRC, VECCHIA_APPROX):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import vecchia_adapted_fixed_lag643_core as core  # noqa: E402
from GEMS_TCO.vecchia_realdata_corridor_width_4x4_lag432 import (  # noqa: E402
    DirectionalRealDataCorridorWidth4x4Lag432VecchiaFit,
)


DTYPE = torch.float64
PARAMETER_NAMES = (
    "sigmasq",
    "range_lat",
    "range_lon",
    "range_time",
    "advec_lat",
    "advec_lon",
    "nugget",
)
CONTRAST_NAMES = ("D_lat", "D_lon", "D_time", "Q_lat", "Q_lon", "Q_time")
SPACE_CONTRASTS = ("D_lat", "D_lon", "Q_lat", "Q_lon")
TIME_CONTRASTS = ("D_time", "Q_time")


class AdaptedLag432FixedZeroVecchiaFit(
    core.FixedZeroNuggetMixin,
    DirectionalRealDataCorridorWidth4x4Lag432VecchiaFit,
):
    """Adapted lag-432 graph with statistical nugget fixed at exactly zero."""


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
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
        json.dumps(json_ready(value), indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def parse_int_list(values: list[int]) -> list[int]:
    out = list(dict.fromkeys(int(value) for value in values))
    if not out or any(value <= 0 for value in out):
        raise ValueError("Scale lists must contain positive integers")
    return out


def parse_nonnegative_int_list(values: list[int]) -> list[int]:
    out = list(dict.fromkeys(int(value) for value in values))
    if not out or any(value < 0 for value in out):
        raise ValueError("Lag lists must contain nonnegative integers")
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic-root",
        type=Path,
        default=REPO
        / "outputs/sim_data/july_st_circulant_realpattern_smooth0p5_nugget0_matched5_090226",
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--day", type=int, default=13)
    parser.add_argument("--smooth", type=float, default=0.5)
    parser.add_argument("--target-chunk-size", type=int, default=64, choices=[64])
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-steps", type=int, default=5)
    parser.add_argument("--lbfgs-eval", type=int, default=20)
    parser.add_argument("--lbfgs-history", type=int, default=10)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    parser.add_argument("--surface-max-lag-cells", type=int, default=6)
    parser.add_argument("--surface-time-lags", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--stencil-space-scales", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--stencil-time-scales", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--jackknife-block-cells", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "simulation_space_time_contrast_diagnostic_20240713_090326",
    )
    parser.add_argument("--force-refit", action="store_true")
    return parser


def loader_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        synthetic_data_root=Path(args.synthetic_root),
        smooth=float(args.smooth),
        truth_nugget=0.0,
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


def fit_or_load(
    asset: core.DayAsset,
    seed: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    fit_path = Path(args.output_dir) / "fit_result.json"
    if fit_path.is_file() and not args.force_refit:
        result = json.loads(fit_path.read_text(encoding="utf-8"))
        print(f"Reusing completed fit: {fit_path}", flush=True)
        return result

    mapped = {
        key: tensor.to(device="cpu", dtype=DTYPE).contiguous()
        for key, tensor in asset.source_map.items()
    }
    model = AdaptedLag432FixedZeroVecchiaFit(
        smooth=float(args.smooth),
        input_map=mapped,
        grid_coords=asset.grid_coords,
        reference_advec_lat=float(seed["seed_lat"]),
        reference_advec_lon=float(seed["seed_lon"]),
        daily_stride=2,
        target_chunk_size=int(args.target_chunk_size),
        min_target_points=1,
    )
    precompute_started = time.perf_counter()
    model.precompute_conditioning_sets()
    precompute_s = time.perf_counter() - precompute_started

    physical_init = {
        **asset.truth,
        "advec_lat": float(seed["seed_lat"]),
        "advec_lon": float(seed["seed_lon"]),
        "nugget": 1e-12,
    }
    raw_init = core.physical_to_raw(physical_init)
    params = [
        Parameter(
            torch.tensor(value, dtype=DTYPE, device="cpu"),
            requires_grad=index != 6,
        )
        for index, value in enumerate(raw_init)
    ]
    optimizer = model.set_optimizer(
        params,
        lr=float(args.lbfgs_lr),
        max_iter=int(args.lbfgs_eval),
        max_eval=int(args.lbfgs_eval),
        history_size=int(args.lbfgs_history),
    )
    likelihood_calls = 0
    original_likelihood = model.vecchia_batched_likelihood

    def counted_likelihood(raw_params: torch.Tensor) -> torch.Tensor:
        nonlocal likelihood_calls
        likelihood_calls += 1
        return original_likelihood(raw_params)

    model.vecchia_batched_likelihood = counted_likelihood
    fit_started = time.perf_counter()
    returned, step_index = model.fit_vecc_lbfgs(
        params,
        optimizer,
        max_steps=int(args.lbfgs_steps),
        grad_tol=float(args.grad_tol),
    )
    fit_s = time.perf_counter() - fit_started
    raw_final = [float(param.detach().item()) for param in params]
    raw_tensor = torch.as_tensor(raw_final, dtype=DTYPE)
    with torch.no_grad():
        final_nll = float(original_likelihood(raw_tensor).detach().item())
        beta = model.get_gls_beta(raw_tensor).detach().cpu().numpy()
    fitted = core.raw_to_physical(raw_final)
    fitted["nugget"] = 0.0
    gradients = [
        abs(float(param.grad.detach().item()))
        for param in params
        if param.grad is not None
    ]
    summary = model.cluster_summary()
    errors = {
        f"error_{name}": float(fitted[name] - asset.truth[name])
        for name in PARAMETER_NAMES
    }
    result = {
        "status": "ok",
        "date": asset.date,
        "geometry": "adapted",
        "lag_pattern": "4/3/2",
        "target_chunk_size": int(args.target_chunk_size),
        "smooth": float(args.smooth),
        "dtype": str(DTYPE),
        "device": "cpu",
        "nugget_mode": "fixed_zero",
        "seed": seed,
        "truth": asset.truth,
        "initial": {**physical_init, "nugget": 0.0},
        "fitted": fitted,
        "errors": errors,
        "raw_final": raw_final,
        "gls_beta": beta.tolist(),
        "lat_mean_val": float(model.lat_mean_val),
        "precompute_s": float(precompute_s),
        "fit_s": float(fit_s),
        "precompute_plus_fit_s": float(precompute_s + fit_s),
        "optimizer_likelihood_calls": int(likelihood_calls),
        "outer_steps": int(step_index) + 1,
        "max_abs_gradient": max(gradients) if gradients else np.nan,
        "fit_returned_nll": float(returned[-1]),
        "final_native_nll": float(final_nll),
        "model_summary": summary,
    }
    atomic_json(fit_path, result)
    del model, mapped, optimizer, params
    gc.collect()
    return result


def make_grid_arrays(
    asset: core.DayAsset,
    fit_result: dict[str, Any],
    truth_raw: dict[str, Any],
) -> dict[str, Any]:
    grid = np.asarray(asset.grid_coords, dtype=np.float64)
    lat_key = np.round(grid[:, 0], 8)
    lon_key = np.round(grid[:, 1], 8)
    lats = np.sort(np.unique(lat_key))
    lons = np.sort(np.unique(lon_key))
    lat_lookup = {float(value): index for index, value in enumerate(lats)}
    lon_lookup = {float(value): index for index, value in enumerate(lons)}
    rows = np.asarray([lat_lookup[float(value)] for value in lat_key], dtype=np.int64)
    cols = np.asarray([lon_lookup[float(value)] for value in lon_key], dtype=np.int64)
    shape = (len(asset.keys), len(lats), len(lons))
    fitted_residual = np.full(shape, np.nan, dtype=np.float64)
    truth_residual = np.full(shape, np.nan, dtype=np.float64)
    source_lat = np.full(shape, np.nan, dtype=np.float64)
    source_lon = np.full(shape, np.nan, dtype=np.float64)
    beta = np.asarray(fit_result["gls_beta"], dtype=np.float64).reshape(-1)
    lat_mean_val = float(fit_result["lat_mean_val"])
    for time_index, key in enumerate(asset.keys):
        tensor = asset.source_map[key].detach().cpu().numpy()
        design = np.column_stack(
            [
                np.ones(tensor.shape[0], dtype=np.float64),
                tensor[:, 0] - lat_mean_val,
                tensor[:, 4:11],
            ]
        )
        y_centered = np.asarray(tensor[:, 2], dtype=np.float64).reshape(-1)
        fitted_values = y_centered - design @ beta
        truth_mean_centered = (
            float(truth_raw["mean_intercept"])
            + float(truth_raw["mean_lat_slope"])
            * (tensor[:, 0] - float(truth_raw["mean_lat_center"]))
            - float(asset.center_value)
        )
        truth_values = y_centered - truth_mean_centered
        fitted_residual[time_index, rows, cols] = fitted_values
        truth_residual[time_index, rows, cols] = truth_values
        source_lat[time_index, rows, cols] = tensor[:, 0]
        source_lon[time_index, rows, cols] = tensor[:, 1]
    return {
        "lats": lats,
        "lons": lons,
        "lat_step": float(np.median(np.diff(lats))),
        "lon_step": float(np.median(np.diff(lons))),
        "fitted_gls": fitted_residual,
        "truth_mean": truth_residual,
        "source_lat": source_lat,
        "source_lon": source_lon,
    }


def matern_correlation(distance: np.ndarray, smooth: float) -> np.ndarray:
    distance = np.asarray(distance, dtype=np.float64)
    if math.isclose(float(smooth), 0.5, rel_tol=0.0, abs_tol=1e-12):
        return np.exp(-distance)
    if math.isclose(float(smooth), 1.5, rel_tol=0.0, abs_tol=1e-12):
        return (1.0 + distance) * np.exp(-distance)
    out = np.ones_like(distance)
    positive = distance > 0.0
    argument = distance[positive]
    constant = 2.0 ** (1.0 - float(smooth)) / gamma(float(smooth))
    out[positive] = constant * argument ** float(smooth) * kv(float(smooth), argument)
    return out


def covariance_values(
    dlat: np.ndarray,
    dlon: np.ndarray,
    dt: np.ndarray | float,
    params: dict[str, float],
    smooth: float,
    separable: bool = False,
) -> np.ndarray:
    dlat = np.asarray(dlat, dtype=np.float64)
    dlon = np.asarray(dlon, dtype=np.float64)
    dt_array = np.asarray(dt, dtype=np.float64)
    if separable:
        spatial_distance = np.sqrt(
            (dlat / float(params["range_lat"])) ** 2
            + (dlon / float(params["range_lon"])) ** 2
        )
        temporal_distance = np.abs(dt_array) / float(params["range_time"])
        correlation = matern_correlation(spatial_distance, smooth) * matern_correlation(
            temporal_distance, smooth
        )
    else:
        shifted_lat = dlat - float(params["advec_lat"]) * dt_array
        shifted_lon = dlon - float(params["advec_lon"]) * dt_array
        distance = np.sqrt(
            (shifted_lat / float(params["range_lat"])) ** 2
            + (shifted_lon / float(params["range_lon"])) ** 2
            + (dt_array / float(params["range_time"])) ** 2
        )
        correlation = matern_correlation(distance, smooth)
    covariance = float(params["sigmasq"]) * correlation
    same = (
        np.isclose(dlat, 0.0, rtol=0.0, atol=1e-12)
        & np.isclose(dlon, 0.0, rtol=0.0, atol=1e-12)
        & np.isclose(dt_array, 0.0, rtol=0.0, atol=1e-12)
    )
    return covariance + same * float(params.get("nugget", 0.0))


def model_candidates(fit_result: dict[str, Any]) -> dict[str, tuple[dict[str, float], bool]]:
    truth = {name: float(fit_result["truth"][name]) for name in PARAMETER_NAMES}
    fitted = {name: float(fit_result["fitted"][name]) for name in PARAMETER_NAMES}
    no_advection = {**fitted, "advec_lat": 0.0, "advec_lon": 0.0}
    reversed_advection = {
        **fitted,
        "advec_lat": -float(fitted["advec_lat"]),
        "advec_lon": -float(fitted["advec_lon"]),
    }
    return {
        "truth": (truth, False),
        "fitted": (fitted, False),
        "no_advection": (no_advection, False),
        "reversed_advection": (reversed_advection, False),
        "separable_same_margins": (fitted, True),
    }


def aligned_slices(length: int, shift: int) -> tuple[slice, slice]:
    if shift >= 0:
        return slice(0, length - shift), slice(shift, length)
    return slice(-shift, length), slice(0, length + shift)


def pair_arrays(
    array: np.ndarray,
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    dr: int,
    dc: int,
    tau: int,
) -> tuple[np.ndarray, ...]:
    time_a = slice(0, array.shape[0] - tau if tau else array.shape[0])
    time_b = slice(tau, array.shape[0])
    row_a, row_b = aligned_slices(array.shape[1], dr)
    col_a, col_b = aligned_slices(array.shape[2], dc)
    index_a = (time_a, row_a, col_a)
    index_b = (time_b, row_b, col_b)
    a = array[index_a]
    b = array[index_b]
    lat_a = source_lat[index_a]
    lat_b = source_lat[index_b]
    lon_a = source_lon[index_a]
    lon_b = source_lon[index_b]
    valid = (
        np.isfinite(a)
        & np.isfinite(b)
        & np.isfinite(lat_a)
        & np.isfinite(lat_b)
        & np.isfinite(lon_a)
        & np.isfinite(lon_b)
    )
    return (
        a[valid],
        b[valid],
        (lat_b - lat_a)[valid],
        (lon_b - lon_a)[valid],
    )


def compute_surfaces(
    grid_arrays: dict[str, Any],
    candidates: dict[str, tuple[dict[str, float], bool]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    max_lag = int(args.surface_max_lag_cells)
    time_lags = sorted(set([0, *parse_nonnegative_int_list(args.surface_time_lags)]))
    lat_step = float(grid_arrays["lat_step"])
    lon_step = float(grid_arrays["lon_step"])
    source_lat = grid_arrays["source_lat"]
    source_lon = grid_arrays["source_lon"]
    for residual_mode in ("fitted_gls", "truth_mean"):
        residual = grid_arrays[residual_mode]
        empirical_variance = float(np.nanmean(residual**2))
        for tau in time_lags:
            for dr in range(-max_lag, max_lag + 1):
                for dc in range(-max_lag, max_lag + 1):
                    a, b, dlat, dlon = pair_arrays(
                        residual, source_lat, source_lon, dr, dc, tau
                    )
                    if a.size == 0:
                        continue
                    empirical_covariance = float(np.mean(a * b))
                    empirical_semivariogram = float(0.5 * np.mean((b - a) ** 2))
                    common = {
                        "residual_mode": residual_mode,
                        "tau": int(tau),
                        "dr": int(dr),
                        "dc": int(dc),
                        "h_lat": float(dr * lat_step),
                        "h_lon": float(dc * lon_step),
                        "n_pairs": int(a.size),
                    }
                    records.append(
                        {
                            **common,
                            "source": "empirical",
                            "covariance": empirical_covariance,
                            "correlation": empirical_covariance / empirical_variance,
                            "semivariogram": empirical_semivariogram,
                        }
                    )
                    for source, (params, separable) in candidates.items():
                        covariance = float(
                            np.mean(
                                covariance_values(
                                    dlat,
                                    dlon,
                                    float(tau),
                                    params,
                                    float(args.smooth),
                                    separable=separable,
                                )
                            )
                        )
                        variance = float(params["sigmasq"] + params.get("nugget", 0.0))
                        records.append(
                            {
                                **common,
                                "source": source,
                                "covariance": covariance,
                                "correlation": covariance / variance,
                                "semivariogram": variance - covariance,
                            }
                        )
    frame = pd.DataFrame(records)
    key_columns = ["residual_mode", "source", "tau", "dr", "dc"]
    lookup = frame.set_index(key_columns)
    interaction: list[float] = []
    asymmetry: list[float] = []
    for row in frame.itertuples(index=False):
        base_key = (row.residual_mode, row.source)
        rho = float(row.correlation)
        rho_space = float(lookup.loc[(*base_key, 0, row.dr, row.dc), "correlation"])
        rho_time = float(lookup.loc[(*base_key, row.tau, 0, 0), "correlation"])
        if row.tau == 0 or (row.dr == 0 and row.dc == 0):
            interaction.append(0.0)
        elif min(rho, rho_space, rho_time) <= 0.0:
            interaction.append(np.nan)
        else:
            interaction.append(float(np.log(rho) - np.log(rho_space) - np.log(rho_time)))
        opposite = float(
            lookup.loc[
                (*base_key, row.tau, -int(row.dr), -int(row.dc)), "semivariogram"
            ]
        )
        asymmetry.append(float(row.semivariogram - opposite))
    frame["log_correlation_interaction"] = interaction
    frame["cross_variogram_asymmetry"] = asymmetry
    return frame


def summarize_surfaces(surface: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for residual_mode in surface["residual_mode"].unique():
        empirical = surface[
            surface["residual_mode"].eq(residual_mode)
            & surface["source"].eq("empirical")
            & surface["tau"].gt(0)
        ].set_index(["tau", "dr", "dc"])
        for source in [value for value in surface["source"].unique() if value != "empirical"]:
            model = surface[
                surface["residual_mode"].eq(residual_mode)
                & surface["source"].eq(source)
                & surface["tau"].gt(0)
            ].set_index(["tau", "dr", "dc"])
            common = empirical.index.intersection(model.index)
            emp = empirical.loc[common]
            mod = model.loc[common]
            non_axis = np.asarray(
                [(dr != 0 or dc != 0) for _, dr, dc in common], dtype=bool
            )
            interaction_valid = (
                non_axis
                & np.isfinite(emp["log_correlation_interaction"].to_numpy())
                & np.isfinite(mod["log_correlation_interaction"].to_numpy())
            )
            rows.append(
                {
                    "residual_mode": residual_mode,
                    "model": source,
                    "semivariogram_rmse": float(
                        np.sqrt(np.mean((emp["semivariogram"] - mod["semivariogram"]) ** 2))
                    ),
                    "asymmetry_rmse": float(
                        np.sqrt(
                            np.mean(
                                (
                                    emp.loc[non_axis, "cross_variogram_asymmetry"]
                                    - mod.loc[non_axis, "cross_variogram_asymmetry"]
                                )
                                ** 2
                            )
                        )
                    ),
                    "log_interaction_rmse": float(
                        np.sqrt(
                            np.mean(
                                (
                                    emp.loc[interaction_valid, "log_correlation_interaction"]
                                    - mod.loc[interaction_valid, "log_correlation_interaction"]
                                )
                                ** 2
                            )
                        )
                    ),
                    "n_surface_cells": int(len(common)),
                }
            )
    return pd.DataFrame(rows)


def six_contrast_matrix() -> np.ndarray:
    # Point order: center, +lat, -lat, +lon, -lon, +time, -time.
    matrix = np.zeros((6, 7), dtype=np.float64)
    root2 = math.sqrt(2.0)
    root6 = math.sqrt(6.0)
    matrix[0, [1, 2]] = [1.0 / root2, -1.0 / root2]
    matrix[1, [3, 4]] = [1.0 / root2, -1.0 / root2]
    matrix[2, [5, 6]] = [1.0 / root2, -1.0 / root2]
    matrix[3, [0, 1, 2]] = [-2.0 / root6, 1.0 / root6, 1.0 / root6]
    matrix[4, [0, 3, 4]] = [-2.0 / root6, 1.0 / root6, 1.0 / root6]
    matrix[5, [0, 5, 6]] = [-2.0 / root6, 1.0 / root6, 1.0 / root6]
    return matrix


def extract_six_stencils(
    residual: np.ndarray,
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    space_cells: int,
    tau: int,
    block_cells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_slice = slice(tau, residual.shape[0] - tau)
    r_slice = slice(space_cells, residual.shape[1] - space_cells)
    c_slice = slice(space_cells, residual.shape[2] - space_cells)
    point_indexes = (
        (t_slice, r_slice, c_slice),
        (t_slice, slice(2 * space_cells, residual.shape[1]), c_slice),
        (t_slice, slice(0, residual.shape[1] - 2 * space_cells), c_slice),
        (t_slice, r_slice, slice(2 * space_cells, residual.shape[2])),
        (t_slice, r_slice, slice(0, residual.shape[2] - 2 * space_cells)),
        (slice(2 * tau, residual.shape[0]), r_slice, c_slice),
        (slice(0, residual.shape[0] - 2 * tau), r_slice, c_slice),
    )
    values = np.stack([residual[index] for index in point_indexes], axis=-1)
    lats = np.stack([source_lat[index] for index in point_indexes], axis=-1)
    lons = np.stack([source_lon[index] for index in point_indexes], axis=-1)
    time_base = np.arange(tau, residual.shape[0] - tau)[:, None, None]
    times = np.stack(
        [
            np.broadcast_to(time_base, values.shape[:-1]),
            np.broadcast_to(time_base, values.shape[:-1]),
            np.broadcast_to(time_base, values.shape[:-1]),
            np.broadcast_to(time_base, values.shape[:-1]),
            np.broadcast_to(time_base, values.shape[:-1]),
            np.broadcast_to(time_base + tau, values.shape[:-1]),
            np.broadcast_to(time_base - tau, values.shape[:-1]),
        ],
        axis=-1,
    )
    rr = np.arange(space_cells, residual.shape[1] - space_cells)[None, :, None]
    cc = np.arange(space_cells, residual.shape[2] - space_cells)[None, None, :]
    group = (
        np.broadcast_to(rr // block_cells, values.shape[:-1])
        * (math.ceil(residual.shape[2] / block_cells) + 1)
        + np.broadcast_to(cc // block_cells, values.shape[:-1])
    )
    valid = (
        np.all(np.isfinite(values), axis=-1)
        & np.all(np.isfinite(lats), axis=-1)
        & np.all(np.isfinite(lons), axis=-1)
    )
    coordinates = np.stack([lats[valid], lons[valid], times[valid]], axis=-1)
    return values[valid], coordinates, group[valid].astype(np.int64)


def clustered_second_moment(
    contrast_values: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_contrasts = contrast_values.shape[1]
    products = np.einsum("ni,nj->nij", contrast_values, contrast_values).reshape(
        len(contrast_values), -1
    )
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    group_sums = np.zeros((len(unique_groups), products.shape[1]), dtype=np.float64)
    np.add.at(group_sums, inverse, products)
    group_counts = np.bincount(inverse).astype(np.int64)
    total_sum = group_sums.sum(axis=0)
    total_count = int(group_counts.sum())
    moment = (total_sum / total_count).reshape(n_contrasts, n_contrasts)
    leave = (total_sum[None, :] - group_sums) / (
        total_count - group_counts[:, None]
    )
    leave_mean = leave.mean(axis=0)
    variance = (len(unique_groups) - 1.0) / len(unique_groups) * np.sum(
        (leave - leave_mean) ** 2, axis=0
    )
    se = np.sqrt(np.maximum(variance, 0.0)).reshape(n_contrasts, n_contrasts)
    return moment, se, contrast_values.mean(axis=0)


def average_model_contrast_covariance(
    coordinates: np.ndarray,
    contrast_matrix: np.ndarray,
    params: dict[str, float],
    smooth: float,
    separable: bool,
    chunk_size: int = 20000,
) -> np.ndarray:
    total = np.zeros((contrast_matrix.shape[0], contrast_matrix.shape[0]), dtype=np.float64)
    count = 0
    for start in range(0, len(coordinates), chunk_size):
        points = coordinates[start : start + chunk_size]
        differences = points[:, :, None, :] - points[:, None, :, :]
        covariance = covariance_values(
            differences[..., 0],
            differences[..., 1],
            differences[..., 2],
            params,
            smooth,
            separable=separable,
        )
        transformed = np.einsum(
            "ai,nij,bj->nab", contrast_matrix, covariance, contrast_matrix, optimize=True
        )
        total += transformed.sum(axis=0)
        count += len(points)
    return total / max(count, 1)


def compute_six_direction_diagnostics(
    grid_arrays: dict[str, Any],
    candidates: dict[str, tuple[dict[str, float], bool]],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix = six_contrast_matrix()
    records: list[dict[str, Any]] = []
    mean_records: list[dict[str, Any]] = []
    for residual_mode in ("fitted_gls", "truth_mean"):
        residual = grid_arrays[residual_mode]
        for space_cells in parse_int_list(args.stencil_space_scales):
            for tau in parse_int_list(args.stencil_time_scales):
                values, coordinates, groups = extract_six_stencils(
                    residual,
                    grid_arrays["source_lat"],
                    grid_arrays["source_lon"],
                    space_cells,
                    tau,
                    int(args.jackknife_block_cells),
                )
                contrasts = values @ matrix.T
                empirical, se, contrast_mean = clustered_second_moment(contrasts, groups)
                model_matrices = {
                    name: average_model_contrast_covariance(
                        coordinates,
                        matrix,
                        params,
                        float(args.smooth),
                        separable,
                    )
                    for name, (params, separable) in candidates.items()
                }
                for index, name in enumerate(CONTRAST_NAMES):
                    mean_records.append(
                        {
                            "residual_mode": residual_mode,
                            "space_cells": space_cells,
                            "tau": tau,
                            "contrast": name,
                            "empirical_mean": float(contrast_mean[index]),
                            "n_stencils": int(len(values)),
                            "n_spatial_blocks": int(len(np.unique(groups))),
                        }
                    )
                for i, name_i in enumerate(CONTRAST_NAMES):
                    for j, name_j in enumerate(CONTRAST_NAMES):
                        row: dict[str, Any] = {
                            "residual_mode": residual_mode,
                            "space_cells": int(space_cells),
                            "tau": int(tau),
                            "contrast_i": name_i,
                            "contrast_j": name_j,
                            "empirical": float(empirical[i, j]),
                            "jackknife_se": float(se[i, j]),
                            "n_stencils": int(len(values)),
                            "n_spatial_blocks": int(len(np.unique(groups))),
                        }
                        for source, model_matrix in model_matrices.items():
                            value = float(model_matrix[i, j])
                            row[source] = value
                            row[f"z_{source}"] = (
                                float((empirical[i, j] - value) / se[i, j])
                                if se[i, j] > 0.0
                                else np.nan
                            )
                        records.append(row)
    return pd.DataFrame(records), pd.DataFrame(mean_records)


def interaction_sector(name_i: str, name_j: str) -> str | None:
    if name_i not in SPACE_CONTRASTS or name_j not in TIME_CONTRASTS:
        return None
    spatial_parity = "odd" if name_i.startswith("D_") else "even"
    temporal_parity = "odd" if name_j.startswith("D_") else "even"
    return f"{spatial_parity}_{temporal_parity}"


def summarize_six_direction(
    frame: pd.DataFrame,
    candidate_names: list[str],
) -> pd.DataFrame:
    subset = frame.copy()
    subset["sector"] = [
        interaction_sector(name_i, name_j)
        for name_i, name_j in zip(subset["contrast_i"], subset["contrast_j"])
    ]
    subset = subset[subset["sector"].notna()]
    rows: list[dict[str, Any]] = []
    group_columns = ["residual_mode", "space_cells", "tau"]
    for keys, group in subset.groupby(group_columns, sort=False):
        for source in candidate_names:
            for sector in ("odd_odd", "even_even", "odd_even", "even_odd", "all_space_time"):
                selected = group if sector == "all_space_time" else group[group["sector"].eq(sector)]
                z = selected[f"z_{source}"].to_numpy(dtype=np.float64)
                raw = (
                    selected["empirical"].to_numpy(dtype=np.float64)
                    - selected[source].to_numpy(dtype=np.float64)
                )
                rows.append(
                    {
                        **dict(zip(group_columns, keys)),
                        "model": source,
                        "sector": sector,
                        "n_entries": int(len(selected)),
                        "rms_z": float(np.sqrt(np.nanmean(z**2))),
                        "max_abs_z": float(np.nanmax(np.abs(z))),
                        "rmse_raw": float(np.sqrt(np.nanmean(raw**2))),
                    }
                )
    return pd.DataFrame(rows)


def extract_rectangle(
    residual: np.ndarray,
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    dr: int,
    dc: int,
    tau: int,
    block_cells: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    row_a, row_b = aligned_slices(residual.shape[1], dr)
    col_a, col_b = aligned_slices(residual.shape[2], dc)
    indexes = (
        (slice(0, residual.shape[0] - tau), row_a, col_a),
        (slice(0, residual.shape[0] - tau), row_b, col_b),
        (slice(tau, residual.shape[0]), row_a, col_a),
        (slice(tau, residual.shape[0]), row_b, col_b),
    )
    values = np.stack([residual[index] for index in indexes], axis=-1)
    lats = np.stack([source_lat[index] for index in indexes], axis=-1)
    lons = np.stack([source_lon[index] for index in indexes], axis=-1)
    time_base = np.arange(residual.shape[0] - tau)[:, None, None]
    times = np.stack(
        [
            np.broadcast_to(time_base, values.shape[:-1]),
            np.broadcast_to(time_base, values.shape[:-1]),
            np.broadcast_to(time_base + tau, values.shape[:-1]),
            np.broadcast_to(time_base + tau, values.shape[:-1]),
        ],
        axis=-1,
    )
    base_rows = np.arange(residual.shape[1])[row_a][None, :, None]
    base_cols = np.arange(residual.shape[2])[col_a][None, None, :]
    group = (
        np.broadcast_to(base_rows // block_cells, values.shape[:-1])
        * (math.ceil(residual.shape[2] / block_cells) + 1)
        + np.broadcast_to(base_cols // block_cells, values.shape[:-1])
    )
    valid = (
        np.all(np.isfinite(values), axis=-1)
        & np.all(np.isfinite(lats), axis=-1)
        & np.all(np.isfinite(lons), axis=-1)
    )
    coordinates = np.stack([lats[valid], lons[valid], times[valid]], axis=-1)
    return values[valid], coordinates, group[valid].astype(np.int64)


def compute_rectangle_diagnostics(
    grid_arrays: dict[str, Any],
    candidates: dict[str, tuple[dict[str, float], bool]],
    args: argparse.Namespace,
) -> pd.DataFrame:
    contrast = np.asarray([[0.5, -0.5, -0.5, 0.5]], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for residual_mode in ("fitted_gls", "truth_mean"):
        residual = grid_arrays[residual_mode]
        for space_cells in parse_int_list(args.stencil_space_scales):
            for tau in parse_int_list(args.stencil_time_scales):
                for axis, unit_dr, unit_dc in (("lat", 1, 0), ("lon", 0, 1)):
                    for sign in (-1, 1):
                        dr = sign * unit_dr * space_cells
                        dc = sign * unit_dc * space_cells
                        values, coordinates, groups = extract_rectangle(
                            residual,
                            grid_arrays["source_lat"],
                            grid_arrays["source_lon"],
                            dr,
                            dc,
                            tau,
                            int(args.jackknife_block_cells),
                        )
                        contrast_values = values @ contrast.T
                        empirical, se, mean = clustered_second_moment(
                            contrast_values, groups
                        )
                        row: dict[str, Any] = {
                            "residual_mode": residual_mode,
                            "axis": axis,
                            "sign": int(sign),
                            "space_cells": int(space_cells),
                            "tau": int(tau),
                            "empirical_variance": float(empirical[0, 0]),
                            "jackknife_se": float(se[0, 0]),
                            "empirical_mean": float(mean[0]),
                            "n_rectangles": int(len(values)),
                            "n_spatial_blocks": int(len(np.unique(groups))),
                        }
                        for source, (params, separable) in candidates.items():
                            model_value = float(
                                average_model_contrast_covariance(
                                    coordinates,
                                    contrast,
                                    params,
                                    float(args.smooth),
                                    separable,
                                )[0, 0]
                            )
                            row[source] = model_value
                            row[f"z_{source}"] = (
                                float((empirical[0, 0] - model_value) / se[0, 0])
                                if se[0, 0] > 0.0
                                else np.nan
                            )
                        rows.append(row)
    return pd.DataFrame(rows)


def matrix_from_rows(frame: pd.DataFrame, column: str) -> np.ndarray:
    lookup = frame.set_index(["contrast_i", "contrast_j"])[column]
    return np.asarray(
        [[lookup.loc[(row, col)] for col in CONTRAST_NAMES] for row in CONTRAST_NAMES],
        dtype=np.float64,
    )


def covariance_to_correlation(covariance: np.ndarray) -> np.ndarray:
    scale = np.sqrt(np.maximum(np.diag(covariance), 1e-15))
    return covariance / scale[:, None] / scale[None, :]


def plot_six_direction(frame: pd.DataFrame, output_dir: Path) -> Path:
    subset = frame[
        frame["residual_mode"].eq("fitted_gls")
        & frame["space_cells"].eq(2)
        & frame["tau"].eq(1)
    ]
    panels = (
        ("empirical", "Empirical contrast correlation", True),
        ("truth", "Truth contrast correlation", True),
        ("fitted", "Fitted contrast correlation", True),
        ("z_fitted", "Z residual: fitted", False),
        ("z_no_advection", "Z residual: no advection", False),
        ("z_separable_same_margins", "Z residual: separable margins", False),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0), constrained_layout=True)
    for axis, (column, title, is_covariance) in zip(axes.flat, panels):
        values = matrix_from_rows(subset, column)
        if is_covariance:
            values = covariance_to_correlation(values)
            limit = 1.0
            label = "correlation"
        else:
            limit = max(3.0, float(np.nanquantile(np.abs(values), 0.95)))
            label = "standardized residual"
        image = axis.imshow(values, cmap="RdBu_r", vmin=-limit, vmax=limit)
        axis.set_xticks(range(len(CONTRAST_NAMES)), CONTRAST_NAMES, rotation=45, ha="right")
        axis.set_yticks(range(len(CONTRAST_NAMES)), CONTRAST_NAMES)
        axis.set_title(title)
        fig.colorbar(image, ax=axis, shrink=0.82, label=label)
        axis.axvline(2.5, color="black", lw=0.8)
        axis.axhline(2.5, color="black", lw=0.8)
    fig.suptitle(
        "Six-direction odd/even contrast diagnostic (h=2 grid cells, tau=1)",
        fontsize=14,
    )
    path = output_dir / "six_direction_contrast_diagnostic.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def surface_grid(
    frame: pd.DataFrame,
    residual_mode: str,
    source: str,
    tau: int,
    column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = frame[
        frame["residual_mode"].eq(residual_mode)
        & frame["source"].eq(source)
        & frame["tau"].eq(tau)
    ]
    pivot = selected.pivot(index="dr", columns="dc", values=column).sort_index().sort_index(axis=1)
    return pivot.index.to_numpy(), pivot.columns.to_numpy(), pivot.to_numpy()


def plot_surface_comparison(surface: pd.DataFrame, output_dir: Path) -> Path:
    fig, axes = plt.subplots(3, 4, figsize=(17.5, 12.0), constrained_layout=True)
    row_specs = (
        ("semivariogram", "Cross-variogram", "viridis"),
        ("cross_variogram_asymmetry", "h vs -h asymmetry", "RdBu_r"),
        ("log_correlation_interaction", "Log-correlation interaction", "RdBu_r"),
    )
    column_specs = (
        ("empirical", 1, "Empirical, tau=1"),
        ("fitted", 1, "Fitted, tau=1"),
        ("empirical", 2, "Empirical, tau=2"),
        ("fitted", 2, "Fitted, tau=2"),
    )
    for row_index, (value_column, row_title, cmap) in enumerate(row_specs):
        panel_values = []
        for source, tau, _ in column_specs:
            _, _, values = surface_grid(surface, "fitted_gls", source, tau, value_column)
            panel_values.append(values)
        if cmap == "RdBu_r":
            limit = max(float(np.nanquantile(np.abs(value), 0.98)) for value in panel_values)
            vmin, vmax = -limit, limit
        else:
            vmin = min(float(np.nanmin(value)) for value in panel_values)
            vmax = max(float(np.nanmax(value)) for value in panel_values)
        for column_index, ((source, tau, title), values) in enumerate(
            zip(column_specs, panel_values)
        ):
            dr, dc, _ = surface_grid(surface, "fitted_gls", source, tau, value_column)
            axis = axes[row_index, column_index]
            image = axis.imshow(
                values,
                origin="lower",
                extent=[dc.min(), dc.max(), dr.min(), dr.max()],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="equal",
            )
            axis.set_title(f"{title}\n{row_title}")
            axis.set_xlabel("longitude lag (cells)")
            axis.set_ylabel("latitude lag (cells)")
            axis.axhline(0, color="0.25", lw=0.5)
            axis.axvline(0, color="0.25", lw=0.5)
            fig.colorbar(image, ax=axis, shrink=0.78)
    fig.suptitle("Oriented space-time surfaces after fitted GLS mean removal", fontsize=15)
    path = output_dir / "cross_variogram_and_interaction_surfaces.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_sector_scores(summary: pd.DataFrame, output_dir: Path) -> Path:
    selected = summary[
        summary["residual_mode"].eq("fitted_gls")
        & summary["sector"].isin(["odd_odd", "even_even", "all_space_time"])
    ].copy()
    selected["scale"] = [
        f"h={space}, t={tau}"
        for space, tau in zip(selected["space_cells"], selected["tau"])
    ]
    scales = list(dict.fromkeys(selected["scale"]))
    models = ["truth", "fitted", "no_advection", "reversed_advection", "separable_same_margins"]
    colors = {
        "truth": "#2ca02c",
        "fitted": "#1f77b4",
        "no_advection": "#ff7f0e",
        "reversed_advection": "#d62728",
        "separable_same_margins": "#9467bd",
    }
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 4.8), constrained_layout=True)
    for axis, sector in zip(axes, ("odd_odd", "even_even", "all_space_time")):
        sector_frame = selected[selected["sector"].eq(sector)]
        for model in models:
            lookup = sector_frame[sector_frame["model"].eq(model)].set_index("scale")["rms_z"]
            axis.plot(
                range(len(scales)),
                [lookup.get(scale, np.nan) for scale in scales],
                marker="o",
                lw=1.5,
                color=colors[model],
                label=model,
            )
        axis.axhline(2.0, color="0.5", ls="--", lw=0.8)
        axis.set_xticks(range(len(scales)), scales, rotation=45, ha="right")
        axis.set_ylabel("RMS standardized residual")
        axis.set_title(sector.replace("_", " "))
        axis.grid(alpha=0.2)
    axes[-1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Which contrast sector detects which misspecification?", fontsize=14)
    path = output_dir / "contrast_sector_sensitivity.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_method_note(output_dir: Path) -> None:
    text = """# Space-time contrast diagnostic prototype

The coefficients of every implemented linear contrast sum to zero.  In
intrinsic-random-function terminology these are authorized linear combinations
of order 0 (ALC-0).  The even central second differences and the four-corner
mixed rectangle additionally annihilate affine trends (ALC-1).

## What each family targets

- Oriented cross-variogram: broad, intuitive mixed-lag check.  Its h versus -h
  difference targets directional time asymmetry/advection, but the raw level
  still mixes spatial, temporal, and joint dependence.
- Log-correlation interaction: removes fitted/empirical pure-space and
  pure-time correlation margins.  It is zero under multiplicative
  separability, so it targets general nonseparability rather than advection
  alone.  It is nonlinear in empirical covariance estimates.
- Six-direction odd/even basis: D_lat/D_lon crossed with D_time is the signed
  odd-odd sector and is most sensitive to advection.  Q_lat/Q_lon crossed with
  Q_time is the symmetric even-even interaction sector.  Odd-even and
  even-odd sectors are useful leakage/stationarity checks.
- Mixed rectangle: the tensor-product first difference (+1,-1,-1,+1) removes
  any additive space-only plus time-only mean exactly.  Its variance measures
  joint local roughness; comparing +h and -h versions restores directional
  information.

A single six-neighbor 3-D Laplacian is retained only implicitly through the Q
contrasts.  Collapsing Q_lat + Q_lon + Q_time to one number would mix marginal
spatial curvature, marginal temporal curvature, and interaction, so it is not
recommended as the primary interaction diagnostic.

Jackknife standard errors treat 8x8 spatial tiles as clusters.  They are useful
for ranking diagnostic sensitivity in this prototype, but neighboring tiles
remain correlated.  Production inference should use multiple independent days
or a larger moving-block/bootstrap calibration.
"""
    (output_dir / "METHOD.md").write_text(text, encoding="utf-8")


def main() -> None:
    workflow_started = time.perf_counter()
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if int(args.surface_max_lag_cells) < 1:
        raise ValueError("surface-max-lag-cells must be positive")
    parse_nonnegative_int_list(args.surface_time_lags)
    parse_int_list(args.stencil_space_scales)
    parse_int_list(args.stencil_time_scales)
    if int(args.jackknife_block_cells) < 2:
        raise ValueError("jackknife-block-cells must be at least 2")

    date = f"{args.year:04d}-{args.month:02d}-{args.day:02d}"
    spec = {
        "dataset_id": f"synthetic_{args.year:04d}{args.month:02d}{args.day:02d}",
        "year": int(args.year),
        "month": int(args.month),
        "day": int(args.day),
        "date": date,
    }
    local_loader_args = loader_args(args)
    load_started = time.perf_counter()
    print(f"Loading synthetic day {date}", flush=True)
    asset = core.load_synthetic_asset(spec, local_loader_args)
    truth_path = Path(asset.source_path).with_name(
        f"sim_july{args.year}_st_circulant_truth.json"
    )
    truth_raw = json.loads(truth_path.read_text(encoding="utf-8"))
    seed = core.m3_q3_seed(asset, local_loader_args)
    load_and_initialize_s = time.perf_counter() - load_started
    atomic_json(args.output_dir / "initializer.json", seed)
    print(
        f"M3/Q3 seed=({seed['seed_lat']:.6f}, {seed['seed_lon']:.6f}); "
        f"truth=({asset.truth['advec_lat']:.6f}, {asset.truth['advec_lon']:.6f})",
        flush=True,
    )

    fit_result = fit_or_load(asset, seed, args)
    print(
        f"Fit NLL={fit_result['final_native_nll']:.10f}; "
        f"fit_s={fit_result['fit_s']:.3f}; fitted advection="
        f"({fit_result['fitted']['advec_lat']:.6f}, "
        f"{fit_result['fitted']['advec_lon']:.6f})",
        flush=True,
    )
    residual_started = time.perf_counter()
    grid_arrays = make_grid_arrays(asset, fit_result, truth_raw)
    candidates = model_candidates(fit_result)
    residual_build_s = time.perf_counter() - residual_started

    surface_started = time.perf_counter()
    print("Computing oriented surfaces", flush=True)
    surfaces = compute_surfaces(grid_arrays, candidates, args)
    surface_summary = summarize_surfaces(surfaces)
    atomic_csv(args.output_dir / "oriented_surface_values.csv", surfaces)
    atomic_csv(args.output_dir / "oriented_surface_summary.csv", surface_summary)
    surface_diagnostic_s = time.perf_counter() - surface_started

    six_started = time.perf_counter()
    print("Computing six-direction odd/even contrast matrices", flush=True)
    six_frame, contrast_means = compute_six_direction_diagnostics(
        grid_arrays, candidates, args
    )
    six_summary = summarize_six_direction(six_frame, list(candidates))
    atomic_csv(args.output_dir / "six_direction_contrast_matrices.csv", six_frame)
    atomic_csv(args.output_dir / "six_direction_contrast_means.csv", contrast_means)
    atomic_csv(args.output_dir / "six_direction_sector_summary.csv", six_summary)
    six_direction_diagnostic_s = time.perf_counter() - six_started

    rectangle_started = time.perf_counter()
    print("Computing mixed rectangle contrasts", flush=True)
    rectangle = compute_rectangle_diagnostics(grid_arrays, candidates, args)
    atomic_csv(args.output_dir / "mixed_rectangle_contrasts.csv", rectangle)
    rectangle_diagnostic_s = time.perf_counter() - rectangle_started

    figure_started = time.perf_counter()
    figure_paths = [
        plot_surface_comparison(surfaces, args.output_dir),
        plot_six_direction(six_frame, args.output_dir),
        plot_sector_scores(six_summary, args.output_dir),
    ]
    write_method_note(args.output_dir)
    figure_and_note_s = time.perf_counter() - figure_started
    config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "source_path": asset.source_path,
        "truth_path": str(truth_path),
        "date": date,
        "n_valid": int(asset.n_valid),
        "n_total": int(asset.n_total),
        "grid_shape": list(grid_arrays["fitted_gls"].shape),
        "lat_step": grid_arrays["lat_step"],
        "lon_step": grid_arrays["lon_step"],
        "fit": {
            "geometry": "adapted",
            "lag_pattern": "4/3/2",
            "block_shape": [4, 4],
            "target_chunk_size": int(args.target_chunk_size),
            "smooth": float(args.smooth),
            "dtype": str(DTYPE),
            "nugget_mode": "fixed_zero",
            "lbfgs_lr": float(args.lbfgs_lr),
            "lbfgs_steps": int(args.lbfgs_steps),
            "lbfgs_eval": int(args.lbfgs_eval),
            "lbfgs_history": int(args.lbfgs_history),
            "grad_tol": float(args.grad_tol),
        },
        "diagnostics": {
            "surface_max_lag_cells": int(args.surface_max_lag_cells),
            "surface_time_lags": sorted(
                set([0, *parse_nonnegative_int_list(args.surface_time_lags)])
            ),
            "stencil_space_scales": parse_int_list(args.stencil_space_scales),
            "stencil_time_scales": parse_int_list(args.stencil_time_scales),
            "jackknife_block_cells": int(args.jackknife_block_cells),
            "residual_modes": ["fitted_gls", "truth_mean"],
            "model_candidates": list(candidates),
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "torch_version": torch.__version__,
        },
        "timings_s": {
            "load_and_initialize": load_and_initialize_s,
            "fit_precompute": float(fit_result["precompute_s"]),
            "fit_optimize": float(fit_result["fit_s"]),
            "fit_precompute_plus_optimize": float(
                fit_result["precompute_plus_fit_s"]
            ),
            "residual_build": residual_build_s,
            "surface_diagnostic": surface_diagnostic_s,
            "six_direction_diagnostic": six_direction_diagnostic_s,
            "rectangle_diagnostic": rectangle_diagnostic_s,
            "figures_and_method_note": figure_and_note_s,
            "current_process_total": time.perf_counter() - workflow_started,
        },
        "figures": [str(path) for path in figure_paths],
    }
    atomic_json(args.output_dir / "run_config.json", config)
    print(f"Saved diagnostic outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
