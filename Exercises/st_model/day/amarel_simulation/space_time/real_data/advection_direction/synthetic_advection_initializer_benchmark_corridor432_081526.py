#!/usr/bin/env python3
"""Synthetic benchmark for fast advection initializers (081526).

This benchmark answers two separate questions.

Stage A -- initializer accuracy
    Compare M0--M5 against known synthetic advection while nuisance covariance
    parameters are oracle-fixed, jointly misspecified, or only range_time is
    free.  The likelihood methods optimize only advection except in the named
    partial-free ablation.

Stage B -- downstream fit
    Use each oracle-nuisance seed to build one fixed directional lag-432
    corridor, then fit all seven covariance parameters on the full benchmark
    domain with identical nuisance starts and optimizer budgets.

Methods
-------
M0_zero
    Zero-advection baseline.
M1_pilot400
    400 spatial max-min points/hour; one L-BFGS call with max_eval=20.
M2_quadrant300
    300 points/hour; four requested quadrant starts, max_eval=5 each; keep the
    fitted advection having the smallest re-evaluated NLL.
M3_empirical
    Pair-count-filtered, Gaussian-smoothed tau=1 cross-semivariogram ridge.
M4_polar_profile
    Profile 8 angles x 3 magnitudes on 300 points/hour, then refine the best
    candidate with one max_eval=5 advection-only L-BFGS call.
M5_oracle
    True synthetic advection, providing an upper-bound seed/corridor.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torch.nn import Parameter


LOCAL_REPO = Path("/Users/joonwonlee/Documents/GEMS_TCO-1")
AMAREL_REPO = Path("/home/jl2815/tco")
REPO = AMAREL_REPO if AMAREL_REPO.exists() else LOCAL_REPO
SRC = REPO if (REPO / "GEMS_TCO").exists() else REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from GEMS_TCO import orderings
from GEMS_TCO.matern_vecchia_engine import fit_vecchia_lbfgs as PointwisePilotVecchia
from GEMS_TCO.vecchia_realdata_corridor_width_4x4_lag432 import (
    BLOCK_SHAPE,
    DIRECTIONAL_SPEC_NAME,
    LAG_COUNTS,
    build_directional_model,
)


METHODS = (
    "M0_zero",
    "M1_pilot400",
    "M2_quadrant300",
    "M3_empirical",
    "M4_polar_profile",
    "M5_oracle",
)
QUADRANT_STARTS = (
    (0.01, 0.10),
    (0.01, -0.10),
    (-0.01, -0.10),
    (-0.01, 0.10),
)
NUISANCE_FACTORS = {
    "oracle_fixed": 1.00,
    "nuisance_minus50": 0.50,
    "nuisance_minus25": 0.75,
    "nuisance_plus25": 1.25,
    "nuisance_plus50": 1.50,
}
P_LABELS = (
    "sigmasq",
    "range_lat",
    "range_lon",
    "range_time",
    "advec_lat",
    "advec_lon",
    "nugget",
)


@dataclass
class DayAsset:
    year: int
    month: int
    day_idx: int
    day: str
    source_map: dict[str, torch.Tensor]
    grid_coords: np.ndarray
    n_valid: int
    n_total: int


def parse_pair(text: str, cast=float) -> list[Any]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected two comma-separated values, got {text!r}")
    return [cast(parts[0]), cast(parts[1])]


def parse_days(text: str) -> list[int]:
    token = str(text).strip().lower()
    parts = [part.strip() for part in token.split(",") if part.strip()]
    if len(parts) == 2:
        start, stop = (int(x) for x in parts)
        return list(range(start, stop))
    return [int(x) for x in parts]


def clean_json(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    return value


def physical_to_raw(params: dict[str, float]) -> list[float]:
    range_lon = float(params["range_lon"])
    phi2 = 1.0 / range_lon
    return [
        float(np.log(float(params["sigmasq"]) * phi2)),
        float(np.log(phi2)),
        float(np.log((range_lon / float(params["range_lat"])) ** 2)),
        float(np.log((range_lon / float(params["range_time"])) ** 2)),
        float(params["advec_lat"]),
        float(params["advec_lon"]),
        float(np.log(float(params["nugget"]))),
    ]


def raw_to_physical(raw: Sequence[float]) -> dict[str, float]:
    values = [float(x) for x in raw[:7]]
    phi2 = float(np.exp(values[1]))
    range_lon = 1.0 / phi2
    return {
        "sigmasq": float(np.exp(values[0]) / phi2),
        "range_lat": float(range_lon / np.sqrt(np.exp(values[2]))),
        "range_lon": range_lon,
        "range_time": float(range_lon / np.sqrt(np.exp(values[3]))),
        "advec_lat": values[4],
        "advec_lon": values[5],
        "nugget": float(np.exp(values[6])),
    }


def nuisance_from_truth(truth: dict[str, float], factor: float) -> dict[str, float]:
    out = dict(truth)
    for key in ("sigmasq", "range_lat", "range_lon", "range_time", "nugget"):
        out[key] = float(truth[key]) * float(factor)
    return out


def vector_metrics(lat: float, lon: float, truth: dict[str, float]) -> dict[str, float]:
    true_lat = float(truth["advec_lat"])
    true_lon = float(truth["advec_lon"])
    error = float(np.hypot(lat - true_lat, lon - true_lon))
    norm = float(np.hypot(lat, lon))
    true_norm = float(np.hypot(true_lat, true_lon))
    if norm <= 1e-12 or true_norm <= 1e-12:
        angle_error = np.nan
    else:
        dot = np.clip((lat * true_lat + lon * true_lon) / (norm * true_norm), -1.0, 1.0)
        angle_error = float(np.degrees(np.arccos(dot)))
    return {
        "seed_error_euclid": error,
        "seed_angle_error_deg": angle_error,
        "seed_norm": norm,
        "true_advec_norm": true_norm,
        "seed_norm_abs_error": abs(norm - true_norm),
        "correct_quadrant": bool(np.sign(lat) == np.sign(true_lat) and np.sign(lon) == np.sign(true_lon)),
    }


def count_valid(source_map: dict[str, torch.Tensor]) -> tuple[int, int]:
    total = sum(int(value.shape[0]) for value in source_map.values())
    valid = sum(int((~torch.isnan(value[:, 2])).sum().item()) for value in source_map.values())
    return valid, total


def load_assets(
    input_path: Path,
    truth: dict[str, Any],
    day_indices: Sequence[int],
    lat_range: Sequence[float],
    lon_range: Sequence[float],
    keep_exact_loc: bool,
) -> list[DayAsset]:
    obj = pd.read_pickle(input_path)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict pickle, got {type(obj)}")
    keys = sorted(obj)
    selected_all = sorted({hour for day in day_indices for hour in range(day * 8, (day + 1) * 8)})
    needed_keys = [keys[idx] for idx in selected_all]

    filtered: dict[str, pd.DataFrame] = {}
    for key in needed_keys:
        frame = obj[key]
        mask = frame["Latitude"].between(lat_range[0], lat_range[1]) & frame["Longitude"].between(
            lon_range[0], lon_range[1]
        )
        filtered[key] = frame.loc[mask].reset_index(drop=True)
    ozone = [pd.to_numeric(frame["ColumnAmountO3"], errors="coerce").to_numpy() for frame in filtered.values()]
    center = float(np.nanmean(np.concatenate(ozone)))
    first_coords = filtered[needed_keys[0]][["Latitude", "Longitude"]].to_numpy(dtype=np.float64)

    assets: list[DayAsset] = []
    for day_idx in day_indices:
        day_keys = keys[day_idx * 8 : (day_idx + 1) * 8]
        source_map: dict[str, torch.Tensor] = {}
        for local_time, key in enumerate(day_keys):
            frame = filtered[key]
            grid_lat = pd.to_numeric(frame["Latitude"], errors="coerce").to_numpy(dtype=np.float64)
            grid_lon = pd.to_numeric(frame["Longitude"], errors="coerce").to_numpy(dtype=np.float64)
            if keep_exact_loc:
                lat = pd.to_numeric(frame["Source_Latitude"], errors="coerce").to_numpy(dtype=np.float64)
                lon = pd.to_numeric(frame["Source_Longitude"], errors="coerce").to_numpy(dtype=np.float64)
                lat = np.where(np.isfinite(lat), lat, grid_lat)
                lon = np.where(np.isfinite(lon), lon, grid_lon)
            else:
                lat, lon = grid_lat, grid_lon
            y = pd.to_numeric(frame["ColumnAmountO3"], errors="coerce").to_numpy(dtype=np.float64) - center
            base = np.column_stack([lat, lon, y, np.full(len(frame), float(local_time))])
            base_tensor = torch.from_numpy(base).to(dtype=torch.double)
            dummies = F.one_hot(torch.tensor([local_time]), num_classes=8).repeat(len(frame), 1)[:, 1:].to(torch.double)
            source_map[key] = torch.cat([base_tensor, dummies], dim=1).contiguous()
        valid, total = count_valid(source_map)
        assets.append(
            DayAsset(
                year=int(truth["year"]),
                month=7,
                day_idx=int(day_idx),
                day=f"{int(truth['year'])}-07-{day_idx + 1:02d}",
                source_map=source_map,
                grid_coords=first_coords.copy(),
                n_valid=valid,
                n_total=total,
            )
        )
    del obj
    gc.collect()
    return assets


def make_subset_map(
    source_map: dict[str, torch.Tensor],
    maxmin_order: np.ndarray,
    n_points: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    keep = torch.as_tensor(maxmin_order[:n_points], dtype=torch.long)
    return {
        key: value.index_select(0, keep).to(device=device, dtype=torch.double).contiguous()
        for key, value in source_map.items()
    }


def build_pilot_model(
    asset: DayAsset,
    maxmin_order: np.ndarray,
    ordered_grid_coords: np.ndarray,
    n_points: int,
    nns_map: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[PointwisePilotVecchia, float]:
    subset_map = make_subset_map(asset.source_map, maxmin_order, n_points, device)
    model = PointwisePilotVecchia(
        smooth=float(args.smooth),
        input_map=subset_map,
        nns_map=nns_map[:n_points],
        mm_cond_number=int(args.pilot_neighbors),
        nheads=int(args.pilot_nheads),
        limit_A=int(args.pilot_limit_a),
        limit_B=int(args.pilot_limit_b),
        limit_C=int(args.pilot_limit_c),
        daily_stride=int(args.daily_stride),
    )
    started = time.perf_counter()
    model.precompute_conditioning_sets()
    return model, float(time.perf_counter() - started)


def parameter_tensor(
    base_raw: Sequence[float],
    advec_lat: torch.Tensor | float,
    advec_lon: torch.Tensor | float,
    device: torch.device,
    range_time_raw: torch.Tensor | None = None,
) -> torch.Tensor:
    values = [torch.as_tensor(value, device=device, dtype=torch.double) for value in base_raw]
    values[4] = torch.as_tensor(advec_lat, device=device, dtype=torch.double).reshape(())
    values[5] = torch.as_tensor(advec_lon, device=device, dtype=torch.double).reshape(())
    if range_time_raw is not None:
        values[3] = range_time_raw.reshape(())
    return torch.stack([value.reshape(()) for value in values])


def evaluate_seed_nll(
    model,
    nuisance: dict[str, float],
    lat: float,
    lon: float,
    device: torch.device,
    range_time_raw: float | None = None,
) -> float:
    raw = physical_to_raw(nuisance)
    with torch.no_grad():
        params = parameter_tensor(raw, lat, lon, device=device)
        if range_time_raw is not None:
            params[3] = float(range_time_raw)
        loss = model.vecchia_batched_likelihood(params)
    return float(loss.detach().cpu().item())


def fit_advection_only(
    model,
    nuisance: dict[str, float],
    start_lat: float,
    start_lon: float,
    max_eval: int,
    args: argparse.Namespace,
    device: torch.device,
    free_range_time: bool = False,
    range_time_start: float | None = None,
) -> dict[str, Any]:
    base = dict(nuisance)
    if range_time_start is not None:
        base["range_time"] = float(range_time_start)
    base_raw = physical_to_raw(base)
    advec_lat = Parameter(torch.tensor(float(start_lat), dtype=torch.double, device=device))
    advec_lon = Parameter(torch.tensor(float(start_lon), dtype=torch.double, device=device))
    optimizer_params: list[Parameter] = [advec_lat, advec_lon]
    range_time_raw = None
    if free_range_time:
        range_time_raw = Parameter(torch.tensor(base_raw[3], dtype=torch.double, device=device))
        optimizer_params.append(range_time_raw)
    optimizer = torch.optim.LBFGS(
        optimizer_params,
        lr=float(args.lbfgs_lr),
        max_iter=int(max_eval),
        max_eval=int(max_eval),
        history_size=min(int(args.lbfgs_history), int(max_eval)),
        line_search_fn="strong_wolfe",
        tolerance_grad=float(args.grad_tol),
    )
    closure_calls = 0

    def closure():
        nonlocal closure_calls
        closure_calls += 1
        optimizer.zero_grad()
        params = parameter_tensor(
            base_raw,
            advec_lat,
            advec_lon,
            device=device,
            range_time_raw=range_time_raw,
        )
        loss = model.vecchia_batched_likelihood(params)
        loss.backward()
        return loss

    started = time.perf_counter()
    optimizer.step(closure)
    elapsed = time.perf_counter() - started
    final_raw = list(base_raw)
    final_raw[4] = float(advec_lat.detach().cpu().item())
    final_raw[5] = float(advec_lon.detach().cpu().item())
    if range_time_raw is not None:
        final_raw[3] = float(range_time_raw.detach().cpu().item())
    with torch.no_grad():
        final_nll = float(
            model.vecchia_batched_likelihood(
                torch.as_tensor(final_raw, device=device, dtype=torch.double)
            ).detach().cpu().item()
        )
    est = raw_to_physical(final_raw)
    return {
        "seed_lat": est["advec_lat"],
        "seed_lon": est["advec_lon"],
        "selection_nll": final_nll,
        "search_s": float(elapsed),
        "closure_calls": int(closure_calls),
        "profile_evals": 0,
        "est_range_time_seed_stage": est["range_time"],
    }


def run_quadrant(
    model,
    nuisance: dict[str, float],
    args: argparse.Namespace,
    device: torch.device,
    free_range_time: bool,
    range_time_start: float | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates: list[dict[str, Any]] = []
    total_s = 0.0
    total_closures = 0
    for idx, (lat, lon) in enumerate(QUADRANT_STARTS, start=1):
        fit = fit_advection_only(
            model,
            nuisance,
            lat,
            lon,
            int(args.quadrant_max_eval_each),
            args,
            device,
            free_range_time=free_range_time,
            range_time_start=range_time_start,
        )
        fit.update({"candidate": f"q{idx}", "start_lat": lat, "start_lon": lon})
        candidates.append(fit)
        total_s += float(fit["search_s"])
        total_closures += int(fit["closure_calls"])
    chosen = min(candidates, key=lambda row: float(row["selection_nll"]))
    result = dict(chosen)
    result["search_s"] = total_s
    result["closure_calls"] = total_closures
    result["selected_candidate"] = chosen["candidate"]
    return result, candidates


def polar_candidates(magnitudes: Sequence[float], n_angles: int) -> list[tuple[str, float, float]]:
    out = []
    for magnitude in magnitudes:
        for idx, theta in enumerate(np.linspace(0.0, 2.0 * np.pi, int(n_angles), endpoint=False)):
            lat = float(magnitude * np.sin(theta))
            lon = float(magnitude * np.cos(theta))
            out.append((f"r{magnitude:g}_a{idx:02d}", lat, lon))
    return out


def run_polar_profile(
    model,
    nuisance: dict[str, float],
    args: argparse.Namespace,
    device: torch.device,
    free_range_time: bool,
    range_time_start: float | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    profile_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for label, lat, lon in polar_candidates(args.polar_magnitudes, args.polar_angles):
        nll = evaluate_seed_nll(model, nuisance, lat, lon, device)
        profile_rows.append(
            {"candidate": label, "start_lat": lat, "start_lon": lon, "profile_nll": nll}
        )
    profile_s = time.perf_counter() - started
    best = min(profile_rows, key=lambda row: float(row["profile_nll"]))
    refined = fit_advection_only(
        model,
        nuisance,
        float(best["start_lat"]),
        float(best["start_lon"]),
        int(args.polar_refine_max_eval),
        args,
        device,
        free_range_time=free_range_time,
        range_time_start=range_time_start,
    )
    refined["search_s"] = float(profile_s + refined["search_s"])
    refined["profile_evals"] = len(profile_rows)
    refined["selected_candidate"] = best["candidate"]
    refined["selected_profile_nll"] = best["profile_nll"]
    return refined, profile_rows


def make_hourly_grids(
    source_map: dict[str, torch.Tensor],
    grid_coords: np.ndarray,
) -> tuple[list[np.ndarray], float, float]:
    lat_key = np.round(grid_coords[:, 0], 6)
    lon_key = np.round(grid_coords[:, 1], 6)
    lats = np.sort(np.unique(lat_key))
    lons = np.sort(np.unique(lon_key))
    lat_lookup = {float(value): idx for idx, value in enumerate(lats)}
    lon_lookup = {float(value): idx for idx, value in enumerate(lons)}
    rows = np.asarray([lat_lookup[float(value)] for value in lat_key], dtype=np.int64)
    cols = np.asarray([lon_lookup[float(value)] for value in lon_key], dtype=np.int64)
    grids = []
    for key in sorted(source_map):
        values = source_map[key][:, 2].detach().cpu().numpy()
        grid = np.full((len(lats), len(lons)), np.nan, dtype=np.float64)
        grid[rows, cols] = values
        grid -= float(np.nanmean(grid))
        grids.append(grid)
    return grids, float(np.median(np.diff(lats))), float(np.median(np.diff(lons)))


def shifted_sum_half_sq(cur: np.ndarray, nxt: np.ndarray, di: int, dj: int) -> tuple[float, int]:
    nlat, nlon = cur.shape
    cur_i = slice(0, nlat - di) if di >= 0 else slice(-di, nlat)
    nxt_i = slice(di, nlat) if di >= 0 else slice(0, nlat + di)
    cur_j = slice(0, nlon - dj) if dj >= 0 else slice(-dj, nlon)
    nxt_j = slice(dj, nlon) if dj >= 0 else slice(0, nlon + dj)
    left, right = cur[cur_i, cur_j], nxt[nxt_i, nxt_j]
    mask = np.isfinite(left) & np.isfinite(right)
    count = int(mask.sum())
    if count == 0:
        return 0.0, 0
    diff = left[mask] - right[mask]
    return float(0.5 * np.sum(diff * diff)), count


def nan_gaussian_filter(array: np.ndarray, sigma: tuple[float, float]) -> np.ndarray:
    mask = np.isfinite(array)
    values = gaussian_filter(np.where(mask, array, 0.0), sigma=sigma, mode="nearest")
    weights = gaussian_filter(mask.astype(float), sigma=sigma, mode="nearest")
    out = values / np.maximum(weights, 1e-12)
    out[weights <= 1e-12] = np.nan
    return out


def empirical_seed(asset: DayAsset, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    grids, lat_step, lon_step = make_hourly_grids(asset.source_map, asset.grid_coords)
    offsets_lat = np.arange(-int(args.empirical_max_lat_offset), int(args.empirical_max_lat_offset) + 1)
    offsets_lon = np.arange(-int(args.empirical_max_lon_offset), int(args.empirical_max_lon_offset) + 1)
    sums = np.zeros((len(offsets_lat), len(offsets_lon)), dtype=np.float64)
    counts = np.zeros_like(sums, dtype=np.int64)
    for hour in range(len(grids) - 1):
        for row, di in enumerate(offsets_lat):
            for col, dj in enumerate(offsets_lon):
                value, count = shifted_sum_half_sq(grids[hour], grids[hour + 1], int(di), int(dj))
                sums[row, col] += value
                counts[row, col] += count
    gamma = np.full_like(sums, np.nan)
    valid = counts > 0
    gamma[valid] = sums[valid] / counts[valid]
    filtered = np.where(counts >= int(args.empirical_min_pair_count), gamma, np.nan)
    smoothed = nan_gaussian_filter(
        filtered,
        (
            float(args.empirical_smooth_bandwidth_deg) / abs(lat_step),
            float(args.empirical_smooth_bandwidth_deg) / abs(lon_step),
        ),
    )
    if not np.isfinite(smoothed).any():
        raise RuntimeError("No finite empirical ridge cell after pair-count filtering")
    row, col = np.unravel_index(int(np.nanargmin(smoothed)), smoothed.shape)
    gamma_min = float(smoothed[row, col])
    tolerance = max(
        float(args.empirical_near_min_abs_tol),
        float(args.empirical_near_min_rel_tol) * abs(gamma_min),
    )
    near_count = int((np.isfinite(smoothed) & (smoothed <= gamma_min + tolerance)).sum())
    return {
        "seed_lat": float(offsets_lat[row] * lat_step),
        "seed_lon": float(offsets_lon[col] * lon_step),
        "selection_nll": np.nan,
        "search_s": float(time.perf_counter() - started),
        "closure_calls": 0,
        "profile_evals": int(smoothed.size),
        "selected_candidate": "tau1_ridge",
        "ridge_gamma_min": gamma_min,
        "ridge_near_min_count": near_count,
        "ridge_is_ambiguous": bool(near_count >= int(args.empirical_ambiguous_min_near_cells)),
        "ridge_n_pairs": int(counts[row, col]),
        "est_range_time_seed_stage": np.nan,
    }


def initializer_row(
    asset: DayAsset,
    scenario: str,
    method: str,
    result: dict[str, Any],
    truth: dict[str, float],
    pilot_points: int,
    precompute_s: float,
    common_eval_nll: float,
) -> dict[str, Any]:
    lat, lon = float(result["seed_lat"]), float(result["seed_lon"])
    return {
        "year": asset.year,
        "day_idx": asset.day_idx,
        "day": asset.day,
        "scenario": scenario,
        "method": method,
        "status": "ok",
        "pilot_points_per_hour": int(pilot_points),
        "seed_lat": lat,
        "seed_lon": lon,
        "selection_nll": result.get("selection_nll", np.nan),
        "common_eval_nll_300": common_eval_nll,
        "precompute_s": float(precompute_s),
        "search_s": float(result.get("search_s", 0.0)),
        "seed_total_s": float(precompute_s + result.get("search_s", 0.0)),
        "closure_calls": int(result.get("closure_calls", 0)),
        "profile_evals": int(result.get("profile_evals", 0)),
        "selected_candidate": result.get("selected_candidate", ""),
        "ridge_near_min_count": result.get("ridge_near_min_count", np.nan),
        "ridge_is_ambiguous": result.get("ridge_is_ambiguous", np.nan),
        "est_range_time_seed_stage": result.get("est_range_time_seed_stage", np.nan),
        "true_advec_lat": float(truth["advec_lat"]),
        "true_advec_lon": float(truth["advec_lon"]),
        **vector_metrics(lat, lon, truth),
    }


def fit_full_corridor(
    asset: DayAsset,
    seed_row: dict[str, Any],
    truth: dict[str, float],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    seed_lat, seed_lon = float(seed_row["seed_lat"]), float(seed_row["seed_lon"])
    mapped = {key: value.to(device=device, dtype=torch.double) for key, value in asset.source_map.items()}
    model = build_directional_model(
        smooth=float(args.smooth),
        input_map=mapped,
        grid_coords=asset.grid_coords,
        reference_advec_lat=seed_lat,
        reference_advec_lon=seed_lon,
        daily_stride=int(args.daily_stride),
        target_chunk_size=int(args.target_chunk_size),
        min_target_points=1,
    )
    pre_started = time.perf_counter()
    model.precompute_conditioning_sets()
    precompute_s = time.perf_counter() - pre_started

    common_init = dict(truth)
    common_init["advec_lat"] = seed_lat
    common_init["advec_lon"] = seed_lon
    raw_init = physical_to_raw(common_init)
    params = [Parameter(torch.tensor(value, dtype=torch.double, device=device)) for value in raw_init]
    optimizer = model.set_optimizer(
        params,
        lr=float(args.lbfgs_lr),
        max_iter=int(args.final_max_eval),
        max_eval=int(args.final_max_eval),
        history_size=min(int(args.lbfgs_history), int(args.final_max_eval)),
    )
    closure_calls = 0

    def closure():
        nonlocal closure_calls
        closure_calls += 1
        optimizer.zero_grad()
        loss = model.vecchia_batched_likelihood(torch.stack([param.reshape(()) for param in params]))
        loss.backward()
        return loss

    fit_started = time.perf_counter()
    for _ in range(int(args.final_outer_steps)):
        optimizer.step(closure)
    fit_s = time.perf_counter() - fit_started
    raw_final = [float(param.detach().cpu().item()) for param in params]
    with torch.no_grad():
        final_nll = float(
            model.vecchia_batched_likelihood(
                torch.as_tensor(raw_final, dtype=torch.double, device=device)
            ).detach().cpu().item()
        )
    est = raw_to_physical(raw_final)
    relative_errors = [
        (float(est[key]) - float(truth[key])) / float(truth[key])
        for key in P_LABELS
    ]
    result = {
        "year": asset.year,
        "day_idx": asset.day_idx,
        "day": asset.day,
        "method": seed_row["method"],
        "status": "ok",
        "seed_lat": seed_lat,
        "seed_lon": seed_lon,
        "seed_error_euclid": seed_row["seed_error_euclid"],
        "seed_angle_error_deg": seed_row["seed_angle_error_deg"],
        "seed_total_s": seed_row["seed_total_s"],
        "final_nll": final_nll,
        "final_precompute_s": float(precompute_s),
        "final_fit_s": float(fit_s),
        "final_total_s": float(precompute_s + fit_s),
        "end_to_end_s": float(seed_row["seed_total_s"] + precompute_s + fit_s),
        "final_closure_calls": int(closure_calls),
        "final_rmsre_7param": float(np.sqrt(np.mean(np.square(relative_errors)))),
        **{f"est_{key}": float(est[key]) for key in P_LABELS},
        **{f"true_{key}": float(truth[key]) for key in P_LABELS},
        **{f"final_{key}": value for key, value in vector_metrics(est["advec_lat"], est["advec_lon"], truth).items()},
    }
    del model, mapped, optimizer, params
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def save_outputs(
    output_root: Path,
    initializer_rows: Sequence[dict[str, Any]],
    quadrant_rows: Sequence[dict[str, Any]],
    polar_rows: Sequence[dict[str, Any]],
    full_rows: Sequence[dict[str, Any]],
) -> None:
    pd.DataFrame(initializer_rows).to_csv(output_root / "initializer_results.csv", index=False, float_format="%.8f")
    pd.DataFrame(quadrant_rows).to_csv(output_root / "quadrant_candidates.csv", index=False, float_format="%.8f")
    pd.DataFrame(polar_rows).to_csv(output_root / "polar_profile_candidates.csv", index=False, float_format="%.8f")
    pd.DataFrame(full_rows).to_csv(output_root / "full_fit_results.csv", index=False, float_format="%.8f")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Users/joonwonlee/Documents/GEMS_DATA/simulation/july_st_circulant_realpattern_smooth0p5"),
    )
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--days", default="0,3")
    parser.add_argument("--full-fit-days", default="0")
    parser.add_argument("--lat-range", default="-3,-1")
    parser.add_argument("--lon-range", default="121,125")
    parser.add_argument("--keep-exact-loc", action="store_true", default=True)
    parser.add_argument("--no-keep-exact-loc", dest="keep_exact_loc", action="store_false")
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-root", type=Path, default=REPO / "outputs/summer_26/synthetic_advection_initializer_benchmark_corridor432_081526")
    parser.add_argument("--smooth", type=float, default=0.5)
    parser.add_argument("--pilot400-points", type=int, default=400)
    parser.add_argument("--pilot300-points", type=int, default=300)
    parser.add_argument("--pilot-neighbors", type=int, default=30)
    parser.add_argument("--pilot-nheads", type=int, default=1)
    parser.add_argument("--pilot-limit-a", type=int, default=20)
    parser.add_argument("--pilot-limit-b", type=int, default=20)
    parser.add_argument("--pilot-limit-c", type=int, default=20)
    parser.add_argument("--daily-stride", type=int, default=2)
    parser.add_argument("--m1-max-eval", type=int, default=20)
    parser.add_argument("--quadrant-max-eval-each", type=int, default=5)
    parser.add_argument("--polar-angles", type=int, default=8)
    parser.add_argument("--polar-magnitudes", nargs="+", type=float, default=[0.10, 0.20, 0.30])
    parser.add_argument("--polar-refine-max-eval", type=int, default=5)
    parser.add_argument("--partial-range-time-start-factor", type=float, default=0.75)
    parser.add_argument("--empirical-max-lat-offset", type=int, default=20)
    parser.add_argument("--empirical-max-lon-offset", type=int, default=20)
    parser.add_argument("--empirical-min-pair-count", type=int, default=1000)
    parser.add_argument("--empirical-smooth-bandwidth-deg", type=float, default=0.063)
    parser.add_argument("--empirical-near-min-rel-tol", type=float, default=0.01)
    parser.add_argument("--empirical-near-min-abs-tol", type=float, default=0.05)
    parser.add_argument("--empirical-ambiguous-min-near-cells", type=int, default=5)
    parser.add_argument("--final-outer-steps", type=int, default=1)
    parser.add_argument("--final-max-eval", type=int, default=10)
    parser.add_argument("--target-chunk-size", type=int, default=128)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-history", type=int, default=10)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.data_root) / f"{args.year}_july_st_circulant" / f"sim_july{args.year}_st_circulant_gridded.pkl"
    truth_path = Path(args.data_root) / f"{args.year}_july_st_circulant" / f"sim_july{args.year}_st_circulant_truth.json"
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    truth_physical = {key: float(truth[key]) for key in P_LABELS}
    day_indices = parse_days(args.days)
    full_fit_days = set(parse_days(args.full_fit_days))
    lat_range = parse_pair(args.lat_range, float)
    lon_range = parse_pair(args.lon_range, float)
    config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "input": input_path,
        "truth": truth_physical,
        "days": day_indices,
        "full_fit_days": sorted(full_fit_days),
        "lat_range": lat_range,
        "lon_range": lon_range,
        "device": str(device),
        "methods": METHODS,
        "nuisance_factors": NUISANCE_FACTORS,
        "arguments": vars(args),
    }
    (output_root / "run_config.json").write_text(json.dumps(clean_json(config), indent=2, sort_keys=True), encoding="utf-8")
    print("input:", input_path)
    print("truth:", truth_physical)
    print("device:", device)
    print("output:", output_root)
    assets = load_assets(input_path, truth, day_indices, lat_range, lon_range, bool(args.keep_exact_loc))

    initializer_rows: list[dict[str, Any]] = []
    quadrant_rows: list[dict[str, Any]] = []
    polar_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    benchmark_started = time.perf_counter()

    for asset in assets:
        print(f"\n=== {asset.day}; rows/hour={len(asset.grid_coords)}, valid={asset.n_valid}/{asset.n_total} ===")
        max_points = max(int(args.pilot400_points), int(args.pilot300_points))
        maxmin_order = orderings.maxmin_cpp(asset.grid_coords[:, [1, 0]].copy())
        ordered_grid_coords = asset.grid_coords[maxmin_order]
        nns_map = orderings.find_nns_l2(ordered_grid_coords[:max_points], max_nn=int(args.pilot_neighbors))
        model400, pre400 = build_pilot_model(
            asset, maxmin_order, ordered_grid_coords, int(args.pilot400_points), nns_map, args, device
        )
        model300, pre300 = build_pilot_model(
            asset, maxmin_order, ordered_grid_coords, int(args.pilot300_points), nns_map, args, device
        )
        empirical = empirical_seed(asset, args)
        base_seeds = {
            "M0_zero": {
                "seed_lat": 0.0, "seed_lon": 0.0, "selection_nll": np.nan,
                "search_s": 0.0, "closure_calls": 0, "profile_evals": 0,
                "selected_candidate": "zero", "est_range_time_seed_stage": np.nan,
            },
            "M3_empirical": empirical,
            "M5_oracle": {
                "seed_lat": truth_physical["advec_lat"], "seed_lon": truth_physical["advec_lon"],
                "selection_nll": np.nan, "search_s": 0.0, "closure_calls": 0,
                "profile_evals": 0, "selected_candidate": "truth",
                "est_range_time_seed_stage": truth_physical["range_time"],
            },
        }

        scenarios = [(name, nuisance_from_truth(truth_physical, factor), False, None) for name, factor in NUISANCE_FACTORS.items()]
        scenarios.append(
            (
                "partial_free_range_time",
                dict(truth_physical),
                True,
                truth_physical["range_time"] * float(args.partial_range_time_start_factor),
            )
        )
        oracle_seed_rows: dict[str, dict[str, Any]] = {}

        for scenario, nuisance, free_range_time, range_time_start in scenarios:
            print(f"-- {scenario} --")
            method_results = dict(base_seeds)
            method_results["M1_pilot400"] = fit_advection_only(
                model400,
                nuisance,
                0.0,
                0.0,
                int(args.m1_max_eval),
                args,
                device,
                free_range_time=free_range_time,
                range_time_start=range_time_start,
            )
            m2, m2_candidates = run_quadrant(
                model300, nuisance, args, device, free_range_time, range_time_start
            )
            method_results["M2_quadrant300"] = m2
            m4, m4_candidates = run_polar_profile(
                model300, nuisance, args, device, free_range_time, range_time_start
            )
            method_results["M4_polar_profile"] = m4

            for candidate in m2_candidates:
                quadrant_rows.append(
                    {
                        "year": asset.year, "day_idx": asset.day_idx, "day": asset.day,
                        "scenario": scenario, "selected": candidate["candidate"] == m2["selected_candidate"],
                        **candidate,
                    }
                )
            for candidate in m4_candidates:
                polar_rows.append(
                    {
                        "year": asset.year, "day_idx": asset.day_idx, "day": asset.day,
                        "scenario": scenario, "selected": candidate["candidate"] == m4["selected_candidate"],
                        **candidate,
                    }
                )

            for method in METHODS:
                result = method_results[method]
                common_nll = evaluate_seed_nll(
                    model300,
                    nuisance,
                    float(result["seed_lat"]),
                    float(result["seed_lon"]),
                    device,
                )
                points = int(args.pilot400_points) if method == "M1_pilot400" else (
                    int(args.pilot300_points) if method in {"M2_quadrant300", "M4_polar_profile"} else 0
                )
                precompute = pre400 if method == "M1_pilot400" else (
                    pre300 if method in {"M2_quadrant300", "M4_polar_profile"} else 0.0
                )
                row = initializer_row(
                    asset, scenario, method, result, truth_physical, points, precompute, common_nll
                )
                initializer_rows.append(row)
                if scenario == "oracle_fixed":
                    oracle_seed_rows[method] = row
                print(
                    f"{method}: seed=({row['seed_lat']:+.4f},{row['seed_lon']:+.4f}) "
                    f"err={row['seed_error_euclid']:.4f} angle={row['seed_angle_error_deg']!s} "
                    f"time={row['seed_total_s']:.3f}s closures={row['closure_calls']}"
                )
            save_outputs(output_root, initializer_rows, quadrant_rows, polar_rows, full_rows)

        if asset.day_idx in full_fit_days:
            print("-- full 7-parameter directional corridor fits (oracle-nuisance seeds) --")
            for method in METHODS:
                started = time.perf_counter()
                try:
                    row = fit_full_corridor(asset, oracle_seed_rows[method], truth_physical, args, device)
                    print(
                        f"{method}: final_nll={row['final_nll']:.6f} "
                        f"final_adv_err={row['final_seed_error_euclid']:.4f} "
                        f"time={row['final_total_s']:.2f}s end_to_end={row['end_to_end_s']:.2f}s"
                    )
                except Exception as exc:
                    row = {
                        "year": asset.year, "day_idx": asset.day_idx, "day": asset.day,
                        "method": method, "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=8),
                        "attempt_s": time.perf_counter() - started,
                    }
                    print("ERROR", method, row["error"])
                full_rows.append(clean_json(row))
                save_outputs(output_root, initializer_rows, quadrant_rows, polar_rows, full_rows)

        del model400, model300
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    elapsed = time.perf_counter() - benchmark_started
    init_df = pd.DataFrame(initializer_rows)
    full_df = pd.DataFrame(full_rows)
    timing = init_df.groupby(["scenario", "method"], as_index=False).agg(
        n_days=("day_idx", "size"),
        mean_seed_s=("seed_total_s", "mean"),
        median_seed_s=("seed_total_s", "median"),
        mean_closure_calls=("closure_calls", "mean"),
        mean_seed_error=("seed_error_euclid", "mean"),
        mean_angle_error_deg=("seed_angle_error_deg", "mean"),
        correct_quadrant_rate=("correct_quadrant", "mean"),
    )
    timing.to_csv(output_root / "initializer_timing_accuracy_summary.csv", index=False, float_format="%.8f")
    if not full_df.empty:
        full_df.to_csv(output_root / "full_fit_results.csv", index=False, float_format="%.8f")
    (output_root / "total_runtime.json").write_text(
        json.dumps({"benchmark_wall_s": elapsed, "benchmark_wall_minutes": elapsed / 60.0}, indent=2),
        encoding="utf-8",
    )
    print(f"\nDONE in {elapsed:.2f}s ({elapsed / 60.0:.2f} min)")
    print("initializer results:", output_root / "initializer_results.csv")
    print("full fit results:", output_root / "full_fit_results.csv")
    print("summary:", output_root / "initializer_timing_accuracy_summary.csv")


if __name__ == "__main__":
    main()
