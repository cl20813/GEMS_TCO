#!/usr/bin/env python3
"""Benchmark refined M3 initializers and a reverse-L likelihood refinement.

Fixed baselines
---------------
M0_zero
    Zero-advection seed.
M3_empirical_loop
    Existing tau=1 regular-grid empirical semivariogram implementation.

Competitors
-----------
M3_fft_exact
    The same pair-count-weighted semivariogram computed with masked FFT cross
    correlations.  This method must reproduce the loop surface and seed.
M3_fft_subgrid
    M3_fft_exact plus a safeguarded 3x3 quadratic minimum refinement.
M3_fft_trimmed_subgrid
    Per-transition masked-FFT semivariograms, median-normalized and 1-from-each-
    tail trimmed before the safeguarded quadratic refinement.
M3_reverseL_s2_refine5 / M3_reverseL_full_refine5
    Start from M3_fft_trimmed_subgrid and optimize only the two advection
    parameters for max_eval=5 using stride-2 or full regular-grid reverse-L
    Vecchia likelihoods.  Nuisance covariance parameters remain oracle-fixed.

Each final seed then defines one directional lag-432 corridor.  The downstream
fit releases all seven covariance parameters with identical nuisance starts
and optimizer budgets; corridor geometry remains fixed during optimization.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate
from torch.nn import Parameter

from synthetic_advection_initializer_benchmark_corridor432_081526 import (
    P_LABELS,
    clean_json,
    empirical_seed,
    fit_full_corridor,
    initializer_row,
    load_assets,
    make_hourly_grids,
    nan_gaussian_filter,
    parse_days,
    parse_pair,
    physical_to_raw,
    raw_to_physical,
    shifted_sum_half_sq,
)


LOCAL_REPO = Path("/Users/joonwonlee/Documents/GEMS_TCO-1")
AMAREL_REPO = Path("/home/jl2815/tco")
REPO = AMAREL_REPO if AMAREL_REPO.exists() else LOCAL_REPO
SRC = REPO if (REPO / "GEMS_TCO").exists() else REPO / "src"

from GEMS_TCO.matern_vecchia_col_batch import ReverseLColumnVecchiaFitBatch


METHODS = (
    "M0_zero",
    "M3_empirical_loop",
    "M3_fft_exact",
    "M3_fft_subgrid",
    "M3_fft_trimmed_subgrid",
    "M3_reverseL_s2_refine5",
    "M3_reverseL_full_refine5",
)


def zero_seed() -> dict[str, Any]:
    return {
        "seed_lat": 0.0,
        "seed_lon": 0.0,
        "selection_nll": np.nan,
        "search_s": 0.0,
        "closure_calls": 0,
        "profile_evals": 0,
        "selected_candidate": "zero",
        "ridge_near_min_count": np.nan,
        "ridge_is_ambiguous": np.nan,
        "est_range_time_seed_stage": np.nan,
    }


def crop_full_correlation(
    full: np.ndarray,
    shape: tuple[int, int],
    offsets_lat: np.ndarray,
    offsets_lon: np.ndarray,
) -> np.ndarray:
    center_lat = int(shape[0] - 1)
    center_lon = int(shape[1] - 1)
    rows = center_lat + offsets_lat.astype(np.int64)
    cols = center_lon + offsets_lon.astype(np.int64)
    return np.asarray(full[np.ix_(rows, cols)], dtype=np.float64)


def fft_pair_squared_difference(
    current: np.ndarray,
    following: np.ndarray,
    offsets_lat: np.ndarray,
    offsets_lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sum squared differences and counts for all requested shifts."""
    valid_a = np.isfinite(current)
    valid_b = np.isfinite(following)
    mask_a = valid_a.astype(np.float64)
    mask_b = valid_b.astype(np.float64)
    a = np.where(valid_a, current, 0.0)
    b = np.where(valid_b, following, 0.0)

    # scipy.signal.correlate(B, A)[center+h] equals sum_x A(x) B(x+h).
    count_full = correlate(mask_b, mask_a, mode="full", method="fft")
    a2_full = correlate(mask_b, a * a, mode="full", method="fft")
    b2_full = correlate(b * b, mask_a, mode="full", method="fft")
    cross_full = correlate(b, a, mode="full", method="fft")
    sumsq_full = a2_full + b2_full - 2.0 * cross_full
    counts = crop_full_correlation(count_full, current.shape, offsets_lat, offsets_lon)
    sumsq = crop_full_correlation(sumsq_full, current.shape, offsets_lat, offsets_lon)
    counts = np.rint(np.maximum(counts, 0.0)).astype(np.int64)
    sumsq = np.maximum(sumsq, 0.0)
    return sumsq, counts


def smooth_semivariogram(
    gamma: np.ndarray,
    lat_step: float,
    lon_step: float,
    bandwidth_deg: float,
) -> np.ndarray:
    return nan_gaussian_filter(
        gamma,
        (
            float(bandwidth_deg) / abs(lat_step),
            float(bandwidth_deg) / abs(lon_step),
        ),
    )


def aggregate_fft_surface(
    grids: Sequence[np.ndarray],
    offsets_lat: np.ndarray,
    offsets_lon: np.ndarray,
    lat_step: float,
    lon_step: float,
    min_pair_count: int,
    bandwidth_deg: float,
) -> dict[str, Any]:
    sumsq = np.zeros((len(offsets_lat), len(offsets_lon)), dtype=np.float64)
    counts = np.zeros_like(sumsq, dtype=np.int64)
    for hour in range(len(grids) - 1):
        pair_sumsq, pair_counts = fft_pair_squared_difference(
            grids[hour], grids[hour + 1], offsets_lat, offsets_lon
        )
        sumsq += pair_sumsq
        counts += pair_counts
    gamma = np.full_like(sumsq, np.nan)
    valid = counts >= int(min_pair_count)
    gamma[valid] = 0.5 * sumsq[valid] / counts[valid]
    smoothed = smooth_semivariogram(gamma, lat_step, lon_step, bandwidth_deg)
    return {"gamma": gamma, "smoothed": smoothed, "counts": counts}


def aggregate_loop_surface(
    grids: Sequence[np.ndarray],
    offsets_lat: np.ndarray,
    offsets_lon: np.ndarray,
    lat_step: float,
    lon_step: float,
    min_pair_count: int,
    bandwidth_deg: float,
) -> dict[str, Any]:
    half_sumsq = np.zeros((len(offsets_lat), len(offsets_lon)), dtype=np.float64)
    counts = np.zeros_like(half_sumsq, dtype=np.int64)
    for row, di in enumerate(offsets_lat):
        for col, dj in enumerate(offsets_lon):
            for hour in range(len(grids) - 1):
                value, count = shifted_sum_half_sq(
                    grids[hour], grids[hour + 1], int(di), int(dj)
                )
                half_sumsq[row, col] += value
                counts[row, col] += count
    gamma = np.full_like(half_sumsq, np.nan)
    valid = counts >= int(min_pair_count)
    gamma[valid] = half_sumsq[valid] / counts[valid]
    smoothed = smooth_semivariogram(gamma, lat_step, lon_step, bandwidth_deg)
    return {"gamma": gamma, "smoothed": smoothed, "counts": counts}


def trimmed_fft_surface(
    grids: Sequence[np.ndarray],
    offsets_lat: np.ndarray,
    offsets_lon: np.ndarray,
    lat_step: float,
    lon_step: float,
    min_pair_count_per_transition: int,
    bandwidth_deg: float,
) -> dict[str, Any]:
    pair_surfaces = []
    pair_counts = []
    for hour in range(len(grids) - 1):
        sumsq, counts = fft_pair_squared_difference(
            grids[hour], grids[hour + 1], offsets_lat, offsets_lon
        )
        gamma = np.full_like(sumsq, np.nan)
        valid = counts >= int(min_pair_count_per_transition)
        gamma[valid] = 0.5 * sumsq[valid] / counts[valid]
        smoothed = smooth_semivariogram(gamma, lat_step, lon_step, bandwidth_deg)
        finite = smoothed[np.isfinite(smoothed)]
        median = float(np.median(finite))
        if not np.isfinite(median) or median <= 0:
            raise RuntimeError(f"Invalid transition median at hour {hour}: {median}")
        pair_surfaces.append(smoothed / median)
        pair_counts.append(counts)
    stack = np.stack(pair_surfaces, axis=0)
    ordered = np.sort(stack, axis=0)
    if ordered.shape[0] >= 5:
        robust = np.nanmean(ordered[1:-1], axis=0)
    else:
        robust = np.nanmedian(ordered, axis=0)
    return {
        "gamma": robust,
        "smoothed": robust,
        "counts": np.sum(pair_counts, axis=0),
        "n_transition_surfaces": int(stack.shape[0]),
    }


def surface_minimum(
    surface: np.ndarray,
    offsets_lat: np.ndarray,
    offsets_lon: np.ndarray,
    lat_step: float,
    lon_step: float,
) -> tuple[int, int, float, float]:
    if not np.isfinite(surface).any():
        raise RuntimeError("No finite semivariogram cells")
    row, col = np.unravel_index(int(np.nanargmin(surface)), surface.shape)
    return (
        int(row),
        int(col),
        float(offsets_lat[row] * lat_step),
        float(offsets_lon[col] * lon_step),
    )


def safeguarded_quadratic_minimum(
    surface: np.ndarray,
    offsets_lat: np.ndarray,
    offsets_lon: np.ndarray,
    lat_step: float,
    lon_step: float,
    max_condition_number: float,
) -> dict[str, Any]:
    row, col, grid_lat, grid_lon = surface_minimum(
        surface, offsets_lat, offsets_lon, lat_step, lon_step
    )
    fallback = {
        "seed_lat": grid_lat,
        "seed_lon": grid_lon,
        "subgrid_accepted": False,
        "subgrid_reason": "fallback",
        "subgrid_delta_lat": 0.0,
        "subgrid_delta_lon": 0.0,
        "subgrid_hessian_condition": np.nan,
        "grid_seed_lat": grid_lat,
        "grid_seed_lon": grid_lon,
    }
    if row < 1 or col < 1 or row >= surface.shape[0] - 1 or col >= surface.shape[1] - 1:
        fallback["subgrid_reason"] = "boundary"
        return fallback
    patch = surface[row - 1 : row + 2, col - 1 : col + 2]
    if not np.isfinite(patch).all():
        fallback["subgrid_reason"] = "nonfinite_patch"
        return fallback

    design = []
    values = []
    for local_row, di in enumerate((-1, 0, 1)):
        for local_col, dj in enumerate((-1, 0, 1)):
            x = float(di * lat_step)
            y = float(dj * lon_step)
            design.append([1.0, x, y, 0.5 * x * x, x * y, 0.5 * y * y])
            values.append(float(patch[local_row, local_col]))
    coefficients, *_ = np.linalg.lstsq(
        np.asarray(design, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
        rcond=None,
    )
    gradient = coefficients[1:3]
    hessian = np.asarray(
        [[coefficients[3], coefficients[4]], [coefficients[4], coefficients[5]]],
        dtype=np.float64,
    )
    eigenvalues = np.linalg.eigvalsh(hessian)
    if not np.all(eigenvalues > 0):
        fallback["subgrid_reason"] = "non_positive_hessian"
        return fallback
    condition = float(np.linalg.cond(hessian))
    fallback["subgrid_hessian_condition"] = condition
    if not np.isfinite(condition) or condition > float(max_condition_number):
        fallback["subgrid_reason"] = "ill_conditioned_hessian"
        return fallback
    delta = -np.linalg.solve(hessian, gradient)
    if abs(float(delta[0])) > abs(lat_step) or abs(float(delta[1])) > abs(lon_step):
        fallback["subgrid_reason"] = "delta_outside_cell"
        return fallback
    return {
        "seed_lat": float(grid_lat + delta[0]),
        "seed_lon": float(grid_lon + delta[1]),
        "subgrid_accepted": True,
        "subgrid_reason": "accepted",
        "subgrid_delta_lat": float(delta[0]),
        "subgrid_delta_lon": float(delta[1]),
        "subgrid_hessian_condition": condition,
        "grid_seed_lat": grid_lat,
        "grid_seed_lon": grid_lon,
    }


def empirical_result(
    surface_result: dict[str, Any],
    offsets_lat: np.ndarray,
    offsets_lon: np.ndarray,
    lat_step: float,
    lon_step: float,
    elapsed: float,
    use_subgrid: bool,
    args: argparse.Namespace,
    label: str,
) -> dict[str, Any]:
    if use_subgrid:
        minimum = safeguarded_quadratic_minimum(
            surface_result["smoothed"],
            offsets_lat,
            offsets_lon,
            lat_step,
            lon_step,
            float(args.subgrid_max_condition_number),
        )
    else:
        _, _, seed_lat, seed_lon = surface_minimum(
            surface_result["smoothed"], offsets_lat, offsets_lon, lat_step, lon_step
        )
        minimum = {
            "seed_lat": seed_lat,
            "seed_lon": seed_lon,
            "subgrid_accepted": False,
            "subgrid_reason": "disabled",
            "subgrid_delta_lat": 0.0,
            "subgrid_delta_lon": 0.0,
            "subgrid_hessian_condition": np.nan,
            "grid_seed_lat": seed_lat,
            "grid_seed_lon": seed_lon,
        }
    return {
        **minimum,
        "selection_nll": np.nan,
        "search_s": float(elapsed),
        "closure_calls": 0,
        "profile_evals": int(surface_result["smoothed"].size),
        "selected_candidate": label,
        "ridge_near_min_count": np.nan,
        "ridge_is_ambiguous": bool(minimum["subgrid_reason"] not in {"accepted", "disabled"}),
        "est_range_time_seed_stage": np.nan,
    }


def run_fft_seed(asset, args: argparse.Namespace, robust: bool, subgrid: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    grids, lat_step, lon_step = make_hourly_grids(asset.source_map, asset.grid_coords)
    offsets_lat = np.arange(-int(args.empirical_max_lat_offset), int(args.empirical_max_lat_offset) + 1)
    offsets_lon = np.arange(-int(args.empirical_max_lon_offset), int(args.empirical_max_lon_offset) + 1)
    if robust:
        result = trimmed_fft_surface(
            grids,
            offsets_lat,
            offsets_lon,
            lat_step,
            lon_step,
            int(args.robust_min_pair_count_per_transition),
            float(args.empirical_smooth_bandwidth_deg),
        )
        label = "tau1_fft_trimmed_subgrid"
    else:
        result = aggregate_fft_surface(
            grids,
            offsets_lat,
            offsets_lon,
            lat_step,
            lon_step,
            int(args.empirical_min_pair_count),
            float(args.empirical_smooth_bandwidth_deg),
        )
        label = "tau1_fft_subgrid" if subgrid else "tau1_fft_exact"
    seed = empirical_result(
        result,
        offsets_lat,
        offsets_lon,
        lat_step,
        lon_step,
        time.perf_counter() - started,
        subgrid,
        args,
        label,
    )
    return seed, {
        "surface": result,
        "grids": grids,
        "offsets_lat": offsets_lat,
        "offsets_lon": offsets_lon,
        "lat_step": lat_step,
        "lon_step": lon_step,
    }


def regular_grid_input_map(
    asset,
    device: torch.device,
    spatial_stride: int,
) -> tuple[dict[str, torch.Tensor], np.ndarray]:
    stride = int(spatial_stride)
    if stride < 1:
        raise ValueError("reverse-L spatial stride must be >= 1")
    coords = np.asarray(asset.grid_coords, dtype=np.float64)
    lat_key = np.round(coords[:, 0], 6)
    lon_key = np.round(coords[:, 1], 6)
    lats = np.sort(np.unique(lat_key))
    lons = np.sort(np.unique(lon_key))
    lat_to_row = {float(value): idx for idx, value in enumerate(lats)}
    lon_to_col = {float(value): idx for idx, value in enumerate(lons)}
    rows = np.asarray([lat_to_row[float(value)] for value in lat_key], dtype=np.int64)
    cols = np.asarray([lon_to_col[float(value)] for value in lon_key], dtype=np.int64)
    keep = (rows % stride == 0) & (cols % stride == 0)
    selected = np.flatnonzero(keep)
    grid_coords = coords[selected]
    grid = torch.as_tensor(grid_coords, device=device, dtype=torch.double)
    output = {}
    for key, value in asset.source_map.items():
        selected_rows = value[selected].to(device=device, dtype=torch.double).clone()
        selected_rows[:, :2] = grid
        output[key] = selected_rows.contiguous()
    return output, grid_coords


def reverse_l_refine(
    asset,
    base_seed: dict[str, Any],
    truth: dict[str, float],
    args: argparse.Namespace,
    device: torch.device,
    spatial_stride: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mapped, reverse_l_grid_coords = regular_grid_input_map(asset, device, spatial_stride)
    model = ReverseLColumnVecchiaFitBatch(
        smooth=float(args.smooth),
        input_map=mapped,
        grid_coords=reverse_l_grid_coords,
        head_right_cols=int(args.reverse_l_head_right_cols),
        above_count=int(args.reverse_l_above_count),
        right_col_count=int(args.reverse_l_right_col_count),
        per_lag_conditioning_count=int(args.reverse_l_per_lag_count),
        lag_count=int(args.reverse_l_lag_count),
        include_lag_self=False,
        target_chunk_size=int(args.reverse_l_target_chunk_size),
        use_data_coords_for_offsets=False,
    )
    pre_started = time.perf_counter()
    model.precompute_conditioning_sets()
    precompute_s = time.perf_counter() - pre_started

    raw = physical_to_raw(truth)
    advec_lat = Parameter(
        torch.tensor(float(base_seed["seed_lat"]), device=device, dtype=torch.double)
    )
    advec_lon = Parameter(
        torch.tensor(float(base_seed["seed_lon"]), device=device, dtype=torch.double)
    )
    optimizer = model.set_optimizer(
        [advec_lat, advec_lon],
        lr=float(args.lbfgs_lr),
        max_iter=int(args.reverse_l_max_eval),
        max_eval=int(args.reverse_l_max_eval),
        history_size=min(int(args.lbfgs_history), int(args.reverse_l_max_eval)),
    )
    closure_calls = 0

    def parameter_tensor() -> torch.Tensor:
        values = [torch.tensor(value, device=device, dtype=torch.double) for value in raw]
        values[4] = advec_lat.reshape(())
        values[5] = advec_lon.reshape(())
        return torch.stack([value.reshape(()) for value in values])

    def closure():
        nonlocal closure_calls
        closure_calls += 1
        optimizer.zero_grad()
        loss = model.vecchia_batched_likelihood(parameter_tensor())
        loss.backward()
        return loss

    fit_started = time.perf_counter()
    optimizer.step(closure)
    refine_s = time.perf_counter() - fit_started
    with torch.no_grad():
        final_nll = float(model.vecchia_batched_likelihood(parameter_tensor()).cpu().item())
    seed = {
        "seed_lat": float(advec_lat.detach().cpu().item()),
        "seed_lon": float(advec_lon.detach().cpu().item()),
        "selection_nll": final_nll,
        "search_s": float(base_seed["search_s"] + refine_s),
        "closure_calls": int(closure_calls),
        "profile_evals": int(base_seed.get("profile_evals", 0)),
        "selected_candidate": f"trimmed_subgrid_then_reverseL_stride{int(spatial_stride)}_eval5",
        "ridge_near_min_count": np.nan,
        "ridge_is_ambiguous": base_seed.get("ridge_is_ambiguous", np.nan),
        "est_range_time_seed_stage": float(truth["range_time"]),
        "reverse_l_precompute_s": float(precompute_s),
        "reverse_l_refine_s": float(refine_s),
        "base_seed_lat": float(base_seed["seed_lat"]),
        "base_seed_lon": float(base_seed["seed_lon"]),
        "reverse_l_spatial_stride": int(spatial_stride),
        "reverse_l_points_per_hour": int(reverse_l_grid_coords.shape[0]),
    }
    diagnostics = {
        "reverse_l_precompute_s": float(precompute_s),
        "reverse_l_refine_s": float(refine_s),
        "reverse_l_closure_calls": int(closure_calls),
        "reverse_l_nll": final_nll,
        "reverse_l_n_heads": int(model.Heads_data.shape[0]),
        "reverse_l_n_tails": int(model.n_tails),
        "reverse_l_spatial_stride": int(spatial_stride),
        "reverse_l_points_per_hour": int(reverse_l_grid_coords.shape[0]),
        "reverse_l_batch_shapes": str(
            [(int(group["max_m"]), int(group["target_idx"].shape[0])) for group in model.Batched_Groups]
        ),
    }
    del model, mapped, optimizer, advec_lat, advec_lon
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return seed, diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Users/joonwonlee/Documents/GEMS_DATA/simulation/july_st_circulant_realpattern_smooth0p5"),
    )
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--days", default="0,3")
    parser.add_argument("--full-fit-days", default="0,3")
    parser.add_argument("--lat-range", default="-3,-1")
    parser.add_argument("--lon-range", default="121,125")
    parser.add_argument("--keep-exact-loc", action="store_true", default=True)
    parser.add_argument("--no-keep-exact-loc", dest="keep_exact_loc", action="store_false")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "outputs/summer_26/synthetic_m3_reverse_l_initialization_benchmark_corridor432_083126",
    )
    parser.add_argument("--smooth", type=float, default=0.5)
    parser.add_argument("--daily-stride", type=int, default=2)
    parser.add_argument("--target-chunk-size", type=int, default=128)
    parser.add_argument("--empirical-max-lat-offset", type=int, default=20)
    parser.add_argument("--empirical-max-lon-offset", type=int, default=20)
    parser.add_argument("--empirical-min-pair-count", type=int, default=1000)
    parser.add_argument("--empirical-smooth-bandwidth-deg", type=float, default=0.063)
    parser.add_argument("--empirical-near-min-rel-tol", type=float, default=0.01)
    parser.add_argument("--empirical-near-min-abs-tol", type=float, default=0.05)
    parser.add_argument("--empirical-ambiguous-min-near-cells", type=int, default=5)
    parser.add_argument("--robust-min-pair-count-per-transition", type=int, default=100)
    parser.add_argument("--subgrid-max-condition-number", type=float, default=100.0)
    parser.add_argument("--reverse-l-head-right-cols", type=int, default=0)
    parser.add_argument("--reverse-l-above-count", type=int, default=2)
    parser.add_argument("--reverse-l-right-col-count", type=int, default=3)
    parser.add_argument("--reverse-l-per-lag-count", type=int, default=14)
    parser.add_argument("--reverse-l-lag-count", type=int, default=2)
    parser.add_argument("--reverse-l-target-chunk-size", type=int, default=1024)
    parser.add_argument("--reverse-l-max-eval", type=int, default=5)
    parser.add_argument("--final-outer-steps", type=int, default=1)
    parser.add_argument("--final-max-eval", type=int, default=10)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-history", type=int, default=10)
    parser.add_argument("--initializer-only", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    input_path = (
        Path(args.data_root)
        / f"{args.year}_july_st_circulant"
        / f"sim_july{args.year}_st_circulant_gridded.pkl"
    )
    truth_path = (
        Path(args.data_root)
        / f"{args.year}_july_st_circulant"
        / f"sim_july{args.year}_st_circulant_truth.json"
    )
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    truth_physical = {key: float(truth[key]) for key in P_LABELS}
    day_indices = parse_days(args.days)
    full_fit_days = set() if args.initializer_only else set(parse_days(args.full_fit_days))
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
        "arguments": vars(args),
    }
    (output_root / "run_config.json").write_text(
        json.dumps(clean_json(config), indent=2, sort_keys=True), encoding="utf-8"
    )
    assets = load_assets(
        input_path, truth, day_indices, lat_range, lon_range, bool(args.keep_exact_loc)
    )
    initializer_rows: list[dict[str, Any]] = []
    method_diagnostics: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    benchmark_started = time.perf_counter()

    print("input:", input_path)
    print("truth:", truth_physical)
    print("methods:", METHODS)
    print("device:", device)
    print("output:", output_root)
    for asset in assets:
        print(f"\n=== {asset.day}; rows/hour={len(asset.grid_coords)}, valid={asset.n_valid}/{asset.n_total} ===")
        baseline = empirical_seed(asset, args)
        fft_exact, fft_exact_work = run_fft_seed(asset, args, robust=False, subgrid=False)
        fft_subgrid, _ = run_fft_seed(asset, args, robust=False, subgrid=True)
        robust_subgrid, _ = run_fft_seed(asset, args, robust=True, subgrid=True)

        # Surface equality is a required correctness test for M3_fft_exact.
        loop_surface = aggregate_loop_surface(
            fft_exact_work["grids"],
            fft_exact_work["offsets_lat"],
            fft_exact_work["offsets_lon"],
            fft_exact_work["lat_step"],
            fft_exact_work["lon_step"],
            int(args.empirical_min_pair_count),
            float(args.empirical_smooth_bandwidth_deg),
        )
        fft_surface = fft_exact_work["surface"]
        max_gamma_diff = float(np.nanmax(np.abs(loop_surface["gamma"] - fft_surface["gamma"])))
        max_smooth_diff = float(
            np.nanmax(np.abs(loop_surface["smoothed"] - fft_surface["smoothed"]))
        )
        max_count_diff = int(np.max(np.abs(loop_surface["counts"] - fft_surface["counts"])))
        equivalence_rows.append(
            {
                "year": asset.year,
                "day_idx": asset.day_idx,
                "day": asset.day,
                "max_abs_gamma_diff": max_gamma_diff,
                "max_abs_smoothed_diff": max_smooth_diff,
                "max_abs_count_diff": max_count_diff,
                "same_seed": bool(
                    np.isclose(baseline["seed_lat"], fft_exact["seed_lat"])
                    and np.isclose(baseline["seed_lon"], fft_exact["seed_lon"])
                ),
            }
        )
        if max_smooth_diff > 1e-8 or max_count_diff != 0:
            raise AssertionError(
                f"FFT equivalence failed on {asset.day}: smooth={max_smooth_diff}, count={max_count_diff}"
            )

        reverse_l_s2, reverse_l_s2_diag = reverse_l_refine(
            asset, robust_subgrid, truth_physical, args, device, spatial_stride=2
        )
        reverse_l_full, reverse_l_full_diag = reverse_l_refine(
            asset, robust_subgrid, truth_physical, args, device, spatial_stride=1
        )
        results = {
            "M0_zero": zero_seed(),
            "M3_empirical_loop": baseline,
            "M3_fft_exact": fft_exact,
            "M3_fft_subgrid": fft_subgrid,
            "M3_fft_trimmed_subgrid": robust_subgrid,
            "M3_reverseL_s2_refine5": reverse_l_s2,
            "M3_reverseL_full_refine5": reverse_l_full,
        }
        day_seed_rows: dict[str, dict[str, Any]] = {}
        for method in METHODS:
            precompute = float(results[method].get("reverse_l_precompute_s", 0.0))
            row = initializer_row(
                asset=asset,
                scenario="oracle_nuisance",
                method=method,
                result=results[method],
                truth=truth_physical,
                pilot_points=0,
                precompute_s=precompute,
                common_eval_nll=np.nan,
            )
            for key in (
                "subgrid_accepted",
                "subgrid_reason",
                "subgrid_delta_lat",
                "subgrid_delta_lon",
                "subgrid_hessian_condition",
                "grid_seed_lat",
                "grid_seed_lon",
                "reverse_l_precompute_s",
                "reverse_l_refine_s",
                "base_seed_lat",
                "base_seed_lon",
                "reverse_l_spatial_stride",
                "reverse_l_points_per_hour",
            ):
                if key in results[method]:
                    row[key] = results[method][key]
            initializer_rows.append(clean_json(row))
            day_seed_rows[method] = row
            print(
                f"{method}: seed=({row['seed_lat']:+.6f},{row['seed_lon']:+.6f}) "
                f"error={row['seed_error_euclid']:.6f} angle={row['seed_angle_error_deg']!s} "
                f"time={row['seed_total_s']:.4f}s closures={row['closure_calls']}"
            )
        method_diagnostics.extend(
            [
                {
                    "year": asset.year,
                    "day_idx": asset.day_idx,
                    "day": asset.day,
                    "method": "M3_reverseL_s2_refine5",
                    **reverse_l_s2_diag,
                },
                {
                    "year": asset.year,
                    "day_idx": asset.day_idx,
                    "day": asset.day,
                    "method": "M3_reverseL_full_refine5",
                    **reverse_l_full_diag,
                },
            ]
        )
        pd.DataFrame(initializer_rows).to_csv(
            output_root / "initializer_results.csv", index=False, float_format="%.10f"
        )
        pd.DataFrame(equivalence_rows).to_csv(
            output_root / "fft_equivalence.csv", index=False, float_format="%.12g"
        )
        pd.DataFrame(method_diagnostics).to_csv(
            output_root / "reverse_l_diagnostics.csv", index=False, float_format="%.10f"
        )

        if asset.day_idx in full_fit_days:
            print("-- full 7-parameter fixed directional-corridor fits --")
            for method in METHODS:
                attempt_started = time.perf_counter()
                try:
                    row = fit_full_corridor(
                        asset, day_seed_rows[method], truth_physical, args, device
                    )
                    print(
                        f"{method}: nll={row['final_nll']:.6f} "
                        f"final_adv_error={row['final_seed_error_euclid']:.6f} "
                        f"fit={row['final_total_s']:.2f}s end_to_end={row['end_to_end_s']:.2f}s"
                    )
                except Exception as exc:
                    row = {
                        "year": asset.year,
                        "day_idx": asset.day_idx,
                        "day": asset.day,
                        "method": method,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=8),
                        "attempt_s": time.perf_counter() - attempt_started,
                    }
                    print("ERROR", method, row["error"])
                full_rows.append(clean_json(row))
                pd.DataFrame(full_rows).to_csv(
                    output_root / "full_fit_results.csv", index=False, float_format="%.10f"
                )

    init_df = pd.DataFrame(initializer_rows)
    full_df = pd.DataFrame(full_rows)
    initializer_summary = init_df.groupby("method", as_index=False).agg(
        n_days=("day_idx", "size"),
        mean_seed_s=("seed_total_s", "mean"),
        median_seed_s=("seed_total_s", "median"),
        mean_seed_error=("seed_error_euclid", "mean"),
        median_seed_error=("seed_error_euclid", "median"),
        max_seed_error=("seed_error_euclid", "max"),
        mean_angle_error_deg=("seed_angle_error_deg", "mean"),
        correct_quadrant_rate=("correct_quadrant", "mean"),
        mean_closure_calls=("closure_calls", "mean"),
    )
    initializer_summary.to_csv(
        output_root / "initializer_summary.csv", index=False, float_format="%.10f"
    )
    if not full_df.empty and "final_nll" in full_df:
        ok = full_df[full_df["status"].eq("ok")]
        full_summary = ok.groupby("method", as_index=False).agg(
            n_days=("day_idx", "size"),
            mean_final_nll=("final_nll", "mean"),
            mean_final_advection_error=("final_seed_error_euclid", "mean"),
            mean_final_rmsre_7param=("final_rmsre_7param", "mean"),
            mean_final_fit_s=("final_total_s", "mean"),
            mean_end_to_end_s=("end_to_end_s", "mean"),
        )
        full_summary.to_csv(
            output_root / "full_fit_summary.csv", index=False, float_format="%.10f"
        )
    elapsed = time.perf_counter() - benchmark_started
    (output_root / "total_runtime.json").write_text(
        json.dumps(
            {"benchmark_wall_s": elapsed, "benchmark_wall_minutes": elapsed / 60.0},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nDONE in {elapsed:.2f}s ({elapsed / 60.0:.2f} min)")
    print(initializer_summary.to_string(index=False))


if __name__ == "__main__":
    main()
