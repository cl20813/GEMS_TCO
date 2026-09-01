#!/usr/bin/env python3
"""Compete six advection seeds for a direction-dependent Vecchia corridor.

The initializer is judged as an engineering component: it should improve the
advection seed and the downstream fixed-corridor fit without costing nearly as
much as that fit.  The methods are

* M0_zero: zero advection.
* M3_fft: exact regular-grid tau=1 FFT semivariogram argmin.
* S1_fixed_stencil: zero-start advection-only L-BFGS on a small, direction-
  neutral reverse-L stencil (stride 3, m <= 16, max_eval 5 by default).
* S2_pairwise_polar: global polar screening using a Gaussian difference
  composite likelihood for tau=1,2, followed by a short local refinement.
* S3_fft_gated: M3-local pairwise refinement when the FFT basin is strong.
  The safe production default keeps M3 unchanged when the basin is weak;
  global S2 fallback remains available as an explicit experiment option.
* S4_iterative_corridor: M3 followed by repeated directional-corridor rebuild
  and short advection-only fits.  This is an intentionally expensive upper
  bound rather than the recommended production method.

Every final fit builds the lag-432 directional corridor once from its selected
seed and releases the same seven covariance parameters with the same budget.
Final parameter vectors are also evaluated on the same oracle-direction
corridor because own-corridor Vecchia objectives are not directly comparable.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from torch.nn import Parameter


REPO = Path("/Users/joonwonlee/Documents/GEMS_TCO-1")
SRC = REPO / "src"
HERE = Path(__file__).resolve().parent
for path in (REPO, SRC, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from synthetic_advection_initializer_benchmark_corridor432_081526 import (
    DayAsset,
    P_LABELS,
    clean_json,
    load_assets,
    parse_pair,
    physical_to_raw,
    vector_metrics,
)
from synthetic_initializer_factorial_robustness_corridor432_083126 import (
    build_corridor_model,
    crop_regular_asset,
    evaluate_raw,
    fit_method_corridor,
    nuisance_start,
    simulate_asset,
    truth_grid,
)
from synthetic_m3_reverse_l_initialization_benchmark_corridor432_083126 import (
    fft_pair_squared_difference,
    make_hourly_grids,
    reverse_l_refine,
    run_fft_seed,
)


METHODS = (
    "M0_zero",
    "M3_fft",
    "S1_fixed_stencil",
    "S2_pairwise_polar",
    "S3_fft_gated",
    "S4_iterative_corridor",
)
NUISANCE_SCENARIOS = (
    "oracle_start",
    "short_time_high_nugget",
    "long_time_low_nugget",
)
POSITIVE_PARAMS = ("sigmasq", "range_lat", "range_lon", "range_time", "nugget")


def parse_float_list(text: str) -> list[float]:
    return [float(token.strip()) for token in str(text).split(",") if token.strip()]


def parse_int_list(text: str, maximum: int) -> list[int]:
    if str(text).strip().lower() == "all":
        return list(range(maximum))
    return [int(token.strip()) for token in str(text).split(",") if token.strip()]


def zero_seed() -> dict[str, Any]:
    return {
        "seed_lat": 0.0,
        "seed_lon": 0.0,
        "seed_total_s": 0.0,
        "search_s": 0.0,
        "closure_calls": 0,
        "selected_candidate": "zero",
    }


def pairwise_sufficient_statistics(asset: DayAsset, args: argparse.Namespace) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    grids, lat_step, lon_step = make_hourly_grids(asset.source_map, asset.grid_coords)
    offsets_lat = np.arange(-int(args.empirical_max_lat_offset), int(args.empirical_max_lat_offset) + 1)
    offsets_lon = np.arange(-int(args.empirical_max_lon_offset), int(args.empirical_max_lon_offset) + 1)
    entries: list[dict[str, np.ndarray | float | int]] = []
    for tau in (1, 2):
        sumsq = np.zeros((len(offsets_lat), len(offsets_lon)), dtype=np.float64)
        counts = np.zeros_like(sumsq, dtype=np.int64)
        for hour in range(len(grids) - tau):
            pair_sumsq, pair_counts = fft_pair_squared_difference(
                grids[hour], grids[hour + tau], offsets_lat, offsets_lon
            )
            sumsq += pair_sumsq
            counts += pair_counts
        min_count = max(20, int(args.pairwise_min_pair_count * (len(grids) - tau) / (len(grids) - 1)))
        valid = counts >= min_count
        lat_mesh, lon_mesh = np.meshgrid(
            offsets_lat.astype(np.float64) * lat_step,
            offsets_lon.astype(np.float64) * lon_step,
            indexing="ij",
        )
        entries.append(
            {
                "tau": tau,
                "h_lat": lat_mesh[valid],
                "h_lon": lon_mesh[valid],
                "sumsq": sumsq[valid],
                "counts": counts[valid].astype(np.float64),
            }
        )
    return {
        "entries": entries,
        "lat_step": float(lat_step),
        "lon_step": float(lon_step),
    }, float(time.perf_counter() - started)


def pairwise_nll_and_grad(
    advection: Sequence[float],
    nuisance: dict[str, float],
    range_time: float,
    stats: dict[str, Any],
    smooth: float,
) -> tuple[float, np.ndarray]:
    a_lat, a_lon = float(advection[0]), float(advection[1])
    sigmasq = float(nuisance["sigmasq"])
    nugget = float(nuisance["nugget"])
    range_lat = max(float(nuisance["range_lat"]), 1e-8)
    range_lon = max(float(nuisance["range_lon"]), 1e-8)
    range_time = max(float(range_time), 1e-8)
    loss_sum = 0.0
    grad_sum = np.zeros(2, dtype=np.float64)
    count_sum = 0.0
    for entry in stats["entries"]:
        tau = float(entry["tau"])
        u_lat = np.asarray(entry["h_lat"]) - tau * a_lat
        u_lon = np.asarray(entry["h_lon"]) - tau * a_lon
        distance = np.sqrt(
            np.maximum(
                (u_lat / range_lat) ** 2
                + (u_lon / range_lon) ** 2
                + (tau / range_time) ** 2,
                1e-14,
            )
        )
        if float(smooth) == 0.5:
            covariance = sigmasq * np.exp(-distance)
            d_cov_d_distance = -covariance
        elif float(smooth) == 1.5:
            exp_term = np.exp(-distance)
            covariance = sigmasq * (1.0 + distance) * exp_term
            d_cov_d_distance = -sigmasq * distance * exp_term
        else:
            raise ValueError(f"Unsupported smooth={smooth}; expected 0.5 or 1.5")
        variance = np.maximum(2.0 * (sigmasq + nugget - covariance), 1e-10)
        counts = np.asarray(entry["counts"])
        sumsq = np.asarray(entry["sumsq"])
        loss_sum += float(0.5 * np.sum(counts * np.log(variance) + sumsq / variance))
        common = 0.5 * (counts / variance - sumsq / variance**2)
        d_distance_d_lat = -tau * u_lat / (range_lat**2 * distance)
        d_distance_d_lon = -tau * u_lon / (range_lon**2 * distance)
        d_variance_d_lat = -2.0 * d_cov_d_distance * d_distance_d_lat
        d_variance_d_lon = -2.0 * d_cov_d_distance * d_distance_d_lon
        grad_sum[0] += float(np.sum(common * d_variance_d_lat))
        grad_sum[1] += float(np.sum(common * d_variance_d_lon))
        count_sum += float(np.sum(counts))
    return loss_sum / count_sum, grad_sum / count_sum


def range_time_candidates(nuisance: dict[str, float], args: argparse.Namespace) -> list[float]:
    base = float(nuisance["range_time"])
    return [base * factor for factor in parse_float_list(args.pairwise_range_time_factors)]


def local_pairwise_refine(
    start: Sequence[float],
    nuisance: dict[str, float],
    selected_range_time: float,
    stats: dict[str, Any],
    args: argparse.Namespace,
    bounds: Sequence[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, float, int]:
    calls = 0

    def objective(x: np.ndarray):
        nonlocal calls
        calls += 1
        return pairwise_nll_and_grad(x, nuisance, selected_range_time, stats, float(args.smooth))

    active_bounds = list(bounds) if bounds is not None else [
        (-float(args.max_seed_component), float(args.max_seed_component))
    ] * 2
    initial = np.asarray(start, dtype=np.float64)
    initial = np.asarray(
        [np.clip(value, lower, upper) for value, (lower, upper) in zip(initial, active_bounds)],
        dtype=np.float64,
    )
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=active_bounds,
        options={
            "maxiter": int(args.pairwise_local_max_iter),
            "maxfun": int(args.pairwise_local_max_eval),
            "ftol": 1e-9,
            "gtol": 1e-6,
            "maxls": 5,
        },
    )
    value, _ = pairwise_nll_and_grad(
        result.x, nuisance, selected_range_time, stats, float(args.smooth)
    )
    return np.asarray(result.x, dtype=np.float64), float(value), int(calls)


def profile_candidates(
    candidates: Sequence[tuple[float, float]],
    nuisance: dict[str, float],
    stats: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[np.ndarray, float, float, int]:
    best: tuple[float, np.ndarray, float] | None = None
    evaluations = 0
    for range_time in range_time_candidates(nuisance, args):
        for candidate in candidates:
            value, _ = pairwise_nll_and_grad(candidate, nuisance, range_time, stats, float(args.smooth))
            evaluations += 1
            if best is None or value < best[0]:
                best = (float(value), np.asarray(candidate, dtype=np.float64), float(range_time))
    if best is None:
        raise RuntimeError("Pairwise profile received no candidates")
    return best[1], best[2], best[0], evaluations


def polar_candidates(args: argparse.Namespace) -> list[tuple[float, float]]:
    candidates: list[tuple[float, float]] = []
    for speed in parse_float_list(args.polar_speeds):
        for angle in np.linspace(0.0, 2.0 * np.pi, int(args.polar_directions), endpoint=False):
            candidates.append((float(speed * np.sin(angle)), float(speed * np.cos(angle))))
    return candidates


def pairwise_polar_seed(
    nuisance: dict[str, float],
    stats: dict[str, Any],
    stats_s: float,
    args: argparse.Namespace,
    include_stats_time: bool = True,
) -> dict[str, Any]:
    started = time.perf_counter()
    start, range_time, screen_nll, profile_evals = profile_candidates(
        polar_candidates(args), nuisance, stats, args
    )
    refined, final_nll, local_evals = local_pairwise_refine(
        start, nuisance, range_time, stats, args
    )
    search_s = float(time.perf_counter() - started)
    return {
        "seed_lat": float(refined[0]),
        "seed_lon": float(refined[1]),
        "search_s": search_s,
        "seed_total_s": float(search_s + (stats_s if include_stats_time else 0.0)),
        "closure_calls": int(local_evals),
        "profile_evals": int(profile_evals),
        "screen_nll": float(screen_nll),
        "selection_nll": float(final_nll),
        "est_range_time_seed_stage": float(range_time),
        "selected_candidate": "pairwise_polar_global",
    }


def fft_signal_diagnostics(fft_diag: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    surface = np.asarray(fft_diag["surface"]["smoothed"], dtype=np.float64)
    finite = np.isfinite(surface)
    row, col = np.unravel_index(int(np.nanargmin(surface)), surface.shape)
    minimum = float(surface[row, col])
    median = float(np.nanmedian(surface))
    scale = max(median - minimum, 1e-12)
    normalized = (surface - minimum) / scale
    near_count = int(np.sum(finite & (normalized <= float(args.fft_near_fraction))))
    outside = finite.copy()
    radius = int(args.fft_exclusion_radius)
    outside[max(0, row - radius) : row + radius + 1, max(0, col - radius) : col + radius + 1] = False
    second_basin = float(np.nanmin(surface[outside])) if outside.any() else np.nan
    basin_gap = (second_basin - minimum) / scale if np.isfinite(second_basin) else np.nan
    boundary = bool(row == 0 or col == 0 or row == surface.shape[0] - 1 or col == surface.shape[1] - 1)
    strong = bool(
        not boundary
        and near_count <= int(args.fft_strong_max_near_cells)
        and np.isfinite(basin_gap)
        and basin_gap >= float(args.fft_strong_min_basin_gap)
    )
    return {
        "fft_strong": strong,
        "fft_near_count": near_count,
        "fft_basin_gap": float(basin_gap),
        "fft_boundary_minimum": boundary,
    }


def fft_gated_seed(
    m3: dict[str, Any],
    fft_diag: dict[str, Any],
    nuisance: dict[str, float],
    stats: dict[str, Any],
    stats_s: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    signal = fft_signal_diagnostics(fft_diag, args)
    if signal["fft_strong"]:
        lat_step = float(stats["lat_step"])
        lon_step = float(stats["lon_step"])
        candidates = [
            (float(m3["seed_lat"] + di * lat_step), float(m3["seed_lon"] + dj * lon_step))
            for di in (-1, 0, 1)
            for dj in (-1, 0, 1)
        ]
        start, range_time, screen_nll, profile_evals = profile_candidates(
            candidates, nuisance, stats, args
        )
        path = "fft_local"
        local_bounds = [
            (
                float(m3["seed_lat"] - 0.5 * lat_step),
                float(m3["seed_lat"] + 0.5 * lat_step),
            ),
            (
                float(m3["seed_lon"] - 0.5 * lon_step),
                float(m3["seed_lon"] + 0.5 * lon_step),
            ),
        ]
    else:
        if str(args.s3_weak_action) == "keep_m3":
            return {
                "seed_lat": float(m3["seed_lat"]),
                "seed_lon": float(m3["seed_lon"]),
                "search_s": 0.0,
                "seed_total_s": float(m3["seed_total_s"]),
                "closure_calls": 0,
                "profile_evals": 0,
                "selection_nll": np.nan,
                "est_range_time_seed_stage": np.nan,
                "selected_candidate": "weak_keep_m3",
                **signal,
            }
        start, range_time, screen_nll, profile_evals = profile_candidates(
            polar_candidates(args), nuisance, stats, args
        )
        path = "global_fallback"
        local_bounds = None
    refined, final_nll, local_evals = local_pairwise_refine(
        start, nuisance, range_time, stats, args, bounds=local_bounds
    )
    added_s = float(time.perf_counter() - started)
    return {
        "seed_lat": float(refined[0]),
        "seed_lon": float(refined[1]),
        "search_s": added_s,
        "seed_total_s": float(m3["seed_total_s"] + stats_s + added_s),
        "closure_calls": int(local_evals),
        "profile_evals": int(profile_evals),
        "screen_nll": float(screen_nll),
        "selection_nll": float(final_nll),
        "est_range_time_seed_stage": float(range_time),
        "selected_candidate": path,
        **signal,
    }


def fixed_stencil_seed(
    asset: DayAsset,
    nuisance: dict[str, float],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    local_args = copy.copy(args)
    local_args.reverse_l_head_right_cols = 0
    local_args.reverse_l_above_count = int(args.s1_above_count)
    local_args.reverse_l_right_col_count = int(args.s1_right_col_count)
    local_args.reverse_l_per_lag_count = int(args.s1_per_lag_count)
    local_args.reverse_l_lag_count = int(args.s1_lag_count)
    local_args.reverse_l_target_chunk_size = int(args.s1_target_chunk_size)
    local_args.reverse_l_max_eval = int(args.s1_max_eval)
    base = zero_seed()
    seed, diag = reverse_l_refine(
        asset, base, nuisance, local_args, device, spatial_stride=int(args.s1_spatial_stride)
    )
    seed.update(diag)
    seed["seed_total_s"] = float(diag["reverse_l_precompute_s"] + diag["reverse_l_refine_s"])
    seed["selected_candidate"] = "zero_then_fixed_stencil"
    return seed


def iterative_corridor_seed(
    asset: DayAsset,
    m3: dict[str, Any],
    nuisance: dict[str, float],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    current = np.asarray([m3["seed_lat"], m3["seed_lon"]], dtype=np.float64)
    total_precompute = 0.0
    total_refine = 0.0
    closure_calls = 0
    round_seeds = []
    raw = physical_to_raw(nuisance)
    for _ in range(int(args.s4_rounds)):
        model = mapped = optimizer = advec_lat = advec_lon = None
        try:
            model, mapped, precompute_s = build_corridor_model(
                asset, float(current[0]), float(current[1]), args, device
            )
            total_precompute += float(precompute_s)
            advec_lat = Parameter(torch.tensor(float(current[0]), dtype=torch.double, device=device))
            advec_lon = Parameter(torch.tensor(float(current[1]), dtype=torch.double, device=device))
            optimizer = model.set_optimizer(
                [advec_lat, advec_lon],
                lr=float(args.lbfgs_lr),
                max_iter=int(args.s4_max_eval_per_round),
                max_eval=int(args.s4_max_eval_per_round),
                history_size=min(int(args.lbfgs_history), int(args.s4_max_eval_per_round)),
            )

            def parameter_tensor():
                values = [torch.tensor(value, dtype=torch.double, device=device) for value in raw]
                values[4] = advec_lat.reshape(())
                values[5] = advec_lon.reshape(())
                return torch.stack(values)

            def closure():
                nonlocal closure_calls
                closure_calls += 1
                optimizer.zero_grad()
                loss = model.vecchia_batched_likelihood(parameter_tensor())
                loss.backward()
                return loss

            started = time.perf_counter()
            optimizer.step(closure)
            total_refine += float(time.perf_counter() - started)
            current = np.asarray(
                [float(advec_lat.detach().cpu()), float(advec_lon.detach().cpu())], dtype=np.float64
            )
            current = np.clip(current, -float(args.max_seed_component), float(args.max_seed_component))
            round_seeds.append(current.tolist())
        finally:
            del model, mapped, optimizer, advec_lat, advec_lon
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return {
        "seed_lat": float(current[0]),
        "seed_lon": float(current[1]),
        "search_s": float(total_precompute + total_refine),
        "seed_total_s": float(m3["seed_total_s"] + total_precompute + total_refine),
        "closure_calls": int(closure_calls),
        "selected_candidate": "m3_iterative_corridor",
        "iterative_precompute_s": float(total_precompute),
        "iterative_refine_s": float(total_refine),
        "iterative_round_seeds": json.dumps(round_seeds),
    }


def initializer_row(
    asset: DayAsset,
    truth_info: dict[str, Any],
    replicate: int,
    nuisance_scenario: str,
    method: str,
    seed: dict[str, Any],
    truth: dict[str, float],
) -> dict[str, Any]:
    return {
        "dataset_id": asset.day,
        "truth_id": truth_info["truth_id"],
        "speed_label": truth_info["speed_label"],
        "direction": truth_info["direction"],
        "replicate": int(replicate),
        "nuisance_scenario": nuisance_scenario,
        "method": method,
        "seed_lat": float(seed["seed_lat"]),
        "seed_lon": float(seed["seed_lon"]),
        "seed_total_s": float(seed.get("seed_total_s", seed.get("search_s", 0.0))),
        "closure_calls": int(seed.get("closure_calls", 0)),
        "profile_evals": int(seed.get("profile_evals", 0)),
        "selected_candidate": seed.get("selected_candidate", ""),
        "fft_strong": seed.get("fft_strong", np.nan),
        "fft_near_count": seed.get("fft_near_count", np.nan),
        "fft_basin_gap": seed.get("fft_basin_gap", np.nan),
        "est_range_time_seed_stage": seed.get("est_range_time_seed_stage", np.nan),
        **vector_metrics(float(seed["seed_lat"]), float(seed["seed_lon"]), truth),
    }


def save_results(output_root: Path, initializer_rows: list[dict], full_rows: list[dict]) -> None:
    init = pd.DataFrame(initializer_rows)
    init.to_csv(output_root / "initializer_results.csv", index=False, float_format="%.10f")
    if len(init):
        summary = (
            init.groupby("method", as_index=False)
            .agg(
                n_cases=("dataset_id", "size"),
                mean_seed_error=("seed_error_euclid", "mean"),
                median_seed_error=("seed_error_euclid", "median"),
                mean_angle_error_deg=("seed_angle_error_deg", "mean"),
                correct_quadrant_rate=("correct_quadrant", "mean"),
                mean_seed_s=("seed_total_s", "mean"),
                median_seed_s=("seed_total_s", "median"),
            )
        )
        m3_time = float(summary.loc[summary.method.eq("M3_fft"), "mean_seed_s"].iloc[0])
        summary["time_ratio_vs_M3"] = summary["mean_seed_s"] / max(m3_time, 1e-12)
        summary.to_csv(output_root / "initializer_summary.csv", index=False, float_format="%.10f")
    full = pd.DataFrame(full_rows)
    if not len(full):
        return
    for column in (
        "converged_grad",
        "common_nll_regret",
        "final_standardized_rmse_7param",
        "final_seed_error_euclid",
        "final_fit_s",
        "end_to_end_s",
    ):
        if column not in full:
            full[column] = np.nan
    full.to_csv(output_root / "full_fit_results.csv", index=False, float_format="%.10f")
    summary = (
        full.groupby("method", as_index=False)
        .agg(
            n_cases=("dataset_id", "size"),
            success_rate=("status", lambda x: float(np.mean(x == "ok"))),
            convergence_rate=("converged_grad", "mean"),
            mean_common_nll_regret=("common_nll_regret", "mean"),
            mean_final_parameter_error=("final_standardized_rmse_7param", "mean"),
            mean_final_advection_error=("final_seed_error_euclid", "mean"),
            mean_final_fit_s=("final_fit_s", "mean"),
            mean_end_to_end_s=("end_to_end_s", "mean"),
        )
    )
    summary.to_csv(output_root / "full_fit_summary.csv", index=False, float_format="%.10f")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Users/joonwonlee/Documents/GEMS_DATA/simulation/july_st_circulant_realpattern_smooth0p5"),
    )
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--truth-indices", default="all")
    parser.add_argument("--nuisance-scenarios", default=",".join(NUISANCE_SCENARIOS))
    parser.add_argument("--lat-range", default="-3,-1")
    parser.add_argument("--lon-range", default="121,125")
    parser.add_argument("--n-lat", type=int, default=32)
    parser.add_argument("--n-lon", type=int, default=48)
    parser.add_argument("--seed", type=int, default=831261)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--initializer-only", action="store_true")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "outputs/summer_26/synthetic_short_model_initializer_competition_corridor432_083126",
    )
    parser.add_argument("--smooth", type=float, default=0.5)
    parser.add_argument("--daily-stride", type=int, default=2)
    parser.add_argument("--target-chunk-size", type=int, default=128)
    parser.add_argument("--final-max-eval", type=int, default=12)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-history", type=int, default=10)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    parser.add_argument("--convergence-grad-threshold", type=float, default=1e-3)
    parser.add_argument("--max-seed-component", type=float, default=0.75)

    parser.add_argument("--empirical-max-lat-offset", type=int, default=20)
    parser.add_argument("--empirical-max-lon-offset", type=int, default=20)
    parser.add_argument("--empirical-min-pair-count", type=int, default=1000)
    parser.add_argument("--empirical-smooth-bandwidth-deg", type=float, default=0.063)
    parser.add_argument("--subgrid-max-condition-number", type=float, default=100.0)

    parser.add_argument("--pairwise-min-pair-count", type=int, default=1000)
    parser.add_argument("--pairwise-range-time-factors", default="0.67,1.0,1.5")
    parser.add_argument("--polar-directions", type=int, default=12)
    parser.add_argument("--polar-speeds", default="0.08,0.16,0.26")
    parser.add_argument("--pairwise-local-max-iter", type=int, default=4)
    parser.add_argument("--pairwise-local-max-eval", type=int, default=8)

    parser.add_argument("--fft-near-fraction", type=float, default=0.05)
    parser.add_argument("--fft-exclusion-radius", type=int, default=2)
    parser.add_argument("--fft-strong-max-near-cells", type=int, default=25)
    parser.add_argument("--fft-strong-min-basin-gap", type=float, default=0.03)
    parser.add_argument(
        "--s3-weak-action",
        choices=("keep_m3", "global_polar"),
        default="keep_m3",
        help="Safe default avoids the nuisance-sensitive global polar fallback.",
    )

    parser.add_argument("--s1-spatial-stride", type=int, default=3)
    parser.add_argument("--s1-above-count", type=int, default=2)
    parser.add_argument("--s1-right-col-count", type=int, default=3)
    parser.add_argument("--s1-per-lag-count", type=int, default=8)
    parser.add_argument("--s1-lag-count", type=int, default=1)
    parser.add_argument("--s1-target-chunk-size", type=int, default=1024)
    parser.add_argument("--s1-max-eval", type=int, default=5)

    parser.add_argument("--s4-rounds", type=int, default=2)
    parser.add_argument("--s4-max-eval-per-round", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    truths = truth_grid()
    truth_indices = parse_int_list(args.truth_indices, len(truths))
    nuisance_scenarios = [token.strip() for token in args.nuisance_scenarios.split(",") if token.strip()]
    unknown = sorted(set(nuisance_scenarios) - set(NUISANCE_SCENARIOS))
    if unknown:
        raise ValueError(f"Unknown nuisance scenarios: {unknown}")

    truth_path = args.data_root / f"{args.year}_july_st_circulant" / f"sim_july{args.year}_st_circulant_truth.json"
    input_path = args.data_root / f"{args.year}_july_st_circulant" / f"sim_july{args.year}_st_circulant_gridded.pkl"
    base_truth_json = json.loads(truth_path.read_text(encoding="utf-8"))
    base_truth = {key: float(base_truth_json[key]) for key in P_LABELS}
    templates = load_assets(
        input_path,
        base_truth_json,
        list(range(int(args.replicates))),
        parse_pair(args.lat_range, float),
        parse_pair(args.lon_range, float),
        False,
    )
    templates = [crop_regular_asset(asset, int(args.n_lat), int(args.n_lon)) for asset in templates]

    config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "methods": METHODS,
        "truth_indices": truth_indices,
        "nuisance_scenarios": nuisance_scenarios,
        "arguments": vars(args),
        "device": str(device),
        "base_truth": base_truth,
    }
    (output_root / "run_config.json").write_text(
        json.dumps(clean_json(config), indent=2, sort_keys=True), encoding="utf-8"
    )

    initializer_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    simulation_rows: list[dict[str, Any]] = []
    benchmark_started = time.perf_counter()
    total_datasets = len(truth_indices) * int(args.replicates)
    dataset_counter = 0

    for truth_index in truth_indices:
        truth_info = truths[truth_index]
        truth = dict(base_truth)
        truth["advec_lat"] = float(truth_info["advec_lat"])
        truth["advec_lon"] = float(truth_info["advec_lon"])
        for replicate in range(int(args.replicates)):
            dataset_counter += 1
            sim_seed = int(args.seed + truth_index * 10_000 + replicate * 101)
            asset, sim_diag = simulate_asset(
                templates[replicate], truth, sim_seed, args, truth_info["truth_id"], replicate
            )
            simulation_rows.append(sim_diag)
            pd.DataFrame(simulation_rows).to_csv(
                output_root / "simulation_diagnostics.csv", index=False, float_format="%.10f"
            )
            print(
                f"\n[{dataset_counter}/{total_datasets}] {asset.day} "
                f"truth=({truth['advec_lat']:+.3f},{truth['advec_lon']:+.3f})",
                flush=True,
            )

            m3, fft_diag = run_fft_seed(asset, args, robust=False, subgrid=False)
            m3["seed_total_s"] = float(m3["search_s"])
            stats, stats_s = pairwise_sufficient_statistics(asset, args)
            common_model = common_mapped = None
            if not args.initializer_only:
                common_model, common_mapped, _ = build_corridor_model(
                    asset, truth["advec_lat"], truth["advec_lon"], args, device
                )
            try:
                for nuisance_index, nuisance_scenario in enumerate(nuisance_scenarios):
                    nuisance = nuisance_start(truth, nuisance_scenario)
                    seeds = {
                        "M0_zero": zero_seed(),
                        "M3_fft": dict(m3),
                    }
                    seeds["S1_fixed_stencil"] = fixed_stencil_seed(asset, nuisance, args, device)
                    seeds["S2_pairwise_polar"] = pairwise_polar_seed(
                        nuisance, stats, stats_s, args
                    )
                    seeds["S3_fft_gated"] = fft_gated_seed(
                        m3, fft_diag, nuisance, stats, stats_s, args
                    )
                    seeds["S4_iterative_corridor"] = iterative_corridor_seed(
                        asset, m3, nuisance, args, device
                    )

                    for method in METHODS:
                        initializer_rows.append(
                            initializer_row(
                                asset,
                                truth_info,
                                replicate,
                                nuisance_scenario,
                                method,
                                seeds[method],
                                truth,
                            )
                        )
                        seed = seeds[method]
                        print(
                            f"  seed {nuisance_scenario:24s} {method:23s} "
                            f"a=({seed['seed_lat']:+.3f},{seed['seed_lon']:+.3f}) "
                            f"err={initializer_rows[-1]['seed_error_euclid']:.3f} "
                            f"time={initializer_rows[-1]['seed_total_s']:.4f}s",
                            flush=True,
                        )
                    save_results(output_root, initializer_rows, full_rows)

                    if args.initializer_only:
                        continue
                    case_indices: list[int] = []
                    rotation = (truth_index + replicate + nuisance_index) % len(METHODS)
                    method_order = METHODS[rotation:] + METHODS[:rotation]
                    for method in method_order:
                        base = {
                            "dataset_id": asset.day,
                            "truth_id": truth_info["truth_id"],
                            "speed_label": truth_info["speed_label"],
                            "direction": truth_info["direction"],
                            "replicate": replicate,
                            "nuisance_scenario": nuisance_scenario,
                            "method": method,
                            **{f"true_{key}": truth[key] for key in P_LABELS},
                        }
                        try:
                            result = fit_method_corridor(
                                asset,
                                method,
                                seeds[method],
                                nuisance,
                                truth,
                                common_model,
                                args,
                                device,
                            )
                            row = {**base, **result}
                            print(
                                f"  fit  {nuisance_scenario:24s} {method:23s} "
                                f"common={row['common_eval_nll']:.6f} "
                                f"param={row['final_standardized_rmse_7param']:.3f} "
                                f"time={row['end_to_end_s']:.2f}s",
                                flush=True,
                            )
                        except Exception as exc:
                            row = {
                                **base,
                                "status": "error",
                                "error": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc(limit=8),
                                "converged_grad": False,
                            }
                            print("  ERROR", nuisance_scenario, method, row["error"], flush=True)
                        full_rows.append(clean_json(row))
                        case_indices.append(len(full_rows) - 1)
                        save_results(output_root, initializer_rows, full_rows)
                    finite_nll = [
                        float(full_rows[index]["common_eval_nll"])
                        for index in case_indices
                        if full_rows[index].get("status") == "ok"
                        and np.isfinite(full_rows[index].get("common_eval_nll", np.nan))
                    ]
                    best_nll = float(np.min(finite_nll)) if finite_nll else np.nan
                    for index in case_indices:
                        row = full_rows[index]
                        row["common_nll_regret"] = (
                            float(row["common_eval_nll"] - best_nll)
                            if row.get("status") == "ok" and np.isfinite(best_nll)
                            else np.nan
                        )
                    save_results(output_root, initializer_rows, full_rows)
            finally:
                del common_model, common_mapped, asset
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            elapsed = time.perf_counter() - benchmark_started
            print(f"  cumulative runtime: {elapsed / 60.0:.2f} min", flush=True)

    save_results(output_root, initializer_rows, full_rows)
    (output_root / "total_runtime.json").write_text(
        json.dumps({"elapsed_s": time.perf_counter() - benchmark_started}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
