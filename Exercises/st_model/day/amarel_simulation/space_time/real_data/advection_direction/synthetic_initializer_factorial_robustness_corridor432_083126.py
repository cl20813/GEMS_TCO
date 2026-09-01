#!/usr/bin/env python3
"""Paired factorial robustness benchmark for three advection initializers.

The experiment crosses four advection quadrants, two speeds, independent
simulation replicates, and three common nuisance-parameter starts.  The three
methods are:

* M3_empirical: regular-grid tau=1 empirical semivariogram minimum.
* M3_fft_subgrid: the same surface via masked FFT plus safeguarded 3x3
  quadratic interpolation.
* M3_reverseL_s2: M3_fft_subgrid followed by five advection-only likelihood
  evaluations on a stride-2 regular-grid reverse-L Vecchia model.  The fixed
  nuisance values in this refinement follow the nuisance-start scenario.

Each seed builds one fixed directional lag-432 corridor.  All seven covariance
parameters are then optimized with a common budget and common nuisance starts.
Because different seeds create different conditioning sets, final estimates
are additionally evaluated on the same truth-direction (oracle) corridor.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import traceback
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn import Parameter


LOCAL_REPO = Path("/Users/joonwonlee/Documents/GEMS_TCO-1")
AMAREL_REPO = Path("/home/jl2815/tco")
REPO = AMAREL_REPO if AMAREL_REPO.exists() else LOCAL_REPO
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from simulate_data.generate_july_st_circulant_real_locations_2022_2025_smooth0p3_051926 import (
    generate_st_field_block,
    set_seed,
)
from synthetic_advection_initializer_benchmark_corridor432_081526 import (
    DayAsset,
    P_LABELS,
    clean_json,
    empirical_seed,
    load_assets,
    parse_pair,
    physical_to_raw,
    raw_to_physical,
    vector_metrics,
)
from synthetic_m3_reverse_l_initialization_benchmark_corridor432_083126 import (
    reverse_l_refine,
    run_fft_seed,
)

from GEMS_TCO.vecchia_realdata_corridor_width_4x4_lag432 import build_directional_model

METHODS = ("M3_empirical", "M3_fft_subgrid", "M3_reverseL_s2")
POSITIVE_PARAMS = ("sigmasq", "range_lat", "range_lon", "range_time", "nugget")
NUISANCE_SCENARIOS = (
    "oracle_start",
    "short_time_high_nugget",
    "long_time_low_nugget",
)


def truth_grid() -> list[dict[str, Any]]:
    """Four quadrants crossed with half/full baseline speed."""
    out: list[dict[str, Any]] = []
    directions = (
        ("NE", +1.0, +1.0),
        ("NW", +1.0, -1.0),
        ("SW", -1.0, -1.0),
        ("SE", -1.0, +1.0),
    )
    speeds = (("slow", 0.5), ("fast", 1.0))
    for speed_label, scale in speeds:
        for direction, lat_sign, lon_sign in directions:
            lat = lat_sign * scale * 0.08
            lon = lon_sign * scale * 0.20
            out.append(
                {
                    "truth_id": f"{speed_label}_{direction}",
                    "speed_label": speed_label,
                    "speed_scale": scale,
                    "direction": direction,
                    "advec_lat": float(lat),
                    "advec_lon": float(lon),
                    "advec_norm": float(np.hypot(lat, lon)),
                }
            )
    return out


def nuisance_start(truth: dict[str, float], scenario: str) -> dict[str, float]:
    out = dict(truth)
    if scenario == "oracle_start":
        return out
    if scenario == "short_time_high_nugget":
        factors = {
            "sigmasq": 0.75,
            "range_lat": 1.25,
            "range_lon": 1.25,
            "range_time": 0.50,
            "nugget": 1.50,
        }
    elif scenario == "long_time_low_nugget":
        factors = {
            "sigmasq": 1.25,
            "range_lat": 0.75,
            "range_lon": 0.75,
            "range_time": 1.50,
            "nugget": 0.50,
        }
    else:
        raise ValueError(f"Unknown nuisance scenario: {scenario}")
    for key, factor in factors.items():
        out[key] = float(truth[key]) * factor
    return out


def crop_regular_asset(asset: DayAsset, n_lat: int, n_lon: int) -> DayAsset:
    coords = np.asarray(asset.grid_coords, dtype=np.float64)
    lat_key = np.round(coords[:, 0], 6)
    lon_key = np.round(coords[:, 1], 6)
    lats = np.sort(np.unique(lat_key))
    lons = np.sort(np.unique(lon_key))
    if n_lat > len(lats) or n_lon > len(lons):
        raise ValueError(f"Requested {n_lat}x{n_lon}, available {len(lats)}x{len(lons)}")
    keep_lat = set(float(value) for value in lats[:n_lat])
    keep_lon = set(float(value) for value in lons[:n_lon])
    keep = np.asarray(
        [(float(lat) in keep_lat) and (float(lon) in keep_lon) for lat, lon in zip(lat_key, lon_key)]
    )
    selected = np.flatnonzero(keep)
    if len(selected) != n_lat * n_lon:
        raise RuntimeError(f"Regular crop is incomplete: {len(selected)} != {n_lat*n_lon}")
    index = torch.as_tensor(selected, dtype=torch.long)
    grid_coords = coords[selected].copy()
    source_map: dict[str, torch.Tensor] = {}
    for key, value in asset.source_map.items():
        subset = value.index_select(0, index).clone().to(dtype=torch.double)
        subset[:, :2] = torch.as_tensor(grid_coords, dtype=torch.double)
        source_map[key] = subset.contiguous()
    n_total = sum(int(value.shape[0]) for value in source_map.values())
    n_valid = sum(int(torch.isfinite(value[:, 2]).sum().item()) for value in source_map.values())
    return replace(asset, source_map=source_map, grid_coords=grid_coords, n_valid=n_valid, n_total=n_total)


def coordinate_indices(grid_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, int, int]:
    coords = np.asarray(grid_coords, dtype=np.float64)
    lat_key = np.round(coords[:, 0], 6)
    lon_key = np.round(coords[:, 1], 6)
    lats = np.sort(np.unique(lat_key))
    lons = np.sort(np.unique(lon_key))
    lat_lookup = {float(value): idx for idx, value in enumerate(lats)}
    lon_lookup = {float(value): idx for idx, value in enumerate(lons)}
    rows = np.asarray([lat_lookup[float(value)] for value in lat_key], dtype=np.int64)
    cols = np.asarray([lon_lookup[float(value)] for value in lon_key], dtype=np.int64)
    return rows, cols, float(np.median(np.diff(lats))), float(np.median(np.diff(lons))), len(lats), len(lons)


def simulate_asset(
    template: DayAsset,
    truth: dict[str, float],
    simulation_seed: int,
    args: argparse.Namespace,
    truth_id: str,
    replicate: int,
) -> tuple[DayAsset, dict[str, Any]]:
    rows, cols, dlat, dlon, n_lat, n_lon = coordinate_indices(template.grid_coords)
    sim_args = argparse.Namespace(
        advec_lat=float(truth["advec_lat"]),
        advec_lon=float(truth["advec_lon"]),
        range_lat=float(truth["range_lat"]),
        range_lon=float(truth["range_lon"]),
        range_time=float(truth["range_time"]),
        sigmasq=float(truth["sigmasq"]),
        smooth=float(args.smooth),
        spline_n_points=4000,
        spline_r_max=30.0,
    )
    set_seed(int(simulation_seed))
    field, embed_diag = generate_st_field_block(n_lat, n_lon, 8, dlat, dlon, sim_args)
    latent = field.detach().cpu().numpy().astype(np.float64)
    rng = np.random.default_rng(int(simulation_seed) + 100_000)
    observed = latent + rng.normal(0.0, math.sqrt(float(truth["nugget"])), size=latent.shape)

    source_map: dict[str, torch.Tensor] = {}
    for hour, key in enumerate(sorted(template.source_map)):
        base = template.source_map[key].clone().to(dtype=torch.double)
        values = observed[rows, cols, hour]
        valid_mask = torch.isfinite(base[:, 2]).detach().cpu().numpy()
        values = values.copy()
        values[~valid_mask] = np.nan
        if np.isfinite(values).any():
            values[np.isfinite(values)] -= float(np.nanmean(values))
        base[:, :2] = torch.as_tensor(template.grid_coords, dtype=torch.double)
        base[:, 2] = torch.as_tensor(values, dtype=torch.double)
        source_map[key] = base.contiguous()
    del field
    n_total = sum(int(value.shape[0]) for value in source_map.values())
    n_valid = sum(int(torch.isfinite(value[:, 2]).sum().item()) for value in source_map.values())
    asset = DayAsset(
        year=int(args.year),
        month=7,
        day_idx=int(replicate),
        day=f"{truth_id}_rep{replicate:02d}",
        source_map=source_map,
        grid_coords=template.grid_coords.copy(),
        n_valid=n_valid,
        n_total=n_total,
    )
    diagnostics = {
        "truth_id": truth_id,
        "replicate": int(replicate),
        "dataset_id": asset.day,
        "simulation_seed": int(simulation_seed),
        "n_lat": n_lat,
        "n_lon": n_lon,
        "n_valid": n_valid,
        "n_total": n_total,
        **embed_diag,
    }
    return asset, diagnostics


def error_metrics(est: dict[str, float], truth: dict[str, float]) -> dict[str, float]:
    rel = [(float(est[key]) - float(truth[key])) / float(truth[key]) for key in P_LABELS]
    positive_log = [math.log(float(est[key]) / float(truth[key])) for key in POSITIVE_PARAMS]
    speed = float(np.hypot(truth["advec_lat"], truth["advec_lon"]))
    standardized = positive_log + [
        (float(est["advec_lat"]) - float(truth["advec_lat"])) / speed,
        (float(est["advec_lon"]) - float(truth["advec_lon"])) / speed,
    ]
    return {
        "final_rmsre_7param": float(np.sqrt(np.mean(np.square(rel)))),
        "final_positive_log_rmse": float(np.sqrt(np.mean(np.square(positive_log)))),
        "final_standardized_rmse_7param": float(np.sqrt(np.mean(np.square(standardized)))),
        **{f"final_{key}": value for key, value in vector_metrics(est["advec_lat"], est["advec_lon"], truth).items()},
    }


def evaluate_raw(model, raw: Sequence[float], device: torch.device) -> float:
    with torch.no_grad():
        loss = model.vecchia_batched_likelihood(
            torch.as_tensor(list(raw), dtype=torch.double, device=device)
        )
    return float(loss.detach().cpu().item())


def optimize_model(
    model,
    init: dict[str, float],
    truth: dict[str, float],
    max_eval: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    raw_init = physical_to_raw(init)
    params = [Parameter(torch.tensor(value, dtype=torch.double, device=device)) for value in raw_init]
    optimizer = model.set_optimizer(
        params,
        lr=float(args.lbfgs_lr),
        max_iter=int(max_eval),
        max_eval=int(max_eval),
        tolerance_grad=float(args.grad_tol),
        history_size=min(int(args.lbfgs_history), int(max_eval)),
    )
    initial_nll = evaluate_raw(model, raw_init, device)
    closure_calls = 0
    loss_history: list[float] = []

    def closure():
        nonlocal closure_calls
        closure_calls += 1
        optimizer.zero_grad()
        loss = model.vecchia_batched_likelihood(torch.stack([param.reshape(()) for param in params]))
        loss.backward()
        loss_history.append(float(loss.detach().cpu().item()))
        return loss

    started = time.perf_counter()
    optimizer.step(closure)
    fit_s = time.perf_counter() - started
    raw_final = [float(param.detach().cpu().item()) for param in params]
    optimizer.zero_grad()
    final_loss_tensor = model.vecchia_batched_likelihood(torch.stack([param.reshape(()) for param in params]))
    final_loss_tensor.backward()
    gradients = [float(param.grad.detach().cpu().item()) if param.grad is not None else np.nan for param in params]
    final_grad_inf = float(np.nanmax(np.abs(gradients)))
    final_nll = float(final_loss_tensor.detach().cpu().item())
    state = optimizer.state.get(params[0], {})
    optimizer_n_iter = int(state.get("n_iter", 0))
    optimizer_func_evals = int(state.get("func_evals", closure_calls))
    est = raw_to_physical(raw_final)
    finite = bool(
        np.isfinite(final_nll)
        and np.isfinite(final_grad_inf)
        and np.all(np.isfinite(raw_final))
        and np.all(np.isfinite([est[key] for key in P_LABELS]))
    )
    result = {
        "raw_final": raw_final,
        "initial_nll": float(initial_nll),
        "final_nll": final_nll,
        "nll_decrease": float(initial_nll - final_nll),
        "fit_s": float(fit_s),
        "closure_calls": int(closure_calls),
        "optimizer_n_iter": optimizer_n_iter,
        "optimizer_func_evals": optimizer_func_evals,
        "final_grad_inf": final_grad_inf,
        "finite_fit": finite,
        "converged_grad": bool(finite and final_grad_inf <= float(args.convergence_grad_threshold)),
        "hit_iteration_budget": bool(optimizer_n_iter >= int(max_eval)),
        "hit_evaluation_budget": bool(optimizer_func_evals >= int(max_eval)),
        "loss_history": loss_history,
        "est": est,
        **error_metrics(est, truth),
    }
    del optimizer, params, final_loss_tensor
    return result


def build_corridor_model(
    asset: DayAsset,
    reference_lat: float,
    reference_lon: float,
    args: argparse.Namespace,
    device: torch.device,
):
    mapped = {key: value.to(device=device, dtype=torch.double) for key, value in asset.source_map.items()}
    model = build_directional_model(
        smooth=float(args.smooth),
        input_map=mapped,
        grid_coords=asset.grid_coords,
        reference_advec_lat=float(reference_lat),
        reference_advec_lon=float(reference_lon),
        daily_stride=int(args.daily_stride),
        target_chunk_size=int(args.target_chunk_size),
        min_target_points=1,
    )
    started = time.perf_counter()
    model.precompute_conditioning_sets()
    return model, mapped, float(time.perf_counter() - started)


def fit_method_corridor(
    asset: DayAsset,
    method: str,
    seed: dict[str, Any],
    nuisance: dict[str, float],
    truth: dict[str, float],
    common_model,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    seed_lat = float(seed["seed_lat"])
    seed_lon = float(seed["seed_lon"])
    model, mapped, precompute_s = build_corridor_model(asset, seed_lat, seed_lon, args, device)
    init = dict(nuisance)
    init["advec_lat"] = seed_lat
    init["advec_lon"] = seed_lon
    fit = optimize_model(model, init, truth, int(args.final_max_eval), args, device)
    common_nll = evaluate_raw(common_model, fit["raw_final"], device)
    result = {
        "method": method,
        "status": "ok" if fit["finite_fit"] else "nonfinite",
        "seed_lat": seed_lat,
        "seed_lon": seed_lon,
        "seed_total_s": float(seed.get("seed_total_s", seed.get("search_s", 0.0))),
        "seed_closure_calls": int(seed.get("closure_calls", 0)),
        "own_initial_nll": fit["initial_nll"],
        "own_final_nll": fit["final_nll"],
        "own_nll_decrease": fit["nll_decrease"],
        "common_eval_nll": float(common_nll),
        "final_precompute_s": precompute_s,
        "final_fit_s": fit["fit_s"],
        "final_total_s": float(precompute_s + fit["fit_s"]),
        "end_to_end_s": float(seed.get("seed_total_s", seed.get("search_s", 0.0)) + precompute_s + fit["fit_s"]),
        "final_closure_calls": fit["closure_calls"],
        "optimizer_n_iter": fit["optimizer_n_iter"],
        "optimizer_func_evals": fit["optimizer_func_evals"],
        "final_grad_inf": fit["final_grad_inf"],
        "converged_grad": fit["converged_grad"],
        "hit_iteration_budget": fit["hit_iteration_budget"],
        "hit_evaluation_budget": fit["hit_evaluation_budget"],
        "finite_fit": fit["finite_fit"],
        "loss_history": json.dumps(clean_json(fit["loss_history"])),
        **{f"est_{key}": float(fit["est"][key]) for key in P_LABELS},
        **fit_error_columns(fit),
    }
    del model, mapped
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def fit_error_columns(fit: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "final_rmsre_7param",
        "final_positive_log_rmse",
        "final_standardized_rmse_7param",
        "final_seed_error_euclid",
        "final_seed_angle_error_deg",
        "final_seed_norm",
        "final_true_advec_norm",
        "final_seed_norm_abs_error",
        "final_correct_quadrant",
    )
    return {key: fit[key] for key in keys}


def initializer_record(
    asset: DayAsset,
    truth_info: dict[str, Any],
    replicate: int,
    nuisance_scenario: str,
    method: str,
    result: dict[str, Any],
    truth: dict[str, float],
) -> dict[str, Any]:
    metrics = vector_metrics(float(result["seed_lat"]), float(result["seed_lon"]), truth)
    return {
        "dataset_id": asset.day,
        "truth_id": truth_info["truth_id"],
        "speed_label": truth_info["speed_label"],
        "direction": truth_info["direction"],
        "replicate": int(replicate),
        "nuisance_scenario": nuisance_scenario,
        "method": method,
        "status": "ok",
        "seed_lat": float(result["seed_lat"]),
        "seed_lon": float(result["seed_lon"]),
        "seed_total_s": float(result.get("search_s", 0.0) + result.get("reverse_l_precompute_s", 0.0)),
        "closure_calls": int(result.get("closure_calls", 0)),
        "subgrid_accepted": result.get("subgrid_accepted", np.nan),
        "subgrid_reason": result.get("subgrid_reason", ""),
        **metrics,
    }


def paired_bootstrap_summary(full: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    metrics = (
        "common_nll_regret",
        "final_standardized_rmse_7param",
        "final_seed_error_euclid",
        "final_fit_s",
        "end_to_end_s",
    )
    baseline = "M3_empirical"
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(int(seed))
    key_cols = ["dataset_id", "nuisance_scenario"]
    ok = full[full["status"].eq("ok")].copy()
    for competitor in (method for method in METHODS if method != baseline):
        left = ok[ok.method.eq(competitor)]
        right = ok[ok.method.eq(baseline)]
        paired = left.merge(right, on=key_cols, suffixes=("_new", "_base"))
        datasets = paired["dataset_id"].unique()
        for metric in metrics:
            diff = paired[f"{metric}_new"] - paired[f"{metric}_base"]
            boot = []
            if len(datasets):
                for _ in range(int(n_boot)):
                    chosen = rng.choice(datasets, size=len(datasets), replace=True)
                    sampled = pd.concat([paired[paired.dataset_id.eq(item)] for item in chosen], ignore_index=True)
                    boot.append(float((sampled[f"{metric}_new"] - sampled[f"{metric}_base"]).mean()))
            rows.append(
                {
                    "baseline": baseline,
                    "competitor": competitor,
                    "metric": metric,
                    "n_pairs": int(len(paired)),
                    "mean_paired_difference_new_minus_base": float(diff.mean()),
                    "median_paired_difference_new_minus_base": float(diff.median()),
                    "paired_win_rate_lower_is_better": float((diff < 0).mean()),
                    "cluster_bootstrap_ci_low": float(np.quantile(boot, 0.025)) if boot else np.nan,
                    "cluster_bootstrap_ci_high": float(np.quantile(boot, 0.975)) if boot else np.nan,
                }
            )
    return pd.DataFrame(rows)


def parameter_variance_tables(full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ok = full[full.status.eq("ok")].copy()
    rows: list[dict[str, Any]] = []
    group_cols = ["truth_id", "speed_label", "direction", "nuisance_scenario", "method"]
    for keys, group in ok.groupby(group_cols, sort=False):
        truth_id, speed_label, direction, nuisance_scenario, method = keys
        truth_speed = float(group["true_advec_norm"].iloc[0])
        standardized = []
        row: dict[str, Any] = {
            "truth_id": truth_id,
            "speed_label": speed_label,
            "direction": direction,
            "nuisance_scenario": nuisance_scenario,
            "method": method,
            "n_replicates": int(len(group)),
        }
        for key in P_LABELS:
            values = group[f"est_{key}"].to_numpy(dtype=float)
            row[f"var_est_{key}"] = float(np.var(values, ddof=1)) if len(values) > 1 else np.nan
            if key in POSITIVE_PARAMS:
                standardized.append(np.log(values / group[f"true_{key}"].to_numpy(dtype=float)))
            else:
                standardized.append(values / truth_speed)
        matrix = np.column_stack(standardized)
        variances = np.var(matrix, axis=0, ddof=1) if len(group) > 1 else np.full(len(P_LABELS), np.nan)
        row["mean_standardized_parameter_variance"] = float(np.mean(variances))
        row["trace_standardized_parameter_covariance"] = float(np.sum(variances))
        rows.append(row)
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby(["method", "nuisance_scenario"], as_index=False)
        .agg(
            n_truth_conditions=("truth_id", "size"),
            mean_standardized_parameter_variance=("mean_standardized_parameter_variance", "mean"),
            median_standardized_parameter_variance=("mean_standardized_parameter_variance", "median"),
            max_standardized_parameter_variance=("mean_standardized_parameter_variance", "max"),
        )
    )
    return detail, summary


def write_summaries(output_root: Path, full_rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    full = pd.DataFrame(full_rows)
    if full.empty:
        return
    full.to_csv(output_root / "full_fit_results.csv", index=False, float_format="%.10f")
    ok = full[full.status.eq("ok")].copy()
    overall = (
        full.groupby("method", as_index=False)
        .agg(
            n_cases=("dataset_id", "size"),
            success_rate=("status", lambda x: float(np.mean(x == "ok"))),
            convergence_rate=("converged_grad", "mean"),
            local_optimum_failure_rate=("local_optimum_failure", "mean"),
            parameter_failure_rate=("parameter_failure", "mean"),
            any_failure_rate=("any_failure", "mean"),
            mean_common_nll_regret=("common_nll_regret", "mean"),
            mean_final_parameter_error=("final_standardized_rmse_7param", "mean"),
            median_final_parameter_error=("final_standardized_rmse_7param", "median"),
            mean_final_advection_error=("final_seed_error_euclid", "mean"),
            mean_optimizer_iterations=("optimizer_n_iter", "mean"),
            mean_function_evals=("optimizer_func_evals", "mean"),
            mean_final_fit_s=("final_fit_s", "mean"),
            mean_end_to_end_s=("end_to_end_s", "mean"),
        )
    )
    overall.to_csv(output_root / "overall_summary.csv", index=False, float_format="%.10f")
    scenario = (
        full.groupby(["method", "nuisance_scenario"], as_index=False)
        .agg(
            n_cases=("dataset_id", "size"),
            success_rate=("status", lambda x: float(np.mean(x == "ok"))),
            convergence_rate=("converged_grad", "mean"),
            local_optimum_failure_rate=("local_optimum_failure", "mean"),
            parameter_failure_rate=("parameter_failure", "mean"),
            mean_common_nll_regret=("common_nll_regret", "mean"),
            mean_final_parameter_error=("final_standardized_rmse_7param", "mean"),
            mean_final_advection_error=("final_seed_error_euclid", "mean"),
            mean_optimizer_iterations=("optimizer_n_iter", "mean"),
            mean_final_fit_s=("final_fit_s", "mean"),
            mean_end_to_end_s=("end_to_end_s", "mean"),
        )
    )
    scenario.to_csv(output_root / "nuisance_summary.csv", index=False, float_format="%.10f")
    truth_summary = (
        ok.groupby(["method", "speed_label", "direction"], as_index=False)
        .agg(
            n_cases=("dataset_id", "size"),
            mean_common_nll_regret=("common_nll_regret", "mean"),
            mean_final_parameter_error=("final_standardized_rmse_7param", "mean"),
            mean_final_advection_error=("final_seed_error_euclid", "mean"),
            local_optimum_failure_rate=("local_optimum_failure", "mean"),
        )
    )
    truth_summary.to_csv(output_root / "direction_speed_summary.csv", index=False, float_format="%.10f")
    paired_bootstrap_summary(full, int(args.bootstrap_replicates), int(args.seed)).to_csv(
        output_root / "paired_comparisons.csv", index=False, float_format="%.10f"
    )
    variance_detail, variance_summary = parameter_variance_tables(full)
    variance_detail.to_csv(output_root / "parameter_variance_by_truth.csv", index=False, float_format="%.10f")
    variance_summary.to_csv(output_root / "parameter_variance_summary.csv", index=False, float_format="%.10f")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/Users/joonwonlee/Documents/GEMS_DATA/simulation/july_st_circulant_realpattern_smooth0p5"),
    )
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--truth-indices", default="all")
    parser.add_argument("--nuisance-scenarios", default=",".join(NUISANCE_SCENARIOS))
    parser.add_argument("--lat-range", default="-3,-1")
    parser.add_argument("--lon-range", default="121,125")
    parser.add_argument("--n-lat", type=int, default=32)
    parser.add_argument("--n-lon", type=int, default=48)
    parser.add_argument("--seed", type=int, default=831260)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "outputs/summer_26/synthetic_initializer_factorial_robustness_corridor432_083126",
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
    parser.add_argument("--subgrid-max-condition-number", type=float, default=100.0)
    parser.add_argument("--reverse-l-head-right-cols", type=int, default=0)
    parser.add_argument("--reverse-l-above-count", type=int, default=2)
    parser.add_argument("--reverse-l-right-col-count", type=int, default=3)
    parser.add_argument("--reverse-l-per-lag-count", type=int, default=14)
    parser.add_argument("--reverse-l-lag-count", type=int, default=2)
    parser.add_argument("--reverse-l-target-chunk-size", type=int, default=1024)
    parser.add_argument("--reverse-l-max-eval", type=int, default=5)
    parser.add_argument("--final-max-eval", type=int, default=20)
    parser.add_argument("--reference-max-eval", type=int, default=35)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-history", type=int, default=10)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    parser.add_argument("--convergence-grad-threshold", type=float, default=1e-3)
    parser.add_argument("--local-nll-regret-threshold", type=float, default=0.002)
    parser.add_argument("--parameter-failure-threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    base_truth_path = (
        Path(args.data_root)
        / f"{args.year}_july_st_circulant"
        / f"sim_july{args.year}_st_circulant_truth.json"
    )
    input_path = (
        Path(args.data_root)
        / f"{args.year}_july_st_circulant"
        / f"sim_july{args.year}_st_circulant_gridded.pkl"
    )
    base_truth_json = json.loads(base_truth_path.read_text(encoding="utf-8"))
    base_truth = {key: float(base_truth_json[key]) for key in P_LABELS}
    all_truths = truth_grid()
    if str(args.truth_indices).lower() == "all":
        selected_truths = list(enumerate(all_truths))
    else:
        indices = [int(token) for token in str(args.truth_indices).split(",") if token.strip()]
        selected_truths = [(idx, all_truths[idx]) for idx in indices]
    nuisance_scenarios = [token.strip() for token in str(args.nuisance_scenarios).split(",") if token.strip()]
    unknown = sorted(set(nuisance_scenarios) - set(NUISANCE_SCENARIOS))
    if unknown:
        raise ValueError(f"Unknown nuisance scenarios: {unknown}")
    lat_range = parse_pair(args.lat_range, float)
    lon_range = parse_pair(args.lon_range, float)
    template_truth = dict(base_truth_json)
    templates = load_assets(
        input_path,
        template_truth,
        list(range(int(args.replicates))),
        lat_range,
        lon_range,
        False,
    )
    templates = [crop_regular_asset(asset, int(args.n_lat), int(args.n_lon)) for asset in templates]

    config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "input_template": input_path,
        "base_truth": base_truth,
        "truths": [item for _, item in selected_truths],
        "nuisance_scenarios": nuisance_scenarios,
        "methods": METHODS,
        "failure_definitions": {
            "convergence_failure": f"final gradient infinity norm > {args.convergence_grad_threshold}",
            "local_optimum_failure": f"common oracle-corridor NLL regret > {args.local_nll_regret_threshold}",
            "parameter_failure": f"standardized 7-parameter RMSE > {args.parameter_failure_threshold}",
        },
        "arguments": vars(args),
        "device": str(device),
    }
    (output_root / "run_config.json").write_text(
        json.dumps(clean_json(config), indent=2, sort_keys=True), encoding="utf-8"
    )

    init_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []
    reference_rows: list[dict[str, Any]] = []
    simulation_rows: list[dict[str, Any]] = []
    benchmark_started = time.perf_counter()
    total_datasets = len(selected_truths) * int(args.replicates)
    dataset_counter = 0
    print("methods:", METHODS, flush=True)
    print("truth conditions:", [item["truth_id"] for _, item in selected_truths], flush=True)
    print("nuisance scenarios:", nuisance_scenarios, flush=True)
    print("replicates:", args.replicates, "grid:", f"{args.n_lat}x{args.n_lon}", "device:", device, flush=True)

    for truth_index, truth_info in selected_truths:
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
                f"truth=({truth['advec_lat']:+.3f},{truth['advec_lon']:+.3f}) "
                f"valid={asset.n_valid}/{asset.n_total}",
                flush=True,
            )

            m3_started = time.perf_counter()
            m3 = empirical_seed(asset, args)
            m3["seed_total_s"] = float(time.perf_counter() - m3_started)
            fft, _ = run_fft_seed(asset, args, robust=False, subgrid=True)
            fft["seed_total_s"] = float(fft["search_s"])

            common_model = None
            common_mapped = None
            dataset_row_indices: list[int] = []
            try:
                common_model, common_mapped, common_precompute_s = build_corridor_model(
                    asset, truth["advec_lat"], truth["advec_lon"], args, device
                )
                reference = optimize_model(
                    common_model, truth, truth, int(args.reference_max_eval), args, device
                )
                reference_rows.append(
                    {
                        "dataset_id": asset.day,
                        "truth_id": truth_info["truth_id"],
                        "speed_label": truth_info["speed_label"],
                        "direction": truth_info["direction"],
                        "replicate": replicate,
                        "status": "ok" if reference["finite_fit"] else "nonfinite",
                        "common_precompute_s": common_precompute_s,
                        "reference_nll": reference["final_nll"],
                        "reference_fit_s": reference["fit_s"],
                        "reference_n_iter": reference["optimizer_n_iter"],
                        "reference_func_evals": reference["optimizer_func_evals"],
                        "reference_grad_inf": reference["final_grad_inf"],
                        **{f"reference_est_{key}": reference["est"][key] for key in P_LABELS},
                    }
                )
                pd.DataFrame(reference_rows).to_csv(
                    output_root / "reference_results.csv", index=False, float_format="%.10f"
                )

                for nuisance_index, nuisance_scenario in enumerate(nuisance_scenarios):
                    nuisance = nuisance_start(truth, nuisance_scenario)
                    reverse_l, reverse_l_diag = reverse_l_refine(
                        asset, fft, nuisance, args, device, spatial_stride=2
                    )
                    reverse_l["seed_total_s"] = float(
                        fft["seed_total_s"]
                        + reverse_l.get("reverse_l_precompute_s", 0.0)
                        + reverse_l.get("reverse_l_refine_s", 0.0)
                    )
                    seeds = {
                        "M3_empirical": m3,
                        "M3_fft_subgrid": fft,
                        "M3_reverseL_s2": reverse_l,
                    }
                    for method in METHODS:
                        init_rows.append(
                            initializer_record(
                                asset,
                                truth_info,
                                replicate,
                                nuisance_scenario,
                                method,
                                seeds[method],
                                truth,
                            )
                        )
                    pd.DataFrame(init_rows).to_csv(
                        output_root / "initializer_results.csv", index=False, float_format="%.10f"
                    )

                    rotation = (truth_index + replicate + nuisance_index) % len(METHODS)
                    method_order = METHODS[rotation:] + METHODS[:rotation]
                    for method in method_order:
                        attempt_started = time.perf_counter()
                        base = {
                            "dataset_id": asset.day,
                            "truth_id": truth_info["truth_id"],
                            "speed_label": truth_info["speed_label"],
                            "speed_scale": truth_info["speed_scale"],
                            "direction": truth_info["direction"],
                            "replicate": replicate,
                            "simulation_seed": sim_seed,
                            "nuisance_scenario": nuisance_scenario,
                            "method": method,
                            **{f"true_{key}": truth[key] for key in P_LABELS},
                            "true_advec_norm": float(truth_info["advec_norm"]),
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
                                f"  {nuisance_scenario:24s} {method:18s} "
                                f"common={row['common_eval_nll']:.6f} "
                                f"err={row['final_standardized_rmse_7param']:.3f} "
                                f"iter={row['optimizer_n_iter']:2d} time={row['end_to_end_s']:.2f}s",
                                flush=True,
                            )
                        except Exception as exc:
                            row = {
                                **base,
                                "status": "error",
                                "error": f"{type(exc).__name__}: {exc}",
                                "traceback": traceback.format_exc(limit=8),
                                "attempt_s": time.perf_counter() - attempt_started,
                                "converged_grad": False,
                                "finite_fit": False,
                            }
                            print("  ERROR", nuisance_scenario, method, row["error"], flush=True)
                        full_rows.append(clean_json(row))
                        dataset_row_indices.append(len(full_rows) - 1)
                        pd.DataFrame(full_rows).to_csv(
                            output_root / "full_fit_results.csv", index=False, float_format="%.10f"
                        )

                candidates = [
                    float(reference_rows[-1]["reference_nll"]),
                    *[
                        float(full_rows[idx]["common_eval_nll"])
                        for idx in dataset_row_indices
                        if full_rows[idx].get("status") == "ok"
                        and np.isfinite(full_rows[idx].get("common_eval_nll", np.nan))
                    ],
                ]
                best_common = float(np.min(candidates))
                for idx in dataset_row_indices:
                    row = full_rows[idx]
                    if row.get("status") == "ok":
                        regret = float(row["common_eval_nll"] - best_common)
                        local_failure = bool(regret > float(args.local_nll_regret_threshold))
                        parameter_failure = bool(
                            float(row["final_standardized_rmse_7param"])
                            > float(args.parameter_failure_threshold)
                        )
                        convergence_failure = not bool(row.get("converged_grad", False))
                        row.update(
                            {
                                "best_observed_common_nll": best_common,
                                "common_nll_regret": regret,
                                "local_optimum_failure": local_failure,
                                "parameter_failure": parameter_failure,
                                "convergence_failure": convergence_failure,
                                "any_failure": bool(local_failure or parameter_failure or convergence_failure),
                            }
                        )
                    else:
                        row.update(
                            {
                                "best_observed_common_nll": best_common,
                                "common_nll_regret": np.nan,
                                "local_optimum_failure": True,
                                "parameter_failure": True,
                                "convergence_failure": True,
                                "any_failure": True,
                            }
                        )
                pd.DataFrame(full_rows).to_csv(
                    output_root / "full_fit_results.csv", index=False, float_format="%.10f"
                )
            finally:
                del common_model, common_mapped
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                del asset
            elapsed = time.perf_counter() - benchmark_started
            print(f"  completed dataset in cumulative {elapsed/60.0:.2f} min", flush=True)

    write_summaries(output_root, full_rows, args)
    init = pd.DataFrame(init_rows)
    if not init.empty:
        init_summary = (
            init.groupby(["method", "nuisance_scenario"], as_index=False)
            .agg(
                n_cases=("dataset_id", "size"),
                mean_seed_error=("seed_error_euclid", "mean"),
                median_seed_error=("seed_error_euclid", "median"),
                max_seed_error=("seed_error_euclid", "max"),
                mean_seed_s=("seed_total_s", "mean"),
                correct_quadrant_rate=("correct_quadrant", "mean"),
            )
        )
        init_summary.to_csv(
            output_root / "initializer_summary.csv", index=False, float_format="%.10f"
        )
    elapsed = time.perf_counter() - benchmark_started
    (output_root / "total_runtime.json").write_text(
        json.dumps(
            {"benchmark_wall_s": elapsed, "benchmark_wall_minutes": elapsed / 60.0},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nDONE in {elapsed:.2f}s ({elapsed/60.0:.2f} min)", flush=True)
    if full_rows:
        print(pd.read_csv(output_root / "overall_summary.csv").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
