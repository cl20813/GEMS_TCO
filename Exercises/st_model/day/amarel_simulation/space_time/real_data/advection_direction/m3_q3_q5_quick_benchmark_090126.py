#!/usr/bin/env python3
"""Quick paired benchmark of M3, safeguarded Q3, and weighted safeguarded Q5.

The experiment is intentionally small enough for a local run of at most twenty
minutes.  It uses four representative synthetic conditions with one common
misspecified nuisance start, three full real GEMS days for initializer timing,
and one real day for short seven-parameter fits.  Every method uses the same
masked-FFT tau=1 semivariogram surface; only the sub-grid calibration differs.

Q3 fits an unweighted quadratic to the 3x3 neighborhood of the discrete M3
minimum.  Q5 fits a Gaussian-weighted quadratic to the 5x5 neighborhood.  Both
fits use grid-cell coordinates and identical positive-curvature, Hessian
conditioning, and half-cell displacement safeguards.
"""

from __future__ import annotations

import argparse
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
from torch.nn import Parameter


REPO = Path("/Users/joonwonlee/Documents/GEMS_TCO-1")
SRC = REPO / "src"
HERE = Path(__file__).resolve().parent
for path in (REPO, SRC, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from m3_surface_diagnostic_subgrid_benchmark_090126 import (
    SIGNAL_REGIMES,
    load_real_assets,
    pooled_surface,
    transition_components,
)
from synthetic_advection_initializer_benchmark_corridor432_081526 import (
    P_LABELS,
    clean_json,
    load_assets,
    make_hourly_grids,
    parse_pair,
    physical_to_raw,
    raw_to_physical,
    vector_metrics,
)
from synthetic_initializer_factorial_robustness_corridor432_083126 import (
    build_corridor_model,
    crop_regular_asset,
    evaluate_raw,
    fit_method_corridor,
    nuisance_start,
    simulate_asset,
)
from synthetic_m3_reverse_l_initialization_benchmark_corridor432_083126 import (
    surface_minimum,
)


METHODS = ("M3_fft", "M3_fft_Q3", "M3_fft_Q5")
DEFAULT_REAL_INIT = {
    "sigmasq": 13.059,
    "range_lat": 0.20,
    "range_lon": 0.25,
    "range_time": 1.50,
    "advec_lat": 0.0218,
    "advec_lon": -0.1689,
    "nugget": 0.247,
}
QUICK_CASES = (
    # Four quadrants, three speeds, and all three signal regimes.
    {"angle_deg": 22.5, "speed_cells": 0.75, "signal": "weak"},
    {"angle_deg": 112.5, "speed_cells": 2.00, "signal": "reference"},
    {"angle_deg": 202.5, "speed_cells": 4.00, "signal": "strong"},
    {"angle_deg": 292.5, "speed_cells": 2.00, "signal": "weak"},
)


def quadratic_minimum(
    surface: np.ndarray,
    offsets_lat: np.ndarray,
    offsets_lon: np.ndarray,
    lat_step: float,
    lon_step: float,
    radius: int,
    max_condition_number: float,
    gaussian_sigma_cells: float | None,
) -> dict[str, Any]:
    """Return a safeguarded local quadratic vertex in grid-cell coordinates."""
    row, col, grid_lat, grid_lon = surface_minimum(
        surface, offsets_lat, offsets_lon, lat_step, lon_step
    )
    result: dict[str, Any] = {
        "seed_lat": float(grid_lat),
        "seed_lon": float(grid_lon),
        "grid_seed_lat": float(grid_lat),
        "grid_seed_lon": float(grid_lon),
        "subgrid_used": False,
        "selection_reason": "fallback",
        "delta_lat_cells": 0.0,
        "delta_lon_cells": 0.0,
        "hessian_condition_cells": np.nan,
        "hessian_eigenvalue_min_cells": np.nan,
        "quadratic_residual_rmse": np.nan,
        "quadratic_residual_df": int((2 * radius + 1) ** 2 - 6),
    }
    if (
        row < radius
        or col < radius
        or row >= surface.shape[0] - radius
        or col >= surface.shape[1] - radius
    ):
        result["selection_reason"] = "boundary"
        return result
    patch = np.asarray(
        surface[row - radius : row + radius + 1, col - radius : col + radius + 1],
        dtype=np.float64,
    )
    if not np.isfinite(patch).all():
        result["selection_reason"] = "nonfinite_patch"
        return result

    design: list[list[float]] = []
    values: list[float] = []
    weights: list[float] = []
    for local_row, u in enumerate(range(-radius, radius + 1)):
        for local_col, v in enumerate(range(-radius, radius + 1)):
            uf, vf = float(u), float(v)
            design.append([1.0, uf, vf, 0.5 * uf * uf, uf * vf, 0.5 * vf * vf])
            values.append(float(patch[local_row, local_col]))
            if gaussian_sigma_cells is None:
                weights.append(1.0)
            else:
                scale = float(gaussian_sigma_cells)
                weights.append(float(np.exp(-0.5 * (uf * uf + vf * vf) / (scale * scale))))
    x = np.asarray(design, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    sqrt_w = np.sqrt(np.asarray(weights, dtype=np.float64))
    coefficients, *_ = np.linalg.lstsq(x * sqrt_w[:, None], y * sqrt_w, rcond=None)
    gradient = coefficients[1:3]
    hessian = np.asarray(
        [[coefficients[3], coefficients[4]], [coefficients[4], coefficients[5]]],
        dtype=np.float64,
    )
    eigenvalues = np.linalg.eigvalsh(hessian)
    result["hessian_eigenvalue_min_cells"] = float(eigenvalues[0])
    fitted = x @ coefficients
    result["quadratic_residual_rmse"] = float(
        np.sqrt(np.average(np.square(y - fitted), weights=np.square(sqrt_w)))
    )
    if not np.all(np.isfinite(eigenvalues)) or not np.all(eigenvalues > 0.0):
        result["selection_reason"] = "non_positive_hessian"
        return result
    condition = float(np.linalg.cond(hessian))
    result["hessian_condition_cells"] = condition
    if not np.isfinite(condition) or condition > float(max_condition_number):
        result["selection_reason"] = "ill_conditioned_hessian"
        return result
    delta = -np.linalg.solve(hessian, gradient)
    if not np.all(np.isfinite(delta)):
        result["selection_reason"] = "nonfinite_vertex"
        return result
    if abs(float(delta[0])) > 0.5 or abs(float(delta[1])) > 0.5:
        result["selection_reason"] = "delta_outside_half_cell"
        return result
    result.update(
        {
            "seed_lat": float(grid_lat + float(delta[0]) * lat_step),
            "seed_lon": float(grid_lon + float(delta[1]) * lon_step),
            "subgrid_used": True,
            "selection_reason": "accepted",
            "delta_lat_cells": float(delta[0]),
            "delta_lon_cells": float(delta[1]),
        }
    )
    return result


def run_initializers(asset, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    grids, lat_step, lon_step = make_hourly_grids(asset.source_map, asset.grid_coords)
    offsets_lat = np.arange(
        -int(args.empirical_max_lat_offset), int(args.empirical_max_lat_offset) + 1
    )
    offsets_lon = np.arange(
        -int(args.empirical_max_lon_offset), int(args.empirical_max_lon_offset) + 1
    )
    fft_started = time.perf_counter()
    sumsq, counts = transition_components(grids, offsets_lat, offsets_lon)
    pooled = pooled_surface(
        sumsq,
        counts,
        np.arange(sumsq.shape[0]),
        lat_step,
        lon_step,
        int(args.empirical_min_pair_count),
        float(args.empirical_smooth_bandwidth_deg),
    )
    fft_s = float(time.perf_counter() - fft_started)
    _, _, grid_lat, grid_lon = surface_minimum(
        pooled["smoothed"], offsets_lat, offsets_lon, lat_step, lon_step
    )

    q3_started = time.perf_counter()
    q3 = quadratic_minimum(
        pooled["smoothed"],
        offsets_lat,
        offsets_lon,
        lat_step,
        lon_step,
        radius=1,
        max_condition_number=float(args.subgrid_max_condition_number),
        gaussian_sigma_cells=None,
    )
    q3_s = float(time.perf_counter() - q3_started)
    q5_started = time.perf_counter()
    q5 = quadratic_minimum(
        pooled["smoothed"],
        offsets_lat,
        offsets_lon,
        lat_step,
        lon_step,
        radius=2,
        max_condition_number=float(args.subgrid_max_condition_number),
        gaussian_sigma_cells=float(args.q5_gaussian_sigma_cells),
    )
    q5_s = float(time.perf_counter() - q5_started)

    rows = [
        {
            "method": "M3_fft",
            "seed_lat": float(grid_lat),
            "seed_lon": float(grid_lon),
            "seed_total_s": fft_s,
            "subgrid_used": False,
            "selection_reason": "discrete_fft_argmin",
            "quadratic_residual_df": np.nan,
        },
        {"method": "M3_fft_Q3", "seed_total_s": fft_s + q3_s, **q3},
        {"method": "M3_fft_Q5", "seed_total_s": fft_s + q5_s, **q5},
    ]
    common = {
        "lat_step": float(lat_step),
        "lon_step": float(lon_step),
        "fft_surface_s": fft_s,
        "q3_extra_s": q3_s,
        "q5_extra_s": q5_s,
        "initializer_wall_s": float(time.perf_counter() - started),
    }
    for row in rows:
        row.update(common)
    return rows, {**common, "surface": pooled["smoothed"]}


def add_seed_truth_metrics(
    row: dict[str, Any], truth: dict[str, float], lat_step: float, lon_step: float
) -> None:
    row.update(vector_metrics(float(row["seed_lat"]), float(row["seed_lon"]), truth))
    row["seed_error_grid_cells"] = float(
        np.hypot(
            (float(row["seed_lat"]) - float(truth["advec_lat"])) / abs(lat_step),
            (float(row["seed_lon"]) - float(truth["advec_lon"])) / abs(lon_step),
        )
    )


def optimize_real_model(
    model, init: dict[str, float], max_eval: int, args: argparse.Namespace, device: torch.device
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

    def closure():
        nonlocal closure_calls
        closure_calls += 1
        optimizer.zero_grad()
        loss = model.vecchia_batched_likelihood(torch.stack([p.reshape(()) for p in params]))
        loss.backward()
        return loss

    fit_started = time.perf_counter()
    optimizer.step(closure)
    fit_s = float(time.perf_counter() - fit_started)
    raw_final = [float(p.detach().cpu().item()) for p in params]
    optimizer.zero_grad()
    final_loss = model.vecchia_batched_likelihood(torch.stack([p.reshape(()) for p in params]))
    final_loss.backward()
    gradients = [
        float(p.grad.detach().cpu().item()) if p.grad is not None else np.nan for p in params
    ]
    final_grad_inf = float(np.nanmax(np.abs(gradients)))
    final_nll = float(final_loss.detach().cpu().item())
    state = optimizer.state.get(params[0], {})
    n_iter = int(state.get("n_iter", 0))
    func_evals = int(state.get("func_evals", closure_calls))
    est = raw_to_physical(raw_final)
    finite = bool(
        np.isfinite(final_nll)
        and np.isfinite(final_grad_inf)
        and np.all(np.isfinite(raw_final))
        and np.all(np.isfinite([est[key] for key in P_LABELS]))
    )
    return {
        "raw_final": raw_final,
        "est": est,
        "own_initial_nll": float(initial_nll),
        "own_final_nll": final_nll,
        "own_nll_decrease": float(initial_nll - final_nll),
        "final_fit_s": fit_s,
        "optimizer_n_iter": n_iter,
        "optimizer_func_evals": func_evals,
        "final_grad_inf": final_grad_inf,
        "finite_fit": finite,
        "converged_grad": bool(
            finite and final_grad_inf <= float(args.convergence_grad_threshold)
        ),
        "hit_evaluation_budget": bool(func_evals >= int(max_eval)),
    }


def synthetic_conditions(lat_step: float, lon_step: float) -> list[dict[str, Any]]:
    output = []
    for index, case in enumerate(QUICK_CASES):
        theta = math.radians(float(case["angle_deg"]))
        speed = float(case["speed_cells"])
        output.append(
            {
                **case,
                "case_index": index,
                "dataset_id": (
                    f"quick_a{int(case['angle_deg']):03d}_"
                    f"s{str(speed).replace('.', 'p')}_{case['signal']}"
                ),
                "advec_lat": float(speed * lat_step * math.cos(theta)),
                "advec_lon": float(speed * lon_step * math.sin(theta)),
            }
        )
    return output


def run_synthetic(
    args: argparse.Namespace,
    output_root: Path,
    benchmark_started: float,
    wall_limit_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth_path = (
        Path(args.synthetic_data_root)
        / f"{args.year}_july_st_circulant"
        / f"sim_july{args.year}_st_circulant_truth.json"
    )
    input_path = (
        Path(args.synthetic_data_root)
        / f"{args.year}_july_st_circulant"
        / f"sim_july{args.year}_st_circulant_gridded.pkl"
    )
    base_json = json.loads(truth_path.read_text(encoding="utf-8"))
    base_truth = {key: float(base_json[key]) for key in P_LABELS}
    template = load_assets(
        input_path,
        base_json,
        [0],
        parse_pair(args.lat_range, float),
        parse_pair(args.lon_range, float),
        False,
    )[0]
    template = crop_regular_asset(template, int(args.n_lat), int(args.n_lon))
    coords = np.asarray(template.grid_coords)
    lats = np.sort(np.unique(np.round(coords[:, 0], 6)))
    lons = np.sort(np.unique(np.round(coords[:, 1], 6)))
    lat_step = float(np.median(np.diff(lats)))
    lon_step = float(np.median(np.diff(lons)))
    conditions = synthetic_conditions(lat_step, lon_step)
    device = torch.device(args.device)
    initializer_rows: list[dict[str, Any]] = []
    full_rows: list[dict[str, Any]] = []

    for condition in conditions:
        if time.perf_counter() - benchmark_started > wall_limit_s - 240.0:
            print("Synthetic stop: wall-time reserve reached.", flush=True)
            break
        truth = dict(base_truth)
        truth["advec_lat"] = float(condition["advec_lat"])
        truth["advec_lon"] = float(condition["advec_lon"])
        signal = str(condition["signal"])
        truth["range_time"] *= float(SIGNAL_REGIMES[signal]["range_time_factor"])
        truth["nugget"] *= float(SIGNAL_REGIMES[signal]["nugget_factor"])
        sim_seed = int(args.seed + int(condition["case_index"]) * 100_003)
        asset, _ = simulate_asset(
            template,
            truth,
            sim_seed,
            args,
            str(condition["dataset_id"]),
            0,
        )
        seeds, meta = run_initializers(asset, args)
        for seed_row in seeds:
            row = {
                **{key: value for key, value in seed_row.items()},
                "dataset_id": condition["dataset_id"],
                "direction_deg_grid": condition["angle_deg"],
                "speed_cells": condition["speed_cells"],
                "signal_regime": signal,
                "true_advec_lat": truth["advec_lat"],
                "true_advec_lon": truth["advec_lon"],
            }
            add_seed_truth_metrics(row, truth, lat_step, lon_step)
            initializer_rows.append(clean_json(row))

        common_model = None
        try:
            common_model, _, common_precompute_s = build_corridor_model(
                asset, truth["advec_lat"], truth["advec_lon"], args, device
            )
            nuisance = nuisance_start(truth, str(args.synthetic_nuisance_scenario))
            rotation = int(condition["case_index"]) % len(METHODS)
            order = METHODS[rotation:] + METHODS[:rotation]
            by_method = {row["method"]: row for row in seeds}
            for method in order:
                fit_started = time.perf_counter()
                try:
                    result = fit_method_corridor(
                        asset,
                        method,
                        by_method[method],
                        nuisance,
                        truth,
                        common_model,
                        args,
                        device,
                    )
                    result.update(
                        {
                            "dataset_id": condition["dataset_id"],
                            "direction_deg_grid": condition["angle_deg"],
                            "speed_cells": condition["speed_cells"],
                            "signal_regime": signal,
                            "nuisance_scenario": args.synthetic_nuisance_scenario,
                            "common_precompute_s": common_precompute_s,
                            **{f"true_{key}": truth[key] for key in P_LABELS},
                        }
                    )
                    full_rows.append(clean_json(result))
                    seed_error = float(
                        np.hypot(
                            float(by_method[method]["seed_lat"]) - float(truth["advec_lat"]),
                            float(by_method[method]["seed_lon"]) - float(truth["advec_lon"]),
                        )
                    )
                    print(
                        f"SYN {condition['dataset_id']} {method}: "
                        f"seed={seed_error:.4f} "
                        f"final_adv={row_value(result, 'final_seed_error_euclid'):.4f} "
                        f"common_nll={result['common_eval_nll']:.6f} "
                        f"{result['end_to_end_s']:.2f}s",
                        flush=True,
                    )
                except Exception as exc:
                    full_rows.append(
                        {
                            "dataset_id": condition["dataset_id"],
                            "method": method,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(limit=6),
                            "attempt_s": float(time.perf_counter() - fit_started),
                        }
                    )
        finally:
            del common_model, asset
            gc.collect()
        pd.DataFrame(initializer_rows).to_csv(
            output_root / "synthetic_initializer_results.csv", index=False, float_format="%.10f"
        )
        pd.DataFrame(full_rows).to_csv(
            output_root / "synthetic_full_fit_results.csv", index=False, float_format="%.10f"
        )
    return pd.DataFrame(initializer_rows), pd.DataFrame(full_rows)


def row_value(row: dict[str, Any], key: str) -> float:
    value = row.get(key, np.nan)
    return float(value) if value is not None else np.nan


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate package."""
    if frame.empty:
        return "Not run."
    shown = frame.copy()
    for column in shown.select_dtypes(include=[np.number]).columns:
        shown[column] = shown[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.6g}"
        )
    columns = [str(column) for column in shown.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in shown.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    return "\n".join(lines)


def run_real(
    args: argparse.Namespace,
    output_root: Path,
    benchmark_started: float,
    wall_limit_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assets, load_meta = load_real_assets(args)
    initializer_rows: list[dict[str, Any]] = []
    seeds_by_day: dict[str, list[dict[str, Any]]] = {}
    for asset in assets:
        rows, _ = run_initializers(asset, args)
        seeds_by_day[asset.day] = rows
        m3 = rows[0]
        for row in rows:
            movement = float(
                np.hypot(
                    (float(row["seed_lat"]) - float(m3["seed_lat"])) / row["lat_step"],
                    (float(row["seed_lon"]) - float(m3["seed_lon"])) / row["lon_step"],
                )
            )
            initializer_rows.append(
                clean_json(
                    {
                        **row,
                        "day": asset.day,
                        "day_idx": asset.day_idx,
                        "points_per_hour": len(asset.grid_coords),
                        "rows_total": asset.n_total,
                        "rows_valid": asset.n_valid,
                        "movement_from_m3_cells": movement,
                    }
                )
            )
        print(
            f"REAL INIT {asset.day}: "
            + " | ".join(
                f"{row['method']}=({row['seed_lat']:+.4f},{row['seed_lon']:+.4f})"
                for row in rows
            ),
            flush=True,
        )
    initializer = pd.DataFrame(initializer_rows)
    initializer.to_csv(
        output_root / "real_initializer_results.csv", index=False, float_format="%.10f"
    )
    (output_root / "real_loading.json").write_text(
        json.dumps(clean_json(load_meta), indent=2, sort_keys=True), encoding="utf-8"
    )

    full_rows: list[dict[str, Any]] = []
    if not assets or time.perf_counter() - benchmark_started > wall_limit_s - 360.0:
        return initializer, pd.DataFrame(full_rows)
    fit_position = min(max(int(args.real_full_fit_day_position), 0), len(assets) - 1)
    asset = assets[fit_position]
    device = torch.device(args.device)
    common_model = None
    common_mapped = None
    try:
        common_model, common_mapped, common_precompute_s = build_corridor_model(
            asset,
            DEFAULT_REAL_INIT["advec_lat"],
            DEFAULT_REAL_INIT["advec_lon"],
            args,
            device,
        )
        seeds = {row["method"]: row for row in seeds_by_day[asset.day]}
        for method in METHODS:
            if time.perf_counter() - benchmark_started > wall_limit_s - 150.0:
                print("Real full-fit stop: wall-time reserve reached.", flush=True)
                break
            seed = seeds[method]
            model = None
            mapped = None
            attempt_started = time.perf_counter()
            try:
                model, mapped, precompute_s = build_corridor_model(
                    asset, seed["seed_lat"], seed["seed_lon"], args, device
                )
                init = dict(DEFAULT_REAL_INIT)
                init["advec_lat"] = float(seed["seed_lat"])
                init["advec_lon"] = float(seed["seed_lon"])
                fit = optimize_real_model(
                    model, init, int(args.real_max_eval), args, device
                )
                common_nll = evaluate_raw(common_model, fit["raw_final"], device)
                row = {
                    "day": asset.day,
                    "method": method,
                    "status": "ok" if fit["finite_fit"] else "nonfinite",
                    "seed_lat": seed["seed_lat"],
                    "seed_lon": seed["seed_lon"],
                    "seed_total_s": seed["seed_total_s"],
                    "subgrid_used": seed["subgrid_used"],
                    "selection_reason": seed["selection_reason"],
                    "common_eval_nll": float(common_nll),
                    "common_corridor_seed_lat": DEFAULT_REAL_INIT["advec_lat"],
                    "common_corridor_seed_lon": DEFAULT_REAL_INIT["advec_lon"],
                    "common_precompute_s": common_precompute_s,
                    "final_precompute_s": precompute_s,
                    "end_to_end_s": float(seed["seed_total_s"] + precompute_s + fit["final_fit_s"]),
                    **{key: value for key, value in fit.items() if key not in ("raw_final", "est")},
                    **{f"est_{key}": fit["est"][key] for key in P_LABELS},
                }
                full_rows.append(clean_json(row))
                print(
                    f"REAL FIT {asset.day} {method}: common_nll={common_nll:.6f} "
                    f"iter={fit['optimizer_n_iter']} time={row['end_to_end_s']:.2f}s",
                    flush=True,
                )
            except Exception as exc:
                full_rows.append(
                    {
                        "day": asset.day,
                        "method": method,
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=6),
                        "attempt_s": float(time.perf_counter() - attempt_started),
                    }
                )
            finally:
                del model, mapped
                gc.collect()
            pd.DataFrame(full_rows).to_csv(
                output_root / "real_full_fit_results.csv", index=False, float_format="%.10f"
            )
    finally:
        del common_model, common_mapped
        gc.collect()
    return initializer, pd.DataFrame(full_rows)


def summarize(
    output_root: Path,
    synthetic_init: pd.DataFrame,
    synthetic_full: pd.DataFrame,
    real_init: pd.DataFrame,
    real_full: pd.DataFrame,
) -> None:
    if not synthetic_init.empty:
        summary = (
            synthetic_init.groupby("method", as_index=False)
            .agg(
                n=("dataset_id", "size"),
                mean_seed_error_cells=("seed_error_grid_cells", "mean"),
                median_seed_error_cells=("seed_error_grid_cells", "median"),
                mean_angle_error_deg=("seed_angle_error_deg", "mean"),
                correct_quadrant_rate=("correct_quadrant", "mean"),
                safeguard_accept_rate=("subgrid_used", "mean"),
                mean_initializer_s=("seed_total_s", "mean"),
            )
        )
        summary.to_csv(
            output_root / "synthetic_initializer_summary.csv", index=False, float_format="%.10f"
        )
    else:
        summary = pd.DataFrame()

    if not synthetic_full.empty and "status" in synthetic_full:
        ok = synthetic_full[synthetic_full.status.eq("ok")].copy()
        full_summary = (
            synthetic_full.groupby("method", as_index=False)
            .agg(
                n_fits=("dataset_id", "size"),
                success_rate=("status", lambda x: float(np.mean(x == "ok"))),
                convergence_rate=("converged_grad", "mean"),
                evaluation_budget_hit_rate=("hit_evaluation_budget", "mean"),
                mean_common_eval_nll=("common_eval_nll", "mean"),
                mean_final_parameter_error=("final_standardized_rmse_7param", "mean"),
                mean_final_advection_error=("final_seed_error_euclid", "mean"),
                mean_optimizer_iterations=("optimizer_n_iter", "mean"),
                mean_fit_s=("final_fit_s", "mean"),
                mean_end_to_end_s=("end_to_end_s", "mean"),
            )
        )
        full_summary.to_csv(
            output_root / "synthetic_full_fit_summary.csv", index=False, float_format="%.10f"
        )
        paired_rows = []
        for competitor in METHODS[1:]:
            new = ok[ok.method.eq(competitor)].set_index("dataset_id")
            base = ok[ok.method.eq("M3_fft")].set_index("dataset_id")
            common_ids = new.index.intersection(base.index)
            for metric in (
                "common_eval_nll",
                "final_standardized_rmse_7param",
                "final_seed_error_euclid",
                "end_to_end_s",
            ):
                difference = new.loc[common_ids, metric] - base.loc[common_ids, metric]
                paired_rows.append(
                    {
                        "baseline": "M3_fft",
                        "competitor": competitor,
                        "metric": metric,
                        "n_pairs": len(common_ids),
                        "mean_difference_competitor_minus_baseline": difference.mean(),
                        "paired_win_rate_lower_is_better": float((difference < 0).mean()),
                    }
                )
        pd.DataFrame(paired_rows).to_csv(
            output_root / "synthetic_paired_comparisons.csv", index=False, float_format="%.10f"
        )
    else:
        full_summary = pd.DataFrame()

    if not real_init.empty:
        real_summary = (
            real_init.groupby("method", as_index=False)
            .agg(
                n_days=("day", "size"),
                mean_initializer_s=("seed_total_s", "mean"),
                safeguard_accept_rate=("subgrid_used", "mean"),
                mean_movement_from_m3_cells=("movement_from_m3_cells", "mean"),
                max_movement_from_m3_cells=("movement_from_m3_cells", "max"),
            )
        )
        real_summary.to_csv(
            output_root / "real_initializer_summary.csv", index=False, float_format="%.10f"
        )
    else:
        real_summary = pd.DataFrame()

    interpretation: list[str] = []
    if not summary.empty:
        seed_best = summary.loc[summary["mean_seed_error_cells"].idxmin()]
        interpretation.append(
            f"- Synthetic seed accuracy was best on average for {seed_best['method']} "
            f"({seed_best['mean_seed_error_cells']:.3f} grid cells), but n=4 is only a screening sample."
        )
    if not full_summary.empty:
        adv_best = full_summary.loc[full_summary["mean_final_advection_error"].idxmin()]
        parameter_best = full_summary.loc[full_summary["mean_final_parameter_error"].idxmin()]
        interpretation.append(
            f"- Short synthetic full fits gave the lowest mean final advection error to "
            f"{adv_best['method']} and the lowest seven-parameter error to {parameter_best['method']}."
        )
        if bool((full_summary["evaluation_budget_hit_rate"] >= 1.0 - 1e-12).all()):
            interpretation.append(
                "- Every synthetic fit hit the short evaluation budget and none met the gradient "
                "criterion; downstream values are budget-matched screening results, not converged estimates."
            )
    if not real_init.empty:
        movement = real_init.pivot(index="day", columns="method", values="movement_from_m3_cells")
        fallback_days = int(
            (
                np.isclose(movement.get("M3_fft_Q3", np.nan), 0.0)
                & np.isclose(movement.get("M3_fft_Q5", np.nan), 0.0)
            ).sum()
        )
        interpretation.append(
            f"- On real data, both refinements fell back to M3 on {fallback_days}/{len(movement)} days; "
            "when accepted, their correction was at most 0.312 grid cells."
        )
    if not real_full.empty and "common_eval_nll" in real_full:
        real_ok = real_full[real_full.status.eq("ok")]
        if not real_ok.empty:
            real_best = real_ok.loc[real_ok["common_eval_nll"].idxmin()]
            interpretation.append(
                f"- For the one non-fallback real-data short fit, {real_best['method']} had the "
                f"lowest fixed-common-corridor NLL ({real_best['common_eval_nll']:.6f}); all methods "
                "still hit the evaluation budget."
            )

    report = [
        "# M3 / Q3 / Q5 quick benchmark (090126)",
        "",
        "This is a deliberately small screening experiment, not the final confirmatory study.",
        "Q3 uses an unweighted 3x3 fit; Q5 uses a Gaussian-weighted 5x5 fit with sigma=1.25 cells.",
        "Both use cell-scaled coordinates and the same positive-curvature, condition-number, and half-cell safeguards.",
        "",
        "## Synthetic initializer summary",
        "",
        markdown_table(summary),
        "",
        "## Synthetic short full-fit summary",
        "",
        markdown_table(full_summary),
        "",
        "## Real-data initializer summary",
        "",
        markdown_table(real_summary),
        "",
        "## Real-data short full fit",
        "",
        markdown_table(real_full) if not real_full.empty else "Not run within the time reserve.",
        "",
        "## Screening interpretation",
        "",
        "\n".join(interpretation) if interpretation else "No completed comparisons.",
        "",
        "Real-data common NLL is evaluated on the pre-specified production corridor seed "
        "(0.0218, -0.1689), independent of the three candidates. Real data have no known "
        "advection truth, so initializer accuracy claims come from the synthetic component.",
    ]
    (output_root / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--synthetic-data-root",
        type=Path,
        default=Path(
            "/Users/joonwonlee/Documents/GEMS_DATA/simulation/"
            "july_st_circulant_realpattern_smooth0p5"
        ),
    )
    parser.add_argument(
        "--real-data-root", type=Path, default=Path("/Users/joonwonlee/Documents/GEMS_DATA")
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "outputs/summer_26/m3_q3_q5_quick_benchmark_090126",
    )
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--days", default="0,1,2")
    parser.add_argument("--lat-range", default="-3,-1")
    parser.add_argument("--lon-range", default="121,125")
    parser.add_argument("--n-lat", type=int, default=32)
    parser.add_argument("--n-lon", type=int, default=48)
    parser.add_argument("--seed", type=int, default=901263)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smooth", type=float, default=0.5)
    parser.add_argument("--daily-stride", type=int, default=2)
    parser.add_argument("--target-chunk-size", type=int, default=128)
    parser.add_argument("--empirical-max-lat-offset", type=int, default=20)
    parser.add_argument("--empirical-max-lon-offset", type=int, default=20)
    parser.add_argument("--empirical-min-pair-count", type=int, default=1000)
    parser.add_argument("--empirical-smooth-bandwidth-deg", type=float, default=0.063)
    parser.add_argument("--subgrid-max-condition-number", type=float, default=100.0)
    parser.add_argument("--q5-gaussian-sigma-cells", type=float, default=1.25)
    parser.add_argument("--synthetic-nuisance-scenario", default="long_time_low_nugget")
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-history", type=int, default=10)
    parser.add_argument("--final-max-eval", type=int, default=12)
    parser.add_argument("--real-max-eval", type=int, default=8)
    parser.add_argument(
        "--real-full-fit-day-position",
        type=int,
        default=1,
        help="Zero-based position within --days used for the real short full fit.",
    )
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    parser.add_argument("--convergence-grad-threshold", type=float, default=1e-3)
    parser.add_argument("--wall-limit-minutes", type=float, default=19.0)
    parser.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild summaries and REPORT.md from existing CSV files without fitting.",
    )
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--skip-real", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    wall_limit_s = float(args.wall_limit_minutes) * 60.0
    config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "methods": METHODS,
        "quick_cases": QUICK_CASES,
        "real_common_corridor_seed": {
            "advec_lat": DEFAULT_REAL_INIT["advec_lat"],
            "advec_lon": DEFAULT_REAL_INIT["advec_lon"],
        },
        "arguments": vars(args),
    }
    if args.summarize_only:
        def read_if_present(name: str) -> pd.DataFrame:
            path = output_root / name
            return pd.read_csv(path) if path.exists() else pd.DataFrame()

        summarize(
            output_root,
            read_if_present("synthetic_initializer_results.csv"),
            read_if_present("synthetic_full_fit_results.csv"),
            read_if_present("real_initializer_results.csv"),
            read_if_present("real_full_fit_results.csv"),
        )
        print(f"Rebuilt summaries: {output_root / 'REPORT.md'}", flush=True)
        return

    (output_root / "run_config.json").write_text(
        json.dumps(clean_json(config), indent=2, sort_keys=True), encoding="utf-8"
    )

    def read_if_present(name: str) -> pd.DataFrame:
        path = output_root / name
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    if args.skip_synthetic:
        synthetic_init = read_if_present("synthetic_initializer_results.csv")
        synthetic_full = read_if_present("synthetic_full_fit_results.csv")
    else:
        synthetic_init, synthetic_full = run_synthetic(
            args, output_root, started, wall_limit_s
        )
    if args.skip_real:
        real_init = read_if_present("real_initializer_results.csv")
        real_full = read_if_present("real_full_fit_results.csv")
    else:
        real_init, real_full = run_real(args, output_root, started, wall_limit_s)
    summarize(output_root, synthetic_init, synthetic_full, real_init, real_full)
    elapsed = float(time.perf_counter() - started)
    (output_root / "total_runtime.json").write_text(
        json.dumps({"wall_s": elapsed, "wall_minutes": elapsed / 60.0}, indent=2),
        encoding="utf-8",
    )
    print(f"DONE in {elapsed / 60.0:.2f} minutes", flush=True)
    print(f"Report: {output_root / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
