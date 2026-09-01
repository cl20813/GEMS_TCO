#!/usr/bin/env python3
"""Compare three fast advection seeds before a directional lag-432 fit.

The experiment deliberately separates two jobs:

1. Estimate a cheap, once-per-day advection seed.
2. Freeze the Vecchia conditioning geometry at that seed and run the same
   full-data directional corridor 4x4 lag-432 fit for every method.

Seed methods
------------
``pilot400``
    Fit a direction-neutral pointwise Vecchia model on the first 400 max-min
    spatial locations per hour for two outer L-BFGS calls.

``quadrant300``
    Fit the same pilot likelihood on 300 max-min locations per hour from four
    requested advection starts.  Each start gets a short L-BFGS run and the
    candidate with the smallest re-evaluated pilot NLL supplies the seed.

``empirical``
    Use the minimum of the pair-count-filtered, Gaussian-smoothed empirical
    cross-semivariogram.  The implementation is adapted from
    GEMS_TCO_EDA/semivariograms/advection_ridge_three_model_compare_081526.ipynb.

The final model uses past conditioning points near ``-v * lag`` because the
covariance distance is based on ``h - v * tau``.  All methods use identical
final optimizer and lag-432 budgets, so reported timing and NLL differences are
attributable to seed acquisition and the resulting fixed corridor geometry.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter
from torch.nn import Parameter


LOCAL_REPO = Path("/Users/joonwonlee/Documents/GEMS_TCO-1")
AMAREL_REPO = Path("/home/jl2815/tco")
REPO = AMAREL_REPO if AMAREL_REPO.exists() else LOCAL_REPO
SRC = REPO if (REPO / "GEMS_TCO").exists() else REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from GEMS_TCO import configuration as config
from GEMS_TCO import orderings
from GEMS_TCO.data_loader import load_data_dynamic_processed
from GEMS_TCO.matern_vecchia_engine import fit_vecchia_lbfgs as PointwisePilotVecchia
from GEMS_TCO.vecchia_realdata_corridor_width_4x4_lag432 import (
    BLOCK_SHAPE,
    DIRECTIONAL_SPEC_NAME,
    LAG_COUNTS,
    build_directional_model,
    directional_model_spec,
)


METHODS = ("pilot400", "quadrant300", "empirical")
P_LABELS = (
    "sigmasq",
    "range_lat",
    "range_lon",
    "range_time",
    "advec_lat",
    "advec_lon",
    "nugget",
)
DEFAULT_INIT_PHYSICAL = {
    "sigmasq": 13.059,
    "range_lat": 0.20,
    "range_lon": 0.25,
    "range_time": 1.50,
    "advec_lat": 0.0218,
    "advec_lon": -0.1689,
    "nugget": 0.247,
}
QUADRANT_STARTS = (
    (0.01, 0.10),
    (0.01, -0.10),
    (-0.01, -0.10),
    (-0.01, 0.10),
)


def parse_pair(text: str, cast=float) -> list[Any]:
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected two comma-separated values, got {text!r}")
    return [cast(parts[0]), cast(parts[1])]


def parse_days(text: str) -> list[int]:
    token = str(text).strip().lower()
    if token == "all":
        return list(range(31))
    parts = [part.strip() for part in token.split(",") if part.strip()]
    if len(parts) == 2:
        start, stop = (int(x) for x in parts)
        return list(range(start, stop))
    return [int(x) for x in parts]


def parse_methods(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        out.extend(part.strip() for part in str(value).split(",") if part.strip())
    unknown = sorted(set(out) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown methods {unknown}; choose from {METHODS}")
    return list(dict.fromkeys(out))


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
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


def physical_to_log_phi(params: dict[str, float]) -> list[float]:
    sigmasq = float(params["sigmasq"])
    range_lat = float(params["range_lat"])
    range_lon = float(params["range_lon"])
    range_time = float(params["range_time"])
    nugget = float(params["nugget"])
    phi2 = 1.0 / range_lon
    return [
        float(np.log(sigmasq * phi2)),
        float(np.log(phi2)),
        float(np.log((range_lon / range_lat) ** 2)),
        float(np.log((range_lon / range_time) ** 2)),
        float(params["advec_lat"]),
        float(params["advec_lon"]),
        float(np.log(nugget)),
    ]


def backmap_params(raw_params: Sequence[float]) -> dict[str, float]:
    raw = [float(x) for x in raw_params[:7]]
    phi2 = float(np.exp(raw[1]))
    range_lon = 1.0 / phi2
    return {
        "sigmasq": float(np.exp(raw[0]) / phi2),
        "range_lat": float(range_lon / np.sqrt(np.exp(raw[2]))),
        "range_lon": float(range_lon),
        "range_time": float(range_lon / np.sqrt(np.exp(raw[3]))),
        "advec_lat": raw[4],
        "advec_lon": raw[5],
        "nugget": float(np.exp(raw[6])),
    }


def make_params_list(
    init_physical: dict[str, float],
    dtype: torch.dtype,
    device: torch.device,
) -> list[Parameter]:
    return [
        Parameter(torch.tensor([value], dtype=dtype, device=device))
        for value in physical_to_log_phi(init_physical)
    ]


def count_valid(day_map: dict[str, torch.Tensor]) -> tuple[int, int]:
    total = sum(int(tensor.shape[0]) for tensor in day_map.values())
    valid = sum(int((~torch.isnan(tensor[:, 2])).sum().item()) for tensor in day_map.values())
    return valid, total


def assert_grid_order_consistent(
    df_map: dict[str, pd.DataFrame],
    keys: Sequence[str],
    base_coords: np.ndarray,
) -> None:
    for key in keys:
        coords = df_map[key][["Latitude", "Longitude"]].to_numpy(dtype=np.float64)
        if coords.shape != base_coords.shape or not np.allclose(coords, base_coords, equal_nan=True):
            raise RuntimeError(f"Regular grid coordinate order differs at {key}")


def reevaluate_nll(model, params_list: Sequence[torch.Tensor]) -> float:
    params = torch.stack([param.reshape(()) for param in params_list])
    with torch.no_grad():
        value = model.vecchia_batched_likelihood(params)
    return float(value.detach().cpu().item())


def fit_model(
    model,
    init_physical: dict[str, float],
    dtype: torch.dtype,
    device: torch.device,
    outer_steps: int,
    lbfgs_eval: int,
    lbfgs_history: int,
    lbfgs_lr: float,
    grad_tol: float,
    suppress_prints: bool,
) -> dict[str, Any]:
    params_list = make_params_list(init_physical, dtype=dtype, device=device)
    optimizer = model.set_optimizer(
        params_list,
        lr=lbfgs_lr,
        max_iter=lbfgs_eval,
        max_eval=lbfgs_eval,
        history_size=lbfgs_history,
    )
    started = time.perf_counter()
    stream = contextlib.redirect_stdout(io.StringIO()) if suppress_prints else contextlib.nullcontext()
    with stream:
        out, steps_raw = model.fit_vecc_lbfgs(
            params_list,
            optimizer,
            max_steps=outer_steps,
            grad_tol=grad_tol,
        )
    fit_s = time.perf_counter() - started
    final_nll = reevaluate_nll(model, params_list)
    raw = [float(param.detach().cpu().item()) for param in params_list]
    return {
        "raw": raw,
        "est": backmap_params(raw),
        "nll": final_nll,
        "optimizer_reported_loss": float(out[-1]),
        "steps": int(steps_raw) + 1,
        "fit_s": float(fit_s),
    }


def make_pilot_map(
    day_map: dict[str, torch.Tensor],
    maxmin_order: np.ndarray,
    n_points: int,
) -> dict[str, torch.Tensor]:
    keep = torch.as_tensor(maxmin_order[:n_points], dtype=torch.long)
    return {key: tensor.index_select(0, keep).contiguous() for key, tensor in day_map.items()}


def run_pilot_candidates(
    day_map: dict[str, torch.Tensor],
    ordered_grid_coords: np.ndarray,
    nns_map: np.ndarray,
    n_points: int,
    starts: Sequence[tuple[str, dict[str, float]]],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[list[dict[str, Any]], float, float]:
    pilot_cpu = make_pilot_map(day_map, args._maxmin_order, n_points)
    pilot_map = {key: value.to(device) for key, value in pilot_cpu.items()}
    model = PointwisePilotVecchia(
        smooth=args.smooth,
        input_map=pilot_map,
        nns_map=nns_map[:n_points],
        mm_cond_number=args.pilot_neighbors,
        nheads=args.pilot_nheads,
        limit_A=args.pilot_limit_a,
        limit_B=args.pilot_limit_b,
        limit_C=args.pilot_limit_c,
        daily_stride=args.daily_stride,
    )
    started = time.perf_counter()
    stream = contextlib.redirect_stdout(io.StringIO()) if args.suppress_fit_prints else contextlib.nullcontext()
    with stream:
        model.precompute_conditioning_sets()
    precompute_s = time.perf_counter() - started

    candidates: list[dict[str, Any]] = []
    for label, init in starts:
        try:
            fit = fit_model(
                model=model,
                init_physical=init,
                dtype=torch.double,
                device=device,
                outer_steps=args._pilot_outer_steps,
                lbfgs_eval=args.pilot_lbfgs_eval,
                lbfgs_history=args.pilot_lbfgs_history,
                lbfgs_lr=args.lbfgs_lr,
                grad_tol=args.grad_tol,
                suppress_prints=args.suppress_fit_prints,
            )
            candidates.append(
                {
                    "candidate": label,
                    "status": "ok" if np.isfinite(fit["nll"]) else "nonfinite",
                    "nll": fit["nll"],
                    "init_advec_lat": init["advec_lat"],
                    "init_advec_lon": init["advec_lon"],
                    "seed_raw_lat": fit["est"]["advec_lat"],
                    "seed_raw_lon": fit["est"]["advec_lon"],
                    "fit_s": fit["fit_s"],
                    "steps": fit["steps"],
                    **{f"pilot_est_{key}": value for key, value in fit["est"].items()},
                }
            )
        except Exception as exc:
            candidates.append(
                {
                    "candidate": label,
                    "status": "error",
                    "nll": np.inf,
                    "init_advec_lat": init["advec_lat"],
                    "init_advec_lon": init["advec_lon"],
                    "seed_raw_lat": np.nan,
                    "seed_raw_lon": np.nan,
                    "fit_s": np.nan,
                    "steps": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    total_s = precompute_s + float(
        np.nansum([candidate.get("fit_s", np.nan) for candidate in candidates])
    )
    del model, pilot_map, pilot_cpu
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return candidates, float(precompute_s), float(total_s)


def make_hourly_grids(
    day_map: dict[str, torch.Tensor],
    regular_grid_coords: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, float, float]:
    lat_key = np.round(regular_grid_coords[:, 0], 6)
    lon_key = np.round(regular_grid_coords[:, 1], 6)
    lats = np.sort(np.unique(lat_key))
    lons = np.sort(np.unique(lon_key))
    lat_to_row = {float(value): idx for idx, value in enumerate(lats)}
    lon_to_col = {float(value): idx for idx, value in enumerate(lons)}
    rows = np.asarray([lat_to_row[float(value)] for value in lat_key], dtype=np.int64)
    cols = np.asarray([lon_to_col[float(value)] for value in lon_key], dtype=np.int64)

    grids: list[np.ndarray] = []
    for key in sorted(day_map):
        values = day_map[key][:, 2].detach().cpu().numpy().astype(np.float64)
        grid = np.full((len(lats), len(lons)), np.nan, dtype=np.float64)
        grid[rows, cols] = values
        if np.isfinite(grid).any():
            grid = grid - float(np.nanmean(grid))
        grids.append(grid)
    lat_step = float(np.nanmedian(np.diff(lats)))
    lon_step = float(np.nanmedian(np.diff(lons)))
    return grids, lats, lons, lat_step, lon_step


def shifted_sum_half_sq(cur: np.ndarray, nxt: np.ndarray, di: int, dj: int) -> tuple[float, int]:
    nlat, nlon = cur.shape
    if abs(di) >= nlat or abs(dj) >= nlon:
        return 0.0, 0
    cur_i = slice(0, nlat - di) if di >= 0 else slice(-di, nlat)
    nxt_i = slice(di, nlat) if di >= 0 else slice(0, nlat + di)
    cur_j = slice(0, nlon - dj) if dj >= 0 else slice(-dj, nlon)
    nxt_j = slice(dj, nlon) if dj >= 0 else slice(0, nlon + dj)
    left = cur[cur_i, cur_j]
    right = nxt[nxt_i, nxt_j]
    mask = np.isfinite(left) & np.isfinite(right)
    n_valid = int(mask.sum())
    if n_valid == 0:
        return 0.0, 0
    diff = left[mask] - right[mask]
    return float(0.5 * np.sum(diff * diff)), n_valid


def nan_gaussian_filter(arr: np.ndarray, sigma: tuple[float, float]) -> np.ndarray:
    if sigma[0] <= 0 and sigma[1] <= 0:
        return arr.copy()
    mask = np.isfinite(arr)
    values = gaussian_filter(np.where(mask, arr, 0.0), sigma=sigma, mode="nearest")
    weights = gaussian_filter(mask.astype(float), sigma=sigma, mode="nearest")
    out = values / np.maximum(weights, 1e-12)
    out[weights <= 1e-12] = np.nan
    return out


def empirical_ridge_seed(
    day_map: dict[str, torch.Tensor],
    regular_grid_coords: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    grids, _, _, lat_step, lon_step = make_hourly_grids(day_map, regular_grid_coords)
    tau = int(args.empirical_tau)
    lat_offsets = np.arange(-args.empirical_max_lat_offset, args.empirical_max_lat_offset + 1)
    lon_offsets = np.arange(-args.empirical_max_lon_offset, args.empirical_max_lon_offset + 1)
    sums = np.zeros((len(lat_offsets), len(lon_offsets)), dtype=np.float64)
    counts = np.zeros_like(sums, dtype=np.int64)
    for hour in range(len(grids) - tau):
        cur, nxt = grids[hour], grids[hour + tau]
        for ii, di in enumerate(lat_offsets):
            for jj, dj in enumerate(lon_offsets):
                value, n_valid = shifted_sum_half_sq(cur, nxt, int(di), int(dj))
                if n_valid:
                    sums[ii, jj] += value
                    counts[ii, jj] += n_valid

    gamma = np.full_like(sums, np.nan)
    valid = counts > 0
    gamma[valid] = sums[valid] / counts[valid]
    filtered = np.where(counts >= args.empirical_min_pair_count, gamma, np.nan)
    sigma = (
        args.empirical_smooth_bandwidth_deg / abs(lat_step),
        args.empirical_smooth_bandwidth_deg / abs(lon_step),
    )
    smoothed = nan_gaussian_filter(filtered, sigma)
    finite = np.isfinite(smoothed)
    if not finite.any():
        raise RuntimeError(
            "Empirical ridge has no finite cells after pair-count filtering; "
            "reduce --empirical-min-pair-count."
        )
    row, col = np.unravel_index(int(np.nanargmin(smoothed)), smoothed.shape)
    gamma_min = float(smoothed[row, col])
    finite_values = np.sort(smoothed[finite].ravel())
    second = float(finite_values[1]) if finite_values.size > 1 else np.nan
    gap = second - gamma_min if np.isfinite(second) else np.nan
    rel_gap = gap / max(abs(gamma_min), 1e-12) if np.isfinite(gap) else np.nan
    tolerance = max(
        args.empirical_near_min_abs_tol,
        args.empirical_near_min_rel_tol * abs(gamma_min),
    )
    near_mask = finite & (smoothed <= gamma_min + tolerance)
    lat_lag = float(lat_offsets[row] * lat_step)
    lon_lag = float(lon_offsets[col] * lon_step)
    elapsed = time.perf_counter() - started
    return {
        "candidate": f"tau{tau}_ridge",
        "status": "ok",
        "nll": np.nan,
        "init_advec_lat": np.nan,
        "init_advec_lon": np.nan,
        "seed_raw_lat": lat_lag / tau,
        "seed_raw_lon": lon_lag / tau,
        "fit_s": float(elapsed),
        "steps": 0,
        "ridge_tau": tau,
        "ridge_lat_lag": lat_lag,
        "ridge_lon_lag": lon_lag,
        "ridge_gamma_min": gamma_min,
        "ridge_gamma_second_min": second,
        "ridge_gamma_gap": gap,
        "ridge_gamma_rel_gap": rel_gap,
        "ridge_near_min_count": int(near_mask.sum()),
        "ridge_near_min_tol": float(tolerance),
        "ridge_is_ambiguous": bool(
            int(near_mask.sum()) >= args.empirical_ambiguous_min_near_cells
        ),
        "ridge_n_pairs": int(counts[row, col]),
        "lat_step": lat_step,
        "lon_step": lon_step,
    }


def stabilize_seed(lat: float, lon: float, max_norm: float) -> tuple[float, float, bool]:
    vector = np.asarray([lat, lon], dtype=np.float64)
    if not np.isfinite(vector).all():
        vector = np.asarray(
            [DEFAULT_INIT_PHYSICAL["advec_lat"], DEFAULT_INIT_PHYSICAL["advec_lon"]],
            dtype=np.float64,
        )
        return float(vector[0]), float(vector[1]), True
    norm = float(np.linalg.norm(vector))
    if max_norm > 0 and norm > max_norm:
        vector *= max_norm / norm
        return float(vector[0]), float(vector[1]), True
    return float(vector[0]), float(vector[1]), False


def direction_degrees(lat: float, lon: float) -> float:
    return float(np.degrees(np.arctan2(lat, lon)))


def angle_difference_degrees(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    if math.hypot(a_lat, a_lon) <= 1e-12 or math.hypot(b_lat, b_lon) <= 1e-12:
        return np.nan
    delta = direction_degrees(a_lat, a_lon) - direction_degrees(b_lat, b_lon)
    return float(abs((delta + 180.0) % 360.0 - 180.0))


def choose_candidate(candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in candidates if row.get("status") == "ok" and np.isfinite(row.get("nll", np.nan))]
    if not valid:
        raise RuntimeError("No finite pilot candidate likelihood")
    return min(valid, key=lambda row: float(row["nll"]))


def fit_final_corridor(
    day_map: dict[str, torch.Tensor],
    regular_grid_coords: np.ndarray,
    seed_lat: float,
    seed_lon: float,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    mapped = {key: value.to(device) for key, value in day_map.items()}
    model = build_directional_model(
        smooth=args.smooth,
        input_map=mapped,
        grid_coords=regular_grid_coords,
        reference_advec_lat=seed_lat,
        reference_advec_lon=seed_lon,
        daily_stride=args.daily_stride,
        target_chunk_size=args.target_chunk_size,
        min_target_points=args.min_target_points,
    )
    started = time.perf_counter()
    stream = contextlib.redirect_stdout(io.StringIO()) if args.suppress_fit_prints else contextlib.nullcontext()
    with stream:
        model.precompute_conditioning_sets()
    precompute_s = time.perf_counter() - started

    init = dict(DEFAULT_INIT_PHYSICAL)
    init["advec_lat"] = float(seed_lat)
    init["advec_lon"] = float(seed_lon)
    fit = fit_model(
        model=model,
        init_physical=init,
        dtype=torch.double,
        device=device,
        outer_steps=args.final_lbfgs_steps,
        lbfgs_eval=args.final_lbfgs_eval,
        lbfgs_history=args.final_lbfgs_history,
        lbfgs_lr=args.lbfgs_lr,
        grad_tol=args.grad_tol,
        suppress_prints=args.suppress_fit_prints,
    )
    summary = model.cluster_summary()
    del model, mapped
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        **fit,
        "precompute_s": float(precompute_s),
        "total_s": float(precompute_s + fit["fit_s"]),
        "cluster_summary": summary,
    }


def default_data_root() -> Path:
    amarel = Path(config.amarel_data_load_path)
    return amarel if amarel.exists() else Path(config.mac_data_load_path)


def default_output_root() -> Path:
    if Path("/home/jl2815").exists():
        return Path(config.amarel_estimates_day_path) / "advection_seed_three_method_corridor432_081626"
    return REPO / "outputs/day/estimates/advection_seed_three_method_corridor432_081626"


def save_state(
    output_root: Path,
    results: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
) -> None:
    result_df = pd.DataFrame(results)
    candidate_df = pd.DataFrame(candidates)
    result_df.to_csv(output_root / "method_results.csv", index=False, float_format="%.8f")
    candidate_df.to_csv(output_root / "seed_candidates.csv", index=False, float_format="%.8f")
    (output_root / "method_results.json").write_text(
        json.dumps(clean_json(list(results)), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if len(result_df) and "status" in result_df:
        ok = result_df.loc[result_df["status"].eq("ok")].copy()
        lines = [
            f"Updated: {datetime.now().isoformat(timespec='seconds')}",
            f"Rows: {len(result_df)}; completed: {len(ok)}",
            "",
        ]
        if len(ok):
            columns = [
                "year", "day_idx", "method", "final_nll", "seed_total_s",
                "final_total_s", "end_to_end_s", "seed_lat", "seed_lon",
                "est_advec_lat", "est_advec_lon",
            ]
            lines.append(ok[[c for c in columns if c in ok]].tail(18).to_string(index=False))
            lines.extend(["", "Median performance by method:"])
            metrics = ["final_nll", "seed_total_s", "final_total_s", "end_to_end_s"]
            lines.append(ok.groupby("method")[[c for c in metrics if c in ok]].median().to_string())
        (output_root / "running_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", default=["2024"])
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--days", default="0", help="0-based day indices: '0', '0,3', or 'all'.")
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--space", default="1,1")
    parser.add_argument("--lat-range", default="-3,2")
    parser.add_argument("--lon-range", default="121,131")
    parser.add_argument("--smooth", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cuda-fallback", choices=["cpu", "error"], default="cpu")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--keep-exact-loc", dest="keep_exact_loc", action="store_true", default=True)
    parser.add_argument("--no-keep-exact-loc", dest="keep_exact_loc", action="store_false")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--suppress-fit-prints", action="store_true")

    parser.add_argument("--pilot400-points", type=int, default=400)
    parser.add_argument("--quadrant300-points", type=int, default=300)
    parser.add_argument("--pilot-neighbors", type=int, default=30)
    parser.add_argument("--pilot-nheads", type=int, default=1)
    parser.add_argument("--pilot-limit-a", type=int, default=20)
    parser.add_argument("--pilot-limit-b", type=int, default=20)
    parser.add_argument("--pilot-limit-c", type=int, default=20)
    parser.add_argument("--pilot400-lbfgs-steps", type=int, default=2)
    parser.add_argument("--quadrant-lbfgs-steps", type=int, default=1)
    parser.add_argument("--pilot-lbfgs-eval", type=int, default=20)
    parser.add_argument("--pilot-lbfgs-history", type=int, default=10)

    parser.add_argument("--empirical-tau", type=int, default=1)
    parser.add_argument("--empirical-max-lat-offset", type=int, default=20)
    parser.add_argument("--empirical-max-lon-offset", type=int, default=20)
    parser.add_argument("--empirical-smooth-bandwidth-deg", type=float, default=0.063)
    parser.add_argument("--empirical-min-pair-count", type=int, default=1000)
    parser.add_argument("--empirical-near-min-rel-tol", type=float, default=0.01)
    parser.add_argument("--empirical-near-min-abs-tol", type=float, default=0.05)
    parser.add_argument("--empirical-ambiguous-min-near-cells", type=int, default=5)

    parser.add_argument("--max-seed-norm", type=float, default=0.75)
    parser.add_argument("--daily-stride", type=int, default=2)
    parser.add_argument("--target-chunk-size", type=int, default=128)
    parser.add_argument("--min-target-points", type=int, default=1)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--final-lbfgs-steps", type=int, default=5)
    parser.add_argument("--final-lbfgs-eval", type=int, default=20)
    parser.add_argument("--final-lbfgs-history", type=int, default=10)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    years = [str(year) for year in args.years]
    days = parse_days(args.days)
    methods = parse_methods(args.methods)
    resolution = parse_pair(args.space, int)
    lat_range = parse_pair(args.lat_range, float)
    lon_range = parse_pair(args.lon_range, float)
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.cuda_fallback == "error":
        raise RuntimeError("CUDA is unavailable and --cuda-fallback=error was requested")
    else:
        device = torch.device("cpu")

    data_root = args.data_root or default_data_root()
    output_root = args.output_root or default_output_root()
    output_root.mkdir(parents=True, exist_ok=True)
    config_payload = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "data_root": data_root,
        "output_root": output_root,
        "years": years,
        "month": args.month,
        "days": days,
        "methods": methods,
        "device": str(device),
        "dtype": str(torch.double),
        "model": {
            "spec_name": DIRECTIONAL_SPEC_NAME,
            "block_shape": BLOCK_SHAPE,
            "lag_counts": LAG_COUNTS,
        },
        "quadrant_starts": QUADRANT_STARTS,
        "arguments": vars(args),
    }
    (output_root / "run_config.json").write_text(
        json.dumps(clean_json(config_payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    results: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    result_path = output_root / "method_results.csv"
    candidate_path = output_root / "seed_candidates.csv"
    if args.skip_existing and result_path.exists():
        results = pd.read_csv(result_path).to_dict(orient="records")
    if args.skip_existing and candidate_path.exists():
        candidate_rows = pd.read_csv(candidate_path).to_dict(orient="records")
    done = {
        (str(row["year"]), int(row["day_idx"]), str(row["method"]))
        for row in results
        if str(row.get("status", "")) == "ok"
    }

    print("device:", device)
    print("data_root:", data_root)
    print("output_root:", output_root)
    print("methods:", methods)
    loader = load_data_dynamic_processed(str(data_root))

    for year in years:
        print(f"\n=== Loading {year}-{args.month:02d} ===")
        df_map, _, _, monthly_mean = loader.load_maxmin_ordered_data_bymonthyear(
            lat_lon_resolution=resolution,
            mm_cond_number=1,
            years_=[year],
            months_=[args.month],
            lat_range=lat_range,
            lon_range=lon_range,
            is_whittle=True,
        )
        keys = sorted(df_map)
        if not keys:
            raise RuntimeError(f"No data loaded for {year}-{args.month:02d}")
        regular_grid_coords = df_map[keys[0]][["Latitude", "Longitude"]].to_numpy(dtype=np.float64)
        requested_pilot_sizes = []
        if "pilot400" in methods:
            requested_pilot_sizes.append(args.pilot400_points)
        if "quadrant300" in methods:
            requested_pilot_sizes.append(args.quadrant300_points)
        max_requested = max(requested_pilot_sizes, default=0)
        if max_requested > len(regular_grid_coords):
            raise ValueError(f"Requested {max_requested} pilot points but grid has {len(regular_grid_coords)}")
        if max_requested:
            # Match the loader's established max-min convention: order in
            # (lon, lat), while covariance/grid APIs use (lat, lon).
            maxmin_input = regular_grid_coords[:, [1, 0]].copy()
            maxmin_order = orderings.maxmin_cpp(maxmin_input)
            ordered_grid_coords = regular_grid_coords[maxmin_order]
            nns_map = orderings.find_nns_l2(
                ordered_grid_coords[:max_requested],
                max_nn=args.pilot_neighbors,
            )
        else:
            maxmin_order = np.arange(len(regular_grid_coords), dtype=np.int64)
            ordered_grid_coords = regular_grid_coords
            nns_map = np.empty((len(regular_grid_coords), 0), dtype=np.int64)
        args._maxmin_order = maxmin_order

        for day_idx in days:
            pending = [method for method in methods if (year, day_idx, method) not in done]
            if not pending:
                print(f"Skipping completed {year}-{args.month:02d}-{day_idx + 1:02d}")
                continue
            day_label = f"{year}-{args.month:02d}-{day_idx + 1:02d}"
            hour_indices = [day_idx * 8, (day_idx + 1) * 8]
            selected_keys = keys[hour_indices[0] : hour_indices[1]]
            print(f"\n--- {day_label}: {pending} ---")
            assert_grid_order_consistent(df_map, selected_keys, regular_grid_coords)
            day_map, _ = loader.load_working_data(
                df_map,
                monthly_mean,
                hour_indices,
                ord_mm=None,
                dtype=torch.double,
                keep_ori=args.keep_exact_loc,
            )
            if len(day_map) != 8:
                raise RuntimeError(f"Expected 8 hourly maps for {day_label}, got {len(day_map)}")
            n_valid, n_total = count_valid(day_map)

            for method in pending:
                method_started = time.perf_counter()
                base = {
                    "year": year,
                    "month": int(args.month),
                    "day_idx": int(day_idx),
                    "day": day_label,
                    "method": method,
                    "status": "error",
                    "error": "",
                    "n_time_slots": len(day_map),
                    "n_rows_total": n_total,
                    "n_valid_o3": n_valid,
                    "valid_rate": n_valid / n_total if n_total else np.nan,
                    "monthly_mean": monthly_mean,
                    "device": str(device),
                    "spec_name": DIRECTIONAL_SPEC_NAME,
                    "block_shape": f"{BLOCK_SHAPE[0]}x{BLOCK_SHAPE[1]}",
                    "lag_pattern": "/".join(str(x) for x in LAG_COUNTS),
                }
                try:
                    if method == "pilot400":
                        init = dict(DEFAULT_INIT_PHYSICAL)
                        args._pilot_outer_steps = args.pilot400_lbfgs_steps
                        candidates, seed_precompute_s, seed_total_s = run_pilot_candidates(
                            day_map=day_map,
                            ordered_grid_coords=ordered_grid_coords,
                            nns_map=nns_map,
                            n_points=args.pilot400_points,
                            starts=[("default", init)],
                            args=args,
                            device=device,
                        )
                        chosen = choose_candidate(candidates)
                        seed_points = args.pilot400_points
                    elif method == "quadrant300":
                        starts = []
                        for idx, (lat, lon) in enumerate(QUADRANT_STARTS, start=1):
                            init = dict(DEFAULT_INIT_PHYSICAL)
                            init["advec_lat"] = lat
                            init["advec_lon"] = lon
                            starts.append((f"q{idx}_lat{lat:+.2f}_lon{lon:+.2f}", init))
                        args._pilot_outer_steps = args.quadrant_lbfgs_steps
                        candidates, seed_precompute_s, seed_total_s = run_pilot_candidates(
                            day_map=day_map,
                            ordered_grid_coords=ordered_grid_coords,
                            nns_map=nns_map,
                            n_points=args.quadrant300_points,
                            starts=starts,
                            args=args,
                            device=device,
                        )
                        chosen = choose_candidate(candidates)
                        seed_points = args.quadrant300_points
                    else:
                        chosen = empirical_ridge_seed(day_map, regular_grid_coords, args)
                        candidates = [chosen]
                        seed_precompute_s = 0.0
                        seed_total_s = float(chosen["fit_s"])
                        seed_points = n_total // len(day_map)

                    seed_lat, seed_lon, clipped = stabilize_seed(
                        float(chosen["seed_raw_lat"]),
                        float(chosen["seed_raw_lon"]),
                        args.max_seed_norm,
                    )
                    for candidate in candidates:
                        candidate_rows.append(
                            {
                                "year": year,
                                "month": int(args.month),
                                "day_idx": int(day_idx),
                                "day": day_label,
                                "method": method,
                                "selected": candidate["candidate"] == chosen["candidate"],
                                "n_points_per_hour": seed_points,
                                **candidate,
                            }
                        )

                    print(
                        f"{method}: raw seed=({chosen['seed_raw_lat']:.5f}, "
                        f"{chosen['seed_raw_lon']:.5f}); used=({seed_lat:.5f}, {seed_lon:.5f})"
                    )
                    final = fit_final_corridor(
                        day_map=day_map,
                        regular_grid_coords=regular_grid_coords,
                        seed_lat=seed_lat,
                        seed_lon=seed_lon,
                        args=args,
                        device=device,
                    )
                    est = final["est"]
                    row = {
                        **base,
                        "status": "ok",
                        "seed_candidate": chosen["candidate"],
                        "seed_points_per_hour": seed_points,
                        "seed_raw_lat": float(chosen["seed_raw_lat"]),
                        "seed_raw_lon": float(chosen["seed_raw_lon"]),
                        "seed_lat": seed_lat,
                        "seed_lon": seed_lon,
                        "seed_norm": float(np.hypot(seed_lat, seed_lon)),
                        "seed_direction_deg": direction_degrees(seed_lat, seed_lon),
                        "seed_was_clipped_or_fallback": bool(clipped),
                        "seed_candidate_nll": chosen.get("nll", np.nan),
                        "seed_precompute_s": seed_precompute_s,
                        "seed_total_s": seed_total_s,
                        "final_nll": final["nll"],
                        "final_optimizer_reported_loss": final["optimizer_reported_loss"],
                        "final_steps": final["steps"],
                        "final_precompute_s": final["precompute_s"],
                        "final_fit_s": final["fit_s"],
                        "final_total_s": final["total_s"],
                        "end_to_end_s": time.perf_counter() - method_started,
                        **{f"est_{key}": float(est[key]) for key in P_LABELS},
                        "est_advec_norm": float(np.hypot(est["advec_lat"], est["advec_lon"])),
                        "est_direction_deg": direction_degrees(est["advec_lat"], est["advec_lon"]),
                        "seed_to_est_angle_deg": angle_difference_degrees(
                            seed_lat,
                            seed_lon,
                            est["advec_lat"],
                            est["advec_lon"],
                        ),
                        **{key: value for key, value in chosen.items() if key.startswith("ridge_")},
                        **{f"cluster_{key}": value for key, value in final["cluster_summary"].items()},
                        "model_spec": json.dumps(directional_model_spec(seed_lat, seed_lon), sort_keys=True),
                    }
                    print(
                        f"{method}: final NLL={row['final_nll']:.8f}, "
                        f"advec=({row['est_advec_lat']:.5f}, {row['est_advec_lon']:.5f}), "
                        f"end-to-end={row['end_to_end_s']:.1f}s"
                    )
                except Exception as exc:
                    row = {
                        **base,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(limit=10),
                        "end_to_end_s": time.perf_counter() - method_started,
                    }
                    print(f"ERROR {day_label} {method}: {row['error']}")
                    traceback.print_exc()
                results.append(clean_json(row))
                save_state(output_root, results, candidate_rows)

            del day_map
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("\nDone")
    print("results:", output_root / "method_results.csv")
    print("candidates:", output_root / "seed_candidates.csv")
    print("summary:", output_root / "running_summary.txt")


if __name__ == "__main__":
    main()
