#!/usr/bin/env python3
"""
Fit five cluster Vecchia temporal-neighbor strategies on pre-generated July ST
circulant simulation assets.

Created 2026-05-22.

This script does not simulate new data.  It reads the reusable simulation
pickles created by:

  simulate_data/generate_july_st_circulant_real_locations_2022_2025.py

Goal:
  Re-test cluster Vecchia geometry because the pointwise hybrid idea may not
  transfer cleanly to 3x3 or 4x4 target clusters.  In cluster form, using both
  target-center and offset-center branches can become redundant and expensive.
  Therefore this sweep intentionally tests only five interpretable strategies:

    1. center_full
       t, t-1, and t-2 all use target-center clusters.  Each time layer keeps
       the full block budget.

    2. center_tapered
       Same centers as center_full, but t-1 keeps about 80% of the t budget and
       t-2 keeps about 50%.  This isolates the "fewer older-lag neighbors"
       effect observed in pointwise Vecchia.

    3. offset_full
       t uses target-center previous same-time clusters.  t-1 uses the cluster
       neighborhood centered at target_lon + 0.063*2, and t-2 uses the cluster
       neighborhood centered at target_lon + 0.063*4.  The original target
       center is not forced into lagged conditioning.

    4. offset_tapered
       Same offset centers as offset_full, with the 80% / 50% lag taper.

    5. offset_tapered_force_center
       Same as offset_tapered, but t-1 and t-2 force the original target-center
       cluster into the conditioning set.  If the target-center cluster already
       appears in the offset-centered candidates, one additional offset-centered
       neighbor is taken; if it does not overlap, the forced target center is
       simply added.  This keeps the comparison from losing one lagged cluster
       just because the offset snapped back to the target neighborhood.

Output policy:
  There is no per-fit fit.csv.  Every completed fit appends exactly one compact
  row to all_fits_summary.csv.  The simulation truth and run configuration are
  written once to JSON files.  Running model and parameter summaries are
  refreshed periodically and include p90-p10, RMSRE, and median absolute error.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import pickle
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


AMAREL_SRC = "/home/jl2815/tco"
LOCAL_SRC = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
SRC = AMAREL_SRC if os.path.exists(AMAREL_SRC) else LOCAL_SRC
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from GEMS_TCO.vecchia_cluster import STRATEGIES, StrategyClusterVecchiaFit


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
ROUND_DECIMALS = 4
DELTA_LAT_BASE = 0.044
DELTA_LON_BASE = 0.063
T_STEPS = 8

P_LABELS = ["sigmasq", "range_lat", "range_lon", "range_time", "advec_lat", "advec_lon", "nugget"]
SPATIAL_KEYS = ["sigmasq", "range_lat", "range_lon"]
ADVECTION_KEYS = ["advec_lat", "advec_lon"]

DEFAULT_STRATEGIES = [
    "center_full",
    "center_tapered",
    "offset_full",
    "offset_tapered",
    "offset_tapered_force_center",
]

ROW_COLUMNS = [
    "fit_id",
    "sim_id",
    "seed",
    "asset_year",
    "asset_day_idx",
    "asset_first_key",
    "data_kind",
    "strategy",
    "block_shape",
    "model_name",
    "smooth",
    "lag0_blocks",
    "lag1_blocks",
    "lag2_blocks",
    "lag1_max_blocks",
    "lag2_max_blocks",
    "temporal_basis",
    "force_target_center",
    "lag1_lon_offset",
    "lag2_lon_offset",
    "n_cond_blocks_nominal",
    "n_cond_points_nominal",
    "n_valid",
    "n_grid",
    "valid_by_t",
    "loss",
    "converged",
    "fit_steps",
    "precompute_s",
    "fit_s",
    "total_s",
    "n_clusters",
    "max_points_per_cluster",
    "n_target_blocks",
    "n_target_points",
    "n_batches",
    "mean_m_by_template",
    "median_m_by_template",
    "max_m_by_template",
    "median_target_size_by_batch",
    "max_target_size_by_batch",
    "overall_rmsre",
    "spatial_rmsre",
    "advec_rmsre",
    "median_abs_error",
    "range_time_re",
    "nugget_re",
    *[f"est_{k}" for k in P_LABELS],
    *[f"abs_error_{k}" for k in P_LABELS],
    *[f"{k}_re" for k in P_LABELS],
    "error",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_range(s: str) -> list[float]:
    vals = [float(x.strip()) for x in str(s).split(",") if x.strip()]
    if len(vals) != 2:
        raise argparse.ArgumentTypeError("range must look like -3,2")
    return [min(vals), max(vals)]


def parse_years(s: str) -> list[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_strategies(s: str) -> list[str]:
    vals = [x.strip() for x in str(s).split(",") if x.strip()]
    bad = [v for v in vals if v not in STRATEGIES]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown strategies: {bad}; allowed={sorted(STRATEGIES)}")
    return vals


def parse_block_shape(s: str) -> tuple[int, int]:
    text = str(s).lower().replace("x", ",")
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != 2:
        raise argparse.ArgumentTypeError("block shape must look like 3x3 or 3,3")
    return vals[0], vals[1]


def parse_block_shapes(s: str) -> list[tuple[int, int]]:
    return [parse_block_shape(x) for x in str(s).split(",") if x.strip()]


def parse_day_idxs(s: str) -> list[int] | str:
    text = str(s).strip().lower()
    if text in {"all", "*"}:
        return "all"
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_gems_hour_key(key: str) -> pd.Timestamp | None:
    pat = r"^y(?P<yy>\d{2})m(?P<mm>\d{2})day(?P<dd>\d{2})_hm(?P<hh>\d{2}):(?P<minute>\d{2})$"
    match = re.match(pat, str(key))
    if not match:
        return None
    parts = {name: int(value) for name, value in match.groupdict().items()}
    return pd.Timestamp(
        year=2000 + parts["yy"],
        month=parts["mm"],
        day=parts["dd"],
        hour=parts["hh"],
        minute=parts["minute"],
        tz="UTC",
    )


def ordered_asset_keys(obj: dict[Any, Any]) -> list[Any]:
    parsed = []
    for key in obj:
        ts = parse_gems_hour_key(str(key))
        if ts is not None:
            parsed.append((ts, key))
    return [key for _, key in sorted(parsed)] if parsed else sorted(obj)


def asset_paths(sim_data_root: Path, year: int, data_kind: str) -> tuple[Path, Path]:
    year_dir = sim_data_root / f"{year}_july_st_circulant"
    prefix = f"sim_july{year}_st_circulant"
    return year_dir / f"{prefix}_{data_kind}.pkl", year_dir / f"{prefix}_truth.json"


def load_truth(truth_path: Path) -> dict[str, float]:
    if not truth_path.exists():
        raise FileNotFoundError(f"Missing simulation truth file: {truth_path}")
    raw = json.loads(truth_path.read_text(encoding="utf-8"))
    truth = {k: float(raw[k]) for k in P_LABELS}
    truth["smooth"] = float(raw.get("smooth", 0.5))
    return truth


def load_asset_map(path: Path) -> dict[Any, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(f"Missing pre-generated simulation pickle: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"{path} must contain a dict of DataFrames")
    return obj


def grid_filtered_frame(df: pd.DataFrame, lat_range: list[float], lon_range: list[float]) -> pd.DataFrame:
    required = {"Latitude", "Longitude", "ColumnAmountO3"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing required columns: {sorted(missing)}")
    out = df.copy()
    out["_grid_lat"] = pd.to_numeric(out["Latitude"], errors="coerce")
    out["_grid_lon"] = pd.to_numeric(out["Longitude"], errors="coerce")
    mask = (
        np.isfinite(out["_grid_lat"])
        & np.isfinite(out["_grid_lon"])
        & (out["_grid_lat"] >= min(lat_range))
        & (out["_grid_lat"] <= max(lat_range))
        & (out["_grid_lon"] >= min(lon_range))
        & (out["_grid_lon"] <= max(lon_range))
    )
    out = out.loc[mask].copy()
    out["_grid_lat_key"] = np.round(out["_grid_lat"].to_numpy(dtype=float), ROUND_DECIMALS)
    out["_grid_lon_key"] = np.round(out["_grid_lon"].to_numpy(dtype=float), ROUND_DECIMALS)
    return out


def build_master_grid(first_df: pd.DataFrame, lat_range: list[float], lon_range: list[float]) -> pd.DataFrame:
    first = grid_filtered_frame(first_df, lat_range, lon_range)
    master = (
        first[["_grid_lat_key", "_grid_lon_key", "_grid_lat", "_grid_lon"]]
        .drop_duplicates(["_grid_lat_key", "_grid_lon_key"])
        .sort_values(["_grid_lat", "_grid_lon"])
        .reset_index(drop=True)
    )
    if master.empty:
        raise ValueError("No grid rows left after lat/lon filtering.")
    return master


def build_regular_master_grid(lat_range: list[float], lon_range: list[float]) -> pd.DataFrame:
    lats = np.arange(min(lat_range), max(lat_range) + 1e-4, DELTA_LAT_BASE, dtype=float)
    lons = np.arange(min(lon_range), max(lon_range) + 1e-4, DELTA_LON_BASE, dtype=float)
    lats = np.round(lats, ROUND_DECIMALS)
    lons = np.round(lons, ROUND_DECIMALS)
    g_lat, g_lon = np.meshgrid(lats, lons, indexing="ij")
    master = pd.DataFrame({"_grid_lat": g_lat.ravel(), "_grid_lon": g_lon.ravel()})
    master["_grid_lat_key"] = np.round(master["_grid_lat"].to_numpy(dtype=float), ROUND_DECIMALS)
    master["_grid_lon_key"] = np.round(master["_grid_lon"].to_numpy(dtype=float), ROUND_DECIMALS)
    return master[["_grid_lat_key", "_grid_lon_key", "_grid_lat", "_grid_lon"]]


def align_to_master_grid(
    df: pd.DataFrame,
    master: pd.DataFrame,
    lat_range: list[float],
    lon_range: list[float],
) -> pd.DataFrame:
    part = grid_filtered_frame(df, lat_range, lon_range)
    finite_y = np.isfinite(pd.to_numeric(part["ColumnAmountO3"], errors="coerce").to_numpy(dtype=float))
    part = part.assign(_finite_y=finite_y).sort_values("_finite_y", ascending=False)
    part = part.drop_duplicates(["_grid_lat_key", "_grid_lon_key"], keep="first")
    part = part.set_index(["_grid_lat_key", "_grid_lon_key"])
    idx = pd.MultiIndex.from_frame(master[["_grid_lat_key", "_grid_lon_key"]])
    aligned = part.reindex(idx).reset_index()
    aligned["_grid_lat"] = master["_grid_lat"].to_numpy(dtype=float)
    aligned["_grid_lon"] = master["_grid_lon"].to_numpy(dtype=float)
    return aligned


def tensor_from_aligned_frame(
    aligned: pd.DataFrame,
    t_idx: int,
    monthly_mean: float,
    center_response: bool,
) -> torch.Tensor:
    y = pd.to_numeric(aligned["ColumnAmountO3"], errors="coerce").to_numpy(dtype=float)
    if center_response:
        y = y - float(monthly_mean)

    if {"Source_Latitude", "Source_Longitude"}.issubset(aligned.columns):
        lat = pd.to_numeric(aligned["Source_Latitude"], errors="coerce").to_numpy(dtype=float)
        lon = pd.to_numeric(aligned["Source_Longitude"], errors="coerce").to_numpy(dtype=float)
    else:
        lat = aligned["_grid_lat"].to_numpy(dtype=float)
        lon = aligned["_grid_lon"].to_numpy(dtype=float)

    valid = np.isfinite(y) & np.isfinite(lat) & np.isfinite(lon)
    y = np.where(valid, y, np.nan)
    lat = np.where(valid, lat, np.nan)
    lon = np.where(valid, lon, np.nan)

    base = np.full((len(aligned), 4), np.nan, dtype=np.float64)
    base[:, 0] = lat
    base[:, 1] = lon
    base[:, 2] = y
    base[:, 3] = float(t_idx)
    base_tensor = torch.from_numpy(base).to(dtype=DTYPE)

    dummies = F.one_hot(torch.tensor([t_idx]), num_classes=T_STEPS).repeat(len(base_tensor), 1)
    dummies = dummies[:, 1:].to(DTYPE)
    return torch.cat([base_tensor, dummies], dim=1).contiguous()


def build_day_asset(
    sim_map: dict[Any, pd.DataFrame],
    year: int,
    day_idx: int,
    lat_range: list[float],
    lon_range: list[float],
    center_response: bool,
    master: pd.DataFrame,
    monthly_mean: float,
) -> dict[str, Any] | None:
    keys = ordered_asset_keys(sim_map)
    day_keys = keys[day_idx * T_STEPS : (day_idx + 1) * T_STEPS]
    if len(day_keys) < T_STEPS:
        return None

    source_map = {}
    valid_by_t = []
    for t_idx, key in enumerate(day_keys):
        aligned = align_to_master_grid(sim_map[key], master, lat_range, lon_range)
        tensor = tensor_from_aligned_frame(aligned, t_idx, monthly_mean, center_response)
        source_map[str(key)] = tensor
        valid_by_t.append(int(torch.isfinite(tensor[:, 2]).sum().item()))

    grid_coords_np = master[["_grid_lat", "_grid_lon"]].to_numpy(dtype=np.float64)
    return {
        "year": int(year),
        "day_idx": int(day_idx),
        "day_keys": [str(k) for k in day_keys],
        "source_map": source_map,
        "grid_coords_np": grid_coords_np,
        "n_grid": int(grid_coords_np.shape[0]),
        "monthly_mean": float(monthly_mean),
        "valid_by_t": valid_by_t,
    }


def build_asset_bank(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, float]]:
    years = parse_years(args.years)
    requested_day_idxs = parse_day_idxs(args.day_idxs)
    bank: list[dict[str, Any]] = []
    truth_ref: dict[str, float] | None = None

    for year in years:
        pickle_path, truth_path = asset_paths(args.sim_data_root, year, args.data_kind)
        truth = load_truth(truth_path)
        if truth_ref is None:
            truth_ref = truth
        else:
            for key in P_LABELS + ["smooth"]:
                if not np.isclose(truth_ref[key], truth[key]):
                    raise ValueError(f"Truth mismatch for {key}: {truth_ref[key]} vs {truth[key]} in {truth_path}")

        sim_map = load_asset_map(pickle_path)
        keys = ordered_asset_keys(sim_map)
        max_day_idx = len(keys) // T_STEPS - 1
        try:
            master = build_master_grid(sim_map[keys[0]], args.lat_range, args.lon_range)
            master_source = "generated_pickle_lat_lon"
        except Exception as exc:
            print(
                f"WARNING: could not build grid from generated pickle Latitude/Longitude ({exc!r}); "
                "falling back to native regular grid.",
                flush=True,
            )
            master = build_regular_master_grid(args.lat_range, args.lon_range)
            master_source = "native_regular_grid_fallback"

        month_vals = []
        for key in keys:
            aligned = align_to_master_grid(sim_map[key], master, args.lat_range, args.lon_range)
            vals = pd.to_numeric(aligned["ColumnAmountO3"], errors="coerce").to_numpy(dtype=float)
            month_vals.append(vals)
        monthly_mean = float(np.nanmean(np.concatenate(month_vals)))

        if requested_day_idxs == "all":
            day_idxs = list(range(max_day_idx + 1))
        else:
            day_idxs = [d for d in requested_day_idxs if d <= max_day_idx]

        print(
            f"Loaded {pickle_path} with {len(keys)} hours; usable day blocks={max_day_idx + 1}; "
            f"grid_source={master_source}, n_grid={len(master):,}",
            flush=True,
        )
        for day_idx in day_idxs:
            asset = build_day_asset(
                sim_map=sim_map,
                year=year,
                day_idx=day_idx,
                lat_range=args.lat_range,
                lon_range=args.lon_range,
                center_response=args.center_response,
                master=master,
                monthly_mean=monthly_mean,
            )
            if asset is None:
                continue
            if sum(asset["valid_by_t"]) == 0:
                print(f"WARNING: skipping year={year} day_idx={day_idx}; no valid observations.", flush=True)
                continue
            bank.append(asset)
            miss = [round(100 * (1 - v / asset["n_grid"]), 2) for v in asset["valid_by_t"]]
            print(
                f"Prepared asset year={year} day_idx={day_idx} "
                f"valid={asset['valid_by_t']} missing%={miss}",
                flush=True,
            )
            if args.max_asset_days > 0 and len(bank) >= args.max_asset_days:
                break
        if args.max_asset_days > 0 and len(bank) >= args.max_asset_days:
            break

    if not bank:
        raise RuntimeError("No pre-generated day assets were loaded.")
    assert truth_ref is not None
    return bank, truth_ref


def true_to_log_params(d: dict[str, float]) -> list[float]:
    phi2 = 1.0 / d["range_lon"]
    phi1 = d["sigmasq"] * phi2
    phi3 = (d["range_lon"] / d["range_lat"]) ** 2
    phi4 = (d["range_lon"] / d["range_time"]) ** 2
    return [
        np.log(phi1),
        np.log(phi2),
        np.log(phi3),
        np.log(phi4),
        d["advec_lat"],
        d["advec_lon"],
        np.log(d["nugget"]),
    ]


def make_random_init(rng: np.random.Generator, true_log: list[float], init_noise: float) -> list[float]:
    noisy = list(true_log)
    for idx in [0, 1, 2, 3, 6]:
        noisy[idx] = true_log[idx] + rng.uniform(-init_noise, init_noise)
    for idx in [4, 5]:
        scale = max(abs(true_log[idx]), 0.05)
        noisy[idx] = true_log[idx] + rng.uniform(-2 * scale, 2 * scale)
    return noisy


def backmap_params(out_params) -> dict[str, float]:
    p = [float(x.item() if isinstance(x, torch.Tensor) else x) for x in out_params[:7]]
    phi2 = np.exp(p[1])
    phi3 = np.exp(p[2])
    phi4 = np.exp(p[3])
    range_lon = 1.0 / phi2
    return {
        "sigmasq": np.exp(p[0]) / phi2,
        "range_lat": range_lon / phi3**0.5,
        "range_lon": range_lon,
        "range_time": range_lon / phi4**0.5,
        "advec_lat": p[4],
        "advec_lon": p[5],
        "nugget": np.exp(p[6]),
    }


def rmsre_for_keys(est: dict[str, float], truth: dict[str, float], keys: list[str], zero_thresh: float = 0.01) -> float:
    vals = []
    for key in keys:
        tv = truth[key]
        if abs(tv) < zero_thresh:
            continue
        vals.append(((est[key] - tv) / abs(tv)) ** 2)
    return float(np.sqrt(np.mean(vals))) if vals else float("nan")


def calculate_metrics(out_params, truth: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    est = backmap_params(out_params)
    abs_errors = {key: abs(est[key] - truth[key]) for key in P_LABELS}
    metrics = {
        "overall_rmsre": rmsre_for_keys(est, truth, P_LABELS),
        "spatial_rmsre": rmsre_for_keys(est, truth, SPATIAL_KEYS),
        "advec_rmsre": rmsre_for_keys(est, truth, ADVECTION_KEYS),
        "median_abs_error": float(np.median([abs_errors[k] for k in P_LABELS])),
        "range_time_re": abs(est["range_time"] - truth["range_time"]) / abs(truth["range_time"]),
        "nugget_re": abs(est["nugget"] - truth["nugget"]) / abs(truth["nugget"]),
    }
    for key in P_LABELS:
        denom = abs(truth[key]) if abs(truth[key]) >= 0.01 else 1.0
        metrics[f"abs_error_{key}"] = abs_errors[key]
        metrics[f"{key}_re"] = abs_errors[key] / denom
    return metrics, est


def template_diagnostics(model) -> dict[str, float]:
    cluster_batches = getattr(model, "_cluster_batches", None)
    if not cluster_batches:
        return {
            "n_batches": 0,
            "mean_m_by_template": 0.0,
            "median_m_by_template": 0.0,
            "max_m_by_template": 0,
            "median_target_size_by_batch": 0.0,
            "max_target_size_by_batch": 0,
        }
    batch_sizes = np.asarray([int(b.X.shape[0]) for b in cluster_batches], dtype=np.int64)
    m_sizes = np.asarray([int(b.max_cond_points) for b in cluster_batches], dtype=np.int64)
    target_sizes = np.asarray([int(b.target_size) for b in cluster_batches], dtype=np.int64)
    return {
        "n_batches": int(len(cluster_batches)),
        "largest_template_n": int(batch_sizes.max()),
        "median_template_n": float(np.median(batch_sizes)),
        "mean_template_n": float(batch_sizes.mean()),
        "mean_m_by_template": float(m_sizes.mean()),
        "median_m_by_template": float(np.median(m_sizes)),
        "max_m_by_template": int(m_sizes.max()),
        "median_target_size_by_batch": float(np.median(target_sizes)),
        "max_target_size_by_batch": int(target_sizes.max()),
    }


def new_params(initial_vals: list[float]) -> list[torch.Tensor]:
    return [torch.tensor([v], device=DEVICE, dtype=DTYPE, requires_grad=True) for v in initial_vals]


def fit_strategy(
    source_map: dict[str, torch.Tensor],
    grid_coords_np: np.ndarray,
    initial_vals: list[float],
    spec: dict[str, Any],
    args: argparse.Namespace,
    truth: dict[str, float],
    smooth: float,
) -> dict[str, Any]:
    params = new_params(initial_vals)
    model = StrategyClusterVecchiaFit(
        smooth=smooth,
        input_map=source_map,
        grid_coords=grid_coords_np,
        block_shape=spec["block_shape"],
        strategy=spec["strategy"],
        lag0_block_count=args.lag0_block_count,
        lag1_keep_fraction=args.lag1_keep_fraction,
        lag2_keep_fraction=args.lag2_keep_fraction,
        daily_stride=args.daily_stride,
        lag1_lon_offset=args.lag1_lon_offset,
        lag2_lon_offset=args.lag2_lon_offset,
        target_chunk_size=args.target_chunk_size,
        min_target_points=args.min_target_points,
    )
    t0 = time.time()
    model.precompute_conditioning_sets()
    pre_s = time.time() - t0
    diag = template_diagnostics(model)
    cluster_diag = model.cluster_summary()

    max_expected = int(spec["block_shape"][0]) * int(spec["block_shape"][1])
    if int(model.max_points_per_cluster) > max_expected:
        raise RuntimeError(
            f"Invalid cluster geometry: max_points_per_cluster={model.max_points_per_cluster}, "
            f"block_shape={spec['block_shape']}"
        )

    opt = model.set_optimizer(
        params,
        lr=1.0,
        max_iter=args.lbfgs_eval,
        max_eval=args.lbfgs_eval,
        history_size=args.lbfgs_hist,
    )
    t1 = time.time()
    out, fit_iter = model.fit_vecc_lbfgs(params, opt, max_steps=args.lbfgs_steps, grad_tol=args.grad_tol)
    fit_s = time.time() - t1

    metrics, est = calculate_metrics(out, truth)
    loss = float(out[-1])
    fit_steps = int(fit_iter) + 1
    n_cond_blocks = (
        int(cluster_diag["lag0_block_count"])
        + int(cluster_diag["lag1_max_blocks"])
        + int(cluster_diag["lag2_max_blocks"])
    )
    row = {
        "strategy": spec["strategy"],
        "block_shape": f"{spec['block_shape'][0]}x{spec['block_shape'][1]}",
        "model_name": (
            f"{spec['strategy']}_block{spec['block_shape'][0]}x{spec['block_shape'][1]}_"
            f"L{cluster_diag['lag0_block_count']}_{cluster_diag['lag1_max_blocks']}_{cluster_diag['lag2_max_blocks']}"
        ),
        "smooth": smooth,
        "lag0_blocks": int(cluster_diag["lag0_block_count"]),
        "lag1_blocks": int(cluster_diag["lag1_block_count"]),
        "lag2_blocks": int(cluster_diag["lag2_block_count"]),
        "lag1_max_blocks": int(cluster_diag["lag1_max_blocks"]),
        "lag2_max_blocks": int(cluster_diag["lag2_max_blocks"]),
        "temporal_basis": cluster_diag["temporal_basis"],
        "force_target_center": int(cluster_diag["force_target_center"]),
        "lag1_lon_offset": float(cluster_diag["lag1_lon_offset"]),
        "lag2_lon_offset": float(cluster_diag["lag2_lon_offset"]),
        "n_cond_blocks_nominal": int(n_cond_blocks),
        "n_cond_points_nominal": int(n_cond_blocks * int(model.max_points_per_cluster)),
        "loss": loss,
        "converged": int(np.isfinite(loss) and fit_steps < int(args.lbfgs_steps)),
        "fit_steps": fit_steps,
        "precompute_s": pre_s,
        "fit_s": fit_s,
        "total_s": pre_s + fit_s,
        **diag,
        **cluster_diag,
        **metrics,
        **{f"est_{k}": v for k, v in est.items()},
        "error": "",
    }
    del model, params, opt
    return row


def round_df(df: pd.DataFrame, digits: int = ROUND_DECIMALS) -> pd.DataFrame:
    out = df.copy()
    cols = out.select_dtypes(include=[np.number]).columns
    out[cols] = out[cols].round(digits)
    return out


def save_csv_rounded(df: pd.DataFrame, path: Path) -> None:
    round_df(df).to_csv(path, index=False, float_format=f"%.{ROUND_DECIMALS}f")


def append_row_csv(path: Path, row: dict[str, Any], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({col: row.get(col, "") for col in columns})


def existing_completed(raw_csv: Path) -> set[tuple[int, str, str]]:
    if not raw_csv.exists():
        return set()
    df = pd.read_csv(raw_csv)
    if df.empty:
        return set()
    ok = df[df["error"].fillna("") == ""] if "error" in df.columns else df
    return {
        (int(r.sim_id), str(r.strategy), str(r.block_shape))
        for r in ok.itertuples(index=False)
        if pd.notna(r.sim_id) and pd.notna(r.strategy) and pd.notna(r.block_shape)
    }


def make_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "error" not in df.columns:
        return pd.DataFrame()
    ok = df[df["error"].fillna("") == ""].copy()
    if ok.empty:
        return pd.DataFrame()
    metric_cols = [
        "loss",
        "overall_rmsre",
        "spatial_rmsre",
        "advec_rmsre",
        "median_abs_error",
        "range_time_re",
        "nugget_re",
        "precompute_s",
        "fit_s",
        "total_s",
        "n_batches",
        "n_target_points",
        "mean_m_by_template",
        "max_m_by_template",
    ]
    rows = []
    group_cols = ["strategy", "block_shape", "model_name"]
    for keys, sub in ok.groupby(group_cols, dropna=False):
        row = {
            "strategy": keys[0],
            "block_shape": keys[1],
            "model_name": keys[2],
            "n": int(len(sub)),
            "n_cond_points_nominal_median": float(sub["n_cond_points_nominal"].median()),
            "converged_rate": float(sub["converged"].astype(float).mean()) if "converged" in sub else np.nan,
        }
        for col in metric_cols:
            if col not in sub.columns:
                continue
            vals = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_median"] = np.nan
                row[f"{col}_p90_p10"] = np.nan
                continue
            p10, p90 = np.percentile(vals, [10, 90])
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_median"] = float(np.median(vals))
            row[f"{col}_p90_p10"] = float(p90 - p10)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["overall_rmsre_median", "median_abs_error_median"])


def make_param_summary(df: pd.DataFrame, truth: dict[str, float]) -> pd.DataFrame:
    if df.empty or "error" not in df.columns:
        return pd.DataFrame()
    ok = df[df["error"].fillna("") == ""].copy()
    if ok.empty:
        return pd.DataFrame()
    rows = []
    for keys, sub in ok.groupby(["strategy", "block_shape", "model_name"], dropna=False):
        for p in P_LABELS:
            est_col = f"est_{p}"
            if est_col not in sub.columns:
                continue
            vals = pd.to_numeric(sub[est_col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            tv = truth[p]
            abs_vals = np.abs(vals - tv)
            denom = abs(tv) if abs(tv) >= 0.01 else 1.0
            re_vals = abs_vals / denom
            p10_abs, p90_abs = np.percentile(abs_vals, [10, 90])
            p10_re, p90_re = np.percentile(re_vals, [10, 90])
            rows.append(
                {
                    "strategy": keys[0],
                    "block_shape": keys[1],
                    "model_name": keys[2],
                    "parameter": p,
                    "true": tv,
                    "n": int(len(vals)),
                    "median_abs_error": float(np.median(abs_vals)),
                    "mean_abs_error": float(np.mean(abs_vals)),
                    "p90_p10_abs_error": float(p90_abs - p10_abs),
                    "rmsre": float(np.sqrt(np.mean(re_vals**2))),
                    "median_re": float(np.median(re_vals)),
                    "p90_p10_re": float(p90_re - p10_re),
                    "estimate_mean": float(np.mean(vals)),
                    "estimate_median": float(np.median(vals)),
                    "estimate_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values(["parameter", "rmsre", "median_abs_error"])


def make_error_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "error" not in df.columns:
        return pd.DataFrame()
    err = df[df["error"].fillna("") != ""].copy()
    if err.empty:
        return pd.DataFrame()
    cols = ["fit_id", "sim_id", "asset_year", "asset_day_idx", "strategy", "block_shape", "error"]
    cols = [c for c in cols if c in err.columns]
    return err[cols].sort_values(["sim_id", "strategy", "block_shape"])


def refresh_running_outputs(
    raw_csv: Path,
    model_csv: Path,
    param_csv: Path,
    error_csv: Path,
    txt_path: Path,
    truth: dict[str, float],
) -> None:
    if not raw_csv.exists():
        return
    df = pd.read_csv(raw_csv)
    model_summary = make_model_summary(df)
    param_summary = make_param_summary(df, truth)
    error_summary = make_error_summary(df)
    if not model_summary.empty:
        save_csv_rounded(model_summary, model_csv)
    if not param_summary.empty:
        save_csv_rounded(param_summary, param_csv)
    if not error_summary.empty:
        error_summary.to_csv(error_csv, index=False)

    ok_n = int((df["error"].fillna("") == "").sum()) if "error" in df else int(len(df))
    err_n = int((df["error"].fillna("") != "").sum()) if "error" in df else 0
    lines = [
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        f"Rows: {len(df)} completed: {ok_n} errors: {err_n}",
        "",
        "Top running model summary:",
    ]
    if not model_summary.empty:
        cols = [
            "strategy",
            "block_shape",
            "n",
            "overall_rmsre_median",
            "overall_rmsre_p90_p10",
            "median_abs_error_median",
            "median_abs_error_p90_p10",
            "total_s_median",
            "n_cond_points_nominal_median",
        ]
        cols = [c for c in cols if c in model_summary.columns]
        lines.append(round_df(model_summary[cols].head(20)).to_string(index=False))
    if not param_summary.empty:
        lines.extend(["", "Parameter summary preview:"])
        cols = [
            "strategy",
            "block_shape",
            "parameter",
            "n",
            "rmsre",
            "median_abs_error",
            "p90_p10_abs_error",
            "median_re",
        ]
        cols = [c for c in cols if c in param_summary.columns]
        lines.append(round_df(param_summary[cols].head(35)).to_string(index=False))
    if not error_summary.empty:
        lines.extend(["", "Recent errors:"])
        lines.append(error_summary.tail(20).to_string(index=False))
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4]), flush=True)
    if not model_summary.empty:
        print(round_df(model_summary.head(12)).to_string(index=False), flush=True)


def make_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = []
    for block_shape in args.block_shapes:
        for strategy in args.strategies:
            specs.append({"strategy": strategy, "block_shape": block_shape})
    return specs


def choose_asset(asset_bank: list[dict[str, Any]], sim_id: int, args: argparse.Namespace) -> dict[str, Any]:
    if args.asset_sampling == "random":
        rng = np.random.default_rng(args.seed + 10_000 + sim_id)
        return asset_bank[int(rng.integers(0, len(asset_bank)))]
    return asset_bank[(sim_id - 1) % len(asset_bank)]


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-sims", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--years", type=str, default="2022,2023,2024,2025")
    parser.add_argument("--day-idxs", type=str, default="all")
    parser.add_argument("--max-asset-days", type=int, default=0)
    parser.add_argument("--asset-sampling", type=str, default="cycle", choices=["cycle", "random"])
    parser.add_argument("--sim-data-root", type=Path, default=Path("/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern"))
    parser.add_argument("--data-kind", type=str, default="real_locations", choices=["gridded", "real_locations"])
    parser.add_argument("--lat-range", type=parse_range, default=parse_range("-3,2"))
    parser.add_argument("--lon-range", type=parse_range, default=parse_range("121,131"))
    parser.add_argument("--strategies", type=parse_strategies, default=DEFAULT_STRATEGIES)
    parser.add_argument("--block-shapes", type=parse_block_shapes, default=parse_block_shapes("3x3,4x4"))
    parser.add_argument("--lag0-block-count", type=int, default=6)
    parser.add_argument("--lag1-keep-fraction", type=float, default=0.80)
    parser.add_argument("--lag2-keep-fraction", type=float, default=0.50)
    parser.add_argument("--lag1-lon-offset", type=float, default=2.0 * DELTA_LON_BASE)
    parser.add_argument("--lag2-lon-offset", type=float, default=4.0 * DELTA_LON_BASE)
    parser.add_argument("--daily-stride", type=int, default=2)
    parser.add_argument("--target-chunk-size", type=int, default=128)
    parser.add_argument("--min-target-points", type=int, default=1)
    parser.add_argument("--lbfgs-steps", type=int, default=5)
    parser.add_argument("--lbfgs-eval", type=int, default=15)
    parser.add_argument("--lbfgs-hist", type=int, default=10)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    parser.add_argument("--init-noise", type=float, default=0.25)
    parser.add_argument("--summary-every", type=int, default=5)
    parser.add_argument("--no-center-response", dest="center_response", action="store_false")
    parser.set_defaults(center_response=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out-dir", type=Path, default=Path("/home/jl2815/tco/exercise_output/estimates/day/cluster_strategy_sweep_052226"))


def main() -> None:
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    if args.require_cuda and DEVICE.type != "cuda":
        raise RuntimeError("--require-cuda was passed, but torch.cuda.is_available() is False")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = args.out_dir / "all_fits_summary.csv"
    model_csv = args.out_dir / "running_model_summary.csv"
    param_csv = args.out_dir / "running_param_summary.csv"
    error_csv = args.out_dir / "running_errors.csv"
    txt_path = args.out_dir / "running_summary.txt"
    truth_json = args.out_dir / "truth_params.json"
    config_json = args.out_dir / "run_config.json"

    set_seed(args.seed)
    print("SRC:", SRC, flush=True)
    print("DEVICE:", DEVICE, flush=True)
    print("torch:", torch.__version__, flush=True)
    print("args:", vars(args), flush=True)
    print("NOTE: loading pre-generated simulation assets; no new simulation is run.", flush=True)

    asset_bank, truth = build_asset_bank(args)
    smooth = float(truth.get("smooth", 0.5))
    true_log = true_to_log_params(truth)
    specs = make_specs(args)

    truth_json.write_text(json.dumps(truth, indent=2), encoding="utf-8")
    config = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "src": SRC,
        "device": str(DEVICE),
        "n_assets_loaded": len(asset_bank),
        "n_specs": len(specs),
        "specs": [
            {"strategy": s["strategy"], "block_shape": f"{s['block_shape'][0]}x{s['block_shape'][1]}"}
            for s in specs
        ],
    }
    config_json.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("truth:", {k: truth[k] for k in P_LABELS}, flush=True)
    print(f"Loaded {len(asset_bank)} day assets; smooth={smooth}; specs={len(specs)}", flush=True)

    done = existing_completed(raw_csv) if args.resume else set()
    fit_id = 0
    if raw_csv.exists():
        old = pd.read_csv(raw_csv)
        if not old.empty and "fit_id" in old:
            old_fit_ids = pd.to_numeric(old["fit_id"], errors="coerce").dropna()
            if not old_fit_ids.empty:
                fit_id = int(old_fit_ids.max())

    for sim_id in range(1, int(args.num_sims) + 1):
        print("\n" + "=" * 100, flush=True)
        print(f"Simulation {sim_id}/{args.num_sims}", flush=True)
        asset = choose_asset(asset_bank, sim_id, args)
        iter_seed = int(args.seed + sim_id)
        set_seed(iter_seed)
        rng = np.random.default_rng(iter_seed)
        initial_vals = make_random_init(rng, true_log, args.init_noise)
        source_map = {
            k: v.to(device=DEVICE, dtype=DTYPE, non_blocking=True).contiguous()
            for k, v in asset["source_map"].items()
        }
        grid_coords_np = np.asarray(asset["grid_coords_np"], dtype=np.float64)
        n_valid = sum(int(torch.isfinite(v[:, 2]).sum().item()) for v in source_map.values())
        print(
            f"asset=year{asset['year']} day_idx={asset['day_idx']} "
            f"n_grid={asset['n_grid']:,} n_valid={n_valid:,} valid={asset['valid_by_t']} "
            f"initial={[round(x, 4) for x in initial_vals]}",
            flush=True,
        )

        for spec in specs:
            block_label = f"{spec['block_shape'][0]}x{spec['block_shape'][1]}"
            key = (sim_id, spec["strategy"], block_label)
            if args.resume and key in done:
                print(f"Skipping completed sim={sim_id} strategy={spec['strategy']} block={block_label}", flush=True)
                continue

            fit_id += 1
            print(f"\n--- fit_id={fit_id} strategy={spec['strategy']} block={block_label} ---", flush=True)
            row_base = {
                "fit_id": fit_id,
                "sim_id": sim_id,
                "seed": iter_seed,
                "asset_year": asset["year"],
                "asset_day_idx": asset["day_idx"],
                "asset_first_key": asset["day_keys"][0],
                "data_kind": args.data_kind,
                "strategy": spec["strategy"],
                "block_shape": block_label,
                "n_valid": int(n_valid),
                "n_grid": int(asset["n_grid"]),
                "valid_by_t": json.dumps(asset["valid_by_t"], separators=(",", ":")),
            }
            try:
                row = fit_strategy(source_map, grid_coords_np, initial_vals, spec, args, truth, smooth)
                row.update(row_base)
                compact = {
                    k: (round(v, 4) if isinstance(v, (float, np.floating)) else v)
                    for k, v in row.items()
                    if k in [
                        "model_name",
                        "loss",
                        "overall_rmsre",
                        "median_abs_error",
                        "spatial_rmsre",
                        "advec_rmsre",
                        "fit_steps",
                        "total_s",
                        "n_cond_points_nominal",
                    ]
                }
                print("RESULT:", compact, flush=True)
            except Exception as exc:
                row = {
                    **row_base,
                    "model_name": f"{spec['strategy']}_block{block_label}",
                    "error": repr(exc),
                }
                print("ERROR:", row, flush=True)

            append_row_csv(raw_csv, row, ROW_COLUMNS)
            if int(args.summary_every) > 0 and fit_id % int(args.summary_every) == 0:
                refresh_running_outputs(raw_csv, model_csv, param_csv, error_csv, txt_path, truth)
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        del source_map
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    refresh_running_outputs(raw_csv, model_csv, param_csv, error_csv, txt_path, truth)
    print("Saved outputs under:", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
