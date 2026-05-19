#!/usr/bin/env python3
"""
Compare four Vecchia variants on pre-generated July ST circulant assets.

This entrypoint intentionally never simulates new data.  It only reads the
pickles created by:

  Exercises/st_model/simulate_data/generate_july_st_circulant_real_locations_2022_2025.py

Models:
  1. kernels_vecchia_hybrid.HybridVecchiaFit
  2. kernels_vecchia_cluster_hybrid.ClusterHybridVecchiaFit
  3. kernel_vecchia_col_batch.ReverseLColumnVecchiaFitBatch
  4. kernels_vecchia_cluster_column_batch.ClusterColumnVecchiaFitBatch
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


AMAREL_SRC = "/home/jl2815/tco"
LOCAL_SRC = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
SRC = AMAREL_SRC if os.path.exists(AMAREL_SRC) else LOCAL_SRC
sys.path.insert(0, SRC)

from GEMS_TCO import orderings as _orderings
from GEMS_TCO.kernel_vecchia_col_batch import ReverseLColumnVecchiaFitBatch
from GEMS_TCO.kernels_vecchia_cluster_column_batch import ClusterColumnVecchiaFitBatch
from GEMS_TCO.kernels_vecchia_cluster_hybrid import ClusterHybridVecchiaFit
from GEMS_TCO.vecchia_candidate.kernels_vecchia_hybrid import HybridVecchiaFit


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
ROUND_DECIMALS = 4
DELTA_LAT_BASE = 0.044
DELTA_LON_BASE = 0.063
CLUSTER_HYBRID_LON_OFFSET = 2.0 * DELTA_LON_BASE
T_STEPS = 8

P_LABELS = ["sigmasq", "range_lat", "range_lon", "range_time", "advec_lat", "advec_lon", "nugget"]
SPATIAL_KEYS = ["sigmasq", "range_lat", "range_lon"]
ADVECTION_KEYS = ["advec_lat", "advec_lon"]

HYBRID_SPEC = {
    "model": "Hybrid_Lean_L08F04_C4F03_Op0p063_exactloc",
    "limit_A": 20,
    "lag1_local_count": 8,
    "lag1_fresh_count": 4,
    "lag2_local_count": 4,
    "lag2_fresh_count": 3,
    "daily_stride": 2,
    "lag1_lon_offset": DELTA_LON_BASE,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_range(s: str) -> list[float]:
    return [float(x.strip()) for x in str(s).split(",")]


def parse_int_list(s: str) -> list[int] | str:
    text = str(s).strip().lower()
    if text in {"all", "*"}:
        return "all"
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_years(s: str) -> list[int]:
    return [int(x.strip()) for x in str(s).split(",") if x.strip()]


def parse_block_shape(s: str) -> tuple[int, int]:
    text = str(s).lower().replace("x", ",")
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != 2:
        raise argparse.ArgumentTypeError("block shape must look like 3x3 or 3,3")
    return vals[0], vals[1]


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


def backmap_params(out_params) -> dict[str, float]:
    p = [float(x.item() if isinstance(x, torch.Tensor) else x) for x in out_params[:7]]
    phi2 = np.exp(p[1])
    phi3 = np.exp(p[2])
    phi4 = np.exp(p[3])
    rlon = 1.0 / phi2
    return {
        "sigmasq": np.exp(p[0]) / phi2,
        "range_lat": rlon / phi3**0.5,
        "range_lon": rlon,
        "range_time": rlon / phi4**0.5,
        "advec_lat": p[4],
        "advec_lon": p[5],
        "nugget": np.exp(p[6]),
    }


def make_random_init(rng: np.random.Generator, true_log: list[float], init_noise: float) -> list[float]:
    noisy = list(true_log)
    for idx in [0, 1, 2, 3, 6]:
        noisy[idx] = true_log[idx] + rng.uniform(-init_noise, init_noise)
    for idx in [4, 5]:
        scale = max(abs(true_log[idx]), 0.05)
        noisy[idx] = true_log[idx] + rng.uniform(-2 * scale, 2 * scale)
    return noisy


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
    metrics = {
        "overall_rmsre": rmsre_for_keys(est, truth, P_LABELS),
        "spatial_rmsre": rmsre_for_keys(est, truth, SPATIAL_KEYS),
        "advec_rmsre": rmsre_for_keys(est, truth, ADVECTION_KEYS),
        "range_time_re": abs(est["range_time"] - truth["range_time"]) / abs(truth["range_time"]),
        "nugget_re": abs(est["nugget"] - truth["nugget"]) / abs(truth["nugget"]),
    }
    for key in P_LABELS:
        denom = abs(truth[key]) if abs(truth[key]) >= 0.01 else 1.0
        metrics[f"{key}_re"] = abs(est[key] - truth[key]) / denom
    return metrics, est


def round_df(df: pd.DataFrame, digits: int = ROUND_DECIMALS) -> pd.DataFrame:
    out = df.copy()
    cols = out.select_dtypes(include=[np.number]).columns
    out[cols] = out[cols].round(digits)
    return out


def save_csv_rounded(df: pd.DataFrame, path: Path) -> None:
    round_df(df).to_csv(path, index=False, float_format=f"%.{ROUND_DECIMALS}f")


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
    """Build the native GEMS regular grid used by the reference simulation code.

    The generated gridded pickles may carry duplicated template rows.  For
    Vecchia geometry, especially block/group Vecchia, the conditioning geometry
    must be built on the native regular grid, not on duplicated template rows.
    """
    lats = np.arange(min(lat_range), max(lat_range) + 1e-4, DELTA_LAT_BASE, dtype=float)
    lons = np.arange(min(lon_range), max(lon_range) + 1e-4, DELTA_LON_BASE, dtype=float)
    lats = np.round(lats, ROUND_DECIMALS)
    lons = np.round(lons, ROUND_DECIMALS)
    g_lat, g_lon = np.meshgrid(lats, lons, indexing="ij")
    master = pd.DataFrame({
        "_grid_lat": g_lat.ravel(),
        "_grid_lon": g_lon.ravel(),
    })
    master["_grid_lat_key"] = np.round(master["_grid_lat"].to_numpy(dtype=float), ROUND_DECIMALS)
    master["_grid_lon_key"] = np.round(master["_grid_lon"].to_numpy(dtype=float), ROUND_DECIMALS)
    return master[["_grid_lat_key", "_grid_lon_key", "_grid_lat", "_grid_lon"]]


def align_to_master_grid(df: pd.DataFrame, master: pd.DataFrame, lat_range: list[float], lon_range: list[float]) -> pd.DataFrame:
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

    valid_y = np.isfinite(y)
    lat = np.where(valid_y & np.isfinite(lat), lat, np.nan)
    lon = np.where(valid_y & np.isfinite(lon), lon, np.nan)

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
    master: pd.DataFrame | None = None,
    monthly_mean: float | None = None,
) -> dict[str, Any] | None:
    keys = ordered_asset_keys(sim_map)
    day_keys = keys[day_idx * T_STEPS : (day_idx + 1) * T_STEPS]
    if len(day_keys) < T_STEPS:
        return None

    if master is None:
        master = build_master_grid(sim_map[day_keys[0]], lat_range, lon_range)
    if monthly_mean is None:
        month_vals = []
        for key in keys:
            aligned = align_to_master_grid(sim_map[key], master, lat_range, lon_range)
            vals = pd.to_numeric(aligned["ColumnAmountO3"], errors="coerce").to_numpy(dtype=float)
            month_vals.append(vals)
        monthly_mean = float(np.nanmean(np.concatenate(month_vals)))

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
        "monthly_mean": monthly_mean,
        "valid_by_t": valid_by_t,
    }


def build_asset_bank(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, float]]:
    years = parse_years(args.years)
    requested_day_idxs = parse_int_list(args.day_idxs)
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
                "falling back to native 0.044 x 0.063 grid.",
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
                print(
                    f"WARNING: skipping asset year={year} day_idx={day_idx}; all 8 slots have zero valid observations.",
                    flush=True,
                )
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


def is_permutation(order: np.ndarray, n: int) -> bool:
    order = np.asarray(order)
    return (
        order.ndim == 1
        and order.size == n
        and order.min(initial=0) >= 0
        and order.max(initial=-1) < n
        and np.unique(order).size == n
    )


def maxmin_fallback_order(coords: np.ndarray) -> np.ndarray:
    """Pure NumPy farthest-point max-min ordering for regular grids.

    The C++ max-min wrapper can return duplicate indices on large perfectly
    regular grids in this environment.  This fallback is O(n^2) but runs once
    per unique grid and returns a true permutation.
    """
    x = np.asarray(coords, dtype=np.float64)
    n = int(x.shape[0])
    order = np.empty(n, dtype=np.int64)
    center = x.mean(axis=0, keepdims=True)
    first = int(np.argmin(np.sum((x - center) ** 2, axis=1)))
    order[0] = first
    min_sq = np.sum((x - x[first]) ** 2, axis=1)
    min_sq[first] = -np.inf
    for i in range(1, n):
        j = int(np.argmax(min_sq))
        order[i] = j
        d_sq = np.sum((x - x[j]) ** 2, axis=1)
        np.minimum(min_sq, d_sq, out=min_sq)
        min_sq[j] = -np.inf
    return order


def compute_grid_ordering(grid_coords_np: np.ndarray, mm_cond_number: int) -> tuple[np.ndarray, np.ndarray]:
    ord_mm = _orderings.maxmin_cpp(grid_coords_np)
    if not is_permutation(ord_mm, len(grid_coords_np)):
        print(
            "WARNING: maxmin_cpp did not return a valid permutation "
            f"(n={len(grid_coords_np)}, unique={np.unique(ord_mm).size}); using NumPy fallback.",
            flush=True,
        )
        ord_mm = maxmin_fallback_order(grid_coords_np)
    nns = _orderings.find_nns_l2(locs=grid_coords_np[ord_mm], max_nn=mm_cond_number)
    return ord_mm, nns


def grid_cache_key(grid_coords_np: np.ndarray) -> tuple[Any, ...]:
    first = tuple(np.round(grid_coords_np[0], 6))
    last = tuple(np.round(grid_coords_np[-1], 6))
    return (int(grid_coords_np.shape[0]), first, last)


def count_valid(day_map: dict[str, torch.Tensor]) -> int:
    return sum(int(torch.isfinite(v[:, 2]).sum().item()) for v in day_map.values())


def template_diagnostics(model) -> dict[str, float]:
    batched = getattr(model, "Batched_Groups", None)
    if batched:
        group_sizes = np.asarray([int(g["target_idx"].shape[0]) for g in batched], dtype=np.int64)
        m_sizes = np.asarray([int(g["max_m"]) for g in batched], dtype=np.int64)
        return {
            "n_templates": np.nan,
            "n_batches": int(len(batched)),
            "largest_template_n": int(group_sizes.max()),
            "median_template_n": float(np.median(group_sizes)),
            "mean_template_n": float(group_sizes.mean()),
            "mean_m_by_template": float(m_sizes.mean()),
            "median_m_by_template": float(np.median(m_sizes)),
            "max_m_by_template": int(m_sizes.max()),
        }

    cluster_batches = getattr(model, "_cluster_batches", None)
    if cluster_batches:
        batch_sizes = np.asarray([int(b.X.shape[0]) for b in cluster_batches], dtype=np.int64)
        m_sizes = np.asarray([int(b.max_cond_points) for b in cluster_batches], dtype=np.int64)
        target_sizes = np.asarray([int(b.target_size) for b in cluster_batches], dtype=np.int64)
        return {
            "n_templates": np.nan,
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

    groups = getattr(model, "Grouped_Batches", [])
    if groups:
        group_sizes = np.asarray([int(g["target_idx"].shape[0]) for g in groups], dtype=np.int64)
        m_sizes = np.asarray([int(g["offsets"].shape[0]) for g in groups], dtype=np.int64)
        return {
            "n_templates": int(len(groups)),
            "n_batches": np.nan,
            "largest_template_n": int(group_sizes.max()),
            "median_template_n": float(np.median(group_sizes)),
            "mean_template_n": float(group_sizes.mean()),
            "mean_m_by_template": float(m_sizes.mean()),
            "median_m_by_template": float(np.median(m_sizes)),
            "max_m_by_template": int(m_sizes.max()),
        }

    return {
        "n_templates": 0,
        "n_batches": 0,
        "largest_template_n": 0,
        "median_template_n": 0.0,
        "mean_template_n": 0.0,
        "mean_m_by_template": 0.0,
        "median_m_by_template": 0.0,
        "max_m_by_template": 0,
    }


def assert_reasonable_cluster_geometry(model, block_shape: tuple[int, int]) -> None:
    max_expected = int(block_shape[0]) * int(block_shape[1])
    max_seen = int(getattr(model, "max_points_per_cluster", 0))
    if max_seen > max_expected:
        raise RuntimeError(
            f"Invalid cluster geometry: max_points_per_cluster={max_seen}, "
            f"but block_shape={block_shape} should allow at most {max_expected}. "
            "Check that grid_coords are the native regular grid, not duplicated template rows."
        )


def new_params(initial_vals: list[float]) -> list[torch.Tensor]:
    return [torch.tensor([v], device=DEVICE, dtype=DTYPE, requires_grad=True) for v in initial_vals]


def fit_hybrid(source_map_ord, nns_grid, initial_vals, args, truth: dict[str, float], smooth: float) -> dict[str, Any]:
    params = new_params(initial_vals)
    model = HybridVecchiaFit(
        smooth=smooth,
        input_map=source_map_ord,
        nns_map=nns_grid,
        mm_cond_number=args.mm_cond_number,
        nheads=0,
        limit_A=HYBRID_SPEC["limit_A"],
        limit_B_local=HYBRID_SPEC["lag1_local_count"],
        limit_C_local=HYBRID_SPEC["lag2_local_count"],
        daily_stride=HYBRID_SPEC["daily_stride"],
        spatial_coords=None,
        lag1_lon_offset=HYBRID_SPEC["lag1_lon_offset"],
        lag1_fresh_count=HYBRID_SPEC["lag1_fresh_count"],
        lag2_fresh_count=HYBRID_SPEC["lag2_fresh_count"],
    )
    t0 = time.time()
    model.precompute_conditioning_sets()
    pre_s = time.time() - t0
    opt = model.set_optimizer(params, lr=1.0, max_iter=args.lbfgs_eval, max_eval=args.lbfgs_eval, history_size=args.lbfgs_hist)
    t1 = time.time()
    out, fit_iter = model.fit_vecc_lbfgs(params, opt, max_steps=args.lbfgs_steps, grad_tol=1e-5)
    fit_s = time.time() - t1
    metrics, est = calculate_metrics(out, truth)
    row = {
        "model": HYBRID_SPEC["model"],
        "kernel": "kernels_vecchia_hybrid",
        "model_family": "point",
        "block_shape": "",
        "coords_used": "real_source_offsets_regular_grid_ordering",
        "loss": float(out[-1]),
        "fit_iter": int(fit_iter),
        "precompute_s": pre_s,
        "fit_s": fit_s,
        "total_s": pre_s + fit_s,
        "total_conditioning_nominal": 41,
        "n_valid": count_valid(source_map_ord),
        "n_heads": int(model.Heads_data.shape[0]),
        "n_tails": int(model.n_tails),
        "n_templates": np.nan,
        **metrics,
        **{f"est_{k}": v for k, v in est.items()},
    }
    del model, params, opt
    return row


def fit_cluster_hybrid(source_map_ord, ordered_grid_coords_np, initial_vals, args, truth: dict[str, float], smooth: float) -> dict[str, Any]:
    params = new_params(initial_vals)
    model = ClusterHybridVecchiaFit(
        smooth=smooth,
        input_map=source_map_ord,
        grid_coords=ordered_grid_coords_np,
        block_shape=args.block_shape,
        n_neighbor_blocks_t=args.cluster_hybrid_lag0_blocks,
        lag1_same_block=args.cluster_hybrid_lag1_same_block,
        lag1_local_blocks=args.cluster_hybrid_lag1_local_blocks,
        lag1_shifted_blocks=args.cluster_hybrid_lag1_shifted_blocks,
        lag2_same_block=args.cluster_hybrid_lag2_same_block,
        lag2_local_blocks=args.cluster_hybrid_lag2_local_blocks,
        lag2_shifted_blocks=args.cluster_hybrid_lag2_shifted_blocks,
        daily_stride=HYBRID_SPEC["daily_stride"],
        lag1_lon_offset=args.cluster_hybrid_lag1_lon_offset,
        lag2_lon_offset=args.cluster_hybrid_lag2_lon_offset,
        target_chunk_size=args.cluster_hybrid_chunk_size,
        min_target_points=args.min_target_points,
    )
    t0 = time.time()
    model.precompute_conditioning_sets()
    pre_s = time.time() - t0
    assert_reasonable_cluster_geometry(model, args.block_shape)
    diag = template_diagnostics(model)
    cluster_diag = model.cluster_summary()
    opt = model.set_optimizer(params, lr=1.0, max_iter=args.lbfgs_eval, max_eval=args.lbfgs_eval, history_size=args.lbfgs_hist)
    t1 = time.time()
    out, fit_iter = model.fit_vecc_lbfgs(params, opt, max_steps=args.lbfgs_steps, grad_tol=1e-5)
    fit_s = time.time() - t1
    metrics, est = calculate_metrics(out, truth)
    final_blocks = (
        args.cluster_hybrid_lag0_blocks
        + int(args.cluster_hybrid_lag1_same_block)
        + args.cluster_hybrid_lag1_local_blocks
        + args.cluster_hybrid_lag1_shifted_blocks
        + int(args.cluster_hybrid_lag2_same_block)
        + args.cluster_hybrid_lag2_local_blocks
        + args.cluster_hybrid_lag2_shifted_blocks
    )
    row = {
        "model": (
            f"ClusterHybrid_block{args.block_shape[0]}x{args.block_shape[1]}_"
            f"A{args.cluster_hybrid_lag0_blocks}_"
            f"Bsame{int(args.cluster_hybrid_lag1_same_block)}L{args.cluster_hybrid_lag1_local_blocks}"
            f"F{args.cluster_hybrid_lag1_shifted_blocks}_"
            f"Csame{int(args.cluster_hybrid_lag2_same_block)}L{args.cluster_hybrid_lag2_local_blocks}"
            f"F{args.cluster_hybrid_lag2_shifted_blocks}_"
            f"O1{args.cluster_hybrid_lag1_lon_offset:.3f}_O2{args.cluster_hybrid_lag2_lon_offset:.3f}".replace(".", "p")
        ),
        "kernel": "kernels_vecchia_cluster_hybrid",
        "model_family": "group",
        "block_shape": f"{args.block_shape[0]}x{args.block_shape[1]}",
        "coords_used": "real_source_offsets_regular_grid_blocks",
        "cluster_lag1_lon_offset": args.cluster_hybrid_lag1_lon_offset,
        "cluster_lag2_lon_offset": args.cluster_hybrid_lag2_lon_offset,
        "loss": float(out[-1]),
        "fit_iter": int(fit_iter),
        "precompute_s": pre_s,
        "fit_s": fit_s,
        "total_s": pre_s + fit_s,
        "total_conditioning_nominal": final_blocks * int(model.max_points_per_cluster),
        "n_valid": count_valid(source_map_ord),
        "n_heads": int(model.Heads_data.shape[0]),
        "n_tails": int(model.n_tails),
        **diag,
        **cluster_diag,
        **metrics,
        **{f"est_{k}": v for k, v in est.items()},
    }
    del model, params, opt
    return row


def fit_column(source_map_ord, ordered_grid_coords_np, initial_vals, args, truth: dict[str, float], smooth: float) -> dict[str, Any]:
    params = new_params(initial_vals)
    above_count = int(args.column_above_count)
    right_col_count = int(args.column_right_col_count)
    per_lag_count = int(args.column_per_lag_count)
    lag_count = int(args.column_lag_count)
    total_conditioning = above_count + per_lag_count * (lag_count + 1)
    model_name = (
        f"ColumnV3Batched_Up{above_count}_Right{right_col_count}_"
        f"Down{per_lag_count}_Lag{lag_count}_head{args.column_head_right_cols}_realloc"
    )
    model = ReverseLColumnVecchiaFitBatch(
        smooth=smooth,
        input_map=source_map_ord,
        mm_cond_number=args.mm_cond_number,
        grid_coords=ordered_grid_coords_np,
        head_right_cols=args.column_head_right_cols,
        above_count=above_count,
        right_col_count=right_col_count,
        per_lag_conditioning_count=per_lag_count,
        lag_count=lag_count,
        include_lag_self=False,
        target_chunk_size=args.column_chunk_size,
        use_data_coords_for_offsets=True,
    )
    t0 = time.time()
    model.precompute_conditioning_sets()
    pre_s = time.time() - t0
    assert_reasonable_cluster_geometry(model, args.block_shape)
    diag = template_diagnostics(model)
    opt = model.set_optimizer(params, lr=1.0, max_iter=args.lbfgs_eval, max_eval=args.lbfgs_eval, history_size=args.lbfgs_hist)
    t1 = time.time()
    out, fit_iter = model.fit_vecc_lbfgs(params, opt, max_steps=args.lbfgs_steps, grad_tol=1e-5)
    fit_s = time.time() - t1
    metrics, est = calculate_metrics(out, truth)
    row = {
        "model": model_name,
        "kernel": "kernel_vecchia_col_batch",
        "model_family": "point",
        "block_shape": "",
        "head_right_cols": int(args.column_head_right_cols),
        "above_count": above_count,
        "right_col_count": right_col_count,
        "per_lag_conditioning_count": per_lag_count,
        "lag_count": lag_count,
        "coords_used": "real_source_offsets_regular_grid_reverse_l",
        "loss": float(out[-1]),
        "fit_iter": int(fit_iter),
        "precompute_s": pre_s,
        "fit_s": fit_s,
        "total_s": pre_s + fit_s,
        "total_conditioning_nominal": total_conditioning,
        "n_valid": count_valid(source_map_ord),
        "n_heads": int(model.Heads_data.shape[0]),
        "n_tails": int(model.n_tails),
        **diag,
        **metrics,
        **{f"est_{k}": v for k, v in est.items()},
    }
    del model, params, opt
    return row


def fit_cluster_column(source_map_ord, ordered_grid_coords_np, initial_vals, args, truth: dict[str, float], smooth: float) -> dict[str, Any]:
    params = new_params(initial_vals)
    model = ClusterColumnVecchiaFitBatch(
        smooth=smooth,
        input_map=source_map_ord,
        grid_coords=ordered_grid_coords_np,
        block_shape=args.block_shape,
        lag0_block_count=args.cluster_column_lag0_blocks,
        lag1_same_block=args.cluster_column_lag1_same_block,
        lag1_block_count=args.cluster_column_lag1_block_count,
        lag2_same_block=args.cluster_column_lag2_same_block,
        lag2_block_count=args.cluster_column_lag2_block_count,
        daily_stride=HYBRID_SPEC["daily_stride"],
        above_block_count=args.cluster_column_above_blocks,
        right_block_count=args.cluster_column_right_blocks,
        target_chunk_size=args.cluster_column_chunk_size,
        min_target_points=args.min_target_points,
    )
    t0 = time.time()
    model.precompute_conditioning_sets()
    pre_s = time.time() - t0
    diag = template_diagnostics(model)
    cluster_diag = model.cluster_summary()
    opt = model.set_optimizer(params, lr=1.0, max_iter=args.lbfgs_eval, max_eval=args.lbfgs_eval, history_size=args.lbfgs_hist)
    t1 = time.time()
    out, fit_iter = model.fit_vecc_lbfgs(params, opt, max_steps=args.lbfgs_steps, grad_tol=1e-5)
    fit_s = time.time() - t1
    metrics, est = calculate_metrics(out, truth)
    final_blocks = (
        args.cluster_column_lag0_blocks
        + int(args.cluster_column_lag1_same_block)
        + args.cluster_column_lag1_block_count
        + int(args.cluster_column_lag2_same_block)
        + args.cluster_column_lag2_block_count
    )
    row = {
        "model": (
            f"ClusterColumn_block{args.block_shape[0]}x{args.block_shape[1]}_"
            f"A{args.cluster_column_lag0_blocks}_"
            f"Bsame{int(args.cluster_column_lag1_same_block)}R{args.cluster_column_lag1_block_count}_"
            f"Csame{int(args.cluster_column_lag2_same_block)}R{args.cluster_column_lag2_block_count}_"
            f"stencilU{args.cluster_column_above_blocks}R{args.cluster_column_right_blocks}"
        ),
        "kernel": "kernels_vecchia_cluster_column_batch",
        "model_family": "group",
        "block_shape": f"{args.block_shape[0]}x{args.block_shape[1]}",
        "coords_used": "real_source_offsets_regular_grid_reverse_l_blocks",
        "loss": float(out[-1]),
        "fit_iter": int(fit_iter),
        "precompute_s": pre_s,
        "fit_s": fit_s,
        "total_s": pre_s + fit_s,
        "total_conditioning_nominal": final_blocks * int(model.max_points_per_cluster),
        "n_valid": count_valid(source_map_ord),
        "n_heads": int(model.Heads_data.shape[0]),
        "n_tails": int(model.n_tails),
        **diag,
        **cluster_diag,
        **metrics,
        **{f"est_{k}": v for k, v in est.items()},
    }
    del model, params, opt
    return row


def make_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    error_col = df.get("error")
    ok = df if error_col is None else df[error_col.fillna("") == ""].copy()
    if ok.empty:
        return pd.DataFrame()
    metric_cols = [
        "loss",
        "overall_rmsre",
        "spatial_rmsre",
        "advec_rmsre",
        "range_time_re",
        "nugget_re",
        "precompute_s",
        "fit_s",
        "total_s",
        "n_batches",
        "n_templates",
        "n_heads",
        "n_tails",
        "mean_m_by_template",
        "max_m_by_template",
    ]
    rows = []
    group_cols = ["model", "kernel", "model_family", "block_shape"]
    for keys, sub in ok.groupby(group_cols, dropna=False):
        row = {
            "model": keys[0],
            "kernel": keys[1],
            "model_family": keys[2],
            "block_shape": keys[3],
            "n": int(len(sub)),
            "total_conditioning_median": float(sub["total_conditioning_nominal"].median())
            if "total_conditioning_nominal" in sub
            else np.nan,
        }
        for col in metric_cols:
            if col not in sub.columns:
                continue
            vals = sub[col].dropna().astype(float).values
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
    return pd.DataFrame(rows).sort_values("overall_rmsre_median")


def make_param_summary(df: pd.DataFrame, truth: dict[str, float]) -> pd.DataFrame:
    error_col = df.get("error")
    ok = df if error_col is None else df[error_col.fillna("") == ""].copy()
    if ok.empty:
        return pd.DataFrame()
    rows = []
    for model, sub in ok.groupby("model"):
        for p in P_LABELS:
            col = f"est_{p}"
            if col not in sub.columns:
                continue
            vals = sub[col].dropna().astype(float).values
            if len(vals) == 0:
                continue
            tv = truth[p]
            denom = abs(tv) if abs(tv) >= 0.01 else 1.0
            re_vals = np.abs((vals - tv) / denom)
            p10, p90 = np.percentile(re_vals, [10, 90])
            rows.append({
                "model": model,
                "parameter": p,
                "true": tv,
                "rmsre": float(np.sqrt(np.mean(re_vals**2))),
                "mean_re": float(np.mean(re_vals)),
                "median_re": float(np.median(re_vals)),
                "p10_re": float(p10),
                "p90_re": float(p90),
                "p90_p10_re": float(p90 - p10),
                "estimate_mean": float(np.mean(vals)),
                "estimate_median": float(np.median(vals)),
                "estimate_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            })
    return pd.DataFrame(rows).sort_values(["model", "parameter"])


def make_error_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "error" not in df.columns:
        return pd.DataFrame()
    err = df[df["error"].fillna("") != ""].copy()
    if err.empty:
        return pd.DataFrame()
    cols = ["iter", "seed", "asset_year", "asset_day_idx", "model", "kernel", "error"]
    cols = [c for c in cols if c in err.columns]
    return err[cols].sort_values(["iter", "model"])


def save_outputs(
    rows: list[dict[str, Any]],
    raw_csv: Path,
    model_csv: Path,
    param_csv: Path,
    error_csv: Path,
    truth: dict[str, float],
) -> None:
    df = pd.DataFrame(rows)
    save_csv_rounded(df, raw_csv)
    model_summary = make_model_summary(df)
    param_summary = make_param_summary(df, truth)
    error_summary = make_error_summary(df)
    if not model_summary.empty:
        save_csv_rounded(model_summary, model_csv)
        print("\nRunning model summary (mean, median, p90-p10)", flush=True)
        print(round_df(model_summary).to_string(index=False), flush=True)
    if not param_summary.empty:
        save_csv_rounded(param_summary, param_csv)
        print("\nRunning parameter summary (includes RMSRE median and p90-p10)", flush=True)
        print(round_df(param_summary).to_string(index=False), flush=True)
    if not error_summary.empty:
        error_summary.to_csv(error_csv, index=False)
        print("\nRunning error summary", flush=True)
        print(error_summary.tail(12).to_string(index=False), flush=True)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--years", type=str, default="2022,2023,2024,2025")
    parser.add_argument("--day-idxs", type=str, default="all")
    parser.add_argument("--max-asset-days", type=int, default=0)
    parser.add_argument("--sim-data-root", type=Path, default=Path("/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern"))
    parser.add_argument("--data-kind", type=str, default="gridded", choices=["gridded", "real_locations"])
    parser.add_argument("--lat-range", type=parse_range, default=parse_range("-3,2"))
    parser.add_argument("--lon-range", type=parse_range, default=parse_range("121,131"))
    parser.add_argument("--mm-cond-number", type=int, default=100)
    parser.add_argument("--lbfgs-steps", type=int, default=5)
    parser.add_argument("--lbfgs-eval", type=int, default=15)
    parser.add_argument("--lbfgs-hist", type=int, default=10)
    parser.add_argument("--init-noise", type=float, default=0.25)
    parser.add_argument("--no-center-response", dest="center_response", action="store_false")
    parser.set_defaults(center_response=True)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=Path("/home/jl2815/tco/exercise_output/estimates/day"))
    parser.add_argument("--out-prefix", type=str, default="sim_vecchia_precomputed_july_st_4model_compare_051826")


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--column-above-count", type=int, default=3)
    parser.add_argument("--column-right-col-count", type=int, default=3)
    parser.add_argument("--column-per-lag-count", type=int, default=14)
    parser.add_argument("--column-lag-count", type=int, default=2)
    parser.add_argument("--column-head-right-cols", type=int, default=0)
    parser.add_argument("--column-chunk-size", type=int, default=512)

    parser.add_argument("--block-shape", type=parse_block_shape, default=(3, 3))
    parser.add_argument("--min-target-points", type=int, default=1)
    parser.add_argument("--cluster-hybrid-chunk-size", type=int, default=128)
    parser.add_argument("--cluster-hybrid-lag0-blocks", type=int, default=6)
    parser.add_argument("--cluster-hybrid-lag1-same-block", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cluster-hybrid-lag1-local-blocks", type=int, default=3)
    parser.add_argument("--cluster-hybrid-lag1-shifted-blocks", type=int, default=1)
    parser.add_argument("--cluster-hybrid-lag2-same-block", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cluster-hybrid-lag2-local-blocks", type=int, default=2)
    parser.add_argument("--cluster-hybrid-lag2-shifted-blocks", type=int, default=1)
    parser.add_argument("--cluster-hybrid-lag1-lon-offset", type=float, default=CLUSTER_HYBRID_LON_OFFSET)
    parser.add_argument("--cluster-hybrid-lag2-lon-offset", type=float, default=CLUSTER_HYBRID_LON_OFFSET)

    parser.add_argument("--cluster-column-chunk-size", type=int, default=64)
    parser.add_argument("--cluster-column-lag0-blocks", type=int, default=6)
    parser.add_argument("--cluster-column-lag1-same-block", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cluster-column-lag1-block-count", type=int, default=2)
    parser.add_argument("--cluster-column-lag2-same-block", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cluster-column-lag2-block-count", type=int, default=1)
    parser.add_argument("--cluster-column-above-blocks", type=int, default=2)
    parser.add_argument("--cluster-column-right-blocks", type=int, default=3)


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    add_model_args(parser)
    args = parser.parse_args()

    if args.require_cuda and DEVICE.type != "cuda":
        raise RuntimeError("--require-cuda was passed, but torch.cuda.is_available() is False")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = args.out_dir / f"{args.out_prefix}_raw.csv"
    model_csv = args.out_dir / f"{args.out_prefix}_model_summary.csv"
    param_csv = args.out_dir / f"{args.out_prefix}_param_summary.csv"
    error_csv = args.out_dir / f"{args.out_prefix}_errors.csv"

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print("SRC:", SRC, flush=True)
    print("DEVICE:", DEVICE, flush=True)
    print("torch:", torch.__version__, flush=True)
    print("args:", vars(args), flush=True)
    print("NOTE: pre-generated simulation assets are loaded from disk; no new simulation data are generated.", flush=True)

    asset_bank, truth = build_asset_bank(args)
    smooth = float(truth.get("smooth", 0.5))
    true_log = true_to_log_params(truth)
    print("truth:", {k: truth[k] for k in P_LABELS}, flush=True)
    print(f"Loaded {len(asset_bank)} day assets; smooth={smooth}", flush=True)

    rows: list[dict[str, Any]] = []
    ordering_cache: dict[tuple[Any, ...], tuple[torch.Tensor, np.ndarray, np.ndarray]] = {}

    for it in range(args.num_iters):
        print("\n" + "=" * 100, flush=True)
        print(f"Iteration {it + 1}/{args.num_iters}", flush=True)
        asset = asset_bank[it % len(asset_bank)]
        iter_seed = args.seed + it
        set_seed(iter_seed)
        initial_vals = make_random_init(rng, true_log, args.init_noise)
        cache_key = grid_cache_key(asset["grid_coords_np"])

        if cache_key not in ordering_cache:
            print("Computing grid maxmin ordering for asset...", flush=True)
            ord_grid, nns_grid = compute_grid_ordering(asset["grid_coords_np"], args.mm_cond_number)
            ordered_grid_coords_np = asset["grid_coords_np"][ord_grid]
            ord_grid_t = torch.as_tensor(ord_grid, device=DEVICE, dtype=torch.long)
            ordering_cache[cache_key] = (ord_grid_t, nns_grid, ordered_grid_coords_np)
        else:
            ord_grid_t, nns_grid, ordered_grid_coords_np = ordering_cache[cache_key]

        source_map_base = {k: v.to(device=DEVICE, dtype=DTYPE, non_blocking=True).contiguous() for k, v in asset["source_map"].items()}
        source_map_ord = {k: v[ord_grid_t].contiguous() for k, v in source_map_base.items()}
        print(
            f"asset=year{asset['year']} day_idx={asset['day_idx']} "
            f"n_grid={asset['n_grid']:,} valid={asset['valid_by_t']} "
            f"initial={[round(x, 4) for x in initial_vals]}",
            flush=True,
        )

        fitters = [
            ("hybrid", lambda: fit_hybrid(source_map_ord, nns_grid, initial_vals, args, truth, smooth)),
            ("cluster_hybrid", lambda: fit_cluster_hybrid(source_map_base, asset["grid_coords_np"], initial_vals, args, truth, smooth)),
            ("column_batch", lambda: fit_column(source_map_ord, ordered_grid_coords_np, initial_vals, args, truth, smooth)),
            ("cluster_column", lambda: fit_cluster_column(source_map_base, asset["grid_coords_np"], initial_vals, args, truth, smooth)),
        ]

        for fit_name, fit_fn in fitters:
            print("\n--- fitting", fit_name, "---", flush=True)
            try:
                row = fit_fn()
                row.update({
                    "iter": it + 1,
                    "seed": iter_seed,
                    "asset_year": asset["year"],
                    "asset_day_idx": asset["day_idx"],
                    "asset_first_key": asset["day_keys"][0],
                    "asset_monthly_mean": asset["monthly_mean"],
                    "error": "",
                })
                row.update({f"true_{k}": truth[k] for k in P_LABELS})
                rows.append(row)
                small = {
                    k: (round(v, 4) if isinstance(v, (float, np.floating)) else v)
                    for k, v in row.items()
                    if k in [
                        "model",
                        "kernel",
                        "loss",
                        "overall_rmsre",
                        "spatial_rmsre",
                        "advec_rmsre",
                        "nugget_re",
                        "n_heads",
                        "n_batches",
                        "n_templates",
                        "total_s",
                    ]
                }
                print("RESULT:", small, flush=True)
            except Exception as exc:
                err = {
                    "iter": it + 1,
                    "seed": iter_seed,
                    "asset_year": asset["year"],
                    "asset_day_idx": asset["day_idx"],
                    "asset_first_key": asset["day_keys"][0],
                    "model": fit_name,
                    "kernel": fit_name,
                    "model_family": "",
                    "block_shape": "",
                    "error": repr(exc),
                }
                rows.append(err)
                print("ERROR:", err, flush=True)

            save_outputs(rows, raw_csv, model_csv, param_csv, error_csv, truth)
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        del source_map_ord, source_map_base
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    print("Saved:", raw_csv, model_csv, param_csv, error_csv, flush=True)


if __name__ == "__main__":
    main()
