#!/usr/bin/env python3
"""
Density sweep for the July space-time corridor Vecchia model.

This script compares how the fitted space-time parameters change as the spatial
grid is made denser by max-min order prefixes.  It fits the fixed real-data
corridor-width 4x4 lag-643 Vecchia model at:

  data_kind in {real, sim}
  smooth in {0.3, 0.5}
  n_first in {1000, 2000, 4000, 18000}
  day_idx in July 1--10 by default

The prefix rule is spatial: for a day, the same max-min ordered grid subset is
used for all eight hourly tensors.  NaNs are left in place so the Vecchia engine
can skip missing observations in the usual way.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import pickle
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn import Parameter


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

AMAREL_SRC = Path("/home/jl2815/tco")
LOCAL_SRC = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/src")
SRC = AMAREL_SRC if AMAREL_SRC.exists() else LOCAL_SRC
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from GEMS_TCO import configuration as config
from GEMS_TCO import orderings
from GEMS_TCO.data_loader import load_data_dynamic_processed
from GEMS_TCO.vecchia_realdata_corridor_width_4x4_lag643 import (
    BLOCK_SHAPE,
    LAG_COUNTS,
    REFERENCE_ADVEC_LON_ABS,
    SPEC_NAME as VECCHIA_SPEC_NAME,
    build_model as build_corridor_width_643_model,
    model_spec as corridor_width_643_spec,
)
from GEMS_TCO.vecchia_st_spline import RealDataCorridorWidth4x4Lag643SplineFit


DTYPE = torch.float64
T_STEPS = 8
ROUND_DECIMALS = 6
DELTA_LAT_BASE = 0.044
DELTA_LON_BASE = 0.063

P_LABELS = [
    "sigmasq",
    "range_lat",
    "range_lon",
    "range_time",
    "advec_lat",
    "advec_lon",
    "nugget",
]

SPATIAL_KEYS = ["sigmasq", "range_lat", "range_lon"]
ADVECTION_KEYS = ["advec_lat", "advec_lon"]

DEFAULT_REAL_INIT_PHYSICAL = {
    "sigmasq": 13.059,
    "range_lat": 0.20,
    "range_lon": 0.25,
    "range_time": 1.50,
    "advec_lat": 0.0218,
    "advec_lon": -0.1689,
    "nugget": 0.247,
}


def parse_int_tokens(values: Iterable[str] | str) -> list[int]:
    if isinstance(values, str):
        raw = [values]
    else:
        raw = list(values)
    out: list[int] = []
    for value in raw:
        out.extend(int(part.strip()) for part in str(value).split(",") if part.strip())
    return out


def parse_float_tokens(values: Iterable[str] | str) -> list[float]:
    if isinstance(values, str):
        raw = [values]
    else:
        raw = list(values)
    out: list[float] = []
    for value in raw:
        out.extend(float(part.strip()) for part in str(value).split(",") if part.strip())
    return out


def parse_pair(text: str, cast=float) -> list[Any]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"Expected two comma-separated values, got {text!r}")
    return [cast(parts[0]), cast(parts[1])]


def parse_day_idxs(text: str) -> list[int]:
    text = str(text).strip().lower()
    if text == "all":
        return list(range(31))
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) == 2:
        start, end = int(parts[0]), int(parts[1])
        return list(range(start, end))
    return [int(p) for p in parts]


def normalize_data_kind(text: str) -> str:
    value = str(text).strip().lower()
    if value in {"real", "real_data", "observed"}:
        return "real"
    if value in {"sim", "simulation", "sim_data"}:
        return "sim"
    raise argparse.ArgumentTypeError("data kind must be real or sim")


def clean_json_value(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [clean_json_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): clean_json_value(v) for k, v in value.items()}
    return value


def default_data_root() -> Path:
    amarel = Path(config.amarel_data_load_path)
    if amarel.exists():
        return amarel
    return Path(config.mac_data_load_path)


def default_output_root() -> Path:
    if Path("/home/jl2815").exists():
        return Path("/home/jl2815/tco/exercise_output/summer/st_corridor_density_sweep_060426")
    return Path("/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer/st_corridor_density_sweep_060426")


def code_float(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def physical_to_log_phi(params: dict[str, float]) -> list[float]:
    sigmasq = float(params["sigmasq"])
    range_lat = float(params["range_lat"])
    range_lon = float(params["range_lon"])
    range_time = float(params["range_time"])
    nugget = float(params["nugget"])
    phi2 = 1.0 / range_lon
    phi1 = sigmasq * phi2
    phi3 = (range_lon / range_lat) ** 2
    phi4 = (range_lon / range_time) ** 2
    return [
        float(np.log(phi1)),
        float(np.log(phi2)),
        float(np.log(phi3)),
        float(np.log(phi4)),
        float(params["advec_lat"]),
        float(params["advec_lon"]),
        float(np.log(nugget)),
    ]


def backmap_params(out_params: Iterable[Any]) -> dict[str, float]:
    p = [float(x.detach().cpu().item() if isinstance(x, torch.Tensor) else x) for x in list(out_params)[:7]]
    phi2 = np.exp(p[1])
    phi3 = np.exp(p[2])
    phi4 = np.exp(p[3])
    range_lon = 1.0 / phi2
    return {
        "sigmasq": float(np.exp(p[0]) / phi2),
        "range_lat": float(range_lon / np.sqrt(phi3)),
        "range_lon": float(range_lon),
        "range_time": float(range_lon / np.sqrt(phi4)),
        "advec_lat": float(p[4]),
        "advec_lon": float(p[5]),
        "nugget": float(np.exp(p[6])),
    }


def make_params_list(init_physical: dict[str, float], dtype: torch.dtype, device: torch.device):
    return [
        Parameter(torch.tensor([val], dtype=dtype, device=device))
        for val in physical_to_log_phi(init_physical)
    ]


def rmsre_for_keys(est: dict[str, float], truth: dict[str, float], keys: list[str], zero_thresh: float = 0.01) -> float:
    vals = []
    for key in keys:
        tv = float(truth[key])
        if abs(tv) < zero_thresh:
            continue
        vals.append(((float(est[key]) - tv) / abs(tv)) ** 2)
    return float(np.sqrt(np.mean(vals))) if vals else float("nan")


def truth_metrics(est: dict[str, float], truth: dict[str, float] | None) -> dict[str, float]:
    if truth is None:
        return {}
    out = {
        "overall_rmsre": rmsre_for_keys(est, truth, P_LABELS),
        "spatial_rmsre": rmsre_for_keys(est, truth, SPATIAL_KEYS),
        "advec_rmsre": rmsre_for_keys(est, truth, ADVECTION_KEYS),
        "range_time_re": abs(est["range_time"] - truth["range_time"]) / abs(truth["range_time"]),
        "nugget_re": abs(est["nugget"] - truth["nugget"]) / abs(truth["nugget"]),
    }
    for key in P_LABELS:
        denom = abs(float(truth[key])) if abs(float(truth[key])) >= 0.01 else 1.0
        out[f"abs_error_{key}"] = abs(float(est[key]) - float(truth[key]))
        out[f"{key}_re"] = out[f"abs_error_{key}"] / denom
    return out


def resolve_device(args: argparse.Namespace) -> torch.device:
    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.cuda_fallback == "error":
        raise RuntimeError("CUDA is not available and --cuda-fallback=error was requested.")
    else:
        device = torch.device("cpu")
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError("--require-cuda was passed, but the selected device is not CUDA.")
    return device


def count_valid(source_map: dict[str, torch.Tensor]) -> tuple[int, int, list[int]]:
    n_valid = 0
    n_total = 0
    valid_by_t = []
    for tensor in source_map.values():
        total = int(tensor.shape[0])
        valid = int(torch.isfinite(tensor[:, 2]).sum().item())
        n_total += total
        n_valid += valid
        valid_by_t.append(valid)
    return n_valid, n_total, valid_by_t


def maxmin_order_from_grid(grid_coords_np: np.ndarray) -> np.ndarray:
    coords = np.asarray(grid_coords_np, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"grid_coords_np must be n x 2, got {coords.shape}")
    lon_lat = np.column_stack([coords[:, 1], coords[:, 0]])
    order = np.asarray(orderings.maxmin_cpp(lon_lat), dtype=np.int64)
    if order.size and order.min() == 1 and order.max() == len(coords):
        order = order - 1
    if order.size != len(coords):
        raise RuntimeError(f"max-min order length {order.size} differs from grid size {len(coords)}")
    return order


def subset_asset_by_maxmin(asset: dict[str, Any], n_first_requested: int) -> dict[str, Any]:
    grid_coords = np.asarray(asset["grid_coords_np"], dtype=np.float64)
    order = asset.get("maxmin_order")
    if order is None:
        order = maxmin_order_from_grid(grid_coords)
        asset["maxmin_order"] = order
    n_use = min(int(n_first_requested), int(len(order)))
    idx = np.asarray(order[:n_use], dtype=np.int64)
    source_map = {k: v[idx].contiguous() for k, v in asset["source_map"].items()}
    n_valid, n_total, valid_by_t = count_valid(source_map)
    out = {
        **asset,
        "source_map": source_map,
        "grid_coords_np": grid_coords[idx],
        "n_grid_full": int(grid_coords.shape[0]),
        "n_first_requested": int(n_first_requested),
        "n_first_used": int(n_use),
        "n_valid_subset": int(n_valid),
        "n_total_subset": int(n_total),
        "valid_by_t_subset": valid_by_t,
    }
    return out


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
    with path.open("rb") as f:
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


def build_sim_day_asset(
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
        "data_kind": "sim",
        "year": int(year),
        "month": 7,
        "day_idx": int(day_idx),
        "day": f"{year}-07-{day_idx + 1:02d}",
        "day_keys": [str(k) for k in day_keys],
        "source_map": source_map,
        "grid_coords_np": grid_coords_np,
        "n_grid": int(grid_coords_np.shape[0]),
        "monthly_mean": float(monthly_mean),
        "valid_by_t_full": valid_by_t,
    }


def load_sim_assets(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, float]]:
    years = parse_int_tokens(args.sim_years)
    day_idxs = parse_day_idxs(args.days)
    lat_range = parse_pair(args.lat_range, float)
    lon_range = parse_pair(args.lon_range, float)
    bank: list[dict[str, Any]] = []
    truth_ref: dict[str, float] | None = None

    for year in years:
        pickle_path, truth_path = asset_paths(args.sim_data_root, year, args.sim_pickle_kind)
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
            master = build_master_grid(sim_map[keys[0]], lat_range, lon_range)
            master_source = "generated_pickle_lat_lon"
        except Exception as exc:
            print(
                f"WARNING: could not build grid from generated pickle Latitude/Longitude ({exc!r}); "
                "falling back to native regular grid.",
                flush=True,
            )
            master = build_regular_master_grid(lat_range, lon_range)
            master_source = "native_regular_grid_fallback"

        month_vals = []
        for key in keys:
            aligned = align_to_master_grid(sim_map[key], master, lat_range, lon_range)
            vals = pd.to_numeric(aligned["ColumnAmountO3"], errors="coerce").to_numpy(dtype=float)
            month_vals.append(vals)
        monthly_mean = float(np.nanmean(np.concatenate(month_vals)))

        print(
            f"Loaded sim {pickle_path} with {len(keys)} hours; usable day blocks={max_day_idx + 1}; "
            f"grid_source={master_source}, n_grid={len(master):,}, monthly_mean={monthly_mean:.6f}",
            flush=True,
        )

        for day_idx in [d for d in day_idxs if d <= max_day_idx]:
            asset = build_sim_day_asset(
                sim_map=sim_map,
                year=year,
                day_idx=day_idx,
                lat_range=lat_range,
                lon_range=lon_range,
                center_response=args.center_response,
                master=master,
                monthly_mean=monthly_mean,
            )
            if asset is None:
                continue
            if sum(asset["valid_by_t_full"]) == 0:
                print(f"WARNING: skipping sim year={year} day_idx={day_idx}; no valid observations.", flush=True)
                continue
            asset["maxmin_order"] = maxmin_order_from_grid(asset["grid_coords_np"])
            bank.append(asset)
            print(
                f"Prepared sim asset year={year} day_idx={day_idx} "
                f"valid={asset['valid_by_t_full']}",
                flush=True,
            )

    if not bank:
        raise RuntimeError("No simulation day assets were loaded.")
    assert truth_ref is not None
    return bank, truth_ref


def load_real_assets(args: argparse.Namespace) -> list[dict[str, Any]]:
    years = parse_int_tokens(args.real_years)
    day_idxs = parse_day_idxs(args.days)
    lat_lon_resolution = [int(x) for x in parse_pair(args.space, int)]
    lat_range = parse_pair(args.lat_range, float)
    lon_range = parse_pair(args.lon_range, float)

    data_root = args.data_root or default_data_root()
    data_loader = load_data_dynamic_processed(str(data_root))
    bank: list[dict[str, Any]] = []

    for year in years:
        print("\n" + "=" * 80, flush=True)
        print(f"Loading real July data for {year}-{args.month:02d}", flush=True)
        print("=" * 80, flush=True)

        df_map, _, _, monthly_mean = data_loader.load_maxmin_ordered_data_bymonthyear(
            lat_lon_resolution=lat_lon_resolution,
            mm_cond_number=1,
            years_=[str(year)],
            months_=[int(args.month)],
            lat_range=lat_range,
            lon_range=lon_range,
            is_whittle=True,
        )
        key_idx = sorted(df_map)
        if not key_idx:
            raise RuntimeError(f"No real data loaded for {year}-{args.month:02d} from {data_root}")

        base_grid_coords_np = df_map[key_idx[0]][["Latitude", "Longitude"]].to_numpy(dtype=np.float64)
        max_order = maxmin_order_from_grid(base_grid_coords_np)
        print(
            f"Loaded real year={year}; hours={len(key_idx)}, n_grid={len(base_grid_coords_np):,}, "
            f"monthly_mean={monthly_mean:.6f}",
            flush=True,
        )

        for day_idx in day_idxs:
            day_keys = key_idx[day_idx * T_STEPS : (day_idx + 1) * T_STEPS]
            if len(day_keys) < T_STEPS:
                print(f"Skipping real year={year} day_idx={day_idx}; found only {len(day_keys)} slots.", flush=True)
                continue
            day_map_cpu, _ = data_loader.load_working_data(
                df_map,
                monthly_mean,
                [day_idx * T_STEPS, (day_idx + 1) * T_STEPS],
                ord_mm=None,
                dtype=DTYPE,
                keep_ori=args.keep_exact_loc,
            )
            n_valid, _, valid_by_t = count_valid(day_map_cpu)
            if n_valid == 0:
                print(f"WARNING: skipping real year={year} day_idx={day_idx}; no valid observations.", flush=True)
                continue
            asset = {
                "data_kind": "real",
                "year": int(year),
                "month": int(args.month),
                "day_idx": int(day_idx),
                "day": f"{year}-{args.month:02d}-{day_idx + 1:02d}",
                "day_keys": [str(k) for k in day_keys],
                "source_map": day_map_cpu,
                "grid_coords_np": base_grid_coords_np,
                "n_grid": int(base_grid_coords_np.shape[0]),
                "monthly_mean": float(monthly_mean),
                "valid_by_t_full": valid_by_t,
                "maxmin_order": max_order,
            }
            bank.append(asset)
            print(f"Prepared real asset year={year} day_idx={day_idx} valid={valid_by_t}", flush=True)

    if not bank:
        raise RuntimeError("No real day assets were loaded.")
    return bank


def existing_completed(csv_path: Path) -> set[tuple[str, int, int, float, int]]:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    if df.empty or "status" not in df.columns:
        return set()
    ok = df[df["status"] == "ok"].copy()
    out: set[tuple[str, int, int, float, int]] = set()
    for row in ok.itertuples(index=False):
        out.add(
            (
                str(getattr(row, "data_kind")),
                int(getattr(row, "year")),
                int(getattr(row, "day_idx")),
                round(float(getattr(row, "smooth")), 6),
                int(getattr(row, "n_first_requested")),
            )
        )
    return out


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(clean_json_value(row), sort_keys=True) + "\n")


def save_rows(csv_path: Path, rows: list[dict[str, Any]], decimals: int = ROUND_DECIMALS) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(decimals)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, float_format=f"%.{decimals}f")
    return df


def make_param_summary(ok: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for param in P_LABELS:
        col = f"est_{param}"
        if col not in ok.columns:
            continue
        group_cols = ["data_kind", "smooth", "n_first_requested", "n_first_used"]
        for keys, sub in ok.groupby(group_cols, dropna=False):
            vals = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(vals) == 0:
                continue
            rows.append(
                {
                    "data_kind": keys[0],
                    "smooth": float(keys[1]),
                    "n_first_requested": int(keys[2]),
                    "n_first_used": int(keys[3]),
                    "parameter": param,
                    "n": int(len(vals)),
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "p10": float(np.quantile(vals, 0.10)),
                    "p90": float(np.quantile(vals, 0.90)),
                    "p90_p10": float(np.quantile(vals, 0.90) - np.quantile(vals, 0.10)),
                }
            )
    return pd.DataFrame(rows)


def make_shift_summary(ok: pd.DataFrame) -> pd.DataFrame:
    rows = []
    id_cols = ["data_kind", "smooth", "year", "day_idx"]
    for param in P_LABELS:
        col = f"est_{param}"
        if col not in ok.columns:
            continue
        for keys, sub in ok.groupby(id_cols, dropna=False):
            sub = sub.sort_values("n_first_requested")
            if sub.empty:
                continue
            baseline_row = sub.iloc[sub["n_first_requested"].to_numpy(dtype=float).argmax()]
            baseline = float(baseline_row[col])
            for _, row in sub.iterrows():
                est = float(row[col])
                abs_shift = abs(est - baseline)
                rel_shift = abs_shift / max(abs(baseline), 1e-8)
                rows.append(
                    {
                        "data_kind": keys[0],
                        "smooth": float(keys[1]),
                        "year": int(keys[2]),
                        "day_idx": int(keys[3]),
                        "n_first_requested": int(row["n_first_requested"]),
                        "n_first_used": int(row["n_first_used"]),
                        "baseline_n_first_requested": int(baseline_row["n_first_requested"]),
                        "parameter": param,
                        "estimate": est,
                        "baseline_estimate": baseline,
                        "abs_shift_vs_baseline": float(abs_shift),
                        "rel_shift_vs_baseline": float(rel_shift),
                    }
                )
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame(rows)
    group_cols = ["data_kind", "smooth", "n_first_requested", "n_first_used", "parameter"]
    summary_rows = []
    for keys, sub in raw.groupby(group_cols, dropna=False):
        abs_vals = sub["abs_shift_vs_baseline"].to_numpy(dtype=float)
        rel_vals = sub["rel_shift_vs_baseline"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "data_kind": keys[0],
                "smooth": float(keys[1]),
                "n_first_requested": int(keys[2]),
                "n_first_used": int(keys[3]),
                "parameter": keys[4],
                "n": int(len(sub)),
                "median_abs_shift_vs_baseline": float(np.median(abs_vals)),
                "p90_abs_shift_vs_baseline": float(np.quantile(abs_vals, 0.90)),
                "median_rel_shift_vs_baseline": float(np.median(rel_vals)),
                "p90_rel_shift_vs_baseline": float(np.quantile(rel_vals, 0.90)),
            }
        )
    return pd.DataFrame(summary_rows)


def plot_parameter_summary(param_summary: pd.DataFrame, path: Path) -> None:
    if param_summary.empty:
        return
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
    axes_flat = axes.ravel()
    styles = {
        ("real", 0.3): ("tab:blue", "o", "-"),
        ("real", 0.5): ("tab:cyan", "s", "-"),
        ("sim", 0.3): ("tab:red", "o", "--"),
        ("sim", 0.5): ("tab:orange", "s", "--"),
    }
    for ax, param in zip(axes_flat, P_LABELS):
        sub_param = param_summary[param_summary["parameter"] == param].copy()
        for (data_kind, smooth), sub in sub_param.groupby(["data_kind", "smooth"], dropna=False):
            sub = sub.sort_values("n_first_requested")
            color, marker, ls = styles.get((str(data_kind), round(float(smooth), 1)), ("0.25", "o", "-"))
            label = f"{data_kind}, smooth={float(smooth):.1f}"
            ax.plot(
                sub["n_first_requested"],
                sub["median"],
                marker=marker,
                linestyle=ls,
                linewidth=1.8,
                color=color,
                label=label,
            )
            ax.fill_between(
                sub["n_first_requested"].to_numpy(dtype=float),
                sub["p10"].to_numpy(dtype=float),
                sub["p90"].to_numpy(dtype=float),
                color=color,
                alpha=0.12,
                linewidth=0,
            )
        ax.set_title(param)
        ax.set_xscale("log")
        ax.grid(alpha=0.25)
    for ax in axes_flat[len(P_LABELS) :]:
        ax.axis("off")
    axes_flat[0].legend(fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("first N grid points in max-min order")
    fig.suptitle("ST corridor Vecchia parameter medians by max-min prefix")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_shift_summary(shift_summary: pd.DataFrame, path: Path) -> None:
    if shift_summary.empty:
        return
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
    axes_flat = axes.ravel()
    styles = {
        ("real", 0.3): ("tab:blue", "o", "-"),
        ("real", 0.5): ("tab:cyan", "s", "-"),
        ("sim", 0.3): ("tab:red", "o", "--"),
        ("sim", 0.5): ("tab:orange", "s", "--"),
    }
    for ax, param in zip(axes_flat, P_LABELS):
        sub_param = shift_summary[shift_summary["parameter"] == param].copy()
        for (data_kind, smooth), sub in sub_param.groupby(["data_kind", "smooth"], dropna=False):
            sub = sub.sort_values("n_first_requested")
            color, marker, ls = styles.get((str(data_kind), round(float(smooth), 1)), ("0.25", "o", "-"))
            ax.plot(
                sub["n_first_requested"],
                sub["median_rel_shift_vs_baseline"],
                marker=marker,
                linestyle=ls,
                linewidth=1.8,
                color=color,
                label=f"{data_kind}, smooth={float(smooth):.1f}",
            )
        ax.axhline(0.0, color="0.6", linewidth=0.8)
        ax.set_title(param)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.grid(alpha=0.25)
    for ax in axes_flat[len(P_LABELS) :]:
        ax.axis("off")
    axes_flat[0].legend(fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("first N grid points in max-min order")
    fig.suptitle("Median relative parameter shift vs densest prefix")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_running_summary(path: Path, df: pd.DataFrame, param_summary: pd.DataFrame, shift_summary: pd.DataFrame) -> None:
    completed = int((df.get("status", pd.Series(dtype=str)) == "ok").sum()) if len(df) else 0
    errors = int((df.get("status", pd.Series(dtype=str)) == "error").sum()) if len(df) else 0
    lines = [
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        f"Rows: {len(df)} completed: {completed} errors: {errors}",
        "",
    ]
    if completed:
        ok = df[df["status"] == "ok"].copy()
        show_cols = [
            "data_kind",
            "smooth",
            "year",
            "day_idx",
            "n_first_requested",
            "loss",
            "fit_s",
            "est_sigmasq",
            "est_range_lat",
            "est_range_lon",
            "est_range_time",
            "est_advec_lat",
            "est_advec_lon",
            "est_nugget",
        ]
        show_cols = [c for c in show_cols if c in ok.columns]
        lines.append("Latest completed fits:")
        lines.append(ok[show_cols].tail(12).to_string(index=False))
        lines.append("")

        if not param_summary.empty:
            lines.append("Parameter median preview:")
            preview = param_summary.sort_values(["data_kind", "smooth", "parameter", "n_first_requested"])
            lines.append(preview.head(28).round(6).to_string(index=False))
            lines.append("")

        if not shift_summary.empty:
            lines.append("Largest median relative shifts vs densest prefix:")
            cols = [
                "data_kind",
                "smooth",
                "n_first_requested",
                "parameter",
                "median_rel_shift_vs_baseline",
                "p90_rel_shift_vs_baseline",
                "n",
            ]
            top = shift_summary.sort_values("median_rel_shift_vs_baseline", ascending=False)[cols].head(20)
            lines.append(top.round(6).to_string(index=False))
            lines.append("")

    if errors:
        err_cols = ["data_kind", "smooth", "year", "day_idx", "n_first_requested", "error"]
        err_cols = [c for c in err_cols if c in df.columns]
        lines.append("Recent errors:")
        lines.append(df[df["status"] == "error"][err_cols].tail(10).to_string(index=False))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_external_monthly_outputs(monthly_out_dir: Path | None, out_dir: Path) -> None:
    if monthly_out_dir is None:
        return
    monthly_out_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "st_corridor_density_sweep_all_fits.csv",
        "parameter_by_nfirst_summary.csv",
        "parameter_shift_vs_densest_summary.csv",
        "parameter_by_nfirst.png",
        "parameter_shift_vs_densest.png",
        "running_summary.txt",
        "run_config.json",
        "simulation_truth.json",
    ]:
        src = out_dir / name
        if not src.exists():
            continue
        dst = monthly_out_dir / name
        if src.suffix.lower() == ".png":
            dst.write_bytes(src.read_bytes())
        else:
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def refresh_outputs(out_dir: Path, rows: list[dict[str, Any]], monthly_out_dir: Path | None = None) -> None:
    raw_csv = out_dir / "st_corridor_density_sweep_all_fits.csv"
    df = save_rows(raw_csv, rows)
    if df.empty:
        return

    for data_kind, sub in df.groupby("data_kind", dropna=False):
        kind_dir = out_dir / str(data_kind)
        kind_dir.mkdir(parents=True, exist_ok=True)
        sub.to_csv(kind_dir / "all_fits.csv", index=False)

    ok = df[df["status"] == "ok"].copy() if "status" in df.columns else pd.DataFrame()
    param_summary = make_param_summary(ok) if not ok.empty else pd.DataFrame()
    shift_summary = make_shift_summary(ok) if not ok.empty else pd.DataFrame()

    if not param_summary.empty:
        save_rows(out_dir / "parameter_by_nfirst_summary.csv", param_summary)
    if not shift_summary.empty:
        save_rows(out_dir / "parameter_shift_vs_densest_summary.csv", shift_summary)

    plot_parameter_summary(param_summary, out_dir / "parameter_by_nfirst.png")
    plot_shift_summary(shift_summary, out_dir / "parameter_shift_vs_densest.png")
    write_running_summary(out_dir / "running_summary.txt", df, param_summary, shift_summary)
    write_external_monthly_outputs(monthly_out_dir, out_dir)


def fit_one(
    asset: dict[str, Any],
    smooth: float,
    n_first: int,
    init_physical: dict[str, float],
    truth: dict[str, float] | None,
    reference_advec_lon_abs: float,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    subset = subset_asset_by_maxmin(asset, n_first)
    source_map = {
        k: v.to(device=device, dtype=DTYPE, non_blocking=True).contiguous()
        for k, v in subset["source_map"].items()
    }
    grid_coords_np = np.asarray(subset["grid_coords_np"], dtype=np.float64)
    n_valid, n_total, valid_by_t = count_valid(source_map)

    params_list = make_params_list(init_physical, dtype=DTYPE, device=device)
    model_spec = corridor_width_643_spec(reference_advec_lon_abs)
    model_spec["smooth_kernel"] = str(args.smooth_kernel)
    if str(args.smooth_kernel) == "spline":
        model = RealDataCorridorWidth4x4Lag643SplineFit(
            smooth=smooth,
            input_map=source_map,
            grid_coords=grid_coords_np,
            lag1_lon_offset=reference_advec_lon_abs,
            daily_stride=args.daily_stride,
            target_chunk_size=args.target_chunk_size,
            min_target_points=args.min_target_points,
            spline_n_points=args.spline_n_points,
            spline_r_max=args.spline_r_max,
        )
    else:
        model = build_corridor_width_643_model(
            smooth=smooth,
            input_map=source_map,
            grid_coords=grid_coords_np,
            device=None,
            reference_advec_lon_abs=reference_advec_lon_abs,
            daily_stride=args.daily_stride,
            target_chunk_size=args.target_chunk_size,
            min_target_points=args.min_target_points,
        )

    t0 = time.time()
    model.precompute_conditioning_sets()
    precompute_s = time.time() - t0

    optimizer = model.set_optimizer(
        params_list,
        lr=args.lbfgs_lr,
        max_iter=args.lbfgs_eval,
        max_eval=args.lbfgs_eval,
        history_size=args.lbfgs_history,
    )

    t1 = time.time()
    if args.suppress_fit_prints:
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            out, steps_ran = model.fit_vecc_lbfgs(params_list, optimizer, max_steps=args.lbfgs_steps, grad_tol=args.grad_tol)
    else:
        out, steps_ran = model.fit_vecc_lbfgs(params_list, optimizer, max_steps=args.lbfgs_steps, grad_tol=args.grad_tol)
    fit_s = time.time() - t1

    est = backmap_params(out)
    cluster_summary = model.cluster_summary()
    row = {
        "status": "ok",
        "error": "",
        "data_kind": subset["data_kind"],
        "year": int(subset["year"]),
        "month": int(subset["month"]),
        "day_idx": int(subset["day_idx"]),
        "day": subset["day"],
        "smooth": float(smooth),
        "n_first_requested": int(n_first),
        "n_first_used": int(subset["n_first_used"]),
        "n_grid_full": int(subset["n_grid_full"]),
        "n_time_slots": int(len(source_map)),
        "n_rows_total": int(n_total),
        "n_valid_o3": int(n_valid),
        "valid_rate": float(n_valid / n_total) if n_total else np.nan,
        "valid_by_t": json.dumps(valid_by_t, separators=(",", ":")),
        "monthly_mean": float(subset["monthly_mean"]),
        "first_slot": subset["day_keys"][0] if subset.get("day_keys") else "",
        "last_slot": subset["day_keys"][-1] if subset.get("day_keys") else "",
        "spec_name": VECCHIA_SPEC_NAME,
        "smooth_kernel": str(args.smooth_kernel),
        "spline_n_points": int(args.spline_n_points) if str(args.smooth_kernel) == "spline" else 0,
        "spline_r_max": float(args.spline_r_max) if str(args.smooth_kernel) == "spline" else np.nan,
        "block_shape": f"{BLOCK_SHAPE[0]}x{BLOCK_SHAPE[1]}",
        "lag_pattern": f"{LAG_COUNTS[0]}/{LAG_COUNTS[1]}/{LAG_COUNTS[2]}",
        "reference_advec_lon_abs": float(reference_advec_lon_abs),
        "loss": float(out[-1]),
        "steps_raw": int(steps_ran),
        "precompute_s": float(precompute_s),
        "fit_s": float(fit_s),
        "total_s": float(precompute_s + fit_s),
        **{f"est_{k}": float(est[k]) for k in P_LABELS},
        **truth_metrics(est, truth),
        **cluster_summary,
        "model_spec": json.dumps(clean_json_value(model_spec), sort_keys=True),
    }

    del model, params_list, optimizer, source_map
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit ST corridor Vecchia density sweep on real and simulation July data.")
    parser.add_argument("--data-kinds", nargs="+", default=["real", "sim"], type=normalize_data_kind)
    parser.add_argument("--smooths", nargs="+", default=["0.3", "0.5"])
    parser.add_argument("--n-first-values", nargs="+", default=["1000", "2000", "4000", "18000"])
    parser.add_argument("--days", default="0,10", help="'0,10' means July day_idx 0..9, i.e. July 1--10.")
    parser.add_argument("--real-years", nargs="+", default=["2024"])
    parser.add_argument("--sim-years", nargs="+", default=["2024"])
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--space", default="1,1")
    parser.add_argument("--lat-range", default="-3,2")
    parser.add_argument("--lon-range", default="121,131")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--sim-data-root", type=Path, default=Path("/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern"))
    parser.add_argument("--sim-pickle-kind", default="real_locations", choices=["real_locations", "gridded"])
    parser.add_argument("--smooth-kernel", choices=["spline", "closed"], default="spline",
                        help="Use spline ST Matern for arbitrary smooth values such as 0.3.")
    parser.add_argument("--spline-n-points", type=int, default=4000)
    parser.add_argument("--spline-r-max", type=float, default=30.0)
    parser.add_argument("--real-reference-advec-lon-abs", type=float, default=REFERENCE_ADVEC_LON_ABS)
    parser.add_argument("--sim-reference-advec-lon-abs", type=float, default=0.2)
    parser.add_argument("--daily-stride", type=int, default=2)
    parser.add_argument("--target-chunk-size", type=int, default=128)
    parser.add_argument("--min-target-points", type=int, default=1)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-steps", type=int, default=5)
    parser.add_argument("--lbfgs-eval", type=int, default=20)
    parser.add_argument("--lbfgs-history", type=int, default=10)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cuda-fallback", choices=["cpu", "error"], default="cpu")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--keep-exact-loc", dest="keep_exact_loc", action="store_true", default=True)
    parser.add_argument("--no-keep-exact-loc", dest="keep_exact_loc", action="store_false")
    parser.add_argument("--center-response", dest="center_response", action="store_true", default=True)
    parser.add_argument("--no-center-response", dest="center_response", action="store_false")
    parser.add_argument("--sim-init", choices=["truth", "real_default"], default="truth")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summary-every", type=int, default=1)
    parser.add_argument("--round-decimals", type=int, default=ROUND_DECIMALS)
    parser.add_argument("--suppress-fit-prints", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--monthly-out-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = resolve_device(args)
    smooths = parse_float_tokens(args.smooths)
    n_first_values = parse_int_tokens(args.n_first_values)
    out_dir = args.out_dir or default_output_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    monthly_out_dir = args.monthly_out_dir
    if monthly_out_dir is not None:
        monthly_out_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "run_config.json"
    jsonl_path = out_dir / "st_corridor_density_sweep_all_fits.jsonl"
    raw_csv = out_dir / "st_corridor_density_sweep_all_fits.csv"

    print("SRC:", SRC, flush=True)
    print("device:", device, flush=True)
    print("torch:", torch.__version__, flush=True)
    print("out_dir:", out_dir, flush=True)
    print("data_kinds:", args.data_kinds, flush=True)
    print("smooths:", smooths, flush=True)
    print("n_first_values:", n_first_values, flush=True)
    print("days:", parse_day_idxs(args.days), flush=True)

    run_config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "src": str(SRC),
        "device": str(device),
        "dtype": str(DTYPE),
        "args": clean_json_value(vars(args)),
        "smooths": smooths,
        "n_first_values": n_first_values,
        "monthly_out_dir": str(monthly_out_dir) if monthly_out_dir is not None else "",
        "model_spec_real": corridor_width_643_spec(args.real_reference_advec_lon_abs),
        "model_spec_sim": corridor_width_643_spec(args.sim_reference_advec_lon_abs),
        "default_real_init_physical": DEFAULT_REAL_INIT_PHYSICAL,
    }
    config_path.write_text(json.dumps(clean_json_value(run_config), indent=2, sort_keys=True), encoding="utf-8")

    assets_by_kind: dict[str, list[dict[str, Any]]] = {}
    truth_by_kind: dict[str, dict[str, float] | None] = {}
    init_by_kind: dict[str, dict[str, float]] = {}
    reference_by_kind = {
        "real": float(args.real_reference_advec_lon_abs),
        "sim": float(args.sim_reference_advec_lon_abs),
    }

    if "real" in args.data_kinds:
        assets_by_kind["real"] = load_real_assets(args)
        truth_by_kind["real"] = None
        init_by_kind["real"] = DEFAULT_REAL_INIT_PHYSICAL

    if "sim" in args.data_kinds:
        sim_assets, sim_truth = load_sim_assets(args)
        assets_by_kind["sim"] = sim_assets
        truth_by_kind["sim"] = sim_truth
        if args.sim_init == "truth":
            init_by_kind["sim"] = {k: float(sim_truth[k]) for k in P_LABELS}
        else:
            init_by_kind["sim"] = DEFAULT_REAL_INIT_PHYSICAL
        (out_dir / "simulation_truth.json").write_text(json.dumps(sim_truth, indent=2, sort_keys=True), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    if args.skip_existing and raw_csv.exists():
        existing = pd.read_csv(raw_csv)
        rows = existing.to_dict(orient="records")
    done = existing_completed(raw_csv) if args.skip_existing else set()

    fit_id = 0
    if rows:
        fit_ids = pd.to_numeric(pd.DataFrame(rows).get("fit_id", pd.Series(dtype=float)), errors="coerce").dropna()
        if not fit_ids.empty:
            fit_id = int(fit_ids.max())

    for data_kind in args.data_kinds:
        for smooth in smooths:
            for asset in assets_by_kind[data_kind]:
                for n_first in n_first_values:
                    key = (
                        data_kind,
                        int(asset["year"]),
                        int(asset["day_idx"]),
                        round(float(smooth), 6),
                        int(n_first),
                    )
                    if args.skip_existing and key in done:
                        print(f"Skipping existing ok fit: {key}", flush=True)
                        continue

                    fit_id += 1
                    print("\n" + "-" * 96, flush=True)
                    print(
                        f"fit_id={fit_id} data={data_kind} smooth={smooth} "
                        f"day={asset['day']} n_first={n_first}",
                        flush=True,
                    )
                    print("-" * 96, flush=True)

                    base = {
                        "fit_id": int(fit_id),
                        "data_kind": data_kind,
                        "year": int(asset["year"]),
                        "month": int(asset["month"]),
                        "day_idx": int(asset["day_idx"]),
                        "day": asset["day"],
                        "smooth": float(smooth),
                        "n_first_requested": int(n_first),
                        "spec_name": VECCHIA_SPEC_NAME,
                        "block_shape": f"{BLOCK_SHAPE[0]}x{BLOCK_SHAPE[1]}",
                        "lag_pattern": f"{LAG_COUNTS[0]}/{LAG_COUNTS[1]}/{LAG_COUNTS[2]}",
                    }
                    try:
                        row = fit_one(
                            asset=asset,
                            smooth=float(smooth),
                            n_first=int(n_first),
                            init_physical=init_by_kind[data_kind],
                            truth=truth_by_kind[data_kind],
                            reference_advec_lon_abs=reference_by_kind[data_kind],
                            device=device,
                            args=args,
                        )
                        row.update(base)
                        print(
                            pd.Series(
                                {
                                    k: row.get(k)
                                    for k in [
                                        "data_kind",
                                        "smooth",
                                        "day",
                                        "n_first_requested",
                                        "n_first_used",
                                        "loss",
                                        "fit_s",
                                        "est_sigmasq",
                                        "est_range_lat",
                                        "est_range_lon",
                                        "est_range_time",
                                        "est_advec_lat",
                                        "est_advec_lon",
                                        "est_nugget",
                                    ]
                                }
                            ).to_string(),
                            flush=True,
                        )
                    except Exception as exc:
                        row = {
                            **base,
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc(limit=10),
                        }
                        print(f"ERROR: {row['error']}", flush=True)
                        traceback.print_exc()

                    rows.append(clean_json_value(row))
                    append_jsonl(jsonl_path, row)
                    if int(args.summary_every) > 0 and fit_id % int(args.summary_every) == 0:
                        refresh_outputs(out_dir, rows, monthly_out_dir)
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

    refresh_outputs(out_dir, rows, monthly_out_dir)
    print("\nDone.", flush=True)
    print("csv:", raw_csv, flush=True)
    print("summary:", out_dir / "running_summary.txt", flush=True)
    print("plots:", out_dir / "parameter_by_nfirst.png", out_dir / "parameter_shift_vs_densest.png", flush=True)


if __name__ == "__main__":
    main()
