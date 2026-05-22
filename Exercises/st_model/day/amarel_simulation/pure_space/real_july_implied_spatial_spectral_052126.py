#!/usr/bin/env python3
"""Real July implied-spatial spectral diagnostics from daily ST Vecchia fits.

What this script does
---------------------
This is the spatio-temporal counterpart of the pure-space daily spectrum plot.
For each smooth/year/day/resolution/variant it fits one ST Vecchia model to the
8 hourly fields in that day, then checks the spatial spectrum implied by the
fitted ST model at temporal lag zero.

The main loop is intentionally:

    for smooth
      for year
        for day in July 1..30
          for resolution x8, x4, x2, x1
            1. full grid에서 fixed thinned grid order 생성
            2. 8 hourly fields를 그 fixed order에 reindex
            3. 하루 8시간을 함께 넣어서 ST Vecchia model 하나 fit
            4. fitted ST model에서 temporal lag=0 slice를 뽑아 implied spatial model 구성
            5. hour별 I(w), hour별 E[I(w)] 계산
            6. plot: gray hourly I, black mean I, red mean E[I], blue ratio
        monthly average plot

Fixed thinned grid order
------------------------
For x8/x4/x2/x1 we first define a fixed downsampled spatial grid from the full
regular grid:

    x8: keep row % 8 == 0 and col % 8 == 0
    x4: keep row % 4 == 0 and col % 4 == 0
    x2: keep row % 2 == 0 and col % 2 == 0
    x1: keep every grid cell

Every hourly tensor is then reindexed to that same fixed thinned grid order.
If an hourly observation is missing at a retained grid cell, we keep the row and
put y=NaN instead of dropping the row.  This matters because the ST cluster
Vecchia code assumes that input_map["hour1"][i], input_map["hour2"][i], and
grid_coords[i] all mean the same spatial location.

Plot layout
-----------
The plot keeps the earlier format:

    columns: x8, x4, x2, x1
    row 1:  nugget fixed 0 ST model
    row 2:  nugget free ST model

Within each panel:

    gray  = 8 hourly spatial residual periodograms I_h(w)
    black = mean_h I_h(w)
    red   = mean_h E_h[I(w)] from the fitted ST model's temporal-lag-zero
            implied spatial covariance
    blue  = mean_h I_h(w) / mean_h E_h[I(w)]

The red curve is not a continuous Matérn spectrum.  It is the finite-sample
expected periodogram after applying the hour-specific mask, taper, and mean
projection, then averaged over the 8 hours.
"""

from __future__ import annotations

import argparse
import gc
import math
import pickle
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.special import gamma, kv


LOCAL_SRC = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/src")
AMAREL_SRC = Path("/home/jl2815/tco")
for candidate in (AMAREL_SRC, LOCAL_SRC):
    if (candidate / "GEMS_TCO").is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
        break

from GEMS_TCO import configuration as config
from GEMS_TCO.vecchia_st_spline import (
    ClusterHybridVecchiaNoNuggetSplineFit,
    ClusterHybridVecchiaSplineFit,
)


DTYPE = torch.float64
EPS = 1e-12
ROUND_DECIMALS = 6


VARIANTS = {
    "nugget0": {
        "class": ClusterHybridVecchiaNoNuggetSplineFit,
        "n_params": 6,
        "row_title": "ST: nugget fixed 0",
    },
    "nugget_free": {
        "class": ClusterHybridVecchiaSplineFit,
        "n_params": 7,
        "row_title": "ST: nugget free",
    },
}


def normalize_range_argv(argv: list[str]) -> list[str]:
    out = []
    i = 0
    range_options = {"--lat-range", "--lon-range"}
    while i < len(argv):
        item = argv[i]
        if item in range_options and i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            out.append(f"{item}={argv[i + 1]}")
            i += 2
            continue
        out.append(item)
        i += 1
    return out


@dataclass
class MonthContext:
    year: int
    month: int
    entries_by_day: dict[int, list[tuple[pd.Timestamp | None, str, pd.DataFrame]]]
    lat_vals: np.ndarray
    lon_vals: np.ndarray
    grid_index: pd.MultiIndex
    local_to_row: np.ndarray
    local_to_col: np.ndarray
    grid_coords_full: np.ndarray
    radial_bins: np.ndarray
    k_full_radial: np.ndarray
    omega_y_full: np.ndarray
    omega_x_full: np.ndarray
    k_max_full: float
    lat_step: float
    lon_step: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily ST implied-spatial spectral diagnostics.")
    p.add_argument("--years", default="2022,2023,2024,2025")
    p.add_argument("--month", type=int, default=7)
    p.add_argument("--days", default="1,30")
    p.add_argument("--smooths", default="0.2,0.25,0.3,0.35,0.4,0.45")
    p.add_argument("--resolutions", default="8,4,2,1")
    p.add_argument("--variants", default="nugget0,nugget_free")
    p.add_argument("--data-root", default=getattr(config, "amarel_data_load_path", "/home/jl2815/tco/data/"))
    p.add_argument("--output-root", default="/home/jl2815/tco/exercise_output/real_data/implied_spatial_spectral_052126")
    p.add_argument("--expanded-bounds", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lat-range", default="-3,7")
    p.add_argument("--lon-range", default="111,131")
    p.add_argument("--x-col", default="Longitude")
    p.add_argument("--y-col", default="Latitude")
    p.add_argument("--source-x-col", default="Source_Longitude")
    p.add_argument("--source-y-col", default="Source_Latitude")
    p.add_argument("--value-col", default="ColumnAmountO3")
    p.add_argument("--time-col", default="Hours_elapsed")
    p.add_argument("--time-origin", type=float, default=477700.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--cuda-fallback", default="cpu", choices=["cpu", "error"])
    p.add_argument("--block-shape", default="3,3")
    p.add_argument("--n-neighbor-blocks-t", type=int, default=6)
    p.add_argument("--lag1-local-blocks", type=int, default=3)
    p.add_argument("--lag1-shifted-blocks", type=int, default=1)
    p.add_argument("--lag2-local-blocks", type=int, default=2)
    p.add_argument("--lag2-shifted-blocks", type=int, default=1)
    p.add_argument("--daily-stride", type=int, default=2)
    p.add_argument("--lag1-lon-offset", type=float, default=0.063)
    p.add_argument("--lag2-lon-offset", type=float, default=None)
    p.add_argument("--target-chunk-size", type=int, default=128)
    p.add_argument("--min-target-points", type=int, default=1)
    p.add_argument("--lbfgs-steps", type=int, default=8)
    p.add_argument("--lbfgs-eval", type=int, default=20)
    p.add_argument("--lbfgs-history", type=int, default=10)
    p.add_argument("--grad-tol", type=float, default=1e-5)
    p.add_argument("--sigmasq-init", type=float, default=13.0)
    p.add_argument("--range-lat-init", type=float, default=0.25)
    p.add_argument("--range-lon-init", type=float, default=0.25)
    p.add_argument("--range-time-init", type=float, default=2.0)
    p.add_argument("--advec-lat-init", type=float, default=0.0)
    p.add_argument("--advec-lon-init", type=float, default=0.0)
    p.add_argument("--nugget-init", type=float, default=2.5)
    p.add_argument("--radial-bins", type=int, default=70)
    p.add_argument("--radial-qmax", type=float, default=0.985)
    p.add_argument("--hann", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--make-monthly-only", action="store_true")
    return p.parse_args(normalize_range_argv(sys.argv[1:]))


def parse_int_list_or_range(text: str) -> list[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(vals) == 2 and vals[1] >= vals[0]:
        return list(range(vals[0], vals[1] + 1))
    return vals


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_names(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_range_pair(text: str) -> tuple[float, float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(vals) != 2:
        raise ValueError(f"Expected two comma-separated values, got {text!r}")
    return vals[0], vals[1]


def parse_shape_pair(text: str) -> tuple[int, int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(vals) != 2:
        raise ValueError(f"Expected block shape 'n_lat,n_lon', got {text!r}")
    return int(vals[0]), int(vals[1])


def smooth_tag(smooth: float) -> str:
    return f"{float(smooth):.6g}".replace(".", "p").replace("-", "m")


def month_file_token(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def expanded_grid_filename(year: int, month: int, lat_range, lon_range) -> str:
    yy = str(year)[2:]
    return (
        f"tco_grid_lat{month_file_token(lat_range[0])}to{month_file_token(lat_range[1])}_"
        f"lon{month_file_token(lon_range[0])}to{month_file_token(lon_range[1])}_{yy}_{month:02d}.pkl"
    )


def default_grid_filename(year: int, month: int) -> str:
    return f"tco_grid_{str(year)[2:]}_{month:02d}.pkl"


def parse_gems_key(key: str) -> pd.Timestamp | None:
    pat = r"y(?P<yy>\d{2})m(?P<mm>\d{2})day(?P<dd>\d{2})_hm(?P<hh>\d{2}):(?P<minute>\d{2})"
    m = re.search(pat, str(key))
    if not m:
        return None
    parts = {k: int(v) for k, v in m.groupdict().items()}
    return pd.Timestamp(
        year=2000 + parts["yy"],
        month=parts["mm"],
        day=parts["dd"],
        hour=parts["hh"],
        minute=parts["minute"],
        tz="UTC",
    )


def select_device(device_arg: str, fallback: str) -> torch.device:
    requested = str(device_arg).lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested not in ("auto", "cuda"):
        return torch.device(device_arg)
    try:
        if torch.cuda.is_available():
            torch.empty(1, device="cuda")
            print(
                f"CUDA ready: count={torch.cuda.device_count()}, "
                f"name={torch.cuda.get_device_name(torch.cuda.current_device())}",
                flush=True,
            )
            return torch.device("cuda")
    except Exception as exc:
        if fallback == "cpu":
            print(f"WARNING: CUDA init failed, falling back to CPU: {exc}", flush=True)
            return torch.device("cpu")
        raise
    if requested == "cuda" and fallback == "error":
        raise RuntimeError("CUDA requested but unavailable.")
    print("CUDA not available; using CPU.", flush=True)
    return torch.device("cpu")


def order_frame(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    return (
        df.copy()
        .assign(_orig_idx=np.arange(len(df)))
        .sort_values([x_col, y_col, "_orig_idx"], kind="mergesort")
        .drop(columns=["_orig_idx"])
        .reset_index(drop=True)
    )


def load_month_pickle(args: argparse.Namespace, year: int, lat_range, lon_range):
    data_root = Path(args.data_root)
    month_dir = data_root / f"pickle_{year}"
    candidates = []
    if args.expanded_bounds:
        candidates.append(month_dir / expanded_grid_filename(year, args.month, lat_range, lon_range))
    candidates.append(month_dir / default_grid_filename(year, args.month))
    for path in candidates:
        if path.exists():
            with open(path, "rb") as f:
                obj = pickle.load(f)
            if not isinstance(obj, dict):
                raise TypeError(f"Expected dict pickle at {path}, got {type(obj)}")
            print(f"Loaded {path}", flush=True)
            return obj, path
    raise FileNotFoundError("Could not find monthly pickle. Checked:\n" + "\n".join(str(p) for p in candidates))


def filter_bounds(df: pd.DataFrame, y_col: str, x_col: str, lat_range, lon_range):
    out = df.copy()
    out[y_col] = pd.to_numeric(out[y_col], errors="coerce")
    out[x_col] = pd.to_numeric(out[x_col], errors="coerce")
    mask = (
        out[y_col].between(lat_range[0], lat_range[1], inclusive="both")
        & out[x_col].between(lon_range[0], lon_range[1], inclusive="both")
    )
    return out.loc[mask].reset_index(drop=True)


def axis_step(axis_vals: np.ndarray) -> float:
    axis_vals = np.asarray(axis_vals, dtype=float)
    return float(np.nanmedian(np.diff(axis_vals))) if len(axis_vals) > 1 else 1.0


def frequency_grid_for_axes(lat_axis: np.ndarray, lon_axis: np.ndarray):
    dlat = axis_step(lat_axis)
    dlon = axis_step(lon_axis)
    fy = np.fft.fftshift(np.fft.fftfreq(len(lat_axis), d=dlat))
    fx = np.fft.fftshift(np.fft.fftfreq(len(lon_axis), d=dlon))
    omega_y = 2.0 * np.pi * fy
    omega_x = 2.0 * np.pi * fx
    ox, oy = np.meshgrid(omega_x, omega_y)
    k = np.sqrt(ox**2 + oy**2)
    return k, oy, ox


def build_month_context(args: argparse.Namespace, year: int, days: list[int]) -> MonthContext:
    lat_range = parse_range_pair(args.lat_range)
    lon_range = parse_range_pair(args.lon_range)
    obj, data_path = load_month_pickle(args, year, lat_range, lon_range)

    entries = []
    fallback_order = 0
    for key, df in obj.items():
        if not isinstance(df, pd.DataFrame):
            continue
        ts = parse_gems_key(str(key))
        if ts is not None:
            if ts.year != year or ts.month != args.month or ts.day not in days:
                continue
            sort_key = (int(ts.day), ts.hour, ts.minute, str(key))
        else:
            sort_key = (fallback_order // 8 + 1, fallback_order % 8, 0, str(key))
            fallback_order += 1
        f = filter_bounds(df, args.y_col, args.x_col, lat_range, lon_range)
        if not f.empty:
            entries.append((sort_key, ts, str(key), f))
    entries.sort(key=lambda x: x[0])

    entries_by_day = {d: [] for d in days}
    for sort_key, ts, key, df in entries:
        day = int(ts.day) if ts is not None else int(sort_key[0])
        if day in entries_by_day:
            entries_by_day[day].append((ts, key, df))

    nonempty_days = [d for d, rows in entries_by_day.items() if rows]
    if not nonempty_days:
        raise ValueError(f"No data entries found for requested days={days} in {data_path}")

    first_df = order_frame(entries_by_day[nonempty_days[0]][0][2], args.x_col, args.y_col)
    grid_coords_full = first_df[[args.y_col, args.x_col]].to_numpy(dtype=np.float64)
    lat_key = np.round(grid_coords_full[:, 0], 10)
    lon_key = np.round(grid_coords_full[:, 1], 10)
    lat_vals = np.sort(np.unique(lat_key))
    lon_vals = np.sort(np.unique(lon_key))
    lat_to_row = {float(v): i for i, v in enumerate(lat_vals)}
    lon_to_col = {float(v): i for i, v in enumerate(lon_vals)}
    local_to_row = np.asarray([lat_to_row[float(v)] for v in lat_key], dtype=np.int64)
    local_to_col = np.asarray([lon_to_col[float(v)] for v in lon_key], dtype=np.int64)
    grid_index = pd.MultiIndex.from_arrays([lat_key, lon_key], names=["_lat_key", "_lon_key"])

    k_full, omega_y_full, omega_x_full = frequency_grid_for_axes(lat_vals, lon_vals)
    positive_k = k_full[np.isfinite(k_full) & (k_full > 0)]
    k_max_full = float(np.quantile(positive_k, float(args.radial_qmax)))
    radial_bins = np.linspace(0.0, k_max_full, int(args.radial_bins) + 1)

    print(
        f"Context year={year} month={args.month:02d}: grid={len(lat_vals)}x{len(lon_vals)} "
        f"n={len(grid_coords_full)} lat_step={axis_step(lat_vals):.5g} lon_step={axis_step(lon_vals):.5g}",
        flush=True,
    )
    return MonthContext(
        year=year,
        month=args.month,
        entries_by_day=entries_by_day,
        lat_vals=lat_vals,
        lon_vals=lon_vals,
        grid_index=grid_index,
        local_to_row=local_to_row,
        local_to_col=local_to_col,
        grid_coords_full=grid_coords_full,
        radial_bins=radial_bins,
        k_full_radial=k_full,
        omega_y_full=omega_y_full,
        omega_x_full=omega_x_full,
        k_max_full=k_max_full,
        lat_step=axis_step(lat_vals),
        lon_step=axis_step(lon_vals),
    )


def hour_tensor_full(ctx: MonthContext, df: pd.DataFrame, hour_idx: int, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    ordered = order_frame(df, args.x_col, args.y_col)
    grid_lat = pd.to_numeric(ordered[args.y_col], errors="coerce").to_numpy(dtype=np.float64)
    grid_lon = pd.to_numeric(ordered[args.x_col], errors="coerce").to_numpy(dtype=np.float64)
    aligned = (
        ordered.assign(_lat_key=np.round(grid_lat, 10), _lon_key=np.round(grid_lon, 10))
        .drop_duplicates(["_lat_key", "_lon_key"], keep="first")
        .set_index(["_lat_key", "_lon_key"])
        .reindex(ctx.grid_index)
    )
    y = pd.to_numeric(aligned.get(args.value_col), errors="coerce").to_numpy(dtype=np.float64)
    grid_lat_full = ctx.grid_coords_full[:, 0].astype(np.float64)
    grid_lon_full = ctx.grid_coords_full[:, 1].astype(np.float64)
    if args.source_y_col in aligned.columns and args.source_x_col in aligned.columns:
        lat = pd.to_numeric(aligned[args.source_y_col], errors="coerce").to_numpy(dtype=np.float64)
        lon = pd.to_numeric(aligned[args.source_x_col], errors="coerce").to_numpy(dtype=np.float64)
        lat = np.where(np.isfinite(lat), lat, grid_lat_full)
        lon = np.where(np.isfinite(lon), lon, grid_lon_full)
    else:
        lat = grid_lat_full
        lon = grid_lon_full
    if args.time_col in aligned.columns and aligned[args.time_col].notna().any():
        h = pd.to_numeric(aligned[args.time_col], errors="coerce").to_numpy(dtype=np.float64)
        med = np.nanmedian(h)
        h = np.where(np.isfinite(h), h, med)
        h = h - float(args.time_origin)
    else:
        h = np.full(len(ctx.grid_coords_full), float(hour_idx), dtype=np.float64)
    base = np.column_stack([lat, lon, y, h])
    dummies = np.zeros((len(ctx.grid_coords_full), 7), dtype=np.float64)
    if hour_idx > 0 and hour_idx <= 7:
        dummies[:, hour_idx - 1] = 1.0
    arr = np.column_stack([base, dummies])
    return torch.as_tensor(arr, dtype=DTYPE, device=device)


def thin_indices(ctx: MonthContext, stride: int) -> np.ndarray:
    keep = (ctx.local_to_row % int(stride) == 0) & (ctx.local_to_col % int(stride) == 0)
    return np.flatnonzero(keep).astype(np.int64)


def build_day_input_map(args, ctx: MonthContext, entries, stride: int, device: torch.device):
    thin_idx = thin_indices(ctx, stride)
    input_map = {}
    full_tensors = []
    for hour_idx, (ts, key, df) in enumerate(entries[:8]):
        full_t = hour_tensor_full(ctx, df, hour_idx, args, device)
        thin_t = full_t[torch.as_tensor(thin_idx, dtype=torch.long, device=device)].contiguous()
        input_map[str(key)] = thin_t
        full_tensors.append(full_t)
    grid_coords = ctx.grid_coords_full[thin_idx].astype(np.float64)
    return input_map, full_tensors, thin_idx, grid_coords


def count_valid_input(input_map: dict[str, torch.Tensor]) -> tuple[int, int]:
    n_grid = int(next(iter(input_map.values())).shape[0]) if input_map else 0
    n_valid = 0
    for t in input_map.values():
        y = t[:, 2]
        n_valid += int((torch.isfinite(y) & torch.isfinite(t[:, 0]) & torch.isfinite(t[:, 1])).sum().item())
    return n_grid, n_valid


def make_params(args, variant: str, device: torch.device):
    range_lon = max(float(args.range_lon_init), EPS)
    range_lat = max(float(args.range_lat_init), EPS)
    range_time = max(float(args.range_time_init), EPS)
    phi2 = 1.0 / range_lon
    phi1 = max(float(args.sigmasq_init), EPS) * phi2
    phi3 = (range_lon / range_lat) ** 2
    phi4 = (range_lon / range_time) ** 2
    vals = [math.log(phi1), math.log(phi2), math.log(phi3), math.log(phi4), float(args.advec_lat_init), float(args.advec_lon_init)]
    if variant == "nugget_free":
        vals.append(math.log(max(float(args.nugget_init), EPS)))
    return [torch.tensor(v, dtype=DTYPE, device=device, requires_grad=True) for v in vals]


def convert_raw(raw: list[float], variant: str) -> dict:
    phi1, phi2, phi3, phi4 = np.exp(raw[0]), np.exp(raw[1]), np.exp(raw[2]), np.exp(raw[3])
    out = {
        "sigmasq": float(phi1 / phi2),
        "range_lon": float(1.0 / phi2),
        "range_lat": float(1.0 / (phi2 * np.sqrt(phi3))),
        "range_time": float(1.0 / (phi2 * np.sqrt(phi4))),
        "advec_lat": float(raw[4]),
        "advec_lon": float(raw[5]),
        "nugget": 0.0,
        "phi1": float(phi1),
        "phi2": float(phi2),
        "phi3": float(phi3),
        "phi4": float(phi4),
    }
    if variant == "nugget_free":
        out["nugget"] = float(np.exp(raw[6]))
    return out


def fit_st_variant(args, ctx, day, stride, variant, smooth, input_map, grid_coords, device):
    n_grid, n_valid_total = count_valid_input(input_map)
    row = {
        "date_str": f"{ctx.year}{ctx.month:02d}{day:02d}",
        "year": int(ctx.year),
        "month": int(ctx.month),
        "day": int(day),
        "smooth": float(smooth),
        "resolution_stride": int(stride),
        "resolution_label": f"x{int(stride)}",
        "variant": variant,
        "n_hours": int(len(input_map)),
        "n_grid": int(n_grid),
        "n_valid_total": int(n_valid_total),
        "valid_fraction": float(n_valid_total / max(n_grid * len(input_map), 1)),
    }
    try:
        cls = VARIANTS[variant]["class"]
        model = cls(
            smooth=float(smooth),
            input_map=input_map,
            grid_coords=grid_coords,
            block_shape=parse_shape_pair(args.block_shape),
            n_neighbor_blocks_t=int(args.n_neighbor_blocks_t),
            lag1_local_blocks=int(args.lag1_local_blocks),
            lag1_shifted_blocks=int(args.lag1_shifted_blocks),
            lag2_local_blocks=int(args.lag2_local_blocks),
            lag2_shifted_blocks=int(args.lag2_shifted_blocks),
            daily_stride=int(args.daily_stride),
            lag1_lon_offset=float(args.lag1_lon_offset),
            lag2_lon_offset=args.lag2_lon_offset,
            target_chunk_size=int(args.target_chunk_size),
            min_target_points=int(args.min_target_points),
        )
        t_pre = time.time()
        model.precompute_conditioning_sets()
        pre_s = time.time() - t_pre
        params = make_params(args, variant, device)
        opt = model.set_optimizer(
            params,
            lr=1.0,
            max_iter=int(args.lbfgs_eval),
            max_eval=int(args.lbfgs_eval),
            history_size=int(args.lbfgs_history),
        )
        t_fit = time.time()
        raw_out, fit_iter = model.fit_vecc_lbfgs(params, opt, max_steps=int(args.lbfgs_steps), grad_tol=float(args.grad_tol))
        fit_s = time.time() - t_fit
        est = convert_raw(raw_out[: VARIANTS[variant]["n_params"]], variant)
        summary = model.cluster_summary()
        out = {
            **row,
            "status": "ok",
            "error": "",
            "loss": float(raw_out[-1]),
            "fit_iter_raw": int(fit_iter),
            "fit_steps_reported": int(fit_iter) + 1,
            "precompute_s": float(pre_s),
            "fit_s": float(fit_s),
            "total_s": float(pre_s + fit_s),
            "est_sigmasq": est["sigmasq"],
            "est_range_lat": est["range_lat"],
            "est_range_lon": est["range_lon"],
            "est_range_time": est["range_time"],
            "est_advec_lat": est["advec_lat"],
            "est_advec_lon": est["advec_lon"],
            "est_nugget": est["nugget"],
            "est_phi1": est["phi1"],
            "est_phi2": est["phi2"],
            "est_phi3": est["phi3"],
            "est_phi4": est["phi4"],
            **summary,
        }
        del model, params, opt
        return out, est
    except Exception as exc:
        out = {
            **row,
            "status": "error",
            "error": repr(exc),
            "loss": np.nan,
            "fit_iter_raw": -1,
            "fit_steps_reported": 0,
            "precompute_s": np.nan,
            "fit_s": np.nan,
            "total_s": np.nan,
            "est_sigmasq": np.nan,
            "est_range_lat": np.nan,
            "est_range_lon": np.nan,
            "est_range_time": np.nan,
            "est_advec_lat": np.nan,
            "est_advec_lon": np.nan,
            "est_nugget": np.nan,
            "est_phi1": np.nan,
            "est_phi2": np.nan,
            "est_phi3": np.nan,
            "est_phi4": np.nan,
        }
        print(f"ERROR ST fit failed: {row['date_str']} {variant} x{stride}: {exc}", flush=True)
        return out, None
    finally:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def trend_design(lat, lon, mean_design="lat"):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat_c = lat - np.nanmean(lat)
    lon_c = lon - np.nanmean(lon)
    if mean_design in ("lat", "base"):
        return np.column_stack([np.ones(len(lat)), lat_c])
    return np.column_stack([np.ones(len(lat)), lat_c, lon_c])


def detrended_residual_grid(ctx: MonthContext, thin_idx: np.ndarray, thin_t: torch.Tensor):
    arr = thin_t.detach().cpu().numpy()
    y = arr[:, 2].astype(float)
    lat = arr[:, 0].astype(float)
    lon = arr[:, 1].astype(float)
    valid = np.isfinite(y) & np.isfinite(lat) & np.isfinite(lon)
    if valid.sum() < 4:
        raise ValueError("Not enough valid points for spectral grid.")
    x = trend_design(lat[valid], lon[valid], mean_design="lat")
    beta, *_ = np.linalg.lstsq(x, y[valid], rcond=None)
    resid = y - trend_design(lat, lon, mean_design="lat") @ beta

    rows_full = ctx.local_to_row[thin_idx]
    cols_full = ctx.local_to_col[thin_idx]
    row_keep = np.sort(np.unique(rows_full))
    col_keep = np.sort(np.unique(cols_full))
    row_pos = {int(r): i for i, r in enumerate(row_keep)}
    col_pos = {int(c): i for i, c in enumerate(col_keep)}
    grid = np.full((len(row_keep), len(col_keep)), np.nan, dtype=float)
    mask = np.zeros_like(grid, dtype=float)
    rr = np.asarray([row_pos[int(r)] for r in rows_full[valid]], dtype=np.int64)
    cc = np.asarray([col_pos[int(c)] for c in cols_full[valid]], dtype=np.int64)
    grid[rr, cc] = resid[valid]
    mask[rr, cc] = 1.0
    return grid, mask, int(valid.sum()), ctx.lat_vals[row_keep].astype(float), ctx.lon_vals[col_keep].astype(float)


def masked_periodogram(grid: np.ndarray, mask: np.ndarray, use_hann: bool):
    obs = mask > 0
    if not np.any(obs):
        raise ValueError("No observed cells in spectral grid.")
    z = np.zeros_like(grid, dtype=float)
    z[obs] = grid[obs]
    win = np.outer(np.hanning(z.shape[0]), np.hanning(z.shape[1])) if use_hann else np.ones_like(z)
    norm = np.sum((mask * win) ** 2)
    norm = norm if norm > EPS else 1.0
    return np.abs(np.fft.fftshift(np.fft.fft2(z * win))) ** 2 / norm


def matern_corr(scaled_distance, smooth):
    scaled_distance = np.asarray(scaled_distance, dtype=float)
    nu = float(smooth)
    if np.isclose(nu, 0.5):
        return np.exp(-scaled_distance)
    if np.isclose(nu, 1.5):
        return (1.0 + scaled_distance) * np.exp(-scaled_distance)
    corr = np.ones_like(scaled_distance, dtype=float)
    pos = scaled_distance > EPS
    if np.any(pos):
        z = np.sqrt(2.0 * nu) * scaled_distance[pos]
        corr[pos] = (2.0 ** (1.0 - nu) / gamma(nu)) * (z**nu) * kv(nu, z)
    return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def spatial_covariance_lag(sigmasq, range_lat, range_lon, nugget, smooth, lag_lat, lag_lon):
    scaled = np.sqrt((np.asarray(lag_lat, dtype=float) / max(float(range_lat), EPS)) ** 2 + (np.asarray(lag_lon, dtype=float) / max(float(range_lon), EPS)) ** 2)
    zero_lag = (np.abs(lag_lat) < EPS) & (np.abs(lag_lon) < EPS)
    return float(sigmasq) * matern_corr(scaled, smooth) + max(float(nugget), 0.0) * zero_lag


def taper_mask_autocorrelation(mask: np.ndarray, use_hann: bool):
    win = np.outer(np.hanning(mask.shape[0]), np.hanning(mask.shape[1])) if use_hann else np.ones_like(mask)
    g = mask * win
    h_norm = float(np.sum(g**2))
    if h_norm <= EPS:
        raise ValueError("Taper/mask normalization is zero.")
    n1, n2 = g.shape
    g_fft = np.fft.fft2(g, s=(2 * n1 - 1, 2 * n2 - 1))
    ac = np.fft.fftshift(np.fft.ifft2(g_fft * np.conj(g_fft)).real) / h_norm
    return ac, h_norm


def autocorr_for_lags(ac, lag1, lag2, n1, n2):
    out = np.zeros_like(lag1, dtype=float)
    valid = (np.abs(lag1) <= n1 - 1) & (np.abs(lag2) <= n2 - 1)
    if np.any(valid):
        out[valid] = ac[(n1 - 1 + lag1[valid]).astype(int), (n2 - 1 + lag2[valid]).astype(int)]
    return out


def covariance_lag_kernel(est, smooth, n1, n2, dlat, dlon):
    lag1_vals = np.arange(-(n1 - 1), n1, dtype=int)
    lag2_vals = np.arange(-(n2 - 1), n2, dtype=int)
    lag1, lag2 = np.meshgrid(lag1_vals, lag2_vals, indexing="ij")
    return spatial_covariance_lag(est["sigmasq"], est["range_lat"], est["range_lon"], est["nugget"], smooth, lag1 * dlat, lag2 * dlon)


def fft_convolve_same(values, kernel):
    out_shape = (values.shape[0] + kernel.shape[0] - 1, values.shape[1] + kernel.shape[1] - 1)
    conv = np.fft.ifft2(np.fft.fft2(values, s=out_shape) * np.fft.fft2(kernel, s=out_shape)).real
    start0 = kernel.shape[0] // 2
    start1 = kernel.shape[1] // 2
    return conv[start0 : start0 + values.shape[0], start1 : start1 + values.shape[1]]


def expected_periodogram_dw_style(args, est, smooth, mask, lat_axis, lon_axis, project_mean=True):
    n1, n2 = mask.shape
    ac, _ = taper_mask_autocorrelation(mask, args.hann)
    dlat = axis_step(lat_axis)
    dlon = axis_step(lon_axis)
    u1, u2 = np.meshgrid(np.arange(n1, dtype=int), np.arange(n2, dtype=int), indexing="ij")
    tilde_cov = np.zeros((n1, n2), dtype=float)
    for shift1 in (0, -n1):
        for shift2 in (0, -n2):
            lag1 = u1 + shift1
            lag2 = u2 + shift2
            ac_lag = autocorr_for_lags(ac, lag1, lag2, n1, n2)
            if not np.any(ac_lag):
                continue
            cov_lag = spatial_covariance_lag(est["sigmasq"], est["range_lat"], est["range_lon"], est["nugget"], smooth, lag1 * dlat, lag2 * dlon)
            tilde_cov += cov_lag * ac_lag
    expected_no_projection = np.fft.fftshift(np.fft.fft2(tilde_cov).real)
    if not project_mean:
        return np.maximum(expected_no_projection, EPS)

    obs = mask > 0
    if obs.sum() <= 2:
        return np.maximum(expected_no_projection, EPS)
    win = np.outer(np.hanning(n1), np.hanning(n2)) if args.hann else np.ones((n1, n2), dtype=float)
    g = mask * win
    h_norm = float(np.sum(g**2))
    lat_grid = np.repeat(np.asarray(lat_axis, dtype=float)[:, None], n2, axis=1)
    lon_grid = np.repeat(np.asarray(lon_axis, dtype=float)[None, :], n1, axis=0)
    x_flat = trend_design(lat_grid.ravel(), lon_grid.ravel(), mean_design="lat")
    x = x_flat.reshape(n1, n2, x_flat.shape[1])
    u = x * mask[..., None]
    xtx = np.einsum("ijp,ijq->pq", u, x)
    b = np.linalg.pinv(xtx)
    kernel = covariance_lag_kernel(est, smooth, n1, n2, axis_step(lat_axis), axis_step(lon_axis))
    cu = np.empty_like(u, dtype=float)
    for p_idx in range(u.shape[2]):
        cu[..., p_idx] = fft_convolve_same(u[..., p_idx], kernel)
    s = np.einsum("ijp,ijq->pq", u, cu)
    h_fft = []
    c_fft = []
    for p_idx in range(u.shape[2]):
        h_fft.append(np.fft.fftshift(np.fft.fft2(g * x[..., p_idx])))
        c_fft.append(np.fft.fftshift(np.fft.fft2(g * cu[..., p_idx])))
    h_fft = np.stack(h_fft, axis=-1)
    c_fft = np.stack(c_fft, axis=-1)
    a_raw = expected_no_projection * h_norm
    cross = np.einsum("...p,pq,...q->...", c_fft, b, np.conj(h_fft))
    mean_quad = np.einsum("...p,pq,...q->...", h_fft, b @ s @ b, np.conj(h_fft))
    expected_projected = (a_raw - 2.0 * np.real(cross) + np.real(mean_quad)) / h_norm
    return np.maximum(expected_projected, EPS)


def radial_average(surface, k_radial, radial_bins, k_max):
    vals = np.asarray(surface, dtype=float).ravel()
    kk = np.asarray(k_radial, dtype=float).ravel()
    good = np.isfinite(vals) & np.isfinite(kk) & (kk > 0) & (kk <= k_max)
    if not np.any(good):
        return pd.DataFrame(columns=["k_bin", "k_mid", "k_mean", "spectrum", "n_freq"])
    bin_idx = np.digitize(kk[good], radial_bins, right=False) - 1
    valid_bin = (bin_idx >= 0) & (bin_idx < len(radial_bins) - 1)
    bin_idx = bin_idx[valid_bin]
    vg = vals[good][valid_bin]
    kg = kk[good][valid_bin]
    rows = []
    for b in range(len(radial_bins) - 1):
        m = bin_idx == b
        if not np.any(m):
            continue
        rows.append({"k_bin": int(b), "k_mid": float(0.5 * (radial_bins[b] + radial_bins[b + 1])), "k_mean": float(np.nanmean(kg[m])), "spectrum": float(np.nanmean(vg[m])), "n_freq": int(m.sum())})
    return pd.DataFrame(rows)


def compute_spectrum_rows(args, ctx, fit_row, est, full_tensors, thin_idx, stride, smooth, device):
    rows = []
    for hour_idx, full_t in enumerate(full_tensors):
        thin_t = full_t[torch.as_tensor(thin_idx, dtype=torch.long, device=device)].contiguous()
        grid, mask, n_valid_spectrum, lat_axis, lon_axis = detrended_residual_grid(ctx, thin_idx, thin_t)
        data_p = masked_periodogram(grid, mask, args.hann)
        expected_p = expected_periodogram_dw_style(args, est, smooth, mask, lat_axis, lon_axis)
        k_data, _, _ = frequency_grid_for_axes(lat_axis, lon_axis)
        data_rad = radial_average(data_p, k_data, ctx.radial_bins, ctx.k_max_full).rename(columns={"spectrum": "data_spectrum"})
        expected_rad = radial_average(expected_p, k_data, ctx.radial_bins, ctx.k_max_full).rename(columns={"spectrum": "theory_spectrum_expected"})
        merged = expected_rad[["k_bin", "k_mid", "k_mean", "theory_spectrum_expected", "n_freq"]].merge(
            data_rad[["k_bin", "data_spectrum"]], on="k_bin", how="left"
        )
        data_k_max = float(data_rad["k_mid"].max()) if not data_rad.empty else np.nan
        for m in merged.itertuples(index=False):
            rows.append({
                **{k: fit_row[k] for k in [
                    "date_str", "year", "month", "day", "smooth", "resolution_stride", "resolution_label", "variant",
                    "est_sigmasq", "est_range_lat", "est_range_lon", "est_range_time", "est_advec_lat", "est_advec_lon", "est_nugget",
                    "est_phi1", "est_phi2", "est_phi3", "est_phi4",
                ]},
                "hour_idx": int(hour_idx),
                "n_valid_spectrum": int(n_valid_spectrum),
                "k_bin": int(m.k_bin),
                "k_mid": float(m.k_mid),
                "k_mean": float(m.k_mean),
                "n_freq": int(m.n_freq) if pd.notna(m.n_freq) else 0,
                "data_k_max": data_k_max,
                "data_spectrum": float(m.data_spectrum) if pd.notna(m.data_spectrum) else np.nan,
                "theory_spectrum_expected": float(m.theory_spectrum_expected) if pd.notna(m.theory_spectrum_expected) else np.nan,
            })
    return rows


def positive_ylim(*series_list, fallback=(1e0, 1e4)):
    vals = []
    for s in series_list:
        if s is None:
            continue
        arr = pd.to_numeric(pd.Series(s), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        arr = arr[arr > 0]
        if not arr.empty:
            vals.append(arr.to_numpy(dtype=float))
    if not vals:
        return fallback
    x = np.concatenate(vals)
    return 10 ** np.floor(np.log10(float(x.min()))), 10 ** np.ceil(np.log10(float(x.max())))


def ratio_frame(df, numerator_col="data_spectrum", denominator_col="theory_spectrum_expected"):
    if df.empty:
        return pd.DataFrame(columns=["k_bin", "k_mid", "ratio"])
    merged = df[["k_bin", "k_mid", numerator_col, denominator_col]].replace([np.inf, -np.inf], np.nan)
    good = merged[numerator_col].notna() & merged[denominator_col].notna() & (merged[numerator_col] > 0) & (merged[denominator_col] > 0)
    out = merged.loc[good, ["k_bin", "k_mid"]].copy()
    out["ratio"] = merged.loc[good, numerator_col].to_numpy(dtype=float) / merged.loc[good, denominator_col].to_numpy(dtype=float)
    return out.sort_values("k_bin")


def add_ratio_axis(ax, ratio_df, ylabel=None):
    if ratio_df is None or ratio_df.empty:
        return None
    r = ratio_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["k_mid", "ratio"])
    r = r[r["ratio"] > 0].sort_values("k_bin")
    if r.empty:
        return None
    ratio_ax = ax.twinx()
    ratio_ax.plot(r["k_mid"], r["ratio"], color="tab:blue", linewidth=1.35, linestyle=":", alpha=0.95, zorder=8)
    ratio_ax.axhline(1.0, color="tab:blue", linewidth=0.8, linestyle="-", alpha=0.35, zorder=7)
    ratio_ax.set_yscale("log")
    vals = r["ratio"].to_numpy(dtype=float)
    lo = max(1e-3, min(0.5, float(np.nanmin(vals)) / 1.2))
    hi = min(1e3, max(2.0, float(np.nanmax(vals)) * 1.2))
    if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
        ratio_ax.set_ylim(lo, hi)
    ratio_ax.tick_params(axis="y", colors="tab:blue", labelsize=7)
    ratio_ax.spines["right"].set_color("tab:blue")
    ratio_ax.grid(False)
    if ylabel:
        ratio_ax.set_ylabel(ylabel, color="tab:blue", fontsize=8)
    return ratio_ax


def aggregate_daily(spectral_df: pd.DataFrame):
    return (
        spectral_df
        .groupby(["variant", "resolution_label", "resolution_stride", "k_bin"], observed=True)
        .agg(
            k_mid=("k_mid", "mean"),
            data_spectrum=("data_spectrum", "mean"),
            theory_spectrum_expected=("theory_spectrum_expected", "mean"),
            data_k_max=("data_k_max", "mean"),
            n_hours=("hour_idx", "nunique"),
        )
        .reset_index()
    )


def format_fit_label(source_df, variant, resolution_label, prefix="ST fit"):
    if source_df is None or source_df.empty:
        return None
    df = source_df[(source_df["variant"].astype(str) == str(variant)) & (source_df["resolution_label"].astype(str) == str(resolution_label))]
    if df.empty:
        return None
    sig = float(pd.to_numeric(df["est_sigmasq"], errors="coerce").median())
    rlat = float(pd.to_numeric(df["est_range_lat"], errors="coerce").median())
    rlon = float(pd.to_numeric(df["est_range_lon"], errors="coerce").median())
    nug = float(pd.to_numeric(df["est_nugget"], errors="coerce").median())
    if not np.isfinite(sig) or not np.isfinite(rlat) or not np.isfinite(rlon):
        return None
    label = f"{prefix}: sigma^2={sig:.3g}\nrange_lat={rlat:.3g}, range_lon={rlon:.3g}"
    if str(variant) == "nugget_free":
        label += f"\nnugget={nug:.3g}"
    return label


def plot_panel_grid(ctx, args, smooth, title, spectral_df, fit_df, out_path, monthly=False):
    labels_order = [f"x{s}" for s in parse_int_list_or_range(args.resolutions)]
    row_specs = [(v, VARIANTS[v]["row_title"]) for v in parse_names(args.variants) if v in VARIANTS]
    plot_df = spectral_df.copy()
    if monthly:
        gray_df = plot_df.copy()
        mean_df = (
            plot_df.groupby(["variant", "resolution_label", "resolution_stride", "k_bin"], observed=True)
            .agg(k_mid=("k_mid", "mean"), data_spectrum=("data_spectrum", "mean"), theory_spectrum_expected=("theory_spectrum_expected", "mean"), data_k_max=("data_k_max", "mean"))
            .reset_index()
        )
    else:
        gray_df = plot_df
        mean_df = aggregate_daily(plot_df)
    ylim = positive_ylim(mean_df.get("data_spectrum"), mean_df.get("theory_spectrum_expected"))
    fig, axes = plt.subplots(len(row_specs), len(labels_order), figsize=(4.4 * len(labels_order), 3.4 * len(row_specs)), sharey=True)
    axes = np.asarray(axes).reshape(len(row_specs), len(labels_order))
    for i, (variant, row_title) in enumerate(row_specs):
        for j, label in enumerate(labels_order):
            ax = axes[i, j]
            sub_mean = mean_df[(mean_df["variant"] == variant) & (mean_df["resolution_label"].astype(str) == label) & (mean_df["k_mid"] > 0)]
            sub_gray = gray_df[(gray_df["variant"] == variant) & (gray_df["resolution_label"].astype(str) == label) & (gray_df["k_mid"] > 0) & gray_df["data_spectrum"].notna()]
            if sub_mean.empty:
                ax.set_visible(False)
                continue
            k_cut = float(sub_mean["data_k_max"].dropna().iloc[0]) if sub_mean["data_k_max"].notna().any() else float(sub_mean["k_mid"].max())
            param_label = format_fit_label(fit_df, variant, label, prefix="fit median" if monthly else "ST fit")
            if param_label:
                ax.plot([], [], color="none", label=param_label)
            red_label = "expected periodogram (30-day mean)" if monthly else "expected periodogram (mean over 8 h)"
            ax.plot(sub_mean["k_mid"], sub_mean["theory_spectrum_expected"], color="tab:red", linewidth=1.9, linestyle="--", label=red_label, zorder=3)
            group_key = "date_str" if monthly else "hour_idx"
            gray_label_base = "daily mean spectra" if monthly else "hourly data spectra"
            for g_i, (_, gs) in enumerate(sub_gray.groupby(group_key)):
                g_label = gray_label_base if (i == 0 and j == 0 and g_i == 0) else None
                ax.plot(gs["k_mid"], gs["data_spectrum"], color="0.35", alpha=0.35 if monthly else 0.55, linewidth=0.85 if monthly else 1.05, label=g_label, zorder=1)
            black_label = "data residual spectrum (30-day mean)" if monthly else "data residual spectrum (mean over 8 h)"
            ax.plot(sub_mean["k_mid"], sub_mean["data_spectrum"], color="black", linewidth=2.2, label=black_label, zorder=4)
            ratio_df = ratio_frame(sub_mean)
            add_ratio_axis(ax, ratio_df, ylabel="I / E[I]" if j == len(labels_order) - 1 else None)
            ax.plot([], [], color="tab:blue", linewidth=1.35, linestyle=":", label="I / E[I] (ratio of means)")
            ax.axvline(k_cut, color="0.45", linewidth=1.0, linestyle=":", alpha=0.95, zorder=2)
            ax.set_xlim(0, ctx.k_max_full)
            ax.set_ylim(*ylim)
            ax.set_title(f"{row_title}, {label}  (data k <= {k_cut:.1f})")
            ax.set_xlabel("radial frequency on full-grid scale")
            if j == 0:
                ax.set_ylabel("spectrum")
            ax.set_yscale("log")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7, handlelength=1.5)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format=f"%.{ROUND_DECIMALS}f")


def process_day(args, ctx: MonthContext, smooth: float, day: int, out_dir: Path, device: torch.device):
    date_str = f"{ctx.year}{ctx.month:02d}{day:02d}"
    daily_dir = out_dir / "daily_csv"
    plot_dir = out_dir / "daily_plots"
    fit_path = daily_dir / f"{date_str}_st_fits.csv"
    spec_path = daily_dir / f"{date_str}_implied_spatial_radial_spectrum.csv"
    plot_path = plot_dir / f"{date_str}_implied_spatial_data_vs_expected_periodogram.png"
    if args.skip_existing and spec_path.exists() and fit_path.exists() and plot_path.exists():
        print(f"Skip existing day {date_str}", flush=True)
        return
    entries = ctx.entries_by_day.get(day, [])
    if len(entries) < 8:
        print(f"WARNING: {date_str} has {len(entries)} hourly entries; expected 8. Processing available hours.", flush=True)
    if not entries:
        return

    fit_rows = []
    spectral_rows = []
    for stride in parse_int_list_or_range(args.resolutions):
        input_map, full_tensors, thin_idx, grid_coords = build_day_input_map(args, ctx, entries, stride, device)
        for variant in [v for v in parse_names(args.variants) if v in VARIANTS]:
            print(f"\n{date_str} smooth={smooth} {variant} x{stride}: daily ST fit", flush=True)
            fit_row, est = fit_st_variant(args, ctx, day, stride, variant, smooth, input_map, grid_coords, device)
            fit_rows.append(fit_row)
            if est is None:
                continue
            try:
                spectral_rows.extend(compute_spectrum_rows(args, ctx, fit_row, est, full_tensors, thin_idx, stride, smooth, device))
            except Exception as exc:
                print(f"ERROR spectrum failed: {date_str} {variant} x{stride}: {exc}", flush=True)

    fit_df = pd.DataFrame(fit_rows)
    spec_df = pd.DataFrame(spectral_rows)
    write_csv(fit_df, fit_path)
    write_csv(spec_df, spec_path)
    if not spec_df.empty:
        title = f"{ctx.year}-{ctx.month:02d}-{day:02d}, smooth={smooth}: ST implied spatial residual spectrum vs expected periodogram"
        plot_panel_grid(ctx, args, smooth, title, spec_df, fit_df, plot_path, monthly=False)
    print(f"Saved day {date_str}: {fit_path}, {spec_path}, {plot_path}", flush=True)


def make_monthly_average(args, ctx: MonthContext, smooth: float, out_dir: Path):
    daily_dir = out_dir / "daily_csv"
    spec_paths = sorted(daily_dir.glob("*_implied_spatial_radial_spectrum.csv"))
    if not spec_paths:
        print(f"No daily spectra found in {daily_dir}; skipping monthly average.", flush=True)
        return
    frames = []
    for path in spec_paths:
        try:
            frames.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            continue
    if not frames:
        return
    spectral_df = pd.concat(frames, ignore_index=True)
    fit_frames = []
    for path in sorted(daily_dir.glob("*_st_fits.csv")):
        try:
            fit_frames.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            continue
    fit_df = pd.concat(fit_frames, ignore_index=True) if fit_frames else spectral_df
    daily_means = (
        spectral_df
        .groupby(["date_str", "variant", "resolution_label", "resolution_stride", "k_bin"], observed=True)
        .agg(
            k_mid=("k_mid", "mean"),
            data_spectrum=("data_spectrum", "mean"),
            theory_spectrum_expected=("theory_spectrum_expected", "mean"),
            data_k_max=("data_k_max", "mean"),
            n_hours=("hour_idx", "nunique"),
        )
        .reset_index()
    )
    monthly = (
        daily_means
        .groupby(["variant", "resolution_label", "resolution_stride", "k_bin"], observed=True)
        .agg(k_mid=("k_mid", "mean"), data_spectrum=("data_spectrum", "mean"), theory_spectrum_expected=("theory_spectrum_expected", "mean"), data_k_max=("data_k_max", "mean"), n_days=("date_str", "nunique"))
        .reset_index()
    )
    write_csv(daily_means, out_dir / "monthly_average" / f"{ctx.year}{ctx.month:02d}_daily_mean_curves.csv")
    write_csv(monthly, out_dir / "monthly_average" / f"{ctx.year}{ctx.month:02d}_30day_mean_curves.csv")
    title = f"{ctx.year}-{ctx.month:02d}, smooth={smooth}: 30-day mean ST implied spatial spectrum vs expected periodogram"
    plot_path = out_dir / "monthly_average" / f"{ctx.year}{ctx.month:02d}_30day_mean_implied_spatial_data_vs_expected_periodogram.png"
    plot_panel_grid(ctx, args, smooth, title, daily_means, fit_df, plot_path, monthly=True)
    print(f"Saved monthly average plot: {plot_path}", flush=True)


def output_dir_for(args: argparse.Namespace, smooth: float, year: int) -> Path:
    return Path(args.output_root) / f"smooth_{smooth_tag(smooth)}" / f"{year}_{args.month:02d}"


def main():
    args = parse_args()
    years = parse_int_list_or_range(args.years)
    days = parse_int_list_or_range(args.days)
    smooths = parse_float_list(args.smooths)
    device = select_device(args.device, args.cuda_fallback)
    print(
        f"Run config: years={years}, month={args.month}, days={days[0]}..{days[-1]}, "
        f"smooths={smooths}, resolutions={parse_int_list_or_range(args.resolutions)}, "
        f"variants={parse_names(args.variants)}, device={device}",
        flush=True,
    )
    for year in years:
        ctx = build_month_context(args, year, days)
        for smooth in smooths:
            out_dir = output_dir_for(args, smooth, year)
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n=== year={year} smooth={smooth} out={out_dir} ===", flush=True)
            if not args.make_monthly_only:
                for day in days:
                    process_day(args, ctx, smooth, day, out_dir, device)
            make_monthly_average(args, ctx, smooth, out_dir)


if __name__ == "__main__":
    main()
