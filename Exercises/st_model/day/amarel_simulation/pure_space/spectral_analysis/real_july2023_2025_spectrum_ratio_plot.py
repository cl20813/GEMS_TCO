#!/usr/bin/env python3
"""Real July 2023-2025 pure-space spectrum ratio diagnostics for Amarel.

This is the canonical pure-space spectrum-ratio plot script for real July GEMS
TCO data over latitude -3..2 and longitude 121..131.

  years:          2023, 2024, 2025 by default
  variants:       Matérn smooth 0.3 baseline and one year-specific generalized Cauchy model
  block prefix:   all only by default, so every fit is x1/full-resolution
  domains:        full -3..2, 121..131 spatial domain by default

Each domain is modeled independently.  This is a pure-space diagnostic, so no
temporal 8x8 whitening is applied.  The monthly summaries average the available
July hourly spectra directly, usually 30 days x 8 hours = 240 hours.

  gray  = hourly data residual spectra
  black = mean data residual spectrum over hours
  red   = mean fitted finite-sample expected periodogram over hours
  blue  = ratio of means, I / E[I]

Daily norm-frequency plots are written first. Monthly plots include norm,
latitude, longitude, and NE-SW diagonal profiles. The main comparison output is
the profiled ratio I / E[I], overlaid across the Matérn baseline and the
year-specific GC candidates.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import pickle
import re
import shutil
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
from scipy.optimize import minimize
from scipy.special import gamma, kv


LOCAL_SRC = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/src")
AMAREL_SRC = Path("/home/jl2815/tco")
for candidate in (AMAREL_SRC, LOCAL_SRC):
    if (candidate / "GEMS_TCO").is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
        break

from GEMS_TCO import configuration as config
from GEMS_TCO import orderings
from GEMS_TCO.kernels_space_iso_cluster_052426 import (
    ClusterSpaceIsoTrendVecchiaFit,
)
from GEMS_TCO.kernels_space_aniso_cauchy_cluster_060326 import (
    ClusterSpaceAnisoCauchyFixedBetaNoNuggetTrendVecchiaFit,
    cauchy_phi_init_from_natural,
)
from GEMS_TCO.matern_bessel_anisotropic import (
    natural_from_raw,
    profiled_vecchia_cluster_nll,
    raw_from_natural,
    smooth_to_raw,
    vecchia_batches_to_numpy,
)


DTYPE = torch.float64
EPS = 1e-12
ROUND_DECIMALS = 6
MONTH_PICKLE_CACHE: dict[str, dict] = {}


VARIANTS = {
    "matern_s03": {
        "family": "matern",
        "smooth": 0.3,
        "row_title": "Matérn smooth 0.3, nugget fixed 0",
        "plot_label": "Matern s=0.3 nugget0",
    },
    "gc_a075_b1": {
        "family": "cauchy",
        "gc_alpha": 0.75,
        "gc_beta": 1.0,
        "row_title": "GC a=0.75 b=1, nugget fixed 0",
        "plot_label": "GC a=0.75 b=1 nugget0",
    },
    "gc_a08_b1": {
        "family": "cauchy",
        "gc_alpha": 0.8,
        "gc_beta": 1.0,
        "row_title": "GC a=0.8 b=1, nugget fixed 0",
        "plot_label": "GC a=0.8 b=1 nugget0",
    },
}

YEAR_VARIANTS = {
    2023: ["matern_s03", "gc_a075_b1"],
    2024: ["matern_s03", "gc_a08_b1"],
    2025: ["matern_s03", "gc_a075_b1"],
}


def normalize_range_argv(argv: list[str]) -> list[str]:
    """Let argparse accept negative comma ranges passed as separate argv items."""
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
    omega2_full: np.ndarray
    omega_lat_full: np.ndarray
    omega_lon_full: np.ndarray
    k_max_full: float
    lat_step: float
    lon_step: float


@dataclass(frozen=True)
class DomainSpec:
    group: str
    label: str
    title: str
    lat_range: tuple[float, float]
    lon_range: tuple[float, float]
    row: int = 0
    col: int = 0
    n_rows: int = 1
    n_cols: int = 1
    lat_upper_inclusive: bool = True
    lon_upper_inclusive: bool = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real July 2023-2025 pure-space Matern/GC spectrum ratio diagnostics.")
    p.add_argument("--years", default="2023,2024,2025")
    p.add_argument("--month", type=int, default=7)
    p.add_argument("--days", default="1,30", help="Inclusive day range or comma list. Default uses July days 1..30.")
    p.add_argument("--smooths", default="0.3")
    p.add_argument("--block-prefixes", default="all")
    p.add_argument("--variants", default="matern_s03,gc_a075_b1,gc_a08_b1")
    p.add_argument("--neighbors", type=int, default=2, help="Deprecated alias; cluster B2 uses --cluster-neighbor-blocks.")
    p.add_argument("--cluster-neighbor-blocks", type=int, default=2)
    p.add_argument("--cluster-block-shape", default="4x4")
    p.add_argument("--mean-design", default="lat", choices=["lat", "base", "latlon", "hour_spatial"])
    p.add_argument("--data-root", default=getattr(config, "amarel_data_load_path", "/home/jl2815/tco/data/"))
    p.add_argument("--output-root", default="/home/jl2815/tco/exercise_output/summer/real_data/real_july2023_2025_pure_space_matern_gc_final_spectrum_ratio_plot")
    p.add_argument("--top-plot-dir", default="", help="Optional top-level folder that receives copies of monthly plot PNGs.")
    p.add_argument("--expanded-bounds", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lat-range", default="-3,2")
    p.add_argument("--lon-range", default="121,131")
    p.add_argument(
        "--domain-modes",
        default="full",
        help="Comma list from full,lat_slices,lon_slices,tile_2x4.",
    )
    p.add_argument("--lat-slice-count", type=int, default=5)
    p.add_argument("--lon-slice-count", type=int, default=5)
    p.add_argument("--lat-slice-width", type=float, default=0.0, help="0 means split --lat-range evenly.")
    p.add_argument("--lon-slice-width", type=float, default=0.0, help="0 means split --lon-range evenly.")
    p.add_argument("--tile-grid", default="2x4")
    p.add_argument("--combined-profiles", default="radial,lat,lon,diag")
    p.add_argument("--combined-ratio-normalize", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-combined-plots", action="store_true")
    p.add_argument("--x-col", default="Longitude")
    p.add_argument("--y-col", default="Latitude")
    p.add_argument("--source-x-col", default="Source_Longitude")
    p.add_argument("--source-y-col", default="Source_Latitude")
    p.add_argument("--value-col", default="ColumnAmountO3")
    p.add_argument("--time-col", default="Hours_elapsed")
    p.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    p.add_argument("--cuda-fallback", default="cpu", choices=["cpu", "error"])
    p.add_argument("--target-chunk-size", type=int, default=512)
    p.add_argument("--lbfgs-steps", type=int, default=8)
    p.add_argument("--lbfgs-eval", type=int, default=20)
    p.add_argument("--lbfgs-history", type=int, default=10)
    p.add_argument("--grad-tol", type=float, default=1e-5)
    p.add_argument("--n-restarts", type=int, default=1)
    p.add_argument("--sigmasq-init", type=float, default=13.0)
    p.add_argument("--range-init", type=float, default=0.25)
    p.add_argument("--range-lat-init", type=float, default=0.35)
    p.add_argument("--range-lon-init", type=float, default=0.35)
    p.add_argument("--range-min", type=float, default=0.03)
    p.add_argument("--range-max", type=float, default=5.0)
    p.add_argument("--nugget-init", type=float, default=2.5)
    p.add_argument("--radial-bins", type=int, default=70)
    p.add_argument("--radial-qmax", type=float, default=0.985)
    p.add_argument("--hann", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--make-monthly-only", action="store_true")
    return p.parse_args(normalize_range_argv(sys.argv[1:]))


def parse_int_list_or_range(text: str) -> list[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(vals) == 2 and vals[1] >= vals[0]:
        return list(range(vals[0], vals[1] + 1))
    return vals


def parse_block_prefixes(text: str) -> list[int]:
    out = []
    for token in [x.strip().lower() for x in str(text).split(",") if x.strip()]:
        out.append(-1 if token in {"all", "full"} else int(token))
    return out


def block_prefix_label(prefix: int) -> str:
    return "all" if int(prefix) <= 0 else f"B{int(prefix)}"


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_range_pair(text: str) -> tuple[float, float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(vals) != 2:
        raise ValueError(f"Expected two comma-separated values, got {text!r}")
    return vals[0], vals[1]


def parse_grid_shape(text: str) -> tuple[int, int]:
    vals = [int(x.strip()) for x in str(text).lower().replace("x", ",").split(",") if x.strip()]
    if len(vals) != 2:
        raise ValueError(f"grid shape must look like 2x4, got {text!r}")
    if vals[0] <= 0 or vals[1] <= 0:
        raise ValueError(f"grid shape must be positive, got {text!r}")
    return vals[0], vals[1]


def range_arg(bounds: tuple[float, float]) -> str:
    return f"{bounds[0]:.12g},{bounds[1]:.12g}"


def value_tag(value: float) -> str:
    s = f"{float(value):.6g}"
    return s.replace("-", "m").replace(".", "p")


def interval_label(prefix: str, lo: float, hi: float) -> str:
    return f"{prefix}_{value_tag(lo)}to{value_tag(hi)}"


def interval_title(prefix: str, lo: float, hi: float) -> str:
    return f"{prefix} {lo:g} to {hi:g}"


def evenly_spaced_intervals(
    bounds: tuple[float, float],
    count: int,
    width: float = 0.0,
) -> list[tuple[float, float]]:
    if int(count) <= 0:
        return []
    lo, hi = float(bounds[0]), float(bounds[1])
    if hi <= lo:
        raise ValueError(f"Range upper bound must exceed lower bound, got {bounds}")
    step = float(width) if float(width) > 0 else (hi - lo) / int(count)
    intervals = []
    for i in range(int(count)):
        a = lo + i * step
        b = min(a + step, hi)
        if b > a + EPS:
            intervals.append((float(a), float(b)))
    return intervals


def build_domain_specs(args: argparse.Namespace) -> list[DomainSpec]:
    lat_range = parse_range_pair(args.lat_range)
    lon_range = parse_range_pair(args.lon_range)
    modes = parse_names(args.domain_modes)
    specs: list[DomainSpec] = []

    if "full" in modes:
        specs.append(
            DomainSpec(
                group="full",
                label="full",
                title=f"full lat {lat_range[0]:g} to {lat_range[1]:g}, lon {lon_range[0]:g} to {lon_range[1]:g}",
                lat_range=lat_range,
                lon_range=lon_range,
            )
        )

    if "lat_slices" in modes:
        intervals = evenly_spaced_intervals(lat_range, int(args.lat_slice_count), float(args.lat_slice_width))
        for i, bounds in enumerate(intervals):
            specs.append(
                DomainSpec(
                    group="lat_slices",
                    label=interval_label("lat", bounds[0], bounds[1]),
                    title=interval_title("lat", bounds[0], bounds[1]),
                    lat_range=bounds,
                    lon_range=lon_range,
                    row=0,
                    col=i,
                    n_rows=1,
                    n_cols=len(intervals),
                    lat_upper_inclusive=(i == len(intervals) - 1),
                    lon_upper_inclusive=True,
                )
            )

    if "lon_slices" in modes:
        intervals = evenly_spaced_intervals(lon_range, int(args.lon_slice_count), float(args.lon_slice_width))
        for i, bounds in enumerate(intervals):
            specs.append(
                DomainSpec(
                    group="lon_slices",
                    label=interval_label("lon", bounds[0], bounds[1]),
                    title=interval_title("lon", bounds[0], bounds[1]),
                    lat_range=lat_range,
                    lon_range=bounds,
                    row=0,
                    col=i,
                    n_rows=1,
                    n_cols=len(intervals),
                    lat_upper_inclusive=True,
                    lon_upper_inclusive=(i == len(intervals) - 1),
                )
            )

    if "tile_2x4" in modes:
        tile_y, tile_x = parse_grid_shape(args.tile_grid)
        y_edges = np.linspace(float(lat_range[0]), float(lat_range[1]), tile_y + 1)
        x_edges = np.linspace(float(lon_range[0]), float(lon_range[1]), tile_x + 1)
        for iy in range(tile_y):
            for ix in range(tile_x):
                lat_bounds = (float(y_edges[iy]), float(y_edges[iy + 1]))
                lon_bounds = (float(x_edges[ix]), float(x_edges[ix + 1]))
                label = f"tile_y{iy + 1:02d}_x{ix + 1:02d}"
                specs.append(
                    DomainSpec(
                        group="tile_2x4",
                        label=label,
                        title=(
                            f"tile y{iy + 1}, x{ix + 1}\n"
                            f"lat {lat_bounds[0]:g} to {lat_bounds[1]:g}, "
                            f"lon {lon_bounds[0]:g} to {lon_bounds[1]:g}"
                        ),
                        lat_range=lat_bounds,
                        lon_range=lon_bounds,
                        row=iy,
                        col=ix,
                        n_rows=tile_y,
                        n_cols=tile_x,
                        lat_upper_inclusive=(iy == tile_y - 1),
                        lon_upper_inclusive=(ix == tile_x - 1),
                    )
                )

    if not specs:
        raise ValueError(f"No domain specs were built from --domain-modes={args.domain_modes!r}")
    return specs


def domain_args(args: argparse.Namespace, spec: DomainSpec) -> argparse.Namespace:
    out = argparse.Namespace(**vars(args))
    out.lat_range = range_arg(spec.lat_range)
    out.lon_range = range_arg(spec.lon_range)
    out.domain_group = spec.group
    out.domain_label = spec.label
    out.domain_title = spec.title
    out.domain_row = int(spec.row)
    out.domain_col = int(spec.col)
    out.domain_n_rows = int(spec.n_rows)
    out.domain_n_cols = int(spec.n_cols)
    out.domain_lat_upper_inclusive = bool(spec.lat_upper_inclusive)
    out.domain_lon_upper_inclusive = bool(spec.lon_upper_inclusive)
    return out


def current_domain_meta(args: argparse.Namespace) -> dict:
    return {
        "domain_group": str(getattr(args, "domain_group", "full")),
        "domain_label": str(getattr(args, "domain_label", "full")),
        "domain_title": str(getattr(args, "domain_title", "full")),
        "domain_row": int(getattr(args, "domain_row", 0)),
        "domain_col": int(getattr(args, "domain_col", 0)),
        "domain_n_rows": int(getattr(args, "domain_n_rows", 1)),
        "domain_n_cols": int(getattr(args, "domain_n_cols", 1)),
        "domain_lat_min": float(parse_range_pair(getattr(args, "lat_range", "-3,2"))[0]),
        "domain_lat_max": float(parse_range_pair(getattr(args, "lat_range", "-3,2"))[1]),
        "domain_lon_min": float(parse_range_pair(getattr(args, "lon_range", "121,131"))[0]),
        "domain_lon_max": float(parse_range_pair(getattr(args, "lon_range", "121,131"))[1]),
        "domain_lat_upper_inclusive": bool(getattr(args, "domain_lat_upper_inclusive", True)),
        "domain_lon_upper_inclusive": bool(getattr(args, "domain_lon_upper_inclusive", True)),
    }


def smooth_tag(smooth: float) -> str:
    s = f"{float(smooth):.6g}"
    return s.replace(".", "p").replace("-", "m")


def domain_path_parts(group: str, label: str) -> list[str]:
    group = str(group)
    label = str(label)
    if group == label:
        return [group]
    return [group, label]


def domain_file_prefix(group: str, label: str) -> str:
    parts = domain_path_parts(group, label)
    return "_".join(parts)


def month_file_token(value: float) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return str(value).replace(".", "p")


def expanded_grid_filename(year: int, month: int, lat_range: tuple[float, float], lon_range: tuple[float, float]) -> str:
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
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
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


def load_month_pickle(args: argparse.Namespace, year: int, lat_range: tuple[float, float], lon_range: tuple[float, float]):
    data_root = Path(args.data_root)
    month_dir = data_root / f"pickle_{year}"
    candidates = []
    if args.expanded_bounds:
        candidates.append(month_dir / expanded_grid_filename(year, args.month, lat_range, lon_range))
        legacy_lat_range = (-3.0, 7.0)
        legacy_lon_range = (111.0, 131.0)
        if tuple(lat_range) != legacy_lat_range or tuple(lon_range) != legacy_lon_range:
            candidates.append(month_dir / expanded_grid_filename(year, args.month, legacy_lat_range, legacy_lon_range))
    candidates.append(month_dir / default_grid_filename(year, args.month))
    for path in candidates:
        if path.exists():
            cache_key = str(path)
            if cache_key in MONTH_PICKLE_CACHE:
                obj = MONTH_PICKLE_CACHE[cache_key]
                print(f"Loaded {path} from in-process cache", flush=True)
            else:
                with open(path, "rb") as f:
                    obj = pickle.load(f)
                MONTH_PICKLE_CACHE[cache_key] = obj
                print(f"Loaded {path}", flush=True)
            if not isinstance(obj, dict):
                raise TypeError(f"Expected dict pickle at {path}, got {type(obj)}")
            return obj, path
    checked = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find monthly pickle. Checked:\n{checked}")


def filter_bounds(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    lat_upper_inclusive: bool = True,
    lon_upper_inclusive: bool = True,
):
    out = df.copy()
    out[y_col] = pd.to_numeric(out[y_col], errors="coerce")
    out[x_col] = pd.to_numeric(out[x_col], errors="coerce")
    lat_mask = out[y_col].ge(lat_range[0])
    lon_mask = out[x_col].ge(lon_range[0])
    if lat_upper_inclusive:
        lat_mask &= out[y_col].le(lat_range[1])
    else:
        lat_mask &= out[y_col].lt(lat_range[1])
    if lon_upper_inclusive:
        lon_mask &= out[x_col].le(lon_range[1])
    else:
        lon_mask &= out[x_col].lt(lon_range[1])
    mask = lat_mask & lon_mask
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
    return k, ox**2 + oy**2, oy, ox


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
        f = filter_bounds(
            df,
            args.y_col,
            args.x_col,
            lat_range,
            lon_range,
            lat_upper_inclusive=bool(getattr(args, "domain_lat_upper_inclusive", True)),
            lon_upper_inclusive=bool(getattr(args, "domain_lon_upper_inclusive", True)),
        )
        if f.empty:
            continue
        entries.append((sort_key, ts, str(key), f))
    entries.sort(key=lambda x: x[0])

    entries_by_day: dict[int, list[tuple[pd.Timestamp | None, str, pd.DataFrame]]] = {d: [] for d in days}
    for sort_key, ts, key, df in entries:
        day = int(ts.day) if ts is not None else int(sort_key[0])
        if day in entries_by_day:
            entries_by_day[day].append((ts, key, df))

    nonempty_days = [d for d, rows in entries_by_day.items() if rows]
    if not nonempty_days:
        raise ValueError(f"No data entries found for requested days={days} in {data_path}")

    first_day = nonempty_days[0]
    first_df = order_frame(entries_by_day[first_day][0][2], args.x_col, args.y_col)
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
    k_full, omega2_full, omega_lat_full, omega_lon_full = frequency_grid_for_axes(lat_vals, lon_vals)
    positive_k = k_full[np.isfinite(k_full) & (k_full > 0)]
    k_max_full = float(np.quantile(positive_k, float(args.radial_qmax)))
    radial_bins = np.linspace(0.0, k_max_full, int(args.radial_bins) + 1)

    print(
        f"Context year={year} month={args.month:02d}: days={nonempty_days[:3]}... "
        f"grid={len(lat_vals)}x{len(lon_vals)} n={len(grid_coords_full)} "
        f"lat_step={axis_step(lat_vals):.5g} lon_step={axis_step(lon_vals):.5g}",
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
        omega2_full=omega2_full,
        omega_lat_full=omega_lat_full,
        omega_lon_full=omega_lon_full,
        k_max_full=k_max_full,
        lat_step=axis_step(lat_vals),
        lon_step=axis_step(lon_vals),
    )


def hour_tensor(ctx: MonthContext, df: pd.DataFrame, hour_idx: int, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    ordered = order_frame(df, args.x_col, args.y_col)
    grid_lat = pd.to_numeric(ordered[args.y_col], errors="coerce").to_numpy(dtype=np.float64)
    grid_lon = pd.to_numeric(ordered[args.x_col], errors="coerce").to_numpy(dtype=np.float64)
    aligned = (
        ordered.assign(
            _lat_key=np.round(grid_lat, 10),
            _lon_key=np.round(grid_lon, 10),
        )
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
        h = np.round(h - 477700.0)
    else:
        h = np.full(len(ctx.grid_coords_full), float(hour_idx), dtype=np.float64)
    base = np.column_stack([lat, lon, y, h])
    dummies = np.zeros((len(ctx.grid_coords_full), 7), dtype=np.float64)
    if hour_idx > 0 and hour_idx <= 7:
        dummies[:, hour_idx - 1] = 1.0
    arr = np.column_stack([base, dummies])
    return torch.as_tensor(arr, dtype=DTYPE, device=device)


def parse_block_shape(text: str) -> tuple[int, int]:
    vals = [int(x.strip()) for x in str(text).lower().replace("x", ",").split(",") if x.strip()]
    if len(vals) != 2:
        raise ValueError(f"block shape must look like 4x4, got {text}")
    return vals[0], vals[1]


def maxmin_block_indices(ctx: MonthContext, block_prefix: int, block_shape_text: str) -> tuple[np.ndarray, dict]:
    block_y, block_x = parse_block_shape(block_shape_text)
    by_block: dict[tuple[int, int], list[int]] = {}
    for idx, (row, col) in enumerate(zip(ctx.local_to_row, ctx.local_to_col)):
        key = (int(row) // block_y, int(col) // block_x)
        by_block.setdefault(key, []).append(int(idx))
    block_keys = sorted(by_block)
    if not block_keys:
        raise ValueError("No grid blocks were built for max-min prefix selection.")
    centers = []
    for key in block_keys:
        idxs = np.asarray(by_block[key], dtype=np.int64)
        centers.append(ctx.grid_coords_full[idxs].mean(axis=0))
    centers_np = np.asarray(centers, dtype=np.float64)
    # maxmin_cpp expects lon/lat order, while grid_coords_full is lat/lon.
    lon_lat = np.column_stack([centers_np[:, 1], centers_np[:, 0]])
    order = np.asarray(orderings.maxmin_cpp(lon_lat), dtype=np.int64)
    if order.size and order.min() == 1 and order.max() == len(block_keys):
        order = order - 1
    if order.size != len(block_keys):
        raise RuntimeError(f"max-min block order length {order.size} != number of blocks {len(block_keys)}")
    requested = int(block_prefix)
    n_use = len(block_keys) if requested <= 0 else min(requested, len(block_keys))
    selected_blocks = [block_keys[int(i)] for i in order[:n_use]]
    idx = np.concatenate([np.asarray(by_block[key], dtype=np.int64) for key in selected_blocks])
    idx = np.asarray(sorted(set(int(i) for i in idx)), dtype=np.int64)
    density_scale = math.sqrt(float(n_use) / float(len(block_keys))) if len(block_keys) else 1.0
    effective_k_max = float(ctx.k_max_full) if requested <= 0 else float(ctx.k_max_full) * float(density_scale)
    meta = {
        "block_prefix_requested": int(requested),
        "block_prefix_used": int(n_use),
        "block_prefix_label": "all" if requested <= 0 else f"B{int(n_use)}",
        "n_blocks_full": int(len(block_keys)),
        "n_grid_selected": int(idx.size),
        "prefix_density_scale": float(density_scale),
        "effective_k_max": float(effective_k_max),
    }
    return idx, meta


def make_cluster_grid(ctx: MonthContext, block_prefix: int, args: argparse.Namespace):
    idx, meta = maxmin_block_indices(ctx, int(block_prefix), str(args.cluster_block_shape))
    coords_regular = np.ascontiguousarray(ctx.grid_coords_full[idx].astype(np.float64))
    return idx, coords_regular, meta


def count_valid(tensor: torch.Tensor) -> int:
    y = tensor[:, 2]
    return int((torch.isfinite(y) & torch.isfinite(tensor[:, 0]) & torch.isfinite(tensor[:, 1])).sum().item())


def space_diag(model) -> dict:
    if hasattr(model, "cluster_summary"):
        return dict(model.cluster_summary())
    groups = getattr(model, "Batched_Groups", []) or []
    if not groups:
        return {"n_batches": 0, "n_tails": 0, "mean_m": 0.0, "max_m": 0, "largest_batch_n": 0}
    ns = np.asarray([int(g["target_idx"].shape[0]) for g in groups], dtype=np.int64)
    ms = np.asarray([int(g["max_m"]) for g in groups], dtype=np.int64)
    return {
        "n_batches": int(len(groups)),
        "n_tails": int(ns.sum()),
        "mean_m": float(ms.mean()),
        "max_m": int(ms.max()),
        "largest_batch_n": int(ns.max()),
    }


def fit_anisotropic_fixed_smooth_from_batches(
    args: argparse.Namespace,
    batches: list[dict],
    n_features: int,
    y_var: float,
    smooth: float,
) -> dict:
    smooth_bounds = (max(1e-6, float(smooth) * 0.5), max(float(smooth) * 1.5, float(smooth) + 1e-6))
    fixed_smooth_raw = smooth_to_raw(float(smooth), smooth_bounds)
    range_bounds = (float(args.range_min), float(args.range_max))
    init = raw_from_natural(
        sigmasq=max(float(args.sigmasq_init), max(float(y_var), 1e-8)),
        range_lat=float(args.range_lat_init),
        range_lon=float(args.range_lon_init),
        smooth=float(smooth),
        nugget=0.0,
        nugget_mode="fixed0",
        smooth_bounds=smooth_bounds,
    )[:3]
    starts = [
        init,
        raw_from_natural(max(float(y_var), 1e-8), 0.25, 0.50, float(smooth), 0.0, "fixed0", smooth_bounds)[:3],
        raw_from_natural(max(float(y_var), 1e-8), 0.50, 0.25, float(smooth), 0.0, "fixed0", smooth_bounds)[:3],
    ][: max(1, int(getattr(args, "n_restarts", 1)))]

    log_phi2_bounds = (math.log(1.0 / range_bounds[1]), math.log(1.0 / range_bounds[0]))
    bounds = [(-40.0, 40.0), log_phi2_bounds, (-8.0, 8.0)]
    param_bounds = {
        "sigmasq": (1e-12, max(float(y_var) * 1e5, 1e-6)),
        "range_lat": range_bounds,
        "range_lon": range_bounds,
        "smooth": smooth_bounds,
        "nugget": (0.0, 0.0),
    }

    def objective(raw3: np.ndarray) -> float:
        raw4 = np.asarray([raw3[0], raw3[1], raw3[2], fixed_smooth_raw], dtype=np.float64)
        return profiled_vecchia_cluster_nll(
            raw4,
            batches=batches,
            n_features=int(n_features),
            nugget_mode="fixed0",
            fixed_nugget=0.0,
            smooth_bounds=smooth_bounds,
            param_bounds=param_bounds,
            jitter=1e-6,
        )

    best = None
    for start in starts:
        res = minimize(
            objective,
            np.asarray(start, dtype=np.float64),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": int(args.lbfgs_eval),
                "maxls": 20,
                "maxcor": int(args.lbfgs_history),
                "ftol": 1e-7,
            },
        )
        loss = float(res.fun) if np.isfinite(res.fun) else np.inf
        if best is None or loss < float(best["loss"]):
            raw4 = np.asarray([res.x[0], res.x[1], res.x[2], fixed_smooth_raw], dtype=np.float64)
            params = natural_from_raw(raw4, "fixed0", 0.0, smooth_bounds)
            rec = params.to_record()
            rec.update(
                {
                    "success": bool(np.isfinite(loss)),
                    "loss": loss,
                    "nll": loss,
                    "message": str(res.message),
                    "n_eval": int(getattr(res, "nfev", 0)),
                    "raw_params": raw4.tolist(),
                    "n_restarts": int(len(starts)),
                }
            )
            best = rec
    if best is None:
        raise RuntimeError("anisotropic fixed-smooth fit failed before producing a result")
    return best


def fit_hour_variant(
    args: argparse.Namespace,
    ctx: MonthContext,
    day: int,
    hour_idx: int,
    time_key: str,
    hour_t: torch.Tensor,
    block_prefix: int,
    variant: str,
    smooth: float,
    ordering_cache: dict,
    device: torch.device,
) -> tuple[dict, dict | None]:
    prefix_label = block_prefix_label(block_prefix)
    if block_prefix not in ordering_cache:
        ordering_cache[block_prefix] = make_cluster_grid(ctx, block_prefix, args)
    selected_idx, selected_grid, prefix_meta = ordering_cache[block_prefix]
    prefix_label = str(prefix_meta["block_prefix_label"])
    selected_t = hour_t[torch.as_tensor(selected_idx, dtype=torch.long, device=device)].contiguous()

    n_grid = int(selected_t.shape[0])
    n_valid = count_valid(selected_t)
    variant_info = VARIANTS.get(str(variant), {})
    base_row = {
        "date_str": f"{ctx.year}{ctx.month:02d}{day:02d}",
        "year": int(ctx.year),
        "month": int(ctx.month),
        "day": int(day),
        "hour_idx": int(hour_idx),
        "time_key": str(time_key),
        "smooth": float(smooth),
        "resolution_stride": int(prefix_meta["block_prefix_used"]),
        "resolution_label": str(prefix_meta["block_prefix_label"]),
        "block_prefix_requested": int(prefix_meta["block_prefix_requested"]),
        "block_prefix_used": int(prefix_meta["block_prefix_used"]),
        "block_prefix_label": str(prefix_meta["block_prefix_label"]),
        "n_blocks_full": int(prefix_meta["n_blocks_full"]),
        "n_grid_selected": int(prefix_meta["n_grid_selected"]),
        "prefix_density_scale": float(prefix_meta["prefix_density_scale"]),
        "effective_k_max": float(prefix_meta["effective_k_max"]),
        "variant": variant,
        "model_family": str(variant_info.get("family", "unknown")),
        "model_label": str(variant_info.get("plot_label", variant)),
        "gc_alpha": float(variant_info.get("gc_alpha", np.nan)),
        "gc_beta": float(variant_info.get("gc_beta", np.nan)),
        "mean_design": args.mean_design,
        "neighbors": int(args.cluster_neighbor_blocks),
        "cluster_block_shape": str(args.cluster_block_shape),
        "cluster_neighbor_blocks": int(args.cluster_neighbor_blocks),
        "n_grid": n_grid,
        "n_valid": n_valid,
        "valid_fraction": float(n_valid / n_grid) if n_grid else np.nan,
        **current_domain_meta(args),
    }

    try:
        y_np = selected_t[:, 2].detach().cpu().numpy().astype(np.float64)
        y_valid = y_np[np.isfinite(y_np)]
        y_var_raw = float(np.nanvar(y_valid, ddof=1)) if y_valid.size > 1 else 1e-8
        y_var = y_var_raw if np.isfinite(y_var_raw) and y_var_raw > 0 else 1e-8
        family = str(variant_info.get("family", "matern"))
        if family == "matern":
            model = ClusterSpaceIsoTrendVecchiaFit(
                smooth=0.5,
                input_map={time_key: selected_t},
                grid_coords=selected_grid,
                block_shape=parse_block_shape(args.cluster_block_shape),
                n_neighbor_blocks=int(args.cluster_neighbor_blocks),
                target_chunk_size=int(args.target_chunk_size),
                min_target_points=1,
                mean_design=args.mean_design,
            )
            t_pre = time.time()
            model.precompute_conditioning_sets()
            pre_s = time.time() - t_pre
            batches = vecchia_batches_to_numpy(model)
            t_fit = time.time()
            fit = fit_anisotropic_fixed_smooth_from_batches(
                args,
                batches=batches,
                n_features=int(model.n_features),
                y_var=y_var,
                smooth=float(variant_info.get("smooth", smooth)),
            )
            fit_s = time.time() - t_fit
            fit_iter = int(fit.get("n_eval", 0))
            fit_success = bool(fit.get("success", True))
            fit_message = str(fit.get("message", ""))
            raw_params = fit.get("raw_params", [])
            del batches
        elif family == "cauchy":
            model = ClusterSpaceAnisoCauchyFixedBetaNoNuggetTrendVecchiaFit(
                smooth=0.5,
                input_map={time_key: selected_t},
                grid_coords=selected_grid,
                block_shape=parse_block_shape(args.cluster_block_shape),
                n_neighbor_blocks=int(args.cluster_neighbor_blocks),
                target_chunk_size=int(args.target_chunk_size),
                min_target_points=1,
                mean_design=args.mean_design,
                gc_alpha=float(variant_info["gc_alpha"]),
                gc_beta=float(variant_info["gc_beta"]),
            )
            t_pre = time.time()
            model.precompute_conditioning_sets()
            pre_s = time.time() - t_pre
            init = cauchy_phi_init_from_natural(
                sigmasq=max(float(args.sigmasq_init), y_var),
                range_lat=float(args.range_lat_init),
                range_lon=float(args.range_lon_init),
                gc_beta=float(variant_info["gc_beta"]),
            )
            params_list = [
                torch.nn.Parameter(torch.tensor(math.log(float(init["phi1"])), dtype=DTYPE, device=device)),
                torch.nn.Parameter(torch.tensor(math.log(float(init["phi2"])), dtype=DTYPE, device=device)),
                torch.nn.Parameter(torch.tensor(math.log(float(init["phi3"])), dtype=DTYPE, device=device)),
            ]
            optimizer = model.set_optimizer(
                params_list,
                lr=1.0,
                max_iter=int(args.lbfgs_eval),
                max_eval=int(args.lbfgs_eval),
                tolerance_grad=float(args.grad_tol),
                history_size=int(args.lbfgs_history),
            )
            t_fit = time.time()
            raw_loss, fit_iter = model.fit_vecc_lbfgs(
                params_list,
                optimizer,
                max_steps=int(args.lbfgs_steps),
                grad_tol=float(args.grad_tol),
            )
            fit_s = time.time() - t_fit
            fit = model._convert_params(raw_loss[:-1])
            fit["loss"] = float(raw_loss[-1])
            fit_success = bool(np.isfinite(float(fit["loss"])))
            fit_message = ""
            raw_params = raw_loss[:-1]
        else:
            raise ValueError(f"Unknown spectral variant family {family!r} for variant={variant!r}")
        est = {
            "sigmasq": float(fit["sigmasq"]),
            "range": float(math.sqrt(float(fit["range_lat"]) * float(fit["range_lon"]))),
            "range_lat": float(fit["range_lat"]),
            "range_lon": float(fit["range_lon"]),
            "smooth": float(variant_info.get("smooth", smooth)),
            "nugget": 0.0,
            "phi1": float(fit["phi1"]),
            "phi2": float(fit["phi2"]),
            "phi3": float(fit["phi3"]),
            "model_family": family,
            "model_label": str(variant_info.get("plot_label", variant)),
            "gc_alpha": float(fit.get("gc_alpha", variant_info.get("gc_alpha", np.nan))),
            "gc_beta": float(fit.get("gc_beta", variant_info.get("gc_beta", np.nan))),
        }
        row = {
            **base_row,
            "status": "ok" if fit_success else "warn",
            "error": "" if fit_success else fit_message,
            "loss": float(fit["loss"]),
            "fit_iter_raw": int(fit_iter),
            "fit_steps_reported": int(fit_iter),
            "precompute_s": float(pre_s),
            "fit_s": float(fit_s),
            "total_s": float(pre_s + fit_s),
            "raw_params": str(raw_params),
            "est_sigmasq": float(est["sigmasq"]),
            "est_range": float(est["range"]),
            "est_range_lat": float(est["range_lat"]),
            "est_range_lon": float(est["range_lon"]),
            "est_smooth": float(est["smooth"]),
            "est_nugget": float(est["nugget"]),
            "est_phi1": float(est["phi1"]),
            "est_phi2": float(est["phi2"]),
            "est_phi3": float(est["phi3"]),
            "est_gc_alpha": float(est["gc_alpha"]),
            "est_gc_beta": float(est["gc_beta"]),
            **space_diag(model),
        }
        del model
        return row, est
    except Exception as exc:
        row = {
            **base_row,
            "status": "error",
            "error": repr(exc),
            "loss": np.nan,
            "fit_iter_raw": -1,
            "fit_steps_reported": 0,
            "precompute_s": np.nan,
            "fit_s": np.nan,
            "total_s": np.nan,
            "est_sigmasq": np.nan,
            "est_range": np.nan,
            "est_range_lat": np.nan,
            "est_range_lon": np.nan,
            "est_smooth": np.nan,
            "est_nugget": np.nan,
            "est_phi1": np.nan,
            "est_phi2": np.nan,
            "est_phi3": np.nan,
            "est_gc_alpha": np.nan,
            "est_gc_beta": np.nan,
            "n_batches": 0,
            "n_tails": 0,
            "mean_m": np.nan,
            "max_m": 0,
            "largest_batch_n": 0,
        }
        print(f"ERROR fit failed: {row['date_str']} h={hour_idx} {variant} {prefix_label}: {exc}", flush=True)
        return row, None
    finally:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def trend_design(lat, lon, mean_design: str, lat_center=None, lon_center=None):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat_center = np.nanmean(lat) if lat_center is None else float(lat_center)
    lon_center = np.nanmean(lon) if lon_center is None else float(lon_center)
    lat_c = lat - lat_center
    lon_c = lon - lon_center
    if mean_design in ("lat", "base"):
        return np.column_stack([np.ones(len(lat)), lat_c])
    return np.column_stack([np.ones(len(lat)), lat_c, lon_c])


def detrended_residual_grid(
    args: argparse.Namespace,
    ctx: MonthContext,
    hour_t: torch.Tensor,
    block_prefix: int,
    device: torch.device,
):
    selected_idx, _, _ = make_cluster_grid(ctx, int(block_prefix), args)
    arr = hour_t[torch.as_tensor(selected_idx, dtype=torch.long, device=device)].detach().cpu().numpy()
    y = arr[:, 2].astype(float)
    lat = arr[:, 0].astype(float)
    lon = arr[:, 1].astype(float)
    valid = np.isfinite(y) & np.isfinite(lat) & np.isfinite(lon)
    if valid.sum() < 4:
        raise ValueError(f"Not enough valid points for spectral grid, block_prefix={block_prefix_label(block_prefix)}")
    lat_center = np.nanmean(lat[valid])
    lon_center = np.nanmean(lon[valid])
    x = trend_design(lat[valid], lon[valid], args.mean_design, lat_center=lat_center, lon_center=lon_center)
    beta, *_ = np.linalg.lstsq(x, y[valid], rcond=None)
    x_all = trend_design(lat, lon, args.mean_design, lat_center=lat_center, lon_center=lon_center)
    resid = y - x_all @ beta

    rows_full = ctx.local_to_row[selected_idx]
    cols_full = ctx.local_to_col[selected_idx]
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
    lat_axis = ctx.lat_vals[row_keep].astype(float)
    lon_axis = ctx.lon_vals[col_keep].astype(float)
    return grid, mask, int(valid.sum()), lat_axis, lon_axis


def masked_periodogram(grid: np.ndarray, mask: np.ndarray, use_hann: bool):
    obs = mask > 0
    if not np.any(obs):
        raise ValueError("No observed cells in spectral grid.")
    z = np.zeros_like(grid, dtype=float)
    z[obs] = grid[obs]
    win = np.outer(np.hanning(z.shape[0]), np.hanning(z.shape[1])) if use_hann else np.ones_like(z)
    zw = z * win
    norm = np.sum((mask * win) ** 2)
    norm = norm if norm > EPS else 1.0
    return np.abs(np.fft.fftshift(np.fft.fft2(zw))) ** 2 / norm


def matern_covariance_lag(sigmasq, range_lat, range_lon, nugget, smooth, lag_lat, lag_lon):
    lag_lat = np.asarray(lag_lat, dtype=float)
    lag_lon = np.asarray(lag_lon, dtype=float)
    r = np.sqrt((lag_lat / max(float(range_lat), EPS)) ** 2 + (lag_lon / max(float(range_lon), EPS)) ** 2)
    nu = float(smooth)
    if np.isclose(nu, 0.5):
        corr = np.exp(-r)
    elif np.isclose(nu, 1.5):
        corr = (1.0 + r) * np.exp(-r)
    else:
        corr = np.ones_like(r, dtype=float)
        positive = r > EPS
        if np.any(positive):
            scaled = np.sqrt(2.0 * nu) * r[positive]
            corr[positive] = (2.0 ** (1.0 - nu) / gamma(nu)) * (scaled**nu) * kv(nu, scaled)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    zero_lag = (np.abs(lag_lat) < EPS) & (np.abs(lag_lon) < EPS)
    return float(sigmasq) * corr + max(float(nugget), 0.0) * zero_lag


def cauchy_covariance_lag(sigmasq, range_lat, range_lon, nugget, gc_alpha, gc_beta, lag_lat, lag_lon):
    lag_lat = np.asarray(lag_lat, dtype=float)
    lag_lon = np.asarray(lag_lon, dtype=float)
    scaled = np.sqrt(
        (lag_lat / max(float(range_lat), EPS)) ** 2
        + (lag_lon / max(float(range_lon), EPS)) ** 2
        + 1e-12
    )
    alpha = max(float(gc_alpha), EPS)
    beta = max(float(gc_beta), EPS)
    corr = (1.0 + scaled**alpha) ** (-beta / alpha)
    zero_lag = (np.abs(lag_lat) < EPS) & (np.abs(lag_lon) < EPS)
    return float(sigmasq) * corr + max(float(nugget), 0.0) * zero_lag


def covariance_lag_from_est(est: dict, lag_lat, lag_lon, nugget_override=None):
    nugget = float(est.get("nugget", 0.0) if nugget_override is None else nugget_override)
    family = str(est.get("model_family", "matern"))
    if family == "cauchy":
        return cauchy_covariance_lag(
            est["sigmasq"],
            est["range_lat"],
            est["range_lon"],
            nugget,
            est.get("gc_alpha", np.nan),
            est.get("gc_beta", np.nan),
            lag_lat,
            lag_lon,
        )
    return matern_covariance_lag(
        est["sigmasq"],
        est["range_lat"],
        est["range_lon"],
        nugget,
        est.get("smooth", 0.3),
        lag_lat,
        lag_lon,
    )


def mask_window_autocorrelation(mask: np.ndarray, use_hann: bool):
    win = np.outer(np.hanning(mask.shape[0]), np.hanning(mask.shape[1])) if use_hann else np.ones_like(mask)
    g = mask * win
    h_norm = float(np.sum(g**2))
    if h_norm <= EPS:
        raise ValueError("Mask/window normalization is zero.")
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


def covariance_lag_kernel(sigmasq, range_lat, range_lon, nugget, smooth, n1, n2, dlat, dlon):
    lag1_vals = np.arange(-(n1 - 1), n1, dtype=int)
    lag2_vals = np.arange(-(n2 - 1), n2, dtype=int)
    lag1, lag2 = np.meshgrid(lag1_vals, lag2_vals, indexing="ij")
    return matern_covariance_lag(sigmasq, range_lat, range_lon, nugget, smooth, lag1 * dlat, lag2 * dlon)


def covariance_lag_kernel_from_est(est: dict, n1, n2, dlat, dlon, nugget_override=None):
    lag1_vals = np.arange(-(n1 - 1), n1, dtype=int)
    lag2_vals = np.arange(-(n2 - 1), n2, dtype=int)
    lag1, lag2 = np.meshgrid(lag1_vals, lag2_vals, indexing="ij")
    return covariance_lag_from_est(est, lag1 * dlat, lag2 * dlon, nugget_override=nugget_override)


def fft_convolve_same(values, kernel):
    values = np.asarray(values, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    out_shape = (values.shape[0] + kernel.shape[0] - 1, values.shape[1] + kernel.shape[1] - 1)
    conv = np.fft.ifft2(np.fft.fft2(values, s=out_shape) * np.fft.fft2(kernel, s=out_shape)).real
    start0 = kernel.shape[0] // 2
    start1 = kernel.shape[1] // 2
    return conv[start0 : start0 + values.shape[0], start1 : start1 + values.shape[1]]


def expected_periodogram_dw_style(
    args: argparse.Namespace,
    sigmasq,
    range_lat,
    range_lon,
    nugget,
    smooth,
    mask,
    lat_axis,
    lon_axis,
    project_mean=True,
):
    n1, n2 = mask.shape
    ac, _ = mask_window_autocorrelation(mask, args.hann)
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
            cov_lag = matern_covariance_lag(sigmasq, range_lat, range_lon, nugget, smooth, lag1 * dlat, lag2 * dlon)
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
    x_flat = trend_design(
        lat_grid.ravel(),
        lon_grid.ravel(),
        args.mean_design,
        lat_center=float(np.nanmean(lat_grid[obs])),
        lon_center=float(np.nanmean(lon_grid[obs])),
    )
    x = x_flat.reshape(n1, n2, x_flat.shape[1])
    u = x * mask[..., None]
    xtx = np.einsum("ijp,ijq->pq", u, x)
    b = np.linalg.pinv(xtx)
    kernel = covariance_lag_kernel(sigmasq, range_lat, range_lon, nugget, smooth, n1, n2, dlat, dlon)
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


def expected_periodogram_from_est(
    args: argparse.Namespace,
    est: dict,
    mask,
    lat_axis,
    lon_axis,
    project_mean=True,
    nugget_override=None,
):
    n1, n2 = mask.shape
    ac, _ = mask_window_autocorrelation(mask, args.hann)
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
            cov_lag = covariance_lag_from_est(est, lag1 * dlat, lag2 * dlon, nugget_override=nugget_override)
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
    x_flat = trend_design(
        lat_grid.ravel(),
        lon_grid.ravel(),
        args.mean_design,
        lat_center=float(np.nanmean(lat_grid[obs])),
        lon_center=float(np.nanmean(lon_grid[obs])),
    )
    x = x_flat.reshape(n1, n2, x_flat.shape[1])
    u = x * mask[..., None]
    xtx = np.einsum("ijp,ijq->pq", u, x)
    b = np.linalg.pinv(xtx)
    kernel = covariance_lag_kernel_from_est(est, n1, n2, dlat, dlon, nugget_override=nugget_override)
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


def matern_spectrum_shape(sigmasq, range_lat, range_lon, smooth, omega_lat, omega_lon, nugget=0.0):
    nu = float(smooth)
    alpha = 2.0 * nu
    omega_scaled = (np.asarray(omega_lat, dtype=float) * max(float(range_lat), EPS)) ** 2 + (
        np.asarray(omega_lon, dtype=float) * max(float(range_lon), EPS)
    ) ** 2
    matern = float(sigmasq) * max(float(range_lat) * float(range_lon), EPS) * (alpha + omega_scaled) ** (-(nu + 1.0))
    return matern + max(float(nugget), 0.0)


def spectrum_shape_from_est(est: dict, ctx: MonthContext, nugget_override=None):
    nugget = float(est.get("nugget", 0.0) if nugget_override is None else nugget_override)
    if str(est.get("model_family", "matern")) == "matern":
        return matern_spectrum_shape(
            est["sigmasq"],
            est["range_lat"],
            est["range_lon"],
            est.get("smooth", 0.3),
            ctx.omega_lat_full,
            ctx.omega_lon_full,
            nugget=nugget,
        )
    n1 = len(ctx.lat_vals)
    n2 = len(ctx.lon_vals)
    kernel = covariance_lag_kernel_from_est(
        est,
        n1,
        n2,
        ctx.lat_step,
        ctx.lon_step,
        nugget_override=nugget,
    )
    center0 = kernel.shape[0] // 2
    center1 = kernel.shape[1] // 2
    embedded = np.zeros((n1, n2), dtype=float)
    embedded[: n1 - center0, : n2 - center1] = kernel[center0:, center1:]
    return np.maximum(np.fft.fftshift(np.fft.fft2(embedded).real), EPS)


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
        rows.append(
            {
                "k_bin": int(b),
                "k_mid": float(0.5 * (radial_bins[b] + radial_bins[b + 1])),
                "k_mean": float(np.nanmean(kg[m])),
                "spectrum": float(np.nanmean(vg[m])),
                "n_freq": int(m.sum()),
            }
        )
    return pd.DataFrame(rows)


DIRECTION_SPECS = {
    "radial": {
        "label": "norm",
        "frequency_label": "norm frequency",
    },
    "lat": {
        "label": "latitude N-S",
        "frequency_label": "lat norm frequency",
    },
    "lon": {
        "label": "longitude E-W",
        "frequency_label": "lon norm frequency",
    },
    "diag": {
        "label": "diagonal NE-SW",
        "frequency_label": "diagonal norm frequency",
    },
}


def min_positive_step(values: np.ndarray) -> float:
    vals = np.unique(np.round(np.abs(np.asarray(values, dtype=float).ravel()), 12))
    vals = vals[np.isfinite(vals) & (vals > EPS)]
    if len(vals) <= 1:
        return 0.0
    return float(np.nanmin(np.diff(np.sort(vals))))


def directional_average(surface, omega_lat, omega_lon, radial_bins, k_max, direction: str):
    vals = np.asarray(surface, dtype=float).ravel()
    wy = np.asarray(omega_lat, dtype=float)
    wx = np.asarray(omega_lon, dtype=float)
    if direction == "lat":
        coord = np.abs(wy)
        perp = np.abs(wx)
        step = min_positive_step(wx)
    elif direction == "lon":
        coord = np.abs(wx)
        perp = np.abs(wy)
        step = min_positive_step(wy)
    elif direction == "diag":
        coord = np.abs((wy + wx) / np.sqrt(2.0))
        perp = np.abs((wy - wx) / np.sqrt(2.0))
        step = min(min_positive_step(wx), min_positive_step(wy))
    else:
        raise ValueError(f"Unknown direction {direction!r}")

    coord_flat = coord.ravel()
    perp_flat = perp.ravel()
    bin_width = float(radial_bins[1] - radial_bins[0]) if len(radial_bins) > 1 else 0.0
    band_tol = max(0.5 * step, 0.5 * bin_width, EPS)
    good = (
        np.isfinite(vals)
        & np.isfinite(coord_flat)
        & np.isfinite(perp_flat)
        & (coord_flat > 0)
        & (coord_flat <= k_max)
        & (perp_flat <= band_tol)
    )
    if not np.any(good):
        return pd.DataFrame(columns=["k_bin", "k_mid", "k_mean", "spectrum", "n_freq"])

    bin_idx = np.digitize(coord_flat[good], radial_bins, right=False) - 1
    valid_bin = (bin_idx >= 0) & (bin_idx < len(radial_bins) - 1)
    bin_idx = bin_idx[valid_bin]
    vg = vals[good][valid_bin]
    kg = coord_flat[good][valid_bin]
    rows = []
    for b in range(len(radial_bins) - 1):
        m = bin_idx == b
        if not np.any(m):
            continue
        rows.append(
            {
                "k_bin": int(b),
                "k_mid": float(0.5 * (radial_bins[b] + radial_bins[b + 1])),
                "k_mean": float(np.nanmean(kg[m])),
                "spectrum": float(np.nanmean(vg[m])),
                "n_freq": int(m.sum()),
            }
        )
    return pd.DataFrame(rows)


def profile_average(surface, k_radial, omega_lat, omega_lon, radial_bins, k_max, direction: str):
    if direction == "radial":
        return radial_average(surface, k_radial, radial_bins, k_max)
    return directional_average(surface, omega_lat, omega_lon, radial_bins, k_max, direction)


def compute_spectrum_rows(
    args: argparse.Namespace,
    ctx: MonthContext,
    fit_row: dict,
    est: dict,
    hour_t: torch.Tensor,
    block_prefix: int,
    smooth: float,
    device: torch.device,
) -> list[dict]:
    grid, mask, n_valid_spectrum, lat_axis, lon_axis = detrended_residual_grid(args, ctx, hour_t, block_prefix, device)
    data_p = masked_periodogram(grid, mask, args.hann)
    k_data, _, omega_lat_data, omega_lon_data = frequency_grid_for_axes(lat_axis, lon_axis)
    expected_p = expected_periodogram_from_est(args, est, mask, lat_axis, lon_axis)
    expected_latent_p = expected_periodogram_from_est(args, est, mask, lat_axis, lon_axis, nugget_override=0.0)
    shape_latent_p = spectrum_shape_from_est(est, ctx, nugget_override=0.0)
    shape_observed_p = spectrum_shape_from_est(est, ctx, nugget_override=est["nugget"])

    directions = ["radial", "lat", "lon", "diag"]
    rows = []
    base_keys = [
        "date_str",
        "year",
        "month",
        "day",
        "hour_idx",
        "time_key",
        "smooth",
        "resolution_stride",
        "resolution_label",
        "block_prefix_requested",
        "block_prefix_used",
        "block_prefix_label",
        "n_blocks_full",
        "n_grid_selected",
        "prefix_density_scale",
        "effective_k_max",
        "variant",
        "model_family",
        "model_label",
        "gc_alpha",
        "gc_beta",
        "mean_design",
        "neighbors",
        "est_sigmasq",
        "est_range",
        "est_range_lat",
        "est_range_lon",
        "est_smooth",
        "est_nugget",
        "est_phi1",
        "est_phi2",
        "est_phi3",
        "est_gc_alpha",
        "est_gc_beta",
        "domain_group",
        "domain_label",
        "domain_title",
        "domain_row",
        "domain_col",
        "domain_n_rows",
        "domain_n_cols",
        "domain_lat_min",
        "domain_lat_max",
        "domain_lon_min",
        "domain_lon_max",
    ]
    for direction in directions:
        spec = DIRECTION_SPECS[direction]
        data_prof = profile_average(
            data_p, k_data, omega_lat_data, omega_lon_data, ctx.radial_bins, ctx.k_max_full, direction
        ).rename(columns={"spectrum": "data_spectrum"})
        expected_prof = profile_average(
            expected_p, k_data, omega_lat_data, omega_lon_data, ctx.radial_bins, ctx.k_max_full, direction
        ).rename(columns={"spectrum": "theory_spectrum_expected"})
        expected_latent_prof = profile_average(
            expected_latent_p, k_data, omega_lat_data, omega_lon_data, ctx.radial_bins, ctx.k_max_full, direction
        ).rename(columns={"spectrum": "theory_spectrum_expected_latent"})
        shape_prof = profile_average(
            shape_latent_p,
            ctx.k_full_radial,
            ctx.omega_lat_full,
            ctx.omega_lon_full,
            ctx.radial_bins,
            ctx.k_max_full,
            direction,
        ).rename(columns={"spectrum": "theory_spectrum_continuous"})
        shape_obs_prof = profile_average(
            shape_observed_p,
            ctx.k_full_radial,
            ctx.omega_lat_full,
            ctx.omega_lon_full,
            ctx.radial_bins,
            ctx.k_max_full,
            direction,
        ).rename(columns={"spectrum": "theory_spectrum_continuous_observed"})
        if shape_prof.empty and expected_prof.empty and data_prof.empty:
            continue
        merged = shape_prof[["k_bin", "k_mid", "k_mean", "theory_spectrum_continuous"]].merge(
            shape_obs_prof[["k_bin", "theory_spectrum_continuous_observed"]],
            on="k_bin",
            how="outer",
        )
        merged = merged.merge(
            expected_prof[["k_bin", "theory_spectrum_expected", "n_freq"]], on="k_bin", how="outer"
        )
        merged = merged.merge(
            expected_latent_prof[["k_bin", "theory_spectrum_expected_latent"]],
            on="k_bin",
            how="outer",
        )
        merged = merged.merge(data_prof[["k_bin", "data_spectrum"]], on="k_bin", how="outer")
        for col in ("k_mid", "k_mean"):
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce")
        if "k_mid" in merged.columns and merged["k_mid"].isna().any():
            fallback = 0.5 * (ctx.radial_bins[:-1] + ctx.radial_bins[1:])
            missing = merged["k_mid"].isna() & merged["k_bin"].notna()
            merged.loc[missing, "k_mid"] = [float(fallback[int(b)]) for b in merged.loc[missing, "k_bin"]]
        raw_data_k_max = float(data_prof["k_mid"].max()) if not data_prof.empty else np.nan
        effective_k_max = float(fit_row.get("effective_k_max", raw_data_k_max))
        data_k_max = min(raw_data_k_max, effective_k_max) if np.isfinite(raw_data_k_max) else effective_k_max
        for m in merged.sort_values("k_bin").itertuples(index=False):
            rows.append(
                {
                    **{k: fit_row[k] for k in base_keys},
                    "profile": direction,
                    "direction": direction,
                    "direction_label": spec["label"],
                    "frequency_label": spec["frequency_label"],
                    "n_valid_spectrum": int(n_valid_spectrum),
                    "k_bin": int(m.k_bin),
                    "k_mid": float(m.k_mid) if pd.notna(m.k_mid) else np.nan,
                    "k_mean": float(m.k_mean) if pd.notna(m.k_mean) else np.nan,
                    "n_freq": int(m.n_freq) if pd.notna(m.n_freq) else 0,
                    "data_k_max": data_k_max,
                    "data_spectrum": float(m.data_spectrum) if pd.notna(m.data_spectrum) else np.nan,
                    "theory_spectrum_expected": float(m.theory_spectrum_expected) if pd.notna(m.theory_spectrum_expected) else np.nan,
                    "theory_spectrum_expected_latent": float(m.theory_spectrum_expected_latent) if pd.notna(m.theory_spectrum_expected_latent) else np.nan,
                    "theory_spectrum_continuous": float(m.theory_spectrum_continuous) if pd.notna(m.theory_spectrum_continuous) else np.nan,
                    "theory_spectrum_continuous_observed": float(m.theory_spectrum_continuous_observed) if pd.notna(m.theory_spectrum_continuous_observed) else np.nan,
                }
        )
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


def ratio_frame(numerator_df, denominator_df, numerator_col, denominator_col, normalize_mean=False, sigma_sq=None, k_max=None):
    if numerator_df.empty or denominator_df.empty:
        return pd.DataFrame(columns=["k_bin", "k_mid", "ratio"])
    left = numerator_df[["k_bin", "k_mid", numerator_col]].copy()
    right = denominator_df[["k_bin", denominator_col]].copy()
    merged = left.merge(right, on="k_bin", how="inner").replace([np.inf, -np.inf], np.nan)
    if k_max is not None and np.isfinite(float(k_max)) and float(k_max) > 0:
        merged = merged[pd.to_numeric(merged["k_mid"], errors="coerce") <= float(k_max)].copy()
    good = merged[numerator_col].notna() & merged[denominator_col].notna() & (merged[numerator_col] > 0) & (merged[denominator_col] > 0)
    out = merged.loc[good, ["k_bin", "k_mid"]].copy()
    raw_ratio = merged.loc[good, numerator_col].to_numpy(dtype=float) / merged.loc[good, denominator_col].to_numpy(dtype=float)
    scale = float(np.nanmean(raw_ratio)) if normalize_mean and len(raw_ratio) else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    out["ratio_raw"] = raw_ratio
    out["ratio_scale_mean"] = scale
    out["ratio"] = raw_ratio / scale if normalize_mean else raw_ratio
    if sigma_sq is not None and np.isfinite(float(sigma_sq)):
        out["sigma_profile"] = float(sigma_sq) * scale
    else:
        out["sigma_profile"] = np.nan
    return out.sort_values("k_bin")


def median_sigmasq(source_df, variant, resolution_label):
    required = {"variant", "resolution_label", "est_sigmasq"}
    if source_df is None or source_df.empty or not required.issubset(source_df.columns):
        return np.nan
    df = source_df[
        (source_df["variant"].astype(str) == str(variant))
        & (source_df["resolution_label"].astype(str) == str(resolution_label))
    ].copy()
    if df.empty:
        return np.nan
    key_cols = [c for c in ["date_str", "hour_idx", "variant", "resolution_label"] if c in df.columns]
    if key_cols:
        df = df.drop_duplicates(key_cols)
    return float(pd.to_numeric(df["est_sigmasq"], errors="coerce").median())


def profile_ratio_label(ratio_df):
    if ratio_df is None or ratio_df.empty or "ratio_scale_mean" not in ratio_df.columns:
        return None
    scale = float(pd.to_numeric(ratio_df["ratio_scale_mean"], errors="coerce").dropna().iloc[0])
    sigma_profile = np.nan
    if "sigma_profile" in ratio_df.columns and ratio_df["sigma_profile"].notna().any():
        sigma_profile = float(pd.to_numeric(ratio_df["sigma_profile"], errors="coerce").dropna().iloc[0])
    label = f"ratio mean scale={scale:.3g}"
    if np.isfinite(sigma_profile):
        label += f"\nprofile sigma^2={sigma_profile:.3g}"
    return label


def label_with_loss(model_label: str, loss: float | None) -> str:
    if loss is None or not np.isfinite(float(loss)):
        return str(model_label)
    return f"{model_label} loss={float(loss):.5f}"


def cutoff_from_frame(df: pd.DataFrame, fallback: float) -> float:
    if df is not None and not df.empty and "data_k_max" in df.columns and df["data_k_max"].notna().any():
        val = float(pd.to_numeric(df["data_k_max"], errors="coerce").dropna().iloc[0])
        if np.isfinite(val) and val > 0:
            return val
    return float(fallback)


def keep_to_cutoff(df: pd.DataFrame, k_cut: float) -> pd.DataFrame:
    if df is None or df.empty or "k_mid" not in df.columns:
        return df
    if not np.isfinite(float(k_cut)) or float(k_cut) <= 0:
        return df
    return df[pd.to_numeric(df["k_mid"], errors="coerce") <= float(k_cut)].copy()


def add_ratio_axis(ax, ratio_df, ylabel=None, color="tab:blue"):
    if ratio_df is None or ratio_df.empty:
        return None
    r = ratio_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["k_mid", "ratio"])
    r = r[r["ratio"] > 0].sort_values("k_bin")
    if r.empty:
        return None
    ratio_ax = ax.twinx()
    ratio_ax.plot(r["k_mid"], r["ratio"], color=color, linewidth=1.35, linestyle=":", alpha=0.95, zorder=8)
    ratio_ax.axhline(1.0, color=color, linewidth=0.8, linestyle="-", alpha=0.35, zorder=7)
    ratio_ax.set_yscale("log")
    vals = r["ratio"].to_numpy(dtype=float)
    lo = max(1e-3, min(0.5, float(np.nanmin(vals)) / 1.2))
    hi = min(1e3, max(2.0, float(np.nanmax(vals)) * 1.2))
    if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
        ratio_ax.set_ylim(lo, hi)
    ratio_ax.tick_params(axis="y", colors=color, labelsize=7)
    ratio_ax.spines["right"].set_color(color)
    ratio_ax.grid(False)
    if ylabel:
        ratio_ax.set_ylabel(ylabel, color=color, fontsize=8)
    return ratio_ax


def format_fit_label(source_df, variant, resolution_label):
    required = {"variant", "resolution_label", "est_sigmasq", "est_range_lat", "est_range_lon", "est_nugget"}
    if source_df is None or source_df.empty or not required.issubset(source_df.columns):
        return None
    df = source_df[
        (source_df["variant"].astype(str) == str(variant))
        & (source_df["resolution_label"].astype(str) == str(resolution_label))
    ].copy()
    if df.empty:
        return None
    key_cols = [c for c in ["date_str", "hour_idx", "variant", "resolution_label"] if c in df.columns]
    if key_cols:
        df = df.drop_duplicates(key_cols)
    sigmasq = float(pd.to_numeric(df["est_sigmasq"], errors="coerce").median())
    range_lat = float(pd.to_numeric(df["est_range_lat"], errors="coerce").median())
    range_lon = float(pd.to_numeric(df["est_range_lon"], errors="coerce").median())
    phi3 = float(pd.to_numeric(df.get("est_phi3", pd.Series(np.nan, index=df.index)), errors="coerce").median())
    nugget = float(pd.to_numeric(df["est_nugget"], errors="coerce").median())
    gc_alpha = float(pd.to_numeric(df.get("est_gc_alpha", pd.Series(np.nan, index=df.index)), errors="coerce").median())
    gc_beta = float(pd.to_numeric(df.get("est_gc_beta", pd.Series(np.nan, index=df.index)), errors="coerce").median())
    loss = float(pd.to_numeric(df.get("loss", pd.Series(np.nan, index=df.index)), errors="coerce").median())
    if not np.isfinite(sigmasq) or not np.isfinite(range_lat) or not np.isfinite(range_lon):
        return None
    label = f"fit median: sigma^2={sigmasq:.3g}\nrange_lat={range_lat:.3g}, range_lon={range_lon:.3g}"
    if np.isfinite(loss):
        label += f"\nloss={loss:.5f}"
    if np.isfinite(phi3):
        label += f"\nphi3={phi3:.3g}"
    if np.isfinite(gc_alpha) and np.isfinite(gc_beta):
        label += f"\nGC a={gc_alpha:.3g}, b={gc_beta:.3g}"
    if np.isfinite(nugget) and abs(nugget) > EPS:
        label += f"\nnugget={nugget:.3g}"
    return label


def median_loss_for_domain(
    args: argparse.Namespace,
    year: int,
    smooth: float,
    spec: DomainSpec,
    variant: str,
    resolution_label: str,
) -> float:
    out_dir = output_dir_for_domain(args, smooth, year, spec)
    frames = []
    for path in sorted((out_dir / "daily_csv").glob("*_fits.csv")):
        try:
            frames.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            continue
    if not frames:
        return np.nan
    df = pd.concat(frames, ignore_index=True)
    if "loss" not in df.columns or "variant" not in df.columns or "resolution_label" not in df.columns:
        return np.nan
    mask = (
        (df["variant"].astype(str) == str(variant))
        & (df["resolution_label"].astype(str) == str(resolution_label))
    )
    if "status" in df.columns:
        mask &= df["status"].astype(str).isin(["ok", "warn"])
    vals = pd.to_numeric(df.loc[mask, "loss"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.median()) if not vals.empty else np.nan


def ensure_profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "profile" not in out.columns:
        out["profile"] = "radial"
    if "direction" not in out.columns:
        out["direction"] = out["profile"].astype(str)
    if "direction_label" not in out.columns:
        out["direction_label"] = out["direction"].map(lambda x: DIRECTION_SPECS.get(str(x), DIRECTION_SPECS["radial"])["label"])
    if "frequency_label" not in out.columns:
        out["frequency_label"] = out["direction"].map(lambda x: DIRECTION_SPECS.get(str(x), DIRECTION_SPECS["radial"])["frequency_label"])
    if "theory_spectrum_continuous_observed" not in out.columns:
        out["theory_spectrum_continuous_observed"] = out.get("theory_spectrum_continuous", np.nan)
    if "theory_spectrum_expected_latent" not in out.columns:
        out["theory_spectrum_expected_latent"] = out.get("theory_spectrum_expected", np.nan)
    return out


def aggregate_daily(spectral_df: pd.DataFrame):
    spectral_df = ensure_profile_columns(spectral_df)
    group_cols = [
        "profile",
        "direction",
        "direction_label",
        "frequency_label",
        "variant",
        "resolution_label",
        "resolution_stride",
        "k_bin",
    ]
    avg_data = (
        spectral_df.dropna(subset=["data_spectrum"])
        .groupby(group_cols, observed=True)
        .agg(
            k_mid=("k_mid", "mean"),
            data_spectrum=("data_spectrum", "mean"),
            n_hours=("hour_idx", "nunique"),
            data_k_max=("data_k_max", "mean"),
        )
        .reset_index()
    )
    avg_theory = (
        spectral_df.dropna(subset=["theory_spectrum_expected"])
        .groupby(group_cols, observed=True)
        .agg(
            k_mid=("k_mid", "mean"),
            theory_spectrum_expected=("theory_spectrum_expected", "mean"),
            theory_spectrum_expected_latent=("theory_spectrum_expected_latent", "mean"),
            theory_spectrum_continuous=("theory_spectrum_continuous", "mean"),
            theory_spectrum_continuous_observed=("theory_spectrum_continuous_observed", "mean"),
        )
        .reset_index()
    )
    return avg_data, avg_theory


def plot_daily(args, ctx: MonthContext, smooth: float, day: int, spectral_df: pd.DataFrame, out_path: Path):
    if spectral_df.empty:
        return
    spectral_df = ensure_profile_columns(spectral_df)
    spectral_df = spectral_df[spectral_df["profile"].astype(str) == "radial"].copy()
    if spectral_df.empty:
        return
    labels_order = [block_prefix_label(s) for s in parse_block_prefixes(args.block_prefixes)]
    avg_data, avg_theory = aggregate_daily(spectral_df)
    ylim = positive_ylim(avg_data.get("data_spectrum"), avg_theory.get("theory_spectrum_expected"))
    row_specs = [(v, VARIANTS[v]["row_title"]) for v in variants_for_year(ctx.year, parse_names(args.variants))]
    fig, axes = plt.subplots(len(row_specs), len(labels_order), figsize=(4.4 * len(labels_order), 3.4 * len(row_specs)), sharey=True)
    axes = np.asarray(axes).reshape(len(row_specs), len(labels_order))
    for i, (variant, row_title) in enumerate(row_specs):
        for j, label in enumerate(labels_order):
            ax = axes[i, j]
            sub_data = avg_data[
                (avg_data["variant"] == variant)
                & (avg_data["resolution_label"].astype(str) == label)
                & (avg_data["k_mid"] > 0)
            ]
            sub_theory = avg_theory[
                (avg_theory["variant"] == variant)
                & (avg_theory["resolution_label"].astype(str) == label)
                & (avg_theory["k_mid"] > 0)
            ]
            hour_sub = spectral_df[
                (spectral_df["variant"] == variant)
                & (spectral_df["resolution_label"].astype(str) == label)
                & (spectral_df["k_mid"] > 0)
                & spectral_df["data_spectrum"].notna()
            ]
            if sub_data.empty or sub_theory.empty:
                ax.set_visible(False)
                continue
            k_cut = cutoff_from_frame(sub_data, sub_data["k_mid"].max())
            sub_data = keep_to_cutoff(sub_data, k_cut)
            sub_theory = keep_to_cutoff(sub_theory, k_cut)
            hour_sub = keep_to_cutoff(hour_sub, k_cut)
            if sub_data.empty or sub_theory.empty:
                ax.set_visible(False)
                continue
            param_label = format_fit_label(spectral_df, variant, label)
            if param_label:
                ax.plot([], [], color="none", label=param_label)
            ax.plot(sub_theory["k_mid"], sub_theory["theory_spectrum_expected"], color="tab:red", linewidth=1.9, linestyle="--", label="expected periodogram (mean over 8 h)", zorder=3)
            for h_i, (_, hs) in enumerate(hour_sub.groupby("hour_idx")):
                hour_label = "hourly data spectra" if (i == 0 and j == 0 and h_i == 0) else None
                ax.plot(hs["k_mid"], hs["data_spectrum"], color="0.35", alpha=0.55, linewidth=1.05, label=hour_label, zorder=1)
            ax.plot(sub_data["k_mid"], sub_data["data_spectrum"], color="black", linewidth=2.2, label="data residual spectrum (mean over 8 h)", zorder=4)
            sigma_sq = median_sigmasq(spectral_df, variant, label)
            ratio_df = ratio_frame(
                sub_data,
                sub_theory,
                "data_spectrum",
                "theory_spectrum_expected",
                normalize_mean=True,
                sigma_sq=sigma_sq,
                k_max=k_cut,
            )
            ratio_label = profile_ratio_label(ratio_df)
            if ratio_label:
                ax.plot([], [], color="none", label=ratio_label)
            add_ratio_axis(ax, ratio_df, ylabel="I / E[I]" if j == len(labels_order) - 1 else None)
            ax.plot([], [], color="tab:blue", linewidth=1.35, linestyle=":", label="profiled I / E[I] (mean=1)")
            ax.set_xlim(0, ctx.k_max_full)
            ax.set_ylim(*ylim)
            ax.set_title(f"{row_title}, {label}  (data k <= {k_cut:.1f})")
            ax.set_xlabel("norm frequency")
            if j == 0:
                ax.set_ylabel("spectrum")
            ax.set_yscale("log")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7, handlelength=1.5)
    fig.suptitle(f"{ctx.year}-{ctx.month:02d}-{day:02d}, smooth={smooth}: norm-frequency residual spectrum vs fitted expected periodogram")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_names(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def variants_for_year(year: int, requested: list[str] | None = None) -> list[str]:
    requested = [v for v in (requested or list(VARIANTS)) if v in VARIANTS]
    allowed = YEAR_VARIANTS.get(int(year), requested)
    return [v for v in allowed if v in requested]


def write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format=f"%.{ROUND_DECIMALS}f")


def existing_day_has_requested_variants(
    fit_path: Path,
    spec_path: Path,
    plot_path: Path,
    variants: list[str],
    labels: list[str],
) -> bool:
    if not (fit_path.exists() and spec_path.exists() and plot_path.exists()):
        return False
    try:
        fit_df = pd.read_csv(fit_path)
        spec_df = pd.read_csv(spec_path)
    except Exception:
        return False
    if fit_df.empty or spec_df.empty or "variant" not in fit_df.columns or "variant" not in spec_df.columns:
        return False
    fit_variants = set(fit_df["variant"].astype(str))
    spec_variants = set(spec_df["variant"].astype(str))
    missing_variants = [v for v in variants if v not in fit_variants or v not in spec_variants]
    if missing_variants:
        print(
            f"Existing day files are incomplete; missing variants {missing_variants}. Recomputing {fit_path.parent.parent.name}.",
            flush=True,
        )
        return False
    if "resolution_label" in fit_df.columns and "resolution_label" in spec_df.columns:
        fit_pairs = set(zip(fit_df["variant"].astype(str), fit_df["resolution_label"].astype(str)))
        spec_pairs = set(zip(spec_df["variant"].astype(str), spec_df["resolution_label"].astype(str)))
        missing_pairs = [(v, label) for v in variants for label in labels if (v, label) not in fit_pairs or (v, label) not in spec_pairs]
        if missing_pairs:
            print(
                f"Existing day files are incomplete; missing variant/domain pairs {missing_pairs[:6]}. Recomputing {fit_path.parent.parent.name}.",
                flush=True,
            )
            return False
    return True


def process_day(
    args: argparse.Namespace,
    ctx: MonthContext,
    smooth: float,
    day: int,
    out_dir: Path,
    device: torch.device,
):
    date_str = f"{ctx.year}{ctx.month:02d}{day:02d}"
    daily_dir = out_dir / "daily_csv"
    plot_dir = out_dir / "daily_plots"
    fit_path = daily_dir / f"{date_str}_fits.csv"
    spec_path = daily_dir / f"{date_str}_spectral_profiles.csv"
    plot_path = plot_dir / f"{date_str}_data_vs_expected_periodogram.png"
    block_prefixes = parse_block_prefixes(args.block_prefixes)
    labels_order = [block_prefix_label(s) for s in block_prefixes]
    variants = variants_for_year(ctx.year, parse_names(args.variants))
    if args.skip_existing and existing_day_has_requested_variants(fit_path, spec_path, plot_path, variants, labels_order):
        print(f"Skip existing day {date_str}", flush=True)
        return
    entries = ctx.entries_by_day.get(day, [])
    if len(entries) < 8:
        print(f"WARNING: {date_str} has {len(entries)} hourly entries; expected 8. Processing available hours.", flush=True)
    fit_rows = []
    spectral_rows = []
    ordering_cache = {}

    for hour_idx, (ts, key, df) in enumerate(entries[:8]):
        hour_t = hour_tensor(ctx, df, hour_idx, args, device)
        for block_prefix in block_prefixes:
            label = block_prefix_label(block_prefix)
            for variant in variants:
                print(f"\n{date_str} smooth={smooth} hour={hour_idx} {variant} {label}", flush=True)
                fit_row, est = fit_hour_variant(
                    args, ctx, day, hour_idx, key, hour_t, block_prefix, variant, smooth, ordering_cache, device
                )
                fit_rows.append(fit_row)
                if est is None:
                    continue
                try:
                    spectral_rows.extend(compute_spectrum_rows(args, ctx, fit_row, est, hour_t, block_prefix, smooth, device))
                except Exception as exc:
                    print(f"ERROR spectrum failed: {date_str} h={hour_idx} {variant} {label}: {exc}", flush=True)

    fit_df = pd.DataFrame(fit_rows)
    spec_df = pd.DataFrame(spectral_rows)
    write_csv(fit_df, fit_path)
    write_csv(spec_df, spec_path)
    fit_variants = sorted(fit_df["variant"].astype(str).unique()) if "variant" in fit_df.columns else []
    spec_variants = sorted(spec_df["variant"].astype(str).unique()) if "variant" in spec_df.columns else []
    print(f"{date_str} requested variants={variants}", flush=True)
    print(f"{date_str} fit variants={fit_variants}", flush=True)
    print(f"{date_str} spectral variants={spec_variants}", flush=True)
    missing_fit = [v for v in variants if v not in fit_variants]
    missing_spec = [v for v in variants if v not in spec_variants]
    if missing_fit or missing_spec:
        raise RuntimeError(
            f"{date_str} missing requested variants after processing: "
            f"fit missing={missing_fit}, spectral missing={missing_spec}. "
            f"Check that the Amarel script/slurm is the current GC-enabled version."
        )
    if not spec_df.empty:
        plot_daily(args, ctx, smooth, day, spec_df, plot_path)
    print(f"Saved day {date_str}: {fit_path}, {spec_path}, {plot_path}", flush=True)


def publish_monthly_plot(args: argparse.Namespace, ctx: MonthContext, smooth: float, plot_path: Path):
    top_dir = Path(args.top_plot_dir) if str(args.top_plot_dir).strip() else Path(args.output_root) / "monthly_plots_top"
    year_dir = top_dir / f"{ctx.year}_{ctx.month:02d}"
    year_dir.mkdir(parents=True, exist_ok=True)
    domain_prefix = domain_file_prefix(
        getattr(args, "domain_group", "full"),
        getattr(args, "domain_label", "full"),
    )
    dest = year_dir / f"smooth_{smooth_tag(smooth)}_{domain_prefix}_{plot_path.name}"
    shutil.copy2(plot_path, dest)
    print(f"Copied monthly plot to top folder: {dest}", flush=True)


def add_continuous_scaled_to_expected(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["theory_spectrum_continuous_scaled"] = np.nan
    out["theory_spectrum_continuous_observed_scaled"] = np.nan
    out["continuous_scale_to_expected"] = np.nan
    out["continuous_observed_scale_to_expected"] = np.nan
    group_cols = ["profile", "variant", "resolution_label"]
    for _, idx in out.groupby(group_cols, observed=True).groups.items():
        sub = out.loc[idx]
        for expected_col, source_col, scaled_col, scale_col in [
            (
                "theory_spectrum_expected_latent",
                "theory_spectrum_continuous",
                "theory_spectrum_continuous_scaled",
                "continuous_scale_to_expected",
            ),
            (
                "theory_spectrum_expected",
                "theory_spectrum_continuous_observed",
                "theory_spectrum_continuous_observed_scaled",
                "continuous_observed_scale_to_expected",
            ),
        ]:
            if expected_col not in sub.columns or source_col not in sub.columns:
                continue
            expected = pd.to_numeric(sub[expected_col], errors="coerce")
            continuous = pd.to_numeric(sub[source_col], errors="coerce")
            good = expected.notna() & continuous.notna() & (expected > 0) & (continuous > 0)
            if not good.any():
                continue
            scale = float(np.nanmean((expected[good] / continuous[good]).to_numpy(dtype=float)))
            if not np.isfinite(scale) or scale <= 0:
                continue
            out.loc[idx, scale_col] = scale
            out.loc[idx, scaled_col] = continuous.to_numpy(dtype=float) * scale
    return out


def plot_monthly_directional_data_expected(
    args: argparse.Namespace,
    ctx: MonthContext,
    smooth: float,
    monthly: pd.DataFrame,
    daily_means: pd.DataFrame,
    fit_df: pd.DataFrame,
    out_dir: Path,
):
    directional = monthly[monthly["profile"].astype(str).isin(["lat", "lon", "diag"])].copy()
    if directional.empty:
        return
    labels_order = [block_prefix_label(s) for s in parse_block_prefixes(args.block_prefixes)]
    row_specs = []
    for variant in variants_for_year(ctx.year, parse_names(args.variants)):
        for direction in ["lat", "lon", "diag"]:
            row_specs.append((variant, direction))
    if not row_specs:
        return
    ylim = positive_ylim(directional.get("data_spectrum"), directional.get("theory_spectrum_expected"))
    fig, axes = plt.subplots(len(row_specs), len(labels_order), figsize=(4.4 * len(labels_order), 2.85 * len(row_specs)), sharey=True)
    axes = np.asarray(axes).reshape(len(row_specs), len(labels_order))
    for i, (variant, direction) in enumerate(row_specs):
        for j, label in enumerate(labels_order):
            ax = axes[i, j]
            sub_data = monthly[
                (monthly["variant"] == variant)
                & (monthly["profile"].astype(str) == direction)
                & (monthly["resolution_label"].astype(str) == label)
                & (monthly["k_mid"] > 0)
            ]
            daily_sub = daily_means[
                (daily_means["variant"] == variant)
                & (daily_means["profile"].astype(str) == direction)
                & (daily_means["resolution_label"].astype(str) == label)
                & (daily_means["k_mid"] > 0)
                & daily_means["data_spectrum"].notna()
            ]
            if sub_data.empty:
                ax.set_visible(False)
                continue
            k_cut = cutoff_from_frame(sub_data, sub_data["k_mid"].max())
            sub_data = keep_to_cutoff(sub_data, k_cut)
            daily_sub = keep_to_cutoff(daily_sub, k_cut)
            if sub_data.empty:
                ax.set_visible(False)
                continue
            param_label = format_fit_label(fit_df, variant, label)
            if param_label:
                ax.plot([], [], color="none", label=param_label)
            ax.plot(sub_data["k_mid"], sub_data["theory_spectrum_expected"], color="tab:red", linewidth=1.9, linestyle="--", label="expected periodogram", zorder=3)
            for d_i, (_, ds) in enumerate(daily_sub.groupby("date_str")):
                day_label = "daily mean spectra" if (i == 0 and j == 0 and d_i == 0) else None
                ax.plot(ds["k_mid"], ds["data_spectrum"], color="0.35", alpha=0.32, linewidth=0.8, label=day_label, zorder=1)
            ax.plot(sub_data["k_mid"], sub_data["data_spectrum"], color="black", linewidth=2.05, label="data residual spectrum", zorder=4)
            sigma_sq = median_sigmasq(fit_df, variant, label)
            ratio_df = ratio_frame(
                sub_data,
                sub_data,
                "data_spectrum",
                "theory_spectrum_expected",
                normalize_mean=True,
                sigma_sq=sigma_sq,
                k_max=k_cut,
            )
            ratio_label = profile_ratio_label(ratio_df)
            if ratio_label:
                ax.plot([], [], color="none", label=ratio_label)
            add_ratio_axis(ax, ratio_df, ylabel="I / E[I]" if j == len(labels_order) - 1 else None)
            ax.plot([], [], color="tab:blue", linewidth=1.35, linestyle=":", label="profiled I / E[I] (mean=1)")
            ax.set_xlim(0, ctx.k_max_full)
            ax.set_ylim(*ylim)
            direction_label = DIRECTION_SPECS[direction]["label"]
            freq_label = DIRECTION_SPECS[direction]["frequency_label"]
            ax.set_title(f"{VARIANTS[variant]['row_title']}: {direction_label}, {label}")
            ax.set_xlabel(freq_label)
            if j == 0:
                ax.set_ylabel("directional spectrum")
            ax.set_yscale("log")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=6.5, handlelength=1.4)
    fig.suptitle(f"{ctx.year}-{ctx.month:02d}, smooth={smooth}: 30-day mean directional residual spectra vs fitted expected periodogram")
    fig.tight_layout()
    plot_path = out_dir / "monthly_average" / f"{ctx.year}{ctx.month:02d}_30day_mean_directional_data_vs_expected_periodogram.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved monthly directional plot: {plot_path}", flush=True)
    publish_monthly_plot(args, ctx, smooth, plot_path)


def plot_monthly_expected_vs_continuous(
    args: argparse.Namespace,
    ctx: MonthContext,
    smooth: float,
    monthly: pd.DataFrame,
    out_dir: Path,
):
    scaled = add_continuous_scaled_to_expected(monthly)
    if scaled.empty or (
        scaled["theory_spectrum_continuous_scaled"].notna().sum() == 0
        and scaled["theory_spectrum_continuous_observed_scaled"].notna().sum() == 0
    ):
        return
    labels_order = [block_prefix_label(s) for s in parse_block_prefixes(args.block_prefixes)]
    profiles = ["radial", "lat", "lon", "diag"]
    row_specs = []
    for variant in variants_for_year(ctx.year, parse_names(args.variants)):
        for profile in profiles:
            row_specs.append((variant, profile))
    ylim = positive_ylim(
        scaled.get("theory_spectrum_expected"),
        scaled.get("theory_spectrum_expected_latent"),
        scaled.get("theory_spectrum_continuous_scaled"),
        scaled.get("theory_spectrum_continuous_observed_scaled"),
    )
    fig, axes = plt.subplots(len(row_specs), len(labels_order), figsize=(4.4 * len(labels_order), 2.55 * len(row_specs)), sharey=True)
    axes = np.asarray(axes).reshape(len(row_specs), len(labels_order))
    for i, (variant, profile) in enumerate(row_specs):
        for j, label in enumerate(labels_order):
            ax = axes[i, j]
            sub = scaled[
                (scaled["variant"] == variant)
                & (scaled["profile"].astype(str) == profile)
                & (scaled["resolution_label"].astype(str) == label)
                & (scaled["k_mid"] > 0)
            ].copy()
            sub = sub.dropna(subset=["theory_spectrum_expected"])
            sub = sub[sub["theory_spectrum_expected"] > 0]
            k_cut = cutoff_from_frame(sub, sub["k_mid"].max())
            sub = keep_to_cutoff(sub, k_cut)
            has_latent = sub["theory_spectrum_continuous_scaled"].notna().any()
            has_observed = sub["theory_spectrum_continuous_observed_scaled"].notna().any()
            if has_latent:
                sub = sub[
                    sub["theory_spectrum_continuous_scaled"].isna()
                    | (sub["theory_spectrum_continuous_scaled"] > 0)
                ]
            if has_observed:
                sub = sub[
                    sub["theory_spectrum_continuous_observed_scaled"].isna()
                    | (sub["theory_spectrum_continuous_observed_scaled"] > 0)
                ]
            if sub.empty:
                ax.set_visible(False)
                continue
            scale_val = float(sub["continuous_scale_to_expected"].dropna().iloc[0]) if sub["continuous_scale_to_expected"].notna().any() else np.nan
            if np.isfinite(scale_val):
                ax.plot([], [], color="none", label=f"latent profile scale={scale_val:.3g}")
            obs_scale_val = float(sub["continuous_observed_scale_to_expected"].dropna().iloc[0]) if sub["continuous_observed_scale_to_expected"].notna().any() else np.nan
            if np.isfinite(obs_scale_val):
                ax.plot([], [], color="none", label=f"observed profile scale={obs_scale_val:.3g}")
            if has_latent:
                latent = sub.dropna(subset=["theory_spectrum_expected_latent", "theory_spectrum_continuous_scaled"])
                ax.plot(
                    latent["k_mid"],
                    latent["theory_spectrum_expected_latent"],
                    color="tab:red",
                    linewidth=2.0,
                    linestyle="--",
                    label="finite-sample E[I] latent",
                    zorder=3,
                )
                ax.plot(
                    latent["k_mid"],
                    latent["theory_spectrum_continuous_scaled"],
                    color="tab:green",
                    linewidth=1.8,
                    label="latent continuous S (no nugget, profiled)",
                    zorder=4,
                )
                ratio_df = ratio_frame(
                    latent,
                    latent,
                    "theory_spectrum_expected_latent",
                    "theory_spectrum_continuous_scaled",
                    k_max=k_cut,
                )
                add_ratio_axis(ax, ratio_df, ylabel="E[I] / scaled S" if j == len(labels_order) - 1 else None, color="tab:blue")
                ax.plot([], [], color="tab:blue", linewidth=1.35, linestyle=":", label="E[I] / scaled latent S")
            if has_observed:
                observed = sub.dropna(subset=["theory_spectrum_expected", "theory_spectrum_continuous_observed_scaled"])
                ax.plot(
                    observed["k_mid"],
                    observed["theory_spectrum_expected"],
                    color="tab:orange",
                    linewidth=1.85,
                    linestyle="--",
                    label="finite-sample E[I] observed",
                    zorder=3,
                )
                ax.plot(
                    observed["k_mid"],
                    observed["theory_spectrum_continuous_observed_scaled"],
                    color="tab:purple",
                    linewidth=1.55,
                    linestyle="-.",
                    label="observed continuous S+nugget (profiled)",
                    zorder=4,
                )
            profile_label = DIRECTION_SPECS.get(profile, DIRECTION_SPECS["radial"])["label"]
            freq_label = DIRECTION_SPECS.get(profile, DIRECTION_SPECS["radial"])["frequency_label"]
            ax.set_xlim(0, ctx.k_max_full)
            ax.set_ylim(*ylim)
            ax.set_title(f"{VARIANTS[variant]['row_title']}: {profile_label}, {label}")
            ax.set_xlabel(freq_label)
            if j == 0:
                ax.set_ylabel("spectrum")
            ax.set_yscale("log")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=6.5, handlelength=1.4)
    fig.suptitle(f"{ctx.year}-{ctx.month:02d}, smooth={smooth}: expected periodogram vs continuous theoretical spectrum")
    fig.tight_layout()
    plot_path = out_dir / "monthly_average" / f"{ctx.year}{ctx.month:02d}_expected_periodogram_vs_continuous_theoretical_scaled.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved expected-vs-continuous plot: {plot_path}", flush=True)
    publish_monthly_plot(args, ctx, smooth, plot_path)


def make_monthly_average(args: argparse.Namespace, ctx: MonthContext, smooth: float, out_dir: Path):
    daily_dir = out_dir / "daily_csv"
    paths = sorted(daily_dir.glob("*_spectral_profiles.csv"))
    if not paths:
        paths = sorted(daily_dir.glob("*_radial_spectrum.csv"))
    if not paths:
        print(f"No daily spectra found in {daily_dir}; skipping monthly average.", flush=True)
        return
    frames = []
    for path in paths:
        try:
            frames.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            continue
    if not frames:
        print("No non-empty daily spectra; skipping monthly average.", flush=True)
        return
    spectral_df = ensure_profile_columns(pd.concat(frames, ignore_index=True))
    fit_paths = sorted(daily_dir.glob("*_fits.csv"))
    fit_frames = []
    for path in fit_paths:
        try:
            fit_frames.append(pd.read_csv(path))
        except pd.errors.EmptyDataError:
            continue
    fit_df = pd.concat(fit_frames, ignore_index=True) if fit_frames else spectral_df

    group_cols_daily = [
        "date_str",
        "profile",
        "direction",
        "direction_label",
        "frequency_label",
        "variant",
        "resolution_label",
        "resolution_stride",
        "k_bin",
    ]
    group_cols_month = [
        "profile",
        "direction",
        "direction_label",
        "frequency_label",
        "variant",
        "resolution_label",
        "resolution_stride",
        "k_bin",
    ]
    daily_means = (
        spectral_df
        .groupby(group_cols_daily, observed=True)
        .agg(
            k_mid=("k_mid", "mean"),
            data_spectrum=("data_spectrum", "mean"),
            theory_spectrum_expected=("theory_spectrum_expected", "mean"),
            theory_spectrum_expected_latent=("theory_spectrum_expected_latent", "mean"),
            theory_spectrum_continuous=("theory_spectrum_continuous", "mean"),
            theory_spectrum_continuous_observed=("theory_spectrum_continuous_observed", "mean"),
            data_k_max=("data_k_max", "mean"),
            n_hours=("hour_idx", "nunique"),
        )
        .reset_index()
    )
    monthly = (
        spectral_df
        .groupby(group_cols_month, observed=True)
        .agg(
            k_mid=("k_mid", "mean"),
            data_spectrum=("data_spectrum", "mean"),
            theory_spectrum_expected=("theory_spectrum_expected", "mean"),
            theory_spectrum_expected_latent=("theory_spectrum_expected_latent", "mean"),
            theory_spectrum_continuous=("theory_spectrum_continuous", "mean"),
            theory_spectrum_continuous_observed=("theory_spectrum_continuous_observed", "mean"),
            data_k_max=("data_k_max", "mean"),
            n_days=("date_str", "nunique"),
            n_hours=("time_key", "nunique"),
        )
        .reset_index()
    )
    for key, value in current_domain_meta(args).items():
        daily_means[key] = value
        monthly[key] = value
    write_csv(daily_means, out_dir / "monthly_average" / f"{ctx.year}{ctx.month:02d}_daily_mean_curves.csv")
    write_csv(monthly, out_dir / "monthly_average" / f"{ctx.year}{ctx.month:02d}_hourly_mean_curves.csv")
    write_csv(monthly, out_dir / "monthly_average" / f"{ctx.year}{ctx.month:02d}_30day_mean_curves.csv")

    monthly_radial = monthly[monthly["profile"].astype(str) == "radial"].copy()
    daily_radial = daily_means[daily_means["profile"].astype(str) == "radial"].copy()
    if monthly_radial.empty:
        print("No norm-frequency monthly spectra; skipping norm-frequency monthly plot.", flush=True)
    else:
        monthly_for_radial_plot = monthly_radial
        daily_for_radial_plot = daily_radial

    labels_order = [block_prefix_label(s) for s in parse_block_prefixes(args.block_prefixes)]
    row_specs = [(v, VARIANTS[v]["row_title"]) for v in variants_for_year(ctx.year, parse_names(args.variants))]
    if not monthly_radial.empty:
        ylim = positive_ylim(monthly_for_radial_plot.get("data_spectrum"), monthly_for_radial_plot.get("theory_spectrum_expected"))
        fig, axes = plt.subplots(len(row_specs), len(labels_order), figsize=(4.4 * len(labels_order), 3.4 * len(row_specs)), sharey=True)
        axes = np.asarray(axes).reshape(len(row_specs), len(labels_order))
        for i, (variant, row_title) in enumerate(row_specs):
            for j, label in enumerate(labels_order):
                ax = axes[i, j]
                sub_data = monthly_for_radial_plot[
                    (monthly_for_radial_plot["variant"] == variant)
                    & (monthly_for_radial_plot["resolution_label"].astype(str) == label)
                    & (monthly_for_radial_plot["k_mid"] > 0)
                ]
                daily_sub = daily_for_radial_plot[
                    (daily_for_radial_plot["variant"] == variant)
                    & (daily_for_radial_plot["resolution_label"].astype(str) == label)
                    & (daily_for_radial_plot["k_mid"] > 0)
                    & daily_for_radial_plot["data_spectrum"].notna()
                ]
                if sub_data.empty:
                    ax.set_visible(False)
                    continue
                k_cut = cutoff_from_frame(sub_data, sub_data["k_mid"].max())
                sub_data = keep_to_cutoff(sub_data, k_cut)
                daily_sub = keep_to_cutoff(daily_sub, k_cut)
                if sub_data.empty:
                    ax.set_visible(False)
                    continue
                param_label = format_fit_label(fit_df, variant, label)
                if param_label:
                    ax.plot([], [], color="none", label=param_label)
                ax.plot(sub_data["k_mid"], sub_data["theory_spectrum_expected"], color="tab:red", linewidth=1.9, linestyle="--", label="expected periodogram (hourly mean)", zorder=3)
                for d_i, (_, ds) in enumerate(daily_sub.groupby("date_str")):
                    day_label = "daily mean spectra" if (i == 0 and j == 0 and d_i == 0) else None
                    ax.plot(ds["k_mid"], ds["data_spectrum"], color="0.35", alpha=0.35, linewidth=0.85, label=day_label, zorder=1)
                ax.plot(sub_data["k_mid"], sub_data["data_spectrum"], color="black", linewidth=2.2, label="data residual spectrum (hourly mean)", zorder=4)
                sigma_sq = median_sigmasq(fit_df, variant, label)
                ratio_df = ratio_frame(
                    sub_data,
                    sub_data,
                    "data_spectrum",
                    "theory_spectrum_expected",
                    normalize_mean=True,
                    sigma_sq=sigma_sq,
                    k_max=k_cut,
                )
                ratio_label = profile_ratio_label(ratio_df)
                if ratio_label:
                    ax.plot([], [], color="none", label=ratio_label)
                add_ratio_axis(ax, ratio_df, ylabel="I / E[I]" if j == len(labels_order) - 1 else None)
                ax.plot([], [], color="tab:blue", linewidth=1.35, linestyle=":", label="profiled I / E[I] (mean=1)")
                ax.set_xlim(0, ctx.k_max_full)
                ax.set_ylim(*ylim)
                ax.set_title(f"{row_title}, {label}  (data k <= {k_cut:.1f})")
                ax.set_xlabel("norm frequency")
                if j == 0:
                    ax.set_ylabel("spectrum")
                ax.set_yscale("log")
                ax.grid(alpha=0.2)
                ax.legend(fontsize=7, handlelength=1.5)
        fig.suptitle(f"{ctx.year}-{ctx.month:02d}, smooth={smooth}: hourly mean norm-frequency residual spectrum vs fitted expected periodogram")
        fig.tight_layout()
        plot_path = out_dir / "monthly_average" / f"{ctx.year}{ctx.month:02d}_hourly_mean_data_vs_expected_periodogram.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved monthly average plot: {plot_path}", flush=True)
        publish_monthly_plot(args, ctx, smooth, plot_path)

    plot_monthly_directional_data_expected(args, ctx, smooth, monthly, daily_means, fit_df, out_dir)
    plot_monthly_expected_vs_continuous(args, ctx, smooth, monthly, out_dir)


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def combined_output_dir(args: argparse.Namespace, smooth: float, year: int) -> Path:
    return Path(args.output_root) / f"{year}_{args.month:02d}" / f"smooth_{smooth_tag(smooth)}" / "combined_domain_plots"


def combined_top_copy(args: argparse.Namespace, year: int, smooth: float, plot_path: Path):
    top_dir = Path(args.top_plot_dir) if str(args.top_plot_dir).strip() else Path(args.output_root) / "monthly_plots_top"
    year_dir = top_dir / f"{year}_{args.month:02d}"
    year_dir.mkdir(parents=True, exist_ok=True)
    dest = year_dir / f"smooth_{smooth_tag(smooth)}_combined_{plot_path.name}"
    shutil.copy2(plot_path, dest)
    print(f"Copied combined plot to top folder: {dest}", flush=True)


def ratio_ylim_from_frames(frames: list[pd.DataFrame], fallback=(0.2, 5.0)) -> tuple[float, float]:
    vals = []
    for df in frames:
        if df is None or df.empty or "ratio" not in df.columns:
            continue
        arr = pd.to_numeric(df["ratio"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        arr = arr[arr > 0]
        if not arr.empty:
            vals.append(arr.to_numpy(dtype=float))
    if not vals:
        return fallback
    x = np.concatenate(vals)
    lo = max(1e-3, min(0.5, float(np.nanpercentile(x, 2)) / 1.25))
    hi = min(1e3, max(2.0, float(np.nanpercentile(x, 98)) * 1.25))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        return fallback
    return lo, hi


def monthly_curves_for_domain(
    args: argparse.Namespace,
    year: int,
    smooth: float,
    spec: DomainSpec,
    profile: str,
    variant: str,
) -> dict:
    out_dir = output_dir_for_domain(args, smooth, year, spec)
    month_prefix = f"{year}{args.month:02d}"
    monthly = ensure_profile_columns(read_csv_or_empty(out_dir / "monthly_average" / f"{month_prefix}_30day_mean_curves.csv"))
    daily_means = ensure_profile_columns(read_csv_or_empty(out_dir / "monthly_average" / f"{month_prefix}_daily_mean_curves.csv"))
    if monthly is None or monthly.empty:
        return {"monthly": pd.DataFrame(), "daily": pd.DataFrame(), "ratio": pd.DataFrame(), "scaled": pd.DataFrame()}

    labels_order = [block_prefix_label(s) for s in parse_block_prefixes(args.block_prefixes)]
    label = labels_order[0] if labels_order else "all"
    median_loss = median_loss_for_domain(args, year, smooth, spec, variant, label)
    sub = monthly[
        (monthly["variant"].astype(str) == str(variant))
        & (monthly["profile"].astype(str) == str(profile))
        & (monthly["resolution_label"].astype(str) == str(label))
        & (pd.to_numeric(monthly["k_mid"], errors="coerce") > 0)
    ].copy()
    if sub.empty:
        return {"monthly": pd.DataFrame(), "daily": pd.DataFrame(), "ratio": pd.DataFrame(), "scaled": pd.DataFrame()}
    k_cut = cutoff_from_frame(sub, sub["k_mid"].max())
    sub = keep_to_cutoff(sub, k_cut)
    scaled = keep_to_cutoff(add_continuous_scaled_to_expected(sub), k_cut)
    monthly_ratio = ratio_frame(
        sub,
        sub,
        "data_spectrum",
        "theory_spectrum_expected",
        normalize_mean=bool(args.combined_ratio_normalize),
        k_max=k_cut,
    )
    if not monthly_ratio.empty:
        for key, value in {
            "year": year,
            "month": int(args.month),
            "smooth": float(smooth),
            "variant": variant,
            "model_label": VARIANTS.get(variant, {}).get("plot_label", variant),
            "median_loss": median_loss,
            "profile": profile,
            "resolution_label": label,
            "domain_group": spec.group,
            "domain_label": spec.label,
            "domain_title": spec.title,
            "domain_row": spec.row,
            "domain_col": spec.col,
            "domain_lat_min": spec.lat_range[0],
            "domain_lat_max": spec.lat_range[1],
            "domain_lon_min": spec.lon_range[0],
            "domain_lon_max": spec.lon_range[1],
            "domain_lat_upper_inclusive": bool(spec.lat_upper_inclusive),
            "domain_lon_upper_inclusive": bool(spec.lon_upper_inclusive),
            "data_k_max": k_cut,
        }.items():
            monthly_ratio[key] = value

    daily_sub = pd.DataFrame()
    daily_ratios = []
    if daily_means is not None and not daily_means.empty:
        daily_sub = daily_means[
            (daily_means["variant"].astype(str) == str(variant))
            & (daily_means["profile"].astype(str) == str(profile))
            & (daily_means["resolution_label"].astype(str) == str(label))
            & (pd.to_numeric(daily_means["k_mid"], errors="coerce") > 0)
        ].copy()
        daily_sub = keep_to_cutoff(daily_sub, k_cut)
        for date_str, one_day in daily_sub.groupby("date_str", observed=True):
            r = ratio_frame(
                one_day,
                one_day,
                "data_spectrum",
                "theory_spectrum_expected",
                normalize_mean=bool(args.combined_ratio_normalize),
                k_max=k_cut,
            )
            if r.empty:
                continue
            r["date_str"] = str(date_str)
            r["domain_group"] = spec.group
            r["domain_label"] = spec.label
            daily_ratios.append(r)
    daily_ratio = pd.concat(daily_ratios, ignore_index=True) if daily_ratios else pd.DataFrame()
    return {
        "monthly": sub,
        "daily": daily_sub,
        "ratio": monthly_ratio,
        "daily_ratio": daily_ratio,
        "scaled": scaled,
        "k_cut": k_cut,
        "resolution_label": label,
    }


def plot_combined_domain_data_expected(
    args: argparse.Namespace,
    year: int,
    smooth: float,
    group: str,
    specs: list[DomainSpec],
    profile: str,
    variant: str,
):
    if not specs:
        return
    panel = {}
    monthly_frames = []
    daily_frames = []
    ratio_frames = []
    for spec in specs:
        curves = monthly_curves_for_domain(args, year, smooth, spec, profile, variant)
        panel[spec.label] = curves
        if not curves["monthly"].empty:
            monthly_frames.append(curves["monthly"])
        if not curves["daily"].empty:
            daily_frames.append(curves["daily"])
        if not curves["ratio"].empty:
            ratio_frames.append(curves["ratio"])
    if not monthly_frames:
        print(f"No combined spectrum data for group={group}, profile={profile}, variant={variant}", flush=True)
        return

    n_rows = max(int(s.n_rows) for s in specs)
    n_cols = max(int(s.n_cols) for s in specs)
    fig_w = max(4.2 * n_cols, 5.0)
    fig_h = max(3.25 * n_rows, 3.7)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(n_rows, n_cols)
    for ax in axes.ravel():
        ax.set_visible(False)

    ylim = positive_ylim(
        *(f.get("data_spectrum") for f in monthly_frames),
        *(f.get("theory_spectrum_expected") for f in monthly_frames),
        *(f.get("data_spectrum") for f in daily_frames),
    )
    freq_label = DIRECTION_SPECS.get(profile, DIRECTION_SPECS["radial"])["frequency_label"]
    for spec in specs:
        ax = axes[int(spec.row), int(spec.col)]
        ax.set_visible(True)
        curves = panel[spec.label]
        monthly = curves["monthly"]
        daily = curves["daily"]
        ratio = curves["ratio"]
        k_cut = curves.get("k_cut", np.nan)
        if not daily.empty:
            for d_i, (_, ds) in enumerate(daily.groupby("date_str", observed=True)):
                ds = ds.replace([np.inf, -np.inf], np.nan).dropna(subset=["k_mid", "data_spectrum"])
                if not ds.empty:
                    label = "daily mean spectra" if spec == specs[0] and d_i == 0 else None
                    ax.plot(ds["k_mid"], ds["data_spectrum"], color="0.35", alpha=0.28, linewidth=0.85, label=label, zorder=1)
        if not monthly.empty:
            expected = monthly.dropna(subset=["k_mid", "theory_spectrum_expected"])
            data = monthly.dropna(subset=["k_mid", "data_spectrum"])
            if not expected.empty:
                ax.plot(
                    expected["k_mid"],
                    expected["theory_spectrum_expected"],
                    color="tab:red",
                    linewidth=1.85,
                    linestyle="--",
                    label="finite-sample E[I]",
                    zorder=3,
                )
            if not data.empty:
                ax.plot(
                    data["k_mid"],
                    data["data_spectrum"],
                    color="black",
                    linewidth=2.15,
                    label="hourly mean I",
                    zorder=4,
                )
        if not ratio.empty:
            add_ratio_axis(ax, ratio, ylabel="I / E[I]" if int(spec.col) == n_cols - 1 else None)
            ax.plot(
                [],
                [],
                color="tab:blue",
                linewidth=1.35,
                linestyle=":",
                label="profiled I / E[I] (mean=1)",
            )
        ax.set_title(f"{spec.title}\ndata k <= {k_cut:.2g}" if np.isfinite(k_cut) else spec.title, fontsize=8.5)
        ax.set_yscale("log")
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.22)
        if int(spec.row) == n_rows - 1:
            ax.set_xlabel(freq_label)
        if int(spec.col) == 0:
            ax.set_ylabel("spectrum")
    first_ax = axes.ravel()[0]
    if first_ax.get_visible():
        first_ax.legend(fontsize=7, handlelength=1.4)

    profile_label = DIRECTION_SPECS.get(profile, DIRECTION_SPECS["radial"])["label"]
    fig.suptitle(
        f"{year}-{args.month:02d}, smooth={smooth}, {variant}: {group} {profile_label} hourly mean I vs E[I], x1 all-grid fits",
        fontsize=12,
    )
    fig.tight_layout()
    out_dir = combined_output_dir(args, smooth, year)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_file = "norm" if profile == "radial" else profile
    plot_path = out_dir / f"{year}{args.month:02d}_{group}_{variant}_data_vs_expected_{profile_file}.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    curve_csv = out_dir / f"{year}{args.month:02d}_{group}_{variant}_data_vs_expected_{profile_file}.csv"
    write_csv(pd.concat(monthly_frames, ignore_index=True), curve_csv)
    if ratio_frames:
        ratio_csv = out_dir / f"{year}{args.month:02d}_{group}_{variant}_ratio_{profile_file}.csv"
        write_csv(pd.concat(ratio_frames, ignore_index=True), ratio_csv)
        print(f"Saved combined ratio CSV: {ratio_csv}", flush=True)
    print(f"Saved combined I-vs-E[I] plot: {plot_path}", flush=True)
    print(f"Saved combined curve CSV: {curve_csv}", flush=True)
    combined_top_copy(args, year, smooth, plot_path)


def plot_combined_domain_expected_continuous(
    args: argparse.Namespace,
    year: int,
    smooth: float,
    group: str,
    specs: list[DomainSpec],
    profile: str,
    variant: str,
):
    if not specs:
        return
    panel = {}
    scaled_frames = []
    for spec in specs:
        curves = monthly_curves_for_domain(args, year, smooth, spec, profile, variant)
        panel[spec.label] = curves
        if not curves["scaled"].empty:
            scaled_frames.append(curves["scaled"])
    if not scaled_frames:
        print(f"No combined E[I]-vs-continuous data for group={group}, profile={profile}, variant={variant}", flush=True)
        return

    n_rows = max(int(s.n_rows) for s in specs)
    n_cols = max(int(s.n_cols) for s in specs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(4.2 * n_cols, 5.0), max(3.1 * n_rows, 3.6)), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(n_rows, n_cols)
    for ax in axes.ravel():
        ax.set_visible(False)

    ylim = positive_ylim(
        *(f.get("theory_spectrum_expected") for f in scaled_frames),
        *(f.get("theory_spectrum_expected_latent") for f in scaled_frames),
        *(f.get("theory_spectrum_continuous_scaled") for f in scaled_frames),
        *(f.get("theory_spectrum_continuous_observed_scaled") for f in scaled_frames),
    )
    freq_label = DIRECTION_SPECS.get(profile, DIRECTION_SPECS["radial"])["frequency_label"]
    for spec in specs:
        ax = axes[int(spec.row), int(spec.col)]
        ax.set_visible(True)
        scaled = panel[spec.label]["scaled"]
        k_cut = panel[spec.label].get("k_cut", np.nan)
        if scaled.empty:
            ax.set_title(spec.title, fontsize=8.5)
            ax.set_yscale("log")
            ax.grid(alpha=0.22)
            continue
        latent = scaled.dropna(subset=["theory_spectrum_expected_latent", "theory_spectrum_continuous_scaled"])
        observed = scaled.dropna(subset=["theory_spectrum_expected", "theory_spectrum_continuous_observed_scaled"])
        if not latent.empty:
            ax.plot(
                latent["k_mid"],
                latent["theory_spectrum_expected_latent"],
                color="tab:red",
                linewidth=1.85,
                linestyle="--",
                label="finite-sample E[I] latent",
                zorder=3,
            )
            ax.plot(
                latent["k_mid"],
                latent["theory_spectrum_continuous_scaled"],
                color="tab:green",
                linewidth=1.75,
                label="continuous S latent (profiled)",
                zorder=4,
            )
            ratio_df = ratio_frame(
                latent,
                latent,
                "theory_spectrum_expected_latent",
                "theory_spectrum_continuous_scaled",
                k_max=k_cut,
            )
            add_ratio_axis(ax, ratio_df, ylabel="E[I] / S" if int(spec.col) == n_cols - 1 else None, color="tab:blue")
            ax.plot([], [], color="tab:blue", linewidth=1.35, linestyle=":", label="E[I] / continuous S")
        if not observed.empty:
            ax.plot(
                observed["k_mid"],
                observed["theory_spectrum_expected"],
                color="tab:orange",
                linewidth=1.65,
                linestyle="--",
                label="finite-sample E[I] observed",
                zorder=3,
            )
            ax.plot(
                observed["k_mid"],
                observed["theory_spectrum_continuous_observed_scaled"],
                color="tab:purple",
                linewidth=1.5,
                linestyle="-.",
                label="continuous S+nugget (profiled)",
                zorder=4,
            )
        ax.set_title(f"{spec.title}\ndata k <= {k_cut:.2g}" if np.isfinite(k_cut) else spec.title, fontsize=8.5)
        ax.set_yscale("log")
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.22)
        if int(spec.row) == n_rows - 1:
            ax.set_xlabel(freq_label)
        if int(spec.col) == 0:
            ax.set_ylabel("spectrum")
    first_ax = axes.ravel()[0]
    if first_ax.get_visible():
        first_ax.legend(fontsize=6.7, handlelength=1.3)

    profile_label = DIRECTION_SPECS.get(profile, DIRECTION_SPECS["radial"])["label"]
    fig.suptitle(
        f"{year}-{args.month:02d}, smooth={smooth}, {variant}: {group} {profile_label} E[I] vs continuous spectrum",
        fontsize=12,
    )
    fig.tight_layout()
    out_dir = combined_output_dir(args, smooth, year)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_file = "norm" if profile == "radial" else profile
    plot_path = out_dir / f"{year}{args.month:02d}_{group}_{variant}_expected_vs_continuous_{profile_file}.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    scaled_csv = out_dir / f"{year}{args.month:02d}_{group}_{variant}_expected_vs_continuous_{profile_file}.csv"
    write_csv(pd.concat(scaled_frames, ignore_index=True), scaled_csv)
    print(f"Saved combined E[I]-vs-continuous plot: {plot_path}", flush=True)
    print(f"Saved combined E[I]-vs-continuous CSV: {scaled_csv}", flush=True)
    combined_top_copy(args, year, smooth, plot_path)


def plot_combined_domain_model_ratio_compare(
    args: argparse.Namespace,
    year: int,
    smooth: float,
    group: str,
    specs: list[DomainSpec],
    profile: str,
    variants: list[str],
):
    if not specs or not variants:
        return
    panel: dict[str, dict[str, pd.DataFrame]] = {}
    ratio_frames = []
    for spec in specs:
        panel[spec.label] = {}
        for variant in variants:
            curves = monthly_curves_for_domain(args, year, smooth, spec, profile, variant)
            ratio = curves.get("ratio", pd.DataFrame())
            if ratio is None or ratio.empty:
                continue
            ratio = ratio.copy()
            base_label = VARIANTS.get(variant, {}).get("plot_label", variant)
            loss = (
                float(pd.to_numeric(ratio["median_loss"], errors="coerce").dropna().iloc[0])
                if "median_loss" in ratio.columns and ratio["median_loss"].notna().any()
                else np.nan
            )
            ratio["model_label"] = label_with_loss(base_label, loss)
            panel[spec.label][variant] = ratio
            ratio_frames.append(ratio)
    if not ratio_frames:
        print(f"No model-comparison ratio data for group={group}, profile={profile}", flush=True)
        return

    n_rows = max(int(s.n_rows) for s in specs)
    n_cols = max(int(s.n_cols) for s in specs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(4.2 * n_cols, 5.0), max(3.1 * n_rows, 3.6)), sharex=True, sharey=True)
    axes = np.asarray(axes).reshape(n_rows, n_cols)
    for ax in axes.ravel():
        ax.set_visible(False)

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["black", "tab:blue", "tab:orange", "tab:green"])
    ylim = ratio_ylim_from_frames(ratio_frames, fallback=(0.25, 4.0))
    freq_label = DIRECTION_SPECS.get(profile, DIRECTION_SPECS["radial"])["frequency_label"]
    for spec in specs:
        ax = axes[int(spec.row), int(spec.col)]
        ax.set_visible(True)
        ax.axhline(1.0, color="0.25", linewidth=0.9, linestyle="--", alpha=0.8)
        for v_i, variant in enumerate(variants):
            ratio = panel.get(spec.label, {}).get(variant, pd.DataFrame())
            if ratio.empty:
                continue
            rr = ratio.replace([np.inf, -np.inf], np.nan).dropna(subset=["k_mid", "ratio"]).sort_values("k_bin")
            rr = rr[rr["ratio"] > 0]
            if rr.empty:
                continue
            ax.plot(
                rr["k_mid"],
                rr["ratio"],
                linewidth=1.85 if variant == "matern_s03" else 1.55,
                linestyle="-" if variant == "matern_s03" else "--",
                color=colors[v_i % len(colors)],
                label=str(rr["model_label"].dropna().iloc[0]) if "model_label" in rr.columns and rr["model_label"].notna().any() else VARIANTS.get(variant, {}).get("plot_label", variant),
            )
        ax.set_title(spec.title, fontsize=8.5)
        ax.set_yscale("log")
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.22)
        if int(spec.row) == n_rows - 1:
            ax.set_xlabel(freq_label)
        if int(spec.col) == 0:
            ax.set_ylabel("profiled I / E[I] ratio")
    first_ax = axes.ravel()[0]
    if first_ax.get_visible():
        first_ax.legend(fontsize=7, handlelength=1.4)

    profile_label = DIRECTION_SPECS.get(profile, DIRECTION_SPECS["radial"])["label"]
    fig.suptitle(
        f"{year}-{args.month:02d}: pure-space hourly mean I / E[I], {group} {profile_label}, no whitening",
        fontsize=12,
    )
    fig.tight_layout()
    out_dir = combined_output_dir(args, smooth, year)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_file = "norm" if profile == "radial" else profile
    plot_path = out_dir / f"{year}{args.month:02d}_{group}_model_compare_profile_out_I_over_EI_ratio_{profile_file}.png"
    fig.savefig(plot_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    ratio_csv = out_dir / f"{year}{args.month:02d}_{group}_model_compare_profile_out_I_over_EI_ratio_{profile_file}.csv"
    write_csv(pd.concat(ratio_frames, ignore_index=True), ratio_csv)
    print(f"Saved model-comparison ratio plot: {plot_path}", flush=True)
    print(f"Saved model-comparison ratio CSV: {ratio_csv}", flush=True)
    combined_top_copy(args, year, smooth, plot_path)


def make_combined_domain_plots(args: argparse.Namespace, specs: list[DomainSpec], year: int, smooth: float):
    if args.skip_combined_plots:
        return
    variants = variants_for_year(year, parse_names(args.variants))
    profiles = [p for p in parse_names(args.combined_profiles) if p in DIRECTION_SPECS]
    groups = []
    for spec in specs:
        if spec.group not in groups:
            groups.append(spec.group)
    for group in groups:
        group_specs = [s for s in specs if s.group == group]
        for profile in profiles:
            plot_combined_domain_model_ratio_compare(args, year, smooth, group, group_specs, profile, variants)
        for variant in variants:
            for profile in profiles:
                plot_combined_domain_data_expected(args, year, smooth, group, group_specs, profile, variant)
                plot_combined_domain_expected_continuous(args, year, smooth, group, group_specs, profile, variant)


def output_dir_for_domain(args: argparse.Namespace, smooth: float, year: int, spec: DomainSpec) -> Path:
    return (
        Path(args.output_root)
        / f"{year}_{args.month:02d}"
        / f"smooth_{smooth_tag(smooth)}"
        / Path(*domain_path_parts(spec.group, spec.label))
    )


def output_dir_for(args: argparse.Namespace, smooth: float, year: int) -> Path:
    return (
        Path(args.output_root)
        / f"{year}_{args.month:02d}"
        / f"smooth_{smooth_tag(smooth)}"
        / Path(*domain_path_parts(
            getattr(args, "domain_group", "full"),
            getattr(args, "domain_label", "full"),
        ))
    )


def main() -> None:
    args = parse_args()
    years = parse_int_list_or_range(args.years)
    days = parse_int_list_or_range(args.days)
    smooths = parse_float_list(args.smooths)
    domain_specs = build_domain_specs(args)
    device = select_device(args.device, args.cuda_fallback)

    print(
        f"Run config: years={years}, month={args.month}, days={days[0]}..{days[-1]}, "
        f"smooths={smooths}, block_prefixes={[block_prefix_label(s) for s in parse_block_prefixes(args.block_prefixes)]}, "
        f"variants={parse_names(args.variants)}, region=lat {args.lat_range} lon {args.lon_range}, "
        f"domain_modes={parse_names(args.domain_modes)}, n_domains={len(domain_specs)}, "
        f"hann_taper={bool(args.hann)}, device={device}",
        flush=True,
    )
    for spec in domain_specs:
        print(
            f"Domain {spec.group}/{spec.label}: lat={range_arg(spec.lat_range)} "
            f"lon={range_arg(spec.lon_range)} panel=({spec.row},{spec.col})",
            flush=True,
        )

    for year in years:
        built_specs = []
        for spec in domain_specs:
            dargs = domain_args(args, spec)
            try:
                ctx = build_month_context(dargs, year, days)
            except Exception as exc:
                print(f"ERROR domain context failed for {spec.group}/{spec.label}: {exc}", flush=True)
                continue
            built_specs.append(spec)
            for smooth in smooths:
                out_dir = output_dir_for(dargs, smooth, year)
                out_dir.mkdir(parents=True, exist_ok=True)
                print(
                    f"\n=== year={year} smooth={smooth} domain={spec.group}/{spec.label} out={out_dir} ===",
                    flush=True,
                )
                if not dargs.make_monthly_only:
                    for day in days:
                        process_day(dargs, ctx, smooth, day, out_dir, device)
                make_monthly_average(dargs, ctx, smooth, out_dir)
        for smooth in smooths:
            make_combined_domain_plots(args, built_specs, year, smooth)


if __name__ == "__main__":
    main()
