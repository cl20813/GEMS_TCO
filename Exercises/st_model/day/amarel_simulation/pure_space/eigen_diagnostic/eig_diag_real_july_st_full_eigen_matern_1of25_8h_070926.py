"""Dense full-eigen diagnostic for a small July space-time subset.

This script is meant as a sanity-check companion to the scalable Vecchia
conditional eigen diagnostics.  Fitting is done once outside this script by the
full-domain Vecchia fit; this script reads that fit summary with
``--vecchia-summary`` and reuses the same estimated parameters.  Only the
diagnostic covariance/eigendecomposition is restricted to either one 5x5
spatial tile or an early max-min spatial subset, with all 8 July time slots for
one day.

The default subset is about 18,000 / 25 * 8 observations.  That is small enough
for a dense covariance/eigen run on a CPU node, but the method is intentionally
not scalable to the full domain.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import linalg
    from scipy.special import gamma, kv
except Exception as exc:  # pragma: no cover - dependency check happens at runtime.
    raise RuntimeError("This dense full-eigen diagnostic requires scipy.") from exc


EPS = 1e-12
BROWN_BRIDGE_Q95 = 1.3581015157406195
ROUND_DECIMALS = 8


@dataclass(frozen=True)
class ModelVariant:
    name: str
    label: str
    smooth: float
    nugget: float
    color: str


@dataclass(frozen=True)
class PhysicalParams:
    sigmasq: float
    range_lat: float
    range_lon: float
    range_time: float
    advec_lat: float
    advec_lon: float
    nugget: float
    source_model_variant: str
    source_loss_per_obs: float | None = None


VARIANTS: dict[str, ModelVariant] = {
    "s03_n1": ModelVariant("s03_n1", "Matern s=0.3 nugget1 rough", 0.3, 1.0, "#d62728"),
    "s05_n0": ModelVariant("s05_n0", "Matern s=0.5 nugget0 under", 0.5, 0.0, "#1f77b4"),
    "s05_n1": ModelVariant("s05_n1", "Matern s=0.5 nugget1 reference", 0.5, 1.0, "#2ca02c"),
    "s05_n2": ModelVariant("s05_n2", "Matern s=0.5 nugget2 over", 0.5, 2.0, "#ff7f0e"),
    "s10_n1": ModelVariant("s10_n1", "Matern s=1.0 nugget1 smooth", 1.0, 1.0, "#9467bd"),
    "s05_n1_rangelon_short": ModelVariant("s05_n1_rangelon_short", "Matern s=0.5 nugget1 range_lon=0.15", 0.5, 1.0, "#1f77b4"),
    "s05_n1_rangelon_true": ModelVariant("s05_n1_rangelon_true", "Matern s=0.5 nugget1 range_lon true", 0.5, 1.0, "#2ca02c"),
    "s05_n1_rangelon_long": ModelVariant("s05_n1_rangelon_long", "Matern s=0.5 nugget1 range_lon=0.60", 0.5, 1.0, "#d62728"),
    "s05_n1_rangetime_short": ModelVariant("s05_n1_rangetime_short", "Matern s=0.5 nugget1 range_time=1", 0.5, 1.0, "#1f77b4"),
    "s05_n1_rangetime_true": ModelVariant("s05_n1_rangetime_true", "Matern s=0.5 nugget1 range_time true", 0.5, 1.0, "#2ca02c"),
    "s05_n1_rangetime_long": ModelVariant("s05_n1_rangetime_long", "Matern s=0.5 nugget1 range_time=4", 0.5, 1.0, "#d62728"),
    "s05_n1_advec_true": ModelVariant("s05_n1_advec_true", "Matern s=0.5 nugget1 advec true", 0.5, 1.0, "#2ca02c"),
    "s05_n1_advec_zero": ModelVariant("s05_n1_advec_zero", "Matern s=0.5 nugget1 advec zero", 0.5, 1.0, "#1f77b4"),
    "s05_n1_advec_large": ModelVariant("s05_n1_advec_large", "Matern s=0.5 nugget1 advec 0.5/0.5", 0.5, 1.0, "#d62728"),
    "s05_n0_advec_true": ModelVariant("s05_n0_advec_true", "Matern s=0.5 nugget0 advec true", 0.5, 0.0, "#2ca02c"),
    "s05_n0_advec_zero": ModelVariant("s05_n0_advec_zero", "Matern s=0.5 nugget0 advec zero", 0.5, 0.0, "#1f77b4"),
    "s05_n0_advec_large": ModelVariant("s05_n0_advec_large", "Matern s=0.5 nugget0 advec 0.5/0.5", 0.5, 0.0, "#d62728"),
}

VARIANT_GROUPS: dict[str, list[str]] = {
    "reference": ["s05_n1"],
    "smoothness": ["s05_n1", "s03_n1", "s10_n1"],
    "nugget": ["s05_n0", "s05_n1", "s05_n2"],
    "range_lon_n1": ["s05_n1_rangelon_short", "s05_n1_rangelon_true", "s05_n1_rangelon_long"],
    "range_time_n1": ["s05_n1_rangetime_short", "s05_n1_rangetime_true", "s05_n1_rangetime_long"],
    "advection_n1": ["s05_n1_advec_true", "s05_n1_advec_zero", "s05_n1_advec_large"],
    "advection_n0": ["s05_n0_advec_true", "s05_n0_advec_zero", "s05_n0_advec_large"],
    "both": ["s05_n1", "s03_n1", "s10_n1", "s05_n0", "s05_n2"],
}

VARIANT_ALIASES: dict[str, list[str]] = {
    "s03_n1": ["matern_s03_n1", "matern_s03_n1_rough", "matern_s03_n0_rough", "matern_s03"],
    "s05_n0": ["matern_s05_n0", "matern_s05_n0_under", "matern_s05_n0_true", "matern_s05"],
    "s05_n1": ["matern_s05_n1", "matern_s05_n1_true", "matern_s05_n1_reference", "matern_s05"],
    "s05_n2": ["matern_s05_n2", "matern_s05_n2_over"],
    "s10_n1": ["matern_s10_n1", "matern_s10_n1_smooth", "matern_s10_n0_smooth", "matern_s10"],
    "s05_n1_rangelon_short": ["matern_s05_n1_rangelon_short"],
    "s05_n1_rangelon_true": ["matern_s05_n1_rangelon_true"],
    "s05_n1_rangelon_long": ["matern_s05_n1_rangelon_long"],
    "s05_n1_rangetime_short": ["matern_s05_n1_rangetime_short"],
    "s05_n1_rangetime_true": ["matern_s05_n1_rangetime_true"],
    "s05_n1_rangetime_long": ["matern_s05_n1_rangetime_long"],
    "s05_n1_advec_true": ["matern_s05_n1_advec_true"],
    "s05_n1_advec_zero": ["matern_s05_n1_advec_zero"],
    "s05_n1_advec_large": ["matern_s05_n1_advec_large"],
    "s05_n0_advec_true": ["matern_s05_n0_advec_true"],
    "s05_n0_advec_zero": ["matern_s05_n0_advec_zero"],
    "s05_n0_advec_large": ["matern_s05_n0_advec_large"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Full dense ST Matern eigen diagnostic on one spatial subset "
            "and 8 July time slots. Works for real or simulated July pickle files."
        )
    )
    p.add_argument("--input", required=True, help="July real-data pickle with yYYmMMdayDD_hmHH:MM keys.")
    p.add_argument("--output-root", required=True)
    p.add_argument("--year", type=int, default=2023)
    p.add_argument("--month", type=int, default=7)
    p.add_argument("--day", type=int, default=1, help="Calendar day in July.")
    p.add_argument("--hours-per-day", type=int, default=8)
    p.add_argument("--hour-mode", default="first", choices=["first", "all"])
    p.add_argument("--time-scale", default="slot", choices=["slot", "utc-hour"])

    p.add_argument("--subset-mode", default="tile", choices=["tile", "maxmin"],
                   help="tile uses one tile from --tile-grid; maxmin uses first --maxmin-points-per-hour spatial locations.")
    p.add_argument("--tile-grid", default="5x5", help="Spatial tile grid, e.g. 5x5.")
    p.add_argument("--tile-row", type=int, default=3, help="1-based row in --tile-grid.")
    p.add_argument("--tile-col", type=int, default=3, help="1-based column in --tile-grid.")
    p.add_argument("--maxmin-points-per-hour", type=int, default=400,
                   help="For --subset-mode maxmin, number of first max-min spatial locations to keep per hour.")
    p.add_argument("--max-spatial-points", type=int, default=0, help="Optional deterministic cap per hour.")
    p.add_argument("--max-total-points", type=int, default=0, help="Optional deterministic cap after ST stacking.")
    p.add_argument("--sample-seed", type=int, default=20240709)

    p.add_argument("--coord-source", default="auto", choices=["auto", "grid", "source"])
    p.add_argument("--x-col", default="auto")
    p.add_argument("--y-col", default="auto")
    p.add_argument("--value-col", default="ColumnAmountO3")
    p.add_argument(
        "--mean-design",
        default="lat_hour",
        choices=["intercept", "lat", "lat_lon", "hour", "lat_hour", "lat_lon_hour"],
    )

    p.add_argument(
        "--model-variants",
        default="reference",
        help=(
            "Comma list of variants or one group: reference, smoothness, nugget, range_lon_n1, "
            "range_time_n1, advection_n1, advection_n0, both. "
            f"Available variants: {','.join(sorted(VARIANTS))}"
        ),
    )
    p.add_argument(
        "--vecchia-summary",
        default="",
        help=(
            "CSV from the full-data Vecchia fit. The full-eigen diagnostic uses "
            "the est_* parameters from this file; fitting is not done on the tile."
        ),
    )
    p.add_argument(
        "--allow-cli-params",
        action="store_true",
        help=(
            "Allow using --sigmasq/--range-* values when --vecchia-summary is not "
            "provided. This is only for debugging, not the intended comparison."
        ),
    )
    p.add_argument("--sigmasq", type=float, default=10.0)
    p.add_argument("--range-lat", type=float, default=0.2)
    p.add_argument("--range-lon", type=float, default=0.3)
    p.add_argument("--range-time", type=float, default=2.0)
    p.add_argument("--advec-lat", type=float, default=0.08)
    p.add_argument("--advec-lon", type=float, default=-0.2)
    p.add_argument("--cov-jitter", type=float, default=1e-8)
    p.add_argument("--eigenvalue-rtol", type=float, default=1e-10)
    p.add_argument("--eigenvalue-atol", type=float, default=1e-12)
    p.add_argument("--brown-bridge-q", type=float, default=BROWN_BRIDGE_Q95)
    p.add_argument("--save-selected-points", action="store_true")
    return p.parse_args()


def parse_grid(text: str) -> tuple[int, int]:
    vals = [int(x.strip()) for x in str(text).lower().replace("x", ",").split(",") if x.strip()]
    if len(vals) != 2 or vals[0] <= 0 or vals[1] <= 0:
        raise ValueError(f"--tile-grid must look like 5x5, got {text!r}")
    return vals[0], vals[1]


def resolve_col(df: pd.DataFrame, requested: str, candidates: list[str], role: str) -> str:
    if requested != "auto":
        if requested not in df.columns:
            raise ValueError(f"{role} column {requested!r} not found. Available: {list(df.columns)}")
        return requested
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not auto-detect {role} column. Tried {candidates}; available: {list(df.columns)}")


def coord_candidates(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    if args.coord_source == "source":
        return ["Source_Longitude", "source_lon", "Longitude", "lon", "x"], [
            "Source_Latitude",
            "source_lat",
            "Latitude",
            "lat",
            "y",
        ]
    if args.coord_source == "grid":
        return ["Longitude", "lon", "x", "Source_Longitude"], [
            "Latitude",
            "lat",
            "y",
            "Source_Latitude",
        ]
    return ["Longitude", "lon", "x", "Source_Longitude", "source_lon"], [
        "Latitude",
        "lat",
        "y",
        "Source_Latitude",
        "source_lat",
    ]


def parse_gems_hour_key(key: str) -> pd.Timestamp | None:
    pat = r"^y(?P<yy>\d{2})m(?P<mm>\d{2})day(?P<dd>\d{2})_hm(?P<hh>\d{2}):(?P<minute>\d{2})$"
    m = re.match(pat, str(key))
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


def load_ordered_hours(path: Path, year: int, month: int) -> list[tuple[pd.Timestamp, str, pd.DataFrame]]:
    obj = pd.read_pickle(path)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected a dict pickle with hourly data frames, got {type(obj)}")
    out: list[tuple[pd.Timestamp, str, pd.DataFrame]] = []
    for key, val in obj.items():
        ts = parse_gems_hour_key(str(key))
        if ts is None or not isinstance(val, pd.DataFrame):
            continue
        if ts.year == int(year) and ts.month == int(month):
            out.append((ts, str(key), val))
    out.sort(key=lambda x: x[0])
    if not out:
        raise ValueError(f"No parseable {year}-{month:02d} hours found in {path}")
    return out


def select_day_hours(
    hours: list[tuple[pd.Timestamp, str, pd.DataFrame]],
    day: int,
    hour_mode: str,
    hours_per_day: int,
) -> list[tuple[pd.Timestamp, str, pd.DataFrame]]:
    selected = [h for h in hours if h[0].day == int(day)]
    selected.sort(key=lambda x: x[0])
    if not selected:
        raise ValueError(f"No hours for day={day}")
    if hour_mode == "first":
        selected = selected[: int(hours_per_day)]
    if len(selected) < int(hours_per_day):
        raise ValueError(f"Only found {len(selected)} hours for day={day}; requested {hours_per_day}.")
    return selected


def clean_hour_frame(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    lon_candidates, lat_candidates = coord_candidates(args)
    x_col = resolve_col(df, args.x_col, lon_candidates, "longitude")
    y_col = resolve_col(df, args.y_col, lat_candidates, "latitude")
    value_col = resolve_col(df, args.value_col, [args.value_col, "ColumnAmountO3", "value", "tco"], "value")
    cols = [y_col, x_col, value_col]
    out = df.loc[:, cols].copy()
    out.columns = ["lat", "lon", "y"]
    for col in ["lat", "lon", "y"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out.groupby(["lat", "lon"], as_index=False)["y"].mean()
    return out


def tile_bounds(base: pd.DataFrame, args: argparse.Namespace) -> tuple[float, float, float, float]:
    nrow, ncol = parse_grid(args.tile_grid)
    row = int(args.tile_row)
    col = int(args.tile_col)
    if row < 1 or row > nrow or col < 1 or col > ncol:
        raise ValueError(f"tile row/col must be inside {nrow}x{ncol}; got row={row}, col={col}")
    lat_edges = np.linspace(float(base["lat"].min()), float(base["lat"].max()) + EPS, nrow + 1)
    lon_edges = np.linspace(float(base["lon"].min()), float(base["lon"].max()) + EPS, ncol + 1)
    return lat_edges[row - 1], lat_edges[row], lon_edges[col - 1], lon_edges[col]


def filter_tile(df: pd.DataFrame, bounds: tuple[float, float, float, float]) -> pd.DataFrame:
    lat0, lat1, lon0, lon1 = bounds
    keep = (df["lat"] >= lat0) & (df["lat"] < lat1) & (df["lon"] >= lon0) & (df["lon"] < lon1)
    return df.loc[keep].sort_values(["lat", "lon"]).reset_index(drop=True)


def maxmin_first_indices(coords: np.ndarray, n_select: int) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    n = int(coords.shape[0])
    n_select = min(max(int(n_select), 1), n)
    mins = np.nanmin(coords, axis=0)
    spans = np.nanmax(coords, axis=0) - mins
    spans = np.where(spans > 0, spans, 1.0)
    scaled = (coords - mins) / spans

    selected = np.empty(n_select, dtype=np.int64)
    first = int(np.argmin(np.sum(scaled * scaled, axis=1)))
    selected[0] = first
    min_d2 = np.sum((scaled - scaled[first]) ** 2, axis=1)
    min_d2[first] = -1.0
    for k in range(1, n_select):
        idx = int(np.argmax(min_d2))
        selected[k] = idx
        d2 = np.sum((scaled - scaled[idx]) ** 2, axis=1)
        np.minimum(min_d2, d2, out=min_d2)
        min_d2[selected[: k + 1]] = -1.0
    return selected


def maxmin_reference_points(base: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    base_sorted = base.sort_values(["lat", "lon"]).reset_index(drop=True)
    coords = base_sorted[["lat", "lon"]].to_numpy(dtype=np.float64)
    idx = maxmin_first_indices(coords, int(args.maxmin_points_per_hour))
    ref = base_sorted.iloc[idx][["lat", "lon"]].copy().reset_index(drop=True)
    ref["maxmin_rank"] = np.arange(1, len(ref) + 1, dtype=np.int64)
    return ref


def filter_maxmin(df: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    out = ref.merge(df, on=["lat", "lon"], how="inner", sort=False)
    return out.sort_values("maxmin_rank").reset_index(drop=True)


def deterministic_cap(df: pd.DataFrame, max_points: int, seed: int) -> pd.DataFrame:
    if int(max_points) <= 0 or len(df) <= int(max_points):
        return df
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(len(df), size=int(max_points), replace=False))
    return df.iloc[idx].reset_index(drop=True)


def stack_space_time_subset(
    selected_hours: list[tuple[pd.Timestamp, str, pd.DataFrame]],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    clean = [(ts, key, clean_hour_frame(df, args)) for ts, key, df in selected_hours]
    bounds = tile_bounds(clean[0][2], args) if args.subset_mode == "tile" else None
    maxmin_ref = maxmin_reference_points(clean[0][2], args) if args.subset_mode == "maxmin" else None

    rows = []
    first_ts = clean[0][0]
    n_spatial_by_hour = []
    for slot, (ts, key, df) in enumerate(clean):
        if args.subset_mode == "tile":
            subset = filter_tile(df, bounds)
            subset = deterministic_cap(subset, int(args.max_spatial_points), int(args.sample_seed) + slot)
        else:
            subset = filter_maxmin(df, maxmin_ref)
        n_spatial_by_hour.append(len(subset))
        if args.time_scale == "utc-hour":
            t_value = (ts - first_ts) / pd.Timedelta(hours=1)
        else:
            t_value = float(slot)
        tmp = subset.copy()
        tmp["time"] = float(t_value)
        tmp["slot"] = int(slot)
        tmp["hour_key"] = key
        tmp["timestamp"] = ts.isoformat()
        rows.append(tmp)

    stacked = pd.concat(rows, ignore_index=True)
    stacked = deterministic_cap(stacked, int(args.max_total_points), int(args.sample_seed) + 99991)
    coords = stacked[["lat", "lon", "time"]].to_numpy(dtype=np.float64)
    y = stacked["y"].to_numpy(dtype=np.float64)
    meta = {
        "subset_mode": str(args.subset_mode),
        "tile_bounds": {
            "lat_min": float(bounds[0]),
            "lat_max": float(bounds[1]),
            "lon_min": float(bounds[2]),
            "lon_max": float(bounds[3]),
        } if bounds is not None else {},
        "maxmin_points_per_hour": int(args.maxmin_points_per_hour) if maxmin_ref is not None else 0,
        "n_spatial_by_hour": [int(x) for x in n_spatial_by_hour],
        "hour_keys": [key for _, key, _ in clean],
        "timestamps": [ts.isoformat() for ts, _, _ in clean],
    }
    return coords, y, stacked, meta


def build_mean_design(coords: np.ndarray, mean_design: str) -> tuple[np.ndarray, list[str]]:
    n = int(coords.shape[0])
    lat = coords[:, 0]
    lon = coords[:, 1]
    hour = coords[:, 2]
    cols = [np.ones(n, dtype=np.float64)]
    names = ["intercept"]
    if mean_design in ("lat", "lat_lon", "lat_hour", "lat_lon_hour"):
        cols.append(lat - float(np.mean(lat)))
        names.append("lat_centered")
    if mean_design in ("lat_lon", "lat_lon_hour"):
        cols.append(lon - float(np.mean(lon)))
        names.append("lon_centered")
    if mean_design in ("hour", "lat_hour", "lat_lon_hour"):
        levels = np.sort(np.unique(hour))
        for lev in levels[1:]:
            cols.append((hour == lev).astype(np.float64))
            names.append(f"hour_{lev:g}")
    x = np.column_stack(cols)
    keep_cols = []
    keep_names = []
    for j, name in enumerate(names):
        trial = x[:, keep_cols + [j]] if keep_cols else x[:, [j]]
        if np.linalg.matrix_rank(trial) > len(keep_cols):
            keep_cols.append(j)
            keep_names.append(name)
    return x[:, keep_cols].astype(np.float64, copy=False), keep_names


def matern_corr_from_distance(dist: np.ndarray, smooth: float) -> np.ndarray:
    nu = float(smooth)
    if nu <= 0:
        raise ValueError(f"Matern smoothness must be positive, got {smooth}")
    if abs(nu - 0.5) < 1e-12:
        np.exp(-dist, out=dist)
        return dist
    if abs(nu - 1.5) < 1e-12:
        z = math.sqrt(3.0) * dist
        corr = (1.0 + z) * np.exp(-z)
        return corr

    z = dist
    z *= math.sqrt(2.0 * nu)
    corr = np.empty_like(z)
    zero = z <= 0
    corr[zero] = 1.0
    positive = ~zero
    coeff = (2.0 ** (1.0 - nu)) / gamma(nu)
    zp = z[positive]
    corr[positive] = coeff * np.power(zp, nu) * kv(nu, zp)
    corr[~np.isfinite(corr)] = 0.0
    corr = np.clip(corr, 0.0, 1.0, out=corr)
    return corr


def build_st_matern_covariance(
    coords: np.ndarray,
    args: argparse.Namespace,
    variant: ModelVariant,
    params: PhysicalParams,
) -> np.ndarray:
    lat = coords[:, 0]
    lon = coords[:, 1]
    time_coord = coords[:, 2]
    u = np.empty((coords.shape[0], 3), dtype=np.float64)
    u[:, 0] = (lat - float(params.advec_lat) * time_coord) / max(float(params.range_lat), EPS)
    u[:, 1] = (lon - float(params.advec_lon) * time_coord) / max(float(params.range_lon), EPS)
    u[:, 2] = time_coord / max(float(params.range_time), EPS)

    norm2 = np.einsum("ij,ij->i", u, u)
    gram = u @ u.T
    dist2 = norm2[:, None] + norm2[None, :] - 2.0 * gram
    del gram
    np.maximum(dist2, 0.0, out=dist2)
    dist = np.sqrt(dist2, out=dist2)
    cov = matern_corr_from_distance(dist, float(variant.smooth))
    cov *= float(params.sigmasq)
    diag_add = max(float(params.nugget), 0.0) + max(float(args.cov_jitter), 0.0)
    cov.flat[:: cov.shape[0] + 1] += diag_add
    cov += cov.T
    cov *= 0.5
    return cov


def eigh_symmetric_inplace(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        return linalg.eigh(cov, overwrite_a=True, check_finite=False, driver="evd")
    except TypeError:
        return np.linalg.eigh(cov)


def full_eigen_diagnostic(
    y: np.ndarray,
    coords: np.ndarray,
    args: argparse.Namespace,
    variant: ModelVariant,
    params: PhysicalParams,
) -> tuple[pd.DataFrame, dict]:
    n = int(len(y))
    x, design_names = build_mean_design(coords, str(args.mean_design))
    rank = int(np.linalg.matrix_rank(x))
    if rank >= n:
        raise ValueError(f"Mean design rank {rank} must be smaller than n={n}")

    t0 = time.time()
    cov = build_st_matern_covariance(coords, args, variant, params)
    cov_seconds = time.time() - t0

    t1 = time.time()
    evals, evecs = eigh_symmetric_inplace(cov)
    eig_seconds = time.time() - t1
    evals = np.asarray(evals, dtype=np.float64)
    evecs = np.asarray(evecs, dtype=np.float64)

    max_eval = max(float(np.max(evals)), EPS)
    threshold = max(float(args.eigenvalue_atol), float(args.eigenvalue_rtol) * max_eval)
    keep = evals > threshold
    if not np.any(keep):
        raise ValueError("No positive covariance eigenvalues survived the threshold.")
    evals = evals[keep]
    evecs = evecs[:, keep]

    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    sqrt_evals = np.sqrt(evals)

    yt = evecs.T @ y
    xt = evecs.T @ x
    ys = yt / sqrt_evals
    xs = xt / sqrt_evals[:, None]

    xtx = xs.T @ xs
    xty = xs.T @ ys
    try:
        beta = linalg.solve(xtx, xty, assume_a="sym", check_finite=False)
        xtx_inv = linalg.inv(xtx, check_finite=False)
    except Exception:
        xtx_inv = np.linalg.pinv(xtx)
        beta = xtx_inv @ xty

    residual_scores = ys - xs @ beta
    y2 = residual_scores * residual_scores
    hat_work = xs @ xtx_inv
    leverage = np.einsum("ij,ij->i", hat_work, xs)
    expected = np.clip(1.0 - leverage, 0.0, None)

    cumsum_y2 = np.cumsum(y2)
    cumsum_expected = np.cumsum(expected)
    residual_df = max(float(np.sum(expected)), EPS)
    frac_expected = cumsum_expected / residual_df
    scaled_cumsum = cumsum_y2 / residual_df
    bridge = scaled_cumsum - frac_expected
    band_width = float(args.brown_bridge_q) * math.sqrt(2.0 / residual_df)
    loss = 0.5 * (float(np.sum(np.log(evals))) + float(np.sum(y2)) + n * math.log(2.0 * math.pi))

    curve = pd.DataFrame(
        {
            "rank_index": np.arange(1, len(evals) + 1, dtype=np.int64),
            "eigenvalue": evals,
            "score2": y2,
            "expected_increment": expected,
            "cumsum_score2": cumsum_y2,
            "cumsum_expected": cumsum_expected,
            "frac_expected": frac_expected,
            "scaled_cumsum": scaled_cumsum,
            "bridge": bridge,
            "band_lower": frac_expected - band_width,
            "band_upper": frac_expected + band_width,
        }
    )
    summary = {
        "variant": variant.name,
        "label": variant.label,
        "smooth": float(variant.smooth),
        "nugget": float(params.nugget),
        "source_model_variant": str(params.source_model_variant),
        "source_vecchia_loss_per_obs": float(params.source_loss_per_obs) if params.source_loss_per_obs is not None else np.nan,
        "est_sigmasq": float(params.sigmasq),
        "est_range_lat": float(params.range_lat),
        "est_range_lon": float(params.range_lon),
        "est_range_time": float(params.range_time),
        "est_advec_lat": float(params.advec_lat),
        "est_advec_lon": float(params.advec_lon),
        "est_nugget": float(params.nugget),
        "n_obs": n,
        "mean_rank": rank,
        "residual_df": residual_df,
        "n_eigen": int(len(evals)),
        "eigen_threshold": threshold,
        "min_kept_eigen": float(np.min(evals)),
        "max_kept_eigen": float(np.max(evals)),
        "sum_score2": float(np.sum(y2)),
        "score2_per_df": float(np.sum(y2) / residual_df),
        "loss": float(loss),
        "loss_per_obs": float(loss / max(n, 1)),
        "max_abs_bridge": float(np.max(np.abs(bridge))),
        "max_abs_bridge_scaled": float(np.max(np.abs(cumsum_y2 - cumsum_expected)) / math.sqrt(2.0 * residual_df)),
        "brown_bridge_band_width": band_width,
        "cov_seconds": cov_seconds,
        "eig_seconds": eig_seconds,
    }
    for name, val in zip(design_names, beta):
        summary[f"beta_{name}"] = float(val)

    del evecs, xs, ys, xtx, xtx_inv, residual_scores, hat_work
    gc.collect()
    return curve, summary


def expand_variants(text: str) -> list[str]:
    entries = [x.strip() for x in str(text).split(",") if x.strip()]
    if not entries:
        raise ValueError("--model-variants is empty")
    out: list[str] = []
    for entry in entries:
        if entry in VARIANT_GROUPS:
            out.extend(VARIANT_GROUPS[entry])
        elif entry in VARIANTS:
            out.append(entry)
        else:
            raise ValueError(f"Unknown model variant/group {entry!r}.")
    deduped = []
    for name in out:
        if name not in deduped:
            deduped.append(name)
    return deduped


def numeric_or_none(value: object) -> float | None:
    try:
        val = float(value)
    except Exception:
        return None
    if not np.isfinite(val):
        return None
    return val


def row_number(row: pd.Series, names: list[str], default: float | None = None) -> float:
    for name in names:
        if name in row.index:
            val = numeric_or_none(row[name])
            if val is not None:
                return val
    if default is not None:
        return float(default)
    raise KeyError(f"None of the columns {names} are present with a finite value.")


def status_ok_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "status" not in df.columns:
        return df
    ok = df[df["status"].astype(str).str.lower() == "ok"].copy()
    return ok if not ok.empty else df


def context_frame(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df
    if "year" in out.columns:
        year_match = out[pd.to_numeric(out["year"], errors="coerce") == int(args.year)]
        if not year_match.empty:
            out = year_match
    if "day_idx" in out.columns:
        day_match = out[pd.to_numeric(out["day_idx"], errors="coerce") == int(args.day) - 1]
        if not day_match.empty:
            out = day_match
    elif "day" in out.columns:
        day_text = out["day"].astype(str)
        day_match = out[day_text.str.contains(f"{int(args.year)}-07-{int(args.day):02d}", regex=False)]
        if day_match.empty:
            day_match = out[day_text.str.endswith(f"-{int(args.day):02d}")]
        if not day_match.empty:
            out = day_match
    return out


def matching_variant_frame(df: pd.DataFrame, name: str, variant: ModelVariant) -> pd.DataFrame:
    if "model_variant" in df.columns:
        aliases = [name, *VARIANT_ALIASES.get(name, [])]
        match = df[df["model_variant"].astype(str).isin(aliases)].copy()
        if not match.empty:
            return match
    smooth_cols = [c for c in ["smooth", "fit_s", "fixed_smooth", "diag_s"] if c in df.columns]
    nugget_cols = [c for c in ["est_nugget", "nugget", "fixed_nugget"] if c in df.columns]
    if smooth_cols:
        smooth_vals = pd.to_numeric(df[smooth_cols[0]], errors="coerce")
        match = df[np.isclose(smooth_vals, float(variant.smooth), atol=1e-10, rtol=0.0)].copy()
        if nugget_cols and not match.empty:
            nugget_vals = pd.to_numeric(match[nugget_cols[0]], errors="coerce")
            nugget_match = match[np.isclose(nugget_vals, float(variant.nugget), atol=1e-8, rtol=0.0)].copy()
            if not nugget_match.empty:
                match = nugget_match
        if not match.empty:
            return match
    return pd.DataFrame()


def load_vecchia_params(
    summary_path: Path,
    variants: list[str],
    args: argparse.Namespace,
) -> dict[str, PhysicalParams]:
    df = pd.read_csv(summary_path)
    df = context_frame(status_ok_frame(df), args)
    if df.empty:
        raise ValueError(f"No usable rows in Vecchia summary: {summary_path}")

    params: dict[str, PhysicalParams] = {}
    for name in variants:
        variant = VARIANTS[name]
        match = matching_variant_frame(df, name, variant)
        if match.empty:
            raise ValueError(
                f"Could not find full-data Vecchia fit row for variant {name!r} "
                f"in {summary_path}. Available model_variant values: "
                f"{sorted(df['model_variant'].astype(str).unique()) if 'model_variant' in df.columns else 'N/A'}"
            )
        if len(match) > 1:
            sort_cols = [c for c in ["vecchia_loss_per_obs", "loss_per_obs", "day_idx"] if c in match.columns]
            if sort_cols:
                match = match.sort_values(sort_cols)
        row = match.iloc[0]
        source_variant = str(row["model_variant"]) if "model_variant" in row.index else name
        loss = None
        for col in ["vecchia_loss_per_obs", "loss_per_obs", "conditional_loss_per_score"]:
            if col in row.index:
                loss = numeric_or_none(row[col])
                if loss is not None:
                    break
        params[name] = PhysicalParams(
            sigmasq=row_number(row, ["est_sigmasq", "sigmasq"]),
            range_lat=row_number(row, ["est_range_lat", "range_lat"]),
            range_lon=row_number(row, ["est_range_lon", "range_lon"]),
            range_time=row_number(row, ["est_range_time", "range_time"]),
            advec_lat=row_number(row, ["est_advec_lat", "advec_lat"]),
            advec_lon=row_number(row, ["est_advec_lon", "advec_lon"]),
            nugget=row_number(row, ["est_nugget", "nugget", "fixed_nugget"], default=variant.nugget),
            source_model_variant=source_variant,
            source_loss_per_obs=loss,
        )
    return params


def cli_params_for_variants(args: argparse.Namespace, variants: list[str]) -> dict[str, PhysicalParams]:
    if not args.allow_cli_params:
        raise ValueError(
            "Provide --vecchia-summary from the full-data Vecchia fit. "
            "Use --allow-cli-params only for debugging fixed-parameter runs."
        )
    out = {}
    for name in variants:
        variant = VARIANTS[name]
        out[name] = PhysicalParams(
            sigmasq=float(args.sigmasq),
            range_lat=float(args.range_lat),
            range_lon=float(args.range_lon),
            range_time=float(args.range_time),
            advec_lat=float(args.advec_lat),
            advec_lon=float(args.advec_lon),
            nugget=float(variant.nugget),
            source_model_variant="cli_fixed_params",
            source_loss_per_obs=None,
        )
    return out


def plot_model_comparison(
    curves: pd.DataFrame,
    summaries: pd.DataFrame,
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    for variant, group in curves.groupby("variant", sort=False):
        info = VARIANTS[str(variant)]
        row = summaries.loc[summaries["variant"] == str(variant)].iloc[0]
        label = f"{info.label} full-loss/obs={float(row['loss_per_obs']):.5f}"
        if "source_vecchia_loss_per_obs" in row.index:
            source_loss = numeric_or_none(row["source_vecchia_loss_per_obs"])
            if source_loss is not None:
                label += f", vecchia={source_loss:.5f}"
        ax.plot(
            group["frac_expected"],
            group["scaled_cumsum"],
            color=info.color,
            linewidth=2.0,
            label=label,
        )
    if "band_lower" in curves.columns:
        first = curves[curves["variant"] == curves["variant"].iloc[0]]
        ax.plot(first["frac_expected"], first["band_lower"], color="0.65", linestyle=(0, (4, 4)), linewidth=1.0)
        ax.plot(first["frac_expected"], first["band_upper"], color="0.65", linestyle=(0, (4, 4)), linewidth=1.0)
    ax.plot([0, 1], [0, 1], color="0.45", linewidth=1.2)
    ax.set_xlim(0.0, 1.0)
    y_max = max(1.05, float(np.nanmax(curves["scaled_cumsum"])) * 1.03)
    ax.set_ylim(0.0, y_max)
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("projected expected df fraction, sorted by full ST covariance eigenvalue")
    ax.set_ylabel("projected cumulative squared score / residual df")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_run_config(
    out_dir: Path,
    args: argparse.Namespace,
    meta: dict,
    variants: list[str],
    params_by_variant: dict[str, PhysicalParams],
) -> None:
    payload = vars(args).copy()
    payload["variants_expanded"] = variants
    payload["subset_meta"] = meta
    payload["vecchia_params_by_variant"] = {
        name: {
            "sigmasq": float(params.sigmasq),
            "range_lat": float(params.range_lat),
            "range_lon": float(params.range_lon),
            "range_time": float(params.range_time),
            "advec_lat": float(params.advec_lat),
            "advec_lon": float(params.advec_lon),
            "nugget": float(params.nugget),
            "source_model_variant": str(params.source_model_variant),
            "source_loss_per_obs": params.source_loss_per_obs,
        }
        for name, params in params_by_variant.items()
    }
    payload["model_defaults"] = {
        "sigmasq": float(args.sigmasq),
        "range_lat": float(args.range_lat),
        "range_lon": float(args.range_lon),
        "range_time": float(args.range_time),
        "advec_lat": float(args.advec_lat),
        "advec_lon": float(args.advec_lon),
        "cov_jitter": float(args.cov_jitter),
    }
    (out_dir / "run_config.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def subset_output_name(args: argparse.Namespace) -> str:
    if args.subset_mode == "maxmin":
        return f"year{int(args.year)}_day{int(args.day):02d}_maxmin{int(args.maxmin_points_per_hour)}_per_hour"
    return f"year{int(args.year)}_day{int(args.day):02d}_tile{int(args.tile_row)}x{int(args.tile_col)}_of_{args.tile_grid}"


def subset_title(args: argparse.Namespace, n_hours: int) -> str:
    if args.subset_mode == "maxmin":
        return (
            f"July {args.year} day={args.day}, {n_hours}h, "
            f"first {int(args.maxmin_points_per_hour)} max-min spatial points/hour: full ST eigen diagnostic"
        )
    return (
        f"July {args.year} day={args.day}, {n_hours}h, "
        f"tile r{args.tile_row}c{args.tile_col} of {args.tile_grid}: full ST eigen diagnostic"
    )


def main() -> None:
    args = parse_args()
    variants = expand_variants(args.model_variants)
    out_dir = Path(args.output_root) / subset_output_name(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    if str(args.vecchia_summary).strip():
        params_by_variant = load_vecchia_params(Path(args.vecchia_summary), variants, args)
    else:
        params_by_variant = cli_params_for_variants(args, variants)

    hours = load_ordered_hours(Path(args.input), int(args.year), int(args.month))
    selected = select_day_hours(hours, int(args.day), str(args.hour_mode), int(args.hours_per_day))
    coords, y, selected_points, meta = stack_space_time_subset(selected, args)
    write_run_config(out_dir, args, meta, variants, params_by_variant)
    if args.save_selected_points:
        selected_points.to_csv(out_dir / "selected_space_time_points.csv", index=False)

    n = int(len(y))
    dense_gib = n * n * 8.0 / (1024.0**3)
    print(
        f"Loaded ST subset: n={n}, hours={len(selected)}, "
        f"spatial/hour={meta['n_spatial_by_hour']}, dense_matrix={dense_gib:.3f} GiB",
        flush=True,
    )
    print(f"Variants: {variants}", flush=True)
    if str(args.vecchia_summary).strip():
        print(f"Using full-data Vecchia fit summary: {args.vecchia_summary}", flush=True)
        for name in variants:
            p = params_by_variant[name]
            print(
                f"  {name}: source={p.source_model_variant}, "
                f"sigmasq={p.sigmasq:.6g}, ranges=({p.range_lat:.6g},{p.range_lon:.6g},{p.range_time:.6g}), "
                f"advec=({p.advec_lat:.6g},{p.advec_lon:.6g}), nugget={p.nugget:.6g}",
                flush=True,
            )
    else:
        print("WARNING: using CLI fixed parameters; no Vecchia summary was provided.", flush=True)
    print(f"Output: {out_dir}", flush=True)

    all_curves = []
    all_summaries = []
    for name in variants:
        variant = VARIANTS[name]
        print(f"\n--- Full eigen diagnostic: {variant.label} ---", flush=True)
        t0 = time.time()
        curve, summary = full_eigen_diagnostic(y, coords, args, variant, params_by_variant[name])
        elapsed = time.time() - t0
        summary["total_seconds"] = float(elapsed)
        for key, val in meta["tile_bounds"].items():
            summary[key] = val
        summary["subset_mode"] = str(args.subset_mode)
        summary["tile_grid"] = str(args.tile_grid)
        summary["tile_row"] = int(args.tile_row)
        summary["tile_col"] = int(args.tile_col)
        summary["maxmin_points_per_hour"] = int(args.maxmin_points_per_hour) if args.subset_mode == "maxmin" else 0
        summary["n_hours"] = int(len(selected))
        curve.insert(0, "variant", name)
        all_curves.append(curve)
        all_summaries.append(summary)
        print(
            f"{variant.label}: loss/obs={summary['loss_per_obs']:.6f}, "
            f"score2/df={summary['score2_per_df']:.6f}, "
            f"eig_seconds={summary['eig_seconds']:.1f}, total={elapsed:.1f}s",
            flush=True,
        )

    curves_df = pd.concat(all_curves, ignore_index=True)
    summary_df = pd.DataFrame(all_summaries)
    curves_df.round(ROUND_DECIMALS).to_csv(out_dir / "st_full_eigen_curves.csv", index=False, float_format="%.8f")
    summary_df.round(ROUND_DECIMALS).to_csv(out_dir / "st_full_eigen_summary.csv", index=False, float_format="%.8f")

    title = subset_title(args, len(selected))
    plot_model_comparison(curves_df, summary_df, out_dir / "st_full_eigen_model_comparison.png", title)
    print("\nDone.", flush=True)
    print(f"Summary: {out_dir / 'st_full_eigen_summary.csv'}", flush=True)
    print(f"Plot: {out_dir / 'st_full_eigen_model_comparison.png'}", flush=True)


if __name__ == "__main__":
    main()
