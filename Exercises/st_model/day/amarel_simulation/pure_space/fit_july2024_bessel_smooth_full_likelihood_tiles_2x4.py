#!/usr/bin/env python3
"""July 2024 tile-wise anisotropic Matern fits with exact full likelihood.

This entrypoint fits the first 240 observed July 2024 hours.  Each hour is
split into a 2x4 lat/lon tile grid, and each tile is fitted independently with
an anisotropic pure-space Matern covariance:

    sigmasq, range_lat, range_lon, smooth, nugget

Smoothness is estimated, not fixed.  The covariance uses the direct Bessel
Matern formula from GEMS_TCO.matern_bessel_anisotropic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, solve_triangular

try:
    from GEMS_TCO.matern_bessel_anisotropic import (
        covariance_from_deltas,
        fit_full_matern,
        make_mean_design,
        natural_from_raw,
        pairwise_deltas,
    )
except ImportError:
    _candidates = [
        Path(__file__).parents[5] / "src",
        Path("/home/jl2815/tco"),
    ]
    for _p in _candidates:
        if (_p / "GEMS_TCO").is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
            break
    from GEMS_TCO.matern_bessel_anisotropic import (
        covariance_from_deltas,
        fit_full_matern,
        make_mean_design,
        natural_from_raw,
        pairwise_deltas,
    )


METHOD = "full_likelihood"


def parse_float_pair(text: str) -> list[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(vals) != 2:
        raise argparse.ArgumentTypeError("range must look like -3,2")
    return [min(vals), max(vals)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="July 2024 2x4 tile Bessel-smooth full likelihood fits.")
    p.add_argument("--mode", choices=["manifest", "fit", "summarize", "all"], required=True)
    p.add_argument("--input", default=os.environ.get("DATA_PATH"))
    p.add_argument("--output-dir", default=os.environ.get(
        "OUTDIR",
        "/home/jl2815/tco/exercise_output/summer/july2024_bessel_smooth/full_likelihood_2x4",
    ))
    p.add_argument("--monthly-output-dir", default=os.environ.get("MONTHLY_OUTDIR", ""))
    p.add_argument("--manifest", default=None)
    p.add_argument("--month", default=os.environ.get("MONTH", "2024-07"))
    p.add_argument("--max-hours", type=int, default=int(os.environ.get("MAX_HOURS", "240")))
    p.add_argument("--expected-hours", type=int, default=int(os.environ.get("EXPECTED_HOURS", "240")))

    p.add_argument("--time-col", default=os.environ.get("TIME_COL", "auto"))
    p.add_argument("--x-col", default=os.environ.get("X_COL", "auto"))
    p.add_argument("--y-col", default=os.environ.get("Y_COL", "auto"))
    p.add_argument("--value-col", default=os.environ.get("VALUE_COL", "auto"))
    p.add_argument("--qa-col", default=os.environ.get("QA_COL", ""))
    p.add_argument("--qa-min", type=float, default=None)

    p.add_argument("--coords", choices=["raw", "lonlat"], default=os.environ.get("COORDS", "raw"))
    p.add_argument("--lat-range", type=parse_float_pair, default=parse_float_pair(os.environ.get("LAT_RANGE", "-3,2")))
    p.add_argument("--lon-range", type=parse_float_pair, default=parse_float_pair(os.environ.get("LON_RANGE", "121,131")))
    p.add_argument("--tile-y", type=int, default=int(os.environ.get("TILE_Y", "2")))
    p.add_argument("--tile-x", type=int, default=int(os.environ.get("TILE_X", "4")))
    p.add_argument("--min-tile-points", type=int, default=int(os.environ.get("MIN_TILE_POINTS", "200")))
    p.add_argument("--tile-max-points", type=int, default=int(os.environ.get("TILE_MAX_POINTS", "0")))
    p.add_argument("--tile-workers", type=int, default=int(os.environ.get("TILE_WORKERS", "1")))
    p.add_argument("--sample-seed", type=int, default=int(os.environ.get("SAMPLE_SEED", "202407")))

    p.add_argument("--nugget-mode", choices=["free", "fixed0"], default=os.environ.get("NUGGET_MODE", "free"))
    p.add_argument("--mean-design", choices=["constant", "lat", "latlon"], default=os.environ.get("MEAN_DESIGN", "lat"))
    p.add_argument("--range-lat-init", type=float, default=float(os.environ.get("RANGE_LAT_INIT", "0.35")))
    p.add_argument("--range-lon-init", type=float, default=float(os.environ.get("RANGE_LON_INIT", "0.35")))
    p.add_argument("--smooth-init", type=float, default=float(os.environ.get("SMOOTH_INIT", "0.5")))
    p.add_argument("--nugget-init", type=float, default=None)
    p.add_argument("--smooth-min", type=float, default=float(os.environ.get("SMOOTH_MIN", "0.05")))
    p.add_argument("--smooth-max", type=float, default=float(os.environ.get("SMOOTH_MAX", "2.5")))
    p.add_argument("--range-min", type=float, default=float(os.environ.get("RANGE_MIN", "0.03")))
    p.add_argument("--range-max", type=float, default=float(os.environ.get("RANGE_MAX", "5.0")))
    p.add_argument("--jitter", type=float, default=float(os.environ.get("JITTER", "1e-6")))
    p.add_argument("--n-restarts", type=int, default=int(os.environ.get("N_RESTARTS", "1")))
    p.add_argument("--maxiter", type=int, default=int(os.environ.get("MAXITER", "80")))
    p.add_argument("--maxfun", type=int, default=int(os.environ.get("MAXFUN", "0")))
    p.add_argument("--maxls", type=int, default=int(os.environ.get("MAXLS", "20")))
    p.add_argument("--maxcor", type=int, default=int(os.environ.get("MAXCOR", "20")))
    p.add_argument("--optimizer-method", default=os.environ.get("OPTIMIZER_METHOD", "L-BFGS-B"))
    p.add_argument(
        "--outlier-whitened-threshold",
        type=float,
        default=float(os.environ.get("OUTLIER_WHITENED_THRESHOLD", "10")),
        help="If >0, fit once, mark |whitened residual| above this value as missing, then refit the tile.",
    )

    p.add_argument("--array-index", type=int, default=None)
    p.add_argument("--hour", default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def output_root(args: argparse.Namespace) -> Path:
    return Path(args.output_dir) / str(args.nugget_mode)


def monthly_output_dir(args: argparse.Namespace) -> Path:
    if args.monthly_output_dir:
        return Path(args.monthly_output_dir)
    return Path(args.output_dir).parent / "monthly_output"


def resolve_manifest(args: argparse.Namespace) -> Path:
    return Path(args.manifest) if args.manifest else Path(args.output_dir) / "manifest_hours.csv"


def require_input(args: argparse.Namespace) -> Path:
    if not args.input:
        raise SystemExit("Missing --input or DATA_PATH.")
    path = Path(args.input).expanduser()
    if not path.exists():
        raise SystemExit(f"Input path does not exist: {path}")
    return path


def read_one_file(path: Path):
    suffix = path.suffix.lower()
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input file type: {path}")


def parse_gems_hour_key(key: str) -> pd.Timestamp | None:
    m = re.match(r"^y(?P<yy>\d{2})m(?P<mm>\d{2})day(?P<dd>\d{2})_hm(?P<hh>\d{2}):(?P<minute>\d{2})$", str(key))
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


def choose_col(df: pd.DataFrame, requested: str, candidates: Iterable[str], role: str) -> str:
    if requested and requested != "auto":
        if requested not in df.columns:
            raise SystemExit(f"{role} column {requested!r} not found. Columns: {list(df.columns)}")
        return requested
    lowered = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lowered:
            return lowered[name.lower()]
    raise SystemExit(f"Could not auto-detect {role} column. Columns: {list(df.columns)}")


def column_names(df: pd.DataFrame, args: argparse.Namespace) -> tuple[str, str, str, str]:
    time_col = choose_col(df, args.time_col, ["time", "datetime", "timestamp", "hour"], "time")
    x_col = choose_col(df, args.x_col, ["Longitude", "longitude", "lon", "x"], "x")
    y_col = choose_col(df, args.y_col, ["Latitude", "latitude", "lat", "y"], "y")
    value_col = choose_col(df, args.value_col, ["ColumnAmountO3", "value", "tco", "column"], "value")
    return time_col, x_col, y_col, value_col


def apply_quality_filter(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.qa_col and args.qa_min is not None:
        if args.qa_col not in df.columns:
            raise SystemExit(f"QA column {args.qa_col!r} not found.")
        return df[df[args.qa_col] >= args.qa_min]
    return df


def month_bounds(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    return start, start + pd.DateOffset(months=1)


def make_manifest(args: argparse.Namespace) -> None:
    obj = read_one_file(require_input(args))
    rows = []
    start, end = month_bounds(args.month)
    if isinstance(obj, dict):
        for key, value in obj.items():
            if not isinstance(value, pd.DataFrame):
                continue
            hour_exact = parse_gems_hour_key(str(key))
            if hour_exact is None:
                continue
            hour = hour_exact.floor("h")
            if hour < start or hour >= end:
                continue
            frame = apply_quality_filter(value, args)
            rows.append({
                "hour": hour.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "hour_exact": hour_exact.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "hour_key": str(key),
                "n_rows": int(len(frame)),
                "source_file": str(require_input(args)),
            })
    else:
        time_col, _, _, _ = column_names(obj, args)
        d = obj.copy()
        d["_time_utc"] = pd.to_datetime(d[time_col], utc=True, errors="coerce")
        d["_hour_utc"] = d["_time_utc"].dt.floor("h")
        d = d[(d["_hour_utc"] >= start) & (d["_hour_utc"] < end)]
        d = apply_quality_filter(d, args)
        rows = [
            {"hour": h.strftime("%Y-%m-%dT%H:%M:%SZ"), "hour_exact": h.strftime("%Y-%m-%dT%H:%M:%SZ"), "hour_key": "", "n_rows": int(n), "source_file": str(require_input(args))}
            for h, n in d.groupby("_hour_utc").size().items()
        ]
    if not rows:
        raise SystemExit(f"No observed hours found for {args.month}.")
    manifest = pd.DataFrame(rows).sort_values("hour").reset_index(drop=True)
    if args.max_hours and len(manifest) > int(args.max_hours):
        manifest = manifest.iloc[: int(args.max_hours)].copy()
    manifest.insert(0, "hour_index", np.arange(len(manifest), dtype=int))
    hour_dt = pd.to_datetime(manifest["hour"], utc=True)
    manifest["day_index"] = hour_dt.dt.day.astype(int)
    manifest["hour_utc"] = hour_dt.dt.hour.astype(int)
    manifest["hour_slot"] = manifest.groupby("day_index", sort=True).cumcount().astype(int)
    manifest = manifest[[
        "hour_index", "day_index", "hour_slot", "hour_utc",
        "hour", "hour_exact", "hour_key", "n_rows", "source_file",
    ]]
    path = resolve_manifest(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(path, index=False)
    print(f"Wrote manifest: {path}")
    print(f"Observed hours kept: {len(manifest)}")
    if args.expected_hours and len(manifest) != args.expected_hours:
        print(f"WARNING: expected {args.expected_hours}, kept {len(manifest)}")


def read_manifest(args: argparse.Namespace) -> pd.DataFrame:
    path = resolve_manifest(args)
    if not path.exists():
        raise SystemExit(f"Manifest does not exist: {path}")
    return pd.read_csv(path)


def choose_hour(args: argparse.Namespace, manifest: pd.DataFrame) -> tuple[int, pd.Timestamp, pd.Series]:
    if args.hour:
        hour = pd.to_datetime(args.hour, utc=True)
        matches = manifest.index[pd.to_datetime(manifest["hour"], utc=True) == hour].tolist()
        if not matches:
            raise SystemExit(f"Hour {args.hour} not found in manifest.")
        idx = int(matches[0])
    else:
        idx = args.array_index
        if idx is None:
            idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if idx < 0 or idx >= len(manifest):
        raise SystemExit(f"array index {idx} outside manifest length {len(manifest)}")
    row = manifest.iloc[int(idx)]
    return int(row["hour_index"]), pd.to_datetime(row["hour"], utc=True), row


def read_hour_table(args: argparse.Namespace, manifest_row: pd.Series) -> pd.DataFrame:
    obj = read_one_file(require_input(args))
    if isinstance(obj, dict):
        key = str(manifest_row["hour_key"])
        if key not in obj:
            raise SystemExit(f"hour_key {key} not found in input pickle")
        out = obj[key].copy()
        out["hour_key"] = key
        out["hour"] = str(manifest_row["hour"])
        return apply_quality_filter(out, args)
    time_col, _, _, _ = column_names(obj, args)
    hour = pd.to_datetime(manifest_row["hour"], utc=True)
    d = obj.copy()
    d["_time_utc"] = pd.to_datetime(d[time_col], utc=True, errors="coerce")
    d["_hour_utc"] = d["_time_utc"].dt.floor("h")
    return apply_quality_filter(d[d["_hour_utc"] == hour].copy(), args)


def lonlat_to_km(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon0 = float(np.nanmedian(lon))
    lat0 = float(np.nanmedian(lat))
    x = (lon - lon0) * 111.320 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 110.574
    return x, y


def prepare_hour_data(df: pd.DataFrame, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    _, x_col, y_col, value_col = column_names(df, args)
    d = df[[x_col, y_col, value_col]].replace([np.inf, -np.inf], np.nan).copy()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[x_col, y_col])
    d = d[
        d[y_col].between(args.lat_range[0], args.lat_range[1])
        & d[x_col].between(args.lon_range[0], args.lon_range[1])
    ].copy()
    if d.empty:
        raise ValueError(f"No rows after region filter lat={args.lat_range}, lon={args.lon_range}.")
    meta = {
        "lat_min": float(d[y_col].min()),
        "lat_max": float(d[y_col].max()),
        "lon_min": float(d[x_col].min()),
        "lon_max": float(d[x_col].max()),
        "n_after_region_filter": int(len(d)),
    }
    d = d.dropna(subset=[value_col])
    d = d.groupby([x_col, y_col], as_index=False)[value_col].mean()
    raw_lon = d[x_col].to_numpy(dtype=float)
    raw_lat = d[y_col].to_numpy(dtype=float)
    if args.coords == "lonlat":
        x, y = lonlat_to_km(raw_lon, raw_lat)
    else:
        x, y = raw_lon, raw_lat
    coords = np.column_stack([y, x])
    values = d[value_col].to_numpy(dtype=float)
    meta["n_used"] = int(len(values))
    meta["value_mean"] = float(np.mean(values))
    meta["value_sd"] = float(np.std(values))
    return coords, values, d.reset_index(drop=True), meta


def assign_tiles(coords: np.ndarray, tile_y_count: int, tile_x_count: int) -> tuple[np.ndarray, dict]:
    y = coords[:, 0]
    x = coords[:, 1]
    eps_y = max((float(y.max()) - float(y.min())) * 1e-12, 1e-12)
    eps_x = max((float(x.max()) - float(x.min())) * 1e-12, 1e-12)
    y_edges = np.linspace(float(y.min()), float(y.max()) + eps_y, tile_y_count + 1)
    x_edges = np.linspace(float(x.min()), float(x.max()) + eps_x, tile_x_count + 1)
    y_idx = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, tile_y_count - 1)
    x_idx = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, tile_x_count - 1)
    return (y_idx * tile_x_count + x_idx).astype(int), {"y_edges": y_edges.tolist(), "x_edges": x_edges.tolist()}


def deterministic_subset(n: int, max_points: int) -> np.ndarray:
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=int)
    return np.unique(np.linspace(0, n - 1, int(max_points)).round().astype(int))


def tile_geometry(tile_id: int, tile_meta: dict, tile_x_count: int) -> dict:
    y_idx = int(tile_id // tile_x_count)
    x_idx = int(tile_id % tile_x_count)
    y_edges = np.asarray(tile_meta["y_edges"], dtype=float)
    x_edges = np.asarray(tile_meta["x_edges"], dtype=float)
    return {
        "tile_id": int(tile_id),
        "tile_y": y_idx,
        "tile_x": x_idx,
        "tile_lat_min": float(y_edges[y_idx]),
        "tile_lat_max": float(y_edges[y_idx + 1]),
        "tile_lon_min": float(x_edges[x_idx]),
        "tile_lon_max": float(x_edges[x_idx + 1]),
        "tile_center_lat": float(0.5 * (y_edges[y_idx] + y_edges[y_idx + 1])),
        "tile_center_lon": float(0.5 * (x_edges[x_idx] + x_edges[x_idx + 1])),
    }


def _append_message(existing: object, addition: str) -> str:
    msg = "" if existing is None or (isinstance(existing, float) and np.isnan(existing)) else str(existing)
    return addition if not msg else f"{msg}; {addition}"


def _fit_has_usable_qc_params(fit: dict) -> bool:
    raw = fit.get("raw_params")
    if raw is None:
        return False
    try:
        raw_arr = np.asarray(raw, dtype=float)
        loss = float(fit.get("loss", fit.get("nll", np.nan)))
    except (TypeError, ValueError):
        return False
    return raw_arr.size > 0 and bool(np.all(np.isfinite(raw_arr))) and bool(np.isfinite(loss))


def full_whitened_residuals(
    y: np.ndarray,
    coords: np.ndarray,
    fit: dict,
    nugget_mode: str,
    mean_design: str,
    smooth_bounds: tuple[float, float],
    jitter: float,
) -> np.ndarray:
    """Return L^{-1}(y-X beta_hat) under a fitted full GP tile model."""
    raw = fit.get("raw_params")
    if raw is None:
        raise ValueError("fit record has no raw_params")
    params = natural_from_raw(raw, nugget_mode, 0.0, smooth_bounds)
    d_lat, d_lon = pairwise_deltas(coords)
    cov = covariance_from_deltas(d_lat, d_lon, params, jitter=float(jitter))
    X = make_mean_design(coords, mean_design)
    c, lower = cho_factor(cov, lower=True, check_finite=False)
    kinv_y = cho_solve((c, lower), y, check_finite=False)
    kinv_X = cho_solve((c, lower), X, check_finite=False)
    xt_k_x = X.T @ kinv_X
    xt_k_y = X.T @ kinv_y
    beta = np.linalg.solve(xt_k_x + np.eye(X.shape[1]) * 1e-8, xt_k_y)
    resid = y - X @ beta
    return solve_triangular(c, resid, lower=lower, check_finite=False)


def _fit_full_tile_once(y: np.ndarray, coords: np.ndarray, task: dict) -> dict:
    return fit_full_matern(
        y=y,
        coords=coords,
        nugget_mode=str(task["nugget_mode"]),
        fixed_nugget=0.0,
        mean_design=str(task["mean_design"]),
        smooth_bounds=tuple(task["smooth_bounds"]),
        range_bounds=tuple(task["range_bounds"]),
        range_lat_init=float(task["range_lat_init"]),
        range_lon_init=float(task["range_lon_init"]),
        smooth_init=float(task["smooth_init"]),
        nugget_init=task["nugget_init"],
        jitter=float(task["jitter"]),
        n_restarts=int(task["n_restarts"]),
        maxiter=int(task["maxiter"]),
        maxfun=int(task["maxfun"]),
        maxls=int(task["maxls"]),
        maxcor=int(task["maxcor"]),
        method=str(task["optimizer_method"]),
    )


def fit_full_tile_with_outlier_qc(
    y: np.ndarray,
    coords: np.ndarray,
    task: dict,
) -> tuple[dict, np.ndarray, dict]:
    """Fit a tile, optionally mask extreme fitted whitened residuals, and refit."""
    threshold = float(task.get("outlier_whitened_threshold", 0.0))
    initial_fit = _fit_full_tile_once(y, coords, task)
    qc = {
        "outlier_whitened_threshold": threshold,
        "n_qc_initial_fit": int(len(y)),
        "n_qc_removed": 0,
        "n_qc_fit": int(len(y)),
        "qc_max_abs_whitened": np.nan,
        "qc_refit": False,
        "qc_initial_success": bool(initial_fit.get("success", False)),
    }
    if threshold <= 0.0 or not _fit_has_usable_qc_params(initial_fit):
        return initial_fit, np.ones(len(y), dtype=bool), qc

    try:
        w = full_whitened_residuals(
            y=y,
            coords=coords,
            fit=initial_fit,
            nugget_mode=str(task["nugget_mode"]),
            mean_design=str(task["mean_design"]),
            smooth_bounds=tuple(task["smooth_bounds"]),
            jitter=float(task["jitter"]),
        )
    except Exception as exc:
        initial_fit["message"] = _append_message(
            initial_fit.get("message", ""),
            f"outlier_qc_skipped_whitening_error {exc}",
        )
        return initial_fit, np.ones(len(y), dtype=bool), qc
    abs_w = np.abs(w)
    bad = np.isfinite(abs_w) & (abs_w > threshold)
    keep = ~bad
    qc["qc_max_abs_whitened"] = float(np.nanmax(abs_w)) if abs_w.size else np.nan
    qc["n_qc_removed"] = int(bad.sum())
    qc["n_qc_fit"] = int(keep.sum())
    if int(bad.sum()) == 0:
        return initial_fit, keep, qc
    min_refit_points = max(10, int(make_mean_design(coords, str(task["mean_design"])).shape[1]) + 2)
    if int(keep.sum()) < min_refit_points:
        initial_fit["message"] = _append_message(
            initial_fit.get("message", ""),
            f"outlier_qc_skipped_too_few_after_removal {len(y)}->{int(keep.sum())}",
        )
        qc["n_qc_fit"] = int(len(y))
        return initial_fit, np.ones(len(y), dtype=bool), qc

    refit = _fit_full_tile_once(y[keep], coords[keep], task)
    refit["message"] = _append_message(
        refit.get("message", ""),
        f"whitened_outlier_qc |r|>{threshold:g} removed {int(bad.sum())}/{len(y)}",
    )
    refit["pre_qc_loss"] = initial_fit.get("loss", np.nan)
    refit["pre_qc_nll"] = initial_fit.get("nll", np.nan)
    qc["qc_refit"] = True
    return refit, keep, qc


def _fit_tile_worker(task: dict) -> dict:
    base = dict(task["base"])
    y_all = np.asarray(task["y"], dtype=np.float64)
    coords_all = np.asarray(task["coords"], dtype=np.float64)
    n_tile = int(y_all.shape[0])
    if n_tile < int(task["min_tile_points"]):
        base.update({
            "n": n_tile,
            "n_fit": 0,
            "success": False,
            "message": "too_few_points",
            "loss": np.nan,
            "nll": np.nan,
        })
        return base
    keep = deterministic_subset(n_tile, int(task["tile_max_points"]))
    y = y_all[keep]
    coords = coords_all[keep]
    try:
        fit, _qc_keep, qc = fit_full_tile_with_outlier_qc(y, coords, task)
        base.update(fit)
        base.update({
            "n": n_tile,
            "n_fit": int(qc.get("n_qc_fit", int(len(y)))),
            "n_initial_fit": int(len(y)),
            "fit_method": METHOD,
            **qc,
        })
        if int(task["tile_max_points"]) > 0 and n_tile > int(task["tile_max_points"]):
            base["message"] = f"{base.get('message', '')}; deterministic_thin {n_tile}->{len(y)}"
        return base
    except Exception as exc:
        base.update({
            "n": n_tile,
            "n_fit": int(len(y)),
            "success": False,
            "message": f"ERROR: {exc}",
            "loss": np.nan,
            "nll": np.nan,
            "fit_method": METHOD,
        })
        return base


def fit_tiles_for_hour(coords: np.ndarray, values: np.ndarray, tile_id: np.ndarray, tile_meta: dict, args: argparse.Namespace) -> pd.DataFrame:
    tasks = []
    tile_count = int(args.tile_y) * int(args.tile_x)
    for tid in range(tile_count):
        mask = tile_id == tid
        base = tile_geometry(tid, tile_meta, int(args.tile_x))
        tasks.append({
            "base": base,
            "y": values[mask],
            "coords": coords[mask],
            "min_tile_points": int(args.min_tile_points),
            "tile_max_points": int(args.tile_max_points),
            "nugget_mode": str(args.nugget_mode),
            "mean_design": str(args.mean_design),
            "smooth_bounds": (float(args.smooth_min), float(args.smooth_max)),
            "range_bounds": (float(args.range_min), float(args.range_max)),
            "range_lat_init": float(args.range_lat_init),
            "range_lon_init": float(args.range_lon_init),
            "smooth_init": float(args.smooth_init),
            "nugget_init": args.nugget_init,
            "jitter": float(args.jitter),
            "n_restarts": int(args.n_restarts),
            "maxiter": int(args.maxiter),
            "maxfun": int(args.maxfun),
            "maxls": int(args.maxls),
            "maxcor": int(args.maxcor),
            "optimizer_method": str(args.optimizer_method),
            "outlier_whitened_threshold": float(args.outlier_whitened_threshold),
        })
    workers = max(1, int(args.tile_workers))
    if workers == 1:
        rows = [_fit_tile_worker(t) for t in tasks]
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(_fit_tile_worker, t) for t in tasks]
            for fut in as_completed(futures):
                rows.append(fut.result())
    return pd.DataFrame(rows).sort_values(["tile_y", "tile_x"]).reset_index(drop=True)


def round_numeric(df: pd.DataFrame, digits: int = 6) -> pd.DataFrame:
    out = df.copy()
    cols = out.select_dtypes(include=[np.number]).columns
    out[cols] = out[cols].round(digits)
    return out


def plot_monthly_tile_parameter_maps(
    summary: pd.DataFrame,
    summary_dir: Path,
    monthly_dir: Path,
    tag: str,
) -> list[Path]:
    specs = [
        ("sigmasq_mean", "sigmasq", "sigmasq"),
        ("sigma_mean", "sigma", "sigma"),
        ("range_lat_mean", "range_lat", "range_lat"),
        ("range_lon_mean", "range_lon", "range_lon"),
        ("smooth_mean", "nu / smooth", "nu"),
        ("nugget_mean", "nugget", "nugget"),
        ("phi1_mean", "phi1", "phi1"),
        ("phi2_mean", "phi2", "phi2"),
        ("phi3_mean", "phi3", "phi3"),
    ]
    available = [spec for spec in specs if spec[0] in summary.columns]
    if not available or summary.empty:
        return []

    try:
        os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: matplotlib unavailable; skipping monthly tile plots: {exc}")
        return []

    n_tile_y = int(summary["tile_y"].max()) + 1
    n_tile_x = int(summary["tile_x"].max()) + 1
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#eeeeee")

    def grid_for(col: str) -> np.ndarray:
        grid = np.full((n_tile_y, n_tile_x), np.nan, dtype=float)
        for row in summary.itertuples(index=False):
            value = getattr(row, col)
            if pd.notna(value):
                grid[int(row.tile_y), int(row.tile_x)] = float(value)
        return grid

    def draw_one(ax, col: str, title: str, cbar_label: str) -> None:
        grid = grid_for(col)
        im = ax.imshow(np.ma.masked_invalid(grid), origin="lower", aspect="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel("tile_x")
        ax.set_ylabel("tile_y")
        ax.set_xticks(np.arange(n_tile_x))
        ax.set_yticks(np.arange(n_tile_y))
        for iy in range(n_tile_y):
            for ix in range(n_tile_x):
                val = grid[iy, ix]
                if np.isfinite(val):
                    ax.text(ix, iy, f"{val:.3g}", ha="center", va="center", color="white", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.82, label=cbar_label)

    def save_to_both(fig, filename: str) -> list[Path]:
        paths = [summary_dir / filename]
        if monthly_dir.resolve() != summary_dir.resolve():
            paths.append(monthly_dir / filename)
        for path in paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=180)
        return paths

    out_paths: list[Path] = []

    n_cols = min(3, len(available))
    n_rows = int(math.ceil(len(available) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.5 * n_cols, 3.6 * n_rows),
        squeeze=False,
        constrained_layout=True,
    )
    for ax, (col, title, cbar_label) in zip(axes.ravel(), available):
        draw_one(ax, col, f"Monthly mean {title}", cbar_label)
    for ax in axes.ravel()[len(available):]:
        ax.axis("off")
    fig.suptitle(f"{tag}: monthly tile parameter means")
    out_paths.extend(save_to_both(fig, f"{tag}_tile_monthly_parameter_maps.png"))
    plt.close(fig)

    legacy = [spec for spec in available if spec[0] in {"nugget_mean", "smooth_mean"}]
    if legacy:
        fig, axes = plt.subplots(
            1,
            len(legacy),
            figsize=(4.8 * len(legacy), 3.8),
            squeeze=False,
            constrained_layout=True,
        )
        for ax, (col, title, cbar_label) in zip(axes.ravel(), legacy):
            draw_one(ax, col, f"Monthly mean {title}", cbar_label)
        fig.suptitle(tag)
        out_paths.extend(save_to_both(fig, f"{tag}_tile_monthly_nugget_nu_maps.png"))
        plt.close(fig)

    for col, title, cbar_label in available:
        fig, ax = plt.subplots(figsize=(4.8, 3.8), constrained_layout=True)
        draw_one(ax, col, f"Monthly mean {title}", cbar_label)
        fig.suptitle(tag)
        safe_label = cbar_label.replace("/", "_").replace(" ", "_")
        out_paths.extend(save_to_both(fig, f"{tag}_tile_monthly_{safe_label}_map.png"))
        plt.close(fig)

    return out_paths


def fit_one_hour(args: argparse.Namespace) -> None:
    manifest = read_manifest(args)
    hour_index, hour, row = choose_hour(args, manifest)
    mode_dir = output_root(args)
    hourly_dir = mode_dir / "hourly"
    hourly_dir.mkdir(parents=True, exist_ok=True)
    stem = f"h{hour_index:03d}_{hour.strftime('%Y%m%dT%H%MZ')}_{METHOD}_{args.nugget_mode}"
    out_csv = hourly_dir / f"{stem}_tiles.csv"
    out_json = hourly_dir / f"{stem}_meta.json"
    if out_csv.exists() and out_json.exists() and not args.overwrite:
        print(f"Skipping existing {out_csv}")
        return

    df_hour = read_hour_table(args, row)
    coords, values, _, meta = prepare_hour_data(df_hour, args)
    tile_id, tile_meta = assign_tiles(coords, int(args.tile_y), int(args.tile_x))
    tile_df = fit_tiles_for_hour(coords, values, tile_id, tile_meta, args)
    tile_df.insert(0, "hour_index", hour_index)
    tile_df.insert(1, "day_index", int(row["day_index"]))
    tile_df.insert(2, "hour_slot", int(row["hour_slot"]))
    tile_df.insert(3, "hour_utc", int(row["hour_utc"]))
    tile_df.insert(4, "hour", hour.strftime("%Y-%m-%dT%H:%M:%SZ"))
    tile_df.insert(5, "month", args.month)
    tile_df.insert(6, "method", METHOD)
    tile_df.insert(7, "nugget_mode", str(args.nugget_mode))
    round_numeric(tile_df, 6).to_csv(out_csv, index=False, float_format="%.6f")
    with out_json.open("w") as f:
        json.dump({
            "hour_index": hour_index,
            "hour": hour.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "method": METHOD,
            "nugget_mode": args.nugget_mode,
            "tile_shape": [int(args.tile_y), int(args.tile_x)],
            "tile_meta": tile_meta,
            "data_meta": meta,
            "args": vars(args),
        }, f, indent=2)
    print(f"Wrote {out_csv}")


def read_hourly_outputs(args: argparse.Namespace) -> pd.DataFrame:
    hourly_dir = output_root(args) / "hourly"
    files = sorted(hourly_dir.glob("*_tiles.csv"))
    if not files:
        raise SystemExit(f"No hourly tile files under {hourly_dir}")
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


def summarize(args: argparse.Namespace) -> None:
    df = read_hourly_outputs(args)
    mode_dir = output_root(args)
    summary_dir = mode_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    monthly_dir = monthly_output_dir(args)
    monthly_dir.mkdir(parents=True, exist_ok=True)
    ym = str(args.month).replace("-", "")
    tag = f"{ym}_{METHOD}_{args.nugget_mode}"

    hourly_path = summary_dir / f"{tag}_hourly_tile_fits.csv"
    round_numeric(df, 6).to_csv(hourly_path, index=False, float_format="%.6f")

    d = df[df["success"].astype(str).str.lower().isin(["true", "1"])].copy()
    keys = ["method", "nugget_mode", "tile_y", "tile_x", "tile_id"]
    value_cols = [
        "sigmasq", "sigma", "range_lat", "range_lon", "smooth", "nugget",
        "phi1", "phi2", "phi3", "loss", "nll", "n", "n_fit",
        "n_initial_fit", "n_qc_removed", "qc_max_abs_whitened",
    ]
    agg = {f"{c}_mean": (c, "mean") for c in value_cols if c in d.columns}
    agg.update({f"{c}_median": (c, "median") for c in ["sigmasq", "range_lat", "range_lon", "smooth", "nugget"] if c in d.columns})
    agg.update({f"{c}_sd": (c, "std") for c in ["sigmasq", "range_lat", "range_lon", "smooth", "nugget"] if c in d.columns})
    summary = d.groupby(keys, as_index=False).agg(
        n_hours=("hour_index", "nunique"),
        tile_center_lat=("tile_center_lat", "mean"),
        tile_center_lon=("tile_center_lon", "mean"),
        tile_lat_min=("tile_lat_min", "mean"),
        tile_lat_max=("tile_lat_max", "mean"),
        tile_lon_min=("tile_lon_min", "mean"),
        tile_lon_max=("tile_lon_max", "mean"),
        **agg,
    )
    summary_path = summary_dir / f"{tag}_tile_monthly_summary.csv"
    monthly_path = monthly_dir / f"{tag}_tile_monthly_summary.csv"
    round_numeric(summary, 6).to_csv(summary_path, index=False, float_format="%.6f")
    round_numeric(summary, 6).to_csv(monthly_path, index=False, float_format="%.6f")
    plot_paths = plot_monthly_tile_parameter_maps(summary, summary_dir, monthly_dir, tag)
    print(f"Wrote {hourly_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {monthly_path}")
    for path in plot_paths:
        print(f"Wrote {path}")


def run_all(args: argparse.Namespace) -> None:
    make_manifest(args)
    manifest = read_manifest(args)
    for idx in range(len(manifest)):
        args.array_index = idx
        fit_one_hour(args)
    summarize(args)


def main() -> None:
    args = parse_args()
    if int(args.tile_y) != 2 or int(args.tile_x) != 4:
        print(f"WARNING: expected 2x4 tiles, got {args.tile_y}x{args.tile_x}")
    if args.mode == "manifest":
        make_manifest(args)
    elif args.mode == "fit":
        fit_one_hour(args)
    elif args.mode == "summarize":
        summarize(args)
    elif args.mode == "all":
        run_all(args)


if __name__ == "__main__":
    main()
