#!/usr/bin/env python3
"""Fit July 2024 hourly spatial Matern nugget profiles.

Workflow:
  1. manifest: scan July 2024 data and write one row per observed hour.
  2. fit:      fit one hourly global spatial model and 3x3 tile nuggets.
  3. summarize: aggregate hourly outputs and write diagnostic plots.

The global model estimates sigmasq, range (isotropic), and a global nugget
for a fixed Matern smoothness using GPU-batched Vecchia (HybridSpaceVecchiaFit,
spatial-only t-conditioning, no t-1/t-2).  Tile nuggets are then profiled with
global sigmasq and range held fixed, using equal-width 3x3 spatial tiles.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree

# Locate GEMS_TCO package (local dev tree or Amarel via PYTHONPATH)
try:
    from GEMS_TCO.kernels_space_050726 import HybridSpaceVecchiaFit as _HybridBase
except ImportError:
    _candidates = [
        Path(__file__).parents[5] / "src",
        Path("/home/jl2815/tco"),
    ]
    for _p in _candidates:
        if (_p / "GEMS_TCO").is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
            break
    from GEMS_TCO.kernels_space_050726 import HybridSpaceVecchiaFit as _HybridBase


TWO_PI_LOG = float(np.log(2.0 * np.pi))


# ---------------------------------------------------------------------------
# GPU-batched spatial Vecchia (t-only conditioning, simplified trend)
# ---------------------------------------------------------------------------

class EDASpaceVecchiaFit(_HybridBase):
    """Isotropic spatial Vecchia: 3 params (sigmasq, range, nugget), intercept+lat GLS mean."""

    def __init__(self, smooth, input_map, nns_map, limit_A=10, target_chunk_size=4096):
        if smooth not in (0.5, 1.5):
            raise ValueError(f"smooth must be 0.5 or 1.5, got {smooth}")
        super().__init__(
            smooth=smooth, input_map=input_map, nns_map=nns_map,
            limit_A=limit_A, target_chunk_size=target_chunk_size,
        )
        self.n_features = 2  # intercept, lat

    def _raw_params(self, params: torch.Tensor):
        # params: [log_sigmasq, log_range, log_nugget]  (isotropic: range_lat = range_lon)
        sigmasq = torch.exp(params[0])
        range_  = torch.exp(params[1])
        nugget  = torch.exp(params[2])
        return sigmasq, range_, range_, nugget

    def _convert_params(self, raw):
        # raw = [log_sigmasq, log_range, log_nugget, final_loss]
        return {
            "sigmasq": float(np.exp(raw[0])),
            "range":   float(np.exp(raw[1])),
            "nugget":  float(np.exp(raw[2])),
        }

    def _design_from_rows(self, rows: torch.Tensor) -> torch.Tensor:
        orig_shape = rows.shape[:-1]
        flat = rows.reshape(-1, rows.shape[-1])
        ones = torch.ones((flat.shape[0], 1), device=self.device, dtype=torch.float64)
        lat = (flat[:, 0:1] - self.lat_mean_val).to(torch.float64)
        X = torch.cat([ones, lat], dim=1)
        return X.reshape(*orig_shape, self.n_features)


def build_nns_map(coords: np.ndarray, m: int) -> list:
    """KD-tree nearest-neighbor map: nns_map[i] = neighbor indices sorted by dist."""
    n = len(coords)
    m = min(max(m, 0), n - 1)
    if m == 0:
        return [np.empty(0, dtype=np.int64) for _ in range(n)]
    tree = cKDTree(coords)
    _, idxs = tree.query(coords, k=min(m + 1, n))
    nns = []
    for i in range(n):
        row = np.atleast_1d(idxs[i])
        row = row[row != i][:m]
        nns.append(row.astype(np.int64))
    return nns


def get_device(args: argparse.Namespace) -> torch.device:
    req = getattr(args, "device", "auto")
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(req)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    success: bool
    nll: float
    sigmasq: float
    sigma: float
    range: float      # isotropic range (degree units when --coords raw)
    nugget: float
    message: str
    n_eval: int


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="July 2024 hourly spatial Matern global/tile nugget fits (GPU batched Vecchia)."
    )
    parser.add_argument("--mode", choices=["manifest", "fit", "summarize", "all"], required=True)
    parser.add_argument("--input", default=os.environ.get("DATA_PATH"))
    parser.add_argument("--input-glob", default=os.environ.get("INPUT_GLOB", "*.parquet"))
    parser.add_argument("--output-dir", default=os.environ.get("OUTDIR", "results/july2024_nugget_3x3"))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--month", default=os.environ.get("MONTH", "2024-07"))
    parser.add_argument("--expected-hours", type=int, default=int(os.environ.get("EXPECTED_HOURS", "248")))

    parser.add_argument("--time-col", default=os.environ.get("TIME_COL", "auto"))
    parser.add_argument("--x-col", default=os.environ.get("X_COL", "auto"))
    parser.add_argument("--y-col", default=os.environ.get("Y_COL", "auto"))
    parser.add_argument("--value-col", default=os.environ.get("VALUE_COL", "auto"))
    parser.add_argument("--qa-col", default=os.environ.get("QA_COL", ""))
    parser.add_argument("--qa-min", type=float, default=None)

    parser.add_argument("--coords", choices=["lonlat", "raw"], default=os.environ.get("COORDS", "lonlat"))
    parser.add_argument("--smooth", type=float, default=float(os.environ.get("SMOOTH", "0.5")))
    parser.add_argument("--neighbors", type=int, default=int(os.environ.get("NEIGHBORS", "8")))
    parser.add_argument("--max-points", type=int, default=int(os.environ.get("MAX_POINTS", "0")))
    parser.add_argument("--min-tile-points", type=int, default=int(os.environ.get("MIN_TILE_POINTS", "80")))
    parser.add_argument("--tiles", type=int, default=int(os.environ.get("TILES", "3")))
    parser.add_argument("--n-restarts", type=int, default=int(os.environ.get("N_RESTARTS", "5")))
    parser.add_argument("--sample-seed", type=int, default=int(os.environ.get("SAMPLE_SEED", "202407")))
    parser.add_argument("--device", default=os.environ.get("DEVICE", "auto"),
                        help="'auto', 'cuda', 'cpu', or 'cuda:N'")
    parser.add_argument("--array-index", type=int, default=None)
    parser.add_argument("--hour", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-every", type=int, default=int(os.environ.get("SUMMARY_EVERY", "8")))

    parser.add_argument("--range-min", type=float, default=None)
    parser.add_argument("--range-max", type=float, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Manifest / IO helpers (unchanged)
# ---------------------------------------------------------------------------

def default_manifest_path(output_dir: str | Path) -> Path:
    return Path(output_dir) / "manifest_hours.csv"


def resolve_manifest(args: argparse.Namespace) -> Path:
    return Path(args.manifest) if args.manifest else default_manifest_path(args.output_dir)


def require_input(args: argparse.Namespace) -> Path:
    if not args.input:
        raise SystemExit("Missing --input or DATA_PATH.")
    path = Path(args.input).expanduser()
    if not path.exists():
        raise SystemExit(f"Input path does not exist: {path}")
    return path


def iter_input_files(path: Path, input_glob: str) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(p for p in path.glob(input_glob) if p.is_file())
    if not files:
        raise SystemExit(f"No files matched {path}/{input_glob}")
    return files


def read_one_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".feather", ".ft"}:
        return pd.read_feather(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported input file type: {path}")


def parse_gems_hour_key(key: str) -> pd.Timestamp | None:
    match = re.match(r"^y(?P<yy>\d{2})m(?P<mm>\d{2})day(?P<dd>\d{2})_hm(?P<hh>\d{2}):(?P<minute>\d{2})$", str(key))
    if not match:
        return None
    parts = {name: int(value) for name, value in match.groupdict().items()}
    year = 2000 + parts["yy"]
    return pd.Timestamp(
        year=year, month=parts["mm"], day=parts["dd"],
        hour=parts["hh"], minute=parts["minute"], tz="UTC",
    )


def dict_pickle_to_table(obj: dict, source_file: Path) -> pd.DataFrame:
    frames = []
    for key, value in obj.items():
        if not isinstance(value, pd.DataFrame):
            continue
        hour_stamp = parse_gems_hour_key(str(key))
        if hour_stamp is None:
            raise ValueError(f"Could not parse GEMS pickle hour key: {key}")
        frame = value.copy()
        frame["hour_key"] = str(key)
        frame["hour"] = hour_stamp.isoformat().replace("+00:00", "Z")
        frame["_source_file"] = str(source_file)
        frames.append(frame)
    if not frames:
        raise ValueError(f"Pickle dict did not contain any DataFrame values: {source_file}")
    return pd.concat(frames, ignore_index=True, copy=False)


def read_input_table(args: argparse.Namespace) -> pd.DataFrame:
    path = require_input(args)
    frames = []
    for file_path in iter_input_files(path, args.input_glob):
        df = read_one_file(file_path)
        if isinstance(df, dict):
            df = dict_pickle_to_table(df, file_path)
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Expected DataFrame or dict of DataFrames from {file_path}, got {type(df)}")
        df["_source_file"] = str(file_path)
        frames.append(df)
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True, copy=False)


def single_pickle_file(args: argparse.Namespace) -> Path | None:
    path = require_input(args)
    files = iter_input_files(path, args.input_glob)
    if len(files) == 1 and files[0].suffix.lower() in {".pkl", ".pickle"}:
        return files[0]
    return None


def make_manifest_from_pickle_dict(args: argparse.Namespace) -> pd.DataFrame | None:
    file_path = single_pickle_file(args)
    if file_path is None:
        return None
    obj = pd.read_pickle(file_path)
    if not isinstance(obj, dict):
        return None

    start, end = month_bounds(args.month)
    rows = []
    for key, value in obj.items():
        if not isinstance(value, pd.DataFrame):
            continue
        hour_exact = parse_gems_hour_key(str(key))
        if hour_exact is None:
            continue
        hour_floor = hour_exact.floor("h")
        if hour_floor < start or hour_floor >= end:
            continue
        frame = apply_quality_filter(value, args)
        rows.append({
            "hour_key": str(key),
            "hour_exact": hour_exact.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "hour": hour_floor.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_rows": int(len(frame)),
            "source_file": str(file_path),
        })
    if not rows:
        raise SystemExit(f"No DataFrame hour keys found for month {args.month} in {file_path}")
    counts = pd.DataFrame(rows).sort_values("hour").reset_index(drop=True)
    counts.insert(0, "hour_index", np.arange(len(counts), dtype=int))
    hour_dt = pd.to_datetime(counts["hour"], utc=True)
    counts["day_index"] = hour_dt.dt.day.astype(int)
    counts["hour_utc"] = hour_dt.dt.hour.astype(int)
    counts["hour_slot"] = counts.groupby("day_index", sort=True).cumcount().astype(int)
    return counts[[
        "hour_index", "day_index", "hour_slot", "hour_utc",
        "hour", "hour_exact", "hour_key", "n_rows", "source_file",
    ]]


def choose_col(df: pd.DataFrame, requested: str, candidates: Iterable[str], role: str) -> str:
    if requested and requested != "auto":
        if requested not in df.columns:
            raise SystemExit(f"{role} column '{requested}' not found. Columns: {list(df.columns)}")
        return requested
    lowered = {str(c).lower(): c for c in df.columns}
    for name in candidates:
        if name.lower() in lowered:
            return lowered[name.lower()]
    raise SystemExit(
        f"Could not auto-detect {role} column. Pass --{role}-col explicitly. "
        f"Columns: {list(df.columns)}"
    )


def column_names(df: pd.DataFrame, args: argparse.Namespace) -> tuple[str, str, str, str]:
    time_col = choose_col(df, args.time_col,
                          ["time", "datetime", "timestamp", "date_time", "utc_time", "hour"], "time")
    x_col = choose_col(df, args.x_col, ["source_longitude", "lon", "longitude", "x", "x_km"], "x")
    y_col = choose_col(df, args.y_col, ["source_latitude", "lat", "latitude", "y", "y_km"], "y")
    value_col = choose_col(
        df, args.value_col,
        ["ColumnAmountO3", "value", "y", "residual", "tco", "column", "column_amount", "vcd", "no2"],
        "value",
    )
    return time_col, x_col, y_col, value_col


def month_bounds(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    end = start + pd.DateOffset(months=1)
    return start, end


def add_hour_column(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = df.copy()
    out["_time_utc"] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    out["_hour_utc"] = out["_time_utc"].dt.floor("h")
    return out


def filter_month(df: pd.DataFrame, args: argparse.Namespace, time_col: str) -> pd.DataFrame:
    out = add_hour_column(df, time_col)
    start, end = month_bounds(args.month)
    out = out[(out["_hour_utc"] >= start) & (out["_hour_utc"] < end)]
    return out


def apply_quality_filter(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.qa_col and args.qa_min is not None:
        if args.qa_col not in df.columns:
            raise SystemExit(f"QA column '{args.qa_col}' not found.")
        return df[df[args.qa_col] >= args.qa_min]
    return df


def make_manifest(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = make_manifest_from_pickle_dict(args)
    if counts is None:
        df = read_input_table(args)
        time_col, _, _, _ = column_names(df, args)
        df = filter_month(df, args, time_col)
        df = apply_quality_filter(df, args)
        if df.empty:
            raise SystemExit(f"No rows found for month {args.month}.")
        counts = df.groupby("_hour_utc", sort=True).size().reset_index(name="n_rows")
        counts.insert(0, "hour_index", np.arange(len(counts), dtype=int))
        counts["hour"] = counts["_hour_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        counts["day_index"] = counts["_hour_utc"].dt.day.astype(int)
        counts["hour_utc"] = counts["_hour_utc"].dt.hour.astype(int)
        counts["hour_slot"] = counts.groupby("day_index", sort=True).cumcount().astype(int)
        counts = counts[["hour_index", "day_index", "hour_slot", "hour_utc", "hour", "n_rows"]]
    manifest_path = resolve_manifest(args)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(manifest_path, index=False)

    print(f"Wrote manifest: {manifest_path}")
    print(f"Observed hours: {len(counts)}")
    if args.expected_hours and len(counts) != args.expected_hours:
        print(
            f"WARNING: expected {args.expected_hours} hours, found {len(counts)}. "
            "Use the manifest length for the array range."
        )


def lonlat_to_km(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    lon0 = float(np.nanmedian(lon))
    lat0 = float(np.nanmedian(lat))
    x = (lon - lon0) * 111.320 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 110.574
    meta = {"lon0": lon0, "lat0": lat0, "coord_units": "km_equirectangular"}
    return x, y, meta


def prepare_hour_data(df: pd.DataFrame, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    _, x_col, y_col, value_col = column_names(df, args)
    cols = [x_col, y_col, value_col]
    d = df[cols].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna()
    if d.empty:
        raise ValueError("No finite rows after coordinate/value filtering.")

    d = d.groupby([x_col, y_col], as_index=False)[value_col].mean()

    raw_x = d[x_col].to_numpy(dtype=float)
    raw_y = d[y_col].to_numpy(dtype=float)
    if args.coords == "lonlat":
        x, y, coord_meta = lonlat_to_km(raw_x, raw_y)
    else:
        x, y = raw_x, raw_y
        coord_meta = {"coord_units": "raw"}
    coords = np.column_stack([y, x])   # col0=y_km (lat dir), col1=x_km (lon dir)
    values = d[value_col].to_numpy(dtype=float)

    if args.max_points and len(values) > args.max_points:
        rng = np.random.default_rng(args.sample_seed)
        keep = np.sort(rng.choice(len(values), size=args.max_points, replace=False))
        coords = coords[keep]
        values = values[keep]
        d = d.iloc[keep].reset_index(drop=True)
        coord_meta["subsampled_to"] = int(args.max_points)
    else:
        d = d.reset_index(drop=True)
    coord_meta["raw_x_col"] = x_col
    coord_meta["raw_y_col"] = y_col
    coord_meta["value_col"] = value_col
    return coords, values, d, coord_meta


# ---------------------------------------------------------------------------
# GPU fitting core
# ---------------------------------------------------------------------------

def _build_gpu_model(
    y: np.ndarray,
    coords: np.ndarray,
    smooth: float,
    neighbors: int,
    device: torch.device,
    chunk: int = 4096,
) -> EDASpaceVecchiaFit:
    n = len(y)
    data = np.zeros((n, 4), dtype=np.float64)
    data[:, 0] = coords[:, 0]   # y_km
    data[:, 1] = coords[:, 1]   # x_km
    data[:, 2] = y
    tensor = torch.from_numpy(data).to(device=device, dtype=torch.float64)
    nns_map = build_nns_map(coords, min(neighbors, n - 1))
    model = EDASpaceVecchiaFit(
        smooth=smooth, input_map={"t0": tensor}, nns_map=nns_map,
        limit_A=min(neighbors, n - 1), target_chunk_size=chunk,
    )
    model.precompute_conditioning_sets()
    return model


def fit_global(
    y: np.ndarray,
    coords: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> FitResult:
    if len(y) < 5:
        raise ValueError("Need at least 5 observations for global fit.")
    smooth = float(args.smooth)
    if smooth not in (0.5, 1.5):
        raise ValueError(f"GPU Vecchia requires smooth 0.5 or 1.5, got {smooth}")

    y_mean, y_std = float(np.nanmean(y)), float(np.nanstd(y))
    print(f"[fit_global] n={len(y)} | y mean={y_mean:.2f} std={y_std:.2f} min={float(np.nanmin(y)):.2f} max={float(np.nanmax(y)):.2f}", flush=True)
    if y_std > 500:
        print(f"[fit_global] WARNING: y_std={y_std:.1f} is unusually large — check data units/columns", flush=True)

    model = _build_gpu_model(y, coords, smooth, args.neighbors, device)

    # Fixed init informed by pure-space EDA (sigmasq~10 DU^2, range~0.2°, nugget~1.0 DU^2)
    params_list = [
        torch.tensor(math.log(10.0),  requires_grad=True, dtype=torch.float64, device=device),  # sigmasq=10
        torch.tensor(math.log(0.2),   requires_grad=True, dtype=torch.float64, device=device),  # range=0.2°
        torch.tensor(math.log(1.0),   requires_grad=True, dtype=torch.float64, device=device),  # nugget=1.0
    ]
    opt = model.set_optimizer(params_list, lr=1.0, max_iter=150, tolerance_grad=1e-5)
    raw, iters = model.fit_vecc_lbfgs(params_list, opt, max_steps=60, grad_tol=1e-5)

    sigmasq = math.exp(raw[0])
    range_  = math.exp(raw[1])
    nugget  = math.exp(raw[2])
    return FitResult(
        success=True,
        nll=raw[-1],
        sigmasq=sigmasq,
        sigma=math.sqrt(sigmasq),
        range=range_,
        nugget=nugget,
        message="lbfgs_gpu",
        n_eval=iters,
    )


def assign_tiles(coords: np.ndarray, tiles: int) -> tuple[np.ndarray, dict]:
    x = coords[:, 1]   # x_km (lon dir) for west-east
    y = coords[:, 0]   # y_km (lat dir) for south-north
    eps_x = max((float(x.max()) - float(x.min())) * 1e-12, 1e-12)
    eps_y = max((float(y.max()) - float(y.min())) * 1e-12, 1e-12)
    x_edges = np.linspace(float(x.min()), float(x.max()) + eps_x, tiles + 1)
    y_edges = np.linspace(float(y.min()), float(y.max()) + eps_y, tiles + 1)
    x_idx = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, tiles - 1)
    y_idx = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, tiles - 1)
    tile_id = y_idx * tiles + x_idx
    meta = {"x_edges": x_edges.tolist(), "y_edges": y_edges.tolist(), "tiles": tiles}
    return tile_id.astype(int), meta


def profile_tile_nuggets(
    y: np.ndarray,
    coords: np.ndarray,
    tile_id: np.ndarray,
    global_fit: FitResult,
    args: argparse.Namespace,
    device: torch.device,
) -> pd.DataFrame:
    rows = []
    var_y = max(float(np.nanvar(y, ddof=1)), 1e-8)
    low  = math.log(var_y * 1e-8)
    high = math.log(var_y * 1e3)
    log_sigmasq = math.log(global_fit.sigmasq)
    log_range   = math.log(global_fit.range)
    smooth = float(args.smooth) if float(args.smooth) in (0.5, 1.5) else 0.5

    for tid in range(args.tiles * args.tiles):
        mask = tile_id == tid
        n_tile = int(mask.sum())
        y_idx = tid // args.tiles
        x_idx = tid % args.tiles
        if n_tile < args.min_tile_points:
            rows.append({
                "tile_id": tid, "tile_x": x_idx, "tile_y": y_idx, "n": n_tile,
                "tile_nugget": np.nan, "tile_nugget_se": np.nan,
                "tile_nll": np.nan, "success": False, "message": "too_few_points",
            })
            continue

        yy = y[mask]
        cc = coords[mask]
        lim = min(args.neighbors, len(yy) - 1)
        tile_model = _build_gpu_model(yy, cc, smooth, lim, device)

        def objective(log_nugget: float) -> float:
            params = torch.tensor(
                [log_sigmasq, log_range, log_nugget],
                dtype=torch.float64, device=device,
            )
            with torch.no_grad():
                return float(tile_model.vecchia_batched_likelihood(params).item())

        res = minimize_scalar(objective, bounds=(low, high), method="bounded",
                              options={"xatol": 1e-4})
        rows.append({
            "tile_id": tid, "tile_x": x_idx, "tile_y": y_idx, "n": n_tile,
            "tile_nugget": float(math.exp(res.x)), "tile_nugget_se": np.nan,
            "tile_nll": float(res.fun), "success": bool(res.success),
            "message": str(res.message),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Manifest / hour helpers (unchanged)
# ---------------------------------------------------------------------------

def read_manifest(args: argparse.Namespace) -> pd.DataFrame:
    manifest_path = resolve_manifest(args)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest does not exist: {manifest_path}. Run --mode manifest first.")
    manifest = pd.read_csv(manifest_path)
    if "hour" not in manifest.columns:
        raise SystemExit(f"Manifest missing 'hour' column: {manifest_path}")
    return manifest


def choose_hour(args: argparse.Namespace, manifest: pd.DataFrame) -> tuple[int, pd.Timestamp, pd.Series]:
    if args.hour:
        hour = pd.to_datetime(args.hour, utc=True)
        matches = manifest.index[pd.to_datetime(manifest["hour"], utc=True) == hour].tolist()
        if not matches:
            raise SystemExit(f"Hour {args.hour} not found in manifest.")
        row = manifest.iloc[int(matches[0])]
        hour_index = int(row["hour_index"]) if "hour_index" in manifest.columns else int(matches[0])
        return hour_index, hour, row
    array_index = args.array_index
    if array_index is None:
        env_idx = os.environ.get("SLURM_ARRAY_TASK_ID")
        array_index = int(env_idx) if env_idx is not None else 0
    if array_index < 0 or array_index >= len(manifest):
        raise SystemExit(f"array index {array_index} outside manifest length {len(manifest)}")
    row = manifest.iloc[array_index]
    hour = pd.to_datetime(row["hour"], utc=True)
    hour_index = int(row["hour_index"]) if "hour_index" in manifest.columns else int(array_index)
    return hour_index, hour, row


def read_selected_hour_table(
    args: argparse.Namespace,
    hour: pd.Timestamp,
    manifest_row: pd.Series,
    pickle_obj: dict | None = None,
) -> pd.DataFrame:
    file_path = single_pickle_file(args)
    if file_path is not None and "hour_key" in manifest_row.index and pd.notna(manifest_row["hour_key"]):
        obj = pickle_obj if pickle_obj is not None else pd.read_pickle(file_path)
        if isinstance(obj, dict):
            hour_key = str(manifest_row["hour_key"])
            if hour_key not in obj:
                raise SystemExit(f"hour_key {hour_key} not found in {file_path}")
            df_hour = obj[hour_key].copy()
            df_hour["hour_key"] = hour_key
            df_hour["hour"] = hour.strftime("%Y-%m-%dT%H:%M:%SZ")
            df_hour["_source_file"] = str(file_path)
            df_hour["_time_utc"] = hour
            df_hour["_hour_utc"] = hour.floor("h")
            return apply_quality_filter(df_hour, args)

    df = read_input_table(args)
    time_col, _, _, _ = column_names(df, args)
    df = filter_month(df, args, time_col)
    df = apply_quality_filter(df, args)
    return df[df["_hour_utc"] == hour].copy()


def manifest_time_labels(hour_index: int, hour: pd.Timestamp, manifest_row: pd.Series) -> tuple[int, int, int]:
    day_index = int(manifest_row["day_index"]) if "day_index" in manifest_row.index else int(hour.day)
    hour_utc  = int(manifest_row["hour_utc"])  if "hour_utc"  in manifest_row.index else int(hour.hour)
    hour_slot = int(manifest_row["hour_slot"]) if "hour_slot" in manifest_row.index else int(hour_index % 8)
    return day_index, hour_slot, hour_utc


def load_single_pickle_dict(args: argparse.Namespace) -> dict | None:
    file_path = single_pickle_file(args)
    if file_path is None:
        return None
    obj = pd.read_pickle(file_path)
    return obj if isinstance(obj, dict) else None


# ---------------------------------------------------------------------------
# Per-hour fit record
# ---------------------------------------------------------------------------

def fit_hour_record(
    args: argparse.Namespace,
    manifest: pd.DataFrame,
    array_index: int,
    device: torch.device,
    pickle_obj: dict | None = None,
) -> tuple[dict, pd.DataFrame]:
    manifest_row = manifest.iloc[array_index]
    hour_index = int(manifest_row["hour_index"]) if "hour_index" in manifest.columns else int(array_index)
    hour = pd.to_datetime(manifest_row["hour"], utc=True)
    day_index, hour_slot, hour_utc = manifest_time_labels(hour_index, hour, manifest_row)
    hour_key = str(manifest_row["hour_key"]) if "hour_key" in manifest_row.index else ""

    df_hour = read_selected_hour_table(args, hour, manifest_row, pickle_obj=pickle_obj)
    if df_hour.empty:
        raise SystemExit(f"No rows found for hour {hour}.")

    coords, values, _, _ = prepare_hour_data(df_hour, args)
    global_fit = fit_global(values, coords, args, device)
    tile_id, _ = assign_tiles(coords, args.tiles)
    tile_df = profile_tile_nuggets(values, coords, tile_id, global_fit, args, device)

    global_row = {
        "day": hour.strftime("%Y-%m-%d"),
        "day_index": day_index,
        "hour_slot": hour_slot,
        "hour_utc": hour_utc,
        "hour_key": hour_key,
        "hour": hour.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_raw": int(len(df_hour)),
        "n_used": int(len(values)),
        "sigmasq": global_fit.sigmasq,
        "sigma": global_fit.sigma,
        "range": global_fit.range,
        "nugget": global_fit.nugget,
        "loss": global_fit.nll,
        "success": bool(global_fit.success),
    }
    tile_df.insert(0, "day", hour.strftime("%Y-%m-%d"))
    tile_df.insert(1, "day_index", day_index)
    tile_df.insert(2, "hour_slot", hour_slot)
    tile_df.insert(3, "hour_utc", hour_utc)
    tile_df.insert(4, "hour_key", hour_key)
    tile_df.insert(5, "hour", hour.strftime("%Y-%m-%dT%H:%M:%SZ"))
    tile_df["global_nugget"] = global_fit.nugget
    return global_row, tile_df


# ---------------------------------------------------------------------------
# Summarization helpers
# ---------------------------------------------------------------------------

def round_numeric(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(digits)
    return out


def csv_line_from_row(row: dict, columns: list[str]) -> str:
    vals = []
    for col in columns:
        val = row.get(col, "")
        if isinstance(val, float):
            vals.append(f"{val:.4f}")
        else:
            vals.append(str(val))
    return ",".join(vals)


def tile_mean_summary(tile_df: pd.DataFrame, tiles: int) -> tuple[pd.DataFrame, np.ndarray]:
    summary = (
        tile_df.groupby(["tile_y", "tile_x", "tile_id"], as_index=False)
        .agg(
            n_hours=("tile_nugget", "count"),
            tile_nugget_mean=("tile_nugget", "mean"),
            tile_nugget_sd=("tile_nugget", "std"),
            tile_nugget_median=("tile_nugget", "median"),
        )
        .sort_values(["tile_y", "tile_x"])
    )
    mat = np.full((tiles, tiles), np.nan)
    for _, row in summary.iterrows():
        mat[int(row["tile_y"]), int(row["tile_x"])] = row["tile_nugget_mean"]
    return summary, mat


def global_nugget_by_hour_slot(global_df: pd.DataFrame) -> pd.DataFrame:
    return (
        global_df.groupby(["hour_slot", "hour_utc"], as_index=False)
        .agg(
            n_days=("day", "nunique"),
            nugget_mean=("nugget", "mean"),
            nugget_sd=("nugget", "std"),
            nugget_median=("nugget", "median"),
            nugget_min=("nugget", "min"),
            nugget_max=("nugget", "max"),
        )
        .sort_values(["hour_slot"])
    )


def global_params_by_hour_slot(global_df: pd.DataFrame) -> pd.DataFrame:
    global_df = ensure_sigma_column(global_df)
    agg = {
        "n_days": ("day", "nunique"),
        "n_hours": ("day", "count"),
        "sigmasq_mean": ("sigmasq", "mean"),
        "sigma_mean": ("sigma", "mean"),
        "range_mean": ("range", "mean"),
        "nugget_mean": ("nugget", "mean"),
        "loss_mean": ("loss", "mean"),
    }
    return (
        global_df.groupby(["hour_slot", "hour_utc"], as_index=False)
        .agg(**agg)
        .sort_values(["hour_slot"])
    )


def matrix_to_text(mat: np.ndarray) -> str:
    lines = []
    for row in mat:
        lines.append("[" + ", ".join("NA" if not np.isfinite(x) else f"{x:.4f}" for x in row) + "]")
    return "\n".join(lines)


def table_to_text(df: pd.DataFrame) -> str:
    return round_numeric(df, 4).to_csv(index=False, float_format="%.4f").strip()


def save_global_nugget_hour_slot_plot(hour_slot_df: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = hour_slot_df.sort_values("hour_slot")
    x = d["hour_slot"].to_numpy(dtype=float)
    y = d["nugget_mean"].to_numpy(dtype=float)
    sd = d["nugget_sd"].fillna(0.0).to_numpy(dtype=float)
    labels = [f"{int(slot)}\n{int(hour):02d}:00" for slot, hour in zip(d["hour_slot"], d["hour_utc"])]

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.errorbar(x, y, yerr=sd, marker="o", lw=1.4, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("hour slot within day / UTC hour")
    ax.set_ylabel("global fitted nugget")
    ax.set_title("Global fitted nugget by hour slot")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def running_summary_text(
    global_df: pd.DataFrame,
    tile_df: pd.DataFrame,
    args: argparse.Namespace,
    label: str,
) -> str:
    _, tile_mat = tile_mean_summary(tile_df, int(args.tiles))
    hour_slot_params = global_params_by_hour_slot(global_df)
    return "\n\n".join([
        label,
        "running avg over fitted hours: 3x3 tile nuggets (rows=south->north, cols=west->east)",
        matrix_to_text(tile_mat),
        "running hourly avg parameters over different days",
        table_to_text(hour_slot_params),
    ])


def write_compact_outputs(global_df: pd.DataFrame, tile_df: pd.DataFrame, args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    month_start = pd.Timestamp(f"{args.month}-01")
    month_label = f"{month_start.strftime('%B').lower()}{month_start.year}"
    global_df = ensure_sigma_column(global_df)

    global_cols = ["day", "hour_slot", "hour_utc", "hour_key", "n_raw", "n_used",
                   "sigmasq", "sigma", "range", "nugget", "loss"]
    global_cols_avail = [c for c in global_cols if c in global_df.columns]
    global_out = round_numeric(global_df[global_cols_avail], 4)
    global_path = output_dir / f"{month_label}_spatial_fit_{len(global_out)}.csv"
    global_out.to_csv(global_path, index=False, float_format="%.4f")

    tile_summary, tile_mat = tile_mean_summary(tile_df, int(args.tiles))
    tile_median_mat = heatmap_matrix(tile_summary, "tile_nugget_median", int(args.tiles))
    tile_summary = round_numeric(tile_summary, 4)
    tile_summary_path = output_dir / "tile_nugget_mean_3x3.csv"
    tile_summary.to_csv(tile_summary_path, index=False, float_format="%.4f")

    hour_slot_nugget = round_numeric(global_nugget_by_hour_slot(global_df), 4)
    hour_slot_params = round_numeric(global_params_by_hour_slot(global_df), 4)
    hour_slot_path = output_dir / "global_params_by_hour_slot.csv"
    hour_slot_params.to_csv(hour_slot_path, index=False, float_format="%.4f")
    global_with_time, global_by_day, global_by_hour_slot = summarize_global_by_time(global_df)
    round_numeric(global_by_day, 4).to_csv(
        output_dir / "global_params_by_day.csv", index=False, float_format="%.4f"
    )

    save_heatmap(
        tile_mat, f"{args.month} mean tile nugget, 3x3", "mean tile nugget",
        output_dir / "tile_nugget_mean_heatmap_3x3.png",
    )
    save_heatmap(
        tile_median_mat, f"{args.month} median tile nugget, 3x3", "median tile nugget",
        output_dir / "tile_nugget_median_heatmap_3x3.png",
    )
    save_global_nugget_hour_slot_plot(hour_slot_nugget, output_dir / "global_nugget_by_hour_slot.png")
    save_global_timeseries(global_df, output_dir / "global_params_timeseries.png")
    save_daily_param_plot(global_by_day, output_dir / "global_params_daily_mean.png")
    save_hour_slot_param_plot(global_by_hour_slot, output_dir / "global_params_hour_slot_mean.png")
    save_day_hour_heatmaps(global_with_time, output_dir)

    summary_text = "\n\n".join([
        "FINAL RUNNING SUMMARY",
        f"main table: {global_path}",
        f"tile summary: {tile_summary_path}",
        f"hour-slot summary: {hour_slot_path}",
        running_summary_text(global_df, tile_df, args, "final running averages"),
    ])
    summary_path = output_dir / "running_summary.txt"
    summary_path.write_text(summary_text + "\n", encoding="utf-8")
    print("\n" + summary_text, flush=True)


def fit_one_hour(args: argparse.Namespace) -> None:
    device = get_device(args)
    output_dir = Path(args.output_dir)
    hourly_dir = output_dir / "hourly"
    hourly_dir.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(args)
    hour_index, hour, manifest_row = choose_hour(args, manifest)
    day_index, hour_slot, hour_utc = manifest_time_labels(hour_index, hour, manifest_row)
    stem = (
        f"obs{hour_index:03d}_day{day_index:02d}_slot{hour_slot:02d}_"
        f"utc{hour_utc:02d}_{hour.strftime('%Y%m%dT%H%MZ')}_nu{args.smooth:g}"
    )
    global_path = hourly_dir / f"{stem}_global.csv"
    tile_path   = hourly_dir / f"{stem}_tiles.csv"
    meta_path   = hourly_dir / f"{stem}_meta.json"
    if global_path.exists() and tile_path.exists() and not args.overwrite:
        print(f"Skipping existing outputs for {hour}: {global_path}")
        return

    df_hour = read_selected_hour_table(args, hour, manifest_row)
    if df_hour.empty:
        raise SystemExit(f"No rows found for hour {hour}.")

    coords, values, _, coord_meta = prepare_hour_data(df_hour, args)
    global_fit = fit_global(values, coords, args, device)
    tile_id, tile_meta = assign_tiles(coords, args.tiles)
    tile_df = profile_tile_nuggets(values, coords, tile_id, global_fit, args, device)

    global_row = {
        "hour_index": hour_index,
        "day_index": day_index,
        "hour_slot": hour_slot,
        "hour_utc": hour_utc,
        "hour": hour.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_raw": int(len(df_hour)),
        "n_fit": int(len(values)),
        "smooth": float(args.smooth),
        "method": "gpu_batched_vecchia",
        "neighbors": int(args.neighbors),
        **asdict(global_fit),
        "resid_var": float(np.nanvar(values, ddof=1)),
    }
    global_df = pd.DataFrame([global_row])
    tile_df.insert(0, "hour_index", hour_index)
    tile_df.insert(1, "day_index", day_index)
    tile_df.insert(2, "hour_slot", hour_slot)
    tile_df.insert(3, "hour_utc", hour_utc)
    tile_df.insert(4, "hour", hour.strftime("%Y-%m-%dT%H:%M:%SZ"))
    tile_df.insert(5, "smooth", float(args.smooth))
    tile_df["global_sigmasq"] = global_fit.sigmasq
    tile_df["global_range"]   = global_fit.range
    tile_df["global_nugget"]  = global_fit.nugget
    tile_df["tile_to_global_nugget"] = tile_df["tile_nugget"] / global_fit.nugget

    global_df.to_csv(global_path, index=False)
    tile_df.to_csv(tile_path, index=False)
    meta = {
        "args": vars(args),
        "coord_meta": coord_meta,
        "tile_meta": tile_meta,
        "hour": hour.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": str(device),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {global_path}")
    print(f"Wrote {tile_path}")


def read_hourly_outputs(output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    hourly_dir = output_dir / "hourly"
    global_files = sorted(hourly_dir.glob("*_global.csv"))
    tile_files   = sorted(hourly_dir.glob("*_tiles.csv"))
    if not global_files or not tile_files:
        raise SystemExit(f"No hourly outputs found under {hourly_dir}")
    global_df = pd.concat([pd.read_csv(p) for p in global_files], ignore_index=True)
    tile_df   = pd.concat([pd.read_csv(p) for p in tile_files],   ignore_index=True)
    return global_df, tile_df


def heatmap_matrix(summary: pd.DataFrame, value_col: str, tiles: int) -> np.ndarray:
    mat = np.full((tiles, tiles), np.nan)
    for _, row in summary.iterrows():
        mat[int(row["tile_y"]), int(row["tile_x"])] = row[value_col]
    return mat


def ensure_sigma_column(global_df: pd.DataFrame) -> pd.DataFrame:
    out = global_df.copy()
    if "sigma" not in out.columns and "sigmasq" in out.columns:
        out["sigma"] = np.sqrt(out["sigmasq"].clip(lower=0.0))
    return out


def save_heatmap(mat: np.ndarray, title: str, cbar_label: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5.4))
    im = ax.imshow(mat, origin="lower", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("tile x: west to east")
    ax.set_ylabel("tile y: south to north")
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_yticks(np.arange(mat.shape[0]))
    for y in range(mat.shape[0]):
        for x in range(mat.shape[1]):
            val = mat[y, x]
            label = "NA" if not np.isfinite(val) else f"{val:.4f}"
            ax.text(x, y, label, ha="center", va="center", color="white", fontsize=9)
    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_global_timeseries(global_df: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = global_df.copy()
    d["hour_dt"] = pd.to_datetime(d["hour"], utc=True)
    d = d.sort_values("hour_dt")
    panels = [
        ("sigma",  "global sigma"),
        ("range",  "global range (deg)"),
        ("nugget", "global nugget"),
    ]
    panels = [(col, lbl) for col, lbl in panels if col in d.columns]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 2.5 * len(panels)), sharex=True)
    if len(panels) == 1:
        axes = [axes]
    for ax, (col, label) in zip(axes, panels):
        ax.plot(d["hour_dt"], d[col], lw=1.2)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("hour")
    fig.suptitle("Hourly global Matern fits")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def add_global_time_features(global_df: pd.DataFrame) -> pd.DataFrame:
    d = ensure_sigma_column(global_df)
    d["hour_dt"] = pd.to_datetime(d["hour"], utc=True)
    d = d.sort_values("hour_dt")
    d["day"] = d["hour_dt"].dt.strftime("%Y-%m-%d")
    d["day_index"] = d["hour_dt"].dt.day.astype(int)
    d["hour_utc"]  = d["hour_dt"].dt.hour.astype(int)
    d["hour_slot"] = d.groupby("day", sort=True).cumcount().astype(int)
    return d


def summarize_global_by_time(global_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = add_global_time_features(global_df)
    param_cols = ["sigmasq", "sigma", "range", "nugget", "loss", "nll", "resid_var"]
    param_cols = [c for c in param_cols if c in d.columns]
    agg_map = {col: ["mean", "median", "std", "min", "max"] for col in param_cols}

    by_day = d.groupby(["day_index", "day"], as_index=False).agg(
        n_hours=("hour", "count"),
        first_hour=("hour", "min"),
        last_hour=("hour", "max"),
        **{f"{col}_{stat}": (col, stat) for col in agg_map for stat in agg_map[col]},
    )
    by_hour_slot = d.groupby(["hour_slot", "hour_utc"], as_index=False).agg(
        n_days=("day", "nunique"),
        n_hours=("hour", "count"),
        **{f"{col}_{stat}": (col, stat) for col in agg_map for stat in agg_map[col]},
    )
    return d, by_day, by_hour_slot


def save_daily_param_plot(by_day: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_specs = [
        ("sigma", "global sigma"),
        ("range", "global range"),
        ("nugget", "global nugget"),
    ]
    plot_specs = [(c, l) for c, l in plot_specs if f"{c}_mean" in by_day.columns]
    fig, axes = plt.subplots(len(plot_specs), 1, figsize=(11, 2.5 * len(plot_specs)), sharex=True)
    if len(plot_specs) == 1:
        axes = [axes]
    x = by_day["day_index"].to_numpy(dtype=float)
    for ax, (col, label) in zip(axes, plot_specs):
        mean = by_day[f"{col}_mean"].to_numpy(dtype=float)
        sd_col = f"{col}_std"
        sd = by_day[sd_col].fillna(0.0).to_numpy(dtype=float) if sd_col in by_day else np.zeros_like(mean)
        ax.plot(x, mean, marker="o", ms=3.5, lw=1.4)
        ax.fill_between(x, mean - sd, mean + sd, alpha=0.18)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("day of month")
    fig.suptitle("Daily mean global spatial parameters, averaged over hourly fits")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_hour_slot_param_plot(by_hour_slot: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_specs = [
        ("sigma", "global sigma"),
        ("range", "global range"),
        ("nugget", "global nugget"),
    ]
    plot_specs = [(c, l) for c, l in plot_specs if f"{c}_mean" in by_hour_slot.columns]
    fig, axes = plt.subplots(len(plot_specs), 1, figsize=(9, 2.5 * len(plot_specs)), sharex=True)
    if len(plot_specs) == 1:
        axes = [axes]
    x = by_hour_slot["hour_slot"].to_numpy(dtype=float)
    labels = [
        f"{int(slot)}\n{int(hour):02d}:00"
        for slot, hour in zip(by_hour_slot["hour_slot"], by_hour_slot["hour_utc"])
    ]
    for ax, (col, label) in zip(axes, plot_specs):
        mean = by_hour_slot[f"{col}_mean"].to_numpy(dtype=float)
        sd_col = f"{col}_std"
        sd = by_hour_slot[sd_col].fillna(0.0).to_numpy(dtype=float) if sd_col in by_hour_slot else np.zeros_like(mean)
        ax.errorbar(x, mean, yerr=sd, marker="o", ms=4, lw=1.4, capsize=3)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels)
    axes[-1].set_xlabel("hour slot within day / UTC hour")
    fig.suptitle("Mean global spatial parameters by hour slot, averaged over 31 days")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_day_hour_heatmaps(global_with_time: pd.DataFrame, summary_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("sigma", "global sigma"),
        ("range", "global range"),
        ("nugget", "global nugget"),
    ]
    for col, label in panels:
        if col not in global_with_time.columns:
            continue
        pivot = global_with_time.pivot_table(
            index="day_index", columns="hour_slot", values=col, aggfunc="mean"
        ).sort_index()
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(pivot.to_numpy(), origin="lower", aspect="auto", cmap="viridis")
        ax.set_title(f"{label}: day x hour slot")
        ax.set_xlabel("hour slot within day")
        ax.set_ylabel("day of month")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels([str(int(c)) for c in pivot.columns])
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([str(int(d)) for d in pivot.index])
        cbar = fig.colorbar(im, ax=ax, shrink=0.86)
        cbar.set_label(label)
        fig.tight_layout()
        fig.savefig(summary_dir / f"global_{col}_day_hour_heatmap.png", dpi=180)
        plt.close(fig)


def save_tile_boxplot(tile_df: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = tile_df.dropna(subset=["tile_nugget"]).copy()
    d["tile_label"] = d["tile_y"].astype(int).astype(str) + "," + d["tile_x"].astype(int).astype(str)
    labels = sorted(d["tile_label"].unique())
    data = [d.loc[d["tile_label"] == label, "tile_nugget"].to_numpy() for label in labels]
    fig, ax = plt.subplots(figsize=(9, 5))
    try:
        ax.boxplot(data, tick_labels=labels, showfliers=False)
    except TypeError:
        ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_xlabel("tile (y,x), south-to-north / west-to-east")
    ax.set_ylabel("tile nugget")
    ax.set_title("Hourly tile nugget distribution")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def summarize(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    global_df, tile_df = read_hourly_outputs(output_dir)

    global_all = summary_dir / "global_results_all.csv"
    tile_all   = summary_dir / "tile_results_all.csv"
    global_df.to_csv(global_all, index=False)
    tile_df.to_csv(tile_all, index=False)

    global_with_time, global_by_day, global_by_hour_slot = summarize_global_by_time(global_df)
    global_with_time.to_csv(summary_dir / "global_results_all_with_time_features.csv", index=False)
    global_by_day_path       = summary_dir / "global_params_by_day.csv"
    global_by_hour_slot_path = summary_dir / "global_params_by_hour_slot.csv"
    global_by_day.to_csv(global_by_day_path, index=False)
    global_by_hour_slot.to_csv(global_by_hour_slot_path, index=False)

    tile_summary = (
        tile_df.groupby(["tile_y", "tile_x", "tile_id"], as_index=False)
        .agg(
            n_hours=("tile_nugget", "count"),
            mean_tile_nugget=("tile_nugget", "mean"),
            median_tile_nugget=("tile_nugget", "median"),
            sd_tile_nugget=("tile_nugget", "std"),
            mean_ratio_to_global=("tile_to_global_nugget", "mean"),
            median_ratio_to_global=("tile_to_global_nugget", "median"),
            mean_tile_n=("n", "mean"),
        )
        .sort_values(["tile_y", "tile_x"])
    )
    tile_summary_path = summary_dir / "tile_nugget_summary_3x3.csv"
    tile_summary.to_csv(tile_summary_path, index=False)

    tiles = int(args.tiles)
    save_heatmap(
        heatmap_matrix(tile_summary, "mean_tile_nugget", tiles),
        "Mean tile nugget, 3x3", "mean tile nugget",
        summary_dir / "tile_nugget_mean_heatmap_3x3.png",
    )
    save_heatmap(
        heatmap_matrix(tile_summary, "median_tile_nugget", tiles),
        "Median tile nugget, 3x3", "median tile nugget",
        summary_dir / "tile_nugget_median_heatmap_3x3.png",
    )
    save_heatmap(
        heatmap_matrix(tile_summary, "mean_ratio_to_global", tiles),
        "Mean tile/global nugget ratio, 3x3", "mean tile/global nugget",
        summary_dir / "tile_to_global_nugget_ratio_mean_heatmap_3x3.png",
    )
    save_global_timeseries(global_df, summary_dir / "global_params_timeseries.png")
    save_daily_param_plot(global_by_day, summary_dir / "global_params_daily_mean.png")
    save_hour_slot_param_plot(global_by_hour_slot, summary_dir / "global_params_hour_slot_mean.png")
    save_day_hour_heatmaps(global_with_time, summary_dir)
    save_tile_boxplot(tile_df, summary_dir / "tile_nugget_boxplot_3x3.png")

    print(f"Wrote {global_all}")
    print(f"Wrote {tile_all}")
    print(f"Wrote {global_by_day_path}")
    print(f"Wrote {global_by_hour_slot_path}")
    print(f"Wrote {tile_summary_path}")
    print(f"Wrote plots under {summary_dir}")


def run_all(args: argparse.Namespace) -> None:
    device = get_device(args)
    make_manifest(args)
    manifest = read_manifest(args)
    pickle_obj = load_single_pickle_dict(args)
    global_rows: list[dict] = []
    tile_frames: list[pd.DataFrame] = []
    running_cols = [
        "day", "hour_slot", "hour_utc", "hour_key",
        "n_raw", "n_used", "sigmasq", "sigma", "range", "nugget", "loss",
    ]
    print("\nFITTED PARAMETERS EACH HOUR", flush=True)
    print(",".join(running_cols), flush=True)
    for idx in range(len(manifest)):
        global_row, tile_df = fit_hour_record(args, manifest, idx, device, pickle_obj=pickle_obj)
        global_rows.append(global_row)
        tile_frames.append(tile_df)
        print(csv_line_from_row(global_row, running_cols), flush=True)

        n_done = idx + 1
        is_last = n_done == len(manifest)
        if args.summary_every > 0 and (n_done % args.summary_every == 0 or is_last):
            global_running = pd.DataFrame(global_rows)
            tile_running   = pd.concat(tile_frames, ignore_index=True)
            print(
                "\n"
                + running_summary_text(
                    global_running, tile_running, args,
                    f"RUNNING SUMMARY after {n_done}/{len(manifest)} fitted hours",
                )
                + "\n",
                flush=True,
            )

    global_df = pd.DataFrame(global_rows)
    tile_df   = pd.concat(tile_frames, ignore_index=True)
    write_compact_outputs(global_df, tile_df, args)


def main() -> None:
    args = parse_args()
    smooth = float(args.smooth)
    if smooth not in (0.5, 1.5):
        raise SystemExit(
            f"ERROR: --smooth {smooth} not supported by GPU Vecchia kernel. "
            "Use 0.5 (Matern-1/2) or 1.5 (Matern-3/2)."
        )
    try:
        if args.mode == "manifest":
            make_manifest(args)
        elif args.mode == "fit":
            fit_one_hour(args)
        elif args.mode == "summarize":
            summarize(args)
        elif args.mode == "all":
            run_all(args)
        else:
            raise ValueError(args.mode)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
