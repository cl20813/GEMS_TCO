#!/usr/bin/env python3
"""Fit hourly July simulation spatial Matern nugget profiles with group Vecchia.

Workflow:
  1. manifest: scan the requested July data and write one row per observed hour.
  2. fit:      fit one hourly global spatial model and tile nuggets.
  3. summarize: aggregate hourly outputs and write diagnostic plots.

The global model estimates sigmasq, range_lat, range_lon, and a global nugget
for a fixed Matern smoothness using GPU-batched cluster/group Vecchia.  The
global spatial scale is fitted with the anisotropic microergodic-style
reparameterization used by the hybrid kernels:
phi2 = 1 / range_lon, phi3 = (range_lon / range_lat)^2, and
sigmasq = phi1 / phi2.  Tile parameters are fitted tile-by-tile with the same
torch/autograd group Vecchia likelihood rather than a dense full-likelihood
profile.

By default this entrypoint runs a two-stage robustification pass: first fit the
hour with group Vecchia, compute Vecchia whitened residuals, remove observations
with |w| > 10, and then refit/profile on the cleaned observations.  The all-mode
default is restricted to the first 10 days and first 8 observed hour slots per
day for the short diagnostic run.

This updated simulation entrypoint is intentionally not pointwise Vecchia.  It
uses 4x4 grid-cell cluster targets with previous max-min cluster blocks as
conditioning sets by default.  It also verifies that the fitted input covers
the expanded region lat [-3, 7], lon [111, 131], and writes geographic tile
heatmaps with latitude/longitude axes plus a GEMS nadir marker at (0, 128).
The default mean design is beta0 + beta1 * centered latitude; this removes the
dominant meridional trend while keeping each hourly pure-space fit separate.
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

# Locate GEMS_TCO package (local dev tree or Amarel via PYTHONPATH)
try:
    from GEMS_TCO.kernels_space_iso_cluster_052426 import ClusterSpaceIsoTrendVecchiaFit as _ClusterBase
except ImportError:
    _candidates = [
        Path(__file__).parents[5] / "src",
        Path("/home/jl2815/tco"),
    ]
    for _p in _candidates:
        if (_p / "GEMS_TCO").is_dir() and str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
            break
    from GEMS_TCO.kernels_space_iso_cluster_052426 import ClusterSpaceIsoTrendVecchiaFit as _ClusterBase


# ---------------------------------------------------------------------------
# GPU-batched cluster/group spatial Vecchia
# ---------------------------------------------------------------------------

class EDAClusterSpaceVecchiaFit(_ClusterBase):
    """Cluster-target anisotropic spatial Vecchia with phi1/phi2/phi3 parameterization.

    Parameters are optimized as log(phi1), log(phi2), log(phi3), log(nugget),
    with sigmasq = phi1 / phi2, range_lon = 1 / phi2, and
    range_lat = 1 / (phi2 * sqrt(phi3)).  This keeps the pure-space fit on the
    same microergodic-style scale as the earlier hybrid kernels, while using
    group/cluster targets instead of pointwise targets.

    For this hourly pure-space diagnostic the default mean is beta0 + latitude.
    The older space-time fits used time dummies and latitude terms, but here
    each slot/hour is fitted separately, so time dummies are deliberately not
    needed.  `--mean-design constant` and `--mean-design latlon` remain
    available for sensitivity checks.
    """

    def __init__(self, *args, mean_design: str = "lat", **kwargs):
        if float(kwargs.get("smooth", args[0] if args else 0.5)) <= 0.0:
            raise ValueError(f"smooth must be positive, got {kwargs.get('smooth', args[0] if args else None)}")
        requested_mean = str(mean_design)
        base_mean = "lat" if requested_mean == "constant" else requested_mean
        super().__init__(*args, mean_design=base_mean, **kwargs)
        self.eda_mean_design = requested_mean
        if requested_mean == "constant":
            self.mean_design = "constant"
            self.n_features = 1

    def _raw_params(self, params: torch.Tensor):
        phi1 = torch.exp(params[0])
        phi2 = torch.exp(params[1])
        phi3 = torch.exp(params[2])
        sigmasq = phi1 / phi2
        range_lon = 1.0 / phi2
        range_lat = 1.0 / (phi2 * torch.sqrt(phi3).clamp_min(1e-12))
        nugget = torch.exp(params[3])
        return sigmasq, range_lat, range_lon, nugget

    def _cov_from_deltas(self, d_lat, d_lon, params: torch.Tensor):
        sigmasq, range_lat, range_lon, _ = self._raw_params(params)
        dist = torch.sqrt(
            d_lat.new_tensor(1e-8)
            + (d_lat / range_lat.clamp_min(1e-12)).pow(2)
            + (d_lon / range_lon.clamp_min(1e-12)).pow(2)
        )
        return sigmasq * self._matern_corr(dist)

    def _convert_params(self, raw):
        phi1 = float(np.exp(raw[0]))
        phi2 = float(np.exp(raw[1]))
        phi3 = float(np.exp(raw[2]))
        sigmasq = phi1 / phi2
        range_lon = 1.0 / phi2
        range_lat = 1.0 / (phi2 * np.sqrt(phi3))
        return {
            "sigmasq": sigmasq,
            "range": float(np.sqrt(range_lat * range_lon)),
            "range_lat": range_lat,
            "range_lon": range_lon,
            "nugget": float(np.exp(raw[3])),
            "phi1": phi1,
            "phi2": phi2,
            "phi3": phi3,
        }

    def _design_from_rows(self, rows: torch.Tensor) -> torch.Tensor:
        if getattr(self, "eda_mean_design", "") != "constant":
            return super()._design_from_rows(rows)
        orig_shape = rows.shape[:-1]
        flat = rows.reshape(-1, rows.shape[-1])
        ones = torch.ones((flat.shape[0], 1), device=self.device, dtype=torch.float64)
        return ones.reshape(*orig_shape, 1)


def get_device(args: argparse.Namespace) -> torch.device:
    req = getattr(args, "device", "auto")
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(req)


def parse_float_pair(text: str) -> list[float]:
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(vals) != 2:
        raise argparse.ArgumentTypeError("range must look like -3,7")
    return [min(vals), max(vals)]


def parse_block_shape(text: str) -> tuple[int, int]:
    vals = [int(x.strip()) for x in str(text).lower().replace("x", ",").split(",") if x.strip()]
    if len(vals) != 2:
        raise argparse.ArgumentTypeError("block shape must look like 4x4")
    if vals[0] <= 0 or vals[1] <= 0:
        raise argparse.ArgumentTypeError("block dimensions must be positive")
    return vals[0], vals[1]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class FitResult:
    success: bool
    nll: float
    sigmasq: float
    sigma: float
    range: float      # geometric mean range, retained for backward-compatible summaries
    range_lat: float
    range_lon: float
    nugget: float
    phi1: float
    phi2: float
    phi3: float
    message: str
    n_eval: int


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hourly July spatial Matern global/tile nugget fits (GPU batched Vecchia)."
    )
    parser.add_argument("--mode", choices=["manifest", "fit", "summarize", "all"], required=True)
    parser.add_argument("--input", default=os.environ.get("DATA_PATH"))
    parser.add_argument("--input-glob", default=os.environ.get("INPUT_GLOB", "*.parquet"))
    parser.add_argument("--output-dir", default=os.environ.get("OUTDIR", "results/july_nugget_4x8_expanded"))
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--month", default=os.environ.get("MONTH", "2024-07"))
    parser.add_argument("--expected-hours", type=int, default=int(os.environ.get("EXPECTED_HOURS", "248")))
    parser.add_argument("--max-days", type=int, default=int(os.environ.get("MAX_DAYS", "10")),
                        help="For --mode all, fit at most this many distinct days. 0 means all days.")
    parser.add_argument("--max-hour-slots-per-day", type=int, default=int(os.environ.get("MAX_HOUR_SLOTS_PER_DAY", "8")),
                        help="For --mode all, fit at most this many observed hour slots per day. 0 means all slots.")

    parser.add_argument("--time-col", default=os.environ.get("TIME_COL", "auto"))
    parser.add_argument("--x-col", default=os.environ.get("X_COL", "auto"))
    parser.add_argument("--y-col", default=os.environ.get("Y_COL", "auto"))
    parser.add_argument("--value-col", default=os.environ.get("VALUE_COL", "auto"))
    parser.add_argument("--qa-col", default=os.environ.get("QA_COL", ""))
    parser.add_argument("--qa-min", type=float, default=None)

    parser.add_argument("--coords", choices=["lonlat", "raw"], default=os.environ.get("COORDS", "lonlat"))
    parser.add_argument("--smooth", type=float, default=float(os.environ.get("SMOOTH", "0.5")))
    parser.add_argument("--neighbors", type=int, default=int(os.environ.get("NEIGHBORS", "8")),
                        help="Legacy pointwise option; retained for CLI compatibility but not used by group Vecchia.")
    parser.add_argument("--cluster-neighbor-blocks", type=int, default=int(os.environ.get("CLUSTER_NEIGHBOR_BLOCKS", "2")),
                        help="Number of previous max-min cluster blocks used for conditioning.")
    parser.add_argument("--cluster-block-shape", type=parse_block_shape, default=parse_block_shape(os.environ.get("CLUSTER_BLOCK_SHAPE", "4x4")),
                        help="Cluster target block shape in grid-cell units, e.g. 4x4.")
    parser.add_argument("--target-chunk-size", type=int, default=int(os.environ.get("TARGET_CHUNK_SIZE", "96")),
                        help="Number of cluster conditionals per GPU batch.")
    parser.add_argument("--min-target-points", type=int, default=int(os.environ.get("MIN_TARGET_POINTS", "1")),
                        help="Minimum observed points required for a target cluster to enter the likelihood.")
    parser.add_argument("--qc-whitened-threshold", type=float, default=float(os.environ.get("QC_WHITENED_THRESHOLD", "10.0")),
                        help="Two-stage QC threshold. Remove observations with |Vecchia whitened residual| above this value before the final fit. <=0 disables QC.")
    parser.add_argument("--qc-min-fit-points", type=int, default=int(os.environ.get("QC_MIN_FIT_POINTS", "80")),
                        help="Skip QC removal if fewer than this many observations would remain.")
    parser.add_argument("--mean-design", default=os.environ.get("MEAN_DESIGN", "lat"),
                        choices=["constant", "lat", "latlon", "base", "hour_spatial"],
                        help="GLS trend design. Default is beta0 + centered latitude for separate hourly pure-space fits.")
    parser.add_argument("--sigmasq-init", type=float, default=float(os.environ.get("SIGMASQ_INIT", "10.0")))
    parser.add_argument("--range-init", type=float, default=float(os.environ.get("RANGE_INIT", "0.2")))
    parser.add_argument("--range-lat-init", type=float, default=float(os.environ.get("RANGE_LAT_INIT", os.environ.get("RANGE_INIT", "0.2"))))
    parser.add_argument("--range-lon-init", type=float, default=float(os.environ.get("RANGE_LON_INIT", os.environ.get("RANGE_INIT", "0.2"))))
    parser.add_argument("--nugget-init", type=float, default=float(os.environ.get("NUGGET_INIT", "1.0")))
    parser.add_argument("--lat-range", type=parse_float_pair, default=parse_float_pair(os.environ.get("LAT_RANGE", "-3,7")),
                        help="Required fitted latitude range; default is the expanded domain -3,7.")
    parser.add_argument("--lon-range", type=parse_float_pair, default=parse_float_pair(os.environ.get("LON_RANGE", "111,131")),
                        help="Required fitted longitude range; default is the expanded domain 111,131.")
    parser.add_argument("--region-tol", type=float, default=float(os.environ.get("REGION_TOL", "0.25")),
                        help="Allowed boundary slack when checking that the expanded domain is covered.")
    parser.add_argument("--require-region-cover", action=argparse.BooleanOptionalAction, default=True,
                        help="Fail if the filtered input does not span the requested expanded region.")
    parser.add_argument("--max-points", type=int, default=int(os.environ.get("MAX_POINTS", "0")))
    parser.add_argument("--min-tile-points", type=int, default=int(os.environ.get("MIN_TILE_POINTS", "80")))
    parser.add_argument("--tile-full-max-points", type=int, default=int(os.environ.get("TILE_FULL_MAX_POINTS", "0")),
                        help="Legacy name retained for Slurm compatibility; now controls deterministic thinning before tile Vecchia. 0 means use all tile points.")
    parser.add_argument("--tile-full-jitter", type=float, default=float(os.environ.get("TILE_FULL_JITTER", "1e-6")),
                        help="Legacy full-likelihood option retained for CLI compatibility; tile Vecchia uses the model's internal jitter.")
    parser.add_argument("--tiles", type=int, default=int(os.environ["TILES"]) if os.environ.get("TILES") else None,
                        help="Legacy square tile count. Use --tile-y/--tile-x for rectangular grids.")
    parser.add_argument("--tile-y", type=int, default=int(os.environ["TILE_Y"]) if os.environ.get("TILE_Y") else None,
                        help="Number of south-north tile rows (default: 4 for expanded grid).")
    parser.add_argument("--tile-x", type=int, default=int(os.environ["TILE_X"]) if os.environ.get("TILE_X") else None,
                        help="Number of west-east tile columns (default: 8 for expanded grid).")
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
    parser.add_argument("--nadir-lat", type=float, default=float(os.environ.get("NADIR_LAT", "0.0")),
                        help="GEMS nadir latitude for tile-distance diagnostics.")
    parser.add_argument("--nadir-lon", type=float, default=float(os.environ.get("NADIR_LON", "128.0")),
                        help="GEMS nadir longitude for tile-distance diagnostics.")
    return parser.parse_args()


def normalize_tile_args(args: argparse.Namespace) -> argparse.Namespace:
    tile_y = args.tile_y
    tile_x = args.tile_x
    legacy_tiles = args.tiles

    if tile_y is None and tile_x is None:
        if legacy_tiles is None:
            tile_y, tile_x = 4, 8
        else:
            tile_y = tile_x = int(legacy_tiles)
    elif tile_y is None:
        tile_y = int(legacy_tiles) if legacy_tiles is not None else int(tile_x)
    elif tile_x is None:
        tile_x = int(legacy_tiles) if legacy_tiles is not None else int(tile_y)

    tile_y = int(tile_y)
    tile_x = int(tile_x)
    if tile_y <= 0 or tile_x <= 0:
        raise SystemExit(f"ERROR: tile dimensions must be positive, got {tile_y}x{tile_x}.")

    args.tile_y = tile_y
    args.tile_x = tile_x
    args.tile_count = tile_y * tile_x
    args.tile_tag = f"{tile_y}x{tile_x}"
    return args


def tile_shape(args: argparse.Namespace) -> tuple[int, int]:
    return int(args.tile_y), int(args.tile_x)


def tile_tag(args: argparse.Namespace) -> str:
    return str(args.tile_tag)


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
    x_col = choose_col(df, args.x_col, ["longitude", "lon", "x", "x_km", "source_longitude"], "x")
    y_col = choose_col(df, args.y_col, ["latitude", "lat", "y", "y_km", "source_latitude"], "y")
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


def validate_region_extent(d: pd.DataFrame, x_col: str, y_col: str, args: argparse.Namespace) -> dict:
    lon_vals = pd.to_numeric(d[x_col], errors="coerce").to_numpy(dtype=float)
    lat_vals = pd.to_numeric(d[y_col], errors="coerce").to_numpy(dtype=float)
    lon_vals = lon_vals[np.isfinite(lon_vals)]
    lat_vals = lat_vals[np.isfinite(lat_vals)]
    if lon_vals.size == 0 or lat_vals.size == 0:
        raise ValueError("No finite longitude/latitude values after expanded-region filtering.")

    meta = {
        "fit_lat_min": float(np.min(lat_vals)),
        "fit_lat_max": float(np.max(lat_vals)),
        "fit_lon_min": float(np.min(lon_vals)),
        "fit_lon_max": float(np.max(lon_vals)),
        "requested_lat_min": float(args.lat_range[0]),
        "requested_lat_max": float(args.lat_range[1]),
        "requested_lon_min": float(args.lon_range[0]),
        "requested_lon_max": float(args.lon_range[1]),
    }
    tol = float(args.region_tol)
    covers = (
        meta["fit_lat_min"] <= meta["requested_lat_min"] + tol
        and meta["fit_lat_max"] >= meta["requested_lat_max"] - tol
        and meta["fit_lon_min"] <= meta["requested_lon_min"] + tol
        and meta["fit_lon_max"] >= meta["requested_lon_max"] - tol
    )
    meta["region_cover_ok"] = bool(covers)
    if not covers:
        msg = (
            "Filtered data do not cover the requested expanded domain. "
            f"requested lat={args.lat_range}, lon={args.lon_range}; "
            f"observed lat=[{meta['fit_lat_min']:.4f}, {meta['fit_lat_max']:.4f}], "
            f"lon=[{meta['fit_lon_min']:.4f}, {meta['fit_lon_max']:.4f}], tol={tol}."
        )
        if bool(args.require_region_cover):
            raise ValueError(msg)
        print("WARNING:", msg, flush=True)
    return meta


def prepare_hour_data(df: pd.DataFrame, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict]:
    _, x_col, y_col, value_col = column_names(df, args)
    cols = [x_col, y_col, value_col]
    d = df[cols].copy()
    d = d.replace([np.inf, -np.inf], np.nan)
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d = d.dropna(subset=[x_col, y_col])
    if d.empty:
        raise ValueError("No finite coordinate rows after coordinate filtering.")

    before_region = int(len(d))
    d = d[
        d[y_col].between(args.lat_range[0], args.lat_range[1])
        & d[x_col].between(args.lon_range[0], args.lon_range[1])
    ].copy()
    if d.empty:
        raise ValueError(
            f"No rows after expanded-region filter lat={args.lat_range}, lon={args.lon_range} "
            f"from {before_region} finite rows."
        )

    # Validate that the input grid itself is the expanded domain.  The finite
    # retrieval values can cover a smaller swath within that grid for a given
    # hour, so the coverage check must happen before dropping missing values.
    region_meta = validate_region_extent(d, x_col, y_col, args)
    n_after_region_filter = int(len(d))

    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna(subset=[value_col])
    if d.empty:
        raise ValueError(
            f"No finite values after expanded-region filter lat={args.lat_range}, lon={args.lon_range}."
        )

    value_lat = d[y_col].to_numpy(dtype=float)
    value_lon = d[x_col].to_numpy(dtype=float)
    region_meta.update({
        "finite_value_lat_min": float(np.min(value_lat)),
        "finite_value_lat_max": float(np.max(value_lat)),
        "finite_value_lon_min": float(np.min(value_lon)),
        "finite_value_lon_max": float(np.max(value_lon)),
    })
    tol = float(args.region_tol)
    finite_covers = (
        region_meta["finite_value_lat_min"] <= region_meta["requested_lat_min"] + tol
        and region_meta["finite_value_lat_max"] >= region_meta["requested_lat_max"] - tol
        and region_meta["finite_value_lon_min"] <= region_meta["requested_lon_min"] + tol
        and region_meta["finite_value_lon_max"] >= region_meta["requested_lon_max"] - tol
    )
    region_meta["finite_value_region_cover_ok"] = bool(finite_covers)
    if not finite_covers:
        msg = (
            "Finite values do not cover the requested expanded domain. "
            f"requested lat={args.lat_range}, lon={args.lon_range}; "
            f"finite values lat=[{region_meta['finite_value_lat_min']:.4f}, "
            f"{region_meta['finite_value_lat_max']:.4f}], "
            f"lon=[{region_meta['finite_value_lon_min']:.4f}, "
            f"{region_meta['finite_value_lon_max']:.4f}], tol={tol}. "
            "The regular grid still covers the requested domain; a smaller finite "
            "retrieval swath can occur for individual hours."
        )
        if bool(getattr(args, "require_finite_value_region_cover", False)):
            raise ValueError(msg)
        print("WARNING:", msg, flush=True)

    d = d.groupby([x_col, y_col], as_index=False)[value_col].mean()
    n_after_value_filter = int(len(d))

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
    coord_meta.update(region_meta)
    coord_meta["n_before_region_filter"] = before_region
    coord_meta["n_after_region_filter"] = n_after_region_filter
    coord_meta["n_after_value_filter"] = n_after_value_filter
    coord_meta["n_used_for_fit"] = int(len(d))
    return coords, values, d, coord_meta


# ---------------------------------------------------------------------------
# GPU fitting core
# ---------------------------------------------------------------------------

def _build_gpu_model(
    y: np.ndarray,
    coords: np.ndarray,
    smooth: float,
    args: argparse.Namespace,
    device: torch.device,
):
    n = len(y)
    data = np.zeros((n, 4), dtype=np.float64)
    data[:, 0] = coords[:, 0]   # y_km
    data[:, 1] = coords[:, 1]   # x_km
    data[:, 2] = y
    tensor = torch.from_numpy(data).to(device=device, dtype=torch.float64)
    model = EDAClusterSpaceVecchiaFit(
        smooth=smooth,
        input_map={"t0": tensor},
        grid_coords=np.asarray(coords, dtype=np.float64),
        block_shape=tuple(args.cluster_block_shape),
        n_neighbor_blocks=int(args.cluster_neighbor_blocks),
        target_chunk_size=int(args.target_chunk_size),
        min_target_points=int(args.min_target_points),
        mean_design=str(args.mean_design),
    )
    model.precompute_conditioning_sets()
    return model


def raw_to_float_list(raw) -> list[float]:
    if isinstance(raw, torch.Tensor):
        return [float(x) for x in raw.detach().cpu().reshape(-1).tolist()]
    out = []
    for x in raw:
        if isinstance(x, torch.Tensor):
            out.append(float(x.detach().cpu().item()))
        else:
            out.append(float(x))
    return out


def fit_global_with_model(
    y: np.ndarray,
    coords: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[FitResult, EDAClusterSpaceVecchiaFit, list[float]]:
    if len(y) < 5:
        raise ValueError("Need at least 5 observations for global fit.")
    smooth = float(args.smooth)
    if smooth <= 0.0:
        raise ValueError(f"smooth must be positive, got {smooth}")

    y_mean, y_std = float(np.nanmean(y)), float(np.nanstd(y))
    print(f"[fit_global] n={len(y)} | y mean={y_mean:.4f} std={y_std:.4f} min={float(np.nanmin(y)):.4f} max={float(np.nanmax(y)):.4f}", flush=True)
    if y_std > 500:
        print(f"[fit_global] WARNING: y_std={y_std:.4f} is unusually large — check data units/columns", flush=True)

    model = _build_gpu_model(y, coords, smooth, args, device)

    # Fixed init informed by pure-space EDA. The optimizer sees
    # log(phi1), log(phi2), log(phi3), log(nugget), with
    # phi2=1/range_lon, phi3=(range_lon/range_lat)^2, sigmasq=phi1/phi2.
    init_sigmasq = float(args.sigmasq_init)
    init_range_lat = float(getattr(args, "range_lat_init", args.range_init))
    init_range_lon = float(getattr(args, "range_lon_init", args.range_init))
    init_nugget = float(args.nugget_init)
    init_phi2 = 1.0 / init_range_lon
    init_phi3 = (init_range_lon / init_range_lat) ** 2
    init_phi1 = init_sigmasq * init_phi2
    params_list = [
        torch.tensor(math.log(init_phi1), requires_grad=True, dtype=torch.float64, device=device),
        torch.tensor(math.log(init_phi2), requires_grad=True, dtype=torch.float64, device=device),
        torch.tensor(math.log(init_phi3), requires_grad=True, dtype=torch.float64, device=device),
        torch.tensor(math.log(init_nugget), requires_grad=True, dtype=torch.float64, device=device),
    ]
    opt = model.set_optimizer(params_list, lr=1.0, max_iter=150, tolerance_grad=1e-5)
    raw, iters = model.fit_vecc_lbfgs(params_list, opt, max_steps=60, grad_tol=1e-5)
    raw_vals = raw_to_float_list(raw)
    raw_params = raw_vals[:4]

    phi1 = math.exp(raw_params[0])
    phi2 = math.exp(raw_params[1])
    phi3 = math.exp(raw_params[2])
    sigmasq = phi1 / phi2
    range_lon = 1.0 / phi2
    range_lat = 1.0 / (phi2 * math.sqrt(phi3))
    range_ = math.sqrt(range_lat * range_lon)
    nugget  = math.exp(raw_params[3])
    fit = FitResult(
        success=True,
        nll=float(raw_vals[-1]),
        sigmasq=sigmasq,
        sigma=math.sqrt(sigmasq),
        range=range_,
        range_lat=range_lat,
        range_lon=range_lon,
        nugget=nugget,
        phi1=phi1,
        phi2=phi2,
        phi3=phi3,
        message="lbfgs_gpu_phi1_phi2_phi3",
        n_eval=iters,
    )
    return fit, model, raw_params


def fit_global(
    y: np.ndarray,
    coords: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> FitResult:
    fit, model, _ = fit_global_with_model(y, coords, args, device)
    del model
    return fit


def vecchia_whitened_residuals_by_obs(
    model: EDAClusterSpaceVecchiaFit,
    raw_params: list[float],
    n_obs: int,
) -> np.ndarray:
    """Return target-row Vecchia whitened residuals mapped to input rows."""
    device = getattr(model, "device", None)
    if device is None:
        device = next(iter(model.input_map.values())).device
    params = torch.as_tensor(raw_params, device=device, dtype=torch.float64)
    n_features = int(model.n_features)
    xt_sinv_x = torch.zeros((n_features, n_features), device=device, dtype=torch.float64)
    xt_sinv_y = torch.zeros((n_features, 1), device=device, dtype=torch.float64)
    chunk_size = max(1, int(model.target_chunk_size))

    with torch.no_grad():
        for batch in getattr(model, "_cluster_batches", []):
            if batch.coords.shape[0] == 0:
                continue
            target_slice = slice(batch.max_cond_points, batch.max_cond_points + batch.target_size)
            for start in range(0, batch.coords.shape[0], chunk_size):
                end = min(start + chunk_size, batch.coords.shape[0])
                k_mat = model._cov_full(batch.coords[start:end], params)
                chol = torch.linalg.cholesky(k_mat)
                z_x = torch.linalg.solve_triangular(chol, batch.X[start:end], upper=False)
                z_y = torch.linalg.solve_triangular(chol, batch.y[start:end], upper=False)
                u_x = z_x[:, target_slice, :].reshape(-1, n_features)
                u_y = z_y[:, target_slice, :].reshape(-1, 1)
                xt_sinv_x += u_x.T @ u_x
                xt_sinv_y += u_x.T @ u_y

        beta = torch.linalg.solve(
            xt_sinv_x + torch.eye(n_features, device=device, dtype=torch.float64) * 1e-8,
            xt_sinv_y,
        )
        out = torch.full((int(n_obs),), float("nan"), device=device, dtype=torch.float64)
        for batch in getattr(model, "_cluster_batches", []):
            if batch.coords.shape[0] == 0:
                continue
            target_slice = slice(batch.max_cond_points, batch.max_cond_points + batch.target_size)
            for start in range(0, batch.coords.shape[0], chunk_size):
                end = min(start + chunk_size, batch.coords.shape[0])
                k_mat = model._cov_full(batch.coords[start:end], params)
                chol = torch.linalg.cholesky(k_mat)
                z_x = torch.linalg.solve_triangular(chol, batch.X[start:end], upper=False)
                z_y = torch.linalg.solve_triangular(chol, batch.y[start:end], upper=False)
                resid = (z_y[:, target_slice, :] - z_x[:, target_slice, :] @ beta).reshape(-1)
                rows = batch.rows[start:end, target_slice].reshape(-1)
                real = rows < int(n_obs)
                out[rows[real].long()] = resid[real]
    return out.detach().cpu().numpy()


def apply_vecchia_whitened_qc(
    values: np.ndarray,
    coords: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    threshold = float(getattr(args, "qc_whitened_threshold", 0.0))
    qc = {
        "qc_whitened_threshold": threshold,
        "qc_max_abs_whitened": np.nan,
        "n_qc_initial_fit": int(len(values)),
        "n_qc_removed": 0,
        "n_qc_fit": int(len(values)),
        "qc_refit": False,
        "qc_message": "disabled" if threshold <= 0.0 else "no_outliers",
    }
    if threshold <= 0.0:
        return values, coords, qc

    try:
        initial_fit, initial_model, raw_params = fit_global_with_model(values, coords, args, device)
        w = vecchia_whitened_residuals_by_obs(initial_model, raw_params, n_obs=int(len(values)))
        del initial_model
        abs_w = np.abs(np.asarray(w, dtype=float))
        finite = np.isfinite(abs_w)
        bad = finite & (abs_w > threshold)
        qc["qc_max_abs_whitened"] = float(np.nanmax(abs_w)) if np.any(finite) else np.nan
        qc["pre_qc_loss"] = float(initial_fit.nll)
        qc["pre_qc_sigmasq"] = float(initial_fit.sigmasq)
        qc["pre_qc_range_lat"] = float(initial_fit.range_lat)
        qc["pre_qc_range_lon"] = float(initial_fit.range_lon)
        qc["pre_qc_nugget"] = float(initial_fit.nugget)
        n_keep = int(len(values) - np.sum(bad))
        if np.any(bad) and n_keep >= int(getattr(args, "qc_min_fit_points", 80)):
            keep = ~bad
            qc["n_qc_removed"] = int(np.sum(bad))
            qc["n_qc_fit"] = n_keep
            qc["qc_refit"] = True
            qc["qc_message"] = "removed_whitened_outliers"
            return values[keep], coords[keep], qc
        if np.any(bad):
            qc["qc_message"] = f"skipped_too_few_after_qc:{n_keep}"
        return values, coords, qc
    except Exception as exc:
        qc["qc_message"] = f"whitened_qc_skipped_error:{exc}"
        return values, coords, qc


def assign_tiles(coords: np.ndarray, tile_y_count: int, tile_x_count: int) -> tuple[np.ndarray, dict]:
    x = coords[:, 1]   # x_km (lon dir) for west-east
    y = coords[:, 0]   # y_km (lat dir) for south-north
    eps_x = max((float(x.max()) - float(x.min())) * 1e-12, 1e-12)
    eps_y = max((float(y.max()) - float(y.min())) * 1e-12, 1e-12)
    x_edges = np.linspace(float(x.min()), float(x.max()) + eps_x, tile_x_count + 1)
    y_edges = np.linspace(float(y.min()), float(y.max()) + eps_y, tile_y_count + 1)
    x_idx = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, tile_x_count - 1)
    y_idx = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, tile_y_count - 1)
    tile_id = y_idx * tile_x_count + x_idx
    meta = {
        "x_edges": x_edges.tolist(),
        "y_edges": y_edges.tolist(),
        "tile_y": int(tile_y_count),
        "tile_x": int(tile_x_count),
        "tile_count": int(tile_y_count * tile_x_count),
    }
    return tile_id.astype(int), meta


def add_tile_geometry(tile_df: pd.DataFrame, tile_meta: dict, args: argparse.Namespace) -> pd.DataFrame:
    """Attach tile centers and nadir-distance diagnostics to per-tile fit rows."""
    out = tile_df.copy()
    x_edges = np.asarray(tile_meta["x_edges"], dtype=float)
    y_edges = np.asarray(tile_meta["y_edges"], dtype=float)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    tile_x = out["tile_x"].astype(int).to_numpy()
    tile_y = out["tile_y"].astype(int).to_numpy()
    center_x = x_centers[tile_x]
    center_y = y_centers[tile_y]
    out["tile_center_x"] = center_x
    out["tile_center_y"] = center_y
    out["tile_x_min"] = x_edges[tile_x]
    out["tile_x_max"] = x_edges[tile_x + 1]
    out["tile_y_min"] = y_edges[tile_y]
    out["tile_y_max"] = y_edges[tile_y + 1]

    if args.coords == "raw":
        nadir_lat = float(args.nadir_lat)
        nadir_lon = float(args.nadir_lon)
        out["tile_center_lon"] = center_x
        out["tile_center_lat"] = center_y
        out["tile_lon_min"] = out["tile_x_min"]
        out["tile_lon_max"] = out["tile_x_max"]
        out["tile_lat_min"] = out["tile_y_min"]
        out["tile_lat_max"] = out["tile_y_max"]
        dlat = center_y - nadir_lat
        dlon = (center_x - nadir_lon) * math.cos(math.radians(nadir_lat))
        out["dist_to_nadir_deg"] = np.hypot(dlat, dlon)
        out["dist_to_nadir_km"] = np.hypot(dlat * 110.574, dlon * 111.320)
    else:
        out["dist_to_nadir_coord"] = np.hypot(center_y - float(args.nadir_lat),
                                              center_x - float(args.nadir_lon))
    return out


def _deterministic_tile_subset(n: int, max_points: int) -> np.ndarray:
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=int)
    # Evenly spaced deterministic thinning keeps results reproducible and avoids
    # tile-to-tile random seed artifacts when a tile is too large.
    return np.unique(np.linspace(0, n - 1, int(max_points)).round().astype(int))


def fit_tile_vecchia_parameters(
    y: np.ndarray,
    coords: np.ndarray,
    tile_id: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> pd.DataFrame:
    """Fit each tile with the same torch/autograd group Vecchia model."""
    rows = []
    tile_y_count, tile_x_count = tile_shape(args)
    max_points = int(getattr(args, "tile_full_max_points", 0) or 0)

    for tid in range(tile_y_count * tile_x_count):
        mask = tile_id == tid
        n_tile = int(mask.sum())
        y_idx = tid // tile_x_count
        x_idx = tid % tile_x_count
        if n_tile < args.min_tile_points:
            rows.append({
                "tile_id": tid, "tile_x": x_idx, "tile_y": y_idx, "n": n_tile,
                "n_vecchia": 0, "tile_nugget": np.nan, "tile_loss": np.nan,
                "success": False, "message": "too_few_points",
                "tile_fit_method": "torch_vecchia_autograd",
            })
            continue

        yy_all = y[mask].astype(np.float64, copy=False)
        cc_all = coords[mask].astype(np.float64, copy=False)
        keep = _deterministic_tile_subset(n_tile, max_points)
        yy = yy_all[keep]
        cc = cc_all[keep]
        n_used = int(len(yy))
        try:
            fit = fit_global(yy, cc, args, device)
            msg = str(fit.message)
            if max_points > 0 and n_tile > max_points:
                msg = f"{msg}; deterministic_thin {n_tile}->{n_used}"
            rows.append({
                "tile_id": tid, "tile_x": x_idx, "tile_y": y_idx, "n": n_tile,
                "n_vecchia": n_used,
                "tile_sigmasq": fit.sigmasq,
                "tile_sigma": fit.sigma,
                "tile_range": fit.range,
                "tile_range_lat": fit.range_lat,
                "tile_range_lon": fit.range_lon,
                "tile_nugget": fit.nugget,
                "tile_phi1": fit.phi1,
                "tile_phi2": fit.phi2,
                "tile_phi3": fit.phi3,
                "tile_loss": fit.nll,
                "success": bool(fit.success),
                "message": msg,
                "tile_fit_method": "torch_vecchia_autograd",
            })
        except Exception as exc:
            rows.append({
                "tile_id": tid, "tile_x": x_idx, "tile_y": y_idx, "n": n_tile,
                "n_vecchia": n_used,
                "tile_nugget": np.nan,
                "tile_loss": np.nan,
                "success": False,
                "message": f"ERROR: {exc}",
                "tile_fit_method": "torch_vecchia_autograd",
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

    coords, values, _, coord_meta = prepare_hour_data(df_hour, args)
    values_fit, coords_fit, qc_meta = apply_vecchia_whitened_qc(values, coords, args, device)
    global_fit = fit_global(values_fit, coords_fit, args, device)
    tile_id, tile_meta = assign_tiles(coords_fit, *tile_shape(args))
    tile_df = fit_tile_vecchia_parameters(values_fit, coords_fit, tile_id, args, device)
    tile_df = add_tile_geometry(tile_df, tile_meta, args)

    global_row = {
        "day": hour.strftime("%Y-%m-%d"),
        "day_index": day_index,
        "hour_slot": hour_slot,
        "hour_utc": hour_utc,
        "hour_key": hour_key,
        "hour": hour.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_raw": int(len(df_hour)),
        "n_used": int(len(values)),
        "n_fit": int(len(values_fit)),
        "lat_min": coord_meta.get("fit_lat_min", np.nan),
        "lat_max": coord_meta.get("fit_lat_max", np.nan),
        "lon_min": coord_meta.get("fit_lon_min", np.nan),
        "lon_max": coord_meta.get("fit_lon_max", np.nan),
        "region_cover_ok": bool(coord_meta.get("region_cover_ok", False)),
        "sigmasq": global_fit.sigmasq,
        "sigma": global_fit.sigma,
        "range": global_fit.range,
        "range_lat": global_fit.range_lat,
        "range_lon": global_fit.range_lon,
        "nugget": global_fit.nugget,
        "phi1": global_fit.phi1,
        "phi2": global_fit.phi2,
        "phi3": global_fit.phi3,
        "loss": global_fit.nll,
        "success": bool(global_fit.success),
        **qc_meta,
    }
    tile_df.insert(0, "day", hour.strftime("%Y-%m-%d"))
    tile_df.insert(1, "day_index", day_index)
    tile_df.insert(2, "hour_slot", hour_slot)
    tile_df.insert(3, "hour_utc", hour_utc)
    tile_df.insert(4, "hour_key", hour_key)
    tile_df.insert(5, "hour", hour.strftime("%Y-%m-%dT%H:%M:%SZ"))
    tile_df["global_nugget"] = global_fit.nugget
    tile_df["global_sigmasq"] = global_fit.sigmasq
    tile_df["global_range"] = global_fit.range
    tile_df["global_range_lat"] = global_fit.range_lat
    tile_df["global_range_lon"] = global_fit.range_lon
    for key, value in qc_meta.items():
        tile_df[key] = value
    return global_row, tile_df


# ---------------------------------------------------------------------------
# Summarization helpers
# ---------------------------------------------------------------------------

def round_numeric(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(digits)
    return out




def round_json_numbers(obj, digits: int = 4):
    if isinstance(obj, dict):
        return {k: round_json_numbers(v, digits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [round_json_numbers(v, digits) for v in obj]
    if isinstance(obj, tuple):
        return [round_json_numbers(v, digits) for v in obj]
    if isinstance(obj, (float, np.floating)):
        if np.isfinite(obj):
            return round(float(obj), digits)
        return None
    return obj

def csv_line_from_row(row: dict, columns: list[str]) -> str:
    vals = []
    for col in columns:
        val = row.get(col, "")
        if isinstance(val, float):
            vals.append(f"{val:.4f}")
        else:
            vals.append(str(val))
    return ",".join(vals)


def merge_tile_geometry_summary(summary: pd.DataFrame, tile_df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    geometry_cols = [
        "tile_center_x", "tile_center_y", "tile_center_lon", "tile_center_lat",
        "tile_x_min", "tile_x_max", "tile_y_min", "tile_y_max",
        "tile_lon_min", "tile_lon_max", "tile_lat_min", "tile_lat_max",
        "dist_to_nadir_deg", "dist_to_nadir_km", "dist_to_nadir_coord",
    ]
    geometry_cols = [c for c in geometry_cols if c in tile_df.columns]
    if summary.empty or not geometry_cols:
        return summary
    geom = tile_df.groupby(keys, as_index=False)[geometry_cols].mean()
    return summary.merge(geom, on=keys, how="left")


def tile_mean_summary(tile_df: pd.DataFrame, tile_y_count: int, tile_x_count: int) -> tuple[pd.DataFrame, np.ndarray]:
    keys = ["tile_y", "tile_x", "tile_id"]
    d = tile_df.dropna(subset=["tile_nugget"]).copy()
    optional_mean_cols = [
        c for c in [
            "tile_sigmasq", "tile_sigma", "tile_range", "tile_range_lat", "tile_range_lon",
            "tile_phi1", "tile_phi2", "tile_phi3", "tile_loss", "n_vecchia",
            "global_sigmasq", "global_range", "global_range_lat", "global_range_lon",
            "global_nugget", "n_qc_removed", "n_qc_fit", "qc_max_abs_whitened",
        ]
        if c in d.columns
    ]
    if d.empty:
        summary = pd.DataFrame(columns=keys + [
            "n_days", "n_hours", "tile_nugget_mean", "tile_nugget_sd", "tile_nugget_median",
        ])
    elif "day_index" in d.columns:
        daily_aggs = {
            "daily_tile_nugget": ("tile_nugget", "mean"),
            "daily_n_hours": ("tile_nugget", "count"),
        }
        for col in optional_mean_cols:
            daily_aggs[f"daily_{col}"] = (col, "mean")
        day_key = "day_index"
        daily = (
            d.groupby(keys + [day_key], as_index=False)
            .agg(**daily_aggs)
        )
        summary_aggs = {
            "n_days": ("daily_tile_nugget", "count"),
            "n_hours": ("daily_n_hours", "sum"),
            "tile_nugget_mean": ("daily_tile_nugget", "mean"),
            "tile_nugget_sd": ("daily_tile_nugget", "std"),
            "tile_nugget_median": ("daily_tile_nugget", "median"),
        }
        for col in optional_mean_cols:
            summary_aggs[f"{col}_mean"] = (f"daily_{col}", "mean")
        summary = (
            daily.groupby(keys, as_index=False)
            .agg(**summary_aggs)
            .sort_values(["tile_y", "tile_x"])
        )
    else:
        aggs = {
            "n_days": ("tile_nugget", "count"),
            "n_hours": ("tile_nugget", "count"),
            "tile_nugget_mean": ("tile_nugget", "mean"),
            "tile_nugget_sd": ("tile_nugget", "std"),
            "tile_nugget_median": ("tile_nugget", "median"),
        }
        for col in optional_mean_cols:
            aggs[f"{col}_mean"] = (col, "mean")
        summary = (
            d.groupby(keys, as_index=False)
            .agg(**aggs)
            .sort_values(["tile_y", "tile_x"])
        )

    summary = merge_tile_geometry_summary(summary, d, keys)

    mat = np.full((tile_y_count, tile_x_count), np.nan)
    for _, row in summary.iterrows():
        mat[int(row["tile_y"]), int(row["tile_x"])] = row["tile_nugget_mean"]
    return summary, mat


def global_nugget_by_hour_slot(global_df: pd.DataFrame) -> pd.DataFrame:
    if "day" not in global_df.columns:
        global_df = add_global_time_features(global_df)
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
    if "day" not in global_df.columns:
        global_df = add_global_time_features(global_df)
    agg = {
        "n_days": ("day", "nunique"),
        "n_hours": ("day", "count"),
    }
    for col in ["sigmasq", "sigma", "range", "range_lat", "range_lon", "nugget", "phi1", "phi2", "phi3", "loss"]:
        if col in global_df.columns:
            agg[f"{col}_mean"] = (col, "mean")
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
    tile_y_count, tile_x_count = tile_shape(args)
    _, tile_mat = tile_mean_summary(tile_df, tile_y_count, tile_x_count)
    hour_slot_params = global_params_by_hour_slot(global_df)
    return "\n\n".join([
        label,
        f"running day-normalized avg: {tile_tag(args)} tile nuggets (rows=south->north, cols=west->east)",
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
                   "n_fit", "n_qc_removed", "qc_max_abs_whitened",
                   "lat_min", "lat_max", "lon_min", "lon_max", "region_cover_ok",
                   "sigmasq", "sigma", "range", "range_lat", "range_lon",
                   "nugget", "phi1", "phi2", "phi3", "loss", "pre_qc_loss",
                   "pre_qc_sigmasq", "pre_qc_range_lat", "pre_qc_range_lon",
                   "pre_qc_nugget", "qc_message"]
    global_cols_avail = [c for c in global_cols if c in global_df.columns]
    global_out = round_numeric(global_df[global_cols_avail], 4)
    global_path = output_dir / f"{month_label}_spatial_fit_{len(global_out)}.csv"
    global_out.to_csv(global_path, index=False, float_format="%.4f")

    tile_y_count, tile_x_count = tile_shape(args)
    tag = tile_tag(args)
    tile_summary, tile_mat = tile_mean_summary(tile_df, tile_y_count, tile_x_count)
    tile_median_mat = heatmap_matrix(tile_summary, "tile_nugget_median", tile_y_count, tile_x_count)
    tile_summary = round_numeric(tile_summary, 4)
    tile_summary_path = output_dir / f"tile_nugget_mean_{tag}.csv"
    tile_summary.to_csv(tile_summary_path, index=False, float_format="%.4f")
    nadir_cols = [
        c for c in [
            "tile_y", "tile_x", "tile_id", "tile_center_lat", "tile_center_lon",
            "tile_lat_min", "tile_lat_max", "tile_lon_min", "tile_lon_max",
            "dist_to_nadir_deg", "dist_to_nadir_km",
            "n_days", "n_hours", "tile_nugget_mean", "tile_nugget_median", "tile_nugget_sd",
        ]
        if c in tile_summary.columns
    ]
    if nadir_cols:
        tile_summary[nadir_cols].to_csv(
            output_dir / f"tile_nugget_vs_nadir_distance_{tag}.csv",
            index=False, float_format="%.4f",
        )

    hour_slot_nugget = round_numeric(global_nugget_by_hour_slot(global_df), 4)
    hour_slot_params = round_numeric(global_params_by_hour_slot(global_df), 4)
    hour_slot_nugget.to_csv(output_dir / "global_nugget_by_hour_slot.csv", index=False, float_format="%.4f")
    hour_slot_path = output_dir / "global_params_by_hour_slot.csv"
    hour_slot_params.to_csv(hour_slot_path, index=False, float_format="%.4f")
    global_with_time, global_by_day, global_by_hour_slot = summarize_global_by_time(global_df)
    round_numeric(global_by_day, 4).to_csv(
        output_dir / "global_params_by_day.csv", index=False, float_format="%.4f"
    )

    save_tile_geographic_heatmap(
        tile_summary, "tile_nugget_mean", tile_y_count, tile_x_count,
        f"{args.month} mean tile nugget, {tag}", "mean tile nugget",
        output_dir / f"tile_nugget_mean_heatmap_{tag}.png",
        args,
    )
    save_tile_geographic_heatmap(
        tile_summary, "tile_nugget_median", tile_y_count, tile_x_count,
        f"{args.month} median tile nugget, {tag}", "median tile nugget",
        output_dir / f"tile_nugget_median_heatmap_{tag}.png",
        args,
    )
    for col, label in [
        ("tile_sigmasq_mean", "mean tile sigmasq"),
        ("tile_range_lat_mean", "mean tile range lat"),
        ("tile_range_lon_mean", "mean tile range lon"),
        ("tile_phi1_mean", "mean tile phi1"),
        ("tile_phi2_mean", "mean tile phi2"),
        ("tile_phi3_mean", "mean tile phi3"),
        ("global_sigmasq_mean", "mean global sigmasq"),
        ("global_range_lat_mean", "mean global range lat"),
        ("global_range_lon_mean", "mean global range lon"),
        ("global_nugget_mean", "mean global nugget"),
        ("n_qc_removed_mean", "mean QC removed count"),
    ]:
        if col in tile_summary.columns:
            save_tile_geographic_heatmap(
                tile_summary, col, tile_y_count, tile_x_count,
                f"{args.month} {label}, {tag}", label,
                output_dir / f"{col}_heatmap_{tag}.png",
                args,
            )
    save_tile_nugget_vs_nadir_plot(
        tile_summary,
        output_dir / f"tile_nugget_vs_nadir_distance_{tag}.png",
        f"{args.month} tile nugget vs GEMS nadir distance ({tag})",
        mean_col="tile_nugget_mean",
        median_col="tile_nugget_median",
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
    values_fit, coords_fit, qc_meta = apply_vecchia_whitened_qc(values, coords, args, device)
    global_fit = fit_global(values_fit, coords_fit, args, device)
    tile_id, tile_meta = assign_tiles(coords_fit, *tile_shape(args))
    tile_df = fit_tile_vecchia_parameters(values_fit, coords_fit, tile_id, args, device)
    tile_df = add_tile_geometry(tile_df, tile_meta, args)

    global_row = {
        "hour_index": hour_index,
        "day_index": day_index,
        "hour_slot": hour_slot,
        "hour_utc": hour_utc,
        "hour": hour.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_raw": int(len(df_hour)),
        "n_used": int(len(values)),
        "n_fit": int(len(values_fit)),
        "lat_min": coord_meta.get("fit_lat_min", np.nan),
        "lat_max": coord_meta.get("fit_lat_max", np.nan),
        "lon_min": coord_meta.get("fit_lon_min", np.nan),
        "lon_max": coord_meta.get("fit_lon_max", np.nan),
        "region_cover_ok": bool(coord_meta.get("region_cover_ok", False)),
        "smooth": float(args.smooth),
        "method": "gpu_cluster_group_vecchia_phi1_phi2_phi3",
        "cluster_block_shape": f"{args.cluster_block_shape[0]}x{args.cluster_block_shape[1]}",
        "cluster_neighbor_blocks": int(args.cluster_neighbor_blocks),
        "mean_design": str(args.mean_design),
        **asdict(global_fit),
        "resid_var": float(np.nanvar(values_fit, ddof=1)),
        **qc_meta,
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
    tile_df["global_range_lat"] = global_fit.range_lat
    tile_df["global_range_lon"] = global_fit.range_lon
    tile_df["global_nugget"]  = global_fit.nugget
    tile_df["tile_to_global_nugget"] = tile_df["tile_nugget"] / global_fit.nugget
    for key, value in qc_meta.items():
        tile_df[key] = value

    round_numeric(global_df, 4).to_csv(global_path, index=False, float_format="%.4f")
    round_numeric(tile_df, 4).to_csv(tile_path, index=False, float_format="%.4f")
    meta = {
        "args": vars(args),
        "coord_meta": coord_meta,
        "qc_meta": qc_meta,
        "tile_meta": tile_meta,
        "hour": hour.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "device": str(device),
    }
    meta_path.write_text(json.dumps(round_json_numbers(meta, 4), indent=2), encoding="utf-8")
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


def heatmap_matrix(summary: pd.DataFrame, value_col: str, tile_y_count: int, tile_x_count: int) -> np.ndarray:
    mat = np.full((tile_y_count, tile_x_count), np.nan)
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

    n_rows, n_cols = mat.shape
    fig_w = max(6.5, 1.0 * n_cols + 2.0)
    fig_h = max(5.4, 0.75 * n_rows + 2.0)
    text_size = 9 if n_rows * n_cols <= 16 else 7
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
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
            ax.text(x, y, label, ha="center", va="center", color="white", fontsize=text_size)
    cbar = fig.colorbar(im, ax=ax, shrink=0.86)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _tile_edges_from_summary(summary: pd.DataFrame, min_col: str, max_col: str, count: int) -> np.ndarray | None:
    if min_col not in summary.columns or max_col not in summary.columns:
        return None
    vals = np.concatenate([
        pd.to_numeric(summary[min_col], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(summary[max_col], errors="coerce").to_numpy(dtype=float),
    ])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None
    rounded = np.unique(np.round(vals, 8))
    if rounded.size >= count + 1:
        return np.sort(rounded)
    return np.linspace(float(np.nanmin(vals)), float(np.nanmax(vals)), count + 1)


def _fmt_geo_tick(x: float) -> str:
    if not np.isfinite(x):
        return ""
    if abs(x - round(x)) < 1e-8:
        return f"{int(round(x))}"
    return f"{x:.2f}".rstrip("0").rstrip(".")


def save_tile_geographic_heatmap(
    summary: pd.DataFrame,
    value_col: str,
    tile_y_count: int,
    tile_x_count: int,
    title: str,
    cbar_label: str,
    path: Path,
    args: argparse.Namespace,
) -> None:
    """Tile heatmap with longitude/latitude axes and GEMS nadir marker."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mat = heatmap_matrix(summary, value_col, tile_y_count, tile_x_count)
    lon_edges = _tile_edges_from_summary(summary, "tile_lon_min", "tile_lon_max", tile_x_count)
    lat_edges = _tile_edges_from_summary(summary, "tile_lat_min", "tile_lat_max", tile_y_count)

    if lon_edges is None or lat_edges is None:
        save_heatmap(mat, title, cbar_label, path)
        return

    fig_w = max(8.2, 1.05 * tile_x_count + 2.2)
    fig_h = max(6.0, 0.85 * tile_y_count + 2.2)
    text_size = 9 if tile_y_count * tile_x_count <= 32 else 7
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    extent = [float(lon_edges[0]), float(lon_edges[-1]), float(lat_edges[0]), float(lat_edges[-1])]
    im = ax.imshow(mat, origin="lower", cmap="viridis", extent=extent, aspect="auto")

    for lon in lon_edges:
        ax.axvline(lon, color="white", lw=0.5, alpha=0.45)
    for lat in lat_edges:
        ax.axhline(lat, color="white", lw=0.5, alpha=0.45)

    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    for y_idx, lat in enumerate(lat_centers):
        for x_idx, lon in enumerate(lon_centers):
            val = mat[y_idx, x_idx]
            label = "NA" if not np.isfinite(val) else f"{val:.4f}"
            ax.text(lon, lat, label, ha="center", va="center", color="white", fontsize=text_size)

    nadir_lon = float(args.nadir_lon)
    nadir_lat = float(args.nadir_lat)
    if extent[0] <= nadir_lon <= extent[1] and extent[2] <= nadir_lat <= extent[3]:
        ax.scatter(
            [nadir_lon], [nadir_lat],
            marker="*", s=230, c="white", edgecolors="black", linewidths=1.1,
            label=f"nadir ({nadir_lon:g}E, {nadir_lat:g}N)", zorder=5,
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xticks(lon_edges)
    ax.set_yticks(lat_edges)
    ax.set_xticklabels([_fmt_geo_tick(x) for x in lon_edges], rotation=0)
    ax.set_yticklabels([_fmt_geo_tick(y) for y in lat_edges])
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
        ("range",  "global range geom mean"),
        ("range_lat",  "global range lat"),
        ("range_lon",  "global range lon"),
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
    param_cols = [
        "sigmasq", "sigma", "range", "range_lat", "range_lon",
        "nugget", "phi1", "phi2", "phi3", "loss", "nll", "resid_var",
    ]
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


PLOT_PARAM_CAPS = {
    "sigma": 50.0,
    "range": 10.0,
    "range_lat": 10.0,
    "range_lon": 10.0,
    "nugget": 5.0,
}


def _readable_ymax(col: str, mean: np.ndarray, sd: np.ndarray | None = None) -> float:
    hard_cap = PLOT_PARAM_CAPS.get(col, 50.0)
    arrays = [mean]
    if sd is not None:
        arrays.append(mean + np.nan_to_num(sd, nan=0.0))
    vals = np.concatenate([np.asarray(a, dtype=float).reshape(-1) for a in arrays])
    vals = vals[np.isfinite(vals) & (vals >= 0.0)]
    if vals.size == 0:
        return 1.0

    in_view = vals[vals <= hard_cap]
    if in_view.size == 0:
        return hard_cap

    robust = float(np.nanpercentile(in_view, 95) * 1.18)
    largest = float(np.nanmax(in_view) * 1.08)
    floor = 1.0 if col in {"sigma", "range", "nugget"} else 1e-6
    return min(max(robust, largest, floor), hard_cap)


def _plot_capped_mean_band(ax, x: np.ndarray, mean: np.ndarray, sd: np.ndarray, col: str, label: str) -> None:
    ymax = _readable_ymax(col, mean, sd)
    lower = np.clip(mean - sd, 0.0, ymax)
    upper = np.clip(mean + sd, 0.0, ymax)
    mean_plot = np.clip(mean, 0.0, ymax)
    clipped = np.isfinite(mean) & (mean > ymax)

    ax.plot(x, mean_plot, marker="o", ms=3.5, lw=1.4)
    ax.fill_between(x, lower, upper, alpha=0.18)
    if clipped.any():
        ax.scatter(x[clipped], np.full(int(clipped.sum()), ymax), marker="^", s=42, color="crimson", zorder=4)
        for xi, actual in zip(x[clipped], mean[clipped]):
            ax.annotate(
                f"{actual:.1f}",
                (xi, ymax),
                textcoords="offset points",
                xytext=(0, -13),
                ha="center",
                fontsize=7,
                color="crimson",
            )
        ax.text(
            0.99, 0.92, f"^ values exceed plotted cap ({ymax:.1f})",
            transform=ax.transAxes, ha="right", va="top", fontsize=8, color="crimson",
        )

    ax.set_ylim(0.0, ymax * 1.04)
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.25)


def save_daily_param_plot(by_day: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_specs = [
        ("sigma", "global sigma"),
        ("range", "global range geom mean"),
        ("range_lat", "global range lat"),
        ("range_lon", "global range lon"),
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
        _plot_capped_mean_band(ax, x, mean, sd, col, label)
    axes[-1].set_xlabel("day of month")
    fig.suptitle("Daily mean global spatial parameters, averaged over hourly fits (readable capped scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_hour_slot_param_plot(by_hour_slot: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_specs = [
        ("sigma", "global sigma"),
        ("range", "global range geom mean"),
        ("range_lat", "global range lat"),
        ("range_lon", "global range lon"),
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
        _plot_capped_mean_band(ax, x, mean, sd, col, label)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels)
    axes[-1].set_xlabel("hour slot within day / UTC hour")
    fig.suptitle("Mean global spatial parameters by hour slot, averaged over observed days")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_day_hour_heatmaps(global_with_time: pd.DataFrame, summary_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [
        ("sigma", "global sigma"),
        ("range", "global range geom mean"),
        ("range_lat", "global range lat"),
        ("range_lon", "global range lon"),
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


def nadir_distance_column(df: pd.DataFrame) -> tuple[str | None, str]:
    if "dist_to_nadir_km" in df.columns:
        return "dist_to_nadir_km", "distance to nadir (km)"
    if "dist_to_nadir_deg" in df.columns:
        return "dist_to_nadir_deg", "distance to nadir (degree-equivalent)"
    if "dist_to_nadir_coord" in df.columns:
        return "dist_to_nadir_coord", "distance to nadir (coordinate units)"
    return None, ""


def save_tile_nugget_vs_nadir_plot(
    tile_summary: pd.DataFrame,
    path: Path,
    title: str,
    mean_col: str,
    median_col: str,
) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dist_col, x_label = nadir_distance_column(tile_summary)
    cols = [dist_col, mean_col, median_col, "tile_x", "tile_y"]
    if dist_col is None or any(c not in tile_summary.columns for c in cols):
        return False

    d = tile_summary[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=[dist_col])
    d = d.sort_values(dist_col)
    if d.empty:
        return False

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    if mean_col in d.columns:
        ax.scatter(d[dist_col], d[mean_col], s=46, label="mean tile nugget", alpha=0.85)
        ax.plot(d[dist_col], d[mean_col], lw=1.0, alpha=0.35)
    if median_col in d.columns:
        ax.scatter(d[dist_col], d[median_col], s=42, marker="s", label="median tile nugget", alpha=0.85)
        ax.plot(d[dist_col], d[median_col], lw=1.0, alpha=0.35)

    for _, row in d.iterrows():
        ax.annotate(
            f"{int(row['tile_y'])},{int(row['tile_x'])}",
            (row[dist_col], row[median_col] if np.isfinite(row[median_col]) else row[mean_col]),
            textcoords="offset points",
            xytext=(3, 3),
            fontsize=6.5,
            alpha=0.75,
        )

    corr_bits = []
    for label, col in [("mean", mean_col), ("median", median_col)]:
        valid = d[[dist_col, col]].dropna()
        if len(valid) >= 3:
            corr_bits.append(f"{label} Spearman={valid[dist_col].corr(valid[col], method='spearman'):.3f}")
    subtitle = " | ".join(corr_bits)
    ax.set_title(title if not subtitle else f"{title}\n{subtitle}")
    ax.set_xlabel(x_label)
    ax.set_ylabel("tile nugget")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def make_summary_tile_table(tile_df: pd.DataFrame) -> pd.DataFrame:
    keys = ["tile_y", "tile_x", "tile_id"]
    d = tile_df.dropna(subset=["tile_nugget"]).copy()
    optional_mean_cols = [
        c for c in [
            "tile_sigmasq", "tile_sigma", "tile_range", "tile_range_lat", "tile_range_lon",
            "tile_phi1", "tile_phi2", "tile_phi3", "tile_loss", "n_vecchia",
            "global_sigmasq", "global_range", "global_range_lat", "global_range_lon",
            "global_nugget", "n_qc_removed", "n_qc_fit", "qc_max_abs_whitened",
        ]
        if c in d.columns
    ]
    if d.empty:
        return pd.DataFrame(columns=keys + [
            "n_days", "n_hours", "mean_tile_nugget", "median_tile_nugget",
            "sd_tile_nugget", "mean_ratio_to_global", "median_ratio_to_global",
            "mean_tile_n",
        ])

    if "tile_to_global_nugget" not in d.columns and "global_nugget" in d.columns:
        d["tile_to_global_nugget"] = d["tile_nugget"] / d["global_nugget"]
    if "tile_to_global_nugget" not in d.columns:
        d["tile_to_global_nugget"] = np.nan

    if "day_index" in d.columns:
        daily_aggs = {
            "daily_tile_nugget": ("tile_nugget", "mean"),
            "daily_ratio_to_global": ("tile_to_global_nugget", "mean"),
            "daily_tile_n": ("n", "mean"),
            "daily_n_hours": ("tile_nugget", "count"),
        }
        for col in optional_mean_cols:
            daily_aggs[f"daily_{col}"] = (col, "mean")
        daily = (
            d.groupby(keys + ["day_index"], as_index=False)
            .agg(**daily_aggs)
        )
        summary_aggs = {
            "n_days": ("daily_tile_nugget", "count"),
            "n_hours": ("daily_n_hours", "sum"),
            "mean_tile_nugget": ("daily_tile_nugget", "mean"),
            "median_tile_nugget": ("daily_tile_nugget", "median"),
            "sd_tile_nugget": ("daily_tile_nugget", "std"),
            "mean_ratio_to_global": ("daily_ratio_to_global", "mean"),
            "median_ratio_to_global": ("daily_ratio_to_global", "median"),
            "mean_tile_n": ("daily_tile_n", "mean"),
        }
        for col in optional_mean_cols:
            summary_aggs[f"mean_{col}"] = (f"daily_{col}", "mean")
        summary = (
            daily.groupby(keys, as_index=False)
            .agg(**summary_aggs)
            .sort_values(["tile_y", "tile_x"])
        )
        return merge_tile_geometry_summary(summary, d, keys)

    aggs = {
        "n_days": ("tile_nugget", "count"),
        "n_hours": ("tile_nugget", "count"),
        "mean_tile_nugget": ("tile_nugget", "mean"),
        "median_tile_nugget": ("tile_nugget", "median"),
        "sd_tile_nugget": ("tile_nugget", "std"),
        "mean_ratio_to_global": ("tile_to_global_nugget", "mean"),
        "median_ratio_to_global": ("tile_to_global_nugget", "median"),
        "mean_tile_n": ("n", "mean"),
    }
    for col in optional_mean_cols:
        aggs[f"mean_{col}"] = (col, "mean")
    summary = (
        d.groupby(keys, as_index=False)
        .agg(**aggs)
        .sort_values(["tile_y", "tile_x"])
    )
    return merge_tile_geometry_summary(summary, d, keys)


def summarize(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    global_df, tile_df = read_hourly_outputs(output_dir)

    global_all = summary_dir / "global_results_all.csv"
    tile_all   = summary_dir / "tile_results_all.csv"
    round_numeric(global_df, 4).to_csv(global_all, index=False, float_format="%.4f")
    round_numeric(tile_df, 4).to_csv(tile_all, index=False, float_format="%.4f")

    global_with_time, global_by_day, global_by_hour_slot = summarize_global_by_time(global_df)
    round_numeric(global_with_time, 4).to_csv(summary_dir / "global_results_all_with_time_features.csv", index=False, float_format="%.4f")
    global_nugget_by_hour_slot_path = summary_dir / "global_nugget_by_hour_slot.csv"
    global_by_day_path       = summary_dir / "global_params_by_day.csv"
    global_by_hour_slot_path = summary_dir / "global_params_by_hour_slot.csv"
    round_numeric(global_nugget_by_hour_slot(global_df), 4).to_csv(
        global_nugget_by_hour_slot_path, index=False, float_format="%.4f"
    )
    round_numeric(global_by_day, 4).to_csv(global_by_day_path, index=False, float_format="%.4f")
    round_numeric(global_by_hour_slot, 4).to_csv(global_by_hour_slot_path, index=False, float_format="%.4f")

    tile_summary = make_summary_tile_table(tile_df)
    tag = tile_tag(args)
    tile_y_count, tile_x_count = tile_shape(args)
    tile_summary_path = summary_dir / f"tile_nugget_summary_{tag}.csv"
    round_numeric(tile_summary, 4).to_csv(tile_summary_path, index=False, float_format="%.4f")
    nadir_cols = [
        c for c in [
            "tile_y", "tile_x", "tile_id", "tile_center_lat", "tile_center_lon",
            "tile_lat_min", "tile_lat_max", "tile_lon_min", "tile_lon_max",
            "dist_to_nadir_deg", "dist_to_nadir_km",
            "n_days", "n_hours", "mean_tile_nugget", "median_tile_nugget", "sd_tile_nugget",
        ]
        if c in tile_summary.columns
    ]
    if nadir_cols:
        round_numeric(tile_summary[nadir_cols], 4).to_csv(
            summary_dir / f"tile_nugget_vs_nadir_distance_{tag}.csv",
            index=False, float_format="%.4f",
        )

    save_tile_geographic_heatmap(
        tile_summary, "mean_tile_nugget", tile_y_count, tile_x_count,
        f"Mean tile nugget, {tag}", "mean tile nugget",
        summary_dir / f"tile_nugget_mean_heatmap_{tag}.png",
        args,
    )
    save_tile_geographic_heatmap(
        tile_summary, "median_tile_nugget", tile_y_count, tile_x_count,
        f"Median tile nugget, {tag}", "median tile nugget",
        summary_dir / f"tile_nugget_median_heatmap_{tag}.png",
        args,
    )
    save_tile_geographic_heatmap(
        tile_summary, "mean_ratio_to_global", tile_y_count, tile_x_count,
        f"Mean tile/global nugget ratio, {tag}", "mean tile/global nugget",
        summary_dir / f"tile_to_global_nugget_ratio_mean_heatmap_{tag}.png",
        args,
    )
    for col, label in [
        ("mean_tile_sigmasq", "mean tile sigmasq"),
        ("mean_tile_range_lat", "mean tile range lat"),
        ("mean_tile_range_lon", "mean tile range lon"),
        ("mean_tile_phi1", "mean tile phi1"),
        ("mean_tile_phi2", "mean tile phi2"),
        ("mean_tile_phi3", "mean tile phi3"),
        ("mean_global_sigmasq", "mean global sigmasq"),
        ("mean_global_range_lat", "mean global range lat"),
        ("mean_global_range_lon", "mean global range lon"),
        ("mean_global_nugget", "mean global nugget"),
        ("mean_n_qc_removed", "mean QC removed count"),
    ]:
        if col in tile_summary.columns:
            save_tile_geographic_heatmap(
                tile_summary, col, tile_y_count, tile_x_count,
                f"{label}, {tag}", label,
                summary_dir / f"{col}_heatmap_{tag}.png",
                args,
            )
    save_tile_nugget_vs_nadir_plot(
        tile_summary,
        summary_dir / f"tile_nugget_vs_nadir_distance_{tag}.png",
        f"{args.month} tile nugget vs GEMS nadir distance ({tag})",
        mean_col="mean_tile_nugget",
        median_col="median_tile_nugget",
    )
    save_global_timeseries(global_df, summary_dir / "global_params_timeseries.png")
    save_daily_param_plot(global_by_day, summary_dir / "global_params_daily_mean.png")
    save_hour_slot_param_plot(global_by_hour_slot, summary_dir / "global_params_hour_slot_mean.png")
    save_day_hour_heatmaps(global_with_time, summary_dir)
    save_tile_boxplot(tile_df, summary_dir / f"tile_nugget_boxplot_{tag}.png")

    print(f"Wrote {global_all}")
    print(f"Wrote {tile_all}")
    print(f"Wrote {global_nugget_by_hour_slot_path}")
    print(f"Wrote {global_by_day_path}")
    print(f"Wrote {global_by_hour_slot_path}")
    print(f"Wrote {tile_summary_path}")
    print(f"Wrote plots under {summary_dir}")


def limited_manifest_indices(manifest: pd.DataFrame, args: argparse.Namespace) -> list[int]:
    max_days = int(getattr(args, "max_days", 0) or 0)
    max_slots = int(getattr(args, "max_hour_slots_per_day", 0) or 0)
    if max_days <= 0 and max_slots <= 0:
        return list(range(len(manifest)))

    seen_days: list[int] = []
    per_day_counts: dict[int, int] = {}
    selected: list[int] = []
    for idx, row in manifest.iterrows():
        hour = pd.to_datetime(row["hour"], utc=True)
        day = int(row["day_index"]) if "day_index" in manifest.columns else int(hour.day)
        if day not in seen_days:
            if max_days > 0 and len(seen_days) >= max_days:
                continue
            seen_days.append(day)
            per_day_counts[day] = 0
        if max_slots > 0 and per_day_counts.get(day, 0) >= max_slots:
            continue
        selected.append(int(idx))
        per_day_counts[day] = per_day_counts.get(day, 0) + 1
    return selected


def run_all(args: argparse.Namespace) -> None:
    device = get_device(args)
    make_manifest(args)
    manifest = read_manifest(args)
    run_indices = limited_manifest_indices(manifest, args)
    if len(run_indices) != len(manifest):
        print(
            f"Restricted --mode all to {len(run_indices)}/{len(manifest)} hours "
            f"(max_days={args.max_days}, max_hour_slots_per_day={args.max_hour_slots_per_day}).",
            flush=True,
        )
    pickle_obj = load_single_pickle_dict(args)
    global_rows: list[dict] = []
    tile_frames: list[pd.DataFrame] = []
    running_cols = [
        "day", "hour_slot", "hour_utc", "hour_key",
        "n_raw", "n_used", "n_fit", "n_qc_removed",
        "lat_min", "lat_max", "lon_min", "lon_max",
        "sigmasq", "sigma", "range", "range_lat", "range_lon", "nugget", "loss",
    ]
    print("\nFITTED PARAMETERS EACH HOUR", flush=True)
    print(",".join(running_cols), flush=True)
    total = len(run_indices)
    for n_done, idx in enumerate(run_indices, start=1):
        global_row, tile_df = fit_hour_record(args, manifest, idx, device, pickle_obj=pickle_obj)
        global_rows.append(global_row)
        tile_frames.append(tile_df)
        print(csv_line_from_row(global_row, running_cols), flush=True)

        is_last = n_done == total
        if args.summary_every > 0 and (n_done % args.summary_every == 0 or is_last):
            global_running = pd.DataFrame(global_rows)
            tile_running   = pd.concat(tile_frames, ignore_index=True)
            print(
                "\n"
                + running_summary_text(
                    global_running, tile_running, args,
                    f"RUNNING SUMMARY after {n_done}/{total} fitted hours",
                )
                + "\n",
                flush=True,
            )

    global_df = pd.DataFrame(global_rows)
    tile_df   = pd.concat(tile_frames, ignore_index=True)
    write_compact_outputs(global_df, tile_df, args)


def main() -> None:
    args = normalize_tile_args(parse_args())
    smooth = float(args.smooth)
    if smooth <= 0.0:
        raise SystemExit(f"ERROR: --smooth must be positive, got {smooth}.")
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
