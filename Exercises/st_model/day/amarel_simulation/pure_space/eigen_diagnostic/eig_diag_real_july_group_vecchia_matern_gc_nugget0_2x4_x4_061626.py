"""Pure-space eigen diagnostics comparing Matern and generalized Cauchy Vecchia fits.

This 2026-06-16 version is a no-QC, nugget-fixed-0 comparison script for real
July GEMS data.  It keeps the group/block Vecchia geometry from the earlier
2-stage QC eigen diagnostic, but removes the outlier-removal refit so the
covariance kernels are compared on the same observations.

Default comparison:

  2023: Matern smooth=0.3 vs GC alpha=0.75 beta=1
  2024: Matern smooth=0.3 vs GC alpha=0.8  beta=1
  2025: Matern smooth=0.3 vs GC alpha=0.75 beta=1

Diagnostics are written for 2x4 coordinate tiles and whole-domain x4 thinning.
Loss columns include both the raw Vecchia negative log-likelihood objective and
loss_per_obs = raw loss / number of valid observations.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from GEMS_TCO.kernels_space_base_engine_052126 import _build_matern_spline_coeffs
    from GEMS_TCO.kernels_space_iso_cluster_052426 import (
        ClusterSpaceIsoTrendVecchiaFit as _ClusterSpaceIsoTrendVecchiaFit,
        ClusterSpaceIsoNoNuggetTrendVecchiaFit as _ClusterSpaceIsoNoNuggetTrendVecchiaFit,
    )
    from GEMS_TCO.kernels_space_aniso_cauchy_cluster_060326 import (
        ClusterSpaceAnisoCauchyFixedBetaNoNuggetTrendVecchiaFit,
        cauchy_phi_init_from_natural,
    )
except ImportError:
    _here = Path(__file__).resolve()
    _candidates = [p / "src" for p in _here.parents] + [Path("/home/jl2815/tco")]
    for candidate in _candidates:
        if (candidate / "GEMS_TCO").is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            break
    from GEMS_TCO.kernels_space_base_engine_052126 import _build_matern_spline_coeffs
    from GEMS_TCO.kernels_space_iso_cluster_052426 import (
        ClusterSpaceIsoTrendVecchiaFit as _ClusterSpaceIsoTrendVecchiaFit,
        ClusterSpaceIsoNoNuggetTrendVecchiaFit as _ClusterSpaceIsoNoNuggetTrendVecchiaFit,
    )
    from GEMS_TCO.kernels_space_aniso_cauchy_cluster_060326 import (
        ClusterSpaceAnisoCauchyFixedBetaNoNuggetTrendVecchiaFit,
        cauchy_phi_init_from_natural,
    )


DTYPE = torch.float64
EPS = 1e-12
ROUND_DECIMALS = 6
BROWN_BRIDGE_Q95 = 1.3581015157406195


class AnisotropicClusterTrendVecchiaFit(_ClusterSpaceIsoTrendVecchiaFit):
    def _raw_params(self, params: torch.Tensor):
        sigmasq = torch.exp(params[0])
        range_lat = torch.exp(params[1])
        range_lon = torch.exp(params[2])
        nugget = torch.exp(params[3])
        return sigmasq, range_lat, range_lon, nugget

    def _cov_from_deltas(self, d_lat, d_lon, params: torch.Tensor):
        sigmasq, range_lat, range_lon, _ = self._raw_params(params)
        scaled = torch.sqrt(
            d_lat.new_tensor(1e-8)
            + (d_lat / range_lat).pow(2)
            + (d_lon / range_lon).pow(2)
        )
        return sigmasq * self._matern_corr(scaled)

    def _convert_params(self, raw):
        return {
            "sigmasq": float(np.exp(raw[0])),
            "range_lat": float(np.exp(raw[1])),
            "range_lon": float(np.exp(raw[2])),
            "nugget": float(np.exp(raw[3])),
        }


class AnisotropicClusterNoNuggetTrendVecchiaFit(_ClusterSpaceIsoNoNuggetTrendVecchiaFit):
    def _raw_params(self, params: torch.Tensor):
        sigmasq = torch.exp(params[0])
        range_lat = torch.exp(params[1])
        range_lon = torch.exp(params[2])
        nugget = params.new_tensor(0.0)
        return sigmasq, range_lat, range_lon, nugget

    def _cov_from_deltas(self, d_lat, d_lon, params: torch.Tensor):
        sigmasq, range_lat, range_lon, _ = self._raw_params(params)
        scaled = torch.sqrt(
            d_lat.new_tensor(1e-8)
            + (d_lat / range_lat).pow(2)
            + (d_lon / range_lon).pow(2)
        )
        return sigmasq * self._matern_corr(scaled)

    def _convert_params(self, raw):
        return {
            "sigmasq": float(np.exp(raw[0])),
            "range_lat": float(np.exp(raw[1])),
            "range_lon": float(np.exp(raw[2])),
            "nugget": 0.0,
        }


VARIANTS = {
    "matern_s03": {
        "class": AnisotropicClusterNoNuggetTrendVecchiaFit,
        "n_params": 3,
        "family": "matern",
        "smooth": 0.3,
        "label": "Matern s=0.3",
        "plot_color": "black",
    },
    "gc_a075_b1": {
        "class": ClusterSpaceAnisoCauchyFixedBetaNoNuggetTrendVecchiaFit,
        "n_params": 3,
        "family": "cauchy",
        "smooth": 0.5,
        "gc_alpha": 0.75,
        "gc_beta": 1.0,
        "label": "GC a=0.75 b=1",
        "plot_color": "#1f77b4",
    },
    "gc_a08_b1": {
        "class": ClusterSpaceAnisoCauchyFixedBetaNoNuggetTrendVecchiaFit,
        "n_params": 3,
        "family": "cauchy",
        "smooth": 0.5,
        "gc_alpha": 0.8,
        "gc_beta": 1.0,
        "label": "GC a=0.8 b=1",
        "plot_color": "#1f77b4",
    },
}

YEAR_VARIANTS = {
    2023: ["matern_s03", "gc_a075_b1"],
    2024: ["matern_s03", "gc_a08_b1"],
    2025: ["matern_s03", "gc_a075_b1"],
}


@dataclass
class UnitSpec:
    family: str
    name: str
    label: str
    tensor: torch.Tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="No-QC real-data July eigenvalue diagnostics comparing Matern and GC pure-space group Vecchia fits.")
    p.add_argument("--input", required=True, help="Real July tco_grid pickle, e.g. tco_grid_lat-3to7_lon111to131_24_07.pkl")
    p.add_argument("--output-root", required=True)
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--month", type=int, default=7)
    p.add_argument("--days", default="1,1", help="Inclusive day range or comma list, e.g. 1,31 or 1,3,7")
    p.add_argument("--hours", default="first", help="'first', 'all', hour range/list such as 0,23, or exact slot list")
    p.add_argument("--hour-match", default="slot", choices=["slot", "utc"],
                   help="For numeric --hours: slot is within-day observed slot; utc uses timestamp hour.")
    p.add_argument("--model-variants", default="auto", help="auto uses year-specific Matern-vs-GC pair; otherwise comma list.")
    p.add_argument("--regions", default="tiles,sparse")
    p.add_argument("--tile-y", type=int, default=2)
    p.add_argument("--tile-x", type=int, default=4)
    p.add_argument("--sparse-strides", default="4")
    p.add_argument("--cluster-neighbor-blocks", type=int, default=2,
                   help="Number of previous max-min cluster blocks used for conditioning.")
    p.add_argument("--cluster-block-shape", default="4x4",
                   help="Cluster target block shape in grid-cell units, e.g. 4x4.")
    p.add_argument("--min-target-points", type=int, default=1)
    p.add_argument("--mean-design", default="lat", choices=["lat", "base", "latlon", "hour_spatial"])
    p.add_argument("--x-col", default="auto")
    p.add_argument("--y-col", default="auto")
    p.add_argument("--value-col", default="ColumnAmountO3")
    p.add_argument("--coords", default="raw", choices=["raw", "lonlat"],
                   help="raw keeps lon/lat degrees; lonlat converts to local km before fitting/eigendecomp.")
    p.add_argument("--device", default="auto")
    p.add_argument("--eig-device", default="same", choices=["same", "auto", "cpu", "cuda"])
    p.add_argument("--cuda-fallback", default="error", choices=["error", "cpu"])
    p.add_argument("--target-chunk-size", type=int, default=1024)
    p.add_argument("--lbfgs-steps", type=int, default=8)
    p.add_argument("--lbfgs-eval", type=int, default=20)
    p.add_argument("--sigmasq-init", type=float, default=10.0)
    p.add_argument("--range-init", type=float, default=0.2)
    p.add_argument("--range-lat-init", type=float, default=None)
    p.add_argument("--range-lon-init", type=float, default=None)
    p.add_argument("--nugget-init", type=float, default=1.0)
    p.add_argument("--min-points", type=int, default=80)
    p.add_argument("--max-points", type=int, default=0)
    p.add_argument("--max-eig-points", type=int, default=0,
                   help="Optional deterministic cap for dense eigendecomposition per unit. 0 uses all selected points.")
    p.add_argument("--sample-seed", type=int, default=202407)
    p.add_argument("--cov-jitter", type=float, default=1e-8)
    p.add_argument(
        "--save-hourly-plots",
        action="store_true",
        help="Also save each hour/unit eigen diagnostic plot. By default only monthly-average outputs are plotted.",
    )
    p.add_argument(
        "--save-hourly-rows",
        action="store_true",
        help="Also save per-hour fit and eigen summary CSV rows. By default only monthly aggregated outputs are written.",
    )
    p.add_argument("--eigenvalue-rtol", type=float, default=1e-10)
    p.add_argument("--eigenvalue-atol", type=float, default=1e-12)
    p.add_argument("--brown-bridge-q", type=float, default=BROWN_BRIDGE_Q95)
    p.add_argument("--save-curves", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def parse_block_shape(text: str) -> tuple[int, int]:
    vals = [int(x.strip()) for x in str(text).lower().replace("x", ",").split(",") if x.strip()]
    if len(vals) != 2 or vals[0] <= 0 or vals[1] <= 0:
        raise ValueError(f"block shape must look like 4x4, got {text!r}")
    return vals[0], vals[1]


def resolve_col(df: pd.DataFrame, requested: str, candidates: list[str], role: str) -> str:
    if requested != "auto":
        if requested not in df.columns:
            raise ValueError(f"Column {requested!r} for {role} not found. Available columns: {list(df.columns)}")
        return requested
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not auto-detect {role} column. Tried {candidates}; available columns: {list(df.columns)}")


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
                f"CUDA ready: device_count={torch.cuda.device_count()}, "
                f"current_device={torch.cuda.current_device()}, "
                f"name={torch.cuda.get_device_name(torch.cuda.current_device())}",
                flush=True,
            )
            return torch.device("cuda")
    except Exception as exc:
        msg = "CUDA initialization failed before fitting."
        if fallback == "cpu":
            print(f"WARNING: {msg}\nCUDA exception: {exc}\nFalling back to CPU.", flush=True)
            return torch.device("cpu")
        raise RuntimeError(f"{msg}\nCUDA exception: {exc}") from exc
    if requested == "cuda":
        msg = "CUDA was requested, but torch.cuda.is_available() is false."
        if fallback == "cpu":
            print(f"WARNING: {msg} Falling back to CPU.", flush=True)
            return torch.device("cpu")
        raise RuntimeError(msg)
    print("CUDA not available; using CPU.", flush=True)
    return torch.device("cpu")


def select_eig_device(arg: str, fit_device: torch.device, fallback: str) -> torch.device:
    if arg == "same":
        return fit_device
    if arg == "cpu":
        return torch.device("cpu")
    return select_device("cuda" if arg == "cuda" else "auto", fallback)


def parse_name_list(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_int_list_or_range(text: str) -> list[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(vals) == 2 and vals[1] >= vals[0]:
        return list(range(vals[0], vals[1] + 1))
    return vals


def parse_hours(text: str) -> str | list[int]:
    text = str(text).strip().lower()
    if text in ("all", "first"):
        return text
    return parse_int_list_or_range(text)


def smooth_tag(smooth: float) -> str:
    return str(float(smooth)).rstrip("0").rstrip(".").replace(".", "p")


def parse_gems_hour_key(key: str) -> pd.Timestamp | None:
    pat = r"^y(?P<yy>\d{2})m(?P<mm>\d{2})day(?P<dd>\d{2})_hm(?P<hh>\d{2}):(?P<minute>\d{2})$"
    m = re.match(pat, str(key))
    if not m:
        return None
    parts = {k: int(v) for k, v in m.groupdict().items()}
    return pd.Timestamp(
        year=2000 + parts["yy"], month=parts["mm"], day=parts["dd"],
        hour=parts["hh"], minute=parts["minute"], tz="UTC",
    )


def load_ordered_hours(path: Path, year: int, month: int) -> list[tuple[pd.Timestamp, str, pd.DataFrame]]:
    obj = pd.read_pickle(path)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict pickle, got {type(obj)}")
    out = []
    for key, df in obj.items():
        ts = parse_gems_hour_key(str(key))
        if ts is None or not isinstance(df, pd.DataFrame):
            continue
        if ts.year == year and ts.month == month:
            out.append((ts, str(key), df.copy()))
    out.sort(key=lambda x: x[0])
    if not out:
        raise ValueError(f"No hours for {year}-{month:02d} in {path}")
    return out


def select_hours(
    hours: list[tuple[pd.Timestamp, str, pd.DataFrame]],
    days: list[int],
    hour_spec: str | list[int],
    hour_match: str,
) -> list[tuple[pd.Timestamp, str, pd.DataFrame]]:
    selected = []
    for day in sorted(set(int(d) for d in days)):
        day_hours = [h for h in hours if h[0].day == day]
        if not day_hours:
            continue
        if hour_spec == "first":
            selected.append(day_hours[0])
        elif hour_spec == "all":
            selected.extend(day_hours)
        elif hour_match == "utc":
            want = set(int(h) for h in hour_spec)
            selected.extend([h for h in day_hours if h[0].hour in want])
        else:
            want = set(int(h) for h in hour_spec)
            selected.extend([h for i, h in enumerate(day_hours) if i in want])
    return selected


def lonlat_to_km(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon0 = float(np.nanmedian(lon))
    lat0 = float(np.nanmedian(lat))
    x = (lon - lon0) * 111.320 * math.cos(math.radians(lat0))
    y = (lat - lat0) * 110.574
    return x, y


def hour_tensor(df: pd.DataFrame, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    x_col = resolve_col(df, args.x_col, ["Longitude", "Source_Longitude", "lon", "x"], "longitude/x")
    y_col = resolve_col(df, args.y_col, ["Latitude", "Source_Latitude", "lat", "y"], "latitude/y")
    value_col = resolve_col(df, args.value_col, [args.value_col, "ColumnAmountO3", "value", "tco"], "value")
    for col in [x_col, y_col, value_col]:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found. Available columns: {list(df.columns)}")
    d = df[[x_col, y_col, value_col]].copy()
    d = d.replace([np.inf, -np.inf], np.nan)
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna()
    d = d.groupby([x_col, y_col], as_index=False)[value_col].mean()
    lon = d[x_col].to_numpy(dtype=float)
    lat = d[y_col].to_numpy(dtype=float)
    if args.coords == "lonlat":
        x, y = lonlat_to_km(lon, lat)
    else:
        x, y = lon, lat
    data = np.zeros((len(d), 4), dtype=np.float64)
    data[:, 0] = y
    data[:, 1] = x
    data[:, 2] = d[value_col].to_numpy(dtype=float)
    data[:, 3] = 0.0
    return torch.from_numpy(data).to(device=device, dtype=DTYPE)


def count_valid(tensor: torch.Tensor) -> int:
    return int(((~torch.isnan(tensor[:, 2])) & (~torch.isnan(tensor[:, 0])) & (~torch.isnan(tensor[:, 1]))).sum().item())


def valid_tensor_view(tensor: torch.Tensor) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    valid = (~torch.isnan(tensor[:, 2])) & (~torch.isnan(tensor[:, 0])) & (~torch.isnan(tensor[:, 1]))
    t = tensor[valid].contiguous()
    arr = t.detach().cpu().numpy()
    return t, arr[:, 0:2].astype(np.float64), arr[:, 2].astype(np.float64)


def maybe_subsample_valid(tensor: torch.Tensor, max_points: int, seed: int) -> torch.Tensor:
    valid_tensor, _, _ = valid_tensor_view(tensor)
    n = int(valid_tensor.shape[0])
    if int(max_points) <= 0 or n <= int(max_points):
        return valid_tensor
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(n, size=int(max_points), replace=False))
    return valid_tensor[torch.as_tensor(idx, device=valid_tensor.device)].contiguous()


def tile_units(tensor: torch.Tensor, tile_y: int, tile_x: int) -> list[UnitSpec]:
    arr = tensor.detach().cpu().numpy()
    y = arr[:, 0]
    x = arr[:, 1]
    y_edges = np.linspace(float(np.nanmin(y)), float(np.nanmax(y)) + EPS, int(tile_y) + 1)
    x_edges = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)) + EPS, int(tile_x) + 1)
    y_idx = np.clip(np.searchsorted(y_edges, y, side="right") - 1, 0, int(tile_y) - 1)
    x_idx = np.clip(np.searchsorted(x_edges, x, side="right") - 1, 0, int(tile_x) - 1)
    units = []
    for iy in range(int(tile_y)):
        for ix in range(int(tile_x)):
            keep = (y_idx == iy) & (x_idx == ix)
            idx = np.flatnonzero(keep).astype(np.int64)
            name = f"tile_r{iy + 1}c{ix + 1}_of_{tile_y}x{tile_x}"
            label = f"tile row {iy + 1}/{tile_y}, col {ix + 1}/{tile_x}"
            units.append(UnitSpec(f"tiles{tile_y}x{tile_x}", name, label, tensor[idx].contiguous()))
    return units


def grid_sparse_unit(tensor: torch.Tensor, stride: int) -> UnitSpec:
    arr = tensor.detach().cpu().numpy()
    y_key = np.round(arr[:, 0], 10)
    x_key = np.round(arr[:, 1], 10)
    y_vals = np.sort(np.unique(y_key))
    x_vals = np.sort(np.unique(x_key))
    grid_size = len(y_vals) * len(x_vals)
    is_regular_enough = int(0.95 * len(arr)) <= grid_size <= int(1.05 * len(arr))
    if is_regular_enough:
        y_to_row = {float(v): i for i, v in enumerate(y_vals)}
        x_to_col = {float(v): i for i, v in enumerate(x_vals)}
        rows = np.asarray([y_to_row[float(v)] for v in y_key], dtype=np.int64)
        cols = np.asarray([x_to_col[float(v)] for v in x_key], dtype=np.int64)
        keep = (rows % int(stride) == 0) & (cols % int(stride) == 0)
        idx = np.flatnonzero(keep).astype(np.int64)
        label = f"whole domain x{int(stride)} grid stride"
    else:
        order = np.lexsort((y_key, x_key))
        idx = np.sort(order[:: int(stride) ** 2]).astype(np.int64)
        label = f"whole domain x{int(stride)} sorted thinning (~1/{int(stride) ** 2})"
    return UnitSpec("sparse", f"x{int(stride)}", label, tensor[idx].contiguous())


def build_units(tensor: torch.Tensor, args: argparse.Namespace) -> list[UnitSpec]:
    regions = set(parse_name_list(args.regions))
    units: list[UnitSpec] = []
    if "global" in regions:
        units.append(UnitSpec("global", "global", "whole expanded domain", tensor.contiguous()))
    if {"tiles", "tiles2x4", "tiles4x4"}.intersection(regions):
        units.extend(tile_units(tensor, args.tile_y, args.tile_x))
    if "sparse" in regions:
        for stride in parse_int_list_or_range(args.sparse_strides):
            units.append(grid_sparse_unit(tensor, stride))
    unknown = regions.difference({"global", "tiles", "tiles2x4", "tiles4x4", "sparse"})
    if unknown:
        raise ValueError(f"Unknown --regions entries: {sorted(unknown)}")
    return units


def build_model(variant: str, smooth: float, input_tensor: torch.Tensor, coords: np.ndarray, args: argparse.Namespace):
    info = VARIANTS[variant]
    kwargs = {
        "smooth": float(info.get("smooth", smooth)),
        "input_map": {"t0": input_tensor},
        "grid_coords": np.asarray(coords, dtype=np.float64),
        "block_shape": parse_block_shape(args.cluster_block_shape),
        "n_neighbor_blocks": int(args.cluster_neighbor_blocks),
        "target_chunk_size": int(args.target_chunk_size),
        "min_target_points": int(args.min_target_points),
        "mean_design": args.mean_design,
    }
    if info.get("family") == "cauchy":
        kwargs["gc_alpha"] = float(info["gc_alpha"])
        kwargs["gc_beta"] = float(info["gc_beta"])
    return info["class"](**kwargs)


def init_params(variant: str, args: argparse.Namespace, device: torch.device) -> list[torch.Tensor]:
    range_lat_init = float(args.range_lat_init) if args.range_lat_init is not None else float(args.range_init)
    range_lon_init = float(args.range_lon_init) if args.range_lon_init is not None else float(args.range_init)
    info = VARIANTS[variant]
    if info.get("family") == "cauchy":
        init = cauchy_phi_init_from_natural(
            sigmasq=float(args.sigmasq_init),
            range_lat=max(range_lat_init, EPS),
            range_lon=max(range_lon_init, EPS),
            gc_beta=float(info["gc_beta"]),
        )
        vals = [init["phi1"], init["phi2"], init["phi3"]]
    else:
        vals = [float(args.sigmasq_init), max(range_lat_init, EPS), max(range_lon_init, EPS)]
    if int(info["n_params"]) != len(vals):
        raise ValueError(f"Unexpected n_params={info['n_params']} for {variant}; init has {len(vals)} values")
    return [torch.tensor(math.log(float(v)), device=device, dtype=DTYPE, requires_grad=True) for v in vals]


def backmap(raw: list[float], variant: str) -> dict:
    info = VARIANTS[variant]
    if info.get("family") == "cauchy":
        phi1 = float(math.exp(raw[0]))
        phi2 = float(math.exp(raw[1]))
        phi3 = float(math.exp(raw[2]))
        range_lon = 1.0 / max(phi2, EPS)
        range_lat = 1.0 / max(phi2 * math.sqrt(max(phi3, EPS)), EPS)
        sigmasq = phi1 / max(phi2, EPS)
        out = {
            "sigmasq": sigmasq,
            "range_lat": range_lat,
            "range_lon": range_lon,
            "range": math.sqrt(max(range_lat * range_lon, EPS)),
            "nugget": 0.0,
            "phi1": phi1,
            "phi2": phi2,
            "phi3": phi3,
            "gc_alpha": float(info["gc_alpha"]),
            "gc_beta": float(info["gc_beta"]),
        }
    else:
        sigmasq = float(math.exp(raw[0]))
        range_lat = float(math.exp(raw[1]))
        range_lon = float(math.exp(raw[2]))
        out = {
            "sigmasq": sigmasq,
            "range_lat": range_lat,
            "range_lon": range_lon,
            "range": math.sqrt(max(range_lat * range_lon, EPS)),
            "nugget": 0.0,
            "gc_alpha": np.nan,
            "gc_beta": np.nan,
        }
    out["model_family"] = str(info["family"])
    out["model_label"] = str(info["label"])
    return out


def raw_to_float_list(raw) -> list[float]:
    vals = []
    for x in raw:
        if isinstance(x, torch.Tensor):
            vals.append(float(x.detach().cpu().item()))
        else:
            vals.append(float(x))
    return vals


def fit_variant_model(
    variant: str,
    smooth: float,
    input_tensor: torch.Tensor,
    coords: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
):
    model = build_model(variant, smooth, input_tensor.contiguous(), coords, args)
    model.precompute_conditioning_sets()
    params = init_params(variant, args, device)
    opt = model.set_optimizer(params, lr=1.0, max_iter=args.lbfgs_eval, max_eval=args.lbfgs_eval, history_size=10)
    raw, _ = model.fit_vecc_lbfgs(params, opt, max_steps=args.lbfgs_steps, grad_tol=1e-5)
    raw_vals = raw_to_float_list(raw)
    n_params = int(VARIANTS[variant]["n_params"])
    raw_params = raw_vals[:n_params]
    est = backmap(raw_params, variant)
    est["raw_params"] = raw_params
    loss = float(raw_vals[-1])
    cluster_summary = model.cluster_summary()
    del params, opt
    return {**est, **cluster_summary}, loss, model, raw_params


def fit_variant(variant: str, smooth: float, input_tensor: torch.Tensor, coords: np.ndarray, args: argparse.Namespace, device: torch.device) -> tuple[dict, float]:
    est, loss, model, _ = fit_variant_model(variant, smooth, input_tensor, coords, args, device)
    del model
    return est, loss


def design_matrix_np(coords: np.ndarray, mean_design: str) -> np.ndarray:
    lat = coords[:, 0].astype(np.float64)
    lon = coords[:, 1].astype(np.float64)
    lat_c = lat - float(np.mean(lat))
    lon_c = lon - float(np.mean(lon))
    if mean_design == "lat":
        return np.column_stack([np.ones(len(coords)), lat_c])
    if mean_design == "base":
        return np.column_stack([np.ones(len(coords)), lat_c])
    if mean_design in ("latlon", "hour_spatial"):
        return np.column_stack([np.ones(len(coords)), lat_c, lon_c])
    raise ValueError(f"Unknown mean_design={mean_design!r}")


def fitted_covariance_torch(coords: np.ndarray, est: dict, smooth: float, eig_device: torch.device, jitter: float) -> torch.Tensor:
    x = torch.as_tensor(coords, device=eig_device, dtype=DTYPE)
    diff = x.unsqueeze(1) - x.unsqueeze(0)
    range_lat = max(float(est.get("range_lat", est.get("range", 1.0))), EPS)
    range_lon = max(float(est.get("range_lon", est.get("range", 1.0))), EPS)
    scaled_dist = torch.sqrt(
        diff[..., 0].pow(2) / (range_lat ** 2)
        + diff[..., 1].pow(2) / (range_lon ** 2)
        + EPS
    )
    family = str(est.get("model_family", "matern"))
    if family == "cauchy":
        alpha = max(float(est.get("gc_alpha", 0.75)), EPS)
        beta = max(float(est.get("gc_beta", 1.0)), EPS)
        corr = torch.pow(1.0 + torch.pow(scaled_dist, alpha), -beta / alpha)
    else:
        nu = float(smooth)
        if nu == 0.5:
            corr = torch.exp(-scaled_dist)
        elif nu == 1.5:
            corr = (1.0 + scaled_dist) * torch.exp(-scaled_dist)
        else:
            r_np = scaled_dist.detach().cpu().numpy()
            coeffs = _build_matern_spline_coeffs(nu)
            r_clip = np.clip(r_np, 0.0, float(coeffs["r_max"]))
            knots = coeffs["knots"]
            idx = np.searchsorted(knots, r_clip.ravel(), side="right") - 1
            idx = np.clip(idx, 0, len(knots) - 2)
            dx = r_clip.ravel() - knots[idx]
            corr_np = (
                coeffs["a"][idx]
                + dx * (coeffs["b"][idx] + dx * (coeffs["c"][idx] + dx * coeffs["d"][idx]))
            ).reshape(r_np.shape)
            corr_np = np.nan_to_num(corr_np, nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0)
            corr = torch.as_tensor(corr_np, device=eig_device, dtype=DTYPE)
    cov = float(est["sigmasq"]) * corr
    cov.diagonal().add_(max(float(est.get("nugget", 0.0)), 0.0) + max(float(jitter), 0.0))
    return cov

def eigen_diagnostic(z: np.ndarray, coords: np.ndarray, est: dict, smooth: float, mean_design: str, args: argparse.Namespace, eig_device: torch.device) -> tuple[pd.DataFrame, dict]:
    n = int(len(z))
    m_np = design_matrix_np(coords, mean_design)
    p = int(np.linalg.matrix_rank(m_np))
    if p >= n:
        raise ValueError(f"Mean design rank {p} is not smaller than n={n}")

    z_t = torch.as_tensor(z, device=eig_device, dtype=DTYPE)
    m_t = torch.as_tensor(m_np, device=eig_device, dtype=DTYPE)
    q, _ = torch.linalg.qr(m_t, mode="reduced")
    rz = z_t - q @ (q.T @ z_t)

    sigma = fitted_covariance_torch(coords, est, smooth, eig_device, args.cov_jitter)
    a = sigma - q @ (q.T @ sigma)
    k_mat = a - (a @ q) @ q.T
    k_mat = 0.5 * (k_mat + k_mat.T)

    evals, evecs = torch.linalg.eigh(k_mat)
    max_eval = torch.clamp(evals.max(), min=torch.as_tensor(EPS, device=eig_device, dtype=DTYPE))
    threshold = max(float(args.eigenvalue_atol), float(args.eigenvalue_rtol) * float(max_eval.detach().cpu().item()))
    keep = evals > threshold
    evals = evals[keep]
    evecs = evecs[:, keep]
    if evals.numel() == 0:
        raise ValueError("No positive projected covariance eigenvalues survived the threshold.")

    order = torch.argsort(evals, descending=True)
    evals = evals[order]
    evecs = evecs[:, order]
    scores = (evecs.T @ rz) / torch.sqrt(evals)
    y2 = scores.pow(2)
    csum = torch.cumsum(y2, dim=0)

    y2_np = y2.detach().cpu().numpy()
    csum_np = csum.detach().cpu().numpy()
    evals_np = evals.detach().cpu().numpy()
    m = int(len(y2_np))
    index = np.arange(1, m + 1, dtype=np.int64)
    width = float(args.brown_bridge_q) * math.sqrt(2.0 * m)
    bridge_d = float(np.max(np.abs(csum_np - index)) / math.sqrt(2.0 * m))
    curve = pd.DataFrame({
        "index": index,
        "eigenvalue": evals_np,
        "y2": y2_np,
        "cumsum_y2": csum_np,
        "expected": index.astype(float),
        "band_lower": index.astype(float) - width,
        "band_upper": index.astype(float) + width,
    })
    summary = {
        "n_obs": n,
        "mean_rank": p,
        "n_eigen": m,
        "eigen_threshold": threshold,
        "min_kept_eigen": float(np.min(evals_np)),
        "max_kept_eigen": float(np.max(evals_np)),
        "sum_y2": float(csum_np[-1]),
        "mean_y2": float(np.mean(y2_np)),
        "max_abs_bridge_scaled": bridge_d,
        "brown_bridge_width": width,
    }
    del sigma, a, k_mat, evals, evecs, scores, y2, csum, q, rz, z_t, m_t
    return curve, summary


def plot_eigen_curve(curve: pd.DataFrame, title: str, out_path: Path) -> None:
    m = int(curve["index"].max())
    y_max = 1.03 * max(float(m), float(np.nanmax(curve["cumsum_y2"].to_numpy(dtype=float))))
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.scatter(curve["index"], curve["cumsum_y2"], s=11, facecolors="none", edgecolors="black", linewidths=0.8, alpha=0.85)
    ax.plot([0, m], [0, m], color="0.45", linewidth=1.5)
    ax.plot(curve["index"], curve["band_lower"], color="0.60", linestyle=(0, (4, 4)), linewidth=1.1)
    ax.plot(curve["index"], curve["band_upper"], color="0.60", linestyle=(0, (4, 4)), linewidth=1.1)
    ax.set_xlim(0, m)
    ax.set_ylim(0, y_max)
    ax.set_xlabel("index")
    ax.set_ylabel("cumulative sum")
    ax.set_title(title)
    ax.grid(alpha=0.20)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_tile_overview(curves: list[tuple[str, pd.DataFrame]], title: str, out_path: Path) -> None:
    if not curves:
        return
    n_panels = len(curves)
    ncol = int(math.ceil(math.sqrt(n_panels)))
    nrow = int(math.ceil(n_panels / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.0 * nrow), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    for ax, (label, curve) in zip(axes.ravel(), curves):
        panel_title = label
        d_label = None
        if "\nD=" in label:
            panel_title, d_value = label.rsplit("\nD=", 1)
            d_label = f"D={d_value}"
        m = int(curve["index"].max())
        y_max = 1.03 * max(float(m), float(np.nanmax(curve["cumsum_y2"].to_numpy(dtype=float))))
        ax.set_visible(True)
        ax.scatter(curve["index"], curve["cumsum_y2"], s=5, facecolors="none", edgecolors="black", linewidths=0.5)
        ax.plot([0, m], [0, m], color="0.45", linewidth=1.0)
        ax.plot(curve["index"], curve["band_lower"], color="0.65", linestyle=(0, (4, 4)), linewidth=0.8)
        ax.plot(curve["index"], curve["band_upper"], color="0.65", linestyle=(0, (4, 4)), linewidth=0.8)
        ax.set_xlim(0, m)
        ax.set_ylim(0, y_max)
        ax.set_title(panel_title, fontsize=8)
        if d_label is not None:
            ax.text(
                0.03,
                0.93,
                d_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7,
                color="0.15",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.0},
            )
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.16)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def resample_curve_for_average(curve: pd.DataFrame, n_grid: int = 200) -> pd.DataFrame:
    m = float(curve["index"].max())
    if not np.isfinite(m) or m <= 0:
        return pd.DataFrame()
    frac = curve["index"].to_numpy(dtype=float) / m
    scaled = curve["cumsum_y2"].to_numpy(dtype=float) / m
    grid = np.linspace(1.0 / n_grid, 1.0, n_grid)
    vals = np.interp(grid, frac, scaled)
    return pd.DataFrame({"frac_index": grid, "scaled_cumsum": vals, "n_eigen": m})


def plot_average_curve(avg: pd.DataFrame, title: str, out_path: Path) -> None:
    if avg.empty:
        return
    x = avg["frac_index"].to_numpy(dtype=float)
    y = avg["scaled_cumsum_mean"].to_numpy(dtype=float)
    sd = avg["scaled_cumsum_std"].fillna(0.0).to_numpy(dtype=float)
    m_bar = max(float(avg["n_eigen_mean"].mean()), 1.0)
    band = BROWN_BRIDGE_Q95 * math.sqrt(2.0 / m_bar)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    ax.plot(x, y, color="black", linewidth=1.6, label="monthly mean")
    ax.fill_between(x, y - sd, y + sd, color="0.2", alpha=0.12, linewidth=0)
    ax.plot([0, 1], [0, 1], color="0.45", linewidth=1.3)
    ax.plot(x, x - band, color="0.60", linestyle=(0, (4, 4)), linewidth=1.0)
    ax.plot(x, x + band, color="0.60", linestyle=(0, (4, 4)), linewidth=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(1.05, float(np.nanmax(y + sd)) * 1.03))
    ax.set_xlabel("eigenvalue rank fraction")
    ax.set_ylabel("cumulative sum / m")
    ax.set_title(title)
    ax.grid(alpha=0.20)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_tile_overview(avg_rows: pd.DataFrame, title: str, out_path: Path, tile_y: int, tile_x: int) -> None:
    if avg_rows.empty:
        return
    fig, axes = plt.subplots(tile_y, tile_x, figsize=(3.0 * tile_x, 2.8 * tile_y), squeeze=False)
    for iy in range(tile_y):
        for ix in range(tile_x):
            unit = f"tile_r{iy + 1}c{ix + 1}_of_{tile_y}x{tile_x}"
            ax = axes[iy, ix]
            d = avg_rows[avg_rows["unit"] == unit]
            if d.empty:
                ax.set_visible(False)
                continue
            x = d["frac_index"].to_numpy(dtype=float)
            y = d["scaled_cumsum_mean"].to_numpy(dtype=float)
            ax.plot(x, y, color="black", linewidth=1.1)
            ax.plot([0, 1], [0, 1], color="0.5", linewidth=0.9)
            ax.set_title(f"r{iy + 1} c{ix + 1}", fontsize=8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, max(1.05, float(np.nanmax(y)) * 1.03))
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.16)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)



def model_plot_label(variant: str, summary: pd.DataFrame | None = None) -> str:
    info = VARIANTS.get(str(variant), {})
    label = str(info.get("label", variant))
    if summary is not None and not summary.empty and "loss_per_obs" in summary.columns:
        vals = pd.to_numeric(summary.loc[summary["variant"] == variant, "loss_per_obs"], errors="coerce").dropna().to_numpy(float)
        if vals.size:
            label += f" loss/obs={float(np.mean(vals)):.5f}"
    return label


def plot_sparse_model_comparison(avg: pd.DataFrame, summary: pd.DataFrame, out_path: Path, year: int) -> None:
    sub = avg[avg["family"] == "sparse"].copy()
    if sub.empty:
        return
    for unit, g_unit in sub.groupby("unit", sort=True):
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        for variant, g in g_unit.groupby("variant", sort=True):
            g = g.sort_values("frac_index")
            color = VARIANTS.get(str(variant), {}).get("plot_color", None)
            label = model_plot_label(str(variant), summary)
            ax.plot(g["frac_index"], g["scaled_cumsum_mean"], color=color, linewidth=2.0, label=label)
            sd = g["scaled_cumsum_std"].fillna(0.0).to_numpy(float)
            x = g["frac_index"].to_numpy(float)
            y = g["scaled_cumsum_mean"].to_numpy(float)
            ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.10, linewidth=0)
        ax.plot([0, 1], [0, 1], color="0.50", linewidth=1.1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, max(1.05, float(np.nanmax(g_unit["scaled_cumsum_mean"].to_numpy(float))) * 1.05))
        ax.set_xlabel("eigenvalue rank fraction")
        ax.set_ylabel("cumulative sum / m")
        ax.set_title(f"Real July {year}: x4 sparse eigen diagnostic, model comparison")
        ax.grid(alpha=0.20)
        ax.legend(fontsize=8)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path.parent / f"model_comparison_sparse_{unit}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_tile_model_comparison(avg: pd.DataFrame, summary: pd.DataFrame, out_path: Path, year: int, tile_y: int, tile_x: int) -> None:
    tile_family = f"tiles{int(tile_y)}x{int(tile_x)}"
    sub = avg[avg["family"] == tile_family].copy()
    if sub.empty:
        return
    fig, axes = plt.subplots(tile_y, tile_x, figsize=(3.3 * tile_x, 3.0 * tile_y), squeeze=False)
    for iy in range(tile_y):
        for ix in range(tile_x):
            unit = f"tile_r{iy + 1}c{ix + 1}_of_{tile_y}x{tile_x}"
            ax = axes[iy, ix]
            d = sub[sub["unit"] == unit]
            if d.empty:
                ax.set_visible(False)
                continue
            for variant, g in d.groupby("variant", sort=True):
                g = g.sort_values("frac_index")
                color = VARIANTS.get(str(variant), {}).get("plot_color", None)
                ax.plot(g["frac_index"], g["scaled_cumsum_mean"], color=color, linewidth=1.5, label=model_plot_label(str(variant), summary))
            ax.plot([0, 1], [0, 1], color="0.55", linewidth=0.9)
            ax.set_title(f"r{iy + 1} c{ix + 1}", fontsize=8)
            ax.set_xlim(0, 1)
            ymax = max(1.05, float(np.nanmax(d["scaled_cumsum_mean"].to_numpy(float))) * 1.05)
            ax.set_ylim(0, ymax)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.16)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8)
    fig.suptitle(f"Real July {year}: {tile_y}x{tile_x} tile eigen diagnostic, Matern vs GC", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_model_comparison_outputs(avg_rows: list[pd.DataFrame], summary_rows: list[dict], out_root: Path, args: argparse.Namespace) -> None:
    if not avg_rows:
        return
    d = pd.concat(avg_rows, ignore_index=True)
    if d.empty:
        return
    keys = ["variant", "family", "unit", "frac_index"]
    avg = (
        d.groupby(keys, as_index=False)
        .agg(
            scaled_cumsum_mean=("scaled_cumsum", "mean"),
            scaled_cumsum_std=("scaled_cumsum", "std"),
            n_curves=("scaled_cumsum", "count"),
            n_eigen_mean=("n_eigen", "mean"),
        )
        .sort_values(keys)
    )
    summary = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()
    plot_sparse_model_comparison(avg, summary, out_root / "model_comparison_sparse_x4.png", int(args.year))
    plot_tile_model_comparison(avg, summary, out_root / f"model_comparison_tiles{int(args.tile_y)}x{int(args.tile_x)}.png", int(args.year), int(args.tile_y), int(args.tile_x))
def write_monthly_average_outputs(avg_rows: list[pd.DataFrame], out_root: Path, args: argparse.Namespace) -> None:
    if not avg_rows:
        return
    d = pd.concat(avg_rows, ignore_index=True)
    if d.empty:
        return
    keys = ["variant", "family", "unit", "frac_index"]
    avg = (
        d.groupby(keys, as_index=False)
        .agg(
            scaled_cumsum_mean=("scaled_cumsum", "mean"),
            scaled_cumsum_std=("scaled_cumsum", "std"),
            n_curves=("scaled_cumsum", "count"),
            n_eigen_mean=("n_eigen", "mean"),
        )
        .sort_values(keys)
    )
    avg.round(ROUND_DECIMALS).to_csv(out_root / "real_eigen_monthly_average_curves.csv", index=False, float_format="%.6f")
    for (variant, family, unit), g in avg.groupby(["variant", "family", "unit"], sort=True):
        if str(family).startswith("tiles"):
            continue
        plot_average_curve(
            g,
            f"monthly average eigen diagnostic: {model_plot_label(str(variant))}, {unit}",
            out_root / f"{variant}_{unit}_monthly_average_eigdiag.png",
        )
    tile_family = f"tiles{int(args.tile_y)}x{int(args.tile_x)}"
    tile_avg = avg[(avg["family"] == tile_family)]
    for variant, g in tile_avg.groupby("variant", sort=True):
        plot_monthly_tile_overview(
            g,
            f"monthly average eigen diagnostic: {model_plot_label(str(variant))}, {args.tile_y}x{args.tile_x} tiles",
            out_root / f"{variant}_{tile_family}_monthly_average_overview.png",
            int(args.tile_y),
            int(args.tile_x),
        )


def write_monthly_fit_summary(summary_rows: list[dict], out_root: Path) -> None:
    if not summary_rows:
        return
    d = pd.DataFrame(summary_rows)
    if d.empty:
        return
    keys = ["variant", "family", "unit"]
    agg: dict[str, tuple[str, str]] = {
        "n_hours": ("hour_key", "nunique"),
        "n_obs_mean": ("n_obs", "mean"),
        "n_obs_min": ("n_obs", "min"),
        "n_obs_max": ("n_obs", "max"),
        "n_eigen_mean": ("n_eigen", "mean"),
        "bridge_D_mean": ("max_abs_bridge_scaled", "mean"),
        "bridge_D_max": ("max_abs_bridge_scaled", "max"),
        "loss_mean": ("loss", "mean"),
        "loss_per_obs_mean": ("loss_per_obs", "mean"),
        "sigmasq_mean": ("sigmasq", "mean"),
        "nugget_mean": ("nugget", "mean"),
    }
    for name in ["range", "range_lat", "range_lon"]:
        if name in d.columns:
            agg[f"{name}_mean"] = (name, "mean")
    optional = {
        "n_qc_removed_sum": ("n_qc_removed", "sum"),
        "n_qc_removed_mean": ("n_qc_removed", "mean"),
        "n_qc_removed_max": ("n_qc_removed", "max"),
        "qc_refit_rate": ("qc_refit", "mean"),
        "qc_max_abs_whitened_mean": ("qc_max_abs_whitened", "mean"),
        "qc_max_abs_whitened_max": ("qc_max_abs_whitened", "max"),
    }
    for out_col, spec in optional.items():
        if spec[0] in d.columns:
            if spec[0] == "qc_refit":
                d[spec[0]] = d[spec[0]].astype(bool).astype(float)
            agg[out_col] = spec
    monthly = d.groupby(keys, as_index=False).agg(**agg).sort_values(keys)
    monthly.round(ROUND_DECIMALS).to_csv(
        out_root / "real_eigen_monthly_fit_summary.csv",
        index=False,
        float_format="%.6f",
    )


def write_math_notes(out_root: Path) -> None:
    text = """Real-data eigenvalue diagnostic math notes
=========================================

This run uses the real expanded-bound July tco_grid pickle, not a simulated ozone field.

Fitting uses pure-space anisotropic block/group Vecchia.  The requested default
geometry is 4x4 grid-cell target blocks, conditioning on the two previous
max-min cluster blocks.

If Z ~ N(mu, Sigma) and Sigma = S Lambda S', then
    Y = Lambda^{-1/2} S' (Z - mu) ~ N(0, I).

The Vecchia fitting likelihood profiles beta by GLS.  For the eigen diagnostic,
mu = M beta is handled through the same residual projection:
    R = I - M (M'M)^{-1} M'
and decomposes
    R Sigma_hat R = S_+ Lambda_+ S_+'.
The plotted scores are
    Y_hat = Lambda_+^{-1/2} S_+' R Z.

Eigenvalues are sorted largest to smallest before cumulating Y_hat_k^2:
    lambda_1 >= ... >= lambda_m.
Small index means large fitted-variance eigenmodes, not necessarily spatial
low-frequency modes. Dashed lines are diagnostic reference bands, not exact
tests, because parameters are fitted.
"""
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "real_eigen_diagnostic_math_notes.txt").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if str(args.model_variants).strip().lower() == "auto":
        variants = list(YEAR_VARIANTS.get(int(args.year), ["matern_s03"]))
    else:
        variants = parse_name_list(args.model_variants)
    for variant in variants:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown model variant {variant!r}; use {sorted(VARIANTS)}")

    fit_device = select_device(args.device, args.cuda_fallback)
    eig_device = select_eig_device(args.eig_device, fit_device, args.cuda_fallback)
    out_root = Path(args.output_root) / f"year{int(args.year)}"
    out_root.mkdir(parents=True, exist_ok=True)
    write_math_notes(out_root)

    hours = load_ordered_hours(Path(args.input), args.year, args.month)
    selected = select_hours(hours, parse_int_list_or_range(args.days), parse_hours(args.hours), args.hour_match)
    if not selected:
        raise ValueError("No selected hours. Check --days, --hours, and --hour-match.")
    print(f"REAL DATA: input={args.input}", flush=True)
    print(
        f"fit_device={fit_device}, eig_device={eig_device}, selected_hours={len(selected)}, "
        f"model_variants={variants}, nugget fixed 0, no QC",
        flush=True,
    )

    all_fit_rows = []
    all_summary_rows = []
    all_curve_rows = []
    all_avg_curve_rows = []

    for hour_i, (ts, key, df) in enumerate(selected):
        day_label = f"{args.year}{args.month:02d}{ts.day:02d}"
        hour_label = f"h{ts.hour:02d}{ts.minute:02d}"
        hour_dir = out_root / f"{day_label}_{hour_label}"
        if args.save_hourly_plots or args.save_curves:
            hour_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[real {day_label} {hour_label}] key={key}, rows={len(df)}", flush=True)

        tensor_full = hour_tensor(df, args, fit_device)
        units = build_units(tensor_full, args)
        overview_curves: dict[tuple[str, str], list[tuple[str, pd.DataFrame]]] = {}

        for unit in units:
            unit_tensor = maybe_subsample_valid(unit.tensor, args.max_points, args.sample_seed + hour_i)
            unit_tensor = maybe_subsample_valid(unit_tensor, args.max_eig_points, args.sample_seed + 100000 + hour_i)
            n_valid = count_valid(unit_tensor)
            if n_valid < int(args.min_points):
                print(f"  {unit.name}: too few valid points ({n_valid}), skip", flush=True)
                continue
            fit_tensor, fit_coords, z = valid_tensor_view(unit_tensor)
            print(f"  {unit.name}: n={n_valid}", flush=True)

            for variant in variants:
                t0 = time.time()
                coords_all = unit_tensor[:, 0:2].detach().cpu().numpy().astype(np.float64)
                variant_smooth = float(VARIANTS[variant].get("smooth", 0.3))
                est, loss = fit_variant(variant, variant_smooth, unit_tensor, coords_all, args, fit_device)
                qc_tensor = unit_tensor
                qc = {
                    "qc_whitened_threshold": 0.0,
                    "qc_max_abs_whitened": np.nan,
                    "n_qc_initial_fit": int(n_valid),
                    "n_qc_removed": 0,
                    "n_qc_fit": int(n_valid),
                    "qc_refit": False,
                }
                fit_seconds = time.time() - t0
                fit_tensor, fit_coords, z = valid_tensor_view(qc_tensor)
                n_after_qc = int(fit_tensor.shape[0])
                if n_after_qc < int(args.min_points):
                    print(
                        f"    {variant}: too few valid points ({n_after_qc}), skip eigen diagnostic",
                        flush=True,
                    )
                    continue

                t1 = time.time()
                curve, summary = eigen_diagnostic(z, fit_coords, est, variant_smooth, args.mean_design, args, eig_device)
                eig_seconds = time.time() - t1

                title = (
                    f"real {day_label} {hour_label}, {variant}, {unit.label}\n"
                    f"{VARIANTS[variant]['label']}, n={summary['n_obs']}, m={summary['n_eigen']}, "
                    f"D={summary['max_abs_bridge_scaled']:.3f}"
                )
                if args.save_hourly_plots:
                    png_path = hour_dir / f"{variant}_{unit.name}_eigdiag.png"
                    plot_eigen_curve(curve, title, png_path)
                if args.save_curves:
                    curve_csv_path = hour_dir / f"{variant}_{unit.name}_curve.csv"
                    curve.round(ROUND_DECIMALS).to_csv(curve_csv_path, index=False, float_format="%.6f")

                row_base = {
                    "data_type": "real",
                    "day": day_label,
                    "hour": ts.hour,
                    "minute": ts.minute,
                    "hour_key": key,
                    "family": unit.family,
                    "unit": unit.name,
                    "unit_label": unit.label,
                    "variant": variant,
                    "model_label": str(VARIANTS[variant]["label"]),
                    "model_family": str(VARIANTS[variant]["family"]),
                    "smooth": float(VARIANTS[variant].get("smooth", np.nan)),
                    "gc_alpha": float(VARIANTS[variant].get("gc_alpha", np.nan)),
                    "gc_beta": float(VARIANTS[variant].get("gc_beta", np.nan)),
                    "coords": args.coords,
                    "loss": loss,
                    "loss_per_obs": float(loss) / max(float(n_after_qc), 1.0),
                    "fit_seconds": fit_seconds,
                    "eig_seconds": eig_seconds,
                    **qc,
                    **est,
                }
                all_fit_rows.append(row_base)
                all_summary_rows.append({**row_base, **summary})
                if args.save_curves:
                    all_curve_rows.append(curve.assign(**row_base))
                avg_curve = resample_curve_for_average(curve).assign(
                    data_type="real",
                    day=day_label,
                    hour=ts.hour,
                    minute=ts.minute,
                    hour_key=key,
                    family=unit.family,
                    unit=unit.name,
                    variant=variant,
                    smooth=float(VARIANTS[variant].get("smooth", np.nan)),
                )
                all_avg_curve_rows.append(avg_curve)
                if args.save_hourly_plots and str(unit.family).startswith("tiles"):
                    tile_label = (
                        f"{unit.name.replace('_of_', '\nof ')}\n"
                        f"D={summary['max_abs_bridge_scaled']:.2f}"
                    )
                    overview_curves.setdefault((variant, unit.family), []).append((tile_label, curve))
                print(
                    f"    {variant}: sigmasq={est['sigmasq']:.4g}, "
                    f"range_lat={est.get('range_lat', np.nan):.4g}, range_lon={est.get('range_lon', np.nan):.4g}, "
                    f"nugget={est['nugget']:.4g}, loss/obs={float(loss) / max(float(n_after_qc), 1.0):.5f}, "
                    f"D={summary['max_abs_bridge_scaled']:.3f}",
                    flush=True,
                )
                gc.collect()
                if fit_device.type == "cuda" or eig_device.type == "cuda":
                    torch.cuda.empty_cache()

        if args.save_hourly_plots:
            for (variant, family), curves in overview_curves.items():
                out_path = hour_dir / f"{variant}_{family}_overview.png"
                plot_tile_overview(curves, f"real {day_label} {hour_label}, {variant}, {family}", out_path)
                print(f"  saved overview {out_path.name}", flush=True)

        if args.save_hourly_rows and all_fit_rows:
            pd.DataFrame(all_fit_rows).round(ROUND_DECIMALS).to_csv(
                out_root / "real_eigen_fit_rows.csv", index=False, float_format="%.6f"
            )
        if args.save_hourly_rows and all_summary_rows:
            pd.DataFrame(all_summary_rows).round(ROUND_DECIMALS).to_csv(
                out_root / "real_eigen_diagnostic_summary.csv", index=False, float_format="%.6f"
            )
        if args.save_curves and all_curve_rows:
            pd.concat(all_curve_rows, ignore_index=True).round(ROUND_DECIMALS).to_csv(
                out_root / "real_eigen_diagnostic_curves.csv", index=False, float_format="%.6f"
            )
        write_monthly_average_outputs(all_avg_curve_rows, out_root, args)
        write_monthly_fit_summary(all_summary_rows, out_root)
        write_model_comparison_outputs(all_avg_curve_rows, all_summary_rows, out_root, args)


if __name__ == "__main__":
    main()
