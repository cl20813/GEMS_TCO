#!/usr/bin/env python3
"""Eigenvalue-decomposition diagnostics for real July GEMS pure-space fits.

This is the real-data companion to
``Exercises/st_model/day/pure_space/simulation/eig_diag_sim_july_pure_space.py``.
It reads the original July ``tco_grid`` pickle, fits a pure-space isotropic
Matérn covariance model to each selected hour/spatial unit, and plots the
Figure 13.5-style cumulative sums of whitened covariance-eigenbasis scores.

The two supported reductions are:

  1. tiles4x4: split the observed domain into 16 coordinate tiles.
  2. sparse:   whole-domain x8 and x4 thinning, using grid row/column indices
               when the coordinates form an expanded grid.
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from GEMS_TCO import orderings
    from GEMS_TCO.kernels_space_multiscale_050826 import _build_matern_spline_coeffs
    from GEMS_TCO.kernels_space_iso_050826 import (
        HybridSpaceIsoTrendVecchiaFit as _HybridSpaceIsoTrendVecchiaFit,
        HybridSpaceIsoNoNuggetTrendVecchiaFit as _HybridSpaceIsoNoNuggetTrendVecchiaFit,
    )
except ImportError:
    for candidate in [Path(__file__).parents[5] / "src", Path("/home/jl2815/tco")]:
        if (candidate / "GEMS_TCO").is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            break
    from GEMS_TCO import orderings
    from GEMS_TCO.kernels_space_multiscale_050826 import _build_matern_spline_coeffs
    from GEMS_TCO.kernels_space_iso_050826 import (
        HybridSpaceIsoTrendVecchiaFit as _HybridSpaceIsoTrendVecchiaFit,
        HybridSpaceIsoNoNuggetTrendVecchiaFit as _HybridSpaceIsoNoNuggetTrendVecchiaFit,
    )


DTYPE = torch.float64
EPS = 1e-12
ROUND_DECIMALS = 6
BROWN_BRIDGE_Q95 = 1.3581015157406195


class MicroergodicIsoTrendVecchiaFit(_HybridSpaceIsoTrendVecchiaFit):
    def _raw_params(self, params: torch.Tensor):
        phi1 = torch.exp(params[0])
        phi2 = torch.exp(params[1])
        sigmasq = phi1 / phi2
        range_space = 1.0 / phi2
        nugget = torch.exp(params[2])
        return sigmasq, range_space, range_space, nugget

    def _convert_params(self, raw):
        phi1 = float(np.exp(raw[0]))
        phi2 = float(np.exp(raw[1]))
        return {"sigmasq": phi1 / phi2, "range": 1.0 / phi2, "nugget": float(np.exp(raw[2]))}


class MicroergodicIsoNoNuggetTrendVecchiaFit(_HybridSpaceIsoNoNuggetTrendVecchiaFit):
    def _raw_params(self, params: torch.Tensor):
        phi1 = torch.exp(params[0])
        phi2 = torch.exp(params[1])
        sigmasq = phi1 / phi2
        range_space = 1.0 / phi2
        nugget = params.new_tensor(0.0)
        return sigmasq, range_space, range_space, nugget

    def _convert_params(self, raw):
        phi1 = float(np.exp(raw[0]))
        phi2 = float(np.exp(raw[1]))
        return {"sigmasq": phi1 / phi2, "range": 1.0 / phi2, "nugget": 0.0}


VARIANTS = {
    "nugget0": {"class": MicroergodicIsoNoNuggetTrendVecchiaFit, "n_params": 2},
    "nugget_free": {"class": MicroergodicIsoTrendVecchiaFit, "n_params": 3},
}


@dataclass
class UnitSpec:
    family: str
    name: str
    label: str
    tensor: torch.Tensor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real-data July eigenvalue diagnostics after pure-space Vecchia fitting.")
    p.add_argument("--input", required=True, help="Real July tco_grid pickle, e.g. tco_grid_lat-3to7_lon111to131_24_07.pkl")
    p.add_argument("--output-root", required=True)
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--month", type=int, default=7)
    p.add_argument("--days", default="1,1", help="Inclusive day range or comma list, e.g. 1,31 or 1,3,7")
    p.add_argument("--hours", default="first", help="'first', 'all', hour range/list such as 0,23, or exact slot list")
    p.add_argument("--hour-match", default="slot", choices=["slot", "utc"],
                   help="For numeric --hours: slot is within-day observed slot; utc uses timestamp hour.")
    p.add_argument("--smooth", type=float, default=0.5)
    p.add_argument("--regions", default="tiles4x4,sparse")
    p.add_argument("--tile-y", type=int, default=4)
    p.add_argument("--tile-x", type=int, default=4)
    p.add_argument("--sparse-strides", default="8,4")
    p.add_argument("--variants", default="nugget0")
    p.add_argument("--neighbors", type=int, default=8)
    p.add_argument("--mean-design", default="base", choices=["base", "latlon", "hour_spatial"])
    p.add_argument("--x-col", default="Source_Longitude")
    p.add_argument("--y-col", default="Source_Latitude")
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
    p.add_argument("--nugget-init", type=float, default=1.0)
    p.add_argument("--min-points", type=int, default=80)
    p.add_argument("--max-points", type=int, default=0)
    p.add_argument("--sample-seed", type=int, default=202407)
    p.add_argument("--cov-jitter", type=float, default=1e-8)
    p.add_argument("--eigenvalue-rtol", type=float, default=1e-10)
    p.add_argument("--eigenvalue-atol", type=float, default=1e-12)
    p.add_argument("--brown-bridge-q", type=float, default=BROWN_BRIDGE_Q95)
    p.add_argument("--save-curves", action="store_true")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


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
    for col in [args.x_col, args.y_col, args.value_col]:
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not found. Available columns: {list(df.columns)}")
    d = df[[args.x_col, args.y_col, args.value_col]].copy()
    d = d.replace([np.inf, -np.inf], np.nan)
    d[args.x_col] = pd.to_numeric(d[args.x_col], errors="coerce")
    d[args.y_col] = pd.to_numeric(d[args.y_col], errors="coerce")
    d[args.value_col] = pd.to_numeric(d[args.value_col], errors="coerce")
    d = d.dropna()
    d = d.groupby([args.x_col, args.y_col], as_index=False)[args.value_col].mean()
    lon = d[args.x_col].to_numpy(dtype=float)
    lat = d[args.y_col].to_numpy(dtype=float)
    if args.coords == "lonlat":
        x, y = lonlat_to_km(lon, lat)
    else:
        x, y = lon, lat
    data = np.zeros((len(d), 4), dtype=np.float64)
    data[:, 0] = y
    data[:, 1] = x
    data[:, 2] = d[args.value_col].to_numpy(dtype=float)
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
            units.append(UnitSpec("tiles4x4", name, label, tensor[idx].contiguous()))
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
    if "tiles4x4" in regions:
        units.extend(tile_units(tensor, args.tile_y, args.tile_x))
    if "sparse" in regions:
        for stride in parse_int_list_or_range(args.sparse_strides):
            units.append(grid_sparse_unit(tensor, stride))
    unknown = regions.difference({"tiles4x4", "sparse"})
    if unknown:
        raise ValueError(f"Unknown --regions entries: {sorted(unknown)}")
    return units


def make_hybrid_ordering(coords: np.ndarray, neighbors: int) -> tuple[np.ndarray, np.ndarray, list]:
    coords = np.ascontiguousarray(coords.astype(np.float64))
    order = orderings.maxmin_cpp(coords).astype(np.int64)
    ordered = np.ascontiguousarray(coords[order])
    nns_map = orderings.find_nns_l2(ordered, max_nn=int(neighbors))
    return order, ordered, nns_map


def build_model(variant: str, smooth: float, input_tensor: torch.Tensor, nns_map: list, args: argparse.Namespace):
    return VARIANTS[variant]["class"](
        smooth=float(smooth),
        input_map={"t0": input_tensor},
        nns_map=nns_map,
        limit_A=int(args.neighbors),
        target_chunk_size=int(args.target_chunk_size),
        mean_design=args.mean_design,
    )


def init_params(variant: str, args: argparse.Namespace, device: torch.device) -> list[torch.Tensor]:
    init_phi2 = 1.0 / max(float(args.range_init), EPS)
    init_phi1 = float(args.sigmasq_init) * init_phi2
    vals = [init_phi1, init_phi2]
    if VARIANTS[variant]["n_params"] == 3:
        vals.append(args.nugget_init)
    return [torch.tensor(math.log(float(v)), device=device, dtype=DTYPE, requires_grad=True) for v in vals]


def backmap(raw: list[float], variant: str) -> dict:
    phi1 = float(math.exp(raw[0]))
    phi2 = float(math.exp(raw[1]))
    return {
        "sigmasq": phi1 / phi2,
        "range": 1.0 / phi2,
        "phi1": phi1,
        "phi2": phi2,
        "nugget": float(math.exp(raw[2])) if VARIANTS[variant]["n_params"] == 3 else 0.0,
    }


def fit_variant(variant: str, smooth: float, input_tensor: torch.Tensor, coords: np.ndarray, args: argparse.Namespace, device: torch.device) -> tuple[dict, float]:
    order, _, nns_map = make_hybrid_ordering(coords, args.neighbors)
    ordered_tensor = input_tensor[order].contiguous()
    model = build_model(variant, smooth, ordered_tensor, nns_map, args)
    model.precompute_conditioning_sets()
    params = init_params(variant, args, device)
    opt = model.set_optimizer(params, lr=1.0, max_iter=args.lbfgs_eval, max_eval=args.lbfgs_eval, history_size=10)
    raw, _ = model.fit_vecc_lbfgs(params, opt, max_steps=args.lbfgs_steps, grad_tol=1e-5)
    est = backmap(raw, variant)
    loss = float(raw[-1])
    del model, params, opt
    return est, loss


def design_matrix_np(coords: np.ndarray, mean_design: str) -> np.ndarray:
    lat = coords[:, 0].astype(np.float64)
    lon = coords[:, 1].astype(np.float64)
    lat_c = lat - float(np.mean(lat))
    lon_c = lon - float(np.mean(lon))
    if mean_design == "base":
        return np.column_stack([np.ones(len(coords)), lat_c])
    if mean_design in ("latlon", "hour_spatial"):
        return np.column_stack([np.ones(len(coords)), lat_c, lon_c])
    raise ValueError(f"Unknown mean_design={mean_design!r}")


def fitted_covariance_torch(coords: np.ndarray, est: dict, smooth: float, eig_device: torch.device, jitter: float) -> torch.Tensor:
    nu = float(smooth)
    x = torch.as_tensor(coords, device=eig_device, dtype=DTYPE)
    scaled_dist = torch.cdist(x, x) / max(float(est["range"]), EPS)
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


def kolmogorov_bridge_sf(d: float, terms: int = 200) -> float:
    if not np.isfinite(d) or d <= 0:
        return 1.0
    total = 0.0
    for j in range(1, terms + 1):
        total += ((-1.0) ** (j - 1)) * math.exp(-2.0 * (j * d) ** 2)
    return float(min(max(2.0 * total, 0.0), 1.0))


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
        "brown_bridge_pvalue_approx": kolmogorov_bridge_sf(bridge_d),
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
        m = int(curve["index"].max())
        y_max = 1.03 * max(float(m), float(np.nanmax(curve["cumsum_y2"].to_numpy(dtype=float))))
        ax.set_visible(True)
        ax.scatter(curve["index"], curve["cumsum_y2"], s=5, facecolors="none", edgecolors="black", linewidths=0.5)
        ax.plot([0, m], [0, m], color="0.45", linewidth=1.0)
        ax.plot(curve["index"], curve["band_lower"], color="0.65", linestyle=(0, (4, 4)), linewidth=0.8)
        ax.plot(curve["index"], curve["band_upper"], color="0.65", linestyle=(0, (4, 4)), linewidth=0.8)
        ax.set_xlim(0, m)
        ax.set_ylim(0, y_max)
        ax.set_title(label, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.16)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_math_notes(out_root: Path) -> None:
    text = """Real-data eigenvalue diagnostic math notes
=========================================

This run uses the real July tco_grid pickle, not a simulated ozone field.

If Z ~ N(mu, Sigma) and Sigma = S Lambda S', then
    Y = Lambda^{-1/2} S' (Z - mu) ~ N(0, I).

For mu = M beta, this script residualizes first:
    R = I - M (M'M)^{-1} M'
and decomposes
    R Sigma_hat R = S_+ Lambda_+ S_+'.
The plotted scores are
    Y_hat = Lambda_+^{-1/2} S_+' R Z.

Eigenvalues are sorted largest to smallest before cumulating Y_hat_k^2.
Small index means broad high-variance spatial modes; large index means finer
spatial modes. Dashed lines are approximate Brownian-bridge bands and should
be read as diagnostics, not exact tests, because parameters are fitted.
"""
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "real_eigen_diagnostic_math_notes.txt").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if float(args.smooth) <= 0.0:
        raise SystemExit("--smooth must be positive.")
    variants = parse_name_list(args.variants)
    for variant in variants:
        if variant not in VARIANTS:
            raise ValueError(f"Unknown variant {variant!r}; use {sorted(VARIANTS)}")

    fit_device = select_device(args.device, args.cuda_fallback)
    eig_device = select_eig_device(args.eig_device, fit_device, args.cuda_fallback)
    out_root = Path(args.output_root) / f"nu{smooth_tag(args.smooth)}"
    out_root.mkdir(parents=True, exist_ok=True)
    write_math_notes(out_root)

    hours = load_ordered_hours(Path(args.input), args.year, args.month)
    selected = select_hours(hours, parse_int_list_or_range(args.days), parse_hours(args.hours), args.hour_match)
    if not selected:
        raise ValueError("No selected hours. Check --days, --hours, and --hour-match.")
    print(f"REAL DATA: input={args.input}", flush=True)
    print(f"fit_device={fit_device}, eig_device={eig_device}, selected_hours={len(selected)}, smooth={args.smooth}", flush=True)

    all_fit_rows = []
    all_summary_rows = []
    all_curve_rows = []

    for hour_i, (ts, key, df) in enumerate(selected):
        day_label = f"{args.year}{args.month:02d}{ts.day:02d}"
        hour_label = f"h{ts.hour:02d}{ts.minute:02d}"
        hour_dir = out_root / f"{day_label}_{hour_label}"
        hour_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[real {day_label} {hour_label}] key={key}, rows={len(df)}", flush=True)

        tensor_full = hour_tensor(df, args, fit_device)
        units = build_units(tensor_full, args)
        overview_curves: dict[tuple[str, str], list[tuple[str, pd.DataFrame]]] = {}

        for unit in units:
            unit_tensor = maybe_subsample_valid(unit.tensor, args.max_points, args.sample_seed + hour_i)
            n_valid = count_valid(unit_tensor)
            if n_valid < max(int(args.min_points), int(args.neighbors) + 2):
                print(f"  {unit.name}: too few valid points ({n_valid}), skip", flush=True)
                continue
            fit_tensor, fit_coords, z = valid_tensor_view(unit_tensor)
            print(f"  {unit.name}: n={n_valid}", flush=True)

            for variant in variants:
                png_path = hour_dir / f"{variant}_{unit.name}_eigdiag.png"
                curve_csv_path = hour_dir / f"{variant}_{unit.name}_curve.csv"
                if args.skip_existing and png_path.exists():
                    print(f"    {variant}: plot exists, skip", flush=True)
                    continue

                t0 = time.time()
                est, loss = fit_variant(variant, args.smooth, fit_tensor, fit_coords, args, fit_device)
                fit_seconds = time.time() - t0

                t1 = time.time()
                curve, summary = eigen_diagnostic(z, fit_coords, est, args.smooth, args.mean_design, args, eig_device)
                eig_seconds = time.time() - t1

                title = (
                    f"real {day_label} {hour_label}, {variant}, {unit.label}\n"
                    f"nu={args.smooth}, n={summary['n_obs']}, m={summary['n_eigen']}, "
                    f"D={summary['max_abs_bridge_scaled']:.3f}, p~{summary['brown_bridge_pvalue_approx']:.3g}"
                )
                plot_eigen_curve(curve, title, png_path)
                if args.save_curves:
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
                    "smooth": float(args.smooth),
                    "coords": args.coords,
                    "loss": loss,
                    "fit_seconds": fit_seconds,
                    "eig_seconds": eig_seconds,
                    **est,
                }
                all_fit_rows.append(row_base)
                all_summary_rows.append({**row_base, **summary})
                if args.save_curves:
                    all_curve_rows.append(curve.assign(**row_base))
                if unit.family == "tiles4x4":
                    overview_curves.setdefault((variant, unit.family), []).append((unit.name.replace("_of_", "\nof "), curve))
                print(
                    f"    {variant}: sigmasq={est['sigmasq']:.4g}, range={est['range']:.4g}, "
                    f"nugget={est['nugget']:.4g}, D={summary['max_abs_bridge_scaled']:.3f}, "
                    f"p~{summary['brown_bridge_pvalue_approx']:.3g}, saved {png_path.name}",
                    flush=True,
                )
                gc.collect()
                if fit_device.type == "cuda" or eig_device.type == "cuda":
                    torch.cuda.empty_cache()

        for (variant, family), curves in overview_curves.items():
            out_path = hour_dir / f"{variant}_{family}_overview.png"
            plot_tile_overview(curves, f"real {day_label} {hour_label}, {variant}, {family}", out_path)
            print(f"  saved overview {out_path.name}", flush=True)

        if all_fit_rows:
            pd.DataFrame(all_fit_rows).round(ROUND_DECIMALS).to_csv(
                out_root / "real_eigen_fit_rows.csv", index=False, float_format="%.6f"
            )
        if all_summary_rows:
            pd.DataFrame(all_summary_rows).round(ROUND_DECIMALS).to_csv(
                out_root / "real_eigen_diagnostic_summary.csv", index=False, float_format="%.6f"
            )
        if args.save_curves and all_curve_rows:
            pd.concat(all_curve_rows, ignore_index=True).round(ROUND_DECIMALS).to_csv(
                out_root / "real_eigen_diagnostic_curves.csv", index=False, float_format="%.6f"
            )


if __name__ == "__main__":
    main()
