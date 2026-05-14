#!/usr/bin/env python3
"""Daily spectral plots for generated July simulation assets.

This is a plot-focused simulation diagnostic. It fits each selected day/hour with
pure-space isotropic Hybrid Vecchia under two variants:

  1. nugget0:      estimate sigmasq and range, nugget fixed at zero
  2. nugget_free:  estimate sigmasq, range, and nugget

Then it creates one daily radial spectrum plot per smooth value. By default,
the output is PNG plots only; optional fit CSVs and partial-profile rows can be
enabled from the command line. The script is
intended for the gridded generated simulation pickle so the 2-D FFT/Nyquist
interpretation is coherent. The real-location pickle can be used for fitting,
but it is not recommended for these FFT spectrum plots.
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
    from GEMS_TCO.kernels_space_iso_050826 import (
        HybridSpaceIsoTrendVecchiaFit,
        HybridSpaceIsoNoNuggetTrendVecchiaFit,
    )
except ImportError:
    for candidate in [Path(__file__).parents[5] / "src", Path("/home/jl2815/tco")]:
        if (candidate / "GEMS_TCO").is_dir() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
            break
    from GEMS_TCO import orderings
    from GEMS_TCO.kernels_space_iso_050826 import (
        HybridSpaceIsoTrendVecchiaFit,
        HybridSpaceIsoNoNuggetTrendVecchiaFit,
    )

DTYPE = torch.float64
EPS = 1e-12
ROUND_DECIMALS = 4


@dataclass
class GridTemplate:
    coords: np.ndarray
    row: np.ndarray
    col: np.ndarray
    lat_vals: np.ndarray
    lon_vals: np.ndarray
    coord_to_idx: dict[tuple[float, float], int]


VARIANTS = {
    "nugget0": {
        "class": HybridSpaceIsoNoNuggetTrendVecchiaFit,
        "labels": ["sigmasq", "range"],
        "title": "full: nugget fixed 0",
    },
    "nugget_free": {
        "class": HybridSpaceIsoTrendVecchiaFit,
        "labels": ["sigmasq", "range", "nugget"],
        "title": "full: nugget free",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Daily simulation spectral plots from generated July gridded assets.")
    p.add_argument("--input", required=True, help="Generated simulation pickle, preferably *_gridded.pkl")
    p.add_argument("--output-root", required=True)
    p.add_argument("--year", type=int, default=2024)
    p.add_argument("--month", type=int, default=7)
    p.add_argument("--days", default="1,31", help="Inclusive day range or comma list, e.g. 1,31 or 1,3,7")
    p.add_argument("--smooths", default="0.3,0.5", help="Comma-separated smooth values, e.g. 0.3,0.5,1.5")
    p.add_argument("--strides", default="8,4,2,1")
    p.add_argument("--neighbors", type=int, default=8)
    p.add_argument("--mean-design", default="base", choices=["base", "latlon", "hour_spatial"],
                   help="For one-hour pure-space fits, base is effectively intercept + centered latitude.")
    p.add_argument("--x-col", default="Longitude")
    p.add_argument("--y-col", default="Latitude")
    p.add_argument("--value-col", default="ColumnAmountO3")
    p.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    p.add_argument("--cuda-fallback", default="error", choices=["error", "cpu"],
                   help="If CUDA initialization fails, either stop with a clear error or continue on CPU.")
    p.add_argument("--target-chunk-size", type=int, default=1024)
    p.add_argument("--lbfgs-steps", type=int, default=8)
    p.add_argument("--lbfgs-eval", type=int, default=20)
    p.add_argument("--profile-steps", type=int, default=8)
    p.add_argument("--radial-bins", type=int, default=70)
    p.add_argument("--radial-qmax", type=float, default=0.985)
    p.add_argument("--sigmasq-init", type=float, default=10.0)
    p.add_argument("--range-init", type=float, default=0.2)
    p.add_argument("--nugget-init", type=float, default=1.0)
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--include-partials", action="store_true",
                   help="Add sigma-only and range-only partial-profile rows anchored at nugget0.")
    p.add_argument("--save-fit-csv", action="store_true",
                   help="Also save per-day fit parameter CSVs. Default output is PNG only.")
    return p.parse_args()


def select_device(device_arg: str, fallback: str) -> torch.device:
    requested = str(device_arg).lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested not in ("auto", "cuda"):
        return torch.device(device_arg)
    try:
        ok = torch.cuda.is_available()
        if ok:
            torch.empty(1, device="cuda")
            print(
                f"CUDA ready: device_count={torch.cuda.device_count()}, "
                f"current_device={torch.cuda.current_device()}, "
                f"name={torch.cuda.get_device_name(torch.cuda.current_device())}",
                flush=True,
            )
            return torch.device("cuda")
    except Exception as exc:
        msg = (
            "CUDA initialization failed before fitting. This is usually a Slurm/node "
            "GPU environment issue rather than a model-code issue. Check "
            "CUDA_VISIBLE_DEVICES, nvidia-smi, and the assigned node."
        )
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


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_int_list_or_range(text: str) -> list[int]:
    vals = [int(x.strip()) for x in str(text).split(",") if x.strip()]
    if len(vals) == 2 and vals[1] >= vals[0]:
        return list(range(vals[0], vals[1] + 1))
    return vals


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


def build_grid_template(df: pd.DataFrame, y_col: str, x_col: str) -> GridTemplate:
    d = df[[y_col, x_col]].copy()
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d = d.dropna().drop_duplicates().sort_values([x_col, y_col], kind="mergesort").reset_index(drop=True)
    coords = d[[y_col, x_col]].to_numpy(dtype=float)
    lat_key = np.round(coords[:, 0], 10)
    lon_key = np.round(coords[:, 1], 10)
    lat_vals = np.sort(np.unique(lat_key))
    lon_vals = np.sort(np.unique(lon_key))
    lat_to_row = {float(v): i for i, v in enumerate(lat_vals)}
    lon_to_col = {float(v): i for i, v in enumerate(lon_vals)}
    row = np.asarray([lat_to_row[float(v)] for v in lat_key], dtype=np.int64)
    col = np.asarray([lon_to_col[float(v)] for v in lon_key], dtype=np.int64)
    coord_to_idx = {(float(la), float(lo)): i for i, (la, lo) in enumerate(zip(lat_key, lon_key))}
    return GridTemplate(coords=coords, row=row, col=col, lat_vals=lat_vals, lon_vals=lon_vals, coord_to_idx=coord_to_idx)


def hour_tensor(df: pd.DataFrame, tmpl: GridTemplate, y_col: str, x_col: str, value_col: str, device: torch.device) -> torch.Tensor:
    data = np.full((len(tmpl.coords), 4), np.nan, dtype=np.float64)
    data[:, 0:2] = tmpl.coords
    d = df[[y_col, x_col, value_col]].copy()
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    d[x_col] = pd.to_numeric(d[x_col], errors="coerce")
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=[y_col, x_col])
    d = d.groupby([y_col, x_col], as_index=False)[value_col].mean()
    for la, lo, val in d[[y_col, x_col, value_col]].itertuples(index=False, name=None):
        idx = tmpl.coord_to_idx.get((float(round(la, 10)), float(round(lo, 10))))
        if idx is not None and np.isfinite(val):
            data[idx, 2] = float(val)
    data[:, 3] = 0.0
    return torch.from_numpy(data).to(device=device, dtype=DTYPE)


def count_valid(tensor: torch.Tensor) -> int:
    return int(((~torch.isnan(tensor[:, 2])) & (~torch.isnan(tensor[:, 0])) & (~torch.isnan(tensor[:, 1]))).sum().item())


def thin_tensor(tensor: torch.Tensor, tmpl: GridTemplate, stride: int) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    keep = (tmpl.row % int(stride) == 0) & (tmpl.col % int(stride) == 0)
    idx = np.flatnonzero(keep).astype(np.int64)
    rows = tmpl.row[idx]
    cols = tmpl.col[idx]
    row_keep = np.sort(np.unique(rows))
    col_keep = np.sort(np.unique(cols))
    return tensor[idx].contiguous(), tmpl.coords[idx], idx, row_keep, col_keep


def make_hybrid_ordering(coords: np.ndarray, neighbors: int) -> tuple[np.ndarray, np.ndarray, list]:
    coords = np.ascontiguousarray(coords.astype(np.float64))
    order = orderings.maxmin_cpp(coords).astype(np.int64)
    ordered = np.ascontiguousarray(coords[order])
    nns_map = orderings.find_nns_l2(ordered, max_nn=int(neighbors))
    return order, ordered, nns_map


def build_model(variant: str, smooth: float, input_tensor: torch.Tensor, nns_map: list, args: argparse.Namespace):
    spec = VARIANTS[variant]
    return spec["class"](
        smooth=float(smooth), input_map={"t0": input_tensor}, nns_map=nns_map,
        limit_A=int(args.neighbors), target_chunk_size=int(args.target_chunk_size),
        mean_design=args.mean_design,
    )


def init_params(variant: str, args: argparse.Namespace, device: torch.device) -> list[torch.Tensor]:
    vals = [args.sigmasq_init, args.range_init]
    if variant == "nugget_free":
        vals.append(args.nugget_init)
    return [torch.tensor(math.log(float(v)), device=device, dtype=DTYPE, requires_grad=True) for v in vals]


def backmap(raw: list[float], variant: str) -> dict:
    labels = VARIANTS[variant]["labels"]
    out = {name: float(math.exp(raw[i])) for i, name in enumerate(labels)}
    out.setdefault("nugget", 0.0)
    return out


def fit_variant(variant: str, smooth: float, input_tensor: torch.Tensor, coords: np.ndarray, args: argparse.Namespace, device: torch.device) -> tuple[dict, float]:
    order, ordered_coords, nns_map = make_hybrid_ordering(coords, args.neighbors)
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


def log_tensor(x: float, like: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(math.log(max(float(x), EPS)), device=like.device, dtype=like.dtype)


def profile_param_tensor(profile_type: str, theta: torch.Tensor, anchor: dict) -> torch.Tensor:
    vals = {
        "sigmasq": log_tensor(anchor["sigmasq"], theta),
        "range": log_tensor(anchor["range"], theta),
    }
    vals["sigmasq" if profile_type == "sigma_only" else "range"] = theta.reshape(())
    return torch.stack([vals["sigmasq"], vals["range"]])


def fit_profile(profile_type: str, smooth: float, anchor: dict, input_tensor: torch.Tensor, coords: np.ndarray, args: argparse.Namespace, device: torch.device) -> tuple[dict, float]:
    order, ordered_coords, nns_map = make_hybrid_ordering(coords, args.neighbors)
    ordered_tensor = input_tensor[order].contiguous()
    model = build_model("nugget0", smooth, ordered_tensor, nns_map, args)
    model.precompute_conditioning_sets()
    free_name = "sigmasq" if profile_type == "sigma_only" else "range"
    theta = torch.tensor([math.log(max(float(anchor[free_name]), EPS))], device=device, dtype=DTYPE, requires_grad=True)
    opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=args.lbfgs_eval, max_eval=args.lbfgs_eval, history_size=10, line_search_fn="strong_wolfe")
    last_loss = None
    for _ in range(args.profile_steps):
        def closure():
            opt.zero_grad()
            loss = model.vecchia_batched_likelihood(profile_param_tensor(profile_type, theta, anchor))
            loss.backward()
            return loss
        last_loss = opt.step(closure)
    est = dict(anchor)
    est[free_name] = float(torch.exp(theta.detach()).cpu().item())
    est["nugget"] = 0.0
    loss = float(last_loss.detach().cpu().item()) if isinstance(last_loss, torch.Tensor) else np.nan
    del model, theta, opt
    return est, loss


def axis_step(axis_vals: np.ndarray) -> float:
    return float(np.median(np.diff(axis_vals))) if len(axis_vals) > 1 else 1.0


def frequency_grid(lat_axis: np.ndarray, lon_axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fy = np.fft.fftshift(np.fft.fftfreq(len(lat_axis), d=axis_step(lat_axis)))
    fx = np.fft.fftshift(np.fft.fftfreq(len(lon_axis), d=axis_step(lon_axis)))
    oy = 2.0 * np.pi * fy
    ox = 2.0 * np.pi * fx
    OX, OY = np.meshgrid(ox, oy)
    return np.sqrt(OX ** 2 + OY ** 2), OX ** 2 + OY ** 2


def trend_design(lat: np.ndarray, lon: np.ndarray, mean_design: str, lat_center: float, lon_center: float) -> np.ndarray:
    lat_c = lat - float(lat_center)
    lon_c = lon - float(lon_center)
    if mean_design == "base":
        return np.column_stack([np.ones(len(lat)), lat_c])
    return np.column_stack([np.ones(len(lat)), lat_c, lon_c])


def residual_grid(input_tensor: torch.Tensor, tmpl: GridTemplate, thin_idx: np.ndarray, row_keep: np.ndarray, col_keep: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = input_tensor.detach().cpu().numpy()
    lat = arr[:, 0]
    lon = arr[:, 1]
    y = arr[:, 2]
    valid = np.isfinite(y) & np.isfinite(lat) & np.isfinite(lon)
    lat_center = np.nanmean(lat[valid])
    lon_center = np.nanmean(lon[valid])
    X = trend_design(lat[valid], lon[valid], args.mean_design, lat_center, lon_center)
    beta, *_ = np.linalg.lstsq(X, y[valid], rcond=None)
    resid = y - trend_design(lat, lon, args.mean_design, lat_center, lon_center) @ beta

    row_pos = {int(r): i for i, r in enumerate(row_keep)}
    col_pos = {int(c): i for i, c in enumerate(col_keep)}
    rows_full = tmpl.row[thin_idx]
    cols_full = tmpl.col[thin_idx]
    grid = np.full((len(row_keep), len(col_keep)), np.nan, dtype=float)
    mask = np.zeros_like(grid)
    for i, ok in enumerate(valid):
        if not ok:
            continue
        rr = row_pos[int(rows_full[i])]
        cc = col_pos[int(cols_full[i])]
        grid[rr, cc] = resid[i]
        mask[rr, cc] = 1.0
    return grid, mask, tmpl.lat_vals[row_keep].astype(float), tmpl.lon_vals[col_keep].astype(float)


def masked_periodogram(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    obs = mask > 0
    z = np.zeros_like(grid, dtype=float)
    z[obs] = grid[obs] - float(np.nanmean(grid[obs]))
    win = np.outer(np.hanning(grid.shape[0]), np.hanning(grid.shape[1]))
    norm = np.sum((mask * win) ** 2)
    norm = norm if norm > EPS else 1.0
    return np.abs(np.fft.fftshift(np.fft.fft2(z * win))) ** 2 / norm


def radial_average(surface: np.ndarray, k_grid: np.ndarray, bins: np.ndarray, k_max: float) -> pd.DataFrame:
    vals = np.asarray(surface).ravel()
    kk = np.asarray(k_grid).ravel()
    good = np.isfinite(vals) & np.isfinite(kk) & (kk > 0) & (kk <= k_max)
    bin_idx = np.digitize(kk[good], bins) - 1
    rows = []
    for b in range(len(bins) - 1):
        m = bin_idx == b
        if np.any(m):
            rows.append({"k_bin": b, "k_mid": 0.5 * (bins[b] + bins[b + 1]), "spectrum": float(np.nanmean(vals[good][m]))})
    return pd.DataFrame(rows, columns=["k_bin", "k_mid", "spectrum"])


def matern_spectrum(sigmasq: float, range_: float, nugget: float, smooth: float, omega2: np.ndarray) -> np.ndarray:
    nu = float(smooth)
    alpha = 2.0 * nu / max(float(range_) ** 2, EPS)
    return float(sigmasq) * (alpha + omega2) ** (-(nu + 1.0)) + max(float(nugget), 0.0)


def scale_to_data(data: np.ndarray, theory: np.ndarray) -> float:
    good = np.isfinite(data) & np.isfinite(theory) & (data > 0) & (theory > 0)
    return float(np.nanmedian(data[good] / theory[good])) if good.any() else 1.0


def spectra_for_est(input_tensor, tmpl, thin_idx, row_keep, col_keep, est, smooth, args, full_k, full_omega2, bins, k_max):
    grid, mask, lat_axis, lon_axis = residual_grid(input_tensor, tmpl, thin_idx, row_keep, col_keep, args)
    data_p = masked_periodogram(grid, mask)
    k_data, _ = frequency_grid(lat_axis, lon_axis)
    data = radial_average(data_p, k_data, bins, k_max).rename(columns={"spectrum": "data"})
    theory = radial_average(matern_spectrum(est["sigmasq"], est["range"], est.get("nugget", 0.0), smooth, full_omega2), full_k, bins, k_max).rename(columns={"spectrum": "theory"})
    merged = theory.merge(data[["k_bin", "data"]], on="k_bin", how="left")
    s = scale_to_data(merged["data"].to_numpy(dtype=float), merged["theory"].to_numpy(dtype=float))
    merged["theory_scaled"] = merged["theory"] * s
    return merged, float(data["k_mid"].max()) if not data.empty else np.nan


def plot_day(day_label: str, smooth: float, rows: list[dict], out_path: Path, args: argparse.Namespace, k_plot_max: float) -> None:
    strides = [int(s) for s in parse_int_list_or_range(args.strides)]
    labels = [f"x{s}" for s in strides]
    row_specs = [
        ("nugget0", "full", "full: nugget fixed 0", "tab:red"),
        ("nugget_free", "full", "full: nugget free", "tab:red"),
    ]
    if args.include_partials:
        row_specs.extend([
            ("nugget0", "sigma_only", "partial: sigma only / nugget0", "tab:blue"),
            ("nugget0", "range_only", "partial: range only / nugget0", "tab:green"),
        ])
    positive = []
    for r in rows:
        d = r["spec"]
        positive.extend(d["data"].dropna().to_list())
        positive.extend(d["theory_scaled"].dropna().to_list())
    positive = np.asarray([x for x in positive if np.isfinite(x) and x > 0], dtype=float)
    ylim = (1e-1, 1e5) if positive.size == 0 else (10 ** np.floor(np.log10(positive.min())), 10 ** np.ceil(np.log10(positive.max())))

    fig, axes = plt.subplots(len(row_specs), len(labels), figsize=(4.3 * len(labels), 3.0 * len(row_specs)), sharey=True)
    for i, (variant, fit_type, title, color) in enumerate(row_specs):
        for j, stride in enumerate(strides):
            ax = axes[i, j]
            sub = [r for r in rows if r["variant"] == variant and r["fit_type"] == fit_type and r["stride"] == stride]
            if not sub:
                ax.set_visible(False)
                continue
            hour_curves = [r["spec"] for r in sub]
            data_stack = pd.concat([d[["k_bin", "k_mid", "data"]].assign(hour=h) for h, d in enumerate(hour_curves)], ignore_index=True)
            data_mean = data_stack.groupby(["k_bin"], as_index=False).agg(k_mid=("k_mid", "mean"), data=("data", "mean"))
            theory_mean = pd.concat([d[["k_bin", "k_mid", "theory_scaled"]] for d in hour_curves], ignore_index=True).groupby("k_bin", as_index=False).agg(k_mid=("k_mid", "mean"), theory_scaled=("theory_scaled", "mean"))
            for d in hour_curves:
                ax.plot(d["k_mid"], d["data"], color="0.75", alpha=0.20, linewidth=0.7)
            ax.plot(data_mean["k_mid"], data_mean["data"], color="black", linewidth=2.0, label="data residual spectrum")
            ax.plot(theory_mean["k_mid"], theory_mean["theory_scaled"], color=color, linestyle="--", linewidth=1.8, label="fitted theory")
            k_cut = np.nanmean([r["data_k_max"] for r in sub])
            if np.isfinite(k_cut):
                ax.axvline(k_cut, color="0.55", linestyle=":", linewidth=1.0)
            ax.set_yscale("log")
            ax.set_ylim(*ylim)
            ax.set_xlim(0, k_plot_max)
            ax.grid(alpha=0.22)
            ax.set_title(f"{title}, x{stride}  (data k <= {k_cut:.4f})")
            ax.set_xlabel("radial frequency on full-grid scale")
            if j == 0:
                ax.set_ylabel("spectrum")
            if i == 0 and j == 0:
                ax.legend(fontsize=8)
    title_tail = "full-fit and partial-profile spectra" if args.include_partials else "full-fit spectra"
    fig.suptitle(f"simulation {day_label}, smooth={smooth}: {title_tail}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = select_device(args.device, args.cuda_fallback)
    hours = load_ordered_hours(Path(args.input), args.year, args.month)
    tmpl = build_grid_template(hours[0][2], args.y_col, args.x_col)
    full_k, full_omega2 = frequency_grid(tmpl.lat_vals.astype(float), tmpl.lon_vals.astype(float))
    pos = full_k[np.isfinite(full_k) & (full_k > 0)]
    k_max = float(np.quantile(pos, args.radial_qmax))
    bins = np.linspace(0.0, k_max, int(args.radial_bins) + 1)
    days = set(parse_int_list_or_range(args.days))
    strides = parse_int_list_or_range(args.strides)
    smooths = parse_float_list(args.smooths)

    print(f"device={device}, grid={len(tmpl.lat_vals)}x{len(tmpl.lon_vals)}, hours={len(hours)}")
    for smooth in smooths:
        tag = smooth_tag(smooth)
        smooth_dir = Path(args.output_root) / f"nu{tag}"
        smooth_dir.mkdir(parents=True, exist_ok=True)
        for day in sorted(days):
            day_hours = [(ts, key, df) for ts, key, df in hours if ts.day == int(day)]
            if not day_hours:
                print(f"[nu={smooth}] day {day:02d}: no hours, skip")
                continue
            day_label = f"{args.year}{args.month:02d}{day:02d}"
            out_path = smooth_dir / f"sim_spectral_{day_label}_nu{tag}.png"
            csv_path = smooth_dir / f"sim_spectral_{day_label}_nu{tag}_fits.csv"
            if args.skip_existing and out_path.exists():
                print(f"[nu={smooth}] {day_label}: exists, skip")
                continue
            print(f"\n[nu={smooth}] {day_label}: {len(day_hours)} hours -> {out_path}")
            plot_rows = []
            fit_rows = []
            for ts, key, df in day_hours:
                tensor_full = hour_tensor(df, tmpl, args.y_col, args.x_col, args.value_col, device)
                for stride in strides:
                    tensor_thin, coords_thin, thin_idx, row_keep, col_keep = thin_tensor(tensor_full, tmpl, stride)
                    n_valid = count_valid(tensor_thin)
                    if n_valid < max(args.neighbors + 2, 10):
                        print(f"  hour={ts.hour:02d}, x{stride}: too few valid ({n_valid}), skip")
                        continue
                    fit_cache = {}
                    for variant in ["nugget0", "nugget_free"]:
                        t0 = time.time()
                        est, loss = fit_variant(variant, smooth, tensor_thin, coords_thin, args, device)
                        spec, data_k_max = spectra_for_est(tensor_thin, tmpl, thin_idx, row_keep, col_keep, est, smooth, args, full_k, full_omega2, bins, k_max)
                        plot_rows.append({"variant": variant, "fit_type": "full", "stride": stride, "hour": ts.hour, "spec": spec, "data_k_max": data_k_max})
                        fit_rows.append({"day": day_label, "hour": ts.hour, "hour_key": key, "stride": stride, "variant": variant, "fit_type": "full", "smooth": smooth, "n_valid": n_valid, "loss": loss, **est, "seconds": time.time() - t0})
                        fit_cache[variant] = est
                    anchor = fit_cache.get("nugget0")
                    if args.include_partials and anchor:
                        for profile_type in ["sigma_only", "range_only"]:
                            t0 = time.time()
                            est, loss = fit_profile(profile_type, smooth, anchor, tensor_thin, coords_thin, args, device)
                            spec, data_k_max = spectra_for_est(tensor_thin, tmpl, thin_idx, row_keep, col_keep, est, smooth, args, full_k, full_omega2, bins, k_max)
                            plot_rows.append({"variant": "nugget0", "fit_type": profile_type, "stride": stride, "hour": ts.hour, "spec": spec, "data_k_max": data_k_max})
                            fit_rows.append({"day": day_label, "hour": ts.hour, "hour_key": key, "stride": stride, "variant": "nugget0", "fit_type": profile_type, "smooth": smooth, "n_valid": n_valid, "loss": loss, **est, "seconds": time.time() - t0})
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
            if plot_rows:
                plot_day(day_label, smooth, plot_rows, out_path, args, k_max)
                print(f"  saved {out_path}")
                if args.save_fit_csv:
                    pd.DataFrame(fit_rows).round(ROUND_DECIMALS).to_csv(csv_path, index=False, float_format="%.4f")
                    print(f"  saved {csv_path}")


if __name__ == "__main__":
    main()
