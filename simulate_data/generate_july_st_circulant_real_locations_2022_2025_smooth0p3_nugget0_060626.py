#!/usr/bin/env python3
"""Reusable July space-time simulation assets on real GEMS locations.

Smoothness-recovery version: the data-generating process uses Matérn
smoothness nu=0.3 with nugget fixed exactly to 0.  The output root is separate
from the earlier smooth0p3 assets, whose default nugget was 1.0, so downstream
smooth-estimation exercises can test recovery without smooth-nugget
confounding.

Outputs are written year-by-year for 2022, 2023, 2024, and 2025.  For each
year this script creates two pickle files:

  1. real-location pickle
     Same dict/DataFrame shape as the original tco_grid_YY_07.pkl.  The
     simulated value is sampled at each row's Source_Latitude/Source_Longitude.

  2. gridded pickle
     Same base grid rows as the original file, but values are assigned to
     regular grid cells by a strict 1-to-1 nearest-cell griddification.

High-resolution simulation grid:
  dlat_hr = 0.044 / lat_factor_hr, default lat_factor_hr=100
  dlon_hr = 0.063 / lon_factor_hr, default lon_factor_hr=10

Griddification threshold:
  After mapping each real source location to its nearest regular grid cell, the
  assignment is accepted only if both coordinate differences are within half a
  native cell:

      abs(source_lat - grid_lat) <= 0.044 / 2
      abs(source_lon - grid_lon) <= 0.063 / 2

  This is an axis-wise half-cell rule, not a radial distance rule.  If multiple
  source locations choose the same grid cell, only the closest source is kept.

For memory reasons, the 3-D circulant embedding is generated in daily
8-hour blocks, not one 248-hour FFT.  With range_time around 2 hours, the
overnight cross-day correlation is negligible for the current daily GEMS use.

Current local July template hour counts:
  2022: 240 hours
  2023: 248 hours
  2024: 248 hours
  2025: 247 hours

The generator uses available ordered July keys up to max_hours and does not
invent missing keys.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.fft
from scipy.interpolate import CubicSpline
from scipy.special import gamma as scipy_gamma
from scipy.special import kv as scipy_kv
from scipy.spatial import cKDTree


DELTA_LAT_BASE = 0.044
DELTA_LON_BASE = 0.063
ROUND_DECIMALS = 4


TRUE_DEFAULTS = {
    "sigmasq": 10.0,
    "range_lat": 0.2,
    "range_lon": 0.3,
    "range_time": 2.0,
    "advec_lat": 0.08,
    "advec_lon": -0.2,
    "nugget": 0.0,
}

_MATERN_SPLINE_CACHE = {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_years(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_range(text: str) -> tuple[float, float]:
    a, b = [float(x.strip()) for x in str(text).split(",")]
    return min(a, b), max(a, b)


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


def ordered_template_keys(obj: dict, max_hours: int) -> list[str]:
    parsed = []
    for key in obj:
        ts = parse_gems_hour_key(key)
        if ts is not None:
            parsed.append((ts, key))
    keys = [key for _, key in sorted(parsed)] if parsed else list(obj)
    if max_hours > 0:
        keys = keys[:max_hours]
    return keys


def build_regular_grid(lat_range: tuple[float, float], lon_range: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    lats = np.arange(lat_range[0], lat_range[1] + 1e-4, DELTA_LAT_BASE, dtype=float)
    lons = np.arange(lon_range[0], lon_range[1] + 1e-4, DELTA_LON_BASE, dtype=float)
    return np.round(lats, 4), np.round(lons, 4)


def build_high_res_grid(
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    lat_factor_hr: int,
    lon_factor_hr: int,
    pad: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    dlat = DELTA_LAT_BASE / float(lat_factor_hr)
    dlon = DELTA_LON_BASE / float(lon_factor_hr)
    if dlat <= 0 or dlon <= 0:
        raise ValueError("High-res grid spacings must be positive.")
    lats = np.arange(lat_range[0] - pad, lat_range[1] + pad + 0.5 * dlat, dlat, dtype=float)
    lons = np.arange(lon_range[0] - pad, lon_range[1] + pad + 0.5 * dlon, dlon, dtype=float)
    return np.round(lats, 6), np.round(lons, 6), float(dlat), float(dlon)


def build_matern_spline_coeffs(smooth: float, n_points: int = 4000, r_max: float = 30.0) -> dict:
    key = (round(float(smooth), 8), int(n_points), float(r_max))
    if key in _MATERN_SPLINE_CACHE:
        return _MATERN_SPLINE_CACHE[key]
    nu = float(smooth)
    if nu <= 0.0:
        raise ValueError(f"smooth must be positive, got {smooth}")
    r_arr = np.linspace(0.0, float(r_max), int(n_points), dtype=np.float64)
    f_arr = np.empty_like(r_arr)
    f_arr[0] = 1.0
    z = np.sqrt(2.0 * nu) * r_arr[1:]
    f_arr[1:] = (2.0 ** (1.0 - nu) / scipy_gamma(nu)) * (z ** nu) * scipy_kv(nu, z)
    f_arr = np.nan_to_num(f_arr, nan=0.0, posinf=1.0, neginf=0.0)
    f_arr = np.clip(f_arr, 0.0, 1.0)
    cs = CubicSpline(r_arr, f_arr, bc_type="natural")
    coeffs = {
        "knots": r_arr,
        "a": cs.c[3].copy(),
        "b": cs.c[2].copy(),
        "c": cs.c[1].copy(),
        "d": cs.c[0].copy(),
        "r_max": float(r_max),
    }
    _MATERN_SPLINE_CACHE[key] = coeffs
    return coeffs


def matern_spline_corr(dist: torch.Tensor, smooth: float, n_points: int, r_max: float) -> torch.Tensor:
    coeffs = build_matern_spline_coeffs(float(smooth), int(n_points), float(r_max))
    device = dist.device
    dtype = dist.dtype
    knots = torch.as_tensor(coeffs["knots"], device=device, dtype=dtype)
    a = torch.as_tensor(coeffs["a"], device=device, dtype=dtype)
    b = torch.as_tensor(coeffs["b"], device=device, dtype=dtype)
    c = torch.as_tensor(coeffs["c"], device=device, dtype=dtype)
    d = torch.as_tensor(coeffs["d"], device=device, dtype=dtype)
    r_c = dist.clamp(0.0, float(coeffs["r_max"]))
    orig_shape = r_c.shape
    r_flat = r_c.reshape(-1)
    idx = torch.searchsorted(knots, r_flat, right=True) - 1
    idx = idx.clamp(0, knots.numel() - 2)
    dx = r_flat - knots[idx]
    vals = a[idx] + dx * (b[idx] + dx * (c[idx] + dx * d[idx]))
    return vals.reshape(orig_shape).clamp(0.0, 1.0)


def matern_corr(dist: torch.Tensor, smooth: float, spline_n_points: int = 4000, spline_r_max: float = 30.0) -> torch.Tensor:
    if smooth == 0.5:
        return torch.exp(-dist)
    if smooth == 1.5:
        return (1.0 + dist) * torch.exp(-dist)
    return matern_spline_corr(dist, float(smooth), int(spline_n_points), float(spline_r_max))


def generate_st_field_block(
    n_lat: int,
    n_lon: int,
    t_steps: int,
    dlat: float,
    dlon: float,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict]:
    cpu = torch.device("cpu")
    f32 = torch.float32
    px, py, pt = 2 * int(n_lat), 2 * int(n_lon), 2 * int(t_steps)

    lx = torch.arange(px, device=cpu, dtype=f32) * float(dlat)
    lx[px // 2 :] -= px * float(dlat)
    ly = torch.arange(py, device=cpu, dtype=f32) * float(dlon)
    ly[py // 2 :] -= py * float(dlon)
    lt = torch.arange(pt, device=cpu, dtype=f32)
    lt[pt // 2 :] -= pt

    lx_g, ly_g, lt_g = torch.meshgrid(lx, ly, lt, indexing="ij")
    u_lat = lx_g - float(args.advec_lat) * lt_g
    u_lon = ly_g - float(args.advec_lon) * lt_g
    dist = torch.sqrt(
        (u_lat / float(args.range_lat)).pow(2)
        + (u_lon / float(args.range_lon)).pow(2)
        + (lt_g / float(args.range_time)).pow(2)
        + 1e-8
    )
    cov = float(args.sigmasq) * matern_corr(
        dist,
        float(args.smooth),
        spline_n_points=int(args.spline_n_points),
        spline_r_max=float(args.spline_r_max),
    )

    spec = torch.fft.fftn(cov)
    min_real = float(spec.real.min().item())
    neg_count = int((spec.real < 0).sum().item())
    spec.real = torch.clamp(spec.real, min=0.0)
    noise = torch.fft.fftn(torch.randn(px, py, pt, device=cpu, dtype=f32))
    field = torch.fft.ifftn(torch.sqrt(spec.real) * noise).real[:n_lat, :n_lon, :t_steps]
    diag = {
        "embedding_px": px,
        "embedding_py": py,
        "embedding_pt": pt,
        "spectrum_min_real_before_clamp": min_real,
        "spectrum_negative_count_before_clamp": neg_count,
    }
    return field, diag


def nearest_hr_indices(
    lat: np.ndarray,
    lon: np.ndarray,
    lats_hr: np.ndarray,
    lons_hr: np.ndarray,
    dlat_hr: float,
    dlon_hr: float,
) -> tuple[np.ndarray, np.ndarray]:
    i = np.rint((lat - float(lats_hr[0])) / dlat_hr).astype(np.int64)
    j = np.rint((lon - float(lons_hr[0])) / dlon_hr).astype(np.int64)
    return np.clip(i, 0, len(lats_hr) - 1), np.clip(j, 0, len(lons_hr) - 1)


def simulated_values_at_sources(
    df: pd.DataFrame,
    field_t: torch.Tensor,
    lats_hr: np.ndarray,
    lons_hr: np.ndarray,
    dlat_hr: float,
    dlon_hr: float,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lat = pd.to_numeric(df["Source_Latitude"], errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(df["Source_Longitude"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(lat) & np.isfinite(lon)
    y = np.full(len(df), np.nan, dtype=float)
    if valid.any():
        ii, jj = nearest_hr_indices(lat[valid], lon[valid], lats_hr, lons_hr, dlat_hr, dlon_hr)
        latent = field_t.numpy()[ii, jj]
        mean = float(args.mean_intercept) + float(args.mean_lat_slope) * (lat[valid] - float(args.mean_lat_center))
        if float(args.nugget) > 0.0:
            eps = rng.normal(0.0, math.sqrt(float(args.nugget)), size=int(valid.sum()))
        else:
            eps = np.zeros(int(valid.sum()), dtype=float)
        y[valid] = mean + latent + eps
    return y, lat, lon, valid


def griddify_one_to_one(
    df: pd.DataFrame,
    y_source: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    valid_source: np.ndarray,
) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    out["ColumnAmountO3"] = np.nan

    grid_lat = pd.to_numeric(out["Latitude"], errors="coerce").to_numpy(dtype=float)
    grid_lon = pd.to_numeric(out["Longitude"], errors="coerce").to_numpy(dtype=float)
    grid_ok = np.isfinite(grid_lat) & np.isfinite(grid_lon)
    grid_coords = np.column_stack([grid_lat[grid_ok], grid_lon[grid_ok]])
    grid_rows = np.flatnonzero(grid_ok)

    src_rows = np.flatnonzero(valid_source & np.isfinite(y_source))
    diag = {
        "n_grid_rows": int(len(out)),
        "n_grid_finite": int(grid_ok.sum()),
        "n_source_valid": int(len(src_rows)),
        "n_assigned_before_threshold": 0,
        "n_assigned_after_threshold": 0,
        "n_rejected_threshold": 0,
        "n_collision_lost": 0,
        "threshold_lat": float(DELTA_LAT_BASE / 2.0),
        "threshold_lon": float(DELTA_LON_BASE / 2.0),
    }
    if len(src_rows) == 0 or len(grid_rows) == 0:
        return out, diag

    tree = cKDTree(grid_coords)
    dist, local_cell = tree.query(np.column_stack([src_lat[src_rows], src_lon[src_rows]]), k=1)
    best_src_for_local = np.full(len(grid_rows), -1, dtype=np.int64)
    best_dist = np.full(len(grid_rows), np.inf, dtype=float)
    for local_i, src_i, d in zip(local_cell, src_rows, dist):
        if d < best_dist[local_i]:
            if best_src_for_local[local_i] >= 0:
                diag["n_collision_lost"] += 1
            best_src_for_local[local_i] = src_i
            best_dist[local_i] = d
        else:
            diag["n_collision_lost"] += 1

    filled_local = np.flatnonzero(best_src_for_local >= 0)
    diag["n_assigned_before_threshold"] = int(len(filled_local))
    if len(filled_local) == 0:
        return out, diag

    src_win = best_src_for_local[filled_local]
    grid_win = grid_rows[filled_local]
    lat_diff = np.abs(src_lat[src_win] - grid_lat[grid_win])
    lon_diff = np.abs(src_lon[src_win] - grid_lon[grid_win])
    keep = (lat_diff <= DELTA_LAT_BASE / 2.0) & (lon_diff <= DELTA_LON_BASE / 2.0)
    diag["n_rejected_threshold"] = int((~keep).sum())
    diag["n_assigned_after_threshold"] = int(keep.sum())

    kept_src = src_win[keep]
    kept_grid = grid_win[keep]
    out.loc[out.index[kept_grid], "ColumnAmountO3"] = y_source[kept_src]
    out.loc[out.index[kept_grid], "Source_Latitude"] = src_lat[kept_src]
    out.loc[out.index[kept_grid], "Source_Longitude"] = src_lon[kept_src]
    return out, diag


def make_real_location_df(df: pd.DataFrame, y_source: np.ndarray, hour_index: int) -> pd.DataFrame:
    out = df.copy()
    out["ColumnAmountO3"] = y_source
    if "Hours_elapsed" in out.columns:
        out["Hours_elapsed"] = float(hour_index)
    return out


def round_df(df: pd.DataFrame, digits: int = ROUND_DECIMALS) -> pd.DataFrame:
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(digits)
    return out


def input_path_for_year(input_root: Path, year: int) -> Path:
    yy = str(year)[2:]
    return input_root / f"pickle_{year}" / f"tco_grid_{yy}_07.pkl"


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def process_year(year: int, args: argparse.Namespace, lats_hr: np.ndarray, lons_hr: np.ndarray, dlat_hr: float, dlon_hr: float) -> None:
    input_path = input_path_for_year(Path(args.input_root), year)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    template = pd.read_pickle(input_path)
    if not isinstance(template, dict):
        raise TypeError(f"{input_path} is not a dict pickle.")

    keys = ordered_template_keys(template, int(args.max_hours))
    if not keys:
        raise ValueError(f"No usable keys found in {input_path}.")

    out_year = Path(args.output_dir) / f"{year}_july_st_circulant"
    out_year.mkdir(parents=True, exist_ok=True)

    real_map = {}
    grid_map = {}
    manifest_rows = []
    grid_diag_rows = []
    embed_diag_rows = []

    n_days = int(math.ceil(len(keys) / int(args.hours_per_day)))
    print(f"\n===== {year}: {len(keys)} hours, {n_days} blocks -> {out_year} =====", flush=True)
    for day_i in range(n_days):
        block_keys = keys[day_i * int(args.hours_per_day):(day_i + 1) * int(args.hours_per_day)]
        if not block_keys:
            continue
        block_seed = int(args.seed + year * 1000 + day_i)
        set_seed(block_seed)
        field, embed_diag = generate_st_field_block(
            len(lats_hr), len(lons_hr), len(block_keys), dlat_hr, dlon_hr, args
        )
        embed_diag.update({"year": year, "block": day_i, "seed": block_seed, "n_hours": len(block_keys)})
        embed_diag_rows.append(embed_diag)

        for local_t, key in enumerate(block_keys):
            hour_i = day_i * int(args.hours_per_day) + local_t
            df = template[key]
            hour_seed = int(block_seed + 10_000 + local_t)
            rng = np.random.default_rng(hour_seed)
            y_source, src_lat, src_lon, valid_source = simulated_values_at_sources(
                df, field[:, :, local_t], lats_hr, lons_hr, dlat_hr, dlon_hr, rng, args
            )
            real_df = make_real_location_df(df, y_source, hour_i)
            grid_df, grid_diag = griddify_one_to_one(df, y_source, src_lat, src_lon, valid_source)
            if "Hours_elapsed" in grid_df.columns:
                grid_df["Hours_elapsed"] = float(hour_i)

            real_map[key] = real_df
            grid_map[key] = grid_df
            ts = parse_gems_hour_key(key)
            manifest_rows.append({
                "year": year,
                "hour_index": hour_i,
                "block": day_i,
                "local_t": local_t,
                "hour_key": key,
                "hour": ts.isoformat() if ts is not None else "",
                "seed_field_block": block_seed,
                "seed_nugget": hour_seed,
                "n_rows": int(len(df)),
                "n_source_valid": int(valid_source.sum()),
                "n_realloc_finite": int(np.isfinite(real_df["ColumnAmountO3"]).sum()),
                "n_grid_finite": int(np.isfinite(grid_df["ColumnAmountO3"]).sum()),
                "mean_realloc": float(np.nanmean(real_df["ColumnAmountO3"])),
                "sd_realloc": float(np.nanstd(real_df["ColumnAmountO3"])),
                "mean_grid": float(np.nanmean(grid_df["ColumnAmountO3"])),
                "sd_grid": float(np.nanstd(grid_df["ColumnAmountO3"])),
            })
            grid_diag.update({"year": year, "hour_index": hour_i, "hour_key": key})
            grid_diag_rows.append(grid_diag)

            if hour_i < 3 or (hour_i + 1) % int(args.hours_per_day) == 0:
                print(
                    f"{year} hour {hour_i + 1}/{len(keys)} {key}: "
                    f"real_n={manifest_rows[-1]['n_realloc_finite']:,}, "
                    f"grid_n={manifest_rows[-1]['n_grid_finite']:,}",
                    flush=True,
                )

        del field

    prefix = f"sim_july{year}_st_circulant"
    real_path = out_year / f"{prefix}_real_locations.pkl"
    grid_path = out_year / f"{prefix}_gridded.pkl"
    manifest_path = out_year / f"{prefix}_manifest.csv"
    grid_diag_path = out_year / f"{prefix}_griddification_diag.csv"
    embed_diag_path = out_year / f"{prefix}_embedding_diag.csv"
    truth_path = out_year / f"{prefix}_truth.json"

    pd.to_pickle(real_map, real_path)
    pd.to_pickle(grid_map, grid_path)
    round_df(pd.DataFrame(manifest_rows)).to_csv(manifest_path, index=False, float_format=f"%.{ROUND_DECIMALS}f")
    round_df(pd.DataFrame(grid_diag_rows)).to_csv(grid_diag_path, index=False, float_format=f"%.{ROUND_DECIMALS}f")
    round_df(pd.DataFrame(embed_diag_rows)).to_csv(embed_diag_path, index=False, float_format=f"%.{ROUND_DECIMALS}f")

    truth = {
        "year": year,
        "input_template": str(input_path),
        "n_hours": int(len(keys)),
        "hours_per_day": int(args.hours_per_day),
        "block_generation": "independent daily 8-hour 3D circulant embeddings",
        "smooth": float(args.smooth),
        "smooth_generation_method": "natural cubic spline Matérn correlation, nugget fixed 0",
        "spline_n_points": int(args.spline_n_points),
        "spline_r_max": float(args.spline_r_max),
        "sigmasq": float(args.sigmasq),
        "sigma": float(math.sqrt(float(args.sigmasq))),
        "range_lat": float(args.range_lat),
        "range_lon": float(args.range_lon),
        "range_time": float(args.range_time),
        "advec_lat": float(args.advec_lat),
        "advec_lon": float(args.advec_lon),
        "nugget": float(args.nugget),
        "mean_intercept": float(args.mean_intercept),
        "mean_lat_slope": float(args.mean_lat_slope),
        "mean_lat_center": float(args.mean_lat_center),
        "lat_range": list(parse_range(args.lat_range)),
        "lon_range": list(parse_range(args.lon_range)),
        "delta_lat_base": DELTA_LAT_BASE,
        "delta_lon_base": DELTA_LON_BASE,
        "lat_factor_hr": int(args.lat_factor_hr),
        "lon_factor_hr": int(args.lon_factor_hr),
        "delta_lat_hr": float(dlat_hr),
        "delta_lon_hr": float(dlon_hr),
        "hr_pad": float(args.hr_pad),
        "griddification_rule": "nearest regular grid cell, 1-to-1, then axis-wise half-cell threshold",
        "griddification_threshold_lat": float(DELTA_LAT_BASE / 2.0),
        "griddification_threshold_lon": float(DELTA_LON_BASE / 2.0),
    }
    write_json(truth_path, truth)

    print(f"Saved {year} real-location pickle: {real_path}", flush=True)
    print(f"Saved {year} gridded pickle:       {grid_path}", flush=True)
    print(f"Saved {year} manifest:             {manifest_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default="2022,2023,2024,2025")
    parser.add_argument("--input-root", type=Path, default=Path("/home/jl2815/tco/data"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/jl2815/tco/exercise_output/sim_data/july_st_circulant_realpattern_smooth0p3_nugget0"))
    parser.add_argument("--max-hours", type=int, default=248)
    parser.add_argument("--hours-per-day", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20240701)
    parser.add_argument("--smooth", type=float, default=0.3)
    parser.add_argument("--spline-n-points", type=int, default=4000)
    parser.add_argument("--spline-r-max", type=float, default=30.0)
    parser.add_argument("--sigmasq", type=float, default=TRUE_DEFAULTS["sigmasq"])
    parser.add_argument("--range-lat", type=float, default=TRUE_DEFAULTS["range_lat"])
    parser.add_argument("--range-lon", type=float, default=TRUE_DEFAULTS["range_lon"])
    parser.add_argument("--range-time", type=float, default=TRUE_DEFAULTS["range_time"])
    parser.add_argument("--advec-lat", type=float, default=TRUE_DEFAULTS["advec_lat"])
    parser.add_argument("--advec-lon", type=float, default=TRUE_DEFAULTS["advec_lon"])
    parser.add_argument("--nugget", type=float, default=TRUE_DEFAULTS["nugget"])
    parser.add_argument("--mean-intercept", type=float, default=260.0)
    parser.add_argument("--mean-lat-slope", type=float, default=1.0)
    parser.add_argument("--mean-lat-center", type=float, default=-0.5)
    parser.add_argument("--lat-range", default="-3,2")
    parser.add_argument("--lon-range", default="121,131")
    parser.add_argument("--lat-factor-hr", type=int, default=100)
    parser.add_argument("--lon-factor-hr", type=int, default=10)
    parser.add_argument("--hr-pad", type=float, default=0.1)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    years = parse_years(args.years)
    lat_range = parse_range(args.lat_range)
    lon_range = parse_range(args.lon_range)
    lats_hr, lons_hr, dlat_hr, dlon_hr = build_high_res_grid(
        lat_range, lon_range, int(args.lat_factor_hr), int(args.lon_factor_hr), float(args.hr_pad)
    )

    print("args:", vars(args), flush=True)
    print("truth:", {k: getattr(args, k) for k in [
        "smooth", "spline_n_points", "spline_r_max",
        "sigmasq", "range_lat", "range_lon", "range_time", "advec_lat", "advec_lon", "nugget"
    ]}, flush=True)
    print(
        f"High-res grid: lat x{args.lat_factor_hr}, lon x{args.lon_factor_hr}; "
        f"dlat={dlat_hr:.8f}, dlon={dlon_hr:.8f}; "
        f"n_lat={len(lats_hr):,}, n_lon={len(lons_hr):,}",
        flush=True,
    )
    print(
        "Griddification threshold: "
        f"lat <= {DELTA_LAT_BASE / 2.0:.6f}, lon <= {DELTA_LON_BASE / 2.0:.6f}",
        flush=True,
    )

    for year in years:
        process_year(year, args, lats_hr, lons_hr, dlat_hr, dlon_hr)

    print("\nAll requested years completed.", flush=True)


if __name__ == "__main__":
    main()
