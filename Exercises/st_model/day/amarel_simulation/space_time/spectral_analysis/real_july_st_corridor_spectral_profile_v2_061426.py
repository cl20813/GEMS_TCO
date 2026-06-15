#!/usr/bin/env python3
"""
Space-time corridor Vecchia Matérn/Cauchy fit + multivariate spectral profile diagnostic.

This is the ST analogue of the pure-space spline-smooth spectral diagnostic:

  1. Fit the 4x4 corridor Vecchia model with lag pattern 6/4/3 and nugget fixed at 0.
  2. Keep the full spatial grid; no max-min prefix thinning is applied.
  3. Build an 8-variate Fourier vector J(omega) for the eight July daytime slots.
  4. Use the fitted covariance, the per-hour missing masks, and a no-taper
     window to compute finite-sample E[J(omega)J(omega)^*].
  5. Compare data I against finite-sample E[I], and finite-sample E[I]
     against a no-window continuous-like covariance spectrum.
  6. Whiten J(omega) with the fitted 8x8 finite-sample cross-periodogram
     matrix, profile the covariance scale, and pool by frequency direction.

The monthly plots are intentionally small in number: each year folder compares
Matérn smooth=0.3 nugget0 against that year's selected generalized Cauchy
nugget0 candidate across norm, latitude, longitude, and diagonal frequencies.

The diagnostic is intentionally no-taper: only the missing-data window enters
the expected periodogram.  This matches the Vecchia fitting target more closely
than a Hamming/Hann taper diagnostic.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


HERE = Path(__file__).resolve().parent
SPACE_TIME_DIR = HERE.parent
REAL_DATA_DIR = SPACE_TIME_DIR / "real_data"
DIAGNOSIS_DIR = SPACE_TIME_DIR / "vecchia_diagnosis"
for path in [HERE, SPACE_TIME_DIR, REAL_DATA_DIR, DIAGNOSIS_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

AMAREL_SRC = Path("/home/jl2815/tco")
LOCAL_SRC = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/src")
SRC = AMAREL_SRC if AMAREL_SRC.exists() else LOCAL_SRC
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from GEMS_TCO.debiased_whittle_1111 import debiased_whittle_likelihood as DWL
from GEMS_TCO.vecchia_realdata_corridor_width_4x4_lag643 import (
    BLOCK_SHAPE,
    LAG_COUNTS,
    REFERENCE_ADVEC_LON_ABS,
    SPEC_NAME as VECCHIA_SPEC_NAME,
    model_spec as corridor_width_643_spec,
)
from GEMS_TCO.vecchia_st_spline import (
    RealDataCorridorWidth4x4Lag643NoNuggetSplineFit,
    _build_matern_spline_coeffs,
)
from GEMS_TCO.vecchia_st_generalized_cauchy import (
    RealDataCorridorWidth4x4Lag643NoNuggetGeneralizedCauchyFit,
)

from fit_real_july2024_corridor_width_4x4_lag643_matern_cauchy_tradeoff_nugget0_prefix_061426 import (
    DEFAULT_REAL_INIT_PHYSICAL,
    DTYPE,
    P_LABELS,
    backmap_params,
    clean_json_value,
    count_valid,
    load_real_assets,
    make_params_list,
    parse_day_idxs,
    parse_int_tokens,
    parse_pair,
    physical_to_log_phi,
    resolve_device,
    save_rows,
)


ROUND_DECIMALS = 6
DELTA_LAT_BASE = 0.044
DELTA_LON_BASE = 0.063
T_STEPS = 8
FIT_CSV = "st_corridor_spectral_all_fits.csv"
PROFILE_CSV = "st_corridor_spectral_profiles.csv"
MONTHLY_SUMMARY_CSV = "st_corridor_spectral_monthly_summary.csv"
BAND_TABLE_CSV = "st_corridor_spectral_representative_frequency_band_table.csv"

VARIANT_SPECS: dict[str, dict[str, Any]] = {
    "matern_s03": {
        "family": "matern",
        "smooth": 0.3,
        "gc_alpha": np.nan,
        "gc_beta": np.nan,
        "label": "Matern s=0.3 nugget0",
    },
    "gc_a075_b1": {
        "family": "cauchy",
        "smooth": np.nan,
        "gc_alpha": 0.75,
        "gc_beta": 1.0,
        "label": "GC a=0.75 b=1 nugget0",
    },
    "gc_a075_b05": {
        "family": "cauchy",
        "smooth": np.nan,
        "gc_alpha": 0.75,
        "gc_beta": 0.5,
        "label": "GC a=0.75 b=0.5 nugget0",
    },
    "gc_a08_b1": {
        "family": "cauchy",
        "smooth": np.nan,
        "gc_alpha": 0.8,
        "gc_beta": 1.0,
        "label": "GC a=0.8 b=1 nugget0",
    },
    "gc_a08_b05": {
        "family": "cauchy",
        "smooth": np.nan,
        "gc_alpha": 0.8,
        "gc_beta": 0.5,
        "label": "GC a=0.8 b=0.5 nugget0",
    },
}

YEAR_VARIANT_DEFAULTS: dict[int, list[str]] = {
    2023: ["matern_s03", "gc_a075_b1", "gc_a075_b05"],
    2024: ["matern_s03", "gc_a08_b1", "gc_a08_b05"],
    2025: ["matern_s03", "gc_a075_b1", "gc_a075_b05", "gc_a08_b05"],
}

DEFAULT_MODEL_VARIANTS = list(dict.fromkeys(v for vals in YEAR_VARIANT_DEFAULTS.values() for v in vals))


def default_output_root() -> Path:
    if Path("/home/jl2815").exists():
        return Path("/home/jl2815/tco/exercise_output/summer/st_corridor_spectral_profile_2023_2025_matern_gc_nugget0_v2_061426")
    return Path("/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/st_corridor_spectral_profile_2023_2025_matern_gc_nugget0_v2_061426")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(clean_json_value(row), sort_keys=True) + "\n")


def no_taper(u, n1: int, n2: int):
    u1, _ = u
    return torch.ones((int(n1), int(n2)), dtype=torch.float64, device=u1.device)


def raw_params_from_est(est: dict[str, float]) -> torch.Tensor:
    raw = physical_to_log_phi({k: float(est[k]) for k in P_LABELS}, nugget_mode="zero")
    return torch.tensor(raw, dtype=torch.float64)


def variant_spec(name: str) -> dict[str, Any]:
    if name not in VARIANT_SPECS:
        raise ValueError(f"Unknown model variant {name!r}. Known: {sorted(VARIANT_SPECS)}")
    return {**VARIANT_SPECS[name], "model_variant": name}


def variants_for_year(year: int, requested_variants: list[str]) -> list[str]:
    allowed = set(YEAR_VARIANT_DEFAULTS.get(int(year), requested_variants))
    return [name for name in requested_variants if name in allowed]


def cauchy_corr_torch(scaled_distance: torch.Tensor, gc_alpha: float, gc_beta: float) -> torch.Tensor:
    alpha = scaled_distance.new_tensor(float(gc_alpha))
    beta = scaled_distance.new_tensor(float(gc_beta))
    scaled = scaled_distance.clamp_min(1e-10)
    return torch.pow(1.0 + torch.pow(scaled, alpha), -beta / alpha)


def matern_corr_spline_torch(r: torch.Tensor, smooth: float, n_points: int, r_max: float) -> torch.Tensor:
    smooth = float(smooth)
    if np.isclose(smooth, 0.5):
        return torch.exp(-r)
    coeffs = _build_matern_spline_coeffs(smooth, n_points=int(n_points), r_max=float(r_max))
    knots = torch.as_tensor(coeffs["knots"], dtype=torch.float64, device=r.device)
    a = torch.as_tensor(coeffs["a"], dtype=torch.float64, device=r.device)
    b = torch.as_tensor(coeffs["b"], dtype=torch.float64, device=r.device)
    c = torch.as_tensor(coeffs["c"], dtype=torch.float64, device=r.device)
    d = torch.as_tensor(coeffs["d"], dtype=torch.float64, device=r.device)
    r_c = r.clamp(0.0, float(coeffs["r_max"]))
    orig = r_c.shape
    flat = r_c.reshape(-1)
    idx = torch.searchsorted(knots, flat, right=True) - 1
    idx = idx.clamp(0, knots.numel() - 2)
    dx = flat - knots[idx]
    vals = a[idx] + dx * (b[idx] + dx * (c[idx] + dx * d[idx]))
    return vals.reshape(orig).clamp_min(0.0)


def cov_x_st_raw(
    u_lat: torch.Tensor,
    u_lon: torch.Tensor,
    t_lag: torch.Tensor,
    params: torch.Tensor,
    spec: dict[str, Any],
    spline_n_points: int,
    spline_r_max: float,
) -> torch.Tensor:
    phi1, phi2, phi3, phi4 = torch.exp(params[0:4])
    advec_lat = params[4]
    advec_lon = params[5]
    nugget = torch.exp(params[6]) if params.numel() > 6 else params.new_tensor(0.0)
    sigmasq = phi1 / phi2

    u_lat_adv = u_lat - advec_lat * t_lag
    u_lon_adv = u_lon - advec_lon * t_lag
    dist = torch.sqrt(
        u_lat.new_tensor(1e-12)
        + u_lat_adv.pow(2) * phi3
        + u_lon_adv.pow(2)
        + t_lag.pow(2) * phi4
    )
    scaled = dist * phi2
    family = str(spec["family"])
    if family == "matern":
        corr = matern_corr_spline_torch(
            scaled,
            smooth=float(spec["smooth"]),
            n_points=int(spline_n_points),
            r_max=float(spline_r_max),
        )
    elif family == "cauchy":
        corr = cauchy_corr_torch(
            scaled,
            gc_alpha=float(spec["gc_alpha"]),
            gc_beta=float(spec["gc_beta"]),
        )
    else:
        raise ValueError(f"Unknown model family {family!r}")
    cov = sigmasq * corr
    zero = (torch.abs(u_lat) < 1e-10) & (torch.abs(u_lon) < 1e-10) & (torch.abs(t_lag) < 1e-10)
    return torch.where(zero, cov + nugget, cov)


def expected_periodogram_raw_st(
    params: torch.Tensor,
    spec: dict[str, Any],
    n1: int,
    n2: int,
    p_time: int,
    taper_autocorr_grid: torch.Tensor,
    delta_lat: float,
    delta_lon: float,
    spline_n_points: int,
    spline_r_max: float,
) -> torch.Tensor:
    device = params.device
    u1_lags = torch.arange(n1, dtype=torch.float64, device=device)
    u2_lags = torch.arange(n2, dtype=torch.float64, device=device)
    u1_mesh, u2_mesh = torch.meshgrid(u1_lags, u2_lags, indexing="ij")
    t_lags = torch.arange(p_time, dtype=torch.float64, device=device)

    def cn_bar(u1, u2, t_diff, q_idx: int, r_idx: int):
        lag_lat = u1 * float(delta_lat)
        lag_lon = u2 * float(delta_lon)
        cov = cov_x_st_raw(
            lag_lat,
            lag_lon,
            t_diff,
            params,
            spec,
            spline_n_points=spline_n_points,
            spline_r_max=spline_r_max,
        )
        idx1 = torch.clamp((n1 - 1 + u1).long(), 0, 2 * n1 - 2)
        idx2 = torch.clamp((n2 - 1 + u2).long(), 0, 2 * n2 - 2)
        if taper_autocorr_grid.ndim == 4:
            tap = taper_autocorr_grid[q_idx, r_idx, idx1, idx2]
        else:
            tap = taper_autocorr_grid[idx1, idx2]
        return cov * tap

    rows = []
    for q in range(p_time):
        cols = []
        for r in range(p_time):
            td = t_lags[q] - t_lags[r]
            grid = (
                cn_bar(u1_mesh, u2_mesh, td, q, r)
                + cn_bar(u1_mesh - n1, u2_mesh, td, q, r)
                + cn_bar(u1_mesh, u2_mesh - n2, td, q, r)
                + cn_bar(u1_mesh - n1, u2_mesh - n2, td, q, r)
            )
            cols.append(grid.to(torch.complex128))
        rows.append(torch.stack(cols, dim=-1))
    tilde = torch.stack(rows, dim=-2)
    result = torch.fft.fft2(tilde, dim=(0, 1)) / (4.0 * math.pi**2)
    return (result + result.conj().transpose(-1, -2)) / 2.0


def design_matrix_from_rows(rows: torch.Tensor, lat_mean: float) -> torch.Tensor:
    flat = rows.reshape(-1, rows.shape[-1])
    ones = torch.ones((flat.shape[0], 1), dtype=torch.float64, device=flat.device)
    lat = (flat[:, 0:1] - float(lat_mean)).to(torch.float64)
    dums = flat[:, 4:11].to(torch.float64)
    return torch.cat([ones, lat, dums], dim=1)


def residual_time_slices(
    asset: dict[str, Any],
    beta: torch.Tensor,
    lat_mean: float,
    device: torch.device,
) -> list[torch.Tensor]:
    grid_coords = torch.as_tensor(np.asarray(asset["grid_coords_np"], dtype=np.float64), dtype=torch.float64, device=device)
    out = []
    for key in sorted(asset["source_map"]):
        rows = asset["source_map"][key].to(device=device, dtype=torch.float64).clone()
        x = design_matrix_from_rows(rows, lat_mean=lat_mean)
        resid = rows[:, 2:3] - x @ beta.reshape(-1, 1)
        rows[:, 0:2] = grid_coords[:, 0:2]
        rows[:, 2] = resid.squeeze(1)
        out.append(rows)
    return out


def safe_cholesky(expected: torch.Tensor, p_time: int) -> torch.Tensor:
    eye = torch.eye(p_time, dtype=torch.complex128, device=expected.device)
    diag = torch.abs(expected.diagonal(dim1=-2, dim2=-1).real)
    base = float(torch.nanmean(diag).detach().cpu().item()) if diag.numel() else 1.0
    for mult in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]:
        try:
            return torch.linalg.cholesky(expected + eye * max(base * mult, 1e-12))
        except RuntimeError:
            continue
    return torch.linalg.cholesky(expected + eye * max(base * 1e-4, 1e-10))


def direction_cases(n1: int, n2: int, delta_lat: float, delta_lon: float) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    freq_lat = np.fft.fftshift(np.fft.fftfreq(n1, d=float(delta_lat)))
    freq_lon = np.fft.fftshift(np.fft.fftfreq(n2, d=float(delta_lon)))
    g_lat, g_lon = np.meshgrid(freq_lat, freq_lon, indexing="ij")
    ii, jj = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
    ci, cj = n1 // 2, n2 // 2
    return {
        "norm": (np.sqrt(g_lat**2 + g_lon**2), np.ones((n1, n2), dtype=bool)),
        "lat": (np.abs(g_lat), np.abs(jj - cj) <= 0),
        "lon": (np.abs(g_lon), np.abs(ii - ci) <= 0),
        "diag": (
            np.sqrt(g_lat**2 + g_lon**2),
            np.abs((ii - ci) / max(ci, 1) - (jj - cj) / max(cj, 1)) <= (1.0 / max(ci, cj, 1)),
        ),
    }


def direction_title(direction: str) -> str:
    return {
        "norm": "Norm frequency",
        "lat": "Latitude frequency",
        "lon": "Longitude frequency",
        "diag": "Diagonal frequency",
    }.get(str(direction), str(direction))


def direction_xlabel(direction: str) -> str:
    return {
        "norm": "norm frequency",
        "lat": "latitude frequency",
        "lon": "longitude frequency",
        "diag": "diagonal frequency",
    }.get(str(direction), "frequency")


def profile_rows_from_spectral_grids(
    data_shifted: np.ndarray,
    expected_shifted: np.ndarray,
    continuous_shifted: np.ndarray,
    whitened_shifted: np.ndarray,
    whitened_scale: float,
    year: int,
    month: int,
    day_idx: int,
    day: str,
    spec: dict[str, Any],
    fit_id: int,
    est: dict[str, float],
    n_bins: int,
    delta_lat: float,
    delta_lon: float,
) -> list[dict[str, Any]]:
    n1, n2 = data_shifted.shape
    expected_profile_shifted = expected_shifted * float(whitened_scale)
    continuous_profile_shifted = continuous_shifted * float(whitened_scale)
    ratio_i_ei = data_shifted / expected_shifted
    ratio_i_ei_profile = data_shifted / expected_profile_shifted
    ratio_ei_cont = expected_shifted / continuous_shifted
    whitened_ratio = whitened_shifted / float(whitened_scale)
    cases = direction_cases(n1, n2, delta_lat=delta_lat, delta_lon=delta_lon)
    rows: list[dict[str, Any]] = []
    for direction, (x_grid, mask2d) in cases.items():
        x_all = x_grid[mask2d].reshape(-1)
        data_all = data_shifted[mask2d].reshape(-1)
        expected_all = expected_shifted[mask2d].reshape(-1)
        expected_profile_all = expected_profile_shifted[mask2d].reshape(-1)
        continuous_all = continuous_shifted[mask2d].reshape(-1)
        continuous_profile_all = continuous_profile_shifted[mask2d].reshape(-1)
        ratio_i_all = ratio_i_ei[mask2d].reshape(-1)
        ratio_i_profile_all = ratio_i_ei_profile[mask2d].reshape(-1)
        ratio_ec_all = ratio_ei_cont[mask2d].reshape(-1)
        whitened_all = whitened_ratio[mask2d].reshape(-1)
        finite = (
            np.isfinite(x_all)
            & np.isfinite(data_all)
            & np.isfinite(expected_all)
            & np.isfinite(expected_profile_all)
            & np.isfinite(continuous_all)
            & np.isfinite(continuous_profile_all)
            & np.isfinite(ratio_i_all)
            & np.isfinite(ratio_i_profile_all)
            & np.isfinite(ratio_ec_all)
            & np.isfinite(whitened_all)
            & (x_all >= 0)
            & (expected_all > 0)
            & (expected_profile_all > 0)
            & (continuous_all > 0)
            & (continuous_profile_all > 0)
        )
        x = x_all[finite]
        if x.size == 0:
            continue
        data_vals_all = data_all[finite]
        expected_vals_all = expected_all[finite]
        expected_profile_vals_all = expected_profile_all[finite]
        continuous_vals_all = continuous_all[finite]
        continuous_profile_vals_all = continuous_profile_all[finite]
        ratio_i_vals_all = ratio_i_all[finite]
        ratio_i_profile_vals_all = ratio_i_profile_all[finite]
        ratio_ec_vals_all = ratio_ec_all[finite]
        whitened_vals_all = whitened_all[finite]
        xmax = float(np.nanmax(x))
        if xmax <= 0:
            bins = np.array([0.0, 1.0])
        else:
            bins = np.linspace(0.0, xmax, int(n_bins) + 1)
        idx = np.digitize(x, bins, right=False) - 1
        idx = np.clip(idx, 0, len(bins) - 2)
        for b in range(len(bins) - 1):
            sel = idx == b
            if not np.any(sel):
                continue
            data_vals = data_vals_all[sel]
            expected_vals = expected_vals_all[sel]
            expected_profile_vals = expected_profile_vals_all[sel]
            continuous_vals = continuous_vals_all[sel]
            continuous_profile_vals = continuous_profile_vals_all[sel]
            ratio_i_vals = ratio_i_vals_all[sel]
            ratio_i_profile_vals = ratio_i_profile_vals_all[sel]
            ratio_ec_vals = ratio_ec_vals_all[sel]
            whitened_vals = whitened_vals_all[sel]
            rows.append(
                {
                    "fit_id": int(fit_id),
                    "year": int(year),
                    "month": int(month),
                    "day_idx": int(day_idx),
                    "day": day,
                    "model_variant": str(spec["model_variant"]),
                    "model_family": str(spec["family"]),
                    "model_label": str(spec["label"]),
                    "smooth": float(spec["smooth"]) if pd.notna(spec["smooth"]) else np.nan,
                    "gc_alpha": float(spec["gc_alpha"]) if pd.notna(spec["gc_alpha"]) else np.nan,
                    "gc_beta": float(spec["gc_beta"]) if pd.notna(spec["gc_beta"]) else np.nan,
                    "nugget_mode": "zero",
                    "direction": direction,
                    "bin_idx": int(b),
                    "k_min": float(bins[b]),
                    "k_max": float(bins[b + 1]),
                    "k_mid": float(0.5 * (bins[b] + bins[b + 1])),
                    "data_spectrum_mean": float(np.nanmean(data_vals)),
                    "expected_spectrum_mean": float(np.nanmean(expected_vals)),
                    "expected_spectrum_profile_mean": float(np.nanmean(expected_profile_vals)),
                    "continuous_spectrum_mean": float(np.nanmean(continuous_vals)),
                    "continuous_spectrum_profile_mean": float(np.nanmean(continuous_profile_vals)),
                    "ratio_I_over_EI_mean": float(np.nanmean(ratio_i_vals)),
                    "ratio_I_over_EI_median": float(np.nanmedian(ratio_i_vals)),
                    "ratio_I_over_EI_p10": float(np.nanquantile(ratio_i_vals, 0.10)),
                    "ratio_I_over_EI_p90": float(np.nanquantile(ratio_i_vals, 0.90)),
                    "ratio_I_over_EI_profile_mean": float(np.nanmean(ratio_i_profile_vals)),
                    "ratio_I_over_EI_profile_median": float(np.nanmedian(ratio_i_profile_vals)),
                    "ratio_I_over_EI_profile_p10": float(np.nanquantile(ratio_i_profile_vals, 0.10)),
                    "ratio_I_over_EI_profile_p90": float(np.nanquantile(ratio_i_profile_vals, 0.90)),
                    "ratio_EI_over_continuous_mean": float(np.nanmean(ratio_ec_vals)),
                    "ratio_EI_over_continuous_median": float(np.nanmedian(ratio_ec_vals)),
                    "whitened_ratio_mean": float(np.nanmean(whitened_vals)),
                    "whitened_ratio_median": float(np.nanmedian(whitened_vals)),
                    "n": int(ratio_i_vals.size),
                    "global_scale": float(whitened_scale),
                    "profile_sigmasq": float(est["sigmasq"] * whitened_scale),
                }
            )
    return rows


def compute_spectral_profile(
    asset: dict[str, Any],
    est: dict[str, float],
    beta: torch.Tensor,
    lat_mean: float,
    spec: dict[str, Any],
    fit_id: int,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    slices = residual_time_slices(asset, beta=beta, lat_mean=lat_mean, device=device)
    j_vec, n1, n2, p_time, taper_grid, obs_masks = DWL.generate_Jvector_tapered_mv(
        slices,
        tapering_func=no_taper,
        lat_col=0,
        lon_col=1,
        val_col=2,
        device=device,
    )
    if p_time != T_STEPS:
        raise RuntimeError(f"Expected {T_STEPS} time slices, got {p_time}")
    taper_auto = DWL.calculate_taper_autocorrelation_multivariate(taper_grid, obs_masks, n1, n2, device)
    params = raw_params_from_est(est).to(device=device)
    with torch.no_grad():
        expected = expected_periodogram_raw_st(
            params=params,
            spec=spec,
            n1=int(n1),
            n2=int(n2),
            p_time=int(p_time),
            taper_autocorr_grid=taper_auto,
            delta_lat=float(args.delta_lat),
            delta_lon=float(args.delta_lon),
            spline_n_points=int(args.spline_n_points),
            spline_r_max=float(args.spline_r_max),
        )
        continuous_auto = torch.ones_like(taper_auto)
        continuous = expected_periodogram_raw_st(
            params=params,
            spec=spec,
            n1=int(n1),
            n2=int(n2),
            p_time=int(p_time),
            taper_autocorr_grid=continuous_auto,
            delta_lat=float(args.delta_lat),
            delta_lon=float(args.delta_lon),
            spline_n_points=int(args.spline_n_points),
            spline_r_max=float(args.spline_r_max),
        )
        chol = safe_cholesky(expected, p_time=int(p_time))
        z = torch.linalg.solve_triangular(chol, j_vec.unsqueeze(-1), upper=False)
        whitened_power = (z.abs() ** 2).squeeze(-1).mean(dim=-1)
        data_scalar = (j_vec.abs() ** 2).mean(dim=-1)
        expected_scalar_t = expected.diagonal(dim1=-2, dim2=-1).real.mean(dim=-1).clamp_min(1e-300)
        continuous_scalar_t = continuous.diagonal(dim1=-2, dim2=-1).real.mean(dim=-1).clamp_min(1e-300)
    whitened_shifted = np.fft.fftshift(whitened_power.detach().cpu().numpy(), axes=(0, 1))
    data_shifted = np.fft.fftshift(data_scalar.detach().cpu().numpy(), axes=(0, 1))
    expected_shifted = np.fft.fftshift(expected_scalar_t.detach().cpu().numpy(), axes=(0, 1))
    continuous_shifted = np.fft.fftshift(continuous_scalar_t.detach().cpu().numpy(), axes=(0, 1))
    raw = whitened_shifted.reshape(-1)
    scale = float(np.nanmean(raw))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"Invalid profile scale: {scale}")
    rows = profile_rows_from_spectral_grids(
        data_shifted=data_shifted,
        expected_shifted=expected_shifted,
        continuous_shifted=continuous_shifted,
        whitened_shifted=whitened_shifted,
        whitened_scale=scale,
        year=int(asset["year"]),
        month=int(asset["month"]),
        day_idx=int(asset["day_idx"]),
        day=str(asset["day"]),
        spec=spec,
        fit_id=int(fit_id),
        est=est,
        n_bins=int(args.n_bins),
        delta_lat=float(args.delta_lat),
        delta_lon=float(args.delta_lon),
    )
    stats = {
        "spectral_n1": int(n1),
        "spectral_n2": int(n2),
        "spectral_p_time": int(p_time),
        "spectral_global_scale": float(scale),
        "spectral_profile_sigmasq": float(est["sigmasq"] * scale),
        "spectral_ratio_mean_after_profile": float(np.nanmean(raw / scale)),
        "spectral_ratio_var_after_profile": float(np.nanvar(raw / scale)),
        "spectral_data_over_expected_mean": float(np.nanmean(data_shifted / expected_shifted)),
        "spectral_expected_over_continuous_mean": float(np.nanmean(expected_shifted / continuous_shifted)),
    }
    del slices, j_vec, taper_auto, continuous_auto, expected, continuous, chol, z, whitened_power
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, stats


def fit_full_asset(
    asset: dict[str, Any],
    spec: dict[str, Any],
    init_physical: dict[str, float],
    reference_advec_lon_abs: float,
    fit_id: int,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], torch.Tensor, float]:
    source_map = {
        k: v.to(device=device, dtype=DTYPE, non_blocking=True).contiguous()
        for k, v in asset["source_map"].items()
    }
    grid_coords_np = np.asarray(asset["grid_coords_np"], dtype=np.float64)
    n_valid, n_total, valid_by_t = count_valid(source_map)
    params_list = make_params_list(init_physical, dtype=DTYPE, device=device, nugget_mode="zero")

    if str(spec["family"]) == "matern":
        model = RealDataCorridorWidth4x4Lag643NoNuggetSplineFit(
            smooth=float(spec["smooth"]),
            input_map=source_map,
            grid_coords=grid_coords_np,
            lag1_lon_offset=float(reference_advec_lon_abs),
            daily_stride=int(args.daily_stride),
            target_chunk_size=int(args.target_chunk_size),
            min_target_points=int(args.min_target_points),
            spline_n_points=int(args.spline_n_points),
            spline_r_max=float(args.spline_r_max),
        )
    elif str(spec["family"]) == "cauchy":
        model = RealDataCorridorWidth4x4Lag643NoNuggetGeneralizedCauchyFit(
            gc_alpha=float(spec["gc_alpha"]),
            gc_beta=float(spec["gc_beta"]),
            input_map=source_map,
            grid_coords=grid_coords_np,
            lag1_lon_offset=float(reference_advec_lon_abs),
            daily_stride=int(args.daily_stride),
            target_chunk_size=int(args.target_chunk_size),
            min_target_points=int(args.min_target_points),
        )
    else:
        raise ValueError(f"Unknown model family {spec['family']!r}")

    t0 = time.time()
    model.precompute_conditioning_sets()
    precompute_s = time.time() - t0
    optimizer = model.set_optimizer(
        params_list,
        lr=float(args.lbfgs_lr),
        max_iter=int(args.lbfgs_eval),
        max_eval=int(args.lbfgs_eval),
        history_size=int(args.lbfgs_history),
    )
    t1 = time.time()
    if args.suppress_fit_prints:
        import contextlib
        import io

        with contextlib.redirect_stdout(io.StringIO()):
            out, steps_ran = model.fit_vecc_lbfgs(
                params_list, optimizer, max_steps=int(args.lbfgs_steps), grad_tol=float(args.grad_tol)
            )
    else:
        out, steps_ran = model.fit_vecc_lbfgs(
            params_list, optimizer, max_steps=int(args.lbfgs_steps), grad_tol=float(args.grad_tol)
        )
    fit_s = time.time() - t1

    params_tensor = torch.stack([p.reshape(()) for p in params_list]).detach()
    beta = model.get_gls_beta(params_tensor).detach()
    lat_mean = float(model.lat_mean_val)
    est = backmap_params(out, nugget_mode="zero")
    cluster_summary = model.cluster_summary()

    row = {
        "fit_id": int(fit_id),
        "status": "ok",
        "error": "",
        "data_kind": "real",
        "year": int(asset["year"]),
        "month": int(asset["month"]),
        "day_idx": int(asset["day_idx"]),
        "day": str(asset["day"]),
        "model_variant": str(spec["model_variant"]),
        "model_family": str(spec["family"]),
        "model_label": str(spec["label"]),
        "smooth": float(spec["smooth"]) if pd.notna(spec["smooth"]) else np.nan,
        "gc_alpha": float(spec["gc_alpha"]) if pd.notna(spec["gc_alpha"]) else np.nan,
        "gc_beta": float(spec["gc_beta"]) if pd.notna(spec["gc_beta"]) else np.nan,
        "nugget_mode": "zero",
        "n_grid_full": int(grid_coords_np.shape[0]),
        "n_time_slots": int(len(source_map)),
        "n_rows_total": int(n_total),
        "n_valid_o3": int(n_valid),
        "valid_rate": float(n_valid / n_total) if n_total else np.nan,
        "valid_by_t": json.dumps(valid_by_t, separators=(",", ":")),
        "monthly_mean": float(asset["monthly_mean"]),
        "first_slot": asset["day_keys"][0] if asset.get("day_keys") else "",
        "last_slot": asset["day_keys"][-1] if asset.get("day_keys") else "",
        "spec_name": VECCHIA_SPEC_NAME,
        "smooth_kernel": "spline" if str(spec["family"]) == "matern" else "generalized_cauchy",
        "spline_n_points": int(args.spline_n_points),
        "spline_r_max": float(args.spline_r_max),
        "block_shape": f"{BLOCK_SHAPE[0]}x{BLOCK_SHAPE[1]}",
        "lag_pattern": f"{LAG_COUNTS[0]}/{LAG_COUNTS[1]}/{LAG_COUNTS[2]}",
        "reference_advec_lon_abs": float(reference_advec_lon_abs),
        "loss": float(out[-1]),
        "steps_raw": int(steps_ran),
        "precompute_s": float(precompute_s),
        "fit_s": float(fit_s),
        "total_s": float(precompute_s + fit_s),
        "gls_lat_mean": float(lat_mean),
        **{f"est_{k}": float(est[k]) for k in P_LABELS},
        **{f"beta_{i}": float(beta.reshape(-1)[i].detach().cpu().item()) for i in range(beta.numel())},
        **cluster_summary,
        "model_spec": json.dumps(clean_json_value(corridor_width_643_spec(reference_advec_lon_abs)), sort_keys=True),
    }
    del model, params_list, optimizer, source_map
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row, beta, lat_mean


def refresh_outputs(out_dir: Path, fit_rows: list[dict[str, Any]], profile_rows: list[dict[str, Any]], top_plot_dir: Path | None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fit_df = save_rows(out_dir / FIT_CSV, fit_rows) if fit_rows else pd.DataFrame()
    profile_df = save_rows(out_dir / PROFILE_CSV, profile_rows) if profile_rows else pd.DataFrame()
    if not fit_df.empty:
        ok = fit_df[fit_df["status"] == "ok"].copy()
        if not ok.empty:
            summary = make_fit_summary(ok)
            save_rows(out_dir / "st_corridor_parameter_monthly_summary.csv", summary)
            plot_parameter_monthly_summary(summary, out_dir / "st_corridor_parameter_monthly_summary.png")
            if top_plot_dir is not None:
                for year, sub_summary in summary.groupby("year", dropna=False):
                    year_dir = top_plot_dir / f"year_{int(year)}"
                    year_dir.mkdir(parents=True, exist_ok=True)
                    save_rows(year_dir / "st_corridor_parameter_monthly_summary.csv", sub_summary)
                    plot_parameter_monthly_summary(sub_summary, year_dir / "st_corridor_parameter_monthly_summary.png")
    if not profile_df.empty:
        monthly = make_profile_monthly_summary(profile_df)
        save_rows(out_dir / MONTHLY_SUMMARY_CSV, monthly)
        band_table = make_ratio_band_table(monthly)
        save_rows(out_dir / BAND_TABLE_CSV, band_table)
        plot_profile_monthly_summary(
            monthly,
            out_dir / "marginal_timeavg_spatial_monthly_I_over_Ediag_profile_sigma_ratio.png",
            metric="ratio_I_over_EI_profile_mean",
        )
        plot_profile_monthly_summary(
            monthly,
            out_dir / "marginal_timeavg_spatial_monthly_Ediag_over_continuous_ratio.png",
            metric="ratio_EI_over_continuous_mean",
        )
        plot_profile_monthly_summary(
            monthly,
            out_dir / "whitened_8x8_monthly_I_over_EI_target1_ratio.png",
            metric="whitened_ratio_mean",
        )
        plot_directional_year_outputs(monthly, out_dir / "monthly_average_plots")
        if top_plot_dir is not None:
            top_plot_dir.mkdir(parents=True, exist_ok=True)
            plot_directional_year_outputs(monthly, top_plot_dir)
            for year, sub_monthly in monthly.groupby("year", dropna=False):
                year_dir = top_plot_dir / f"year_{int(year)}"
                year_dir.mkdir(parents=True, exist_ok=True)
                save_rows(year_dir / MONTHLY_SUMMARY_CSV, sub_monthly)
                sub_band = band_table[band_table["year"] == int(year)].copy() if not band_table.empty else pd.DataFrame()
                save_rows(year_dir / BAND_TABLE_CSV, sub_band)

    lines = [
        f"Updated: {datetime.now().isoformat(timespec='seconds')}",
        f"Fits: {len(fit_rows)}",
        f"Profile rows: {len(profile_rows)}",
    ]
    if fit_rows:
        lines.append(pd.DataFrame(fit_rows).tail(12).to_string(index=False))
    (out_dir / "running_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_fit_summary(ok: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for param in P_LABELS:
        col = f"est_{param}"
        if col not in ok.columns:
            continue
        for keys, sub in ok.groupby(["year", "month", "model_variant", "model_label"], dropna=False):
            vals = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            rows.append(
                {
                    "year": int(keys[0]),
                    "month": int(keys[1]),
                    "model_variant": str(keys[2]),
                    "model_label": str(keys[3]),
                    "parameter": param,
                    "n": int(vals.size),
                    "mean": float(np.mean(vals)),
                    "median": float(np.median(vals)),
                    "sd": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                    "p10": float(np.quantile(vals, 0.10)),
                    "p90": float(np.quantile(vals, 0.90)),
                }
            )
    return pd.DataFrame(rows)


def plot_parameter_monthly_summary(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
    axes_flat = axes.ravel()
    for ax, param in zip(axes_flat, P_LABELS):
        sub_param = summary[summary["parameter"] == param].copy()
        for model_variant, sub in sub_param.groupby("model_variant", dropna=False):
            sub = sub.sort_values(["year", "month"])
            labels = [f"{int(y)}-{int(m):02d}" for y, m in zip(sub["year"], sub["month"])]
            x = np.arange(len(labels))
            label = str(sub["model_label"].dropna().iloc[0]) if sub["model_label"].notna().any() else str(model_variant)
            ax.plot(x, sub["median"], marker="o", linewidth=1.8, label=label)
            ax.fill_between(x, sub["p10"].to_numpy(float), sub["p90"].to_numpy(float), alpha=0.12)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_title(param)
        ax.grid(alpha=0.25)
    for ax in axes_flat[len(P_LABELS) :]:
        ax.axis("off")
    axes_flat[0].legend(fontsize=8)
    fig.suptitle("ST corridor Vecchia parameter monthly summaries, full grid")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_profile_monthly_summary(profile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["year", "month", "model_variant", "model_label", "direction", "bin_idx"]
    for keys, sub in profile.groupby(group_cols, dropna=False):
        vals = pd.to_numeric(sub["ratio_I_over_EI_mean"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        vals_profile = (
            pd.to_numeric(sub["ratio_I_over_EI_profile_mean"], errors="coerce").dropna().to_numpy(dtype=float)
            if "ratio_I_over_EI_profile_mean" in sub.columns
            else vals
        )
        ratio_ei_cont = pd.to_numeric(sub["ratio_EI_over_continuous_mean"], errors="coerce").dropna().to_numpy(dtype=float)
        whitened = pd.to_numeric(sub["whitened_ratio_mean"], errors="coerce").dropna().to_numpy(dtype=float)
        expected_profile_col = "expected_spectrum_profile_mean" if "expected_spectrum_profile_mean" in sub.columns else "expected_spectrum_mean"
        continuous_profile_col = "continuous_spectrum_profile_mean" if "continuous_spectrum_profile_mean" in sub.columns else "continuous_spectrum_mean"
        rows.append(
            {
                "year": int(keys[0]),
                "month": int(keys[1]),
                "model_variant": str(keys[2]),
                "model_label": str(keys[3]),
                "direction": str(keys[4]),
                "bin_idx": int(keys[5]),
                "k_mid": float(np.nanmean(pd.to_numeric(sub["k_mid"], errors="coerce"))),
                "n_days": int(vals.size),
                "data_spectrum_mean": float(np.nanmean(pd.to_numeric(sub["data_spectrum_mean"], errors="coerce"))),
                "expected_spectrum_mean": float(np.nanmean(pd.to_numeric(sub["expected_spectrum_mean"], errors="coerce"))),
                "expected_spectrum_profile_mean": float(np.nanmean(pd.to_numeric(sub[expected_profile_col], errors="coerce"))),
                "continuous_spectrum_mean": float(np.nanmean(pd.to_numeric(sub["continuous_spectrum_mean"], errors="coerce"))),
                "continuous_spectrum_profile_mean": float(np.nanmean(pd.to_numeric(sub[continuous_profile_col], errors="coerce"))),
                "ratio_I_over_EI_mean": float(np.mean(vals)),
                "ratio_I_over_EI_median": float(np.median(vals)),
                "ratio_I_over_EI_p10": float(np.quantile(vals, 0.10)),
                "ratio_I_over_EI_p90": float(np.quantile(vals, 0.90)),
                "ratio_I_over_EI_profile_mean": float(np.mean(vals_profile)) if vals_profile.size else np.nan,
                "ratio_I_over_EI_profile_median": float(np.median(vals_profile)) if vals_profile.size else np.nan,
                "ratio_I_over_EI_profile_p10": float(np.quantile(vals_profile, 0.10)) if vals_profile.size else np.nan,
                "ratio_I_over_EI_profile_p90": float(np.quantile(vals_profile, 0.90)) if vals_profile.size else np.nan,
                "ratio_EI_over_continuous_mean": float(np.mean(ratio_ei_cont)) if ratio_ei_cont.size else np.nan,
                "ratio_EI_over_continuous_median": float(np.median(ratio_ei_cont)) if ratio_ei_cont.size else np.nan,
                "ratio_EI_over_continuous_p10": float(np.quantile(ratio_ei_cont, 0.10)) if ratio_ei_cont.size else np.nan,
                "ratio_EI_over_continuous_p90": float(np.quantile(ratio_ei_cont, 0.90)) if ratio_ei_cont.size else np.nan,
                "whitened_ratio_mean": float(np.mean(whitened)) if whitened.size else np.nan,
                "whitened_ratio_median": float(np.median(whitened)) if whitened.size else np.nan,
                "whitened_ratio_p10": float(np.quantile(whitened, 0.10)) if whitened.size else np.nan,
                "whitened_ratio_p90": float(np.quantile(whitened, 0.90)) if whitened.size else np.nan,
                "global_scale_mean": float(np.nanmean(pd.to_numeric(sub["global_scale"], errors="coerce"))),
                "profile_sigmasq_mean": float(np.nanmean(pd.to_numeric(sub["profile_sigmasq"], errors="coerce"))),
            }
        )
    return pd.DataFrame(rows)


def ratio_band(bin_idx: int, max_bin: int) -> str:
    b = int(bin_idx)
    if b == 0:
        return "lowest_frequency_bin"
    if 1 <= b <= 5:
        return "low_frequency_bins_1_5"
    frac = b / max(max_bin, 1)
    if 0.35 <= frac <= 0.55:
        return "mid_frequency_band"
    if frac >= 0.80:
        return "high_frequency_band"
    return ""


def make_ratio_band_table(monthly: pd.DataFrame) -> pd.DataFrame:
    if monthly.empty:
        return pd.DataFrame()
    df = monthly.copy()
    max_bins = df.groupby(["year", "model_variant", "direction"], dropna=False)["bin_idx"].transform("max")
    df["frequency_band"] = [ratio_band(b, m) for b, m in zip(df["bin_idx"], max_bins)]
    df = df[df["frequency_band"] != ""].copy()
    if df.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["year", "month", "model_variant", "model_label", "direction", "frequency_band"]
    for keys, sub in df.groupby(group_cols, dropna=False):
        ratio_profile_col = "ratio_I_over_EI_profile_mean" if "ratio_I_over_EI_profile_mean" in sub.columns else "ratio_I_over_EI_mean"
        ratio_profile_median_col = (
            "ratio_I_over_EI_profile_median" if "ratio_I_over_EI_profile_median" in sub.columns else "ratio_I_over_EI_median"
        )
        expected_profile_col = "expected_spectrum_profile_mean" if "expected_spectrum_profile_mean" in sub.columns else "expected_spectrum_mean"
        continuous_profile_col = "continuous_spectrum_profile_mean" if "continuous_spectrum_profile_mean" in sub.columns else "continuous_spectrum_mean"
        rows.append(
            {
                "year": int(keys[0]),
                "month": int(keys[1]),
                "model_variant": str(keys[2]),
                "model_label": str(keys[3]),
                "direction": str(keys[4]),
                "frequency_band": str(keys[5]),
                "bin_idx_min": int(pd.to_numeric(sub["bin_idx"], errors="coerce").min()),
                "bin_idx_max": int(pd.to_numeric(sub["bin_idx"], errors="coerce").max()),
                "k_mid_mean": float(np.nanmean(pd.to_numeric(sub["k_mid"], errors="coerce"))),
                "n_bins": int(len(sub)),
                "ratio_I_over_EI_mean": float(np.nanmean(pd.to_numeric(sub["ratio_I_over_EI_mean"], errors="coerce"))),
                "ratio_I_over_EI_median": float(np.nanmedian(pd.to_numeric(sub["ratio_I_over_EI_median"], errors="coerce"))),
                "ratio_I_over_EI_profile_mean": float(np.nanmean(pd.to_numeric(sub[ratio_profile_col], errors="coerce"))),
                "ratio_I_over_EI_profile_median": float(np.nanmedian(pd.to_numeric(sub[ratio_profile_median_col], errors="coerce"))),
                "ratio_EI_over_continuous_mean": float(np.nanmean(pd.to_numeric(sub["ratio_EI_over_continuous_mean"], errors="coerce"))),
                "ratio_EI_over_continuous_median": float(np.nanmedian(pd.to_numeric(sub["ratio_EI_over_continuous_median"], errors="coerce"))),
                "whitened_ratio_mean": float(np.nanmean(pd.to_numeric(sub["whitened_ratio_mean"], errors="coerce"))),
                "data_spectrum_mean": float(np.nanmean(pd.to_numeric(sub["data_spectrum_mean"], errors="coerce"))),
                "expected_spectrum_mean": float(np.nanmean(pd.to_numeric(sub["expected_spectrum_mean"], errors="coerce"))),
                "expected_spectrum_profile_mean": float(np.nanmean(pd.to_numeric(sub[expected_profile_col], errors="coerce"))),
                "continuous_spectrum_mean": float(np.nanmean(pd.to_numeric(sub["continuous_spectrum_mean"], errors="coerce"))),
                "continuous_spectrum_profile_mean": float(np.nanmean(pd.to_numeric(sub[continuous_profile_col], errors="coerce"))),
            }
        )
    return pd.DataFrame(rows)


def plot_profile_monthly_summary(monthly: pd.DataFrame, path: Path, metric: str = "ratio_I_over_EI_mean") -> None:
    if monthly.empty:
        return
    if metric not in monthly.columns:
        return
    band_cols = {
        "ratio_I_over_EI_mean": ("ratio_I_over_EI_p10", "ratio_I_over_EI_p90"),
        "ratio_I_over_EI_profile_mean": ("ratio_I_over_EI_profile_p10", "ratio_I_over_EI_profile_p90"),
        "ratio_EI_over_continuous_mean": ("ratio_EI_over_continuous_p10", "ratio_EI_over_continuous_p90"),
        "whitened_ratio_mean": ("whitened_ratio_p10", "whitened_ratio_p90"),
    }
    directions = ["norm", "lat", "lon", "diag"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for ax, direction in zip(axes.ravel(), directions):
        sub_dir = monthly[monthly["direction"] == direction].copy()
        for (year, model_variant), sub in sub_dir.groupby(["year", "model_variant"], dropna=False):
            sub = sub.sort_values("k_mid")
            model_label = str(sub["model_label"].dropna().iloc[0]) if sub["model_label"].notna().any() else str(model_variant)
            label = f"{int(year)}, {model_label}"
            line = ax.plot(sub["k_mid"], sub[metric], linewidth=1.6, label=label)[0]
            lo_col, hi_col = band_cols.get(metric, ("", ""))
            if lo_col in sub.columns and hi_col in sub.columns:
                x = pd.to_numeric(sub["k_mid"], errors="coerce").to_numpy(dtype=float)
                lo = pd.to_numeric(sub[lo_col], errors="coerce").to_numpy(dtype=float)
                hi = pd.to_numeric(sub[hi_col], errors="coerce").to_numpy(dtype=float)
                ok = np.isfinite(x) & np.isfinite(lo) & np.isfinite(hi) & (lo > 0) & (hi > 0)
                if ok.any():
                    ax.fill_between(x[ok], lo[ok], hi[ok], color=line.get_color(), alpha=0.12, linewidth=0)
        ax.axhline(1.0, color="0.25", linestyle="--", linewidth=1.0)
        ax.set_title(direction_title(direction))
        ax.set_xlabel(direction_xlabel(direction))
        ax.set_yscale("log")
        ax.set_ylim(0.2, 5.0)
        ax.grid(alpha=0.25, which="both")
    ylabel = {
        "ratio_I_over_EI_mean": "data I / finite-sample E[I]",
        "ratio_I_over_EI_profile_mean": "time-averaged marginal spatial I / diagonal E[I] (profile sigma)",
        "ratio_EI_over_continuous_mean": "finite-sample E[I] / continuous-like spectrum",
        "whitened_ratio_mean": "8x8-whitened I / E[I] quadratic power (target = 1)",
    }.get(metric, metric)
    axes[0, 0].set_ylabel(ylabel)
    axes[1, 0].set_ylabel(ylabel)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"ST Vecchia spectral diagnostic: monthly mean {ylabel}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_model_pair_lines(
    ax: plt.Axes,
    sub_year_dir: pd.DataFrame,
    y_cols: tuple[str, str],
    labels: tuple[str, str],
) -> None:
    for model_variant, sub_model in sub_year_dir.groupby("model_variant", dropna=False):
        sub_model = sub_model.sort_values("k_mid")
        model_label = str(sub_model["model_label"].dropna().iloc[0]) if sub_model["model_label"].notna().any() else str(model_variant)
        line = ax.plot(
            pd.to_numeric(sub_model["k_mid"], errors="coerce"),
            pd.to_numeric(sub_model[y_cols[0]], errors="coerce"),
            linewidth=1.9,
            label=f"{model_label}: {labels[0]}",
        )[0]
        ax.plot(
            pd.to_numeric(sub_model["k_mid"], errors="coerce"),
            pd.to_numeric(sub_model[y_cols[1]], errors="coerce"),
            color=line.get_color(),
            linestyle="--",
            linewidth=1.7,
            label=f"{model_label}: {labels[1]}",
        )


def plot_directional_year_outputs(monthly: pd.DataFrame, base_dir: Path) -> None:
    if monthly.empty:
        return
    directions = ["norm", "lat", "lon", "diag"]
    for year, sub_year in monthly.groupby("year", dropna=False):
        year_dir = base_dir / f"year_{int(year)}"
        year_dir.mkdir(parents=True, exist_ok=True)
        for direction in directions:
            sub = sub_year[sub_year["direction"] == direction].copy()
            if sub.empty:
                continue

            fig, ax = plt.subplots(figsize=(8.5, 5.4))
            _plot_model_pair_lines(
                ax,
                sub,
                ("data_spectrum_mean", "expected_spectrum_profile_mean"),
                ("I", "E[I]"),
            )
            ax.set_title(f"Real July {int(year)}: marginal time-averaged spatial I vs diagonal E[I], {direction_title(direction)}")
            ax.set_xlabel(direction_xlabel(direction))
            ax.set_ylabel("Marginal time-averaged spatial spectrum with profile sigma: observed I and diagonal fitted E[I]")
            ax.set_yscale("log")
            ax.grid(alpha=0.25, which="both")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(year_dir / f"marginal_timeavg_spatial_I_vs_Ediag_profile_sigma_{direction}.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8.5, 5.4))
            for model_variant, sub_model in sub.groupby("model_variant", dropna=False):
                sub_model = sub_model.sort_values("k_mid")
                model_label = str(sub_model["model_label"].dropna().iloc[0]) if sub_model["model_label"].notna().any() else str(model_variant)
                ax.plot(
                    pd.to_numeric(sub_model["k_mid"], errors="coerce"),
                    pd.to_numeric(sub_model["ratio_I_over_EI_profile_mean"], errors="coerce"),
                    linewidth=1.9,
                    label=model_label,
                )
            ax.axhline(1.0, color="0.25", linestyle="--", linewidth=1.0)
            ax.set_title(f"Real July {int(year)}: marginal time-averaged spatial I / diagonal E[I], {direction_title(direction)}")
            ax.set_xlabel(direction_xlabel(direction))
            ax.set_ylabel("Marginal time-averaged spatial I / diagonal E[I] with profile sigma (target = 1)")
            ax.set_yscale("log")
            ax.set_ylim(0.2, 5.0)
            ax.grid(alpha=0.25, which="both")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(year_dir / f"marginal_timeavg_spatial_I_over_Ediag_profile_sigma_target1_{direction}.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8.5, 5.4))
            _plot_model_pair_lines(
                ax,
                sub,
                ("expected_spectrum_profile_mean", "continuous_spectrum_profile_mean"),
                ("E[I]", "theoretic continuous"),
            )
            ax.set_title(f"Real July {int(year)}: marginal time-averaged diagonal E[I] vs continuous spectrum, {direction_title(direction)}")
            ax.set_xlabel(direction_xlabel(direction))
            ax.set_ylabel("Marginal time-averaged spatial spectrum with profile sigma: diagonal fitted E[I] and theoretical continuous spectrum")
            ax.set_yscale("log")
            ax.grid(alpha=0.25, which="both")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(year_dir / f"marginal_timeavg_spatial_Ediag_vs_continuous_profile_sigma_{direction}.png", dpi=180, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(8.5, 5.4))
            for model_variant, sub_model in sub.groupby("model_variant", dropna=False):
                sub_model = sub_model.sort_values("k_mid")
                model_label = str(sub_model["model_label"].dropna().iloc[0]) if sub_model["model_label"].notna().any() else str(model_variant)
                ax.plot(
                    pd.to_numeric(sub_model["k_mid"], errors="coerce"),
                    pd.to_numeric(sub_model["whitened_ratio_mean"], errors="coerce"),
                    linewidth=1.9,
                    label=model_label,
                )
            ax.axhline(1.0, color="0.25", linestyle="--", linewidth=1.0)
            ax.set_title(f"Real July {int(year)}: 8x8-whitened I vs E[I], {direction_title(direction)}")
            ax.set_xlabel(direction_xlabel(direction))
            ax.set_ylabel("8x8-whitened I / E[I] quadratic power (target = 1)")
            ax.set_yscale("log")
            ax.set_ylim(0.2, 5.0)
            ax.grid(alpha=0.25, which="both")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(year_dir / f"whitened_8x8_I_vs_EI_target1_{direction}.png", dpi=180, bbox_inches="tight")
            plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit full-grid ST corridor Vecchia and compute 8x8-whitened spectral profiles.")
    parser.add_argument("--model-variants", nargs="+", default=DEFAULT_MODEL_VARIANTS)
    parser.add_argument("--days", default="0,30", help="'0,30' means July day_idx 0..29.")
    parser.add_argument("--real-years", nargs="+", default=["2023", "2024", "2025"])
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--space", default="1,1")
    parser.add_argument("--lat-range", default="-3,2")
    parser.add_argument("--lon-range", default="121,131")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--delta-lat", type=float, default=DELTA_LAT_BASE)
    parser.add_argument("--delta-lon", type=float, default=DELTA_LON_BASE)
    parser.add_argument("--spline-n-points", type=int, default=4000)
    parser.add_argument("--spline-r-max", type=float, default=30.0)
    parser.add_argument("--real-reference-advec-lon-abs", type=float, default=REFERENCE_ADVEC_LON_ABS)
    parser.add_argument("--daily-stride", type=int, default=2)
    parser.add_argument("--target-chunk-size", type=int, default=128)
    parser.add_argument("--min-target-points", type=int, default=1)
    parser.add_argument("--lbfgs-lr", type=float, default=1.0)
    parser.add_argument("--lbfgs-steps", type=int, default=5)
    parser.add_argument("--lbfgs-eval", type=int, default=20)
    parser.add_argument("--lbfgs-history", type=int, default=10)
    parser.add_argument("--grad-tol", type=float, default=1e-5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cuda-fallback", choices=["cpu", "error"], default="cpu")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--keep-exact-loc", dest="keep_exact_loc", action="store_true", default=True)
    parser.add_argument("--no-keep-exact-loc", dest="keep_exact_loc", action="store_false")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summary-every", type=int, default=1)
    parser.add_argument("--n-bins", type=int, default=80)
    parser.add_argument("--suppress-fit-prints", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--monthly-out-dir", type=Path, default=None)
    return parser


def load_existing_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    return pd.read_csv(csv_path).to_dict(orient="records")


def completed_keys(rows: list[dict[str, Any]]) -> set[tuple[int, int, str]]:
    out = set()
    for row in rows:
        if str(row.get("status", "")) == "ok":
            out.add((int(row["year"]), int(row["day_idx"]), str(row["model_variant"])))
    return out


def main() -> None:
    args = build_arg_parser().parse_args()
    device = resolve_device(args)
    model_variants = []
    for token in args.model_variants:
        model_variants.extend(part.strip() for part in str(token).split(",") if part.strip())
    for name in model_variants:
        variant_spec(name)
    years = parse_int_tokens(args.real_years)
    out_dir = args.out_dir or default_output_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    top_plot_dir = args.monthly_out_dir
    if top_plot_dir is not None:
        top_plot_dir.mkdir(parents=True, exist_ok=True)

    print("SRC:", SRC, flush=True)
    print("device:", device, flush=True)
    print("out_dir:", out_dir, flush=True)
    print("years:", years, "model_variants:", model_variants, "days:", parse_day_idxs(args.days), flush=True)
    print("year_variant_defaults:", YEAR_VARIANT_DEFAULTS, flush=True)
    print("region:", parse_pair(args.lat_range, float), parse_pair(args.lon_range, float), flush=True)

    run_config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "src": str(SRC),
        "device": str(device),
        "args": clean_json_value(vars(args)),
        "model_variants": {name: clean_json_value(variant_spec(name)) for name in model_variants},
        "year_variant_defaults": YEAR_VARIANT_DEFAULTS,
        "years": years,
        "model_spec_real": corridor_width_643_spec(args.real_reference_advec_lon_abs),
        "default_real_init_physical": DEFAULT_REAL_INIT_PHYSICAL,
        "spectral_diagnostic": "raw residual field, no taper, missing-window cross autocorrelation, 8x8 Cholesky whitening, global mean profile-out",
    }
    (out_dir / "run_config.json").write_text(json.dumps(clean_json_value(run_config), indent=2, sort_keys=True), encoding="utf-8")

    fit_rows = load_existing_rows(out_dir / FIT_CSV) if args.skip_existing else []
    profile_rows = load_existing_rows(out_dir / PROFILE_CSV) if args.skip_existing else []
    done = completed_keys(fit_rows) if args.skip_existing else set()
    fit_id = int(max([0] + [int(r.get("fit_id", 0)) for r in fit_rows]))

    assets = load_real_assets(args)

    for asset in assets:
        asset_model_variants = variants_for_year(int(asset["year"]), model_variants)
        if not asset_model_variants:
            print(f"Skipping {asset['day']}: no requested variants are enabled for year {asset['year']}", flush=True)
            continue
        for model_variant in asset_model_variants:
            spec = variant_spec(model_variant)
            key = (int(asset["year"]), int(asset["day_idx"]), str(model_variant))
            if args.skip_existing and key in done:
                print(f"Skipping existing ok fit: {key}", flush=True)
                continue
            fit_id += 1
            print("\n" + "-" * 96, flush=True)
            print(f"fit_id={fit_id} variant={model_variant} day={asset['day']} full_grid={asset.get('n_grid', len(asset['grid_coords_np']))}", flush=True)
            print("-" * 96, flush=True)
            base = {
                "fit_id": int(fit_id),
                "status": "error",
                "data_kind": "real",
                "year": int(asset["year"]),
                "month": int(asset["month"]),
                "day_idx": int(asset["day_idx"]),
                "day": str(asset["day"]),
                "model_variant": str(model_variant),
                "model_family": str(spec["family"]),
                "model_label": str(spec["label"]),
                "smooth": float(spec["smooth"]) if pd.notna(spec["smooth"]) else np.nan,
                "gc_alpha": float(spec["gc_alpha"]) if pd.notna(spec["gc_alpha"]) else np.nan,
                "gc_beta": float(spec["gc_beta"]) if pd.notna(spec["gc_beta"]) else np.nan,
                "nugget_mode": "zero",
                "spec_name": VECCHIA_SPEC_NAME,
                "block_shape": f"{BLOCK_SHAPE[0]}x{BLOCK_SHAPE[1]}",
                "lag_pattern": f"{LAG_COUNTS[0]}/{LAG_COUNTS[1]}/{LAG_COUNTS[2]}",
            }
            try:
                row, beta, lat_mean = fit_full_asset(
                    asset=asset,
                    spec=spec,
                    init_physical=DEFAULT_REAL_INIT_PHYSICAL,
                    reference_advec_lon_abs=float(args.real_reference_advec_lon_abs),
                    fit_id=int(fit_id),
                    device=device,
                    args=args,
                )
                est = {k: float(row[f"est_{k}"]) for k in P_LABELS}
                prof, spectral_stats = compute_spectral_profile(
                    asset=asset,
                    est=est,
                    beta=beta,
                    lat_mean=lat_mean,
                    spec=spec,
                    fit_id=int(fit_id),
                    device=device,
                    args=args,
                )
                row.update(spectral_stats)
                fit_rows.append(clean_json_value(row))
                profile_rows.extend(clean_json_value(r) for r in prof)
                append_jsonl(out_dir / "st_corridor_spectral_all_fits.jsonl", row)
                print(
                    pd.Series(
                        {
                            k: row.get(k)
                            for k in [
                                "day",
                                "model_label",
                                "loss",
                                "fit_s",
                                "est_sigmasq",
                                "est_range_lat",
                                "est_range_lon",
                                "est_range_time",
                                "est_advec_lat",
                                "est_advec_lon",
                                "est_nugget",
                                "spectral_global_scale",
                                "spectral_data_over_expected_mean",
                                "spectral_expected_over_continuous_mean",
                                "spectral_ratio_var_after_profile",
                            ]
                        }
                    ).to_string(),
                    flush=True,
                )
            except Exception as exc:
                row = {**base, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(limit=10)}
                fit_rows.append(clean_json_value(row))
                append_jsonl(out_dir / "st_corridor_spectral_all_fits.jsonl", row)
                print(f"ERROR: {row['error']}", flush=True)
                traceback.print_exc()
            if int(args.summary_every) > 0 and fit_id % int(args.summary_every) == 0:
                refresh_outputs(out_dir, fit_rows, profile_rows, top_plot_dir)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    refresh_outputs(out_dir, fit_rows, profile_rows, top_plot_dir)
    print("\nDone.", flush=True)
    print("fits:", out_dir / FIT_CSV, flush=True)
    print("profiles:", out_dir / PROFILE_CSV, flush=True)
    print(
        "marginal time-averaged spatial I/Ediag monthly ratio plot:",
        out_dir / "marginal_timeavg_spatial_monthly_I_over_Ediag_profile_sigma_ratio.png",
        flush=True,
    )
    print("directional monthly plots:", out_dir / "monthly_average_plots", flush=True)
    print("ratio band table:", out_dir / BAND_TABLE_CSV, flush=True)


if __name__ == "__main__":
    main()
