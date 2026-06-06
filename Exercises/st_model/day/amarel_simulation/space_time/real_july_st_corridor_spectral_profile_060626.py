#!/usr/bin/env python3
"""
Space-time corridor Vecchia fit + multivariate spectral profile diagnostic.

This is the ST analogue of the pure-space spline-smooth spectral diagnostic:

  1. Fit the 4x4 corridor Vecchia model with lag pattern 6/4/3.
  2. Keep the full spatial grid; no max-min prefix thinning is applied.
  3. Build an 8-variate Fourier vector J(omega) for the eight July daytime slots.
  4. Use the fitted covariance, the per-hour missing masks, and a no-taper
     window to compute E[J(omega)J(omega)^*].
  5. Whiten by the 8x8 Cholesky factor at each spatial frequency.
  6. Profile out one global scale by mean(raw whitened power).

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
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

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
    RealDataCorridorWidth4x4Lag643SplineFit,
    _build_matern_spline_coeffs,
)

from fit_july2024_st_corridor_density_sweep_060426 import (
    DEFAULT_REAL_INIT_PHYSICAL,
    DELTA_LAT_BASE,
    DELTA_LON_BASE,
    DTYPE,
    P_LABELS,
    T_STEPS,
    backmap_params,
    clean_json_value,
    count_valid,
    load_real_assets,
    make_params_list,
    parse_day_idxs,
    parse_float_tokens,
    parse_int_tokens,
    parse_pair,
    physical_to_log_phi,
    resolve_device,
    save_rows,
)


ROUND_DECIMALS = 6
FIT_CSV = "st_corridor_spectral_all_fits.csv"
PROFILE_CSV = "st_corridor_spectral_profiles.csv"


def default_output_root() -> Path:
    if Path("/home/jl2815").exists():
        return Path("/home/jl2815/tco/exercise_output/summer/st_corridor_spectral_profile_060626")
    return Path("/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/summer_26/st_corridor_spectral_profile_060626")


def code_float(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(clean_json_value(row), sort_keys=True) + "\n")


def no_taper(u, n1: int, n2: int):
    u1, _ = u
    return torch.ones((int(n1), int(n2)), dtype=torch.float64, device=u1.device)


def raw_params_from_est(est: dict[str, float]) -> torch.Tensor:
    raw = physical_to_log_phi({k: float(est[k]) for k in P_LABELS})
    return torch.tensor(raw, dtype=torch.float64)


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
    smooth: float,
    spline_n_points: int,
    spline_r_max: float,
) -> torch.Tensor:
    phi1, phi2, phi3, phi4 = torch.exp(params[0:4])
    advec_lat = params[4]
    advec_lon = params[5]
    nugget = torch.exp(params[6])
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
    cov = sigmasq * matern_corr_spline_torch(scaled, smooth, spline_n_points, spline_r_max)
    zero = (torch.abs(u_lat) < 1e-10) & (torch.abs(u_lon) < 1e-10) & (torch.abs(t_lag) < 1e-10)
    return torch.where(zero, cov + nugget, cov)


def expected_periodogram_raw_st(
    params: torch.Tensor,
    smooth: float,
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
            smooth,
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


def profile_rows_from_power(
    power_shifted: np.ndarray,
    scale: float,
    year: int,
    month: int,
    day_idx: int,
    day: str,
    smooth: float,
    fit_id: int,
    est: dict[str, float],
    n_bins: int,
    delta_lat: float,
    delta_lon: float,
) -> list[dict[str, Any]]:
    n1, n2, p_time = power_shifted.shape
    ratio = power_shifted / float(scale)
    freq_lat = np.fft.fftshift(np.fft.fftfreq(n1, d=float(delta_lat)))
    freq_lon = np.fft.fftshift(np.fft.fftfreq(n2, d=float(delta_lon)))
    g_lat, g_lon = np.meshgrid(freq_lat, freq_lon, indexing="ij")
    ii, jj = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
    ci, cj = n1 // 2, n2 // 2

    cases = {
        "radial": (np.sqrt(g_lat**2 + g_lon**2), np.ones((n1, n2), dtype=bool)),
        "latitude_ns": (np.abs(g_lat), np.abs(jj - cj) <= 0),
        "longitude_ew": (np.abs(g_lon), np.abs(ii - ci) <= 0),
        "diagonal_ne_sw": (
            np.sqrt(g_lat**2 + g_lon**2),
            np.abs((ii - ci) / max(ci, 1) - (jj - cj) / max(cj, 1)) <= (1.0 / max(ci, cj, 1)),
        ),
    }
    rows: list[dict[str, Any]] = []
    for direction, (x_grid, mask2d) in cases.items():
        x = np.repeat(x_grid[mask2d].reshape(-1), p_time)
        y = ratio[mask2d, :].reshape(-1)
        finite = np.isfinite(x) & np.isfinite(y) & (x >= 0)
        x = x[finite]
        y = y[finite]
        if len(y) == 0:
            continue
        xmax = float(np.nanmax(x))
        if xmax <= 0:
            bins = np.array([0.0, 1.0])
        else:
            bins = np.linspace(0.0, xmax, int(n_bins) + 1)
        idx = np.digitize(x, bins, right=False) - 1
        idx = np.clip(idx, 0, len(bins) - 2)
        for b in range(len(bins) - 1):
            vals = y[idx == b]
            if vals.size == 0:
                continue
            rows.append(
                {
                    "fit_id": int(fit_id),
                    "year": int(year),
                    "month": int(month),
                    "day_idx": int(day_idx),
                    "day": day,
                    "smooth": float(smooth),
                    "direction": direction,
                    "bin_idx": int(b),
                    "k_min": float(bins[b]),
                    "k_max": float(bins[b + 1]),
                    "k_mid": float(0.5 * (bins[b] + bins[b + 1])),
                    "ratio_mean": float(np.nanmean(vals)),
                    "ratio_median": float(np.nanmedian(vals)),
                    "ratio_p10": float(np.nanquantile(vals, 0.10)),
                    "ratio_p90": float(np.nanquantile(vals, 0.90)),
                    "n": int(vals.size),
                    "global_scale": float(scale),
                    "profile_sigmasq": float(est["sigmasq"] * scale),
                }
            )
    return rows


def compute_spectral_profile(
    asset: dict[str, Any],
    est: dict[str, float],
    beta: torch.Tensor,
    lat_mean: float,
    smooth: float,
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
            smooth=float(smooth),
            n1=int(n1),
            n2=int(n2),
            p_time=int(p_time),
            taper_autocorr_grid=taper_auto,
            delta_lat=float(args.delta_lat),
            delta_lon=float(args.delta_lon),
            spline_n_points=int(args.spline_n_points),
            spline_r_max=float(args.spline_r_max),
        )
        chol = safe_cholesky(expected, p_time=int(p_time))
        z = torch.linalg.solve_triangular(chol, j_vec.unsqueeze(-1), upper=False)
        power = (z.abs() ** 2).squeeze(-1)
    power_shifted = np.fft.fftshift(power.detach().cpu().numpy(), axes=(0, 1))
    raw = power_shifted.reshape(-1)
    scale = float(np.nanmean(raw))
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"Invalid profile scale: {scale}")
    rows = profile_rows_from_power(
        power_shifted=power_shifted,
        scale=scale,
        year=int(asset["year"]),
        month=int(asset["month"]),
        day_idx=int(asset["day_idx"]),
        day=str(asset["day"]),
        smooth=float(smooth),
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
    }
    del slices, j_vec, taper_auto, expected, chol, z, power
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, stats


def fit_full_asset(
    asset: dict[str, Any],
    smooth: float,
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
    params_list = make_params_list(init_physical, dtype=DTYPE, device=device)

    model = RealDataCorridorWidth4x4Lag643SplineFit(
        smooth=float(smooth),
        input_map=source_map,
        grid_coords=grid_coords_np,
        lag1_lon_offset=float(reference_advec_lon_abs),
        daily_stride=int(args.daily_stride),
        target_chunk_size=int(args.target_chunk_size),
        min_target_points=int(args.min_target_points),
        spline_n_points=int(args.spline_n_points),
        spline_r_max=float(args.spline_r_max),
    )

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
    est = backmap_params(out)
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
        "smooth": float(smooth),
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
        "smooth_kernel": "spline",
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
                top_plot_dir.mkdir(parents=True, exist_ok=True)
                plot_parameter_monthly_summary(summary, top_plot_dir / "st_corridor_parameter_monthly_summary.png")
    if not profile_df.empty:
        monthly = make_profile_monthly_summary(profile_df)
        save_rows(out_dir / "st_corridor_spectral_monthly_summary.csv", monthly)
        plot_profile_monthly_summary(monthly, out_dir / "st_corridor_spectral_monthly_profile.png")
        if top_plot_dir is not None:
            top_plot_dir.mkdir(parents=True, exist_ok=True)
            plot_profile_monthly_summary(monthly, top_plot_dir / "st_corridor_spectral_monthly_profile.png")

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
        for keys, sub in ok.groupby(["year", "month", "smooth"], dropna=False):
            vals = pd.to_numeric(sub[col], errors="coerce").dropna().to_numpy(dtype=float)
            if vals.size == 0:
                continue
            rows.append(
                {
                    "year": int(keys[0]),
                    "month": int(keys[1]),
                    "smooth": float(keys[2]),
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
        for smooth, sub in sub_param.groupby("smooth", dropna=False):
            sub = sub.sort_values(["year", "month"])
            labels = [f"{int(y)}-{int(m):02d}" for y, m in zip(sub["year"], sub["month"])]
            x = np.arange(len(labels))
            ax.plot(x, sub["median"], marker="o", linewidth=1.8, label=f"smooth={float(smooth):g}")
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
    group_cols = ["year", "month", "smooth", "direction", "bin_idx"]
    for keys, sub in profile.groupby(group_cols, dropna=False):
        vals = pd.to_numeric(sub["ratio_mean"], errors="coerce").dropna().to_numpy(dtype=float)
        if vals.size == 0:
            continue
        rows.append(
            {
                "year": int(keys[0]),
                "month": int(keys[1]),
                "smooth": float(keys[2]),
                "direction": str(keys[3]),
                "bin_idx": int(keys[4]),
                "k_mid": float(np.nanmean(pd.to_numeric(sub["k_mid"], errors="coerce"))),
                "n_days": int(vals.size),
                "ratio_mean": float(np.mean(vals)),
                "ratio_median": float(np.median(vals)),
                "ratio_p10": float(np.quantile(vals, 0.10)),
                "ratio_p90": float(np.quantile(vals, 0.90)),
                "global_scale_mean": float(np.nanmean(pd.to_numeric(sub["global_scale"], errors="coerce"))),
                "profile_sigmasq_mean": float(np.nanmean(pd.to_numeric(sub["profile_sigmasq"], errors="coerce"))),
            }
        )
    return pd.DataFrame(rows)


def plot_profile_monthly_summary(monthly: pd.DataFrame, path: Path) -> None:
    if monthly.empty:
        return
    directions = ["radial", "latitude_ns", "longitude_ew", "diagonal_ne_sw"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for ax, direction in zip(axes.ravel(), directions):
        sub_dir = monthly[monthly["direction"] == direction].copy()
        for (year, smooth), sub in sub_dir.groupby(["year", "smooth"], dropna=False):
            sub = sub.sort_values("k_mid")
            label = f"{int(year)}, s={float(smooth):g}"
            ax.plot(sub["k_mid"], sub["ratio_mean"], linewidth=1.6, label=label)
        ax.axhline(1.0, color="0.25", linestyle="--", linewidth=1.0)
        ax.set_title(direction)
        ax.set_xlabel("spatial frequency")
        ax.set_yscale("log")
        ax.set_ylim(0.2, 5.0)
        ax.grid(alpha=0.25, which="both")
    axes[0, 0].set_ylabel("profiled whitened power")
    axes[1, 0].set_ylabel("profiled whitened power")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("ST Vecchia spectral diagnostic: monthly mean after 8x8 whitening + global profile-out")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit full-grid ST corridor Vecchia and compute 8x8-whitened spectral profiles.")
    parser.add_argument("--smooths", nargs="+", default=["0.35", "0.4", "0.5"])
    parser.add_argument("--days", default="0,30", help="'0,30' means July day_idx 0..29.")
    parser.add_argument("--real-years", nargs="+", default=["2022", "2023", "2024", "2025"])
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


def completed_keys(rows: list[dict[str, Any]]) -> set[tuple[int, int, float]]:
    out = set()
    for row in rows:
        if str(row.get("status", "")) == "ok":
            out.add((int(row["year"]), int(row["day_idx"]), round(float(row["smooth"]), 6)))
    return out


def main() -> None:
    args = build_arg_parser().parse_args()
    device = resolve_device(args)
    smooths = parse_float_tokens(args.smooths)
    years = parse_int_tokens(args.real_years)
    out_dir = args.out_dir or default_output_root()
    out_dir.mkdir(parents=True, exist_ok=True)
    top_plot_dir = args.monthly_out_dir
    if top_plot_dir is not None:
        top_plot_dir.mkdir(parents=True, exist_ok=True)

    print("SRC:", SRC, flush=True)
    print("device:", device, flush=True)
    print("out_dir:", out_dir, flush=True)
    print("years:", years, "smooths:", smooths, "days:", parse_day_idxs(args.days), flush=True)
    print("region:", parse_pair(args.lat_range, float), parse_pair(args.lon_range, float), flush=True)

    run_config = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "src": str(SRC),
        "device": str(device),
        "args": clean_json_value(vars(args)),
        "smooths": smooths,
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

    for smooth in smooths:
        for asset in assets:
            key = (int(asset["year"]), int(asset["day_idx"]), round(float(smooth), 6))
            if args.skip_existing and key in done:
                print(f"Skipping existing ok fit: {key}", flush=True)
                continue
            fit_id += 1
            print("\n" + "-" * 96, flush=True)
            print(f"fit_id={fit_id} smooth={smooth} day={asset['day']} full_grid={asset['n_grid']}", flush=True)
            print("-" * 96, flush=True)
            base = {
                "fit_id": int(fit_id),
                "status": "error",
                "data_kind": "real",
                "year": int(asset["year"]),
                "month": int(asset["month"]),
                "day_idx": int(asset["day_idx"]),
                "day": str(asset["day"]),
                "smooth": float(smooth),
                "spec_name": VECCHIA_SPEC_NAME,
                "block_shape": f"{BLOCK_SHAPE[0]}x{BLOCK_SHAPE[1]}",
                "lag_pattern": f"{LAG_COUNTS[0]}/{LAG_COUNTS[1]}/{LAG_COUNTS[2]}",
            }
            try:
                row, beta, lat_mean = fit_full_asset(
                    asset=asset,
                    smooth=float(smooth),
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
                    smooth=float(smooth),
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
                                "smooth",
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
    print("monthly plot:", out_dir / "st_corridor_spectral_monthly_profile.png", flush=True)


if __name__ == "__main__":
    main()
