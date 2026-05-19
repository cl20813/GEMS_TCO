"""
sim_vecchia_irregular_realpattern_colv3fill_heads_compare_050726.py

Amarel simulation for real-data-like irregular GEMS observation patterns.

Question:
  With known truth, does reverse-L conditioning geometry improve RMSRE relative
  to Hybrid Lean when conditioning sizes are matched around 40?

Design:
  - high-resolution latent field
  - real Source_Latitude/Source_Longitude matching
  - regular grid only defines ordering and reverse-L scan geometry
  - covariance offsets for both Hybrid and Column are real source-location offsets
  - Column missing neighbors are skipped and later reverse-L candidates are scanned
    until the per-lag cap is filled when possible

Models:
  1. Hybrid Lean exact-location, nominal m=41
  2. Column V3 Batched, head_right_cols=3, nominal tail m=42
  3. Column V3 Batched, head_right_cols=0, nominal tail m=42
"""

from __future__ import annotations

import argparse
import gc
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.fft
from sklearn.neighbors import BallTree

AMAREL_SRC = "/home/jl2815/tco"
LOCAL_SRC = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
SRC = AMAREL_SRC if os.path.exists(AMAREL_SRC) else LOCAL_SRC
sys.path.insert(0, SRC)

from GEMS_TCO import configuration as config
from GEMS_TCO import orderings as _orderings
from GEMS_TCO.data_loader import load_data_dynamic_processed
from GEMS_TCO.kernel_vecchia_col_batch import ReverseLColumnVecchiaFitBatch
from GEMS_TCO.vecchia_candidate.kernels_vecchia_hybrid import HybridVecchiaFit


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
ROUND_DECIMALS = 4
DELTA_LAT_BASE = 0.044
DELTA_LON_BASE = 0.063
T_STEPS = 8
SMOOTH = 0.5

P_LABELS = ["sigmasq", "range_lat", "range_lon", "range_time", "advec_lat", "advec_lon", "nugget"]
SPATIAL_KEYS = ["sigmasq", "range_lat", "range_lon"]
ADVECTION_KEYS = ["advec_lat", "advec_lon"]

TRUE_DICT = {
    "sigmasq": 10.0,
    "range_lat": 0.30,
    "range_lon": 0.40,
    "range_time": 2.0,
    "advec_lat": 0.08,
    "advec_lon": -0.16,
    "nugget": 2.5,
}

HYBRID_SPEC = {
    "model": "Hybrid_Lean_L08F04_C4F03_Op0p063_exactloc",
    "limit_A": 20,
    "lag1_local_count": 8,
    "lag1_fresh_count": 4,
    "lag2_local_count": 4,
    "lag2_fresh_count": 3,
    "daily_stride": 2,
    "lag1_lon_offset": 0.063,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_range(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",")]


def parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def true_to_log_params(d: dict[str, float]) -> list[float]:
    phi2 = 1.0 / d["range_lon"]
    phi1 = d["sigmasq"] * phi2
    phi3 = (d["range_lon"] / d["range_lat"]) ** 2
    phi4 = (d["range_lon"] / d["range_time"]) ** 2
    return [
        np.log(phi1),
        np.log(phi2),
        np.log(phi3),
        np.log(phi4),
        d["advec_lat"],
        d["advec_lon"],
        np.log(d["nugget"]),
    ]


def backmap_params(out_params) -> dict[str, float]:
    p = [float(x.item() if isinstance(x, torch.Tensor) else x) for x in out_params[:7]]
    phi2 = np.exp(p[1])
    phi3 = np.exp(p[2])
    phi4 = np.exp(p[3])
    rlon = 1.0 / phi2
    return {
        "sigmasq": np.exp(p[0]) / phi2,
        "range_lat": rlon / phi3**0.5,
        "range_lon": rlon,
        "range_time": rlon / phi4**0.5,
        "advec_lat": p[4],
        "advec_lon": p[5],
        "nugget": np.exp(p[6]),
    }


def make_random_init(rng: np.random.Generator, true_log: list[float], init_noise: float) -> list[float]:
    noisy = list(true_log)
    for idx in [0, 1, 2, 3, 6]:
        noisy[idx] = true_log[idx] + rng.uniform(-init_noise, init_noise)
    for idx in [4, 5]:
        scale = max(abs(true_log[idx]), 0.05)
        noisy[idx] = true_log[idx] + rng.uniform(-2 * scale, 2 * scale)
    return noisy


def rmsre_for_keys(est: dict[str, float], truth: dict[str, float], keys: list[str], zero_thresh: float = 0.01) -> float:
    vals = []
    for key in keys:
        tv = truth[key]
        if abs(tv) < zero_thresh:
            continue
        vals.append(((est[key] - tv) / abs(tv)) ** 2)
    return float(np.sqrt(np.mean(vals))) if vals else float("nan")


def calculate_metrics(out_params, truth: dict[str, float]):
    est = backmap_params(out_params)
    metrics = {
        "overall_rmsre": rmsre_for_keys(est, truth, P_LABELS),
        "spatial_rmsre": rmsre_for_keys(est, truth, SPATIAL_KEYS),
        "advec_rmsre": rmsre_for_keys(est, truth, ADVECTION_KEYS),
        "range_time_re": abs(est["range_time"] - truth["range_time"]) / abs(truth["range_time"]),
        "nugget_re": abs(est["nugget"] - truth["nugget"]) / abs(truth["nugget"]),
    }
    for key in P_LABELS:
        denom = abs(truth[key]) if abs(truth[key]) >= 0.01 else 1.0
        metrics[f"{key}_re"] = abs(est[key] - truth[key]) / denom
    return metrics, est


def round_df(df: pd.DataFrame, digits: int = ROUND_DECIMALS) -> pd.DataFrame:
    out = df.copy()
    cols = out.select_dtypes(include=[np.number]).columns
    out[cols] = out[cols].round(digits)
    return out


def save_csv_rounded(df: pd.DataFrame, path: Path) -> None:
    round_df(df).to_csv(path, index=False, float_format=f"%.{ROUND_DECIMALS}f")


def get_covariance_on_grid(lx, ly, lt, params):
    params = torch.clamp(params, min=-15.0, max=15.0)
    phi1, phi2, phi3, phi4 = (torch.exp(params[i]) for i in range(4))
    u_lat = lx - params[4] * lt
    u_lon = ly - params[5] * lt
    dist = torch.sqrt(u_lat.pow(2) * phi3 + u_lon.pow(2) + lt.pow(2) * phi4 + 1e-8)
    return (phi1 / phi2) * torch.exp(-dist * phi2)


def build_grid_coords(lat_range: list[float], lon_range: list[float]):
    lats = torch.arange(min(lat_range), max(lat_range) + 0.0001, DELTA_LAT_BASE, device=DEVICE, dtype=DTYPE)
    lons = torch.arange(lon_range[0], lon_range[1] + 0.0001, DELTA_LON_BASE, device=DEVICE, dtype=DTYPE)
    lats = torch.round(lats * 10000) / 10000
    lons = torch.round(lons * 10000) / 10000
    g_lat, g_lon = torch.meshgrid(lats, lons, indexing="ij")
    return lats, lons, torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)


def build_high_res_grid(lat_range: list[float], lon_range: list[float], lat_factor: int, lon_factor: int):
    dlat = DELTA_LAT_BASE / lat_factor
    dlon = DELTA_LON_BASE / lon_factor
    lats = torch.arange(min(lat_range) - 0.1, max(lat_range) + 0.1, dlat, device=DEVICE, dtype=DTYPE)
    lons = torch.arange(lon_range[0] - 0.1, lon_range[1] + 0.1, dlon, device=DEVICE, dtype=DTYPE)
    return lats, lons, dlat, dlon


def generate_field_values(lats_hr, lons_hr, t_steps: int, params, dlat: float, dlon: float):
    cpu = torch.device("cpu")
    f32 = torch.float32
    nx, ny, nt = len(lats_hr), len(lons_hr), t_steps
    px, py, pt = 2 * nx, 2 * ny, 2 * nt

    lx = torch.arange(px, device=cpu, dtype=f32) * dlat
    lx[px // 2 :] -= px * dlat
    ly = torch.arange(py, device=cpu, dtype=f32) * dlon
    ly[py // 2 :] -= py * dlon
    lt = torch.arange(pt, device=cpu, dtype=f32)
    lt[pt // 2 :] -= pt

    params_cpu = params.cpu().float()
    lx_g, ly_g, lt_g = torch.meshgrid(lx, ly, lt, indexing="ij")
    cov = get_covariance_on_grid(lx_g, ly_g, lt_g, params_cpu)
    spec = torch.fft.fftn(cov)
    spec.real = torch.clamp(spec.real, min=0)
    noise = torch.fft.fftn(torch.randn(px, py, pt, device=cpu, dtype=f32))
    field = torch.fft.ifftn(torch.sqrt(spec.real) * noise).real[:nx, :ny, :nt]
    return field.to(dtype=DTYPE, device=DEVICE)


def apply_step3_1to1(src_np_valid: np.ndarray, grid_coords_np: np.ndarray, grid_tree: BallTree) -> np.ndarray:
    n_grid = len(grid_coords_np)
    if len(src_np_valid) == 0:
        return np.full(n_grid, -1, dtype=np.int64)

    dist_to_cell, cell_for_obs = grid_tree.query(np.radians(src_np_valid), k=1)
    dist_to_cell = dist_to_cell.flatten()
    cell_for_obs = cell_for_obs.flatten()

    assignment = np.full(n_grid, -1, dtype=np.int64)
    best_dist = np.full(n_grid, np.inf)
    for obs_i, (cell_j, dist) in enumerate(zip(cell_for_obs, dist_to_cell)):
        if dist < best_dist[cell_j]:
            assignment[cell_j] = obs_i
            best_dist[cell_j] = dist

    filled = assignment >= 0
    if filled.any():
        win_obs = assignment[filled]
        lat_diff = np.abs(src_np_valid[win_obs, 0] - grid_coords_np[filled, 0])
        lon_diff = np.abs(src_np_valid[win_obs, 1] - grid_coords_np[filled, 1])
        too_far = (lat_diff > DELTA_LAT_BASE / 2) | (lon_diff > DELTA_LON_BASE / 2)
        assignment[np.where(filled)[0][too_far]] = -1
    return assignment


def precompute_mapping_indices(ref_day_map, lats_hr, lons_hr, grid_coords, sorted_keys):
    hr_lat_g, hr_lon_g = torch.meshgrid(lats_hr, lons_hr, indexing="ij")
    hr_coords_np = torch.stack([hr_lat_g.flatten(), hr_lon_g.flatten()], dim=1).detach().cpu().numpy()
    hr_tree = BallTree(np.radians(hr_coords_np), metric="haversine")
    del hr_coords_np, hr_lat_g, hr_lon_g
    gc.collect()

    grid_coords_np = grid_coords.detach().cpu().numpy()
    n_grid = len(grid_coords_np)
    grid_tree = BallTree(np.radians(grid_coords_np), metric="haversine")

    step3_assignment_per_t = []
    hr_idx_per_t = []
    src_locs_per_t = []
    for key in sorted_keys:
        df = ref_day_map.get(key)
        if df is None or len(df) == 0 or not {"Source_Latitude", "Source_Longitude"}.issubset(df.columns):
            step3_assignment_per_t.append(np.full(n_grid, -1, dtype=np.int64))
            hr_idx_per_t.append(torch.zeros(0, dtype=torch.long, device=DEVICE))
            src_locs_per_t.append(torch.zeros((0, 2), dtype=DTYPE, device=DEVICE))
            continue

        src_np = df[["Source_Latitude", "Source_Longitude"]].values
        valid_mask = ~np.isnan(src_np).any(axis=1)
        src_np_valid = src_np[valid_mask]
        assignment = apply_step3_1to1(src_np_valid, grid_coords_np, grid_tree)
        step3_assignment_per_t.append(assignment)

        if len(src_np_valid) > 0:
            _, hr_idx = hr_tree.query(np.radians(src_np_valid), k=1)
            hr_idx_per_t.append(torch.tensor(hr_idx.flatten(), device=DEVICE))
        else:
            hr_idx_per_t.append(torch.zeros(0, dtype=torch.long, device=DEVICE))
        src_locs_per_t.append(torch.tensor(src_np_valid, device=DEVICE, dtype=DTYPE))

    return step3_assignment_per_t, hr_idx_per_t, src_locs_per_t


def assemble_source_map(field, step3_assignment_per_t, hr_idx_per_t, src_locs_per_t, sorted_keys, grid_coords, true_params, t_offset=21.0):
    nugget_std = torch.sqrt(torch.exp(true_params[6]))
    n_grid = grid_coords.shape[0]
    field_flat = field.reshape(-1, T_STEPS)
    source_map = {}

    for t_idx, key in enumerate(sorted_keys):
        t_val = float(t_offset + t_idx)
        assign = step3_assignment_per_t[t_idx]
        hr_idx = hr_idx_per_t[t_idx]
        src_locs = src_locs_per_t[t_idx]
        n_valid = hr_idx.shape[0]

        dummy = torch.zeros(7, device=DEVICE, dtype=DTYPE)
        if t_idx > 0:
            dummy[t_idx - 1] = 1.0

        rows = torch.full((n_grid, 11), float("nan"), device=DEVICE, dtype=DTYPE)
        rows[:, 3] = t_val
        rows[:, 4:] = dummy.unsqueeze(0).expand(n_grid, -1)

        if n_valid > 0:
            gp_vals = field_flat[hr_idx, t_idx]
            sim_vals = gp_vals + torch.randn(n_valid, device=DEVICE, dtype=DTYPE) * nugget_std
            assign_t = torch.tensor(assign, device=DEVICE, dtype=torch.long)
            filled = assign_t >= 0
            win_obs = assign_t[filled]
            rows[filled, 0] = src_locs[win_obs, 0]
            rows[filled, 1] = src_locs[win_obs, 1]
            rows[filled, 2] = sim_vals[win_obs]

        source_map[key] = rows.detach()
    return source_map


def compute_grid_ordering(grid_coords, mm_cond_number: int):
    coords_np = grid_coords.detach().cpu().numpy()
    ord_mm = _orderings.maxmin_cpp(coords_np)
    nns = _orderings.find_nns_l2(locs=coords_np[ord_mm], max_nn=mm_cond_number)
    return ord_mm, nns


def count_valid(day_map: dict[str, torch.Tensor]) -> int:
    return sum(int((~torch.isnan(v[:, 2])).sum().item()) for v in day_map.values())


def template_diagnostics(model) -> dict[str, float]:
    batched = getattr(model, "Batched_Groups", None)
    if batched:
        group_sizes = np.asarray([int(g["target_idx"].shape[0]) for g in batched], dtype=np.int64)
        m_sizes = np.asarray([int(g["max_m"]) for g in batched], dtype=np.int64)
        return {
            "n_templates": np.nan,
            "n_batches": int(len(batched)),
            "largest_template_n": int(group_sizes.max()),
            "median_template_n": float(np.median(group_sizes)),
            "mean_template_n": float(group_sizes.mean()),
            "mean_m_by_template": float(m_sizes.mean()),
            "median_m_by_template": float(np.median(m_sizes)),
            "max_m_by_template": int(m_sizes.max()),
        }
    groups = getattr(model, "Grouped_Batches", [])
    if not groups:
        return {
            "n_templates": 0,
            "n_batches": 0,
            "largest_template_n": 0,
            "median_template_n": 0.0,
            "mean_template_n": 0.0,
            "mean_m_by_template": 0.0,
            "median_m_by_template": 0.0,
            "max_m_by_template": 0,
        }
    group_sizes = np.asarray([int(g["target_idx"].shape[0]) for g in groups], dtype=np.int64)
    m_sizes = np.asarray([int(g["offsets"].shape[0]) for g in groups], dtype=np.int64)
    return {
        "n_templates": int(len(groups)),
        "n_batches": np.nan,
        "largest_template_n": int(group_sizes.max()),
        "median_template_n": float(np.median(group_sizes)),
        "mean_template_n": float(group_sizes.mean()),
        "mean_m_by_template": float(m_sizes.mean()),
        "median_m_by_template": float(np.median(m_sizes)),
        "max_m_by_template": int(m_sizes.max()),
    }


def fit_hybrid(source_map_ord, nns_grid, initial_vals, args):
    params = [torch.tensor([v], device=DEVICE, dtype=DTYPE, requires_grad=True) for v in initial_vals]
    model = HybridVecchiaFit(
        smooth=SMOOTH,
        input_map=source_map_ord,
        nns_map=nns_grid,
        mm_cond_number=args.mm_cond_number,
        nheads=0,
        limit_A=HYBRID_SPEC["limit_A"],
        limit_B_local=HYBRID_SPEC["lag1_local_count"],
        limit_C_local=HYBRID_SPEC["lag2_local_count"],
        daily_stride=HYBRID_SPEC["daily_stride"],
        spatial_coords=None,
        lag1_lon_offset=HYBRID_SPEC["lag1_lon_offset"],
        lag1_fresh_count=HYBRID_SPEC["lag1_fresh_count"],
        lag2_fresh_count=HYBRID_SPEC["lag2_fresh_count"],
    )
    t0 = time.time()
    model.precompute_conditioning_sets()
    pre_s = time.time() - t0
    opt = model.set_optimizer(params, lr=1.0, max_iter=args.lbfgs_eval, max_eval=args.lbfgs_eval, history_size=args.lbfgs_hist)
    t1 = time.time()
    out, fit_iter = model.fit_vecc_lbfgs(params, opt, max_steps=args.lbfgs_steps, grad_tol=1e-5)
    fit_s = time.time() - t1
    metrics, est = calculate_metrics(out, TRUE_DICT)
    row = {
        "model": HYBRID_SPEC["model"],
        "kernel": "hybrid_lean_exactloc",
        "head_right_cols": np.nan,
        "coords_used": "real_source",
        "loss": float(out[-1]),
        "fit_iter": int(fit_iter),
        "precompute_s": pre_s,
        "fit_s": fit_s,
        "total_s": pre_s + fit_s,
        "total_conditioning_nominal": 41,
        "n_valid": count_valid(source_map_ord),
        "n_heads": int(model.Heads_data.shape[0]),
        "n_tails": sum(int(x.shape[0]) for x in [model.X_A, model.X_AB, model.X_ABC] if x is not None),
        "n_templates": np.nan,
        **metrics,
        **{f"est_{k}": v for k, v in est.items()},
    }
    del model, params, opt
    return row


def fit_column(source_map_ord, ordered_grid_coords_np, initial_vals, args, head_right_cols: int):
    params = [torch.tensor([v], device=DEVICE, dtype=DTYPE, requires_grad=True) for v in initial_vals]
    model_name = f"ColumnV3Batched_Up3_Right3_Down14_Lag2_head{head_right_cols}_realloc"
    model = ReverseLColumnVecchiaFitBatch(
        smooth=SMOOTH,
        input_map=source_map_ord,
        mm_cond_number=args.mm_cond_number,
        grid_coords=ordered_grid_coords_np,
        head_right_cols=head_right_cols,
        above_count=3,
        right_col_count=3,
        per_lag_conditioning_count=14,
        lag_count=2,
        include_lag_self=False,
        target_chunk_size=args.column_chunk_size,
        use_data_coords_for_offsets=True,
    )
    t0 = time.time()
    model.precompute_conditioning_sets()
    pre_s = time.time() - t0
    diag = template_diagnostics(model)
    opt = model.set_optimizer(params, lr=1.0, max_iter=args.lbfgs_eval, max_eval=args.lbfgs_eval, history_size=args.lbfgs_hist)
    t1 = time.time()
    out, fit_iter = model.fit_vecc_lbfgs(params, opt, max_steps=args.lbfgs_steps, grad_tol=1e-5)
    fit_s = time.time() - t1
    metrics, est = calculate_metrics(out, TRUE_DICT)
    row = {
        "model": model_name,
        "kernel": "column_reverse_l_v3_batched_realloc",
        "head_right_cols": int(head_right_cols),
        "coords_used": "real_source_offsets_regular_grid_scan",
        "loss": float(out[-1]),
        "fit_iter": int(fit_iter),
        "precompute_s": pre_s,
        "fit_s": fit_s,
        "total_s": pre_s + fit_s,
        "total_conditioning_nominal": 42,
        "n_valid": count_valid(source_map_ord),
        "n_heads": int(model.Heads_data.shape[0]),
        "n_tails": int(model.n_tails),
        **diag,
        **metrics,
        **{f"est_{k}": v for k, v in est.items()},
    }
    del model, params, opt
    return row


def make_model_summary(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df.get("error", "").fillna("") == ""].copy()
    if ok.empty:
        return pd.DataFrame()
    metric_cols = [
        "loss",
        "overall_rmsre",
        "spatial_rmsre",
        "advec_rmsre",
        "range_time_re",
        "nugget_re",
        "precompute_s",
        "fit_s",
        "total_s",
        "n_templates",
        "n_heads",
        "n_tails",
        "mean_m_by_template",
        "max_m_by_template",
    ]
    rows = []
    for keys, sub in ok.groupby(["model", "kernel", "head_right_cols"], dropna=False):
        row = {
            "model": keys[0],
            "kernel": keys[1],
            "head_right_cols": keys[2],
            "n": int(len(sub)),
            "total_conditioning": float(sub["total_conditioning_nominal"].median()),
        }
        for col in metric_cols:
            if col not in sub.columns:
                continue
            vals = sub[col].dropna().astype(float).values
            if len(vals) == 0:
                row[f"{col}_mean"] = np.nan
                row[f"{col}_p90_p10"] = np.nan
                continue
            p10, p90 = np.percentile(vals, [10, 90])
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_p90_p10"] = float(p90 - p10)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("overall_rmsre_mean")


def make_param_summary(df: pd.DataFrame) -> pd.DataFrame:
    ok = df[df.get("error", "").fillna("") == ""].copy()
    if ok.empty:
        return pd.DataFrame()
    rows = []
    for model, sub in ok.groupby("model"):
        for p in P_LABELS:
            vals = sub[f"est_{p}"].dropna().astype(float).values
            if len(vals) == 0:
                continue
            tv = TRUE_DICT[p]
            denom = abs(tv) if abs(tv) >= 0.01 else 1.0
            re = np.abs((vals - tv) / denom)
            p10, p90 = np.percentile(re, [10, 90])
            rows.append({
                "model": model,
                "parameter": p,
                "true": tv,
                "rmsre": float(np.sqrt(np.mean(re**2))),
                "mean_re": float(np.mean(re)),
                "median_re": float(np.median(re)),
                "p10_re": float(p10),
                "p90_re": float(p90),
                "p90_p10_re": float(p90 - p10),
                "estimate_mean": float(np.mean(vals)),
                "estimate_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            })
    return pd.DataFrame(rows).sort_values(["model", "parameter"])


def save_outputs(rows: list[dict], raw_csv: Path, model_csv: Path, param_csv: Path) -> None:
    df = pd.DataFrame(rows)
    save_csv_rounded(df, raw_csv)
    model_summary = make_model_summary(df)
    param_summary = make_param_summary(df)
    if not model_summary.empty:
        save_csv_rounded(model_summary, model_csv)
        print("\nRunning model summary")
        print(round_df(model_summary).to_string(index=False), flush=True)
    if not param_summary.empty:
        save_csv_rounded(param_summary, param_csv)
        print("\nRunning parameter summary")
        print(round_df(param_summary).to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--years", type=str, default="2024")
    parser.add_argument("--month", type=int, default=7)
    parser.add_argument("--day-idxs", type=str, default="2")
    parser.add_argument("--lat-range", type=parse_range, default=parse_range("-3,2"))
    parser.add_argument("--lon-range", type=parse_range, default=parse_range("121,131"))
    parser.add_argument("--lat-factor-hr", type=int, default=100)
    parser.add_argument("--lon-factor-hr", type=int, default=10)
    parser.add_argument("--mm-cond-number", type=int, default=100)
    parser.add_argument("--lbfgs-steps", type=int, default=5)
    parser.add_argument("--lbfgs-eval", type=int, default=15)
    parser.add_argument("--lbfgs-hist", type=int, default=10)
    parser.add_argument("--init-noise", type=float, default=0.25)
    parser.add_argument("--column-chunk-size", type=int, default=512)
    parser.add_argument("--out-dir", type=Path, default=Path("/home/jl2815/tco/exercise_output/estimates/day"))
    parser.add_argument("--out-prefix", type=str, default="sim_vecchia_irregular_realpattern_colv3fill_heads_compare_050726")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = args.out_dir / f"{args.out_prefix}_raw.csv"
    model_csv = args.out_dir / f"{args.out_prefix}_model_summary.csv"
    param_csv = args.out_dir / f"{args.out_prefix}_param_summary.csv"

    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    years = [y.strip() for y in args.years.split(",") if y.strip()]
    day_idxs = parse_int_list(args.day_idxs)

    print("SRC:", SRC, flush=True)
    print("DEVICE:", DEVICE, flush=True)
    print("torch:", torch.__version__, flush=True)
    print("args:", vars(args), flush=True)
    print("truth:", TRUE_DICT, flush=True)

    data_path = config.amarel_data_load_path if os.path.exists(config.amarel_data_load_path) else config.mac_data_load_path
    loader = load_data_dynamic_processed(data_path)

    lats_grid, lons_grid, grid_coords = build_grid_coords(args.lat_range, args.lon_range)
    n_grid = grid_coords.shape[0]
    print(f"Base grid: {len(lats_grid)} x {len(lons_grid)} = {n_grid:,}", flush=True)

    lats_hr, lons_hr, dlat_hr, dlon_hr = build_high_res_grid(
        args.lat_range, args.lon_range, args.lat_factor_hr, args.lon_factor_hr
    )
    print(f"High-res grid: {len(lats_hr)} x {len(lons_hr)} = {len(lats_hr) * len(lons_hr):,}", flush=True)

    print("Computing grid maxmin ordering...", flush=True)
    ord_grid, nns_grid = compute_grid_ordering(grid_coords, args.mm_cond_number)
    ordered_grid_coords_np = grid_coords[ord_grid].detach().cpu().numpy()

    pattern_bank = []
    for year in years:
        df_map, _, _, _ = loader.load_maxmin_ordered_data_bymonthyear(
            lat_lon_resolution=[1, 1],
            mm_cond_number=args.mm_cond_number,
            years_=[year],
            months_=[args.month],
            lat_range=args.lat_range,
            lon_range=args.lon_range,
            is_whittle=False,
        )
        all_keys = sorted(df_map.keys())
        yy = str(year)[2:]
        tco_path = Path(data_path) / f"pickle_{year}" / f"tco_grid_{yy}_{args.month:02d}.pkl"
        if tco_path.exists():
            with open(tco_path, "rb") as f:
                tco_map = pickle.load(f)
            print(f"Loaded tco_grid {tco_path} ({len(tco_map)} slots)", flush=True)
        else:
            tco_map = {}
            print(f"WARNING: missing tco_grid, falling back to df_map: {tco_path}", flush=True)

        for day_idx in day_idxs:
            day_keys = all_keys[day_idx * T_STEPS : (day_idx + 1) * T_STEPS]
            if len(day_keys) < T_STEPS:
                continue
            ref_day = {}
            for k in day_keys:
                suffix = k.split("_", 2)[-1]
                ref_day[k] = tco_map.get(suffix, df_map.get(k))
            s3, hr_idx, src = precompute_mapping_indices(ref_day, lats_hr, lons_hr, grid_coords, day_keys)
            valid_by_t = [int((a >= 0).sum()) for a in s3]
            pattern_bank.append({
                "year": year,
                "day_idx": day_idx,
                "day_keys": day_keys,
                "s3": s3,
                "hr_idx": hr_idx,
                "src": src,
                "valid_by_t": valid_by_t,
            })
            miss = [round(100 * (1 - v / n_grid), 2) for v in valid_by_t]
            print(f"Pattern year={year} day_idx={day_idx} valid={valid_by_t} missing%={miss}", flush=True)
        del df_map
        gc.collect()

    if not pattern_bank:
        raise RuntimeError("No real observation patterns were built.")

    true_log = true_to_log_params(TRUE_DICT)
    true_params = torch.tensor(true_log, device=DEVICE, dtype=DTYPE)
    rows = []

    for it in range(args.num_iters):
        print("\n" + "=" * 100, flush=True)
        print(f"Iteration {it + 1}/{args.num_iters}", flush=True)
        pattern = pattern_bank[it % len(pattern_bank)]
        iter_seed = args.seed + it
        set_seed(iter_seed)
        initial_vals = make_random_init(rng, true_log, args.init_noise)
        print(
            f"pattern={pattern['year']} day_idx={pattern['day_idx']} "
            f"initial={[round(x, 4) for x in initial_vals]}",
            flush=True,
        )

        field = generate_field_values(lats_hr, lons_hr, T_STEPS, true_params, dlat_hr, dlon_hr)
        source_map = assemble_source_map(
            field, pattern["s3"], pattern["hr_idx"], pattern["src"], pattern["day_keys"], grid_coords, true_params
        )
        del field
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        source_map_ord = {k: v[ord_grid].contiguous() for k, v in source_map.items()}
        del source_map

        fitters = [
            ("hybrid", lambda: fit_hybrid(source_map_ord, nns_grid, initial_vals, args)),
            ("column_head3", lambda: fit_column(source_map_ord, ordered_grid_coords_np, initial_vals, args, 3)),
            ("column_head0", lambda: fit_column(source_map_ord, ordered_grid_coords_np, initial_vals, args, 0)),
        ]

        for fit_name, fit_fn in fitters:
            print("\n--- fitting", fit_name, "---", flush=True)
            try:
                row = fit_fn()
                row.update({
                    "iter": it + 1,
                    "seed": iter_seed,
                    "pattern_year": pattern["year"],
                    "pattern_day_idx": pattern["day_idx"],
                    "error": "",
                })
                row.update({f"true_{k}": v for k, v in TRUE_DICT.items()})
                rows.append(row)
                small = {
                    k: (round(v, 4) if isinstance(v, (float, np.floating)) else v)
                    for k, v in row.items()
                    if k in [
                        "model",
                        "loss",
                        "overall_rmsre",
                        "spatial_rmsre",
                        "advec_rmsre",
                        "nugget_re",
                        "head_right_cols",
                        "n_heads",
                        "n_templates",
                        "total_s",
                    ]
                }
                print("RESULT:", small, flush=True)
            except Exception as exc:
                err = {
                    "iter": it + 1,
                    "seed": iter_seed,
                    "pattern_year": pattern["year"],
                    "pattern_day_idx": pattern["day_idx"],
                    "model": fit_name,
                    "kernel": fit_name,
                    "error": repr(exc),
                }
                rows.append(err)
                print("ERROR:", err, flush=True)
            save_outputs(rows, raw_csv, model_csv, param_csv)
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

        del source_map_ord
        gc.collect()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    print("Saved:", raw_csv, model_csv, param_csv, flush=True)


if __name__ == "__main__":
    main()
