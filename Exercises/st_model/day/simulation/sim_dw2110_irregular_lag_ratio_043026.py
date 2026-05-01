"""
sim_dw2110_irregular_lag_ratio_043026.py

Amarel simulation study for the lag-budget question on real-data-like
irregular GEMS observation patterns.

Compares six Vecchia-Irregular models on exactly the same simulated data:

  1. Irr_Full_A20_B20_C20
     Original 20/20/20 irregular Vecchia budget.

  2. Irr_Ratio_A20_B16_C10
     Same current-time budget, reduced temporal budgets:
       t-1 local neighbors: 16 ~= 0.80 * 20
       t-2 local neighbors: 10  = 0.50 * 20

  3-6. Targeted ablations to identify whether t-1 or t-2 drives advection
       recovery in the real-data-like irregular setting:
       A20_B18_C15, A20_B18_C12, A20_B20_C10, A20_B16_C20.

The data pipeline follows sim_dw2110_three_model_041626.py:
  high-resolution FFT field -> real GEMS source locations -> irregular
  N_grid-row maps with NaNs for unobserved rows.  Conditioning order is computed
  on the regular grid template, then applied to the irregular source-location
  rows, matching the real-data Vecchia pipeline.

Example:
  conda activate faiss_env
  python sim_dw2110_irregular_lag_ratio_043026.py --num-iters 1
"""

import gc
import os
import pickle
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.fft
import typer
from sklearn.neighbors import BallTree

AMAREL_SRC = "/home/jl2815/tco"
LOCAL_SRC = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
_src = AMAREL_SRC if os.path.exists(AMAREL_SRC) else LOCAL_SRC
sys.path.insert(0, _src)

from GEMS_TCO import configuration as config
from GEMS_TCO import kernels_vecchia
from GEMS_TCO import orderings as _orderings
from GEMS_TCO.data_loader import load_data_dynamic_processed


is_amarel = os.path.exists(config.amarel_data_load_path)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
DELTA_LAT_BASE = 0.044
DELTA_LON_BASE = 0.063
T_STEPS = 8

P_LABELS = [
    "sigmasq", "range_lat", "range_lon", "range_time",
    "advec_lat", "advec_lon", "nugget",
]
P_COLS = [
    "sigmasq_est", "range_lat_est", "range_lon_est", "range_t_est",
    "advec_lat_est", "advec_lon_est", "nugget_est",
]
SPATIAL_KEYS = ["sigmasq", "range_lat", "range_lon"]
ADVECTION_KEYS = ["advec_lat", "advec_lon"]

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_covariance_on_grid(lx, ly, lt, params):
    params = torch.clamp(params, min=-15.0, max=15.0)
    phi1, phi2, phi3, phi4 = (torch.exp(params[i]) for i in range(4))
    u_lat = lx - params[4] * lt
    u_lon = ly - params[5] * lt
    dist = torch.sqrt(u_lat.pow(2) * phi3 + u_lon.pow(2) + lt.pow(2) * phi4 + 1e-8)
    return (phi1 / phi2) * torch.exp(-dist * phi2)


def build_high_res_grid(lat_range, lon_range, lat_factor=100, lon_factor=10):
    dlat = DELTA_LAT_BASE / lat_factor
    dlon = DELTA_LON_BASE / lon_factor
    lat_max, lat_min = max(lat_range), min(lat_range)
    lats = torch.arange(lat_min - 0.1, lat_max + 0.1, dlat, device=DEVICE, dtype=DTYPE)
    lons = torch.arange(lon_range[0] - 0.1, lon_range[1] + 0.1, dlon, device=DEVICE, dtype=DTYPE)
    return lats, lons, dlat, dlon


def generate_field_values(lats_hr, lons_hr, t_steps, params, dlat, dlon):
    """Generate one FFT circulant-embedding field on the high-res grid."""
    cpu = torch.device("cpu")
    f32 = torch.float32
    nx, ny, nt = len(lats_hr), len(lons_hr), t_steps
    px, py, pt = 2 * nx, 2 * ny, 2 * nt

    lx = torch.arange(px, device=cpu, dtype=f32) * dlat
    lx[px // 2:] -= px * dlat
    ly = torch.arange(py, device=cpu, dtype=f32) * dlon
    ly[py // 2:] -= py * dlon
    lt = torch.arange(pt, device=cpu, dtype=f32)
    lt[pt // 2:] -= pt

    params_cpu = params.cpu().float()
    lx_g, ly_g, lt_g = torch.meshgrid(lx, ly, lt, indexing="ij")
    cov = get_covariance_on_grid(lx_g, ly_g, lt_g, params_cpu)
    spec = torch.fft.fftn(cov)
    spec.real = torch.clamp(spec.real, min=0)
    noise = torch.fft.fftn(torch.randn(px, py, pt, device=cpu, dtype=f32))
    field = torch.fft.ifftn(torch.sqrt(spec.real) * noise).real[:nx, :ny, :nt]
    return field.to(dtype=DTYPE, device=DEVICE)


def apply_step3_1to1(src_np_valid, grid_coords_np, grid_tree):
    """Obs-to-cell 1:1 assignment, matching step3 behavior."""
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
    hr_coords_np = torch.stack([hr_lat_g.flatten(), hr_lon_g.flatten()], dim=1).cpu().numpy()
    hr_tree = BallTree(np.radians(hr_coords_np), metric="haversine")

    grid_coords_np = grid_coords.cpu().numpy()
    n_grid = len(grid_coords_np)
    grid_tree = BallTree(np.radians(grid_coords_np), metric="haversine")

    step3_assignment_per_t = []
    hr_idx_per_t = []
    src_locs_per_t = []

    for key in sorted_keys:
        df = ref_day_map.get(key)
        if df is None or len(df) == 0:
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


def compute_grid_ordering(grid_coords, mm_cond_number):
    coords_np = grid_coords.cpu().numpy()
    ord_mm = _orderings.maxmin_cpp(coords_np)
    nns = _orderings.find_nns_l2(locs=coords_np[ord_mm], max_nn=mm_cond_number)
    return ord_mm, nns


def assemble_irregular_map(
    field,
    step3_assignment_per_t,
    hr_idx_per_t,
    src_locs_per_t,
    sorted_keys,
    grid_coords,
    true_params,
    t_offset=21.0,
):
    """Build irregular N_grid-row maps with NaN rows for unobserved cells."""
    nugget_std = torch.sqrt(torch.exp(true_params[6]))
    n_grid = grid_coords.shape[0]
    field_flat = field.reshape(-1, T_STEPS)

    irr_map = {}
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

        irr_map[key] = rows.detach()

    return irr_map


def true_to_log_params(true_dict):
    phi2 = 1.0 / true_dict["range_lon"]
    phi1 = true_dict["sigmasq"] * phi2
    phi3 = (true_dict["range_lon"] / true_dict["range_lat"]) ** 2
    phi4 = (true_dict["range_lon"] / true_dict["range_time"]) ** 2
    return [
        np.log(phi1), np.log(phi2), np.log(phi3), np.log(phi4),
        true_dict["advec_lat"], true_dict["advec_lon"], np.log(true_dict["nugget"]),
    ]


def backmap_params(out_params):
    p = out_params
    if isinstance(p[0], torch.Tensor):
        p = [x.item() if x.numel() == 1 else x[0].item() for x in p[:7]]
    else:
        p = [float(x) for x in p[:7]]
    phi2 = np.exp(p[1])
    phi3 = np.exp(p[2])
    phi4 = np.exp(p[3])
    rlon = 1.0 / phi2
    return {
        "sigmasq": np.exp(p[0]) / phi2,
        "range_lat": rlon / phi3 ** 0.5,
        "range_lon": rlon,
        "range_time": rlon / phi4 ** 0.5,
        "advec_lat": p[4],
        "advec_lon": p[5],
        "nugget": np.exp(p[6]),
    }


def rmsre_for_keys(est, true_dict, keys, zero_thresh=0.01):
    vals = []
    for key in keys:
        tv = true_dict[key]
        if abs(tv) < zero_thresh:
            continue
        vals.append(((est[key] - tv) / abs(tv)) ** 2)
    return float(np.sqrt(np.mean(vals))) if vals else float("nan")


def calculate_metrics(out_params, true_dict):
    est = backmap_params(out_params)
    metrics = {
        "overall_rmsre": rmsre_for_keys(est, true_dict, P_LABELS),
        "spatial_rmsre": rmsre_for_keys(est, true_dict, SPATIAL_KEYS),
        "range_time_re": abs(est["range_time"] - true_dict["range_time"]) / abs(true_dict["range_time"]),
        "advec_rmsre": rmsre_for_keys(est, true_dict, ADVECTION_KEYS),
        "nugget_re": abs(est["nugget"] - true_dict["nugget"]) / abs(true_dict["nugget"]),
    }
    return metrics, est


def make_random_init(rng, true_log, init_noise):
    noisy = list(true_log)
    for idx in [0, 1, 2, 3, 6]:
        noisy[idx] = true_log[idx] + rng.uniform(-init_noise, init_noise)
    for idx in [4, 5]:
        scale = max(abs(true_log[idx]), 0.05)
        noisy[idx] = true_log[idx] + rng.uniform(-2 * scale, 2 * scale)
    return noisy


def total_conditioning(limit_a, limit_b, limit_c):
    return int(limit_a + (1 + limit_b) + (1 + limit_c))


def fit_irregular_model(
    spec: Dict,
    irr_map_ord: Dict[str, torch.Tensor],
    nns_grid,
    initial_vals,
    smooth,
    mm_cond_number,
    nheads,
    daily_stride,
    lbfgs_lr,
    lbfgs_eval,
    lbfgs_hist,
    lbfgs_steps,
):
    params = [
        torch.tensor([val], device=DEVICE, dtype=DTYPE, requires_grad=True)
        for val in initial_vals
    ]
    model = kernels_vecchia.fit_vecchia_lbfgs(
        smooth=smooth,
        input_map=irr_map_ord,
        nns_map=nns_grid,
        mm_cond_number=mm_cond_number,
        nheads=nheads,
        limit_A=spec["limit_A"],
        limit_B=spec["limit_B"],
        limit_C=spec["limit_C"],
        daily_stride=daily_stride,
    )
    model.precompute_conditioning_sets()
    optimizer = model.set_optimizer(
        params, lr=lbfgs_lr, max_iter=lbfgs_eval, history_size=lbfgs_hist
    )
    t0 = time.time()
    out, fit_iter = model.fit_vecc_lbfgs(params, optimizer, max_steps=lbfgs_steps, grad_tol=1e-5)
    elapsed = time.time() - t0
    loss = float(out[-1])
    return out, loss, int(fit_iter) + 1, elapsed


def build_model_summary(df):
    metric_cols = [
        "loss", "overall_rmsre", "spatial_rmsre", "range_time_re",
        "advec_rmsre", "nugget_re", "total_s", "fit_iter",
    ]
    rows = []
    ok = df[df["error"].fillna("") == ""].copy()
    for keys, sub in ok.groupby(["model", "allocation", "limit_A", "limit_B", "limit_C", "total_conditioning"]):
        row = {
            "model": keys[0],
            "allocation": keys[1],
            "limit_A": keys[2],
            "limit_B": keys[3],
            "limit_C": keys[4],
            "total_conditioning": keys[5],
            "n": len(sub),
        }
        for col in metric_cols:
            vals = sub[col].dropna().values
            if len(vals) == 0:
                row[f"{col}_mean"] = float("nan")
                row[f"{col}_median"] = float("nan")
                row[f"{col}_p10"] = float("nan")
                row[f"{col}_p90"] = float("nan")
                row[f"{col}_p90_p10"] = float("nan")
                continue
            p10, p90 = np.percentile(vals, [10, 90])
            row[f"{col}_mean"] = float(np.mean(vals))
            row[f"{col}_median"] = float(np.median(vals))
            row[f"{col}_p10"] = float(p10)
            row[f"{col}_p90"] = float(p90)
            row[f"{col}_p90_p10"] = float(p90 - p10)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("overall_rmsre_mean")


def build_param_summary(df, true_dict):
    rows = []
    ok = df[df["error"].fillna("") == ""].copy()
    for model, sub in ok.groupby("model"):
        for par, col in zip(P_LABELS, P_COLS):
            tv = true_dict[par]
            denom = abs(tv) if abs(tv) >= 0.01 else 1.0
            vals = sub[col].dropna().values
            if len(vals) == 0:
                continue
            re = np.abs((vals - tv) / denom)
            p10, p90 = np.percentile(re, [10, 90])
            rows.append({
                "model": model,
                "parameter": par,
                "true": tv,
                "rmsre": float(np.sqrt(np.mean(re ** 2))),
                "mean_re": float(np.mean(re)),
                "median_re": float(np.median(re)),
                "p10_re": float(p10),
                "p90_re": float(p90),
                "p90_p10_re": float(p90 - p10),
                "estimate_mean": float(sub[col].mean()),
                "estimate_sd": float(sub[col].std()),
            })
    return pd.DataFrame(rows)


@app.command()
def cli(
    v: float = typer.Option(0.5, help="Matern smoothness"),
    mm_cond_number: int = typer.Option(100, help="Maxmin neighbor list size"),
    nheads: int = typer.Option(0, help="Vecchia head points per time step"),
    base_limit_a: int = typer.Option(20, help="Full model Set A neighbors"),
    base_limit_b: int = typer.Option(20, help="Full model Set B local neighbors"),
    base_limit_c: int = typer.Option(20, help="Full model Set C local neighbors"),
    ratio_limit_a: int = typer.Option(20, help="Reduced model Set A neighbors"),
    ratio_limit_b: int = typer.Option(16, help="Reduced model Set B local neighbors"),
    ratio_limit_c: int = typer.Option(10, help="Reduced model Set C local neighbors"),
    daily_stride: int = typer.Option(2, help="Set C stride; 2 means t-2"),
    num_iters: int = typer.Option(300, help="Simulation iterations"),
    years: str = typer.Option("2022,2023,2024,2025", help="Years for observation patterns"),
    month: int = typer.Option(7, help="Reference month"),
    lat_range: str = typer.Option("-3,2", help="lat_min,lat_max"),
    lon_range: str = typer.Option("121,131", help="lon_min,lon_max"),
    lat_factor: int = typer.Option(100, help="High-res lat multiplier"),
    lon_factor: int = typer.Option(10, help="High-res lon multiplier"),
    init_noise: float = typer.Option(0.7, help="Uniform log-space init noise"),
    lbfgs_steps: int = typer.Option(5, help="Outer LBFGS steps"),
    lbfgs_eval: int = typer.Option(20, help="LBFGS max_iter per outer step"),
    lbfgs_hist: int = typer.Option(10, help="LBFGS history size"),
    seed: int = typer.Option(42, help="Random seed"),
    out_prefix: str = typer.Option("sim_dw2110_irregular_lag_ratio_candidates", help="Output prefix"),
) -> None:
    set_seed(seed)
    rng = np.random.default_rng(seed)

    lat_r = [float(x) for x in lat_range.split(",")]
    lon_r = [float(x) for x in lon_range.split(",")]
    years_list = [y.strip() for y in years.split(",")]

    print(f"Device : {DEVICE}")
    print(f"Region : lat {lat_r}, lon {lon_r}")
    print(f"Years  : {years_list} month={month}")
    print(f"High-res : lat x{lat_factor}, lon x{lon_factor}")
    print(f"Iterations : {num_iters}")

    output_path = Path(config.amarel_estimates_day_path if is_amarel else config.mac_estimates_day_path)
    output_path.mkdir(parents=True, exist_ok=True)
    date_tag = datetime.now().strftime("%m%d%y")
    raw_csv = output_path / f"{out_prefix}_{date_tag}_raw.csv"
    model_summary_csv = output_path / f"{out_prefix}_{date_tag}_model_summary.csv"
    param_summary_csv = output_path / f"{out_prefix}_{date_tag}_param_summary.csv"

    # Same active DW2110-like scenario as sim_dw2110_three_model_041626.py.
    true_dict = {
        "sigmasq": 10.0,
        "range_lat": 0.5,
        "range_lon": 0.6,
        "range_time": 2.5,
        "advec_lat": 0.25,
        "advec_lon": -0.16,
        "nugget": 1.2,
    }
    true_log = true_to_log_params(true_dict)
    true_params = torch.tensor(true_log, device=DEVICE, dtype=DTYPE)

    model_specs = [
        {
            "model": f"Irr_Full_A{base_limit_a:02d}_B{base_limit_b:02d}_C{base_limit_c:02d}",
            "allocation": "baseline full lag budget",
            "limit_A": base_limit_a,
            "limit_B": base_limit_b,
            "limit_C": base_limit_c,
        },
        {
            "model": f"Irr_Ratio_A{ratio_limit_a:02d}_B{ratio_limit_b:02d}_C{ratio_limit_c:02d}",
            "allocation": "same current, t-1 near 0.8x, t-2 near 0.5x",
            "limit_A": ratio_limit_a,
            "limit_B": ratio_limit_b,
            "limit_C": ratio_limit_c,
        },
        {
            "model": "Irr_Cand_A20_B18_C15",
            "allocation": "candidate: t-1 0.9x, t-2 0.75x",
            "limit_A": 20,
            "limit_B": 18,
            "limit_C": 15,
        },
        {
            "model": "Irr_Cand_A20_B18_C12",
            "allocation": "candidate: t-1 0.9x, t-2 0.6x",
            "limit_A": 20,
            "limit_B": 18,
            "limit_C": 12,
        },
        {
            "model": "Irr_Cand_A20_B20_C10",
            "allocation": "candidate: full t-1, t-2 0.5x",
            "limit_A": 20,
            "limit_B": 20,
            "limit_C": 10,
        },
        {
            "model": "Irr_Cand_A20_B16_C20",
            "allocation": "candidate: t-1 0.8x, full t-2",
            "limit_A": 20,
            "limit_B": 16,
            "limit_C": 20,
        },
    ]
    for spec in model_specs:
        spec["total_conditioning"] = total_conditioning(
            spec["limit_A"], spec["limit_B"], spec["limit_C"]
        )
        spec["lag1_ratio_actual"] = spec["limit_B"] / max(spec["limit_A"], 1)
        spec["lag2_ratio_actual"] = spec["limit_C"] / max(spec["limit_A"], 1)

    print("\nModel specs")
    for spec in model_specs:
        print(
            f"  {spec['model']}: A={spec['limit_A']} B={spec['limit_B']} C={spec['limit_C']} "
            f"total={spec['total_conditioning']} "
            f"(B/A={spec['lag1_ratio_actual']:.2f}, C/A={spec['lag2_ratio_actual']:.2f})"
        )

    print("\n[Setup 1/5] Loading GEMS observation patterns...")
    data_path = config.amarel_data_load_path if is_amarel else config.mac_data_load_path
    data_loader = load_data_dynamic_processed(data_path)
    year_dfmaps = {}
    year_tco_maps = {}
    for yr in years_list:
        df_map_yr, _, _, _ = data_loader.load_maxmin_ordered_data_bymonthyear(
            lat_lon_resolution=[1, 1],
            mm_cond_number=mm_cond_number,
            years_=[yr],
            months_=[month],
            lat_range=lat_r,
            lon_range=lon_r,
            is_whittle=False,
        )
        year_dfmaps[yr] = df_map_yr
        print(f"  {yr}-{month:02d}: {len(df_map_yr)} time slots loaded")

        yr2 = str(yr)[2:]
        tco_path = Path(data_path) / f"pickle_{yr}" / f"tco_grid_{yr2}_{month:02d}.pkl"
        if tco_path.exists():
            with open(tco_path, "rb") as f:
                year_tco_maps[yr] = pickle.load(f)
            print(f"    tco_grid: {len(year_tco_maps[yr])} slots loaded")
        else:
            year_tco_maps[yr] = {}
            print(f"    [WARN] Missing tco_grid: {tco_path}")

    print("[Setup 2/5] Building regular target grid...")
    lats_grid = torch.arange(min(lat_r), max(lat_r) + 0.0001, DELTA_LAT_BASE, device=DEVICE, dtype=DTYPE)
    lons_grid = torch.arange(lon_r[0], lon_r[1] + 0.0001, DELTA_LON_BASE, device=DEVICE, dtype=DTYPE)
    lats_grid = torch.round(lats_grid * 10000) / 10000
    lons_grid = torch.round(lons_grid * 10000) / 10000
    g_lat, g_lon = torch.meshgrid(lats_grid, lons_grid, indexing="ij")
    grid_coords = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)
    n_grid = grid_coords.shape[0]
    print(f"  Grid: {len(lats_grid)} lat x {len(lons_grid)} lon = {n_grid} cells")

    print("[Setup 3/5] Building high-res grid and precomputing mappings...")
    lats_hr, lons_hr, dlat_hr, dlon_hr = build_high_res_grid(lat_r, lon_r, lat_factor, lon_factor)
    print(f"  High-res: {len(lats_hr)} lat x {len(lons_hr)} lon = {len(lats_hr) * len(lons_hr):,} cells")

    dummy_keys = [f"t{i}" for i in range(T_STEPS)]
    all_day_mappings = []
    for yr in years_list:
        df_map_yr = year_dfmaps[yr]
        all_sorted = sorted(df_map_yr.keys())
        n_days_yr = len(all_sorted) // T_STEPS
        print(f"  {yr}: precomputing {n_days_yr} day-patterns...", flush=True)
        for d_idx in range(n_days_yr):
            day_keys = all_sorted[d_idx * T_STEPS : (d_idx + 1) * T_STEPS]
            if len(day_keys) < T_STEPS:
                continue
            ref_day = {
                k: year_tco_maps[yr].get(k.split("_", 2)[-1], pd.DataFrame())
                for k in day_keys
            }
            s3, hr_idx, src = precompute_mapping_indices(ref_day, lats_hr, lons_hr, grid_coords, day_keys)
            all_day_mappings.append((yr, d_idx, s3, hr_idx, src))

    if not all_day_mappings:
        raise RuntimeError("No usable day-patterns were found.")
    print(f"  Total available day-patterns: {len(all_day_mappings)}")

    print("[Setup 4/5] Computing shared grid-based maxmin ordering...")
    ord_grid, nns_grid = compute_grid_ordering(grid_coords, mm_cond_number)
    print(f"  Ordering complete: N_grid={n_grid}, mm_cond_number={mm_cond_number}")

    print("[Setup 5/5] Verifying one assembled irregular map...")
    yr0, d0, s3_0, hr0, src0 = all_day_mappings[0]
    field0 = generate_field_values(lats_hr, lons_hr, T_STEPS, true_params, dlat_hr, dlon_hr)
    irr0 = assemble_irregular_map(field0, s3_0, hr0, src0, dummy_keys, grid_coords, true_params)
    del field0
    first = list(irr0.values())[0]
    n_valid = int((~torch.isnan(first[:, 2])).sum().item())
    print(f"  Sample pattern {yr0} day {d0}: {n_valid}/{n_grid} valid rows at t0")
    del irr0
    torch.cuda.empty_cache()
    gc.collect()

    lbfgs_lr = 1.0
    records = []

    for it in range(num_iters):
        print(f"\n{'=' * 72}")
        print(f"Iteration {it + 1}/{num_iters}")
        print(f"{'=' * 72}")

        yr_it, d_it, step3_assignment_per_t, hr_idx_per_t, src_locs_per_t = all_day_mappings[
            rng.integers(len(all_day_mappings))
        ]
        initial_vals = make_random_init(rng, true_log, init_noise)
        init_orig = backmap_params(initial_vals)
        print(f"Obs pattern: {yr_it} day {d_it}")
        print(
            f"Init: sigmasq={init_orig['sigmasq']:.3f}, "
            f"range_lon={init_orig['range_lon']:.3f}, nugget={init_orig['nugget']:.3f}"
        )

        try:
            field = generate_field_values(lats_hr, lons_hr, T_STEPS, true_params, dlat_hr, dlon_hr)
            irr_map = assemble_irregular_map(
                field, step3_assignment_per_t, hr_idx_per_t, src_locs_per_t,
                dummy_keys, grid_coords, true_params,
            )
            del field
            irr_map_ord = {k: v[ord_grid] for k, v in irr_map.items()}
            del irr_map
        except Exception as exc:
            print(f"[SKIP] Iteration assembly failed: {type(exc).__name__}: {exc}")
            continue

        for spec in model_specs:
            print(f"\n--- {spec['model']} ---")
            pre_t = time.time()
            error_msg = ""
            try:
                out, loss, fit_iter, fit_s = fit_irregular_model(
                    spec, irr_map_ord, nns_grid, initial_vals, v, mm_cond_number,
                    nheads, daily_stride, lbfgs_lr, lbfgs_eval, lbfgs_hist, lbfgs_steps,
                )
                total_s = time.time() - pre_t
                metrics, est = calculate_metrics(out, true_dict)
                print(
                    f"loss={loss:.4f} overall={metrics['overall_rmsre']:.4f} "
                    f"spatial={metrics['spatial_rmsre']:.4f} time={total_s:.1f}s"
                )
            except Exception as exc:
                total_s = time.time() - pre_t
                loss = fit_iter = fit_s = float("nan")
                metrics = {
                    "overall_rmsre": float("nan"),
                    "spatial_rmsre": float("nan"),
                    "range_time_re": float("nan"),
                    "advec_rmsre": float("nan"),
                    "nugget_re": float("nan"),
                }
                est = {k: float("nan") for k in P_LABELS}
                error_msg = f"{type(exc).__name__}: {exc}"
                print(f"FAILED: {error_msg}")

            records.append({
                "iter": it + 1,
                "obs_year": yr_it,
                "obs_day": d_it,
                "model": spec["model"],
                "allocation": spec["allocation"],
                "limit_A": spec["limit_A"],
                "limit_B": spec["limit_B"],
                "limit_C": spec["limit_C"],
                "lag1_ratio_actual": spec["lag1_ratio_actual"],
                "lag2_ratio_actual": spec["lag2_ratio_actual"],
                "total_conditioning": spec["total_conditioning"],
                "loss": round(loss, 6) if np.isfinite(loss) else np.nan,
                "overall_rmsre": round(metrics["overall_rmsre"], 6),
                "spatial_rmsre": round(metrics["spatial_rmsre"], 6),
                "range_time_re": round(metrics["range_time_re"], 6),
                "advec_rmsre": round(metrics["advec_rmsre"], 6),
                "nugget_re": round(metrics["nugget_re"], 6),
                "fit_iter": fit_iter,
                "fit_s": round(fit_s, 3) if np.isfinite(fit_s) else np.nan,
                "total_s": round(total_s, 3),
                "sigmasq_est": round(est["sigmasq"], 6),
                "range_lat_est": round(est["range_lat"], 6),
                "range_lon_est": round(est["range_lon"], 6),
                "range_t_est": round(est["range_time"], 6),
                "advec_lat_est": round(est["advec_lat"], 6),
                "advec_lon_est": round(est["advec_lon"], 6),
                "nugget_est": round(est["nugget"], 6),
                "init_sigmasq": round(init_orig["sigmasq"], 6),
                "init_range_lon": round(init_orig["range_lon"], 6),
                "init_nugget": round(init_orig["nugget"], 6),
                "error": error_msg,
            })

            torch.cuda.empty_cache()
            gc.collect()

        df_now = pd.DataFrame(records)
        df_now.to_csv(raw_csv, index=False)
        ok_now = df_now[df_now["error"].fillna("") == ""]
        if not ok_now.empty:
            model_summary = build_model_summary(df_now)
            param_summary = build_param_summary(df_now, true_dict)
            model_summary.to_csv(model_summary_csv, index=False)
            param_summary.to_csv(param_summary_csv, index=False)
            if not model_summary.empty:
                print("\nRunning model summary")
                cols = [
                    "model", "n", "loss_mean", "overall_rmsre_mean",
                    "overall_rmsre_p90_p10", "spatial_rmsre_mean",
                    "spatial_rmsre_p90_p10", "advec_rmsre_mean",
                    "advec_rmsre_p90_p10", "total_s_mean", "total_s_p90_p10",
                ]
                print(model_summary[cols].to_string(index=False))

            if not param_summary.empty:
                print("\nRunning parameter summary")
                param_cols_show = [
                    "model", "parameter", "rmsre", "mean_re", "median_re",
                    "p10_re", "p90_re", "p90_p10_re",
                ]
                print(param_summary[param_cols_show].to_string(index=False))

        del irr_map_ord
        torch.cuda.empty_cache()
        gc.collect()

    df_final = pd.DataFrame(records)
    df_final.to_csv(raw_csv, index=False)
    model_summary = build_model_summary(df_final)
    param_summary = build_param_summary(df_final, true_dict)
    model_summary.to_csv(model_summary_csv, index=False)
    param_summary.to_csv(param_summary_csv, index=False)

    print(f"\nSaved raw results: {raw_csv}")
    print(f"Saved model summary: {model_summary_csv}")
    print(f"Saved parameter summary: {param_summary_csv}")
    if not model_summary.empty:
        print("\nFinal model summary")
        print(model_summary.to_string(index=False))


if __name__ == "__main__":
    app()
