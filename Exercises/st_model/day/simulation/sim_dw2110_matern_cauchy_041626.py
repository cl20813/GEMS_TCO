"""
sim_matern_cauchy_dw_comparison_040226.py

Three-way simulation study comparing:
  1. Vecc_Matern : Vecchia-Irregular at actual GEMS source obs locations  (Matérn ν=0.5)
  2. Vecc_Cauchy : Vecchia-Irregular at actual GEMS source obs locations  (Generalized Cauchy β)
  3. DW          : Debiased Whittle on regular grid (step3 re-gridded data)

DGP: Matérn ν=0.5 (exponential), generated via FFT circulant embedding.
All three models are fitted to the same simulated dataset.
Vecc_Cauchy is intentionally misspecified relative to the Matérn DGP.

Data pipeline (same as sim_three_model_comparison_031926.py):
  ┌─ FFT high-res field  (lat × lat_factor, lon × lon_factor of base resolution)
  │
  ├─► [Simulated actual observations]  (irregular — used by Vecc_Matern & Vecc_Cauchy)
  │     valid obs → nearest high-res point → + nugget noise
  │     stored as [N_grid, 11] tensor: NaN rows for unobserved cells
  │
  └─► [Simulated tco_grid]  (regular, step3 re-gridded — used by DW)
        step3: each obs → nearest grid cell, 1:1, no duplicates
        unassigned cells → NaN in value column

sbatch:
  cd ./jobscript/tco/gp_exercise
  nano sim_matern_cauchy_dw_040226.sh
  sbatch sim_matern_cauchy_dw_040226.sh
"""

import sys
import os
import time
import pickle
from datetime import datetime
import json
import numpy as np
import torch
import torch.fft
import pandas as pd
import typer
from pathlib import Path
from typing import List
from sklearn.neighbors import BallTree

sys.path.append("/cache/home/jl2815/tco")

from GEMS_TCO import kernels_vecchia
from GEMS_TCO import kernels_vecchia_cauchy
from GEMS_TCO import orderings as _orderings
from GEMS_TCO import debiased_whittle_2110 as debiased_whittle
from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data_dynamic_processed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64
DELTA_LAT_BASE = 0.044
DELTA_LON_BASE = 0.063

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})


# ── Covariance kernel (Matérn ν=0.5 — for DGP field generation only) ─────────

def get_covariance_on_grid(lx, ly, lt, params):
    params = torch.clamp(params, min=-15.0, max=15.0)
    phi1, phi2, phi3, phi4 = (torch.exp(params[i]) for i in range(4))
    u_lat = lx - params[4] * lt
    u_lon = ly - params[5] * lt
    dist  = torch.sqrt(u_lat.pow(2) * phi3 + u_lon.pow(2) + lt.pow(2) * phi4 + 1e-8)
    return (phi1 / phi2) * torch.exp(-dist * phi2)


# ── High-resolution FFT field ─────────────────────────────────────────────────

def build_high_res_grid(lat_range, lon_range, lat_factor=100, lon_factor=20):
    dlat = DELTA_LAT_BASE / lat_factor
    dlon = DELTA_LON_BASE / lon_factor
    lat_max, lat_min = max(lat_range), min(lat_range)
    lats = torch.arange(lat_min - 0.1, lat_max + 0.1, dlat, device=DEVICE, dtype=DTYPE)
    lons = torch.arange(lon_range[0] - 0.1, lon_range[1] + 0.1, dlon, device=DEVICE, dtype=DTYPE)
    return lats, lons, dlat, dlon


def generate_field_values(lats_hr, lons_hr, t_steps, params, dlat, dlon):
    """Matérn ν=0.5 FFT circulant embedding field. Returns [Nx, Ny, Nt] on DEVICE."""
    CPU = torch.device("cpu")
    F32 = torch.float32
    Nx, Ny, Nt = len(lats_hr), len(lons_hr), t_steps
    Px, Py, Pt = 2*Nx, 2*Ny, 2*Nt

    lx = torch.arange(Px, device=CPU, dtype=F32) * dlat; lx[Px//2:] -= Px * dlat
    ly = torch.arange(Py, device=CPU, dtype=F32) * dlon; ly[Py//2:] -= Py * dlon
    lt = torch.arange(Pt, device=CPU, dtype=F32);        lt[Pt//2:] -= Pt

    params_cpu = params.cpu().float()
    Lx, Ly, Lt = torch.meshgrid(lx, ly, lt, indexing='ij')
    C = get_covariance_on_grid(Lx, Ly, Lt, params_cpu)
    S = torch.fft.fftn(C)
    S.real = torch.clamp(S.real, min=0)
    noise = torch.fft.fftn(torch.randn(Px, Py, Pt, device=CPU, dtype=F32))
    field = torch.fft.ifftn(torch.sqrt(S.real) * noise).real[:Nx, :Ny, :Nt]
    return field.to(dtype=DTYPE, device=DEVICE)


# ── Index precomputation ──────────────────────────────────────────────────────

def apply_step3_1to1(src_np_valid, grid_coords_np, grid_tree):
    N_grid = len(grid_coords_np)
    if len(src_np_valid) == 0:
        return np.full(N_grid, -1, dtype=np.int64)

    dist_to_cell, cell_for_obs = grid_tree.query(np.radians(src_np_valid), k=1)
    dist_to_cell = dist_to_cell.flatten()
    cell_for_obs = cell_for_obs.flatten()

    assignment = np.full(N_grid, -1, dtype=np.int64)
    best_dist  = np.full(N_grid, np.inf)
    for obs_i, (cell_j, d) in enumerate(zip(cell_for_obs, dist_to_cell)):
        if d < best_dist[cell_j]:
            assignment[cell_j] = obs_i
            best_dist[cell_j]  = d

    # Threshold: match step3_enforce_regular_grid (lat_thresh=DELTA_LAT/2, lon_thresh=DELTA_LON/2)
    filled = assignment >= 0
    if filled.any():
        win_obs  = assignment[filled]
        lat_diff = np.abs(src_np_valid[win_obs, 0] - grid_coords_np[filled, 0])
        lon_diff = np.abs(src_np_valid[win_obs, 1] - grid_coords_np[filled, 1])
        too_far  = (lat_diff > DELTA_LAT_BASE / 2) | (lon_diff > DELTA_LON_BASE / 2)
        assignment[np.where(filled)[0][too_far]] = -1
    return assignment


def precompute_mapping_indices(ref_day_map, lats_hr, lons_hr, grid_coords, sorted_keys):
    """obs pool = tco_grid Source_Latitude/Source_Longitude (1 obs per cell)."""
    hr_lat_g, hr_lon_g = torch.meshgrid(lats_hr, lons_hr, indexing='ij')
    hr_coords_np = torch.stack(
        [hr_lat_g.flatten(), hr_lon_g.flatten()], dim=1).cpu().numpy()
    hr_tree = BallTree(np.radians(hr_coords_np), metric='haversine')

    grid_coords_np = grid_coords.cpu().numpy()
    N_grid = len(grid_coords_np)
    grid_tree = BallTree(np.radians(grid_coords_np), metric='haversine')

    step3_assignment_per_t, hr_idx_per_t, src_locs_per_t = [], [], []
    for key in sorted_keys:
        df = ref_day_map.get(key)
        if df is None or len(df) == 0:
            step3_assignment_per_t.append(np.full(N_grid, -1, dtype=np.int64))
            hr_idx_per_t.append(torch.zeros(0, dtype=torch.long, device=DEVICE))
            src_locs_per_t.append(torch.zeros((0, 2), dtype=DTYPE, device=DEVICE))
            continue
        src_np = df[['Source_Latitude', 'Source_Longitude']].values
        valid_mask   = ~np.isnan(src_np).any(axis=1)
        src_np_valid = src_np[valid_mask]

        assignment = apply_step3_1to1(src_np_valid, grid_coords_np, grid_tree)
        step3_assignment_per_t.append(assignment)

        if len(src_np_valid) > 0:
            _, hr_idx = hr_tree.query(np.radians(src_np_valid), k=1)
            hr_idx_per_t.append(torch.tensor(hr_idx.flatten(), device=DEVICE))
        else:
            hr_idx_per_t.append(torch.zeros(0, dtype=torch.long, device=DEVICE))

        src_locs_per_t.append(
            torch.tensor(src_np_valid, device=DEVICE, dtype=DTYPE))

    return step3_assignment_per_t, hr_idx_per_t, src_locs_per_t


def compute_grid_ordering(grid_coords, mm_cond_number):
    coords_np = grid_coords.cpu().numpy()
    ord_mm    = _orderings.maxmin_cpp(coords_np)
    nns       = _orderings.find_nns_l2(locs=coords_np[ord_mm], max_nn=mm_cond_number)
    return ord_mm, nns


# ── Dataset assembly ──────────────────────────────────────────────────────────

def assemble_datasets(field, step3_assignment_per_t, hr_idx_per_t, src_locs_per_t,
                      sorted_keys, grid_coords, true_params, t_offset=21.0):
    """
    Produce two datasets from one FFT realization:
      irr_map : irregular source locations (for Vecchia models)
      reg_map : regular grid locations (for DW)
    """
    nugget_std = torch.sqrt(torch.exp(true_params[6]))
    N_grid     = grid_coords.shape[0]
    field_flat = field.reshape(-1, 8)

    irr_map, irr_list = {}, []
    reg_map, reg_list = {}, []

    for t_idx, key in enumerate(sorted_keys):
        t_val    = float(t_offset + t_idx)
        assign   = step3_assignment_per_t[t_idx]
        hr_idx   = hr_idx_per_t[t_idx]
        src_locs = src_locs_per_t[t_idx]
        N_valid  = hr_idx.shape[0]

        dummy = torch.zeros(7, device=DEVICE, dtype=DTYPE)
        if t_idx > 0:
            dummy[t_idx - 1] = 1.0

        if N_valid > 0:
            gp_vals  = field_flat[hr_idx, t_idx]
            sim_vals = gp_vals + torch.randn(N_valid, device=DEVICE, dtype=DTYPE) * nugget_std
        else:
            sim_vals = torch.zeros(0, device=DEVICE, dtype=DTYPE)

        NaN = float('nan')
        dummy_row = dummy.unsqueeze(0).expand(N_grid, -1)

        irr_rows = torch.full((N_grid, 11), NaN, device=DEVICE, dtype=DTYPE)
        irr_rows[:, 3]  = t_val
        irr_rows[:, 4:] = dummy_row

        reg_rows = torch.full((N_grid, 11), NaN, device=DEVICE, dtype=DTYPE)
        reg_rows[:, :2] = grid_coords
        reg_rows[:, 3]  = t_val
        reg_rows[:, 4:] = dummy_row

        if N_valid > 0:
            assign_t = torch.tensor(assign, device=DEVICE, dtype=torch.long)
            filled   = assign_t >= 0
            win_obs  = assign_t[filled]

            irr_rows[filled, 0] = src_locs[win_obs, 0]
            irr_rows[filled, 1] = src_locs[win_obs, 1]
            irr_rows[filled, 2] = sim_vals[win_obs]

            reg_rows[filled, 2] = sim_vals[win_obs]

        irr_map[key] = irr_rows.detach()
        irr_list.append(irr_rows.detach())
        reg_map[key] = reg_rows.detach()
        reg_list.append(reg_rows.detach())

    return (irr_map, torch.cat(irr_list, dim=0)), (reg_map, torch.cat(reg_list, dim=0))


# ── Metrics ───────────────────────────────────────────────────────────────────

def backmap_params(out_params):
    p = out_params
    if isinstance(p[0], torch.Tensor):
        p = [x.item() if x.numel() == 1 else x[0].item() for x in p]
    p = [float(x) for x in p]
    phi2 = np.exp(p[1]); phi3 = np.exp(p[2]); phi4 = np.exp(p[3])
    rlon = 1.0 / phi2
    return {
        'sigmasq':    np.exp(p[0]) / phi2,
        'range_lat':  rlon / phi3 ** 0.5,
        'range_lon':  rlon,
        'range_time': rlon / phi4 ** 0.5,
        'advec_lat':  p[4],
        'advec_lon':  p[5],
        'nugget':     np.exp(p[6]),
    }


def calculate_rmsre(out_params, true_dict):
    est = backmap_params(out_params)
    est_arr  = np.array([est['sigmasq'],        est['range_lat'],   est['range_lon'],
                         est['range_time'],      est['advec_lat'],   est['advec_lon'],
                         est['nugget']])
    true_arr = np.array([true_dict['sigmasq'],   true_dict['range_lat'], true_dict['range_lon'],
                         true_dict['range_time'],true_dict['advec_lat'], true_dict['advec_lon'],
                         true_dict['nugget']])
    return float(np.sqrt(np.mean(((est_arr - true_arr) / np.abs(true_arr)) ** 2))), est


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command()
def cli(
    v:              float = typer.Option(0.5,   help="Matérn smoothness (DGP & Vecc_Matern)"),
    gc_beta:        float = typer.Option(1.0,   help="Cauchy β for Vecc_Cauchy"),
    mm_cond_number: int   = typer.Option(100,   help="Vecchia neighbors"),
    nheads:         int   = typer.Option(0,     help="Vecchia head points"),
    limit_a:        int   = typer.Option(20,    help="Set A neighbors"),
    limit_b:        int   = typer.Option(20,    help="Set B neighbors"),
    limit_c:        int   = typer.Option(20,    help="Set C neighbors"),
    daily_stride:   int   = typer.Option(2,     help="Set C daily stride"),
    num_iters:      int   = typer.Option(50,    help="Simulation iterations"),
    years:          str   = typer.Option("2022,2023,2024,2025", help="Years to sample obs patterns from"),
    month:          int   = typer.Option(7,     help="Reference month"),
    lat_range:      str   = typer.Option("-3,2",    help="lat_min,lat_max"),
    lon_range:      str   = typer.Option("121,131", help="lon_min,lon_max"),
    lat_factor:     int   = typer.Option(100,   help="High-res lat multiplier"),
    lon_factor:     int   = typer.Option(10,    help="High-res lon multiplier"),
    init_noise:     float = typer.Option(0.7,   help="Log-space init perturbation half-width"),
    seed:           int   = typer.Option(42,    help="Random seed"),
) -> None:

    import random as _random
    rng = np.random.default_rng(seed)
    _random.seed(seed)

    lat_r      = [float(x) for x in lat_range.split(',')]
    lon_r      = [float(x) for x in lon_range.split(',')]
    years_list = [y.strip() for y in years.split(',')]

    print(f"Device   : {DEVICE}")
    print(f"Region   : lat {lat_r}, lon {lon_r}")
    print(f"Years    : {years_list}  month={month}")
    print(f"High-res : lat×{lat_factor}, lon×{lon_factor}")
    print(f"DGP      : Matérn ν={v}")
    print(f"Models   : Vecc_Matern(ν={v})  Vecc_Cauchy(β={gc_beta})  DW")

    output_path = Path(config.amarel_estimates_day_path) / "sim_matern_cauchy_dw"
    output_path.mkdir(parents=True, exist_ok=True)
    date_tag    = datetime.now().strftime("%m%d%y")
    beta_tag    = f"b{int(gc_beta * 10):02d}"
    csv_raw     = f"sim_matern_cauchy_dw_{beta_tag}_{date_tag}.csv"
    csv_summary = f"sim_matern_cauchy_dw_{beta_tag}_{date_tag}_summary.csv"

    # ── True parameters (Matérn DGP) ──────────────────────────────────────────
    true_dict = {
        'sigmasq':    13.059,
        'range_lat':  0.154,
        'range_lon':  0.195,
        'range_time': 1.0,
        'advec_lat':  0.0218,
        'advec_lon':  -0.1689,
        'nugget':     0.247,
    }

    phi2 = 1.0 / true_dict['range_lon']
    phi1 = true_dict['sigmasq'] * phi2
    phi3 = (true_dict['range_lon'] / true_dict['range_lat'])  ** 2
    phi4 = (true_dict['range_lon'] / true_dict['range_time']) ** 2
    true_log = [
        np.log(phi1), np.log(phi2), np.log(phi3), np.log(phi4),
        true_dict['advec_lat'], true_dict['advec_lon'], np.log(true_dict['nugget'])
    ]
    true_params = torch.tensor(true_log, device=DEVICE, dtype=DTYPE)

    def make_random_init(rng):
        noisy = list(true_log)
        for i in [0, 1, 2, 3, 6]:
            noisy[i] = true_log[i] + rng.uniform(-init_noise, init_noise)
        for i in [4, 5]:
            scale = max(abs(true_log[i]), 0.05)
            noisy[i] = true_log[i] + rng.uniform(-2 * scale, 2 * scale)
        return noisy

    # ── Load obs patterns ──────────────────────────────────────────────────────
    print("\n[Setup 1/5] Loading GEMS obs patterns for all years...")
    is_amarel   = os.path.exists(config.amarel_data_load_path)
    data_path   = config.amarel_data_load_path if is_amarel else config.mac_data_load_path
    data_loader = load_data_dynamic_processed(data_path)
    all_day_mappings = []
    year_dfmaps, year_means, year_tco_maps = {}, {}, {}

    for yr in years_list:
        df_map_yr, _, _, monthly_mean_yr = data_loader.load_maxmin_ordered_data_bymonthyear(
            lat_lon_resolution=[1, 1],
            mm_cond_number=mm_cond_number,
            years_=[yr], months_=[month],
            lat_range=lat_r, lon_range=lon_r,
            is_whittle=False
        )
        year_dfmaps[yr] = df_map_yr
        year_means[yr]  = monthly_mean_yr
        print(f"  {yr}-{month:02d}: {len(df_map_yr)} time slots loaded")

        yr2      = str(yr)[2:]
        tco_path = Path(data_path) / f"pickle_{yr}" / f"tco_grid_{yr2}_{month:02d}.pkl"
        if tco_path.exists():
            with open(tco_path, 'rb') as f:
                year_tco_maps[yr] = pickle.load(f)
            print(f"  tco_grid: {len(year_tco_maps[yr])} slots loaded")
        else:
            year_tco_maps[yr] = {}
            print(f"  [WARN] tco_grid not found: {tco_path}")

    # ── Regular target grid ────────────────────────────────────────────────────
    print("[Setup 2/5] Building regular target grid...")
    lats_grid = torch.arange(min(lat_r), max(lat_r) + 0.0001,  DELTA_LAT_BASE,
                              device=DEVICE, dtype=DTYPE)
    lons_grid = torch.arange(lon_r[0],   lon_r[1]  + 0.0001,  DELTA_LON_BASE,
                              device=DEVICE, dtype=DTYPE)
    lats_grid = torch.round(lats_grid * 10000) / 10000
    lons_grid = torch.round(lons_grid * 10000) / 10000
    g_lat, g_lon = torch.meshgrid(lats_grid, lons_grid, indexing='ij')
    grid_coords  = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)
    N_grid = grid_coords.shape[0]
    print(f"  Grid: {len(lats_grid)} lat × {len(lons_grid)} lon = {N_grid} cells")

    # ── High-res grid & precompute mappings ───────────────────────────────────
    print("[Setup 3/5] Building high-res grid and precomputing mappings...")
    lats_hr, lons_hr, dlat_hr, dlon_hr = build_high_res_grid(lat_r, lon_r,
                                                               lat_factor, lon_factor)
    print(f"  High-res: {len(lats_hr)} lat × {len(lons_hr)} lon "
          f"= {len(lats_hr)*len(lons_hr):,} cells")

    DUMMY_KEYS = [f't{i}' for i in range(8)]

    for yr in years_list:
        df_map_yr  = year_dfmaps[yr]
        all_sorted = sorted(df_map_yr.keys())
        n_days_yr  = len(all_sorted) // 8
        print(f"  {yr}: precomputing {n_days_yr} days...", flush=True)
        for d_idx in range(n_days_yr):
            day_keys = all_sorted[d_idx * 8 : (d_idx + 1) * 8]
            if len(day_keys) < 8:
                continue
            # orbit_map keys have "YYYY_MM_" prefix; tco_grid keys do not — strip prefix
            ref_day = {k: year_tco_maps[yr].get(k.split('_', 2)[-1], pd.DataFrame())
                       for k in day_keys}
            s3, hr_idx, src = precompute_mapping_indices(
                ref_day, lats_hr, lons_hr, grid_coords, day_keys)
            all_day_mappings.append((yr, d_idx, s3, hr_idx, src))

    print(f"  Total available day-patterns: {len(all_day_mappings)}")

    # ── Shared grid ordering ───────────────────────────────────────────────────
    print("[Setup 4/5] Computing maxmin ordering (shared across all Vecchia models)...")
    ord_grid, nns_grid = compute_grid_ordering(grid_coords, mm_cond_number)
    print(f"  N_grid={N_grid}, mm_cond_number={mm_cond_number}")

    # ── Verify dataset structure ──────────────────────────────────────────────
    print("[Setup 5/5] Verifying dataset structure with sample field...")
    _yr0, _d0, _s3_0, _hr0, _src0 = all_day_mappings[0]
    field0 = generate_field_values(lats_hr, lons_hr, 8, true_params, dlat_hr, dlon_hr)
    (irr0, _), (reg0, _) = assemble_datasets(
        field0, _s3_0, _hr0, _src0, DUMMY_KEYS, grid_coords, true_params)
    del field0
    n_irr = (~torch.isnan(list(irr0.values())[0][:, 2])).sum().item()
    n_reg = (~torch.isnan(list(reg0.values())[0][:, 2])).sum().item()
    print(f"  irr_map: {n_irr}/{N_grid} valid per step  |  reg_map: {n_reg}/{N_grid}")

    # ── Shared optimizer settings ──────────────────────────────────────────────
    LBFGS_LR    = 1.0
    LBFGS_STEPS = 5
    LBFGS_HIST  = 10
    LBFGS_EVAL  = 20
    DWL_STEPS   = 5
    LAT_COL, LON_COL, VAL_COL, TIME_COL = 0, 1, 2, 3

    MODELS     = ['Vecc_Matern', 'Vecc_Cauchy', 'DW']
    param_cols = ['sigmasq_est', 'range_lat_est', 'range_lon_est',
                  'range_t_est', 'advec_lat_est', 'advec_lon_est', 'nugget_est']
    p_labels_  = ['sigmasq', 'range_lat', 'range_lon', 'range_t',
                  'advec_lat', 'advec_lon', 'nugget']
    true_vals_ = [true_dict['sigmasq'],    true_dict['range_lat'], true_dict['range_lon'],
                  true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
                  true_dict['nugget']]

    records = []
    skipped = 0

    # ──────────────────────────────────────────────────────────────────────────
    # SIMULATION LOOP
    # ──────────────────────────────────────────────────────────────────────────
    for it in range(num_iters):
        print(f"\n{'='*60}")
        print(f"  Iteration {it+1}/{num_iters}  (skipped so far: {skipped})")
        print(f"{'='*60}")

        yr_it, d_it, step3_assignment_per_t, hr_idx_per_t, src_locs_per_t = \
            all_day_mappings[rng.integers(len(all_day_mappings))]
        print(f"  Obs pattern : {yr_it} day {d_it}")

        initial_vals = make_random_init(rng)
        init_orig    = backmap_params(initial_vals)
        print(f"  Init        : sigmasq={init_orig['sigmasq']:.3f}  "
              f"range_lon={init_orig['range_lon']:.3f}  nugget={init_orig['nugget']:.3f}")

        try:
            field = generate_field_values(lats_hr, lons_hr, 8, true_params, dlat_hr, dlon_hr)
            (irr_map, irr_agg), (reg_map, reg_agg) = assemble_datasets(
                field, step3_assignment_per_t, hr_idx_per_t, src_locs_per_t,
                DUMMY_KEYS, grid_coords, true_params)
            del field

            irr_map_ord = {k: v[ord_grid] for k, v in irr_map.items()}

            # ──────────────────────────────────────────────────────────────────
            # Model 1 : Vecc_Matern  (Matérn ν=0.5, irregular source locations)
            # ──────────────────────────────────────────────────────────────────
            print("--- Model 1: Vecc_Matern ---")
            p_mat = [torch.tensor([val], device=DEVICE, dtype=DTYPE, requires_grad=True)
                     for val in initial_vals]
            model_mat = kernels_vecchia.fit_vecchia_lbfgs(
                smooth=v, input_map=irr_map_ord,
                nns_map=nns_grid, mm_cond_number=mm_cond_number, nheads=nheads,
                limit_A=limit_a, limit_B=limit_b, limit_C=limit_c, daily_stride=daily_stride
            )
            model_mat.precompute_conditioning_sets()
            opt_mat = model_mat.set_optimizer(p_mat, lr=LBFGS_LR, max_iter=LBFGS_EVAL,
                                               max_eval=LBFGS_EVAL, history_size=LBFGS_HIST)
            t0 = time.time()
            out_mat, _ = model_mat.fit_vecc_lbfgs(p_mat, opt_mat,
                                                   max_steps=LBFGS_STEPS, grad_tol=1e-5)
            t_mat          = time.time() - t0
            rmsre_mat, est_mat = calculate_rmsre(out_mat, true_dict)
            print(f"  RMSRE = {rmsre_mat:.4f}  ({t_mat:.1f}s)")

            # ──────────────────────────────────────────────────────────────────
            # Model 2 : Vecc_Cauchy  (Generalized Cauchy β, same irregular locs)
            # ──────────────────────────────────────────────────────────────────
            print(f"--- Model 2: Vecc_Cauchy (β={gc_beta}) ---")
            p_cau = [torch.tensor([val], device=DEVICE, dtype=DTYPE, requires_grad=True)
                     for val in initial_vals]
            model_cau = kernels_vecchia_cauchy.fit_cauchy_vecchia_lbfgs(
                smooth=v, gc_beta=gc_beta,
                input_map=irr_map_ord,
                nns_map=nns_grid, mm_cond_number=mm_cond_number, nheads=nheads,
                limit_A=limit_a, limit_B=limit_b, limit_C=limit_c, daily_stride=daily_stride
            )
            model_cau.precompute_conditioning_sets()
            opt_cau = model_cau.set_optimizer(p_cau, lr=LBFGS_LR, max_iter=LBFGS_EVAL,
                                               max_eval=LBFGS_EVAL, history_size=LBFGS_HIST)
            t0 = time.time()
            out_cau, _ = model_cau.fit_vecc_lbfgs(p_cau, opt_cau,
                                                   max_steps=LBFGS_STEPS, grad_tol=1e-5)
            t_cau          = time.time() - t0
            rmsre_cau, est_cau = calculate_rmsre(out_cau, true_dict)
            print(f"  RMSRE = {rmsre_cau:.4f}  ({t_cau:.1f}s)")

            # ──────────────────────────────────────────────────────────────────
            # Model 3 : DW  (Debiased Whittle, regular re-gridded data)
            # ──────────────────────────────────────────────────────────────────
            print("--- Model 3: DW ---")
            p_dw = [torch.tensor([val], device=DEVICE, dtype=DTYPE, requires_grad=True)
                    for val in initial_vals]
            dwl = debiased_whittle.debiased_whittle_likelihood()

            db = debiased_whittle.debiased_whittle_preprocess(
                [reg_agg], [reg_map], day_idx=0,
                params_list=[
                    true_dict['sigmasq'],    true_dict['range_lat'], true_dict['range_lon'],
                    true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
                    true_dict['nugget']
                ],
                lat_range=lat_r, lon_range=lon_r
            )
            cur_df   = db.generate_spatially_filtered_days(
                lat_r[0], lat_r[1], lon_r[0], lon_r[1]).to(DEVICE)
            unique_t = torch.unique(cur_df[:, TIME_COL])
            time_slices = [cur_df[cur_df[:, TIME_COL] == t] for t in unique_t]

            J_vec, n1, n2, p_time, taper, obs_masks = dwl.generate_Jvector_tapered_mv(
                time_slices, dwl.cgn_hamming, LAT_COL, LON_COL, VAL_COL, DEVICE)
            I_samp = dwl.calculate_sample_periodogram_vectorized(J_vec)
            t_auto = dwl.calculate_taper_autocorrelation_multivariate(
                taper, obs_masks, n1, n2, DEVICE)
            del obs_masks

            opt_dw = torch.optim.LBFGS(
                p_dw, lr=1.0, max_iter=LBFGS_EVAL, max_eval=LBFGS_EVAL,
                history_size=LBFGS_HIST, line_search_fn="strong_wolfe",
                tolerance_grad=1e-5)
            t0 = time.time()
            _, _, _, loss_dw, _ = dwl.run_lbfgs_tapered(
                params_list=p_dw, optimizer=opt_dw, I_sample=I_samp,
                n1=n1, n2=n2, p_time=p_time, taper_autocorr_grid=t_auto,
                max_steps=DWL_STEPS, device=DEVICE)
            t_dw           = time.time() - t0
            out_dw         = [p.item() for p in p_dw]
            rmsre_dw, est_dw = calculate_rmsre(out_dw, true_dict)
            print(f"  RMSRE = {rmsre_dw:.4f}  ({t_dw:.1f}s)")

        except Exception as e:
            skipped += 1
            import traceback
            print(f"  [SKIP] Iteration {it+1}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        # ── Record ────────────────────────────────────────────────────────────
        for model_name, est_d, rmsre_val, elapsed in [
            ('Vecc_Matern', est_mat, rmsre_mat, t_mat),
            ('Vecc_Cauchy', est_cau, rmsre_cau, t_cau),
            ('DW',          est_dw,  rmsre_dw,  t_dw),
        ]:
            records.append({
                'iter':          it + 1,
                'obs_year':      yr_it,
                'obs_day':       d_it,
                'model':         model_name,
                'gc_beta':       gc_beta,
                'rmsre':         round(rmsre_val, 6),
                'time_s':        round(elapsed,   2),
                'sigmasq_est':   round(est_d['sigmasq'],    6),
                'range_lat_est': round(est_d['range_lat'],  6),
                'range_lon_est': round(est_d['range_lon'],  6),
                'range_t_est':   round(est_d['range_time'], 6),
                'advec_lat_est': round(est_d['advec_lat'],  6),
                'advec_lon_est': round(est_d['advec_lon'],  6),
                'nugget_est':    round(est_d['nugget'],     6),
                'init_sigmasq':  round(init_orig['sigmasq'],    4),
                'init_range_lon':round(init_orig['range_lon'],  4),
            })

        pd.DataFrame(records).to_csv(output_path / csv_raw, index=False)

        # ── Running summary ────────────────────────────────────────────────────
        n_done = len([r for r in records if r['model'] == 'Vecc_Matern'])
        print(f"\n  ── Running summary ({n_done} completed / {it+1} attempted) ──")
        cw = 12
        hdr = f"  {'param':<11} {'true':>{cw}}" + "".join(f"  {m:>{cw}}" for m in MODELS)
        print(hdr)
        print(f"  {'-'*60}")
        for lbl, col, tv in zip(p_labels_, param_cols, true_vals_):
            row = f"  {lbl:<11} {tv:>{cw}.4f}"
            for m in MODELS:
                vals = [r[col] for r in records if r['model'] == m]
                row += f"  {np.mean(vals):>{cw}.4f}" if vals else f"  {'---':>{cw}}"
            print(row)

        print(f"\n  [P90-P10 per param]")
        for lbl, col, tv in zip(p_labels_, param_cols, true_vals_):
            row = f"  {lbl:<11} {'':>{cw}}"
            for m in MODELS:
                vals = [r[col] for r in records if r['model'] == m]
                if len(vals) >= 2:
                    p90p10 = np.percentile(vals, 90) - np.percentile(vals, 10)
                    row += f"  {p90p10:>{cw}.4f}"
                else:
                    row += f"  {'---':>{cw}}"
            print(row)

    # ── Final summary ─────────────────────────────────────────────────────────
    df_final = pd.DataFrame(records)

    def param_rmsre(sub, col, tv):
        return float(np.sqrt(np.mean(((sub[col].values - tv) / abs(tv)) ** 2)))

    def param_mdare(sub, col, tv):
        return float(np.median(np.abs((sub[col].values - tv) / abs(tv))))

    def param_p90p10(sub, col, tv):
        return float(np.percentile(sub[col].values, 90) - np.percentile(sub[col].values, 10))

    col_w = 14
    print(f"\n{'='*75}")
    print(f"  FINAL SUMMARY — {num_iters} iterations  DGP=Matérn(ν={v})  Cauchy(β={gc_beta})")
    print(f"{'='*75}")
    header = f"  {'Parameter':<14} {'True':>10}" + "".join(f"  {m:>{col_w}}" for m in MODELS)
    print(header)
    print(f"  {'-'*75}")

    for lbl, col, tv in zip(p_labels_, param_cols, true_vals_):
        row_str = f"  {lbl:<14} {tv:>10.4f}"
        for m in MODELS:
            sub = df_final[df_final['model'] == m]
            row_str += f"  {param_rmsre(sub, col, tv):>{col_w}.4f}"
        print(row_str)

    print(f"  {'-'*75}")
    for metric_lbl, fn in [('Overall RMSRE', param_rmsre), ('Overall MdARE', param_mdare),
                            ('Overall P90P10', param_p90p10)]:
        row = f"  {metric_lbl:<14} {'':>10}"
        for m in MODELS:
            sub = df_final[df_final['model'] == m]
            vals = [fn(sub, col, tv) for col, tv in zip(param_cols, true_vals_)]
            row += f"  {np.mean(vals):>{col_w}.4f}"
        print(row)

    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary_rows = []
    for lbl, col, tv in zip(p_labels_, param_cols, true_vals_):
        row = {'parameter': lbl, 'true': tv}
        for m in MODELS:
            sub = df_final[df_final['model'] == m]
            row[f'{m}_rmsre']   = round(param_rmsre(sub, col, tv),  6)
            row[f'{m}_p90p10']  = round(param_p90p10(sub, col, tv), 6)
            row[f'{m}_mean']    = round(sub[col].mean(), 6)
            row[f'{m}_sd']      = round(sub[col].std(),  6)
        summary_rows.append(row)
    overall = {'parameter': 'Overall', 'true': float('nan')}
    for m in MODELS:
        sub = df_final[df_final['model'] == m]
        overall[f'{m}_rmsre']  = round(np.mean([param_rmsre(sub, c, tv)
                                                 for c, tv in zip(param_cols, true_vals_)]), 6)
        overall[f'{m}_p90p10'] = round(np.mean([param_p90p10(sub, c, tv)
                                                  for c, tv in zip(param_cols, true_vals_)]), 6)
        overall[f'{m}_mean'] = float('nan')
        overall[f'{m}_sd']   = float('nan')
    summary_rows.append(overall)

    pd.DataFrame(summary_rows).to_csv(output_path / csv_summary, index=False)
    print(f"\n  Saved: {csv_raw}")
    print(f"  Saved: {csv_summary}")
    print(f"  Total skipped: {skipped}/{num_iters}")


if __name__ == "__main__":
    app()
