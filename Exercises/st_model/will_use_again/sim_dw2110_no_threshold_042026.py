"""
sim_dw2110_no_threshold_042026.py

Single-condition version of sim_dw2110_missing_sensitivity:
no distance threshold — every grid cell is matched to its nearest observation.

Assignment rule (obs → cell direction, 1:1):
  For each observation, find nearest grid cell via BallTree.
  If multiple observations compete for the same cell, closest obs wins.
  Grid cells that no observation wins → NaN (missing).
  A small fraction of cells will still be missing due to this deduplication.

Three models:
  Vecc_Irr : Vecchia at actual GEMS obs locations
  Vecc_Reg : Vecchia on regular grid (with NaN for unmatched cells)
  DW       : Debiased Whittle on same regular grid

Usage (local test):
  python sim_dw2110_no_threshold_042026.py --num-iters 3 --lat-factor 10 --lon-factor 4

Usage (Amarel):
  python sim_dw2110_no_threshold_042026.py --num-iters 100
"""

import sys
import os
import time
import pickle
from datetime import datetime
import numpy as np
import torch
import torch.fft
import pandas as pd
import typer
from pathlib import Path
from sklearn.neighbors import BallTree

AMAREL_SRC = "/home/jl2815/tco"
LOCAL_SRC  = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
_src = AMAREL_SRC if os.path.exists(AMAREL_SRC) else LOCAL_SRC
sys.path.insert(0, _src)

from GEMS_TCO import kernels_vecchia
from GEMS_TCO import orderings as _orderings
from GEMS_TCO import debiased_whittle_2110 as debiased_whittle
from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data_dynamic_processed

is_amarel = os.path.exists(config.amarel_data_load_path)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64
DELTA_LAT_BASE = 0.044
DELTA_LON_BASE = 0.063

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})


# ── Covariance kernel ─────────────────────────────────────────────────────────

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
    lats = torch.arange(min(lat_range) - 0.1, max(lat_range) + 0.1, dlat,
                        device=DEVICE, dtype=DTYPE)
    lons = torch.arange(lon_range[0] - 0.1, lon_range[1] + 0.1, dlon,
                        device=DEVICE, dtype=DTYPE)
    return lats, lons, dlat, dlon


def generate_field_values(lats_hr, lons_hr, t_steps, params, dlat, dlon):
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


# ── Obs → cell assignment (no threshold, 1:1 deduplication) ──────────────────

def assign_obs_to_cells(src_np_valid, grid_coords_np, grid_tree):
    """
    For each observation, find nearest grid cell (obs→cell direction).
    If multiple observations compete for the same cell, closest obs wins.
    Returns assignment[cell_j] = obs_i, or -1 if no obs won.
    A small fraction of cells will be -1 (missing) due to deduplication.
    """
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

    return assignment


# ── Precomputed mapping (per day-pattern) ─────────────────────────────────────

def precompute_mapping_indices(ref_day_map, lats_hr, lons_hr, grid_coords, sorted_keys):
    """
    Build per-time-step BallTree queries.
    Returns per time-step:
      step3_per_t : list of (N_grid,) int64 assignment arrays (no threshold)
      hr_idx_per_t: list of (N_valid,) long tensors (high-res indices for obs)
      src_locs_per_t: list of (N_valid, 2) tensors (obs lat/lon)
    """
    hr_lat_g, hr_lon_g = torch.meshgrid(lats_hr, lons_hr, indexing='ij')
    hr_coords_np = torch.stack(
        [hr_lat_g.flatten(), hr_lon_g.flatten()], dim=1).cpu().numpy()
    hr_tree = BallTree(np.radians(hr_coords_np), metric='haversine')

    grid_coords_np = grid_coords.cpu().numpy()
    N_grid         = len(grid_coords_np)
    grid_tree      = BallTree(np.radians(grid_coords_np), metric='haversine')

    step3_per_t  = []
    hr_idx_per_t = []
    src_locs_per_t = []

    for key in sorted_keys:
        df = ref_day_map.get(key)
        if df is None or len(df) == 0:
            step3_per_t.append(np.full(N_grid, -1, dtype=np.int64))
            hr_idx_per_t.append(torch.zeros(0, dtype=torch.long, device=DEVICE))
            src_locs_per_t.append(torch.zeros((0, 2), dtype=DTYPE, device=DEVICE))
            continue

        src_np      = df[['Source_Latitude', 'Source_Longitude']].values
        valid_mask  = ~np.isnan(src_np).any(axis=1)
        src_np_valid = src_np[valid_mask]

        assignment = assign_obs_to_cells(src_np_valid, grid_coords_np, grid_tree)
        step3_per_t.append(assignment)

        if len(src_np_valid) > 0:
            _, hr_idx = hr_tree.query(np.radians(src_np_valid), k=1)
            hr_idx_per_t.append(torch.tensor(hr_idx.flatten(), device=DEVICE))
        else:
            hr_idx_per_t.append(torch.zeros(0, dtype=torch.long, device=DEVICE))

        src_locs_per_t.append(
            torch.tensor(src_np_valid, device=DEVICE, dtype=DTYPE))

    return step3_per_t, hr_idx_per_t, src_locs_per_t


# ── Dataset assembly ──────────────────────────────────────────────────────────

def assemble_datasets(field, step3_per_t, hr_idx_per_t, src_locs_per_t,
                      sorted_keys, grid_coords, true_params, t_offset=21.0):
    """
    Build irr_map (actual obs locations) and reg_map (regular grid).
    Both use the same no-threshold assignment. Cells with no winning obs → NaN.
    """
    nugget_std = torch.sqrt(torch.exp(true_params[6]))
    N_grid     = grid_coords.shape[0]
    field_flat = field.reshape(-1, 8)

    irr_map, irr_list = {}, []
    reg_map, reg_list = {}, []
    n_obs_counts = []

    for t_idx, key in enumerate(sorted_keys):
        t_val    = float(t_offset + t_idx)
        assign   = step3_per_t[t_idx]
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

        NaN       = float('nan')
        dummy_row = dummy.unsqueeze(0).expand(N_grid, -1)
        assign_t  = torch.tensor(assign, device=DEVICE, dtype=torch.long)
        filled    = assign_t >= 0
        win_obs   = assign_t[filled]

        # ── irr_map: actual obs lat/lon ──────────────────────────────────────
        irr_rows = torch.full((N_grid, 11), NaN, device=DEVICE, dtype=DTYPE)
        irr_rows[:, 3]  = t_val
        irr_rows[:, 4:] = dummy_row
        if N_valid > 0:
            irr_rows[filled, 0] = src_locs[win_obs, 0]
            irr_rows[filled, 1] = src_locs[win_obs, 1]
            irr_rows[filled, 2] = sim_vals[win_obs]

        # ── reg_map: grid coords, NaN value where unmatched ──────────────────
        reg_rows = torch.full((N_grid, 11), NaN, device=DEVICE, dtype=DTYPE)
        reg_rows[:, :2] = grid_coords
        reg_rows[:, 3]  = t_val
        reg_rows[:, 4:] = dummy_row
        if N_valid > 0:
            reg_rows[filled, 2] = sim_vals[win_obs]

        n_obs_t = int(filled.sum().item())
        n_obs_counts.append(n_obs_t)

        irr_map[key] = irr_rows.detach()
        irr_list.append(irr_rows.detach())
        reg_map[key] = reg_rows.detach()
        reg_list.append(reg_rows.detach())

    n_obs_mean = float(np.mean(n_obs_counts))
    return (irr_map, torch.cat(irr_list, dim=0)), \
           (reg_map, torch.cat(reg_list, dim=0)), \
           n_obs_mean


# ── Spatial ordering ──────────────────────────────────────────────────────────

def compute_grid_ordering(grid_coords, mm_cond_number):
    coords_np = grid_coords.cpu().numpy()
    ord_mm    = _orderings.maxmin_cpp(coords_np)
    nns       = _orderings.find_nns_l2(locs=coords_np[ord_mm], max_nn=mm_cond_number)
    return ord_mm, nns


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
    est     = backmap_params(out_params)
    est_arr = np.array([est['sigmasq'], est['range_lat'], est['range_lon'],
                        est['range_time'], est['advec_lat'], est['advec_lon'],
                        est['nugget']])
    tru_arr = np.array([true_dict['sigmasq'], true_dict['range_lat'], true_dict['range_lon'],
                        true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
                        true_dict['nugget']])
    return float(np.sqrt(np.mean(((est_arr - tru_arr) / np.abs(tru_arr)) ** 2))), est


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command()
def cli(
    v: float            = typer.Option(0.5,   help="Matern smoothness"),
    mm_cond_number: int = typer.Option(100,   help="Vecchia neighbors"),
    nheads: int         = typer.Option(0,     help="Vecchia head points per time step"),
    limit_a: int        = typer.Option(20,    help="Set A neighbors"),
    limit_b: int        = typer.Option(20,    help="Set B neighbors"),
    limit_c: int        = typer.Option(20,    help="Set C neighbors"),
    daily_stride: int   = typer.Option(8,     help="Daily stride for Set C"),
    num_iters: int      = typer.Option(100,   help="Simulation iterations"),
    years: str          = typer.Option("2022,2024,2025", help="Years for obs patterns"),
    month: int          = typer.Option(7,     help="Reference month"),
    lat_range: str      = typer.Option("-3,2",    help="lat_min,lat_max"),
    lon_range: str      = typer.Option("121,131", help="lon_min,lon_max"),
    lat_factor: int     = typer.Option(100,   help="High-res lat multiplier"),
    lon_factor: int     = typer.Option(20,    help="High-res lon multiplier"),
    init_noise: float   = typer.Option(0.7,   help="Uniform noise half-width in log space"),
    seed: int           = typer.Option(42,    help="Random seed"),
) -> None:

    import random as _random
    rng = np.random.default_rng(seed)
    _random.seed(seed)

    lat_r      = [float(x) for x in lat_range.split(',')]
    lon_r      = [float(x) for x in lon_range.split(',')]
    years_list = [y.strip() for y in years.split(',')]

    print(f"Device      : {DEVICE}")
    print(f"Assignment  : no threshold — nearest obs, 1:1 dedup (missing from dedup only)")
    print(f"Region      : lat {lat_r}, lon {lon_r}")
    print(f"Years       : {years_list}  month={month}")
    print(f"High-res    : lat×{lat_factor}, lon×{lon_factor}")
    print(f"Init noise  : ±{init_noise} log-space")

    output_path = Path(config.amarel_estimates_day_path if is_amarel
                       else config.mac_estimates_day_path)
    output_path.mkdir(parents=True, exist_ok=True)
    date_tag    = datetime.now().strftime("%m%d%y")
    csv_raw     = f"sim_dw2110_no_thr_{date_tag}.csv"
    csv_summary = f"sim_dw2110_no_thr_summary_{date_tag}.csv"

    # ── True parameters ───────────────────────────────────────────────────────
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
    true_log = [np.log(phi1), np.log(phi2), np.log(phi3), np.log(phi4),
                true_dict['advec_lat'], true_dict['advec_lon'], np.log(true_dict['nugget'])]
    true_params = torch.tensor(true_log, device=DEVICE, dtype=DTYPE)

    def make_random_init(rng_):
        noisy = list(true_log)
        for i in [0, 1, 2, 3, 6]:
            noisy[i] = true_log[i] + rng_.uniform(-init_noise, init_noise)
        for i in [4, 5]:
            scale    = max(abs(true_log[i]), 0.05)
            noisy[i] = true_log[i] + rng_.uniform(-2 * scale, 2 * scale)
        return noisy

    # ── [Setup 1/5] Load GEMS obs patterns ───────────────────────────────────
    print("\n[Setup 1/5] Loading GEMS obs patterns...")
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
            is_whittle=False)
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

    # ── [Setup 2/5] Build regular target grid ────────────────────────────────
    print("[Setup 2/5] Building regular target grid...")
    lats_grid = torch.arange(min(lat_r), max(lat_r) + 0.0001, DELTA_LAT_BASE,
                              device=DEVICE, dtype=DTYPE)
    lons_grid = torch.arange(lon_r[0],   lon_r[1]  + 0.0001,  DELTA_LON_BASE,
                              device=DEVICE, dtype=DTYPE)
    lats_grid = torch.round(lats_grid * 10000) / 10000
    lons_grid = torch.round(lons_grid * 10000) / 10000
    g_lat, g_lon = torch.meshgrid(lats_grid, lons_grid, indexing='ij')
    grid_coords  = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)
    N_grid = grid_coords.shape[0]
    print(f"  Grid: {len(lats_grid)} lat × {len(lons_grid)} lon = {N_grid} cells")

    # ── [Setup 3/5] Build high-res grid & precompute mappings ────────────────
    print("[Setup 3/5] Building high-res grid and precomputing mappings...")
    lats_hr, lons_hr, dlat_hr, dlon_hr = build_high_res_grid(lat_r, lon_r,
                                                               lat_factor, lon_factor)
    print(f"  High-res: {len(lats_hr)} lat × {len(lons_hr)} lon = "
          f"{len(lats_hr)*len(lons_hr):,} cells")

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
            ref_day = {k: year_tco_maps[yr].get(k.split('_', 2)[-1], pd.DataFrame())
                       for k in day_keys}
            s3, hr_idx, src = precompute_mapping_indices(
                ref_day, lats_hr, lons_hr, grid_coords, day_keys)
            all_day_mappings.append((yr, d_idx, s3, hr_idx, src))

    print(f"  Total available day-patterns: {len(all_day_mappings)}")

    # Report missing rate preview for first day
    if all_day_mappings:
        _s3_0 = all_day_mappings[0][2]
        n_obs_preview = int((_s3_0[0] >= 0).sum())
        miss_pct = 100.0 * (1 - n_obs_preview / N_grid)
        print(f"\n  Missing preview (first day, t=0): "
              f"{n_obs_preview}/{N_grid} matched  "
              f"({miss_pct:.1f}% missing from dedup)")

    # ── [Setup 4/5] Grid-based maxmin ordering ────────────────────────────────
    print("[Setup 4/5] Computing grid-based maxmin ordering...")
    ord_grid, nns_grid = compute_grid_ordering(grid_coords, mm_cond_number)
    print(f"  N_grid={N_grid}, mm_cond_number={mm_cond_number}")

    # ── [Setup 5/5] Verify dataset structure ─────────────────────────────────
    print("[Setup 5/5] Verifying dataset structure...")
    _yr0, _d0, _s3_0, _hr0, _src0 = all_day_mappings[0]
    field0 = generate_field_values(lats_hr, lons_hr, 8, true_params, dlat_hr, dlon_hr)
    (irr0, _), (reg0, _), n_obs0 = assemble_datasets(
        field0, _s3_0, _hr0, _src0, DUMMY_KEYS, grid_coords, true_params)
    del field0
    n_irr_valid = (~torch.isnan(list(irr0.values())[0][:, 2])).sum().item()
    n_reg_valid = (~torch.isnan(list(reg0.values())[0][:, 2])).sum().item()
    print(f"  irr_map: {n_irr_valid}/{N_grid} valid rows")
    print(f"  reg_map: {n_reg_valid}/{N_grid} valid rows  "
          f"({100*(N_grid-n_reg_valid)/N_grid:.1f}% missing from dedup)")

    LBFGS_LR    = 1.0
    LBFGS_STEPS = 5
    LBFGS_HIST  = 10
    LBFGS_EVAL  = 20
    DWL_STEPS   = 5
    LAT_COL, LON_COL, VAL_COL, TIME_COL = 0, 1, 2, 3

    MODELS   = ['Vecc_Irr', 'Vecc_Reg', 'DW']
    P_COLS   = ['sigmasq_est', 'range_lat_est', 'range_lon_est',
                'range_t_est', 'advec_lat_est', 'advec_lon_est', 'nugget_est']
    P_LABELS = ['sigmasq', 'range_lat', 'range_lon', 'range_time',
                'advec_lat', 'advec_lon', 'nugget']
    TV_LIST  = [true_dict['sigmasq'], true_dict['range_lat'], true_dict['range_lon'],
                true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
                true_dict['nugget']]

    records = []
    skipped = 0

    # ─────────────────────────────────────────────────────────────────────────
    # SIMULATION LOOP
    # ─────────────────────────────────────────────────────────────────────────
    for it in range(num_iters):
        print(f"\n{'='*65}")
        print(f"  Iteration {it+1}/{num_iters}  (skipped: {skipped})")
        print(f"{'='*65}")

        yr_it, d_it, step3_per_t, hr_idx_per_t, src_locs_per_t = \
            all_day_mappings[rng.integers(len(all_day_mappings))]
        print(f"  Obs pattern: {yr_it} day {d_it}")

        initial_vals = make_random_init(rng)
        init_orig    = backmap_params(initial_vals)
        print(f"  Init: sigmasq={init_orig['sigmasq']:.3f}  "
              f"range_lon={init_orig['range_lon']:.3f}  "
              f"nugget={init_orig['nugget']:.3f}")

        try:
            field = generate_field_values(lats_hr, lons_hr, 8, true_params, dlat_hr, dlon_hr)

            (irr_map, irr_agg), (reg_map, reg_agg), n_obs_reg = assemble_datasets(
                field, step3_per_t, hr_idx_per_t, src_locs_per_t,
                DUMMY_KEYS, grid_coords, true_params)
            del field

            irr_map_ord = {k: v[ord_grid] for k, v in irr_map.items()}
            reg_map_ord = {k: v[ord_grid] for k, v in reg_map.items()}
            n_irr_obs   = int((~torch.isnan(list(irr_map.values())[0][:, 2])).sum())
            miss_pct    = 100.0 * (1 - n_obs_reg / N_grid)
            print(f"  obs: {n_irr_obs} irr | {int(n_obs_reg)} reg / {N_grid}  "
                  f"({miss_pct:.1f}% missing)")

            MIN_OBS = max(limit_a, limit_b, limit_c) * 2
            if n_irr_obs < MIN_OBS:
                print(f"  [SKIP] only {n_irr_obs} obs < MIN_OBS={MIN_OBS}")
                skipped += 1
                continue

            # ── Vecchia-Irregular ─────────────────────────────────────────────
            print("  [Vecc_Irr]")
            p_irr   = [torch.tensor([val], device=DEVICE, dtype=DTYPE, requires_grad=True)
                       for val in initial_vals]
            model_irr = kernels_vecchia.fit_vecchia_lbfgs(
                smooth=v, input_map=irr_map_ord,
                nns_map=nns_grid, mm_cond_number=mm_cond_number, nheads=nheads,
                limit_A=limit_a, limit_B=limit_b, limit_C=limit_c,
                daily_stride=daily_stride)
            model_irr.precompute_conditioning_sets()
            opt_irr = model_irr.set_optimizer(p_irr, lr=LBFGS_LR, max_iter=LBFGS_EVAL,
                                               history_size=LBFGS_HIST)
            t0 = time.time()
            out_irr, _ = model_irr.fit_vecc_lbfgs(p_irr, opt_irr,
                                                    max_steps=LBFGS_STEPS, grad_tol=1e-5)
            t_irr = time.time() - t0
            rmsre_irr, est_irr = calculate_rmsre(out_irr, true_dict)
            print(f"  RMSRE={rmsre_irr:.4f}  ({t_irr:.1f}s)")

            # ── Vecchia-Regular ───────────────────────────────────────────────
            print("  [Vecc_Reg]")
            p_reg   = [torch.tensor([val], device=DEVICE, dtype=DTYPE, requires_grad=True)
                       for val in initial_vals]
            model_reg = kernels_vecchia.fit_vecchia_lbfgs(
                smooth=v, input_map=reg_map_ord,
                nns_map=nns_grid, mm_cond_number=mm_cond_number, nheads=nheads,
                limit_A=limit_a, limit_B=limit_b, limit_C=limit_c,
                daily_stride=daily_stride)
            model_reg.precompute_conditioning_sets()
            opt_reg = model_reg.set_optimizer(p_reg, lr=LBFGS_LR, max_iter=LBFGS_EVAL,
                                               history_size=LBFGS_HIST)
            t0 = time.time()
            out_reg, _ = model_reg.fit_vecc_lbfgs(p_reg, opt_reg,
                                                    max_steps=LBFGS_STEPS, grad_tol=1e-5)
            t_reg = time.time() - t0
            rmsre_reg, est_reg = calculate_rmsre(out_reg, true_dict)
            print(f"  RMSRE={rmsre_reg:.4f}  ({t_reg:.1f}s)")

            # ── Debiased Whittle ──────────────────────────────────────────────
            print("  [DW]")
            p_dw = [torch.tensor([val], device=DEVICE, dtype=DTYPE, requires_grad=True)
                    for val in initial_vals]
            dwl = debiased_whittle.debiased_whittle_likelihood()
            db  = debiased_whittle.debiased_whittle_preprocess(
                [reg_agg], [reg_map], day_idx=0,
                params_list=[true_dict['sigmasq'], true_dict['range_lat'],
                             true_dict['range_lon'], true_dict['range_time'],
                             true_dict['advec_lat'], true_dict['advec_lon'],
                             true_dict['nugget']],
                lat_range=lat_r, lon_range=lon_r)
            cur_df      = db.generate_spatially_filtered_days(
                lat_r[0], lat_r[1], lon_r[0], lon_r[1]).to(DEVICE)
            unique_t    = torch.unique(cur_df[:, TIME_COL])
            time_slices = [cur_df[cur_df[:, TIME_COL] == t] for t in unique_t]

            J_vec, n1, n2, p_time, taper, obs_masks = dwl.generate_Jvector_tapered_mv(
                time_slices, dwl.cgn_hamming, LAT_COL, LON_COL, VAL_COL, DEVICE)
            I_samp = dwl.calculate_sample_periodogram_vectorized(J_vec)
            t_auto = dwl.calculate_taper_autocorrelation_multivariate(
                taper, obs_masks, n1, n2, DEVICE)
            del obs_masks

            opt_dw = torch.optim.LBFGS(
                p_dw, lr=1.0, max_iter=20, max_eval=20,
                history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-5)
            t0 = time.time()
            _, _, _, loss_dw, _ = dwl.run_lbfgs_tapered(
                params_list=p_dw, optimizer=opt_dw, I_sample=I_samp,
                n1=n1, n2=n2, p_time=p_time, taper_autocorr_grid=t_auto,
                max_steps=DWL_STEPS, device=DEVICE)
            t_dw = time.time() - t0
            out_dw = [p.item() for p in p_dw]
            rmsre_dw, est_dw = calculate_rmsre(out_dw, true_dict)
            print(f"  RMSRE={rmsre_dw:.4f}  ({t_dw:.1f}s)")

        except Exception as e:
            skipped += 1
            import traceback
            print(f"  [SKIP] iter {it+1}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        # ── Record ───────────────────────────────────────────────────────────
        for model_name, est_d, rmsre_val, elapsed in [
            ('Vecc_Irr', est_irr, rmsre_irr, t_irr),
            ('Vecc_Reg', est_reg, rmsre_reg, t_reg),
            ('DW',       est_dw,  rmsre_dw,  t_dw),
        ]:
            records.append({
                'iter':          it + 1,
                'obs_year':      yr_it,
                'obs_day':       d_it,
                'model':         model_name,
                'n_obs_reg':     round(n_obs_reg, 1),
                'n_obs_irr':     n_irr_obs,
                'n_grid':        N_grid,
                'miss_pct':      round(miss_pct, 2),
                'rmsre':         round(rmsre_val, 6),
                'time_s':        round(elapsed,   2),
                'sigmasq_est':   round(est_d['sigmasq'],    6),
                'range_lat_est': round(est_d['range_lat'],  6),
                'range_lon_est': round(est_d['range_lon'],  6),
                'range_t_est':   round(est_d['range_time'], 6),
                'advec_lat_est': round(est_d['advec_lat'],  6),
                'advec_lon_est': round(est_d['advec_lon'],  6),
                'nugget_est':    round(est_d['nugget'],     6),
                'init_sigmasq':  round(init_orig['sigmasq'],   4),
                'init_rlon':     round(init_orig['range_lon'], 4),
            })

        pd.DataFrame(records).to_csv(output_path / csv_raw, index=False)

        # ── Running summary (every 10 iters) ─────────────────────────────────
        n_done = len(records) // len(MODELS)
        if n_done % 10 == 0 or n_done == 1:
            print(f"\n  ── Running summary ({n_done} done / {it+1} attempted) ──")
            cw = 9
            for m in MODELS:
                sub = [r for r in records if r['model'] == m]
                if not sub: continue
                print(f"\n  [{m}]")
                print(f"  {'param':<12} {'true':>{cw}}  {'mean':>{cw}}  {'median':>{cw}}  "
                      f"{'bias':>{cw}}  {'RMSRE':>{cw}}  {'RMSRE_med':>{cw}}  {'P90-P10':>{cw}}")
                print(f"  {'-'*94}")
                for lbl, col, tv in zip(P_LABELS, P_COLS, TV_LIST):
                    vals = np.array([r[col] for r in sub])
                    cm   = float(np.mean(vals))
                    med  = float(np.median(vals))
                    bi   = cm - tv
                    denom = max(abs(tv), 1e-10)
                    rm   = float(np.sqrt(np.mean(((vals - tv) / denom) ** 2)))
                    rmd  = float(np.median(np.abs((vals - tv) / denom)))
                    p9p1 = float(np.percentile(vals, 90) - np.percentile(vals, 10))
                    print(f"  {lbl:<12} {tv:>{cw}.4f}  {cm:>{cw}.4f}  {med:>{cw}.4f}  "
                          f"{bi:>{cw}.4f}  {rm:>{cw}.4f}  {rmd:>{cw}.4f}  {p9p1:>{cw}.4f}")
                rv   = np.array([r['rmsre'] for r in sub])
                print(f"  {'-'*94}")
                print(f"  {'Overall':<12} {'':>{cw}}  {'':>{cw}}  {'':>{cw}}  {'':>{cw}}  "
                      f"{np.mean(rv):>{cw}.4f}  {np.median(rv):>{cw}.4f}  "
                      f"{float(np.percentile(rv,90)-np.percentile(rv,10)):>{cw}.4f}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  DONE: {len(records)//len(MODELS)} iters, {skipped} skipped")
    print(f"{'='*65}")

    df_final = pd.DataFrame(records)
    df_final.to_csv(output_path / csv_raw, index=False)

    cw = 9
    summary_rows = []
    for m in MODELS:
        df_m = df_final[df_final['model'] == m]
        if df_m.empty: continue
        print(f"\n  [{m}]  (n={len(df_m)})")
        print(f"  {'param':<12} {'true':>{cw}}  {'mean':>{cw}}  {'median':>{cw}}  "
              f"{'bias':>{cw}}  {'RMSRE':>{cw}}  {'RMSRE_med':>{cw}}  {'P90-P10':>{cw}}")
        print(f"  {'-'*94}")
        for lbl, col, tv in zip(P_LABELS, P_COLS, TV_LIST):
            if col not in df_m.columns: continue
            vals = df_m[col].values
            cm   = float(np.mean(vals))
            med  = float(np.median(vals))
            bi   = cm - tv
            denom = max(abs(tv), 1e-10)
            rm   = float(np.sqrt(np.mean(((vals - tv) / denom) ** 2)))
            rmd  = float(np.median(np.abs((vals - tv) / denom)))
            p9p1 = float(np.percentile(vals, 90) - np.percentile(vals, 10))
            print(f"  {lbl:<12} {tv:>{cw}.4f}  {cm:>{cw}.4f}  {med:>{cw}.4f}  "
                  f"{bi:>{cw}.4f}  {rm:>{cw}.4f}  {rmd:>{cw}.4f}  {p9p1:>{cw}.4f}")
            summary_rows.append({
                'model': m, 'param': lbl, 'true': tv,
                'mean': round(cm, 6), 'median': round(med, 6),
                'bias': round(bi, 6), 'std': round(float(np.std(vals)), 6),
                'RMSRE': round(rm, 6), 'RMSRE_median': round(rmd, 6),
                'P10': round(float(np.percentile(vals, 10)), 6),
                'P90': round(float(np.percentile(vals, 90)), 6),
                'P90_P10': round(p9p1, 6),
            })
        rv   = df_m['rmsre'].values
        p9p1_rv = float(np.percentile(rv, 90) - np.percentile(rv, 10))
        print(f"  {'-'*94}")
        print(f"  {'Overall':<12} {'':>{cw}}  {'':>{cw}}  {'':>{cw}}  {'':>{cw}}  "
              f"{np.mean(rv):>{cw}.4f}  {np.median(rv):>{cw}.4f}  {p9p1_rv:>{cw}.4f}")
        summary_rows.append({
            'model': m, 'param': 'Overall', 'true': float('nan'),
            'mean': float('nan'), 'median': float('nan'), 'bias': float('nan'),
            'std': float('nan'), 'RMSRE': round(float(np.mean(rv)), 6),
            'RMSRE_median': round(float(np.median(rv)), 6),
            'P10': float('nan'), 'P90': float('nan'),
            'P90_P10': round(p9p1_rv, 6),
        })

    # ── Distribution: Q1 / Median / Q3 / Min / Max ───────────────────────────
    print(f"\n{'='*65}")
    print(f"  FINAL DISTRIBUTION — Q1 / Median / Q3 / Min / Max")
    print(f"{'='*65}")
    cw2 = 9
    for m in MODELS:
        df_m = df_final[df_final['model'] == m]
        if df_m.empty: continue
        print(f"\n  [{m}]")
        print(f"  {'param':<12} {'true':>{cw2}}  {'Q1':>{cw2}}  {'median':>{cw2}}  "
              f"{'Q3':>{cw2}}  {'min':>{cw2}}  {'max':>{cw2}}")
        print(f"  {'-'*80}")
        for lbl, col, tv in zip(P_LABELS, P_COLS, TV_LIST):
            if col not in df_m.columns: continue
            vals = df_m[col].values
            q1   = float(np.percentile(vals, 25))
            med  = float(np.median(vals))
            q3   = float(np.percentile(vals, 75))
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            print(f"  {lbl:<12} {tv:>{cw2}.4f}  {q1:>{cw2}.4f}  {med:>{cw2}.4f}  "
                  f"{q3:>{cw2}.4f}  {vmin:>{cw2}.4f}  {vmax:>{cw2}.4f}")

    # ── Missing rate summary ──────────────────────────────────────────────────
    df_reg = df_final[df_final['model'] == 'Vecc_Reg']
    if len(df_reg):
        print(f"\n  Missing rate (from deduplication only):")
        print(f"  mean={df_reg['miss_pct'].mean():.2f}%  "
              f"min={df_reg['miss_pct'].min():.2f}%  "
              f"max={df_reg['miss_pct'].max():.2f}%")

    pd.DataFrame(summary_rows).to_csv(output_path / csv_summary, index=False)
    print(f"\nSaved:\n  {output_path / csv_raw}\n  {output_path / csv_summary}")


if __name__ == "__main__":
    app()
