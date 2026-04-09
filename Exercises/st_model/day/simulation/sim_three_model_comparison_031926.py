"""
sim_three_model_comparison_031926.py

Three-way simulation study comparing:
  1. Vecchia-Irregular : Vecchia at actual GEMS source observation locations
  2. Vecchia-Regular   : Vecchia on regular grid (step3 obs→cell, 1:1, no duplicates)
  3. Debiased Whittle  : DW on the same regular grid data as Vecchia-Regular

Data pipeline (mimics real GEMS processing pipeline):
  ┌─ FFT high-res field (lat × lat_factor, lon × lon_factor of base resolution)
  │
  ├─► [Simulated actual observations]  (irregular)
  │     valid obs location → nearest high-res point → + nugget noise
  │     stored as [N_grid, 11] tensor: NaN rows for unobserved cells
  │     (source lat/lon, irregular spacing)
  │
  └─► [Simulated tco_grid]  (regular, step3 applied)
        step3: each obs → nearest grid cell (obs→cell direction, 1:1, no duplicates)
        winner obs value → assigned to that grid cell
        unassigned cells → NaN value
        (grid lat/lon always present, NaN only in value column)

Both datasets use N_grid rows per time step.  Unobserved rows are flagged with
NaN in the value column (column 2).  kernels_vecchia handles NaN via is_nan_mask.

Ordering:
  maxmin ordering computed ONCE on all N_grid grid-cell coordinates.
  Same ordering applied to BOTH irr_map and reg_map.
  (mirrors real pipeline: ordering uses grid template, not source locations)

conda activate faiss_env
cd /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/local_computer
python sim_three_model_comparison_031926.py --num-iters 1 --lat-factor 10 --lon-factor 4
"""
import sys
import time
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

sys.path.append("/Users/joonwonlee/Documents/GEMS_TCO-1/src")

from GEMS_TCO import kernels_vecchia
from GEMS_TCO import orderings as _orderings
from GEMS_TCO import debiased_whittle
from GEMS_TCO import configuration as config
from GEMS_TCO import alg_optimization, BaseLogger
from GEMS_TCO.data_loader import load_data_dynamic_processed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64
DELTA_LAT_BASE = 0.044   # 1×1 grid base spacing (degrees)
DELTA_LON_BASE = 0.063

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})


# ── Covariance kernel (Matern-like, exp-class) ────────────────────────────────

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
    """Generate one FFT circulant embedding realization on the high-res grid.
    FFT computed on CPU in float32 to keep memory manageable for large grids.
    Returns field [Nx, Ny, Nt] on DEVICE in DTYPE (float64)."""
    CPU   = torch.device("cpu")
    F32   = torch.float32
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


# ── Precomputed index structures ──────────────────────────────────────────────

def apply_step3_1to1(src_np_valid, grid_coords_np, grid_tree):
    """
    Step3: obs → cell direction, 1:1 assignment, no duplicates.

    For each valid obs, find its nearest grid cell.
    If multiple obs compete for the same cell, keep the nearest one.

    Returns:
      assignment [N_grid] int array: obs_i (index in valid obs) for each cell,
                                     -1 if no obs assigned to that cell.
    """
    N_grid = len(grid_coords_np)
    if len(src_np_valid) == 0:
        return np.full(N_grid, -1, dtype=np.int64)

    dist_to_cell, cell_for_obs = grid_tree.query(np.radians(src_np_valid), k=1)
    dist_to_cell = dist_to_cell.flatten()   # [N_valid]
    cell_for_obs = cell_for_obs.flatten()   # [N_valid]

    assignment = np.full(N_grid, -1, dtype=np.int64)
    best_dist  = np.full(N_grid, np.inf)

    for obs_i, (cell_j, d) in enumerate(zip(cell_for_obs, dist_to_cell)):
        if d < best_dist[cell_j]:
            assignment[cell_j] = obs_i
            best_dist[cell_j]  = d

    return assignment


def precompute_mapping_indices(ref_day_map, lats_hr, lons_hr, grid_coords, sorted_keys):
    """
    Precompute BallTree queries (stable across iterations).

    For each time step t:
      - valid obs: non-NaN source lat/lon rows of ref_day_map[t]
      - step3_assignment_per_t[t]: [N_grid] int array
            cell_j → obs_i (index into valid obs), or -1 if unassigned
      - hr_idx_per_t[t]: [N_valid] long tensor
            for each valid obs, nearest high-res cell index
      - src_locs_per_t[t]: [N_valid, 2] tensor
            source lat/lon for valid obs

    Ordering is computed separately from grid coordinates.
    """
    # High-res BallTree (for sampling FFT field)
    hr_lat_g, hr_lon_g = torch.meshgrid(lats_hr, lons_hr, indexing='ij')
    hr_coords_np = torch.stack(
        [hr_lat_g.flatten(), hr_lon_g.flatten()], dim=1).cpu().numpy()
    hr_tree = BallTree(np.radians(hr_coords_np), metric='haversine')

    # Grid BallTree (for step3 obs→cell assignment)
    grid_coords_np = grid_coords.cpu().numpy()
    grid_tree = BallTree(np.radians(grid_coords_np), metric='haversine')

    step3_assignment_per_t = []
    hr_idx_per_t           = []
    src_locs_per_t         = []

    for key in sorted_keys:
        ref_t   = ref_day_map[key].to(DEVICE)
        src_locs = ref_t[:, :2]           # [N_grid, 2], NaN for missing obs
        src_np   = src_locs.cpu().numpy()

        # Valid obs: non-NaN source lat/lon
        valid_mask = ~np.isnan(src_np).any(axis=1)
        src_np_valid = src_np[valid_mask]

        # Step3: obs → cell, 1:1, no duplicates
        assignment = apply_step3_1to1(src_np_valid, grid_coords_np, grid_tree)
        step3_assignment_per_t.append(assignment)

        # High-res index for each valid obs
        if valid_mask.sum() > 0:
            _, hr_idx = hr_tree.query(np.radians(src_np_valid), k=1)
            hr_idx_per_t.append(torch.tensor(hr_idx.flatten(), device=DEVICE))
        else:
            hr_idx_per_t.append(torch.zeros(0, dtype=torch.long, device=DEVICE))

        src_locs_per_t.append(src_locs[valid_mask])

    return step3_assignment_per_t, hr_idx_per_t, src_locs_per_t


# ── Spatial ordering (grid-based, same for both models) ──────────────────────

def compute_grid_ordering(grid_coords, mm_cond_number):
    """
    maxmin ordering on all N_grid grid-cell coordinates.
    Used for BOTH irregular and regular Vecchia models.
    """
    coords_np = grid_coords.cpu().numpy()
    ord_mm    = _orderings.maxmin_cpp(coords_np)
    nns       = _orderings.find_nns_l2(locs=coords_np[ord_mm], max_nn=mm_cond_number)
    return ord_mm, nns


# ── Dataset assembly (per iteration) ─────────────────────────────────────────

def assemble_datasets(field, step3_assignment_per_t, hr_idx_per_t, src_locs_per_t,
                      sorted_keys, grid_coords, true_params, t_offset=21.0):
    """
    Produce two datasets from one FFT field realization.

    irr_map[key]: [N_grid, 11]
      Row j (assigned cell): [src_lat, src_lon, sim_val, t, D1..D7]
      Row j (unassigned)   : [NaN,     NaN,     NaN,     t, D1..D7]

    reg_map[key]: [N_grid, 11]
      Row j (assigned cell): [grid_lat, grid_lon, sim_val, t, D1..D7]
      Row j (unassigned)   : [grid_lat, grid_lon, NaN,     t, D1..D7]

    NaN in column 2 (value) → excluded by kernels_vecchia (is_nan_mask).
    """
    nugget_std = torch.sqrt(torch.exp(true_params[6]))
    N_grid     = grid_coords.shape[0]
    field_flat = field.reshape(-1, 8)   # [N_hr, T]

    irr_map, irr_list = {}, []
    reg_map, reg_list = {}, []

    for t_idx, key in enumerate(sorted_keys):
        t_val    = float(t_offset + t_idx)
        assign   = step3_assignment_per_t[t_idx]   # [N_grid] int64 numpy
        hr_idx   = hr_idx_per_t[t_idx]             # [N_valid]
        src_locs = src_locs_per_t[t_idx]           # [N_valid, 2]
        N_valid  = hr_idx.shape[0]

        # Time dummy: t_idx=0 → all zeros (reference), t_idx>0 → one-hot
        dummy = torch.zeros(7, device=DEVICE, dtype=DTYPE)
        if t_idx > 0:
            dummy[t_idx - 1] = 1.0

        # Sample FFT field at valid obs locations + nugget
        if N_valid > 0:
            gp_vals  = field_flat[hr_idx, t_idx]
            sim_vals = gp_vals + torch.randn(N_valid, device=DEVICE, dtype=DTYPE) * nugget_std
        else:
            sim_vals = torch.zeros(0, device=DEVICE, dtype=DTYPE)

        # ── Build full [N_grid, 11] rows ──────────────────────────────────────
        NaN = float('nan')
        dummy_row = dummy.unsqueeze(0).expand(N_grid, -1)  # [N_grid, 7]

        # Irregular: start with all NaN for lat/lon/val; fill time & dummies
        irr_rows = torch.full((N_grid, 11), NaN, device=DEVICE, dtype=DTYPE)
        irr_rows[:, 3]  = t_val
        irr_rows[:, 4:] = dummy_row

        # Regular: grid lat/lon always present; val starts NaN
        reg_rows = torch.full((N_grid, 11), NaN, device=DEVICE, dtype=DTYPE)
        reg_rows[:, :2] = grid_coords
        reg_rows[:, 3]  = t_val
        reg_rows[:, 4:] = dummy_row

        # Fill assigned cells (vectorised)
        if N_valid > 0:
            assign_t  = torch.tensor(assign, device=DEVICE, dtype=torch.long)
            filled    = assign_t >= 0                        # [N_grid] bool
            win_obs   = assign_t[filled]                     # [N_filled] obs indices

            irr_rows[filled, 0] = src_locs[win_obs, 0]      # src lat
            irr_rows[filled, 1] = src_locs[win_obs, 1]      # src lon
            irr_rows[filled, 2] = sim_vals[win_obs]          # sim value

            reg_rows[filled, 2] = sim_vals[win_obs]          # sim value

        irr_map[key] = irr_rows.detach()
        irr_list.append(irr_rows.detach())
        reg_map[key] = reg_rows.detach()
        reg_list.append(reg_rows.detach())

    return (irr_map, torch.cat(irr_list, dim=0)), (reg_map, torch.cat(reg_list, dim=0))


# ── Metrics ───────────────────────────────────────────────────────────────────

def backmap_params(out_params):
    """Convert log-reparametrized estimates → original parameter space."""
    p = out_params
    if isinstance(p[0], torch.Tensor):
        p = [x.item() if x.numel() == 1 else x[0].item() for x in p]
    p = [float(x) for x in p]
    phi2 = np.exp(p[1])
    phi3 = np.exp(p[2])
    phi4 = np.exp(p[3])
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
    est_arr  = np.array([est['sigmasq'], est['range_lat'], est['range_lon'],
                         est['range_time'], est['advec_lat'], est['advec_lon'],
                         est['nugget']])
    true_arr = np.array([true_dict['sigmasq'], true_dict['range_lat'], true_dict['range_lon'],
                         true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
                         true_dict['nugget']])
    return float(np.sqrt(np.mean(((est_arr - true_arr) / np.abs(true_arr)) ** 2))), est


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command()
def cli(
    v: float = typer.Option(0.5,    help="Matern smoothness"),
    mm_cond_number: int = typer.Option(100,  help="Vecchia neighbors"),
    nheads: int = typer.Option(0,    help="Vecchia head points per time step"),
    limit_a: int = typer.Option(20,  help="Set A neighbors"),
    limit_b: int = typer.Option(20,  help="Set B neighbors"),
    limit_c: int = typer.Option(20,  help="Set C neighbors"),
    daily_stride: int = typer.Option(8, help="Daily stride for Set C"),
    num_iters: int = typer.Option(10,  help="Simulation iterations"),
    years: str = typer.Option("2022,2024,2025", help="Years to sample obs patterns from"),
    month: int = typer.Option(7,    help="Reference month"),
    lat_range: str = typer.Option("-3,2",    help="lat_min,lat_max"),
    lon_range: str = typer.Option("121,131", help="lon_min,lon_max"),
    lat_factor: int = typer.Option(100, help="High-res lat multiplier"),
    lon_factor: int = typer.Option(20,  help="High-res lon multiplier"),
    init_noise: float = typer.Option(0.7, help="Uniform noise half-width in log space for random init"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:

    import random as _random
    rng = np.random.default_rng(seed)
    _random.seed(seed)

    lat_r = [float(x) for x in lat_range.split(',')]
    lon_r = [float(x) for x in lon_range.split(',')]
    years_list = [y.strip() for y in years.split(',')]
    print(f"Device : {DEVICE}")
    print(f"Region : lat {lat_r}, lon {lon_r}")
    print(f"Years  : {years_list}  month={month}  (obs pattern sampled randomly per iter)")
    print(f"High-res : lat×{lat_factor}, lon×{lon_factor}")
    print(f"Init noise: ±{init_noise} in log space (×{np.exp(init_noise):.2f} in original scale)")

    output_path = Path(config.amarel_estimates_day_path)
    output_path.mkdir(parents=True, exist_ok=True)
    date_tag = datetime.now().strftime("%m%d%y")   # e.g. 032026
    csv_raw     = f"sim_three_model_comparison_{date_tag}.csv"
    csv_summary = f"sim_three_model_summary_{date_tag}.csv"

    # ── True parameters ────────────────────────────────────────────────────────
    # Scenario A — original real-data fitted parameters
    #true_dict = {
    #    'sigmasq':    13.059,
    #    'range_lat':  0.154,
    #    'range_lon':  0.195,
    #    'range_time': 1.0,
    #    'advec_lat':  0.0218,
    #    'advec_lon':  -0.1689,
    #    'nugget':     0.247,
    #}

    # Scenario B (revised)
    # true_dict = {
    #     'sigmasq':    5.5,
    #     'range_lat':  0.30,
    #     'range_lon':  0.38,
    #     'range_time': 2.5,
    #     'advec_lat':  -0.12,
    #     'advec_lon':  -0.20,
    #     'nugget':     0.82,
    # }

    # Scenario C (active) — wider range, higher nugget, moderate lat / strong lon advection
    #true_dict = {
    #    'sigmasq':    10.0,
    #    'range_lat':  0.5,
    #    'range_lon':  0.6,
    #    'range_time': 2.5,
    #    'advec_lat':  0.15,
    #    'advec_lon':  -0.25,
    #    'nugget':     1.2,
    #}

    # Scenario D — strong lat advection, moderate lon advection
    true_dict = {
        'sigmasq':    10.0,
        'range_lat':  0.5,
        'range_lon':  0.6,
        'range_time': 2.5,
        'advec_lat':  0.25,
        'advec_lon':  -0.16,
        'nugget':     1.2,
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

    # ── Random initial value generator ────────────────────────────────────────
    def make_random_init(rng):
        """Perturb true_log for random optimization start.
        Log-space params (0,1,2,3,6): ±init_noise uniform in log space
            → original scale ×/÷ exp(init_noise)  (default ×/÷2.0)
        Advection params (4,5): ±2×|true| additive noise
        """
        noisy = list(true_log)
        for i in [0, 1, 2, 3, 6]:
            noisy[i] = true_log[i] + rng.uniform(-init_noise, init_noise)
        for i in [4, 5]:
            scale = max(abs(true_log[i]), 0.05)
            noisy[i] = true_log[i] + rng.uniform(-2 * scale, 2 * scale)
        return noisy

    # ── Load obs patterns from all years upfront ───────────────────────────────
    print("\n[Setup 1/5] Loading GEMS obs patterns for all years...")
    data_loader = load_data_dynamic_processed(config.amarel_data_load_path)
    all_day_mappings = []   # list of (year, day_idx, step3_per_t, hr_idx_per_t, src_locs_per_t)

    # Defer grid & high-res setup to after we know N_grid; do it in setup 2/3 below.
    # First pass: just collect sorted keys per year to count available days.
    year_dfmaps = {}
    year_means  = {}
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
        n_keys = len(df_map_yr)
        print(f"  {yr}-{month:02d}: {n_keys} time slots loaded")

    # Determine N_grid from first available DataFrame
    first_df = None
    for yr in years_list:
        if year_dfmaps[yr]:
            first_df = list(year_dfmaps[yr].values())[0]
            break
    n_obs_first = len(first_df) if first_df is not None else 0
    print(f"  N_grid (rows per time step): {n_obs_first}")

    # ── Build regular target grid ──────────────────────────────────────────────
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

    # ── Build high-res grid & precompute mapping indices for ALL days ─────────
    print("[Setup 3/5] Building high-res grid and precomputing mappings for all days...")
    lats_hr, lons_hr, dlat_hr, dlon_hr = build_high_res_grid(lat_r, lon_r,
                                                               lat_factor, lon_factor)
    print(f"  High-res: {len(lats_hr)} lat × {len(lons_hr)} lon "
          f"= {len(lats_hr)*len(lons_hr):,} cells")

    DUMMY_KEYS = [f't{i}' for i in range(8)]   # fixed key names for assembled maps

    for yr in years_list:
        df_map_yr     = year_dfmaps[yr]
        monthly_mean_yr = year_means[yr]
        all_sorted    = sorted(df_map_yr.keys())
        n_days_yr     = len(all_sorted) // 8
        print(f"  {yr}: precomputing {n_days_yr} days...", flush=True)
        for d_idx in range(n_days_yr):
            hour_indices = [d_idx * 8, (d_idx + 1) * 8]
            ref_day_map, _ = data_loader.load_working_data(
                df_map_yr, monthly_mean_yr, hour_indices,
                ord_mm=None, dtype=DTYPE, keep_ori=True
            )
            day_keys = sorted(ref_day_map.keys())[:8]
            if len(day_keys) < 8:
                continue
            s3, hr_idx, src = precompute_mapping_indices(
                ref_day_map, lats_hr, lons_hr, grid_coords, day_keys)
            all_day_mappings.append((yr, d_idx, s3, hr_idx, src))

    print(f"  Total available day-patterns: {len(all_day_mappings)}")
    # Report assignment stats from first entry
    if all_day_mappings:
        _s3 = all_day_mappings[0][2]
        n_assigned = [(a >= 0).sum() for a in _s3]
        print(f"  Step3 (sample day): {n_assigned[0]}–{max(n_assigned)} cells filled per time step")

    # ── Grid-based ordering (same for both models) ────────────────────────────
    print("[Setup 4/5] Computing grid-based maxmin ordering (shared by both models)...")
    ord_grid, nns_grid = compute_grid_ordering(grid_coords, mm_cond_number)
    print(f"  Ordering: N_grid={N_grid}, mm_cond_number={mm_cond_number}")

    # ── Verify dataset structure with one sample field ────────────────────────
    print("[Setup 5/5] Verifying dataset structure with sample field...")
    _yr0, _d0, _s3_0, _hr0, _src0 = all_day_mappings[0]
    field0 = generate_field_values(lats_hr, lons_hr, 8, true_params, dlat_hr, dlon_hr)
    (irr0, _), (reg0, _) = assemble_datasets(
        field0, _s3_0, _hr0, _src0, DUMMY_KEYS, grid_coords, true_params)
    del field0
    first_irr = list(irr0.values())[0]
    first_reg = list(reg0.values())[0]
    n_irr_valid = (~torch.isnan(first_irr[:, 2])).sum().item()
    n_reg_valid = (~torch.isnan(first_reg[:, 2])).sum().item()
    print(f"  irr_map first step: {n_irr_valid}/{N_grid} valid rows (src loc + val)")
    print(f"  reg_map first step: {n_reg_valid}/{N_grid} valid rows (grid loc + val)")

    # ── Shared optimization settings ──────────────────────────────────────────
    LBFGS_LR    = 1.0
    LBFGS_STEPS = 5
    LBFGS_HIST  = 10
    LBFGS_EVAL  = 20
    DWL_STEPS   = 5
    LAT_COL, LON_COL, VAL_COL, TIME_COL = 0, 1, 2, 3

    records = []

    # ─────────────────────────────────────────────────────────────────────────
    # SIMULATION LOOP
    # ─────────────────────────────────────────────────────────────────────────
    skipped = 0
    for it in range(num_iters):
        print(f"\n{'='*60}")
        print(f"  Iteration {it+1}/{num_iters}  (skipped so far: {skipped})")
        print(f"{'='*60}")

        # ── Randomly sample obs pattern (year + day) ──────────────────────────
        yr_it, d_it, step3_assignment_per_t, hr_idx_per_t, src_locs_per_t = \
            all_day_mappings[rng.integers(len(all_day_mappings))]
        print(f"  Obs pattern: {yr_it} day {d_it}")

        # ── Random starting point ─────────────────────────────────────────────
        initial_vals = make_random_init(rng)
        init_orig    = backmap_params(initial_vals)
        print(f"  Init sigmasq={init_orig['sigmasq']:.3f}  "
              f"range_lon={init_orig['range_lon']:.3f}  "
              f"nugget={init_orig['nugget']:.3f}")

        try:
            # Generate new FFT field
            field = generate_field_values(lats_hr, lons_hr, 8, true_params, dlat_hr, dlon_hr)

            # Assemble irregular and regular datasets from the same field
            (irr_map, irr_agg), (reg_map, reg_agg) = assemble_datasets(
                field, step3_assignment_per_t, hr_idx_per_t, src_locs_per_t,
                DUMMY_KEYS, grid_coords, true_params)
            del field

            # Apply grid-based maxmin ordering to BOTH models
            irr_map_ord = {k: v[ord_grid] for k, v in irr_map.items()}
            reg_map_ord = {k: v[ord_grid] for k, v in reg_map.items()}

            # ──────────────────────────────────────────────────────────────────────
            # Model 1: Vecchia-Irregular
            # actual GEMS source obs locations (NaN rows excluded internally)
            # ──────────────────────────────────────────────────────────────────────
            print("--- Model 1: Vecchia-Irregular ---")
            p_irr = [torch.tensor([val], device=DEVICE, dtype=DTYPE, requires_grad=True)
                     for val in initial_vals]
            model_irr = kernels_vecchia.fit_vecchia_lbfgs(
                smooth=v, input_map=irr_map_ord,
                nns_map=nns_grid, mm_cond_number=mm_cond_number, nheads=nheads,
                limit_A=limit_a, limit_B=limit_b, limit_C=limit_c, daily_stride=daily_stride
            )
            model_irr.precompute_conditioning_sets()
            opt_irr = model_irr.set_optimizer(p_irr, lr=LBFGS_LR, max_iter=LBFGS_EVAL,
                                              history_size=LBFGS_HIST)
            t0 = time.time()
            out_irr, _ = model_irr.fit_vecc_lbfgs(p_irr, opt_irr,
                                                   max_steps=LBFGS_STEPS, grad_tol=1e-5)
            t_irr          = time.time() - t0
            rmsre_irr, est_irr = calculate_rmsre(out_irr, true_dict)
            print(f"  RMSRE = {rmsre_irr:.4f}  ({t_irr:.1f}s)")

            # ──────────────────────────────────────────────────────────────────────
            # Model 2: Vecchia-Regular
            # regular grid locations (step3, 1:1); NaN rows excluded internally
            # ──────────────────────────────────────────────────────────────────────
            print("--- Model 2: Vecchia-Regular ---")
            p_reg = [torch.tensor([val], device=DEVICE, dtype=DTYPE, requires_grad=True)
                     for val in initial_vals]
            model_reg = kernels_vecchia.fit_vecchia_lbfgs(
                smooth=v, input_map=reg_map_ord,
                nns_map=nns_grid, mm_cond_number=mm_cond_number, nheads=nheads,
                limit_A=limit_a, limit_B=limit_b, limit_C=limit_c, daily_stride=daily_stride
            )
            model_reg.precompute_conditioning_sets()
            opt_reg = model_reg.set_optimizer(p_reg, lr=LBFGS_LR, max_iter=LBFGS_EVAL,
                                              history_size=LBFGS_HIST)
            t0 = time.time()
            out_reg, _ = model_reg.fit_vecc_lbfgs(p_reg, opt_reg,
                                                   max_steps=LBFGS_STEPS, grad_tol=1e-5)
            t_reg          = time.time() - t0
            rmsre_reg, est_reg = calculate_rmsre(out_reg, true_dict)
            print(f"  RMSRE = {rmsre_reg:.4f}  ({t_reg:.1f}s)")

            # ──────────────────────────────────────────────────────────────────────
            # Model 3: Debiased Whittle
            # ──────────────────────────────────────────────────────────────────────
            print("--- Model 3: Debiased Whittle ---")
            p_dw = [torch.tensor([val], device=DEVICE, dtype=DTYPE, requires_grad=True)
                    for val in initial_vals]
            dwl = debiased_whittle.debiased_whittle_likelihood()

            db = debiased_whittle.debiased_whittle_preprocess(
                [reg_agg], [reg_map], day_idx=0,
                params_list=[
                    true_dict['sigmasq'], true_dict['range_lat'], true_dict['range_lon'],
                    true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
                    true_dict['nugget']
                ],
                lat_range=lat_r, lon_range=lon_r
            )
            cur_df      = db.generate_spatially_filtered_days(
                lat_r[0], lat_r[1], lon_r[0], lon_r[1]).to(DEVICE)
            unique_t    = torch.unique(cur_df[:, TIME_COL])
            time_slices = [cur_df[cur_df[:, TIME_COL] == t] for t in unique_t]

            J_vec, n1, n2, p_time, taper, obs_masks = dwl.generate_Jvector_tapered_mv(
                time_slices, dwl.cgn_hamming, LAT_COL, LON_COL, VAL_COL, DEVICE)
            I_samp = dwl.calculate_sample_periodogram_vectorized(J_vec)
            t_auto = dwl.calculate_taper_autocorrelation_multivariate(taper, obs_masks, n1, n2, DEVICE)
            del obs_masks

            opt_dw = torch.optim.LBFGS(
                p_dw, lr=1.0, max_iter=20, max_eval=20,
                history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-5)
            t0 = time.time()
            _, _, _, loss_dw, _ = dwl.run_lbfgs_tapered(
                params_list=p_dw, optimizer=opt_dw, I_sample=I_samp,
                n1=n1, n2=n2, p_time=p_time, taper_autocorr_grid=t_auto,
                max_steps=DWL_STEPS, device=DEVICE)
            t_dw               = time.time() - t0
            out_dw             = [p.item() for p in p_dw]
            rmsre_dw, est_dw   = calculate_rmsre(out_dw, true_dict)
            print(f"  RMSRE = {rmsre_dw:.4f}  ({t_dw:.1f}s)")

        except Exception as e:
            skipped += 1
            print(f"  [SKIP] Iteration {it+1} failed: {type(e).__name__}: {e}")
            print(f"  Skipping to next iteration. (total skipped: {skipped})")
            continue

        # ── Record & save ─────────────────────────────────────────────────────
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

        df_now = pd.DataFrame(records)
        df_now.to_csv(output_path / csv_raw, index=False)

        # ── Running summary table ──────────────────────────────────────────────
        MODELS_     = ['Vecc_Irr', 'Vecc_Reg', 'DW']
        p_cols_     = ['sigmasq_est', 'range_lat_est', 'range_lon_est',
                       'range_t_est', 'advec_lat_est', 'advec_lon_est', 'nugget_est']
        p_labels_   = ['sigmasq', 'range_lat', 'range_lon', 'range_t',
                       'advec_lat', 'advec_lon', 'nugget']
        true_vals_  = [true_dict['sigmasq'],    true_dict['range_lat'], true_dict['range_lon'],
                       true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
                       true_dict['nugget']]

        n_done = len([r for r in records if r['model'] == 'Vecc_Irr'])
        print(f"\n  ── Running summary ({n_done} completed / {it+1} attempted) ──")
        cw = 10
        hdr = f"  {'param':<11} {'true':>{cw}}" + "".join(f"  {m:>{cw}}" for m in MODELS_)
        print(hdr)
        print(f"  {'-'*55}")
        for lbl, col, tv in zip(p_labels_, p_cols_, true_vals_):
            row = f"  {lbl:<11} {tv:>{cw}.4f}"
            for m in MODELS_:
                vals = [r[col] for r in records if r['model'] == m]
                row += f"  {np.mean(vals):>{cw}.4f}"
            print(row)
        print(f"  {'-'*55}")
        # Min | Q1 | Q2 | Q3 | Max per parameter
        print(f"\n  [Min | Q1 | Q2(med) | Q3 | Max]")
        for lbl, col, tv in zip(p_labels_, p_cols_, true_vals_):
            print(f"  {lbl} (true={tv:.4f})")
            for m in MODELS_:
                vals = np.array([r[col] for r in records if r['model'] == m])
                vmin = vals.min()
                q1, q2, q3 = np.percentile(vals, [25, 50, 75])
                vmax = vals.max()
                print(f"    {m:<12} {vmin:.4f} | {q1:.4f} | {q2:.4f} | {q3:.4f} | {vmax:.4f}")
        # Per-parameter RMSRE rows
        per_param_by_model = {}
        per_param_mdare_by_model = {}
        per_param_med_by_model = {}
        for m in MODELS_:
            sub_recs = [r for r in records if r['model'] == m]
            per_param_by_model[m] = [
                float(np.sqrt(np.mean([((r[col] - tv) / abs(tv)) ** 2 for r in sub_recs])))
                for col, tv in zip(p_cols_, true_vals_)
            ]
            per_param_mdare_by_model[m] = [
                float(np.median([abs((r[col] - tv) / abs(tv)) for r in sub_recs]))
                for col, tv in zip(p_cols_, true_vals_)
            ]
            per_param_med_by_model[m] = [
                float(np.percentile([r[col] for r in sub_recs], 90)
                      - np.percentile([r[col] for r in sub_recs], 10))
                for col, tv in zip(p_cols_, true_vals_)
            ]
        for metric_lbl, model_dict in [
            ('RMSRE',   per_param_by_model),
            ('MdARE',   per_param_mdare_by_model),
            ('P90-P10', per_param_med_by_model),
        ]:
            print(f"\n  [{metric_lbl} per param]")
            for lbl, idx in zip(p_labels_, range(len(p_labels_))):
                row = f"  {lbl:<11} {'':>{cw}}"
                for m in MODELS_:
                    row += f"  {model_dict[m][idx]:>{cw}.4f}"
                print(row)
            overall_row = f"  {'Overall':<11} {'':>{cw}}"
            for m in MODELS_:
                overall_row += f"  {np.mean(model_dict[m]):>{cw}.4f}"
            print(overall_row)

    # ── Final summary ─────────────────────────────────────────────────────────
    df_final   = pd.DataFrame(records)
    MODELS     = ['Vecc_Irr', 'Vecc_Reg', 'DW']
    param_cols = ['sigmasq_est', 'range_lat_est', 'range_lon_est',
                  'range_t_est', 'advec_lat_est', 'advec_lon_est', 'nugget_est']
    param_labels = ['sigmasq', 'range_lat', 'range_lon', 'range_time',
                    'advec_lat', 'advec_lon', 'nugget']
    true_vals  = [true_dict['sigmasq'],    true_dict['range_lat'], true_dict['range_lon'],
                  true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
                  true_dict['nugget']]

    # Per-parameter RMSRE: sqrt(mean(((est_k - true_k)/|true_k|)^2)) over iterations
    def param_rmsre(sub, col, tv):
        return float(np.sqrt(np.mean(((sub[col].values - tv) / abs(tv)) ** 2)))

    def param_mdare(sub, col, tv):
        return float(np.median(np.abs((sub[col].values - tv) / abs(tv))))

    def param_med_rmsre(sub, col, tv):
        return float(np.percentile(sub[col].values, 90) - np.percentile(sub[col].values, 10))

    # ── Per-parameter RMSRE table (printed) ───────────────────────────────────
    print(f"\n{'='*75}")
    print(f"  FINAL SUMMARY — Per-parameter RMSRE  ({num_iters} iterations)")
    print(f"{'='*75}")
    col_w = 14
    header = f"  {'Parameter':<14} {'True':>10}" + "".join(f"  {m:>{col_w}}" for m in MODELS)
    print(header)
    print(f"  {'-'*73}")
    for lbl, col, tv in zip(param_labels, param_cols, true_vals):
        row_str = f"  {lbl:<14} {tv:>10.4f}"
        for m in MODELS:
            sub = df_final[df_final['model'] == m]
            row_str += f"  {param_rmsre(sub, col, tv):>{col_w}.4f}"
        print(row_str)

    # Overall RMSRE row
    print(f"  {'-'*73}")
    overall_str = f"  {'Overall RMSRE':<14} {'':>10}"
    for m in MODELS:
        sub = df_final[df_final['model'] == m]
        per_param_rmsres = [param_rmsre(sub, col, tv) for col, tv in zip(param_cols, true_vals)]
        overall_str += f"  {np.mean(per_param_rmsres):>{col_w}.4f}"
    print(overall_str)
    overall_mdare_str = f"  {'Overall MdARE':<14} {'':>10}"
    for m in MODELS:
        sub = df_final[df_final['model'] == m]
        per_param_mdare = [param_mdare(sub, col, tv) for col, tv in zip(param_cols, true_vals)]
        overall_mdare_str += f"  {np.mean(per_param_mdare):>{col_w}.4f}"
    print(overall_mdare_str)
    overall_med_str = f"  {'Overall P90-P10':<14} {'':>10}"
    for m in MODELS:
        sub = df_final[df_final['model'] == m]
        per_param_med = [param_med_rmsre(sub, col, tv) for col, tv in zip(param_cols, true_vals)]
        overall_med_str += f"  {np.mean(per_param_med):>{col_w}.4f}"
    print(overall_med_str)

    # ── Per-parameter mean & SD table (printed) ───────────────────────────────
    print(f"\n  Mean estimate (SD) across {num_iters} iterations")
    print(f"  {'Parameter':<14} {'True':>10}" + "".join(f"  {'mean(SD)':>{col_w}}" for _ in MODELS))
    print(f"  {'-'*73}")
    for lbl, col, tv in zip(param_labels, param_cols, true_vals):
        row_str = f"  {lbl:<14} {tv:>10.4f}"
        for m in MODELS:
            sub = df_final[df_final['model'] == m]
            me, sd = sub[col].mean(), sub[col].std()
            row_str += f"  {me:>6.3f}({sd:.3f})"
        print(row_str)

    # ── 5-Number summary (Min | Q1 | Q2/Med | Q3 | Max) ──────────────────────
    print(f"\n{'='*75}")
    print(f"  5-NUMBER SUMMARY  (Min | Q1 | Median | Q3 | Max)")
    print(f"{'='*75}")
    for lbl, col, tv in zip(param_labels, param_cols, true_vals):
        print(f"\n  {lbl}  (true = {tv:.4f})")
        hdr5 = f"    {'Model':<12}  {'Min':>8}  {'Q1':>8}  {'Median':>8}  {'Q3':>8}  {'Max':>8}"
        print(hdr5)
        print(f"    {'-'*58}")
        for m in MODELS:
            sub  = df_final[df_final['model'] == m][col].dropna().values
            vmin = sub.min()
            q1, q2, q3 = np.percentile(sub, [25, 50, 75])
            vmax = sub.max()
            print(f"    {m:<12}  {vmin:>8.4f}  {q1:>8.4f}  {q2:>8.4f}  {q3:>8.4f}  {vmax:>8.4f}")

    # ── Save summary CSV (parameter × model) ──────────────────────────────────
    summary_rows = []
    for lbl, col, tv in zip(param_labels, param_cols, true_vals):
        row = {'parameter': lbl, 'true': tv}
        for m in MODELS:
            sub = df_final[df_final['model'] == m]
            row[f'{m}_rmsre']     = round(param_rmsre(sub, col, tv),     6)
            row[f'{m}_med_rmsre'] = round(param_med_rmsre(sub, col, tv), 6)
            row[f'{m}_mean']      = round(sub[col].mean(), 6)
            row[f'{m}_sd']        = round(sub[col].std(),  6)
        summary_rows.append(row)
    # Overall row
    overall_row = {'parameter': 'Overall_RMSRE', 'true': float('nan')}
    for m in MODELS:
        sub = df_final[df_final['model'] == m]
        per_param_rmsres = [param_rmsre(sub, col, tv)     for col, tv in zip(param_cols, true_vals)]
        per_param_med    = [param_med_rmsre(sub, col, tv) for col, tv in zip(param_cols, true_vals)]
        overall_row[f'{m}_rmsre']     = round(np.mean(per_param_rmsres), 6)
        overall_row[f'{m}_med_rmsre'] = round(np.mean(per_param_med),    6)
        overall_row[f'{m}_mean']      = float('nan')
        overall_row[f'{m}_sd']        = float('nan')
    summary_rows.append(overall_row)

    pd.DataFrame(summary_rows).to_csv(output_path / csv_summary, index=False)
    print(f"\n  Saved: {csv_raw}  (all {num_iters} iterations, raw)")
    print(f"  Saved: {csv_summary}     (per-parameter RMSRE table)")

    # ── Distribution plots (saved as PNG) ─────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')   # non-interactive — works on Amarel / no display
        import matplotlib.pyplot as plt
        from scipy.stats import gaussian_kde

        plot_dir = output_path / "plots"
        plot_dir.mkdir(exist_ok=True)

        MODEL_COLORS  = {'Vecc_Irr': '#2196F3', 'Vecc_Reg': '#FF9800', 'DW': '#4CAF50'}
        MODEL_MARKERS = {'Vecc_Irr': 'o',        'Vecc_Reg': 's',       'DW': '^'}

        # ── 1. Per-parameter plot: 3 panels (one per model), hist + KDE ──────
        for lbl, col, tv in zip(param_labels, param_cols, true_vals):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            fig.suptitle(f"Distribution of estimates: {lbl}  (true = {tv:.4f})",
                         fontsize=12, fontweight='bold')
            for ax, m in zip(axes, MODELS):
                sub  = df_final[df_final['model'] == m][col].dropna().values
                c    = MODEL_COLORS[m]
                n_b  = max(5, min(20, len(sub) // 3 + 1))
                ax.hist(sub, bins=n_b, alpha=0.35, color=c, density=True,
                        edgecolor='white', linewidth=0.5)
                if len(sub) >= 3:
                    try:
                        kde = gaussian_kde(sub)
                        xs  = np.linspace(sub.min(), sub.max(), 300)
                        ax.plot(xs, kde(xs), color=c, lw=2.0)
                    except Exception:
                        pass
                ax.axvline(tv, color='black',  lw=1.5, ls='--', label=f'true={tv:.3f}')
                ax.axvline(np.median(sub), color=c, lw=1.5, ls=':',
                           label=f'median={np.median(sub):.3f}')
                q1, q3 = np.percentile(sub, [25, 75])
                ax.axvspan(q1, q3, alpha=0.10, color=c)
                ax.set_title(m, fontsize=11)
                ax.set_xlabel(lbl, fontsize=9)
                ax.legend(fontsize=8, framealpha=0.7)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
            plt.tight_layout()
            plt.savefig(plot_dir / f"{col}_dist.png", dpi=130, bbox_inches='tight')
            plt.close()

        # ── 2. Overview: all 7 params, 3 models overlaid per panel ──────────
        n_params = len(param_labels)
        n_cols   = 2
        n_rows   = (n_params + 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = axes.flatten()
        for i, (lbl, col, tv) in enumerate(zip(param_labels, param_cols, true_vals)):
            ax = axes[i]
            for m in MODELS:
                sub = df_final[df_final['model'] == m][col].dropna().values
                c   = MODEL_COLORS[m]
                if len(sub) >= 3:
                    try:
                        kde = gaussian_kde(sub)
                        xs  = np.linspace(sub.min(), sub.max(), 300)
                        ax.plot(xs, kde(xs), color=c, lw=2.0, label=m)
                        ax.fill_between(xs, kde(xs), alpha=0.10, color=c)
                    except Exception:
                        ax.hist(sub, bins=10, alpha=0.3, color=c, density=True, label=m)
                else:
                    ax.hist(sub, bins=5, alpha=0.3, color=c, density=True, label=m)
            ax.axvline(tv, color='black', lw=1.5, ls='--', label=f'true={tv:.3f}')
            ax.set_title(f"{lbl}  (true={tv:.3f})", fontsize=10)
            ax.legend(fontsize=8, framealpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        for j in range(n_params, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(f"Parameter Estimate Distributions  ({num_iters} iterations)",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plot_dir / "all_params_overview.png", dpi=130, bbox_inches='tight')
        plt.close()

        # ── 3. Boxplot comparison across models ──────────────────────────────
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
        axes = axes.flatten()
        for i, (lbl, col, tv) in enumerate(zip(param_labels, param_cols, true_vals)):
            ax = axes[i]
            data_bp = [df_final[df_final['model'] == m][col].dropna().values for m in MODELS]
            bp = ax.boxplot(data_bp, labels=MODELS, patch_artist=True, widths=0.5,
                            medianprops={'color': 'black', 'lw': 2})
            for patch, m in zip(bp['boxes'], MODELS):
                patch.set_facecolor(MODEL_COLORS[m])
                patch.set_alpha(0.6)
            ax.axhline(tv, color='black', lw=1.5, ls='--', label=f'true={tv:.3f}')
            ax.set_title(f"{lbl}  (true={tv:.3f})", fontsize=10)
            ax.legend(fontsize=8, framealpha=0.7)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        for j in range(n_params, len(axes)):
            axes[j].set_visible(False)
        fig.suptitle(f"Boxplot of Parameter Estimates  ({num_iters} iterations)",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(plot_dir / "all_params_boxplot.png", dpi=130, bbox_inches='tight')
        plt.close()

        print(f"\n  Plots saved → {plot_dir}/")
        print(f"  - {col}_dist.png  × {n_params}  (per-param, 3-panel hist+KDE)")
        print(f"  - all_params_overview.png  (KDE overlay, all params)")
        print(f"  - all_params_boxplot.png   (boxplot comparison)")

    except ImportError as ie:
        print(f"\n  [Plot skipped — missing library: {ie}]")
    except Exception as pe:
        import traceback
        print(f"\n  [Plot generation failed: {pe}]")
        traceback.print_exc()


if __name__ == "__main__":
    app()
