"""
sim_dw_filter_local_032626.py

LOCAL TEST version of sim_dw_filter_comparison_032626.py.

Differences from the Amarel version:
  - Paths hardcoded to local Mac paths (no is_amarel detection)
  - Small defaults: num_iters=3, lat_factor=10, lon_factor=4, dw_steps=3
  - DEVICE forced to CPU

Usage:
  cd /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/will_use_again
  conda activate faiss_env
  python sim_dw_filter_local_032626.py
  python sim_dw_filter_local_032626.py --num-iters 2 --dw-steps 5
"""

import sys
import time
import os
import pickle
from datetime import datetime
import numpy as np
import torch
import torch.fft
import pandas as pd
import typer
from pathlib import Path
from sklearn.neighbors import BallTree

# ── Path setup (local only) ───────────────────────────────────────────────────
LOCAL_SRC = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
sys.path.insert(0, LOCAL_SRC)

from GEMS_TCO import configuration as config
from GEMS_TCO import debiased_whittle_raw   as dw_raw
from GEMS_TCO import debiased_whittle_2110  as dw_2110
from GEMS_TCO import debiased_whittle_1111  as dw_1111
from GEMS_TCO import debiased_whittle_mixed as dw_mixed
from GEMS_TCO.data_loader import load_data_dynamic_processed

# ── Always local ──────────────────────────────────────────────────────────────
DATA_PATH   = config.mac_data_load_path
OUTPUT_PATH = Path(config.mac_estimates_day_path)

DEVICE    = torch.device("cpu")
DEVICE_DW = torch.device("cpu")
DTYPE     = torch.float64
DELTA_LAT_BASE = 0.044
DELTA_LON_BASE = 0.063
FREQ_ALPHA     = 0.45

MODELS       = ['raw', 'filt_2110', 'filt_1111', 'mixed']
MODEL_LABELS = ['Raw', '2-1-1-0', '1-1-1-1', f'Mixed(α={FREQ_ALPHA})']
P_LABELS     = ['sigmasq', 'range_lat', 'range_lon', 'range_t',
                'advec_lat', 'advec_lon', 'nugget']

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

def build_high_res_grid(lat_range, lon_range, lat_factor, lon_factor):
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


# ── Step3 obs→cell assignment ─────────────────────────────────────────────────

def precompute_mapping_indices(ref_day_map, lats_hr, lons_hr, grid_coords, sorted_keys):
    """obs pool = tco_grid Source_Latitude/Source_Longitude (1 obs per cell)."""
    hr_lat_g, hr_lon_g = torch.meshgrid(lats_hr, lons_hr, indexing='ij')
    hr_coords_np = torch.stack(
        [hr_lat_g.flatten(), hr_lon_g.flatten()], dim=1).cpu().numpy()
    hr_tree = BallTree(np.radians(hr_coords_np), metric='haversine')

    grid_coords_np = grid_coords.cpu().numpy()
    N_grid = len(grid_coords_np)
    grid_tree = BallTree(np.radians(grid_coords_np), metric='haversine')

    step3_per_t   = []
    hr_idx_per_t  = []
    src_locs_per_t = []

    for key in sorted_keys:
        df = ref_day_map.get(key)
        if df is None or len(df) == 0:
            step3_per_t.append(np.full(N_grid, -1, dtype=np.int64))
            hr_idx_per_t.append(torch.zeros(0, dtype=torch.long, device=DEVICE))
            src_locs_per_t.append(torch.zeros((0, 2), dtype=DTYPE, device=DEVICE))
            continue

        snp  = df[['Source_Latitude', 'Source_Longitude']].values
        mask = ~np.isnan(snp).any(axis=1)
        snp  = snp[mask]

        if len(snp) == 0:
            step3_per_t.append(np.full(N_grid, -1, dtype=np.int64))
            hr_idx_per_t.append(torch.zeros(0, dtype=torch.long, device=DEVICE))
            src_locs_per_t.append(torch.zeros((0, 2), dtype=DTYPE, device=DEVICE))
            continue

        # obs→cell: each obs assigned to nearest grid cell (1:1, nearest-wins)
        _, cell_for_obs = grid_tree.query(np.radians(snp), k=1)
        cell_for_obs = cell_for_obs.flatten()
        assignment   = np.full(N_grid, -1, dtype=np.int64)
        best_dist    = np.full(N_grid, np.inf)
        dist_all, _  = grid_tree.query(np.radians(snp), k=1)
        dist_all     = dist_all.flatten()
        for obs_i, (cell_j, d) in enumerate(zip(cell_for_obs, dist_all)):
            if d < best_dist[cell_j]:
                assignment[cell_j] = obs_i
                best_dist[cell_j]  = d

        step3_per_t.append(assignment)

        _, hr_idx = hr_tree.query(np.radians(snp), k=1)
        hr_idx_per_t.append(torch.tensor(hr_idx.flatten(), device=DEVICE))
        src_locs_per_t.append(torch.tensor(snp, device=DEVICE, dtype=DTYPE))

    return step3_per_t, hr_idx_per_t, src_locs_per_t


# ── Dataset assembly ──────────────────────────────────────────────────────────

def assemble_reg_dataset(field, step3_per_t, hr_idx_per_t, src_locs_per_t,
                         sorted_keys, grid_coords, true_params):
    nugget_std = torch.sqrt(torch.exp(true_params[6]))
    N_grid     = grid_coords.shape[0]
    field_flat = field.reshape(-1, 8)

    reg_map, reg_list = {}, []

    for t_idx, key in enumerate(sorted_keys):
        assign   = step3_per_t[t_idx]
        hr_idx   = hr_idx_per_t[t_idx]
        src_locs = src_locs_per_t[t_idx]
        N_valid  = hr_idx.shape[0]

        t_val = float(21.0 + t_idx)
        dummy = torch.zeros(7, device=DEVICE, dtype=DTYPE)
        if t_idx > 0:
            dummy[t_idx - 1] = 1.0

        if N_valid > 0:
            gp_vals  = field_flat[hr_idx, t_idx]
            sim_vals = gp_vals + torch.randn(N_valid, device=DEVICE, dtype=DTYPE) * nugget_std
        else:
            sim_vals = torch.zeros(0, device=DEVICE, dtype=DTYPE)

        dummy_row = dummy.unsqueeze(0).expand(N_grid, -1)
        reg_rows  = torch.full((N_grid, 11), float('nan'), device=DEVICE, dtype=DTYPE)
        reg_rows[:, :2] = grid_coords
        reg_rows[:, 3]  = t_val
        reg_rows[:, 4:] = dummy_row

        if N_valid > 0:
            assign_t = torch.tensor(assign, device=DEVICE, dtype=torch.long)
            filled   = assign_t >= 0
            win_obs  = assign_t[filled]
            reg_rows[filled, 2] = sim_vals[win_obs]

        reg_map[key] = reg_rows.detach()
        reg_list.append(reg_rows.detach())

    return reg_map, torch.cat(reg_list, dim=0)


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
    num_iters:  int   = typer.Option(3,     help="Simulation iterations"),
    years:      str   = typer.Option("2022", help="Years to sample obs patterns"),
    month:      int   = typer.Option(7,     help="Reference month"),
    lat_range:  str   = typer.Option("-3,2",    help="lat_min,lat_max"),
    lon_range:  str   = typer.Option("121,131", help="lon_min,lon_max"),
    lat_factor: int   = typer.Option(10,    help="High-res lat multiplier (small for local)"),
    lon_factor: int   = typer.Option(4,     help="High-res lon multiplier (small for local)"),
    init_noise: float = typer.Option(0.7,   help="Uniform noise half-width in log space"),
    dw_steps:   int   = typer.Option(3,     help="LBFGS max steps per run"),
    seed:       int   = typer.Option(42,    help="Random seed"),
) -> None:

    rng        = np.random.default_rng(seed)
    lat_r      = [float(x) for x in lat_range.split(',')]
    lon_r      = [float(x) for x in lon_range.split(',')]
    years_list = [y.strip() for y in years.split(',')]

    print(f"[LOCAL TEST]")
    print(f"Device     : {DEVICE}")
    print(f"Models     : {MODEL_LABELS}")
    print(f"Region     : lat {lat_r}, lon {lon_r}")
    print(f"Years      : {years_list}  month={month}")
    print(f"Iterations : {num_iters}  |  DW steps/iter: {dw_steps}")
    print(f"High-res   : lat×{lat_factor}, lon×{lon_factor}")
    print(f"Init noise : ±{init_noise} log-space")
    print(f"Data path  : {DATA_PATH}")
    print(f"Output path: {OUTPUT_PATH}")

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    date_tag = datetime.now().strftime("%m%d%y_%H%M")
    csv_out  = f"sim_dw_4model_local_{date_tag}.csv"

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
    phi2     = 1.0 / true_dict['range_lon']
    phi1     = true_dict['sigmasq'] * phi2
    phi3     = (true_dict['range_lon'] / true_dict['range_lat'])  ** 2
    phi4     = (true_dict['range_lon'] / true_dict['range_time']) ** 2
    true_log = [np.log(phi1), np.log(phi2), np.log(phi3), np.log(phi4),
                true_dict['advec_lat'], true_dict['advec_lon'], np.log(true_dict['nugget'])]
    true_params = torch.tensor(true_log, device=DEVICE, dtype=DTYPE)

    TRUE_VALS = [true_dict['sigmasq'],    true_dict['range_lat'], true_dict['range_lon'],
                 true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
                 true_dict['nugget']]

    def make_random_init(rng):
        noisy = list(true_log)
        for i in [0, 1, 2, 3, 6]:
            noisy[i] = true_log[i] + rng.uniform(-init_noise, init_noise)
        for i in [4, 5]:
            scale    = max(abs(true_log[i]), 0.05)
            noisy[i] = true_log[i] + rng.uniform(-2*scale, 2*scale)
        return noisy

    # ── [Setup 1/4] Load GEMS obs patterns ───────────────────────────────────
    print("\n[Setup 1/4] Loading GEMS obs patterns...")
    data_loader      = load_data_dynamic_processed(DATA_PATH)
    all_day_mappings = []
    year_dfmaps, year_tco_maps = {}, {}

    for yr in years_list:
        df_map_yr, _, _, _ = data_loader.load_maxmin_ordered_data_bymonthyear(
            lat_lon_resolution=[1, 1], mm_cond_number=8,
            years_=[yr], months_=[month],
            lat_range=lat_r, lon_range=lon_r, is_whittle=False)
        year_dfmaps[yr] = df_map_yr
        print(f"  {yr}-{month:02d}: {len(df_map_yr)} time slots")

        yr2      = str(yr)[2:]
        tco_path = Path(DATA_PATH) / f"pickle_{yr}" / f"tco_grid_{yr2}_{month:02d}.pkl"
        if tco_path.exists():
            with open(tco_path, 'rb') as f:
                year_tco_maps[yr] = pickle.load(f)
            print(f"  tco_grid: {len(year_tco_maps[yr])} slots loaded")
        else:
            year_tco_maps[yr] = {}
            print(f"  [WARN] tco_grid not found: {tco_path}")

    # ── [Setup 2/4] Build regular target grid ────────────────────────────────
    print("[Setup 2/4] Building regular target grid...")
    lats_grid = torch.arange(min(lat_r), max(lat_r) + 0.0001, DELTA_LAT_BASE,
                              device=DEVICE, dtype=DTYPE)
    lons_grid = torch.arange(lon_r[0],   lon_r[1]  + 0.0001,  DELTA_LON_BASE,
                              device=DEVICE, dtype=DTYPE)
    lats_grid = torch.round(lats_grid * 10000) / 10000
    lons_grid = torch.round(lons_grid * 10000) / 10000
    g_lat, g_lon = torch.meshgrid(lats_grid, lons_grid, indexing='ij')
    grid_coords  = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)
    N_grid       = grid_coords.shape[0]
    n_lat, n_lon = len(lats_grid), len(lons_grid)
    print(f"  Grid: {n_lat} lat × {n_lon} lon = {N_grid} cells")
    print(f"  After diff filter: {n_lat-1}×{n_lon-1}")

    # ── [Setup 3/4] High-res grid & obs mappings ──────────────────────────────
    print("[Setup 3/4] Building high-res grid and precomputing obs mappings...")
    lats_hr, lons_hr, dlat_hr, dlon_hr = build_high_res_grid(lat_r, lon_r,
                                                               lat_factor, lon_factor)
    print(f"  High-res: {len(lats_hr)}×{len(lons_hr)} = {len(lats_hr)*len(lons_hr):,} cells")

    DUMMY_KEYS = [f't{i}' for i in range(8)]
    for yr in years_list:
        all_sorted = sorted(year_dfmaps[yr].keys())
        n_days_yr  = len(all_sorted) // 8
        print(f"  {yr}: precomputing {n_days_yr} days...", flush=True)
        for d_idx in range(n_days_yr):
            day_keys = all_sorted[d_idx * 8 : (d_idx + 1) * 8]
            if len(day_keys) < 8:
                continue
            # orbit_map keys have "YYYY_MM_" prefix; tco_grid keys do not — strip prefix
            ref_day = {k: year_tco_maps[yr].get(k.split('_', 2)[-1], pd.DataFrame())
                       for k in day_keys}
            s3, hr_i, src = precompute_mapping_indices(
                ref_day, lats_hr, lons_hr, grid_coords, day_keys)
            all_day_mappings.append((yr, d_idx, s3, hr_i, src))
    print(f"  Total available day-patterns: {len(all_day_mappings)}")

    if not all_day_mappings:
        print("[ERROR] No day patterns available. Check data path and year/month.")
        return

    # ── [Setup 4/4] Verify first day ──────────────────────────────────────────
    print("[Setup 4/4] Verifying dataset structure...")
    _yr0, _d0, _s3_0, _hr0, _src0 = all_day_mappings[0]
    f0    = generate_field_values(lats_hr, lons_hr, 8, true_params, dlat_hr, dlon_hr)
    r0, _ = assemble_reg_dataset(f0, _s3_0, _hr0, _src0, DUMMY_KEYS, grid_coords, true_params)
    del f0
    fv = (~torch.isnan(list(r0.values())[0][:, 2])).sum().item()
    print(f"  reg_map first step: {fv}/{N_grid} valid  ({100*fv/N_grid:.1f}%)")

    # ── DW likelihood objects ─────────────────────────────────────────────────
    dwl_raw   = dw_raw.debiased_whittle_likelihood()
    dwl_2110  = dw_2110.debiased_whittle_likelihood()
    dwl_1111  = dw_1111.debiased_whittle_likelihood()
    dwl_mixed = dw_mixed.debiased_whittle_likelihood()

    LC, NC, VC, TC = 0, 1, 2, 3
    records = []
    skipped = 0

    # ── SIMULATION LOOP ───────────────────────────────────────────────────────
    for it in range(num_iters):
        print(f"\n{'='*60}")
        print(f"  Iter {it+1}/{num_iters}  (skipped: {skipped})")
        print(f"{'='*60}")

        yr_it, d_it, step3_per_t, hr_idx_per_t, src_locs_per_t = \
            all_day_mappings[rng.integers(len(all_day_mappings))]
        initial_vals = make_random_init(rng)
        init_orig    = backmap_params(initial_vals)
        print(f"  Obs: {yr_it} day {d_it}  |  "
              f"init sig={init_orig['sigmasq']:.3f}  "
              f"rl={init_orig['range_lon']:.3f}  "
              f"nug={init_orig['nugget']:.3f}")

        rec = {'iter': it+1, 'obs_year': yr_it, 'obs_day': d_it}

        try:
            # Generate field and reg_map
            field = generate_field_values(lats_hr, lons_hr, 8, true_params, dlat_hr, dlon_hr)
            reg_map, reg_agg = assemble_reg_dataset(
                field, step3_per_t, hr_idx_per_t, src_locs_per_t,
                DUMMY_KEYS, grid_coords, true_params)
            del field

            params_kwargs = dict(
                params_list=[true_dict['sigmasq'], true_dict['range_lat'], true_dict['range_lon'],
                             true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
                             true_dict['nugget']],
                lat_range=lat_r, lon_range=lon_r)

            # Raw preprocessing (model 1 + mixed raw part)
            db_raw  = dw_raw.debiased_whittle_preprocess([reg_agg], [reg_map], day_idx=0,
                                                          **params_kwargs)
            cur_raw = db_raw.generate_spatially_filtered_days(
                lat_r[0], lat_r[1], lon_r[0], lon_r[1]).to(DEVICE_DW)
            sl_raw  = [cur_raw[cur_raw[:, TC]==t] for t in torch.unique(cur_raw[:, TC])]

            J_raw, n1, n2, p_time, tap_raw, om_raw = dwl_raw.generate_Jvector_tapered_mv(
                sl_raw, dwl_raw.cgn_hamming, LC, NC, VC, DEVICE_DW)
            I_raw  = dwl_raw.calculate_sample_periodogram_vectorized(J_raw)
            ta_raw = dwl_raw.calculate_taper_autocorrelation_multivariate(
                tap_raw, om_raw, n1, n2, DEVICE_DW)
            del om_raw

            # 2-1-1-0 preprocessing (model 2)
            db_old  = dw_2110.debiased_whittle_preprocess([reg_agg], [reg_map], day_idx=0,
                                                           **params_kwargs)
            cur_old = db_old.generate_spatially_filtered_days(
                lat_r[0], lat_r[1], lon_r[0], lon_r[1]).to(DEVICE_DW)
            sl_old  = [cur_old[cur_old[:, TC]==t] for t in torch.unique(cur_old[:, TC])]

            J_old, n1o, n2o, _, tap_old, om_old = dwl_raw.generate_Jvector_tapered_mv(
                sl_old, dwl_raw.cgn_hamming, LC, NC, VC, DEVICE_DW)
            I_old  = dwl_raw.calculate_sample_periodogram_vectorized(J_old)
            ta_old = dwl_raw.calculate_taper_autocorrelation_multivariate(
                tap_old, om_old, n1o, n2o, DEVICE_DW)
            del om_old

            # 1-1-1-1 preprocessing (model 3 + mixed diff part)
            db_new  = dw_1111.debiased_whittle_preprocess([reg_agg], [reg_map], day_idx=0,
                                                           **params_kwargs)
            cur_new = db_new.generate_spatially_filtered_days(
                lat_r[0], lat_r[1], lon_r[0], lon_r[1]).to(DEVICE_DW)
            sl_new  = [cur_new[cur_new[:, TC]==t] for t in torch.unique(cur_new[:, TC])]

            J_new, n1d, n2d, _, tap_new, om_new = dwl_raw.generate_Jvector_tapered_mv(
                sl_new, dwl_raw.cgn_hamming, LC, NC, VC, DEVICE_DW)
            I_new  = dwl_raw.calculate_sample_periodogram_vectorized(J_new)
            ta_new = dwl_raw.calculate_taper_autocorrelation_multivariate(
                tap_new, om_new, n1d, n2d, DEVICE_DW)
            del om_new

            print(f"  Raw:{n1}×{n2}  2110:{n1o}×{n2o}  1111:{n1d}×{n2d}  p={p_time}")

            # ── Model 1: Raw ──────────────────────────────────────────────────
            p1   = [torch.tensor([v], device=DEVICE_DW, dtype=DTYPE, requires_grad=True)
                    for v in initial_vals]
            opt1 = torch.optim.LBFGS(p1, lr=1.0, max_iter=20, max_eval=100,
                                      history_size=10, line_search_fn="strong_wolfe",
                                      tolerance_grad=1e-5)
            t0 = time.time()
            _, _, _, loss1, _ = dwl_raw.run_lbfgs_tapered(
                params_list=p1, optimizer=opt1, I_sample=I_raw,
                n1=n1, n2=n2, p_time=p_time, taper_autocorr_grid=ta_raw,
                max_steps=dw_steps, device=DEVICE_DW)
            t1 = time.time() - t0
            rmsre1, est1 = calculate_rmsre([p.item() for p in p1], true_dict)
            print(f"  [raw]       RMSRE={rmsre1:.4f}  ({t1:.1f}s)")

            # ── Model 2: 2-1-1-0 ─────────────────────────────────────────────
            p2   = [torch.tensor([v], device=DEVICE_DW, dtype=DTYPE, requires_grad=True)
                    for v in initial_vals]
            opt2 = torch.optim.LBFGS(p2, lr=1.0, max_iter=20, max_eval=100,
                                      history_size=10, line_search_fn="strong_wolfe",
                                      tolerance_grad=1e-5)
            t0 = time.time()
            _, _, _, loss2, _ = dwl_2110.run_lbfgs_tapered(
                params_list=p2, optimizer=opt2, I_sample=I_old,
                n1=n1o, n2=n2o, p_time=p_time, taper_autocorr_grid=ta_old,
                max_steps=dw_steps, device=DEVICE_DW)
            t2 = time.time() - t0
            rmsre2, est2 = calculate_rmsre([p.item() for p in p2], true_dict)
            print(f"  [filt_2110] RMSRE={rmsre2:.4f}  ({t2:.1f}s)")

            # ── Model 3: 1-1-1-1 ─────────────────────────────────────────────
            p3   = [torch.tensor([v], device=DEVICE_DW, dtype=DTYPE, requires_grad=True)
                    for v in initial_vals]
            opt3 = torch.optim.LBFGS(p3, lr=1.0, max_iter=20, max_eval=100,
                                      history_size=10, line_search_fn="strong_wolfe",
                                      tolerance_grad=1e-5)
            t0 = time.time()
            _, _, _, loss3, _ = dwl_1111.run_lbfgs_tapered(
                params_list=p3, optimizer=opt3, I_sample=I_new,
                n1=n1d, n2=n2d, p_time=p_time, taper_autocorr_grid=ta_new,
                max_steps=dw_steps, device=DEVICE_DW)
            t3 = time.time() - t0
            rmsre3, est3 = calculate_rmsre([p.item() for p in p3], true_dict)
            print(f"  [filt_1111] RMSRE={rmsre3:.4f}  ({t3:.1f}s)")

            # ── Model 4: Mixed ────────────────────────────────────────────────
            K1  = int(n1 * FREQ_ALPHA)
            K2  = int(n2 * FREQ_ALPHA)
            p4  = [torch.tensor([v], device=DEVICE_DW, dtype=DTYPE, requires_grad=True)
                   for v in initial_vals]
            opt4 = torch.optim.LBFGS(p4, lr=1.0, max_iter=20, max_eval=100,
                                      history_size=10, line_search_fn="strong_wolfe",
                                      tolerance_grad=1e-5)
            t0 = time.time()
            _, _, _, loss4, _ = dwl_mixed.run_lbfgs_mixed(
                params_list=p4, optimizer=opt4,
                I_samp_raw=I_raw, I_samp_diff=I_new,
                n1=n1, n2=n2, n1d=n1d, n2d=n2d,
                p_time=p_time,
                taper_auto_raw=ta_raw, taper_auto_diff=ta_new,
                K1=K1, K2=K2,
                max_steps=dw_steps, device=DEVICE_DW)
            t4 = time.time() - t0
            rmsre4, est4 = calculate_rmsre([p.item() for p in p4], true_dict)
            print(f"  [mixed]     RMSRE={rmsre4:.4f}  ({t4:.1f}s)  K1={K1} K2={K2}")

        except Exception as e:
            skipped += 1
            import traceback
            print(f"  [SKIP] iter {it+1}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        # ── Record ───────────────────────────────────────────────────────────
        for m, est, rmsre, loss, t_s, n1_, n2_ in [
            ('raw',      est1, rmsre1, loss1, t1, n1,  n2),
            ('filt_2110', est2, rmsre2, loss2, t2, n1o, n2o),
            ('filt_1111', est3, rmsre3, loss3, t3, n1d, n2d),
            ('mixed',    est4, rmsre4, loss4, t4, n1,  n2),
        ]:
            rec[f'rmsre_{m}']         = round(rmsre,        6)
            rec[f'time_s_{m}']        = round(t_s,          2)
            rec[f'loss_{m}']          = round(float(loss),  6)
            rec[f'sigmasq_est_{m}']   = round(est['sigmasq'],    6)
            rec[f'range_lat_est_{m}'] = round(est['range_lat'],  6)
            rec[f'range_lon_est_{m}'] = round(est['range_lon'],  6)
            rec[f'range_t_est_{m}']   = round(est['range_time'], 6)
            rec[f'advec_lat_est_{m}'] = round(est['advec_lat'],  6)
            rec[f'advec_lon_est_{m}'] = round(est['advec_lon'],  6)
            rec[f'nugget_est_{m}']    = round(est['nugget'],     6)
            rec[f'n1_{m}'] = n1_; rec[f'n2_{m}'] = n2_

        records.append(rec)
        pd.DataFrame(records).to_csv(OUTPUT_PATH / csv_out, index=False)

        # ── Quick summary ─────────────────────────────────────────────────────
        print(f"\n  ── Summary ({len(records)} done) ──")
        cw = 9
        for m, ml in zip(MODELS, MODEL_LABELS):
            rmsre_vals = [r[f'rmsre_{m}'] for r in records]
            print(f"  [{ml:<20}]  RMSRE mean={np.mean(rmsre_vals):.4f}  "
                  f"median={np.median(rmsre_vals):.4f}")

    print(f"\n[Done]  {len(records)} iters completed, {skipped} skipped.")
    print(f"Output: {OUTPUT_PATH / csv_out}")


if __name__ == "__main__":
    app()
