"""
sim_dw_filter_comparison_032626.py

Two-way simulation study comparing Debiased Whittle with two prewhitening filters:
  DW_old : filter [[-2,1],[1,0]]  — sum-of-differences, DC-only exclusion
  DW_new : filter [[-1,1],[1,-1]] — separable product,   w1=0 row + w2=0 col exclusion

Both models use the same:
  - simulated FFT field (ground truth params)
  - regular gridded data (step3 obs→cell mapping)
  - random initialization (perturbed from true)

Imported explicitly to avoid any ambiguity:
  debiased_whittle      → old filter logic  (debiased_whittle.py)
  debiased_whittle_new  → new filter logic  (debiased_whittle_new.py)

Usage (local):
  conda activate faiss_env
  cd /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation
  python sim_dw_filter_comparison_032626.py --num-iters 20 --lat-factor 10 --lon-factor 4

Usage (Amaral):
  python sim_dw_filter_comparison_032626.py --num-iters 100
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

# ── Path setup ────────────────────────────────────────────────────────────────
import os
AMAREL_SRC = "/home/jl2815/tco"
LOCAL_SRC  = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
_src = AMAREL_SRC if os.path.exists(AMAREL_SRC) else LOCAL_SRC
sys.path.insert(0, _src)

from GEMS_TCO import configuration as config
from GEMS_TCO import debiased_whittle          as dw_old   # old filter, DC-only exclusion
from GEMS_TCO import debiased_whittle_new      as dw_new   # new filter, axis exclusion
from GEMS_TCO.data_loader import load_data_dynamic_processed

# Detect environment by checking if Amaral data path exists
is_amarel = os.path.exists(config.amarel_data_load_path)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64
DELTA_LAT_BASE = 0.044
DELTA_LON_BASE = 0.063

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})


# ── Covariance kernel (shared ground truth) ───────────────────────────────────

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
    CPU   = torch.device("cpu")
    F32   = torch.float32
    Nx, Ny, Nt = len(lats_hr), len(lons_hr), t_steps
    Px, Py, Pt = 2*Nx, 2*Ny, 2*Nt
    lx = torch.arange(Px, device=CPU, dtype=F32) * dlat; lx[Px//2:] -= Px * dlat
    ly = torch.arange(Py, device=CPU, dtype=F32) * dlon; ly[Py//2:] -= Py * dlon
    lt = torch.arange(Pt, device=CPU, dtype=F32);        lt[Pt//2:] -= Pt
    params_cpu = params.cpu().float()
    Lx, Ly, Lt = torch.meshgrid(lx, ly, lt, indexing='ij')
    C     = get_covariance_on_grid(Lx, Ly, Lt, params_cpu)
    S     = torch.fft.fftn(C)
    S.real = torch.clamp(S.real, min=0)
    noise = torch.fft.fftn(torch.randn(Px, Py, Pt, device=CPU, dtype=F32))
    field = torch.fft.ifftn(torch.sqrt(S.real) * noise).real[:Nx, :Ny, :Nt]
    return field.to(dtype=DTYPE, device=DEVICE)


# ── Step3: obs → cell, 1:1 assignment ────────────────────────────────────────

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
    return assignment


def precompute_mapping_indices(ref_day_map, lats_hr, lons_hr, grid_coords, sorted_keys):
    hr_lat_g, hr_lon_g = torch.meshgrid(lats_hr, lons_hr, indexing='ij')
    hr_coords_np = torch.stack(
        [hr_lat_g.flatten(), hr_lon_g.flatten()], dim=1).cpu().numpy()
    hr_tree   = BallTree(np.radians(hr_coords_np), metric='haversine')
    grid_coords_np = grid_coords.cpu().numpy()
    grid_tree = BallTree(np.radians(grid_coords_np), metric='haversine')

    step3_per_t, hr_idx_per_t, src_locs_per_t = [], [], []
    for key in sorted_keys:
        ref_t     = ref_day_map[key].to(DEVICE)
        src_locs  = ref_t[:, :2]
        src_np    = src_locs.cpu().numpy()
        valid_mask = ~np.isnan(src_np).any(axis=1)
        src_np_valid = src_np[valid_mask]
        assignment = apply_step3_1to1(src_np_valid, grid_coords_np, grid_tree)
        step3_per_t.append(assignment)
        if valid_mask.sum() > 0:
            _, hr_idx = hr_tree.query(np.radians(src_np_valid), k=1)
            hr_idx_per_t.append(torch.tensor(hr_idx.flatten(), device=DEVICE))
        else:
            hr_idx_per_t.append(torch.zeros(0, dtype=torch.long, device=DEVICE))
        src_locs_per_t.append(src_locs[valid_mask])
    return step3_per_t, hr_idx_per_t, src_locs_per_t


# ── Regular-grid dataset assembly ─────────────────────────────────────────────

def assemble_reg_dataset(field, step3_per_t, hr_idx_per_t, src_locs_per_t,
                          sorted_keys, grid_coords, true_params, t_offset=21.0):
    """Produce regular-grid dataset (same as reg_map in three-model sim)."""
    nugget_std = torch.sqrt(torch.exp(true_params[6]))
    N_grid     = grid_coords.shape[0]
    field_flat = field.reshape(-1, 8)

    reg_map, reg_list = {}, []
    for t_idx, key in enumerate(sorted_keys):
        t_val   = float(t_offset + t_idx)
        assign  = step3_per_t[t_idx]
        hr_idx  = hr_idx_per_t[t_idx]
        src_locs = src_locs_per_t[t_idx]
        N_valid = hr_idx.shape[0]

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
        reg_rows  = torch.full((N_grid, 11), NaN, device=DEVICE, dtype=DTYPE)
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


# ── Parameter backmap & RMSRE ─────────────────────────────────────────────────

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
    est      = backmap_params(out_params)
    est_arr  = np.array([est[k] for k in
                         ['sigmasq','range_lat','range_lon','range_time',
                          'advec_lat','advec_lon','nugget']])
    true_arr = np.array([true_dict[k] for k in
                         ['sigmasq','range_lat','range_lon','range_time',
                          'advec_lat','advec_lon','nugget']])
    return float(np.sqrt(np.mean(((est_arr - true_arr) / np.abs(true_arr)) ** 2))), est


# ── Run one DW model ──────────────────────────────────────────────────────────

def run_dw(dw_module, reg_map, reg_agg, lat_r, lon_r, initial_vals,
           true_dict, device, dw_steps=3):
    """Run one DW estimation using the given module (dw_new or dw_old)."""
    LAT_COL, LON_COL, VAL_COL, TIME_COL = 0, 1, 2, 3
    dwl = dw_module.debiased_whittle_likelihood()
    db  = dw_module.debiased_whittle_preprocess(
        [reg_agg], [reg_map], day_idx=0,
        params_list=[true_dict['sigmasq'], true_dict['range_lat'], true_dict['range_lon'],
                     true_dict['range_time'], true_dict['advec_lat'],
                     true_dict['advec_lon'], true_dict['nugget']],
        lat_range=lat_r, lon_range=lon_r
    )
    cur_df      = db.generate_spatially_filtered_days(
        lat_r[0], lat_r[1], lon_r[0], lon_r[1]).to(device)
    unique_t    = torch.unique(cur_df[:, TIME_COL])
    time_slices = [cur_df[cur_df[:, TIME_COL] == t] for t in unique_t]

    p_dw = [torch.tensor([val], device=device, dtype=DTYPE, requires_grad=True)
            for val in initial_vals]
    J_vec, n1, n2, p_time, taper, obs_masks = dwl.generate_Jvector_tapered_mv(
        time_slices, dwl.cgn_hamming, LAT_COL, LON_COL, VAL_COL, device)
    I_samp = dwl.calculate_sample_periodogram_vectorized(J_vec)
    t_auto = dwl.calculate_taper_autocorrelation_multivariate(taper, obs_masks, n1, n2, device)
    del obs_masks
    opt_dw = torch.optim.LBFGS(
        p_dw, lr=1.0, max_iter=20, max_eval=20,
        history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-5)

    t0 = time.time()
    _, _, _, loss_dw, _ = dwl.run_lbfgs_tapered(
        params_list=p_dw, optimizer=opt_dw, I_sample=I_samp,
        n1=n1, n2=n2, p_time=p_time, taper_autocorr_grid=t_auto,
        max_steps=dw_steps, device=device)
    elapsed = time.time() - t0

    out = [p.item() for p in p_dw]
    rmsre, est = calculate_rmsre(out, true_dict)
    return rmsre, est, loss_dw, elapsed


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command()
def cli(
    num_iters:   int   = typer.Option(20,    help="Simulation iterations"),
    years:       str   = typer.Option("2022,2024,2025", help="Years for obs patterns"),
    month:       int   = typer.Option(7,     help="Reference month"),
    lat_range:   str   = typer.Option("-3,2",    help="lat_min,lat_max"),
    lon_range:   str   = typer.Option("121,131", help="lon_min,lon_max"),
    lat_factor:  int   = typer.Option(100,   help="High-res lat multiplier"),
    lon_factor:  int   = typer.Option(20,    help="High-res lon multiplier"),
    init_noise:  float = typer.Option(0.7,   help="Uniform noise half-width in log space"),
    dw_steps:    int   = typer.Option(3,     help="LBFGS steps per DW run"),
    seed:        int   = typer.Option(42,    help="Random seed"),
) -> None:

    rng = np.random.default_rng(seed)
    lat_r      = [float(x) for x in lat_range.split(',')]
    lon_r      = [float(x) for x in lon_range.split(',')]
    years_list = [y.strip() for y in years.split(',')]

    print(f"Device     : {DEVICE}")
    print(f"Region     : lat {lat_r}, lon {lon_r}")
    print(f"Filters    : DW_old=[[-2,1],[1,0]]  DW_new=[[-1,1],[1,-1]]")
    print(f"High-res   : lat×{lat_factor}, lon×{lon_factor}")
    print(f"Init noise : ±{init_noise} log-space (×{np.exp(init_noise):.2f} original scale)")

    output_path = Path(config.amarel_estimates_day_path if is_amarel
                       else config.mac_estimates_day_path)
    output_path.mkdir(parents=True, exist_ok=True)
    date_tag    = datetime.now().strftime("%m%d%y")
    csv_raw     = output_path / f"sim_dw_filter_comparison_{date_tag}.csv"
    csv_summary = output_path / f"sim_dw_filter_summary_{date_tag}.csv"

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

    def make_random_init(rng):
        noisy = list(true_log)
        for i in [0, 1, 2, 3, 6]:
            noisy[i] = true_log[i] + rng.uniform(-init_noise, init_noise)
        for i in [4, 5]:
            scale = max(abs(true_log[i]), 0.05)
            noisy[i] = true_log[i] + rng.uniform(-2 * scale, 2 * scale)
        return noisy

    # ── Load obs patterns ─────────────────────────────────────────────────────
    print("\n[Setup 1/4] Loading GEMS obs patterns...")
    data_loader = load_data_dynamic_processed(
        config.amarel_data_load_path if is_amarel else config.mac_data_load_path)

    year_dfmaps, year_means = {}, {}
    for yr in years_list:
        df_map_yr, _, _, monthly_mean_yr = data_loader.load_maxmin_ordered_data_bymonthyear(
            lat_lon_resolution=[1, 1], mm_cond_number=8,
            years_=[yr], months_=[month],
            lat_range=lat_r, lon_range=lon_r, is_whittle=False)
        year_dfmaps[yr] = df_map_yr
        year_means[yr]  = monthly_mean_yr
        print(f"  {yr}-{month:02d}: {len(df_map_yr)} time slots")

    # ── Regular target grid ───────────────────────────────────────────────────
    print("[Setup 2/4] Building regular target grid...")
    lats_grid = torch.arange(max(lat_r), min(lat_r) - 0.0001, -DELTA_LAT_BASE,
                              device=DEVICE, dtype=DTYPE)
    lons_grid = torch.arange(lon_r[0], lon_r[1] + 0.0001,  DELTA_LON_BASE,
                              device=DEVICE, dtype=DTYPE)
    lats_grid = torch.round(lats_grid * 10000) / 10000
    lons_grid = torch.round(lons_grid * 10000) / 10000
    g_lat, g_lon = torch.meshgrid(lats_grid, lons_grid, indexing='ij')
    grid_coords  = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)
    N_grid = grid_coords.shape[0]
    print(f"  Grid: {len(lats_grid)}×{len(lons_grid)} = {N_grid} cells")

    # ── High-res grid & mapping ───────────────────────────────────────────────
    print("[Setup 3/4] Building high-res grid and precomputing mappings...")
    lats_hr, lons_hr, dlat_hr, dlon_hr = build_high_res_grid(lat_r, lon_r,
                                                               lat_factor, lon_factor)
    print(f"  High-res: {len(lats_hr)}×{len(lons_hr)} = {len(lats_hr)*len(lons_hr):,} cells")

    DUMMY_KEYS = [f't{i}' for i in range(8)]
    all_day_mappings = []
    for yr in years_list:
        df_map_yr       = year_dfmaps[yr]
        monthly_mean_yr = year_means[yr]
        all_sorted      = sorted(df_map_yr.keys())
        n_days_yr       = len(all_sorted) // 8
        print(f"  {yr}: precomputing {n_days_yr} days...", flush=True)
        for d_idx in range(n_days_yr):
            hour_indices = [d_idx * 8, (d_idx + 1) * 8]
            ref_day_map, _ = data_loader.load_working_data(
                df_map_yr, monthly_mean_yr, hour_indices,
                ord_mm=None, dtype=DTYPE, keep_ori=True)
            day_keys = sorted(ref_day_map.keys())[:8]
            if len(day_keys) < 8:
                continue
            s3, hr_idx, src = precompute_mapping_indices(
                ref_day_map, lats_hr, lons_hr, grid_coords, day_keys)
            all_day_mappings.append((yr, d_idx, s3, hr_idx, src))
    print(f"  Total available day-patterns: {len(all_day_mappings)}")

    # ── Verify structure ──────────────────────────────────────────────────────
    print("[Setup 4/4] Verifying dataset structure...")
    _yr0, _d0, _s3_0, _hr0, _src0 = all_day_mappings[0]
    field0 = generate_field_values(lats_hr, lons_hr, 8, true_params, dlat_hr, dlon_hr)
    reg0, reg_agg0 = assemble_reg_dataset(field0, _s3_0, _hr0, _src0,
                                           DUMMY_KEYS, grid_coords, true_params)
    del field0
    n_valid0 = (~torch.isnan(list(reg0.values())[0][:, 2])).sum().item()
    print(f"  reg_map first step: {n_valid0}/{N_grid} valid cells")

    # ── Simulation loop ───────────────────────────────────────────────────────
    MODELS = ['DW_old', 'DW_new']
    records        = []
    skipped        = 0

    for it in range(num_iters):
        print(f"\n{'='*60}")
        print(f"  Iteration {it+1}/{num_iters}  (skipped so far: {skipped})")
        print(f"{'='*60}")

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
            reg_map, reg_agg = assemble_reg_dataset(
                field, step3_per_t, hr_idx_per_t, src_locs_per_t,
                DUMMY_KEYS, grid_coords, true_params)
            del field

            # ── DW_old: old filter + DC-only likelihood ────────────────────
            print("--- DW_old: filter [[-2,1],[1,0]] ---")
            rmsre_old, est_old, loss_old, t_old = run_dw(
                dw_old, reg_map, reg_agg, lat_r, lon_r, initial_vals,
                true_dict, DEVICE, dw_steps)
            print(f"  RMSRE={rmsre_old:.4f}  loss={loss_old:.4f}  ({t_old:.1f}s)")

            # ── DW_new: new filter + axis-exclusion likelihood ─────────────
            print("--- DW_new: filter [[-1,1],[1,-1]] ---")
            rmsre_new, est_new, loss_new, t_new = run_dw(
                dw_new, reg_map, reg_agg, lat_r, lon_r, initial_vals,
                true_dict, DEVICE, dw_steps)
            print(f"  RMSRE={rmsre_new:.4f}  loss={loss_new:.4f}  ({t_new:.1f}s)")

        except Exception as e:
            skipped += 1
            import traceback
            print(f"  [SKIP] iter {it+1}: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        # ── Record ─────────────────────────────────────────────────────────
        for model_name, est_d, rmsre_val, elapsed, loss_val in [
            ('DW_old', est_old, rmsre_old, t_old, loss_old),
            ('DW_new', est_new, rmsre_new, t_new, loss_new),
        ]:
            records.append({
                'iter':          it + 1,
                'obs_year':      yr_it,
                'obs_day':       d_it,
                'model':         model_name,
                'rmsre':         round(rmsre_val,       6),
                'loss':          round(float(loss_val), 6),
                'time_s':        round(elapsed,         2),
                'sigmasq_est':   round(est_d['sigmasq'],    6),
                'range_lat_est': round(est_d['range_lat'],  6),
                'range_lon_est': round(est_d['range_lon'],  6),
                'range_t_est':   round(est_d['range_time'], 6),
                'advec_lat_est': round(est_d['advec_lat'],  6),
                'advec_lon_est': round(est_d['advec_lon'],  6),
                'nugget_est':    round(est_d['nugget'],     6),
                'init_sigmasq':  round(init_orig['sigmasq'],   4),
                'init_range_lon':round(init_orig['range_lon'], 4),
            })

        # ── Save & print running summary ───────────────────────────────────
        pd.DataFrame(records).to_csv(csv_raw, index=False)

        p_cols   = ['sigmasq_est','range_lat_est','range_lon_est','range_t_est',
                    'advec_lat_est','advec_lon_est','nugget_est']
        p_labels = ['sigmasq','range_lat','range_lon','range_t',
                    'advec_lat','advec_lon','nugget']
        true_vals = [true_dict['sigmasq'],   true_dict['range_lat'], true_dict['range_lon'],
                     true_dict['range_time'],true_dict['advec_lat'], true_dict['advec_lon'],
                     true_dict['nugget']]

        n_done = len([r for r in records if r['model'] == 'DW_old'])
        print(f"\n  ── Running summary ({n_done} completed / {it+1} attempted) ──")
        cw = 12
        print(f"  {'param':<12} {'true':>{cw}}  {'DW_old':>{cw}}  {'DW_new':>{cw}}")
        print(f"  {'-'*54}")
        for lbl, col, tv in zip(p_labels, p_cols, true_vals):
            row = f"  {lbl:<12} {tv:>{cw}.4f}"
            for m in MODELS:
                vals = [r[col] for r in records if r['model'] == m]
                row += f"  {np.mean(vals):>{cw}.4f}"
            print(row)
        print(f"  {'-'*54}")
        # Per-param metrics: each computed across iterations for that parameter
        # Overall = mean over 7 params  (matches sim_three_model methodology)
        def _are(sub_recs, col, tv):
            return np.abs((np.array([r[col] for r in sub_recs]) - tv) / abs(tv))

        metric_defs = [
            ('RMSRE',   lambda s, c, t: float(np.sqrt(np.mean(_are(s,c,t)**2)))),
            ('MdARE',   lambda s, c, t: float(np.median(_are(s,c,t)))),
            ('P90-P10', lambda s, c, t: float(np.percentile(_are(s,c,t),90) - np.percentile(_are(s,c,t),10))),
        ]
        for metric_lbl, mfunc in metric_defs:
            # per-parameter row
            print(f"\n  [{metric_lbl} per param]")
            for lbl, col, tv in zip(p_labels, p_cols, true_vals):
                row = f"  {lbl:<12} {tv:>{cw}.4f}"
                for m in MODELS:
                    row += f"  {mfunc([r for r in records if r['model']==m], col, tv):>{cw}.4f}"
                print(row)
            # overall = mean of 7
            row = f"  {'Overall':<12} {'—':>{cw}}"
            for m in MODELS:
                sub_recs = [r for r in records if r['model'] == m]
                row += f"  {np.mean([mfunc(sub_recs,col,tv) for col,tv in zip(p_cols,true_vals)]):>{cw}.4f}"
            print(row)

    # ── Final summary CSV ─────────────────────────────────────────────────────
    if records:
        df_all = pd.DataFrame(records)
        summary_rows = []
        for m in MODELS:
            sub = df_all[df_all['model'] == m]
            per_param_rmsre   = []
            per_param_mare    = []
            per_param_p90_p10 = []
            row = {'model': m, 'n': len(sub)}
            for col, lbl, tv in zip(p_cols, p_labels, true_vals):
                are = (sub[col] - tv).abs() / abs(tv)
                p_rmsre   = float(np.sqrt(np.mean(((sub[col] - tv) / abs(tv)) ** 2)))
                p_mare    = float(are.median())
                p_p90_p10 = float(np.percentile(are, 90) - np.percentile(are, 10))
                per_param_rmsre.append(p_rmsre)
                per_param_mare.append(p_mare)
                per_param_p90_p10.append(p_p90_p10)
                row[f'{lbl}_mean']    = sub[col].mean()
                row[f'{lbl}_bias']    = sub[col].mean() - tv
                row[f'{lbl}_rmsre']   = p_rmsre
                row[f'{lbl}_mare']    = p_mare
                row[f'{lbl}_p90_p10'] = p_p90_p10
            # Overall = mean of 7 per-parameter values (matches sim_three_model methodology)
            row['overall_rmsre']   = float(np.mean(per_param_rmsre))
            row['overall_mare']    = float(np.mean(per_param_mare))
            row['overall_p90_p10'] = float(np.mean(per_param_p90_p10))
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(csv_summary, index=False)
        print(f"\nSaved raw    : {csv_raw}")
        print(f"Saved summary: {csv_summary}")
    else:
        print("\nNo valid iterations completed.")


if __name__ == "__main__":
    app()
