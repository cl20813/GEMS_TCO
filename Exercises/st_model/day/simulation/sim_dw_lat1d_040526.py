"""
sim_dw_lat1d_040526.py

Simulation study: Debiased Whittle with lat-only 1D differencing.

Comparison target: sim_three_model_comparison_031926.py (DW column)
  - Same data generation pipeline (FFT circulant + step3 obs→cell)
  - Same true parameters (Scenario D)
  - Same random init scheme
  - Only change: debiased_whittle_lat1d  (Z(i,j) = X(i+1,j) - X(i,j))
    vs.          debiased_whittle        (Z(i,j) = -2X(i,j)+X(i+1,j)+X(i,j+1))

Output grid after lat-only diff:
  (nlat-1) × nlon   (lon dimension unchanged, vs. (nlat-1)×(nlon-1) in v05)

Usage:
  cd /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/simulation
  python sim_dw_lat1d_040526.py --num-iters 1 --lat-factor 10 --lon-factor 4
  python sim_dw_lat1d_040526.py --num-iters 1000
"""

import sys
import time
from datetime import datetime
import numpy as np
import torch
import torch.fft
import pandas as pd
import typer
from pathlib import Path
from sklearn.neighbors import BallTree

sys.path.append("/Users/joonwonlee/Documents/GEMS_TCO-1/src")

from GEMS_TCO import debiased_whittle_lat1d as debiased_whittle
from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data_dynamic_processed

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
    """FFT circulant embedding realization on high-res grid."""
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


# ── Obs → grid mapping (step3) ────────────────────────────────────────────────

def apply_step3_1to1(src_np_valid, grid_coords_np, grid_tree):
    N_grid = len(grid_coords_np)
    if len(src_np_valid) == 0:
        return np.full(N_grid, -1, dtype=np.int64)
    dist_to_cell, cell_for_obs = grid_tree.query(np.radians(src_np_valid), k=1)
    dist_to_cell = dist_to_cell.flatten()
    cell_for_obs = cell_for_obs.flatten()
    assignment   = np.full(N_grid, -1, dtype=np.int64)
    best_dist    = np.full(N_grid, np.inf)
    for obs_i, (cell_j, d) in enumerate(zip(cell_for_obs, dist_to_cell)):
        if d < best_dist[cell_j]:
            assignment[cell_j] = obs_i
            best_dist[cell_j]  = d
    return assignment


def precompute_mapping_indices(ref_day_map, lats_hr, lons_hr, grid_coords, sorted_keys):
    hr_lat_g, hr_lon_g = torch.meshgrid(lats_hr, lons_hr, indexing='ij')
    hr_coords_np = torch.stack(
        [hr_lat_g.flatten(), hr_lon_g.flatten()], dim=1).cpu().numpy()
    hr_tree = BallTree(np.radians(hr_coords_np), metric='haversine')

    grid_coords_np = grid_coords.cpu().numpy()
    grid_tree = BallTree(np.radians(grid_coords_np), metric='haversine')

    step3_per_t, hr_idx_per_t, src_locs_per_t = [], [], []

    for key in sorted_keys:
        ref_t      = ref_day_map[key].to(DEVICE)
        src_locs   = ref_t[:, :2]
        src_np     = src_locs.cpu().numpy()
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


# ── Dataset assembly ──────────────────────────────────────────────────────────

def assemble_reg_dataset(field, step3_per_t, hr_idx_per_t, src_locs_per_t,
                         sorted_keys, grid_coords, true_params, t_offset=21.0):
    """
    Build regular-grid dataset (reg_map, reg_agg) from one FFT field realization.

    reg_map[key]: [N_grid, 11]
      Assigned cell: [grid_lat, grid_lon, sim_val, t, D1..D7]
      Unassigned   : [grid_lat, grid_lon, NaN,     t, D1..D7]

    This is the input to debiased_whittle_lat1d.debiased_whittle_preprocess.
    """
    nugget_std = torch.sqrt(torch.exp(true_params[6]))
    N_grid     = grid_coords.shape[0]
    field_flat = field.reshape(-1, 8)

    reg_map, reg_list = {}, []

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

        dummy_row = dummy.unsqueeze(0).expand(N_grid, -1)

        reg_rows = torch.full((N_grid, 11), float('nan'), device=DEVICE, dtype=DTYPE)
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


# ── Parameter back-mapping ────────────────────────────────────────────────────

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
    est_arr  = np.array([est['sigmasq'],       est['range_lat'],  est['range_lon'],
                         est['range_time'],    est['advec_lat'],  est['advec_lon'],
                         est['nugget']])
    true_arr = np.array([true_dict['sigmasq'], true_dict['range_lat'], true_dict['range_lon'],
                         true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
                         true_dict['nugget']])
    return float(np.sqrt(np.mean(((est_arr - true_arr) / np.abs(true_arr)) ** 2))), est


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command()
def cli(
    v: float = typer.Option(0.5,    help="Matern smoothness"),
    mm_cond_number: int = typer.Option(100, help="(unused: kept for API parity with three-model script)"),
    num_iters: int = typer.Option(1000, help="Simulation iterations"),
    years: str = typer.Option("2022,2024,2025", help="Years to sample obs patterns from"),
    month: int = typer.Option(7,    help="Reference month"),
    lat_range: str = typer.Option("-3,2",    help="lat_min,lat_max"),
    lon_range: str = typer.Option("121,131", help="lon_min,lon_max"),
    lat_factor: int = typer.Option(100, help="High-res lat multiplier"),
    lon_factor: int = typer.Option(20,  help="High-res lon multiplier"),
    init_noise: float = typer.Option(0.7, help="Uniform noise half-width in log space"),
    seed: int = typer.Option(42,    help="Random seed"),
) -> None:

    rng = np.random.default_rng(seed)

    lat_r  = [float(x) for x in lat_range.split(',')]
    lon_r  = [float(x) for x in lon_range.split(',')]
    years_list = [y.strip() for y in years.split(',')]

    print(f"Device     : {DEVICE}")
    print(f"Filter     : lat-only 1D diff  Z(i,j)=X(i+1,j)-X(i,j)")
    print(f"Region     : lat {lat_r}, lon {lon_r}")
    print(f"Years      : {years_list}  month={month}")
    print(f"Iterations : {num_iters}")
    print(f"Init noise : ±{init_noise} log-space")

    output_path = Path(config.amarel_estimates_day_path)
    output_path.mkdir(parents=True, exist_ok=True)
    date_tag    = datetime.now().strftime("%m%d%y")
    csv_raw     = f"sim_dw_lat1d_{date_tag}.csv"
    csv_summary = f"sim_dw_lat1d_summary_{date_tag}.csv"

    # ── True parameters: Scenario D (same as three-model script) ─────────────
    true_dict = {
        'sigmasq':    10.0,
        'range_lat':  0.5,
        'range_lon':  0.6,
        'range_time': 2.5,
        'advec_lat':  0.25,
        'advec_lon':  -0.16,
        'nugget':     1.2,
    }
    phi2      = 1.0 / true_dict['range_lon']
    phi1      = true_dict['sigmasq'] * phi2
    phi3      = (true_dict['range_lon'] / true_dict['range_lat'])  ** 2
    phi4      = (true_dict['range_lon'] / true_dict['range_time']) ** 2
    true_log  = [np.log(phi1), np.log(phi2), np.log(phi3), np.log(phi4),
                 true_dict['advec_lat'], true_dict['advec_lon'], np.log(true_dict['nugget'])]
    true_params = torch.tensor(true_log, device=DEVICE, dtype=DTYPE)

    def make_random_init(rng):
        noisy = list(true_log)
        for i in [0, 1, 2, 3, 6]:
            noisy[i] = true_log[i] + rng.uniform(-init_noise, init_noise)
        for i in [4, 5]:
            scale   = max(abs(true_log[i]), 0.05)
            noisy[i] = true_log[i] + rng.uniform(-2 * scale, 2 * scale)
        return noisy

    # ── Load GEMS obs patterns ────────────────────────────────────────────────
    print("\n[Setup 1/4] Loading GEMS obs patterns...")
    data_loader = load_data_dynamic_processed(config.amarel_data_load_path)
    all_day_mappings = []
    year_dfmaps, year_means = {}, {}

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

    # ── Build regular target grid ─────────────────────────────────────────────
    print("[Setup 2/4] Building regular target grid...")
    lats_grid = torch.arange(max(lat_r), min(lat_r) - 0.0001, -DELTA_LAT_BASE,
                              device=DEVICE, dtype=DTYPE)
    lons_grid = torch.arange(lon_r[0],   lon_r[1]  + 0.0001,  DELTA_LON_BASE,
                              device=DEVICE, dtype=DTYPE)
    lats_grid = torch.round(lats_grid * 10000) / 10000
    lons_grid = torch.round(lons_grid * 10000) / 10000
    g_lat, g_lon = torch.meshgrid(lats_grid, lons_grid, indexing='ij')
    grid_coords  = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)
    N_grid       = grid_coords.shape[0]
    n_lat        = len(lats_grid)
    n_lon        = len(lons_grid)
    print(f"  Grid: {n_lat} lat × {n_lon} lon = {N_grid} cells")
    print(f"  After lat-only diff: {n_lat-1} lat × {n_lon} lon = {(n_lat-1)*n_lon} cells per time step")

    # ── Build high-res grid & precompute mappings ─────────────────────────────
    print("[Setup 3/4] Building high-res grid and precomputing obs mappings...")
    lats_hr, lons_hr, dlat_hr, dlon_hr = build_high_res_grid(lat_r, lon_r,
                                                               lat_factor, lon_factor)
    print(f"  High-res: {len(lats_hr)} lat × {len(lons_hr)} lon = {len(lats_hr)*len(lons_hr):,} cells")

    DUMMY_KEYS = [f't{i}' for i in range(8)]

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

    # ── Verify dataset structure ──────────────────────────────────────────────
    print("[Setup 4/4] Verifying dataset structure...")
    _yr0, _d0, _s3_0, _hr0, _src0 = all_day_mappings[0]
    field0   = generate_field_values(lats_hr, lons_hr, 8, true_params, dlat_hr, dlon_hr)
    reg0, _  = assemble_reg_dataset(field0, _s3_0, _hr0, _src0,
                                     DUMMY_KEYS, grid_coords, true_params)
    del field0
    first_reg  = list(reg0.values())[0]
    n_reg_valid = (~torch.isnan(first_reg[:, 2])).sum().item()
    print(f"  reg_map first step: {n_reg_valid}/{N_grid} valid (assigned) cells")

    # ── DW optimization settings ──────────────────────────────────────────────
    DWL_STEPS = 5
    LAT_COL, LON_COL, VAL_COL, TIME_COL = 0, 1, 2, 3
    dwl = debiased_whittle.debiased_whittle_likelihood()

    P_COLS   = ['sigmasq_est', 'range_lat_est', 'range_lon_est',
                'range_t_est', 'advec_lat_est', 'advec_lon_est', 'nugget_est']
    P_LABELS = ['sigmasq', 'range_lat', 'range_lon', 'range_t',
                'advec_lat', 'advec_lon', 'nugget']
    TRUE_VALS = [true_dict['sigmasq'],    true_dict['range_lat'], true_dict['range_lon'],
                 true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
                 true_dict['nugget']]

    records = []
    skipped = 0

    # ─────────────────────────────────────────────────────────────────────────
    # SIMULATION LOOP
    # ─────────────────────────────────────────────────────────────────────────
    for it in range(num_iters):
        print(f"\n{'='*55}")
        print(f"  Iter {it+1}/{num_iters}  (skipped: {skipped})")
        print(f"{'='*55}")

        # Sample obs pattern and make random init
        yr_it, d_it, step3_per_t, hr_idx_per_t, src_locs_per_t = \
            all_day_mappings[rng.integers(len(all_day_mappings))]
        initial_vals = make_random_init(rng)
        init_orig    = backmap_params(initial_vals)
        print(f"  Obs: {yr_it} day {d_it}  |  "
              f"init sigmasq={init_orig['sigmasq']:.3f}  "
              f"range_lon={init_orig['range_lon']:.3f}  "
              f"nugget={init_orig['nugget']:.3f}")

        try:
            # Generate simulated field
            field    = generate_field_values(lats_hr, lons_hr, 8, true_params, dlat_hr, dlon_hr)
            reg_map, reg_agg = assemble_reg_dataset(
                field, step3_per_t, hr_idx_per_t, src_locs_per_t,
                DUMMY_KEYS, grid_coords, true_params)
            del field

            # ── DW with lat-only 1D diff ──────────────────────────────────────
            p_dw = [torch.tensor([val], device=DEVICE, dtype=DTYPE, requires_grad=True)
                    for val in initial_vals]

            db = debiased_whittle.debiased_whittle_preprocess(
                [reg_agg], [reg_map], day_idx=0,
                params_list=[
                    true_dict['sigmasq'], true_dict['range_lat'],  true_dict['range_lon'],
                    true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
                    true_dict['nugget']
                ],
                lat_range=lat_r, lon_range=lon_r
            )

            # generate_spatially_filtered_days applies lat-only diff:
            #   Z(i,j) = X(i+1,j) - X(i,j)
            # output grid: (nlat-1) × nlon
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

            print(f"  DW grid: {n1}×{n2}, {p_time} time steps  (lat-only diff: nlon={n2} unchanged)")

            opt_dw = torch.optim.LBFGS(
                p_dw, lr=1.0, max_iter=20, max_eval=100,
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
        records.append({
            'iter':          it + 1,
            'obs_year':      yr_it,
            'obs_day':       d_it,
            'model':         'DW_lat1d',
            'rmsre':         round(rmsre_dw,         6),
            'time_s':        round(t_dw,             2),
            'loss_dw':       round(float(loss_dw),   6),
            'sigmasq_est':   round(est_dw['sigmasq'],    6),
            'range_lat_est': round(est_dw['range_lat'],  6),
            'range_lon_est': round(est_dw['range_lon'],  6),
            'range_t_est':   round(est_dw['range_time'], 6),
            'advec_lat_est': round(est_dw['advec_lat'],  6),
            'advec_lon_est': round(est_dw['advec_lon'],  6),
            'nugget_est':    round(est_dw['nugget'],     6),
            'init_sigmasq':  round(init_orig['sigmasq'],   4),
            'init_range_lon':round(init_orig['range_lon'], 4),
            'n1': n1, 'n2': n2,
        })

        # Save after every iteration
        pd.DataFrame(records).to_csv(output_path / csv_raw, index=False)

        # ── Running summary ───────────────────────────────────────────────────
        n_done = len(records)
        print(f"\n  ── Running summary ({n_done} done / {it+1} attempted) ──")
        cw = 10
        print(f"  {'param':<12} {'true':>{cw}}  {'mean':>{cw}}  {'bias':>{cw}}  {'RMSRE':>{cw}}  {'MdARE':>{cw}}  {'P90-P10':>{cw}}")
        print(f"  {'-'*80}")
        for lbl, col, tv in zip(P_LABELS, P_COLS, TRUE_VALS):
            vals = np.array([r[col] for r in records])
            mean_   = np.mean(vals)
            bias_   = mean_ - tv
            rmsre_  = np.sqrt(np.mean(((vals - tv) / abs(tv)) ** 2))
            mdare_  = np.median(np.abs((vals - tv) / abs(tv)))
            p90p10_ = np.percentile(vals, 90) - np.percentile(vals, 10)
            print(f"  {lbl:<12} {tv:>{cw}.4f}  {mean_:>{cw}.4f}  {bias_:>{cw}.4f}  {rmsre_:>{cw}.4f}  {mdare_:>{cw}.4f}  {p90p10_:>{cw}.4f}")
        rmsre_all = np.array([r['rmsre'] for r in records])
        print(f"  {'-'*80}")
        print(f"  {'Overall':<12} {'':>{cw}}  {'':>{cw}}  {'':>{cw}}  {np.mean(rmsre_all):>{cw}.4f}  {np.median(rmsre_all):>{cw}.4f}  {np.percentile(rmsre_all,90)-np.percentile(rmsre_all,10):>{cw}.4f}")

    # ── Final summary CSV ─────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  DONE: {len(records)} completed, {skipped} skipped")
    print(f"{'='*55}")

    df_final = pd.DataFrame(records)
    df_final.to_csv(output_path / csv_raw, index=False)

    summary_rows = []
    for lbl, col, tv in zip(P_LABELS, P_COLS, TRUE_VALS):
        vals = df_final[col].values
        summary_rows.append({
            'param':      lbl,
            'true':       tv,
            'mean':       round(float(np.mean(vals)),   6),
            'median':     round(float(np.median(vals)), 6),
            'bias':       round(float(np.mean(vals) - tv), 6),
            'std':        round(float(np.std(vals)),    6),
            'RMSRE':      round(float(np.sqrt(np.mean(((vals - tv) / abs(tv)) ** 2))), 6),
            'MdARE':      round(float(np.median(np.abs((vals - tv) / abs(tv)))), 6),
            'P10':        round(float(np.percentile(vals, 10)), 6),
            'P90':        round(float(np.percentile(vals, 90)), 6),
            'P90_P10':    round(float(np.percentile(vals, 90) - np.percentile(vals, 10)), 6),
        })
    summary_rows.append({
        'param':  'Overall',
        'true':   float('nan'),
        'mean':   float('nan'),
        'median': float('nan'),
        'bias':   float('nan'),
        'std':    float('nan'),
        'RMSRE':  round(float(np.mean(df_final['rmsre'].values)), 6),
        'MdARE':  round(float(np.median(df_final['rmsre'].values)), 6),
        'P10':    float('nan'), 'P90': float('nan'), 'P90_P10': float('nan'),
    })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(output_path / csv_summary, index=False)

    print(f"\nSaved:\n  {output_path / csv_raw}\n  {output_path / csv_summary}")
    print(f"\nFinal summary:\n{df_summary[['param','true','mean','bias','RMSRE','MdARE']].to_string(index=False)}")


if __name__ == "__main__":
    app()
