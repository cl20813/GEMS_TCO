"""
sim_cond_effect_031926.py

Simulation study: effect of conditioning number (n_cond) on Vecchia-Irregular
estimation quality, with nheads=0 FIXED.

Goal: Show that simply increasing the Vecchia conditioning set (more "local data")
does NOT substitute for long-range temporal heads.
COND_LIST = [4, 6, 8, 12, 16] applied uniformly to limit_A, limit_B, limit_C.
nheads = 0 throughout.

conda activate faiss_env
cd /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/local_computer
python sim_cond_effect_031926.py --num-iters 1 --lat-factor 10 --lon-factor 4
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

from GEMS_TCO import kernels_vecchia
from GEMS_TCO import orderings as _orderings
from GEMS_TCO import configuration as config
from GEMS_TCO import alg_optimization, BaseLogger
from GEMS_TCO.data_loader import load_data_dynamic_processed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64
DELTA_LAT_BASE = 0.044
DELTA_LON_BASE = 0.063

MM_COND_DATA = 16   # mm for data loading (fixed)

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})


# ── Covariance kernel ─────────────────────────────────────────────────────────

def get_covariance_on_grid(lx, ly, lt, params):
    params = torch.clamp(params, min=-15.0, max=15.0)
    phi1, phi2, phi3, phi4 = (torch.exp(params[i]) for i in range(4))
    u_lat = lx - params[4] * lt
    u_lon = ly - params[5] * lt
    dist  = torch.sqrt(u_lat.pow(2) * phi3 + u_lon.pow(2) + lt.pow(2) * phi4 + 1e-8)
    return (phi1 / phi2) * torch.exp(-dist * phi2)


# ── FFT field generation ──────────────────────────────────────────────────────

def build_high_res_grid(lat_range, lon_range, lat_factor=100, lon_factor=10):
    dlat = DELTA_LAT_BASE / lat_factor
    dlon = DELTA_LON_BASE / lon_factor
    lat_max, lat_min = max(lat_range), min(lat_range)
    lats = torch.arange(lat_min - 0.1, lat_max + 0.1, dlat, device=DEVICE, dtype=DTYPE)
    lons = torch.arange(lon_range[0] - 0.1, lon_range[1] + 0.1, dlon, device=DEVICE, dtype=DTYPE)
    return lats, lons, dlat, dlon


def generate_field_values(lats_hr, lons_hr, t_steps, params, dlat, dlon):
    CPU = torch.device("cpu"); F32 = torch.float32
    Nx, Ny, Nt = len(lats_hr), len(lons_hr), t_steps
    Px, Py, Pt = 2*Nx, 2*Ny, 2*Nt
    lx = torch.arange(Px, device=CPU, dtype=F32) * dlat; lx[Px//2:] -= Px * dlat
    ly = torch.arange(Py, device=CPU, dtype=F32) * dlon; ly[Py//2:] -= Py * dlon
    lt = torch.arange(Pt, device=CPU, dtype=F32);        lt[Pt//2:] -= Pt
    params_cpu = params.cpu().float()
    Lx, Ly, Lt = torch.meshgrid(lx, ly, lt, indexing='ij')
    C = get_covariance_on_grid(Lx, Ly, Lt, params_cpu)
    S = torch.fft.fftn(C); S.real = torch.clamp(S.real, min=0)
    noise = torch.fft.fftn(torch.randn(Px, Py, Pt, device=CPU, dtype=F32))
    field = torch.fft.ifftn(torch.sqrt(S.real) * noise).real[:Nx, :Ny, :Nt]
    return field.to(dtype=DTYPE, device=DEVICE)


# ── Step3 1:1 obs→cell ────────────────────────────────────────────────────────

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
    hr_coords_np = torch.stack([hr_lat_g.flatten(), hr_lon_g.flatten()], dim=1).cpu().numpy()
    hr_tree   = BallTree(np.radians(hr_coords_np), metric='haversine')
    grid_coords_np = grid_coords.cpu().numpy()
    grid_tree = BallTree(np.radians(grid_coords_np), metric='haversine')
    step3_assignment_per_t, hr_idx_per_t, src_locs_per_t = [], [], []
    for key in sorted_keys:
        ref_t    = ref_day_map[key].to(DEVICE)
        src_locs = ref_t[:, :2]
        src_np   = src_locs.cpu().numpy()
        valid_mask = ~np.isnan(src_np).any(axis=1)
        src_np_valid = src_np[valid_mask]
        assignment = apply_step3_1to1(src_np_valid, grid_coords_np, grid_tree)
        step3_assignment_per_t.append(assignment)
        if valid_mask.sum() > 0:
            _, hr_idx = hr_tree.query(np.radians(src_np_valid), k=1)
            hr_idx_per_t.append(torch.tensor(hr_idx.flatten(), device=DEVICE))
        else:
            hr_idx_per_t.append(torch.zeros(0, dtype=torch.long, device=DEVICE))
        src_locs_per_t.append(src_locs[valid_mask])
    return step3_assignment_per_t, hr_idx_per_t, src_locs_per_t


def compute_grid_ordering(grid_coords, mm_cond_number):
    coords_np = grid_coords.cpu().numpy()
    ord_mm    = _orderings.maxmin_cpp(coords_np)
    nns       = _orderings.find_nns_l2(locs=coords_np[ord_mm], max_nn=mm_cond_number)
    return ord_mm, nns


def assemble_irr_dataset(field, step3_assignment_per_t, hr_idx_per_t, src_locs_per_t,
                         sorted_keys, grid_coords, true_params, t_offset=21.0):
    """Returns irr_map only (irregular obs locations)."""
    nugget_std = torch.sqrt(torch.exp(true_params[6]))
    N_grid     = grid_coords.shape[0]
    field_flat = field.reshape(-1, 8)
    irr_map = {}
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
        if N_valid > 0:
            assign_t = torch.tensor(assign, device=DEVICE, dtype=torch.long)
            filled   = assign_t >= 0
            win_obs  = assign_t[filled]
            irr_rows[filled, 0] = src_locs[win_obs, 0]
            irr_rows[filled, 1] = src_locs[win_obs, 1]
            irr_rows[filled, 2] = sim_vals[win_obs]
        irr_map[key] = irr_rows.detach()
    return irr_map


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
    est_arr  = np.array([est['sigmasq'], est['range_lat'], est['range_lon'],
                         est['range_time'], est['advec_lat'], est['advec_lon'], est['nugget']])
    true_arr = np.array([true_dict['sigmasq'], true_dict['range_lat'], true_dict['range_lon'],
                         true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
                         true_dict['nugget']])
    return float(np.sqrt(np.mean(((est_arr - true_arr) / np.abs(true_arr)) ** 2))), est


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command()
def cli(
    v: float = typer.Option(0.5,   help="Matern smoothness"),
    cond_list: str = typer.Option("6,8,12,20", help="Comma-separated nc values to test"),
    nheads: int = typer.Option(0, help="Fixed nheads for all nc configs"),
    mm_cond_max: int = typer.Option(100, help="Max neighbors stored in NNS map"),
    daily_stride: int = typer.Option(2, help="Daily stride for Set C"),
    num_iters: int = typer.Option(200, help="Simulation iterations"),
    years: str = typer.Option("2022,2024,2025", help="Years to sample obs patterns from"),
    month: int = typer.Option(7,   help="Reference month"),
    lat_range: str = typer.Option("-3,2",    help="lat_min,lat_max"),
    lon_range: str = typer.Option("121,131", help="lon_min,lon_max"),
    lat_factor: int = typer.Option(100, help="High-res lat multiplier"),
    lon_factor: int = typer.Option(10,  help="High-res lon multiplier"),
    init_noise: float = typer.Option(0.7, help="Uniform noise half-width in log space"),
    seed: int = typer.Option(42, help="Random seed"),
) -> None:

    import random as _random
    rng = np.random.default_rng(seed)
    _random.seed(seed)

    COND_LIST    = [int(x) for x in cond_list.split(',')]
    NHEADS_FIXED = nheads
    MM_COND_MAX  = mm_cond_max
    lat_r = [float(x) for x in lat_range.split(',')]
    lon_r = [float(x) for x in lon_range.split(',')]
    years_list = [y.strip() for y in years.split(',')]

    print(f"Device     : {DEVICE}")
    print(f"COND_LIST  : {COND_LIST}  (nheads={NHEADS_FIXED} fixed)")
    print(f"MM_COND_MAX: {MM_COND_MAX}")
    print(f"Region     : lat {lat_r}, lon {lon_r}")

    output_path = Path(config.amarel_estimates_day_path)
    output_path.mkdir(parents=True, exist_ok=True)
    date_tag    = datetime.now().strftime("%m%d%y")
    csv_raw     = f"sim_cond_effect_{date_tag}.csv"
    csv_summary = f"sim_cond_summary_{date_tag}.csv"

    # ── True parameters ────────────────────────────────────────────────────────
    # Scenario A — original GEMS-fitted parameters
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
    print("\n[Setup 1/5] Loading GEMS obs patterns...")
    data_loader = load_data_dynamic_processed(config.amarel_data_load_path)
    all_day_mappings = []
    year_dfmaps, year_means = {}, {}
    for yr in years_list:
        df_map_yr, _, _, monthly_mean_yr = data_loader.load_maxmin_ordered_data_bymonthyear(
            lat_lon_resolution=[1, 1], mm_cond_number=MM_COND_DATA,
            years_=[yr], months_=[month], lat_range=lat_r, lon_range=lon_r, is_whittle=False)
        year_dfmaps[yr] = df_map_yr
        year_means[yr]  = monthly_mean_yr
        print(f"  {yr}: {len(df_map_yr)} time slots")

    # ── Build grids ────────────────────────────────────────────────────────────
    print("[Setup 2/5] Building grids...")
    lats_grid = torch.arange(max(lat_r), min(lat_r) - 0.0001, -DELTA_LAT_BASE, device=DEVICE, dtype=DTYPE)
    lons_grid = torch.arange(lon_r[0], lon_r[1] + 0.0001, DELTA_LON_BASE, device=DEVICE, dtype=DTYPE)
    lats_grid = torch.round(lats_grid * 10000) / 10000
    lons_grid = torch.round(lons_grid * 10000) / 10000
    g_lat, g_lon = torch.meshgrid(lats_grid, lons_grid, indexing='ij')
    grid_coords  = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)
    N_grid = grid_coords.shape[0]
    print(f"  Grid: {len(lats_grid)}×{len(lons_grid)} = {N_grid} cells")

    print("[Setup 3/5] Building high-res grid and precomputing mappings...")
    lats_hr, lons_hr, dlat_hr, dlon_hr = build_high_res_grid(lat_r, lon_r, lat_factor, lon_factor)
    DUMMY_KEYS = [f't{i}' for i in range(8)]

    for yr in years_list:
        df_map_yr = year_dfmaps[yr]; monthly_mean_yr = year_means[yr]
        all_sorted = sorted(df_map_yr.keys()); n_days_yr = len(all_sorted) // 8
        for d_idx in range(n_days_yr):
            hour_indices = [d_idx * 8, (d_idx + 1) * 8]
            ref_day_map, _ = data_loader.load_working_data(
                df_map_yr, monthly_mean_yr, hour_indices, ord_mm=None, dtype=DTYPE, keep_ori=True)
            day_keys = sorted(ref_day_map.keys())[:8]
            if len(day_keys) < 8:
                continue
            s3, hr_idx, src = precompute_mapping_indices(
                ref_day_map, lats_hr, lons_hr, grid_coords, day_keys)
            all_day_mappings.append((yr, d_idx, s3, hr_idx, src))
    print(f"  Total day-patterns: {len(all_day_mappings)}")

    print("[Setup 4/5] Computing grid ordering (mm_cond_number={MM_COND_MAX})...")
    ord_grid, nns_grid = compute_grid_ordering(grid_coords, MM_COND_MAX)

    LBFGS_LR = 1.0; LBFGS_STEPS = 3; LBFGS_HIST = 100; LBFGS_EVAL = 100
    OUTLIER_THRESH = 50.0

    p_labels  = ['sigmasq', 'range_lat', 'range_lon', 'range_t', 'advec_lat', 'advec_lon', 'nugget']
    p_cols    = ['sigmasq_est', 'range_lat_est', 'range_lon_est', 'range_t_est',
                 'advec_lat_est', 'advec_lon_est', 'nugget_est']
    true_vals = [true_dict['sigmasq'], true_dict['range_lat'], true_dict['range_lon'],
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
        print(f"  Init sigmasq={init_orig['sigmasq']:.3f}  "
              f"range_lon={init_orig['range_lon']:.3f}  nugget={init_orig['nugget']:.3f}")

        try:
            field = generate_field_values(lats_hr, lons_hr, 8, true_params, dlat_hr, dlon_hr)
            irr_map = assemble_irr_dataset(
                field, step3_per_t, hr_idx_per_t, src_locs_per_t,
                DUMMY_KEYS, grid_coords, true_params)
            del field
            irr_map_ord = {k: v[ord_grid] for k, v in irr_map.items()}

            iter_results = {}   # n_cond → (est, rmsre, elapsed)

            for nc in COND_LIST:
                print(f"  -- n_cond={nc}  (nheads={NHEADS_FIXED}) --")
                p = [torch.tensor([val], device=DEVICE, dtype=DTYPE, requires_grad=True)
                     for val in initial_vals]
                model = kernels_vecchia.fit_vecchia_lbfgs(
                    smooth=v, input_map=irr_map_ord,
                    nns_map=nns_grid, mm_cond_number=MM_COND_MAX, nheads=NHEADS_FIXED,
                    limit_A=nc, limit_B=nc, limit_C=nc, daily_stride=daily_stride
                )
                model.precompute_conditioning_sets()
                opt = model.set_optimizer(p, lr=LBFGS_LR, max_iter=LBFGS_EVAL,
                                          history_size=LBFGS_HIST)
                t0 = time.time()
                out, _ = model.fit_vecc_lbfgs(p, opt, max_steps=LBFGS_STEPS, grad_tol=1e-7)
                elapsed = time.time() - t0
                rmsre, est = calculate_rmsre(out, true_dict)
                iter_results[nc] = (est, rmsre, elapsed)
                print(f"     RMSRE={rmsre:.4f}  ({elapsed:.1f}s)")

        except Exception as e:
            skipped += 1
            print(f"  [SKIP] {type(e).__name__}: {e}  (total skipped: {skipped})")
            continue

        # ── Outlier check ─────────────────────────────────────────────────────
        def _is_outlier(est_d):
            return any([
                abs(est_d['sigmasq'])    > abs(true_dict['sigmasq'])    * OUTLIER_THRESH,
                abs(est_d['range_lat'])  > abs(true_dict['range_lat'])  * OUTLIER_THRESH,
                abs(est_d['range_lon'])  > abs(true_dict['range_lon'])  * OUTLIER_THRESH,
                abs(est_d['range_time']) > abs(true_dict['range_time']) * OUTLIER_THRESH,
                abs(est_d['advec_lat'])  > abs(true_dict['advec_lat'])  * OUTLIER_THRESH,
                abs(est_d['advec_lon'])  > abs(true_dict['advec_lon'])  * OUTLIER_THRESH,
                abs(est_d['nugget'])     > abs(true_dict['nugget'])     * OUTLIER_THRESH,
            ])

        outlier_conds = [nc for nc, (est, _, _) in iter_results.items() if _is_outlier(est)]
        if outlier_conds:
            skipped += 1
            print(f"  [SKIP] Extreme estimate in n_cond={outlier_conds}. "
                  f"Skipping all. (total skipped: {skipped})")
            continue

        # ── Record ────────────────────────────────────────────────────────────
        for nc, (est_d, rmsre_val, elapsed) in iter_results.items():
            records.append({
                'iter':          it + 1,
                'obs_year':      yr_it,
                'obs_day':       d_it,
                'n_cond':        nc,
                'nheads':        NHEADS_FIXED,
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
                'init_range_lon':round(init_orig['range_lon'], 4),
            })

        pd.DataFrame(records).to_csv(output_path / csv_raw, index=False)

        # ── Running summary table ─────────────────────────────────────────────
        n_done = len([r for r in records if r['n_cond'] == COND_LIST[0]])
        print(f"\n  ── Running summary ({n_done} completed / {it+1} attempted) ──")
        cw = 9
        hdr = f"  {'param':<11} {'true':>{cw}}" + "".join(f"  {'nc='+str(c):>{cw}}" for c in COND_LIST)
        print(hdr)
        print(f"  {'-'*65}")
        for lbl, col, tv in zip(p_labels, p_cols, true_vals):
            row = f"  {lbl:<11} {tv:>{cw}.4f}"
            for nc in COND_LIST:
                vals = [r[col] for r in records if r['n_cond'] == nc]
                row += f"  {np.mean(vals):>{cw}.4f}"
            print(row)
        print(f"  {'-'*65}")
        per_param_by_cond = {}
        per_param_med_by_cond = {}
        for nc in COND_LIST:
            sub_recs = [r for r in records if r['n_cond'] == nc]
            per_param_by_cond[nc] = [
                float(np.sqrt(np.mean([((r[col] - tv) / abs(tv)) ** 2 for r in sub_recs])))
                for col, tv in zip(p_cols, true_vals)
            ]
            per_param_med_by_cond[nc] = [
                float(np.sqrt(np.median([((r[col] - tv) / abs(tv)) ** 2 for r in sub_recs])))
                for col, tv in zip(p_cols, true_vals)
            ]
        for lbl, idx in zip(p_labels, range(len(p_labels))):
            rmsre_p = f"  {'RMSRE_'+lbl:<11} {'':>{cw}}"
            for nc in COND_LIST:
                rmsre_p += f"  {per_param_by_cond[nc][idx]:>{cw}.4f}"
            print(rmsre_p)
        print(f"  {'-'*65}")
        rmsre_row = f"  {'RMSRE':<11} {'':>{cw}}"
        for nc in COND_LIST:
            rmsre_row += f"  {np.mean(per_param_by_cond[nc]):>{cw}.4f}"
        print(rmsre_row)
        med_rmsre_row = f"  {'MedRMSRE':<11} {'':>{cw}}"
        for nc in COND_LIST:
            med_rmsre_row += f"  {np.mean(per_param_med_by_cond[nc]):>{cw}.4f}"
        print(med_rmsre_row)
        time_row = f"  {'time(s)':<11} {'':>{cw}}"
        for nc in COND_LIST:
            time_row += f"  {np.mean([r['time_s'] for r in records if r['n_cond'] == nc]):>{cw}.1f}"
        print(time_row)

    # ── Final summary ─────────────────────────────────────────────────────────
    df_final = pd.DataFrame(records)
    n_valid  = len(df_final[df_final['n_cond'] == COND_LIST[0]])

    def param_rmsre(sub, col, tv):
        return float(np.sqrt(np.mean(((sub[col].values - tv) / abs(tv)) ** 2)))

    def param_med_rmsre(sub, col, tv):
        return float(np.sqrt(np.median(((sub[col].values - tv) / abs(tv)) ** 2)))

    print(f"\n{'='*75}")
    print(f"  FINAL SUMMARY — Per-parameter RMSRE  ({n_valid} valid iterations)")
    print(f"  nheads={NHEADS_FIXED} fixed; varying n_cond (limit_A=limit_B=limit_C=n_cond)")
    print(f"{'='*75}")
    cw2 = 10
    print(f"  {'Parameter':<14} {'True':>8}" + "".join(f"  {'nc='+str(c):>{cw2}}" for c in COND_LIST))
    print(f"  {'-'*73}")
    for lbl, col, tv in zip(p_labels, p_cols, true_vals):
        row = f"  {lbl:<14} {tv:>8.4f}"
        for nc in COND_LIST:
            sub = df_final[df_final['n_cond'] == nc]
            row += f"  {param_rmsre(sub, col, tv):>{cw2}.4f}"
        print(row)
    print(f"  {'-'*73}")
    overall_row = f"  {'Overall RMSRE':<14} {'':>8}"
    for nc in COND_LIST:
        sub = df_final[df_final['n_cond'] == nc]
        per_param_rmsres = [param_rmsre(sub, col, tv) for col, tv in zip(p_cols, true_vals)]
        overall_row += f"  {np.mean(per_param_rmsres):>{cw2}.4f}"
    print(overall_row)
    overall_med_row = f"  {'Overall Med':<14} {'':>8}"
    for nc in COND_LIST:
        sub = df_final[df_final['n_cond'] == nc]
        per_param_med = [param_med_rmsre(sub, col, tv) for col, tv in zip(p_cols, true_vals)]
        overall_med_row += f"  {np.mean(per_param_med):>{cw2}.4f}"
    print(overall_med_row)

    print(f"\n  Mean estimate (SD) — {n_valid} iterations")
    print(f"  {'Parameter':<14} {'True':>8}" + "".join(f"  {'nc='+str(c):>{cw2}}" for c in COND_LIST))
    print(f"  {'-'*73}")
    for lbl, col, tv in zip(p_labels, p_cols, true_vals):
        row = f"  {lbl:<14} {tv:>8.4f}"
        for nc in COND_LIST:
            sub = df_final[df_final['n_cond'] == nc]
            row += f"  {sub[col].mean():>5.3f}({sub[col].std():.3f})"
        print(row)

    print(f"\n  Mean wall-clock time (s) per n_cond config:")
    time_row = f"  {'time_s':<14} {'':>8}"
    for nc in COND_LIST:
        sub = df_final[df_final['n_cond'] == nc]
        time_row += f"  {sub['time_s'].mean():>{cw2}.2f}"
    print(time_row)

    # ── Save CSVs ──────────────────────────────────────────────────────────────
    summary_rows = []
    for lbl, col, tv in zip(p_labels, p_cols, true_vals):
        row = {'parameter': lbl, 'true': tv}
        for nc in COND_LIST:
            sub = df_final[df_final['n_cond'] == nc]
            row[f'nc{nc}_rmsre']     = round(param_rmsre(sub, col, tv),     6)
            row[f'nc{nc}_med_rmsre'] = round(param_med_rmsre(sub, col, tv), 6)
            row[f'nc{nc}_mean']      = round(sub[col].mean(), 6)
            row[f'nc{nc}_sd']        = round(sub[col].std(),  6)
        summary_rows.append(row)
    overall = {'parameter': 'Overall_RMSRE', 'true': float('nan')}
    for nc in COND_LIST:
        sub = df_final[df_final['n_cond'] == nc]
        per_param_rmsres = [param_rmsre(sub, col, tv)     for col, tv in zip(p_cols, true_vals)]
        per_param_med    = [param_med_rmsre(sub, col, tv) for col, tv in zip(p_cols, true_vals)]
        overall[f'nc{nc}_rmsre']     = round(np.mean(per_param_rmsres), 6)
        overall[f'nc{nc}_med_rmsre'] = round(np.mean(per_param_med),    6)
        overall[f'nc{nc}_mean']      = float('nan')
        overall[f'nc{nc}_sd']        = float('nan')
    summary_rows.append(overall)

    timing = {'parameter': 'mean_time_s', 'true': float('nan')}
    for nc in COND_LIST:
        sub = df_final[df_final['n_cond'] == nc]
        timing[f'nc{nc}_rmsre'] = float('nan')
        timing[f'nc{nc}_mean']  = round(sub['time_s'].mean(), 2)
        timing[f'nc{nc}_sd']    = round(sub['time_s'].std(),  2)
    summary_rows.append(timing)

    pd.DataFrame(summary_rows).to_csv(output_path / csv_summary, index=False)
    print(f"\n  Saved: {csv_raw}")
    print(f"  Saved: {csv_summary}")


if __name__ == "__main__":
    app()
