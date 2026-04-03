"""
sim_cauchy_beta_comparison_032626.py

Misspecification robustness study comparing three Generalized Cauchy kernels:
  1. Cauchy β=0.5  — heavy polynomial tail  C(d) = σ²/(1+d/α)^0.5
  2. Cauchy β=1.0  — standard Cauchy        C(d) = σ²/(1+d/α)^1.0
  3. Cauchy β=2.0  — faster polynomial decay C(d) = σ²/(1+d/α)^2.0

Data-generating process: Matérn ν=0.5 (exponential decay), generated via FFT
circulant embedding — identical to sim_three_model_comparison_031926.py.

All three models are misspecified relative to the true (Matérn) DGP.
Purpose: assess which β is most robust to misspecification, i.e., which β
recovers the Matérn parameters most faithfully when the true kernel is
exponential.

Data pipeline:
  ┌─ FFT high-res field (Matérn ν=0.5 covariance)
  └─► [Simulated actual observations]  (irregular source locations)
        valid obs → nearest high-res point → + nugget noise
        stored as [N_grid, 11] tensor  (NaN rows for unobserved cells)

All three Cauchy models fit the same irregular-location dataset.

Reference
---------
Gneiting & Schlather (2004). Stochastic Models That Separate Fractal Dimension
and the Hurst Effect. SIAM Review, 46(2), 269-282.

Run locally:
  conda activate faiss_env
  cd /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/local_computer
  python sim_cauchy_beta_comparison_local_032626.py --num-iters 1
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
from typing import List
from sklearn.neighbors import BallTree

sys.path.append("/Users/joonwonlee/Documents/GEMS_TCO-1/src")
sys.path.append("/Users/joonwonlee/Documents/GEMS_TCO-1")

from GEMS_TCO import kernels_vecchia
from GEMS_TCO import kernels_vecchia_cauchy
from GEMS_TCO import orderings as _orderings
from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data_dynamic_processed

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64
DELTA_LAT_BASE = 0.044
DELTA_LON_BASE = 0.063

BETA_VALUES = [0.5, 1.0, 2.0]
MODEL_NAMES = ['Cauchy_b05', 'Cauchy_b10', 'Cauchy_b20']

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})


# ── Covariance kernel (Matérn ν=0.5) — used only for FFT data generation ─────

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
    FFT computed on CPU in float32; returns field [Nx, Ny, Nt] on DEVICE."""
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


# ── Step3 obs→cell assignment ──────────────────────────────────────────────────

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
    hr_tree = BallTree(np.radians(hr_coords_np), metric='haversine')

    grid_coords_np = grid_coords.cpu().numpy()
    grid_tree = BallTree(np.radians(grid_coords_np), metric='haversine')

    step3_assignment_per_t, hr_idx_per_t, src_locs_per_t = [], [], []
    for key in sorted_keys:
        ref_t     = ref_day_map[key].to(DEVICE)
        src_locs  = ref_t[:, :2]
        src_np    = src_locs.cpu().numpy()
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


# ── Grid ordering ──────────────────────────────────────────────────────────────

def compute_grid_ordering(grid_coords, mm_cond_number):
    coords_np = grid_coords.cpu().numpy()
    ord_mm    = _orderings.maxmin_cpp(coords_np)
    nns       = _orderings.find_nns_l2(locs=coords_np[ord_mm], max_nn=mm_cond_number)
    return ord_mm, nns


# ── Dataset assembly (irregular only) ─────────────────────────────────────────

def assemble_irr_dataset(field, step3_assignment_per_t, hr_idx_per_t, src_locs_per_t,
                         sorted_keys, grid_coords, true_params, t_offset=21.0):
    """Produce irregular-location dataset from one FFT field realization."""
    nugget_std = torch.sqrt(torch.exp(true_params[6]))
    N_grid     = grid_coords.shape[0]
    field_flat = field.reshape(-1, 8)

    irr_map, irr_list = {}, []
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
        irr_list.append(irr_rows.detach())

    return irr_map, torch.cat(irr_list, dim=0)


# ── Metrics ───────────────────────────────────────────────────────────────────

def backmap_params(out_params):
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
    est_arr  = np.array([est['sigmasq'],        est['range_lat'], est['range_lon'],
                         est['range_time'],      est['advec_lat'], est['advec_lon'],
                         est['nugget']])
    true_arr = np.array([true_dict['sigmasq'],   true_dict['range_lat'], true_dict['range_lon'],
                         true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
                         true_dict['nugget']])
    return float(np.sqrt(np.mean(((est_arr - true_arr) / np.abs(true_arr)) ** 2))), est


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command()
def cli(
    v:              float = typer.Option(0.5,    help="Matérn smoothness (data generation only)"),
    mm_cond_number: int   = typer.Option(100,    help="Vecchia neighbors"),
    nheads:         int   = typer.Option(0,      help="Vecchia head points per time step"),
    limit_a:        int   = typer.Option(20,      help="Set A neighbors"),
    limit_b:        int   = typer.Option(20,      help="Set B neighbors"),
    limit_c:        int   = typer.Option(20,      help="Set C neighbors"),
    daily_stride:   int   = typer.Option(4,      help="Daily stride for Set C"),
    num_iters:      int   = typer.Option(3,      help="Simulation iterations"),
    years:          str   = typer.Option("2024", help="Years to sample obs patterns from"),
    month:          int   = typer.Option(7,      help="Reference month"),
    lat_range:      str   = typer.Option("-3,2",    help="lat_min,lat_max"),
    lon_range:      str   = typer.Option("121,131", help="lon_min,lon_max"),
    lat_factor:     int   = typer.Option(10,     help="High-res lat multiplier"),
    lon_factor:     int   = typer.Option(4,      help="High-res lon multiplier"),
    init_noise:     float = typer.Option(0.7,    help="Uniform noise half-width in log space"),
    seed:           int   = typer.Option(42,     help="Random seed"),
) -> None:

    import random as _random
    rng = np.random.default_rng(seed)
    _random.seed(seed)

    lat_r      = [float(x) for x in lat_range.split(',')]
    lon_r      = [float(x) for x in lon_range.split(',')]
    years_list = [y.strip() for y in years.split(',')]

    print(f"Device : {DEVICE}")
    print(f"Region : lat {lat_r}, lon {lon_r}")
    print(f"Years  : {years_list}  month={month}")
    print(f"High-res : lat×{lat_factor}, lon×{lon_factor}")
    print(f"Init noise: ±{init_noise} in log space")
    print(f"Models : Cauchy β ∈ {BETA_VALUES}  (data from Matérn ν={v})")

    output_path = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/cauchy_beta")
    output_path.mkdir(parents=True, exist_ok=True)
    date_tag    = datetime.now().strftime("%m%d%y")
    csv_raw     = f"sim_cauchy_beta_comparison_{date_tag}_local.csv"
    csv_summary = f"sim_cauchy_beta_summary_{date_tag}_local.csv"

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
    data_loader    = load_data_dynamic_processed(config.amarel_data_load_path)
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

    # ── Build regular target grid ──────────────────────────────────────────────
    print("[Setup 2/5] Building regular target grid...")
    lats_grid = torch.arange(max(lat_r), min(lat_r) - 0.0001, -DELTA_LAT_BASE,
                              device=DEVICE, dtype=DTYPE)
    lons_grid = torch.arange(lon_r[0],   lon_r[1]  + 0.0001,  DELTA_LON_BASE,
                              device=DEVICE, dtype=DTYPE)
    lats_grid = torch.round(lats_grid * 10000) / 10000
    lons_grid = torch.round(lons_grid * 10000) / 10000
    g_lat, g_lon = torch.meshgrid(lats_grid, lons_grid, indexing='ij')
    grid_coords  = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)
    N_grid = grid_coords.shape[0]
    print(f"  Grid: {len(lats_grid)} lat × {len(lons_grid)} lon = {N_grid} cells")

    # ── Build high-res grid & precompute mappings ─────────────────────────────
    print("[Setup 3/5] Building high-res grid and precomputing mappings...")
    lats_hr, lons_hr, dlat_hr, dlon_hr = build_high_res_grid(lat_r, lon_r, lat_factor, lon_factor)
    print(f"  High-res: {len(lats_hr)} lat × {len(lons_hr)} lon "
          f"= {len(lats_hr)*len(lons_hr):,} cells")

    DUMMY_KEYS = [f't{i}' for i in range(8)]

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
                ord_mm=None, dtype=DTYPE, keep_ori=True
            )
            day_keys = sorted(ref_day_map.keys())[:8]
            if len(day_keys) < 8:
                continue
            s3, hr_idx, src = precompute_mapping_indices(
                ref_day_map, lats_hr, lons_hr, grid_coords, day_keys)
            all_day_mappings.append((yr, d_idx, s3, hr_idx, src))

    print(f"  Total available day-patterns: {len(all_day_mappings)}")

    # ── Grid-based ordering ────────────────────────────────────────────────────
    print("[Setup 4/5] Computing grid-based maxmin ordering...")
    ord_grid, nns_grid = compute_grid_ordering(grid_coords, mm_cond_number)
    print(f"  N_grid={N_grid}, mm_cond_number={mm_cond_number}")

    # ── Verify dataset structure ──────────────────────────────────────────────
    print("[Setup 5/5] Verifying dataset structure with sample field...")
    _yr0, _d0, _s3_0, _hr0, _src0 = all_day_mappings[0]
    field0 = generate_field_values(lats_hr, lons_hr, 8, true_params, dlat_hr, dlon_hr)
    irr0, _ = assemble_irr_dataset(field0, _s3_0, _hr0, _src0, DUMMY_KEYS, grid_coords, true_params)
    del field0
    first_irr = list(irr0.values())[0]
    n_irr_valid = (~torch.isnan(first_irr[:, 2])).sum().item()
    print(f"  irr_map first step: {n_irr_valid}/{N_grid} valid rows")

    # ── Optimization settings ─────────────────────────────────────────────────
    LBFGS_LR    = 1.0
    LBFGS_STEPS = 5
    LBFGS_HIST  = 10
    LBFGS_EVAL  = 20

    records = []

    # ── SIMULATION LOOP ───────────────────────────────────────────────────────
    skipped = 0
    for it in range(num_iters):
        print(f"\n{'='*60}")
        print(f"  Iteration {it+1}/{num_iters}  (skipped so far: {skipped})")
        print(f"{'='*60}")

        yr_it, d_it, step3_assignment_per_t, hr_idx_per_t, src_locs_per_t = \
            all_day_mappings[rng.integers(len(all_day_mappings))]
        print(f"  Obs pattern: {yr_it} day {d_it}")

        initial_vals = make_random_init(rng)
        init_orig    = backmap_params(initial_vals)
        print(f"  Init sigmasq={init_orig['sigmasq']:.3f}  "
              f"range_lon={init_orig['range_lon']:.3f}  "
              f"nugget={init_orig['nugget']:.3f}")

        try:
            field = generate_field_values(lats_hr, lons_hr, 8, true_params, dlat_hr, dlon_hr)
            irr_map, _ = assemble_irr_dataset(
                field, step3_assignment_per_t, hr_idx_per_t, src_locs_per_t,
                DUMMY_KEYS, grid_coords, true_params)
            del field

            irr_map_ord = {k: v[ord_grid] for k, v in irr_map.items()}

            iter_results = {}

            for beta, mname in zip(BETA_VALUES, MODEL_NAMES):
                print(f"--- {mname}  (β={beta}) ---")
                p_vals = [torch.tensor([val], device=DEVICE, dtype=DTYPE, requires_grad=True)
                          for val in initial_vals]
                model = kernels_vecchia_cauchy.fit_cauchy_vecchia_lbfgs(
                    smooth=v, gc_beta=beta,
                    input_map=irr_map_ord, nns_map=nns_grid,
                    mm_cond_number=mm_cond_number, nheads=nheads,
                    limit_A=limit_a, limit_B=limit_b, limit_C=limit_c,
                    daily_stride=daily_stride
                )
                model.precompute_conditioning_sets()
                opt = model.set_optimizer(p_vals, lr=LBFGS_LR, max_iter=LBFGS_EVAL,
                                          max_eval=LBFGS_EVAL, history_size=LBFGS_HIST)
                t0 = time.time()
                out, _ = model.fit_vecc_lbfgs(p_vals, opt,
                                              max_steps=LBFGS_STEPS, grad_tol=1e-5)
                elapsed = time.time() - t0
                rmsre, est = calculate_rmsre(out, true_dict)
                print(f"  RMSRE = {rmsre:.4f}  ({elapsed:.1f}s)")
                iter_results[mname] = (rmsre, est, elapsed)

        except Exception as e:
            skipped += 1
            print(f"  [SKIP] Iteration {it+1} failed: {type(e).__name__}: {e}")
            print(f"  Skipping to next iteration. (total skipped: {skipped})")
            continue

        # ── Record ────────────────────────────────────────────────────────────
        for mname, beta in zip(MODEL_NAMES, BETA_VALUES):
            rmsre_val, est_d, elapsed = iter_results[mname]
            records.append({
                'iter':          it + 1,
                'obs_year':      yr_it,
                'obs_day':       d_it,
                'model':         mname,
                'beta':          beta,
                'rmsre':         round(rmsre_val,         6),
                'time_s':        round(elapsed,           2),
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

        # ── Running summary ────────────────────────────────────────────────────
        p_cols_   = ['sigmasq_est', 'range_lat_est', 'range_lon_est',
                     'range_t_est', 'advec_lat_est', 'advec_lon_est', 'nugget_est']
        p_labels_ = ['sigmasq', 'range_lat', 'range_lon', 'range_t',
                     'advec_lat', 'advec_lon', 'nugget']
        true_vals_ = [true_dict['sigmasq'],    true_dict['range_lat'], true_dict['range_lon'],
                      true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
                      true_dict['nugget']]

        n_done = len([r for r in records if r['model'] == MODEL_NAMES[0]])
        print(f"\n  ── Running summary ({n_done} completed / {it+1} attempted) ──")
        cw  = 11
        hdr = f"  {'param':<11} {'true':>{cw}}" + "".join(f"  {m:>{cw}}" for m in MODEL_NAMES)
        print(hdr)
        print(f"  {'-'*60}")
        for lbl, col, tv in zip(p_labels_, p_cols_, true_vals_):
            row = f"  {lbl:<11} {tv:>{cw}.4f}"
            for m in MODEL_NAMES:
                vals = [r[col] for r in records if r['model'] == m]
                row += f"  {np.mean(vals):>{cw}.4f}"
            print(row)
        print(f"  {'-'*60}")

        per_param_rmsre  = {}
        per_param_mdare  = {}
        per_param_p90p10 = {}
        for m in MODEL_NAMES:
            sub_recs = [r for r in records if r['model'] == m]
            ares = [[abs((r[col] - tv) / abs(tv)) for r in sub_recs]
                    for col, tv in zip(p_cols_, true_vals_)]
            per_param_rmsre[m]  = [float(np.sqrt(np.mean(np.array(a)**2))) for a in ares]
            per_param_mdare[m]  = [float(np.median(a))                     for a in ares]
            per_param_p90p10[m] = [
                float(np.percentile([r[col] for r in sub_recs], 90)
                      - np.percentile([r[col] for r in sub_recs], 10))
                for col, _ in zip(p_cols_, true_vals_)
            ]

        for metric_lbl, model_dict in [
            ('RMSRE',   per_param_rmsre),
            ('MdARE',   per_param_mdare),
            ('P90-P10', per_param_p90p10),
        ]:
            print(f"\n  [{metric_lbl} per param]")
            for lbl, idx in zip(p_labels_, range(len(p_labels_))):
                row = f"  {lbl:<11} {'':>{cw}}"
                for m in MODEL_NAMES:
                    row += f"  {model_dict[m][idx]:>{cw}.4f}"
                print(row)
            overall_row = f"  {'Overall':<11} {'':>{cw}}"
            for m in MODEL_NAMES:
                overall_row += f"  {np.mean(model_dict[m]):>{cw}.4f}"
            print(overall_row)

    # ── Final summary ─────────────────────────────────────────────────────────
    df_final = pd.DataFrame(records)
    param_cols   = ['sigmasq_est', 'range_lat_est', 'range_lon_est',
                    'range_t_est', 'advec_lat_est', 'advec_lon_est', 'nugget_est']
    param_labels = ['sigmasq', 'range_lat', 'range_lon', 'range_time',
                    'advec_lat', 'advec_lon', 'nugget']
    true_vals    = [true_dict['sigmasq'],    true_dict['range_lat'], true_dict['range_lon'],
                    true_dict['range_time'], true_dict['advec_lat'], true_dict['advec_lon'],
                    true_dict['nugget']]

    def param_rmsre(sub, col, tv):
        return float(np.sqrt(np.mean(((sub[col].values - tv) / abs(tv)) ** 2)))

    def param_mdare(sub, col, tv):
        return float(np.median(np.abs((sub[col].values - tv) / abs(tv))))

    def param_p90p10(sub, col, tv):
        return float(np.percentile(sub[col].values, 90) - np.percentile(sub[col].values, 10))

    print(f"\n{'='*75}")
    print(f"  FINAL SUMMARY — Cauchy β comparison  ({num_iters} iterations)")
    print(f"  True DGP: Matérn ν={v}  (all Cauchy models are misspecified)")
    print(f"{'='*75}")
    col_w = 14

    for metric_fn, metric_lbl in [(param_rmsre, 'RMSRE'), (param_mdare, 'MdARE'),
                                   (param_p90p10, 'P90-P10')]:
        print(f"\n  [{metric_lbl} per param]")
        header = f"  {'Parameter':<14} {'True':>10}" + "".join(f"  {m:>{col_w}}" for m in MODEL_NAMES)
        print(header)
        print(f"  {'-'*76}")
        per_model_vals = {m: [] for m in MODEL_NAMES}
        for lbl, col, tv in zip(param_labels, param_cols, true_vals):
            row_str = f"  {lbl:<14} {tv:>10.4f}"
            for m in MODEL_NAMES:
                sub = df_final[df_final['model'] == m]
                val = metric_fn(sub, col, tv)
                per_model_vals[m].append(val)
                row_str += f"  {val:>{col_w}.4f}"
            print(row_str)
        print(f"  {'-'*76}")
        overall_str = f"  {'Overall':<14} {'':>10}"
        for m in MODEL_NAMES:
            overall_str += f"  {np.mean(per_model_vals[m]):>{col_w}.4f}"
        print(overall_str)

    # ── Mean ± SD table ────────────────────────────────────────────────────────
    print(f"\n  [Mean estimate (SD) across {num_iters} iterations]")
    print(f"  {'Parameter':<14} {'True':>10}" + "".join(f"  {'mean(SD)':>{col_w}}" for _ in MODEL_NAMES))
    print(f"  {'-'*76}")
    for lbl, col, tv in zip(param_labels, param_cols, true_vals):
        row_str = f"  {lbl:<14} {tv:>10.4f}"
        for m in MODEL_NAMES:
            sub = df_final[df_final['model'] == m]
            row_str += f"  {sub[col].mean():>6.3f}({sub[col].std():.3f})"
        print(row_str)

    # ── Save summary CSV ───────────────────────────────────────────────────────
    summary_rows = []
    for lbl, col, tv in zip(param_labels, param_cols, true_vals):
        row = {'parameter': lbl, 'true': tv}
        for m in MODEL_NAMES:
            sub = df_final[df_final['model'] == m]
            row[f'{m}_rmsre']   = round(param_rmsre(sub, col, tv),   6)
            row[f'{m}_mare']    = round(param_mdare(sub, col, tv),    6)
            row[f'{m}_p90p10']  = round(param_p90p10(sub, col, tv),  6)
            row[f'{m}_mean']    = round(sub[col].mean(), 6)
            row[f'{m}_sd']      = round(sub[col].std(),  6)
        summary_rows.append(row)

    overall_row = {'parameter': 'Overall', 'true': float('nan')}
    for m in MODEL_NAMES:
        sub = df_final[df_final['model'] == m]
        overall_row[f'{m}_rmsre']  = round(np.mean([param_rmsre(sub, c, tv)  for c, tv in zip(param_cols, true_vals)]), 6)
        overall_row[f'{m}_mare']   = round(np.mean([param_mdare(sub, c, tv)  for c, tv in zip(param_cols, true_vals)]), 6)
        overall_row[f'{m}_p90p10'] = round(np.mean([param_p90p10(sub, c, tv) for c, tv in zip(param_cols, true_vals)]), 6)
        overall_row[f'{m}_mean']   = float('nan')
        overall_row[f'{m}_sd']     = float('nan')
    summary_rows.append(overall_row)

    pd.DataFrame(summary_rows).to_csv(output_path / csv_summary, index=False)
    print(f"\n  Saved: {csv_raw}")
    print(f"  Saved: {csv_summary}")


if __name__ == "__main__":
    app()
