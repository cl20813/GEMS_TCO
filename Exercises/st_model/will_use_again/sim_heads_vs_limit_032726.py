"""
sim_heads_vs_limit_032726.py

Vecchia approximation: heads-as-low-frequency-diagnostic study.

Research question
-----------------
Does nheads act as a low-frequency diagnostic?
  - Short range (0.20/0.25°): local Vecchia already captures structure → heads negligible
  - Long  range (1.50/2.00°): global skeleton matters → heads should win iso-compute

Design
------
Range cases : 'short' (range_lat=0.20, range_lon=0.25)
              'long'  (range_lat=1.50, range_lon=2.00)
Iso-compute :
    (h=0,   L=24, 'limit_all')  ← limit에 올인
    (h=400, L=16, 'balanced')   ← 절충
    (h=800, L=12, 'heads_all')  ← heads에 올인
→ 2 × 3 = 6 fits per iteration (one field generated per range case)

gc_beta = 1.0 fixed (Cauchy), correctly specified for both DGP and model.
MM_COND = 30 (fixed NNS map; must be >= max limit = 24).

Key prediction
--------------
  short range : RMSRE(limit_all) ≈ RMSRE(balanced) ≈ RMSRE(heads_all)
  long  range : RMSRE(heads_all) < RMSRE(balanced)  < RMSRE(limit_all)

Output
------
    estimates/day/sim_heads_diagnostic_{date}_j{job_id}.csv
    estimates/day/sim_heads_diagnostic_summary_{date}_j{job_id}.csv

Parallel use (recommended: 10 jobs × 10 iters each = 100 total iters)
    sbatch --array=0-9
    Merge CSVs afterward: pd.concat([pd.read_csv(f) for f in glob("*_j*.csv")])
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

sys.path.append("/cache/home/jl2815/tco")

from GEMS_TCO import kernels_vecchia_cauchy
from GEMS_TCO import orderings as _orderings
from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data_dynamic_processed

app    = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64
DELTA_LAT_BASE = 0.044
DELTA_LON_BASE = 0.063
GC_BETA        = 1.0

MM_COND = 30  # fixed NNS map; must be >= max(limit) = 24

ISO_COMBOS = [
    (0,   24, 'limit_all'),
    (400, 16, 'balanced'),
    (800, 12, 'heads_all'),
]

RANGE_CASES = {
    'short': {'sigmasq': 10.0, 'range_lat': 0.20, 'range_lon': 0.25, 'range_time': 1.5,
              'advec_lat': 0.02, 'advec_lon': -0.17, 'nugget': 0.25},
    'long':  {'sigmasq': 10.0, 'range_lat': 1.50, 'range_lon': 2.00, 'range_time': 1.5,
              'advec_lat': 0.02, 'advec_lon': -0.17, 'nugget': 0.25},
}


# ── Cauchy covariance for FFT data generation ─────────────────────────────────

def get_covariance_on_grid_cauchy(lx, ly, lt, params, gc_beta=GC_BETA):
    params = torch.clamp(params, min=-15.0, max=15.0)
    phi1, phi2, phi3, phi4 = (torch.exp(params[i]) for i in range(4))
    u_lat = lx - params[4] * lt
    u_lon = ly - params[5] * lt
    dist  = torch.sqrt(u_lat.pow(2) * phi3 + u_lon.pow(2) + lt.pow(2) * phi4 + 1e-8)
    return (phi1 / phi2) * torch.pow(1.0 + dist * phi2, -gc_beta)


def build_high_res_grid(lat_range, lon_range, lat_factor=100, lon_factor=10):
    dlat = DELTA_LAT_BASE / lat_factor
    dlon = DELTA_LON_BASE / lon_factor
    lats = torch.arange(min(lat_range) - 0.1, max(lat_range) + 0.1, dlat,
                        device=DEVICE, dtype=DTYPE)
    lons = torch.arange(lon_range[0] - 0.1, lon_range[1] + 0.1, dlon,
                        device=DEVICE, dtype=DTYPE)
    return lats, lons, dlat, dlon


def generate_field_values(lats_hr, lons_hr, t_steps, params, dlat, dlon):
    """Cauchy FFT circulant embedding realization [Nx, Ny, Nt]."""
    CPU, F32 = torch.device("cpu"), torch.float32
    Nx, Ny, Nt = len(lats_hr), len(lons_hr), t_steps
    Px, Py, Pt = 2 * Nx, 2 * Ny, 2 * Nt

    lx = torch.arange(Px, device=CPU, dtype=F32) * dlat; lx[Px // 2:] -= Px * dlat
    ly = torch.arange(Py, device=CPU, dtype=F32) * dlon; ly[Py // 2:] -= Py * dlon
    lt = torch.arange(Pt, device=CPU, dtype=F32);        lt[Pt // 2:] -= Pt

    Lx, Ly, Lt = torch.meshgrid(lx, ly, lt, indexing='ij')
    C     = get_covariance_on_grid_cauchy(Lx, Ly, Lt, params.cpu().float())
    S     = torch.fft.fftn(C);  S.real = torch.clamp(S.real, min=0)
    noise = torch.fft.fftn(torch.randn(Px, Py, Pt, device=CPU, dtype=F32))
    field = torch.fft.ifftn(torch.sqrt(S.real) * noise).real[:Nx, :Ny, :Nt]
    return field.to(dtype=DTYPE, device=DEVICE)


# ── Obs-location pipeline ─────────────────────────────────────────────────────

def apply_step3_1to1(src_np_valid, grid_coords_np, grid_tree):
    N_grid     = len(grid_coords_np)
    if len(src_np_valid) == 0:
        return np.full(N_grid, -1, dtype=np.int64)
    dist_to_cell, cell_for_obs = grid_tree.query(np.radians(src_np_valid), k=1)
    assignment = np.full(N_grid, -1, dtype=np.int64)
    best_dist  = np.full(N_grid, np.inf)
    for obs_i, (cell_j, d) in enumerate(zip(cell_for_obs.flatten(), dist_to_cell.flatten())):
        if d < best_dist[cell_j]:
            assignment[cell_j] = obs_i
            best_dist[cell_j]  = d
    return assignment


def precompute_mapping_indices(ref_day_map, lats_hr, lons_hr, grid_coords, sorted_keys):
    hr_lat_g, hr_lon_g = torch.meshgrid(lats_hr, lons_hr, indexing='ij')
    hr_coords_np = torch.stack([hr_lat_g.flatten(), hr_lon_g.flatten()], dim=1).cpu().numpy()
    hr_tree      = BallTree(np.radians(hr_coords_np), metric='haversine')
    grid_coords_np = grid_coords.cpu().numpy()
    grid_tree    = BallTree(np.radians(grid_coords_np), metric='haversine')

    step3_per_t, hr_idx_per_t, src_locs_per_t = [], [], []
    for key in sorted_keys:
        ref_t    = ref_day_map[key].to(DEVICE)
        src_locs = ref_t[:, :2]
        src_np   = src_locs.cpu().numpy()
        valid    = ~np.isnan(src_np).any(axis=1)
        assignment = apply_step3_1to1(src_np[valid], grid_coords_np, grid_tree)
        step3_per_t.append(assignment)
        if valid.sum() > 0:
            _, hr_idx = hr_tree.query(np.radians(src_np[valid]), k=1)
            hr_idx_per_t.append(torch.tensor(hr_idx.flatten(), device=DEVICE))
        else:
            hr_idx_per_t.append(torch.zeros(0, dtype=torch.long, device=DEVICE))
        src_locs_per_t.append(src_locs[valid])
    return step3_per_t, hr_idx_per_t, src_locs_per_t


def assemble_irr_dataset(field, step3_per_t, hr_idx_per_t, src_locs_per_t,
                         sorted_keys, grid_coords, true_params, t_offset=21.0):
    nugget_std = torch.sqrt(torch.exp(true_params[6]))
    N_grid     = grid_coords.shape[0]
    field_flat = field.reshape(-1, 8)
    irr_map, irr_list = {}, []
    for t_idx, key in enumerate(sorted_keys):
        t_val    = float(t_offset + t_idx)
        hr_idx   = hr_idx_per_t[t_idx]
        src_locs = src_locs_per_t[t_idx]
        N_valid  = hr_idx.shape[0]
        dummy    = torch.zeros(7, device=DEVICE, dtype=DTYPE)
        if t_idx > 0:
            dummy[t_idx - 1] = 1.0
        if N_valid > 0:
            gp_vals  = field_flat[hr_idx, t_idx]
            sim_vals = gp_vals + torch.randn(N_valid, device=DEVICE, dtype=DTYPE) * nugget_std
        else:
            sim_vals = torch.zeros(0, device=DEVICE, dtype=DTYPE)
        NaN       = float('nan')
        irr_rows  = torch.full((N_grid, 11), NaN, device=DEVICE, dtype=DTYPE)
        irr_rows[:, 3]  = t_val
        irr_rows[:, 4:] = dummy.unsqueeze(0).expand(N_grid, -1)
        if N_valid > 0:
            assign_t = torch.tensor(step3_per_t[t_idx], device=DEVICE, dtype=torch.long)
            filled   = assign_t >= 0
            win_obs  = assign_t[filled]
            irr_rows[filled, 0] = src_locs[win_obs, 0]
            irr_rows[filled, 1] = src_locs[win_obs, 1]
            irr_rows[filled, 2] = sim_vals[win_obs]
        irr_map[key] = irr_rows.detach()
        irr_list.append(irr_rows.detach())
    return irr_map, torch.cat(irr_list, dim=0)


# ── Parameter helpers ─────────────────────────────────────────────────────────

def true_dict_to_log(true_dict):
    phi2 = 1.0 / true_dict['range_lon']
    phi1 = true_dict['sigmasq'] * phi2
    phi3 = (true_dict['range_lon'] / true_dict['range_lat'])  ** 2
    phi4 = (true_dict['range_lon'] / true_dict['range_time']) ** 2
    return [np.log(phi1), np.log(phi2), np.log(phi3), np.log(phi4),
            true_dict['advec_lat'], true_dict['advec_lon'], np.log(true_dict['nugget'])]


def make_init(rng, true_log, init_noise):
    noisy = list(true_log)
    for i in [0, 1, 2, 3, 6]:
        noisy[i] = true_log[i] + rng.uniform(-init_noise, init_noise)
    for i in [4, 5]:
        noisy[i] = true_log[i] + rng.uniform(-2 * max(abs(true_log[i]), 0.05),
                                               2 * max(abs(true_log[i]), 0.05))
    return noisy


def backmap_params(p):
    if isinstance(p[0], torch.Tensor):
        p = [x.item() for x in p]
    p = [float(x) for x in p]
    phi2 = np.exp(p[1]); phi3 = np.exp(p[2]); phi4 = np.exp(p[3])
    rlon = 1.0 / phi2
    return {'sigmasq':    np.exp(p[0]) / phi2,
            'range_lat':  rlon / phi3 ** 0.5,
            'range_lon':  rlon,
            'range_time': rlon / phi4 ** 0.5,
            'advec_lat':  p[4],
            'advec_lon':  p[5],
            'nugget':     np.exp(p[6])}


def calc_rmsre(out_params, true_dict):
    est = backmap_params(out_params)
    keys = ['sigmasq', 'range_lat', 'range_lon', 'range_time',
            'advec_lat', 'advec_lon', 'nugget']
    est_arr  = np.array([est[k]        for k in keys])
    true_arr = np.array([true_dict[k]  for k in keys])
    return float(np.sqrt(np.mean(((est_arr - true_arr) / np.abs(true_arr)) ** 2))), est


# ── CLI ───────────────────────────────────────────────────────────────────────

@app.command()
def cli(
    num_iters:    int   = typer.Option(10,                   help="Iterations per job"),
    job_id:       int   = typer.Option(0,                    help="Job index for parallel runs"),
    daily_stride: int   = typer.Option(2,                    help="Daily stride for Set C"),
    years:        str   = typer.Option("2022,2023,2024,2025", help="Years for obs patterns"),
    month:        int   = typer.Option(7,                    help="Reference month"),
    lat_range:    str   = typer.Option("-3,2",               help="lat_min,lat_max"),
    lon_range:    str   = typer.Option("121,131",            help="lon_min,lon_max"),
    lat_factor:   int   = typer.Option(100,                  help="High-res lat multiplier"),
    lon_factor:   int   = typer.Option(10,                   help="High-res lon multiplier"),
    init_noise:   float = typer.Option(0.5,                  help="Log-space init noise half-width"),
    seed:         int   = typer.Option(42,                   help="Base seed (actual = seed+job_id)"),
) -> None:

    actual_seed = seed + job_id
    rng = np.random.default_rng(actual_seed)
    lat_r      = [float(x) for x in lat_range.split(',')]
    lon_r      = [float(x) for x in lon_range.split(',')]
    years_list = [y.strip() for y in years.split(',')]

    print(f"Device  : {DEVICE}")
    print(f"Job     : {job_id}  (seed={actual_seed})")
    print(f"DGP     : Cauchy β={GC_BETA}  (correctly specified)")
    print(f"Range cases : {list(RANGE_CASES.keys())}")
    print(f"Iso-combos  : {[(h, L, name) for h, L, name in ISO_COMBOS]}")
    print(f"MM_COND : {MM_COND}")
    print(f"Iters   : {num_iters}  lat×{lat_factor}  lon×{lon_factor}")

    output_path = Path(config.amarel_estimates_day_path)
    output_path.mkdir(parents=True, exist_ok=True)
    date_tag    = datetime.now().strftime("%m%d%y")
    csv_raw     = output_path / f"sim_heads_diagnostic_{date_tag}_j{job_id}.csv"
    csv_summary = output_path / f"sim_heads_diagnostic_summary_{date_tag}_j{job_id}.csv"

    # ── Load obs patterns ──────────────────────────────────────────────────────
    print("\n[Setup 1/5] Loading GEMS obs patterns...")
    data_loader      = load_data_dynamic_processed(config.amarel_data_load_path)
    all_day_mappings = []
    year_dfmaps, year_means = {}, {}
    for yr in years_list:
        df_map_yr, _, _, monthly_mean_yr = data_loader.load_maxmin_ordered_data_bymonthyear(
            lat_lon_resolution=[1, 1], mm_cond_number=MM_COND,
            years_=[yr], months_=[month], lat_range=lat_r, lon_range=lon_r, is_whittle=False)
        year_dfmaps[yr] = df_map_yr
        year_means[yr]  = monthly_mean_yr
        print(f"  {yr}-{month:02d}: {len(df_map_yr)} time slots")

    # ── Regular target grid ────────────────────────────────────────────────────
    print("[Setup 2/5] Building regular target grid...")
    lats_grid = torch.round(torch.arange(max(lat_r), min(lat_r) - 1e-4, -DELTA_LAT_BASE,
                                          device=DEVICE, dtype=DTYPE) * 10000) / 10000
    lons_grid = torch.round(torch.arange(lon_r[0], lon_r[1] + 1e-4, DELTA_LON_BASE,
                                          device=DEVICE, dtype=DTYPE) * 10000) / 10000
    g_lat, g_lon = torch.meshgrid(lats_grid, lons_grid, indexing='ij')
    grid_coords  = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)
    N_grid = grid_coords.shape[0]
    print(f"  Grid: {len(lats_grid)} lat × {len(lons_grid)} lon = {N_grid} cells")

    # ── High-res grid & obs mappings ───────────────────────────────────────────
    print("[Setup 3/5] Building high-res grid and precomputing obs mappings...")
    lats_hr, lons_hr, dlat_hr, dlon_hr = build_high_res_grid(lat_r, lon_r, lat_factor, lon_factor)
    print(f"  High-res: {len(lats_hr)} lat × {len(lons_hr)} lon = {len(lats_hr)*len(lons_hr):,}")
    DUMMY_KEYS = [f't{i}' for i in range(8)]
    for yr in years_list:
        all_sorted = sorted(year_dfmaps[yr].keys())
        n_days_yr  = len(all_sorted) // 8
        print(f"  {yr}: {n_days_yr} days...", flush=True)
        for d_idx in range(n_days_yr):
            ref_day_map, _ = data_loader.load_working_data(
                year_dfmaps[yr], year_means[yr], [d_idx * 8, (d_idx + 1) * 8],
                ord_mm=None, dtype=DTYPE, keep_ori=True)
            day_keys = sorted(ref_day_map.keys())[:8]
            if len(day_keys) < 8:
                continue
            s3, hr_idx, src = precompute_mapping_indices(
                ref_day_map, lats_hr, lons_hr, grid_coords, day_keys)
            all_day_mappings.append((yr, d_idx, s3, hr_idx, src))
    print(f"  Total day-patterns: {len(all_day_mappings)}")

    # ── Shared NNS ordering (MM_COND=30, fixed for all combos) ────────────────
    print("[Setup 4/5] Computing maxmin ordering (MM_COND=30)...")
    ord_grid = _orderings.maxmin_cpp(grid_coords.cpu().numpy())
    nns_grid = _orderings.find_nns_l2(locs=grid_coords.cpu().numpy()[ord_grid], max_nn=MM_COND)
    print(f"  N_grid={N_grid}  MM_COND={MM_COND}")

    # ── Sanity check ──────────────────────────────────────────────────────────
    print("[Setup 5/5] Sanity check...")
    _short_log    = true_dict_to_log(RANGE_CASES['short'])
    _short_params = torch.tensor(_short_log, device=DEVICE, dtype=DTYPE)
    field0 = generate_field_values(lats_hr, lons_hr, 8, _short_params, dlat_hr, dlon_hr)
    irr0, _ = assemble_irr_dataset(field0, *all_day_mappings[0][2:], DUMMY_KEYS, grid_coords, _short_params)
    del field0
    n_v = (~torch.isnan(list(irr0.values())[0][:, 2])).sum().item()
    print(f"  Sample step valid obs: {n_v}/{N_grid}")

    LBFGS_LR, LBFGS_STEPS, LBFGS_HIST, LBFGS_EVAL = 1.0, 3, 100, 30
    records = []
    skipped = 0

    # ── SIMULATION LOOP ────────────────────────────────────────────────────────
    for it in range(num_iters):
        print(f"\n{'='*70}")
        print(f"  Iteration {it+1}/{num_iters}  [job {job_id}]  (skipped: {skipped})")
        print(f"{'='*70}")

        yr_it, d_it, step3_per_t, hr_idx_per_t, src_locs_per_t = \
            all_day_mappings[rng.integers(len(all_day_mappings))]
        print(f"  Obs pattern: {yr_it} day {d_it}")

        for case_name, true_dict in RANGE_CASES.items():
            print(f"\n  ── Range case: {case_name} "
                  f"(r_lat={true_dict['range_lat']}, r_lon={true_dict['range_lon']}) ──")

            true_log    = true_dict_to_log(true_dict)
            true_params = torch.tensor(true_log, device=DEVICE, dtype=DTYPE)
            initial_vals = make_init(rng, true_log, init_noise)
            init_est     = backmap_params(initial_vals)
            print(f"    init σ²={init_est['sigmasq']:.3f}  "
                  f"r_lon={init_est['range_lon']:.3f}  nugget={init_est['nugget']:.3f}")

            try:
                field = generate_field_values(lats_hr, lons_hr, 8, true_params, dlat_hr, dlon_hr)
                irr_map, _ = assemble_irr_dataset(
                    field, step3_per_t, hr_idx_per_t, src_locs_per_t,
                    DUMMY_KEYS, grid_coords, true_params)
                del field
                irr_map_ord = {k: v[ord_grid] for k, v in irr_map.items()}
            except Exception as e:
                skipped += 1
                print(f"    [SKIP] Field generation failed: {e}")
                continue

            for nheads, limit, combo_name in ISO_COMBOS:
                tag = f"{case_name}/{combo_name}(h={nheads},L={limit})"
                print(f"    {tag}", flush=True)
                try:
                    p_vals = [torch.tensor([v], device=DEVICE, dtype=DTYPE, requires_grad=True)
                              for v in initial_vals]
                    model  = kernels_vecchia_cauchy.fit_cauchy_vecchia_lbfgs(
                        smooth=0.5, gc_beta=GC_BETA,
                        input_map=irr_map_ord, nns_map=nns_grid,
                        mm_cond_number=MM_COND, nheads=nheads,
                        limit_A=limit, limit_B=limit, limit_C=limit,
                        daily_stride=daily_stride,
                    )
                    opt = model.set_optimizer(p_vals, lr=LBFGS_LR,
                                              max_iter=LBFGS_EVAL, history_size=LBFGS_HIST)
                    t0      = time.time()
                    out, _  = model.fit_vecc_lbfgs(p_vals, opt,
                                                   max_steps=LBFGS_STEPS, grad_tol=1e-7)
                    elapsed = time.time() - t0

                    raw_params = out[:-1];  nll_val = float(out[-1])
                    rmsre_val, est_d = calc_rmsre(raw_params, true_dict)
                    print(f"      NLL={nll_val:.4f}  RMSRE={rmsre_val:.4f}  ({elapsed:.1f}s)")

                    records.append({
                        'iter':          it + 1,
                        'job_id':        job_id,
                        'obs_year':      yr_it,
                        'obs_day':       d_it,
                        'range_case':    case_name,
                        'combo':         combo_name,
                        'nheads':        nheads,
                        'limit':         limit,
                        'loss':          round(nll_val,         4),
                        'time_s':        round(elapsed,         2),
                        'rmsre':         round(rmsre_val,       6),
                        'sigmasq_est':   round(est_d['sigmasq'],    4),
                        'range_lat_est': round(est_d['range_lat'],  4),
                        'range_lon_est': round(est_d['range_lon'],  4),
                        'range_t_est':   round(est_d['range_time'], 4),
                        'advec_lat_est': round(est_d['advec_lat'],  4),
                        'advec_lon_est': round(est_d['advec_lon'],  4),
                        'nugget_est':    round(est_d['nugget'],     4),
                    })

                except Exception as e:
                    import traceback
                    print(f"      [FAIL] {tag}: {type(e).__name__}: {e}")
                    traceback.print_exc()

        pd.DataFrame(records).to_csv(csv_raw, index=False)
        _print_running_summary(records, it + 1)

    _print_final_summary(records, csv_summary)
    print(f"\nDone. {len(records)} records → {csv_raw.name}")


# ── Summary helpers ───────────────────────────────────────────────────────────

def _print_combo_table(df, col, fmt='.4f'):
    cw = 12
    header = f"  {'range_case':<8}  {'combo':<12}  {'nheads':>6}  {'limit':>5}  {col:>{cw}}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for case_name in ['short', 'long']:
        for nheads, limit, combo_name in ISO_COMBOS:
            sub = df[(df['range_case'] == case_name) & (df['combo'] == combo_name)][col]
            val = f"{np.mean(sub):{cw}{fmt}}" if len(sub) > 0 else f"{'---':>{cw}}"
            print(f"  {case_name:<8}  {combo_name:<12}  {nheads:>6}  {limit:>5}  {val}")
        print()


def _print_running_summary(records, n_done):
    if not records:
        return
    df = pd.DataFrame(records)
    n_iters = df['iter'].nunique()
    print(f"\n  ── Running summary ({n_iters} complete iters) ──")
    for label, col, fmt in [('Mean NLL',    'loss',   '.4f'),
                             ('Mean time(s)','time_s', '.1f'),
                             ('Mean RMSRE',  'rmsre',  '.4f')]:
        print(f"\n  [{label}]")
        _print_combo_table(df, col, fmt)


def _print_final_summary(records, csv_summary):
    if not records:
        return
    df = pd.DataFrame(records)
    print(f"\n{'='*70}\n  FINAL SUMMARY\n{'='*70}")

    param_cols   = ['sigmasq_est', 'range_lat_est', 'range_lon_est',
                    'range_t_est', 'advec_lat_est', 'advec_lon_est', 'nugget_est']
    param_labels = ['sigmasq', 'range_lat', 'range_lon', 'range_time',
                    'advec_lat', 'advec_lon', 'nugget']

    rows = []
    for case_name, true_dict in RANGE_CASES.items():
        true_vals = [true_dict[k] for k in param_labels]
        for nheads, limit, combo_name in ISO_COMBOS:
            sub = df[(df['range_case'] == case_name) & (df['combo'] == combo_name)]
            if len(sub) == 0:
                continue
            row = {
                'range_case':  case_name,
                'combo':       combo_name,
                'nheads':      nheads,
                'limit':       limit,
                'n':           len(sub),
                'mean_loss':   round(sub['loss'].mean(),   4),
                'sd_loss':     round(sub['loss'].std(),    4),
                'mean_time_s': round(sub['time_s'].mean(), 2),
                'sd_time_s':   round(sub['time_s'].std(),  2),
                'mean_rmsre':  round(sub['rmsre'].mean(),  6),
                'sd_rmsre':    round(sub['rmsre'].std(),   6),
            }
            for col, lbl, tv in zip(param_cols, param_labels, true_vals):
                row[f'rmsre_{lbl}'] = round(
                    float(np.sqrt(np.mean(((sub[col].values - tv) / abs(tv)) ** 2))), 6)
            rows.append(row)

    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(csv_summary, index=False)
    print(f"\n  Summary CSV → {csv_summary.name}")

    for label, col, fmt in [('Final Mean NLL',    'loss',   '.4f'),
                             ('Final Mean time(s)','time_s', '.1f'),
                             ('Final Mean RMSRE',  'rmsre',  '.4f')]:
        print(f"\n  [{label}]")
        _print_combo_table(df, col, fmt)

    # Key diagnostic: RMSRE gap between limit_all and heads_all per range case
    print("\n  [Diagnostic: RMSRE gap  (limit_all - heads_all)  per range case]")
    print(f"  {'range_case':<8}  {'limit_all RMSRE':>16}  {'heads_all RMSRE':>16}  {'gap':>8}  {'verdict':>20}")
    print("  " + "-" * 75)
    for case_name in ['short', 'long']:
        sub_lim  = df[(df['range_case'] == case_name) & (df['combo'] == 'limit_all')]['rmsre']
        sub_head = df[(df['range_case'] == case_name) & (df['combo'] == 'heads_all')]['rmsre']
        if len(sub_lim) == 0 or len(sub_head) == 0:
            continue
        r_lim  = np.mean(sub_lim)
        r_head = np.mean(sub_head)
        gap    = r_lim - r_head
        verdict = 'heads wins (low-freq)' if gap > 0.005 else 'no clear difference'
        print(f"  {case_name:<8}  {r_lim:>16.4f}  {r_head:>16.4f}  {gap:>8.4f}  {verdict:>20}")


if __name__ == "__main__":
    app()
