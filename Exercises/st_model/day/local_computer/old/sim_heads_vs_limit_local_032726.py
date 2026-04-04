"""
sim_heads_vs_limit_local_032726.py  —  LOCAL TEST VERSION
(Amarel version: Exercises/st_model/day/simulation/sim_heads_vs_limit_032726.py)

Quick local test with reduced grid and small data.

Run:
  conda activate faiss_env
  cd /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/local_computer
  python sim_heads_vs_limit_local_032726.py --num-iters 1 --lat-factor 10 --lon-factor 4
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
sys.path.append("/Users/joonwonlee/Documents/GEMS_TCO-1")

from GEMS_TCO import kernels_vecchia_cauchy
from GEMS_TCO import orderings as _orderings
from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data_dynamic_processed

app    = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.float64
DELTA_LAT_BASE = 0.044
DELTA_LON_BASE = 0.063
GC_BETA_DGP    = 1.0

# Reduced grid for local testing
HEADS_LIST  = [0, 100, 200, 400]
LIMITS_LIST = [4, 8, 16, 24]
COMBOS      = [(h, L) for h in HEADS_LIST for L in LIMITS_LIST]   # 16 combos
MM_COND     = 30


def get_covariance_on_grid_cauchy(lx, ly, lt, params, gc_beta=GC_BETA_DGP):
    params = torch.clamp(params, min=-15.0, max=15.0)
    phi1, phi2, phi3, phi4 = (torch.exp(params[i]) for i in range(4))
    u_lat = lx - params[4] * lt
    u_lon = ly - params[5] * lt
    dist  = torch.sqrt(u_lat.pow(2) * phi3 + u_lon.pow(2) + lt.pow(2) * phi4 + 1e-8)
    return (phi1 / phi2) * torch.pow(1.0 + dist * phi2, -gc_beta)


def build_high_res_grid(lat_range, lon_range, lat_factor=10, lon_factor=4):
    dlat = DELTA_LAT_BASE / lat_factor
    dlon = DELTA_LON_BASE / lon_factor
    lats = torch.arange(min(lat_range) - 0.1, max(lat_range) + 0.1, dlat,
                        device=DEVICE, dtype=DTYPE)
    lons = torch.arange(lon_range[0] - 0.1, lon_range[1] + 0.1, dlon,
                        device=DEVICE, dtype=DTYPE)
    return lats, lons, dlat, dlon


def generate_field_values(lats_hr, lons_hr, t_steps, params, dlat, dlon):
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


def apply_step3_1to1(src_np_valid, grid_coords_np, grid_tree):
    N_grid = len(grid_coords_np)
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
        NaN      = float('nan')
        irr_rows = torch.full((N_grid, 11), NaN, device=DEVICE, dtype=DTYPE)
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


def backmap_params(p):
    if isinstance(p[0], torch.Tensor):
        p = [x.item() for x in p]
    p = [float(x) for x in p]
    phi2 = np.exp(p[1]); phi3 = np.exp(p[2]); phi4 = np.exp(p[3])
    rlon = 1.0 / phi2
    return {'sigmasq': np.exp(p[0]) / phi2, 'range_lat': rlon / phi3 ** 0.5,
            'range_lon': rlon, 'range_time': rlon / phi4 ** 0.5,
            'advec_lat': p[4], 'advec_lon': p[5], 'nugget': np.exp(p[6])}


def calc_rmsre(out_params, true_dict):
    est = backmap_params(out_params)
    keys = ['sigmasq', 'range_lat', 'range_lon', 'range_time', 'advec_lat', 'advec_lon', 'nugget']
    est_arr  = np.array([est[k] for k in keys])
    true_arr = np.array([true_dict[k] for k in keys])
    return float(np.sqrt(np.mean(((est_arr - true_arr) / np.abs(true_arr)) ** 2))), est


@app.command()
def cli(
    num_iters:    int   = typer.Option(2,                    help="Iterations"),
    daily_stride: int   = typer.Option(2,                    help="Daily stride for Set C"),
    years:        str   = typer.Option("2024",               help="Years for obs patterns"),
    month:        int   = typer.Option(7,                    help="Reference month"),
    lat_range:    str   = typer.Option("-3,2",               help="lat_min,lat_max"),
    lon_range:    str   = typer.Option("121,131",            help="lon_min,lon_max"),
    lat_factor:   int   = typer.Option(10,                   help="High-res lat multiplier"),
    lon_factor:   int   = typer.Option(4,                    help="High-res lon multiplier"),
    init_noise:   float = typer.Option(0.5,                  help="Log-space init noise"),
    seed:         int   = typer.Option(42,                   help="Random seed"),
) -> None:

    rng = np.random.default_rng(seed)
    lat_r      = [float(x) for x in lat_range.split(',')]
    lon_r      = [float(x) for x in lon_range.split(',')]
    years_list = [y.strip() for y in years.split(',')]

    print(f"Device  : {DEVICE}")
    print(f"DGP     : Cauchy β={GC_BETA_DGP}  (correctly specified)")
    print(f"Grid    : {len(COMBOS)} combos (local reduced)")
    print(f"Iters   : {num_iters}  lat×{lat_factor}  lon×{lon_factor}")

    output_path = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/heads_vs_limit")
    output_path.mkdir(parents=True, exist_ok=True)
    date_tag    = datetime.now().strftime("%m%d%y")
    csv_raw     = output_path / f"sim_heads_vs_limit_{date_tag}_local.csv"
    csv_summary = output_path / f"sim_heads_vs_limit_summary_{date_tag}_local.csv"

    true_dict = {'sigmasq': 10.0, 'range_lat': 0.2, 'range_lon': 0.25, 'range_time': 1.5,
                 'advec_lat': 0.02, 'advec_lon': -0.17, 'nugget': 0.25}
    phi2 = 1.0 / true_dict['range_lon']
    phi1 = true_dict['sigmasq'] * phi2
    phi3 = (true_dict['range_lon'] / true_dict['range_lat'])  ** 2
    phi4 = (true_dict['range_lon'] / true_dict['range_time']) ** 2
    true_log    = [np.log(phi1), np.log(phi2), np.log(phi3), np.log(phi4),
                   true_dict['advec_lat'], true_dict['advec_lon'], np.log(true_dict['nugget'])]
    true_params = torch.tensor(true_log, device=DEVICE, dtype=DTYPE)

    def make_init(rng):
        noisy = list(true_log)
        for i in [0, 1, 2, 3, 6]:
            noisy[i] = true_log[i] + rng.uniform(-init_noise, init_noise)
        for i in [4, 5]:
            noisy[i] = true_log[i] + rng.uniform(-2 * max(abs(true_log[i]), 0.05),
                                                    2 * max(abs(true_log[i]), 0.05))
        return noisy

    print("\n[Setup] Loading GEMS obs patterns...")
    data_loader = load_data_dynamic_processed(config.amarel_data_load_path)
    all_day_mappings = []
    year_dfmaps, year_means = {}, {}
    for yr in years_list:
        df_map_yr, _, _, monthly_mean_yr = data_loader.load_maxmin_ordered_data_bymonthyear(
            lat_lon_resolution=[1, 1], mm_cond_number=MM_COND,
            years_=[yr], months_=[month], lat_range=lat_r, lon_range=lon_r, is_whittle=False)
        year_dfmaps[yr] = df_map_yr
        year_means[yr]  = monthly_mean_yr

    lats_grid = torch.round(torch.arange(max(lat_r), min(lat_r) - 1e-4, -DELTA_LAT_BASE,
                                          device=DEVICE, dtype=DTYPE) * 10000) / 10000
    lons_grid = torch.round(torch.arange(lon_r[0], lon_r[1] + 1e-4, DELTA_LON_BASE,
                                          device=DEVICE, dtype=DTYPE) * 10000) / 10000
    g_lat, g_lon = torch.meshgrid(lats_grid, lons_grid, indexing='ij')
    grid_coords  = torch.stack([g_lat.flatten(), g_lon.flatten()], dim=1)
    N_grid = grid_coords.shape[0]

    lats_hr, lons_hr, dlat_hr, dlon_hr = build_high_res_grid(lat_r, lon_r, lat_factor, lon_factor)
    DUMMY_KEYS = [f't{i}' for i in range(8)]
    for yr in years_list:
        for d_idx in range(len(sorted(year_dfmaps[yr].keys())) // 8):
            ref_day_map, _ = data_loader.load_working_data(
                year_dfmaps[yr], year_means[yr], [d_idx * 8, (d_idx + 1) * 8],
                ord_mm=None, dtype=DTYPE, keep_ori=True)
            day_keys = sorted(ref_day_map.keys())[:8]
            if len(day_keys) < 8:
                continue
            s3, hr_idx, src = precompute_mapping_indices(
                ref_day_map, lats_hr, lons_hr, grid_coords, day_keys)
            all_day_mappings.append((yr, d_idx, s3, hr_idx, src))
    print(f"  {len(all_day_mappings)} day-patterns  N_grid={N_grid}")

    ord_grid = _orderings.maxmin_cpp(grid_coords.cpu().numpy())
    nns_grid = _orderings.find_nns_l2(locs=grid_coords.cpu().numpy()[ord_grid], max_nn=MM_COND)

    LBFGS_LR, LBFGS_STEPS, LBFGS_HIST, LBFGS_EVAL = 1.0, 5, 10, 20
    records  = []

    for it in range(num_iters):
        print(f"\n{'='*60}  Iter {it+1}/{num_iters}  {'='*60}")
        yr_it, d_it, step3_per_t, hr_idx_per_t, src_locs_per_t = \
            all_day_mappings[rng.integers(len(all_day_mappings))]
        initial_vals = make_init(rng)
        print(f"  Obs: {yr_it} day {d_it}")

        try:
            field = generate_field_values(lats_hr, lons_hr, 8, true_params, dlat_hr, dlon_hr)
            irr_map, _ = assemble_irr_dataset(
                field, step3_per_t, hr_idx_per_t, src_locs_per_t,
                DUMMY_KEYS, grid_coords, true_params)
            del field
            irr_map_ord = {k: v[ord_grid] for k, v in irr_map.items()}
        except Exception as e:
            print(f"  [SKIP] {e}")
            continue

        for nheads, limit in COMBOS:
            tag = f"h{nheads}_L{limit}"
            print(f"  {tag}", flush=True)
            try:
                p_vals = [torch.tensor([v], device=DEVICE, dtype=DTYPE, requires_grad=True)
                          for v in initial_vals]
                model  = kernels_vecchia_cauchy.fit_cauchy_vecchia_lbfgs(
                    smooth=0.5, gc_beta=GC_BETA_DGP,
                    input_map=irr_map_ord, nns_map=nns_grid,
                    mm_cond_number=MM_COND, nheads=nheads,
                    limit_A=limit, limit_B=limit, limit_C=limit,
                    daily_stride=daily_stride)
                opt = model.set_optimizer(p_vals, lr=LBFGS_LR,
                                          max_iter=LBFGS_EVAL, max_eval=LBFGS_EVAL,
                                          history_size=LBFGS_HIST)
                t0 = time.time()
                out, _ = model.fit_vecc_lbfgs(p_vals, opt, max_steps=LBFGS_STEPS, grad_tol=1e-5)
                elapsed = time.time() - t0
                raw_params = out[:-1];  nll_val = float(out[-1])
                rmsre_val, est_d = calc_rmsre(raw_params, true_dict)
                print(f"    NLL={nll_val:.4f}  RMSRE={rmsre_val:.4f}  ({elapsed:.1f}s)")
                records.append({
                    'iter': it+1, 'obs_year': yr_it, 'obs_day': d_it,
                    'nheads': nheads, 'limit': limit,
                    'loss': round(nll_val, 4), 'time_s': round(elapsed, 2),
                    'rmsre': round(rmsre_val, 6),
                    'sigmasq_est':   round(est_d['sigmasq'], 4),
                    'range_lat_est': round(est_d['range_lat'], 4),
                    'range_lon_est': round(est_d['range_lon'], 4),
                    'range_t_est':   round(est_d['range_time'], 4),
                    'advec_lat_est': round(est_d['advec_lat'], 4),
                    'advec_lon_est': round(est_d['advec_lon'], 4),
                    'nugget_est':    round(est_d['nugget'], 4),
                })
            except Exception as e:
                print(f"    [FAIL] {tag}: {e}")

        pd.DataFrame(records).to_csv(csv_raw, index=False)

        # Quick running summary
        df = pd.DataFrame(records)
        cw = 8
        for label, col, fmt in [('NLL', 'loss', '.4f'), ('time(s)', 'time_s', '.1f'),
                                 ('RMSRE', 'rmsre', '.4f')]:
            print(f"\n  [{label}]  h\\L  " + "  ".join(f"{L:>{cw}}" for L in LIMITS_LIST))
            for h in HEADS_LIST:
                row = f"  {h:<8}"
                for L in LIMITS_LIST:
                    sub = df[(df['nheads'] == h) & (df['limit'] == L)][col]
                    row += f"  {np.mean(sub):{cw}{fmt}}" if len(sub) > 0 else f"  {'---':>{cw}}"
                print(row)

    pd.DataFrame(records).to_csv(csv_raw, index=False)
    print(f"\nDone → {csv_raw}")


if __name__ == "__main__":
    app()
