"""
sim_GIM_vecc_irr_dw_local_031926.py  —  Local (mac) version

Identical logic to sim_GIM_vecc_irr_dw_031926.py (Amarel).
Differences:
  - sys.path  → mac src path
  - data path → config.mac_data_load_path  (/Users/joonwonlee/Documents/GEMS_DATA/)
  - est CSVs  → GEMS_TCO-1/outputs/day/july_22_24_25/
  - output    → GEMS_TCO-1/outputs/day/GIM/
  - defaults  → num_sims=20, lat_factor=4, lon_factor=2  (faster for local testing)

conda activate faiss_env
cd /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/local_computer
python sim_GIM_vecc_irr_dw_local_031926.py --sample-year 2024 --sample-day 1 --num-sims 20
"""
import sys
import gc
import time
import math
import random
import numpy as np
import pandas as pd
import torch
import typer
from pathlib import Path
from typing import List
from sklearn.neighbors import BallTree

sys.path.append("/Users/joonwonlee/Documents/GEMS_TCO-1/src")

from GEMS_TCO import kernels_vecchia
from GEMS_TCO import orderings as _orderings
from GEMS_TCO import debiased_whittle
from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data_dynamic_processed

# --- GLOBAL SETTINGS ---
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE     = torch.float64
DELTA_LAT = 0.044
DELTA_LON = 0.063

LOCAL_EST_PATH   = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/july_22_24_25")
LOCAL_OUTPUT_DIR = Path("/Users/joonwonlee/Documents/GEMS_TCO-1/outputs/day/GIM")


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Covariance kernel ─────────────────────────────────────────────────────────

def get_model_covariance_on_grid(lags_x, lags_y, lags_t, params):
    params = torch.clamp(params, min=-15.0, max=15.0)
    phi1, phi2, phi3, phi4 = (torch.exp(params[i]) for i in range(4))
    u_lat = lags_x - params[4] * lags_t
    u_lon = lags_y - params[5] * lags_t
    dist  = torch.sqrt(u_lat.pow(2) * phi3 + u_lon.pow(2) + lags_t.pow(2) * phi4 + 1e-8)
    return (phi1 / phi2) * torch.exp(-dist * phi2)


# ── Regular-grid simulation for DW bootstrap ─────────────────────────────────

def generate_regular_data(params_tensor, grid_config):
    """Circulant-embedding simulation on the DW regular grid (CPU float32 for speed)."""
    lats   = grid_config['lats']
    lons   = grid_config['lons']
    t_def  = grid_config['t_def']
    Nx, Ny, Nt = len(lats), len(lons), t_def
    dlat = float(lats[1] - lats[0]) if Nx > 1 else DELTA_LAT
    dlon = float(lons[1] - lons[0]) if Ny > 1 else DELTA_LON
    Px, Py, Pt = 2*Nx, 2*Ny, 2*Nt

    lx = torch.arange(Px, device=DEVICE, dtype=DTYPE) * dlat; lx[Px//2:] -= Px * dlat
    ly = torch.arange(Py, device=DEVICE, dtype=DTYPE) * dlon; ly[Py//2:] -= Py * dlon
    lt = torch.arange(Pt, device=DEVICE, dtype=DTYPE);        lt[Pt//2:] -= Pt

    L_x, L_y, L_t = torch.meshgrid(lx, ly, lt, indexing='ij')
    C = get_model_covariance_on_grid(L_x, L_y, L_t, params_tensor)
    S = torch.fft.fftn(C); S.real = torch.clamp(S.real, min=0)
    field = torch.fft.ifftn(torch.sqrt(S.real) * torch.fft.fftn(
        torch.randn(Px, Py, Pt, device=DEVICE, dtype=DTYPE))).real[:Nx, :Ny, :Nt]

    nugget_std = torch.sqrt(torch.exp(params_tensor[6]))
    grid_lat, grid_lon = torch.meshgrid(lats, lons, indexing='ij')
    flat_lats, flat_lons = grid_lat.flatten(), grid_lon.flatten()

    agg_list = []
    for t in range(t_def):
        obs = field[:, :, t].flatten() + torch.randn_like(flat_lats) * nugget_std
        row = torch.stack([flat_lats, flat_lons, obs,
                           torch.full_like(flat_lats, 21.0 + t)], dim=1).detach()
        agg_list.append(row)
    return torch.cat(agg_list, dim=0)


# ── High-res FFT pipeline ─────────────────────────────────────────────────────

def build_high_res_grid(lat_range, lon_range, lat_factor, lon_factor):
    dlat = DELTA_LAT / lat_factor
    dlon = DELTA_LON / lon_factor
    lats = torch.arange(lat_range[0] - 0.1, lat_range[1] + 0.1, dlat, device=DEVICE, dtype=DTYPE)
    lons = torch.arange(lon_range[0] - 0.1, lon_range[1] + 0.1, dlon, device=DEVICE, dtype=DTYPE)
    return lats, lons, dlat, dlon


def generate_field_values(lats_hr, lons_hr, t_steps, params, dlat, dlon):
    CPU = torch.device("cpu"); F32 = torch.float32
    Nx, Ny, Nt = len(lats_hr), len(lons_hr), t_steps
    Px, Py, Pt = 2*Nx, 2*Ny, 2*Nt

    lx = torch.arange(Px, device=CPU, dtype=F32) * dlat; lx[Px//2:] -= Px * dlat
    ly = torch.arange(Py, device=CPU, dtype=F32) * dlon; ly[Py//2:] -= Py * dlon
    lt = torch.arange(Pt, device=CPU, dtype=F32);        lt[Pt//2:] -= Pt

    params_cpu = params.detach().cpu().float()
    Lx, Ly, Lt = torch.meshgrid(lx, ly, lt, indexing='ij')
    C = get_model_covariance_on_grid(Lx, Ly, Lt, params_cpu)
    S = torch.fft.fftn(C); S.real = torch.clamp(S.real, min=0)
    field = torch.fft.ifftn(torch.sqrt(S.real) *
                             torch.fft.fftn(torch.randn(Px, Py, Pt, device=CPU, dtype=F32))
                             ).real[:Nx, :Ny, :Nt]
    return field.to(dtype=DTYPE, device=DEVICE)


def apply_step3_1to1(src_np_valid, grid_coords_np, grid_tree):
    N_grid = len(grid_coords_np)
    if len(src_np_valid) == 0:
        return np.full(N_grid, -1, dtype=np.int64)
    dist_to_cell, cell_for_obs = grid_tree.query(np.radians(src_np_valid), k=1)
    dist_to_cell = dist_to_cell.flatten()
    cell_for_obs = cell_for_obs.flatten()
    assignment = np.full(N_grid, -1, dtype=np.int64)
    best_dist   = np.full(N_grid, np.inf)
    for obs_i, (cell_j, d) in enumerate(zip(cell_for_obs, dist_to_cell)):
        if d < best_dist[cell_j]:
            assignment[cell_j] = obs_i
            best_dist[cell_j]  = d
    return assignment


def precompute_mapping_indices(ref_irr_map, grid_coords, lats_hr, lons_hr, sorted_keys):
    hr_lat_g, hr_lon_g = torch.meshgrid(lats_hr, lons_hr, indexing='ij')
    hr_coords_np = torch.stack([hr_lat_g.flatten(), hr_lon_g.flatten()], dim=1).cpu().numpy()
    hr_tree = BallTree(np.radians(hr_coords_np), metric='haversine')

    grid_coords_np = grid_coords.cpu().numpy()
    grid_tree = BallTree(np.radians(grid_coords_np), metric='haversine')

    step3_per_t, hr_idx_per_t, src_locs_per_t, valid_rows_per_t = [], [], [], []
    for key in sorted_keys:
        ref_t      = ref_irr_map[key].to(DEVICE)
        src_np     = ref_t[:, :2].cpu().numpy()
        valid_mask = ~np.isnan(src_np).any(axis=1)

        step3_per_t.append(apply_step3_1to1(src_np[valid_mask], grid_coords_np, grid_tree))
        valid_rows_per_t.append(torch.tensor(valid_mask, device=DEVICE))

        if valid_mask.sum() > 0:
            _, hr_idx = hr_tree.query(np.radians(src_np[valid_mask]), k=1)
            hr_idx_per_t.append(torch.tensor(hr_idx.flatten(), device=DEVICE))
        else:
            hr_idx_per_t.append(torch.zeros(0, dtype=torch.long, device=DEVICE))

        src_locs_per_t.append(ref_t[valid_mask, :2])

    return step3_per_t, hr_idx_per_t, src_locs_per_t, valid_rows_per_t


def generate_bootstrap_pair(params, ref_irr_map, grid_coords, lats_hr, lons_hr, dlat_hr, dlon_hr,
                             step3_per_t, hr_idx_per_t, src_locs_per_t, valid_rows_per_t,
                             sorted_keys, time_vals):
    t_def = len(sorted_keys)
    Nx_hr, Ny_hr = len(lats_hr), len(lons_hr)

    field      = generate_field_values(lats_hr, lons_hr, t_def, params, dlat_hr, dlon_hr)
    field_flat = field.reshape(Nx_hr * Ny_hr, t_def)

    nugget_std = torch.sqrt(torch.exp(params[6]))
    N_grid     = grid_coords.shape[0]
    NaN        = float('nan')

    irr_map  = {}
    reg_list = []

    for t_idx, key in enumerate(sorted_keys):
        t_val      = time_vals[t_idx]
        assign     = step3_per_t[t_idx]
        hr_idx     = hr_idx_per_t[t_idx]
        valid_rows = valid_rows_per_t[t_idx]
        N_valid    = hr_idx.shape[0]

        if N_valid > 0:
            gp_vals  = field_flat[hr_idx, t_idx]
            sim_vals = gp_vals + torch.randn(N_valid, device=DEVICE, dtype=DTYPE) * nugget_std
        else:
            sim_vals = torch.zeros(0, device=DEVICE, dtype=DTYPE)

        # irr_map: clone real structure, replace value column only
        irr_rows = ref_irr_map[key].clone()
        irr_rows[:, 2] = NaN
        if N_valid > 0:
            valid_indices = torch.where(valid_rows)[0]
            irr_rows[valid_indices, 2] = sim_vals
        irr_map[key] = irr_rows.detach()

        # reg_agg: step3 re-gridded [lat, lon, val, time] for DW
        reg_rows = torch.full((N_grid, 4), NaN, device=DEVICE, dtype=DTYPE)
        reg_rows[:, :2] = grid_coords
        reg_rows[:, 3]  = t_val
        if N_valid > 0:
            assign_t = torch.tensor(assign, device=DEVICE, dtype=torch.long)
            filled   = assign_t >= 0
            win_obs  = assign_t[filled]
            reg_rows[filled, 2] = sim_vals[win_obs]
        reg_list.append(reg_rows.detach())

    return irr_map, torch.cat(reg_list, dim=0)


# ── Utilities ─────────────────────────────────────────────────────────────────

def transform_raw_to_log_phi(raw: list) -> list:
    sigmasq, range_lat, range_lon, range_time, advec_lat, advec_lon, nugget = raw
    phi2 = 1.0 / max(range_lon, 1e-4)
    phi1 = sigmasq * phi2
    phi3 = (range_lon / max(range_lat, 1e-4)) ** 2
    phi4 = (range_lon / max(range_time, 1e-4)) ** 2
    return [math.log(phi1), math.log(phi2), math.log(phi3), math.log(phi4),
            advec_lat, advec_lon, math.log(max(nugget, 1e-8))]


def transform_log_phi_to_physical(p: torch.Tensor) -> torch.Tensor:
    phi1, phi2, phi3, phi4 = (torch.exp(p[i]) for i in range(4))
    rlon = 1.0 / phi2
    return torch.stack([phi1/phi2, rlon/torch.sqrt(phi3), rlon,
                        rlon/torch.sqrt(phi4), p[4], p[5], torch.exp(p[6])])


# ── Finite-difference Hessian ─────────────────────────────────────────────────

def finite_diff_hessian(nll_fn, p, eps=1e-4):
    n = p.shape[0]
    H = torch.zeros(n, n, device=p.device, dtype=p.dtype)
    for i in range(n):
        p_p = p.detach().clone(); p_p[i] += eps; p_p.requires_grad_(True)
        p_m = p.detach().clone(); p_m[i] -= eps; p_m.requires_grad_(True)
        g_p = torch.autograd.grad(nll_fn(p_p), p_p)[0].detach()
        g_m = torch.autograd.grad(nll_fn(p_m), p_m)[0].detach()
        H[i] = (g_p - g_m) / (2.0 * eps)
    return (H + H.T) / 2.0


# ── Main CLI ──────────────────────────────────────────────────────────────────

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})

@app.command()
def cli(
    sample_year: str = typer.Option('2024', help="Year of representative day"),
    sample_day:  int = typer.Option(1,      help="Day of month (1-based)"),
    month:       int = typer.Option(7,      help="Month"),
    v:           float = typer.Option(0.5,  help="Matern smoothness"),
    mm_cond_number: int = typer.Option(100, help="Vecchia neighbors"),
    nheads:      int = typer.Option(200,    help="Head points"),
    limit_a:     int = typer.Option(8,      help="Set A neighbors"),
    limit_b:     int = typer.Option(8,      help="Set B neighbors"),
    limit_c:     int = typer.Option(8,      help="Set C neighbors"),
    daily_stride: int = typer.Option(2,     help="Set C stride"),
    num_sims:    int = typer.Option(20,     help="GIM bootstrap samples (local default=20)"),
    lat_factor:  int = typer.Option(4,      help="High-res FFT lat factor (local default=4)"),
    lon_factor:  int = typer.Option(2,      help="High-res FFT lon factor (local default=2)"),
) -> None:

    set_seed(2025)
    day_str = f"{sample_year}-{month:02d}-{sample_day}"
    day_idx = sample_day - 1
    print(f"Device: {DEVICE}")
    print(f"Day: {day_str}  num_sims={num_sims}  lat_factor={lat_factor}  lon_factor={lon_factor}")

    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = LOCAL_OUTPUT_DIR / f"GIM_{day_str}_nsims{num_sims}_local.csv"

    # ── Load fitted estimates ─────────────────────────────────────────────────
    dw_csv   = LOCAL_EST_PATH / "real_dw_july_22_24_25.csv"
    vecc_csv = LOCAL_EST_PATH / "real_vecc_july_22_24_25_h1000_mm16.csv"
    if not dw_csv.exists() or not vecc_csv.exists():
        print(f"[Error] CSV not found in {LOCAL_EST_PATH}")
        raise SystemExit(1)

    _param_cols = ['sigma', 'range_lat', 'range_lon', 'range_time',
                   'advec_lat', 'advec_lon', 'nugget']
    dw_df   = pd.read_csv(dw_csv)
    vecc_df = pd.read_csv(vecc_csv)
    dw_by_day   = {row['day']: [row[c] for c in _param_cols] for _, row in dw_df.iterrows()}
    vecc_by_day = {row['day']: [row[c] for c in _param_cols] for _, row in vecc_df.iterrows()}

    if day_str not in dw_by_day or day_str not in vecc_by_day:
        print(f"[Error] No estimates for {day_str}. Available: {sorted(dw_by_day.keys())[:5]} ...")
        raise SystemExit(1)

    print(f"DW   params: {[round(x,4) for x in dw_by_day[day_str]]}")
    print(f"Vecc params: {[round(x,4) for x in vecc_by_day[day_str]]}")

    # ── Load spatial data ─────────────────────────────────────────────────────
    data_load_instance = load_data_dynamic_processed(config.mac_data_load_path)
    df_map, ord_mm, nns_map, monthly_mean = data_load_instance.load_maxmin_ordered_data_bymonthyear(
        mm_cond_number=mm_cond_number,
        years_=[sample_year], months_=[month],
        lat_range=[-3, 2], lon_range=[121, 131],
        is_whittle=False
    )
    hour_indices = [day_idx * 8, (day_idx + 1) * 8]

    p_names = ["SigmaSq", "RangeLat", "RangeLon", "RangeTime", "AdvecLat", "AdvecLon", "Nugget"]
    dwl = debiased_whittle.debiased_whittle_likelihood()
    t0  = time.time()

    # ── Load both DW and Vecchia data ─────────────────────────────────────────
    dw_map, dw_agg = data_load_instance.load_working_data(
        df_map, monthly_mean, hour_indices,
        ord_mm=None, dtype=DTYPE, keep_ori=False
    )
    real_agg_dw = dw_agg.to(DEVICE)

    vecc_map, _ = data_load_instance.load_working_data(
        df_map, monthly_mean, hour_indices,
        ord_mm=ord_mm, dtype=DTYPE, keep_ori=True
    )
    real_map_vecc = {k: v.to(DEVICE) for k, v in vecc_map.items()}

    # ── Precompute mapping indices once ───────────────────────────────────────
    sorted_keys = sorted(real_map_vecc.keys())
    time_vals   = [float(real_map_vecc[k][0, 3]) for k in sorted_keys]
    t_def       = len(sorted_keys)

    grid_lats = torch.unique(real_agg_dw[:, 0])
    grid_lons = torch.unique(real_agg_dw[:, 1])
    glat_g, glon_g = torch.meshgrid(grid_lats, grid_lons, indexing='ij')
    grid_coords = torch.stack([glat_g.flatten(), glon_g.flatten()], dim=1)

    lat_range_data = [float(grid_lats.min()), float(grid_lats.max())]
    lon_range_data = [float(grid_lons.min()), float(grid_lons.max())]
    lats_hr, lons_hr, dlat_hr, dlon_hr = build_high_res_grid(
        lat_range_data, lon_range_data, lat_factor, lon_factor)
    print(f"  High-res grid: {len(lats_hr)} × {len(lons_hr)}  (factor {lat_factor}×{lon_factor})")

    step3_per_t, hr_idx_per_t, src_locs_per_t, valid_rows_per_t = precompute_mapping_indices(
        real_map_vecc, grid_coords, lats_hr, lons_hr, sorted_keys)

    obs_rate = sum(v.float().mean().item() for v in valid_rows_per_t) / t_def
    print(f"  Obs rate: {obs_rate:.3f}  |  grid: {grid_coords.shape[0]} cells  |  T={t_def}")

    # ── DW GIM ────────────────────────────────────────────────────────────────
    print(f"\n[1/2] DW GIM  ({day_str})...")
    dw_log_phi = torch.tensor(
        transform_raw_to_log_phi(dw_by_day[day_str]),
        device=DEVICE, dtype=DTYPE, requires_grad=True
    )

    db = debiased_whittle.debiased_whittle_preprocess(
        [real_agg_dw], [dw_map], day_idx=0,
        params_list=dw_by_day[day_str],
        lat_range=[-3, 2], lon_range=[121.0, 131.0]
    )
    cur_df       = db.generate_spatially_filtered_days(-3, 2, 121, 131).to(DEVICE)
    unique_times = torch.unique(cur_df[:, 3])
    time_slices  = [cur_df[cur_df[:, 3] == t] for t in unique_times]

    J_vec, n1, n2, p_time, taper_grid = dwl.generate_Jvector_tapered(
        time_slices, dwl.cgn_hamming, 0, 1, 2, DEVICE)
    I_obs  = dwl.calculate_sample_periodogram_vectorized(J_vec)
    t_auto = dwl.calculate_taper_autocorrelation_fft(taper_grid, n1, n2, DEVICE)

    def nll_dw(p):
        loss = dwl.whittle_likelihood_loss_tapered(
            p, I_obs, n1, n2, p_time, t_auto, DELTA_LAT, DELTA_LON)
        return loss[0] if isinstance(loss, tuple) else loss

    H_dw       = finite_diff_hessian(nll_dw, dw_log_phi)
    eigvals_dw = torch.linalg.eigvalsh(H_dw)
    print(f"  H_dw eigenvalues: {eigvals_dw.detach().cpu().numpy().round(4)}")
    print(f"  H_dw condition number: {(eigvals_dw.max()/eigvals_dw.abs().min()).item():.2e}")
    H_inv_dw = torch.linalg.inv(H_dw + torch.eye(7, device=DEVICE) * 1e-5)

    # ── DW J: parametric bootstrap via proper data pipeline ───────────────────
    # Pipeline mirrors real data:
    #   circulant embedding (high-res) → sample at irregular obs locations
    #   → step3 re-grid → DW score
    # Uses dw_log_phi (DW MLE) for simulation, then centers gradients.
    nan_mask_dw = torch.isnan(real_agg_dw[:, 2])
    obs_rate_dw = 1.0 - nan_mask_dw.float().mean().item()
    print(f"  DW obs rate: {obs_rate_dw:.3f}")

    grads_dw = []
    for i in range(num_sims):
        with torch.no_grad():
            _, b_reg = generate_bootstrap_pair(
                dw_log_phi, real_map_vecc, grid_coords,
                lats_hr, lons_hr, dlat_hr, dlon_hr,
                step3_per_t, hr_idx_per_t, src_locs_per_t, valid_rows_per_t,
                sorted_keys, time_vals)
        b_reg = b_reg.clone()
        b_reg[nan_mask_dw, 2] = float('nan')
        b_slices = [b_reg[b_reg[:, 3] == t] for t in unique_times]
        J_b, bn1, bn2, _, bt = dwl.generate_Jvector_tapered(
            b_slices, dwl.cgn_hamming, 0, 1, 2, DEVICE)
        I_b = dwl.calculate_sample_periodogram_vectorized(J_b)
        ta  = t_auto if (bn1 == n1 and bn2 == n2) else \
              dwl.calculate_taper_autocorrelation_fft(bt, bn1, bn2, DEVICE)
        if dw_log_phi.grad is not None: dw_log_phi.grad.zero_()
        loss = dwl.whittle_likelihood_loss_tapered(
            dw_log_phi, I_b, bn1, bn2, p_time, ta, DELTA_LAT, DELTA_LON)
        (loss[0] if isinstance(loss, tuple) else loss).backward()
        grads_dw.append(dw_log_phi.grad.detach().clone())
        del b_reg, b_slices, J_b, I_b, loss
        if (i + 1) % 5 == 0:
            print(f"  DW bootstrap {i+1}/{num_sims}")

    grads_stack  = torch.stack(grads_dw)           # [num_sims, 7]
    mean_grad_dw = grads_stack.mean(dim=0)          # [7]
    centered_dw  = grads_stack - mean_grad_dw       # centered
    J_dw         = centered_dw.T @ centered_dw / num_sims

    grad_norms_dw = [g.norm().item() for g in grads_dw]
    print(f"  DW grad norm — mean: {np.mean(grad_norms_dw):.4e}  std: {np.std(grad_norms_dw):.4e}")
    print(f"  DW mean_grad (sanity, should be ~0): {mean_grad_dw.cpu().numpy().round(6)}")
    print(f"  J_dw diag:  {torch.diag(J_dw).detach().cpu().numpy().round(4)}")
    print(f"  H_dw diag:  {torch.diag(H_dw).detach().cpu().numpy().round(4)}")
    GIM_dw = H_inv_dw @ J_dw @ H_inv_dw
    Jac_dw = torch.autograd.functional.jacobian(transform_log_phi_to_physical, dw_log_phi)
    SE_dw  = torch.sqrt(torch.diag(Jac_dw @ GIM_dw @ Jac_dw.T)).detach().cpu().numpy()
    Pt_dw  = transform_log_phi_to_physical(dw_log_phi).detach().cpu().numpy()

    del real_agg_dw, dw_map, H_dw, J_dw, GIM_dw
    gc.collect()

    # ── Vecchia-Irr GIM ───────────────────────────────────────────────────────
    print(f"\n[2/2] Vecchia-Irr GIM  ({day_str})...")
    vc_log_phi = torch.tensor(
        transform_raw_to_log_phi(vecc_by_day[day_str]),
        device=DEVICE, dtype=DTYPE, requires_grad=True
    )

    model_vc = kernels_vecchia.fit_vecchia_lbfgs(
        smooth=v, input_map=real_map_vecc,
        nns_map=nns_map, mm_cond_number=mm_cond_number, nheads=nheads,
        limit_A=limit_a, limit_B=limit_b, limit_C=limit_c, daily_stride=daily_stride
    )
    model_vc.precompute_conditioning_sets()

    def nll_vc(p): return model_vc.vecchia_batched_likelihood(p)

    H_vc       = finite_diff_hessian(nll_vc, vc_log_phi)
    eigvals_vc = torch.linalg.eigvalsh(H_vc)
    print(f"  H_vc eigenvalues: {eigvals_vc.detach().cpu().numpy().round(4)}")
    print(f"  H_vc condition number: {(eigvals_vc.max()/eigvals_vc.abs().min()).item():.2e}")
    H_inv_vc = torch.linalg.inv(H_vc + torch.eye(7, device=DEVICE) * 1e-5)

    model_vc.input_map = real_map_vecc

    grads_vc = []
    for i in range(num_sims):
        with torch.no_grad():
            b_irr, _ = generate_bootstrap_pair(
                vc_log_phi, real_map_vecc, grid_coords,
                lats_hr, lons_hr, dlat_hr, dlon_hr,
                step3_per_t, hr_idx_per_t, src_locs_per_t, valid_rows_per_t,
                sorted_keys, time_vals)
        model_vc.input_map = b_irr
        model_vc.refresh_y_from_input_map()
        if vc_log_phi.grad is not None: vc_log_phi.grad.zero_()
        model_vc.vecchia_batched_likelihood(vc_log_phi).backward()
        grads_vc.append(vc_log_phi.grad.detach().clone())
        del b_irr
        if (i + 1) % 5 == 0:
            print(f"  Vecchia bootstrap {i+1}/{num_sims}")

    J_vc          = torch.stack(grads_vc).T @ torch.stack(grads_vc) / num_sims
    grad_norms_vc = [g.norm().item() for g in grads_vc]
    print(f"  VC grad norm  — mean: {np.mean(grad_norms_vc):.4e}  std: {np.std(grad_norms_vc):.4e}")
    print(f"  J_vc diag:  {torch.diag(J_vc).detach().cpu().numpy().round(6)}")
    print(f"  H_vc diag:  {torch.diag(H_vc).detach().cpu().numpy().round(6)}")
    GIM_vc = H_inv_vc @ J_vc @ H_inv_vc
    Jac_vc = torch.autograd.functional.jacobian(transform_log_phi_to_physical, vc_log_phi)
    SE_vc  = torch.sqrt(torch.diag(Jac_vc @ GIM_vc @ Jac_vc.T)).detach().cpu().numpy()
    Pt_vc  = transform_log_phi_to_physical(vc_log_phi).detach().cpu().numpy()

    # ── Save & print ──────────────────────────────────────────────────────────
    row = {"day": day_str, "num_sims": num_sims, "lat_factor": lat_factor, "lon_factor": lon_factor}
    for k, name in enumerate(p_names):
        row[f"DW_Est_{name}"]     = round(float(Pt_dw[k]), 4)
        row[f"DW_SE_{name}"]      = round(float(SE_dw[k]), 4)
        row[f"VC_Irr_Est_{name}"] = round(float(Pt_vc[k]), 4)
        row[f"VC_Irr_SE_{name}"]  = round(float(SE_vc[k]), 4)
    pd.DataFrame([row]).to_csv(out_file, index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  GIM results — {day_str}  ({elapsed:.1f}s total)")
    print(f"{'='*65}")
    print(f"  {'Param':<10} | {'DW Est':>8} | {'DW SE':>8} || {'VC Est':>8} | {'VC SE':>8}")
    print(f"  {'-'*63}")
    for k, name in enumerate(p_names):
        print(f"  {name:<10} | {Pt_dw[k]:>8.4f} | {SE_dw[k]:>8.4f} || "
              f"{Pt_vc[k]:>8.4f} | {SE_vc[k]:>8.4f}")
    print(f"\n  Saved: {out_file}")

    del real_map_vecc, vecc_map, model_vc, H_vc, J_vc, GIM_vc, grads_vc
    gc.collect()
    print("\n[Done]")


if __name__ == "__main__":
    app()
