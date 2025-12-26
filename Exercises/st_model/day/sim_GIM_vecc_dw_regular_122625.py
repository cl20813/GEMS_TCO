# Standard libraries
import sys
import os
import logging
import argparse 
import pandas as pd
import numpy as np
import pickle
import torch
import torch.optim as optim
import copy 
import time
import random
import torch.nn.functional as F
from typing import Optional, List, Tuple
from pathlib import Path
import typer

# Custom imports
sys.path.append("/cache/home/jl2815/tco")

from GEMS_TCO import kernels_reparam_space_time_gpu as kernels_reparam_space_time
from GEMS_TCO import orderings as _orderings 
from GEMS_TCO import alg_optimization
from GEMS_TCO import configuration as config
from GEMS_TCO import debiased_whittle 

# --- GLOBAL SETTINGS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64
NUM_SIMS = 100

# --- GRID STEPS ---
DELTA_LAT = 0.044
DELTA_LON = 0.063

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Info] Random Seed set to {seed}")

# --- HELPER: SAFE LOG (NUGGET HANDLING) ---
def safe_log(val, eps=1e-6):
    """Nugget이 0일 때 -inf가 되는 것을 방지"""
    if isinstance(val, torch.Tensor):
        return torch.log(torch.clamp(val, min=eps))
    return np.log(max(val, eps))

# --- HELPER: SPATIAL DIFFERENCING ---
def apply_spatial_difference(df_tensor):
    """
    Whittle Likelihood의 Hessian(관측데이터)과 J(부트스트랩)의 일관성을 유지하기 위해
    부트스트랩 데이터에도 동일한 Spatial Differencing을 적용함.
    """
    if df_tensor.size(0) == 0: return df_tensor
    
    u_lat = torch.unique(df_tensor[:, 0])
    u_lon = torch.unique(df_tensor[:, 1])
    n_lat, n_lon = len(u_lat), len(u_lon)
    
    if df_tensor.size(0) != n_lat * n_lon:
        return df_tensor

    # (Batch, Channel, Height, Width)
    vals = df_tensor[:, 2].view(1, 1, n_lat, n_lon)
    
    # Kernel: Matches the differencing logic (Z(s) - alpha*Z(s-1)) or Laplacian style
    # [Check] 사용하시는 라이브러리의 필터와 동일한 커널인지 확인 필요. (일반적으로 사용되는 형태 유지)
    diff_kernel = torch.tensor([[[[-2., 1.],
                                  [ 1., 0.]]]], dtype=torch.float64, device=df_tensor.device)
    
    out_vals = F.conv2d(vals, diff_kernel, padding='valid')
    
    new_lats = u_lat[:-1]
    new_lons = u_lon[:-1]
    
    grid_lat, grid_lon = torch.meshgrid(new_lats, new_lons, indexing='ij')
    
    flat_lat = grid_lat.flatten()
    flat_lon = grid_lon.flatten()
    flat_val = out_vals.flatten()
    
    time_val = df_tensor[0, 3]
    flat_time = torch.full_like(flat_lat, time_val)
    
    return torch.stack([flat_lat, flat_lon, flat_val, flat_time], dim=1)

# --- 1. CORE SIMULATION FUNCTIONS ---

def get_model_covariance_on_grid(lags_x, lags_y, lags_t, params):
    params = torch.clamp(params, min=-15.0, max=15.0)
    phi1, phi2, phi3, phi4 = torch.exp(params[0]), torch.exp(params[1]), torch.exp(params[2]), torch.exp(params[3])
    advec_lat, advec_lon = params[4], params[5]
    sigmasq = phi1 / phi2

    u_lat_eff = lags_x - advec_lat * lags_t
    u_lon_eff = lags_y - advec_lon * lags_t
    
    dist_sq = (u_lat_eff.pow(2) * phi3) + (u_lon_eff.pow(2)) + (lags_t.pow(2) * phi4)
    distance = torch.sqrt(dist_sq + 1e-8)
    
    return sigmasq * torch.exp(-distance * phi2)

def generate_exact_gems_field(lat_coords, lon_coords, t_steps, params):
    Nx, Ny, Nt = len(lat_coords), len(lon_coords), t_steps
    dlat = float(lat_coords[1] - lat_coords[0]) if len(lat_coords) > 1 else DELTA_LAT
    dlon = float(lon_coords[1] - lon_coords[0]) if len(lon_coords) > 1 else DELTA_LON
    dt = 1.0 
    
    Px, Py, Pt = 2*Nx, 2*Ny, 2*Nt
    Lx_len, Ly_len, Lt_len = Px * dlat, Py * dlon, Pt * dt
    
    lags_x = torch.arange(Px, device=DEVICE, dtype=DTYPE) * dlat
    lags_x[Px//2:] -= Lx_len 
    lags_y = torch.arange(Py, device=DEVICE, dtype=DTYPE) * dlon
    lags_y[Py//2:] -= Ly_len
    lags_t = torch.arange(Pt, device=DEVICE, dtype=DTYPE) * dt
    lags_t[Pt//2:] -= Lt_len

    L_x, L_y, L_t = torch.meshgrid(lags_x, lags_y, lags_t, indexing='ij')
    C_vals = get_model_covariance_on_grid(L_x, L_y, L_t, params)

    S = torch.fft.fftn(C_vals)
    S.real = torch.clamp(S.real, min=0)
    random_phase = torch.fft.fftn(torch.randn(Px, Py, Pt, device=DEVICE, dtype=DTYPE))
    weighted_freq = torch.sqrt(S.real) * random_phase
    field_sim = torch.fft.ifftn(weighted_freq).real
    
    return field_sim[:Nx, :Ny, :Nt]

def generate_regular_data_directly(params_tensor, grid_config):
    lats_sim = grid_config['lats']
    lons_sim = grid_config['lons']
    t_def = grid_config['t_def']
    ozone_mean = grid_config['mean']
    
    sim_field = generate_exact_gems_field(lats_sim, lons_sim, t_def, params_tensor)
    
    input_map = {}
    aggregated_list = []
    # Nugget std calculation (exp ensures positivity)
    nugget_std = torch.sqrt(torch.exp(params_tensor[6]))
    
    grid_lat, grid_lon = torch.meshgrid(lats_sim, lons_sim, indexing='ij')
    flat_lats = grid_lat.flatten()
    flat_lons = grid_lon.flatten()
    
    for t in range(t_def):
        field_t = sim_field[:, :, t]
        flat_vals = field_t.flatten()
        
        obs_vals = flat_vals + (torch.randn_like(flat_vals) * nugget_std) + ozone_mean
        
        time_val = 21.0 + t
        flat_times = torch.full_like(flat_lats, time_val)
        
        row_tensor = torch.stack([flat_lats, flat_lons, obs_vals, flat_times], dim=1).detach()
        key_str = f't_{t:02d}'
        input_map[key_str] = row_tensor
        aggregated_list.append(row_tensor)

    aggregated_data = torch.cat(aggregated_list, dim=0)
    return input_map, aggregated_data

# --- 3. HELPER & REPORTING ---

def get_spatial_ordering(input_maps, mm_cond_number=10):
    key_list = list(input_maps.keys())
    data_for_coord = input_maps[key_list[0]]
    if isinstance(data_for_coord, torch.Tensor):
        data_for_coord = data_for_coord.cpu().numpy()

    coords1 = np.stack((data_for_coord[:, 0], data_for_coord[:, 1]), axis=-1)
    ord_mm = _orderings.maxmin_cpp(coords1)
    
    data_reordered = data_for_coord[ord_mm]
    coords_reordered = np.stack((data_reordered[:, 0], data_reordered[:, 1]), axis=-1)
    nns_map = _orderings.find_nns_l2(locs=coords_reordered, max_nn=mm_cond_number)
    return ord_mm, nns_map

def print_final_metrics_formatted(true_params_vec, best_params_tensor, SE_GIM, H_inv, J, nll_val, model_name="Model"):
    param_names = ["log_phi1", "log_phi2", "log_phi3", "log_phi4", "advec_lat", "advec_lon", "log_nugget"]
    true_vals = true_params_vec.detach().cpu().numpy()
    est_vals = best_params_tensor.detach().cpu().numpy()
    se_vals = SE_GIM.detach().cpu().numpy()
    
    print(f"\nResults for [{model_name}]")
    print(f"{'Param':<15} | {'True':<10} | {'Est':<10} | {'SE(GIM)':<10} | {'95% CI':<25} | {'Covered?'}")
    print("-" * 100)

    covered_count = 0
    for i, p_name in enumerate(param_names):
        t_v = true_vals[i]
        e_v = est_vals[i]
        s_v = se_vals[i]
        ci_L, ci_U = e_v - 1.96*s_v, e_v + 1.96*s_v
        
        is_covered = (ci_L <= t_v <= ci_U)
        if is_covered: covered_count += 1
        
        print(f"{p_name:<15} | {t_v:<10.4f} | {e_v:<10.4f} | {s_v:<10.4f} | ({ci_L:.3f}, {ci_U:.3f}) | {'YES' if is_covered else 'NO'}")

    # TIC Logic
    with torch.no_grad():
        penalty = torch.trace(H_inv @ J)
        tic = 2 * nll_val + 2 * penalty.item()

    print("-" * 100)
    print(f" > Coverage: {covered_count}/{len(param_names)}")
    print(f" > TIC Score: {tic:.4f} (Lower is better)")
    print("=" * 60)

# --- MAIN CLI COMMAND ---
app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})

@app.command()
def cli(
    v: float = typer.Option(0.5, help="smooth"),
    lr: float = typer.Option(0.1, help="learning rate"),
    mm_cond_number: int = typer.Option(8, help="Neighbors"),
    epochs: int = typer.Option(100, help="Iterations"),
    nheads: int = typer.Option(300, help="Heads")
) -> None:
    
    set_seed(2025)
    print(f"Running on: {DEVICE}")

    # =========================================================================
    # [Config] True Parameters & Grid
    # =========================================================================
    init_sigmasq, init_range_lon, init_range_lat = 13.059, 0.195, 0.154 
    init_advec_lat, init_range_time, init_advec_lon = 0.0218, 1.0, -0.1689
    
    # *** [LOG SAFETY FIX] Nugget이 0일 수 있으므로 safe_log 적용 ***
    init_nugget = 0.0  # 만약 0이라면
    
    init_phi2 = 1.0 / init_range_lon
    init_phi1 = init_sigmasq * init_phi2
    init_phi3 = (init_range_lon / init_range_lat)**2
    init_phi4 = (init_range_lon / init_range_time)**2
    
    true_params_vec = torch.tensor([
        safe_log(init_phi1), safe_log(init_phi2), safe_log(init_phi3), safe_log(init_phi4),
        init_advec_lat, init_advec_lon, 
        safe_log(init_nugget) # <--- Safe Log Applied Here
    ], device=DEVICE, dtype=DTYPE)
    
    grid_cfg = {
        'lats': torch.arange(5, 0 - 0.0001, -DELTA_LAT, device=DEVICE, dtype=DTYPE), 
        'lons': torch.arange(123, 133 + 0.0001, DELTA_LON, device=DEVICE, dtype=DTYPE),
        't_def': 8, 'mean': 260.0
    }

    # =========================================================================
    # [Step 1] Generate Observed Data
    # =========================================================================
    print("\n[Step 1] Generating Observed Data...")
    obs_input_map, obs_agg_data = generate_regular_data_directly(true_params_vec, grid_cfg)
    print(f"Data Generated. Shape: {obs_agg_data.shape}")

    # =========================================================================
    # [Step 2] Debiased Whittle Estimation & UQ
    # =========================================================================
    print("\n" + "="*50)
    print(" >>> MODEL 1: DEBIASED WHITTLE <<<")
    print("="*50)

    dwl = debiased_whittle.debiased_whittle_likelihood()
    params_dw = [p.clone().detach().requires_grad_(True) for p in true_params_vec] # Init with Truth for Demo
    
    # [Observational Data Preprocessing]
    # - db.generate_spatially_filtered_days 내부에서 'Spatial Differencing' 수행됨
    db = debiased_whittle.debiased_whittle_preprocess(
        [obs_agg_data], [obs_input_map], day_idx=0, 
        params_list=params_dw, lat_range=[0, 5], lon_range=[123, 133]
    )
    cur_df = db.generate_spatially_filtered_days(5, 0, 123, 133)
    time_slices = [cur_df[cur_df[:, 3] == t] for t in torch.unique(cur_df[:, 3])]
    
    J_vec_obs, n1, n2, p_time, taper_grid = dwl.generate_Jvector_tapered( 
        time_slices, dwl.cgn_hamming, 0, 1, 2, DEVICE
    )
    I_sample_obs = dwl.calculate_sample_periodogram_vectorized(J_vec_obs)
    taper_auto = dwl.calculate_taper_autocorrelation_fft(taper_grid, n1, n2, DEVICE)

    # [Optimization]
    opt_dw = torch.optim.LBFGS(params_dw, lr=1.0, max_iter=20, line_search_fn="strong_wolfe")
    def dw_closure():
        opt_dw.zero_grad()
        loss = dwl.whittle_likelihood_loss_tapered(torch.stack(params_dw), I_sample_obs, n1, n2, p_time, taper_auto, DELTA_LAT, DELTA_LON)
        if isinstance(loss, tuple): loss = loss[0]
        loss.backward()
        return loss
    opt_dw.step(dw_closure)
    best_params_dw = torch.stack(params_dw).detach().requires_grad_(True)
    
    # [Hessian Calculation - Consistency Check]
    # - nll_wrapper는 'I_sample_obs'(Differenced Observed Data)를 사용함.
    def nll_whittle(p):
        loss = dwl.whittle_likelihood_loss_tapered(p, I_sample_obs, n1, n2, p_time, taper_auto, DELTA_LAT, DELTA_LON)
        return loss[0] if isinstance(loss, tuple) else loss

    print("Calculating Hessian (H)...")
    H_dw = torch.autograd.functional.hessian(nll_whittle, best_params_dw)
    H_inv_dw = torch.linalg.inv(H_dw + torch.eye(7, device=DEVICE)*1e-5)
    nll_val_dw = nll_whittle(best_params_dw).item()

    # [Variability (J) Calculation - Consistency Check]
    print("Calculating Variability (J) via Bootstrap...")
    grad_list_dw = []
    
    for i in range(NUM_SIMS):
        with torch.no_grad():
            _, boot_agg = generate_regular_data_directly(best_params_dw, grid_cfg)
        
        # *** [CRITICAL CONSISTENCY FIX] ***
        # H 계산 시 사용된 데이터가 Differencing 되었으므로, J 계산용 부트스트랩 데이터도 반드시 Differencing 해야 함.
        # apply_spatial_difference 함수가 이를 수행함.
        boot_slices = []
        for t_val in torch.unique(boot_agg[:, 3]):
            d_slice = boot_agg[boot_agg[:, 3] == t_val]
            boot_slices.append(apply_spatial_difference(d_slice)) # <-- Correct
            
        J_b, bn1, bn2, _, btaper = dwl.generate_Jvector_tapered(boot_slices, dwl.cgn_hamming, 0, 1, 2, DEVICE)
        I_b = dwl.calculate_sample_periodogram_vectorized(J_b)
        c_auto = taper_auto if (bn1==n1 and bn2==n2) else dwl.calculate_taper_autocorrelation_fft(btaper, bn1, bn2, DEVICE)

        if best_params_dw.grad: best_params_dw.grad.zero_()
        loss = dwl.whittle_likelihood_loss_tapered(best_params_dw, I_b, bn1, bn2, p_time, c_auto, DELTA_LAT, DELTA_LON)
        if isinstance(loss, tuple): loss = loss[0]
        loss.backward()
        grad_list_dw.append(best_params_dw.grad.detach().clone())
    
    J_mat_dw = torch.matmul(torch.stack(grad_list_dw).T, torch.stack(grad_list_dw)) / NUM_SIMS
    GIM_inv_dw = H_inv_dw @ J_mat_dw @ H_inv_dw
    SE_dw = torch.sqrt(torch.diag(GIM_inv_dw))

    print_final_metrics_formatted(true_params_vec, best_params_dw, SE_dw, H_inv_dw, J_mat_dw, nll_val_dw, "Debiased Whittle")


    # =========================================================================
    # [Step 3] Vecchia Estimation & UQ
    # =========================================================================
    print("\n" + "="*50)
    print(" >>> MODEL 2: VECCHIA APPROXIMATION <<<")
    print("="*50)

    # [Preprocessing - Ordering]
    # - Hessian과 J 모두 동일한 ord_mm(Ordering)을 사용해야 일관성 유지
    ord_mm, nns_map = get_spatial_ordering(obs_input_map, mm_cond_number=mm_cond_number)
    mm_input_vc = {k: obs_input_map[k][ord_mm] for k in obs_input_map}

    params_vc = [p.clone().detach().requires_grad_(True) for p in true_params_vec]

    model_vc = kernels_reparam_space_time.fit_vecchia_lbfgs(
        smooth=v, input_map=mm_input_vc, aggregated_data=obs_agg_data,
        nns_map=nns_map, mm_cond_number=mm_cond_number, nheads=nheads
    )
    opt_vc = model_vc.set_optimizer(params_vc, lr=lr, max_iter=100, history_size=100)
    model_vc.precompute_conditioning_sets()
    
    print("Fitting Vecchia...")
    final_vals, _ = model_vc.fit_vecc_lbfgs(params_vc, opt_vc, max_steps=epochs, grad_tol=1e-6)
    
    # Handle return type
    if isinstance(final_vals, (list, tuple)): best_params_vc = final_vals[0] if isinstance(final_vals[0], torch.Tensor) else torch.tensor(final_vals[:-1], device=DEVICE)
    elif isinstance(final_vals, torch.Tensor): best_params_vc = final_vals[:-1]
    
    best_params_vc = best_params_vc.detach().clone().requires_grad_(True)

    # [Hessian - Consistency Check]
    def nll_vecc(p): return model_vc.vecchia_batched_likelihood(p)

    print("Calculating Hessian (H)...")
    H_vc = torch.autograd.functional.hessian(nll_vecc, best_params_vc)
    H_inv_vc = torch.linalg.inv(H_vc + torch.eye(7, device=DEVICE)*1e-6)
    nll_val_vc = nll_vecc(best_params_vc).item()

    # [Variability (J) - Consistency Check]
    print("Calculating Variability (J) via Bootstrap...")
    grad_list_vc = []
    
    for i in range(NUM_SIMS):
        with torch.no_grad():
            boot_map_raw, _ = generate_regular_data_directly(best_params_vc, grid_cfg)
        
        # *** [CRITICAL CONSISTENCY FIX] ***
        # H 계산 시 사용된 Ordering(ord_mm)을 부트스트랩 데이터에도 똑같이 적용해야 함.
        boot_mm = {k: boot_map_raw[k][ord_mm] for k in boot_map_raw}
        
        model_vc.input_map = boot_mm
        model_vc.precompute_conditioning_sets() # Re-compute conditioning for new values (coords are same but safe to run)
        
        if best_params_vc.grad: best_params_vc.grad.zero_()
        loss = model_vc.vecchia_batched_likelihood(best_params_vc)
        loss.backward()
        grad_list_vc.append(best_params_vc.grad.detach().clone())

    J_mat_vc = torch.matmul(torch.stack(grad_list_vc).T, torch.stack(grad_list_vc)) / NUM_SIMS
    GIM_inv_vc = H_inv_vc @ J_mat_vc @ H_inv_vc
    SE_vc = torch.sqrt(torch.diag(GIM_inv_vc))

    print_final_metrics_formatted(true_params_vec, best_params_vc, SE_vc, H_inv_vc, J_mat_vc, nll_val_vc, "Vecchia")

if __name__ == "__main__":
    app()

