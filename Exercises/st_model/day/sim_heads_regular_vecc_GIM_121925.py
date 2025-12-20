# Standard libraries
import sys
# Add your custom path
sys.path.append("/cache/home/jl2815/tco")
import os
import logging
import argparse # Argument parsing

# Data manipulation and analysis
import pandas as pd
import numpy as np
import pickle
import torch
import torch.optim as optim
import copy                    # clone tensor
import time
import random
# Custom imports


from GEMS_TCO import kernels_reparam_space_time_gpu as kernels_reparam_space_time
from GEMS_TCO import orderings as _orderings 
from GEMS_TCO import alg_optimization, alg_opt_Encoder
from GEMS_TCO import kernels_gpu_st_simulation_column as kernels_gpu_st_simulation_column


from typing import Optional, List, Tuple
from pathlib import Path
import typer
import json
from json import JSONEncoder
from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data2, exact_location_filter
from GEMS_TCO import debiased_whittle
from torch.nn import Parameter
import torch
import torch.fft


# --- GLOBAL SETTINGS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

NUM_SIMS = 100


def set_seed(seed=42):
    """재현성을 위한 시드 고정 함수"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Info] Random Seed set to {seed}")

# --- MODEL HELPER FUNCTIONS ---
def get_model_covariance_on_grid(lags_x, lags_y, lags_t, params):
    phi1, phi2, phi3, phi4 = torch.exp(params[0]), torch.exp(params[1]), torch.exp(params[2]), torch.exp(params[3])
    advec_lat, advec_lon = params[4], params[5]
    sigmasq = phi1 / phi2

    u_lat_eff = lags_x - advec_lat * lags_t
    u_lon_eff = lags_y - advec_lon * lags_t
    
    dist_sq = (u_lat_eff.pow(2) * phi3) + (u_lon_eff.pow(2)) + (lags_t.pow(2) * phi4)
    distance = torch.sqrt(dist_sq + 1e-8)
    
    return sigmasq * torch.exp(-distance * phi2)

def generate_exact_gems_field(lat_coords, lon_coords, t_steps, params):
    Nx = len(lat_coords)
    Ny = len(lon_coords)
    Nt = t_steps
    
    # 1. Padding
    dlat = float(lat_coords[1] - lat_coords[0])
    dlon = float(lon_coords[1] - lon_coords[0])
    dt = 1.0 
    Px, Py, Pt = 2*Nx, 2*Ny, 2*Nt
    
    # 2. Lags
    Lx_len = Px * dlat   
    lags_x = torch.arange(Px, device=DEVICE, dtype=DTYPE) * dlat
    lags_x[Px//2:] -= Lx_len 
    
    Ly_len = Py * dlon   
    lags_y = torch.arange(Py, device=DEVICE, dtype=DTYPE) * dlon
    lags_y[Py//2:] -= Ly_len

    Lt_len = Pt * dt     
    lags_t = torch.arange(Pt, device=DEVICE, dtype=DTYPE) * dt
    lags_t[Pt//2:] -= Lt_len

    L_x, L_y, L_t = torch.meshgrid(lags_x, lags_y, lags_t, indexing='ij')
    C_vals = get_model_covariance_on_grid(L_x, L_y, L_t, params)

    # 3. FFT
    S = torch.fft.fftn(C_vals)
    S.real = torch.clamp(S.real, min=0)

    random_phase = torch.fft.fftn(torch.randn(Px, Py, Pt, device=DEVICE, dtype=DTYPE))
    weighted_freq = torch.sqrt(S.real) * random_phase
    field_sim = torch.fft.ifftn(weighted_freq).real
    
    return field_sim[:Nx, :Ny, :Nt]

def generate_synthetic_data(params_tensor, grid_config):
    lats_sim = grid_config['lats']
    lons_sim = grid_config['lons']
    t_def = grid_config['t_def']
    ozone_mean = grid_config['mean']
    
    # FFT Simulation
    sim_field = generate_exact_gems_field(lats_sim, lons_sim, t_def, params_tensor)
    
    input_map = {}
    aggregated_list = []
    nugget_std = torch.sqrt(torch.exp(params_tensor[6]))
    
    lats_flip = torch.flip(lats_sim, dims=[0])
    lons_flip = torch.flip(lons_sim, dims=[0])
    grid_lat, grid_lon = torch.meshgrid(lats_flip, lons_flip, indexing='ij')
    flat_lats = grid_lat.flatten()
    flat_lons = grid_lon.flatten()
    
    for t in range(t_def):
        field_t = sim_field[:, :, t] 
        field_t_flipped = torch.flip(field_t, dims=[0, 1]) 
        flat_vals = field_t_flipped.flatten()
        
        # Add Noise
        obs_vals = flat_vals + (torch.randn_like(flat_vals) * nugget_std) + ozone_mean
        
        time_val = 21.0 + t
        flat_times = torch.full_like(flat_lats, time_val)
        
        row_tensor = torch.stack([flat_lats, flat_lons, obs_vals, flat_times], dim=1).detach()
        key_str = f't_{t:02d}'
        input_map[key_str] = row_tensor
        aggregated_list.append(row_tensor)

    aggregated_data = torch.cat(aggregated_list, dim=0)
    return input_map, aggregated_data

def get_spatial_ordering(input_maps, mm_cond_number=10) -> Tuple[np.ndarray, np.ndarray]:
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

# --- MAIN CLI COMMAND ---
app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})
@app.command()

def cli(
    v: float = typer.Option(0.5, help="smooth"),
    lr: float = typer.Option(0.1, help="learning rate"),
    step: int = typer.Option(80, help="Number of iterations in optimization"),

    gamma_par: float = typer.Option(0.5, help="decreasing factor for learning rate"),
    space: List[str] = typer.Option(['20', '20'], help="spatial resolution"),
    days: List[str] = typer.Option(['0', '31'], help="Number of nearest neighbors in Vecchia approx."),
    
    mm_cond_number: int = typer.Option(8, help="Number of nearest neighbors in Vecchia approx."),
    params: List[str] = typer.Option(['20', '8.25', '5.25', '.2', '.2', '.05', '5'], help="Initial parameters"),
    ## mm-cond-number should be called in command line not mm_cond_number
    ## negative number can be a problem when parsing with typer
    epochs: int = typer.Option(120, help="Number of iterations in optimization"),
    nheads: int = typer.Option(300, help="Number of iterations in optimization"),
    keep_exact_loc: bool = typer.Option(True, help="whether to keep exact location data or not")
) -> None:
    
    SEED_VAL = 2025
    set_seed(SEED_VAL)
    print(f"Running on: {DEVICE}")

    # 1. Setup True Parameters
    init_sigmasq   = 13.059
    init_range_lon = 0.195 
    init_range_lat = 0.154 
    init_advec_lat = 0.0218
    init_range_time = 1.0
    init_advec_lon = -0.1689
    init_nugget    = 0.247

    # Reparameterization
    init_phi2 = 1.0 / init_range_lon
    init_phi1 = init_sigmasq * init_phi2
    init_phi3 = (init_range_lon / init_range_lat)**2
    init_phi4 = (init_range_lon / init_range_time)**2
    
    true_params_vec = torch.tensor([
        np.log(init_phi1), np.log(init_phi2), np.log(init_phi3), np.log(init_phi4),
        init_advec_lat, init_advec_lon, np.log(init_nugget)
    ], device=DEVICE, dtype=DTYPE)
    
    TRUE_VALUES = {
        "log_phi1": true_params_vec[0].item(),
        "log_phi2": true_params_vec[1].item(),
        "log_phi3": true_params_vec[2].item(),
        "log_phi4": true_params_vec[3].item(),
        "advec_lat": true_params_vec[4].item(),
        "advec_lon": true_params_vec[5].item(),
        "log_nugget": true_params_vec[6].item()
    }

    grid_cfg = {
        'lats': torch.arange(0, 5.0 + 0.001, 0.044, device=DEVICE, dtype=DTYPE),
        'lons': torch.arange(123.0, 133.0 + 0.001, 0.063, device=DEVICE, dtype=DTYPE),
        't_def': 8,
        'mean': 260.0
    }

    # 2. Generate Observed Data (Truth)
    print("\n[Step 1] Generating Observed Data from Truth...")
    obs_input_map, obs_agg_data = generate_synthetic_data(true_params_vec, grid_cfg)

    # 3. Spatial Ordering
    print("[Step 2] Calculating Spatial Ordering...")
    ord_mm, nns_map = get_spatial_ordering(obs_input_map, mm_cond_number=mm_cond_number)

    mm_input_map = {}
    for key in obs_input_map:
        mm_input_map[key] = obs_input_map[key][ord_mm]

    # 4. Initialize & Fit Model
    print("[Step 3] Initializing & Fitting Model...")
    
    # Start from True Params (or you can perturb them)
    params_list = [
        torch.tensor([val], requires_grad=True, dtype=DTYPE, device=DEVICE) 
        for val in true_params_vec.tolist()
    ]
    
    model = kernels_reparam_space_time.fit_vecchia_lbfgs(
        smooth=v,
        input_map=mm_input_map,
        aggregated_data=obs_agg_data,
        nns_map=nns_map,
        mm_cond_number=mm_cond_number,
        nheads=nheads
    )
    
    optimizer = model.set_optimizer(params_list, lr=1.0, max_iter=100, history_size=100)
    model.precompute_conditioning_sets() 
    
    start_t = time.time()
    final_params_raw, steps = model.fit_vecc_lbfgs(params_list, optimizer, max_steps=epochs, grad_tol=1e-7)
    print(f"Optimization done in {time.time() - start_t:.2f}s. Steps: {steps}")

    # 5. GIM Calculation & Coverage Check
    print(f'\n{"="*50}')
    print(f"--- STARTING GIM CALCULATION (Bootstrap J) ---")
    print(f'{"="*50}')

    best_params_tensor = torch.tensor(final_params_raw[:-1], device=DEVICE, dtype=DTYPE, requires_grad=True)

    # (A) Hessian (H) - from Observed Data
    print("1. Calculating Hessian (H)...")
    def nll_wrapper(p_tensor):
        return model.vecchia_batched_likelihood(p_tensor)

    H = torch.autograd.functional.hessian(nll_wrapper, best_params_tensor)
    
    try:
        H_inv = torch.linalg.inv(H)
        print("   Hessian Inverted Successfully.")
    except torch.linalg.LinAlgError:
        print("   [Warning] Hessian is singular. Adding jitter.")
        H_inv = torch.linalg.inv(H + torch.eye(H.shape[0], device=DEVICE)*1e-6)

    # (B) Variability (J) - from 100 Simulations (Gradient ONLY)
    
    print(f"2. Calculating Variability Matrix (J) with {NUM_SIMS} simulations...")
    
    grad_list = []
    
    for i in range(NUM_SIMS):
        # 1. Generate Bootstrap Data from Estimated Theta (NOT Truth)
        with torch.no_grad():
            boot_raw_map, _ = generate_synthetic_data(best_params_tensor, grid_cfg)
        
        # 2. Format & Inject
        boot_mm_map = {}
        for key in boot_raw_map:
            boot_mm_map[key] = boot_raw_map[key][ord_mm]
            
        model.input_map = boot_mm_map
        model.precompute_conditioning_sets() # Refresh internal tensors
        
        # 3. Calculate Gradient (NO Optimization step here)
        if best_params_tensor.grad is not None:
            best_params_tensor.grad.zero_()
            
        loss = model.vecchia_batched_likelihood(best_params_tensor)
        loss.backward()
        
        grad_list.append(best_params_tensor.grad.detach().cpu().clone())
        
        if (i+1) % 10 == 0:
            print(f"   Sim {i+1}/{NUM_SIMS} complete.")

    # Restore Observed Data
    model.input_map = mm_input_map
    model.precompute_conditioning_sets()

    # (C) Coverage Check
    grads = torch.stack(grad_list).to(DEVICE)
    J = torch.matmul(grads.T, grads) / NUM_SIMS
    GIM_inv = H_inv @ J @ H_inv 
    SE_GIM = torch.sqrt(torch.diag(GIM_inv))
    
    print(f'\n{"="*50}')
    print(f'heads {nheads}')
    print(f"--- COVERAGE ANALYSIS (95% CI) ---")
    print(f'{"="*50}')
    print(f"{'Param':<15} | {'True':<10} | {'Est':<10} | {'SE(GIM)':<10} | {'95% CI (Lower, Upper)':<30} | {'Covered?'}")
    print("-" * 105)
    
    param_names = list(TRUE_VALUES.keys())
    true_vals_list = true_params_vec.cpu().numpy()
    est_vals_list = best_params_tensor.detach().cpu().numpy()
    se_vals_list = SE_GIM.cpu().numpy()
    
    covered_count = 0
    for i in range(len(param_names)):
        p_name = param_names[i]
        true_v = true_vals_list[i]
        est_v = est_vals_list[i]
        se_v = se_vals_list[i]
        
        ci_lower = est_v - 1.96 * se_v
        ci_upper = est_v + 1.96 * se_v
        
        is_covered = (true_v >= ci_lower) and (true_v <= ci_upper)
        if is_covered: covered_count += 1
        cover_str = "YES" if is_covered else "NO"
        
        print(f"{p_name:<15} | {true_v:<10.4f} | {est_v:<10.4f} | {se_v:<10.4f} | ({ci_lower:.4f}, {ci_upper:.4f})     | {cover_str}")

    print("-" * 105)
    print(f"Total Coverage: {covered_count}/{len(param_names)}")


    # [1] Avg SE (GIM): 불확실성의 평균 크기 (작을수록 좋음)
    # SE_GIM은 이미 위에서 계산된 텐서입니다.
    avg_se = torch.mean(SE_GIM).item()
    
    # [2] Avg Bias (MAPE): 참값과 얼마나 틀렸는지 평균 % 오차 (작을수록 좋음)
    # (추정값 - 참값)의 절대값 비율 평균
    mape = torch.mean(torch.abs((true_params_vec - best_params_tensor) / true_params_vec)).item() * 100
    
    # [3] TIC (Takeuchi Information Criterion): 모델 적합도 (작을수록 좋음)
    # TIC = -2 * LogLikelihood + 2 * Trace(H_inv @ J)
    # H_inv와 J는 위에서 이미 계산했습니다.
    with torch.no_grad():
        nll = model.vecchia_batched_likelihood(best_params_tensor) # NLL (Negative Log Likelihood)
        penalty = torch.trace(H_inv @ J)
        tic = 2 * nll.item() + 2 * penalty.item()

    print(f"\n=== [FINAL MODEL PERFORMANCE METRICS] ===")
    print(f"1. Avg SE (GIM): {avg_se:.6f}  <-- 작을수록 정밀함")
    print(f"2. Avg Bias (%): {mape:.4f}%  <-- 작을수록 정확함")
    print(f"3. TIC Score   : {tic:.4f}    <-- 작을수록 적합함")
    print("=" * 50)


if __name__ == "__main__":
    app()



