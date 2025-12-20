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
from typing import Optional, List, Tuple
from pathlib import Path
import typer
import json
from json import JSONEncoder

# Custom imports (사용자 환경에 맞게 경로 확인 필요)
sys.path.append("/cache/home/jl2815/tco")

from GEMS_TCO import kernels_reparam_space_time_gpu as kernels_reparam_space_time
from GEMS_TCO import orderings as _orderings 
from GEMS_TCO import alg_optimization
from GEMS_TCO import configuration as config
from GEMS_TCO import debiased_whittle # Whittle 모듈

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
from typing import Optional, List, Tuple
from pathlib import Path
import typer
import json
from json import JSONEncoder

# Custom imports (사용자 환경에 맞게 경로 확인 필요)
sys.path.append("/cache/home/jl2815/tco")

from GEMS_TCO import kernels_reparam_space_time_gpu as kernels_reparam_space_time
from GEMS_TCO import orderings as _orderings 
from GEMS_TCO import alg_optimization
from GEMS_TCO import configuration as config
from GEMS_TCO import debiased_whittle # Whittle 모듈 (제공해주신 클래스가 여기에 있다고 가정)

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
from typing import Optional, List, Tuple
from pathlib import Path
import typer
import json
from json import JSONEncoder
from sklearn.neighbors import BallTree 

# Custom imports
sys.path.append("/cache/home/jl2815/tco")

from GEMS_TCO import kernels_reparam_space_time_gpu as kernels_reparam_space_time
from GEMS_TCO import orderings as _orderings 
from GEMS_TCO import alg_optimization
from GEMS_TCO import configuration as config
from GEMS_TCO import debiased_whittle 

# --- GLOBAL SETTINGS ---
# [Safety Check] 에러 발생 시 원인 파악을 위해 CPU 사용 권장. 
# 문제 해결 후 GPU 사용 시: DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

# --- 1. CORE SIMULATION FUNCTIONS ---

def get_model_covariance_on_grid(lags_x, lags_y, lags_t, params):
    # [수정] 수치 안정성을 위한 파라미터 클램핑
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
    dlat = float(lat_coords[1] - lat_coords[0]) if len(lat_coords) > 1 else 0.044
    dlon = float(lon_coords[1] - lon_coords[0]) if len(lon_coords) > 1 else 0.063
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

# --- 2. REGULAR DATA GENERATION (DIRECT) ---

def generate_regular_data_directly(params_tensor, grid_config):
    lats_sim = grid_config['lats']
    lons_sim = grid_config['lons']
    t_def = grid_config['t_def']
    ozone_mean = grid_config['mean']
    
    sim_field = generate_exact_gems_field(lats_sim, lons_sim, t_def, params_tensor)
    
    input_map = {}
    aggregated_list = []
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

# --- 3. HELPER & GIM ENGINE ---

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

def print_metrics_table(title, true_vec, est_vec, se_vec, param_names):
    true_v = true_vec.cpu().detach().numpy()
    est_v = est_vec.cpu().detach().numpy()
    se_v = se_vec.cpu().detach().numpy()

    mape = np.mean(np.abs((true_v - est_v) / true_v)) * 100

    print(f'\n{"="*105}')
    print(f" TABLE: {title} (MAPE: {mape:.4f}%)")
    print(f'{"="*105}')
    print(f"{'Param':<15} | {'True':<10} | {'Est':<10} | {'SE(GIM)':<10} | {'95% CI (Lower, Upper)':<30} | {'Covered?'}")
    print("-" * 105)
    
    covered_cnt = 0
    for i in range(len(param_names)):
        p = param_names[i]
        t, e, s = true_v[i], est_v[i], se_v[i]
        low, high = e - 1.96*s, e + 1.96*s
        covered = (t >= low) and (t <= high)
        if covered: covered_cnt += 1
        
        print(f"{p:<15} | {t:<10.4f} | {e:<10.4f} | {s:<10.4f} | ({low:.4f}, {high:.4f})     | {'YES' if covered else 'NO'}")
    print("-" * 105)
    print(f"Total Coverage: {covered_cnt}/{len(param_names)}")
    print(f'{"="*105}\n')

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

    # True Parameters
    init_sigmasq, init_range_lon, init_range_lat = 13.059, 0.195, 0.154 
    init_advec_lat, init_range_time, init_advec_lon, init_nugget = 0.0218, 1.0, -0.1689, 0.247

    init_phi2 = 1.0 / init_range_lon
    init_phi1 = init_sigmasq * init_phi2
    init_phi3 = (init_range_lon / init_range_lat)**2
    init_phi4 = (init_range_lon / init_range_time)**2
    
    true_params_vec = torch.tensor([
        np.log(init_phi1), np.log(init_phi2), np.log(init_phi3), np.log(init_phi4),
        init_advec_lat, init_advec_lon, np.log(init_nugget)
    ], device=DEVICE, dtype=DTYPE)
    
    PARAM_NAMES = ["log_phi1", "log_phi2", "log_phi3", "log_phi4", "advec_lat", "advec_lon", "log_nugget"]

    # Simulation Grid Config
    grid_cfg = {
        'lats': torch.arange(5.0, 0.0 - 0.0001, -DELTA_LAT, device=DEVICE, dtype=DTYPE), 
        'lons': torch.arange(123.0, 133.0 + 0.0001, DELTA_LON, device=DEVICE, dtype=DTYPE),
        't_def': 8,
        'mean': 260.0
    }

    # [Step 1] Generate Data (Regular)
    print("\n[Step 1] Generating Observed Data (Direct Regular Grid)...")
    obs_input_map, obs_agg_data = generate_regular_data_directly(true_params_vec, grid_cfg)
    print(f"Data Generated. Shape: {obs_agg_data.shape}")

    # [Step 2] Debiased Whittle
    print("\n" + "="*50)
    print(" >>> MODEL 1: DEBIASED WHITTLE <<<")
    print("="*50)

    dwl = debiased_whittle.debiased_whittle_likelihood()
    TAPERING_FUNC = dwl.cgn_hamming 
    
    daily_aggregated_tensors = [obs_agg_data]
    daily_hourly_maps = [obs_input_map]
    
    params_dw = [p.clone().detach().requires_grad_(True) for p in true_params_vec]
    
    db = debiased_whittle.debiased_whittle_preprocess(
        daily_aggregated_tensors, daily_hourly_maps, day_idx=0, 
        params_list=params_dw, lat_range=[0,5], lon_range=[123.0, 133.0]
    )
    cur_df = db.generate_spatially_filtered_days(0, 5, 123, 133)
    unique_times = torch.unique(cur_df[:, 3])
    time_slices_list = [cur_df[cur_df[:, 3] == t_val] for t_val in unique_times]
    
    J_vec_obs, n1, n2, p_time, taper_grid = dwl.generate_Jvector_tapered( 
        time_slices_list, tapering_func=TAPERING_FUNC, 
        lat_col=0, lon_col=1, val_col=2, device=DEVICE
    )
    I_sample_obs = dwl.calculate_sample_periodogram_vectorized(J_vec_obs)
    taper_autocorr_grid = dwl.calculate_taper_autocorrelation_fft(taper_grid, n1, n2, DEVICE)

    optimizer_dw = torch.optim.LBFGS(params_dw, lr=1.0, max_iter=20, line_search_fn="strong_wolfe")
    print("Fitting Whittle...")
    
    def dw_closure():
        optimizer_dw.zero_grad()
        # [수정] cat -> stack
        p_tensor = torch.stack(params_dw)
        
        # [수정] 현재 I_sample_obs의 크기에 맞춰 n1, n2 전달
        curr_n1, curr_n2 = I_sample_obs.shape[0], I_sample_obs.shape[1]
        
        nll = dwl.whittle_likelihood_loss_tapered(
            p_tensor, I_sample_obs, curr_n1, curr_n2, p_time, taper_autocorr_grid, DELTA_LAT, DELTA_LON
        )
        
        if isinstance(nll, tuple): nll = nll[0]
        nll.backward()
        return nll

    optimizer_dw.step(dw_closure)
    best_params_dw = torch.stack(params_dw).detach().requires_grad_(True)
    
    def nll_whittle_wrapper(p_tensor):
        curr_n1, curr_n2 = I_sample_obs.shape[0], I_sample_obs.shape[1]
        loss = dwl.whittle_likelihood_loss_tapered(
            p_tensor, I_sample_obs, curr_n1, curr_n2, p_time, taper_autocorr_grid, DELTA_LAT, DELTA_LON
        )
        if isinstance(loss, tuple): return loss[0]
        return loss

    print("Calculating Hessian (Whittle)...")
    H_dw = torch.autograd.functional.hessian(nll_whittle_wrapper, best_params_dw)
    H_inv_dw = torch.linalg.inv(H_dw + torch.eye(len(best_params_dw), device=DEVICE)*1e-5)

    print("Calculating Whittle Variability (J)...")
    grad_list_dw = []
    
    # 관측 데이터의 그리드 크기 저장
    obs_n1, obs_n2 = I_sample_obs.shape[0], I_sample_obs.shape[1]

    for i in range(NUM_SIMS):
        with torch.no_grad():
            _, boot_agg = generate_regular_data_directly(best_params_dw, grid_cfg)
        
        boot_slices = [boot_agg[boot_agg[:, 3] == t_val] for t_val in unique_times]
        J_vec_boot, boot_n1, boot_n2, _, boot_taper_grid = dwl.generate_Jvector_tapered(
             boot_slices, tapering_func=TAPERING_FUNC, lat_col=0, lon_col=1, val_col=2, device=DEVICE
        )
        I_sample_boot = dwl.calculate_sample_periodogram_vectorized(J_vec_boot)
        
        # [중요] 부트스트랩 샘플 크기가 다를 경우 taper grid 재계산 (Safety Check)
        if (boot_n1 != obs_n1) or (boot_n2 != obs_n2):
            current_taper_autocorr = dwl.calculate_taper_autocorrelation_fft(boot_taper_grid, boot_n1, boot_n2, DEVICE)
        else:
            current_taper_autocorr = taper_autocorr_grid

        if best_params_dw.grad is not None: best_params_dw.grad.zero_()
        
        loss = dwl.whittle_likelihood_loss_tapered(
            best_params_dw, I_sample_boot, boot_n1, boot_n2, p_time, current_taper_autocorr, DELTA_LAT, DELTA_LON
        )
        if isinstance(loss, tuple): loss = loss[0]
        
        loss.backward()
        grad_list_dw.append(best_params_dw.grad.detach().clone())
    
    grads_dw = torch.stack(grad_list_dw)
    J_mat_dw = torch.matmul(grads_dw.T, grads_dw) / NUM_SIMS
    GIM_inv_dw = H_inv_dw @ J_mat_dw @ H_inv_dw
    SE_dw = torch.sqrt(torch.diag(GIM_inv_dw))

    print_metrics_table("DEBIASED WHITTLE", true_params_vec, best_params_dw, SE_dw, PARAM_NAMES)

    # [Step 3] Vecchia Approximation
    print("\n" + "="*50)
    print(" >>> MODEL 2: VECCHIA APPROXIMATION <<<")
    print("="*50)

    ord_mm, nns_map = get_spatial_ordering(obs_input_map, mm_cond_number=mm_cond_number)
    mm_input_map_vecc = {}
    for key in obs_input_map:
        mm_input_map_vecc[key] = obs_input_map[key][ord_mm]

    params_vecc = [p.clone().detach().requires_grad_(True) for p in true_params_vec]

    model_vecc = kernels_reparam_space_time.fit_vecchia_lbfgs(
        smooth=v, input_map=mm_input_map_vecc, aggregated_data=obs_agg_data,
        nns_map=nns_map, mm_cond_number=mm_cond_number, nheads=nheads
    )
    optimizer_vecc = model_vecc.set_optimizer(params_vecc, lr=lr, max_iter=100, history_size=100)
    model_vecc.precompute_conditioning_sets()
    
    print("Fitting Vecchia...")
    final_vals, _ = model_vecc.fit_vecc_lbfgs(params_vecc, optimizer_vecc, max_steps=epochs, grad_tol=1e-6)
    best_params_vecc = torch.tensor(final_vals[:-1], device=DEVICE, dtype=DTYPE, requires_grad=True)

    def nll_vecc_wrapper(p_tensor):
        return model_vecc.vecchia_batched_likelihood(p_tensor)

    print("Calculating Hessian (Vecchia)...")
    H_vecc = torch.autograd.functional.hessian(nll_vecc_wrapper, best_params_vecc)
    H_inv_vecc = torch.linalg.inv(H_vecc + torch.eye(len(best_params_vecc), device=DEVICE)*1e-6)

    print("Calculating Vecchia Variability (J)...")
    grad_list_vecc = []
    
    for i in range(NUM_SIMS):
        with torch.no_grad():
            boot_map_raw, _ = generate_regular_data_directly(best_params_vecc, grid_cfg)
        
        boot_mm_map = {}
        for key in boot_map_raw:
            boot_mm_map[key] = boot_map_raw[key][ord_mm]
        
        model_vecc.input_map = boot_mm_map
        model_vecc.precompute_conditioning_sets()
        
        if best_params_vecc.grad is not None: best_params_vecc.grad.zero_()
        loss = model_vecc.vecchia_batched_likelihood(best_params_vecc)
        loss.backward()
        grad_list_vecc.append(best_params_vecc.grad.detach().clone())

    grads_vecc = torch.stack(grad_list_vecc)
    J_mat_vecc = torch.matmul(grads_vecc.T, grads_vecc) / NUM_SIMS
    GIM_inv_vecc = H_inv_vecc @ J_mat_vecc @ H_inv_vecc
    SE_vecc = torch.sqrt(torch.diag(GIM_inv_vecc))

    print_metrics_table("VECCHIA APPROX", true_params_vec, best_params_vecc, SE_vecc, PARAM_NAMES)

if __name__ == "__main__":
    app()

