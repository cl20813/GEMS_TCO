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
import cmath
import math
import gc

# Custom imports
sys.path.append("/cache/home/jl2815/tco")

from GEMS_TCO import kernels_reparam_space_time_gpu as kernels_reparam_space_time
from GEMS_TCO import orderings as _orderings 
from GEMS_TCO import alg_optimization
from GEMS_TCO import configuration as config
from GEMS_TCO import debiased_whittle 
from GEMS_TCO.data_loader import load_data2 

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

# --- HELPER: SPATIAL DIFFERENCING ---
def apply_spatial_difference(df_tensor):
    if df_tensor.size(0) == 0: return df_tensor
    u_lat = torch.unique(df_tensor[:, 0])
    u_lon = torch.unique(df_tensor[:, 1])
    n_lat, n_lon = len(u_lat), len(u_lon)
    if df_tensor.size(0) != n_lat * n_lon: return df_tensor

    vals = df_tensor[:, 2].view(1, 1, n_lat, n_lon)
    diff_kernel = torch.tensor([[[[-2., 1.], [ 1., 0.]]]], dtype=torch.float64, device=df_tensor.device)
    out_vals = F.conv2d(vals, diff_kernel, padding='valid')
    
    new_lats = u_lat[:-1]
    new_lons = u_lon[:-1]
    grid_lat, grid_lon = torch.meshgrid(new_lats, new_lons, indexing='ij')
    
    flat_lat, flat_lon, flat_val = grid_lat.flatten(), grid_lon.flatten(), out_vals.flatten()
    time_val = df_tensor[0, 3]
    flat_time = torch.full_like(flat_lat, time_val)
    
    return torch.stack([flat_lat, flat_lon, flat_val, flat_time], dim=1)

# --- CORE SIMULATION FUNCTIONS ---
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
    lags_x = torch.arange(Px, device=DEVICE, dtype=DTYPE) * dlat; lags_x[Px//2:] -= Lx_len 
    lags_y = torch.arange(Py, device=DEVICE, dtype=DTYPE) * dlon; lags_y[Py//2:] -= Ly_len
    lags_t = torch.arange(Pt, device=DEVICE, dtype=DTYPE) * dt;   lags_t[Pt//2:] -= Lt_len

    L_x, L_y, L_t = torch.meshgrid(lags_x, lags_y, lags_t, indexing='ij')
    C_vals = get_model_covariance_on_grid(L_x, L_y, L_t, params)

    S = torch.fft.fftn(C_vals); S.real = torch.clamp(S.real, min=0)
    random_phase = torch.fft.fftn(torch.randn(Px, Py, Pt, device=DEVICE, dtype=DTYPE))
    field_sim = torch.fft.ifftn(torch.sqrt(S.real) * random_phase).real
    return field_sim[:Nx, :Ny, :Nt]

def generate_regular_data_compatible(params_tensor, grid_config):
    lats_sim, lons_sim, t_def = grid_config['lats'], grid_config['lons'], grid_config['t_def']
    sim_field = generate_exact_gems_field(lats_sim, lons_sim, t_def, params_tensor)
    
    input_map, aggregated_list = {}, []
    nugget_std = torch.sqrt(torch.exp(params_tensor[6]))
    grid_lat, grid_lon = torch.meshgrid(lats_sim, lons_sim, indexing='ij')
    flat_lats, flat_lons = grid_lat.flatten(), grid_lon.flatten()
    
    for t in range(t_def):
        obs_vals = sim_field[:, :, t].flatten() + (torch.randn_like(flat_lats) * nugget_std) + grid_config['mean']
        row_tensor = torch.stack([flat_lats, flat_lons, obs_vals, torch.full_like(flat_lats, 21.0 + t)], dim=1).detach()
        input_map[f't_{t:02d}'] = row_tensor
        aggregated_list.append(row_tensor)
    return input_map, torch.cat(aggregated_list, dim=0)

def transform_raw_to_model_params(raw_params: list) -> list:
    sigma_sq, range_lat, range_lon, range_time, advec_lat, advec_lon, nugget = raw_params
    phi2 = 1.0 / max(range_lon, 1e-4)
    phi1 = sigma_sq * phi2
    phi3 = (range_lon / max(range_lat, 1e-4))**2
    phi4 = (range_lon / max(range_time, 1e-4))**2
    return [math.log(phi1), math.log(phi2), math.log(phi3), math.log(phi4), advec_lat, advec_lon, math.log(max(nugget, 1e-8))]

def transform_model_to_physical_tensor(model_params: torch.Tensor) -> torch.Tensor:
    phi1 = torch.exp(model_params[0])
    phi2 = torch.exp(model_params[1])
    phi3 = torch.exp(model_params[2])
    phi4 = torch.exp(model_params[3])
    advec_lat = model_params[4]
    advec_lon = model_params[5]
    nugget = torch.exp(model_params[6])
    range_lon = 1.0 / phi2
    sigma_sq = phi1 / phi2
    range_lat = range_lon / torch.sqrt(phi3)
    range_time = range_lon / torch.sqrt(phi4)
    return torch.stack([sigma_sq, range_lat, range_lon, range_time, advec_lat, advec_lon, nugget])

# --- MAIN CLI COMMAND ---
app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})

# ... (앞부분 import 및 설정은 동일) ...

@app.command()
def cli(
    start_day: int = typer.Option(..., help="Start day (e.g., 1)"),
    end_day: int = typer.Option(..., help="End day (e.g., 5)"),
    v: float = typer.Option(0.5, help="smooth"),
    mm_cond_number: int = typer.Option(8, help="Neighbors"),
    nheads: int = typer.Option(300, help="Heads"),
    keep_exact_loc: bool = typer.Option(True, help="Use exact locations")
) -> None:
    
    set_seed(2025)
    print(f"Running on: {DEVICE}")
    print(f"Processing Batch: Day {start_day} to Day {end_day}")

    # =========================================================
    # 1. INITIAL SETUP (메타데이터만 로드)
    # =========================================================
    print("\n[1] Initializing Data Loader...")
    # 해상도가 [1, 1]이면 데이터가 작지만, [20, 20] 등 고해상도라면 엄청 큽니다.
    lat_lon_resolution = [1, 1] 
    years = ['2024']; month_range =[7]
    data_load_instance = load_data2(config.amarel_data_load_path)

    # 전체 Raw Data Map은 한 번 로드해야 합니다 (이건 어쩔 수 없지만, working data보단 작음)
    df_map, ord_mm, nns_map = data_load_instance.load_maxmin_ordered_data_bymonthyear(
        lat_lon_resolution=lat_lon_resolution, mm_cond_number=mm_cond_number,
        years_=years, months_=month_range, lat_range=[-3, 2], lon_range=[121, 131] 
    )
    
    # [수정됨] 여기서 daily_data_list 같은 리스트를 만들지 않습니다!
    print(f"Metadata Loaded. Starting Day-by-Day Processing.")

    # =========================================================
    # 2. PARAMETER LOADING
    # =========================================================
    path = Path("/cache/home/jl2815/tco/exercise_output/estimates/day/real_fit_dw_and_vecc_july24")
    vecc_path = path / 'real_vecc_july24_h1000_mm16.csv'
    dw_path = path / 'real_dw_july24.csv'
    dw_real_df = pd.read_csv(dw_path).iloc[:28, 4:(4+7)]
    vecc_real_df = pd.read_csv(vecc_path).iloc[:28, 4:(4+7)]

    # =========================================================
    # 3. MAIN LOOP (Load -> Process -> Delete)
    # =========================================================
    results = []
    target_range = range(start_day - 1, end_day)
    
    for day in target_range:
        if day >= 28: 
            print(f"Day {day+1} is out of range. Stopping.")
            break

        # [MEMORY] 시작 전 청소
        gc.collect(); torch.cuda.empty_cache()
        print(f"\n>>> Processing Day {day+1} ...")

        # ---------------------------------------------------------
        # [NEW] 여기서 데이터를 로드합니다 (Just-In-Time)
        # ---------------------------------------------------------
        hour_indices = [day * 8, (day + 1) * 8]
        
        # 1. Whittle 데이터 로드 & GPU 이동
        print("   -> Loading Data for current day...")
        dw_map_tmp, dw_agg_tmp = data_load_instance.load_working_data(
            df_map, hour_indices, ord_mm=None, dtype=DTYPE, keep_ori=keep_exact_loc
        )
        real_agg_dw = dw_agg_tmp.to(DEVICE) # 바로 GPU로
        
        # dw_map_tmp는 Whittle에서 안쓰면 바로 삭제 (메모리 절약)
        del dw_map_tmp, dw_agg_tmp

        # ---------------------------------------------------------
        # [MODEL 1] Whittle SE
        # ---------------------------------------------------------
        print("   [1/2] Whittle SE...")
        
        # Grid Config 생성
        grid_cfg_day = {
            'lats': torch.unique(real_agg_dw[:, 0]), 'lons': torch.unique(real_agg_dw[:, 1]),
            't_def': len(torch.unique(real_agg_dw[:, 3])), 'mean': 260.0
        }

        dw_params_val = torch.tensor(transform_raw_to_model_params(dw_real_df.iloc[day].tolist()), device=DEVICE, dtype=DTYPE)
        dw_params_grad = dw_params_val.clone().detach().requires_grad_(True)
        dwl = debiased_whittle.debiased_whittle_likelihood()
        
        # Real Data Diff
        cur_df = real_agg_dw  
        time_slices_obs = []
        for t_val in torch.unique(cur_df[:, 3]):
            day_slice_diff = apply_spatial_difference(cur_df[cur_df[:, 3] == t_val]) 
            time_slices_obs.append(day_slice_diff)

        J_vec_obs, n1, n2, p_time, taper_grid = dwl.generate_Jvector_tapered(time_slices_obs, dwl.cgn_hamming, 0, 1, 2, DEVICE)
        I_sample_obs = dwl.calculate_sample_periodogram_vectorized(J_vec_obs)
        taper_auto = dwl.calculate_taper_autocorrelation_fft(taper_grid, n1, n2, DEVICE)

        def nll_dw_fn(p):
            loss = dwl.whittle_likelihood_loss_tapered(p, I_sample_obs, n1, n2, p_time, taper_auto, DELTA_LAT, DELTA_LON)
            return loss[0] if isinstance(loss, tuple) else loss

        try:
            H_dw = torch.autograd.functional.hessian(nll_dw_fn, dw_params_grad)
            H_inv_dw = torch.linalg.inv(H_dw + torch.eye(7, device=DEVICE)*1e-5)
        except Exception:
            torch.cuda.empty_cache()
            H_dw = torch.autograd.functional.hessian(nll_dw_fn, dw_params_grad)
            H_inv_dw = torch.linalg.inv(H_dw + torch.eye(7, device=DEVICE)*1e-5)
        
        del H_dw, J_vec_obs, I_sample_obs
        
        grad_list_dw = []
        for i in range(NUM_SIMS):
            with torch.no_grad(): _, boot_agg = generate_regular_data_compatible(dw_params_grad, grid_cfg_day)
            boot_slices = [apply_spatial_difference(boot_agg[boot_agg[:, 3] == t]) for t in torch.unique(boot_agg[:, 3])]
            J_b, bn1, bn2, _, btaper = dwl.generate_Jvector_tapered(boot_slices, dwl.cgn_hamming, 0, 1, 2, DEVICE)
            I_b = dwl.calculate_sample_periodogram_vectorized(J_b)
            c_auto = taper_auto if (bn1==n1 and bn2==n2) else dwl.calculate_taper_autocorrelation_fft(btaper, bn1, bn2, DEVICE)
            
            if dw_params_grad.grad is not None: dw_params_grad.grad.zero_()
            loss = dwl.whittle_likelihood_loss_tapered(dw_params_grad, I_b, bn1, bn2, p_time, c_auto, DELTA_LAT, DELTA_LON)
            if isinstance(loss, tuple): loss = loss[0]
            loss.backward()
            grad_list_dw.append(dw_params_grad.grad.detach().clone())
            del boot_agg, boot_slices, J_b, I_b, loss

        J_mat_dw = torch.matmul(torch.stack(grad_list_dw).T, torch.stack(grad_list_dw)) / NUM_SIMS
        Cov_dw = H_inv_dw @ J_mat_dw @ H_inv_dw
        Jacobian_dw = torch.autograd.functional.jacobian(transform_model_to_physical_tensor, dw_params_val)
        SE_phys_dw = torch.sqrt(torch.diag(Jacobian_dw @ Cov_dw @ Jacobian_dw.T)).detach().cpu().numpy()
        Pt_phys_dw = transform_model_to_physical_tensor(dw_params_val).detach().cpu().numpy()

        # [MEMORY] Whittle 끝났으니 데이터 삭제
        del grad_list_dw, J_mat_dw, Cov_dw, dwl, cur_df, real_agg_dw
        gc.collect(); torch.cuda.empty_cache()

        # ---------------------------------------------------------
        # [MODEL 2] Vecchia SE
        # ---------------------------------------------------------
        print("   [2/2] Vecchia SE...")
        
        # 2. Vecchia 데이터 로드 (필요할 때 로드)
        vecc_map_tmp, vecc_agg_tmp = data_load_instance.load_working_data(
            df_map, hour_indices, ord_mm=ord_mm, dtype=DTYPE, keep_ori=keep_exact_loc
        )
        real_agg_vecc = vecc_agg_tmp.to(DEVICE)
        real_map_vecc = {k: v.to(DEVICE) for k, v in vecc_map_tmp.items()}
        del vecc_map_tmp, vecc_agg_tmp  # CPU 원본 삭제

        vecc_params_val = torch.tensor(transform_raw_to_model_params(vecc_real_df.iloc[day].tolist()), device=DEVICE, dtype=DTYPE)
        vecc_params_grad = vecc_params_val.clone().detach().requires_grad_(True)
        
        model_vc = kernels_reparam_space_time.fit_vecchia_lbfgs(
            smooth=v, input_map=real_map_vecc, aggregated_data=real_agg_vecc,
            nns_map=nns_map, mm_cond_number=mm_cond_number, nheads=nheads
        )
        model_vc.precompute_conditioning_sets()
        def nll_vecc_fn(p): return model_vc.vecchia_batched_likelihood(p)

        try:
            H_vc = torch.autograd.functional.hessian(nll_vecc_fn, vecc_params_grad)
            H_inv_vc = torch.linalg.inv(H_vc + torch.eye(7, device=DEVICE)*1e-5)
        except Exception:
            torch.cuda.empty_cache()
            H_vc = torch.autograd.functional.hessian(nll_vecc_fn, vecc_params_grad)
            H_inv_vc = torch.linalg.inv(H_vc + torch.eye(7, device=DEVICE)*1e-5)
        del H_vc
        
        grad_list_vc = []
        for i in range(NUM_SIMS):
            with torch.no_grad(): boot_map, _ = generate_regular_data_compatible(vecc_params_grad, grid_cfg_day)
            boot_mm = {k: boot_map[k][ord_mm] for k in boot_map}
            model_vc.input_map = boot_mm
            model_vc.precompute_conditioning_sets()
            if vecc_params_grad.grad is not None: vecc_params_grad.grad.zero_()
            loss = model_vc.vecchia_batched_likelihood(vecc_params_grad)
            loss.backward()
            grad_list_vc.append(vecc_params_grad.grad.detach().clone())
            del boot_map, boot_mm, loss

        J_mat_vc = torch.matmul(torch.stack(grad_list_vc).T, torch.stack(grad_list_vc)) / NUM_SIMS
        Cov_vc = H_inv_vc @ J_mat_vc @ H_inv_vc
        Jacobian_vc = torch.autograd.functional.jacobian(transform_model_to_physical_tensor, vecc_params_val)
        SE_phys_vc = torch.sqrt(torch.diag(Jacobian_vc @ Cov_vc @ Jacobian_vc.T)).detach().cpu().numpy()
        Pt_phys_vc = transform_model_to_physical_tensor(vecc_params_val).detach().cpu().numpy()

        row = {
            "Day": day + 1,
            "DW_Est_SigmaSq": Pt_phys_dw[0], "DW_SE_SigmaSq": SE_phys_dw[0],
            # ... (나머지 저장 코드 동일) ...
            "VC_Est_Nugget": Pt_phys_vc[6],   "VC_SE_Nugget": SE_phys_vc[6],
        }
        results.append(row)
        
        # [MEMORY] Vecchia 데이터 및 모델 삭제
        del real_agg_vecc, real_map_vecc, model_vc, H_inv_vc, grad_list_vc, J_mat_vc, Cov_vc
        gc.collect(); torch.cuda.empty_cache()

    df_res = pd.DataFrame(results)
    out_file = f"se_results_days_{start_day}_to_{end_day}.csv"
    df_res.to_csv(out_file, index=False)
    print(f"\n[Success] Saved to {out_file}")

if __name__ == "__main__":
    app()