# Standard libraries
import sys
import os
import time
import json
import copy
from pathlib import Path
import logging
from typing import Optional, List, Tuple

# 시나리오 2, 위치 오차 시뮬레이션을 위한 추가 라이브러리 근데 이거 다시 해야할듯
# sim_scenario2 or 3 반영해야할듯

# Data manipulation and analysis
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import typer
from scipy.interpolate import RegularGridInterpolator

# --- Custom Imports Path (사용자 환경에 맞게 수정 필수) ---
sys.path.append("/cache/home/jl2815/tco") 

# GEMS_TCO modules
from GEMS_TCO import kernels_reparam_space_time_cpu_010126 as kernels_reparam_space_time_cpu
from GEMS_TCO import configuration as config
from GEMS_TCO import alg_optimization, BaseLogger

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})

# ==============================================================================
# 1. Helper Functions
# ==============================================================================

def generate_exact_gems_field(lat_coords, lon_coords, t_steps, params, device, dtype):
    """FFT Simulation for Ground Truth Latent Field"""
    Nx = len(lat_coords)
    Ny = len(lon_coords)
    Nt = t_steps
    
    dlat = float(abs(lat_coords[1] - lat_coords[0]))
    dlon = float(abs(lon_coords[1] - lon_coords[0]))
    dt = 1.0 
    
    Px, Py, Pt = 2*Nx, 2*Ny, 2*Nt
    
    Lx_len = Px * dlat   
    lags_x = torch.arange(Px, device=device, dtype=dtype) * dlat
    lags_x[Px//2:] -= Lx_len 
    
    Ly_len = Py * dlon   
    lags_y = torch.arange(Py, device=device, dtype=dtype) * dlon
    lags_y[Py//2:] -= Ly_len

    Lt_len = Pt * dt     
    lags_t = torch.arange(Pt, device=device, dtype=dtype) * dt
    lags_t[Pt//2:] -= Lt_len

    L_x, L_y, L_t = torch.meshgrid(lags_x, lags_y, lags_t, indexing='ij')

    phi1, phi2 = torch.exp(params[0]), torch.exp(params[1])
    phi3, phi4 = torch.exp(params[2]), torch.exp(params[3])
    adv_lat, adv_lon = params[4], params[5]
    
    sigmasq = phi1 / phi2
    
    u_x = L_x - adv_lat * L_t
    u_y = L_y - adv_lon * L_t
    
    dist_sq = (u_x * torch.sqrt(phi3) * phi2)**2 + (u_y * phi2)**2 + (L_t * torch.sqrt(phi4) * phi2)**2
    dist = torch.sqrt(dist_sq + 1e-12)
    C_vals = sigmasq * torch.exp(-dist)

    S = torch.fft.fftn(C_vals)
    S.real = torch.clamp(S.real, min=0)

    random_phase = torch.fft.fftn(torch.randn(Px, Py, Pt, device=device, dtype=dtype))
    weighted_freq = torch.sqrt(S.real) * random_phase
    field_sim = torch.fft.ifftn(weighted_freq).real
    
    return field_sim[:Nx, :Ny, :Nt]

def calculate_rmsre(est_params_tensor, true_params_dict):
    est_t = est_params_tensor.detach().cpu().flatten()
    
    phi1_e, phi2_e = torch.exp(est_t[0]), torch.exp(est_t[1])
    phi3_e, phi4_e = torch.exp(est_t[2]), torch.exp(est_t[3])
    adv_lat_e, adv_lon_e = est_t[4], est_t[5]  
    nugget_e = torch.exp(est_t[6])

    sigmasq_e = phi1_e / phi2_e
    range_lon_e = 1.0 / phi2_e
    range_lat_e = range_lon_e / torch.sqrt(phi3_e)
    range_time_e = range_lon_e / torch.sqrt(phi4_e)

    est_array = torch.tensor([
        sigmasq_e, range_lat_e, range_lon_e, range_time_e, adv_lat_e, adv_lon_e, nugget_e
    ], dtype=torch.float64)
    
    true_array = torch.tensor([
        true_params_dict['sigmasq'], true_params_dict['range_lat'], true_params_dict['range_lon'],
        true_params_dict['range_time'], true_params_dict['advec_lat'], true_params_dict['advec_lon'], 
        true_params_dict['nugget']
    ], dtype=torch.float64)

    relative_error = (est_array - true_array) / true_array
    return torch.sqrt(torch.mean(relative_error ** 2)).item(), est_array.numpy()

def run_fl_optimization(model_instance, init_params_list, max_steps=100):
    optimizer = torch.optim.LBFGS(
        init_params_list, 
        lr=1.0, 
        max_iter=20,
        history_size=100,
        line_search_fn="strong_wolfe"
    )

    final_loss = None
    
    def closure():
        optimizer.zero_grad()
        params = torch.stack(init_params_list)
        loss = model_instance.compute_nll(params)
        loss.backward()
        return loss

    print("    Starting L-BFGS Optimization...")
    for i in range(max_steps):
        loss = optimizer.step(closure)
        final_loss = loss.item()
        if (i+1) % 10 == 0:
            print(f"    Step {i+1}/{max_steps} | Loss: {final_loss:.5f}")
            
    return final_loss, init_params_list

# ==============================================================================
# 2. Main Execution CLI
# ==============================================================================

@app.command()
def main(
    num_iters: int = typer.Option(5, help="Number of simulation iterations"),
    save_path: str = typer.Option(config.amarel_estimates_day_path, help="Path to save results")
):
    # --- A. Setup Environment ---
    DEVICE = torch.device("cpu") 
    DTYPE = torch.float64
    
    print(f"Running on {DEVICE} with {DTYPE}")
    
    output_path = Path(save_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Constants
    OZONE_MEAN = 260.0
    LOC_ERR_STD = 0.03 # Irregularity magnitude
    T_STEPS = 8

    # True Parameters
    true_params_dict = {
        'sigmasq': 13.059, 'range_lat': 0.154, 'range_lon': 0.195,
        'range_time': 1.0, 'advec_lat': 0.042, 'advec_lon': -0.1689, 'nugget': 0.247
    }

    # Parameter Transformation
    init_phi2 = 1.0 / true_params_dict['range_lon']
    init_phi1 = true_params_dict['sigmasq'] * init_phi2
    init_phi3 = (true_params_dict['range_lon'] / true_params_dict['range_lat'])**2
    init_phi4 = (true_params_dict['range_lon'] / true_params_dict['range_time'])**2

    true_params_raw = [
        np.log(init_phi1), np.log(init_phi2), np.log(init_phi3), np.log(init_phi4),
        true_params_dict['advec_lat'], true_params_dict['advec_lon'], np.log(true_params_dict['nugget'])
    ]
    true_params_tensor = [torch.tensor([v], device=DEVICE, dtype=DTYPE) for v in true_params_raw]

    # --- B. Grid Setup (Target Regular Domain) ---
    target_lat_start = -1.0 
    target_lat_end   = -3.0
    target_lon_start = 121.0
    target_lon_end   = 125.0
    step_lat = -0.044
    step_lon = 0.063

    lats = torch.arange(target_lat_start, target_lat_end - 0.0001, step_lat, device=DEVICE, dtype=DTYPE)
    lats = torch.round(lats * 10000) / 10000
    
    lons = torch.arange(target_lon_start, target_lon_end + 0.0001, step_lon, device=DEVICE, dtype=DTYPE)
    lons = torch.round(lons * 10000) / 10000
    
    # 격자 정보 (Binning의 기준이 됨)
    grid_lat, grid_lon = torch.meshgrid(lats, lons, indexing='ij')
    flat_lats_reg = grid_lat.flatten()
    flat_lons_reg = grid_lon.flatten()
    
    print(f"\n[Grid Info]")
    print(f"  Lat Step: {step_lat}, Lon Step: {step_lon}")
    print("-" * 60)

    results_summary = []
    seeds = [42 + i for i in range(num_iters)]

    # ==========================================================================
    # 3. Simulation Loop
    # ==========================================================================
    for i, seed in enumerate(seeds):
        iter_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Iteration {i+1}/{num_iters} | Seed: {seed}")
        print(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # ------------------------------------------------------------------
        # Step 1. Generate Latent Truth (FFT)
        # ------------------------------------------------------------------
        print("  Generating True Latent Field...")
        sim_field = generate_exact_gems_field(lats, lons, T_STEPS, true_params_tensor, DEVICE, DTYPE)
        
        # Setup Interpolator
        sim_field_np = sim_field.cpu().numpy()
        lats_np = lats.cpu().numpy()
        lons_np = lons.cpu().numpy()
        times_np = np.arange(T_STEPS, dtype=np.float64)
        
        interpolator = RegularGridInterpolator(
            (lats_np, lons_np, times_np), 
            sim_field_np, 
            method='linear', bounds_error=False, fill_value=None
        )

        # ------------------------------------------------------------------
        # Step 2. Create "Descendant 2" (Irregular Data = Original Observation)
        # ------------------------------------------------------------------
        print("  Creating Irregular Data (Original Observation)...")
        irr_map = {}
        irr_list = []
        nugget_std = torch.sqrt(torch.exp(true_params_tensor[6]))

        # 베이스가 되는 레귤러 좌표 (흔들기 전)
        base_lats = flat_lats_reg
        base_lons = flat_lons_reg

        for t in range(T_STEPS):
            # 2-1. 좌표 흔들기 (Perturbation)
            lat_noise = torch.randn_like(base_lats) * LOC_ERR_STD
            lon_noise = torch.randn_like(base_lons) * LOC_ERR_STD
            
            # 실제 관측된 이레귤러 좌표
            obs_lats = base_lats + lat_noise
            obs_lons = base_lons + lon_noise
            
            # 2-2. 해당 위치에서 참값 보간 (Interpolation)
            pts = np.stack([obs_lats.cpu().numpy(), obs_lons.cpu().numpy(), np.full(len(obs_lats), t)], axis=1)
            true_vals_at_irr = torch.from_numpy(interpolator(pts)).to(DEVICE, dtype=DTYPE)
            
            # 2-3. 관측 노이즈 추가
            obs_vals = true_vals_at_irr + (torch.randn_like(true_vals_at_irr) * nugget_std) + OZONE_MEAN
            
            # Tensor 생성 [Lat, Lon, Val, Time]
            flat_times = torch.full_like(obs_lats, float(t))
            irr_tensor = torch.stack([obs_lats, obs_lons, obs_vals, flat_times], dim=1).detach()
            
            irr_map[f'time_{t}'] = irr_tensor
            irr_list.append(irr_tensor)
            
        irr_aggregated = torch.cat(irr_list, dim=0)

        # ------------------------------------------------------------------
        # Step 3. Create "Descendant 3" (Regularized Data = Gridded via Binning)
        # ------------------------------------------------------------------
        print("  Creating Regularized Data (Forced Gridding)...")
        # Logic: 이레귤러 좌표를 가장 가까운 정규 격자점으로 반올림(Snapping) -> 평균
        
        reg_derived_map = {}
        reg_derived_list = []
        
        # Grid Parameters for snapping
        lat_min = lats[0].item()
        lat_step = step_lat
        lon_min = lons[0].item()
        lon_step = step_lon
        
        for t in range(T_STEPS):
            # 원본(Irregular) 데이터 가져오기
            irr_data_t = irr_map[f'time_{t}'] # [Lat, Lon, Val, Time]
            raw_lats = irr_data_t[:, 0]
            raw_lons = irr_data_t[:, 1]
            raw_vals = irr_data_t[:, 2]
            
            # 3-1. Snap to Grid (Nearest Neighbor Binning)
            # 인덱스로 변환 후 반올림
            lat_idx = torch.round((raw_lats - lat_min) / lat_step)
            lon_idx = torch.round((raw_lons - lon_min) / lon_step)
            
            # 다시 좌표로 복원 (Snap)
            snapped_lats = lat_min + lat_idx * lat_step
            snapped_lons = lon_min + lon_idx * lon_step
            
            # Rounding error 방지
            snapped_lats = torch.round(snapped_lats * 10000) / 10000
            snapped_lons = torch.round(snapped_lons * 10000) / 10000
            
            # 3-2. Grouping & Averaging (Pandas 활용이 빠름)
            df_temp = pd.DataFrame({
                'lat': snapped_lats.cpu().numpy(),
                'lon': snapped_lons.cpu().numpy(),
                'val': raw_vals.cpu().numpy()
            })
            
            # 같은 격자점에 떨어진 값들 평균 내기 (Spatial Averaging)
            df_grouped = df_temp.groupby(['lat', 'lon'], as_index=False)['val'].mean()
            
            # 텐서로 변환
            final_lats = torch.tensor(df_grouped['lat'].values, device=DEVICE, dtype=DTYPE)
            final_lons = torch.tensor(df_grouped['lon'].values, device=DEVICE, dtype=DTYPE)
            final_vals = torch.tensor(df_grouped['val'].values, device=DEVICE, dtype=DTYPE)
            final_times = torch.full_like(final_lats, float(t))
            
            reg_tensor = torch.stack([final_lats, final_lons, final_vals, final_times], dim=1).detach()
            
            reg_derived_map[f'time_{t}'] = reg_tensor
            reg_derived_list.append(reg_tensor)
            
        reg_derived_aggregated = torch.cat(reg_derived_list, dim=0)
        
        print(f"     >> Data Points: Irregular {len(irr_aggregated)} -> Regularized {len(reg_derived_aggregated)}")

        # ------------------------------------------------------------------
        # Step 4. Fit Model on Irregular Data (Original)
        # ------------------------------------------------------------------
        print("  -> [Fitting] Full Likelihood on ORIGINAL IRREGULAR data...")
        params_fit_irr = [torch.tensor([val], requires_grad=True, dtype=DTYPE, device=DEVICE) for val in true_params_raw]
        
        model_irr = kernels_reparam_space_time_cpu.FullLikelihoodCPU(
            smooth=0.5, input_map=irr_map, aggregated_data=irr_aggregated
        )
        
        try:
            nll_irr, final_params_irr = run_fl_optimization(model_irr, params_fit_irr)
            rmsre_irr, est_vals_irr = calculate_rmsre(torch.stack(final_params_irr), true_params_dict)
            print(f"     >> [Original Irregular] RMSRE: {rmsre_irr:.4f} | NLL: {nll_irr:.2f}")
        except Exception as e:
            print(f"     >> [Error] Irregular Fit Failed: {e}")
            nll_irr, rmsre_irr = np.nan, np.nan
            est_vals_irr = np.full(7, np.nan)

        # ------------------------------------------------------------------
        # Step 5. Fit Model on Regularized Data (Forced Grid)
        # ------------------------------------------------------------------
        print("  -> [Fitting] Full Likelihood on FORCED REGULARIZED data...")
        params_fit_reg = [torch.tensor([val], requires_grad=True, dtype=DTYPE, device=DEVICE) for val in true_params_raw]
        
        model_reg = kernels_reparam_space_time_cpu.FullLikelihoodCPU(
            smooth=0.5, input_map=reg_derived_map, aggregated_data=reg_derived_aggregated
        )
        
        try:
            nll_reg, final_params_reg = run_fl_optimization(model_reg, params_fit_reg)
            rmsre_reg, est_vals_reg = calculate_rmsre(torch.stack(final_params_reg), true_params_dict)
            print(f"     >> [Forced Regularized] RMSRE: {rmsre_reg:.4f} | NLL: {nll_reg:.2f}")
        except Exception as e:
            print(f"     >> [Error] Regularized Fit Failed: {e}")
            nll_reg, rmsre_reg = np.nan, np.nan
            est_vals_reg = np.full(7, np.nan)

        # ------------------------------------------------------------------
        # Step 6. Record & Save
        # ------------------------------------------------------------------
        # Note: 'irr' is now the reference, 'reg' is the derived one.
        record = {
            "iteration": i + 1, "seed": seed,
            "original_irr_rmsre": rmsre_irr, "original_irr_nll": nll_irr,
            "forced_reg_rmsre": rmsre_reg, "forced_reg_nll": nll_reg,
            "original_irr_params": est_vals_irr.tolist(),
            "forced_reg_params": est_vals_reg.tolist()
        }
        results_summary.append(record)
        
        timestamp = time.strftime("%Y%m%d_%H%M")
        json_path = output_path / f"sim_grid_bias_interm_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results_summary, f, indent=4)
            
        print(f"  Iteration {i+1} Finished. Duration: {time.time() - iter_start_time:.2f}s")

    final_csv_path = output_path / "final_simulation_grid_bias_comparison.csv"
    df = pd.DataFrame(results_summary)
    df.to_csv(final_csv_path, index=False)
    
    print(f"\nAll iterations completed. Results saved to {final_csv_path}")
    print(df[["iteration", "original_irr_rmsre", "forced_reg_rmsre"]])

if __name__ == "__main__":
    app()


