# Standard libraries
import sys
import os
import time
import json
import copy
from pathlib import Path
import logging
from typing import Optional, List, Tuple

# Data manipulation and analysis
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import typer

# location error scenario에서 좌표 섭동을 위한 난수 생성에 사용

# --- Custom Imports Path (사용자 환경에 맞게 수정 필수) ---
sys.path.append("/cache/home/jl2815/tco") 

# GEMS_TCO modules
# [주의] 사용하시는 환경에 맞는 모듈명인지 확인해주세요. (cpu 버전 사용 권장)
from GEMS_TCO import kernels_reparam_space_time_cpu_010126 as kernels_reparam_space_time_cpu
from GEMS_TCO import configuration as config
from GEMS_TCO import alg_optimization, BaseLogger

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})

# ==============================================================================
# 1. Helper Functions
# ==============================================================================

def generate_exact_gems_field(lat_coords, lon_coords, t_steps, params, device, dtype):
    """
    정확한 GEMS 필드를 생성하는 함수.
    입력받은 좌표(lat_coords, lon_coords)의 크기에 맞춰 FFT 시뮬레이션을 수행합니다.
    """
    Nx = len(lat_coords)
    Ny = len(lon_coords)
    Nt = t_steps
    
    # 격자 간격 계산 (절대값 사용으로 방향 무관하게 처리)
    dlat = float(abs(lat_coords[1] - lat_coords[0]))
    dlon = float(abs(lon_coords[1] - lon_coords[0]))
    dt = 1.0 
    
    # 패딩을 2배로 주어 Circulant Embedding 수행
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

    # 파라미터 추출
    phi1, phi2 = torch.exp(params[0]), torch.exp(params[1])
    phi3, phi4 = torch.exp(params[2]), torch.exp(params[3])
    adv_lat, adv_lon = params[4], params[5]
    
    sigmasq = phi1 / phi2
    range_lon = 1.0 / phi2
    # range_lat, range_time은 커널 계산식에 내재됨
    
    # Spatiotemporal Distance with Advection (이류 효과 포함)
    u_x = L_x - adv_lat * L_t
    u_y = L_y - adv_lon * L_t
    
    # Matern Kernel (nu=0.5 -> Exponential) 계산
    # Scaling Factors: phi3(Lat ratio^2), phi4(Time ratio^2)
    dist_sq = (u_x * torch.sqrt(phi3) * phi2)**2 + (u_y * phi2)**2 + (L_t * torch.sqrt(phi4) * phi2)**2
    dist = torch.sqrt(dist_sq + 1e-12)
    C_vals = sigmasq * torch.exp(-dist)

    # FFT Synthesis
    S = torch.fft.fftn(C_vals)
    S.real = torch.clamp(S.real, min=0) # 음수 파워 제거 (수치 오차 방지)

    random_phase = torch.fft.fftn(torch.randn(Px, Py, Pt, device=device, dtype=dtype))
    weighted_freq = torch.sqrt(S.real) * random_phase
    field_sim = torch.fft.ifftn(weighted_freq).real
    
    # 원래 크기만큼 잘라서 반환
    return field_sim[:Nx, :Ny, :Nt]

def calculate_rmsre(est_params_tensor, true_params_dict):
    """
    추정된 파라미터와 참값 사이의 RMSRE(Root Mean Squared Relative Error)를 계산합니다.
    """
    est_t = est_params_tensor.detach().cpu().flatten()
    
    # 모델 파라미터(Log scale) -> 물리적 파라미터 변환
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

    # RMSRE 계산
    relative_error = (est_array - true_array) / true_array
    return torch.sqrt(torch.mean(relative_error ** 2)).item(), est_array.numpy()

def run_fl_optimization(model_instance, init_params_list, max_steps=100):
    """
    Full Likelihood 최적화를 수행하는 헬퍼 함수 (L-BFGS 사용)
    """
    optimizer = torch.optim.LBFGS(
        init_params_list, 
        lr=1.0, 
        max_iter=20,          # step 한 번당 최대 평가 횟수
        history_size=100,
        line_search_fn="strong_wolfe" # Full Likelihood에는 필수
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
        
        # 진행 상황 모니터링 (10 step마다 출력)
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
    # Full Likelihood의 거대 행렬 역산은 CPU MKL(BLAS)이 안정적이고 빠를 수 있음
    DEVICE = torch.device("cpu") 
    DTYPE = torch.float64 # 수치 안정성을 위해 Double Precision 사용
    
    print(f"Running on {DEVICE} with {DTYPE}")
    
    output_path = Path(save_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Constants
    OZONE_MEAN = 260.0
    LOC_ERR_STD = 0.03 # Irregular noise std (좌표 섭동 크기)
    T_STEPS = 8

    # True Parameters (Ground Truth)
    true_params_dict = {
        'sigmasq': 13.059, 'range_lat': 0.154, 'range_lon': 0.195,
        'range_time': 1.0, 'advec_lat': 0.042, 'advec_lon': -0.1689, 'nugget': 0.247
    }

    # Parameter Transformation for Model Input
    init_phi2 = 1.0 / true_params_dict['range_lon']
    init_phi1 = true_params_dict['sigmasq'] * init_phi2
    init_phi3 = (true_params_dict['range_lon'] / true_params_dict['range_lat'])**2
    init_phi4 = (true_params_dict['range_lon'] / true_params_dict['range_time'])**2

    # Raw tensors for generation (Log space for positivity constraint)
    true_params_raw = [
        np.log(init_phi1), np.log(init_phi2), np.log(init_phi3), np.log(init_phi4),
        true_params_dict['advec_lat'], true_params_dict['advec_lon'], np.log(true_params_dict['nugget'])
    ]
    true_params_tensor = [torch.tensor([v], device=DEVICE, dtype=DTYPE) for v in true_params_raw]

    # --- B. Grid Setup (Reduced Target Domain) ---
    # 요청하신 범위: Lat [-3, -1], Lon [121, 125]
    # Step이 음수(-0.044)이므로 Start는 큰 값(-1.0), End는 작은 값(-3.0)이어야 함
    target_lat_start = -1.0 
    target_lat_end   = -3.0
    target_lon_start = 121.0
    target_lon_end   = 125.0
    
    step_lat = -0.044
    step_lon = 0.063

    # Rounding fix included to match resolution exactly
    lats = torch.arange(target_lat_start, target_lat_end - 0.0001, step_lat, device=DEVICE, dtype=DTYPE)
    lats = torch.round(lats * 10000) / 10000
    
    lons = torch.arange(target_lon_start, target_lon_end + 0.0001, step_lon, device=DEVICE, dtype=DTYPE)
    lons = torch.round(lons * 10000) / 10000
    
    # 격자 생성
    grid_lat, grid_lon = torch.meshgrid(lats, lons, indexing='ij')
    flat_lats = grid_lat.flatten()
    flat_lons = grid_lon.flatten()
    
    N_space = len(flat_lats)
    N_total = N_space * T_STEPS
    
    print(f"\n[Grid Info]")
    print(f"  Lat Range: {lats[0].item()} ~ {lats[-1].item()} (Size: {len(lats)})")
    print(f"  Lon Range: {lons[0].item()} ~ {lons[-1].item()} (Size: {len(lons)})")
    print(f"  Spatial Points: {N_space}")
    print(f"  Total Data Points (x8 time steps): {N_total}")
    print(f"  Est. Memory for Cov Matrix: {N_total**2 * 8 / 1e9:.2f} GB")
    print("-" * 60)

    # Results Container
    results_summary = []

    # [중요] 일관된 데이터 생성을 위한 시드 리스트 (5회 반복)
    seeds = [42 + i for i in range(num_iters)]

    # ==========================================================================
    # 3. Simulation Loop
    # ==========================================================================
    for i, seed in enumerate(seeds):
        iter_start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Iteration {i+1}/{num_iters} | Seed: {seed}")
        print(f"{'='*60}")

        # ----------------------------------------------------------------------
        # Step 1: Data Generation (Consistent Seed)
        # ----------------------------------------------------------------------
        # 시드 고정: 매 반복마다 동일한 조건에서 시작
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # 1-1. True Field Generation (Latent)
        print("  Generating True Latent Field...")
        sim_field = generate_exact_gems_field(lats, lons, T_STEPS, true_params_tensor, DEVICE, DTYPE)
        
        # 1-2. Create Regular Dataset
        reg_map = {}
        reg_aggregated_list = []
        nugget_std = torch.sqrt(torch.exp(true_params_tensor[6]))

        for t in range(T_STEPS):
            field_t = sim_field[:, :, t]
            flat_vals = field_t.flatten()
            
            # Observation Noise 추가
            obs_vals = flat_vals + (torch.randn_like(flat_vals) * nugget_std) + OZONE_MEAN
            
            # Time column
            flat_times = torch.full_like(flat_lats, float(t))
            
            # Regular Tensor: [Lat, Lon, Val, Time]
            # detach()로 그래프 끊기 (중요)
            row_tensor = torch.stack([flat_lats, flat_lons, obs_vals, flat_times], dim=1).detach()
            
            key_str = f'time_{t}'
            reg_map[key_str] = row_tensor
            reg_aggregated_list.append(row_tensor)
            
        reg_aggregated_tensor = torch.cat(reg_aggregated_list, dim=0)

        # 1-3. Create Irregular Dataset (Perturbation)
        # 중요: Regular 데이터를 복사한 뒤 좌표만 흔듭니다. 값(Val)은 유지하여 비교 공정성 확보
        irreg_map = {}
        irreg_aggregated_list = []
        
        # 좌표 노이즈용 난수 (이 시점의 랜덤 상태 사용)
        for key, tensor in reg_map.items():
            # Copy tensor to avoid modifying original regular data
            irreg_tensor = tensor.clone()
            
            # Add location noise
            lat_noise = torch.randn_like(irreg_tensor[:, 0]) * LOC_ERR_STD
            lon_noise = torch.randn_like(irreg_tensor[:, 1]) * LOC_ERR_STD
            
            irreg_tensor[:, 0] += lat_noise
            irreg_tensor[:, 1] += lon_noise
            
            irreg_map[key] = irreg_tensor
            irreg_aggregated_list.append(irreg_tensor)
            
        irreg_aggregated_tensor = torch.cat(irreg_aggregated_list, dim=0)

        print("  Data Generation Complete.")

        # ----------------------------------------------------------------------
        # Step 2: Fit Full Likelihood (Regular)
        # ----------------------------------------------------------------------
        print("  -> [Fitting] Full Likelihood on REGULAR grid...")
        
        # Reset Init Params for fitting (매번 초기화)
        params_fit_reg = [
            torch.tensor([val], requires_grad=True, dtype=DTYPE, device=DEVICE)
            for val in true_params_raw
        ]
        
        # Model Instance
        model_reg = kernels_reparam_space_time_cpu.FullLikelihoodCPU(
            smooth=0.5,
            input_map=reg_map,
            aggregated_data=reg_aggregated_tensor
        )
        
        # Optimization
        try:
            nll_reg, final_params_reg = run_fl_optimization(model_reg, params_fit_reg)
            rmsre_reg, est_vals_reg = calculate_rmsre(torch.stack(final_params_reg), true_params_dict)
            print(f"     >> [Regular Result] RMSRE: {rmsre_reg:.4f} | NLL: {nll_reg:.2f}")
        except Exception as e:
            print(f"     >> [Regular Error] Optimization Failed: {e}")
            nll_reg, rmsre_reg = np.nan, np.nan
            est_vals_reg = np.full(7, np.nan)

        # ----------------------------------------------------------------------
        # Step 3: Fit Full Likelihood (Irregular)
        # ----------------------------------------------------------------------
        print("  -> [Fitting] Full Likelihood on IRREGULAR grid...")

        # Reset Init Params for fitting
        params_fit_irr = [
            torch.tensor([val], requires_grad=True, dtype=DTYPE, device=DEVICE)
            for val in true_params_raw
        ]
        
        # Model Instance
        model_irr = kernels_reparam_space_time_cpu.FullLikelihoodCPU(
            smooth=0.5,
            input_map=irreg_map,
            aggregated_data=irreg_aggregated_tensor
        )
        
        # Optimization
        try:
            nll_irr, final_params_irr = run_fl_optimization(model_irr, params_fit_irr)
            rmsre_irr, est_vals_irr = calculate_rmsre(torch.stack(final_params_irr), true_params_dict)
            print(f"     >> [Irregular Result] RMSRE: {rmsre_irr:.4f} | NLL: {nll_irr:.2f}")
        except Exception as e:
            print(f"     >> [Irregular Error] Optimization Failed: {e}")
            nll_irr, rmsre_irr = np.nan, np.nan
            est_vals_irr = np.full(7, np.nan)

        # ----------------------------------------------------------------------
        # Step 4: Record Results
        # ----------------------------------------------------------------------
        record = {
            "iteration": i + 1,
            "seed": seed,
            "reg_rmsre": rmsre_reg,
            "reg_nll": nll_reg,
            "irr_rmsre": rmsre_irr,
            "irr_nll": nll_irr,
            "reg_est_params": est_vals_reg.tolist(),
            "irr_est_params": est_vals_irr.tolist()
        }
        results_summary.append(record)
        
        # Save intermediate results (안전장치)
        timestamp = time.strftime("%Y%m%d_%H%M")
        # 파일명을 고정하지 않고 타임스탬프를 넣어 덮어쓰기 방지
        json_path = output_path / f"sim_fl_interm_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results_summary, f, indent=4)
            
        print(f"  Iteration {i+1} Finished. Duration: {time.time() - iter_start_time:.2f}s")

    # --- C. Final Save ---
    final_csv_path = output_path / "final_simulation_fl_comparison.csv"
    df = pd.DataFrame(results_summary)
    df.to_csv(final_csv_path, index=False)
    
    print(f"\nAll iterations completed. Results saved to {final_csv_path}")
    print(df[["iteration", "seed", "reg_rmsre", "irr_rmsre"]])

if __name__ == "__main__":
    app()



