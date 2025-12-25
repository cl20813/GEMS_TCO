# Standard libraries
import sys
import os
import logging
import argparse 
import time
import copy
import json
from pathlib import Path
from typing import Optional, List, Tuple

# Data manipulation and analysis
import pandas as pd
import numpy as np
import pickle
import torch
import torch.optim as optim
import typer
from datetime import datetime  # <--- 이 줄을 추가하세요

# --- Custom Imports ---
sys.path.append("/cache/home/jl2815/tco") 

from GEMS_TCO import kernels_reparam_space_time_gpu as kernels_reparam_space_time
from GEMS_TCO import orderings as _orderings 
from GEMS_TCO import alg_optimization, BaseLogger
from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data2

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})

@app.command()
def cli(
    v: float = typer.Option(0.5, help="smooth"),
    lr: float = typer.Option(0.1, help="learning rate"), # Not used in LBFGS direct calling but kept for compatibility
    space: List[str] = typer.Option(['20', '20'], help="spatial resolution"),
    days: List[str] = typer.Option(['0', '31'], help="Start and End day index (0-30)"), 
    mm_cond_number: int = typer.Option(8, help="Number of nearest neighbors in Vecchia approx."),
    params: List[str] = typer.Option(['20', '8.25', '5.25', '.2', '.2', '.05', '5'], help="Initial parameters"),
    nheads: int = typer.Option(300, help="Number of iterations in optimization"),
    keep_exact_loc: bool = typer.Option(True, help="whether to keep exact location data or not")
) -> None:

    # 1. 설정 파싱
    lat_lon_resolution = [int(s) for s in space[0].split(',')]
    days_s_e = list(map(int, days[0].split(',')))
    days_list = list(range(days_s_e[0], days_s_e[1]))

    # Device 설정 (Vecchia는 GPU 권장)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    print(f"Target Days: {days_list}")

    # 2. 경로 및 데이터 로더 설정
    years = ['2024']
    month_range =[7]

    output_path = Path(config.amarel_estimates_day_path)
    output_path.mkdir(parents=True, exist_ok=True) 
    
    data_load_instance = load_data2(config.amarel_data_load_path)

    print("\nLoading MaxMin Ordered Data (Metadata)...")
    df_map, ord_mm, nns_map = data_load_instance.load_maxmin_ordered_data_bymonthyear(
        lat_lon_resolution=lat_lon_resolution, 
        mm_cond_number=mm_cond_number,
        years_=years, 
        months_=month_range,
        lat_range=[-3, 2],      
        lon_range=[121, 131] 
    )

    print("Pre-loading daily tensors (Vecchia)...")
    daily_aggregated_tensors_vecc = [] 
    daily_hourly_maps_vecc = []   

    for day_index in range(31):
        hour_start_index = day_index * 8
        hour_end_index = (day_index + 1) * 8
        hour_indices = [hour_start_index, hour_end_index]
        
        # Vecchia용 데이터 (MaxMin Ordered)
        day_hourly_map, day_aggregated_tensor = data_load_instance.load_working_data(
            df_map, 
            hour_indices, 
            ord_mm=ord_mm,  # Vecchia needs ordering
            dtype=torch.float64, 
            keep_ori=keep_exact_loc
        )
        daily_aggregated_tensors_vecc.append(day_aggregated_tensor)
        daily_hourly_maps_vecc.append(day_hourly_map)

    # Global L-BFGS Settings
    LBFGS_LR = 1.0
    LBFGS_MAX_STEPS = 10      
    LBFGS_HISTORY_SIZE = 100   
    LBFGS_MAX_EVAL = 100       

    # --- Main Loop ---
    for day_idx in days_list:
        print(f'\n{"="*50}')
        print(f'--- Processing Real Data (Vecchia Only): Day {day_idx+1} ---')
        print(f'{"="*50}')

        try:
            # Data Prepare
            daily_hourly_map_vecc = daily_hourly_maps_vecc[day_idx]
            daily_aggregated_tensor_vecc = daily_aggregated_tensors_vecc[day_idx].to(DEVICE)

            if daily_aggregated_tensor_vecc.shape[0] == 0:
                print(f"Skipping Day {day_idx+1}: No data.")
                continue

            # --- Parameter Initialization ---
            # (Whittle 없이 바로 초기값 사용)
            init_sigmasq   = 13.059
            init_range_lat = 0.154 
            init_range_lon = 0.195
            init_range_time = 1.0
            init_advec_lat = 0.0218
            init_advec_lon = -0.1689
            init_nugget    = 0.247
            
            # Map to 'phi' reparameterization
            init_phi2 = 1.0 / init_range_lon
            init_phi1 = init_sigmasq * init_phi2
            init_phi3 = (init_range_lon / init_range_lat)**2
            init_phi4 = (init_range_lon / init_range_time)**2

            initial_vals = [np.log(init_phi1), np.log(init_phi2), np.log(init_phi3), 
                            np.log(init_phi4), init_advec_lat, init_advec_lon, np.log(init_nugget)]

            # Create Params Tensor on GPU
            params_list_vecc = [
                torch.tensor([val], requires_grad=True, dtype=torch.float64, device=DEVICE)
                for val in initial_vals
            ]

            # -------------------------------------------------------
            # STEP: Vecchia Optimization
            # -------------------------------------------------------
            print(f"\n--- Starting Vecchia Optimization (mm={mm_cond_number}) ---")
            
            # Instantiate Model
            model_instance = kernels_reparam_space_time.fit_vecchia_lbfgs(
                smooth=v,
                input_map=daily_hourly_map_vecc,
                aggregated_data=daily_aggregated_tensor_vecc,
                nns_map=nns_map,
                mm_cond_number=mm_cond_number,
                nheads=nheads
            )

            # Set Optimizer
            optimizer_vecc = model_instance.set_optimizer(
                params_list_vecc,     
                lr=LBFGS_LR,            
                max_iter=LBFGS_MAX_EVAL,        
                history_size=LBFGS_HISTORY_SIZE 
            )

            start_time = time.time()
            
            # Run Optimization
            out, steps_ran = model_instance.fit_vecc_lbfgs(
                params_list_vecc,
                optimizer_vecc,
                max_steps=LBFGS_MAX_STEPS, 
                grad_tol=1e-7
            )

            epoch_time = time.time() - start_time
            print(f"Vecchia Optimization finished in {epoch_time:.2f}s. Results: {out}")

            # -------------------------------------------------------
            # Save Results (Append Mode)
            # -------------------------------------------------------
            grid_res = int(daily_aggregated_tensor_vecc.shape[0] / 8)
            
            # alg_optimization 객체 생성
            res = alg_optimization(
                day=f"{years[0]}-07-{day_idx+1}", 
                cov_name=f"Vecc_Only_mm{mm_cond_number}", 
                space_size=grid_res, 
                lr=LBFGS_LR, 
                params=out, 
                time=epoch_time, 
                rmsre=0.0 # Real data
            )
            
            date = datetime.now().strftime("%m%d%y")

            # 파일명 설정 (real_vecc_summary...)
            common_filename = f"real_vecc_summary_mm{mm_cond_number}_{date}"
            
            # 1. JSON 저장
            json_filepath = output_path / f"{common_filename}.json"
            current_data = BaseLogger.load_list(json_filepath)
            current_data.append(res.__dict__)
            
            with json_filepath.open('w', encoding='utf-8') as f:
                json.dump(current_data, f, separators=(",", ":"), indent=4)
            
            # 2. CSV 저장
            csv_filepath = output_path / f"{common_filename}.csv"
            pd.DataFrame(current_data).to_csv(csv_filepath, index=False)
            
            print(f"✅ Day {day_idx+1} Saved to {common_filename}.[json/csv]")

        except Exception as e:
            print(f"🔴 Day {day_idx+1} Failed: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    app()



