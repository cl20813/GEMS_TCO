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
from GEMS_TCO import debiased_whittle_gpu as debiased_whittle

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})

@app.command()
def cli(
    v: float = typer.Option(0.5, help="smooth"),
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

    # Device 설정 (Whittle=CPU, Vecchia=GPU)
    DEVICE_DW = torch.device("cpu")
    DEVICE_VECC = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Configurations:")
    print(f" - Whittle Device: {DEVICE_DW}")
    print(f" - Vecchia Device: {DEVICE_VECC}")
    print(f" - Target Days: {days_list}")

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

    print("Pre-loading daily tensors...")
    daily_aggregated_tensors_dw = [] 
    daily_hourly_maps_dw = []      

    for day_index in range(31):
        hour_start_index = day_index * 8
        hour_end_index = (day_index + 1) * 8
        hour_indices = [hour_start_index, hour_end_index]
        
        day_hourly_map, day_aggregated_tensor = data_load_instance.load_working_data(
            df_map, 
            hour_indices, 
            ord_mm=None, 
            dtype=torch.float64, 
            keep_ori=keep_exact_loc
        )
        daily_aggregated_tensors_dw.append(day_aggregated_tensor)
        daily_hourly_maps_dw.append(day_hourly_map)
    
    # Global Constants
    dwl = debiased_whittle.debiased_whittle_likelihood()
    TAPERING_FUNC = dwl.cgn_hamming 
    DWL_MAX_STEPS = 20         
    LAT_COL, LON_COL, VAL_COL, TIME_COL = 0, 1, 2, 3

    # --- Main Loop ---
    for day_idx in days_list:
        print(f'\n{"="*50}')
        print(f'--- Processing Real Data: Day {day_idx+1} (2024-07-{day_idx+1}) ---')
        print(f'{"="*50}')

        try:
            # Data Prepare
            daily_hourly_map_dw = daily_hourly_maps_dw[day_idx]
            daily_aggregated_tensor_dw = daily_aggregated_tensors_dw[day_idx].to(DEVICE_DW)

            if daily_aggregated_tensor_dw.shape[0] == 0:
                print(f"Skipping Day {day_idx+1}: No data.")
                continue

            # Parameter Init
            init_sigmasq   = 13.059
            init_range_lat = 0.154 
            init_range_lon = 0.195
            init_advec_lat = 0.0218
            init_range_time = 1.0
            init_advec_lon = -0.1689
            init_nugget    = 0.247
            
            init_phi2 = 1.0 / init_range_lon
            init_phi1 = init_sigmasq * init_phi2
            init_phi3 = (init_range_lon / init_range_lat)**2
            init_phi4 = (init_range_lon / init_range_time)**2

            initial_vals = [np.log(init_phi1), np.log(init_phi2), np.log(init_phi3), 
                            np.log(init_phi4), init_advec_lat, init_advec_lon, np.log(init_nugget)]

            params_list = [
                torch.tensor([val], requires_grad=True, dtype=torch.float64, device=DEVICE_DW)
                for val in initial_vals
            ]

            # -------------------------------------------------------
            # STEP 1: Debiased Whittle Optimization (ON CPU)
            # -------------------------------------------------------
            print(f"\n--- [Phase 1] Debiased Whittle Optimization (Device: {DEVICE_DW}) ---")
            
            raw_init_floats = [init_sigmasq, init_range_lat, init_range_lon, init_range_time, 
                               init_advec_lat, init_advec_lon, init_nugget]

            db = debiased_whittle.debiased_whittle_preprocess(
                daily_aggregated_tensors_dw, daily_hourly_maps_dw, day_idx=day_idx, 
                params_list=raw_init_floats, lat_range=[0,5], lon_range=[123.0, 133.0]
            )

            cur_df = db.generate_spatially_filtered_days(0, 5, 123, 133).to(DEVICE_DW)
            unique_times = torch.unique(cur_df[:, TIME_COL])
            time_slices_list = [cur_df[cur_df[:, TIME_COL] == t_val] for t_val in unique_times]

            print("Pre-computing J-vector...")
            J_vec, n1, n2, p_time, taper_grid = dwl.generate_Jvector_tapered( 
                time_slices_list, tapering_func=TAPERING_FUNC, 
                lat_col=LAT_COL, lon_col=LON_COL, val_col=VAL_COL, device=DEVICE_DW
            )
            
            I_sample = dwl.calculate_sample_periodogram_vectorized(J_vec)
            taper_autocorr_grid = dwl.calculate_taper_autocorrelation_fft(taper_grid, n1, n2, DEVICE_DW)

            optimizer_dw = torch.optim.LBFGS(
                params_list, lr=1.0, max_iter=20, history_size=100, 
                line_search_fn="strong_wolfe", tolerance_grad=1e-7
            )

            # 🟢 [타이머 시작]
            start_time = time.time()

            nat_str, phi_str, raw_str, loss, steps = dwl.run_lbfgs_tapered(
                params_list=params_list, optimizer=optimizer_dw, I_sample=I_sample,
                n1=n1, n2=n2, p_time=p_time, taper_autocorr_grid=taper_autocorr_grid, 
                max_steps=DWL_MAX_STEPS, device=DEVICE_DW
            )

            # 🟢 [타이머 종료 및 시간 계산]
            epoch_time = time.time() - start_time
            print(f"Whittle Optimization finished in {epoch_time:.2f}s.")
            
            # -------------------------------------------------------
            # STEP 2: Save Results (SINGLE FILE APPEND MODE)
            # -------------------------------------------------------
            
            loss_scaled = loss * n1 * n2 * 8  
            dw_estimates_values = [p.item() for p in params_list]
            dw_estimates_loss = dw_estimates_values + [loss_scaled]
            
            grid_res = int(daily_aggregated_tensor_dw.shape[0] / 8)
            
            # 🟢 [저장] time 인자에 epoch_time 전달
            res = alg_optimization(
                day=f"{years[0]}-07-{day_idx+1}", 
                cov_name="DW_Real", 
                space_size=grid_res, 
                lr=1.0, 
                params=dw_estimates_loss, 
                time=epoch_time,  # 0.0 -> epoch_time
                rmsre=0.0
            )
            
            date = datetime.now().strftime("%m%d%y")

            common_filename = f"real_dw_summary_LBFGS_{grid_res}_{date}"
            
            # 1. JSON 저장
            json_filepath = output_path / f"{common_filename}.json"
            current_data = BaseLogger.load_list(json_filepath) 
            current_data.append(res.__dict__)
            
            with json_filepath.open('w', encoding='utf-8') as f:
                json.dump(current_data, f, separators=(",", ":"), indent=4)
            
            # 2. CSV 저장
            csv_filepath = output_path / f"{common_filename}.csv"
            pd.DataFrame(current_data).to_csv(csv_filepath, index=False)
            
            print(f"✅ Day {day_idx+1} Whittle Finished. Appended to {common_filename}.[json/csv]")

        except Exception as e:
            print(f"🔴 Day {day_idx+1} Failed: {e}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    app()


