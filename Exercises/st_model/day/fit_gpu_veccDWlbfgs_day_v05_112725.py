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

# Custom imports


from GEMS_TCO import kernels_reparam_space_time_gpu as kernels_reparam_space_time
from GEMS_TCO import orderings as _orderings 
from GEMS_TCO import alg_optimization, alg_opt_Encoder

from typing import Optional, List, Tuple
from pathlib import Path
import typer
import json
from json import JSONEncoder
from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data2, exact_location_filter
from GEMS_TCO import debiased_whittle
from torch.nn import Parameter

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

    lat_lon_resolution = [int(s) for s in space[0].split(',')]
    days_s_e = list(map(int, days[0].split(',')))
    days_list = list(range(days_s_e[0], days_s_e[1]))


    # parsed_params = [float(p) for p in params[0].split(',')]
    # params = torch.tensor(parsed_params, requires_grad=True)

    ############################## 
    ## initialize setting
    years = ['2024']
    month_range =[7]

    output_path = input_path = Path(config.amarel_estimates_day_path)
    data_load_instance = load_data2(config.amarel_data_load_path)

    df_map, ord_mm, nns_map = data_load_instance.load_maxmin_ordered_data_bymonthyear(
    lat_lon_resolution=lat_lon_resolution, 
    mm_cond_number=mm_cond_number,
    years_=years, 
    months_=month_range,
    lat_range=[0.0, 5.0],      
    lon_range=[123.0, 133.0] 
    )

    daily_aggregated_tensors_dw = [] 
    daily_hourly_maps_dw = []      

    daily_aggregated_tensors_vecc = [] 
    daily_hourly_maps_vecc = []   


    for day_index in range(31):
        hour_start_index = day_index * 8
        hour_end_index = (day_index + 1) * 8
        #hour_end_index = day_index*8 + 1
        hour_indices = [hour_start_index, hour_end_index]
        
        day_hourly_map, day_aggregated_tensor = data_load_instance.load_working_data(
        df_map, 
        hour_indices, 
        ord_mm= None,  # or just omit it
        dtype=torch.float64, # or just omit it 
        keep_ori=keep_exact_loc
        )

        daily_aggregated_tensors_dw.append( day_aggregated_tensor )
        daily_hourly_maps_dw.append( day_hourly_map )
    
        day_hourly_map, day_aggregated_tensor = data_load_instance.load_working_data(
        df_map, 
        hour_indices, 
        ord_mm= ord_mm,  # or just omit it
        dtype=torch.float64, # or just omit it 
        keep_ori=keep_exact_loc
        )

        daily_aggregated_tensors_vecc.append( day_aggregated_tensor )
        daily_hourly_maps_vecc.append( day_hourly_map )





    # 5/09/24 try larger range parameters
    # init_estimates =  Path(config.amarel_estimates_day_saved_path) / config.amarel_full_day_v05_range_plus2_csv

  
    # --- Assume global variables are set: ---
    # daily_hourly_maps, daily_aggregated_tensors, nns_map
    # lat_lon_resolution, v, mm_cond_number, nheads
    # lr, patience, factor, epochs

# --- 1. Global Configuration & Constants ---
    dwl = debiased_whittle.debiased_whittle_likelihood()
    TAPERING_FUNC = dwl.cgn_hamming 
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # Global L-BFGS Settings
    LBFGS_LR = 1.0
    LBFGS_MAX_STEPS = 10      # 10 to 20  
    LBFGS_HISTORY_SIZE = 100   
    LBFGS_MAX_EVAL = 80       # line search from 50 to 80
    DWL_MAX_STEPS = 20         

    DELTA_LAT, DELTA_LON = 0.044, 0.063 
    LAT_COL, LON_COL, VAL_COL, TIME_COL = 0, 1, 2, 3

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")



    # --- 2. Run optimization loop over pre-loaded data ---

    for day_idx in days_list:  # 0-based

        print(f'\n{"="*40}')
        print(f'--- Starting Processing for Day {day_idx+1} (2024-07-{day_idx+1}) ---')
        print(f'{"="*40}')

        # Assuming data access is correct
        daily_hourly_map_dw = daily_hourly_maps_dw[day_idx]
        daily_aggregated_tensor_dw = daily_aggregated_tensors_dw[day_idx]

        daily_hourly_map_vecc = daily_hourly_maps_vecc[day_idx]
        daily_aggregated_tensor_vecc = daily_aggregated_tensors_vecc[day_idx]

        if isinstance(daily_aggregated_tensor_vecc, torch.Tensor):
            daily_aggregated_tensor_vecc = daily_aggregated_tensor_vecc.to(DEVICE)


        # Create 1-item lists to satisfy class interfaces designed for lists
        # This prevents loading 30 days of unused data

        # --- Parameter Initialization (SPATIO-TEMPORAL) ---
        '''  
        init_sigmasq   = 15.0
        init_range_lat = 0.66 
        init_range_lon = 0.7 
        init_nugget    = 1.5
        init_range_time = 1.0
        init_advec_lat = 0.02
        init_advec_lon = -0.08
        '''
        
        init_sigmasq   = 13.059
        init_range_lat = 0.154 
        init_range_lon = 0.195
        init_advec_lat = 0.0218
        init_range_time = 1.0
        init_advec_lon = -0.1689
        init_nugget    = 0.247
        
        # Map model parameters to the 'phi' reparameterization
        init_phi2 = 1.0 / init_range_lon                # 1/range_lon
        init_phi1 = init_sigmasq * init_phi2            # sigmasq / range_lon
        init_phi3 = (init_range_lon / init_range_lat)**2  # (range_lon / range_lat)^2
        init_phi4 = (init_range_lon / init_range_time)**2      # (range_lon / range_time)^2

        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Create Initial Parameters (Float64, Requires Grad)
        initial_vals = [np.log(init_phi1), np.log(init_phi2), np.log(init_phi3), 
                        np.log(init_phi4), init_advec_lat, init_advec_lon, np.log(init_nugget)]

        params_list = [
            torch.tensor([val], requires_grad=True, dtype=torch.float64, device=DEVICE)
            for val in initial_vals
        ]


# -------------------------------------------------------
        # STEP C: Phase 1 - Debiased Whittle Optimization
        # -------------------------------------------------------
        print("\n--- Phase 1: Debiased Whittle Initialization ---")
        
        # Helper list for preprocess (can use raw values or params_list)
        # We use the raw floats for the preprocessor init
        raw_init_floats = [init_sigmasq, init_range_lat, init_range_lon, init_range_time, 
                           init_advec_lat, init_advec_lon, init_nugget]

        # Initialize Preprocessor with the wrapper lists (0-index because list has length 1)
        db = debiased_whittle.debiased_whittle_preprocess(
            daily_aggregated_tensors_dw, 
            daily_hourly_maps_dw, 
            day_idx= day_idx, # It is index 0 in our wrapper list # this automatically lead to day_idx
            params_list=raw_init_floats, 
            lat_range=[0,5], 
            lon_range=[123.0, 133.0]
        )

        cur_df = db.generate_spatially_filtered_days(0, 5, 123, 133)
        
        if cur_df.numel() == 0 or cur_df.shape[1] <= max(LAT_COL, LON_COL, VAL_COL, TIME_COL):
            print(f"Error: Data for Day {day_idx+1} is empty or invalid.")
            exit()

        unique_times = torch.unique(cur_df[:, TIME_COL])
        time_slices_list = [cur_df[cur_df[:, TIME_COL] == t_val] for t_val in unique_times]

        # --- 1. Pre-compute J-vector, Taper Grid, and Taper Autocorrelation ---
        print("Pre-computing J-vector (Hamming taper)...")
        
        # --- 💥 REVISED: Renamed 'p' to 'p_time' 💥 ---
        J_vec, n1, n2, p_time, taper_grid = dwl.generate_Jvector_tapered( 
            time_slices_list,
            tapering_func=TAPERING_FUNC, 
            lat_col=LAT_COL, lon_col=LON_COL, val_col=VAL_COL,
            device=DEVICE
        )

        if J_vec is None or J_vec.numel() == 0 or n1 == 0 or n2 == 0 or p_time == 0:
            print(f"Error: J-vector generation failed for Day {day_idx+1}.")
            exit()
        

        I_sample = dwl.calculate_sample_periodogram_vectorized(J_vec)
        taper_autocorr_grid = dwl.calculate_taper_autocorrelation_fft(taper_grid, n1, n2, DEVICE)

        # Set up Optimizer for Whittle
        optimizer_dw = torch.optim.LBFGS(
            params_list, lr=1.0, max_iter=20, history_size=100, 
            line_search_fn="strong_wolfe", tolerance_grad=1e-7  # 1e-5 to 1e-7
        )

        print(f"Running Whittle L-BFGS on {DEVICE}...")
        # --- END REVISION ---

        # Run Whittle Optimization      
        nat_str, phi_str, raw_str, loss, steps = dwl.run_lbfgs_tapered(
            params_list=params_list,
            optimizer=optimizer_dw,
            I_sample=I_sample,
            n1=n1, n2=n2, p_time=p_time,
            taper_autocorr_grid=taper_autocorr_grid, 
            max_steps=DWL_MAX_STEPS,
            device=DEVICE
        )

# -------------------------------------------------------
# -------------------------------------------------------
        # STEP D: Handoff - Create 'new_params_list'
        # -------------------------------------------------------
        
        # Scale loss for reporting (optional, based on your logic)
        loss_scaled = loss * n1 * n2 * 8  
        
        # 1. Create fresh tensors for Vecchia
        # We DETACH from the Whittle graph so Vecchia starts fresh
        new_params_list = [
            p.detach().clone().to(DEVICE).requires_grad_(True)
            for p in params_list
        ]

        # 2. Apply lower bound to nugget (index -1)
        with torch.no_grad():
            new_params_list[-1].clamp_min_(-2.0)

        # Log Initial estimates (Whittle Results)
        dw_estimates_loss = [x.item() for x in new_params_list] + [loss_scaled]
        
        # (Your JSON/CSV saving code remains here...)
        input_filepath = output_path / f"gpu_debiased_whittle_day_v05_LBFGS_NOV25_{ ( daily_aggregated_tensor_vecc.shape[0]/8 ) }.json"
        res = alg_optimization( f"2024-07-{day_idx+1}", "DW_Nov25", ( daily_aggregated_tensor_vecc[0].shape[0]/8 ) , lr,  step , dw_estimates_loss, 0 , 0)
        # ... saving logic ...

        
        # --- 💥 Instantiate the GPU Batched Class ---
        # NOTE: Ensure fit_vecchia_lbfgs is the NEW class we defined
        model_instance = kernels_reparam_space_time.fit_vecchia_lbfgs(
                smooth = v,
                input_map = daily_hourly_map_vecc,
                aggregated_data = daily_aggregated_tensor_vecc,
                nns_map = nns_map,
                mm_cond_number = mm_cond_number,
                nheads = nheads
            )




        # --- 💥 Set L-BFGS Optimizer ---
        optimizer_vecc = model_instance.set_optimizer(
                    new_params_list,     
                    lr=LBFGS_LR,            
                    max_iter=LBFGS_MAX_EVAL,        
                    history_size=LBFGS_HISTORY_SIZE 
                )

        print(f"\n--- Starting Phase 2: Vecchia Optimization (Day {day_idx+1}) ---")
        start_time = time.time()
        
        # --- 💥 Call the Batched Fit Method ---
        # REMOVED: model_instance.matern_cov_aniso_STABLE_log_reparam
        out, steps_ran = model_instance.fit_vecc_lbfgs(
                new_params_list,
                optimizer_vecc,
                # covariance_function argument is GONE
                max_steps=LBFGS_MAX_STEPS, 
                grad_tol=1e-7
            )

        end_time = time.time()
        epoch_time = end_time - start_time
        
        print(f"Vecchia Optimization finished in {epoch_time:.2f}s. Results: {out}")

        # (Your Result Saving Logic...)
        input_filepath = output_path / f"gpu_vecchiaDW_day_v05_LBFGS_NOV25_{ ( daily_aggregated_tensors_vecc[0].shape[0]/8 ) }.json"
        
        res = alg_optimization( f"2024-07-{day_idx+1}", "Vecc_Nov25", ( daily_aggregated_tensors_vecc[0].shape[0]/8 ) , lr,  step , out, epoch_time, 0)
        loaded_data = res.load(input_filepath)
        loaded_data.append( res.toJSON() )
        res.save(input_filepath,loaded_data)
        fieldnames = ['day', 'cov_name', 'lat_lon_resolution', 'lr', 'stepsize',  'sigma','range_lat','range_lon','advec_lat','advec_lon','beta','nugget','loss', 'time', 'epoch'] # 0 for epoch

        csv_filepath = input_path/f"gpu_vecchiaDW_v{int(v*100):03d}_LBFGS_NOV25_{(daily_aggregated_tensors_vecc[0].shape[0]/8 )}.csv"
        res.tocsv( loaded_data, fieldnames,csv_filepath )

if __name__ == "__main__":
    app()



