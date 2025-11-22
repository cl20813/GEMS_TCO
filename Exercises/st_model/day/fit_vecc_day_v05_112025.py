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


from GEMS_TCO import kernels_reparam_space_time as kernels_reparam_space_time
from GEMS_TCO import orderings as _orderings 
from GEMS_TCO import alg_optimization, alg_opt_Encoder

from typing import Optional, List, Tuple
from pathlib import Path
import typer
import json
from json import JSONEncoder
from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data2, exact_location_filter


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

    daily_aggregated_tensors = [] 
    daily_hourly_maps = []      

    for day_index in range(31):
        hour_start_index = day_index * 8
        hour_end_index = (day_index + 1) * 8
        #hour_end_index = day_index*8 + 1
        hour_indices = [hour_start_index, hour_end_index]
        
        day_hourly_map, day_aggregated_tensor = data_load_instance.load_working_data(
        df_map, 
        hour_indices, 
        ord_mm= ord_mm,  # or just omit it
        dtype=torch.float, # or just omit it 
        keep_ori=keep_exact_loc
        )

        daily_aggregated_tensors.append( day_aggregated_tensor )
        daily_hourly_maps.append( day_hourly_map )


    # 5/09/24 try larger range parameters
    # init_estimates =  Path(config.amarel_estimates_day_saved_path) / config.amarel_full_day_v05_range_plus2_csv

  
    # --- Assume global variables are set: ---
    # daily_hourly_maps, daily_aggregated_tensors, nns_map
    # lat_lon_resolution, v, mm_cond_number, nheads
    # lr, patience, factor, epochs

    # --- L-BFGS SPECIFIC GLOBAL PARAMETERS ---
    LBFGS_LR = 1.0
    LBFGS_MAX_STEPS = 10       # Number of outer optimization steps
    LBFGS_HISTORY_SIZE = 100   # Memory for Hessian approximation
    LBFGS_MAX_EVAL = 50        # Max evaluations (line search) per step

    # --- 2. Run optimization loop over pre-loaded data ---

    for day_idx in days_list:  # 0-based

        # Assuming data access is correct
        daily_hourly_map = daily_hourly_maps[day_idx]
        daily_aggregated_tensor = daily_aggregated_tensors[day_idx]

        # --- Parameter Initialization (SPATIO-TEMPORAL) ---
        '''  
        init_sigmasq   = 15.0
        init_range_lat = 0.66 
        init_range_lon = 0.7 
        init_nugget    = 1.5
        init_range_time = 0.1
        init_advec_lat = 0.02
        init_advec_lon = -0.08
        '''
        init_sigmasq   = 13.059
        init_range_lat = 0.154 
        init_range_lon = 0.195 
        init_nugget    = 0.247
        init_range_time = 0.7
        init_advec_lat = 0.0218
        init_advec_lon = -0.1689

        
        # Map model parameters to the 'phi' reparameterization
        init_phi2 = 1.0 / init_range_lon                # 1/range_lon
        init_phi1 = init_sigmasq * init_phi2            # sigmasq / range_lon
        init_phi3 = (init_range_lon / init_range_lat)**2  # (range_lon / range_lat)^2
        init_phi4 = (init_range_lon / init_range_time)**2      # (range_lon / range_time)^2

        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 7-parameter spatio-temporal list (Log/Linear)
        params_list = [
            torch.tensor([np.log(init_phi1)],      requires_grad=True, dtype=torch.float64, device=device_str ), # [0] log(phi1)
            torch.tensor([np.log(init_phi2)],      requires_grad=True, dtype=torch.float64, device=device_str ), # [1] log(phi2)
            torch.tensor([np.log(init_phi3)],      requires_grad=True, dtype=torch.float64, device=device_str ), # [2] log(phi3)
            torch.tensor([np.log(init_phi4)],      requires_grad=True, dtype=torch.float64, device=device_str ), # [3] log(phi4)
            torch.tensor([init_advec_lat],         requires_grad=True, dtype=torch.float64, device=device_str ), # [4] advec_lat (linear)
            torch.tensor([init_advec_lon],         requires_grad=True, dtype=torch.float64, device=device_str ), # [5] advec_lon (linear)
            torch.tensor([np.log(init_nugget)],    requires_grad=True, dtype=torch.float64, device=device_str )  # [6] log(nugget)
        ]

        # --- Define parameter groups ---
        lr_all = LBFGS_LR
        all_indices = [0, 1, 2, 3, 4, 5, 6] 
        
        # L-BFGS requires the parameters to be iterable in a single list or group
        param_groups = [
            {'params': [params_list[idx] for idx in all_indices], 'lr': lr_all, 'name': 'all_params'}
        ]

        # --- Print Job Info (using placeholder print variables) ---

        print(f'\n--- Starting Day {day_idx+1} (2024-07-{day_idx+1}) ---')
        print(f'Data size per day: { daily_aggregated_tensor.shape[0]/8}, smooth: {v}')
        print(f'mm_cond_number: {mm_cond_number},\ninitial parameters: \n')
        for i, p in enumerate(params_list):
            print(f"  Param {i}: {p.item():.4f}")
                
        # --- 💥 Instantiate the L-BFGS Class ---
        # NOTE: Assuming fit_vecchia_lbfgs is available via kernels_reparam_space_time
        model_instance = kernels_reparam_space_time.fit_vecchia_lbfgs(
                smooth = v,
                input_map = daily_hourly_map,
                aggregated_data = daily_aggregated_tensor,
                nns_map = nns_map,
                mm_cond_number = mm_cond_number,
                nheads = nheads
            )

        start_time = time.time()
        
        # --- 💥 Set L-BFGS Optimizer ---
        # L-BFGS specific arguments are passed here
        optimizer = model_instance.set_optimizer(
                param_groups,     
                lr=LBFGS_LR,            
                max_iter=LBFGS_MAX_EVAL,        # max_iter in LBFGS is the line search limit
                history_size=LBFGS_HISTORY_SIZE 
            )

        # --- 💥 Call the L-BFGS Fit Method ---
        out, steps_ran = model_instance.fit_vecc_lbfgs(
                params_list,
                optimizer,
                model_instance.matern_cov_aniso_STABLE_log_reparam, 
                max_steps=LBFGS_MAX_STEPS # Outer loop steps
            )

        end_time = time.time()
        epoch_time = end_time - start_time
        
        print(f"Day {day_idx+1} optimization finished in {epoch_time:.2f}s over {steps_ran+1} L-BFGS steps.")
        print(f"Day {day_idx+1} final results (raw params + loss): {out}")


        input_filepath = output_path / f"vecchia_day_v05_LBFGS_NOV25_{ ( daily_aggregated_tensors[0].shape[0]/8 ) }.json"
        
        res = alg_optimization( f"2024-07-{day_idx+1}", "Vecc_Nov25", ( daily_aggregated_tensors[0].shape[0]/8 ) , lr,  step , out, epoch_time, 0)
        loaded_data = res.load(input_filepath)
        loaded_data.append( res.toJSON() )
        res.save(input_filepath,loaded_data)
        fieldnames = ['day', 'cov_name', 'lat_lon_resolution', 'lr', 'stepsize',  'sigma','range_lat','range_lon','advec_lat','advec_lon','beta','nugget','loss', 'time', 'epoch'] # 0 for epoch

        csv_filepath = input_path/f"vecchia_v{int(v*100):03d}_LBFGS_NOV25_{(daily_aggregated_tensors[0].shape[0]/8 )}.csv"
        res.tocsv( loaded_data, fieldnames,csv_filepath )

if __name__ == "__main__":
    app()



