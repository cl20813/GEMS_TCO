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
import GEMS_TCO
from GEMS_TCO import kernels
from GEMS_TCO import data_preprocess 
from GEMS_TCO import kernels 
from GEMS_TCO import orderings as _orderings 
from GEMS_TCO import load_data_amarel

# Configure logging to a specific file path
log_file_path = '/home/jl2815/tco/exercise_output/logs/fit_full.log'
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(message)s')

from typing import Optional, List, Tuple
from pathlib import Path
import typer

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})
@app.command()

def cli(
    v: float = typer.Option(0.5, help="smooth"),
    lr: float = typer.Option(0.01, help="learning rate"),
    space: List[str] = typer.Option(['20', '20'], help="spatial resolution"),
    days: int = typer.Option(1, help="Number of nearest neighbors in Vecchia approx."),
    mm_cond_number: int = typer.Option(1, help="Number of nearest neighbors in Vecchia approx."),
    params: List[str] = typer.Option(['20', '8.25', '5.25', '.2', '.2', '.05', '5'], help="Initial parameters"),
    ## mm-cond-number should be called in command line
    ## negative number can be a problem when parsing with typer
    epochs: int = typer.Option(100, help="Number of iterations in optimization"),
    nheads: int = typer.Option(200, help="Number of iterations in optimization"),
) -> None:

 
    lat_lon_resolution = [int(s) for s in space[0].split(',')]
    parsed_params = [float(p) for p in params[0].split(',')]
    params = torch.tensor(parsed_params, requires_grad=True)

    rho_lat = lat_lon_resolution[0]          
    rho_lon = lat_lon_resolution[1]
    ############################## 

    ## load initial estimates 

    input_path = "/home/jl2815/tco/exercise_output/estimates/day/"
    input_filename = "full_day_v(0.5)_1250_july24.pkl"
    input_filepath = os.path.join(input_path, input_filename)
    # Load pickle
    with open(input_filepath, 'rb') as pickle_file:
        amarel_map1250= pickle.load(pickle_file)
   
    df_1250 = pd.DataFrame()
    for key in amarel_map1250:
        tmp = pd.DataFrame(amarel_map1250[key][0].reshape(1, -1), columns=['sigmasq', 'range_lat', 'range_lon', 'advec_lat', 'advec_lon', 'beta', 'nugget'])
        tmp['loss'] = amarel_map1250[key][1]
        df_1250 = pd.concat((df_1250, tmp), axis=0)

    date_range = pd.date_range(start='07-01-24', end='07-31-24')

    # Ensure the number of dates matches the number of rows in df_1250
    if len(date_range) == len(df_1250):
        df_1250.index = date_range
    else:
        print("The number of dates does not match the number of rows in the DataFrame.")

    ##

    # Set spaital coordinates
    years = ['2024']
    month_range =[7,8]
    
    instance = load_data_amarel()
    map, ord_mm, nns_map = instance.load_mm20k_data_bymonthyear( lat_lon_resolution = lat_lon_resolution, mm_cond_number = mm_cond_number,years_ = years, months_ = month_range)
    
    result_morning = {}
    result_noon = {}


    for day in range(days):
        idx_morning = [8*day,8*day+4]
        idx_noon = [8*day+4,8*day+8]

        analysis_data_map_morning, aggregated_data_morning = instance.load_working_data_byday( map, ord_mm, nns_map, idx_for_datamap = idx_morning)
        analysis_data_map_noon, aggregated_data_noon = instance.load_working_data_byday( map, ord_mm, nns_map, idx_for_datamap = idx_noon)

        lenth_of_analysis = idx_morning[1]- idx_morning[0]
        print(f'day {day+1}, data size per hour: {aggregated_data_morning.shape[0]/lenth_of_analysis}, smooth: {v}')
        print(lat_lon_resolution, mm_cond_number, idx_morning, params, v,lr)
        
        params = list(df_1250.iloc[day][:-1])
        params = torch.tensor(params, dtype=torch.float64, requires_grad=True)

        model_morning = kernels.model_fitting(
                smooth=v,
                input_map=analysis_data_map_morning,
                aggregated_data=aggregated_data_morning,
                nns_map=nns_map,
                mm_cond_number=mm_cond_number,
                nheads = nheads
            )

        model_noon = kernels.model_fitting(
                smooth=v,
                input_map=analysis_data_map_noon,
                aggregated_data=aggregated_data_noon,
                nns_map=nns_map,
                mm_cond_number=mm_cond_number,
                nheads = nheads
            )
        

        start_time = time.time()
        # optimizer = optim.Adam([params], lr=0.01)  # For Adam
        optimizer_m, scheduler_m = model_morning.optimizer_fun(params, lr=lr, betas=(0.9, 0.8), eps=1e-8, step_size=40, gamma=0.9)    
        optimizer_n, scheduler_n = model_noon.optimizer_fun(params, lr= lr, betas=(0.9, 0.8), eps=1e-8, step_size=40, gamma=0.9)
        out_m = model_morning.run_full(params, optimizer_m,scheduler_m, model_morning.matern_cov_anisotropy_v05, epochs=epochs)
        out_n = model_noon.run_full(params, optimizer_n,scheduler_n, model_noon.matern_cov_anisotropy_v05, epochs=epochs)
        result_morning[day+1] = out_m
        result_noon[day+1] = out_n

        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'day {day + 1} took {epoch_time/2:.2f} for each morning and noon')

    output_filename_m = f"full_morning_v({v})_estimation_{int((200/rho_lat)*(100/rho_lon))}_july24.pkl"
    output_filename_n = f"full_noon_v({v})_estimation_{int((200/rho_lat)*(100/rho_lon))}_july24.pkl"

    # base_path = "/home/jl2815/tco/data/pickle_data"
    output_path = "/home/jl2815/tco/exercise_output/estimates/halfday"

    output_filepath_m = os.path.join(output_path, output_filename_m)
    with open(output_filepath_m, 'wb') as pickle_file:
        pickle.dump(result_morning, pickle_file)    

    output_filepath_n = os.path.join(output_path, output_filename_n)
    with open(output_filepath_n, 'wb') as pickle_file:
        pickle.dump(result_noon, pickle_file)  

if __name__ == '__main__':
    app()


