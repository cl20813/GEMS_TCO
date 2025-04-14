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
from GEMS_TCO import orbitmap 
from GEMS_TCO import kernels 
from GEMS_TCO import orderings as _orderings 
from GEMS_TCO import load_data_amarel

from typing import Optional, List, Tuple
from pathlib import Path
import typer

# Configure logging to a specific file path
log_file_path = '/home/jl2815/tco/exercise_output/logs/fit_vecc.log'
logging.basicConfig(filename=log_file_path, level=logging.INFO, format='%(asctime)s - %(message)s')


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
      
    v = v
    # key_for_dict = [0, 8]
    lr = lr
    epochs = epochs
    nheads = nheads

    lat_lon_resolution = [int(s) for s in space[0].split(',')]
    parsed_params = [float(p) for p in params[0].split(',')]

    params = torch.tensor(parsed_params, requires_grad=True)

    days = days
    mm_cond_number = mm_cond_number
   
    
    rho_lat = lat_lon_resolution[0]
    rho_lon = lat_lon_resolution[1]

    # Argument parser

    ############################## 

    input_path = "/home/jl2815/tco/exercise_output/estimates/"
    input_filename = "estimation_1250_july24.pkl"
    input_filepath = os.path.join(input_path, input_filename)
    # Load pickle
    with open(input_filepath, 'rb') as pickle_file:
        amarel_map1250= pickle.load(pickle_file)

    # Assuming df_1250 is your DataFrame
    df_1250 = pd.DataFrame()
    for key in amarel_map1250:
        tmp = pd.DataFrame(amarel_map1250[key][0].reshape(1, -1), columns=['sigmasq', 'range_lat', 'range_lon', 'advec_lat', 'advec_lon', 'beta', 'nugget'])
        tmp['loss'] = amarel_map1250[key][1]
        df_1250 = pd.concat((df_1250, tmp), axis=0)

    # Generate date range
    date_range = pd.date_range(start='07-01-24', end='07-31-24')

    # Ensure the number of dates matches the number of rows in df_1250
    if len(date_range) == len(df_1250):
        df_1250.index = date_range
    else:
        print("The number of dates does not match the number of rows in the DataFrame.")

    ######

    # Load the one dictionary to set spaital coordinates
    years = ['2024']
    month_range =[7,8]
    
    instance = load_data_amarel()
    map, ord_mm, nns_map= instance.load_mm20k_data_bymonthyear( lat_lon_resolution= lat_lon_resolution, mm_cond_number= mm_cond_number ,years_=years, months_=month_range)
   
    result = {}
    key_for_dict = [0,8]
    for day in range(days):
        idx_for_datamap= [8*day,8*(day+1)]
        analysis_data_map, aggregated_data = instance.load_working_data_byday( map, ord_mm, nns_map, idx_for_datamap = idx_for_datamap)

        lenth_of_analysis = idx_for_datamap[1]-idx_for_datamap[0]
        print(f'day {day+1}, data size per hour: {aggregated_data.shape[0]/lenth_of_analysis}')
        print(lat_lon_resolution, mm_cond_number, idx_for_datamap, params, v,lr)

        # params = [24.42, 1.92, 1.92, 0.001, -0.045, 0.237, 3.34]
        # params = torch.tensor(params, requires_grad=True)
        params = list(df_1250.iloc[day-1][:-1])
        params = torch.tensor(params, dtype=torch.float64, requires_grad=True)

        # different approximations 
        key_order = [0,1,2,4,3,5,7,6]
        keys = list(analysis_data_map.keys())
        reordered_dict = {keys[key]: analysis_data_map[keys[key]] for key in key_order}

        model_instance = kernels.model_fitting(
                smooth=v,
                input_map= reordered_dict,
                aggregated_data=aggregated_data,
                nns_map=nns_map,
                mm_cond_number=mm_cond_number,
                nheads = nheads 
            )

        start_time = time.time()
        # optimizer = optim.Adam([params], lr=0.01)  # For Adam
        optimizer, scheduler = model_instance.optimizer_fun(params, lr=0.01, betas=(0.9, 0.8), eps=1e-8, step_size=20, gamma=0.9)    
        out = model_instance.run_vecc_extrapolate(params, optimizer,scheduler, epochs=epochs)
        result[day+1] = out

        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'day {day + 1} took {epoch_time:.2f}')

    output_filename = f"vecc_extra_estimates_{int((200/rho_lat)*(100/rho_lon))}_july24.pkl"

    # base_path = "/home/jl2815/tco/data/pickle_data"
    output_path = "/home/jl2815/tco/exercise_output/estimates"
    output_filepath = os.path.join(output_path, output_filename)
    with open(output_filepath, 'wb') as pickle_file:
        pickle.dump(result, pickle_file)    


if __name__ == "__main__":
    app()



