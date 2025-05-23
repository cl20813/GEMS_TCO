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

import torch
from collections import defaultdict

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import copy  # clone tensor

# Custom imports
import GEMS_TCO
from GEMS_TCO import kernels
from GEMS_TCO import data_preprocess 
from GEMS_TCO import kernels 
from GEMS_TCO import orderings as _orderings 
from GEMS_TCO import load_data_amarel
from GEMS_TCO import alg_optimization, alg_opt_Encoder

import torch
from collections import defaultdict

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import copy                    # clone tensor

from pathlib import Path
import json
from json import JSONEncoder
from typing import Optional, List, Tuple
import typer

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})
@app.command()

def cli(
    v: float = typer.Option(0.5, help="smooth"),
    lr: float = typer.Option(0.01, help="learning rate"),
    step: int = typer.Option(200, help="Number of iterations in optimization"),
    space: List[str] = typer.Option(['20', '20'], help="spatial resolution"),
    days: int = typer.Option(1, help="Number of nearest neighbors in Vecchia approx."),
    mm_cond_number: int = typer.Option(1, help="Number of nearest neighbors in Vecchia approx."),
    params: List[str] = typer.Option(['20', '8.25', '5.25', '.2', '.2', '.05', '5'], help="Initial parameters"),
    ## mm-cond-number should be called in command line
    ## negative number can be a problem when parsing with typer
    epochs: int = typer.Option(100, help="Number of iterations in optimization"),
    nheads: int = typer.Option(200, help="Number of iterations in optimization")
) -> None:

    input_path = "/home/jl2815/tco/exercise_output/estimates/day/"  

    input_filename = "full_day_v(0.5)_1250_july24.pkl"
    input_filepath = os.path.join(input_path, input_filename)
    # Load pickle
    with open(input_filepath, 'rb') as pickle_file:
        amarel_map1250 = pickle.load(pickle_file)

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

  
    df = df_1250

    lat_lon_resolution = [int(s) for s in space[0].split(',')]
    print(f'lat_lon_look: {lat_lon_resolution}')
    years = ['2024']
    month_range = [7, 8]

    
    # 50 for 10 work best for competitor 2 lags reordered, cahced
    #  50 for resolution 10: result1 [11,10,9] result2 = [9,11,10]
    #  200 for resolution 4, result1 [21,2,7] result2=[7,7,16]
    # 300 for resolution 4, result1 [23, 1, 6]  result2 = [6,10,14]

    instance = load_data_amarel()
    map, ord_mm, nns_map = instance.load_mm20k_data_bymonthyear(lat_lon_resolution=lat_lon_resolution, mm_cond_number=mm_cond_number, years_=years, months_=month_range)
    for day in range(days):
        print(f'\n Day {day+1} data size per day: { (200 / lat_lon_resolution[0]) * (100 / lat_lon_resolution[0]) } \n')

        idx_for_datamap = [8 * (day), 8 * (day+1)]
        # params = [27.25, 2.18, 2.294, 4.099e-4, -0.07915, 0.0999, 3.65]   #200
        params = list(df.iloc[day][:-1])
        params = torch.tensor(params, dtype=torch.float64, requires_grad=True)
        
        # data
        analysis_data_map, aggregated_data = instance.load_working_data_byday(map, ord_mm, nns_map, idx_for_datamap=idx_for_datamap)

        # different approximations
        key_order = [0, 1, 2, 4, 3, 5, 7, 6]

        reordered_dict, reordered_df = instance.reorder_data(analysis_data_map, aggregated_data, key_order)

        model_instance = kernels.model_fitting(
                smooth=v,
                input_map=reordered_dict,
                aggregated_data=reordered_df,
                nns_map=nns_map,
                mm_cond_number=mm_cond_number,
                nheads = nheads
            )
        
        model_instance2 = kernels.model_fitting(
                smooth=v,
                input_map=analysis_data_map,
                aggregated_data=aggregated_data,
                nns_map=nns_map,
                mm_cond_number=mm_cond_number,
                nheads = nheads
            )
        
        start_time = time.time()
        params = list(df.iloc[day][:-1])

      
    
        params = torch.tensor(params, dtype=torch.float64, requires_grad=True)
        optimizer, scheduler = model_instance.optimizer_fun(params, lr= lr, betas=(0.9, 0.80), eps=1e-8, step_size=step, gamma=0.9)    

        out1, epoch1 = model_instance2.run_vecc_interpolate(params, optimizer,scheduler, model_instance.matern_cov_anisotropy_v05, epochs=epochs)
    
        end_time = time.time()
        epoch_time1 = end_time - start_time
        print(f'day vecc b2 {day + 1} took {epoch_time1:.2f}')


        start_time = time.time()
        params = list(df.iloc[day][:-1])
        params = torch.tensor(params, dtype=torch.float64, requires_grad=True)
        # optimizer = optim.Adam([params], lr=0.01)  # For Adam
        optimizer, scheduler = model_instance.optimizer_testing(params, lr= lr, betas=(0.9, 0.80), eps=1e-8, step_size= step, gamma=0.9)    
        out2 , epoch2= model_instance.run_vecc_testing(params, optimizer,scheduler, model_instance.matern_cov_anisotropy_v05, epochs=epochs)
       
        end_time = time.time()
        epoch_time2 = end_time - start_time
        print(f'day testing {day + 1} took {epoch_time2:.2f}')

        start_time = time.time()
        params = list(df.iloc[day][:-1])
        params = torch.tensor(params, dtype=torch.float64, requires_grad=True)
        optimizer, scheduler = model_instance.optimizer_fun(params, lr= lr, betas=(0.9, 0.80), eps=1e-8, step_size= step/2, gamma=0.9)    

        out3, epoch3 = model_instance2.run_full(params, optimizer,scheduler, model_instance.matern_cov_anisotropy_v05, epochs=epochs)
        end_time = time.time()
        epoch_time3 = end_time - start_time
        print(f'day full {day + 1} took {epoch_time3:.2f}')


        # ndarray, set is not serializable
        
        input_path = Path("/home/jl2815/tco/exercise_output/optimization/output/")
        input_filepath = input_path / f"vecc_alg_opt_{ (200 / lat_lon_resolution[0]) * (100 / lat_lon_resolution[0]) }.json"
        res1 = alg_optimization( f"2025-07-{day+1}", "Vecc_b2", (200 / lat_lon_resolution[0]) * (100 / lat_lon_resolution[0]) , lr,  step , out1, epoch_time1, epoch1)
        res2 = alg_optimization( f"2025-07-{day+1}", "Vecc_testing",(200 / lat_lon_resolution[0]) * (100 / lat_lon_resolution[0]) , lr, step , out2, epoch_time2, epoch2)
        res3 = alg_optimization( f"2025-07-{day+1}", "full_like", (200 / lat_lon_resolution[0]) * (100 / lat_lon_resolution[0]) , lr, step , out3, epoch_time3, epoch3)

        loaded_data = res1.load(input_filepath)
        loaded_data.append( res1.toJSON() )
        loaded_data.append( res2.toJSON() )
        loaded_data.append( res3.toJSON() )

        res1.save(input_filepath,loaded_data)

        fieldnames = ['day', 'cov_name', 'lat_lon_resolution', 'lr', 'stepsize',  'sigma','range_lat','range_lon','advec_lat','advec_lon','beta','nugget','loss', 'time', 'epoch']

        csv_filepath = input_path/f"vecc_alg_opt_{(200 / lat_lon_resolution[0]) * (100 / lat_lon_resolution[0])}.csv"
        res1.tocsv( loaded_data, fieldnames,csv_filepath )
if __name__ == "__main__":
    app()
