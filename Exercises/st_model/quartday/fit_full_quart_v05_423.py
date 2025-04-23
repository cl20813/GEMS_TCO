# Standard libraries
import sys
# Add your custom path
sys.path.append("/cache/home/jl2815/tco")
import os


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
from GEMS_TCO import alg_optimization, alg_opt_Encoder
import json
from json import JSONEncoder

from typing import Optional, List, Tuple
from pathlib import Path
import typer

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})
@app.command()

def cli(
    v: float = typer.Option(0.5, help="smooth"),
    lr: float = typer.Option(0.01, help="learning rate"),
    step: int = typer.Option(40, help="Number of iterations in optimization"),
    avg_days: int = typer.Option(4, help="average days"),
    space: List[str] = typer.Option(['20', '20'], help="spatial resolution"),
    days: int = typer.Option(1, help="Number of nearest neighbors in Vecchia approx."),
    mm_cond_number: int = typer.Option(1, help="Number of nearest neighbors in Vecchia approx."),
    params: List[str] = typer.Option(['20', '8.25', '5.25', '.2', '.2', '.05', '5'], help="Initial parameters"),
    ## mm-cond-number should be called in command line
    ## negative number can be a problem when parsing with typer
    epochs: int = typer.Option(100, help="Number of iterations in optimization"),
    nheads: int = typer.Option(200, help="Number of iterations in optimization")
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

    df = df_1250.copy()
    ##

    # Set spaital coordinates
    years = ['2024']
    month_range =[7,8]
    
    # datapath = "C:\\Users\\joonw\\tco\\Extracted_data\\"  window
    datapath = "/home/jl2815/tco/data/pickle_data/"
    instance = load_data_amarel(datapath)
    coarse_dicts, ord_mm, nns_map = instance.load_mm20k_data_bymonthyear( lat_lon_resolution = lat_lon_resolution, mm_cond_number = mm_cond_number,years_ = years, months_ = month_range)
    
    result_q1 = {}
    result_q2 = {}
    result_q3 = {}
    result_q4 = {}
 

    for group_idx in range(1, 8):
        print(f'\n Group {group_idx} data size per day: { (200/lat_lon_resolution[0])*(100/lat_lon_resolution[0])  } \n')

        # parameters

        params = list(df.iloc[ 4*(group_idx-1)][:-1])
        params = torch.tensor(params, dtype=torch.float64, requires_grad=True)
        analysis_data_map_q1, entire_data_q1 = instance.load_working_data_by_quarterday( coarse_dicts, ord_mm, nns_map, which_group=group_idx, qrt_idx=1, avg_days= avg_days)
        
        params = list(df.iloc[ 4*(group_idx-1)][:-1])
        params = torch.tensor(params, dtype=torch.float64, requires_grad=True)
        analysis_data_map_q2, entire_data_q2 = instance.load_working_data_by_quarterday( coarse_dicts, ord_mm, nns_map, which_group=group_idx, qrt_idx=2, avg_days= avg_days)
        
        params = list(df.iloc[ 4*(group_idx-1)][:-1])
        params = torch.tensor(params, dtype=torch.float64, requires_grad=True)
        analysis_data_map_q3, entire_data_q3 = instance.load_working_data_by_quarterday( coarse_dicts, ord_mm, nns_map, which_group=group_idx, qrt_idx=3, avg_days= avg_days)
        
        params = list(df.iloc[ 4*(group_idx-1)][:-1])
        params = torch.tensor(params, dtype=torch.float64, requires_grad=True)
        analysis_data_map_q4, entire_data_q4 = instance.load_working_data_by_quarterday( coarse_dicts, ord_mm, nns_map, which_group=group_idx, qrt_idx=4, avg_days= avg_days)
        
        print(f'group_idx {group_idx+1}, data size per hour: {entire_data_q1.shape[0]}, smooth: {v}')
        print(lat_lon_resolution, mm_cond_number , params, v,lr)
        
        params = list(df_1250.iloc[group_idx][:-1])
        params = torch.tensor(params, dtype=torch.float64, requires_grad=True)

        model_q1 = kernels.model_fitting(
                smooth=v,
                input_map= analysis_data_map_q1,
                aggregated_data= entire_data_q1,
                nns_map=nns_map,
                mm_cond_number=mm_cond_number,
                nheads = nheads
            )

        model_q2 = kernels.model_fitting(
                smooth=v,
                input_map=analysis_data_map_q2,
                aggregated_data=entire_data_q2,
                nns_map=nns_map,
                mm_cond_number=mm_cond_number,
                nheads = nheads
            )

        model_q3 = kernels.model_fitting(
                smooth=v,
                input_map=analysis_data_map_q3,
                aggregated_data=entire_data_q3,
                nns_map=nns_map,
                mm_cond_number=mm_cond_number,
                nheads = nheads
            )

        model_q4 = kernels.model_fitting(
                smooth=v,
                input_map=analysis_data_map_q4,
                aggregated_data=entire_data_q4,
                nns_map=nns_map,
                mm_cond_number=mm_cond_number,
                nheads = nheads
            )
        

        start_time = time.time()
        # optimizer = optim.Adam([params], lr=0.01)  # For Adam
        optimizer, scheduler = model_q1.optimizer_fun(params, lr=lr, betas=(0.9, 0.8), eps=1e-8, step_size=step, gamma=0.9)    
        
        out_q1, epoch_q1 = model_q1.run_full(params, optimizer,scheduler, model_q1.matern_cov_anisotropy_v05, epochs=epochs)
        out_q2, epoch_q2 = model_q2.run_full(params, optimizer,scheduler, model_q2.matern_cov_anisotropy_v05, epochs=epochs)
        out_q3, epoch_q3 = model_q3.run_full(params, optimizer,scheduler, model_q3.matern_cov_anisotropy_v05, epochs=epochs)
        out_q4, epoch_q4 = model_q4.run_full(params, optimizer,scheduler, model_q4.matern_cov_anisotropy_v05, epochs=epochs)
        result_q1[group_idx] = out_q1
        result_q2[group_idx] = out_q2
        result_q3[group_idx] = out_q3
        result_q4[group_idx] = out_q4

        out_q1 = [ round(x, 4) for x in out_q1 ]
        out_q2 = [ round(x, 4) for x in out_q2 ]
        out_q3 = [ round(x, 4) for x in out_q3 ]
        out_q4 = [ round(x, 4) for x in out_q4 ]
        
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'group_idx {group_idx + 1} took {epoch_time/4:.2f} for each quarter of day on average')


        input_path = Path("/home/jl2815/tco/exercise_output/estimates/quartday/")
        input_filepath = input_path / f"fit_full_quarterv05_{ (200 / lat_lon_resolution[0]) * (100 / lat_lon_resolution[0]) }.json"

        res1 = alg_optimization( f"2025-07: q1_{group_idx}", "full_like", (200 / lat_lon_resolution[0]) * (100 / lat_lon_resolution[0]) , lr,  step , out_q1, epoch_time, epoch_q1)
        res2 = alg_optimization( f"2025-07: q2_{group_idx}", "full_like", (200 / lat_lon_resolution[0]) * (100 / lat_lon_resolution[0]) , lr,  step , out_q2, epoch_time, epoch_q2)
        res3 = alg_optimization( f"2025-07: q3_{group_idx}", "full_like", (200 / lat_lon_resolution[0]) * (100 / lat_lon_resolution[0]) , lr,  step , out_q3, epoch_time, epoch_q3) 
        res4 = alg_optimization( f"2025-07: q4_{group_idx}", "full_like", (200 / lat_lon_resolution[0]) * (100 / lat_lon_resolution[0]) , lr,  step , out_q4, epoch_time, epoch_q4) 

        
        loaded_data = res1.load(input_filepath)
        loaded_data.append( res1.toJSON() )
        loaded_data.append( res2.toJSON() )
        loaded_data.append( res3.toJSON() )
        loaded_data.append( res4.toJSON() )
        
        res1.save(input_filepath,loaded_data)
        fieldnames = ['day', 'cov_name', 'lat_lon_resolution', 'lr', 'stepsize',  'sigma','range_lat','range_lon','advec_lat','advec_lon','beta','nugget','loss', 'time', 'epoch']

        csv_filepath = input_path/f"fit_full_quarterv05_{(200 / lat_lon_resolution[0]) * (100 / lat_lon_resolution[0])}.csv"
        res1.tocsv( loaded_data, fieldnames,csv_filepath )

    output_filename_q1 = f"full_q1_v{v}_estimation_{int((200/rho_lat)*(100/rho_lon))}_july24.pkl"
    output_filename_q2 = f"full_q2_v{v}_estimation_{int((200/rho_lat)*(100/rho_lon))}_july24.pkl"
    output_filename_q3 = f"full_q3_v{v}_estimation_{int((200/rho_lat)*(100/rho_lon))}_july24.pkl"
    output_filename_q4 = f"full_q4_v{v}_estimation_{int((200/rho_lat)*(100/rho_lon))}_july24.pkl"

    output_filepath_q1 = os.path.join(input_path, output_filename_q1)
    with open(output_filepath_q1, 'wb') as pickle_file:
        pickle.dump(result_q1, pickle_file)    

    output_filepath_q2 = os.path.join(input_path, output_filename_q2)
    with open(output_filepath_q2, 'wb') as pickle_file:
        pickle.dump(result_q2, pickle_file)
    output_filepath_q3 = os.path.join(input_path, output_filename_q3)
    with open(output_filepath_q3, 'wb') as pickle_file:
        pickle.dump(result_q3, pickle_file)
    output_filepath_q4 = os.path.join(input_path, output_filename_q4)
    with open(output_filepath_q4, 'wb') as pickle_file:
        pickle.dump(result_q4, pickle_file)
        
    '''
    output_filename_q1 = f"full_q1_v({v})_estimation_{int((200/rho_lat)*(100/rho_lon))}_july24.pkl"
    output_filename_q2 = f"full_q2_v({v})_estimation_{int((200/rho_lat)*(100/rho_lon))}_july24.pkl"
    output_filename_q3 = f"full_q3_v({v})_estimation_{int((200/rho_lat)*(100/rho_lon))}_july24.pkl"
    output_filename_q4 = f"full_q4_v({v})_estimation_{int((200/rho_lat)*(100/rho_lon))}_july24.pkl"
    
    # base_path = "/home/jl2815/tco/data/pickle_data"
    output_path = "/home/jl2815/tco/exercise_output/estimates/quartday"

    output_filepath_q1 = os.path.join(output_path, output_filename_q1)
    output_filepath_q2 = os.path.join(output_path, output_filename_q2)
    output_filepath_q3 = os.path.join(output_path, output_filename_q3)
    output_filepath_q4 = os.path.join(output_path, output_filename_q4)

    with open(output_filepath_q1, 'wb') as pickle_file:
        pickle.dump(result_q1, pickle_file)    
    with open(output_filepath_q2, 'wb') as pickle_file:
        pickle.dump(result_q2, pickle_file) 
    with open(output_filepath_q3, 'wb') as pickle_file:
        pickle.dump(result_q3, pickle_file) 
    with open(output_filepath_q4, 'wb') as pickle_file:
        pickle.dump(result_q4, pickle_file) 
    '''


if __name__ == '__main__':
    app()