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
from GEMS_TCO import alg_optimization, alg_opt_Encoder, likelihood_comparison


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
    map, ord_mm, nns_map= instance.load_mm20k_data_bymonthyear( lat_lon_resolution= lat_lon_resolution, mm_cond_number=mm_cond_number,years_=years, months_=month_range)
    
  
    input_path = Path("/home/jl2815/tco/exercise_output/likelihood_exercise/")
    for day in range(days):
        idx_for_datamap= [8*day,8*(day+1)]
        analysis_data_map, aggregated_data = instance.load_working_data_byday( map, ord_mm, nns_map, idx_for_datamap = idx_for_datamap)

        lenth_of_analysis =idx_for_datamap[1]-idx_for_datamap[0]
        print(f'2025-07-{day+1}, data size per hour: {aggregated_data.shape[0]/lenth_of_analysis}, smooth: {v}')
        
        params_item = list(df_1250.iloc[day][:-1])
        params = torch.tensor(params_item, dtype=torch.float64, requires_grad=True)
        
        print(f'mm_cond_number {mm_cond_number}, params: {params} ')
        

        
        key_order = [0,1,2,4,3,5,7,6]
        reordered_dict, reordered_df = instance.reorder_data(analysis_data_map, aggregated_data, key_order)

        ord_instance = kernels.model_fitting(
                smooth=v,
                input_map=analysis_data_map,
                aggregated_data=aggregated_data,
                nns_map=nns_map,
                mm_cond_number=mm_cond_number,
                nheads = nheads
            )

        reord_instance = kernels.model_fitting(
                smooth = v,
                input_map = reordered_dict,
                aggregated_data = reordered_df,
                nns_map = nns_map,
                mm_cond_number = mm_cond_number,
                nheads = nheads
            )

        start_time = time.time()
        # optimizer = optim.Adam([params], lr=0.01)  # For Adam
        ord_vec = ord_instance.compute_vecc_nll_interpolate( params, ord_instance.matern_cov_anisotropy_v05 )   
        end_time = time.time()
        epoch_time1 = end_time - start_time 
        
        start_time = time.time()
        reord_vec = reord_instance.compute_vecc_nll_testing( params, reord_instance.matern_cov_anisotropy_v05 )
        end_time = time.time()
        epoch_time2 = end_time - start_time 
        
        '''  exclude for resolution 5000
        start_time = time.time()
        full = ord_instance.compute_full_nll( params, reord_instance.matern_cov_anisotropy_v05 )
        end_time = time.time()
        epoch_time3 = end_time - start_time 
        print(f'ordered_vecc: {ord_vec.item()}, reordered_vecc: {reord_vec.item()}, full: {full.item()}')
        '''
       
        params_item = [round(p, 3) for p in params_item]
        ord_out =  params_item + [ round(ord_vec.item(),3)]
        reord_out = params_item + [  round(reord_vec.item(),3)]
        # full_out = params_item + [  round(full.item(),3)]
      
        input_filepath = input_path / f"likelihood_comparison_{ (200 / lat_lon_resolution[0]) * (100 / lat_lon_resolution[0]) }.json"
        res1 = likelihood_comparison( f"2025-07-{day+1}", "vecc ord_contender", (200 / lat_lon_resolution[0]) * (100 / lat_lon_resolution[0]), ord_out,  round(epoch_time1,3))
        res2 = likelihood_comparison( f"2025-07-{day+1}", "vecc Reord_contender", (200 / lat_lon_resolution[0]) * (100 / lat_lon_resolution[0]), reord_out,  round(epoch_time2,3))
        
        # res3 = likelihood_comparison( f"2025-07-{day+1}", "Full", (200 / lat_lon_resolution[0]) * (100 / lat_lon_resolution[0]), full_out,  round(epoch_time3,3))
        loaded_data = res1.load(input_filepath)
        loaded_data.append( res1.toJSON() )
        loaded_data.append( res2.toJSON() )
        # loaded_data.append( res3.toJSON() )
        res1.save(input_filepath,loaded_data)

        fieldnames = ['day', 'cov_name', 'lat_lon_resolution', 'sigma','range_lat','range_lon','advec_lat','advec_lon','beta','nugget','loss', 'time']

        csv_filepath = input_path/f"likelihood_comparison_{(200 / lat_lon_resolution[0]) * (100 / lat_lon_resolution[0])}.csv"
        res1.tocsv( loaded_data, fieldnames,csv_filepath )


if __name__ == '__main__':
    app()


