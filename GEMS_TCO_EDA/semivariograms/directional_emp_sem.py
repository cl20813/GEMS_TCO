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

# Custom imports
import GEMS_TCO
from GEMS_TCO import kernels
from GEMS_TCO import data_preprocess 
from GEMS_TCO import kernels 
from GEMS_TCO import orderings as _orderings 
from GEMS_TCO import load_data
from GEMS_TCO import alg_optimization, alg_opt_Encoder
from GEMS_TCO import evaluate
from typing import Optional, List, Tuple
from pathlib import Path
import typer
import json
from json import JSONEncoder
from GEMS_TCO import configuration as config


app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})
@app.command()

def cli(
    space: List[str] = typer.Option(['20', '20'], help="spatial resolution"),
    days: List[str] = typer.Option(['0', '31'], help="Number of nearest neighbors in Vecchia approx."),
    mm_cond_number: int = typer.Option(1, help="Number of nearest neighbors in Vecchia approx."),
) -> None:
    
    ## initialize setting
    lat_lon_resolution = [int(s) for s in space[0].split(',')]
    days_s_e = list(map(int, days[0].split(',')))
    days_list = list(range(days_s_e[0], days_s_e[1]))
    years = ['2024']
    month_range =[7,8]
    output_path = input_path = Path(config.amarel_estimates_day_path)


    data_load_instance = load_data(config.amarel_data_load_path)
    df_map, ord_mm, nns_map= data_load_instance.load_mm20k_data_bymonthyear(lat_lon_resolution = lat_lon_resolution, mm_cond_number=mm_cond_number,years_=years, months_=month_range)


   # rho_lat = lat_lon_resolution[0]          
   # rho_lon = lat_lon_resolution[1]
    ############################## 

    # Set the days of the month
    

    #############
    ############# longitude semivariogram 

    instance_sem = evaluate.CrossVariogram()
    for day in days_list:
        print(f'2024-07-{day+1}, data size per hour: { (int(158.7 / lat_lon_resolution[0] * (113.63 / lat_lon_resolution[0]))) } ')
        idx_for_datamap= [8*day,8*(day+1)]
        analysis_data_map, aggregated_data = data_load_instance.load_working_data_byday( df_map, ord_mm, nns_map, idx_for_datamap = idx_for_datamap)


        tmp = np.concatenate((np.linspace(-2, -0.2, 10), [-0.1, 0, 0.1], np.linspace(0.2, 2, 10)))
        lon_deltas = [ (0, round(a,1)) for a in tmp]
        lat_deltas = [ ( round(a,1),0 ) for a in tmp]
     
        tolerance = 0.02
        # days= list(np.arange(1,32))  # Example days you want to plot

        lon_lag_sem = instance_sem.cross_lon_lat(lon_deltas, analysis_data_map, days, tolerance)
        lat_lag_sem = instance_sem.cross_lon_lat(lat_deltas, analysis_data_map, days, tolerance)

        output_filename = f"empirical_lon_sem_{int((200/rho_lat)*(100/rho_lon))}_july24.pkl"

        # base_path = "/home/jl2815/tco/data/pickle_data"
        output_path = "/home/jl2815/tco/exercise_output/eda"
        output_filepath = os.path.join(output_path, output_filename)
        with open(output_filepath, 'wb') as pickle_file:
            pickle.dump(lon_lag_sem, pickle_file)    

    #############
    ############# latitude semivariogram 


if __name__ == '__main__':
    app()