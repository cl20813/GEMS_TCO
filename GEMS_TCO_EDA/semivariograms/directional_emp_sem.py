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
        analysis_data_map, _ = data_load_instance.load_working_data_byday( df_map, ord_mm, nns_map, idx_for_datamap = idx_for_datamap)

        '''
        In this study, the center matched data is used. 
        '''
        tmp_lon = np.concatenate((np.linspace(-2, -0.2, 10), [-0.12, -0.06, 0, 0.06, 0.12], np.linspace(0.2, 2, 10)))
        tmp_lat = np.concatenate((np.linspace(-2, -0.2, 10), [-0.13, -0.04, 0, 0.04, 0.12], np.linspace(0.2, 2, 10)))

        lon_deltas = [ (0, round(a,1)) for a in tmp_lon]
        lat_deltas = [ ( round(a,1),0 ) for a in tmp_lat]
     
        tolerance = 0.02
        # days= list(np.arange(1,32))  # Example days you want to plot

        lon_lag_sem = instance_sem.cross_lon_lat(lon_deltas, analysis_data_map, days, tolerance)
        lat_lag_sem = instance_sem.cross_lon_lat(lat_deltas, analysis_data_map, days, tolerance)

        print(lon_lag_sem)
        print(lat_lag_sem)


    #############
    ############# latitude semivariogram 


if __name__ == '__main__':
    app()