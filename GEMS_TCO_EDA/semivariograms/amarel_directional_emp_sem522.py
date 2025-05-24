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


    ############################## 

    instance_sem = evaluate.CrossVariogram(config.amarel_save_computed_semi_path,7)

    # load entire days in July
    idx_for_datamap= [0, 248]
    analysis_data_map, _ = data_load_instance.load_working_data_byday( df_map, ord_mm, nns_map, idx_for_datamap = idx_for_datamap)

    '''
    In this study, the center matched data is used. 
    '''
    tmp_lon = np.concatenate([
        -np.arange(0.18, 2, 0.063 * 3)[::-1],
        [-0.126, -0.063, 0, 0.063, 0.126],
        np.arange(0.18, 2, 0.063 * 3)
    ])

    tmp_lat = np.concatenate([
        -np.arange(0.176, 2.3, 0.044 * 5)[::-1],
        [-0.132, -0.044, 0, 0.044, 0.132],
        np.arange(0.176, 2.3, 0.044 * 5)
    ])

    lon_deltas = [ (0, round(a,3)) for a in tmp_lon]
    lat_deltas = [ ( round(a,3),0 ) for a in tmp_lat]
    tolerance = 0.02
    
    # Avoid redundant computation of symmetric semivariograms 
    lat_lag_sem = instance_sem.compute_directional_semivariogram(lat_deltas[len(lat_deltas)//2 :], analysis_data_map, days_list, tolerance)
    lon_lag_sem = instance_sem.compute_directional_semivariogram(lon_deltas[len(lon_deltas)//2 :], analysis_data_map, days_list, tolerance)

    cross_lat_lag_sem = instance_sem.compute_cross_lon_lat(lat_deltas, analysis_data_map, days_list, tolerance)
    cross_lon_lag_sem = instance_sem.compute_cross_lon_lat(lon_deltas, analysis_data_map, days_list, tolerance)

    print(lon_lag_sem)
    print(lat_lag_sem)

    lat_filename = f"empirical_lat_sem_july24.pkl"
    lon_filename = f"empirical_lon_sem_july24.pkl"

    cross_lat_filename = f"empirical_cross_lat_sem_july24.pkl"
    cross_lon_filename = f"empirical_cross_lon_sem_july24.pkl"

    output_path = instance_sem.save_path

    lat_filepath = os.path.join(output_path, lat_filename)
    with open(lat_filepath, 'wb') as pickle_file:
        pickle.dump(lat_lag_sem, pickle_file)  

    lon_filepath = os.path.join(output_path, lon_filename)
    with open(lon_filepath, 'wb') as pickle_file:
        pickle.dump(lon_lag_sem, pickle_file)   

    cross_lat_filepath = os.path.join(output_path, cross_lat_filename)
    with open(cross_lat_filepath, 'wb') as pickle_file:
        pickle.dump(cross_lat_lag_sem, pickle_file)  

    cross_lon_filepath = os.path.join(output_path, cross_lon_filename)
    with open(cross_lon_filepath, 'wb') as pickle_file:
        pickle.dump(cross_lon_lag_sem, pickle_file)  


if __name__ == '__main__':
    app()