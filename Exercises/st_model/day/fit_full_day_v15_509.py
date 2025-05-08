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
from GEMS_TCO import load_data
from GEMS_TCO import alg_optimization, alg_opt_Encoder

from typing import Optional, List, Tuple
from pathlib import Path
import typer
import json
from json import JSONEncoder
from GEMS_TCO import configuration as config

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})
@app.command()

def cli(
    v: float = typer.Option(0.5, help="smooth"),
    lr: float = typer.Option(0.01, help="learning rate"),
    step: int = typer.Option(80, help="Number of iterations in optimization"),
    coarse_factor: int = typer.Option(100, help="coarse factor in spline learning"),

    gamma_par: float = typer.Option(0.5, help="decreasing factor for learning rate"),
    space: List[str] = typer.Option(['20', '20'], help="spatial resolution"),
    days: List[str] = typer.Option(['0', '31'], help="Number of nearest neighbors in Vecchia approx."),
    
    mm_cond_number: int = typer.Option(1, help="Number of nearest neighbors in Vecchia approx."),
    params: List[str] = typer.Option(['20', '8.25', '5.25', '.2', '.2', '.05', '5'], help="Initial parameters"),
    ## mm-cond-number should be called in command line
    ## negative number can be a problem when parsing with typer
    epochs: int = typer.Option(100, help="Number of iterations in optimization"),
    nheads: int = typer.Option(200, help="Number of iterations in optimization")

) -> None:

    ############################## 

    ## initialize setting
    lat_lon_resolution = [int(s) for s in space[0].split(',')]
    days_s_e = list(map(int, days[0].split(',')))
    days_list = list(range(days_s_e[0], days_s_e[1]))
    years = ['2024']
    month_range =[7,8]
    output_path = input_path = Path(config.amarel_estimates_day_path)

    data_load_instance = load_data(config.amarel_data_load_path)
    df_map, ord_mm, nns_map= data_load_instance.load_mm20k_data_bymonthyear( lat_lon_resolution= lat_lon_resolution, mm_cond_number=mm_cond_number,years_=years, months_=month_range)

    # 5/09/24 try larger range parameters
    init_estimates =  Path(config.amarel_estimates_day_saved_path) / config.amarel_full_day_v05_range_plus2_csv
    estimates_df = pd.read_csv(init_estimates)
    
    for day in days_list:

        params = list(estimates_df.iloc[day][5:-3])
        params = torch.tensor(params, dtype=torch.float64, requires_grad=True)
        print(f'2024-07-{day+1}, data size per day: { (200/lat_lon_resolution[0])*(100/lat_lon_resolution[0]) }, smooth: {v}')
        print(f'mm_cond_number: {mm_cond_number},\ninitial parameters: \n {params}')
                
        idx_for_datamap= [8*day,8*(day+1)]
        analysis_data_map, aggregated_data = data_load_instance.load_working_data_byday( df_map, ord_mm, nns_map, idx_for_datamap = idx_for_datamap)

        model_instance = kernels.model_fitting(
                smooth=v,
                input_map=analysis_data_map,
                aggregated_data=aggregated_data,
                nns_map=nns_map,
                mm_cond_number=mm_cond_number,
                nheads = nheads
            )

        start_time = time.time()

        # optimizer = optim.Adam([params], lr=0.01)  # For Adam
        optimizer, scheduler = model_instance.optimizer_fun(params, lr=lr, betas=(0.9, 0.99), eps=1e-8, step_size=step, gamma = gamma_par)    
        out, epoch = model_instance.run_full(params, optimizer,scheduler, model_instance.matern_cov_anisotropy_v15, epochs=epochs)
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'End 2024-07-{day+1} for lr:{lr}, step size {step}, betas(_,b2):{0.99}, gamma:{gamma_par} took {epoch_time:.2f}, epoch {epochs}')
        print(f'params and loss {out}')


        input_filepath = output_path / f"full_day_v15_{ (200 / lat_lon_resolution[0]) * (100 / lat_lon_resolution[0]) }.json"
        
        res = alg_optimization( f"2024-07-{day+1}", f"Vecc_b2 and b2{0.99}", (200 / lat_lon_resolution[0]) * (100 / lat_lon_resolution[0]) , lr,  step , out, epoch_time, epoch)
        loaded_data = res.load(input_filepath)
        loaded_data.append( res.toJSON() )
        res.save(input_filepath,loaded_data)
        fieldnames = ['day', 'cov_name', 'lat_lon_resolution', 'lr', 'stepsize',  'sigma','range_lat','range_lon','advec_lat','advec_lon','beta','nugget','loss', 'time', 'epoch']

        csv_filepath = output_path/f"full_day_v15_{(200 / lat_lon_resolution[0]) * (100 / lat_lon_resolution[0])}.csv"
        res.tocsv( loaded_data, fieldnames,csv_filepath )

if __name__ == '__main__':
    app()


