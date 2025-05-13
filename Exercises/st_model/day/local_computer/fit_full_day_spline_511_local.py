# Standard libraries
import sys
# Add your custom path
gems_tco_path = "/Users/joonwonlee/Documents/GEMS_TCO-1/src"
sys.path.append(gems_tco_path)
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
from GEMS_TCO import configuration as config

from typing import Optional, List, Tuple
from pathlib import Path
import typer
import json
from json import JSONEncoder

# 20, 20
# /opt/anaconda3/envs/faiss_env/bin/python /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/local_computer/fit_full_day_spline_511_local.py --v 0.5 --lr 0.03 --step 100 --space "20,20" --days "0,1" --mm-cond-number 10 --params 20,8.25,5.25,.2,.2,.05,5 --epochs 700 --nheads 200

# 10, 10 
# /opt/anaconda3/envs/faiss_env/bin/python /Users/joonwonlee/Documents/GEMS_TCO-1/Exercises/st_model/day/local_computer/fit_full_day_spline_511_local.py --v 0.5 --lr 0.03 --step 100 --space "10,10" --days "0,1" --mm-cond-number 10 --params 20,8.25,5.25,.2,.2,.05,5 --epochs 700 --nheads 200


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
    # initialization
    lat_lon_resolution = [int(s) for s in space[0].split(',')]
    days_s_e = list(map(int, days[0].split(',')))
    days_list = list(range(days_s_e[0], days_s_e[1]))
    years = ['2024']
    month_range =[7,8]
    output_path = input_path = Path(config.mac_estimates_day_path)

    ## load ozone data from amarel
    data_load_instance = load_data(config.mac_data_load_path)
    df_map, ord_mm, nns_map= data_load_instance.load_mm20k_data_bymonthyear(lat_lon_resolution = lat_lon_resolution, mm_cond_number=mm_cond_number,years_=years, months_=month_range)  
    init_estimates =  Path(config.mac_estimates_day_path) / config.mac_full_day_v05_csv
    estimates_df = pd.read_csv(init_estimates)
    
    # only fit spline once because space are all same
    # load first data of analysis_data_map and aggregated_data to initialize spline_instance
    first_day_idx_for_datamap= [0,8]
    first_day_analysis_data_map, first_day_aggregated_data = data_load_instance.load_working_data_byday(df_map, ord_mm, nns_map, idx_for_datamap= first_day_idx_for_datamap)
    spline_instance = kernels.spline(
            epsilon = 1e-17, 
            coarse_factor= coarse_factor, 
            smooth = v, 
            input_map= first_day_analysis_data_map, 
            aggregated_data= first_day_aggregated_data, 
            nns_map=nns_map, 
            mm_cond_number= mm_cond_number)
    
    for day in days_list:  
        params = list(estimates_df.iloc[day][:-1])
        params = torch.tensor(params, dtype=torch.float64, requires_grad=True)
        print(f'2024-07-{day+1}, data size per day: { (200/lat_lon_resolution[0])*(100/lat_lon_resolution[0]) }, smooth: {v}')
        print(f'mm_cond_number: {mm_cond_number},\ninitial parameters: \n {params}')
                
        idx_for_datamap= [8*day,8*(day+1)]
        analysis_data_map, aggregated_data = data_load_instance.load_working_data_byday( df_map, ord_mm, nns_map, idx_for_datamap= idx_for_datamap)

        spline_instance.new_aggregated_data = aggregated_data[:,:4]
        spline_instance.new_aggregated_response = aggregated_data[:,2]

        start_time = time.time()
        optimizer, scheduler = spline_instance.optimizer_fun(params, lr= lr , betas=(0.9, 0.99), eps=1e-8, step_size= step, gamma= gamma_par)  
        out, epoch = spline_instance.run_full(params, optimizer,scheduler, epochs= epochs)
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'End 2024-07-{day+1} for lr:{lr}, step size {step}, betas(_,b2):{0.99}, gamma:{gamma_par} took {epoch_time:.2f}, epoch {epochs}')
        print(f'params and loss {out}')

        input_filepath = output_path / f"full_day_v{int(v*100):03d}_spline{ (200 / lat_lon_resolution[0]) * (100 / lat_lon_resolution[0]) }.json"
        
        res = alg_optimization( f"2024-07-{day+1}", f"full likelihood", (200 / lat_lon_resolution[0]) * (100 / lat_lon_resolution[0]) , lr,  step , out, epoch_time, epoch)
        loaded_data = res.load(input_filepath)
        loaded_data.append( res.toJSON() )
        res.save(input_filepath,loaded_data)
        fieldnames = ['day', 'cov_name', 'lat_lon_resolution', 'lr', 'stepsize',  'sigma','range_lat','range_lon','advec_lat','advec_lon','beta','nugget','loss', 'time', 'epoch']

        csv_filepath = output_path/f"full_day_v{int(v*100):03d}_spline{(200 / lat_lon_resolution[0]) * (100 / lat_lon_resolution[0])}.csv"
        res.tocsv( loaded_data, fieldnames,csv_filepath )

if __name__ == '__main__':
    app()


