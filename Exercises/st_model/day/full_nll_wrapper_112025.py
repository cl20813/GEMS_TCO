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


from GEMS_TCO import kernels_reparam_space_time as kernels_reparam_space_time
from GEMS_TCO import orderings as _orderings 
from GEMS_TCO import alg_optimization, alg_opt_Encoder

from typing import Optional, List, Tuple
from pathlib import Path
import typer
import json
from json import JSONEncoder
from GEMS_TCO import configuration as config
from GEMS_TCO.data_loader import load_data2, exact_location_filter
from GEMS_TCO import debiased_whittle

app = typer.Typer(context_settings={"help_option_names": ["--help", "-h"]})
@app.command()

def cli(
    v: float = typer.Option(0.5, help="smooth"),
    space: List[str] = typer.Option(['20', '20'], help="spatial resolution"),
    days: List[str] = typer.Option(['0', '31'], help="Number of nearest neighbors in Vecchia approx."),
    mm_cond_number: int = typer.Option(10, help="Number of nearest neighbors in Vecchia approx."),
    nheads: int = typer.Option(200, help="Number of iterations in optimization")

) -> None:
      
    lat_lon_resolution = [int(s) for s in space[0].split(',')]
    days_s_e = list(map(int, days[0].split(',')))
    days_list = list(range(days_s_e[0], days_s_e[1]))


    # parsed_params = [float(p) for p in params[0].split(',')]
    # params = torch.tensor(parsed_params, requires_grad=True)

    ############################## 
    ## initialize setting
    lat_lon_resolution = [int(s) for s in space[0].split(',')]
    days_s_e = list(map(int, days[0].split(',')))
    days_list = list(range(days_s_e[0], days_s_e[1]))
    years = ['2024']
    month_range =[7,8]


    output_path = input_path = Path(config.amarel_estimates_day_path)
    data_load_instance = load_data2(config.amarel_data_load_path)


    df_map, ord_mm, nns_map = data_load_instance.load_maxmin_ordered_data_bymonthyear(
    lat_lon_resolution=lat_lon_resolution, 
    mm_cond_number=mm_cond_number,
    years_=years, 
    months_=month_range,
    lat_range=[0.0, 5.0],      
    lon_range=[123.0, 133.0] 
    )

    daily_aggregated_tensors = [] 
    daily_hourly_maps = []      

    for day_index in range(31):
        hour_start_index = day_index * 8
        hour_end_index = (day_index + 1) * 8
        #hour_end_index = day_index*8 + 1
        hour_indices = [hour_start_index, hour_end_index]
        
        day_hourly_map, day_aggregated_tensor = data_load_instance.load_working_data(
        df_map, 
        hour_indices, 
        ord_mm= ord_mm,  # or just omit it
        dtype=torch.float, # or just omit it 
        keep_ori=False
        )

        daily_aggregated_tensors.append( day_aggregated_tensor )
        daily_hourly_maps.append( day_hourly_map )



    for day_idx in days_list:  # 0-based

        # Assuming data access is correct
        daily_hourly_map = daily_hourly_maps[day_idx]
        daily_aggregated_tensor = daily_aggregated_tensors[day_idx]

        lat_range= [0,5]
        lon_range= [123.0, 133.0]
        nn = daily_aggregated_tensors[0].shape[0]

        param1 = [3.36336504695141, 1.4000489357246877, -1.5226391671330102, -3.122542129387705, 0.10834442729589913, -0.22068132726903175, 1.1186604833527585]
        model_params = param1
        instance = debiased_whittle.full_vecc_dw_likelihoods(daily_aggregated_tensors, daily_hourly_maps, day_idx=day_idx, params_list=model_params, lat_range=lat_range, lon_range=lon_range)
        instance.initiate_model_instance_vecchia(v, nns_map, mm_cond_number, nheads)
        res = instance.likelihood_wrapper()
        print(f' full likelihood: {torch.round(res[0]*nn, decimals=2)},\n vecchia: {torch.round(res[1]*nn, decimals=2)}, \n whittle de-biased: {torch.round(res[2], decimals = 2)}')
if __name__ == "__main__":
    app()



