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
    nheads: int = typer.Option(200, help="Number of iterations in optimization"),
    keep_exact_loc: bool = typer.Option(True, help="whether to keep exact location data or not")
) -> None:
      

    # parsed_params = [float(p) for p in params[0].split(',')]
    # params = torch.tensor(parsed_params, requires_grad=True)

    ############################## 
    ## initialize setting
    lat_lon_resolution = [int(s) for s in space[0].split(',')]
    days_s_e = list(map(int, days[0].split(',')))
    days_list = list(range(days_s_e[0], days_s_e[1]))
    years = ['2024']
    month_range =[7]

    lat_range= [1,3]
    lon_range= [125, 130]

    data_load_instance = load_data2(config.amarel_data_load_path)

    df_map, ord_mm, nns_map = data_load_instance.load_maxmin_ordered_data_bymonthyear(
    lat_lon_resolution=lat_lon_resolution, 
    mm_cond_number=mm_cond_number,
    years_=years, 
    months_=month_range,
    lat_range= lat_range,
    lon_range= lon_range
    )

    daily_aggregated_tensors_dw = [] 
    daily_hourly_maps_dw = []      

    daily_aggregated_tensors_vecc = [] 
    daily_hourly_maps_vecc = []   


    for day_index in range(31):
        hour_start_index = day_index * 8
        hour_end_index = (day_index + 1) * 8
        #hour_end_index = day_index*8 + 1
        hour_indices = [hour_start_index, hour_end_index]
        
        day_hourly_map, day_aggregated_tensor = data_load_instance.load_working_data(
        df_map, 
        hour_indices, 
        ord_mm= None,  # or just omit it
        dtype=torch.float64, # or just omit it 
        keep_ori=keep_exact_loc
        )

        daily_aggregated_tensors_dw.append( day_aggregated_tensor )
        daily_hourly_maps_dw.append( day_hourly_map )
    
        day_hourly_map, day_aggregated_tensor = data_load_instance.load_working_data(
        df_map, 
        hour_indices, 
        ord_mm= ord_mm,  # or just omit it
        dtype=torch.float64, # or just omit it 
        keep_ori=keep_exact_loc
        )

        daily_aggregated_tensors_vecc.append( day_aggregated_tensor )
        daily_hourly_maps_vecc.append( day_hourly_map )


    day1_va = [4.22817, 1.664023, 0.481917, -3.77204, 0.02213, -0.16318, -1.737487]
    day1_vl = [4.2866, 1.7396, 0.4891, -3.777, 0.02048, -0.16411, -12.05573]
    day1_vl2 = [4.2818, 1.7106, 0.4882, -3.7703, 0.0202, -0.1617, -4.9082]
    day1_vl3 = [4.2843, 1.7136, 0.4887, -3.7712, 0.0202, -0.1616, -14.7220]
    day1_dwl = [4.2739, 1.8060, 0.7948, -3.3599, 0.0223, -0.1672, -11.8381]


    day2_va = [3.7634, 1.2864, 0.6458, -4.05860, 0.001777, -0.22191, 0.7242916]
    day2_vl = [3.7503, 1.2538, 0.6472, -4.09016, 0.001728, -0.222897, 0.73606]
    day2_vl2 = [3.7440, 1.2168, 0.6473, -4.0569, 0.00105, -0.22017, 0.74023]
    day2_vl3 = [3.7440, 1.2160, 0.6473, -4.0566, 0.0011, -0.2202, 0.7403]
    day2_dwl =[4.1200, 1.6540, 0.8909, -3.4966, -0.0263, -0.2601, -0.0986]

    day3_va = [4.61865, 1.86892, 0.54694, -4.21337, -0.04020, -0.24562, -0.7427]
    day3_vl = [4.4038, 1.6321, 0.50344, -4.3653, -0.0417, -0.2480, 0.2393]
    day3_vl2 = [4.39415, 1.6056, 0.5026, -4.3046, -0.03894, -0.2451, 0.26051]
    day3_vl3 = [4.39425, 1.60585, 0.50261, -4.30459, -0.03894, -0.2451, 0.26052]
    
    day3_dwl = [4.0950, 1.6663, 0.6876, -3.3118, -0.0500, -0.2666, -0.5033]

    day4_va = [4.1117, 1.6978, 0.7622, -4.0126, 0.028246, -0.14168, -0.27482]
    day4_vl = [3.962231, 1.4687, 0.7822, -4.0332, 0.03072, -0.14823, 0.0994]
    day4_vl2 = [3.95552, 1.4425, 0.7809, -4.0119, 0.0301, -0.1465, 0.1117]
    day4_vl3 = [3.9555, 1.4425, 0.7809, -4.0119, 0.03009, -0.1465, 0.1118]
    
    day4_dwl = [3.9351, 1.8070, 1.0980, -3.5154, 0.0214, -0.1712, -0.5348]


    day1 = [day1_va, day1_vl, day1_vl2, day1_vl3, day1_dwl]
    day2 = [day2_va, day2_vl, day2_vl2, day2_vl3, day2_dwl]
    day3 = [day3_va, day3_vl, day3_vl2, day3_vl3, day3_dwl]
    day4 = [day4_va, day4_vl, day4_vl2, day4_vl3, day4_dwl]

    whole_params = [day1, day2,day3, day4]

    nn = daily_aggregated_tensors_vecc[0].shape[0]

    print(f'nn: {nn}')
    for day_idx in days_list:  # 0-based
        for i, model_params in enumerate(whole_params[day_idx]):
            print(f"Day {day_idx+1}, Model {i+1} params: {[round(p,4) for p in model_params]}")

            instance = debiased_whittle.full_vecc_dw_likelihoods(daily_aggregated_tensors_vecc, daily_hourly_maps_vecc, day_idx= day_idx, params_list=model_params, lat_range=lat_range, lon_range=lon_range)
     
            instance.initiate_model_instance_vecchia(v, nns_map, mm_cond_number, nheads)
    
            res = instance.likelihood_wrapper(daily_aggregated_tensors_dw, daily_hourly_maps_dw)
            print(f' full likelihood: {torch.round(res[0]*nn, decimals=2)},\n vecchia: {torch.round(res[1]*nn, decimals=2)}, \n whittle de-biased: {torch.round(res[2], decimals = 2)}')
            print(res[3:])
if __name__ == "__main__":
    app()



